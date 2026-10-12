#!/bin/bash
# Stage-1 smoke test on one A30, before any Stage-1 training (brief §9: parameters, step time and peak memory of every
# idea; and a check that each variant trains, scores and evaluates end to end). Each spec trains one epoch on N_TRAIN
# training graphs of a baseline run (spread evenly over its train tar: ttbar and HNL events for arm iii) plus its
# N_BIG largest training graphs (so the peak memory is the production worst case), monitoring N_VAL val graphs, then
# scores the same N_VAL val graphs, checks the scores and runs evaluate_gnn on them. Use the committed configs
# without overrides for numbers that stand for the production runs.
# Spec: NAME=CONFIG[:key=value[:key=value...]] (runner --set overrides). Run dirs: runs/stage1_smoke/job<ID>/<NAME>.
# Usage: sbatch slurm/stage1/smoke_variants.sbatch ARM SEED SPEC [SPEC...]
#SBATCH -J s1_smoke
#SBATCH -p submit-gpu-express
#SBATCH --constraint=nvidia_a30
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
export RUN_OVERRIDE=/ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/stage1_smoke/job$SLURM_JOB_ID
source /work/submit/jinboz1/hnl/hnl-displaced/slurm/baseline/_common.sh "$1" "$2"
shift 2
N_TRAIN=${N_TRAIN:-200}
N_VAL=${N_VAL:-40}
N_BIG=${N_BIG:-10}
GRAPHS=$HNL_DATA/runs/baseline_${TAG}/arm${ARM}_s${SEED}/${GRAPHS_SUBDIR:-ml_graphs}   # GRAPHS_SUBDIR=ltf_union_graphs: Canto-L graphs
python -I - "$GRAPHS" "$LOCAL/in" "$N_TRAIN" "$N_VAL" "$N_BIG" <<'EOF'
import os, sys, tarfile
src, dest, n_train, n_val, n_big = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
for tar_name, split, n, big in (("trainset.tar", "trainset", n_train, n_big), ("valset.tar", "valset", n_val, 0)):
    with tarfile.open(os.path.join(src, tar_name)) as t:
        members = [m for m in t.getmembers() if m.name.endswith(".pyg.gz")]
        pick = members[::max(1, len(members) // n)][:n]
        largest = sorted(members, key=lambda m: m.size)[len(members) - big:] if big else []
        pick += [m for m in largest if m not in pick]
        sizes = sorted(m.size for m in pick)
        for m in pick:
            m.name = os.path.basename(m.name)
            t.extract(m, os.path.join(dest, split), filter="data")
    print(f"staged {split}: {len(pick)} of {len(members)} graphs ({len(largest)} largest), "
          f"file size median {sizes[len(sizes) // 2] / 1e6:.2f} MB, max {sizes[-1] / 1e6:.2f} MB")
EOF
mkdir -p "$LOCAL/val"
cp -r "$LOCAL/in/valset" "$LOCAL/val/valset"

cat > "$LOCAL/check_scores.py" <<'EOF'
import glob, sys
import torch
from acorn.utils.loading_utils import load_pyg
n, s_true, s_fake, extra = 0, [], [], set()
for f in sorted(glob.glob(sys.argv[1] + "/*.pyg.gz")):
    g = load_pyg(f[:-3], weights_only=False)
    sc = g.scores if "scores" in g else g.edge_scores
    assert sc.shape[0] == g.edge_index.shape[1], (f, sc.shape, g.edge_index.shape)
    assert torch.isfinite(sc).all() and sc.min() >= 0 and sc.max() <= 1, f
    if hasattr(g, "edge_vtx_scores"):
        v = g.edge_vtx_scores
        assert v.shape == sc.shape and torch.isfinite(v).all(), f
        extra.add("edge_vtx_scores")
    y = (g.y if "y" in g else g.edge_y).bool()          # acorn saves edge_y under the unprefixed name y
    s_true.append(sc[y])
    s_fake.append(sc[~y])
    n += 1
assert n > 0, "no scored graphs"
t, k = torch.cat(s_true), torch.cat(s_fake)
print(f"scores ok in {n} graphs; mean score edge_y=1 {t.mean():.3f} edge_y=0 {k.mean():.3f}; "
      f"kept at 0.5: {(t > 0.5).float().mean():.3f} / {(k > 0.5).float().mean():.4f}; extra {sorted(extra)}")
EOF

# every step chained with && (errexit does not act inside a tested command list)
for spec in "$@"; do
    name=${spec%%=*}
    rest=${spec#*=}
    IFS=':' read -ra parts <<< "$rest"
    cfg=${parts[0]}
    sets=()
    for kv in "${parts[@]:1}"; do sets+=(--set "$kv"); done
    echo "==== $name: $cfg ${sets[*]}"
    if $R train --stage gnn --arm "$ARM" --seed 0 --run-dir "$RUN/$name" --input-dir "$LOCAL/in" --max-epochs 1 \
            --config "$cfg" "${sets[@]}" \
        && $R infer --stage gnn --ckpt "$RUN/$name/gnn/artifacts/last.ckpt" --input-dir "$LOCAL/val" \
            --out-dir "$LOCAL/out_$name" \
        && cp "$LOCAL/out_$name"/infer_*.json "$RUN/$name/" \
        && python -I "$LOCAL/check_scores.py" "$LOCAL/out_$name/valset" \
        && python -m hnl.eval.evaluate_gnn --graphs "$LOCAL/out_$name/valset" --out "$LOCAL/eval_$name" \
            --cuts 0.5 > /dev/null; then
        echo "SMOKE OK $name"
    else
        echo "SMOKE FAILED $name"
    fi
done

python -I - "$RUN" <<'EOF'
import glob, json, os, sys
import pandas as pd
rows = []
for d in sorted(glob.glob(os.path.join(sys.argv[1], "*"))):
    runs = glob.glob(os.path.join(d, "gnn", "run_*.json"))
    if not runs:
        rows.append({"name": os.path.basename(d), "status": "no run"})
        continue
    m = json.load(open(runs[0]))
    mets = [pd.read_csv(p) for p in glob.glob(os.path.join(d, "gnn", "lightning_logs", "*", "metrics.csv"))]
    met = pd.concat(mets) if mets else pd.DataFrame()
    st = met["train/step_time_seconds"].dropna() if "train/step_time_seconds" in met else pd.Series(dtype=float)
    inf = glob.glob(os.path.join(d, "infer_*.json"))
    rows.append({"name": os.path.basename(d), "status": m.get("state"), "params": m.get("n_parameters"),
                 "peak_mem_gib": m.get("peak_gpu_mem_gib"), "steps": m.get("global_step"),
                 "fit_s": m.get("fit_seconds"), "step_s_median": round(float(st.median()), 4) if len(st) else None,
                 "n_step_logs": len(st),
                 "predict_s": json.load(open(inf[0])).get("predict_seconds") if inf else None})
pd.set_option("display.width", 200)
print(pd.DataFrame(rows).to_string(index=False))
EOF
