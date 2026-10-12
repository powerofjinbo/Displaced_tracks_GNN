#!/bin/bash
# Smoke test of the X1 metric learning (plan D3; configs/trackml/ml_train_x1.yaml) on one GPU type, everything on
# node-local disk (nothing written to /ceph but the log):
#  1. feature store with the cluster features: parallel (spawned) staging equals sequential staging; the stored
#     hit_log_* attributes equal the Embedder's inputs on real events;
#  2. training on 40 training events and 5 valmon events (events_per_epoch 40, 3 epochs = 120 steps): FRNN on this GPU,
#     step time, peak memory;
#  3. pre-timeout stop: SIGUSR1 after the first epoch -> preempt.ckpt; resume_ckpt (slurm/trackml/_common.sh) must pick
#     it; the resumed run must end at step 120 and the two runs together must have trained on exactly the events of an
#     uninterrupted run (HNL_TML_EVENT_LOG: one line per training step).
# Usage: sbatch -p submit-gpu-express --constraint=<gpu feature> --nice=100 slurm/trackml/ml_smoke_x1.sbatch
#SBATCH -J tml_x1_smoke
#SBATCH -p submit-gpu-express
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:40:00
#SBATCH -o /work/submit/jinboz1/hnl/logs/%x_%j.out
source /work/submit/jinboz1/hnl/hnl-displaced/slurm/trackml/_common.sh
RUN=$LOCAL/run                                   # node-local: the smoke run never touches the run directories
ML_CONFIG=configs/trackml/ml_train_x1.yaml
NTRAIN=40
EPOCHS=3
echo "smoke X1 on $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader); node-local /tmp: $(df -h /tmp | tail -n 1)"
TR=$(events train | python -I -c "import sys; print(','.join(sys.stdin.read().strip().split(',')[:$NTRAIN]))")
VM=$(events valmon | python -I -c "import sys; print(','.join(sys.stdin.read().strip().split(',')[:5]))")

# 1. feature store
python -m hnl.trackml.graphs fs --events "$(cut -d, -f1-6 <<< "$TR")" --dest "$LOCAL/fs_seq" --ml-config "$ML_CONFIG" --nproc 1
python -m hnl.trackml.graphs fs --events "$(cut -d, -f1-6 <<< "$TR")" --dest "$LOCAL/fs_par" --ml-config "$ML_CONFIG" --nproc 3
python -m hnl.trackml.graphs fs --events "$TR" --dest "$LOCAL/fs/trainset" --ml-config "$ML_CONFIG" --nproc 8
python -m hnl.trackml.graphs fs --events "$VM" --dest "$LOCAL/fs/valset" --ml-config "$ML_CONFIG" --nproc 5
du -sh "$LOCAL/fs/trainset" "$LOCAL/fs/valset"
python - "$LOCAL" "$ML_CONFIG" <<'PYEOF'
import gzip, io, os, sys, torch, yaml
from hnl.trackml import events as E
loc, cfgp = sys.argv[1:]
cfg = yaml.safe_load(open(cfgp))
ld = lambda p: torch.load(io.BytesIO(gzip.decompress(open(p, "rb").read())), weights_only=False)
seq, par = sorted(os.listdir(f"{loc}/fs_seq")), sorted(os.listdir(f"{loc}/fs_par"))
assert seq == par and len(seq) == 6, (seq, par)
for f in seq:
    a, b = ld(f"{loc}/fs_seq/{f}"), ld(f"{loc}/fs_par/{f}")
    assert set(a.keys()) == set(b.keys())
    for k in a.keys():
        assert (torch.equal(a[k], b[k]) if isinstance(a[k], torch.Tensor) else a[k] == b[k]), (f, k)
print("fs check: parallel staging == sequential staging on", len(seq), "events")
g = ld(f"{loc}/fs_seq/{seq[0]}")
ev = next(E.iter_events([E.event_from_eid(g.event_id)], particles=False))
x = E.Embedder(None, hparams=cfg, seed=0).inputs(ev)
for j, (f, s) in enumerate(zip(cfg["node_features"], cfg["node_scales"])):
    assert torch.equal((g[f] / s).float(), x[:, j]), f
    print(f"  {f:16s} min {x[:, j].min():9.4f} median {x[:, j].median():9.4f} max {x[:, j].max():9.4f}")
print("fs check: stored cluster features == Embedder inputs on event", ev.event)
PYEOF

# 2. + 3. training, pre-timeout stop, resume
export HNL_TML_EVENT_LOG=$LOCAL/events.log
R="python -m hnl.train.acorn_runner train --stage ml --arm trackml --seed 0 --run-dir $RUN --input-dir $LOCAL/fs \
   --max-epochs $EPOCHS --config $ML_CONFIG --set ml_seed=0 --set events_per_epoch=$NTRAIN"
T0=$(date +%s)
$R > "$LOCAL/run1.out" 2>&1 &
P1=$!
until grep -q "^\[epoch\] 0 " "$LOCAL/run1.out" || ! kill -0 $P1 2>/dev/null; do sleep 2; done
sleep 5
kill -USR1 $P1 2>/dev/null && echo "SIGUSR1 sent $(( $(date +%s) - T0 ))s after start"
wait $P1 || { echo "run 1 failed"; tail -n 40 "$LOCAL/run1.out"; exit 1; }
grep -E "^\[epoch\]|Traceback|Error" "$LOCAL/run1.out" | cut -c1-250 || true
ls -la --time-style=+%T "$RUN/ml/artifacts" 2>/dev/null || true
RES=$(resume_ckpt ml)
echo "resume_ckpt -> $RES"
cp "$RUN/ml/run_${SLURM_JOB_ID}.json" "$LOCAL/run1.json"
$R --resume "$RES" > "$LOCAL/run2.out" 2>&1 || { echo "run 2 failed"; tail -n 40 "$LOCAL/run2.out"; exit 1; }
grep -E "^\[epoch\]|Traceback|Error|resuming" "$LOCAL/run2.out" | cut -c1-250 || true
cp "$RUN/ml/run_${SLURM_JOB_ID}.json" "$LOCAL/run2.json"
SMOKE_LOG_COPY=/work/submit/jinboz1/hnl/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_events.txt PL_GLOBAL_SEED=0 \
python - "$LOCAL" "$RUN" "$RES" "$NTRAIN" "$EPOCHS" <<'PYEOF'
import glob, json, os, sys
import numpy as np, pandas as pd, torch
from hnl.trackml.ml import epoch_events
loc, run, res, n, epochs = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
r1, r2 = json.load(open(f"{loc}/run1.json")), json.load(open(f"{loc}/run2.json"))
pre = torch.load(f"{run}/ml/artifacts/preempt.ckpt", map_location="cpu", weights_only=False)["global_step"]
print("run 1:", r1["state"], "global_step", r1["global_step"], "preempt.ckpt step", pre, "| run 2:", r2["state"],
      "global_step", r2["global_step"], "epochs", r2["epochs_completed"], "| gpu", r2["gpu"], "peak GiB",
      max(r1["peak_gpu_mem_gib"], r2["peak_gpu_mem_gib"]))
ok = r1["state"] == "preempted" and r2["state"] == "done" and r2["global_step"] == n * epochs
ok &= os.path.basename(res) == "preempt.ckpt" and pre == r1["global_step"]
mid = pre % n != 0
eids = sorted(f[len("event"):-len("-graph.pyg.gz")] for f in os.listdir(f"{loc}/fs/trainset"))
log = pd.read_csv(f"{loc}/events.log", sep=" ", names=["step", "epoch", "bidx", "eid"], dtype={"eid": str})
want = [eids[j] for e in range(epochs) for j in epoch_events(e, n, n, 0, int(os.getenv("PL_GLOBAL_SEED", 0)))]
seq_ok = log.step.tolist() == list(range(n * epochs)) and log.eid.tolist() == want
bad = [i for i, (a, b) in enumerate(zip(log.eid.tolist(), want)) if a != b]
print("event sequence of the two runs == uninterrupted sequence:", seq_ok, f"({len(log)} steps; stop mid-epoch: {mid};",
      f"first mismatch at row {bad[0] if bad else None}; steps unique: {log.step.is_unique})")
log.assign(want=(want + [""] * len(log))[:len(log)]).to_csv(os.environ["SMOKE_LOG_COPY"], sep=" ", index=False)
m = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(f"{run}/ml/lightning_logs/*/metrics.csv"))])
st = m["train/step_time_seconds"].dropna() if "train/step_time_seconds" in m else pd.Series(dtype=float)
print("step time [s] (Lightning throughput, per logged step): median", round(float(st.median()), 3) if len(st) else None,
      "| fit seconds run1", r1["fit_seconds"], "run2", r2["fit_seconds"])
print("SMOKE", "PASS" if ok and seq_ok else "FAIL")
sys.exit(0 if ok and seq_ok else 1)
PYEOF
