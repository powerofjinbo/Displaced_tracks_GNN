"""Feedback round 3, task A2: the acorn baselines with Canto's full track-builder menu, chosen on val, each new choice
evaluated once on test (post-freeze, descriptive).

Rounds 1 and 2 (hnl.eval.strong_baselines, hnl.eval.strong_baselines_armiii) gave acorn (ttbar), acorn (HNL) and
acorn (ttbar+HNL) (Stage-0 arms i, ii, iii; seeds 0, 1, 2; runs/baseline_v1/arm<A>_s<S>) connected components, acorn's
own CCandWalk (plain and score-guided), walkthrough + merging and MWS-layer + merging. Canto's own menu also holds the
two-pass Mutex Watershed + merging and the learned fragment merging. This module adds them and extends every family's
grid where its val choice sits at an edge:

  mws2passmerge  two-pass same-layer MWS + merging: the rows of RUN/trackbuild_val_ext (s1 0.2-0.5 x s2 0.02-0.1, as
                 Canto's extended grid), every acorn run has them;
  mwslmerge      learned merging: a GBDT per run trained on that run's own GNN scores of 2000 of its training graphs,
                 exactly as Canto's (slurm/feedback/fb3_lmerge.sbatch: sample rule of score_split_sample.sbatch,
                 hnl.trackbuild.learned_merge data / train), val grid RUN/trackbuild_val_fb3_lmerge (tau 0.005-0.5);
  edges          a family whose val choice (builder rule at FR* with the one-standard-error tie rule) has a parameter
                 at the lowest or highest value of that parameter in the family's grid is extended past that edge on val
                 (`extension`: the edge slice's configurations with the next values of the parameter's progression,
                 clipped at the parameter's range), on all nine acorn runs, until every choice is interior or the range
                 ends (RUN/trackbuild_val_fb3_ext<k>, `grid`); the parts are united per run in RUN/trackbuild_val_fb3_all
                 (`merge`, a data copy, as round 1's CCandWalk union).

Selection (val): per run and family the builder rule at FR* with the one-standard-error tie rule (hnl.eval.ranking,
tie_se); per run over all families (CC at its best cut <= FR*, CCandWalk plain and score-guided, walkthrough + merging,
MWS-layer + merging, two-pass + merging, learned merging) the same rule over the families' choices (the two-stage form of
hnl.eval.ranking's "best family", with CC included): the arm's val-best configuration per seed.

    python -m hnl.eval.strong_baselines_v3 pick --dir DIR --n 2000                 (graph names drawn, rng 0)
    python -m hnl.eval.strong_baselines_v3 provenance-lmerge ...                   (inside fb3_lmerge.sbatch)
    python -m hnl.eval.strong_baselines_v3 grid --graphs DIR --out OUT --configs 'fam,p,..;fam,p,..' [--merge-model M]
    python -m hnl.eval.strong_baselines_v3 inventory --step K                       (choices and edges on the current grids)
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import time

import numpy as np
import pandas as pd

from hnl.eval import bootstrap as BS_
from hnl.eval import ranking as RK
from hnl.eval import ranking_test as RT
from hnl.eval import strong_baselines as SB

DATA = RT.DATA
BASE = os.path.join(DATA, "runs", "baseline_v1")
FB3 = os.path.join(DATA, "runs", "feedback_v3", "strong_baselines_a2")
ARMS = (("i", "acorn (ttbar)"), ("ii", "acorn (HNL)"), ("iii", "acorn (ttbar+HNL)"))
SEEDS = ("0", "1", "2")
LMERGE_GRID = "trackbuild_val_fb3_lmerge"
ALL_GRID = "trackbuild_val_fb3_all"
EXCLUDE = ("submit00,submit01,submit77-1,submit60,submit62,submit63,submit64,submit65,submit66,submit67,submit68,"
           "submit71,submit72,submit73")
N_TRAIN_GRAPHS = 2000


def _repo(p: str) -> str:
    return RT._repo(p)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git() -> str:
    try:
        h = subprocess.run(["git", "-C", RT.REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        d = subprocess.run(["git", "-C", RT.REPO, "status", "--porcelain", "--", "src", "slurm"], capture_output=True,
                           text=True).stdout.strip()
        return h + (f" (src/slurm dirty: {len(d.splitlines())} files)" if d else "")
    except OSError:
        return "unknown"


# ------------------------------------------------------------------------------------------------- learned merging
def pick_graphs(names: list[str], n: int, seed: int = 0) -> list[str]:
    """The graphs drawn for the learned merging: n of the sorted names uniformly at random with numpy
    default_rng(seed).choice (sorted indices), as slurm/stage1/score_split_sample.sbatch (and learned_merge data
    --max-events) draw them; all names if n >= their number."""
    names = sorted(names)
    if n >= len(names):
        return names
    pick = np.sort(np.random.default_rng(seed).choice(len(names), n, replace=False))
    return [names[k] for k in pick]


def _infer_sha(d: str) -> list[str]:
    return sorted({json.load(open(p))["ckpt_sha256"] for p in glob.glob(os.path.join(d, "infer_*.json"))})


def provenance_lmerge(run: str, out: str, ckpt: str, pairs: str, scored: str, started: str) -> dict:
    """Checks and records one learned-merging model: the checkpoint that scored the training graphs is byte for byte the
    one behind the run's stored val and test scores (else: checksum mismatch, stop); the pair data are stored
    compressed (the same arrays, checked); counts of the drawn graphs per sample code."""
    sha = _sha256(ckpt)
    val_sha, test_sha = _infer_sha(os.path.join(run, "gnn_scores")), _infer_sha(os.path.join(run, "gnn_scores_test"))
    if val_sha != [sha] or test_sha != [sha]:
        raise SystemExit(f"checksum mismatch: checkpoint {sha} / val scores {val_sha} / test scores {test_sha}")
    z = np.load(pairs)
    arrays = {k: z[k] for k in z.files}
    dest = os.path.join(out, "pairs_train.npz")
    np.savez_compressed(dest + ".tmp.npz", **arrays)
    os.replace(dest + ".tmp.npz", dest)
    back = np.load(dest)
    for k, v in arrays.items():
        if not np.array_equal(back[k], v, equal_nan=v.dtype.kind == "f"):
            raise SystemExit(f"{dest}: array {k} differs after compression")
    picked = [ln.strip() for ln in open(os.path.join(out, "picked_graphs.txt")) if ln.strip()]
    scored_names = sorted(f for f in os.listdir(scored) if f.endswith(".pyg.gz"))
    if scored_names != sorted(picked):
        raise SystemExit(f"{scored}: the scored graphs are not the {len(picked)} drawn ones")
    codes: dict = {}
    for f in picked:
        c = f[len("event"):][0]
        codes[c] = codes.get(c, 0) + 1
    info = json.load(open(os.path.join(out, "model.joblib.json")))
    prov = {"what": "learned fragment merging (hnl.trackbuild.learned_merge) trained on this acorn run's own GNN scores "
                    "of its training graphs, as Canto's learned merging (feedback round 3, task A2; val)",
            "run": run, "graphs": os.path.join(run, "ml_graphs", "trainset.tar"), "n_drawn": len(picked),
            "draw": "numpy default_rng(0).choice over the sorted graph names (score_split_sample.sbatch rule)",
            "per_sample_code": codes, "ckpt": ckpt, "ckpt_sha256": sha, "val_scores_ckpt_sha256": val_sha[0],
            "test_scores_ckpt_sha256": test_sha[0], "pairs": int(len(arrays["y"])),
            "positive_frac": float(np.mean(arrays["y"])), "model": os.path.join(out, "model.joblib"),
            "model_sha256": _sha256(os.path.join(out, "model.joblib")), "model_info": info,
            "pairs_stored": dest + " (np.savez_compressed of the learned_merge data output; arrays checked equal)",
            "git": _git(), "slurm_job": os.environ.get("SLURM_JOB_ID", "local"), "host": os.uname().nodename,
            "started_utc": started, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(os.path.join(out, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=1)
    print(json.dumps(prov, indent=1))
    return prov


# ------------------------------------------------------------------------------------------------- families and edges
FAMILIES = (  # key, family, base val grid, label (rounds 1-2 labels kept), short
    ("acwalk", "acwalk", "trackbuild_val_acwalk_all", "acorn's CC + walkthrough (CCandWalk)", "CCandWalk"),
    ("acwalkg", "acwalkg", "trackbuild_val_acwalk_all", "acorn's CC + walkthrough, score-guided (CCandWalk)",
     "CCandWalk-g"),
    ("walkmerge", "walkmerge", "trackbuild_val_ext", "walkthrough + merging (Canto builder family)", "walk+M"),
    ("mwsm", "mwslayermerge", "trackbuild_val_ext", "MWS-layer + merging (Canto builder)", "MWS+M"),
    ("mws2pm", "mws2passmerge", "trackbuild_val_ext", "two-pass MWS-layer + merging (Canto builder family)",
     "2pass+M"),
    ("lmerge", "mwslmerge", LMERGE_GRID, "MWS-layer + learned merging (Canto-L's builder)", "lmerge"),
)
FAMILY_OF = {k: f for k, f, *_ in FAMILIES}
PARAM_NAMES = {"acwalk": ("score_cut_cc", "walk min", "walk add"), "acwalkg": ("score_cut_cc", "walk min", "walk add"),
               "walkmerge": ("walk min", "walk add", "merging rms"), "mwslayermerge": ("s0",),
               "mws2passmerge": ("s1", "s2"), "mwslmerge": ("tau",)}
EXT_PREFIX = "trackbuild_val_fb3_ext"
# acorn's walk (build_roads) enumerates every road whose branches pass walk add; at walk add 0.1 one ttbar val event
# needs more than 6 GB (acorn (ttbar) seed 0) and every step-1 job ran out of memory: the computable range of walk add
# ends at 0.2 (docs/feedback/round3_A2_decisions.md, 10:32 UTC)
ACWALK_MIN_ADD = 0.2
# the 1-2-5 sequence of round 1's CCandWalk extension (0.01, 0.005, 0.002, ..., 0.0001, then 0) and above it
SEQ125 = tuple(sorted({round(m * 10.0 ** e, 12) for e in range(-4, 1) for m in (1, 2, 5)}))


def next_values(values, edge: str, n: int = 2, hi: float = 1.0) -> list[float]:
    """The next n values beyond the edge ("lowest" / "highest") of a parameter whose grid values are `values`: the
    grid's own linear step where its values next to the edge are evenly spaced (two values: their difference), else
    the 1-2-5 sequence; downwards a linear step that would reach 0 switches to the 1-2-5 sequence below the edge
    value, which ends at 0 (the range end, as round 1's CCandWalk extension); upwards the values stop at hi (the range
    end, included once)."""
    v = sorted({round(float(x), 12) for x in values})
    near = v[:3] if edge == "lowest" else v[-3:]
    steps = np.diff(near)
    linear = len(near) >= 2 and bool(np.allclose(steps, steps[0], rtol=0, atol=1e-9))
    step = float(steps[0]) if linear else None
    out, x = [], (v[0] if edge == "lowest" else v[-1])
    while len(out) < n:
        if edge == "lowest":
            if x <= 0.0:
                break
            if linear and x - step > 1e-12:
                x = round(x - step, 12)
            else:
                linear = False
                below = [s for s in SEQ125 if s < x - 1e-15]
                x = below[-1] if below else 0.0
        else:
            if x >= hi - 1e-12:
                break
            if linear:
                x = round(x + step, 12)
            else:
                above = [s for s in SEQ125 if s > x + 1e-15]
                x = above[0] if above else hi
            x = min(x, hi)
        out.append(x)
    return out


def constrain(fam: str, cfg: tuple) -> tuple | None:
    """A configuration within the family's range: walk add below walk min is walk add = walk min (every outward edge
    above min may branch: the range end), acorn's walk min below its CC cut is the cut (the walk never sees the removed
    edges); two-pass needs s2 < s1; thresholds in [0, 1]. None if outside the range."""
    p = [float(x) for x in cfg[1:]]
    if any(x < -1e-12 or x > 1.0 + 1e-12 for x in p):
        return None
    if fam == "walkmerge":
        if p[0] <= 0.0 or p[0] >= 1.0:
            return None
        p[1] = max(p[1], p[0])
    elif fam in ("acwalk", "acwalkg"):
        p[1] = max(p[1], p[0])
        p[2] = max(p[2], p[1])
        if p[2] < ACWALK_MIN_ADD - 1e-12:            # acorn's road enumeration is not computable below (decisions)
            return None
    elif fam == "mwslayermerge":
        if p[0] <= 0.0 or p[0] >= 1.0:
            return None
    elif fam == "mws2passmerge":
        if not (0.0 < p[1] < p[0] < 1.0):
            return None
    return (fam, *[round(x, 12) for x in p])


def edge_flags(cfgs: list, fam: str, values: list[float]) -> list[tuple[str, float, str, int]]:
    """(parameter, value, 'lowest' | 'highest', index) for every parameter whose chosen value is the lowest / highest
    value of that parameter among the family's configurations (one value in the grid: fixed, not an edge); round 2's
    edge check (hnl.eval.strong_baselines_armiii.edge_flags) with the parameter index."""
    names = PARAM_NAMES.get(fam, tuple(f"p{j + 1}" for j in range(len(values))))
    rows = [c[1:] for c in cfgs if c[0] == fam]
    out = []
    for j, v in enumerate(values):
        col = sorted({float(r[j]) for r in rows})
        if len(col) < 2:
            continue
        if np.isclose(v, col[0], rtol=0, atol=1e-12):
            out.append((names[j], float(v), "lowest", j))
        elif np.isclose(v, col[-1], rtol=0, atol=1e-12):
            out.append((names[j], float(v), "highest", j))
    return out


def extension(cfgs: list, fam: str, edges: list[tuple[str, float, str, int]], n: int = 2) -> list[tuple]:
    """New configurations past the given edges of one family: every configuration of the edge slice (the parameter at
    its edge value) with that parameter set to each of the next n values (next_values), within the family's range
    (constrain); configurations already in the grid are dropped. An empty list: the range ends at this edge."""
    have = {tuple([c[0], *[round(float(x), 12) for x in c[1:]]]) for c in cfgs if c[0] == fam}
    rows = [c for c in cfgs if c[0] == fam]
    out: list[tuple] = []
    for _, v, where, j in edges:
        col = [float(r[1 + j]) for r in rows]
        for x in next_values(col, where, n):
            for c in rows:
                if not np.isclose(float(c[1 + j]), v, rtol=0, atol=1e-12):
                    continue
                new = list(c)
                new[1 + j] = x
                k = constrain(fam, tuple(new))
                if k is not None and k not in have and k not in out:
                    out.append(k)
    return out


def ext_parts(run: str) -> list[str]:
    """The extension grids of a run (RUN/trackbuild_val_fb3_ext<step>_<tag>), in step order."""
    parts = [os.path.basename(d) for d in glob.glob(os.path.join(run, EXT_PREFIX + "*"))
             if os.path.exists(os.path.join(d, "configs.json"))]
    key = lambda g: (int(re.match(re.escape(EXT_PREFIX) + r"(\d+)", g).group(1)), g)
    return sorted(parts, key=key)


def family_parts(run: str, fam: str) -> list[str]:
    """The family's base grid and every extension part that holds the family (a run's grid of this family is their
    union)."""
    base = {f: g for _, f, g, *_ in FAMILIES}[fam]
    parts = [base] if os.path.exists(os.path.join(run, base, "configs.json")) else []
    for g in ext_parts(run):
        if any(c[0] == fam for c in json.load(open(os.path.join(run, g, "configs.json")))):
            parts.append(g)
    return parts


class GridCache:
    """Curves (hnl.eval.baseline_val.curves) of every configuration of a run's grids, read once."""

    def __init__(self, info: pd.DataFrame):
        self.info, self.c = info, {}

    def get(self, run: str, grid: str):
        key = (run, grid)
        if key not in self.c:
            from hnl.eval.baseline_val import curves
            d = os.path.join(run, grid)
            cfgs = json.load(open(os.path.join(d, "configs.json")))
            pp = pd.read_parquet(os.path.join(d, "per_particle.parquet"))
            pe = pd.read_parquet(os.path.join(d, "per_event.parquet"))
            self.c[key] = (cfgs, curves(pp, pe, self.info), frozenset(pe.event_id))
        return self.c[key]


def family_choice(cache: GridCache, run: str, fam: str, fr: float, parts: list[str] | None = None) -> dict:
    """The builder rule at FR* with the one-standard-error tie rule (hnl.eval.trackbuild_select.select, tie_se) over the
    union of the family's grid parts of one run (the same choice as hnl.eval.ranking on the united grid); parts:
    the grids to use (default: the family's base grid and extension parts)."""
    from hnl.eval.trackbuild_select import select
    cfgs_all, rows, src, events = [], [], [], None
    for g in (parts if parts is not None else family_parts(run, fam)):
        cfgs, c, ev = cache.get(run, g)
        if events is not None and ev != events:
            raise SystemExit(f"{run}: {g} covers other events")
        events = ev
        idx = [i for i, x in enumerate(cfgs) if x[0] == fam]
        sub = c.loc[idx].copy()
        sub.index = np.arange(len(cfgs_all), len(cfgs_all) + len(idx))
        cfgs_all += [cfgs[i] for i in idx]
        rows.append(sub)
        src += [g] * len(idx)
    if not rows:
        raise SystemExit(f"{run}: no grid with family {fam}")
    c = pd.concat(rows)
    keys = [tuple([x[0], *[round(float(v), 12) for v in x[1:]]]) for x in cfgs_all]
    if len(set(keys)) != len(keys):
        raise SystemExit(f"{run}: a {fam} configuration appears in two grid parts")
    tab, none_ok = select(c, cfgs_all, fr, tie_se=True)
    if tab.empty:
        return {"run": run, "family": fam, "params": None, "grid": None, "edges": [], "n_configs": len(cfgs_all)}
    r = tab.iloc[0]
    k = [i for i, x in enumerate(cfgs_all) if ",".join(str(v) for v in x[1:]) == r["params"]]
    if len(k) != 1:
        raise SystemExit(f"{run}: {fam} {r['params']} found {len(k)} times")
    values = [float(v) for v in cfgs_all[k[0]][1:]]
    return {"run": run, "family": fam, "params": r["params"], "grid": src[k[0]], "eff_primary": float(r.eff_primary),
            "fake_rate_hnl": float(r.fake_rate_hnl), "n_primary": float(r.n_primary), "n_configs": len(cfgs_all),
            "edges": edge_flags(cfgs_all, fam, values), "cfgs": cfgs_all}


def runs_all() -> list[str]:
    return [os.path.join(BASE, f"arm{a}_s{s}") for a, _ in ARMS for s in SEEDS]


def inventory(families: list[str], out_dir: str | None = None, step: int | None = None, n: int = 2) -> dict:
    """Choices and edges of the given families on every acorn run's current grids (base + extension parts); with
    step: the next extension (union over the runs of every edge's new configurations), written to
    OUT_DIR/step<k>.json with the inventory table (nothing is run here)."""
    from hnl.eval.particles import particle_table
    from hnl.eval.strong_baselines import fr_star
    fr = fr_star()
    ev = np.unique(pd.read_parquet(os.path.join(BASE, "armii_s0", "eval_val", "per_event.parquet"),
                                   columns=["event_id"]).event_id)
    cache = GridCache(particle_table(ev))
    rows, new = [], {f: [] for f in families}
    for run in runs_all():
        for fam in families:
            ch = family_choice(cache, run, fam, fr)
            ext = extension(ch.get("cfgs", []), fam, ch["edges"], n) if ch["edges"] else []
            for k in ext:
                if k not in new[fam]:
                    new[fam].append(k)
            rows.append({"run": os.path.basename(run), "family": fam, "params": ch["params"], "grid": ch["grid"],
                         "eff_primary": ch.get("eff_primary"), "fake_rate_hnl": ch.get("fake_rate_hnl"),
                         "n_configs": ch["n_configs"],
                         "edges": "; ".join(f"{a} {v:g} {w}" for a, v, w, _ in ch["edges"]) or "interior",
                         "range_end": bool(ch["edges"]) and not ext,
                         "extension": ";".join(",".join([k[0], *[f"{x:g}" for x in k[1:]]]) for k in ext)})
    tab = pd.DataFrame(rows)
    plan = {"step": step, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "fr_star": fr,
            "n_values_per_edge": n, "families": families,
            "configs": {f: [list(k) for k in v] for f, v in new.items()},
            "rule": "edge extension (docs/feedback/round3_A2_decisions.md): the edge slice's configurations with the next "
                    "values of the parameter's progression, within the family's range; run on all nine acorn runs"}
    if out_dir is not None and step is not None:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"step{step}.json")
        if os.path.exists(path):
            raise SystemExit(f"{path} exists")
        tab.to_csv(os.path.join(out_dir, f"step{step}_inventory.csv"), index=False)
        with open(path, "w") as f:
            json.dump(plan, f, indent=1)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    pd.set_option("display.max_rows", 500)
    print(tab.drop(columns=["extension"]).to_string())
    for f, v in new.items():
        print(f, len(v), ";".join(",".join([k[0], *[f"{x:g}" for x in k[1:]]]) for k in v))
    return {"table": tab, "plan": plan}


EXT_GROUPS = (("w", ("walkmerge",)), ("l", ("mwslmerge",)), ("a", ("acwalk", "acwalkg")),
              ("m", ("mwslayermerge", "mws2passmerge")))
EXT_RESOURCES = {"w": (16, "32G", "01:30:00", None), "l": (16, "32G", "03:00:00", None),
                 "a": (8, "16G", "01:30:00", 4), "m": (8, "16G", "01:30:00", None)}   # cpus, mem, time, workers


def merge_model(run: str) -> str:
    return os.path.join(FB3, os.path.basename(run), "lmerge", "model.joblib")


def ext_jobs(step: int, out_dir: str) -> list[str]:
    """sbatch commands of one extension step (slurm/feedback/fb3_grid_val.sbatch), one job per run and group
    (walkthrough + merging; learned merging, with the run's merge model; acorn's CCandWalk, with half the workers for
    memory; MWS-layer and two-pass + merging), for the grids that do not exist yet; the configuration lists are written
    next to the plan."""
    plan = json.load(open(os.path.join(out_dir, f"step{step}.json")))
    cmds = []
    for tag, fams in EXT_GROUPS:
        cfgs = [tuple(c) for f in fams for c in plan["configs"].get(f, [])]
        if not cfgs:
            continue
        cf = os.path.join(out_dir, f"step{step}_{tag}.txt")
        text = config_text(cfgs)
        if os.path.exists(cf) and open(cf).read() != text:
            raise SystemExit(f"{cf} exists with other configurations")
        with open(cf, "w") as f:
            f.write(text)
        for run in runs_all():
            g = f"{EXT_PREFIX}{step}_{tag}"
            if os.path.exists(os.path.join(run, g, "per_particle.parquet")):
                continue
            mm = f" {merge_model(run)}" if tag == "l" else ""
            cpus, mem, tlim, workers = EXT_RESOURCES[tag]
            w = f"WORKERS={workers} " if workers else ""
            cmds.append(f"env -u OMP_NUM_THREADS {w}sbatch -J fb3_ext{step}{tag}_{os.path.basename(run)} -c {cpus} "
                        f"--mem={mem} -t {tlim} slurm/feedback/fb3_grid_val.sbatch {run} {g} {cf}{mm}")
    return cmds


# ------------------------------------------------------------------------------------------------- grids
def config_text(cfgs) -> str:
    return ";".join(",".join([str(c[0]), *[repr(float(x)) for x in c[1:]]]) for c in cfgs)


def parse_configs(text: str) -> list[tuple]:
    """'fam,p1,p2;fam,p1' -> [(fam, p1, p2), (fam, p1)] (floats), the configuration tuples of hnl.eval.trackbuild_val."""
    out = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.split(",")
        out.append((parts[0], *[float(x) for x in parts[1:]]))
    if len({tuple(c) for c in out}) != len(out):
        raise ValueError("a configuration is listed twice")
    return out


def run_grid(graphs: str, out: str, cfgs: list[tuple], workers: int, merge_model: str | None = None) -> dict:
    """hnl.eval.trackbuild_val on an explicit list of configurations (its own evaluation code: _chunk / evaluate_event,
    the same output files), for grid extensions whose values are not in trackbuild_val.configs."""
    import concurrent.futures as cf
    from hnl.eval import trackbuild_val as TV
    from hnl.eval.evaluate_gnn import event_truth
    if os.path.exists(os.path.join(out, "per_particle.parquet")):
        raise SystemExit(f"{out} exists: refusing to overwrite")
    if merge_model:
        os.environ["HNL_MERGE_MODEL"] = os.path.abspath(merge_model)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    paths = sorted(glob.glob(os.path.join(graphs, "*.pyg.gz")))
    eids = [os.path.basename(p).split("event")[-1].split(".")[0] for p in paths]
    pairs, keep = event_truth(eids)
    chunks = [(paths[k:k + 10], cfgs, pairs, keep) for k in range(0, len(paths), 10)]
    with cf.ProcessPoolExecutor(max_workers=max(workers, 1)) as ex:
        res = list(ex.map(TV._chunk, chunks))
    pd.concat([r[0] for r in res], ignore_index=True).to_parquet(os.path.join(out, "per_particle.parquet"), index=False)
    pd.concat([r[1] for r in res], ignore_index=True).to_parquet(os.path.join(out, "per_event.parquet"), index=False)
    with open(os.path.join(out, "configs.json"), "w") as f:
        json.dump([list(map(lambda x: x if isinstance(x, str) else float(x), c)) for c in cfgs], f)
    info = {"n_events": len(paths), "n_configs": len(cfgs), "elapsed_s": round(time.time() - t0, 1)}
    print(json.dumps(info))
    return info


# ------------------------------------------------------------------------------------------------- val
V1, V2 = "strong_baselines_v1", "strong_baselines_v2_armiii"
NAME = "strong_baselines_v3"
NEW = "Canto-L (frozen: learned merging)"
BEST = "val-best builder of the full menu (chosen per seed)"
MENU_PARTS = ("trackbuild_val_ext", "trackbuild_val_acwalk_all", LMERGE_GRID)
CANTO_GRIDS = ["trackbuild_val_ext", "trackbuild_val_lmerge", "trackbuild_val"]      # hnl.eval.final_select's grids
CANTO_B_RUNS = tuple(os.path.join(DATA, "runs", "stage1_final", "l1t", f"armi_s{s}") for s in SEEDS)
CANTO_L_RUNS = tuple(os.path.join(DATA, "runs", "stage1_final_learned", "l1tl", f"armi_s{s}") for s in SEEDS)
CANTO_FAMILIES = ("walkmerge", "mwsm", "mws2pm", "lmerge")
CC_ROWS = {  # arm -> CC rows of rounds 1-2 (labels of results/strong_baselines_v2_armiii_spec_val.json): (label, rule)
    "i": (("acorn (ttbar) + CC, preregistered cut (= best cut)", "s4"),),
    "ii": (("acorn (HNL) + CC, best cut <= FR*", "best"),
           ("acorn (HNL) + CC, preregistered cut (claims' reference)", "s4")),
    "iii": (("acorn (ttbar+HNL) + CC, best cut <= FR*", "best"), ("acorn (ttbar+HNL) + CC, preregistered cut rule", "s4")),
}
ALAB = dict(ARMS)
BLAB = {k: blab for k, _, _, blab, _ in FAMILIES}
BSHORT = {k: s for k, _, _, _, s in FAMILIES}
CANTO_B_CC = "Canto-B + CC, best cut <= FR*"
CANTO_B_FROZEN = "Canto-B (frozen: learned merging)"
TOL = 1e-12


def family_label(arm: str, key: str) -> str:
    return f"{ALAB[arm]} + {BEST}" if key == "best" else f"{ALAB[arm]} + {BLAB[key]}"


def canto_label(which: str, key: str) -> str:
    if which == "B":
        return CANTO_B_CC if key == "cc" else (CANTO_B_FROZEN if key == "lmerge" else f"Canto-B + {BLAB[key]}")
    return NEW if key == "lmerge" else f"Canto-L + {BLAB[key]}, same builder (not the frozen builder)"


def menu_choice(run: str, fr: float, info: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """The val-best configuration of one run over the full menu: each family's choice on the united grid (builder rule
    at FR* with the one-standard-error tie rule, hnl.eval.ranking) and CC at its best cut <= FR* (hnl.eval.ranking's
    'best' rule; not a candidate when FR* is unreachable), then the same tie rule over these candidates
    (hnl.eval.trackbuild_select.tie_best: within one binomial standard error of the highest primary efficiency, the
    lowest fake rate). Returns the winning spec entry and the candidate table."""
    cands = []
    for key, fam, *_ in FAMILIES:
        try:
            _, row, wp, _ = RK._grid({"run": run, "source": [ALL_GRID], "family": fam}, fr, info, tie_se=True)
        except ValueError:
            continue
        cands.append({"run": run, "key": key, "family": fam, "working_point": wp, "eff_primary": float(row.eff_primary),
                      "fake_rate_hnl": float(row.fake_rate_hnl), "n_primary": float(row.n_primary),
                      "entry": {"run": run, "source": [ALL_GRID], "family": fam}})
    _, row, wp, _ = RK._cc({"run": run, "rule": "best"}, fr, info)
    cands.append({"run": run, "key": "cc", "family": "cc", "working_point": wp, "eff_primary": float(row.eff_primary),
                  "fake_rate_hnl": float(row.fake_rate_hnl), "n_primary": float(row.n_primary),
                  "entry": {"run": run, "source": "eval_val", "rule": "best"}})
    t = menu_tie(pd.DataFrame(cands), fr)
    i = int(np.flatnonzero(t.chosen.to_numpy())[0])
    return cands[i]["entry"], t.drop(columns="entry")


def menu_tie(t: pd.DataFrame, fr: float) -> pd.DataFrame:
    """The one-standard-error rule over candidate rows (eff_primary, fake_rate_hnl, n_primary): eligible = fake rate
    <= FR*; tied = eligible and within SE = sqrt(e_max (1 - e_max) / n) of the highest efficiency; chosen = the tied
    row with the lowest fake rate, then the higher efficiency (hnl.eval.trackbuild_select.tie_best)."""
    from hnl.eval.trackbuild_select import tie_best
    t = t.reset_index(drop=True).copy()
    t["eligible"] = t.fake_rate_hnl <= fr + 1e-12
    ok = t[t.eligible]
    if ok.empty:
        raise ValueError("no candidate with fake rate <= FR*")
    n = float(ok.loc[ok.eff_primary.idxmax(), "n_primary"])
    i = tie_best(ok, n)
    e_max = float(ok.eff_primary.max())
    se = float(np.sqrt(e_max * (1 - e_max) / n))
    t["tied"] = t.eligible & (t.eff_primary >= e_max - se - 1e-12)
    t["chosen"] = t.index == i
    return t


def build_val_spec(fr: float, info: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    """The val spec: Canto-L (frozen; round 1), per arm the CC rows of rounds 1-2, the six family rows on the united
    grid and the val-best row of the full menu (per seed); the Canto-B rows of the decomposition (CC at its best cut,
    the four families on Canto-B's own grids: the per-seed configurations of the frozen selection's table) and Canto-L
    with the other families (same-builder rows)."""
    v2 = {m["label"]: m for m in json.load(open(_repo(f"results/{V2}_spec_val.json")))}
    spec = [{**v2[NEW], "group": "reference", "kind": "canto"}]
    cands = []
    for arm, alab in ARMS:
        runs = [os.path.join(BASE, f"arm{arm}_s{s}") for s in SEEDS]
        for lab, rule in CC_ROWS[arm]:
            m = {k: v for k, v in v2[lab].items() if k not in ("cc_rule", "prefix", "key")}
            if [e.get("rule") for e in m["entries"]] != [rule] * 3:
                raise SystemExit(f"{lab}: rules {[e.get('rule') for e in m['entries']]} != {rule}")
            spec.append({**m, "group": alab, "arm": arm, "kind": "cc", "cc_rule": rule})
        for key, fam, _, blab, bshort in FAMILIES:
            spec.append({"label": family_label(arm, key), "short": f"{alab[6:]} {bshort}", "group": alab, "arm": arm,
                         "key": key, "kind": "family",
                         "entries": [{"run": r, "source": [ALL_GRID], "family": fam} for r in runs]})
        ents = []
        for r in runs:
            e, t = menu_choice(r, fr, info)
            ents.append(e)
            cands.append(t.assign(arm=arm, seed=r[-1]))
        spec.append({"label": family_label(arm, "best"), "short": f"{alab[6:]} val-best", "group": alab, "arm": arm,
                     "key": "best", "kind": "best", "entries": ents})
    spec.append({"label": CANTO_B_CC, "short": "Canto-B CC*", "group": "decomposition", "kind": "canto", "key": "cc",
                 "entries": [{"run": r, "source": "eval_val", "rule": "best"} for r in CANTO_B_RUNS]})
    for key in CANTO_FAMILIES:
        spec.append({"label": canto_label("B", key), "short": f"Canto-B {BSHORT[key]}", "group": "decomposition",
                     "kind": "canto", "key": key,
                     "entries": [{"run": r, "source": CANTO_GRIDS, "family": FAMILY_OF[key]} for r in CANTO_B_RUNS]})
    for key in CANTO_FAMILIES[:-1]:
        spec.append({"label": canto_label("L", key), "short": f"Canto-L {BSHORT[key]}", "group": "Canto-L, same builder",
                     "kind": "canto", "key": key,
                     "entries": [{"run": r, "source": CANTO_GRIDS, "family": FAMILY_OF[key]} for r in CANTO_L_RUNS]})
    return spec, pd.concat(cands, ignore_index=True)


def _cmp(checks: list, what: str, mine, ref, tol: float = TOL) -> bool:
    if isinstance(ref, str) or isinstance(mine, str):
        ok = str(mine) == str(ref)
    else:
        ok = abs(float(mine) - float(ref)) <= tol
    checks.append((what, mine, ref, ok))
    return ok


def _fp(wp: str) -> tuple[str, str]:
    """(family, params) of a working point 'fam params (grid)' or ('CC', cut)."""
    m = re.match(r"^CC (\d+\.\d+)", wp)
    if m:
        return "CC", m.group(1)
    f, p, _ = SB._parse_wp(wp)
    return f, p


def val_checks(tab: pd.DataFrame, ent: pd.DataFrame) -> tuple[list, pd.DataFrame]:
    """Identity checks of the val ranking against rounds 1-2, ranking v3 and the frozen selection's table; and the list
    of family choices that changed from rounds 1-2 (not failures: the menu and grids are larger)."""
    checks: list = []
    ref_t = pd.concat([pd.read_csv(_repo(f"results/{V1}_val.csv")), pd.read_csv(_repo(f"results/{V2}_val.csv"))])
    ref_t = ref_t.drop_duplicates("label").set_index("label")
    ref_e = pd.concat([pd.read_csv(_repo(f"results/{V1}_val_entries.csv")),
                       pd.read_csv(_repo(f"results/{V2}_val_entries.csv"))]).drop_duplicates(["label", "run"])
    t = tab.set_index("label")
    changes = []
    for lab in t.index:
        if lab not in ref_t.index:
            continue
        mine_e = ent[ent.label == lab].reset_index(drop=True)
        old_e = ref_e[ref_e.label == lab].set_index("run")
        if not set(mine_e.run) <= set(old_e.index):
            continue
        same = all(_fp(r.working_point) == _fp(old_e.loc[r.run, "working_point"]) for _, r in mine_e.iterrows())
        for _, r in mine_e.iterrows():
            o = old_e.loc[r.run]
            if _fp(r.working_point) == _fp(o.working_point):
                for k in RK.KEYS:
                    _cmp(checks, f"rounds 1-2 val entry {lab} {os.path.basename(r.run)} {k}", float(r[k]), float(o[k]))
            else:
                changes.append({"label": lab, "run": os.path.basename(r.run), "old": o.working_point,
                                "new": r.working_point, "old_eff": float(o.eff_primary), "new_eff": float(r.eff_primary),
                                "old_fake": float(o.fake_rate_hnl), "new_fake": float(r.fake_rate_hnl)})
        if same:
            for k in RK.KEYS:
                _cmp(checks, f"rounds 1-2 val row {lab} {k}", float(t.loc[lab, k]), float(ref_t.loc[lab, k]))
                _cmp(checks, f"rounds 1-2 val row {lab} {k}_sd", float(t.loc[lab, k + "_sd"]),
                     float(ref_t.loc[lab, k + "_sd"]))
            for c in ("k_n_d0_100", "k_n_d0_300"):
                _cmp(checks, f"rounds 1-2 val row {lab} {c}", str(t.loc[lab, c]), str(ref_t.loc[lab, c]))
    # Canto-B and Canto-L family rows: the frozen selection's per-seed table (configuration and numbers)
    ss = pd.read_csv(_repo("results/final_selection_v1_seeds.csv"))
    for which, pipe, runs in (("B", "l1t", CANTO_B_RUNS), ("L", "l1tl", CANTO_L_RUNS)):
        for key in CANTO_FAMILIES:
            lab = canto_label(which, key)
            if lab not in t.index:
                continue
            for s, run in zip(SEEDS, runs):
                r = ent[(ent.label == lab) & (ent.run == run)].iloc[0]
                o = ss[(ss.pipeline == pipe) & (ss.seed == int(s)) & (ss.family == FAMILY_OF[key])].iloc[0]
                f, p = _fp(r.working_point)
                _cmp(checks, f"final_selection_v1_seeds {lab} s{s} configuration", f"{f},{p}", o.config)
                for k in RK.KEYS:
                    _cmp(checks, f"final_selection_v1_seeds {lab} s{s} {k}", float(r[k]), float(o[k]))
    v3 = pd.read_csv(_repo("results/ranking/ranking_val_v3.csv")).set_index("label")
    r3 = v3.loc["Canto-B (full budget): base graph, learned merging (rule R6)"]
    for k in RK.KEYS:
        _cmp(checks, f"ranking v3 Canto-B row {k}", float(t.loc[CANTO_B_FROZEN, k]), float(r3[k]))
    wp = json.load(open(_repo(RT.WP)))
    for arm in ("i", "ii", "iii"):
        for lab, rule in CC_ROWS[arm]:
            if rule != "s4":
                continue
            for _, r in ent[ent.label == lab].iterrows():
                cut = float(re.match(r"CC (\d+\.\d+)", r.working_point).group(1))
                _cmp(checks, f"preregistered cut {os.path.basename(r.run)}", cut,
                     float(wp["cuts_at_FR_star"][arm][r.run[-1]]), 1e-9)
    return checks, pd.DataFrame(changes)


def determinism_checks(checks: list) -> None:
    """Configurations computed twice (the extension against an older grid of the same run holding the same
    configuration, e.g. MWS-layer + merging 0.05 in RUN/trackbuild_val_mwslayer): identical per-particle and per-event
    rows required."""
    for run in runs_all():
        for g in ext_parts(run):
            cfgs = json.load(open(os.path.join(run, g, "configs.json")))
            for old in ("trackbuild_val_mwslayer", "trackbuild_val_2pass", "trackbuild_val_lmerge"):
                p = os.path.join(run, old, "configs.json")
                if not os.path.exists(p):
                    continue
                oc = json.load(open(p))
                if old == "trackbuild_val_lmerge":
                    oc = [c for c in oc if c[0] != "mwslmerge"]          # another merge model there
                for i, c in enumerate(cfgs):
                    k = [j for j, x in enumerate(oc) if x[0] == c[0] and np.allclose(x[1:], c[1:], rtol=0, atol=1e-12)]
                    if not k:
                        continue
                    for f, key in (("per_particle.parquet", ["event_id", "particle_id"]), ("per_event.parquet",
                                                                                           ["event_id"])):
                        a = pd.read_parquet(os.path.join(run, g, f), filters=[("config", "==", i)])
                        b = pd.read_parquet(os.path.join(run, old, f), filters=[("config", "==", k[0])])
                        a = a.drop(columns="config").sort_values(key).reset_index(drop=True)
                        b = b.drop(columns="config").sort_values(key).reset_index(drop=True)
                        ok = a.shape == b.shape and bool((a.values == b.values).all())
                        checks.append((f"determinism {os.path.basename(run)} {g} {c} = {old} [{f}]", len(a), len(b), ok))


def edges_table(info: pd.DataFrame, fr: float) -> pd.DataFrame:
    """Final grid position of every family choice on the united grid (RUN/trackbuild_val_fb3_all), with the extension
    that would come next (empty: interior, or the range ends)."""
    cache = GridCache(info)
    rows = []
    for run in runs_all():
        for key, fam, *_ in FAMILIES:
            ch = family_choice(cache, run, fam, fr, parts=[ALL_GRID])
            ext = extension(ch.get("cfgs", []), fam, ch["edges"]) if ch["edges"] else []
            n_steps = len({re.match(re.escape(EXT_PREFIX) + r"(\d+)", g).group(1) for g in ext_parts(run)
                           if any(c[0] == fam for c in json.load(open(os.path.join(run, g, "configs.json"))))})
            rows.append({"run": os.path.basename(run), "family": fam, "label_key": key, "params": ch["params"],
                         "eff_primary": ch.get("eff_primary"), "fake_rate_hnl": ch.get("fake_rate_hnl"),
                         "n_family_configs": ch["n_configs"], "extension_steps": n_steps,
                         "edges": "; ".join(f"{a} {v:g} {w}" for a, v, w, _ in ch["edges"]) or "interior",
                         "status": "interior" if not ch["edges"] else ("range end" if not ext else "EDGE: extend"),
                         "next_extension": config_text(ext)})
    return pd.DataFrame(rows)


def val_models(spec: list[dict], ent: pd.DataFrame):
    """Bootstrap models (hnl.eval.bootstrap_tables, val tables B = 1000, seed 20261005) of every spec row at its val
    working points."""
    from hnl.eval import bootstrap_tables as BT
    rs = BT.Resampler.build(BT.VAL_SETS, BT.VAL_CODE, os.path.join(BT.OUT_RUNS, "tables", f"val_b{BT.B_VAL}"),
                            BT.B_VAL, BT.SEED_VAL, f"val tables (B = {BT.B_VAL}, seed {BT.SEED_VAL})")
    ld = BT.ValLoader(rs)
    models = {}
    for m in spec:
        runs = [e["run"] for e in m["entries"]]
        e = ent[ent.label == m["label"]].set_index("run")
        wps = [e.loc[r, "working_point"] for r in runs]
        models[m["label"]] = BT.Model(m["label"], m.get("short", ""), [ld.seed(r, w) for r, w in zip(runs, wps)], rs,
                                      wps, runs, m.get("group", ""))
    return models


VAL_Q = SB.VAL_Q


def pairs(labels: list[str]) -> list[tuple[str, str, str]]:
    """(new, baseline, block): Canto-L minus every acorn row; the val-best rows against each other; the decomposition
    on acorn (ttbar) scores in both orders (builder first: CC -> each family -> Canto-B with that family; classifier
    first: CC -> Canto-B CC -> Canto-B with each family)."""
    have = set(labels)
    acorn = [x for x in labels if x.startswith("acorn (")]
    out = [(NEW, x, "Canto-L minus acorn") for x in acorn]
    b = {a: f"{alab} + {BEST}" for a, alab in (("i", "acorn (ttbar)"), ("ii", "acorn (HNL)"),
                                               ("iii", "acorn (ttbar+HNL)"))}
    for x, y in (("iii", "ii"), ("iii", "i"), ("ii", "i")):
        if b[x] in have and b[y] in have:
            out.append((b[x], b[y], "val-best against val-best"))
    if "i" not in ALAB:
        return [(a, c, k) for a, c, k in out if a in have and c in have]
    cc = CC_ROWS["i"][0][0]
    for key, *_ in FAMILIES:
        out.append((family_label("i", key), cc, "decomposition, builder first: builder on acorn (ttbar) scores"))
    out.append((family_label("i", "best"), cc, "decomposition, builder first: builder on acorn (ttbar) scores"))
    for key in CANTO_FAMILIES:
        if canto_label("B", key) in have:
            out.append((canto_label("B", key), family_label("i", key),
                        "decomposition, builder first: then the Canto-B classifier"))
    if CANTO_B_CC in have:
        out.append((CANTO_B_CC, cc, "decomposition, classifier first: Canto-B classifier with CC"))
        for key in CANTO_FAMILIES:
            if canto_label("B", key) in have:
                out.append((canto_label("B", key), CANTO_B_CC, "decomposition, classifier first: then the builder"))
    for key in CANTO_FAMILIES[:-1]:
        if canto_label("L", key) in have:
            out.append((canto_label("L", key), family_label("ii", key), "same builder: Canto-L minus acorn (HNL)"))
    return [(a, c, k) for a, c, k in out if a in have and c in have]


def val_differences(models: dict, prs: list[tuple[str, str, str]]) -> pd.DataFrame:
    """new minus baseline on val (paired event bootstrap of the val tables + both seed terms, as rounds 1-2); where both
    rows are the same trained networks also the seed term paired by seed (hnl.eval.bootstrap_tables.difference)."""
    from hnl.eval import bootstrap_tables as BT
    rows = []
    for a, b, blk in prs:
        new, base = models[a], models[b]
        paired = [os.path.normpath(r) for r in new.runs] == [os.path.normpath(r) for r in base.runs]
        for q, bq in VAL_Q:
            d = BT.difference(new, base, bq, paired_seeds=paired)
            sn, sb = BT.summary(new, bq), BT.summary(base, bq)
            rows.append({"set": "val", "block": blk, "new": a, "baseline": b, "quantity": q,
                         "n_per_seed": sb.get("n_per_seed"), "value_new": sn["value"], "sigma_new": sn["combined"],
                         "s_seeds_new": sn["seed_sd"], "value_base": sb["value"], "sigma_base": sb["combined"],
                         "s_seeds_base": sb["seed_sd"], "difference": d["value"], "sigma_boot": d["boot_sd"],
                         "sigma_seeds": d["seed_sd"], "sigma": d["combined"], "z": d["z"], "B": d["B"],
                         "seed_term_paired": d.get("seed_term_paired", np.nan),
                         "sigma_paired": d.get("combined_paired", np.nan), "per_seed": d.get("per_seed", "")})
    return pd.DataFrame(rows)


def _seed(run: str) -> str:
    return re.search(r"_s(\d+)$", run).group(1)


def existing_test_outputs() -> dict:
    """(run, configuration) -> test output directory of rounds 1-2 (their val-fixed specs; provenance checked when the
    test summary is built)."""
    out = {}
    for f in (f"results/{V1}_spec_test.json", f"results/{V2}_spec_test.json"):
        for m in json.load(open(_repo(f)))["models"]:
            for e in m["entries"]:
                if "config" in e:
                    out[(os.path.normpath(e["run"]), e["config"])] = {"sub": e["sub"], "label": m["label"], "spec": f}
    return out


def test_spec_from_val(spec: list[dict], ent: pd.DataFrame) -> list[dict]:
    """The test entries of every row, fixed by its val working points: a configuration evaluated on test in rounds 1-2
    (same run, same configuration) is reused, any other is a new output under runs/feedback_v3/strong_baselines_a2;
    learned merging carries the run's merge model; CC rows as rounds 1-2 (frozen eval_test or their post-freeze
    outputs); Canto-B CC at its best cut: a new post-freeze CC evaluation of Canto-B's stored test scores."""
    old = existing_test_outputs()
    v2t = {m["label"]: m for m in json.load(open(_repo(f"results/{V2}_spec_test.json")))["models"]}
    ref = {m["label"]: m for m in json.load(open(_repo("results/ranking/spec_test_v1.json")))}
    v3_of = {"acorn (HNL) + CC, best cut <= FR*": "acorn IGNN, HNL-trained + CC (best cut <= FR*)",
             "acorn (HNL) + CC, preregistered cut (claims' reference)": "acorn IGNN, HNL-trained + CC (prereg. cut rule)",
             "acorn (ttbar) + CC, preregistered cut (= best cut)": "acorn IGNN, ttbar-trained + CC (prereg. cut rule)",
             NEW: "Canto-L (full budget): learned triplet union, learned merging (rule R6)",
             CANTO_B_FROZEN: "Canto-B (full budget): base graph, learned merging (rule R6)"}
    by_label = {}
    tspec = []
    for m in spec:
        lab = m["label"]
        es = []
        if lab in v3_of:
            for e in ref[v3_of[lab]]["entries"]:
                es.append({k: e[k] for k in ("run", "sub", "cut", "wp")} | {"source": "results/ranking/spec_test_v1.json"})
        elif lab in v2t and m.get("kind") == "cc":
            for e in v2t[lab]["entries"]:
                es.append({k: e[k] for k in ("run", "sub", "cut", "wp")} | {"source": f"results/{V2}_spec_test.json"})
        elif lab == CANTO_B_CC:
            for e in m["entries"]:
                r = ent[(ent.label == lab) & (ent.run == e["run"])].iloc[0]
                if "unreachable" in r.working_point:
                    raise SystemExit(f"{lab} {e['run']}: FR* unreachable on val")
                cut = float(re.match(r"CC (\d+\.\d+)", r.working_point).group(1))
                es.append({"run": e["run"], "sub": os.path.join(FB3, f"canto_b_s{_seed(e['run'])}", "eval_test_bestcut"),
                           "cut": cut, "val_wp": r.working_point,
                           "wp": RT._wp_text(f"CC {cut:.2f}", ["post-freeze CC on the stored test scores"])})
        elif m.get("kind") in ("family", "best"):
            for e in m["entries"]:
                r = ent[(ent.label == lab) & (ent.run == e["run"])].iloc[0]
                if r.working_point.startswith("CC "):                      # val-best = CC at its best cut
                    cc_lab = [x for x, rule in CC_ROWS[m["arm"]] if rule == "best" or len(CC_ROWS[m["arm"]]) == 1][0]
                    ce = [x for x in by_label[cc_lab] if x["run"] == e["run"]][0]
                    es.append(dict(ce))
                    continue
                fam, params, grid = SB._parse_wp(r.working_point)
                cfg = f"{fam},{params}"
                key = [k for k, f, *_ in FAMILIES if f == fam][0]
                prev = old.get((os.path.normpath(e["run"]), cfg))
                sub = prev["sub"] if prev else os.path.join(FB3, os.path.basename(e["run"]), f"{key}_test")
                x = {"run": e["run"], "sub": sub, "cut": 0, "config": cfg, "val_grid": grid, "val_wp": r.working_point,
                     "wp": f"{fam} {params} (chosen on val in {grid}; test configuration 0)",
                     "reused": prev["spec"] if prev else None}
                if fam == "mwslmerge":
                    x["merge_model"] = merge_model(e["run"])
                    x["merge_model_sha256"] = _sha256(x["merge_model"])
                es.append(x)
        else:
            continue                                                    # val-only rows (Canto-B / Canto-L families)
        by_label[lab] = es
        tspec.append({k: m[k] for k in ("label", "short", "group") if k in m}
                     | {k: m[k] for k in ("arm", "key", "kind", "cc_rule") if k in m} | {"entries": es})
    # one output per run and configuration: rows that share a configuration share its output
    seen = {}
    for m in tspec:
        for e in m["entries"]:
            if "config" in e:
                k = (os.path.normpath(e["run"]), e["config"])
                if k in seen and seen[k] != e["sub"]:
                    raise SystemExit(f"{k}: two test outputs {seen[k]} and {e['sub']}")
                seen[k] = e["sub"]
    subs = {}
    for k, s in seen.items():
        if s in subs and subs[s] != k:
            raise SystemExit(f"{s}: one output for two configurations {subs[s]} and {k}")
        subs[s] = k
    return tspec


def run_val(out: str, name: str) -> None:
    from hnl.eval.particles import particle_table
    fr = SB.fr_star()
    ev = np.unique(pd.read_parquet(os.path.join(BASE, "armii_s0", "eval_val", "per_event.parquet"),
                                   columns=["event_id"]).event_id)
    info = particle_table(ev)
    for run in runs_all():
        if not os.path.exists(os.path.join(run, ALL_GRID, "configs.json")):
            raise SystemExit(f"{run}: no {ALL_GRID} (run `merge` after the last extension step)")
    spec, cands = build_val_spec(fr, info)
    tab, ent = RK.rank(spec, fr, tie_se=True)
    checks, changes = val_checks(tab, ent)
    determinism_checks(checks)
    # the per-seed val-best entries agree with the family rows they name
    for m in spec:
        if m.get("kind") != "best":
            continue
        for e in m["entries"]:
            mine = ent[(ent.label == m["label"]) & (ent.run == e["run"])].iloc[0]
            if e["source"] == "eval_val":
                other = ent[(ent.label == [x for x, rl in CC_ROWS[m["arm"]] if rl == "best" or
                                           len(CC_ROWS[m["arm"]]) == 1][0]) & (ent.run == e["run"])].iloc[0]
            else:
                key = [k for k, f, *_ in FAMILIES if f == e["family"]][0]
                other = ent[(ent.label == family_label(m["arm"], key)) & (ent.run == e["run"])].iloc[0]
            _cmp(checks, f"val-best entry = its family row {os.path.basename(e['run'])}", mine.working_point,
                 other.working_point)
            for k in RK.KEYS:
                _cmp(checks, f"val-best entry = its family row {os.path.basename(e['run'])} {k}", float(mine[k]),
                     float(other[k]))
    edges = edges_table(info, fr)
    # round 2's rule for comparison: the arm's row with the highest three-seed-mean primary efficiency (ties: lower
    # fake rate) among its family and CC rows (hnl.eval.strong_baselines_armiii.val_best)
    from hnl.eval.strong_baselines_armiii import val_best
    r2 = {}
    for arm, alab in ARMS:
        rows = tab[tab.label.str.startswith(alab + " + ") & (tab.label != family_label(arm, "best"))]
        r2[arm] = val_best(rows, alab)
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(checks, columns=["check", "ours", "reference", "ok"]).to_csv(
        os.path.join(out, f"{name}_val_checks.csv"), index=False)
    bad = [c for c in checks if not c[3]]
    if bad:
        for c in bad[:30]:
            print("FAIL", c)
        raise SystemExit(f"{len(bad)} of {len(checks)} val checks failed")
    print(f"val checks: {len(checks)} of {len(checks)} ok")
    if (edges.status == "EDGE: extend").any():
        print(edges[edges.status == "EDGE: extend"].to_string())
        raise SystemExit("a family choice is still at a grid edge: run the next extension step first")
    tspec = test_spec_from_val(spec, ent)
    models = val_models(spec, ent)
    prs = pairs(list(tab.label))
    vd = val_differences(models, prs)
    # round 1's and round 2's val differences of unchanged pairs reproduced (every column, 1e-12)
    for f in (f"results/{V1}_val_differences.csv", f"results/{V2}_val_differences.csv"):
        ref = pd.read_csv(_repo(f))
        for (a, b), g in vd.groupby(["new", "baseline"], sort=False):
            r = ref[(ref.new == a) & (ref.baseline == b)]
            if r.empty or a in set(changes.get("label", [])) or b in set(changes.get("label", [])):
                continue
            r = r.drop_duplicates("quantity").set_index("quantity")
            for _, x in g.iterrows():
                for c in ("value_new", "value_base", "difference", "sigma_boot", "sigma_seeds", "sigma"):
                    _cmp(checks, f"{f}: {a} - {b} {x.quantity} {c}", float(x[c]), float(r.loc[x.quantity, c]))
    pd.DataFrame(checks, columns=["check", "ours", "reference", "ok"]).to_csv(
        os.path.join(out, f"{name}_val_checks.csv"), index=False)
    bad = [c for c in checks if not c[3]]
    if bad:
        for c in bad[:30]:
            print("FAIL", c)
        raise SystemExit(f"{len(bad)} of {len(checks)} val checks failed (differences)")
    print(f"val checks incl. rounds 1-2 differences: {len(checks)} of {len(checks)} ok")
    spec_file = os.path.join(out, f"{name}_spec_test.json")
    json.dump(spec, open(os.path.join(out, f"{name}_spec_val.json"), "w"), indent=1)
    json.dump({"fr_star": fr, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "rule": "per run and family: builder rule at FR* with the one-standard-error tie rule (hnl.eval.ranking, "
                       "tie_se) on the united grid RUN/trackbuild_val_fb3_all; val-best: the same rule over the "
                       "families' choices and CC at its best cut <= FR* (menu_choice); CC rows: rounds 1-2",
               "round2_rule_best": r2, "models": tspec}, open(spec_file, "w"), indent=1)
    tab.insert(0, "set", "val")
    ent.insert(0, "set", "val")
    tab.to_csv(os.path.join(out, f"{name}_val.csv"), index=False)
    ent.to_csv(os.path.join(out, f"{name}_val_entries.csv"), index=False)
    vd.to_csv(os.path.join(out, f"{name}_val_differences.csv"), index=False)
    edges.to_csv(os.path.join(out, f"{name}_edges.csv"), index=False)
    cands.to_csv(os.path.join(out, f"{name}_menu_candidates.csv"), index=False)
    changes.to_csv(os.path.join(out, f"{name}_val_changes.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 100)
    pd.set_option("display.max_rows", 500)
    print(RK.markdown(tab.drop(columns="set"), fr))
    print(cands.to_string())
    print(changes.to_string())
    print("round 2's rule (highest three-seed mean):", r2)
    p = vd[vd.quantity == "primary"]
    for _, r in p.iterrows():
        print(f"[{r.block}] {r.new} - {r.baseline}: {100 * r.difference:+.2f} +- {100 * r.sigma:.2f} (z {r.z:+.1f})")


# ------------------------------------------------------------------------------------------------- test jobs
def test_jobs(out: str, name: str, cpus: int = 8, mem: str = "16G", tlim: str = "02:00:00") -> list[str]:
    """sbatch commands (one per run and configuration) for the test outputs that do not exist yet, and the post-freeze
    CC job of Canto-B at its best cut."""
    spec_file = os.path.abspath(os.path.join(out, f"{name}_spec_test.json"))
    ts = json.load(open(spec_file))["models"]
    cmds, done, bestcut = [], set(), []
    for m in ts:
        for e in m["entries"]:
            if "config" in e and not e.get("reused"):
                if e["sub"] in done or os.path.exists(os.path.join(e["sub"], "provenance.json")):
                    continue
                done.add(e["sub"])
                mm = f"MM={e['merge_model']} " if e.get("merge_model") else ""
                if e["config"].startswith("acwalk"):           # acorn's CCandWalk: more memory per worker
                    mm += f"WORKERS={cpus // 2} "
                nm = os.path.basename(e["sub"])
                cmds.append(f"env -u OMP_NUM_THREADS SPEC={spec_file} {mm}sbatch -J fb3_test_{nm}_"
                            f"{os.path.basename(e['run'])} -c {cpus} --mem={mem} -t {tlim} slurm/feedback/fb3_test.sbatch "
                            f"{e['run']} {os.path.dirname(e['sub'])} {nm}={e['config']}")
            elif m["label"] == CANTO_B_CC and not os.path.exists(os.path.join(e["sub"], "provenance.json")):
                bestcut.append(_seed(e["run"]))
    if bestcut:
        cmds.append(f"env -u OMP_NUM_THREADS sbatch -J fb3_bestcut_canto_b -p submit,submit-gpu --exclude={EXCLUDE} "
                    f"-c 8 --mem=16G -t 01:00:00 slurm/run_py.sbatch hnl.eval.strong_baselines_v3 bestcut "
                    f"--seeds {','.join(sorted(set(bestcut)))} --out {os.path.abspath(out)} --name {name}")
    return cmds


def cmd_bestcut(seeds: list[str], out: str, name: str) -> None:
    """CC of Canto-B at its val-best cut <= FR* on its stored test scores (RUN/gnn_scores/testset.tar; post-freeze,
    descriptive, once), with its frozen e* cut alongside as an identity check against the frozen RUN/eval_test."""
    import shutil
    from hnl.eval import evaluate_gnn as EG
    from hnl.eval.published_defs import _same_rows
    spec_file = os.path.abspath(os.path.join(out, f"{name}_spec_test.json"))
    ts = json.load(open(spec_file))["models"]
    m = next(x for x in ts if x["label"] == CANTO_B_CC)
    job = os.environ.get("SLURM_JOB_ID", "local")
    for s in seeds:
        e = next(x for x in m["entries"] if _seed(x["run"]) == str(s))
        run, cut, d = e["run"], float(e["cut"]), e["sub"]
        if os.path.exists(os.path.join(d, "provenance.json")):
            raise SystemExit(f"{d} exists: the post-freeze evaluation is done once")
        frozen = os.path.join(run, "eval_test")
        ce = sorted({round(float(x), 4) for x in pd.read_parquet(os.path.join(frozen, "per_event.parquet"),
                                                                   columns=["cut"]).cut})
        if len(ce) != 1:
            raise SystemExit(f"{frozen}: {ce} cuts, expected the single e* cut")
        local = os.path.join("/tmp", os.environ.get("USER", "u"), f"fb3_bestcut_{job}_s{s}")
        os.makedirs(os.path.join(local, "testset"), exist_ok=True)
        tar = os.path.join(run, "gnn_scores", "testset.tar")
        cuts = sorted({cut, ce[0]})
        try:
            subprocess.run(["tar", "xf", tar, "-C", os.path.join(local, "testset")], check=True)
            EG.main(["--graphs", os.path.join(local, "testset"), "--out", d, "--cuts",
                     ",".join(f"{c:.4f}" for c in cuts)])
        finally:
            shutil.rmtree(local, ignore_errors=True)
        ok = {}
        for f, keys in (("per_particle.parquet", ["event_id", "particle_id"]), ("per_event.parquet", ["event_id"])):
            new = pd.read_parquet(os.path.join(d, f))
            old = pd.read_parquet(os.path.join(frozen, f))
            ok[f] = bool(_same_rows(new[np.isclose(new.cut, ce[0], atol=1e-6)], old, keys))
        prov = {"what": "post-freeze CC evaluation of Canto-B on its stored one-time test scores at the best cut <= FR* "
                        "fixed on val (descriptive, not a claim; hnl.eval.strong_baselines_v3 bestcut)",
                "run": run, "scores": tar, "cut_best_val": cut, "spec": spec_file, "spec_sha256": _sha256(spec_file),
                "cuts_evaluated": cuts, "identity_cut": ce[0], "identity_check_frozen_eval_test": ok, "git": _git(),
                "slurm_job": job, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        with open(os.path.join(d, "provenance.json"), "w") as fh:
            json.dump(prov, fh, indent=1)
        print(json.dumps(prov, indent=1))
        if not all(ok.values()):
            raise SystemExit(f"{os.path.basename(run)}: e* rows differ from the frozen eval_test")


# ------------------------------------------------------------------------------------------------- test
def _check_new_output(e: dict, label: str, spec_file: str) -> None:
    """A new test output of this task: written once from this very val-fixed spec (unchanged since: sha256), holding
    exactly the entry's run and configuration (and merge model, by sha256)."""
    p = json.load(open(os.path.join(e["sub"], "provenance.json")))
    cfgs = json.load(open(os.path.join(e["sub"], "configs.json")))
    want = e["config"].split(",")
    same = (len(cfgs) == 1 and str(cfgs[0][0]) == want[0]
            and np.allclose([float(x) for x in cfgs[0][1:]], [float(x) for x in want[1:]], rtol=0, atol=1e-12))
    if not (same and os.path.normpath(p["run"]) == os.path.normpath(e["run"]) and p["config"] == e["config"]):
        raise SystemExit(f"{e['sub']}: configuration {cfgs} / provenance {p} is not the val-chosen {e['config']}")
    if os.path.normpath(p.get("spec", "")) != os.path.normpath(spec_file) or p.get("spec_sha256") != _sha256(spec_file):
        raise SystemExit(f"{e['sub']}: written from {p.get('spec')} ({p.get('spec_sha256')}), not from the current "
                         f"{spec_file} ({_sha256(spec_file)}): checksum mismatch, stop")
    if e.get("merge_model"):
        cur = _sha256(e["merge_model"])
        if not (p.get("merge_model_sha256") == e["merge_model_sha256"] == cur
                and os.path.normpath(p.get("merge_model") or "") == os.path.normpath(e["merge_model"])):
            raise SystemExit(f"{e['sub']}: merge model checksum mismatch ({p.get('merge_model_sha256')}, "
                             f"{e['merge_model_sha256']}, {cur}): stop")


def _check_reused_output(e: dict) -> None:
    """A test output of rounds 1-2 reused for the same run and configuration: its provenance and configuration (and, for
    round 2, its spec's checksum) checked as there."""
    SB._check_test_outputs({"entries": [e]})
    p = json.load(open(os.path.join(e["sub"], "provenance.json")))
    if p.get("spec") and _sha256(p["spec"]) != p.get("spec_sha256"):
        raise SystemExit(f"{e['sub']}: its spec {p['spec']} changed since its test job: checksum mismatch, stop")


def _check_bestcut_canto_b(e: dict, spec_file: str) -> None:
    f = os.path.join(e["sub"], "provenance.json")
    p = json.load(open(f))
    ok = (os.path.normpath(p["run"]) == os.path.normpath(e["run"]) and abs(float(p["cut_best_val"]) - e["cut"]) < 1e-6
          and all(p["identity_check_frozen_eval_test"].values()) and RT._holds(e["sub"], e["cut"]))
    if not ok:
        raise SystemExit(f"{f}: not run {e['run']} at cut {e['cut']}, or its identity check failed")
    if os.path.normpath(p["spec"]) != os.path.normpath(spec_file) or p["spec_sha256"] != _sha256(spec_file):
        raise SystemExit(f"{f}: spec checksum mismatch: stop")


def build_test_spec(out: str, name: str) -> tuple[list[dict], dict]:
    spec_file = os.path.abspath(os.path.join(out, f"{name}_spec_test.json"))
    meta = json.load(open(spec_file))
    rows = []
    for m in meta["models"]:
        for e in m["entries"]:
            if "config" in e:
                if e.get("reused"):
                    _check_reused_output(e)
                else:
                    _check_new_output(e, m["label"], spec_file)
            elif m["label"] == CANTO_B_CC:
                _check_bestcut_canto_b(e, spec_file)
            else:                                # rounds 1-2 / release-card entries: CC tables or frozen builder tables
                d = os.path.join(e["run"], e["sub"])
                if not os.path.exists(os.path.join(d, "configs.json")) and not RT._holds(d, float(e["cut"])):
                    raise SystemExit(f"{d}: cut {e['cut']} not evaluated")
        rows.append({k: m[k] for k in ("label", "short", "group") if k in m}
                    | {"entries": [{k: e[k] for k in ("run", "sub", "cut", "wp")} for e in m["entries"]]})
    return rows, meta


def test_checks(tab: pd.DataFrame, more: dict, spec: list[dict]) -> list:
    """Rows of rounds 1-2 whose test outputs are reused unchanged reproduce their test rows (1e-12); Canto-L and Canto-B
    (frozen) reproduce the release card's test rows (results/ranking/ranking_test_v1.csv)."""
    checks: list = []
    ref = pd.concat([pd.read_csv(_repo(f"results/{V1}.csv")), pd.read_csv(_repo(f"results/{V2}.csv"))])
    ref = ref[ref.set == "test"].drop_duplicates("label").set_index("label")
    ref_e = pd.concat([pd.read_csv(_repo(f"results/{V1}_entries.csv")), pd.read_csv(_repo(f"results/{V2}_entries.csv"))])
    ref_e = ref_e[ref_e.set == "test"].drop_duplicates(["label", "run"])
    t = tab.set_index("label")
    subs = {m["label"]: [os.path.normpath(os.path.join(e["run"], e["sub"])) for e in m["entries"]] for m in spec}
    v1s = {m["label"]: m for m in json.load(open(_repo(f"results/{V1}_spec_test.json")))["models"]}
    v2s = {m["label"]: m for m in json.load(open(_repo(f"results/{V2}_spec_test.json")))["models"]}
    for lab in t.index:
        if lab not in ref.index:
            continue
        old = v1s.get(lab) or v2s.get(lab)
        if old is not None:
            if [os.path.normpath(os.path.join(e["run"], e["sub"])) for e in old["entries"]] != subs[lab]:
                continue                                                  # another configuration now
        for k in RK.KEYS:
            _cmp(checks, f"rounds 1-2 test row {lab} {k}", float(t.loc[lab, k]), float(ref.loc[lab, k]))
            _cmp(checks, f"rounds 1-2 test row {lab} {k}_sd", float(t.loc[lab, k + "_sd"]), float(ref.loc[lab, k + "_sd"]))
        for c in ("k_n_d0_100", "k_n_d0_300"):
            _cmp(checks, f"rounds 1-2 test row {lab} {c}", str(t.loc[lab, c]), str(ref.loc[lab, c]))
        p = more[lab]["primary"]
        _cmp(checks, f"rounds 1-2 test row {lab} primary sigma", p["sigma"], float(ref.loc[lab, "primary_sigma_total"]))
    rc = pd.read_csv(_repo("results/ranking/ranking_test_v1.csv")).set_index("label")
    for mine, lab in ((NEW, "Canto-L (full budget): learned triplet union, learned merging (rule R6)"),
                      (CANTO_B_FROZEN, "Canto-B (full budget): base graph, learned merging (rule R6)")):
        if mine in t.index:
            for k in RK.KEYS:
                _cmp(checks, f"ranking_test_v1 row {lab} {k}", float(t.loc[mine, k]), float(rc.loc[lab, k]))
    return checks


def test_differences(spec: list[dict], prs: list[tuple[str, str, str]]) -> pd.DataFrame:
    """new minus baseline on test with the claims' estimator (hnl.eval.bootstrap.linear_combination: paired event
    bootstrap of the preregistered test tables, B = 500, + both seed terms; as hnl.eval.strong_baselines.differences),
    and for pairs of the same trained networks also the seed term paired by seed."""
    from hnl.eval import test_stats as TS
    from hnl.eval.particles import particle_table
    bs = BS_.Bootstrap()
    want = {x for a, b, _ in prs for x in (a, b)}
    models = {m["label"]: [TS.RunTest(e["run"], float(e["cut"]), sub=e["sub"]) for e in m["entries"]]
              for m in spec if m["label"] in want}
    runs = {m["label"]: [os.path.normpath(e["run"]) for e in m["entries"]] for m in spec}
    ev = np.unique(np.concatenate([rt.pp.event_id.unique() for rr in models.values() for rt in rr]))
    info = particle_table(ev)
    vals = {}
    for q, kind, mask, sets in SB.QUANTITIES:
        for lab, rr in models.items():
            if kind == "eff":
                p, r, n = TS.efficiency(bs, rr, info, mask(), sets)
            else:
                p, r = TS.event_ratio(bs, rr, "n_fake", "n_tracks_ge5", sets)
                n = None
            vals[(q, lab)] = (p, r, n)
    rows = []
    for new, b, blk in prs:
        paired = runs[new] == runs[b]
        for q, *_ in SB.QUANTITIES:
            pn, rn, nn = vals[(q, new)]
            pb, rb, nb = vals[(q, b)]
            d = BS_.linear_combination([(1.0, pn, rn), (-1.0, pb, rb)])
            sn, sb = BS_.seed_mean(pn, rn), BS_.seed_mean(pb, rb)
            row = {"set": "test", "block": blk, "new": new, "baseline": b, "quantity": q, "n_per_seed": nb,
                   "value_new": sn["mean"], "sigma_new": sn["sigma"], "s_seeds_new": sn["s_seeds"],
                   "value_base": sb["mean"], "sigma_base": sb["sigma"], "s_seeds_base": sb["s_seeds"],
                   "difference": d["value"], "sigma_boot": d["sigma_boot"], "sigma_seeds": d["sigma_seeds"],
                   "sigma": d["sigma"], "z": d["z"], "B": BS_.B}
            if paired:
                ds = np.asarray(pn, dtype=float) - np.asarray(pb, dtype=float)
                st = float(np.std(ds, ddof=1) / np.sqrt(len(ds)))
                row.update({"seed_term_paired": st, "sigma_paired": float(np.sqrt(d["sigma_boot"] ** 2 + st ** 2)),
                            "per_seed": ";".join(f"{x:.10g}" for x in ds)})
            rows.append(row)
    return pd.DataFrame(rows)


def run_test(out: str, name: str, notes_file: str | None) -> None:
    fr = SB.fr_star()
    spec, meta = build_test_spec(out, name)
    tab, ent, more = RT.rank(spec)
    checks = test_checks(tab, more, spec)
    prs = pairs(list(tab.label))
    d = test_differences(spec, prs)
    claims = json.load(open(_repo("results/claims_test_v1.json")))["result"]["primary"]["difference"]
    ref = d[(d.new == NEW) & (d.quantity == "primary")
            & (d.baseline == "acorn (HNL) + CC, preregistered cut (claims' reference)")].iloc[0]
    for k in ("value", "sigma_boot", "sigma_seeds", "sigma"):
        _cmp(checks, f"claims primary difference {k}", float(ref["difference" if k == "value" else k]), float(claims[k]))
    for f in (f"results/{V1}_differences.csv", f"results/{V2}_differences.csv"):
        r0 = pd.read_csv(_repo(f))
        r0 = r0[r0.set == "test"] if "set" in r0 else r0
        same_rows = set(c[0].split("test row ")[1].rsplit(" ", 1)[0] for c in checks if "rounds 1-2 test row" in c[0])
        for (a, b), g in d.groupby(["new", "baseline"], sort=False):
            r = r0[(r0.new == a) & (r0.baseline == b)]
            if r.empty or a not in same_rows | {NEW} or b not in same_rows | {NEW}:
                continue
            r = r.drop_duplicates("quantity").set_index("quantity")
            for _, x in g.iterrows():
                for c in ("value_new", "value_base", "difference", "sigma_boot", "sigma_seeds", "sigma"):
                    _cmp(checks, f"{f}: {a} - {b} {x.quantity} {c}", float(x[c]), float(r.loc[x.quantity, c]))
    pd.DataFrame(checks, columns=["check", "ours", "reference", "ok"]).to_csv(
        os.path.join(out, f"{name}_test_checks.csv"), index=False)
    bad = [c for c in checks if not c[3]]
    if bad:
        for c in bad[:30]:
            print("FAIL", c)
        raise SystemExit(f"{len(bad)} of {len(checks)} test checks failed")
    print(f"test checks: {len(checks)} of {len(checks)} ok")
    tab.insert(0, "set", "test")
    ent.insert(0, "set", "test")
    for lab in tab.label:
        p = more[lab]["primary"]
        tab.loc[tab.label == lab, "primary_sigma_total"] = p["sigma"]
        tab.loc[tab.label == lab, "primary_sigma_boot"] = p["sigma_boot"]
    hnl_best = family_label("ii", "best")
    ps = SB.per_set(spec, [(NEW, hnl_best)])
    ps.to_csv(os.path.join(out, f"{name}_test_per_set.csv"), index=False)
    vt = pd.read_csv(os.path.join(out, f"{name}_val.csv"))
    cols = list(vt.columns) + [c for c in tab.columns if c not in vt.columns]
    pd.concat([vt, tab], ignore_index=True)[cols].to_csv(os.path.join(out, f"{name}.csv"), index=False)
    ve = pd.read_csv(os.path.join(out, f"{name}_val_entries.csv"))
    pd.concat([ve, ent], ignore_index=True).to_csv(os.path.join(out, f"{name}_entries.csv"), index=False)
    vd = pd.read_csv(os.path.join(out, f"{name}_val_differences.csv"))
    pd.concat([vd, d], ignore_index=True).to_csv(os.path.join(out, f"{name}_differences.csv"), index=False)
    notes = [] if notes_file is None else [ln.strip()[2:] for ln in open(notes_file) if ln.startswith("- ")]
    md = markdown(vt, tab, more, vd, d, fr, out, name, ps, notes)
    with open(os.path.join(out, f"{name}.md"), "w") as f:
        f.write(md)
    print(md)


# ------------------------------------------------------------------------------------------------- markdown
def _cell(r, q: str) -> str:
    z = f"z {r.z:+.1f}" if np.isfinite(r.z) else "identical"
    return f"{100 * r.difference:+.2f} ± {100 * r.sigma:.2f} ({z})"


def _diff_rows(d: pd.DataFrame, block: str, paired: bool = False) -> list[str]:
    qs = [q for q, *_ in SB.QUANTITIES]
    lines = ["| new | baseline | " + " | ".join(SB.QLAB[q] for q in qs) + " |", "|---|---|" + "---|" * len(qs)]
    sel = d[d.block == block]
    for (a, b), g in sel.groupby(["new", "baseline"], sort=False):
        g = g.set_index("quantity")
        cells = []
        for q in qs:
            r = g.loc[q]
            c = _cell(r, q)
            if paired and "sigma_paired" in r and np.isfinite(r.sigma_paired):
                c += f" [paired seeds ± {100 * r.sigma_paired:.2f}]"
            cells.append(c)
        lines.append(f"| {a} | {b} | " + " | ".join(cells) + " |")
    return lines


def _arm_tables(t: pd.DataFrame) -> list[str]:
    L = []
    for arm, alab in ARMS:
        sub = t[t.label.str.startswith(alab + " + ")].sort_values("eff_primary", ascending=False, kind="stable")
        L += [f"**{alab}**", ""] + SB._table(sub) + [""]
    other = t[~t.label.str.startswith("acorn (")].sort_values("eff_primary", ascending=False, kind="stable")
    L += ["**Canto (reference, decomposition and same-builder rows)**", ""] + SB._table(other) + [""]
    return L


def markdown(vt: pd.DataFrame, tt: pd.DataFrame, more: dict, vd: pd.DataFrame, d: pd.DataFrame, fr: float, out: str,
             name: str, ps: pd.DataFrame, notes: list[str]) -> str:
    cand = pd.read_csv(os.path.join(out, f"{name}_menu_candidates.csv"))
    edges = pd.read_csv(os.path.join(out, f"{name}_edges.csv"), dtype=str).fillna("")
    L = ["# The acorn baselines with Canto's full track-builder menu (feedback round 3, task A2)", "",
         "acorn (ttbar), acorn (HNL), acorn (ttbar+HNL) = acorn's interaction GNN trained on ttbar only, on the HNL "
         "signal (Wang et al.) and on both (Stage-0 arms i, ii, iii; three seeds each, the Stage-0 scores). Builders on "
         "those scores: connected components (CC), acorn's own CC + walkthrough (CCandWalk, plain and score-guided), "
         "walkthrough + merging, MWS-layer + merging, two-pass MWS-layer + merging and MWS-layer + learned merging (a "
         "fragment-merging GBDT per run, trained on that run's own GNN scores of 2000 of its training graphs, as "
         "Canto's). Grids: those of rounds 1-2 and Canto's extended grid, extended on val past every edge at which a "
         f"choice sat (edge-extension rule, docs/feedback/round3_A2_decisions.md). Every configuration is chosen per "
         f"seed on val by the builder rule at FR* = {fr:.6f} (pooled HNL val) with the one-standard-error tie rule; "
         "the val-best row of an arm applies the same rule per seed over the families' choices and CC at its best cut. "
         "Nothing is chosen on test. Cells: mean ± seed sd over three seeds; |d0| bins: HNL-daughter targets pooled over "
         "the seeds (k/n).", "", "## Validation (pooled HNL val 1,999 events + ttbar val 1,000 events)", ""]
    L += _arm_tables(vt)
    L += ["## Val-best configuration per arm and seed (full menu)", "",
          "| arm | seed | chosen | primary | fake rate | candidates within one standard error (tied) |", "|---|---|---|---|---|---|"]
    for (arm, seed), g in cand.groupby(["arm", "seed"], sort=False):
        c = g[g.chosen].iloc[0]
        tied = "; ".join(f"{r.working_point.split(' (')[0]} {r.eff_primary:.4f} / {r.fake_rate_hnl:.4f}"
                         for _, r in g[g.tied].sort_values("eff_primary", ascending=False).iterrows())
        L.append(f"| {ALAB[str(arm)]} | {seed} | {c.working_point.split(' (trackbuild')[0]} | {c.eff_primary:.4f} | "
                 f"{c.fake_rate_hnl:.4f} | {tied} |")
    r2 = json.load(open(os.path.join(out, f"{name}_spec_test.json"))).get("round2_rule_best", {})
    if r2:
        L += ["", "For comparison, round 2's rule (the arm's row with the highest three-seed-mean val primary efficiency, "
              "ties to the lower fake rate): " + "; ".join(f"{ALAB[a]}: {lab.split(' + ', 1)[1]}" for a, lab in r2.items()) + "."]
    L += ["", "## Grid edges after the extension (val)", "",
          "| run | family | chosen | primary | fake rate | configurations | extension steps | edges | status |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in edges.iterrows():
        f4 = lambda x: f"{float(x):.4f}" if str(x) not in ("", "nan", "None") else "-"
        L.append(f"| {r.run} | {r.family} | {r.params} | {f4(r.eff_primary)} | {f4(r.fake_rate_hnl)} | "
                 f"{r.n_family_configs} | {r.extension_steps} | {r.edges} | {r.status} |")
    L += ["", "## Test, post-freeze descriptive (not claims; the preregistered claims stay as they are)", "",
          "Pooled HNL test sets (hnl10_100_test + hnl15_200_test + hnl15_100_external, 6,999 events) for primary, |d0|, "
          "W muons and fake rate; ttbar_test (1,000 events) for ttbar prompt. The val-chosen configurations applied once "
          "to the stored test scores (an output of rounds 1-2 is reused where the run and configuration are the same); "
          "summary code path of hnl.eval.ranking_test.", ""]
    L += _arm_tables(tt)
    L += ["Primary efficiency with the preregistered total sigma (event bootstrap + seed term):", ""]
    for _, r in tt.sort_values("eff_primary", ascending=False, kind="stable").iterrows():
        p = more[r.label]["primary"]
        L.append(f"- {r.label}: {p['mean']:.4f} ± {p['sigma']:.4f} (bootstrap {p['sigma_boot']:.4f}, seed sd "
                 f"{p['s_seeds']:.4f})")
    for title, dd in (("Differences on test (new minus baseline; paired event bootstrap of the preregistered test "
                       "tables, B = 500, + both seed terms: the claims' estimator; post-freeze descriptive)", d),
                      ("Differences on val (new minus baseline; paired event bootstrap of the val tables of feedback "
                       "point 5, B = 1000, + both seed terms; validation)", vd)):
        L += ["", f"## {title}", ""]
        for blk in dd.block.drop_duplicates():
            L += [f"**{blk}**", ""] + _diff_rows(dd, blk, paired=blk.startswith("decomposition")) + [""]
    if ps is not None and len(ps):
        L += ["Primary efficiency per test set (descriptive; seed mean ± total sigma; difference with the same "
              "estimator):", "", "| test set | primary particles per seed | new | value | baseline | value | difference |",
              "|---|---|---|---|---|---|---|"]
        for _, r in ps.iterrows():
            L.append(f"| {r.set} | {int(r.n_per_seed)} | {r.new} | {r.value_new:.4f} ± {r.sigma_new:.4f} | {r.baseline} | "
                     f"{r.value_base:.4f} ± {r.sigma_base:.4f} | {100 * r.difference:+.2f} ± {100 * r.sigma:.2f} "
                     f"(z {r.z:+.1f}) |")
        L.append("")
    L += ["Points = percentage points (fake rate: a negative difference means the new row has the lower fake rate). "
          "[paired seeds]: the same trained networks on both sides, seed term of the per-seed differences. A row's own "
          "sigma (bootstrap + seed term) is in the CSV files.", "", "## Notes", ""] + [f"- {n}" for n in notes]
    return "\n".join(L) + "\n"


def cmd_merge() -> None:
    """RUN/trackbuild_val_fb3_all = the union of every grid part of the menu (data copy; hnl.eval.strong_baselines
    merge_grids: same events and particles required, no configuration twice)."""
    for run in runs_all():
        parts = [g for g in MENU_PARTS if os.path.exists(os.path.join(run, g, "configs.json"))] + ext_parts(run)
        if len(parts) < len(MENU_PARTS):
            raise SystemExit(f"{run}: missing grid parts ({parts})")
        print(SB.merge_grids(run, parts, ALL_GRID), parts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pick")
    p.add_argument("--dir", required=True)
    p.add_argument("--n", type=int, default=N_TRAIN_GRAPHS)
    p.add_argument("--seed", type=int, default=0)
    q = sub.add_parser("provenance-lmerge")
    for k in ("run", "out", "ckpt", "pairs", "scored", "started"):
        q.add_argument(f"--{k}", required=True)
    g = sub.add_parser("grid")
    g.add_argument("--graphs", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--configs", required=True, help="'fam,p,..;fam,p,..' or @FILE holding that text")
    g.add_argument("--merge-model", default=None)
    g.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    v = sub.add_parser("inventory")
    v.add_argument("--families", default=",".join(f for _, f, *_ in FAMILIES))
    v.add_argument("--step", type=int, default=None, help="write the extension plan of this step")
    v.add_argument("--n", type=int, default=2, help="values per edge and step")
    v.add_argument("--out-dir", default=os.path.join(FB3, "extensions"))
    j = sub.add_parser("ext-jobs")
    j.add_argument("--step", type=int, required=True)
    j.add_argument("--out-dir", default=os.path.join(FB3, "extensions"))
    sub.add_parser("merge")
    for c in ("val", "jobs", "bestcut", "test"):
        x = sub.add_parser(c)
        x.add_argument("--out", default=_repo("results"))
        x.add_argument("--name", default=NAME)
        if c == "bestcut":
            x.add_argument("--seeds", default="0,1,2")
        if c == "test":
            x.add_argument("--notes", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "pick":
        names = [f for f in os.listdir(a.dir) if f.endswith(".pyg.gz")]
        print("\n".join(pick_graphs(names, a.n, a.seed)))
    elif a.cmd == "provenance-lmerge":
        provenance_lmerge(a.run, a.out, a.ckpt, a.pairs, a.scored, a.started)
    elif a.cmd == "grid":
        text = open(a.configs[1:]).read() if a.configs.startswith("@") else a.configs
        run_grid(a.graphs, a.out, parse_configs(text), a.workers, a.merge_model)
    elif a.cmd == "inventory":
        inventory(a.families.split(","), a.out_dir if a.step is not None else None, a.step, a.n)
    elif a.cmd == "ext-jobs":
        print("\n".join(ext_jobs(a.step, a.out_dir)))
    elif a.cmd == "merge":
        cmd_merge()
    elif a.cmd == "val":
        run_val(a.out, a.name)
    elif a.cmd == "jobs":
        print("\n".join(test_jobs(a.out, a.name)))
    elif a.cmd == "bestcut":
        cmd_bestcut(a.seeds.split(","), a.out, a.name)
    elif a.cmd == "test":
        run_test(a.out, a.name, a.notes)


if __name__ == "__main__":
    main()
