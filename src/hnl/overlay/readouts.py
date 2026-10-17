"""Read-outs of addendum H section 7 that the H1 code does not compute (docs/preregistration_pileup.md sections 6 and 7),
on the outputs of the frozen Canto-L (TrackML) for one overlay set in one condition. No claim rests on them.

Per event and working point (WP-A, WP-S and WP-R-P3; on the pilot the dry configurations of the grid):
  all particles   the frozen counts of hnl.trackml.evaluate.event_counts (gun targets included), pooled as in
                  hnl.trackml.evaluate.pooled: fake rate n_fake_ge5 / n_tracks_ge5 and duplicate rate
                  dup_sum_targets / n_target (section 6: both fake rates of H1 are reported);
  gun hits        fake tracks (>= 5 hits and no double-majority particle, as n_fake_ge5) that hold at least one gun hit,
                  per gun LLP (10 per event; the LLPs with a particle row are counted as well);
  host            the counts of hnl.overlay.blind.blind_counts (the V2 definition: gun particles removed, gun hits as
                  noise, tracks with half or more gun hits removed): host prompt efficiency, host fake rate and, in
                  C-full, the TrackML score; each also split by the collision-twin status of the host's hard scatter
                  (--twins, hnl.overlay.signatures).
Per gun target, at WP-A (required on production sets; the first dry configuration on the pilot): the stage at which it is lost,
hnl.trackml.displaced_diag.diagnose and outcome (found, graph, gnn, split, contaminated), with the P_P and P_D flags
of hnl.overlay.h1.gun_targets and the sign of eta.

Inputs, always checked against the stored read-outs so that the tracks are the evaluated ones:
  - a set and condition with a grid (C-1GeV and C-hs; C-full on H-test): the scores the grid was built from. The
    tracks are built again with the frozen builders exactly as hnl.overlay.tracks builds them, and the frozen counts
    of every event and configuration must equal the grid's per_event row.
  - H-val in C-full (no grid): the sealed directory, opened with hnl.overlay.blind.open_sealed (the step-5 read, only
    after the version record is committed and with the manifest sha256 of RESULTS.md); its raw track labels and
    scores, with the blinded counts recomputed from the labels and checked against the sealed per_event rows, as
    hnl.overlay.blind.sealed_per_target does. The pilot's scratch sealed directory needs its manifest sha256 (--expect).
Grants (hnl.overlay.grant): kind eval in the condition, with --out inside the grant's out (dry on the pilot, with the
explicit inputs below); on H-val and H-test only while the set's launch record accepts HNL_H_LAUNCH (decision D3).

Writes under --out only (one block: blocks/b<K>of<N>; refuses to overwrite): per_event.parquet, loss.parquet and
summary.json; `merge` pools the blocks and writes the summary of the set.

    python -m hnl.overlay.readouts run --set SET --cond full|1gev|hs --out DIR [--twins FILE] [--block K --n-blocks N]
        [--workers W] [--scores DIR --grid DIR --sealed DIR --expect SHA --root ROOT]   (explicit inputs: pilot only)
    python -m hnl.overlay.readouts merge --out DIR [--twins FILE]
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import time

import numpy as np
import pandas as pd

from hnl.overlay import blind as B
from hnl.overlay import grant as G

LOSS_WP = "WP-A"
N_LLP = 10                            # isolated (1-8) and dense (9-10) LLPs per event (hnl.overlay.gun)
EVENT_COLS = ("n_tracks", "n_tracks_ge5", "n_fake_ge5", "dup_sum_targets", "n_target", "m_target",
              "n_prompt_target", "m_prompt_target")
HOST_COLS = ("n_tracks_ge5", "n_fake_ge5", "n_prompt_target", "m_prompt_target")
B_BOOT, SEED_BOOT = 500, 12345


# ------------------------------------------------------------------------------------------------ per event
def gun_hit_fakes(pid, labels) -> tuple[int, int]:
    """(fake tracks with >= 5 hits, those holding at least one gun hit): the matching of
    hnl.trackml.evaluate.event_counts (a track is fake when no particle holds more than half of its hits and more than
    half of its own hits; noise is particle 0)."""
    from hnl.trackml.evaluate import _match
    pid = np.asarray(pid, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    particles, n_particle, tu, n_track, pt, pp, n_shared = _match(pid, labels)
    T = len(tu)
    if T == 0:
        return 0, 0
    real = particles[pp] != 0 if len(pp) else np.zeros(0, dtype=bool)
    dm = (2 * n_shared > n_track[pt]) & (2 * n_shared > n_particle[pp]) & real
    matched = np.zeros(T, dtype=bool)
    matched[pt[dm]] = True
    fake = (n_track >= 5) & ~matched
    on = labels >= 0
    gun_hit = np.zeros(T, dtype=bool)
    gt = np.unique(labels[on & B.is_gun(pid)])
    gun_hit[np.searchsorted(tu, gt)] = True
    return int(fake.sum()), int((fake & gun_hit).sum())


def n_llp(particles: pd.DataFrame) -> int:
    v = B.vertex_field(particles.particle_id.to_numpy(np.int64))
    return int(len(np.unique(v[B.is_gun(particles.particle_id.to_numpy(np.int64))])))


def event_rows(ev, ev_full, labels: dict[int, np.ndarray], names: list[str], cond: str) -> list[dict]:
    """Per configuration index: the frozen counts, the gun-hit fakes and the host (blinded) counts of one event."""
    from hnl.trackml.evaluate import event_counts
    hid = ev.full_rows + 1
    w = ev.weight if cond == "full" else None
    out = []
    for i, lab in labels.items():
        c, _ = event_counts(ev.pid, lab, ev.particles, hid, w)
        cb, _ = B.blind_counts(ev.pid, lab, ev.particles, hid, w)
        nf, nfg = gun_hit_fakes(ev.pid, lab)
        if nf != c["n_fake_ge5"]:
            raise RuntimeError(f"event {ev.event} config {i}: {nf} fake tracks by the gun-hit count, "
                               f"{c['n_fake_ge5']} by event_counts")
        row = {"event": int(ev.event), "config": int(i), "name": names[i], **{k: c[k] for k in EVENT_COLS},
               "n_fake_ge5_gun_hit": nfg, "n_llp_rows": n_llp(ev_full.particles), "n_llp": N_LLP,
               **{f"host_{k}": cb[k] for k in HOST_COLS},
               "host_trackml_score": cb.get("trackml_score", np.nan)}
        out.append(row)
    return out


def loss_rows(ev, a, b, sc, lab) -> pd.DataFrame:
    """hnl.trackml.displaced_diag.diagnose on the gun targets of one event, with its outcome and the H1 flags."""
    from hnl.overlay.h1 import gun_targets
    from hnl.trackml import displaced_diag as DD
    d = DD.diagnose(ev, a, b, sc, lab)
    gt = gun_targets(ev.particles.assign(event=np.int64(ev.event)))
    gt = gt[gt.target.astype(bool)][["particle_id", "llp", "iso", "pp", "pd", "soft_sibling", "eta"]]
    d = d.merge(gt.drop(columns="eta"), on="particle_id", how="inner", validate="one_to_one")
    if len(d) != len(gt):
        raise RuntimeError(f"event {ev.event}: {len(gt)} gun targets, {len(d)} diagnosed")
    d["outcome"] = DD.outcome(d).to_numpy()
    d["eta_sign"] = np.where(d.eta.to_numpy() >= 0, "+", "-")
    return d


# ------------------------------------------------------------------------------------------------ inputs
def _sealed_inputs(set_name: str, k: int, sealed: str | None, expect: str | None, repo: str) -> tuple[str, list]:
    """(sealed directory, configurations) of a C-full set without a grid, opened by its rule."""
    if k == 0:
        sealed = B.sealed_dir(set_name, "full", B.RUNS_H)
        B.open_sealed(sealed, expect, repo)
    else:
        if not sealed or not expect:
            raise PermissionError("a scratch sealed directory is read only with --sealed and its --expect")
        if B._under(sealed, B.RUNS_H):
            raise PermissionError("the pilot's sealed directory is a scratch directory")
        bad = B.verify(sealed, expect)
        if bad:
            raise PermissionError(f"{sealed} does not verify: {bad[:5]}")
    with open(os.path.join(sealed, "blind.json")) as fh:
        rec = json.load(fh)
    if rec["set"] != set_name or int(rec["k"]) != k:
        raise ValueError(f"{sealed}: a sealed directory of {rec['set']}, not {set_name}")
    if k == 9 and not rec["scores"].get("placeholder"):
        raise PermissionError("a pilot directory holds truth-made placeholder outputs only")
    with open(os.path.join(sealed, "configs.json")) as fh:
        configs = json.load(fh)
    return sealed, configs


def _grid_inputs(set_name: str, k: int, cond: str, scores: str | None, grid_dir: str | None, root: str | None,
                 events) -> tuple[str, str, list, pd.DataFrame]:
    """(scores, grid, configurations, grid per_event rows) of a set and condition with a grid."""
    from hnl.overlay import tracks as TR
    if k == 9:
        if not scores or not grid_dir:
            raise PermissionError("the pilot takes explicit --scores and --grid (placeholders)")
        rec = TR.grid_record(grid_dir, set_name, cond)
    else:
        if scores or grid_dir:
            raise PermissionError("production sets read the scores and grid of hnl.overlay.chain paths only")
        from hnl.overlay.chain import paths
        scores = paths(set_name, cond)["scores"]
        grid_dir = paths(set_name, cond)["grid"]
        if not grid_dir:
            raise PermissionError(f"{set_name} {cond} has no grid")
        rec = TR.grid_provenance(grid_dir, set_name, cond, root, events=None)
    if os.path.realpath(rec["scores"]) != os.path.realpath(scores):
        raise ValueError(f"{grid_dir}: built from {rec['scores']}, not {scores}")
    pe = pd.read_parquet(os.path.join(grid_dir, "per_event.parquet"))
    return scores, grid_dir, rec["configs"], pe


def _merge_model(configs: list[dict], k: int) -> str | None:
    if not any(c["family"] == "mwslmerge" for c in configs):
        return None
    if k == 9:
        raise PermissionError("no learned component runs on a pilot event (mwslmerge refused on k = 9)")
    with open(B.SELECTION) as fh:
        mm = json.load(fh)["inputs"]["merge_model"]
    if B.sha256_file(mm["path"]) != mm["sha256"]:
        raise ValueError(f"{mm['path']}: sha256 differs from selection_v1.json")
    return mm["path"]


def sealed_labels(sealed: str, events, config: int) -> dict[int, np.ndarray]:
    """hnl.overlay.blind.read_labels with the events in the parquet filter as well (one row group per event and
    configuration, so only the wanted row groups are read); the same checks."""
    import pyarrow.parquet as pq
    want = sorted({int(e) for e in events})
    out: dict[int, np.ndarray] = {}
    for p in sorted(glob.glob(os.path.join(sealed, "tracks", "*.parquet"))):
        t = pq.read_table(p, filters=[("config", "==", int(config)), ("event", "in", want)]).to_pandas()
        for e, g in t.groupby("event"):
            if int(e) in out:
                raise ValueError(f"{sealed}: event {e} appears twice in tracks/")
            g = g.sort_values("row")
            if not np.array_equal(g.row.to_numpy(), np.arange(len(g))):
                raise ValueError(f"{sealed}: the labels of event {e} are not rows 0..N-1")
            out[int(e)] = g.label.to_numpy(np.int64)
    missing = sorted(set(want) - set(out))
    if missing:
        raise KeyError(f"{sealed}: no track labels for events {missing[:5]} at configuration {config}")
    return out


# ------------------------------------------------------------------------------------------------ one chunk
def _chunk(args):
    from hnl.overlay.conditions import with_condition
    from hnl.trackml import events as E
    from hnl.trackml import trackbuild as TB
    events, cond, root, src, configs, loss_i, merge_model, ref = args
    names = [c["name"] for c in configs]
    model = None
    if merge_model:
        import joblib
        model = joblib.load(merge_model)
    sc_all = TB.read_scores(src["scores"], events, cond)
    labels_all = None
    if src["kind"] == "sealed":
        labels_all = {i: sealed_labels(src["sealed"], events, i) for i in range(len(configs))}
    rows, loss = [], []
    for ev0 in E.iter_events(events, root=root):
        ev = with_condition(ev0, cond)
        a, b, sc = sc_all[ev.event]
        if len(a) and max(int(np.max(a)), int(np.max(b))) >= ev.n:
            raise ValueError(f"event {ev.event}: scored edges index beyond the {ev.n} hits of condition {cond}")
        if labels_all is None:
            st = TB.EventState(ev, a, b, sc, model)
            labels = {i: np.asarray(TB.build(B.config_tuple(c), st)) for i, c in enumerate(configs)}
        else:
            labels = {i: labels_all[i][ev.event] for i in range(len(configs))}
            if any(len(v) != ev.n for v in labels.values()):
                raise ValueError(f"event {ev.event}: stored labels do not match the {ev.n} hits")
        r = event_rows(ev, ev0, labels, names, cond)
        _check_reference(ev, labels, r, ref, src["kind"], cond)
        rows.extend(r)
        loss.append(loss_rows(ev, a, b, sc, labels[loss_i]).assign(config=np.int16(loss_i)))
    return rows, (pd.concat(loss, ignore_index=True) if loss else pd.DataFrame())


def _check_reference(ev, labels, rows, ref: pd.DataFrame, kind: str, cond: str) -> None:
    """The tracks are those that were evaluated: the frozen counts equal the grid's per_event rows, or the blinded
    counts equal the sealed per_event rows (H-val C-full)."""
    if kind == "grid":
        for r in rows:
            g = ref[(ref.event == r["event"]) & (ref.config == r["config"])]
            if len(g) != 1:
                raise ValueError(f"event {r['event']} config {r['config']}: {len(g)} grid rows")
            for col in EVENT_COLS:
                if int(g[col].iloc[0]) != int(r[col]):
                    raise ValueError(f"event {r['event']} config {r['config']}: {col} {r[col]} here, "
                                     f"{int(g[col].iloc[0])} in the grid")
    else:
        hid = ev.full_rows + 1
        for i, lab in labels.items():
            cb, _ = B.blind_counts(ev.pid, lab, ev.particles, hid, ev.weight)
            s = ref[(ref.event == ev.event) & (ref.config == i)]
            if len(s) != 1:
                raise ValueError(f"event {ev.event} config {i}: {len(s)} sealed per_event rows")
            for key, v in cb.items():
                sv = s[key].iloc[0]
                same = (np.isclose(float(sv), float(v), rtol=0, atol=1e-12) or (np.isnan(float(sv)) and
                        np.isnan(float(v)))) if isinstance(v, float) else int(sv) == int(v)
                if not same:
                    raise ValueError(f"event {ev.event}: blinded {key} {v} from the labels differs from the sealed {sv}")


# ------------------------------------------------------------------------------------------------ run and summary
def run(set_name: str, cond: str, out: str, twins: str | None = None, block: int = 0, n_blocks: int = 1,
        workers: int = 1, scores: str | None = None, grid_dir: str | None = None, sealed: str | None = None,
        expect: str | None = None, root: str | None = None, repo: str = B.REPO) -> dict:
    import concurrent.futures as cf
    split = B.load_split()
    k = B.set_index(set_name, split)
    if cond not in B.CONDITIONS_H:
        raise ValueError(f"condition {cond!r} not in {B.CONDITIONS_H}")
    if k == 1:
        raise PermissionError("the closure set holds no gun particle: no read-out of this module")
    if k == 9 and B._under(out, B.RUNS_H):
        raise PermissionError("placeholder outputs never go under the production runs root")
    if k != 9 and any(x is not None for x in (scores, grid_dir, sealed, root)):
        raise PermissionError("--scores, --grid, --sealed and --root are for the pilot only")
    if k == 2 and not twins:
        raise PermissionError("H-test read-outs need --twins (the H-test twins file of hnl.overlay.signatures, H-T1)")
    all_events = B.set_ids(set_name, split)
    per = -(-len(all_events) // n_blocks)
    events = all_events[block * per:(block + 1) * per]
    if not events:
        raise SystemExit(f"block {block} of {n_blocks} is empty")
    G.check(events, "dry" if k == 9 else "eval", cond, out=out)
    if k in (0, 2):
        bad = B.check_launch(set_name, os.environ.get("HNL_H_LAUNCH"), B.runs_root(repo), repo)
        if bad:
            raise PermissionError(f"{set_name}: refused by its launch record: {bad[:5]}")
    root = B.overlay_root(k, root)
    d = out if n_blocks == 1 else os.path.join(out, "blocks", B.block_name(block, n_blocks))
    if os.path.exists(os.path.join(d, "summary.json")) or os.path.exists(os.path.join(d, "per_event.parquet")):
        raise SystemExit(f"{d} holds read-outs already: refusing to overwrite them")
    if k == 0 and cond == "full" or (k == 9 and cond == "full" and sealed):
        sd, configs = _sealed_inputs(set_name, k, sealed, expect, repo)
        ref = pd.read_parquet(os.path.join(sd, "per_event.parquet"))
        src = {"kind": "sealed", "sealed": sd, "scores": os.path.join(sd, "scores")}
        provenance = {"sealed": sd, "manifest_sha256": B.sha256_file(os.path.join(sd, B.MANIFEST))}
    else:
        sc_dir, gd, configs, ref = _grid_inputs(set_name, k, cond, scores, grid_dir, root, events)
        src = {"kind": "grid", "scores": sc_dir}
        provenance = {"grid": gd, "scores": sc_dir,
                      "grid_per_event_sha256": B.sha256_file(os.path.join(gd, "per_event.parquet"))}
    names = [c["name"] for c in configs]
    if k != 9 and LOSS_WP not in names:
        raise ValueError(f"no {LOSS_WP} among the configurations {names} of a production set")
    loss_i = names.index(LOSS_WP) if LOSS_WP in names else 0
    merge_model = _merge_model(configs, k)
    t0 = time.time()
    chunks = [(events[i:i + 1], cond, root, src, configs, loss_i, merge_model, ref[ref.event == events[i]])
              for i in range(len(events))]
    with cf.ProcessPoolExecutor(max_workers=max(1, min(workers, len(chunks)))) as ex:
        res = list(ex.map(_chunk, chunks))
    pe = pd.DataFrame([r for rr, _ in res for r in rr])
    loss = pd.concat([lo for _, lo in res if len(lo)], ignore_index=True) if any(len(lo) for _, lo in res) \
        else pd.DataFrame()
    rec = {"set": set_name, "k": k, "condition": cond, "events": [int(e) for e in events], "n_events": len(events),
           "block": block, "n_blocks": n_blocks, "root": root or B.DERIVED_OVERLAY, "configs": configs,
           "loss_config": names[loss_i], "inputs": {**provenance, "kind": src["kind"]}, "merge_model": merge_model,
           "elapsed_s": round(time.time() - t0, 1), "git": B._git_head(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
           "host": os.uname().nodename, "code_sha256": B.sha256_file(os.path.abspath(__file__)),
           "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    os.makedirs(d, exist_ok=True)
    pe.to_parquet(os.path.join(d, "per_event.parquet"), index=False)
    loss.to_parquet(os.path.join(d, "loss.parquet"), index=False)
    rec["summary"] = summarize(pe, loss, configs, _load_twins(twins), cond) if n_blocks == 1 else None
    if twins:
        rec["twins"] = {"path": os.path.abspath(twins), "sha256": B.sha256_file(twins)}
    with open(os.path.join(d, "summary.json"), "w") as fh:
        json.dump(rec, fh, indent=1, default=_json_default)
    return rec


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not JSON: {type(o)}")


def _load_twins(path: str | None) -> dict[int, bool] | None:
    if not path:
        return None
    with open(path) as fh:
        return {int(h): bool(t) for h, t in json.load(fh)["hs_twin"].items()}


def _host_of(event: int) -> int:
    return int(event) % 1_000_000


def _ratio_boot(pe: pd.DataFrame, num: str, den: str, idx: np.ndarray) -> tuple[float, float]:
    """Pooled num / den over the events and its event-bootstrap sd (index table of hnl.trackml.evaluate.bootstrap)."""
    n, d = pe[num].to_numpy(np.float64), pe[den].to_numpy(np.float64)
    val = float(n.sum() / d.sum()) if d.sum() else float("nan")
    if not len(pe):
        return val, float("nan")
    ns, ds = n[idx].sum(axis=1), d[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = ns / ds
    return val, float(np.nanstd(r, ddof=1))


def _boot_index(n: int) -> np.ndarray:
    return np.random.default_rng(SEED_BOOT).integers(0, n, size=(B_BOOT, n))


def summarize(pe: pd.DataFrame, loss: pd.DataFrame, configs: list[dict], twins: dict[int, bool] | None,
              cond: str) -> dict:
    """Pooled read-outs per configuration (events in overlay-id order for the bootstrap index table)."""
    out = {}
    for i, c in enumerate(configs):
        q = pe[pe.config == i].sort_values("event").reset_index(drop=True)
        idx = _boot_index(len(q))
        r = {}
        for key, (num, den) in {"fake_rate": ("n_fake_ge5", "n_tracks_ge5"),
                                "duplicate_rate": ("dup_sum_targets", "n_target"),
                                "efficiency_all_targets": ("m_target", "n_target"),
                                "fake_ge5_with_gun_hit_per_llp": ("n_fake_ge5_gun_hit", "n_llp"),
                                "share_of_fakes_with_gun_hit": ("n_fake_ge5_gun_hit", "n_fake_ge5"),
                                "host_prompt_efficiency": ("host_m_prompt_target", "host_n_prompt_target"),
                                "host_fake_rate": ("host_n_fake_ge5", "host_n_tracks_ge5")}.items():
            v, s = _ratio_boot(q, num, den, idx)
            r[key] = {"value": v, "sd_boot": s, "num": int(q[num].sum()), "den": int(q[den].sum())}
        if cond == "full":
            sc = q.host_trackml_score.dropna()
            r["host_trackml_score"] = {"value": float(sc.mean()) if len(sc) else float("nan"), "events": int(len(sc))}
        r["llp_rows_per_event"] = float(q.n_llp_rows.mean()) if len(q) else float("nan")
        if twins is not None:
            tw = q.event.map(lambda e: twins.get(_host_of(e)))
            if tw.isna().any():
                raise KeyError(f"no twin status for hosts {sorted(set(q.event[tw.isna()].map(_host_of)))[:5]}")
            by = {}
            for flag, g in q.groupby(tw.astype(bool)):
                g = g.reset_index(drop=True)
                gi = _boot_index(len(g))
                e = {"events": int(len(g))}
                for key, (num, den) in {"host_prompt_efficiency": ("host_m_prompt_target", "host_n_prompt_target"),
                                        "host_fake_rate": ("host_n_fake_ge5", "host_n_tracks_ge5")}.items():
                    v, s = _ratio_boot(g, num, den, gi)
                    e[key] = {"value": v, "sd_boot": s, "num": int(g[num].sum()), "den": int(g[den].sum())}
                if cond == "full":
                    sc = g.host_trackml_score.dropna()
                    e["host_trackml_score"] = float(sc.mean()) if len(sc) else float("nan")
                by["twin" if flag else "no_twin"] = e
            r["by_hs_twin"] = by
        out[c["name"]] = r
    if len(loss):
        lo = loss.copy()
        lo["population"] = np.select([lo.pp, lo.pd], ["P_P", "P_D"], "other gun targets")
        tabs = {}
        for pop in ("P_P", "P_D", "other gun targets", "all gun targets"):
            g = lo if pop == "all gun targets" else lo[lo.population == pop]
            t = pd.crosstab(g.eta_sign, g.outcome) if len(g) else pd.DataFrame()
            tabs[pop] = {"n": int(len(g)),
                         "outcome": {str(o): int(n) for o, n in g.outcome.value_counts().sort_index().items()},
                         "by_eta_sign": {str(s): {str(o): int(t.loc[s, o]) for o in t.columns} for s in t.index}}
        out["loss_stage"] = {"config": configs[int(lo.config.iloc[0])]["name"], **tabs}
    return out


def merge(out: str, twins: str | None = None) -> dict:
    dirs = sorted(glob.glob(os.path.join(out, "blocks", "b*of*")))
    if not dirs:
        raise SystemExit(f"{out}: no blocks")
    recs = [json.load(open(os.path.join(d, "summary.json"))) for d in dirs]
    keys = {json.dumps([r["set"], r["condition"], r["configs"], r["root"], r["inputs"].get("kind")], sort_keys=True)
            for r in recs}
    if len(keys) != 1 or len(dirs) != recs[0]["n_blocks"]:
        raise SystemExit(f"{out}: blocks differ in set, condition, configurations or inputs, or are missing")
    want = B.set_ids(recs[0]["set"])
    pe = pd.concat([pd.read_parquet(os.path.join(d, "per_event.parquet")) for d in dirs], ignore_index=True)
    loss = pd.concat([pd.read_parquet(os.path.join(d, "loss.parquet")) for d in dirs], ignore_index=True)
    per = pe.groupby("config").event.agg(lambda e: sorted(int(x) for x in e))
    if len(per) != len(recs[0]["configs"]) or any(v != want for v in per):
        raise SystemExit(f"{out}: blocks do not hold every event of {recs[0]['set']} once per configuration")
    if os.path.exists(os.path.join(out, "summary.json")):
        raise SystemExit(f"{out} holds a merged summary already")
    pe.to_parquet(os.path.join(out, "per_event.parquet"), index=False)
    loss.to_parquet(os.path.join(out, "loss.parquet"), index=False)
    rec = {**{k: recs[0][k] for k in ("set", "k", "condition", "root", "configs", "loss_config", "merge_model")},
           "inputs": recs[0]["inputs"], "events": want, "n_events": len(want), "n_blocks": recs[0]["n_blocks"],
           "blocks": [{k: r[k] for k in ("block", "elapsed_s", "slurm_job", "host", "git", "code_sha256")}
                      for r in recs],
           "summary": summarize(pe, loss, recs[0]["configs"], _load_twins(twins), recs[0]["condition"]),
           "git": B._git_head(), "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    if twins:
        rec["twins"] = {"path": os.path.abspath(twins), "sha256": B.sha256_file(twins)}
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(rec, fh, indent=1, default=_json_default)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--set", required=True)
    r.add_argument("--cond", required=True, choices=B.CONDITIONS_H)
    r.add_argument("--out", required=True)
    r.add_argument("--twins", default=None)
    r.add_argument("--block", type=int, default=0)
    r.add_argument("--n-blocks", type=int, default=1)
    r.add_argument("--workers", type=int, default=1)
    for x in ("scores", "grid", "sealed", "expect", "root"):
        r.add_argument(f"--{x}", default=None)
    m = sub.add_parser("merge")
    m.add_argument("--out", required=True)
    m.add_argument("--twins", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        rec = run(a.set, a.cond, a.out, a.twins, a.block, a.n_blocks, a.workers, a.scores, a.grid, a.sealed,
                  a.expect, a.root)
    else:
        rec = merge(a.out, a.twins)
    print(json.dumps({k: rec[k] for k in ("set", "condition", "n_events")}, default=_json_default))


if __name__ == "__main__":
    main()
