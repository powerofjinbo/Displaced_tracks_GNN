"""Addendum G step T1 on the extension `generic_iso_x1` (truth only; T1 deviation 13 in RESULTS.md,
docs/new_samples_spec.md section 9, results/new_samples_sufficiency_v1.md sections 6.1-6.4). Committed before the
extension exists; no model reads any of its events here or before this step and the ledger entry are committed.

The extension holds events 8000-23999 of the isolated component (component string `iso`), stored as 32 uncompressed
tars `generic_iso_x1/events_<first>_<last>.tar` of `<event>.npz` members (read with hnl.data.shards) and the truth
sidecars `generic_iso_x1_truth/{truth,gun_truth}_<first>.npz`. The outputs of hnl.eval.addendum_g_t1 for the T1
samples are not touched; this module writes new files only.

    repro    reproduction check before production (spec section 9): chunk 7500-7999 of generic_iso_v1 regenerated
             with the extension's code (slurm/acts/produce.sbatch, PACK=tar) into scratch must equal the stored events
             and sidecars array by array (dtype, shape and bytes); and the reader identity test of the sufficiency
             report 6.1: the 500 stored npz files packed as members of a scratch tar read back through
             hnl.data.shards give the same bytes and arrays as the files.
    scan     checksums of every tar and sidecar; the shard structure (32 tars, contiguous 8000-23999, one member per
             event, nothing else in the directories); per event, through the shard reader, the format invariants and
             the corruption rule of hnl.eval.addendum_g_t1 (an unreadable member is listed and dropped); the
             per-particle truth table with the populations of hnl.eval.addendum_g_t1 (P_G: gun targets with exact
             |d0| >= 10 mm or r_v >= 100 mm). Also a regression of the shared code against the committed T1 truth
             table of generic_iso_v1 (events 0-499 and 7500-7999 must give identical rows).
    summary  split lists by the hash rule of addendum G section 3 with component "iso" (splits/generic_iso_x1_
             {train,val,test}.txt, dropped events removed); the union test list splits/generic_iso_union_test.txt
             (the T1 test list splits/generic_iso_test.txt plus the extension's test events, sorted by event number);
             target counts and P_G per split; the comparison with generic_iso_v1 below; the shared-collision check of
             T1 deviation 14 for the extension; results/addendum_g_t1_x1_v1.{md,csv} and
             results/addendum_g_t1_x1_compare_v1.csv.

Comparison with generic_iso_v1 (fixed here, before the extension exists). Both samples come from the same generator
settings, so every difference must be consistent with chance. All events of both samples (all splits; truth only),
dropped events removed. Nine quantities:
  1. P_G targets per event (mean)                       6. species shares of the P_G targets (e, mu, pi, K, p)
  2. r_v of the P_G targets (distribution)              7. hits per P_G target (distribution)
  3. exact |d0| of the P_G targets (distribution)       8. targets per event (all targets; mostly ttbar)
  4. pT of the P_G targets (distribution)               9. hits per target (all targets)
  5. eta of the P_G targets (distribution)
Statistic: difference of the per-event means (1, 8), Kolmogorov-Smirnov distance (2-5, 7, 9), chi-square of the 2 x 5
table (6). Targets of one event are correlated (the two daughters of an LLP share r_v and species), so the null
distribution permutes whole events between the two samples (B = 10,000 permutations, numpy default_rng(20261014));
p = (1 + #{T_perm >= T_obs}) / (B + 1). Rule: a difference beyond expectation is a p-value below 0.01 / 9 (Bonferroni,
family-wise 1 %); then the verdict is STOP and no model runs on the extension. The number of p-values below 0.05 is
reported beside it (0.45 expected). Control (reported, not part of the rule): the same nine tests between
generic_iso_v1 events 0-3999 and 4000-7999.

    python -m hnl.eval.addendum_g_t1_x1 repro --repro-dir DIR --pack-dir DIR      (sbatch, after the scratch chunk)
    python -m hnl.eval.addendum_g_t1_x1 scan [--workers 16]                         (sbatch)
    python -m hnl.eval.addendum_g_t1_x1 summary [--jobs JOBID,...]                  (sbatch)
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import glob
import io
import json
import os
import re
import subprocess
import tarfile

import numpy as np
import pandas as pd

from hnl.data import npz, shards
from hnl.eval import addendum_g_t1 as T1

X1 = npz.Sample("generic_iso_x1", "generic_iso_x1", 16000, "generic", False, root=npz.GENERATED_ROOT)
V1 = npz.GENERATED["generic_iso"]
X1_FIRST, X1_LAST, CHUNK = 8000, 23999, 500
COMPONENT = "iso"
X1_DIR = os.path.join(npz.GENERATED_ROOT, X1.dirname)
X1_TRUTH = X1_DIR + "_truth"
V1_DIR = os.path.join(npz.GENERATED_ROOT, V1.dirname)
OUT_MD = os.path.join(T1.REPO, "results", "addendum_g_t1_x1_v1.md")
OUT_CSV = os.path.join(T1.REPO, "results", "addendum_g_t1_x1_v1.csv")
OUT_CMP = os.path.join(T1.REPO, "results", "addendum_g_t1_x1_compare_v1.csv")
REPRO_JSON = os.path.join(T1.OUT_REPO, "x1_repro_check.json")
SCAN_JSON = os.path.join(T1.OUT_REPO, "scan_generic_iso_x1.json")
N_PERM = 10_000
PERM_SEED = 20261014
N_TESTS = 9
ALPHA_FW = 0.01
SPECIES = (11, 13, 211, 321, 2212)


def chunks() -> list[tuple[int, int]]:
    return [(f, f + CHUNK - 1) for f in range(X1_FIRST, X1_LAST + 1, CHUNK)]


def same(a: np.ndarray, b: np.ndarray) -> bool:
    """Bitwise identity of two arrays (dtype, shape, bytes; NaN payloads included)."""
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def compare_arrays(x: dict, y: dict) -> list[str]:
    """Keys whose arrays differ (or are missing on one side)."""
    out = sorted(set(x) ^ set(y))
    for k in sorted(set(x) & set(y)):
        if not same(np.asarray(x[k]), np.asarray(y[k])):
            out.append(k)
    return out


def _load_npz(path: str) -> dict:
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


# ------------------------------------------------------------------------------------------------- repro
def cmd_repro(a) -> None:
    first, n = a.first, a.events
    last = first + n - 1
    res = {"stored": os.path.relpath(V1_DIR, npz.GENERATED_ROOT), "repro_dir": a.repro_dir, "first": first,
           "events": n, "git": T1._git(), "slurm_job": os.environ.get("SLURM_JOB_ID")}
    # 1. regenerated events (tar, through the shard reader) against the stored files
    diffs, n_same = [], 0
    got = dict(shards.iter_events_from(a.repro_dir, range(first, last + 1)))
    if sorted(got) != list(range(first, last + 1)):
        raise SystemExit(f"repro tar holds events {min(got)}..{max(got)} ({len(got)}), expected {first}..{last}")
    for i in range(first, last + 1):
        ref = _load_npz(os.path.join(V1_DIR, f"{i}.npz"))
        d = compare_arrays(ref, got[i])
        if d:
            diffs.append({"event": i, "keys": d})
        else:
            n_same += 1
    res["events_identical"] = n_same
    res["events_different"] = diffs[:20]
    res["n_events_different"] = len(diffs)
    # 2. truth sidecars
    side = {}
    for stem in ("truth", "gun_truth"):
        ref = _load_npz(os.path.join(V1_DIR + "_truth", f"{stem}_{first}.npz"))
        new = _load_npz(os.path.join(a.repro_dir + "_truth", f"{stem}_{first}.npz"))
        side[stem] = {"keys": len(ref), "different": compare_arrays(ref, new)}
    res["sidecars"] = side
    # 3. reader identity: the stored files packed into a scratch tar, read back through hnl.data.shards
    os.makedirs(a.pack_dir, exist_ok=True)
    tar = os.path.join(a.pack_dir, shards.shard_name(first, last))
    if os.path.exists(tar):
        os.remove(tar)
    raw = {}
    with tarfile.open(tar, "w", format=tarfile.USTAR_FORMAT) as tf:
        for i in range(first, last + 1):
            with open(os.path.join(V1_DIR, f"{i}.npz"), "rb") as f:
                raw[i] = f.read()
            ti = tarfile.TarInfo(f"{i}.npz")
            ti.size = len(raw[i])
            ti.mtime, ti.mode, ti.uid, ti.gid, ti.uname, ti.gname = 0, 0o444, 0, 0, "", ""
            tf.addfile(ti, io.BytesIO(raw[i]))
    shards.list_shards.cache_clear()
    rd = {"events": 0, "bytes_identical": 0, "arrays_identical_load_event_from": 0, "arrays_identical_iter": 0}
    it = dict(shards.iter_events_from(a.pack_dir, range(first, last + 1)))
    path = shards.shard_for(a.pack_dir, first)
    mem = shards._members(path)
    with open(path, "rb") as f:
        for i in range(first, last + 1):
            rd["events"] += 1
            ti = mem[f"{i}.npz"]
            f.seek(ti.offset_data)
            rd["bytes_identical"] += int(f.read(ti.size) == raw[i])
            ref = _load_npz(os.path.join(V1_DIR, f"{i}.npz"))
            rd["arrays_identical_load_event_from"] += int(not compare_arrays(ref, shards.load_event_from(a.pack_dir, i)))
            rd["arrays_identical_iter"] += int(not compare_arrays(ref, it[i]))
    os.remove(tar)
    res["reader_identity"] = rd
    ok = (len(diffs) == 0 and n_same == n and all(not v["different"] for v in side.values())
          and rd["bytes_identical"] == rd["arrays_identical_load_event_from"] == rd["arrays_identical_iter"] == n)
    res["pass"] = bool(ok)
    res["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    os.makedirs(T1.OUT_REPO, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    if not ok:
        raise SystemExit("reproduction check FAILED: production must not start")


# ------------------------------------------------------------------------------------------------- scan
def structure() -> dict:
    """The expected files and nothing else; shard ranges and members."""
    want_tars = [shards.shard_name(f, l) for f, l in chunks()]
    want_side = sorted([f"truth_{f}.npz" for f, _ in chunks()] + [f"gun_truth_{f}.npz" for f, _ in chunks()])
    have_tars, have_side = sorted(os.listdir(X1_DIR)), sorted(os.listdir(X1_TRUTH))
    out = {"missing": sorted(set(want_tars) - set(have_tars)) + sorted(set(want_side) - set(have_side)),
           "unexpected": sorted(set(have_tars) - set(want_tars)) + sorted(set(have_side) - set(want_side))}
    if out["missing"] or out["unexpected"]:
        raise SystemExit(f"generic_iso_x1 directory content: {out}")
    sh = shards.list_shards(X1_DIR)
    if [(f, l) for f, l, _ in sh] != chunks():
        raise SystemExit("shard ranges are not the 32 chunks of 8000-23999")
    bad = []
    for f, l, path in sh:
        names = sorted(shards._members(path))
        if names != sorted(f"{i}.npz" for i in range(f, l + 1)):
            bad.append(os.path.basename(path))
    if bad:
        raise SystemExit(f"shards whose members do not match their range: {bad}")
    out.update({"tars": len(sh), "sidecars": len(want_side), "members": sum(len(shards._members(p)) for *_, p in sh)})
    return out


def scan_shard(args):
    first, last = args
    rows, parts, unread = [], [], []
    for i in range(first, last + 1):
        try:
            ev = shards.load_event_from(X1_DIR, i, check=True)
        except Exception as e:          # unreadable member: listed and dropped (addendum G section 3)
            unread.append({"event": i, "error": repr(e)[:300]})
            continue
        r, p = T1.scan_loaded(ev, i)
        rows.append(r)
        parts.append(p)
    return rows, parts, unread


def _scan_v1(i: int):
    return T1.scan_loaded(_load_npz(os.path.join(V1_DIR, f"{i}.npz")), i)


def regression(workers: int) -> dict:
    """The shared code (scan_loaded, truth_table) on generic_iso_v1 events 0-499 and 7500-7999 must reproduce the
    committed T1 truth table rows of these events."""
    meta = json.load(open(os.path.join(T1.OUT_REPO, "scan_generic_iso.json")))
    if T1.sha256_file(meta["truth_table"]) != meta["truth_table_sha256"]:
        raise SystemExit("the T1 truth table of generic_iso_v1 does not match its recorded sha256")
    ev = list(range(0, 500)) + list(range(7500, 8000))
    with cf.ProcessPoolExecutor(max(workers, 1)) as ex:
        res = list(ex.map(_scan_v1, ev, chunksize=10))
    parts = pd.concat([r[1] for r in res], ignore_index=True)
    t, _ = T1.truth_table(V1, parts, True, None)
    ref = pd.read_parquet(meta["truth_table"])
    ref = ref[ref.event.isin(ev)]
    key = ["event", "particle_id"]
    t = t.sort_values(key).reset_index(drop=True)
    ref = ref.sort_values(key).reset_index(drop=True)
    ok = list(t.columns) == list(ref.columns) and t.equals(ref)
    diff = []
    if not ok:
        diff = [c for c in ref.columns if c not in t.columns or len(t) != len(ref) or str(t[c].dtype) != str(ref[c].dtype)
                or not t[c].equals(ref[c])]
    return {"events": len(ev), "rows": len(t), "rows_ref": len(ref), "identical": bool(ok), "columns_differing": diff}


def cmd_scan(a) -> None:
    os.makedirs(T1.OUT_DATA, exist_ok=True)
    os.makedirs(T1.OUT_REPO, exist_ok=True)
    st = structure()
    print(f"[t1x1] structure {st}", flush=True)
    files = sorted(glob.glob(os.path.join(X1_DIR, "*")) + glob.glob(os.path.join(X1_TRUTH, "*")), key=T1._natural)
    with cf.ProcessPoolExecutor(a.workers) as ex:
        sums = list(ex.map(T1.sha256_file, files))
    with open(os.path.join(T1.OUT_REPO, f"{X1.dirname}.sha256"), "w") as f:
        for h, p in zip(sums, files):
            f.write(f"{h}  {os.path.relpath(p, npz.GENERATED_ROOT)}\n")
    print(f"[t1x1] {len(files)} files checksummed", flush=True)
    with cf.ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(scan_shard, chunks()))
    ev = pd.DataFrame([r for x in res for r in x[0]])
    parts = pd.concat([p for x in res for p in x[1]], ignore_index=True)
    unread = [u for x in res for u in x[2]]
    for u in unread:
        ev = pd.concat([ev, pd.DataFrame([{"event": u["event"], "format_ok": False, "corrupted": False,
                                           "dropped": True, "unreadable": True}])], ignore_index=True)
    ev["unreadable"] = ev.get("unreadable", pd.Series(False, index=ev.index)).fillna(False).astype(bool)
    ev = ev.sort_values("event").reset_index(drop=True)
    if sorted(ev.event) != list(range(X1_FIRST, X1_LAST + 1)):
        raise SystemExit("event table does not cover 8000-23999")
    print(f"[t1x1] {len(ev)} events, {int(ev.dropped.sum())} dropped ({len(unread)} unreadable), "
          f"{len(parts)} particles with hits", flush=True)
    t, chk = T1.truth_table(X1, parts, True, None)
    # sidecar-level checks
    tr = T1.load_sidecar(X1, "truth")
    g = T1.load_sidecar(X1, "gun_truth")
    xm, comp = set(), set()
    for f in glob.glob(os.path.join(X1_TRUTH, "truth_*.npz")):
        with np.load(f) as z:
            xm.add(str(z["xmode"]))
    for f in glob.glob(os.path.join(X1_TRUTH, "gun_truth_*.npz")):
        with np.load(f) as z:
            comp.add(str(z["component"]))
    chk.update({
        "sidecar_xmode": sorted(xm), "gun_component": sorted(comp),
        "sidecar_events": int(len(np.unique(tr["e_event"]))),
        "sidecar_event_range": [int(tr["e_event"].min()), int(tr["e_event"].max())],
        "gun_events": int(len(np.unique(g["event"]))),
        "gun_particles_per_event": sorted(set(np.unique(g["event"], return_counts=True)[1].tolist())),
        "dropped_duplicate_hits": int(tr["e_n_dup"].sum()), "measurements": int(tr["e_n_meas"].sum()),
        "pull_abs_gt5": int(tr["e_res_abs_gt5"].sum()),
    })
    if chk["sidecar_xmode"] != ["measured"] or chk["gun_component"] != ["iso"] or chk["sidecar_events"] != 16000:
        raise SystemExit(f"sidecar checks failed: {chk}")
    reg = regression(a.workers)
    print(f"[t1x1] regression on generic_iso_v1: {reg}", flush=True)
    if not reg["identical"]:
        raise SystemExit("the shared T1 code does not reproduce the committed generic_iso_v1 truth table")
    tpath = os.path.join(T1.OUT_DATA, f"truth_{X1.key}.parquet")
    epath = os.path.join(T1.OUT_DATA, f"events_{X1.key}.parquet")
    t.to_parquet(tpath, index=False)
    ev.to_parquet(epath, index=False)
    meta = {"sample": X1.key, "dir": X1.dirname, "n_files": len(files), "structure": st, "n_events": len(ev),
            "dropped": ev.loc[ev.dropped, "event"].astype(int).tolist(), "unreadable": unread, "checks": chk,
            "regression_generic_iso_v1": reg,
            "truth_table": tpath, "truth_table_sha256": T1.sha256_file(tpath),
            "events_table": epath, "events_table_sha256": T1.sha256_file(epath),
            "git": T1._git(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    with open(SCAN_JSON, "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))


# ------------------------------------------------------------------------------------------------- comparison
def _ks_stat_sorted(lab_sorted: np.ndarray, ends: np.ndarray) -> float:
    """KS distance from the sample labels (True = extension) of the values in sorted order; ends: last index of each
    group of equal values."""
    c1 = np.cumsum(lab_sorted)
    n1 = c1[-1]
    n0 = len(lab_sorted) - n1
    c0 = np.arange(1, len(lab_sorted) + 1) - c1
    return float(np.max(np.abs(c1[ends] / n1 - c0[ends] / n0)))


def perm_ks(values: np.ndarray, ev_idx: np.ndarray, lab: np.ndarray, perms: np.ndarray) -> tuple[float, float]:
    """KS distance between the targets of the two samples; null by permuting event labels (perms: [B, events] bool,
    the permuted label vectors)."""
    o = np.argsort(values, kind="stable")
    v = values[o]
    ends = np.r_[np.flatnonzero(v[1:] != v[:-1]), len(v) - 1]
    e = ev_idx[o]
    t_obs = _ks_stat_sorted(lab[e], ends)
    t = np.array([_ks_stat_sorted(L[e], ends) for L in perms])
    return t_obs, (1 + int((t >= t_obs - 1e-12).sum())) / (len(perms) + 1)


def perm_hist(H: np.ndarray, lab: np.ndarray, perms: np.ndarray, kind: str) -> tuple[float, float]:
    """H [events, K] per-event counts of K categories (ordered). kind ks: KS distance of the ordered categories;
    chi2: chi-square of the 2 x K table. Null by permuting event labels."""
    def stat(L):                       # L [m, events] float
        h1 = L @ H
        h0 = H.sum(0)[None, :] - h1
        if kind == "ks":
            c1 = np.cumsum(h1, 1) / h1.sum(1, keepdims=True)
            c0 = np.cumsum(h0, 1) / h0.sum(1, keepdims=True)
            return np.max(np.abs(c1 - c0), 1)
        tot = h1 + h0
        n1, n0 = h1.sum(1, keepdims=True), h0.sum(1, keepdims=True)
        e1, e0 = tot * n1 / (n1 + n0), tot * n0 / (n1 + n0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.nansum((h1 - e1) ** 2 / e1 + (h0 - e0) ** 2 / e0, 1)
    H = H.astype(np.float64)
    t_obs = float(stat(lab[None, :].astype(np.float64))[0])
    ts = np.concatenate([stat(perms[k:k + 500].astype(np.float64)) for k in range(0, len(perms), 500)])
    return t_obs, (1 + int((ts >= t_obs - 1e-9).sum())) / (len(perms) + 1)


def perm_mean(x: np.ndarray, lab: np.ndarray, perms: np.ndarray) -> tuple[float, float]:
    def stat(L):
        L = L.astype(np.float64)
        n1 = L.sum(-1)
        return np.abs(L @ x / n1 - (x.sum() - L @ x) / (len(x) - n1))
    t_obs = float(stat(lab))
    ts = np.concatenate([stat(perms[k:k + 500]) for k in range(0, len(perms), 500)])
    return t_obs, (1 + int((ts >= t_obs - 1e-12).sum())) / (len(perms) + 1)


def _mean_se(x: np.ndarray) -> tuple[float, float]:
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


def _ratio_se(num: np.ndarray, den: np.ndarray) -> tuple[float, float]:
    """Event-clustered mean of a per-target quantity: sum(num) / sum(den) over events, with its delta-method SE."""
    r = num.sum() / den.sum()
    n = len(num)
    se = np.sqrt(np.sum((num - r * den) ** 2) / (n * (n - 1))) / den.mean()
    return float(r), float(se)


def compare(t_v1: pd.DataFrame, t_x1: pd.DataFrame, ev_v1: np.ndarray, ev_x1: np.ndarray) -> pd.DataFrame:
    events = np.r_[ev_v1, ev_x1]
    lab = np.r_[np.zeros(len(ev_v1), bool), np.ones(len(ev_x1), bool)]
    pos = pd.Series(np.arange(len(events)), index=events)
    t = pd.concat([t_v1[t_v1.event.isin(ev_v1)], t_x1[t_x1.event.isin(ev_x1)]], ignore_index=True)
    tg = t[t.is_target]
    pg = t[t.in_pop]
    e_tg = pos.loc[tg.event].to_numpy()
    e_pg = pos.loc[pg.event].to_numpy()
    rng = np.random.default_rng(PERM_SEED)
    perms = np.array([lab[rng.permutation(len(events))] for _ in range(N_PERM)])     # [B, events] permuted labels
    n_pg = np.bincount(e_pg, minlength=len(events)).astype(np.float64)
    n_tg = np.bincount(e_tg, minlength=len(events)).astype(np.float64)
    sp = np.searchsorted(SPECIES, np.abs(pg.pdg.to_numpy()))
    if not np.array_equal(np.array(SPECIES)[np.clip(sp, 0, 4)], np.abs(pg.pdg.to_numpy())):
        raise SystemExit("a P_G target outside the five gun species")
    H_sp = np.zeros((len(events), len(SPECIES)))
    np.add.at(H_sp, (e_pg, sp), 1)

    def hits_hist(e, nh):
        lv = np.unique(nh)
        H = np.zeros((len(events), len(lv)))
        np.add.at(H, (e, np.searchsorted(lv, nh)), 1)
        return H
    rows = []

    def add(name, kind, t_obs, p, v1, x1, unit=""):
        rows.append({"quantity": name, "statistic": kind, "T_obs": t_obs, "p_perm": p,
                     "v1": v1, "x1": x1, "unit": unit})
    m = lambda x, sel: _mean_se(x[sel])  # noqa: E731
    t_obs, p = perm_mean(n_pg, lab, perms)
    add("P_G targets per event", "|mean difference|", t_obs, p, m(n_pg, ~lab), m(n_pg, lab))
    for col, name, unit in (("r_v", "r_v of P_G targets", "mm"), ("d0", "exact |d0| of P_G targets", "mm"),
                            ("pt", "pT of P_G targets", "GeV"), ("eta", "eta of P_G targets", "")):
        x = pg[col].to_numpy(np.float64)
        t_obs, p = perm_ks(x, e_pg, lab, perms)
        num = np.bincount(e_pg, weights=x, minlength=len(events))
        add(name, "KS distance", t_obs, p, _ratio_se(num[~lab], n_pg[~lab]), _ratio_se(num[lab], n_pg[lab]), unit)
    t_obs, p = perm_hist(H_sp, lab, perms, "chi2")
    sh = lambda sel: tuple(float(v) for v in H_sp[sel].sum(0) / H_sp[sel].sum())  # noqa: E731
    add("species shares of P_G targets (e, mu, pi, K, p)", "chi-square (4 dof)", t_obs, p, sh(~lab), sh(lab))
    nh = pg.n_hits.to_numpy()
    t_obs, p = perm_hist(hits_hist(e_pg, nh), lab, perms, "ks")
    num = np.bincount(e_pg, weights=nh, minlength=len(events))
    add("hits per P_G target", "KS distance", t_obs, p, _ratio_se(num[~lab], n_pg[~lab]), _ratio_se(num[lab], n_pg[lab]))
    t_obs, p = perm_mean(n_tg, lab, perms)
    add("targets per event (all)", "|mean difference|", t_obs, p, m(n_tg, ~lab), m(n_tg, lab))
    nh = tg.n_hits.to_numpy()
    t_obs, p = perm_hist(hits_hist(e_tg, nh), lab, perms, "ks")
    num = np.bincount(e_tg, weights=nh, minlength=len(events))
    add("hits per target (all)", "KS distance", t_obs, p, _ratio_se(num[~lab], n_tg[~lab]), _ratio_se(num[lab], n_tg[lab]))
    out = pd.DataFrame(rows)
    assert len(out) == N_TESTS
    out["threshold"] = ALPHA_FW / N_TESTS
    out["beyond_expectation"] = out.p_perm < out.threshold
    return out


def shared_collisions(tables: dict[str, pd.DataFrame]) -> dict:
    """Events whose ttbar collision vertex coincides exactly with that of another event (T1 deviation 14). The
    collision vertex of an event is the most common production vertex of its generator particles with hits (barcode
    generation 0, sub-particle 0, secondary vertex below the gun's 1000); usually these carry secondary vertex 0, but
    ACTS can number the collision vertex 1 (generic_iso_x1 event 22550)."""
    keys = []
    for name, t in tables.items():
        p = t[(t.gen == 0) & (t["sub"] == 0) & (t.vsec < 1000)]
        c = p.groupby(["event", "p_vx", "p_vy", "p_vz"]).size().reset_index(name="n")
        v = c.sort_values(["event", "n"], ascending=[True, False], kind="stable").drop_duplicates("event")
        v = v.drop(columns="n")
        v["sample"] = name
        keys.append(v)
    v = pd.concat(keys, ignore_index=True)
    dup = v[v.duplicated(["p_vx", "p_vy", "p_vz"], keep=False)]
    groups = [sorted(f"{s} {int(e)}" for s, e in zip(g["sample"], g.event))
              for _, g in dup.groupby(["p_vx", "p_vy", "p_vz"])]
    x1 = [g for g in groups if any(s.startswith("generic_iso_x1 ") for s in g)]
    return {"events_with_vertex": {k: int((v["sample"] == k).sum()) for k in tables},
            "coinciding_groups": len(groups), "groups_with_extension": x1, "all_groups": groups}


# ------------------------------------------------------------------------------------------------- summary
def _read_list(path: str) -> np.ndarray:
    return np.array([int(x) for x in open(path) if x.strip() and not x.startswith("#")], dtype=np.int64)


def _split_row(sample: str, split: str, t: pd.DataFrame, idx) -> dict:
    tt = t[t.event.isin(idx)]
    return {"sample": sample, "split": split, "events": len(idx), "targets": int(tt.is_target.sum()),
            "population": int(tt.in_pop.sum()), "prompt": int(tt.prompt.sum()),
            "gun_particles_with_hits": int(tt.gun.sum()), "gun_targets": int((tt.gun & tt.is_target).sum()),
            "population_rv_ge_100": int((tt.in_pop & (tt.r_v >= 100)).sum()),
            "population_rv_lt_100": int((tt.in_pop & (tt.r_v < 100)).sum()),
            "population_d0_ge_300": int((tt.in_pop & (tt.d0 >= 300)).sum())}


def _sacct(jobs: list[str]) -> pd.DataFrame:
    if not jobs:
        return pd.DataFrame()
    r = subprocess.run(["sacct", "-X", "-n", "-P", "-j", ",".join(jobs),
                        "--format=JobID,JobName,NCPUS,ElapsedRaw,State,NodeList"], capture_output=True, text=True)
    rows = [ln.split("|") for ln in r.stdout.splitlines() if ln.strip()]
    d = pd.DataFrame(rows, columns=["job", "name", "ncpus", "elapsed_s", "state", "node"])
    d["cpu_h"] = d.ncpus.astype(int) * d.elapsed_s.astype(int) / 3600.0
    return d


def cmd_summary(a) -> None:
    m1 = json.load(open(os.path.join(T1.OUT_REPO, "scan_generic_iso.json")))
    mx = json.load(open(SCAN_JSON))
    md = json.load(open(os.path.join(T1.OUT_REPO, "scan_generic_dense.json")))
    for m in (m1, mx, md):
        if T1.sha256_file(m["truth_table"]) != m["truth_table_sha256"]:
            raise SystemExit(f"{m['truth_table']} does not match its recorded sha256")
    t1 = pd.read_parquet(m1["truth_table"])
    tx = pd.read_parquet(mx["truth_table"])
    e1 = pd.read_parquet(m1["events_table"])
    ex = pd.read_parquet(mx["events_table"])
    drop_x = np.sort(ex.loc[ex.dropped, "event"].to_numpy())
    drop_1 = np.sort(e1.loc[e1.dropped, "event"].to_numpy())
    # split lists of the extension (hash rule, component "iso")
    lab = {i: T1.split_of(COMPONENT, i) for i in range(X1_FIRST, X1_LAST + 1)}
    splits = {}
    for sp in ("train", "val", "test"):
        idx = np.setdiff1d([i for i, s in lab.items() if s == sp], drop_x)
        splits[sp] = idx
        T1.write_split(os.path.join(T1.SPLITS_DIR, f"generic_iso_x1_{sp}.txt"),
                       f"# generic_iso_x1 {sp}: {len(idx)} events (event numbers 8000-23999 of the isolated component, "
                       f"T1 deviation 13); preregistration addendum G section 3: first 8 hex digits of "
                       f"sha256('{T1.SALT}:{COMPONENT}:<event>') mod 10 (0-5 train, 6 val, 7-9 test); dropped events "
                       f"removed (results/addendum_g_t1_x1_v1.md). No model reads the train or val events under "
                       f"addendum G (deviation 13c).\n", idx)
    # the T1 test list, checked against the rule, and the union
    v1_test = _read_list(os.path.join(T1.SPLITS_DIR, "generic_iso_test.txt"))
    v1_rule = np.setdiff1d([i for i in range(V1.n_events) if T1.split_of(COMPONENT, i) == "test"], drop_1)
    if not np.array_equal(np.sort(v1_test), v1_rule):
        raise SystemExit("splits/generic_iso_test.txt does not follow the hash rule")
    union = np.union1d(v1_test, splits["test"])
    if len(union) != len(v1_test) + len(splits["test"]):
        raise SystemExit("the T1 and extension test lists overlap")
    T1.write_split(os.path.join(T1.SPLITS_DIR, "generic_iso_union_test.txt"),
                   f"# generic_iso union test: {len(union)} events = {len(v1_test)} of splits/generic_iso_test.txt "
                   f"(generic_iso_v1, event numbers 0-7999) + {len(splits['test'])} of splits/generic_iso_x1_test.txt "
                   f"(generic_iso_x1, 8000-23999), sorted by event number; the generic_iso test split of the claims "
                   f"(T1 deviation 13a), evaluated once\n", union)
    rows = [_split_row("generic_iso_x1", sp, tx, splits[sp]) for sp in ("train", "val", "test")]
    rows.append(_split_row("generic_iso_v1", "test", t1, v1_test))
    tu = pd.concat([t1[t1.event.isin(v1_test)], tx[tx.event.isin(splits["test"])]], ignore_index=True)
    rows.append(_split_row("generic_iso (union)", "test", tu, union))
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT_CSV, index=False)
    # comparison with generic_iso_v1 (all events, truth only)
    ev_1 = np.setdiff1d(np.arange(V1.n_events), drop_1)
    ev_x = np.setdiff1d(np.arange(X1_FIRST, X1_LAST + 1), drop_x)
    cmp_ = compare(t1, tx, ev_1, ev_x)
    verdict = "PASS" if not cmp_.beyond_expectation.any() else "STOP"
    ctrl = compare(t1, t1, ev_1[ev_1 < 4000], ev_1[ev_1 >= 4000])
    cmp_.assign(v1=cmp_.v1.astype(str), x1=cmp_.x1.astype(str), comparison="generic_iso_v1 vs generic_iso_x1") \
        .pipe(lambda d: pd.concat([d, ctrl.assign(v1=ctrl.v1.astype(str), x1=ctrl.x1.astype(str),
                                                  comparison="control: generic_iso_v1 0-3999 vs 4000-7999")]))\
        .to_csv(OUT_CMP, index=False)
    td = pd.read_parquet(md["truth_table"])
    coll = shared_collisions({"generic_iso_v1": t1, "generic_iso_x1": tx, "generic_dense_v1": td})
    repro = json.load(open(REPRO_JSON)) if os.path.exists(REPRO_JSON) else None
    acct = _sacct([j for j in (a.jobs or "").split(",") if j])
    env = {"max_excess_mm": float(ex.env_max_excess_mm.max()), "hits_outside": int(ex.n_env_bad.sum()),
           "hits_unseen_module": int(ex.n_env_unseen_module.sum()),
           "v1_max_excess_mm": float(e1.env_max_excess_mm.max()), "v1_hits_outside": int(e1.n_env_bad.sum())}
    text = markdown(tab, cmp_, verdict, mx, coll, repro, acct, len(drop_x), ctrl, env)
    with open(OUT_MD, "w") as f:
        f.write(text)
    print(text)
    print(f"verdict: {verdict}")


def _fmt(v, unit=""):
    if isinstance(v, (tuple, list)) and len(v) == 2:
        return f"{v[0]:.4g} +- {v[1]:.2g}{(' ' + unit) if unit else ''}"
    if isinstance(v, (tuple, list)):
        return " / ".join(f"{x:.3f}" for x in v)
    return str(v)


def markdown(tab, cmp_, verdict, mx, coll, repro, acct, n_drop, ctrl, env) -> str:
    sha = mx["truth_table_sha256"][:16]
    L = ["# Addendum G, step T1 on the extension generic_iso_x1 (truth only)", "",
         f"Written {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by `hnl.eval.addendum_g_t1_x1` "
         f"(git {T1._git()[:7]}), before any model has read an event of the extension. The extension is T1 deviation "
         "13 (RESULTS.md): events 8000-23999 of the isolated component, the settings of `docs/new_samples_spec.md` "
         "section 3, base seed 260001 (gun 260008, Fatras and digitisation 260014), section 9 of the spec. "
         "Definitions in the module docstrings of `hnl.eval.addendum_g_t1` and `hnl.eval.addendum_g_t1_x1`; the "
         "T1 outputs of `results/addendum_g_t1_v1.md` are unchanged.", ""]
    L += ["## Production", ""]
    if repro is not None:
        rd = repro["reader_identity"]
        L += [f"- Reproduction check before production (job {repro['slurm_job']}, git {repro['git'][:7]}): chunk "
              f"{repro['first']}-{repro['first'] + repro['events'] - 1} of generic_iso_v1 (base seed 250001) "
              f"regenerated with the extension's code path (`slurm/acts/produce.sbatch`, PACK=tar) into scratch on "
              f"/work: {repro['events_identical']} of {repro['events']} events identical to the stored npz in every "
              f"array (dtype, shape and bytes), sidecars "
              + ", ".join(f"`{k}_{repro['first']}.npz` {v['keys']} arrays, "
                          f"{'identical' if not v['different'] else 'DIFFERENT: ' + ', '.join(v['different'])}"
                          for k, v in repro["sidecars"].items())
              + f". {'PASS' if repro['pass'] else 'FAIL'}.",
              f"- Reader identity (sufficiency report 6.1): the {rd['events']} stored npz files of that chunk packed "
              f"as tar members read back through `hnl.data.shards`: bytes identical {rd['bytes_identical']}, arrays "
              f"identical {rd['arrays_identical_load_event_from']} (`load_event_from`) and "
              f"{rd['arrays_identical_iter']} (`iter_events_from`)."]
    if len(acct):
        L += ["- Jobs (sacct; CPU-h = CPUs x elapsed):", "", "| job | name | CPUs | elapsed (s) | state | node | CPU-h |",
              "|---|---|---|---|---|---|---|"]
        for _, r in acct.iterrows():
            L.append(f"| {r.job} | {r['name']} | {r.ncpus} | {r.elapsed_s} | {r.state} | {r.node} | {r.cpu_h:.2f} |")
        L += ["", f"  Total {acct.cpu_h.sum():.1f} CPU-h; no GPU."]
    st = mx["structure"]
    c = mx["checks"]
    L += ["", "## Files and events", "",
          f"- {st['tars']} tars `generated/generic_iso_x1/events_<first>_<last>.tar` (500 members `<event>.npz` each, "
          f"{st['members']} in all, ranges contiguous 8000-23999) and {st['sidecars']} sidecars in "
          f"`generated/generic_iso_x1_truth/`; nothing else in either directory. Checksums: "
          f"`results/addendum_g_t1/generic_iso_x1.sha256` ({mx['n_files']} files, sha256sum format).",
          f"- Every event read through `hnl.data.shards` with the format invariants and the corruption rule of T1: "
          f"{mx['n_events']} events, {n_drop} dropped"
          + (f" ({', '.join(map(str, mx['dropped']))})" if mx["dropped"] else "")
          + f", {len(mx['unreadable'])} unreadable. Largest excess over the padded Zenodo module envelopes "
          f"{env['max_excess_mm']:.2f} mm (rule 50 mm), {env['hits_outside']} hits outside them, "
          f"{env['hits_unseen_module']} hits on modules unseen in the Zenodo scan (generic_iso_v1: "
          f"{env['v1_max_excess_mm']:.2f} mm, {env['v1_hits_outside']} hits).",
          "- Truth checks: " + ", ".join(f"{k}: {v if not isinstance(v, float) else f'{v:.3g}'}" for k, v in c.items())
          + ".",
          f"- Shared code: on generic_iso_v1 events 0-499 and 7500-7999 the T1 functions reproduce the committed T1 "
          f"truth table ({mx['regression_generic_iso_v1']['rows']} rows, identical: "
          f"{mx['regression_generic_iso_v1']['identical']}).",
          f"- Truth table: `{os.path.relpath(mx['truth_table'], npz.DATA_ROOT)}` sha256 `{sha}`.", ""]
    L += ["## Splits and populations", "",
          "Split lists (addendum G section 3, component `iso`): `splits/generic_iso_x1_{train,val,test}.txt`. Union "
          "test list of the claims (deviation 13a): `splits/generic_iso_union_test.txt` = `splits/generic_iso_test.txt` "
          "(T1, unchanged) + `splits/generic_iso_x1_test.txt`, sorted by event number. The extension's train and val "
          "events are not read by any model under addendum G (deviation 13c); the T3 training list "
          "`splits/generic_train_g7999.txt` is unchanged (13b).", "",
          "| sample | split | events | targets | P_G | prompt reference | gun targets | P_G r_v >= 100 mm | "
          "P_G r_v < 100 mm | P_G abs(d0) >= 300 mm |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in tab.iterrows():
        L.append(f"| {r['sample']} | {r['split']} | {r['events']} | {r['targets']} | {r['population']} | "
                 f"{r['prompt']} | {r['gun_targets']} | {r['population_rv_ge_100']} | {r['population_rv_lt_100']} | "
                 f"{r['population_d0_ge_300']} |")
    t = tab.set_index(["sample", "split"])
    n_u = int(t.loc[("generic_iso (union)", "test"), "population"])
    n_xt = int(t.loc[("generic_iso_x1", "test"), "events"])
    rate = int(t.loc[("generic_iso_v1", "test"), "population"]) / int(t.loc[("generic_iso_v1", "test"), "events"])
    L += ["", f"P_G on the union test split: {n_u} targets "
          f"({int(t.loc[('generic_iso_v1', 'test'), 'population'])} from T1 + "
          f"{int(t.loc[('generic_iso_x1', 'test'), 'population'])} from the extension). Expected before production "
          f"(sufficiency report 6.1): about 4,770 extension test events and about 8,700 P_G targets on the union; "
          f"here {n_xt} extension test events (30 % of 16,000: 4,800 +- 58), and {rate * n_xt:.0f} extension P_G "
          f"targets at the T1 test rate of {rate:.3f} per event.", ""]
    L += ["## Comparison with generic_iso_v1 (all events, truth only)", "",
          f"Rule fixed in the module docstring before the extension existed: event-permutation tests (B = {N_PERM:,}, "
          f"default_rng({PERM_SEED})); a p-value below {ALPHA_FW} / {N_TESTS} = {ALPHA_FW / N_TESTS:.5f} is a "
          "difference beyond expectation and stops the work. Means with event-clustered standard errors; species "
          "shares e / mu / pi / K / p.", "",
          "| quantity | statistic | T | p (permutation) | generic_iso_v1 | generic_iso_x1 |", "|---|---|---|---|---|---|"]
    for _, r in cmp_.iterrows():
        L.append(f"| {r.quantity} | {r.statistic} | {r.T_obs:.4g} | {r.p_perm:.4f} | {_fmt(r.v1, r.unit)} | "
                 f"{_fmt(r.x1, r.unit)} |")
    n05 = int((cmp_.p_perm < 0.05).sum())
    L += ["", f"Smallest p-value {cmp_.p_perm.min():.4f} (threshold {ALPHA_FW / N_TESTS:.5f}); {n05} of {N_TESTS} "
          f"below 0.05 ({0.05 * N_TESTS:.2f} expected). **Verdict: {verdict}**"
          + (" (no difference beyond expectation; the extension joins the generic_iso test split as fixed by "
             "deviation 13)." if verdict == "PASS" else " (a difference beyond expectation: no model runs on the "
                                                          "extension until it is understood)."), "",
          f"Control (not part of the rule): the same tests between generic_iso_v1 events 0-3999 and 4000-7999 give "
          f"p-values " + ", ".join(f"{p:.3f}" for p in ctrl.p_perm) + f" (smallest {ctrl.p_perm.min():.4f}; "
          f"{int((ctrl.p_perm < 0.05).sum())} below 0.05).", ""]
    gx = coll["groups_with_extension"]
    L += ["## Shared ttbar collisions (T1 deviation 14)", "",
          f"Events whose ttbar collision vertex (the most common production vertex of the generator particles with "
          f"hits) equals that of another event, among generic_iso_v1, generic_iso_x1 and generic_dense_v1 "
          f"({', '.join(f'{k} {v}' for k, v in coll['events_with_vertex'].items())} events): "
          f"{coll['coinciding_groups']} groups in all ("
          + ("; ".join(", ".join(g) for g in coll["all_groups"]) if coll["all_groups"] else "none")
          + f"); {len(gx)} involve an extension event"
          + (": " + "; ".join(", ".join(g) for g in gx) if gx else "") + ".", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("repro")
    p.add_argument("--repro-dir", required=True, help="scratch sample dir with events_<first>_<last>.tar")
    p.add_argument("--pack-dir", required=True, help="scratch dir for the reader identity tar")
    p.add_argument("--first", type=int, default=7500)
    p.add_argument("--events", type=int, default=500)
    p.add_argument("--out", default=REPRO_JSON)
    p = sub.add_parser("scan")
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    p = sub.add_parser("summary")
    p.add_argument("--jobs", default="", help="job ids for the CPU-h table (production, reproduction, T1)")
    a = ap.parse_args(argv)
    {"repro": cmd_repro, "scan": cmd_scan, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
