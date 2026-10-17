"""Long-strip sensitivity of addendum H section 7 (truth only): the P_P and P_D counts of an overlay set with the hit
drop rate of the long-strip disks (volumes 16 and 18) raised until the emulation misses 2.6 % of the Fatras crossings
there, the middle of TrackML's 2.2-3.0 % (docs/preregistration_pileup.md sections 2 and 7). The long-strip disks could
not be measured on TrackML and take the pooled strip rate, with which the emulation misses 0.7-1.0 % of the crossings
there (results/pileup_hp/emulation_overlay_v1.md).

The set's own Fatras hits (the WORK directory of slurm/overlay/overlay_set.sbatch: gun/gun_events.json, gun/fatras_in
and gun/fatras_out/simhits.parquet) go through hnl.overlay.writer.gun_event again, unchanged and with the same seeds.
Only the per-hit drop rate of the long-strip disk layers is replaced, in a copy of the emulation tables, by eps_ls. The
emulation draws one uniform number per crossing and one per hit in a fixed order that does not depend on the rates, so
a higher eps_ls removes the same hits as before plus further hits on the long-strip disks; every other hit, every stop
and every cell stays as it was. Each gun event is converted again (hnl.overlay.writer.convert_gun), and its targets,
P_P and P_D are those of hnl.overlay.h1.gun_targets.

eps_ls is the smallest rate, to 1e-6, at which the missed share of the set's long-strip disk crossings reaches the
target (2.6 %), in the metric of the emulation gate that gave TrackML's 2.2-3.0 % and the emulation's 0.7-1.0 %
(hnl.overlay.emulate.miss_rates): a crossing is the run of one particle's hits on one layer (as in
hnl.overlay.emulate); the interior crossings are those with a later kept hit on the same particle; the share is that of
interior long-strip disk crossings without a kept hit.
Check: with the tables unchanged the rerun reproduces every stored gun particle row of the set (n_hits and is_target)
and the sidecar's dropped and stopped hit counts of every event; the run stops otherwise.

    python -m hnl.overlay.longstrip run --set hval --work DIR --out FILE.json [--target 0.026]
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os

import numpy as np
import pandas as pd

LONG_STRIP_DISKS = (16, 18)
TARGET_MISS = 0.026
TOL = 1e-6


def _load(set_name: str, work: str, root: str):
    from hnl.overlay import digitise as D
    from hnl.overlay import emulate as EM
    from hnl.overlay import writer as W
    from hnl.trackml import convert as C
    with open(os.path.join(root, f"meta_{set_name}.json")) as fh:
        meta = json.load(fh)
    with open(os.path.join(root, f"sidecar_{set_name}.json")) as fh:
        side = json.load(fh)
    gun = os.path.join(work, "gun")
    with open(os.path.join(gun, "gun_events.json")) as fh:
        gev = json.load(fh)
    tables_path, lookup_path = meta["emulation_tables"], meta["lookup"]
    if EM.emulation_hash(tables_path) != meta["emulation_hash"]:
        raise ValueError(f"{tables_path}: emulation hash differs from meta_{set_name}.json")
    if C.sha256_file(lookup_path) != meta["lookup_sha256"]:
        raise ValueError(f"{lookup_path}: sha256 differs from meta_{set_name}.json")
    if int(gev["base_seed"]) != int(meta["base_seed"]):
        raise ValueError("the WORK directory's gun is not the stored set's (base seed)")
    want = {str(int(e["overlay_event"])): int(e["fatras_event"]) for e in gev["events"]}
    if want != {str(k): int(v) for k, v in meta["fatras_events"].items()}:
        raise ValueError("the WORK directory's gun events are not the stored set's (overlay ids, Fatras numbers)")
    return {"meta": meta, "side": {int(e["overlay_event"]): e for e in side["events"]}, "gev": gev, "gun": gun,
            "sim": pd.read_parquet(os.path.join(gun, "fatras_out", "simhits.parquet")),
            "tables": EM.Tables.load(tables_path), "tables_path": tables_path, "lookup": D.load_lookup(lookup_path),
            "mods": D.Modules.load(), "det": C.load_detectors(), "W": W, "EM": EM}


def _with_eps(tables, eps_ls: float | None):
    """A copy of the emulation tables with the drop rate of every long-strip disk layer set to eps_ls."""
    if eps_ls is None:
        return tables
    t = copy.copy(tables)
    t.eps = tables.eps.copy()
    codes = [v * 100 + lay for v in LONG_STRIP_DISKS for lay in range(0, 100)]
    t.eps[codes] = np.where(tables.eps[codes] > 0, eps_ls, 0.0)
    return t


def _event_inputs(L: dict, ev: dict):
    """(overlay id, Fatras input particles, sim hits) of one gun event, read once."""
    W = L["W"]
    oid, fe = int(ev["overlay_event"]), int(ev["fatras_event"])
    cache = L.setdefault("cache", {})
    if oid not in cache:
        parts = pd.read_csv(W.FIO_path(L["gun"], fe), dtype={"particle_id": np.uint64})
        s = L["sim"][L["sim"].event == fe].reset_index(drop=True)
        cache[oid] = (parts, s)
    return (oid, *cache[oid])


def _emulation(L: dict, s: pd.DataFrame, parts: pd.DataFrame, oid: int, tables) -> tuple[dict, pd.DataFrame]:
    """The emulation flags of one event as hnl.overlay.writer.gun_event computes them, with the hits' crossings."""
    EM, W = L["EM"], L["W"]
    pinfo = pd.DataFrame({"particle_id": parts.particle_id.to_numpy(np.uint64),
                          "pdg": parts.particle_type.to_numpy(np.int64),
                          "p": np.sqrt(parts.px ** 2 + parts.py ** 2 + parts.pz ** 2).to_numpy()})
    rng = np.random.default_rng([int(L["gev"]["base_seed"]) + 13, oid, W.EMULATION_STREAM])
    vol, lay, mod = L["lookup"].from_geometry_id(s.geometry_id.to_numpy(np.uint64))
    hits = pd.DataFrame({"particle_id": s.particle_id.to_numpy(np.uint64), "volume": vol, "layer": lay,
                         "module": mod, "order": s["index"].to_numpy(),
                         **{c: s[c].to_numpy(np.float64) for c in ("tx", "ty", "tpx", "tpy", "tpz")}})
    em = EM.emulate(hits, pinfo, tables, rng)
    return em, hits


def crossing_misses(hits: pd.DataFrame, em: dict) -> tuple[int, int]:
    """(interior long-strip disk crossings, those without a kept hit), the metric of the emulation gate's miss rate
    against Fatras (hnl.overlay.emulate.miss_rates on stop_table: the share of interior expected crossings, those with
    a later observed hit on the same particle, that hold no observed hit), with the Fatras crossings as the expected
    ones and the kept hits as the observed ones. TrackML's 2.2-3.0 % on the long-strip disks is in this metric."""
    if not len(hits):
        return 0, 0
    pid = hits.particle_id.to_numpy(np.uint64)
    o = np.lexsort((hits.order.to_numpy(), pid))
    code = hits.volume.to_numpy(np.int64)[o] * 100 + hits.layer.to_numpy(np.int64)[o]
    ps = pid[o]
    newp = np.r_[True, ps[1:] != ps[:-1]]
    newc = newp | np.r_[False, code[1:] != code[:-1]]
    cross = np.cumsum(newc) - 1
    part = np.cumsum(newp) - 1
    keep = em["keep"][o]
    last_kept = np.full(int(part[-1]) + 1, -1, np.int64)
    np.maximum.at(last_kept, part[keep], cross[keep])
    df = pd.DataFrame({"c": cross, "part": part, "ls": np.isin(code // 100, LONG_STRIP_DISKS), "keep": keep})
    g = df.groupby("c").agg(part=("part", "first"), ls=("ls", "first"), kept=("keep", "any"))
    g = g[g.ls & (g.index.to_numpy() < last_kept[g.part.to_numpy()])]
    return int(len(g)), int((~g.kept).sum())


def miss_share(L: dict, eps_ls: float | None) -> dict:
    tab = _with_eps(L["tables"], eps_ls)
    n = m = 0
    for ev in L["gev"]["events"]:
        oid, parts, s = _event_inputs(L, ev)
        em, hits = _emulation(L, s, parts, oid, tab)
        a, b = crossing_misses(hits, em)
        n, m = n + a, m + b
    return {"crossings": n, "missed": m, "share": m / n if n else float("nan")}


def solve_eps(L: dict, target: float = TARGET_MISS) -> tuple[float, dict]:
    """Smallest eps_ls (to TOL) whose missed share reaches the target; the share is non-decreasing in eps_ls."""
    lo, hi = 0.0, 1.0
    r_hi = miss_share(L, hi)
    if not r_hi["share"] >= target:
        raise ValueError(f"even eps_ls = 1 misses only {r_hi['share']:.4f} of the long-strip disk crossings")
    while hi - lo > TOL:
        mid = 0.5 * (lo + hi)
        if miss_share(L, mid)["share"] >= target:
            hi = mid
        else:
            lo = mid
    return hi, miss_share(L, hi)


def gun_particles(L: dict, eps_ls: float | None) -> tuple[pd.DataFrame, dict]:
    """Converted gun particles of every event of the set with the given long-strip disk rate, and the per-event
    stopped and dropped hit counts."""
    W = L["W"]
    tab = _with_eps(L["tables"], eps_ls)
    base = int(L["gev"]["base_seed"])
    rows, counts = [], {}
    for ev in L["gev"]["events"]:
        oid, parts, s = _event_inputs(L, ev)
        g = W.gun_event(s, parts, L["lookup"], L["mods"], tab, base, oid)
        gtab, _ = W.convert_gun(g["gz"], oid, L["det"])
        rows.append(gtab["particles"].to_pandas())
        counts[oid] = {"stopped_hits": int(g["emulation"]["stopped"].sum()),
                       "dropped_hits": int(g["emulation"]["dropped"].sum())}
    return pd.concat(rows, ignore_index=True), counts


def counts_of(particles: pd.DataFrame) -> dict:
    from hnl.overlay.h1 import gun_targets
    g = gun_targets(particles)
    return {"gun_particles": int(len(g)), "targets": int(g.target.sum()), "P_P": int(g.pp.sum()),
            "P_D": int(g.pd.sum())}


def check_nominal(L: dict, set_name: str, root: str) -> dict:
    """The rerun with the tables unchanged against the stored set: every gun row's n_hits and is_target, and the
    sidecar's stopped and dropped hits per event."""
    from hnl.trackml.convert import read_table
    p0, c0 = gun_particles(L, None)
    ids = sorted(int(e["overlay_event"]) for e in L["gev"]["events"])
    st = read_table("particles", ids, ["event", "particle_id", "n_hits", "is_target"], root=root)
    st = st[(st.particle_id.to_numpy(np.int64) >> 52) >= 1001]
    m = st.merge(p0[["event", "particle_id", "n_hits", "is_target"]], on=["event", "particle_id"], how="outer",
                 suffixes=("_stored", "_rerun"), indicator=True)
    bad_rows = int((m._merge != "both").sum())
    both = m[m._merge == "both"]
    bad_hits = int((both.n_hits_stored != both.n_hits_rerun).sum())
    bad_tgt = int((both.is_target_stored.astype(bool) != both.is_target_rerun.astype(bool)).sum())
    bad_side = [e for e, c in c0.items() if c["stopped_hits"] != L["side"][e]["stopped_hits"]
                or c["dropped_hits"] != L["side"][e]["dropped_hits"]]
    ok = bad_rows == 0 and bad_hits == 0 and bad_tgt == 0 and not bad_side
    return {"pass": ok, "gun_rows": int(len(st)), "rows_unmatched": bad_rows, "n_hits_differ": bad_hits,
            "is_target_differ": bad_tgt, "events_sidecar_differ": bad_side[:10], "particles": p0}


def run(set_name: str, work: str, out: str, target: float = TARGET_MISS, root: str | None = None) -> dict:
    from hnl.overlay import blind as B
    root = root or B.DERIVED_OVERLAY
    if os.path.exists(out):
        raise SystemExit(f"{out} exists: refusing to overwrite it")
    L = _load(set_name, work, root)
    chk = check_nominal(L, set_name, root)
    p_nom = chk.pop("particles")
    if not chk["pass"]:
        raise SystemExit(f"the rerun does not reproduce the stored {set_name} gun particles: {chk}")
    codes = [v * 100 + lay for v in LONG_STRIP_DISKS for lay in range(0, 100)]
    nominal = sorted({float(x) for x in L["tables"].eps[codes] if x > 0})
    share0 = miss_share(L, None)
    eps_ls, share1 = solve_eps(L, target)
    p_ls, _ = gun_particles(L, eps_ls)
    c_nom, c_ls = counts_of(p_nom), counts_of(p_ls)
    rec = {"set": set_name, "events": len(L["gev"]["events"]), "work": os.path.abspath(work), "root": root,
           "target_miss_share": target, "eps_nominal_long_strip_disks": nominal, "eps_ls": eps_ls,
           "miss_share_nominal": share0, "miss_share_raised": share1, "check_nominal_rerun": chk,
           "counts_nominal": c_nom, "counts_raised": c_ls,
           "change": {k: c_ls[k] - c_nom[k] for k in ("targets", "P_P", "P_D")},
           "overlay_hash": L["meta"]["overlay_hash"], "emulation_tables": L["tables_path"],
           "code_sha256": B.sha256_file(os.path.abspath(__file__)), "git": B._git_head(),
           "slurm_job": os.environ.get("SLURM_JOB_ID"), "host": os.uname().nodename,
           "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(rec, fh, indent=1)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--set", required=True, choices=("hval", "htest", "pilot"))
    r.add_argument("--work", required=True, help="the set's WORK directory of slurm/overlay/overlay_set.sbatch")
    r.add_argument("--out", required=True)
    r.add_argument("--target", type=float, default=TARGET_MISS)
    r.add_argument("--root", default=None, help="the overlay tables (default: the production root)")
    a = ap.parse_args(argv)
    rec = run(a.set, a.work, a.out, a.target, a.root)
    print(json.dumps({k: rec[k] for k in ("set", "events", "eps_nominal_long_strip_disks", "eps_ls",
                                           "miss_share_nominal", "miss_share_raised", "counts_nominal",
                                           "counts_raised", "change")}, indent=1))


if __name__ == "__main__":
    main()
