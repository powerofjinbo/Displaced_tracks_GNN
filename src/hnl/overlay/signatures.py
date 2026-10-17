"""Generator-level collision signatures of TrackML events, and the collision twins of overlay hosts (addendum H,
docs/preregistration_pileup.md sections 3 and 7; docs/pileup_overlay_plan.md section 6). Truth only.

The throughput events reuse generator-level collisions: the same collision, translated and simulated again, appears in
several events. A host collision with a "twin" in the training events of the frozen Canto-L (TrackML) is not unseen at
the collision level. Section 3 asks for the twin fractions of the H-test hosts at H-T1, computed from truth and
matched against the training signatures, and section 7 splits the host prompt efficiency, fake rate and TrackML score
by twin status.

Signature (the definition of gnn-displaced gd.prep.signatures, signatures_v1, repeated here so that it runs on the
tables of this project): the generation-0 particles of one vertex field (particle_id bits [52, 64); generation = bits
[24, 36), hnl.trackml.displaced_diag.generation) as a multiset of rows (particle_type, q, px, py, pz), the momenta
rounded to SIG_DIGITS = 5 significant digits, sorted lexicographically and hashed with blake2b-64 (16 hex digits).
Vertex positions are left out (they are drawn again for each reuse). The tables are those of
hnl.trackml.convert (particles: px, py, pz as float32 and q) with the throughput side table ptype (particle_type).

A host's twin status is that of its hard-scatter collision (vertex field 1): twin if its signature equals the signature
of any vertex of any training event. hnl.trackml.collision_reuse counted the val twins with the gnn-displaced file;
`check-tp` repeats that count with this module on the tp_v1 tables (val against train) as the test of the definition.

    python -m hnl.overlay.signatures build --root ROOT --events FILE|train|val --out FILE.parquet
    python -m hnl.overlay.signatures twins --hosts FILE.parquet --train FILE.parquet --out FILE.json
    python -m hnl.overlay.signatures check-tp --out FILE.json        (tp_v1 val against tp_v1 train)
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os

import numpy as np
import pandas as pd

SIG_DIGITS = 5
HS_VERTEX = 1
VERSION = "signatures_h_v1"
GD_SIGNATURES = "/ceph/submit/data/user/j/jinboz1/gnn-displaced/gd/signatures/clusters_v1.parquet"


def quantize(x, digits: int = SIG_DIGITS) -> np.ndarray:
    """Integer code of each value rounded to `digits` significant digits (two values share a code iff they round to the
    same decimal number; 0 has the code 500). The code of gd.prep.signatures.quantize, unchanged."""
    x = np.asarray(x, dtype=np.float64)
    ax = np.abs(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.floor(np.log10(np.where(ax > 0, ax, 1.0))).astype(np.int64)
    m = np.rint(x / np.power(10.0, e - digits + 1)).astype(np.int64)
    over = np.abs(m) >= 10 ** digits
    if over.any():
        e = np.where(over, e + 1, e)
        m = np.where(over, np.rint(x / np.power(10.0, e - digits + 1)).astype(np.int64), m)
    m = np.where(ax > 0, m, 0)
    e = np.where(ax > 0, e, 0)
    return m * 1000 + (e + 500)


def key_rows(particle_type, q, px, py, pz, digits: int = SIG_DIGITS) -> np.ndarray:
    return np.stack([np.asarray(particle_type, dtype=np.int64), np.asarray(q, dtype=np.int64),
                     quantize(px, digits), quantize(py, digits), quantize(pz, digits)], axis=1)


def multiset_signature(rows: np.ndarray) -> str:
    """blake2b-64 (16 hex digits) of the lexicographically sorted rows: independent of the row order."""
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    rows = rows[np.lexsort(rows.T[::-1])] if len(rows) else rows
    return hashlib.blake2b(rows.tobytes(), digest_size=8).hexdigest()


def vertex_field(pid) -> np.ndarray:
    return (np.asarray(pid).astype(np.uint64) >> np.uint64(52)).astype(np.int64)


def generation(pid) -> np.ndarray:
    return ((np.asarray(pid).astype(np.uint64) >> np.uint64(24)) & np.uint64(0xFFF)).astype(np.int64)


def event_signatures(p: pd.DataFrame) -> pd.DataFrame:
    """One row per (event, vertex field with at least one generation-0 particle): event, vtx, n_gen0, signature.
    p: event, particle_id, particle_type, q, px, py, pz (any number of events)."""
    pid = p.particle_id.to_numpy(np.int64)
    g = p[generation(pid) == 0].assign(vtx=lambda d: vertex_field(d.particle_id.to_numpy(np.int64)))
    g = g.sort_values(["event", "vtx"], kind="stable")
    rows = key_rows(g.particle_type, g.q, g.px, g.py, g.pz)
    ev, vtx = g.event.to_numpy(np.int64), g.vtx.to_numpy(np.int64)
    if not len(g):
        return pd.DataFrame({"event": pd.Series(dtype=np.int64), "vtx": pd.Series(dtype=np.int64),
                             "n_gen0": pd.Series(dtype=np.int64), "signature": pd.Series(dtype=object)})
    cut = np.flatnonzero((np.diff(ev) != 0) | (np.diff(vtx) != 0)) + 1
    starts, ends = np.r_[0, cut], np.r_[cut, len(g)]
    return pd.DataFrame({"event": ev[starts], "vtx": vtx[starts], "n_gen0": ends - starts,
                         "signature": [multiset_signature(rows[s:e]) for s, e in zip(starts, ends)]})


def read_particles(events, root: str) -> pd.DataFrame:
    """particles (event, particle_id, q, px, py, pz) joined with ptype (particle_type) of the given events under root;
    every particles row must have its particle_type."""
    from hnl.trackml.convert import read_table
    events = sorted(int(e) for e in events)
    p = read_table("particles", events, ["event", "particle_id", "q", "px", "py", "pz"], root=root)
    t = read_table("ptype", events, ["event", "particle_id", "particle_type"], root=root)
    m = p.merge(t, on=["event", "particle_id"], how="left", validate="one_to_one")
    if m.particle_type.isna().any():
        raise ValueError(f"{int(m.particle_type.isna().sum())} particles rows without a particle_type under {root}")
    if sorted(m.event.unique().tolist()) != events:
        missing = sorted(set(events) - set(m.event.unique().tolist()))
        raise KeyError(f"events without particles under {root}: {missing[:5]}")
    return m


def build(events, root: str, chunk: int = 100) -> pd.DataFrame:
    events = sorted(int(e) for e in events)
    out = [event_signatures(read_particles(events[i:i + chunk], root)) for i in range(0, len(events), chunk)]
    return pd.concat(out, ignore_index=True)


def twins(hosts: pd.DataFrame, train: pd.DataFrame) -> dict:
    """Twin status of each host event's hard-scatter collision (vtx 1) and the twin fraction of all its collisions,
    against every collision of the training events."""
    if set(hosts.event) & set(train.event):
        raise ValueError("a host event is also a training event")
    known = set(train.signature)
    h = hosts.assign(twin=hosts.signature.isin(known))
    hs = h[h.vtx == HS_VERTEX].set_index("event")
    missing = sorted(set(h.event) - set(hs.index))
    if missing:
        raise ValueError(f"host events without a hard-scatter collision (vertex field 1): {missing[:5]}")
    return {"hs_twin": {str(int(e)): bool(t) for e, t in hs.twin.items()},
            "n_hosts": int(len(hs)), "n_hs_twin": int(hs.twin.sum()),
            "n_collisions": int(len(h)), "n_collisions_twin": int(h.twin.sum()),
            "collision_twin_fraction": float(h.twin.mean()) if len(h) else float("nan"),
            "n_train_events": int(train.event.nunique()), "n_train_collisions": int(len(train))}


def _sha_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _events_arg(spec: str) -> list[int]:
    if spec in ("train", "val"):
        from hnl.trackml.config import event_set
        return [int(e) for e in event_set(spec)]
    with open(spec) as fh:
        return [int(x) for x in fh.read().split()]


def _write_parquet(df: pd.DataFrame, path: str, rec: dict) -> None:
    if os.path.exists(path):
        raise SystemExit(f"{path} exists: refusing to overwrite it")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_parquet(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    with open(path + ".json", "w") as fh:
        json.dump({**rec, "sha256": _sha_file(path)}, fh, indent=1)


def check_tp(out: str | None = None) -> dict:
    """The test of the definition on truth: tp_v1 val collisions against tp_v1 train with this module, beside the same
    count with the gnn-displaced signatures (hnl.trackml.collision_reuse, results/trackml_collision_reuse_v1.txt), and
    the partition of the (event, vertex) rows into equal signatures against the gnn-displaced signature ids."""
    from hnl.trackml.config import event_set
    from hnl.trackml.events import DERIVED_TP
    tr, va = [int(e) for e in event_set("train")], [int(e) for e in event_set("val")]
    s_tr, s_va = build(tr, DERIVED_TP), build(va, DERIVED_TP)
    ours = twins(s_va, s_tr)
    gd = pd.read_parquet(GD_SIGNATURES, columns=["event_id", "vtx", "sid"])
    gd = gd[gd.event_id.isin(tr + va)]
    in_train = set(gd[gd.event_id.isin(tr)].sid)
    gv = gd[gd.event_id.isin(va)].assign(twin=lambda d: d.sid.isin(in_train))
    both = pd.concat([s_tr, s_va]).merge(gd.rename(columns={"event_id": "event"}), on=["event", "vtx"], how="outer",
                                         indicator=True)
    matched = both[both._merge == "both"]
    a = matched.groupby("signature").sid.nunique()
    b = matched.groupby("sid").signature.nunique()
    rec = {
        "ours": {k: v for k, v in ours.items() if k != "hs_twin"},
        "gnn_displaced": {"n_collisions": int(len(gv)), "n_collisions_twin": int(gv.twin.sum()),
                          "n_hs": int((gv.vtx == HS_VERTEX).sum()),
                          "n_hs_twin": int(gv[gv.vtx == HS_VERTEX].twin.sum())},
        "rows": {"ours_only": int((both._merge == "left_only").sum()),
                 "gnn_displaced_only": int((both._merge == "right_only").sum()), "both": int(len(matched))},
        "partition": {"signatures_with_several_sids": int((a > 1).sum()),
                      "sids_with_several_signatures": int((b > 1).sum())},
        "hs_twin_agree": int(sum(ours["hs_twin"][str(int(e))] == bool(t)
                                 for e, t in gv[gv.vtx == HS_VERTEX].set_index("event_id").twin.items())),
    }
    rec["pass"] = (rec["ours"]["n_collisions"] == rec["gnn_displaced"]["n_collisions"]
                   and rec["ours"]["n_collisions_twin"] == rec["gnn_displaced"]["n_collisions_twin"]
                   and rec["ours"]["n_hs_twin"] == rec["gnn_displaced"]["n_hs_twin"]
                   and rec["hs_twin_agree"] == rec["gnn_displaced"]["n_hs"]
                   and rec["partition"]["signatures_with_several_sids"] == 0
                   and rec["partition"]["sids_with_several_signatures"] == 0)
    rec["written_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    if out:
        with open(out, "w") as fh:
            json.dump(rec, fh, indent=1)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--root", required=True)
    b.add_argument("--events", required=True, help="train, val (tp_v1, hnl.trackml.config) or a file of event ids")
    b.add_argument("--out", required=True)
    t = sub.add_parser("twins")
    t.add_argument("--hosts", required=True)
    t.add_argument("--train", required=True)
    t.add_argument("--out", required=True)
    c = sub.add_parser("check-tp")
    c.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        ev = _events_arg(a.events)
        df = build(ev, a.root)
        _write_parquet(df, a.out, {"version": VERSION, "root": os.path.abspath(a.root), "events": len(ev),
                                   "events_spec": a.events, "rows": int(len(df)), "sig_digits": SIG_DIGITS,
                                   "code_sha256": _sha_file(os.path.abspath(__file__)),
                                   "slurm_job": os.environ.get("SLURM_JOB_ID"),
                                   "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
        print(json.dumps({"events": len(ev), "rows": int(len(df)), "out": a.out}))
    elif a.cmd == "twins":
        if os.path.exists(a.out):
            raise SystemExit(f"{a.out} exists: refusing to overwrite it")
        h, tr = pd.read_parquet(a.hosts), pd.read_parquet(a.train)
        r = {**twins(h, tr), "hosts": {"path": os.path.abspath(a.hosts), "sha256": _sha_file(a.hosts)},
             "train": {"path": os.path.abspath(a.train), "sha256": _sha_file(a.train)}, "version": VERSION,
             "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
        with open(a.out, "w") as fh:
            json.dump(r, fh, indent=1)
        print(json.dumps({k: v for k, v in r.items() if k != "hs_twin"}, indent=1))
    else:
        r = check_tp(a.out)
        print(json.dumps(r, indent=1))
        if not r["pass"]:
            raise SystemExit("check-tp: the signatures of this module do not reproduce the gnn-displaced twins")


if __name__ == "__main__":
    main()
