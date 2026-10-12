"""Whole-dataset scans (brief §2.6): data-format checks, corrupted events, and event overlap.

One pass over every event of every sample collects
  * per-event format checks of brief §2.3 (hid permutation, pid membership, E = N_hits - N_particles_with_hits,
    same-pid edges, chain structure, edges between different modules, unused geometry-id bits, labels),
  * hard-range checks: any hit with r < 29 mm, r > 1030 mm or |z| > 2960 mm,
  * per-hit (module, r, phi, z) for robust per-module envelopes,
  * a 64-bit hash of every hit (float32 bits of r, phi, z + geometry id), a multiset hash of every particle's hits,
    and a kinematic signature of every particle with hits and nonzero momentum (PDG code + float64 bits of px, py,
    pz). The signature catches events that share generated particles but not hits (the paired 10 GeV / 15 GeV 200 mm
    events have identical momenta for most particles while most of their hits differ).
Then
  * module envelopes: 0.1-99.9 percentiles of r, z and phi (relative to the module's mean direction), from events
    that pass the hard-range check, padded by 5 mm (phi: 5 mm / r); every hit outside its module's envelope is
    counted per event,
  * overlaps: for every event, the largest fraction of its hits / particles (hit sets) / particle signatures found
    in any single other event of any sample at any index, the partner event, and every pair above `--pair-min`.

Run as a CPU Slurm job (slurm/scan_events.sbatch):  python -m hnl.data.scans --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import numpy as np
import pandas as pd

from hnl.data.npz import COL, SAMPLES, UNPACKED, load_event, particle_ids
from hnl.geometry.ids import decode_geometry_id

HARD_R_MIN, HARD_R_MAX, HARD_ABSZ_MAX = 29.0, 1030.0, 2960.0
ENVELOPE_Q = (0.001, 0.999)
ENVELOPE_PAD_MM = 5.0
EXPECTED_VOLUMES = (7, 8, 9, 12, 13, 14, 16, 17, 18)

_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_GOLD = np.uint64(0x9E3779B97F4A7C15)


def mix64(x: np.ndarray) -> np.ndarray:
    """splitmix64 finaliser, vectorised (uint64 arithmetic wraps mod 2**64)."""
    x = x.astype(np.uint64, copy=True)
    x ^= x >> np.uint64(30)
    x *= _M1
    x ^= x >> np.uint64(27)
    x *= _M2
    x ^= x >> np.uint64(31)
    return x


def hash_words(*words: np.ndarray) -> np.ndarray:
    """Order-dependent 64-bit hash of equally long uint64 columns."""
    h = np.full(words[0].shape, _GOLD, dtype=np.uint64)
    for i, w in enumerate(words):
        h = mix64(h ^ (w.astype(np.uint64) + np.uint64(((i + 1) * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF)))
    return h


def hit_hashes(x: np.ndarray, geoid: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    w1 = (bits[:, 0] << np.uint64(32)) | bits[:, 1]
    w2 = bits[:, 2]
    return hash_words(w1, w2, geoid.view(np.uint64))


def multiset_hash(group: np.ndarray, values: np.ndarray, n_groups: int) -> np.ndarray:
    """Order-independent hash of the multiset of `values` in each group 0..n_groups-1 (sum, xor and count)."""
    a = mix64(values ^ np.uint64(0x5BD1E9955BD1E995))
    b = mix64(values ^ np.uint64(0x27D4EB2F165667C5))
    s = np.zeros(n_groups, dtype=np.uint64)
    np.add.at(s, group, a)
    xr = np.zeros(n_groups, dtype=np.uint64)
    np.bitwise_xor.at(xr, group, b)
    cnt = np.bincount(group, minlength=n_groups).astype(np.uint64)
    return hash_words(s, xr, cnt)


def event_checks(ev: dict[str, np.ndarray]) -> tuple[dict, dict]:
    """Format checks of brief §2.3 for one event. Returns (scalar stats, per-hit arrays for later passes)."""
    x, hid, pid, gid = ev["x"], ev["hid"], ev["pid"], ev["layers"]
    te, P = ev["layerless_true_edges"], ev["particles"]
    n = len(pid)
    r, phi, z = x[:, 0], x[:, 1], x[:, 2]
    ids = particle_ids(P)
    dec = decode_geometry_id(gid)
    upid, pinv = np.unique(pid, return_inverse=True)
    st = {
        "n_hits": n,
        "n_particles": len(P),
        "n_particles_with_hits": len(upid),
        "n_true_edges": te.shape[1],
        "hid_is_perm": bool(np.array_equal(np.sort(hid), np.arange(n))),
        "pid_in_table": bool(np.isin(upid, ids).all()),
        "particle_ids_unique": bool(len(np.unique(ids)) == len(ids)),
        "edges_eq_hits_minus_particles": bool(te.shape[1] == n - len(upid)),
        "n_edges_diff_pid": int((pid[te[0]] != pid[te[1]]).sum()) if te.size else 0,
        "n_edges_same_module": int((gid[te[0]] == gid[te[1]]).sum()) if te.size else 0,
        "max_out_degree": int(np.bincount(te[0], minlength=n).max()) if te.size else 0,
        "max_in_degree": int(np.bincount(te[1], minlength=n).max()) if te.size else 0,
        "edge_rows_in_range": bool(te.size == 0 or (te.min() >= 0 and te.max() < n)),
        "frac_edges_r_out": float((r[te[1]] > r[te[0]]).mean()) if te.size else np.nan,
        "frac_edges_rho3_out": float((np.hypot(r[te[1]], z[te[1]]) > np.hypot(r[te[0]], z[te[0]])).mean())
        if te.size else np.nan,
        "n_bad_geoid_bits": int(((dec["boundary"] != 0) | (dec["approach"] != 0) | (dec["sensitive"] == 0)).sum()),
        "n_unexpected_volume": int((~np.isin(dec["volume"], EXPECTED_VOLUMES)).sum()),
        "n_odd_or_bad_layer": int(((dec["layer"] % 2 != 0) | (dec["layer"] < 2) | (dec["layer"] > 14)).sum()),
        "n_nonfinite_x": int((~np.isfinite(x)).any(axis=1).sum()),
        "n_r_low": int((r < HARD_R_MIN).sum()),
        "n_r_high": int((r > HARD_R_MAX).sum()),
        "n_absz_high": int((np.abs(z) > HARD_ABSZ_MAX).sum()),
        "r_min": float(r.min()) if n else np.nan,
        "r_max": float(r.max()) if n else np.nan,
        "absz_max": float(np.abs(z).max()) if n else np.nan,
        "n_parent_50": int((np.abs(P[:, COL["parent_pid"]]) == 50).sum()),
        "n_parent_24": int((np.abs(P[:, COL["parent_pid"]]) == 24).sum()),
        "n_parent_nonzero": int((P[:, COL["parent_pid"]] != 0).sum()),
    }
    # chain structure: each particle's edges form one path, i.e. n_edges(p) == n_hits(p) - 1 and degrees <= 1
    if te.size:
        ne_p = np.bincount(pinv[te[0]], minlength=len(upid))
    else:
        ne_p = np.zeros(len(upid), dtype=np.int64)
    nh_p = np.bincount(pinv, minlength=len(upid))
    st["n_particles_not_chain"] = int((ne_p != nh_p - 1).sum())
    # hashes
    hh = hit_hashes(x, gid)
    st["n_dup_hit_hash_in_event"] = int(n - len(np.unique(hh)))
    ph = multiset_hash(pinv, hh, len(upid))
    sorter = np.argsort(ids)
    rows = sorter[np.minimum(np.searchsorted(ids, upid, sorter=sorter), len(ids) - 1)]
    pt = P[rows, COL["particle_type"]].astype(np.int64).view(np.uint64)
    pxyz = np.ascontiguousarray(P[rows][:, [COL["px"], COL["py"], COL["pz"]]]).view(np.uint64)
    kin = hash_words(pt, pxyz[:, 0], pxyz[:, 1], pxyz[:, 2])
    # particles with exactly zero momentum (e.g. stopped e+-) share the signature (pdg, 0, 0, 0) across unrelated
    # events: leave them out of the kinematic overlap
    p_zero = (P[rows][:, [COL["px"], COL["py"], COL["pz"]]] == 0).all(axis=1)
    st["n_particles_with_hits_p0"] = int(p_zero.sum())
    arrays = {"geoid": gid, "r": r, "phi": phi, "z": z, "hit_hash": hh, "particle_hash": ph, "kin_hash": kin[~p_zero]}
    return st, arrays


def collect(samples: list[str], root: str = UNPACKED, limit: int | None = None, log_every: int = 2000):
    """One pass over the events. Returns (per-event DataFrame, per-hit arrays, per-particle arrays)."""
    stats, hits, parts = [], {k: [] for k in ("event", "geoid", "r", "phi", "z", "hit_hash")}, \
        {k: [] for k in ("event", "particle_hash", "kin_event", "kin_hash")}
    gev = 0
    t0 = time.time()
    for s in samples:
        n_ev = SAMPLES[s].n_events if limit is None else min(limit, SAMPLES[s].n_events)
        for i in range(n_ev):
            ev = load_event(s, i, root)
            st, a = event_checks(ev)
            st.update(sample=s, index=i, event=gev)
            stats.append(st)
            nh, npart = len(a["r"]), len(a["particle_hash"])
            hits["event"].append(np.full(nh, gev, dtype=np.int32))
            for k in ("geoid", "r", "phi", "z", "hit_hash"):
                hits[k].append(a[k])
            parts["event"].append(np.full(npart, gev, dtype=np.int32))
            parts["particle_hash"].append(a["particle_hash"])
            parts["kin_event"].append(np.full(len(a["kin_hash"]), gev, dtype=np.int32))
            parts["kin_hash"].append(a["kin_hash"])
            gev += 1
            if log_every and gev % log_every == 0:
                print(f"[collect] {gev} events, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(stats)
    hits = {k: np.concatenate(v) for k, v in hits.items()}
    parts = {k: np.concatenate(v) for k, v in parts.items()}
    return df, hits, parts


def _group_quantiles(group: np.ndarray, values: np.ndarray, n_groups: int, qs=ENVELOPE_Q):
    """Per-group quantiles (nearest-rank, lower/upper) via one lexsort."""
    order = np.lexsort((values, group))
    g, v = group[order], values[order]
    counts = np.bincount(g, minlength=n_groups)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    out = []
    for q, rounding in zip(qs, (np.floor, np.ceil)):
        idx = starts + rounding(q * np.maximum(counts - 1, 0)).astype(np.int64)
        res = np.full(n_groups, np.nan)
        ok = counts > 0
        res[ok] = v[idx[ok]]
        out.append(res)
    return out, counts


def module_envelopes(df: pd.DataFrame, hits: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Robust (r, z, dphi) envelopes per geometry id from hard-range-clean events.

    Returns (table, inverse index hit -> module row, dphi of every hit w.r.t. its module's mean direction)."""
    clean_ev = df.loc[(df.n_r_low + df.n_r_high + df.n_absz_high) == 0, "event"].to_numpy()
    use = np.isin(hits["event"], clean_ev)
    ugeo, inv = np.unique(hits["geoid"], return_inverse=True)
    ng = len(ugeo)
    phi = hits["phi"].astype(np.float64)
    c = np.bincount(inv[use], weights=np.cos(phi[use]), minlength=ng)
    s = np.bincount(inv[use], weights=np.sin(phi[use]), minlength=ng)
    phi_c = np.arctan2(s, c)
    dphi = np.remainder(phi - phi_c[inv] + np.pi, 2 * np.pi) - np.pi
    del phi
    (r_lo, r_hi), cnt = _group_quantiles(inv[use], hits["r"][use].astype(np.float64), ng)
    (z_lo, z_hi), _ = _group_quantiles(inv[use], hits["z"][use].astype(np.float64), ng)
    (p_lo, p_hi), _ = _group_quantiles(inv[use], dphi[use], ng)
    dec = decode_geometry_id(ugeo)
    env = pd.DataFrame({
        "geoid": ugeo, "volume": dec["volume"], "layer": dec["layer"], "sensitive": dec["sensitive"],
        "n_hits_clean": cnt, "r_lo": r_lo, "r_hi": r_hi, "z_lo": z_lo, "z_hi": z_hi,
        "phi_c": phi_c, "dphi_lo": p_lo, "dphi_hi": p_hi,
    })
    return env, inv, dphi


def envelope_flags(df: pd.DataFrame, hits: dict, env: pd.DataFrame, inv: np.ndarray, dphi: np.ndarray) -> pd.DataFrame:
    """Per event: hits outside their module's envelope (padded), hits on modules never seen in a clean event,
    and the largest excess in mm. Works one coordinate at a time to bound memory."""
    pad = ENVELOPE_PAD_MM
    n_hits = len(inv)
    excess = np.zeros(n_hits)
    r = hits["r"].astype(np.float64)
    for name, val, pad_v, scale in (
        ("r", r, pad, 1.0),
        ("z", hits["z"].astype(np.float64), pad, 1.0),
        ("dphi", dphi, pad / np.maximum(r, 1.0), r),
    ):
        lo = env[f"{name}_lo"].to_numpy()[inv]
        hi = env[f"{name}_hi"].to_numpy()[inv]
        ex = np.maximum(lo - pad_v - val, val - hi - pad_v) * scale
        np.maximum(excess, np.nan_to_num(ex, nan=0.0), out=excess)
        del lo, hi, ex
    unseen = np.isnan(env["r_lo"].to_numpy())[inv]          # module never seen in a clean event
    bad = (excess > 0) | unseen
    nev = len(df)
    max_excess = np.zeros(nev)
    np.maximum.at(max_excess, hits["event"][bad], excess[bad])
    return pd.DataFrame({
        "event": np.arange(nev),
        "n_env_bad": np.bincount(hits["event"][bad], minlength=nev),
        "n_env_unseen_module": np.bincount(hits["event"][unseen], minlength=nev),
        "env_max_excess_mm": max_excess,
    })


def _pair_counts(event: np.ndarray, key: np.ndarray, n_events: int) -> np.ndarray:
    """For every pair of distinct events (a < b) that share keys: rows (a, b, number of distinct shared keys)."""
    order = np.lexsort((event, key))
    k, e = key[order], event[order].astype(np.int64)
    first = np.r_[True, (k[1:] != k[:-1]) | (e[1:] != e[:-1])]          # unique (key, event)
    k, e = k[first], e[first]
    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    sizes = np.diff(np.r_[starts, len(k)])
    multi = np.flatnonzero(sizes >= 2)
    if len(multi) == 0:
        return np.empty((0, 3), dtype=np.int64)
    pairs = []
    for m in np.unique(sizes[multi]):                                   # groups of equal size, vectorised
        g = multi[sizes[multi] == m]
        members = e[starts[g][:, None] + np.arange(m)[None, :]]
        iu, ju = np.triu_indices(m, 1)
        pairs.append(np.stack([members[:, iu].ravel(), members[:, ju].ravel()], axis=1))
    pairs = np.concatenate(pairs)
    a, b = np.minimum(pairs[:, 0], pairs[:, 1]), np.maximum(pairs[:, 0], pairs[:, 1])
    uc, cnt = np.unique(a * n_events + b, return_counts=True)
    return np.stack([uc // n_events, uc % n_events, cnt], axis=1)


def overlaps(df: pd.DataFrame, hits: dict, parts: dict, pair_min: float):
    """Per event, for hits / particle hit-sets / particle kinematic signatures: the largest fraction of the event's
    items found in any single other event, and that partner; plus every pair with a fraction >= pair_min."""
    n_ev = len(df)
    per_event = pd.DataFrame({"event": np.arange(n_ev)})
    denom = {"hit": df.n_hits.to_numpy(), "particle": df.n_particles_with_hits.to_numpy(),
             "kin": (df.n_particles_with_hits - df.n_particles_with_hits_p0).to_numpy()}
    keys = {"hit": (hits["event"], hits["hit_hash"]), "particle": (parts["event"], parts["particle_hash"]),
            "kin": (parts["kin_event"], parts["kin_hash"])}
    all_pairs = []
    for name, (ev, key) in keys.items():
        pc = _pair_counts(ev, key, n_ev)
        frac_best = np.zeros(n_ev)
        partner = np.full(n_ev, -1, dtype=np.int64)
        if len(pc):
            me = np.concatenate([pc[:, 0], pc[:, 1]])
            other = np.concatenate([pc[:, 1], pc[:, 0]])
            frac = np.concatenate([pc[:, 2], pc[:, 2]]) / np.maximum(denom[name][me], 1)
            order = np.lexsort((frac, me))
            last = order[np.flatnonzero(np.r_[me[order][1:] != me[order][:-1], True])]
            frac_best[me[last]] = frac[last]
            partner[me[last]] = other[last]
            fa = pc[:, 2] / np.maximum(denom[name][pc[:, 0]], 1)
            fb = pc[:, 2] / np.maximum(denom[name][pc[:, 1]], 1)
            keep = np.maximum(fa, fb) >= pair_min
            all_pairs.append(pd.DataFrame({"metric": name, "event_a": pc[keep, 0], "event_b": pc[keep, 1],
                                           "shared": pc[keep, 2], "frac_a": fa[keep], "frac_b": fb[keep]}))
        per_event[f"max_{name}_frac"] = frac_best
        per_event[f"max_{name}_partner"] = partner
    pairs = pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame()
    return per_event, pairs


CORRUPT_EXCESS_MM = 50.0


def corrupted_events(event_scan: pd.DataFrame, threshold_mm: float = CORRUPT_EXCESS_MM) -> pd.DataFrame:
    """Dropping rule (DECISIONS.md 2026-10-03): an event is corrupted if any hit lies more than `threshold_mm`
    outside its module's padded (r, z, phi) envelope. On the full scan this selects exactly the 4 events with hits
    at r < 29 mm (max excess 414-595 mm); every other event has all hits within 3.4 mm of its padded envelope."""
    return event_scan[event_scan.env_max_excess_mm > threshold_mm]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="output directory (on /ceph, under derived/)")
    ap.add_argument("--samples", default=",".join(SAMPLES))
    ap.add_argument("--limit", type=int, default=None, help="events per sample (testing)")
    ap.add_argument("--pair-min", type=float, default=0.01)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    df, hits, parts = collect(a.samples.split(","), limit=a.limit)
    print(f"[scan] collected {len(df)} events, {len(hits['r'])} hits, {len(parts['event'])} particles "
          f"in {time.time() - t0:.0f}s", flush=True)
    env, inv, dphi = module_envelopes(df, hits)
    flags = envelope_flags(df, hits, env, inv, dphi)
    df = df.merge(flags, on="event")
    ov, pairs = overlaps(df, hits, parts, a.pair_min)
    df = df.merge(ov, on="event")
    key = df.set_index("event")[["sample", "index"]]
    for name in ("hit", "particle", "kin"):
        p = df[f"max_{name}_partner"].to_numpy()
        df[f"max_{name}_partner_sample"] = np.where(p >= 0, key["sample"].reindex(p).to_numpy(), "")
        df[f"max_{name}_partner_index"] = np.where(p >= 0, key["index"].reindex(p).fillna(-1).to_numpy(), -1)
    if len(pairs):
        for side in ("a", "b"):
            pairs[f"sample_{side}"] = key["sample"].reindex(pairs[f"event_{side}"]).to_numpy()
            pairs[f"index_{side}"] = key["index"].reindex(pairs[f"event_{side}"]).to_numpy()
    df.to_parquet(os.path.join(a.out, "event_scan.parquet"), index=False)
    env.to_parquet(os.path.join(a.out, "module_envelopes.parquet"), index=False)
    pairs.to_parquet(os.path.join(a.out, "overlap_pairs.parquet"), index=False)
    meta = {
        "samples": a.samples, "limit": a.limit, "pair_min": a.pair_min, "n_events": len(df),
        "n_hits": int(len(hits["r"])), "n_modules": int(len(env)), "elapsed_s": round(time.time() - t0, 1),
        "git": subprocess.run(["git", "-C", os.path.dirname(__file__), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
    }
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
