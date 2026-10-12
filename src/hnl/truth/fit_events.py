"""Per-particle truth quantities from each particle's own true hits (brief §5b), event by event.

For every particle with >= 3 hits: hits ordered along the true-edge chain; a robust fixed-R circle fit (R from the
table pT, rotation sense sigma = field_sign * sign(q)) and a free-R fit; perigee d0 for both; z0 and cot(theta)
from a robust s-z line fit; first-hit radius, pixel-hit count, labels. For HNL-daughter pairs: the two-track vertex.

    python -m hnl.truth.fit_events --out <dir> [--field-sign +1|-1] [--samples ...] [--limit N]

Hit-resolution scales of the robust (Huber) loss are per subsystem and orientation (brief §2.4: smeared
digitisation, no cells): see SCALES; they set where the loss turns linear, not a hit weight.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from hnl.data.npz import COL, SAMPLES, load_event, particle_ids
from hnl.geometry.ids import decode_barcode, decode_geometry_id, cylindrical_to_cartesian, is_pixel
from hnl.truth import helix as hx
from hnl.truth import labels as L

# Huber scales [mm]: transverse (distance to the circle) and longitudinal (z), per volume.
# Barrel strips measure r-phi finely and z coarsely; endcap strips measure r-phi finely and r coarsely (the r error
# enters the transverse residual for tracks that are not radial).
SCALES = {
    # volume: (transverse, z)
    7: (0.05, 0.10), 8: (0.05, 0.10), 9: (0.05, 0.10),        # pixel endcap-, barrel, endcap+
    12: (0.30, 0.20), 13: (0.10, 0.60), 14: (0.30, 0.20),      # short strips
    16: (1.00, 0.30), 17: (0.15, 3.00), 18: (1.00, 0.30),      # long strips
}


def hit_scales(volume: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.zeros(len(volume))
    z = np.zeros(len(volume))
    for v, (st, sz) in SCALES.items():
        m = volume == v
        t[m], z[m] = st, sz
    return t, z


def _pad_like(p: hx.Padded, values: np.ndarray, fill=1.0) -> np.ndarray:
    out = np.full(p.rows.shape, fill, dtype=np.float64)
    ok = p.rows >= 0
    out[ok] = values[p.rows[ok]]
    return out


def fit_event(ev: dict, field_sign: int, min_hits: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per-particle table, per-hit residual table of the fixed-R fit)."""
    x, pid, te, P = ev["x"], ev["pid"], ev["layerless_true_edges"], ev["particles"]
    ids = particle_ids(P)
    order = hx.order_hits(pid, te)
    pid_o = pid[order]
    up, cnt = np.unique(pid_o, return_counts=True)
    fit_ids = up[cnt >= min_hits]
    sel = np.isin(pid_o, fit_ids)
    xyz = cylindrical_to_cartesian(x)
    vol = decode_geometry_id(ev["layers"])["volume"]
    p = hx.pad_tracks(pid_o[sel], xyz[order][sel], rows=order[sel])
    sorter = np.argsort(ids)
    prow = sorter[np.searchsorted(ids, p.track_ids, sorter=sorter)]
    q = P[prow, COL["q"]]
    pt = P[prow, COL["pt"]]
    R0 = hx.MM_PER_GEV * np.maximum(pt, 1e-6)
    sigma = (field_sign * np.sign(q)).astype(np.int64)
    sigma[sigma == 0] = 1
    st, sz = hit_scales(vol)
    scale_t = _pad_like(p, st)
    scale_z = _pad_like(p, sz)

    # centre-based fixed-R fit for a robust start, then perigee-parameter fits (well conditioned for any pT):
    # "fix" keeps kappa = sigma / R(table pT), "free" fits kappa too
    f0 = hx.fit_circle(p, R0, sigma, scale=scale_t)
    pg0 = hx.perigee(f0.xc, f0.yc, f0.R, f0.sigma)
    k0 = sigma / R0
    fits = {"fix": hx.fit_perigee(p, pg0.d0, pg0.phi0, k0, free_kappa=False, scale=scale_t)}
    fits["free"] = hx.fit_perigee(p, fits["fix"].d0, fits["fix"].phi0, k0, free_kappa=True, scale=scale_t)
    out = {}
    for tag, f in fits.items():
        s = hx.arc_length_perigee(p, f.d0, f.phi0, f.kappa)
        z0, ct, zrms = hx.fit_line_sz(s, p.z, p.mask, scale=scale_z)
        xc, yc = f.centre()
        out.update({f"{tag}_d0": f.d0, f"{tag}_phi0": f.phi0, f"{tag}_kappa": f.kappa, f"{tag}_z0": z0,
                    f"{tag}_cot": ct, f"{tag}_rms": f.rms, f"{tag}_zrms": zrms, f"{tag}_nout": f.n_out,
                    f"{tag}_conv": f.converged, f"{tag}_sigma": f.sigma, f"{tag}_xc": xc, f"{tag}_yc": yc,
                    f"{tag}_R": f.R, f"{tag}_xpca": f.d0 * np.sin(f.phi0), f"{tag}_ypca": -f.d0 * np.cos(f.phi0)})
    fx = fits["fix"]
    # hit-level residuals of the fixed-R fit (for resolution studies)
    rho = hx.perigee_residuals(p, fx.d0, fx.phi0, fx.kappa)
    ok = p.mask
    res = pd.DataFrame({"track": np.nonzero(ok)[0], "volume": vol[p.rows[ok]], "res_t": rho[ok]})

    bc = decode_barcode(p.track_ids)
    pix = is_pixel(vol)
    n_pix = np.array([pix[p.rows[i, : p.n_hits[i]]].sum() for i in range(len(p.n_hits))])
    r_first = np.hypot(p.x[:, 0], p.y[:, 0])
    lab_table = L.labels_from_table(P)
    df = pd.DataFrame({
        "particle_id": p.track_ids, "pdg": P[prow, COL["particle_type"]].astype(np.int64), "q": q, "pt": pt,
        "eta": P[prow, COL["eta"]], "process": P[prow, COL["process"]].astype(np.int64),
        "parent_pdg": P[prow, COL["parent_pid"]].astype(np.int64),
        "vsec": bc["vsec"], "gen": bc["gen"], "n_hits": p.n_hits, "n_pixel_hits": n_pix, "r_first": r_first,
        "z_first": p.z[:, 0], "label_hnl": lab_table["hnl_daughter"][prow], "label_w": lab_table["w_muon"][prow],
        "rule_hnl": L.hnl_daughter_rule(P)[prow], "rule_w": L.w_muon_rule(P)[prow], **out,
    })
    return df, res


def pair_vertices(df: pd.DataFrame, which: str = "label_hnl", tag: str = "fix") -> pd.DataFrame:
    """Two-track vertex of the HNL-daughter pair of one event (both daughters must have a fit)."""
    d = df[df[which]]
    if len(d) != 2:
        return pd.DataFrame()
    h = [dict(xc=d[f"{tag}_xc"].to_numpy()[[i]], yc=d[f"{tag}_yc"].to_numpy()[[i]], R=d[f"{tag}_R"].to_numpy()[[i]],
              sigma=d[f"{tag}_sigma"].to_numpy()[[i]], x_pca=d[f"{tag}_xpca"].to_numpy()[[i]],
              y_pca=d[f"{tag}_ypca"].to_numpy()[[i]], z0=d[f"{tag}_z0"].to_numpy()[[i]],
              cot=d[f"{tag}_cot"].to_numpy()[[i]]) for i in (0, 1)]
    v = hx.two_track_vertex(h[0], h[1], np.array([d.r_first.min()]))
    return pd.DataFrame({"pid_a": [d.particle_id.iloc[0]], "pid_b": [d.particle_id.iloc[1]], "vx": v.x, "vy": v.y,
                         "vz": v.z, "rv": np.hypot(v.x, v.y), "dz": v.dz, "n_int": v.n_intersections,
                         "inside": v.inside_first_hits, "ok": v.ok})


def _fit_chunk(args):
    """Worker: fit events [i0, i1) of one sample; returns (particles, vertices, W-muon residuals)."""
    s, idx, field_sign = args
    labeled = SAMPLES[s].labeled
    parts, verts, resid = [], [], []
    for i in idx:
        df, res = fit_event(load_event(s, i), field_sign)
        df.insert(0, "event", i)
        parts.append(df)
        res["is_w"] = df["label_w" if labeled else "rule_w"].to_numpy()[res.track]
        resid.append(res[res.is_w].drop(columns=["track", "is_w"]).assign(event=i))
        v = pair_vertices(df, "label_hnl" if labeled else "rule_hnl")
        if len(v):
            v.insert(0, "event", i)
            verts.append(v)
    cat = lambda x: pd.concat(x, ignore_index=True) if x else pd.DataFrame()
    return cat(parts), cat(verts), cat(resid)


def main(argv=None):
    import concurrent.futures as cf

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--field-sign", type=int, required=True, choices=(-1, 1))
    ap.add_argument("--samples", default=",".join(SAMPLES))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--every", type=int, default=1, help="use every n-th event (calibration passes)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    ap.add_argument("--chunk", type=int, default=100)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    dropped = pd.read_csv(os.path.join(repo, "docs", "dropped_events.tsv"), sep="\t", comment="#")
    drop = set(zip(dropped["sample"], dropped["event"]))
    t0 = time.time()
    for s in a.samples.split(","):
        n = SAMPLES[s].n_events if a.limit is None else min(a.limit, SAMPLES[s].n_events)
        events = [i for i in range(0, n, a.every) if (s, i) not in drop]
        chunks = [(s, events[k:k + a.chunk], a.field_sign) for k in range(0, len(events), a.chunk)]
        results = []
        if a.workers > 1:
            with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
                for k, r in enumerate(ex.map(_fit_chunk, chunks)):
                    results.append(r)
                    if (k + 1) % 10 == 0:
                        print(f"[fit] {s} {min((k + 1) * a.chunk, len(events))}/{len(events)} events "
                              f"{time.time() - t0:.0f}s", flush=True)
        else:
            results = [_fit_chunk(c) for c in chunks]
        parts, verts, resid = (pd.concat([r[j] for r in results], ignore_index=True) for j in range(3))
        parts.to_parquet(os.path.join(a.out, f"particles_{s}.parquet"), index=False)
        if len(verts):
            verts.to_parquet(os.path.join(a.out, f"vertices_{s}.parquet"), index=False)
        resid.to_parquet(os.path.join(a.out, f"wmuon_residuals_{s}.parquet"), index=False)
        print(f"[fit] {s} done: {len(events)} events, {len(parts)} particles, {time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump({"field_sign": a.field_sign, "samples": a.samples, "limit": a.limit, "every": a.every,
                   "scales": SCALES, "slurm_job": os.environ.get("SLURM_JOB_ID"), "elapsed_s": time.time() - t0},
                  f, indent=1, default=str)


if __name__ == "__main__":
    main()
