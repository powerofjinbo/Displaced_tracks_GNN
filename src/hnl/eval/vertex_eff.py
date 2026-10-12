"""Vertex efficiency of a track builder on scored graphs (brief §6D: "both daughters reconstructed and paired" vs r_v).

For every event whose two HNL daughters are both targets: vertex_found = both daughters are DM-matched (each to its own
track) and those two tracks form one accepted two-track vertex of hnl.vertex.pairing.find_vertices (cuts fixed on val
before any test use, DECISIONS.md 2026-10-03). Every event (HNL and ttbar) also records the number of accepted
vertices; in an HNL event all accepted vertices other than the daughter pair are fake. Only the geometry of the
reconstructed tracks enters the vertexing (hit volumes from the radius, as in hnl.trackbuild.merge_fragments); truth
is used only to score it. Truth r_v of the pair (hnl.eval.particles) is attached for the preregistered r_v bins.

    python -m hnl.eval.vertex_eff --graphs DIR --out OUT --builder cc:0.40 [--builder walkmerge:0.1:0.5:1.0 ...]

Writes OUT/per_event.parquet (event_id, builder, n_vertices, and for pair events pair_both, vertex_found, rv_reco,
rv) and OUT/builders.json; print a summary per builder and sample.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd

from hnl.eval import matching as M
from hnl.eval.evaluate_gnn import event_truth
from hnl.eval.trackbuild_val import build, load_scored
from hnl.trackbuild.merge_fragments import _volume_from_radius
from hnl.truth.report import RV_BINS
from hnl.vertex.pairing import find_vertices, fit_tracks


def parse_builder(spec: str) -> tuple:
    """"cc:0.4" -> ("cc", 0.4); "walkmerge:0.1:0.5:1.0" -> ("walkmerge", 0.1, 0.5, 1.0)."""
    kind, *vals = spec.split(":")
    return (kind, *[float(v) for v in vals])


def evaluate_event(path: str, cfgs: list[tuple], hnl_pairs: dict) -> list[dict]:
    eid, pid, r, phi, z, R, a, b, sc, vtx = load_scored(path, with_vtx=True)
    vol = _volume_from_radius(r, z)
    pair = hnl_pairs.get(eid)
    rows = []
    for i, cfg in enumerate(cfgs):
        lab = build(cfg, len(pid), a, b, sc, R, (r, phi, z), vtx)
        verts, radii = find_vertices(fit_tracks(r, phi, z, vol, lab))     # (V, 2) track-label pairs
        row = {"event_id": int(eid), "builder": i, "n_vertices": int(len(verts))}
        if pair is not None:
            part, _ = M.match_event(pid, lab)
            ta, tb = (int(part.dm_track.get(p, -1)) for p in pair)
            row["pair_both"] = ta >= 0 and tb >= 0
            hit = [k for k, (x, y) in enumerate(verts) if row["pair_both"] and {int(x), int(y)} == {ta, tb}]
            row["vertex_found"] = bool(hit)
            row["rv_reco"] = float(radii[hit[0]]) if hit else np.nan
        rows.append(row)
    return rows


def _chunk(args):
    paths, cfgs, pairs = args
    return [row for p in paths for row in evaluate_event(p, cfgs, pairs)]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Vertex efficiency per builder and sample, overall and in the preregistered r_v bins (+ "no vertex"), and the
    fake vertices per event (HNL events: accepted vertices other than the pair; ttbar: all accepted vertices)."""
    from hnl.eval.particles import split_event_id
    from hnl.data.acorn_store import SAMPLE_CODE
    code_sample = {v: k for k, v in SAMPLE_CODE.items()}
    df = df.copy()
    df["sample"] = [code_sample[int(c)] for c in split_event_id(df.event_id.to_numpy())[0]]
    found = df["vertex_found"].fillna(False).astype(int) if "vertex_found" in df else 0
    df["fake_vertices"] = df.n_vertices - found
    out = []
    for (bld, smp), g in df.groupby(["builder", "sample"]):
        row = {"builder": bld, "sample": smp, "n_events": len(g), "fake_vertices_per_event": g.fake_vertices.mean()}
        pe = g[g.pair_both.notna()] if "pair_both" in g else g.iloc[:0]
        if len(pe):
            row.update(n_pair_events=len(pe), pair_eff=pe.pair_both.astype(bool).mean(),
                       vertex_eff=pe.vertex_found.astype(bool).mean())
            b = pd.cut(pe.rv, RV_BINS, right=False)
            for iv, gg in pe.groupby(b, observed=True):
                row[f"vertex_eff_rv[{iv.left:g},{iv.right:g})"] = gg.vertex_found.astype(bool).mean()
            nv = pe[pe.rv.isna()]
            row["vertex_eff_rv_no_vertex"] = nv.vertex_found.astype(bool).mean() if len(nv) else np.nan
        out.append(row)
    return pd.DataFrame(out)


def main(argv=None):
    import concurrent.futures as cf
    from hnl.eval.particles import particle_table
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--builder", action="append", required=True, help="kind:param[:param...], see trackbuild_val")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    ap.add_argument("--max-events", type=int, default=None)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    paths = sorted(glob.glob(os.path.join(a.graphs, "*.pyg.gz")))[:a.max_events]
    eids = [os.path.basename(p).split("event")[-1].split(".")[0] for p in paths]
    pairs, _ = event_truth(eids)
    cfgs = [parse_builder(s) for s in a.builder]
    chunks = [(paths[k:k + 10], cfgs, pairs) for k in range(0, len(paths), 10)]
    with cf.ProcessPoolExecutor(max_workers=max(a.workers, 1)) as ex:
        rows = [r for res in ex.map(_chunk, chunks) for r in res]
    df = pd.DataFrame(rows)
    # truth r_v of the pair (the same for both daughters; NaN = no two-track vertex in the truth fit)
    pt = particle_table(df.event_id.unique())
    first = {eid: pa for eid, (pa, _) in pairs.items()}
    rv = pt.set_index(["event_id", "particle_id"]).rv
    df["rv"] = [rv.get((e, first[str(e)]), np.nan) if str(e) in first else np.nan for e in df.event_id]
    df.to_parquet(os.path.join(a.out, "per_event.parquet"), index=False)
    with open(os.path.join(a.out, "builders.json"), "w") as f:
        json.dump([[c[0], *map(float, c[1:])] for c in cfgs], f)
    s = summarize(df)
    s.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(s.round(4).to_string(index=False))
    print(json.dumps({"n_events": len(paths), "builders": a.builder, "elapsed_s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
