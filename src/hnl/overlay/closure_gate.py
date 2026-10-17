"""Closure truth gate of addendum H (docs/preregistration_pileup.md section 9.2), before the closure model run. Truth
only: no model is run and no model output is read.

Cells: those of hnl.overlay.closure_checks.hits_per_layer: the detector region of the hit's volume (pixel, short-strip,
long-strip; barrel, endcap: hnl.overlay.truth_checks.region) times the |eta| band of the native (0-1, 1-2, 2-3, 3-4),
with the natives of populations (a)-(c) pooled. A crossed layer is a layer with at least one hit of the track. Per cell
and source (native track in tp_v1, its Fatras hits before the emulation, the stored closure track): tracks, hits,
crossed layers, hits per crossed layer.

Rule (flag gate, section 9.3): in every cell with |eta| < 3 and at least GATE_MIN_LAYERS = 2,000 crossed layers on
each side (native and closure), the mean number of hits per crossed layer of the re-simulated tracks lies within
GATE_TOL = 0.05 of the natives'. Reported per cell with it: hits and layers per track, the share of natives (with hits
in the cell) whose re-simulated version is still a target, and the Fatras hits per crossed layer, which separate
Fatras from the emulation. The cells at |eta| = 3-4 are reported without the rule.

Outcome: "fail" if a counted cell fails, "pass" if at least one cell counts and none fails, and "not informative" if
no cell counts: reported, neither a pass nor a fail, and it does not block the closure model run. One or two counted
cells are reported as "few cells" (`few_cells`).

    python -m hnl.overlay.closure_gate run --work DIR --root ROOT --out-json J
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from hnl.overlay import closure as CL
from hnl.overlay import closure_checks as CC
from hnl.overlay import digitise as D
from hnl.overlay import truth_checks as TC

ABS_ETA_BANDS = (0.0, 1.0, 2.0, 3.0, 4.0)          # = hnl.overlay.closure_checks.ABS_ETA_BANDS
GATE_BAND_MAX = 3.0
GATE_MIN_LAYERS = 2000
GATE_TOL = 0.05
FEW_CELLS = 3                                       # fewer counted cells than this: "few cells"
NOT_INFORMATIVE = "not informative"
SOURCES = ("native", "sim", "closure")


def layer_rows(ctx: dict) -> pd.DataFrame:
    """One row per source, native and region: hits and crossed layers (the inputs of
    hnl.overlay.closure_checks.hits_per_layer), the native's |eta| and whether its closure version is a target."""
    lookup = D.load_lookup(ctx["meta"]["lookup"])
    sim = pd.read_parquet(ctx["sim_path"], columns=["event", "particle_id", "geometry_id"])
    vol, lay, _ = lookup.from_geometry_id(sim.geometry_id.to_numpy(np.uint64))
    sim = sim.assign(volume=vol, layer=lay)
    rows = []
    for e in sorted(ctx["events"]):
        hh = ctx["hosts"][e]["hits"].to_pandas()
        cs = CC.closure_slices(ctx, e)
        sh = cs["hits"].to_pandas()
        cp = cs["particles"].to_pandas()
        tgt_c = dict(zip(CL.native_particle_id(cp.particle_id.to_numpy(np.int64)).tolist(),
                         cp.is_target.to_numpy(bool).tolist()))
        t = ctx["truth"][ctx["truth"].event == e]
        se = sim[sim.event == int(t.fatras_event.iloc[0])]
        src = {"native": hh[hh.pid.isin(t.native_particle_id)].assign(key=lambda d: d.pid),
               "sim": se.assign(key=se.particle_id.map(dict(zip(t.acts_barcode.to_numpy(np.uint64).tolist(),
                                                                t.native_particle_id.tolist())))),
               "closure": sh.assign(key=CL.native_particle_id(sh.pid.to_numpy(np.int64)))}
        eta = dict(zip(t.native_particle_id, t.eta))
        for name, h in src.items():
            g = h.assign(region=TC.region(h.volume), vl=h.volume.to_numpy(np.int64) * 1000
                         + h.layer.to_numpy(np.int64)).groupby(["key", "region"]).agg(
                hits=("vl", "size"), layers=("vl", "nunique"))
            for (k, reg), r in g.iterrows():
                rows.append({"source": name, "event": int(e), "native": int(k), "region": reg,
                             "abs_eta": abs(float(eta[k])), "hits": int(r.hits), "layers": int(r.layers),
                             "closure_target": bool(tgt_c.get(int(k), False))})
    return pd.DataFrame(rows)


def cells(rows: pd.DataFrame) -> list[dict]:
    """Per region and |eta| band: the numbers of hnl.overlay.closure_checks.hits_per_layer, per track, and the share of
    natives still a target."""
    d = rows.assign(band=pd.cut(rows.abs_eta, ABS_ETA_BANDS, right=False))
    out = []
    for (reg, band), g in d.groupby(["region", "band"], observed=True):
        rec = {"region": reg, "abs_eta": str(band), "band_lo": float(band.left)}
        for name in SOURCES:
            x = g[g.source == name]
            rec.update({f"tracks_{name}": int(len(x)), f"hits_{name}": int(x.hits.sum()),
                        f"layers_{name}": int(x.layers.sum()),
                        f"hits_per_layer_{name}": float(x.hits.sum() / x.layers.sum()) if len(x) else None,
                        f"hits_per_track_{name}": float(x.hits.mean()) if len(x) else None,
                        f"layers_per_track_{name}": float(x.layers.mean()) if len(x) else None})
        nat = g[g.source == "native"]
        rec["closure_target_share"] = float(nat.closure_target.mean()) if len(nat) else None
        out.append(rec)
    return out


def gate(cell_rows: list[dict], min_layers: int = GATE_MIN_LAYERS, tol: float = GATE_TOL) -> dict:
    """The rule of the module docstring on the cells of `cells` (or of closure_checks.hits_per_layer, whose band
    labels are the same)."""
    out, failed, counted = [], [], 0
    for c in cell_rows:
        lo = c.get("band_lo", float(str(c["abs_eta"]).strip("[(").split(",")[0]))
        r = dict(c)
        r["in_rule"] = lo < GATE_BAND_MAX
        r["counted"] = bool(r["in_rule"] and c["layers_native"] >= min_layers and c["layers_closure"] >= min_layers)
        if r["counted"]:
            counted += 1
            r["diff"] = c["hits_per_layer_closure"] - c["hits_per_layer_native"]
            r["pass"] = bool(abs(r["diff"]) <= tol)
            if not r["pass"]:
                failed.append({"region": c["region"], "abs_eta": c["abs_eta"], "diff": r["diff"]})
        else:
            r["diff"] = (c["hits_per_layer_closure"] - c["hits_per_layer_native"]
                         if c.get("hits_per_layer_closure") is not None and c.get("hits_per_layer_native") is not None
                         else None)
            r["pass"] = None
        out.append(r)
    outcome = "fail" if failed else ("pass" if counted else NOT_INFORMATIVE)
    return {"rule": {"abs_eta_max": GATE_BAND_MAX, "min_crossed_layers_each_side": min_layers, "tol": tol,
                     "not_informative": "no counted cell: neither a pass nor a fail; does not block",
                     "few_cells": f"fewer than {FEW_CELLS} counted cells, reported"},
            "cells": out, "counted": counted, "failed": failed, "outcome": outcome,
            "pass": {"pass": True, "fail": False}.get(outcome), "no_counted_cell": counted == 0,
            "few_cells": bool(0 < counted < FEW_CELLS)}


def cmd_run(a) -> dict:
    t0 = time.time()
    ctx = CC.load_context(a.work, a.root)
    rows = layer_rows(ctx)
    out = {"set": ctx["set"], "root": a.root, "work": a.work, "events": len(ctx["events"]),
           "kind": "flag gate (addendum H section 9.3), before the closure model run", **gate(cells(rows))}
    out["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out_json)), exist_ok=True)
    with open(a.out_json, "w") as fh:
        json.dump(out, fh, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    print(f"[closure truth gate] {out['outcome']}{' (few cells)' if out['few_cells'] else ''}: {out['counted']} "
          f"counted cells, failed {out['failed']} ({out['seconds']} s)", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--work", required=True)
    r.add_argument("--root", required=True)
    r.add_argument("--out-json", required=True)
    cmd_run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
