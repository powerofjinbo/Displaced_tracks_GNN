"""Addendum H, step H-T0.5.1 (truth only): species shares of the H-val gun particles against the generic train split
(docs/preregistration_pileup.md section 9.2, "Content (descriptive)"; plan section 10). No model output is read.

H-val: gun particles (vertex field 1001-1010; 1001-1008 isolated, 1009-1010 dense) with at least one stored hit, by
|particle_type| of the stored ptype table. Generic train split: gun particles with at least one hit in the spec v1
generic samples (derived/generated_v1/t1/truth_generic_{iso,dense}.parquet, train ids of splits/generic_{iso,dense}_
train.txt), by |pdg|. The two setups differ (TrackML-like detector, RZ field map and emulation against the HNL setup
in a constant 2 T field), so the comparison has no rule. Shares are also given for the targets (pT > 1 GeV, |eta| < 4,
>= 5 hits) of each side.

    python results/pileup_hp/h_t05_truth_species.py --out-json results/pileup_hp/h_t05_truth_species_v1.json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from hnl.overlay import writer as W

EXT = "/ceph/submit/data/user/j/jinboz1/hnl_lrt"
REPO = "/work/submit/jinboz1/hnl/hnl-displaced"
NAMES = {11: "e", 13: "mu", 211: "pi", 321: "K", 2212: "p"}


def stored(root, set_name, table, columns):
    with open(os.path.join(root, f"meta_{set_name}.json")) as fh:
        meta = json.load(fh)
    return pd.concat([pd.read_parquet(os.path.join(root, sh[table]["file"]), columns=columns)
                      for _, sh in sorted(meta["shards"].items())], ignore_index=True)


def shares(species: pd.Series) -> dict:
    s = species.map(lambda x: NAMES.get(int(x), "other")).value_counts()
    n = int(s.sum())
    return {"n": n, **{k: round(float(s.get(k, 0)) / n, 4) if n else None for k in (*NAMES.values(), "other")}}


def hval(root, set_name):
    p = stored(root, set_name, "particles", ["event", "particle_id", "n_hits", "is_target"])
    y = stored(root, set_name, "ptype", ["event", "particle_id", "particle_type"])
    v = p.particle_id.to_numpy(np.int64) >> 52
    g = p[(v >= 1001) & (v <= 1010)].merge(y, on=["event", "particle_id"], how="left")
    if g.particle_type.isna().any():
        raise RuntimeError("gun particles without a type")
    g["iso"] = (g.particle_id.to_numpy(np.int64) >> 52) <= 1008
    g["sp"] = g.particle_type.abs().astype(int)
    out = {}
    for comp, sel in (("iso", g.iso), ("dense", ~g.iso)):
        h = g[sel & (g.n_hits > 0)]
        out[comp] = {"with_hits": shares(h.sp), "targets": shares(h[h.is_target.astype(bool)].sp)}
    return out


def generic():
    out = {}
    for comp in ("iso", "dense"):
        ids = [int(l) for l in open(f"{REPO}/splits/generic_{comp}_train.txt") if l.strip() and not l.startswith("#")]
        t = pd.read_parquet(f"{EXT}/derived/generated_v1/t1/truth_generic_{comp}.parquet",
                            columns=["event", "pdg", "n_hits", "is_target", "gun"])
        t = t[t.event.isin(ids) & t.gun.astype(bool) & (t.n_hits > 0)]
        sp = t.pdg.abs().astype(int)
        out[comp] = {"train_events": len(ids), "with_hits": shares(sp), "targets": shares(sp[t.is_target.astype(bool)])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="hval", choices=["hval", "pilot"])
    ap.add_argument("--root", default=W.DERIVED_OVERLAY)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    out = {"what": "addendum H H-T0.5.1: species shares of gun particles with hits, H-val against the generic train "
                   "split (descriptive, truth only)", "set": a.set, "root": a.root,
           "hval": hval(a.root, a.set), "generic_train": generic(),
           "code": "results/pileup_hp/h_t05_truth_species.py"}
    with open(a.out_json, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
