"""Per-hit outlier flags of muons (DECISIONS.md: "edge-level results also report a 'clean' variant for muon targets
that excludes edges touching an outlier hit").

A muon hit is an outlier if its transverse residual to the particle's own fixed-R perigee fit (truth_fits/v1:
fix_d0, fix_phi0, fix_kappa) exceeds 3 Huber scales of its volume (hnl.truth.fit_events.SCALES): the same rule as
the fit's stored outlier count fix_nout, which is checked per particle.

    python -m hnl.truth.outlier_hits [--samples ...]
writes derived/truth_fits/v1/muon_outlier_hits_<sample>.parquet (event, row, particle_id) and prints the check.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from hnl.data.convert import read_table
from hnl.truth import helix as hx
from hnl.truth.fit_events import hit_scales

DERIVED = os.path.join(os.environ.get("HNL_DATA", "/ceph/submit/data/user/j/jinboz1/hnl_lrt"), "derived")
SAMPLES = ("ttbar", "hnl10_100", "hnl15_200", "hnl15_100")


def flag_sample(sample: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    fits = pd.read_parquet(os.path.join(DERIVED, "truth_fits", "v1", f"particles_{sample}.parquet"),
                           columns=["event", "particle_id", "pdg", "fix_d0", "fix_phi0", "fix_kappa", "fix_nout"])
    mu = fits[(fits.pdg.abs() == 13) & fits.fix_d0.notna()]
    hits = read_table(os.path.join(DERIVED, "events", "v1"), "hits", sample, events=np.unique(mu.event.to_numpy()),
                      columns=["event", "row", "x", "y", "z", "volume", "pid"])
    h = hits.merge(mu[["event", "particle_id"]], left_on=["event", "pid"], right_on=["event", "particle_id"], how="inner")
    h = h.sort_values(["event", "particle_id", "row"]).reset_index(drop=True)
    # one padded "track" per (event, particle)
    grp = pd.factorize(pd.MultiIndex.from_arrays([h.event, h.particle_id]))[0]
    p = hx.pad_tracks(grp, h[["x", "y", "z"]].to_numpy(np.float64), rows=np.arange(len(h)))
    first = h.groupby(grp, sort=True).first()
    par = first[["event", "particle_id"]].merge(mu, on=["event", "particle_id"], how="left")
    e = hx.perigee_residuals(p, par.fix_d0.to_numpy(), par.fix_phi0.to_numpy(), par.fix_kappa.to_numpy())
    st, _ = hit_scales(h.volume.to_numpy())
    scale = np.where(p.rows >= 0, st[np.maximum(p.rows, 0)], 1.0)
    out_mask = p.mask & (np.abs(e) > 3.0 * scale)
    flagged_rows = p.rows[out_mask]
    flags = h.loc[flagged_rows, ["event", "row", "particle_id"]].reset_index(drop=True)
    n_out = out_mask.sum(1)
    check = par[["event", "particle_id", "fix_nout"]].assign(n_flagged=n_out)
    return flags, check


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=",".join(SAMPLES))
    a = ap.parse_args(argv)
    for s in a.samples.split(","):
        flags, check = flag_sample(s)
        path = os.path.join(DERIVED, "truth_fits", "v1", f"muon_outlier_hits_{s}.parquet")
        flags.to_parquet(path, index=False)
        agree = (check.fix_nout == check.n_flagged).mean()
        print(f"{s}: {len(check)} fitted muons, {len(flags)} flagged hits, per-particle agreement with fix_nout "
              f"{agree:.4f} (max |diff| {int((check.fix_nout - check.n_flagged).abs().max())})", flush=True)


if __name__ == "__main__":
    main()
