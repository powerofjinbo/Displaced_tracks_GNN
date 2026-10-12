"""Data check of the module-frame resolution model and of attach v2's pulls (val events, truth tracks).

For truth tracks of a population (ttbar prompt muons by pT; HNL W muons and HNL daughters), per volume of the hit:
  fit: the leverage-corrected pulls of the weighted helix fit of all the track's hits (hnl.trackbuild.helix_fit; no
    multiple-scattering term, so they widen at low momentum);
  loo: the attach pulls of a left-out hit against the fit of the track's other hits, computed by
    hnl.trackbuild.attach.candidates itself (module-frame sigma plus Highland scattering over the gap): the
    distribution of the attach features for true candidates.
Robust widths (1.4826 MAD for the signed fit pulls; median |pull| / 0.6745 for the attach pulls, which are absolute),
and the candidate window's recall of the left-out hits.
A calibrated model gives about 1 everywhere; the strip endcaps (12, 14, 16, 18) of displaced daughters against prompt
tracks test the anisotropic endcap-strip treatment.

    python -m hnl.eval.attach_pulls [--events 200]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from hnl.geometry import resolution as RS


def _populations(t: pd.DataFrame, sample: str):
    if sample == "ttbar":
        mu = (t.pdg.abs() == 13) & (t.free_d0.abs() < 1.0) & (t.n_hits >= 6)
        return {"ttbar prompt mu pT 1-3": t[mu & (t.pt > 1) & (t.pt <= 3)],
                "ttbar prompt mu pT 3-10": t[mu & (t.pt > 3) & (t.pt <= 10)],
                "ttbar prompt mu pT > 10": t[mu & (t.pt > 10)],
                "ttbar W mu": t[t.label_w & (t.n_hits >= 6)]}
    return {"hnl10_100 W mu": t[t.label_w & (t.n_hits >= 6)],
            "hnl10_100 HNL daughters": t[t.label_hnl & (t.n_hits >= 6)]}


def collect(sample: str, events, max_left_out: int = 14):
    from hnl.trackbuild import attach as AT
    from hnl.trackbuild import helix_fit as HF
    fit_rows, loo_rows = [], []
    for e, g, t in RS._tracks(sample, events):
        x, y, z = g.x.to_numpy(), g.y.to_numpy(), g.z.to_numpy()
        vol = g.volume.to_numpy().astype(np.int64)
        lay = vol * 1000 + g.layer.to_numpy()
        phi_c = RS.module_phi_c(g.geoid.to_numpy())
        geo = {"volume": vol, "phi_c": phi_c, "layer_id": lay}
        pid = g.pid.to_numpy()
        r, phi = np.hypot(x, y), np.arctan2(y, x)
        for name, sel in _populations(t, sample).items():
            if not len(sel):
                continue
            inpop = np.isin(pid, sel.particle_id.to_numpy())
            lab = np.where(inpop, pid, -1)
            f = HF.fit(x, y, z, lab, vol, phi_c, min_hits=6, leverage=True)
            if len(f["track"]):
                d = HF.hit_table(f)
                fit_rows.append(pd.DataFrame({"pop": name, "volume": d["volume"], "pull_t": d["pull_t"],
                                              "pull_z": d["pull_z"]}))
            # leave-one-out: pass m frees the m-th hit (in R order) of every track of the population
            R = np.hypot(r, z)
            order = np.lexsort((R, lab))
            rank = np.zeros(len(lab), dtype=np.int64)
            ok = order[lab[order] >= 0]
            starts = np.r_[0, np.flatnonzero(np.diff(lab[ok]) != 0) + 1]
            cnt = np.diff(np.r_[starts, len(ok)])
            rank[ok] = np.arange(len(ok)) - np.repeat(starts, cnt)
            for m in range(max_left_out):
                out = (lab >= 0) & (rank == m)
                if not out.any():
                    break
                lm = np.where(out, -1, lab)
                rows, X = AT.candidates(lm, r, phi, z, geo=geo)
                left = pd.DataFrame({"pop": name, "volume": vol[out], "hit": np.flatnonzero(out)})
                if len(rows):
                    true = out[rows[:, 1]] & (lab[rows[:, 1]] == rows[:, 0])
                    c = pd.DataFrame({"hit": rows[true, 1], **{k: X[true, AT.FEATURES.index(k)]
                                                                for k in ("pull_t", "pull_z", "gap")}})
                    left = left.merge(c, on="hit", how="left")
                else:
                    left[["pull_t", "pull_z", "gap"]] = np.nan
                loo_rows.append(left)
    return pd.concat(fit_rows, ignore_index=True), pd.concat(loo_rows, ignore_index=True)


def widths(fit: pd.DataFrame, loo: pd.DataFrame) -> pd.DataFrame:
    mad = lambda x: float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) >= 10 else np.nan
    half = lambda x: float(np.median(np.abs(x)) / 0.6745) if len(x) >= 10 else np.nan
    rows = []
    for (pop, v), g in fit.groupby(["pop", "volume"]):
        lo = loo[(loo["pop"] == pop) & (loo.volume == v)]
        found = lo.dropna(subset=["pull_t"])
        rows.append({"pop": pop, "volume": int(v), "n_fit": len(g), "fit_t": mad(g.pull_t), "fit_z": mad(g.pull_z),
                     "n_loo": len(lo), "window_recall": len(found) / max(len(lo), 1), "loo_t": half(found.pull_t),
                     "loo_z": half(found.pull_z)})
    return pd.DataFrame(rows)


def main(argv=None):
    from hnl.data.splits import read_split
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=200)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    parts = [collect("ttbar", read_split("ttbar", "val")[:a.events]),
             collect("hnl10_100", read_split("hnl10_100", "val")[:a.events])]
    fit = pd.concat([p[0] for p in parts], ignore_index=True)
    loo = pd.concat([p[1] for p in parts], ignore_index=True)
    w = widths(fit, loo)
    if a.out:
        w.to_csv(a.out, index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(w.round(3).to_string(index=False))
    print(json.dumps({"events_per_sample": a.events, "resolution": RS.RES_PROVENANCE[:60],
                      "elapsed_s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
