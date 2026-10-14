"""Compare two sample_stats.py summaries (a sample we simulate against the Zenodo reference) as a markdown table.

    python scripts/acts/compare_stats.py OURS.json ZENODO.json [--label-a A --label-b B] > table.md
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def mean_sem(d):
    return d["mean"], d["sd"] / np.sqrt(max(d.get("n", 0) or 1, 1))


def ks(h1, h2):
    a, b = np.cumsum(h1, dtype=float), np.cumsum(h2, dtype=float)
    if a[-1] == 0 or b[-1] == 0:
        return float("nan")
    return float(np.max(np.abs(a / a[-1] - b / b[-1])))


def frac(d, keys=None, top=8):
    tot = sum(d.values())
    ks_ = keys or sorted(d, key=lambda k: -d[k])[:top]
    return {k: d.get(k, 0) / tot for k in ks_}, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--label-a", default="ours")
    ap.add_argument("--label-b", default="Zenodo")
    o = ap.parse_args()
    A, B = json.load(open(o.a)), json.load(open(o.b))
    na, nb = A["n_events"], B["n_events"]
    L = [f"| quantity | {o.label_a} ({na} events) | {o.label_b} ({nb} events) | difference / sigma |", "|---|---|---|---|"]

    def row(name, k):
        ma, sa = A[k]["mean"], A[k]["sd"] / np.sqrt(na)
        mb, sb = B[k]["mean"], B[k]["sd"] / np.sqrt(nb)
        L.append(f"| {name} | {ma:.1f} +- {sa:.1f} | {mb:.1f} +- {sb:.1f} | {(ma - mb) / np.hypot(sa, sb):+.1f} |")

    row("hits per event", "n_hits")
    row("particle-table rows per event", "n_particles")
    row("particles with hits per event", "n_particles_with_hits")
    row("primaries with hits per event", "n_gen0")
    row("secondaries with hits per event", "n_genpos")

    def ratio(name, kn, kd):
        a, b = A[kn]["sum"] / A[kd]["sum"], B[kn]["sum"] / B[kd]["sum"]
        L.append(f"| {name} | {100 * a:.2f} % | {100 * b:.2f} % | |")

    ratio("secondary fraction among particles with hits", "n_genpos", "n_particles_with_hits")
    ratio("true edges pointing outward in r", "edges_out_r", "n_edges")
    ratio("true edges pointing outward in sqrt(r^2+z^2)", "edges_out_R", "n_edges")
    ratio("chains monotone in sqrt(r^2+z^2)", "chains_R_monotone", "chains")
    for k, name in [("edges_same_module", "true edges within one module"),
                    ("particles_repeated_module", "particles with two hits on one module"),
                    ("n_edges_minus_expected", "edges minus (hits - particles with hits)")]:
        L.append(f"| {name} (total) | {A[k]['sum']:.0f} | {B[k]['sum']:.0f} | |")
    hp_a, hp_b = np.array(A["hits_per_particle"]), np.array(B["hits_per_particle"])
    nh = np.arange(len(hp_a))
    L.append(f"| mean hits per particle with hits | {(hp_a * nh).sum() / hp_a.sum():.2f} | "
             f"{(hp_b * nh).sum() / hp_b.sum():.2f} | KS {ks(hp_a, hp_b):.3f} |")
    for k, name in [("pt_all", "pT of particles with hits"), ("pt_gen0", "pT of primaries with hits"),
                    ("eta_all", "eta of particles with hits"), ("p_gen0_low", "p of primaries with hits, 0-2 GeV")]:
        L.append(f"| {name}: shape | | | KS {ks(A[k], B[k]):.3f} |")
    for v in ("vol13", "vol17"):
        a, b = A[f"zres_{v}_mad"], B[f"zres_{v}_mad"]
        L.append(f"| z residual of barrel {'short' if v == 'vol13' else 'long'}-strip hits, sigma_MAD (mm) | "
                 f"{a['sigma_mad']:.3f} ({a['n']}) | {b['sigma_mad']:.3f} ({b['n']}) | |")
    L.append("")
    L.append("Hits per (volume, layer), share of all hits:")
    L.append("")
    L.append(f"| volume_layer | {o.label_a} | {o.label_b} |")
    L.append("|---|---|---|")
    ta, tb = sum(A["vol_layer"].values()), sum(B["vol_layer"].values())
    for k in sorted(set(A["vol_layer"]) | set(B["vol_layer"]), key=lambda s: tuple(map(int, s.split("_")))):
        L.append(f"| {k} | {100 * A['vol_layer'].get(k, 0) / ta:.2f} % | {100 * B['vol_layer'].get(k, 0) / tb:.2f} % |")
    for key, name in [("pdg_gen0", "primaries with hits"), ("pdg_genpos", "secondaries with hits"),
                      ("process", "process code of particles with hits")]:
        fb, _ = frac(B[key])
        fa, _ = frac(A[key], keys=list(fb))
        L.append("")
        L.append(f"{name}: share by |PDG| (or code), top of {o.label_b}:")
        L.append("")
        L.append(f"| code | {o.label_a} | {o.label_b} |")
        L.append("|---|---|---|")
        for k in fb:
            L.append(f"| {k} | {100 * fa[k]:.2f} % | {100 * fb[k]:.2f} % |")
    L.append("")
    L.append("Label-exchange test (hnl.eval.dense_jet_check): exchanged / tested pairs by transverse separation (mm):")
    L.append("")
    sep = A["edges"]["sep"]
    head = " | ".join(f"{sep[i]:g}-{sep[i + 1]:g}" for i in range(len(sep) - 1))
    L.append(f"| pairs | {head} |")
    L.append("|---" * len(sep) + "|")
    for lab, S in ((o.label_a, A), (o.label_b, B)):
        for kind, nm in (("same", "same module"), ("ctrl", "different modules of one layer (control)")):
            n, x = np.array(S[f"swap_{kind}_n"]), np.array(S[f"swap_{kind}_x"])
            cells = " | ".join(f"{100 * xx / nn:.1f} % ({xx}/{nn})" if nn else "-" for xx, nn in zip(x, n))
            L.append(f"| {lab}, {nm} | {cells} |")
    print("\n".join(L))


if __name__ == "__main__":
    main()
