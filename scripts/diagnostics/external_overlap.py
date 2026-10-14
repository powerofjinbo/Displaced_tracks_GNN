"""Do the events of Wang et al.'s evaluation sample (hnl15_100, our external test set) overlap the events of our
training and validation sets? Truth and hits only: no model output is read.

Three checks of every external event against every event of ttbar, hnl10_100 and hnl15_200 (all splits, reported per
split) and against the other external events:
  1. identical hits: an order-independent 64-bit hash of the event's hit coordinates (x, y, z rounded to 1e-3 mm)
     and the hit count;
  2. identical generator record: the same hash of the generation-0 particles (pdg and table momentum rounded to
     1e-4 GeV) and their count;
  3. shared collision content: generation-0 particles of the underlying event (category "other", table pT > 0.5 GeV)
     keyed by (pdg, pT to 0.01 GeV, eta to 0.01, phi to 0.01; coarse, so that a re-simulated copy of the same
     collision still matches although the table momenta are final-state values); for each external event the
     largest number of such particles shared with any one other event, against the same count between events of
     different samples that cannot share collisions by construction (the distribution of chance coincidences).

Writes results/external_overlap_v1.txt.

    sbatch ... --wrap "source env/activate.sh; python scripts/diagnostics/external_overlap.py"
"""
import glob
import os
import time

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DERIVED = os.path.join(os.environ.get("HNL_DATA", "/ceph/submit/data/user/j/jinboz1/hnl_lrt"), "derived")
OUT = os.path.join(REPO, "results", "external_overlap_v1.txt")
SAMPLES = {"ttbar": ("train", "val", "test"), "hnl10_100": ("train", "val", "test"),
           "hnl15_200": ("train", "val", "test"), "hnl15_100": ("external",)}
MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


def mix(k: np.ndarray) -> np.ndarray:
    """splitmix64 finaliser on uint64 keys (vectorised; wraps modulo 2^64)."""
    with np.errstate(over="ignore"):
        z = k.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def combine(*cols: np.ndarray) -> np.ndarray:
    h = np.zeros(len(cols[0]), dtype=np.uint64)
    with np.errstate(over="ignore"):
        for c in cols:
            h = mix(h ^ np.asarray(c).astype(np.int64).view(np.uint64))
    return h


def event_hash(event: np.ndarray, key: np.ndarray) -> pd.DataFrame:
    """Order-independent per-event hash: the wrapped sum of mixed keys, plus the count."""
    df = pd.DataFrame({"event": event, "h": mix(key).view(np.int64)})
    g = df.groupby("event").h
    return pd.DataFrame({"h": g.apply(lambda x: int(np.sum(x.to_numpy().view(np.uint64), dtype=np.uint64))),
                         "n": g.size()})


def split_of(sample: str) -> dict[int, str]:
    from hnl.data.splits import read_split
    out = {}
    for sp in SAMPLES[sample]:
        for e in read_split(sample, sp):
            out[int(e)] = sp
    return out


def main() -> None:
    t0 = time.time()
    if os.path.exists(OUT):
        raise SystemExit(f"{OUT} exists")
    hits_h, gen_h, ue = {}, {}, []
    for s in SAMPLES:
        sp = split_of(s)
        hf = sorted(glob.glob(os.path.join(DERIVED, "events", "v1", "hits", s, "*.parquet")))
        hh = []
        for f in hf:
            h = pd.read_parquet(f, columns=["event", "x", "y", "z"])
            key = combine(np.round(h.x.to_numpy() * 1000), np.round(h.y.to_numpy() * 1000),
                          np.round(h.z.to_numpy() * 1000))
            hh.append(event_hash(h.event.to_numpy(), key))
        hits_h[s] = pd.concat(hh)
        pf = sorted(glob.glob(os.path.join(DERIVED, "events", "v1", "particles", s, "*.parquet")))
        p = pd.concat([pd.read_parquet(f, columns=["event", "pdg", "px", "py", "pz", "pt", "eta", "gen", "category"])
                       for f in pf], ignore_index=True)
        g0 = p[p.gen == 0]
        key = combine(g0.pdg.to_numpy(), np.round(g0.px.to_numpy() * 1e4), np.round(g0.py.to_numpy() * 1e4),
                      np.round(g0.pz.to_numpy() * 1e4))
        gen_h[s] = event_hash(g0.event.to_numpy(), key)
        u = g0[(g0.category == "other") & (g0.pt > 0.5)]
        phi = np.arctan2(u.py.to_numpy(), u.px.to_numpy())
        ue.append(pd.DataFrame({"sample": s, "event": u.event.to_numpy(),
                                "split": u.event.map(sp).to_numpy(),
                                "k": combine(u.pdg.to_numpy(), np.round(u.pt.to_numpy() * 100),
                                             np.round(u.eta.to_numpy() * 100), np.round(phi * 100)).view(np.int64)}))
        for name, d in (("hits", hits_h[s]), ("gen0", gen_h[s])):
            d["split"] = d.index.map(sp)
            d["sample"] = s
        print(f"{s}: {len(hits_h[s])} events hashed, {len(u)} UE particles ({time.time() - t0:.0f} s)", flush=True)
    lines = ["# External sample (hnl15_100, Wang et al.'s evaluation events) against our other events", "",
             f"Job {os.environ.get('SLURM_JOB_ID', 'local')}, {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}; "
             "truth and hits only (scripts/diagnostics/external_overlap.py).", ""]
    ext_sp = split_of("hnl15_100")
    n_ext = len(ext_sp)
    lines.append(f"External events: {n_ext} (splits/hnl15_100_external.txt).")
    for name, H in (("identical hits (x, y, z to 1e-3 mm)", hits_h), ("identical generation-0 record", gen_h)):
        e = H["hnl15_100"]
        e = e[e.index.isin(ext_sp)]
        dup_in = int(e.duplicated(["h", "n"], keep=False).sum())
        lines.append(f"\n## {name}\n")
        lines.append(f"- external events sharing the hash with another external event: {dup_in}")
        for s in ("ttbar", "hnl10_100", "hnl15_200"):
            o = H[s]
            m = e.reset_index().merge(o.reset_index(), on=["h", "n"], suffixes=("_ext", "_oth"))
            per = m.groupby("split_oth").event_ext.nunique().to_dict() if len(m) else {}
            lines.append(f"- {s}: {o.shape[0]} events; external events with an identical one: {m.event_ext.nunique()}"
                         f" (by split of the match: {per})")
    u = pd.concat(ue, ignore_index=True)
    ext = u[(u["sample"] == "hnl15_100") & u.event.isin(ext_sp)]
    lines.append("\n## Shared underlying-event particles (generation 0, category other, pT > 0.5 GeV, coarse key)\n")
    lines.append(f"- external: {len(ext)} particles in {ext.event.nunique()} events "
                 f"({len(ext) / max(ext.event.nunique(), 1):.1f} per event)")
    for s in ("ttbar", "hnl10_100", "hnl15_200", "hnl15_100"):
        o = u[u["sample"] == s]
        m = ext.merge(o, on="k", suffixes=("_ext", "_oth"))
        if s == "hnl15_100":
            m = m[m.event_ext != m.event_oth]
        c = m.groupby(["event_ext", "event_oth"]).size()
        best = c.groupby(level=0).max().reindex(sorted(ext_sp), fill_value=0)
        top = None
        if s != "hnl15_100" and len(c):
            sp_oth = o.drop_duplicates("event").set_index("event").split
            top = c.reset_index(name="n").sort_values("n", ascending=False).head(5)
            top["split_oth"] = top.event_oth.map(sp_oth)
        q = np.percentile(best, [50, 90, 99, 100])
        lines.append(f"- against {s} ({o.event.nunique()} events): per external event, the largest number shared with "
                     f"one {s} event: median {q[0]:.0f}, 90% {q[1]:.0f}, 99% {q[2]:.0f}, max {q[3]:.0f}; external "
                     f"events with >= 5 shared: {int((best >= 5).sum())}, >= 10: {int((best >= 10).sum())}")
        if top is not None:
            lines.append("  largest pairs (external event, other event, its split, shared): "
                         + "; ".join(f"({int(r.event_ext)}, {int(r.event_oth)}, {r.split_oth}, {int(r.n)})"
                                     for r in top.itertuples()))
    lines.append(f"\nElapsed {time.time() - t0:.0f} s.")
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
