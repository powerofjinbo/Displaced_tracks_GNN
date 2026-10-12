"""eta-phi sectors with halos for TrackML events (docs/trackml_plan.md section 6.3).

Why. A TrackML event (about 110k hits, 16 % noise, about 1.8 M graph pairs at the plan's density target of 16.5 pairs
per hit) does not fit one Canto classifier forward on a 24 GB A30, so the edge classifier runs on sectors. Only the
edge classifier is sectorised (plan 6.3): the embedding, both radius searches, the triplet support and the track
builders run on full events; each sector forward sees every edge with both hits in its core plus halo, only the edges
the sector owns keep their scores, and the builders run once on the stitched full-event scores, so in the plan's path
no track crosses a sector boundary. `stitch_tracks` (sector-level track labels -> event tracks) is the contingency
path for a per-sector builder (e.g. if gate G3, the builder runtime on full events, fails).

Coordinates. A hit's sector coordinates are its azimuth phi = atan2(y, x) and its pseudorapidity seen from the origin,
eta = asinh(z / r). They are used for partitioning only, never as a network input (Canto's inputs stay invariant).

Cores. n_phi equal azimuthal slices, slice k = [phi0 + k w, phi0 + (k + 1) w), w = 2 pi / n_phi, times the eta slices
cut at the interior boundaries eta_edges (none: one eta slice, i.e. the plan's phi-only sectors). Sector index
s = k_phi * n_eta + k_eta. A hit belongs to exactly one core (its "core sector").

Halos. Sector s also holds every hit within halo_phi of its core in azimuth (circular distance from the core centre at
most w / 2 + halo_phi) and, with eta slices, within halo_eta of its core in eta.

Ownership. An edge (a, b) is owned by the sector whose core contains the edge midpoint: the short-arc midpoint of
phi[a] and phi[b] (= their circular mean) and the mean of eta[a] and eta[b]. Every edge has exactly one owner.

Sector graph. The hits of sector s are its core and halo hits plus both end hits of every edge s owns (so an owned edge
is always present, even one longer than the halo: such "extra" hits are counted); its edges are all event edges with
both hits among them (the induced subgraph). Owned edges keep their sector scores; the others are context.

Fixed configurations (plan section 6.3):
  INFERENCE              8 phi sectors: 45 deg cores, +-22.5 deg halos (scoring, valmon checkpoint selection, gate G2)
  INFERENCE_G2_FALLBACK  the same with +-45 deg halos (if gate G2 fails: "the halos are doubled to +-45 deg")
  TRAIN_WINDOW_CORE / TRAIN_WINDOW_HALO   training windows: a 22.5 deg core at a random azimuth with +-11.25 deg halos
                         (1/8 of the event; loss on the core-owned edges only), built by `window_spec`
The feasibility study's 16 phi x 8 eta layout belonged to the ttbar-trained graph (17.8 M pairs per event); the plan
replaced it by 8 phi sectors for the retrained graph. eta slices stay available (eta_edges, halo_eta) for a G2 or G0
fallback, by a dated decision only.

    python -m hnl.trackml.sectors describe          (the fixed configurations as JSON)
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

TWO_PI = 2.0 * np.pi
DEG = np.pi / 180.0


# ------------------------------------------------------------------------------------------------ specification
@dataclass(frozen=True)
class SectorSpec:
    """Sector layout: n_phi azimuthal cores from phi0 (radians), interior eta core boundaries eta_edges (increasing),
    halo widths halo_phi (radians, each side) and halo_eta (each side)."""
    n_phi: int
    halo_phi: float
    phi0: float = -np.pi
    eta_edges: tuple = field(default_factory=tuple)
    halo_eta: float = 0.0

    def __post_init__(self):
        if self.n_phi < 1:
            raise ValueError("n_phi must be >= 1")
        if self.halo_phi < 0 or self.halo_eta < 0:
            raise ValueError("halos must be >= 0")
        e = np.asarray(self.eta_edges, dtype=np.float64)
        if e.size and not np.all(np.diff(e) > 0):
            raise ValueError("eta_edges must be strictly increasing")
        object.__setattr__(self, "eta_edges", tuple(float(x) for x in e))

    @property
    def width(self) -> float:
        return TWO_PI / self.n_phi

    @property
    def n_eta(self) -> int:
        return len(self.eta_edges) + 1

    @property
    def n_sectors(self) -> int:
        return self.n_phi * self.n_eta

    def to_dict(self) -> dict:
        d = asdict(self)
        d["eta_edges"] = list(self.eta_edges)
        d.update({"core_width_deg": math.degrees(self.width), "halo_phi_deg": math.degrees(self.halo_phi),
                  "n_sectors": self.n_sectors})
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SectorSpec":
        return cls(n_phi=int(d["n_phi"]), halo_phi=float(d["halo_phi"]), phi0=float(d.get("phi0", -np.pi)),
                   eta_edges=tuple(d.get("eta_edges", ())), halo_eta=float(d.get("halo_eta", 0.0)))

    @classmethod
    def from_config(cls, d: dict) -> "SectorSpec":
        """From a config block in degrees: n_phi, halo_phi_deg, [phi0_deg], [eta_edges], [halo_eta]."""
        return cls(n_phi=int(d["n_phi"]), halo_phi=float(d["halo_phi_deg"]) * DEG,
                   phi0=float(d.get("phi0_deg", -180.0)) * DEG, eta_edges=tuple(d.get("eta_edges", ()) or ()),
                   halo_eta=float(d.get("halo_eta", 0.0)))


INFERENCE = SectorSpec(n_phi=8, halo_phi=22.5 * DEG)
INFERENCE_G2_FALLBACK = SectorSpec(n_phi=8, halo_phi=45.0 * DEG)
# one sector holding the whole event, hits and edges in the event's order: the full-event forward that gate G2 compares
# the stitched sectors with (S5 after G2 failed with both halos, DECISIONS 2026-10-08)
FULL_EVENT = SectorSpec(n_phi=1, halo_phi=0.0)
TRAIN_WINDOW_CORE = 22.5 * DEG
TRAIN_WINDOW_HALO = 11.25 * DEG


def window_spec(phi_center: float, core_width: float = TRAIN_WINDOW_CORE,
                halo: float = TRAIN_WINDOW_HALO) -> SectorSpec:
    """A training window as sector 0 of a layout whose core 0 is [phi_center - core_width / 2, phi_center +
    core_width / 2): 2 pi / core_width must be an integer (16 for the plan's 22.5 deg)."""
    n = TWO_PI / core_width
    if abs(n - round(n)) > 1e-9:
        raise ValueError("2 pi / core_width must be an integer")
    return SectorSpec(n_phi=int(round(n)), halo_phi=float(halo), phi0=float(phi_center) - 0.5 * core_width)


# ------------------------------------------------------------------------------------------------ coordinates
def wrap(x) -> np.ndarray:
    """Angle(s) to [-pi, pi)."""
    return np.remainder(np.asarray(x, dtype=np.float64) + np.pi, TWO_PI) - np.pi


def hit_eta(r, z) -> np.ndarray:
    """Pseudorapidity of the hit position seen from the origin (float64)."""
    r = np.asarray(r, dtype=np.float64)
    return np.arcsinh(np.asarray(z, dtype=np.float64) / np.maximum(r, 1e-9))


def phi_slice(phi, spec: SectorSpec) -> np.ndarray:
    u = np.remainder(np.asarray(phi, dtype=np.float64) - spec.phi0, TWO_PI)
    return np.minimum(np.floor(u / spec.width).astype(np.int64), spec.n_phi - 1)


def eta_slice(eta, spec: SectorSpec) -> np.ndarray:
    if spec.n_eta == 1:
        return np.zeros(np.shape(eta), dtype=np.int64)
    return np.searchsorted(np.asarray(spec.eta_edges), np.asarray(eta, dtype=np.float64), side="right").astype(np.int64)


def core_sector(phi, eta, spec: SectorSpec) -> np.ndarray:
    """The sector whose core contains each (phi, eta): the hit-to-sector map."""
    return phi_slice(phi, spec) * spec.n_eta + eta_slice(eta, spec)


def edge_midpoints(phi, eta, a, b) -> tuple[np.ndarray, np.ndarray]:
    """Short-arc azimuth midpoint (circular mean) and mean eta of the edges (a, b)."""
    pa = np.asarray(phi, dtype=np.float64)[a]
    pb = np.asarray(phi, dtype=np.float64)[b]
    mid = wrap(pa + 0.5 * wrap(pb - pa))
    e = np.asarray(eta, dtype=np.float64)
    return mid, 0.5 * (e[a] + e[b])


def edge_owner(phi, eta, a, b, spec: SectorSpec) -> np.ndarray:
    pm, em = edge_midpoints(phi, eta, a, b)
    return core_sector(pm, em, spec)


def sector_bounds(spec: SectorSpec, s: int) -> dict:
    """Core centre and half width in phi, core eta range [lo, hi) of sector s (radians; +-inf at the ends)."""
    kphi, keta = divmod(int(s), spec.n_eta)
    edges = (-np.inf, *spec.eta_edges, np.inf)
    return {"phi_center": float(wrap(spec.phi0 + (kphi + 0.5) * spec.width)), "phi_half": 0.5 * spec.width,
            "eta_lo": float(edges[keta]), "eta_hi": float(edges[keta + 1])}


def in_sector(phi, eta, spec: SectorSpec, s: int, halo: bool = True) -> np.ndarray:
    """Hits inside the core (halo=False: by the core map) or the core plus halo of sector s."""
    if not halo:
        return core_sector(phi, eta, spec) == s
    b = sector_bounds(spec, s)
    ok = np.abs(wrap(np.asarray(phi, dtype=np.float64) - b["phi_center"])) <= b["phi_half"] + spec.halo_phi
    if spec.n_eta > 1:
        e = np.asarray(eta, dtype=np.float64)
        ok &= (e >= b["eta_lo"] - spec.halo_eta) & (e < b["eta_hi"] + spec.halo_eta)
    return ok


def membership(phi, eta, spec: SectorSpec) -> np.ndarray:
    """(n_hits, n_sectors) bool: hit in the core or halo of each sector (each hit is in at least its core sector)."""
    core = core_sector(phi, eta, spec)
    m = np.stack([in_sector(phi, eta, spec, s) for s in range(spec.n_sectors)], axis=1)
    m[np.arange(len(core)), core] = True
    return m


# ------------------------------------------------------------------------------------------------ sector graphs
@dataclass
class Sector:
    """One sector graph of an event: hits = global hit ids (sorted; local id = position), edges = global edge ids
    (sorted), local_edges (2, E_s) in local hit ids, owned (E_s,) bool. n_core / n_halo / n_extra: hits in the core,
    in the halo only, and added only as end hits of owned edges."""
    index: int
    hits: np.ndarray
    edges: np.ndarray
    local_edges: np.ndarray
    owned: np.ndarray
    n_core: int
    n_halo: int
    n_extra: int

    @property
    def n_hits(self) -> int:
        return int(len(self.hits))

    @property
    def n_edges(self) -> int:
        return int(len(self.edges))


def split(phi, eta, edges, spec: SectorSpec, sectors=None) -> list[Sector]:
    """Sector graphs of one event. phi, eta: per hit; edges: (2, E) hit pairs (any orientation, the event's edge list;
    edge ids are column positions). sectors: the sector indices to build (default all)."""
    phi = np.asarray(phi, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    a, b = (np.asarray(x, dtype=np.int64) for x in edges)
    n = len(phi)
    owner = edge_owner(phi, eta, a, b, spec)
    core = core_sector(phi, eta, spec)
    out = []
    for s in (range(spec.n_sectors) if sectors is None else sectors):
        s = int(s)
        is_core = core == s
        m = in_sector(phi, eta, spec, s) | is_core
        own = owner == s
        ends = np.zeros(n, dtype=bool)
        ends[a[own]] = True
        ends[b[own]] = True
        extra = ends & ~m
        m |= extra
        e = np.flatnonzero(m[a] & m[b])
        hits = np.flatnonzero(m)
        loc = np.full(n, -1, dtype=np.int64)
        loc[hits] = np.arange(len(hits))
        n_core = int(is_core.sum())
        out.append(Sector(index=s, hits=hits, edges=e, local_edges=np.stack([loc[a[e]], loc[b[e]]]), owned=own[e],
                          n_core=n_core, n_halo=int(len(hits)) - n_core - int(extra.sum()), n_extra=int(extra.sum())))
    return out


def check_partition(sectors: list[Sector], n_edges: int) -> None:
    """Every event edge is owned by exactly one of the given sectors (all sectors of a layout)."""
    cnt = np.zeros(n_edges, dtype=np.int64)
    for s in sectors:
        np.add.at(cnt, s.edges[s.owned], 1)
    if not np.all(cnt == 1):
        raise ValueError(f"edge ownership is not a partition: {int((cnt == 0).sum())} edges unowned, "
                         f"{int((cnt > 1).sum())} owned twice")


def stitch_scores(n_edges: int, sectors: list[Sector], scores: list) -> np.ndarray:
    """Full-event edge scores from the sector forwards: each edge takes the score its owner sector gave it.
    scores[i] holds one score per edge of sectors[i] (local order = sectors[i].edges)."""
    out = np.full(n_edges, np.nan, dtype=np.float32)
    cnt = np.zeros(n_edges, dtype=np.int64)
    for s, sc in zip(sectors, scores, strict=True):
        sc = np.asarray(sc)
        if sc.shape != (s.n_edges,):
            raise ValueError(f"sector {s.index}: {sc.shape} scores for {s.n_edges} edges")
        g = s.edges[s.owned]
        out[g] = sc[s.owned]
        np.add.at(cnt, g, 1)
    if not np.all(cnt == 1):
        raise ValueError(f"stitching: {int((cnt == 0).sum())} edges without a score, {int((cnt > 1).sum())} scored twice")
    return out


# ------------------------------------------------------------------------------------------------ sector-level tracks
def _union_find(n: int):
    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    return parent, find


def stitch_tracks(n_hits: int, sectors: list[Sector], labels: list, phi, eta, spec: SectorSpec,
                  merge_frac: float = 0.5, min_hits: int = 2) -> np.ndarray:
    """Event track labels (per hit, -1 = no track) from per-sector track labels (labels[i]: one label per local hit
    of sectors[i], -1 = none). Contingency path; the plan builds tracks on the stitched full-event scores.

    1. A sector-level track (>= min_hits hits) is kept only by its owner, the sector whose core contains the midpoint
       of its hits (circular mean phi, mean eta): a track seen whole by two overlapping sectors is kept once.
    2. Two kept tracks (from different sectors) that share hits are merged (union-find) when they agree on the hits
       both sectors see: the shared hits are at least merge_frac of each track's hits that the other track's sector
       also holds (one particle seen in part by two sectors agrees exactly there). Otherwise every hit claimed by
       several (merged) tracks goes to the one owned by the hit's core sector, else to the one with the most hits
       (ties: the lower owner sector, then the lower track number).
    3. Tracks left with fewer than min_hits hits get -1; the others are numbered 0, 1, ... by their smallest hit id."""
    phi = np.asarray(phi, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    core = core_sector(phi, eta, spec)
    seen = np.zeros((n_hits, spec.n_sectors), dtype=bool)
    for s in sectors:
        seen[s.hits, s.index] = True
    tracks = []                                # (owner sector, global hit ids)
    for s, lab in zip(sectors, labels, strict=True):
        lab = np.asarray(lab)
        if lab.shape != (s.n_hits,):
            raise ValueError(f"sector {s.index}: {lab.shape} labels for {s.n_hits} hits")
        ok = lab >= 0
        if not ok.any():
            continue
        u, inv, cnt = np.unique(lab[ok], return_inverse=True, return_counts=True)
        g = s.hits[ok]
        c = np.bincount(inv, weights=np.cos(phi[g]), minlength=len(u))
        si = np.bincount(inv, weights=np.sin(phi[g]), minlength=len(u))
        em = np.bincount(inv, weights=eta[g], minlength=len(u)) / cnt
        owner = core_sector(np.arctan2(si, c), em, spec)
        order = np.argsort(inv, kind="stable")
        starts = np.r_[0, np.cumsum(cnt)]
        for t in range(len(u)):
            if cnt[t] >= min_hits and owner[t] == s.index:
                tracks.append((int(owner[t]), np.sort(g[order[starts[t]:starts[t + 1]]])))
    out = np.full(n_hits, -1, dtype=np.int64)
    if not tracks:
        return out
    T = len(tracks)
    tid = np.concatenate([np.full(len(h), k) for k, (_, h) in enumerate(tracks)])
    hid = np.concatenate([h for _, h in tracks])
    size = np.array([len(h) for _, h in tracks])
    # shared-hit counts between kept tracks (only hits claimed more than once matter)
    parent, find = _union_find(T)
    o = np.lexsort((tid, hid))
    hs, ts = hid[o], tid[o]
    multi = np.flatnonzero(np.r_[hs[1:] == hs[:-1], False] | np.r_[False, hs[1:] == hs[:-1]])
    if len(multi):
        pairs = {}
        grp_start = np.flatnonzero(np.r_[True, hs[multi][1:] != hs[multi][:-1]])
        grp_end = np.r_[grp_start[1:], len(multi)]
        for g0, g1 in zip(grp_start, grp_end):
            ids = ts[multi[g0:g1]]
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    key = (int(min(ids[i], ids[j])), int(max(ids[i], ids[j])))
                    pairs[key] = pairs.get(key, 0) + 1
        for (i, j), sh in sorted(pairs.items()):
            (si, hi), (sj, hj) = tracks[i], tracks[j]
            if sh >= merge_frac * max(int(seen[hi, sj].sum()), int(seen[hj, si].sum())):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    root = np.array([find(k) for k in range(T)])
    # merged groups: owners and hit counts (a hit counted once per group)
    owners = {}
    for k in range(T):
        owners.setdefault(int(root[k]), set()).add(tracks[k][0])
    key = np.unique(root[tid].astype(np.int64) * (n_hits + 1) + hid)
    g_of, h_of = key // (n_hits + 1), key % (n_hits + 1)
    gsize = pd.Series(g_of).value_counts()
    # resolve hits claimed by several groups
    oh = np.lexsort((g_of, h_of))
    g_of, h_of = g_of[oh], h_of[oh]
    first = np.r_[True, h_of[1:] != h_of[:-1]]
    starts = np.flatnonzero(first)
    ends = np.r_[starts[1:], len(h_of)]
    assign = np.full(n_hits, -1, dtype=np.int64)
    min_owner = {g: min(o) for g, o in owners.items()}
    for s0, s1 in zip(starts, ends):
        h = int(h_of[s0])
        cands = g_of[s0:s1].tolist()
        if len(cands) == 1:
            assign[h] = cands[0]
            continue
        home = [g for g in cands if int(core[h]) in owners[g]]
        pool = home if home else cands
        assign[h] = min(pool, key=lambda g: (-int(gsize[g]), min_owner[g], g))
    on = assign >= 0
    if not on.any():
        return out
    u, inv, cnt = np.unique(assign[on], return_inverse=True, return_counts=True)
    good = cnt[inv] >= min_hits
    idx = np.flatnonzero(on)[good]
    if len(idx):
        g = assign[idx]
        first_hit = pd.Series(idx).groupby(g).min()
        rank = {int(k): r for r, k in enumerate(first_hit.sort_values().index)}
        out[idx] = np.array([rank[int(x)] for x in g], dtype=np.int64)
    return out


# ------------------------------------------------------------------------------------------------ sizes and memory
# Per-forward memory model of the L1 SymmetricTripletIGNN (hidden 128, 8 iterations, triplet top-k 4). The classifier
# enumerates sum over hits of C(deg, 2) candidate triplets before its top-k cut.
#  * TrackML, measured (hnl.trackml.probe memory, CPU peak RSS increase of one pass, 16 threads; Slurm jobs 6683826 and
#    6683957; ttbar-trained embedding at 16.5 pairs per hit on dev event 1000): inference 7.6 GB at 0.396 M pairs and
#    12.6 M candidate triplets, 10.8 GB at 0.547 M / 23.8 M, 18.1 GB at 0.875 M / 68.9 M -> about 17.7 kB per pair +
#    47 B per candidate triplet (within 3 %); training (forward + backward with activation checkpointing, no optimizer
#    step) 10.5 GB at 0.192 M pairs / 5.9 M triplets, 14.9 GB at 0.272 M / 12.0 M -> about 54.5 kB per pair.
#  * Earlier estimates: 18 kB per pair + 117 B per candidate triplet (docs/trackml_feasibility.md section 3, CPU
#    inference of TrackML wedges of the ttbar-scale graph) and 73 kB per pair for training (ttbar: 6.4 GB peak at 88k
#    pairs on an A30).
# On an A30 add the CUDA context (about 0.5 GB) and caching-allocator slack (take 10-20 %); gate G0 measures it directly
# (hnl.trackml.g0 sectors).
INFER_BYTES_PER_PAIR = 17.7e3
INFER_BYTES_PER_TRIPLET = 47.0
TRAIN_BYTES_PER_PAIR = 54.5e3
INFER_BYTES_PER_PAIR_FEASIBILITY = 18e3
INFER_BYTES_PER_TRIPLET_FEASIBILITY = 117.0
TRAIN_BYTES_PER_PAIR_TTBAR = 73e3
A30_BYTES = 24 * 2**30                       # 24 GiB board memory
CUDA_CONTEXT_BYTES = 0.5e9


def candidate_triplets(n_hits: int, local_edges: np.ndarray) -> int:
    """sum over hits of C(deg, 2): the unordered edge pairs sharing a hit that SymmetricTripletIGNN enumerates."""
    deg = np.bincount(np.asarray(local_edges).ravel(), minlength=n_hits).astype(np.int64)
    return int((deg * (deg - 1) // 2).sum())


def memory_estimate(pairs: int, cand_triplets: int) -> dict:
    """Host-measured tensor memory of one pass (GB): inference and training with the TrackML coefficients, the
    feasibility study's inference estimate and the ttbar training rate; and the A30 estimate (x1.15 + context)."""
    infer = INFER_BYTES_PER_PAIR * pairs + INFER_BYTES_PER_TRIPLET * cand_triplets
    train = TRAIN_BYTES_PER_PAIR * pairs
    a30 = lambda b: (1.15 * b + CUDA_CONTEXT_BYTES) / 1e9
    return {"infer_gb": infer / 1e9, "train_gb": train / 1e9,
            "infer_gb_feasibility": (INFER_BYTES_PER_PAIR_FEASIBILITY * pairs
                                     + INFER_BYTES_PER_TRIPLET_FEASIBILITY * cand_triplets) / 1e9,
            "train_gb_ttbar_rate": TRAIN_BYTES_PER_PAIR_TTBAR * pairs / 1e9,
            "a30_infer_gb": a30(infer), "a30_train_gb": a30(train)}


def sector_table(sectors: list[Sector], n_hits_event: int, n_edges_event: int) -> pd.DataFrame:
    rows = []
    for s in sectors:
        ct = candidate_triplets(s.n_hits, s.local_edges)
        rows.append({"sector": s.index, "hits": s.n_hits, "core_hits": s.n_core, "halo_hits": s.n_halo,
                     "extra_hits": s.n_extra, "edges": s.n_edges, "owned_edges": int(s.owned.sum()),
                     "frac_event_hits": s.n_hits / max(n_hits_event, 1), "frac_event_edges": s.n_edges / max(n_edges_event, 1),
                     "candidate_triplets": ct, **memory_estimate(s.n_edges, ct)})
    return pd.DataFrame(rows)


def describe() -> dict:
    return {"INFERENCE": INFERENCE.to_dict(), "INFERENCE_G2_FALLBACK": INFERENCE_G2_FALLBACK.to_dict(),
            "TRAIN_WINDOW": {"core_width_deg": math.degrees(TRAIN_WINDOW_CORE),
                             "halo_phi_deg": math.degrees(TRAIN_WINDOW_HALO),
                             "example": window_spec(0.0).to_dict()},
            "memory_model": {"infer_bytes_per_pair": INFER_BYTES_PER_PAIR,
                             "infer_bytes_per_candidate_triplet": INFER_BYTES_PER_TRIPLET,
                             "train_bytes_per_pair": TRAIN_BYTES_PER_PAIR,
                             "train_bytes_per_pair_ttbar": TRAIN_BYTES_PER_PAIR_TTBAR}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("describe")
    a = ap.parse_args(argv)
    if a.cmd == "describe":
        print(json.dumps(describe(), indent=1))


if __name__ == "__main__":
    main()
