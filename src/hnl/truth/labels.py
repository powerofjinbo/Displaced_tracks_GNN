"""Particle categories: HNL daughters and the W muon (brief §2.3, §5d).

Labeled samples (ttbar, 10 GeV 100 mm, 15 GeV 200 mm) carry the parent's signed PDG code in particles[:, 12]:
|parent| = 50 marks an HNL daughter, |parent| = 24 a W daughter. The 15 GeV 100 mm sample has 0 everywhere, so its
categories come from barcode rules, validated here on every labeled HNL event:

  HNL-daughter rule (brief): the two opposite-charge generation-0 muons that share one secondary-vertex number
      vsec > 1 and are the only generation-0 particles at that vsec.
  HNL-daughter rule allowing FSR photons: as above, but other generation-0 particles at that vsec may be photons.
  W-muon rule: the only generation-0 muon with vsec = 1. (A generation-0 FSR photon of the W muon can share vsec 1,
      so "the only generation-0 particle at vsec 1" would miss those events.)
"""

from __future__ import annotations

import numpy as np

from hnl.data.npz import COL, particle_ids
from hnl.geometry.ids import decode_barcode

MUON = 13


def labels_from_table(P: np.ndarray) -> dict[str, np.ndarray]:
    """Categories from column 12 (labeled samples only)."""
    parent = np.abs(P[:, COL["parent_pid"]])
    is_mu = np.abs(P[:, COL["particle_type"]]) == MUON
    return {"hnl_daughter": parent == 50, "w_muon": (parent == 24) & is_mu, "w_daughter": parent == 24}


def _gen0_vsec_counts(bc: dict[str, np.ndarray]) -> dict[int, int]:
    g0 = bc["gen"] == 0
    v, c = np.unique(bc["vsec"][g0], return_counts=True)
    return dict(zip(v.tolist(), c.tolist()))


def hnl_daughter_rule(P: np.ndarray, allow_photons: bool = False) -> np.ndarray:
    """Brief's rule: per vsec > 1, exactly two generation-0 particles, both muons, opposite charge.
    allow_photons: other generation-0 particles at that vsec may be photons (FSR)."""
    ids = particle_ids(P)
    bc = decode_barcode(ids)
    g0 = bc["gen"] == 0
    pdg = np.abs(P[:, COL["particle_type"]])
    is_mu = pdg == MUON
    q = P[:, COL["q"]]
    out = np.zeros(len(P), dtype=bool)
    for v in _gen0_vsec_counts(bc):
        if v <= 1:
            continue
        sel = np.flatnonzero(g0 & (bc["vsec"] == v))
        mu = sel[is_mu[sel]]
        others = sel[~is_mu[sel]]
        if len(mu) != 2 or q[mu[0]] * q[mu[1]] >= 0:
            continue
        if len(others) and not (allow_photons and np.all(pdg[others] == 22)):
            continue
        out[mu] = True
    return out


def hnl_daughter_rule_photons(P: np.ndarray) -> np.ndarray:
    return hnl_daughter_rule(P, allow_photons=True)


def hnl_daughter_rule_first_pair(P: np.ndarray) -> np.ndarray:
    """Alternative (tie-breaker for events where the brief's rule finds more than one pair): among the qualifying
    vsec values, the smallest one."""
    cand = hnl_daughter_rule(P)
    if cand.sum() <= 2:
        return cand
    bc = decode_barcode(particle_ids(P))
    vmin = bc["vsec"][cand].min()
    return cand & (bc["vsec"] == vmin)


def w_muon_rule(P: np.ndarray) -> np.ndarray:
    ids = particle_ids(P)
    bc = decode_barcode(ids)
    mu1 = (bc["gen"] == 0) & (bc["vsec"] == 1) & (np.abs(P[:, COL["particle_type"]]) == MUON)
    if mu1.sum() != 1:
        return np.zeros(len(P), dtype=bool)
    return mu1
