"""Displaced-vertex gun of the TrackML pileup overlay (docs/pileup_overlay_plan.md sections 5.3 and 7; decisions D10,
D11): the content of spec v1 section 3 (scripts/acts/gun.py, isolated and dense components) without Pythia.

Per host event:
- origin: the hard-scatter vertex of the host, i.e. the production vertex shared by most particles with vertex field
  1 (particle_id >> 52) and a production radius below 1 mm (`hard_scatter_vertex`; vertex 1 is the hard scatter,
  plan section 6). Truth-only read of the host's particles table.
- N_ISO = 8 isolated LLPs (index 1-8) and N_DENSE = 2 dense LLPs (index 9-10). Each LLP is drawn as in spec v1
  (scripts/acts/gun.py `draw_llp`, `nbody`, `boost`, the eta and r_v ranges and the |z_v| redraw); then its azimuth is
  redrawn until its direction is at least Delta R = DR_MIN from every LLP placed before it in the event (after
  MAX_PHI_TRIES failed azimuths its eta and r_v are redrawn as well). The redraw counts go to the truth table.
- Fatras input: the charged decay products with ACTS barcodes (vertex primary 1, vertex secondary 1000 + LLP index,
  particle 1..n; the barcode of spec v1). The LLP itself is not written.
- TrackML particle ids (plan section 5.3): vertex [52, 64) = 1000 + LLP index (1001-1010), primary [36, 52) = ACTS
  particle index, generation [24, 36) = ACTS generation, secondary [12, 24) = ACTS sub-particle, process [0, 12) = 0.
  A vertex field of 2048 or more would set the int64 sign bit (pandas reads such ids as uint64 and the converter
  stores them negative), so the encoder refuses anything outside 1001-1010. Gun hit weights are 0 (D10, set by the
  digitiser's truth member).
- Random numbers: numpy.random.default_rng([base_seed + 7, overlay event id]) per event, as spec v1 section 4.

    python -m hnl.overlay.gun --hosts 21117,21124 --set pilot --base-seed 990001 --out DIR
writes DIR/fatras_in/event<i>-particles.csv (i = position in the host list), DIR/gun_truth.parquet and
DIR/gun_events.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os

import numpy as np
import pandas as pd

from hnl.geometry.ids import decode_barcode
from hnl.overlay import fatras_io as FIO

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
GUN_V1 = os.path.join(REPO, "scripts", "acts", "gun.py")
N_ISO, N_DENSE = 8, 2
DR_MIN = 0.8
MAX_PHI_TRIES = 1000
VERTEX0 = 1000
LLP_INDICES = tuple(range(1, N_ISO + N_DENSE + 1))
ETA_RANGE, RV_RANGE, ZV_MAX = (-3.0, 3.0), (1.0, 800.0), 2900.0      # spec v1 section 3 (scripts/acts/gun.py)
SEED_OFFSET = 7
TRUTH_COLUMNS = ["event", "host", "particle_id", "acts_barcode", "llp", "component", "n_prongs", "llp_mass",
                 "llp_betagamma", "llp_eta", "llp_phi", "r_v", "vx", "vy", "vz", "pdg", "q", "px", "py", "pz",
                 "phi_redraws", "eta_redraws", "hs_vx", "hs_vy", "hs_vz"]


def _load_gun_v1():
    spec = importlib.util.spec_from_file_location("hnl_overlay_gun_v1", GUN_V1)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G = _load_gun_v1()            # the frozen spec-v1 gun functions (draw_llp, nbody, boost, barcode, signed_pdg, MASS)


def component_of(llp: int) -> str:
    if not 1 <= llp <= N_ISO + N_DENSE:
        raise ValueError(f"LLP index {llp} outside 1..{N_ISO + N_DENSE}")
    return "iso" if llp <= N_ISO else "dense"


def hard_scatter_vertex(particles: pd.DataFrame) -> dict:
    """Production vertex of the host's hard scatter from its particles table (particle_id, vx, vy, vz): the most
    frequent exact vertex among vertex-field-1 particles within 1 mm of the beam line."""
    pid = particles.particle_id.to_numpy().astype(np.uint64)
    v = particles[((pid >> np.uint64(52)) == 1) & (np.hypot(particles.vx, particles.vy) < 1.0)]
    if not len(v):
        raise ValueError("no vertex-1 particle within 1 mm of the beam line")
    key = v[["vx", "vy", "vz"]].astype(np.float64).round(6)
    cnt = key.value_counts()
    best = cnt.index[0]
    return {"vx": float(best[0]), "vy": float(best[1]), "vz": float(best[2]), "n_at_vertex": int(cnt.iloc[0]),
            "n_vertex1_prompt": int(len(v))}


def delta_r(eta1, phi1, eta2, phi2) -> np.ndarray:
    d = np.asarray(phi1) - np.asarray(phi2)
    d = (d + np.pi) % (2 * np.pi) - np.pi
    return np.hypot(np.asarray(eta1) - np.asarray(eta2), d)


def _draw_eta_rv(rng, zpv: float) -> tuple[float, float]:
    eta = rng.uniform(*ETA_RANGE)
    while True:
        rv = rng.uniform(*RV_RANGE)
        zv = zpv + rv * np.sinh(eta)
        if abs(zv) <= ZV_MAX:
            return eta, rv
        eta = rng.uniform(*ETA_RANGE)


def place_llp(rng, zpv: float, placed: list[tuple[float, float]]) -> tuple[float, float, float, int, int]:
    """eta, phi, r_v of one LLP at least DR_MIN from the placed directions; also the azimuth and eta redraw counts.
    The first draw follows spec v1 (eta, phi, then r_v with the |z_v| redraw of eta)."""
    eta = rng.uniform(*ETA_RANGE)
    phi = rng.uniform(0, 2 * np.pi)
    while True:
        rv = rng.uniform(*RV_RANGE)
        if abs(zpv + rv * np.sinh(eta)) <= ZV_MAX:
            break
        eta = rng.uniform(*ETA_RANGE)
    n_phi = n_eta = 0
    while placed and np.min(delta_r(eta, phi, *np.array(placed).T)) < DR_MIN:
        n_phi += 1
        if n_phi % MAX_PHI_TRIES == 0:
            eta, rv = _draw_eta_rv(rng, zpv)
            n_eta += 1
        phi = rng.uniform(0, 2 * np.pi)
    return eta, phi, rv, n_phi, n_eta


def make_gun_event(hs: dict, rng, event: int, host: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(Fatras input particles with ACTS barcodes, truth table) of one overlay event."""
    pv = np.array([hs["vx"], hs["vy"], hs["vz"], 0.0])
    rows, truth, placed = [], [], []
    for k in LLP_INDICES:
        comp = component_of(k)
        ds, qs, m, bg = G.draw_llp(comp, rng)
        eta, phi, rv, n_phi, n_eta = place_llp(rng, pv[2], placed)
        placed.append((eta, phi))
        zv = pv[2] + rv * np.sinh(eta)
        u = np.array([np.cos(phi) / np.cosh(eta), np.sin(phi) / np.cosh(eta), np.tanh(eta)])
        beta = bg / np.sqrt(1 + bg * bg)
        L = rv * np.cosh(eta)
        xv = np.array([pv[0] + rv * np.cos(phi), pv[1] + rv * np.sin(phi), zv, pv[3] + L / (beta * G.C_MM_PER_NS)])
        p4 = G.boost(G.nbody(m, [G.MASS[d] for d in ds], rng), beta * u)
        for i, (d, q) in enumerate(zip(ds, qs), 1):
            bc = G.barcode(1, VERTEX0 + k, i)
            pdg = G.signed_pdg(d, q)
            rows.append([bc, pdg, 0, *xv, *p4[i - 1, 1:], G.MASS[d], q])
            truth.append([event, host, trackml_particle_id(bc), bc, k, comp, len(ds), m, bg, eta, phi, rv, *xv[:3],
                          pdg, q, *p4[i - 1, 1:], n_phi, n_eta, *pv[:3]])
    parts = pd.DataFrame(rows, columns=FIO.PARTICLE_COLUMNS)
    parts["particle_id"] = parts.particle_id.astype(np.uint64)
    tr = pd.DataFrame(truth, columns=TRUTH_COLUMNS)
    tr["acts_barcode"] = tr.acts_barcode.astype(np.uint64)
    tr["particle_id"] = tr.particle_id.astype(np.int64)
    return parts, tr


def trackml_particle_id(acts_barcode) -> np.ndarray | int:
    """ACTS barcode of a gun particle (or of a Fatras secondary of it) -> TrackML particle_id (module docstring)."""
    scalar = np.ndim(acts_barcode) == 0
    b = np.atleast_1d(np.asarray(acts_barcode).astype(np.uint64))
    d = decode_barcode(b.view(np.int64))
    if (d["vprim"] != 1).any():
        raise ValueError("gun barcode with vertex primary != 1")
    vert = d["vsec"]
    if ((vert < VERTEX0 + LLP_INDICES[0]) | (vert > VERTEX0 + LLP_INDICES[-1])).any():
        raise ValueError(f"gun vertex field outside {VERTEX0 + LLP_INDICES[0]}-{VERTEX0 + LLP_INDICES[-1]}")
    if (d["particle"] >= 1 << 16).any() or (d["gen"] >= 1 << 12).any() or (d["sub"] >= 1 << 12).any():
        raise ValueError("barcode field too large for the TrackML layout")
    tml = ((vert.astype(np.int64) << 52) | (d["particle"].astype(np.int64) << 36) | (d["gen"].astype(np.int64) << 24)
           | (d["sub"].astype(np.int64) << 12))
    return int(tml[0]) if scalar else tml


def trackml_fields(pid) -> dict[str, np.ndarray]:
    """TrackML particle_id fields (plan section 5.3), read as int64."""
    p = np.asarray(pid, np.int64)
    return {"vertex": p >> 52, "primary": (p >> 36) & 0xFFFF, "generation": (p >> 24) & 0xFFF,
            "secondary": (p >> 12) & 0xFFF, "process": p & 0xFFF}


def make_gun_set(host_vertices: dict[int, dict], overlay_ids: dict[int, int], base_seed: int, out_dir: str) -> dict:
    """Fatras input and truth of a set of overlay events. host_vertices: {host: hard_scatter_vertex}; overlay_ids:
    {host: overlay id}. Fatras event i = position of the host in sorted order of the overlay ids."""
    fin = os.path.join(out_dir, "fatras_in")
    os.makedirs(fin, exist_ok=True)
    order = sorted(overlay_ids, key=lambda h: overlay_ids[h])
    T, events = [], []
    for i, host in enumerate(order):
        oid = overlay_ids[host]
        rng = np.random.default_rng([base_seed + SEED_OFFSET, oid])
        parts, tr = make_gun_event(host_vertices[host], rng, oid, host)
        FIO.write_particles(parts, fin, i)
        tr["fatras_event"] = i
        T.append(tr)
        events.append({"fatras_event": i, "overlay_event": oid, "host": host, "n_particles": int(len(parts)),
                       "hs_vertex": host_vertices[host]})
    truth = pd.concat(T, ignore_index=True)
    truth.to_parquet(os.path.join(out_dir, "gun_truth.parquet"), index=False)
    rec = {"base_seed": base_seed, "seed_rule": "default_rng([base_seed + 7, overlay event id])",
           "fatras_seed": base_seed + 13, "n_iso": N_ISO, "n_dense": N_DENSE, "dr_min": DR_MIN,
           "gun_v1": os.path.relpath(GUN_V1, REPO), "events": events,
           "phi_redraws_per_llp": float(truth.drop_duplicates(["event", "llp"]).phi_redraws.mean()),
           "eta_redraws": int(truth.drop_duplicates(["event", "llp"]).eta_redraws.sum())}
    with open(os.path.join(out_dir, "gun_events.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    return rec


def main(argv=None):
    from hnl.overlay import writer as W
    from hnl.trackml.convert import read_table
    from hnl.trackml.throughput import DERIVED_TP
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hosts", required=True, help="comma-separated host event ids (tp_v1)")
    ap.add_argument("--set", required=True, choices=sorted(W.SETS))
    ap.add_argument("--base-seed", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    hosts = [int(x) for x in a.hosts.split(",")]
    k, base = W.SETS[a.set]
    base = a.base_seed if a.base_seed is not None else base
    if base is None:
        raise SystemExit(f"set {a.set} has no base seed; pass --base-seed")
    p = read_table("particles", hosts, ["event", "particle_id", "vx", "vy", "vz"], root=DERIVED_TP)
    hv = {h: hard_scatter_vertex(g) for h, g in p.groupby("event")}
    rec = make_gun_set(hv, {h: W.overlay_id(a.set, h) for h in hosts}, base, a.out)
    print(json.dumps({k_: v for k_, v in rec.items() if k_ != "events"}, indent=1))


if __name__ == "__main__":
    main()
