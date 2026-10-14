"""Displaced-vertex particle gun of the generic displaced sample (docs/new_samples_spec.md section 3), added to
Pythia8 ttbar events.

For each event of a chunk: read event<N>-particles_gen.csv (one ttbar collision from acts_sim.py pythia), place
N_LLP neutral long-lived particles at the collision vertex, let each fly in a straight line to a transverse decay
radius r_v and decay there, and write the ttbar particles plus the charged decay products as
event<N>-particles_input.csv for acts_sim.py sim. The LLP itself is not written (it is neutral and leaves no hits).
Truth of every gun particle goes to gun_truth_<first>.npz in the same directory.

Components (fixed before production; docs/new_samples_spec.md):
  iso    two-body decays to e+e-, mu+mu-, pi+pi-, K+K- or p pbar (equal shares), m ~ logU(m_lo, 20 GeV) with
         m_lo = max(0.5 GeV, 2.2 m_daughter), betagamma ~ logU(0.3, 30)
  dense  N-body phase-space decays to N_ch = 3..8 charged particles (uniform), each pi (70 %), K (15 %), p (5 %),
         e (5 %) or mu (5 %), charges summing to 0 or +-1, m ~ logU(max(2 GeV, 1.2 sum m_i), 40 GeV),
         betagamma ~ logU(1, 20)
  both   2 LLPs per event; eta ~ U(-3, 3), phi ~ U(0, 2 pi); r_v ~ U(1, 800) mm, z_v = z_PV + r_v sinh(eta),
         redrawn while |z_v| > 2900 mm; decay isotropic in the LLP rest frame.
Barcodes of gun particles: primary vertex 1 (the ttbar collision), secondary vertex 1000 + LLP index, particle
1..n, generation 0 (hnl.data.npz barcode bits), so they never collide with the generator's secondary vertices.

    python scripts/acts/gun.py DIR --component iso|dense --first N --events N --seed S
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

MASS = {11: 0.000510999, 13: 0.105658, 211: 0.139570, 321: 0.493677, 2212: 0.938272}
COLS = ["particle_id", "particle_type", "process", "vx", "vy", "vz", "vt", "px", "py", "pz", "m", "q"]
C_MM_PER_NS = 299.792458
VSEC0 = 1000


def barcode(vprim, vsec, particle, gen=0, sub=0):
    return (vprim << 52) | (vsec << 40) | (particle << 24) | (gen << 16) | sub


def pstar(M, m1, m2):
    a = (M * M - (m1 + m2) ** 2) * (M * M - (m1 - m2) ** 2)
    return np.sqrt(max(a, 0.0)) / (2 * M)


def iso_dir(rng):
    c = rng.uniform(-1, 1)
    s = np.sqrt(1 - c * c)
    f = rng.uniform(0, 2 * np.pi)
    return np.array([s * np.cos(f), s * np.sin(f), c])


def boost(p4, b):
    """Lorentz boost of the four-momenta p4 [n, 4] (E, px, py, pz) by velocity b [3]."""
    b2 = float(b @ b)
    if b2 <= 0:
        return p4
    g = 1 / np.sqrt(1 - b2)
    bp = p4[:, 1:] @ b
    e = g * (p4[:, 0] + bp)
    p = p4[:, 1:] + ((g - 1) * bp / b2 + g * p4[:, 0])[:, None] * b[None, :]
    return np.c_[e, p]


def nbody(M, masses, rng):
    """Unweighted N-body phase space (Raubold-Lynch / GENBOD with accept-reject); returns [n, 4] in the rest frame."""
    n = len(masses)
    if n == 2:
        p = pstar(M, *masses)
        d = iso_dir(rng)
        return np.array([[np.sqrt(p * p + masses[0] ** 2), *(p * d)], [np.sqrt(p * p + masses[1] ** 2), *(-p * d)]])
    T = M - sum(masses)
    # weight bound: product of the two-body momenta at the largest possible effective masses
    cum = np.cumsum(masses)
    wmax = 1.0
    for k in range(1, n):                              # each factor at its own maximum: M_k + 1 largest, M_k smallest
        wmax *= pstar(T + cum[k], cum[k - 1], masses[k])
    while True:
        r = np.sort(rng.uniform(0, 1, n - 2))
        meff = cum + np.r_[0.0, r, 1.0] * T            # M_1 = m_1, ..., M_n = M
        meff[0] = masses[0]
        ps = [pstar(meff[k], meff[k - 1], masses[k]) for k in range(1, n)]
        if rng.uniform(0, wmax) <= np.prod(ps):
            break
    # build up: particle 0 and 1 in the frame of M_2, then add one particle at a time
    p = ps[0]
    d = iso_dir(rng)
    vecs = np.array([[np.sqrt(p * p + masses[0] ** 2), *(p * d)], [np.sqrt(p * p + masses[1] ** 2), *(-p * d)]])
    for k in range(2, n):
        p = ps[k - 1]
        d = iso_dir(rng)
        # the system of the first k particles recoils against particle k in the frame of M_{k+1}
        esys = np.sqrt(p * p + meff[k - 1] ** 2)
        vecs = boost(vecs, -p * d / esys)
        new = np.array([[np.sqrt(p * p + masses[k] ** 2), *(p * d)]])
        vecs = np.r_[vecs, new]
    return vecs


def draw_llp(component, rng):
    if component == "iso":
        pdg = rng.choice([11, 13, 211, 321, 2212])
        ds = [pdg, pdg]
        qs = [1, -1]
        lo = max(0.5, 2.2 * MASS[pdg])
        m = float(np.exp(rng.uniform(np.log(lo), np.log(20.0))))
        bg = float(np.exp(rng.uniform(np.log(0.3), np.log(30.0))))
    else:
        nch = int(rng.integers(3, 9))
        ds = list(rng.choice([211, 321, 2212, 11, 13], size=nch, p=[0.70, 0.15, 0.05, 0.05, 0.05]))
        qs = rng.permutation([1, -1] * (nch // 2) + ([int(rng.choice([1, -1]))] if nch % 2 else []))
        lo = max(2.0, 1.2 * sum(MASS[d] for d in ds))
        m = float(np.exp(rng.uniform(np.log(lo), np.log(40.0))))
        bg = float(np.exp(rng.uniform(np.log(1.0), np.log(20.0))))
    return [int(d) for d in ds], [int(q) for q in qs], m, bg


def signed_pdg(pdg, q):
    if pdg in (11, 13):        # leptons: positive code for the negative particle
        return -pdg if q > 0 else pdg
    return pdg if q > 0 else -pdg


def make_event(gen: pd.DataFrame, component: str, n_llp: int, rng, event: int):
    prim = gen[((gen.particle_id.astype(np.uint64) >> np.uint64(40)) & np.uint64(0xFFF)) == 0]
    pv = prim[["vx", "vy", "vz", "vt"]].iloc[0].to_numpy(float) if len(prim) else gen[["vx", "vy", "vz", "vt"]].iloc[0].to_numpy(float)
    rows, truth = [], []
    for k in range(1, n_llp + 1):
        ds, qs, m, bg = draw_llp(component, rng)
        eta, phi = rng.uniform(-3, 3), rng.uniform(0, 2 * np.pi)
        while True:
            rv = rng.uniform(1.0, 800.0)
            zv = pv[2] + rv * np.sinh(eta)
            if abs(zv) <= 2900.0:
                break
            eta = rng.uniform(-3, 3)
        u = np.array([np.cos(phi) / np.cosh(eta), np.sin(phi) / np.cosh(eta), np.tanh(eta)])
        beta = bg / np.sqrt(1 + bg * bg)
        L = rv * np.cosh(eta)
        xv = np.array([pv[0] + rv * np.cos(phi), pv[1] + rv * np.sin(phi), zv, pv[3] + L / (beta * C_MM_PER_NS)])
        p4 = boost(nbody(m, [MASS[d] for d in ds], rng), beta * u)
        for i, (d, q) in enumerate(zip(ds, qs), 1):
            pid = barcode(1, VSEC0 + k, i)
            rows.append([pid, signed_pdg(d, q), 0, *xv, *p4[i - 1, 1:], MASS[d], q])
            truth.append([event, pid, k, len(ds), m, bg, eta, phi, rv, *xv, signed_pdg(d, q), q, *p4[i - 1, 1:]])
    return pd.DataFrame(rows, columns=COLS), truth


TRUTH_COLS = ["event", "particle_id", "llp", "n_prongs", "llp_mass", "llp_betagamma", "llp_eta", "llp_phi", "r_v",
              "vx", "vy", "vz", "vt", "pdg", "q", "px", "py", "pz"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--component", choices=["iso", "dense"], required=True)
    ap.add_argument("--first", type=int, required=True)
    ap.add_argument("--events", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-llp", type=int, default=2)
    a = ap.parse_args()
    truth = []
    for ev in range(a.first, a.first + a.events):
        rng = np.random.default_rng([a.seed, ev])         # one stream per event: chunks reproduce independently
        gen = pd.read_csv(os.path.join(a.dir, f"event{ev:09d}-particles_gen.csv"), dtype={"particle_id": np.uint64})
        gun, t = make_event(gen, a.component, a.n_llp, rng, ev)
        gun["particle_id"] = gun.particle_id.astype(np.uint64)     # concat of uint64 and int64 would give float64
        out = pd.concat([gen[COLS], gun], ignore_index=True)
        if out.particle_id.dtype != np.uint64:
            raise TypeError(f"event {ev}: particle_id dtype {out.particle_id.dtype}")
        if out.particle_id.duplicated().any():
            raise ValueError(f"event {ev}: duplicate barcode")
        out.to_csv(os.path.join(a.dir, f"event{ev:09d}-particles_input.csv"), index=False, float_format="%.10g")
        truth += t
    t = np.array(truth, dtype=object)
    arrs = {c: np.array(t[:, i].tolist(), dtype=(np.uint64 if c == "particle_id" else np.int64 if c in
                                                 ("event", "llp", "n_prongs", "pdg", "q") else np.float64))
            for i, c in enumerate(TRUTH_COLS)}
    np.savez(os.path.join(a.dir, f"gun_truth_{a.first}.npz"), component=np.array(a.component), **arrs)


if __name__ == "__main__":
    main()
