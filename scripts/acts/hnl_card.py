"""Pythia8 settings of the HNL samples, pp -> W -> mu N, N -> mu mu nu (docs/hnl_points_spec.md).

The card that reproduces the Zenodo HNL samples (Wang et al., arXiv:2203.08800; Zenodo 10.5281/zenodo.6812533) on
truth (results/hnl_points_closure_v1.md). Pure Python, so that both the ACTS v15 generator (acts_sim.py, inside the
build container) and the stand-alone Pythia8 driver (hnl_pythia.cc, which also records the generator's parent codes)
take their settings from the same function.

- W production: f fbar' -> W (Pythia's WeakSingleBoson:ffbar2W), the W forced to mu N: W+ -> mu+ N and, by charge
  conjugation, W- -> mu- Nbar (the Zenodo files carry parent +50 with a W+ and -50 with a W-, in every event).
  Initial-state showers start at the factorization scale (SpaceShower:pTmaxMatch = 1) instead of Pythia's default
  for a hard process without final-state partons (emissions up to the kinematic limit): only then does the W-muon
  pT spectrum above the Jacobian peak agree with the Zenodo samples (90/95/99 % quantiles 43.0 / 48.0 / 63.4 GeV
  against Zenodo 42.4 / 47.2 / 60.8 GeV; default shower 44.7 / 51.7 / 72.5 GeV).
- Beam energy 14 TeV (the ACTS v15 default; the Zenodo record says 13 TeV, but at 13 TeV the particle multiplicity is
  2.4-4.5 % below the Zenodo samples, at 14 TeV it agrees).
- N: a new neutral spin-1/2 particle with code 50 and a distinct antiparticle, mass m_N, zero width, proper lifetime
  c tau0 (Pythia's tau0 is in mm/c); never a resonance (Pythia makes particles above 20 GeV resonances by default,
  which would decay them in the hard process) and always allowed to decay (Pythia allows decays by default only for
  tau0 < 1000 mm).
- N decay, forced to mu+ mu- and a muon neutrino. DECAYS lists the forms the closure compared. Pythia's meMode 22
  weight for three products (p1, p2, p3) is (P.p1)(p2.p3) (measured on generator-level samples: the product listed
  first takes the smaller energy share). The V-A matrix element of N -> mu- W+*, W+* -> mu+ nu_mu is
  (P.p_mu+)(p_mu-.p_nu), i.e. products (-13, 13, 14); the lepton-number-violating decay of a Majorana N,
  N -> mu+ W-*, W-* -> mu- nubar_mu, is (P.p_mu-)(p_mu+.p_nubar), products (13, -13, -14).
- Hadron decays: Pythia defaults (every particle with tau0 < 1000 mm is decayed by the generator, K_S and Lambda
  included), unlike the ttbar of acts_sim.py (ParticleDecays:limitTau0 = on); the Zenodo HNL tables hold no K_S or
  Lambda but their daughters (parent codes 310, 3122), the Zenodo ttbar tables hold K_S and Lambda.
"""
from __future__ import annotations

HNL_ID = 50
ECM_GEV = 14000.0          # ACTS v15 default; the Zenodo record says 13 TeV, the closure requires 14 TeV
BEAMSPOT_MM = (0.0125, 0.0125, 55.5)
BEAM_T_NS = 5.0            # ACTS v15 Pythia8 example default; measured on the Zenodo W muons (4.9 ns)

# (branching ratio, meMode, products)
DECAYS = {
    "majorana_va": [(0.5, 22, (-13, 13, 14)), (0.5, 22, (13, -13, -14))],
    "dirac_va": [(1.0, 22, (-13, 13, 14))],
    "phase_space": [(1.0, 0, (14, 13, -13))],
    "va_nu_first": [(1.0, 22, (14, 13, -13))],
    "dirac_va_conj": [(1.0, 22, (13, -13, 14))],
}
DEFAULT_DECAY = "majorana_va"


def settings(mass: float, ctau: float, decay: str = DEFAULT_DECAY, extra: tuple[str, ...] = ()) -> list[str]:
    """Pythia8 settings for one HNL point (mass in GeV, c tau0 in mm)."""
    if not 0.5 < mass < 75.0:
        raise ValueError(f"m_N = {mass} GeV outside the W -> mu N range used here")
    if ctau <= 0:
        raise ValueError("c tau0 must be positive")
    channels = DECAYS[decay]
    if abs(sum(br for br, _, _ in channels) - 1.0) > 1e-9:
        raise ValueError(f"{decay}: branching ratios do not sum to 1")
    out = [
        "WeakSingleBoson:ffbar2W = on",
        "24:onMode = off",
        f"24:addChannel = 1 1.0 100 -13 {HNL_ID}",
        "SpaceShower:pTmaxMatch = 1",
        f"{HNL_ID}:new = N Nbar 2 0 0 {mass:.6g} 0.0 0.0 0.0 {ctau:.6g}",
        f"{HNL_ID}:isResonance = false",
        f"{HNL_ID}:mayDecay = on",
    ]
    for br, me, prod in channels:
        # mu+ mu- and one muon neutrino: guards against a truncated product list
        if sorted(prod[:]) not in ([-13, 13, 14], [-14, -13, 13]):
            raise ValueError(f"N decay products {prod}")
        out.append(f"{HNL_ID}:addChannel = 1 {br:.6g} {int(me)} " + " ".join(str(int(x)) for x in prod))
    return out + [x.strip() for x in extra if x.strip()]


def write_cmnd(path: str, mass: float, ctau: float, decay: str = DEFAULT_DECAY, extra: tuple[str, ...] = ()) -> list[str]:
    s = settings(mass, ctau, decay, extra)
    with open(path, "w") as f:
        f.write("! HNL point m_N = %g GeV, c tau0 = %g mm, N decay %s (scripts/acts/hnl_card.py)\n"
                % (mass, ctau, decay))
        for line in s:
            f.write(line + "\n")
    return s


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="write the Pythia8 card of one HNL point")
    ap.add_argument("out")
    ap.add_argument("--mass", type=float, required=True)
    ap.add_argument("--ctau", type=float, required=True)
    ap.add_argument("--decay", default=DEFAULT_DECAY, choices=sorted(DECAYS))
    ap.add_argument("--extra", default="", help="further Pythia settings separated by ';' (pilot studies only)")
    a = ap.parse_args()
    for line in write_cmnd(a.out, a.mass, a.ctau, a.decay, tuple(a.extra.split(";"))):
        print(line)
