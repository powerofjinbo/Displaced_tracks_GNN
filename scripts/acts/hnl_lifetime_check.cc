// Lifetime check of the HNL card (docs/hnl_points_spec.md section 10). Generator level only, no simulation.
//
// For every event it records the proper lifetime tau of the N as Pythia 8.244 assigns it in the process record
// (pythia.process, where ResonanceDecays draws it when the W decays) and in the final event record (every copy of the
// N: the shower copies it when it takes the recoil of a photon emitted by the W muon), so that a bias of the draw, a
// change of tau between copies, or an event selection that depends on tau can be told apart.
//
// The random numbers come, as in hnl_pythia.cc, from one std::mt19937 per event seeded with std::seed_seq{seed,
// event}, after four standard normals for the beam spot (RNG=engine), or from Pythia's own generator seeded once
// (RNG=pythia).
//
// Build (ACTS v15 build container, LCG_97apython3 view):
//   g++ -O2 -std=c++17 hnl_lifetime_check.cc -I$PYTHIA8/include -L$PYTHIA8/lib -Wl,-rpath,$PYTHIA8/lib -lpythia8 -ldl
// Run:
//   hnl_lifetime_check CARD FIRST NEVENTS SEED ECM_GEV OUT.csv engine|pythia
// Output, one row per accepted event: event, tries, n copies of the N in the process and event records, tau in the
// process record, tau of the first and last copy in the event record, number of distinct tau values among the copies,
// status of the last copy, whether it decayed, the W mass, |p| and m of the last copy, and the proper decay length
// |v_dec - v_prod| m / |p| of the last copy.

#include "Pythia8/Pythia.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
#include <string>

using namespace Pythia8;

namespace {
constexpr int kHnl = 50;
struct EngineRndm : public RndmEngine {
  std::mt19937* rng = nullptr;
  double flat() override { return std::uniform_real_distribution<double>(0.0, 1.0)(*rng); }
};
}  // namespace

int main(int argc, char** argv) {
  if (argc != 8) {
    std::fprintf(stderr, "usage: %s CARD FIRST NEVENTS SEED ECM_GEV OUT.csv engine|pythia\n", argv[0]);
    return 2;
  }
  const std::string card = argv[1], outPath = argv[6], mode = argv[7];
  const long first = std::atol(argv[2]), nev = std::atol(argv[3]);
  const unsigned long seed = std::strtoul(argv[4], nullptr, 10);
  const double ecm = std::atof(argv[5]);
  if (mode != "engine" && mode != "pythia") return 2;

  Pythia pythia("", false);
  pythia.readString("Print:quiet = on");
  if (!pythia.readFile(card)) return 2;
  pythia.settings.mode("Beams:idA", 2212);
  pythia.settings.mode("Beams:idB", 2212);
  pythia.settings.mode("Beams:frameType", 1);
  pythia.settings.parm("Beams:eCM", ecm);
  if (mode == "pythia") {
    pythia.readString("Random:setSeed = on");
    pythia.settings.mode("Random:seed", static_cast<int>((seed + first) % 900000000));
  }
  if (!pythia.init()) return 3;

  FILE* fo = std::fopen(outPath.c_str(), "w");
  if (!fo) return 4;
  std::fprintf(fo, "event,tries,n_proc,n_evt,tau_proc,tau_first,tau_last,n_tau_distinct,status_last,decayed,"
                   "m_w,p_last,m_last,ctau_vertex\n");
  EngineRndm engine;
  for (long ev = first; ev < first + nev; ++ev) {
    std::seed_seq sq{static_cast<uint32_t>(seed), static_cast<uint32_t>(ev)};
    std::mt19937 rng(sq);
    if (mode == "engine") {
      engine.rng = &rng;
      std::normal_distribution<double> normal(0.0, 1.0);
      for (int k = 0; k < 4; ++k) normal(rng);  // the beam spot draws of hnl_pythia.cc
      pythia.rndm.rndmEnginePtr(&engine);
    }
    int tries = 0;
    bool ok = false;
    while (!ok && tries < 100) {
      ++tries;
      ok = pythia.next();
    }
    if (!ok) return 5;
    const Event& P = pythia.process;
    const Event& E = pythia.event;
    int nProc = 0, nEvt = 0, iLast = -1;
    double tauProc = -1, tauFirst = -1, mW = -1;
    std::set<double> taus;
    for (int i = 0; i < P.size(); ++i) {
      if (P[i].idAbs() == kHnl) {
        ++nProc;
        tauProc = P[i].tau();
      }
      if (P[i].idAbs() == 24) mW = P[i].m();
    }
    for (int i = 0; i < E.size(); ++i) {
      if (E[i].idAbs() != kHnl) continue;
      ++nEvt;
      if (tauFirst < 0) tauFirst = E[i].tau();
      taus.insert(E[i].tau());
      iLast = i;
    }
    double tauLast = -1, pLast = -1, mLast = -1, ctauV = -1;
    int statusLast = 0, decayed = 0;
    if (iLast >= 0) {
      const Particle& N = E[iLast];
      tauLast = N.tau();
      statusLast = N.status();
      decayed = N.daughter1() > 0 ? 1 : 0;
      pLast = N.pAbs();
      mLast = N.m();
      const double dx = N.xDec() - N.xProd(), dy = N.yDec() - N.yProd(), dz = N.zDec() - N.zProd();
      ctauV = std::sqrt(dx * dx + dy * dy + dz * dz) * mLast / pLast;
    }
    std::fprintf(fo, "%ld,%d,%d,%d,%.9g,%.9g,%.9g,%zu,%d,%d,%.6g,%.9g,%.9g,%.9g\n", ev, tries, nProc, nEvt, tauProc,
                 tauFirst, tauLast, taus.size(), statusLast, decayed, mW, pLast, mLast, ctauV);
  }
  std::fclose(fo);
  return 0;
}
