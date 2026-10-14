// Pythia8 244 driver for the HNL points (docs/hnl_points_spec.md).
//
// Generates pp events with the settings of a Pythia8 card (scripts/acts/hnl_card.py) and writes, per event,
// event<N>-particles_gen.csv in the format and with the barcodes of the ACTS v15 generator chain
// (ActsExamples::Pythia8Generator inside ActsExamples::EventGenerator, one hard interaction, Gaussian beam spot), so
// that `acts_sim.py sim --stem particles_gen` simulates it exactly as an ACTS-generated event. On top of what the
// ACTS generator keeps, it writes the generator truth that the ACTS generator drops:
//   event<N>-parents.csv   per written particle: barcode, PDG code of the generator mother (mother1 of the particle's
//                          top copy, so that a muon that radiated keeps the W or N as parent), Pythia status
//   hnl_gen_<first>.csv    per event: W charge and momentum, N four-momentum, production and decay vertex, proper
//                          decay length, the neutrino momentum, the barcodes of the prompt muon and the N daughters,
//                          number of next() calls
//
// Emulated from ACTS v15 (Examples/Algorithms/GeneratorsPythia8/.../Pythia8ProcessGenerator.cpp, Generators/
// EventGenerator.cpp, VertexGenerators.hpp, Io/Csv/src/CsvParticleWriter.cpp):
//   - one std::mt19937 per event; the primary vertex is drawn first (four standard normals x sigma), then Pythia
//     draws from the same engine through a RndmEngine wrapper; Pythia is initialised with its own default generator;
//   - particles: final and visible, beam particles (HepMC status 4) skipped, in event-record order; particle number =
//     1 + index among them; a particle with a production vertex (Pythia hasVertex) gets secondary vertex
//     1 + index of its exact (x, y, z, t) among the vertices met so far; primary vertex 1;
//   - position = Pythia production vertex (mm, time in mm/c) + primary vertex; mass = nominal mass m0; charge in e;
//   - CSV columns particle_id, particle_type, process, vx, vy, vz, vt [ns], px, py, pz [GeV], m, q, values as float32,
//     rows in barcode order (the ACTS particle container is ordered by barcode).
// The per-event seed is std::seed_seq{seed, event}: events are reproducible chunk by chunk.
//
// Build (in the ACTS v15 build container with the LCG_97apython3 view):
//   g++ -O2 -std=c++17 hnl_pythia.cc -I$PYTHIA8/include -L$PYTHIA8/lib -Wl,-rpath,$PYTHIA8/lib -lpythia8 -ldl
// Run:
//   hnl_pythia CARD OUTDIR FIRST NEVENTS SEED ECM_GEV SIGMA_XY_MM SIGMA_Z_MM SIGMA_T_NS

#include "Pythia8/Pythia.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using namespace Pythia8;

namespace {

constexpr double kMmPerNs = 299.792458;  // ACTS: 1 ns = 299.792458 mm (c = 1)
constexpr int kHnl = 50;

struct EngineRndm : public RndmEngine {
  std::mt19937* rng = nullptr;
  double flat() override { return std::uniform_real_distribution<double>(0.0, 1.0)(*rng); }
};

uint64_t barcode(uint64_t vprim, uint64_t vsec, uint64_t particle) {
  return (vprim << 52) | (vsec << 40) | (particle << 24);
}

struct Row {
  uint64_t id;
  int pdg;
  float vx, vy, vz, vt, px, py, pz, m, q;
  int parent, status;
};

// PDG code of the generator mother: mother1 of the top copy (carbon copies and radiating copies of the same
// particle are followed upwards).
int parentCode(const Event& ev, int i) {
  int top = ev[i].iTopCopyId();
  int m = ev[top].mother1();
  return m > 0 ? ev[m].id() : 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 10) {
    std::fprintf(stderr, "usage: %s CARD OUTDIR FIRST NEVENTS SEED ECM_GEV SIGMA_XY_MM SIGMA_Z_MM SIGMA_T_NS\n",
                 argv[0]);
    return 2;
  }
  const std::string card = argv[1], out = argv[2];
  const long first = std::atol(argv[3]), nev = std::atol(argv[4]);
  const unsigned long seed = std::strtoul(argv[5], nullptr, 10);
  const double ecm = std::atof(argv[6]);
  const double sxy = std::atof(argv[7]), sz = std::atof(argv[8]), st = std::atof(argv[9]);

  Pythia pythia("", false);
  pythia.readString("Print:quiet = on");
  if (!pythia.readFile(card)) {
    std::fprintf(stderr, "cannot read card %s\n", card.c_str());
    return 2;
  }
  pythia.settings.mode("Beams:idA", 2212);
  pythia.settings.mode("Beams:idB", 2212);
  pythia.settings.mode("Beams:frameType", 1);
  pythia.settings.parm("Beams:eCM", ecm);
  if (!pythia.init()) {
    std::fprintf(stderr, "Pythia init failed\n");
    return 3;
  }

  const std::string truthPath = out + "/hnl_gen_" + std::to_string(first) + ".csv";
  FILE* ft = std::fopen(truthPath.c_str(), "w");
  if (!ft) return 4;
  std::fprintf(ft,
               "event,n_tries,w_charge,w_px,w_py,w_pz,w_e,n_id,n_px,n_py,n_pz,n_e,n_m,"
               "n_prod_x,n_prod_y,n_prod_z,n_prod_t,n_dec_x,n_dec_y,n_dec_z,n_dec_t,n_tau,"
               "nu_id,nu_px,nu_py,nu_pz,n_visible_daughters,mu_w_id,d1_id,d2_id,"
               "pv_x,pv_y,pv_z,pv_t,n_written\n");

  EngineRndm engine;
  for (long ev = first; ev < first + nev; ++ev) {
    std::seed_seq sq{static_cast<uint32_t>(seed), static_cast<uint32_t>(ev)};
    std::mt19937 rng(sq);
    engine.rng = &rng;
    // primary vertex first, as ActsExamples::EventGenerator::read
    std::normal_distribution<double> normal(0.0, 1.0);
    const double n0 = normal(rng), n1 = normal(rng), n2 = normal(rng), n3 = normal(rng);
    const double pv[4] = {n0 * sxy, n1 * sxy, n2 * sz, n3 * st * kMmPerNs};  // time in mm, as ACTS
    pythia.rndm.rndmEnginePtr(&engine);
    int tries = 0;
    bool ok = false;
    while (!ok && tries < 100) {
      ++tries;
      ok = pythia.next();
    }
    if (!ok) {
      std::fprintf(stderr, "event %ld: Pythia next() failed 100 times\n", ev);
      return 5;
    }
    const Event& E = pythia.event;

    std::vector<Row> rows;
    std::vector<std::array<double, 4>> vertices;
    uint64_t muW = 0, d1 = 0, d2 = 0;
    int nVis = 0;
    for (int i = 0; i < E.size(); ++i) {
      const Particle& p = E[i];
      if (p.statusHepMC() == 4) continue;
      if (!p.isFinal()) continue;
      if (!p.isVisible()) continue;
      const std::array<double, 4> v4 = {p.xProd(), p.yProd(), p.zProd(), p.tProd()};
      uint64_t vsec = 0;
      if (p.hasVertex()) {
        auto it = std::find(vertices.begin(), vertices.end(), v4);
        if (it == vertices.end()) {
          vertices.push_back(v4);
          vsec = vertices.size();
        } else {
          vsec = 1 + static_cast<uint64_t>(std::distance(vertices.begin(), it));
        }
      }
      const uint64_t id = barcode(1, vsec, 1 + rows.size());
      const double pabs = std::sqrt(p.px() * p.px() + p.py() * p.py() + p.pz() * p.pz());
      Row r;
      r.id = id;
      r.pdg = p.id();
      r.vx = static_cast<float>(pv[0] + v4[0]);
      r.vy = static_cast<float>(pv[1] + v4[1]);
      r.vz = static_cast<float>(pv[2] + v4[2]);
      r.vt = static_cast<float>((pv[3] + v4[3]) / kMmPerNs);
      // ACTS stores direction and |p|; px = |p| * unit direction
      r.px = static_cast<float>(pabs > 0 ? pabs * (p.px() / pabs) : 0.0);
      r.py = static_cast<float>(pabs > 0 ? pabs * (p.py() / pabs) : 0.0);
      r.pz = static_cast<float>(pabs > 0 ? pabs * (p.pz() / pabs) : 0.0);
      r.m = static_cast<float>(p.m0());
      r.q = static_cast<float>(p.charge());
      r.parent = parentCode(E, i);
      r.status = p.status();
      rows.push_back(r);
      if (std::abs(r.parent) == 24 && std::abs(r.pdg) == 13) muW = id;
      if (std::abs(r.parent) == kHnl) {
        ++nVis;
        if (d1 == 0) d1 = id; else d2 = id;
      }
    }
    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.id < b.id; });

    char name[64];
    std::snprintf(name, sizeof(name), "/event%09ld-particles_gen.csv", ev);
    FILE* fp = std::fopen((out + name).c_str(), "w");
    std::snprintf(name, sizeof(name), "/event%09ld-parents.csv", ev);
    FILE* fq = std::fopen((out + name).c_str(), "w");
    if (!fp || !fq) return 4;
    std::fprintf(fp, "particle_id,particle_type,process,vx,vy,vz,vt,px,py,pz,m,q\n");
    std::fprintf(fq, "particle_id,parent_pdg,status\n");
    for (const Row& r : rows) {
      std::fprintf(fp, "%" PRIu64 ",%d,0,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n", r.id, r.pdg, r.vx, r.vy,
                   r.vz, r.vt, r.px, r.py, r.pz, r.m, r.q);
      std::fprintf(fq, "%" PRIu64 ",%d,%d\n", r.id, r.parent, r.status);
    }
    std::fclose(fp);
    std::fclose(fq);

    // generator truth: the W (last copy), the N (last copy, at its decay) and its invisible daughter
    int iW = -1, iN = -1;
    for (int i = 0; i < E.size(); ++i) {
      if (E[i].idAbs() == 24) iW = i;
      if (E[i].idAbs() == kHnl) iN = i;
    }
    double w[4] = {0, 0, 0, 0}, nn[5] = {0, 0, 0, 0, 0}, nprod[4] = {0, 0, 0, 0}, ndec[4] = {0, 0, 0, 0};
    double nu[3] = {0, 0, 0}, tau = -1;
    int wq = 0, nid = 0, nuid = 0;
    if (iW >= 0) {
      wq = E[iW].id() > 0 ? 1 : -1;
      w[0] = E[iW].px(); w[1] = E[iW].py(); w[2] = E[iW].pz(); w[3] = E[iW].e();
    }
    if (iN >= 0) {
      const Particle& N = E[iN];
      nid = N.id();
      nn[0] = N.px(); nn[1] = N.py(); nn[2] = N.pz(); nn[3] = N.e(); nn[4] = N.m();
      nprod[0] = pv[0] + N.xProd(); nprod[1] = pv[1] + N.yProd(); nprod[2] = pv[2] + N.zProd();
      nprod[3] = (pv[3] + N.tProd()) / kMmPerNs;
      ndec[0] = pv[0] + N.xDec(); ndec[1] = pv[1] + N.yDec(); ndec[2] = pv[2] + N.zDec();
      ndec[3] = (pv[3] + N.tDec()) / kMmPerNs;
      tau = N.tau();
      for (int d : N.daughterList()) {
        if (E[d].idAbs() == 12 || E[d].idAbs() == 14 || E[d].idAbs() == 16) {
          nuid = E[d].id();
          nu[0] = E[d].px(); nu[1] = E[d].py(); nu[2] = E[d].pz();
        }
      }
    }
    std::fprintf(ft,
                 "%ld,%d,%d,%.9g,%.9g,%.9g,%.9g,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,"
                 "%.9g,%.9g,%d,%.9g,%.9g,%.9g,%d,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%.9g,%.9g,%.9g,%.9g,%zu\n",
                 ev, tries, wq, w[0], w[1], w[2], w[3], nid, nn[0], nn[1], nn[2], nn[3], nn[4], nprod[0], nprod[1],
                 nprod[2], nprod[3], ndec[0], ndec[1], ndec[2], ndec[3], tau, nuid, nu[0], nu[1], nu[2], nVis, muW,
                 d1, d2, pv[0], pv[1], pv[2], pv[3] / kMmPerNs, rows.size());
  }
  std::fclose(ft);
  return 0;
}
