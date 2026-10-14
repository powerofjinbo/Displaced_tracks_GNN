"""ACTS v15 steps of the new-sample production (docs/new_samples_spec.md). Runs inside the build container with the
ACTS Python bindings (slurm/acts/produce.sbatch sets up the environment):

    geometry  DIR                                   detectors.csv of the GenericDetector
    pythia    DIR --process ttbar|hss15|hss40 --first N --events N --seed S
              Pythia8 events at 14 TeV with the beam spot below, written as event<N>-particles_gen.csv
    pythia    DIR --process hnl --mass M --ctau C [--decay majorana_va] [--ecm GEV] ...
              W -> mu N, N -> mu mu nu with the card of hnl_card.py (docs/hnl_points_spec.md); the production of
              the HNL points uses hnl_pythia.cc, which writes the same particles plus the generator's parent codes,
              and this ACTS-generator form serves as its cross-check
    sim       DIR --stem STEM --first N --events N --seed S
              reads event<N>-<STEM>.csv, runs Fatras (EM processes on, hits on sensitive surfaces, pMin 0.5 GeV) and
              the smearing digitisation of the GenericDetector, writes particles_initial / particles_final / hits /
              measurements / measurement-simhit-map CSVs

Event numbers are global within a sample, so the per-event random streams (seed + Cantor(algorithm, event)) never
repeat between chunks.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import acts
import acts.examples

u = acts.UnitConstants
ACTS_SRC = Path(os.environ.get("ACTS_SRC", "/work/submit/jinboz1/hnl/acts/src-v15"))
DIGI_JSON = ACTS_SRC / "Examples/Algorithms/Digitization/share/default-smearing-config-generic.json"
# beam spot: sigma_z of the Zenodo samples (about 55 mm, docs/brief.md section 2.4); transverse size as the ACTS
# full-chain examples
BEAMSPOT = acts.Vector4(0.0125 * u.mm, 0.0125 * u.mm, 55.5 * u.mm, 0.0)

PROCESSES = {
    # ttbar as in the Zenodo sample: particles with tau0 > 10 mm (K_S, Lambda, ...) are not decayed by the generator
    # and Fatras does not decay them either (checked on the Zenodo ttbar: K_S, K_L, Lambda in the particle table).
    "ttbar": ["Top:qqbar2ttbar = on", "Top:gg2ttbar = on", "ParticleDecays:limitTau0 = on"],
    # gg -> H(125) -> s s with a long-lived scalar s (PDG 35) decaying to b bbar (70 %) or e+ e- (30 %); every
    # unstable hadron is decayed by the generator (Pythia default), including K_S and Lambda.
    "hss15": ["HiggsSM:gg2H = on", "25:onMode = off", "25:addChannel = 1 1.0 100 35 35",
              "35:m0 = 15.0", "35:mWidth = 0.001", "35:mMin = 14.95", "35:mMax = 15.05", "35:tau0 = 100.0",
              "35:onMode = off", "35:oneChannel = 1 0.7 101 5 -5", "35:addChannel = 1 0.3 101 11 -11"],
    "hss40": ["HiggsSM:gg2H = on", "25:onMode = off", "25:addChannel = 1 1.0 100 35 35",
              "35:m0 = 40.0", "35:mWidth = 0.001", "35:mMin = 39.95", "35:mMax = 40.05", "35:tau0 = 300.0",
              "35:onMode = off", "35:oneChannel = 1 0.7 101 5 -5", "35:addChannel = 1 0.3 101 11 -11"],
}


def detector():
    det, geo, deco = acts.examples.GenericDetector.create()
    return det, geo, deco


def run_geometry(out: str):
    det, geo, _ = detector()      # keep det: it owns the detector elements the surfaces point to
    s = acts.examples.Sequencer(events=1, numThreads=1, logLevel=acts.logging.WARNING)
    s.addWriter(acts.examples.CsvTrackingGeometryWriter(
        level=acts.logging.WARNING, trackingGeometry=geo, outputDir=out, writePerEvent=False))
    s.run()
    del det


def run_pythia(out: str, process: str, first: int, events: int, seed: int, threads: int,
               settings: list[str] | None = None, ecm_gev: float = 14000.0):
    rnd = acts.examples.RandomNumbers(seed=seed)
    gen = acts.examples.EventGenerator.Generator(
        multiplicity=acts.examples.FixedMultiplicityGenerator(n=1),
        vertex=acts.examples.GaussianVertexGenerator(stddev=BEAMSPOT, mean=acts.Vector4(0, 0, 0, 0)),
        particles=acts.examples.pythia8.Pythia8Generator(
            level=acts.logging.WARNING, pdgBeam0=acts.PdgParticle.eProton, pdgBeam1=acts.PdgParticle.eProton,
            cmsEnergy=ecm_gev * u.GeV, settings=settings if settings is not None else PROCESSES[process]))
    evgen = acts.examples.EventGenerator(level=acts.logging.WARNING, generators=[gen],
                                         outputParticles="particles_gen", randomNumbers=rnd)
    s = acts.examples.Sequencer(events=events, skip=first, numThreads=threads, logLevel=acts.logging.WARNING)
    s.addReader(evgen)
    s.addWriter(acts.examples.CsvParticleWriter(level=acts.logging.WARNING, inputParticles="particles_gen",
                                                outputDir=out, outputStem="particles_gen"))
    s.run()


def run_sim(out: str, stem: str, first: int, events: int, seed: int, threads: int):
    det, geo, _ = detector()      # keep det alive while the sequencer runs
    field = acts.ConstantBField(acts.Vector3(0, 0, 2 * u.T))
    rnd = acts.examples.RandomNumbers(seed=seed)
    # the reader's event range is the files in DIR, and skip counts from its first event: DIR holds one chunk
    s = acts.examples.Sequencer(events=events, numThreads=threads, logLevel=acts.logging.WARNING)
    s.addReader(acts.examples.CsvParticleReader(level=acts.logging.WARNING, inputDir=out, inputStem=stem,
                                                outputParticles="particles_input"))
    sim = acts.examples.FatrasSimulation(
        level=acts.logging.WARNING, inputParticles="particles_input", outputParticlesInitial="particles_initial",
        outputParticlesFinal="particles_final", outputSimHits="simhits", randomNumbers=rnd,
        trackingGeometry=geo, magneticField=field, generateHitsOnSensitive=True,
        emScattering=True, emEnergyLossIonisation=True, emEnergyLossRadiation=True, emPhotonConversion=True,
        pMin=0.5 * u.GeV)
    s.addAlgorithm(sim)
    digi_cfg = acts.examples.DigitizationConfig(acts.examples.readDigiConfigFromJson(str(DIGI_JSON)),
                                                trackingGeometry=geo, randomNumbers=rnd, inputSimHits="simhits")
    digi = acts.examples.DigitizationAlgorithm(digi_cfg, acts.logging.WARNING)
    s.addAlgorithm(digi)
    for coll in ("particles_initial", "particles_final"):
        s.addWriter(acts.examples.CsvParticleWriter(level=acts.logging.WARNING, inputParticles=coll,
                                                    outputDir=out, outputStem=coll))
    s.addWriter(acts.examples.CsvSimHitWriter(level=acts.logging.WARNING, inputSimHits="simhits",
                                              outputDir=out, outputStem="hits"))
    s.addWriter(acts.examples.CsvMeasurementWriter(
        level=acts.logging.WARNING, inputMeasurements=digi.config.outputMeasurements,
        inputClusters="", inputSimHits="simhits",
        inputMeasurementSimHitsMap=digi.config.outputMeasurementSimHitsMap, outputDir=out))
    s.run()
    del det


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["geometry", "pythia", "sim"])
    ap.add_argument("dir")
    ap.add_argument("--process", choices=sorted(PROCESSES) + ["hnl"])
    # HNL points (hnl_card.py); the other processes keep 14 TeV
    ap.add_argument("--mass", type=float)
    ap.add_argument("--ctau", type=float)
    ap.add_argument("--ecm", type=float, default=None)
    ap.add_argument("--decay", default=None, help="N decay form of hnl_card.DECAYS")
    ap.add_argument("--stem", default="particles_input")
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--events", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.dir, exist_ok=True)
    if a.mode == "geometry":
        run_geometry(a.dir)
    elif a.mode == "pythia" and a.process == "hnl":
        import hnl_card
        st = hnl_card.settings(a.mass, a.ctau, a.decay or hnl_card.DEFAULT_DECAY)
        run_pythia(a.dir, a.process, a.first, a.events, a.seed, a.threads, settings=st,
                   ecm_gev=a.ecm if a.ecm is not None else hnl_card.ECM_GEV)
    elif a.mode == "pythia":
        run_pythia(a.dir, a.process, a.first, a.events, a.seed, a.threads,
                   ecm_gev=a.ecm if a.ecm is not None else 14000.0)
    else:
        run_sim(a.dir, a.stem, a.first, a.events, a.seed, a.threads)


if __name__ == "__main__":
    main()
