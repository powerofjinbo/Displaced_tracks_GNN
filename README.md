<p align="center">
  <img src="branding/canto_banner.gif" width="768"
       alt="Tercet, a small orange gecko, hangs the letters C, A, N, T, O on a wire in the dark">
</p>

# Canto: displaced-particle tracking with a graph neural network trained on ttbar events

**Canto** (**C**hained-triplet, origin-**A**gnostic **N**etworks for **T**racking **O**ff-vertex) is a graph neural
network (GNN) tracking pipeline for particles produced away from the collision point. A search for long-lived
particles whose properties are not known in advance cannot train its track reconstruction on the signal it looks
for, so every learned stage of Canto is trained on generic top-quark pair (ttbar) events only and evaluated on a
public benchmark of heavy neutral leptons (HNLs), `pp -> W -> mu N`, `N -> nu mu+ mu-`, in the ACTS generic detector
without pileup. Canto keeps the three stages of the Exa.TrkX pipeline and changes each: an edge classifier with
messages between hit triplets and inputs independent of the production point, track builders that keep the hits of
one layer apart and merge helix fragments, and a graph that adds the legs of helix-consistent triplets.

A paper is in preparation. This repository holds the code, configurations, job scripts, tests, data splits, result
tables and figures.

## Results

HNL benchmark (validation and test are kept separate; uncertainties are statistical):

- **Validation.** Trained on ttbar only, Canto-L finds as many displaced HNL daughters as the same pipeline trained
  on the signal, (+0.08 +- 0.33) points apart; acorn (Exa.TrkX) gains (1.48 +- 0.35) points from training on the
  signal.
- **One-time test** of the frozen models under a preregistered protocol: the ttbar-trained Canto-L finds
  (99.08 +- 0.15) % of the displaced daughters and its three preregistered claims hold. In a post-freeze comparison
  it finds (1.10 +- 0.22) points more than the signal-trained acorn configuration chosen on validation.

TrackML challenge, throughput phase (pileup 200; Canto-L retrained on throughput-phase events, one seed, evaluated
once on 300 test events): (99.65 +- 0.01) % of the targets at a fake-track rate of (1.68 +- 0.01) %. Recounted with
the evaluation code of three published GNN studies on this phase, none of their efficiencies is higher; with all
hits and the EggNet code, at a fake rate not resolved from EggNet's, Canto-L finds 99.37 % of the particles against
95.87 %.

Release cards: `figures/release/` (`canto_card_full` for every dataset and model, `canto_card*` for the HNL
benchmark, `canto_l_trackml_card` for TrackML). The TrackML model card is
`docs/model_card_canto_l_trackml.md`.

## Models

All variants are trained on ttbar only:

| variant | graph | edge classifier | track builder |
|---|---|---|---|
| Canto-B | metric-learning graph (radius r*) | triplet interaction GNN with inner-hit truncation augmentation (`configs/stage1/triplet_symregion_trunc.yaml`) | same-layer Mutex Watershed (one or two passes) + fragment merging (hand-cut or learned) |
| Canto-H | + wider-radius candidates with hand-cut helix-triplet support (`hnl.graphs.helix_union`) | same | same |
| Canto-L | + learned triplet support instead of the hand cuts (`hnl.graphs.learned_triplet_filter`) | same | same |

Baselines: acorn (the Exa.TrkX pipeline in the ATLAS GNN4ITk framework) retrained on ttbar, on the HNL signal and on
both, and the interaction network of DeZoort et al. (`src/hnl/models/dezoort_in.py`).

## Layout

- `src/hnl/` — the package: `data`, `truth`, `geometry`, `graphs`, `models`, `train`, `trackbuild`, `vertex`,
  `eval`, `plots`, and `trackml` (conversion, evaluation and the ports of the published evaluation codes,
  `trackml/refports`).
- `configs/` — model and pipeline configurations; `configs/trackml/` for TrackML, including the data split.
- `slurm/` — job scripts and orchestrators; `tests/` — unit tests (`pytest`; tests marked `data` or `gpu` need the
  samples or a CUDA device).
- `splits/` — the event lists of every HNL and ttbar split; `results/` — result tables; `figures/` — figures.
- `env/` — environment build; `patches/acorn/` — the patches applied to acorn at install time.

## Installation

The environment was built and run on the MIT subMIT cluster (NVIDIA A30, CUDA 12.6 toolchain):

```
export HNL_ROOT=/path/outside/home         # caches, environment, logs; env/env.sh refuses locations under $HOME
git clone https://github.com/powerofjinbo/Displaced_tracks_GNN.git $HNL_ROOT/hnl-displaced
cd $HNL_ROOT/hnl-displaced
bash env/make_env.sh --stage deps          # micromamba, Python 3.12, torch 2.11 (cu126), PyG 2.7, lightning 2.6
bash env/make_env.sh --stage build,smoke   # on a GPU node: FRNN, acorn 99d15341 + patches/acorn, this package
source env/activate.sh
```

`HNL_ROOT` and `HNL_DATA` (the samples and derived data) must be exported in every shell before `env/activate.sh`;
otherwise `env/env.sh` falls back to the subMIT locations. `env/lock.txt` records the versions of the environment
used for the results. The job scripts in `slurm/` carry the
site's paths (`#SBATCH -o` lines, `source` lines and partition names) and need adapting elsewhere.

## Data

- HNL and ttbar samples: X. Ju, "Large Radius Tracking Events in pp collisions with a Generic detector",
  doi:10.5281/zenodo.6812533. The splits are in `splits/`.
- TrackML: throughput phase doi:10.5281/zenodo.4730157; the accuracy-phase events (Kaggle `train_1`) for the
  cross-phase test. The split rule and archive checksums are in `configs/trackml/split_tp_v1.json`.

## Provenance

The project was developed in a private repository whose history is kept as the full record. Commit hashes cited in
comments and documents (for example the selection freeze `aecb23f`) refer to that history; the development log is
not part of this repository. Model checkpoints are not included.

Third-party material is listed in `NOTICE`.
