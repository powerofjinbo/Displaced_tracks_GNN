<!-- Written 2026-10-03 by the acorn-map workflow (five read-only subsystem readers + a synthesizer that resolved contradictions from source; raw maps: /work/submit/jinboz1/hnl/tmp/acorn_map_result.json; scratch checks: /work/submit/jinboz1/hnl/tmp/acorn_map/synth/). Reviewed by the lead. Decisions on its open questions: DECISIONS.md 2026-10-03. -->

# acorn map: building the acorn baseline for the HNL npz data

Scope: everything needed to run the brief §5f baseline (metric learning on (r, phi, z), radius graph, optional filter,
InteractionGNN with hidden 128 and 8 iterations, connected components) on the HNL / ttbar npz events, which acorn
has no reader for.

- acorn: `/work/submit/jinboz1/hnl/env/src/acorn`, commit `99d153418981` + `patches/acorn/0001-0007` (the working
  tree is the patched tree). Bare paths below are relative to the acorn root. `src/`, `docs/`, `splits/`, `env/`,
  `tests/` are this repo. `site-packages/` means `/work/submit/jinboz1/hnl/env/venv/lib/python3.12/site-packages`.
  Bare file names such as `metric_learning.py` are unique in acorn, except `connected_components.py`, which always
  means `acorn/stages/track_building/models/connected_components.py`. `:N` alone continues the last file named.
  `ExN:L` is a line of `examples/Example_N/<stage>_train.yaml`.
- Environment: torch 2.11.0+cu126, torch-geometric 2.7.0, lightning 2.6.1, pandas 3.0.2 (`env/lock.txt`).
- Sources: five subsystem maps (data, metric learning, GNN, track building, running). Where they disagreed, the acorn
  and Lightning source decided (Appendix A). CPU-only checks made for this page are the scratch scripts
  `synth/s1_sampler.py`, `s3_match.py`, `s4_sizes.py`, `s5_chain.py`, `s6_resolver.py`, `s7_predict_trainer.py` and
  `s8_cc_inmem.py` in `/work/submit/jinboz1/hnl/tmp/acorn_map/` (not part of the repo). Nothing was trained and no GPU
  was used, so GPU step time and memory are not measured.

## 0. Rules that matter most

1. Write events with acorn's short names. Never name a field `y` (it becomes `edge_y`) or a per-hit field
   `particle_id` (it becomes the per-true-edge `track_particle_id`).
2. `event_id` is a 9-digit string that is unique across samples: `f"{code}{index:08d}"`.
3. The metric-learning input is flat: `<feature_store>/<split>/event<ID>-graph.pyg.gz`, one file per event.
4. No `hard_cuts` in any training config (brief §5f: no truth-based node cut), and `hard_cuts: {}` in every infer
   config.
5. Set `undirected: true` in metric learning and in the GNN. Its default is `false`, which mislabels the ~1 % of
   true edges that point inward in r²+z².
6. Our pT is in GeV. The acorn examples use MeV thresholds (`1000.`), so translate them to `1.0`.
7. Always pass `-c <checkpoint>`, and use a fresh `stage_dir` per run. Set `log_wandb: false`, `accelerator`,
   `devices`, `nodes` and `num_workers` explicitly.
8. Seed with `seed_everything(seed, workers=True)` before the module is built, because acorn seeds nothing. acorn's
   trainer then shuffles training events with that seed.
9. Keep per-event files on node-local disk and persist only packed archives. The budget is counted on the
   `hnl_lrt` subtree after all trainings are done (user, 2026-10-03, DECISIONS.md).
10. Compute every number we report ourselves (`hnl.eval.matching`, `hnl.graphs.ceiling`). acorn's own metrics use
    different definitions (§5).

## 1. Pipeline at the pinned commit

### 1.1 Patches

| Patch | Where | Effect |
|---|---|---|
| 0001 | `acorn/stages/track_building/utils.py:41-43` | `load_reconstruction_df` maps `track_edges` through `hit_id`. Before the patch it was wrong after node cuts. |
| 0002 | `acorn/stages/track_building/models/connected_components.py:80-84` | `on_true_graph_edges` selects edges with `edge_y`. Before, `graph.y` was None and every edge was kept. |
| 0003 | `acorn/stages/graph_construction/utils.py:206-238` | `within` / `not_within` are fixed in the graph-construction copy of `get_condition_lambda`. |
| 0004 | `acorn/stages/graph_construction/utils.py:146-181` | `map_value_to_edges` raises when a condition length matches none of E, T, N. Before, the condition was silently ignored. |
| 0005 | `acorn/utils/loading_utils.py:37-49` + savers | `flatten_event_id`: outputs are named `event<ID>`, not `event['<ID>']`. |
| 0006 | `acorn/stages/data_reading/models/trackml_reader.py` | Deterministic TrackML split. We do not use the reader. |
| 0007 | `acorn/stages/edge_classifier/models/filter.py:185-202` | The Filter evaluates in chunks of `eval_chunk_size` (default 2,000,000) instead of catching OOM. |

The repo's characterisation tests for the brief §4.5 pitfalls are in `tests/acorn_pitfalls/test_p01..p12`.

### 1.2 Stages and classes

| `stage:` / `model:` | Code | Reads | Writes per event (`stage_dir/<split>/`) |
|---|---|---|---|
| `graph_construction` / `MetricLearning` | `acorn/stages/graph_construction/models/metric_learning.py:38-504` | feature store via `EventDataset` (`acorn/stages/graph_construction/graph_construction_stage.py:241-432`) | `event<ID>.pyg.gz`, adding `edge_index`, `y`, `truth_map`, `hit_R` (`metric_learning.py:491-504`) |
| `edge_classifier` / `Filter` (optional) | `acorn/stages/edge_classifier/models/filter.py:89-202` | metric-learning outputs via `GraphDataset` (`acorn/stages/edge_classifier/edge_classifier_stage.py:613-846`) | `event<ID>.pyg.gz`, adding `scores`, `edge_weights` |
| `edge_classifier` / `InteractionGNN` | `acorn/stages/edge_classifier/models/interaction_gnn.py:36-242` | metric-learning outputs, or filter outputs + `input_cut` | `event<ID>.pyg.gz`, adding `scores`, `edge_weights`; `truth_map` recomputed (`edge_classifier_stage.py:535-558`) |
| `track_building` / `ConnectedComponents` | `acorn/stages/track_building/models/connected_components.py:41-178` | GNN outputs via track-building `GraphDataset` (`acorn/stages/track_building/track_building_stage.py:390-484`) | `<prefix>event<ID>.pyg.gz` with `hit_track_labels`, `time_taken`, plus `stage_dir/<split>_tracks/<prefix>event<ID>.txt` |

We do not use the data-reading stage (`EventReader`, `acorn/stages/data_reading/data_reading_stage.py`). It needs
per-event CSVs, a `hardware` column and production vertices, and our npz `vx, vy, vz` are end-state values
(`src/hnl/data/npz.py`). acorn has no npz reader. We write the feature store ourselves (§2).

Variants not to use: `InteractionGNNWithPyG` / `InteractionGNN2WithPyG` cannot be constructed with PyG 2.7. The
jittable classes take `forward(x, edge_index, ...)`, so the stage cannot train them. `GravNetMetricLearning` crashes
in `Trainer.predict`. `Walkthrough` is not exported (`acorn/stages/track_building/__init__.py`). (Reported by the
GNN, metric-learning and track maps.)

### 1.3 Data flow and what is saved where

```
npz -> hnl.data.convert -> derived/events/v1/{hits,particles,edges}/<sample>/shard_*.parquet   (500 events per shard)
    -> [ours] feature-store writer -> <fs>/{trainset,valset,testset}/event<ID>-graph.pyg.gz
    -> MetricLearning  train: <ml>/artifacts/{best-*,last-*}.ckpt
                       infer: <ml>/<split>/event<ID>.pyg.gz
    -> [Filter]        train/infer, same layout under <filter>/
    -> InteractionGNN  train: <gnn>/artifacts/*.ckpt   infer: <gnn>/<split>/event<ID>.pyg.gz
    -> CC (acorn infer, or in memory, §3.5) -> hit_track_labels -> our metrics (§5)
```

- Checkpoints go to `stage_dir/artifacts/`. They are named `best-<id>-<metric>=<v>-epoch=<e>.ckpt` (top-k 1, with
  `<metric>` the monitored metric) and `last-<id>.ckpt`, where `<id>` is the W&B run id or empty
  (`acorn/core/core_utils.py:134-155`).
- Logs: with `log_wandb: false` they go to a `CSVLogger` at `stage_dir/lightning_logs/version_N/{metrics.csv,hparams.yaml}`.
  Otherwise a `WandbLogger` writes under `stage_dir` (`acorn/core/core_utils.py:121-132`). `env/env.sh` sets
  `WANDB_MODE=offline`.
- `acorn infer` builds a `Trainer` without a logger argument (`acorn/core/infer_stage.py:107-111`), so Lightning's
  default logger writes `./lightning_logs/` in the current directory.
- In a Slurm batch job, `default_root_dir` is `./$SLURM_JOB_ID` (`acorn/core/core_utils.py:45-54`). Lightning's
  pre-timeout `hpc_ckpt_<n>.ckpt` files go there (§4.4).
- `acorn eval` of track building writes `stage_dir/results_summary_<style>.txt` and
  `track_reconstruction_eff_vs_<var>_<style>.png` (`acorn/stages/track_building/track_building_stage.py:243-284`).

Per step:

- **Metric-learning training** (`metric_learning.py:216-292`):
  1. Embed all hits without gradients.
  2. The queries are `points_per_batch` random hits. That is every hit for us (N ≤ 3.4k < 50k).
  3. Add hard negatives with `build_edges(r_train, knn, FRNN)` (`:150-170`), `int(randomisation × n_queries)`
     random pairs (`:172-184`), and the signal true edges (`:186-199`).
  4. Take `torch.unique` and re-embed the included hits with gradients.
  5. Label the pairs with `get_truth` (`:273-287`): the edges are flipped to point outward in `hit_R`, then
     intersected with `track_edges`.
  6. Weight them with `acorn.utils.loading_utils.handle_weighting`.
  7. Apply a hinge loss with margin² on the squared distance, taking separate means over negatives and positives
     (`:325-367`).
- **Metric-learning evaluation and inference** (`shared_evaluation`, `metric_learning.py:369-411`) build the full
  radius graph:
  - validation: `r_train` / `knn_val` (`:441-448`)
  - test: `r_train` / 1000 (`:450-454`)
  - predict: `r_infer` / `knn_infer` (`:471-489`)

  The saved `edge_index` has one column per unordered pair, oriented outward in `hit_R` and sorted. So the saved E
  is already the graph size in unique unordered pairs.
- **InteractionGNN**:
  - It appends the reversed edges (`interaction_gnn.py:200-202`), runs 8 message steps with shared weights and
    activation checkpointing (`:204-215`), and returns the mean of the two directed logits (`:188-192`).
  - The loss is weighted BCE divided by Σ|w| (`edge_classifier_stage.py:284-343`).
  - Prediction saves sigmoid scores on every input edge, with no threshold.
- **ConnectedComponents** keeps edges with `edge_scores > score_cut` (strict) and runs scipy connected components.
  Hits with no kept edge get label −1 (`connected_components.py:62-112`).

## 2. Event format our converter must produce

### 2.1 Layout

There is no particle table. A tensor field has one of three layouts, decided only by its name prefix
(`acorn/utils/mapping_utils.py:61-69`). The prefix controls how hard cuts slice a tensor, how weighting conditions are
mapped, and how score cuts apply.

- **node-like** (`hit_*`): one value per hit, [N].
- **track-like** (`track_*`): one value per true edge, [T], aligned with the columns of `track_edges` [2, T].
  Particle quantities are repeated on each of the particle's edges. A particle with fewer than 2 hits does not exist
  for acorn.
- **edge-like** (`edge_*`, including `edge_index`): one value per constructed edge, [E].
- A field without one of these prefixes is "other" and is never sliced, for example `event_id`, `config`,
  `num_nodes`.

### 2.2 Fields the converter writes

Names are as stored on disk (acorn default naming, `variable_with_prefix` unset). The in-memory name follows the
arrow.

| On disk → in memory | Level | dtype, shape | Required by |
|---|---|---|---|
| `hit_id` | hit | int64 [N], `arange(N)` = npz row = node index | `infer_num_nodes` (`acorn/utils/loading_utils.py:605-612`); CC track lists (`connected_components.py:130`); acorn eval (`acorn/stages/track_building/utils.py:34-43`) |
| `x` → `hit_x` | hit | float32 [N], Cartesian x = r cos phi [mm] | REQUIRED by `test_data` (`graph_construction_stage.py:121`, `edge_classifier_stage.py:174-180`); CC `num_nodes` (`connected_components.py:90-93`) |
| `r` → `hit_r`, `phi` → `hit_phi`, `z` → `hit_z` | hit | float32 [N] [mm, rad, mm], as stored in the npz | `node_features` (`metric_learning.py:133-139`, `interaction_gnn.py:197-199`); `hit_R` (`acorn/stages/track_building/utils.py:329-337`); CC hit order (`connected_components.py:132`) |
| `eta` → `hit_eta` | hit | float32 [N], `asinh(z/r)` | acorn eval plots only (`acorn/utils/eval_utils.py:112`, `:369`) |
| `track_edges` | true edge | int64 [2, T], npz `layerless_true_edges` (rows, chain order) | REQUIRED by `test_data`; truth (`metric_learning.py:273-287`); EC truth map (`edge_classifier_stage.py:539-546`); eval (`acorn/stages/track_building/utils.py:43`) |
| `particle_id` → `track_particle_id` | true edge | int64 [T], particle of each true edge | acorn CC eval (`acorn/stages/track_building/utils.py:44`, `:60`) |
| `pt` → `track_particle_pt` | true edge | float32 [T], table pT in GeV (final state) | acorn eval (`acorn/stages/track_building/utils.py:61`; `acorn/utils/eval_utils.py:110`); weighting / `target_tracks` conditions |
| `eta_particle` → `track_particle_eta` | true edge | float32 [T], table eta | acorn eval eta column (`acorn/stages/track_building/utils.py:63-64`); `target_tracks` |
| `nhits` → `track_particle_nhits` | true edge | int64 [T], `n_hits` = number of space points of the particle | only through weighting / `target_tracks` conditions |
| `event_id` | event | str of 9 digits | output names (`metric_learning.py:496-502`, `edge_classifier_stage.py:550-556`, `track_building_stage.py:322-356`); `int(event_id)` in eval (`acorn/stages/track_building/utils.py:210-213`); `_shard_path` regex (`acorn/utils/loading_utils.py:59-64`) |
| `config` | event | `[]` (a list) | every saver calls `.append(hparams)` (`metric_learning.py:492`, `edge_classifier_stage.py:538`, `connected_components.py:128`) |
| `hit_y` (optional) | hit | float32 [N] | nothing in acorn; keeps the y coordinate |
| `hit_particle_id` (ours) | hit | int64 [N], npz `pid` | our DM matching directly on acorn outputs (§5) |

Fields the stages add (do not write them): `edge_index` int64 [2, E]; `y` → `edge_y` bool [E]; `truth_map` →
`track_to_edge_map` int64 [T] (−1 = true edge not in the graph); `hit_R` float [N] in scaled units
((r/s_r)² + (z/s_z)² with the `node_scales`, i.e. (r/1000)² + (z/1000)² here; never unscaled); `scores` →
`edge_scores` float32 [E]; `edge_weights` float32 [E]; `hit_track_labels` int64 [N] (−1 = no track); `time_taken`
float; `batch`, `ptr`, `num_nodes`.

Stage outputs are PyG `DataBatch` objects: `config` nests one level per stage and `event_id` is flattened by patch
0005. Keep `batch_size` 1, as acorn does: `track_edges` has no "index" in its name, so collation neither increments
it nor concatenates it along the edge dimension.

### 2.3 Naming rules

- With `variable_with_prefix` unset, loading renames short names through `variable_name_prefix_map`
  (`acorn/utils/loading_utils.py:441-527`, applied at `:530-534`), and saving strips the prefixes again (`:537-544`).
  Config keys are rewritten the same way (`:547-602`): `node_features`, `edge_features`, weighting conditions,
  `target_tracks`, `hard_cuts`, `feature_sets`. All TrackML examples use this scheme. Use it, and keep it identical in
  every stage.
- `y` ↔ `edge_y` (`:517`), `truth_map` ↔ `track_to_edge_map` (`:518`), `scores` ↔ `edge_scores` (`:525`). So write
  `y: true` in configs, and expect `y`, `truth_map`, `scores` in the files.
- Do not use these short names for anything else:
  - `y`: never a hit coordinate (use `hit_y`).
  - `particle_id`: per true edge (`:509`). A per-hit particle id must be `hit_particle_id`.
  - `eta`: hit eta (`:446`). Particle eta is `eta_particle` (`:515`).
- A name that is not in the map and has no `hit_` / `edge_` / `track_` prefix is "other". It is never sliced and is
  invisible to conditions. Give new per-hit fields a `hit_` prefix, for example `hit_hid` or `hit_geoid`.
- In `target_tracks` of the CC eval, `eta` is rewritten to `hit_eta`, which is node-level, and evaluation fails. Use
  `eta_particle` there.

### 2.4 Files, directories, ids

- Feature store: `<fs>/<split>/event<ID>-graph.pyg[.gz]` for `split` in `trainset`, `valset`, `testset`.
  - `EventDataset` lists the directory non-recursively (`graph_construction_stage.py:324-325`). Never set
    `subdir_size` on the feature store.
  - It takes the last digit group of each filename as the id and sorts ids as strings (`:331-333`). It keeps the first
    `data_split[i]` ids and asserts that this many exist (`:338-342`).
  - It loads `-graph.pyg` before `.pyg` (`:278-291`).
  - A file named `<prefix>_event<ID>` is found only if `event_prefix` is set (`:345-366`).
- `save_pyg` writes `path + ".gz"` at gzip level 1 through an atomic `.gz.tmp` rename. It creates directories only
  when `subdir` is set, so the converter must `os.makedirs(<fs>/<split>)` itself (`acorn/utils/loading_utils.py:68-86`).
  `load_pyg` tries `.gz` first (`:89-101`).
- Id scheme: `event_id = f"{code}{index:08d}"` with ttbar 1, hnl10_100 2, hnl15_100 3, hnl15_200 4. Why:
  - Metric-learning and EC outputs are named `event<ID>.pyg.gz` without `event_prefix` (`metric_learning.py:502`,
    `edge_classifier_stage.py:556`). Arms (ii) and (iii) mix samples whose indices overlap, so a per-sample index
    would make files overwrite each other.
  - Ids of equal length sort as strings in numeric order.
  - `int()` and the `event(\d+)` regex both work.
- Which events: exactly those in `splits/*.txt` (dropped events are already excluded). Events per sample: ttbar
  7999/1000/1000, hnl10_100 7999/1000/1000, hnl15_200 8000/999/1000, hnl15_100 4999 (external).
- `data_split` takes the first N string-sorted ids, so with mixed samples it selects by sample code, not at random.
  Build each split directory with exactly the events you want, and set `data_split` to those counts.

### 2.5 Minimal converter (checked through acorn's metric learning, IGNN, CC and tracking_efficiency on CPU: `synth/s5_chain.py`)

```python
import os, numpy as np, pandas as pd, torch
from torch_geometric.data import Data
from acorn.utils.loading_utils import save_pyg

SAMPLE_CODE = {"ttbar": 1, "hnl10_100": 2, "hnl15_100": 3, "hnl15_200": 4}

def acorn_event(hits: pd.DataFrame, edges: pd.DataFrame, parts: pd.DataFrame, event_id: str) -> Data:
    """One event of derived/events/v1 (hnl.data.convert) -> acorn feature-store event (default naming)."""
    hits = hits.sort_values("row")
    n = len(hits)
    assert np.array_equal(hits.row.to_numpy(), np.arange(n))           # node index == npz row
    r, phi = hits.r.to_numpy(np.float32), hits.phi.to_numpy(np.float32)
    z = hits.z.to_numpy(np.float32)                                    # float64 copy of the stored float32 z
    pid = hits.pid.to_numpy(np.int64)
    te = np.stack([edges.src.to_numpy(np.int64), edges.dst.to_numpy(np.int64)])   # [2,T] rows, chain order
    tpid = pid[te[0]]
    p = parts.set_index("particle_id").loc[tpid]                       # one row per true edge
    g = Data(
        hit_id=torch.arange(n, dtype=torch.int64),
        x=torch.from_numpy(hits.x.to_numpy(np.float32)),             # -> hit_x (required by test_data)
        r=torch.from_numpy(r), phi=torch.from_numpy(phi), z=torch.from_numpy(z),
        eta=torch.from_numpy(np.arcsinh(z / r).astype(np.float32)),   # -> hit_eta (acorn plots only)
        track_edges=torch.from_numpy(te),
        particle_id=torch.from_numpy(tpid),                            # -> track_particle_id
        pt=torch.from_numpy(p.pt.to_numpy(np.float32)),                # -> track_particle_pt [GeV]
        eta_particle=torch.from_numpy(p.eta.to_numpy(np.float32)),     # -> track_particle_eta
        nhits=torch.from_numpy(p.n_hits.to_numpy(np.int64)),           # -> track_particle_nhits
    )
    g.hit_y = torch.from_numpy(hits.y.to_numpy(np.float32))            # never "y"
    g.hit_particle_id = torch.from_numpy(pid)                          # never "particle_id"
    g.event_id = event_id                                              # f"{SAMPLE_CODE[s]}{index:08d}"
    g.config = []
    return g

def write_event(g: Data, fs: str, split: str) -> None:
    os.makedirs(os.path.join(fs, split), exist_ok=True)                # save_pyg does not mkdir
    save_pyg(g, os.path.join(fs, split, f"event{g.event_id}-graph.pyg"))   # -> .pyg.gz
```

## 3. Per-stage configuration for the baseline

Keys every stage needs:
- `stage`, `model` (`acorn/core/core_utils.py:37-42`).
- `input_dir`, `stage_dir`.
- `data_split: [n_train, n_val, n_test]`. The EC / track-building `GraphDataset` needs ints, asserts it finds exactly
  n files (`acorn/utils/loading_utils.py:123-127`), and skips a split with 0 (`edge_classifier_stage.py:157`). The
  metric-learning `EventDataset` asserts that at least n exist and takes the first n; `null` means all
  (`graph_construction_stage.py:338-342`).
- For Lightning stages, in both train and infer YAMLs: `accelerator`, `devices`, `nodes`
  (`acorn/core/core_utils.py:144-171`, `acorn/core/infer_stage.py:104-111`).
- For training: `max_epochs`, and `log_wandb: false` (the default `true` builds a `WandbLogger` and then needs
  `project`).

### 3.1 Metric learning (template: `examples/Example_2/metric_learning_train.yaml`, the 3-feature TrackML config; Example_3 is the same with 12 cell features)

| Key | Baseline | Example | Note |
|---|---|---|---|
| `node_features` / `node_scales` | `[r, phi, z]` / `[1000, 3.14, 1000]` | Ex2:42-43 | keep the r and z scales equal (they set the `hit_R` orientation) |
| `emb_hidden`, `nb_layer`, `emb_dim`, `activation` | 1024, 4, 12, Tanh | Ex2:44-47 | 3,165,196 parameters; LayerNorm without affine is hard-coded (`metric_learning.py:48-54`) |
| `r_train`, `knn`, `knn_val` | 0.1, 100, 1000 | Ex3:48-50 (Ex2: knn 50) | defaults: `knn_val` 500; `knn` and `r_train` required |
| `randomisation`, `points_per_batch` | 1, 100000 | Ex3:46-47 | above N, so every hit is a query |
| `margin`, `lr`, `factor`, `patience`, `warmup`, `max_epochs` | 0.1, 0.01, 0.7, 10, 5, 100 | Ex2:55-60 | AdamW(amsgrad) + StepLR always; the `scheduler` key is ignored (`:105-126`) |
| `metric_to_monitor` / `metric_mode` | `f1` / `max` | Ex2:61-62 | default `val_loss`/`min`; see open question 3 |
| `undirected`, `train_undirected` | true, true | Ex2:40-41 | `undirected` defaults to false (`:389`) |
| `hard_cuts` | absent (train), `{}` (infer) | Ex2:36-37 has pt ≥ 1000 MeV | forbidden by brief §5f; a checkpoint value is inherited at infer (`acorn/core/infer_stage.py:93`) |
| `weighting` | see below | Ex2:19-32 | required (KeyError); every rule needs a `y` key (`acorn/stages/graph_construction/utils.py:23`) |
| `num_workers` | `[4, 4, 4]` | — | default 16 per loader (`metric_learning.py:62-96`) |
| `r_infer`, `knn_infer` (infer YAML) | from the val scan (§3.2) | Ex2 infer: 0.1 / 1000; Ex3: 0.1 / 100 | `r_infer` required; `knn_infer` defaults to 500 (`:480-483`) |

Weighting. Rules apply in order and later rules overwrite earlier ones. The default weight is 0 for true edges and 1
for fake edges (`acorn/utils/loading_utils.py:213-214`).
- **N (natural, the brief's "all tracks")**: `weighting: [{weight: 1.0, conditions: {y: true}}]`.
- **R (recipe, in GeV)**:
  ```yaml
  weighting:
    - {weight: 1.0, conditions: {y: false}}
    - {weight: 0.0, conditions: {y: true}}
    - {weight: 3.0, conditions: {y: true, pt: [1.0, .inf], nhits: [3, .inf]}}
  ```
- Never copy Example_2's `primary: True` or `pdgId: [not_in, [11, -11]]`. We have no `primary` flag, and that rule
  would zero the displaced daughters.
- The choice is open question 1.

Keys that silently change behaviour:
- `undirected: false`: about 1 % of true edges point inward in r²+z² (RESULTS.md: 98.8-99.2 % point outward). They are
  saved as `y = false`, `truth_map = -1`, and the GNN trains on those labels.
- `hard_cuts` inherited from a checkpoint removes hits from saved graphs (`graph_construction_stage.py:375-394`).
- Condition syntax (`acorn/utils/mapping_utils.py:22-51`):
  - Valid: a bool, `[lo, hi]` (inclusive), `[in|not_in|within|not_within, ...]`.
  - Other scalars raise ValueError, even though the example comments say "Single value: track[key] == value".
  - `[is, v]` / `[is_not, v]` are broken.
  - `hit_*` conditions crash the loss weighting. Use only `edge_*` / `track_*` conditions.
- Thresholds written in MeV (1000.) select nothing in GeV data. With rule R, the loss is then NaN, because there are no
  positive edges.
- `node_scales` with unequal r and z scales changes the stored edge orientation (`hit_R` is computed in scaled units).
- FRNN is used only if CUDA is visible when acorn is imported (`acorn/stages/graph_construction/models/utils.py:24-36`).
  Otherwise the PyG `radius` fallback is used, which differs when the `knn` cap binds. FRNN keeps the first k
  neighbours it finds, not the nearest k, and the query uses one of the k slots (brief §4.5,
  `tests/acorn_pitfalls/test_p03_frnn_first_k.py`).
- Validation builds its graphs with `r_train`, not `r_infer`. Validation metrics are averaged per event.
- `log_metrics` divides by the number of signal edges and by eff+pur (`metric_learning.py:413-439`). A validation event
  with zero signal edges raises ZeroDivisionError. It is not called during the sanity check.

### 3.2 Radius graph (metric-learning inference)

- `acorn infer ml_infer.yaml -c <ckpt>` writes every split with `data_split > 0` (all three dataloaders,
  `metric_learning.py:98-103`). It overwrites existing files and has no skip option.
- Besides `stage`, `model`, `input_dir`, `stage_dir`, `accelerator`, `devices` and `nodes`, put `hard_cuts: {}`,
  `undirected: true`, `r_infer`, `knn_infer`, `data_split` and `num_workers` in the infer YAML. Every other key comes
  from the checkpoint.
- (r, knn) scan on val (brief §5f: ≥ 99 % true-edge efficiency):
  1. Load the checkpoint and merge the infer overrides into `_hparams`, prefixed with
     `add_variable_name_prefix_in_config` as `acorn/core/infer_stage.py:93-99` does.
  2. Call `setup("predict")`, which scales the inputs.
  3. Per val event: `apply_embedding` once, then for each (r, k) call
     `build_edges(emb, emb, r_max=r, k_max=k, backend="FRNN")` and `get_truth(batch, edges, True)`. These are the
     same calls as `shared_evaluation` (`metric_learning.py:369-390`).
  4. Pool the unordered efficiency, the graph size (columns after `get_truth` = unique pairs), and the fraction of
     hits where the k cap binds (keep it < 5 %, brief §4.5). For the cap, call `build_edges(..., return_indices=True)`
     and count rows of `idxs` with all k slots filled.
  5. Run on the GPU so that FRNN is used.

  Calling `shared_evaluation` or `predict_step` without a Trainer raises RuntimeError (`hasattr(self, "trainer")`
  at `:398` does not catch it). Either attach a Trainer or call the pieces as above.

### 3.3 Filter (optional; template `examples/Example_3/filter_train.yaml`)

- Settings: `hidden: 1024`, `nb_layer: 4`, `layernorm: true`, `batchnorm: true`, `hidden_activation: ReLU`,
  `edge_cut: 0.5`, `ratio: 5` (required), `warmup: 10`, `scheduler: StepLR`, `lr: 0.001`, `factor: 0.8`,
  `patience: 10`, `max_epochs: 50`, `metric_to_monitor: auc`, `metric_mode: max` (Ex3:37-56). Use
  `node_features: [r, phi, z]` and `node_scales: [1000, 3.14, 1000]`. That is 700,161 parameters (Example_3 with
  cells: 718,593).
- The filter removes nothing. It saves scores on all edges. The GNN prunes only if `input_cut` is set (keeps
  `edge_scores >= input_cut`, `edge_classifier_stage.py:726-738`, `:827-840`). Without it, the GNN sees every edge.
  Example_3's README (line 64) says to "comment out" `input_cut`, which is the wrong way round. Choose `input_cut` on
  val (Example_3 evaluates the filter at 0.2).
- With `track_running_stats` false (the default), BatchNorm uses batch statistics in evaluation too. Scores therefore
  depend on which edges are evaluated together (training subsamples, inference uses the full graph).
- Training subsamples negatives with unseeded `torch.randint` (`filter.py:28-86`).

### 3.4 InteractionGNN (template `examples/Example_3/gnn_train.yaml`; this recipe has 263,937 parameters, the brief's "264k")

| Key | Baseline | Example | Note |
|---|---|---|---|
| `hidden`, `n_graph_iters` | 128, 8 | Ex3:37-38 | node and edge networks are shared across iterations by default (`interaction_gnn.py:51-60`) |
| `nb_node_layer`, `nb_edge_layer`, `layernorm`, `batchnorm` | 2, 2, true, false | Ex3:39-42 | |
| `aggregation` | `sum` | Ex3:43 (Ex2: `[sum, mean, max, std]`, 362,241 parameters) | |
| `hidden_activation`, `output_activation` | SiLU, Tanh | Ex3:44-45 | |
| `node_features`, `node_scales` | `[r, phi, z]`, `[1000, 3.14, 1000]` | Ex3:35-36 | `edge_features` are ignored by `InteractionGNN` |
| `edge_cut` | 0.5 | Ex3:32 | logged metrics only (`sigmoid > edge_cut`); required |
| `undirected`, `undirected_message_passing` | true, true | Ex3:33-34 | the second is read at forward time, so an infer YAML can change it |
| `warmup`, `scheduler`, `lr`, `factor`, `patience`, `max_epochs` | 5, StepLR, 0.001, 0.9, 10, 100 | Ex3:48-54 | `warmup` must exist (KeyError at `edge_classifier_stage.py:481`); `ReduceLROnPlateau` raises TypeError under torch 2.11 (`verbose=`, `acorn/utils/ml_utils.py:111-117`) |
| `metric_to_monitor` | absent → `val_loss` / `min` | Ex3 sets none | |
| `weighting` | N: `[{weight: 1.0, conditions: {y: true}}]` (equals omitting the key); R: `[{weight: 1.0, conditions: {y: true, pt: [1.0, .inf], nhits: [3, .inf]}}]` | Ex3:19-24 | with the key absent, every edge has weight 1 (`edge_classifier_stage.py:716-722`) |
| `hard_cuts` | absent (train), `{}` (infer) | Ex3:28-29 | edge-level cut, inherited at infer |
| `num_workers` | `[4, 4, 4]` | — | default `[1, 1, 1]` |
| `input_cut` | only after a filter | Ex3:12 (commented) | |

Keys that silently change behaviour:
- `undirected`: it is used only when saving, to recompute `truth_map` (`edge_classifier_stage.py:539-546`). `y` is
  never recomputed, so the training labels are the metric-learning labels.
- `weighting` present vs absent: when it is present, true edges default to weight 0.
- `fit` sets `torch.set_float32_matmul_precision("medium")` (TF32); predict and test set `"highest"`
  (`edge_classifier_stage.py:120-122`).
- `max_training_graph_size` skips events.
- `dataset_class: HeteroGraphDataset` turns off the averaging of the two directed logits (`interaction_gnn.py:188-192`).
- The `ratio` key in Example_3's GNN config is unused.

### 3.5 Connected components

- `score_cut` (default 0.5, strict `>`, `connected_components.py:86`, `:163`). Example_3 uses 0.8. The brief and the
  pre-registration choose s* on val over the grid 0.05-0.95.
- `max_workers` (default 1; any other value runs a process pool).
- `save_graph`, `save_tracks` (default true, true), `on_true_graph_edges` (the ceiling on true edges, patch 0002).
- Run the threshold scan in memory. It writes no files:
  ```python
  from acorn.stages.track_building import ConnectedComponents
  from acorn.utils.loading_utils import load_pyg, add_variable_name_prefix_in_pyg
  cc = ConnectedComponents({})                                   # synth/s8_cc_inmem.py
  g = add_variable_name_prefix_in_pyg(load_pyg(path_without_gz, weights_only=False))
  labels = cc._build_event(g, score_cut=s).hit_track_labels      # needs hit_x for num_nodes
  ```
- acorn's own evaluation (`acorn eval cc_eval.yaml -d valset`), for comparison only:
  - Settings: `matching_style: two_way`, `matching_fraction: 0.5`, `target_tracks: {pt: [1.0, .inf], eta_particle:
    [-4.0, 4.0], n_true_hits: [5, .inf]}`, `min_track_length` (5 = acorn default, 1 = no length gate),
    `plots: {tracking_efficiency: {variables: {pt: {x_lim: [1, 50]}}}}`.
  - It reads `stage_dir/<split>`, so it needs `save_graph: true`.
  - It raises ZeroDivisionError if nothing is reconstructed (`track_building_stage.py:226-228`).

## 4. Running

### 4.1 CLI and programmatic use

- CLI (`acorn` = `acorn/core/entrypoint_stage.py:cli`):
  - `acorn train CFG [-c CKPT] [-r DIR] [--load_only_model_parameters True]`
  - `acorn infer CFG [-c CKPT]`
  - `acorn eval CFG [-c CKPT] [-d trainset|valset|testset]` (default `valset`)
- Every entry point does the same setup:
  1. Load the YAML.
  2. Prefix the config unless `variable_with_prefix` is set (`acorn/core/train_stage.py:88-89`).
  3. Resolve the class with `getattr(getattr(acorn.stages, stage), model)` (`acorn/core/core_utils.py:37-42`).
- `train` builds the trainer with `get_trainer` (`acorn/core/core_utils.py:91-171`):
  - `DDPStrategy(find_unused_parameters=False, static_graph=True)`, also on one device;
  - a `ModelCheckpoint` (best + last);
  - `ThroughputCallback`, `LearningRateMonitor`;
  - `precision` from the config (default 32).
- `infer` merges `{**checkpoint hparams, **yaml}` (`acorn/core/infer_stage.py:93`), calls `setup("predict")` and
  `Trainer.predict`.
- `eval` merges the hparams the same way but never loads the weights. `stage.evaluate(config)` builds an untrained
  `cls(config)` and reads the saved outputs from `stage_dir` (`acorn/core/eval_stage.py:76-97`). To evaluate with
  weights, call `EdgeClassifierStage.evaluate(config, checkpoint=path)` (`edge_classifier_stage.py:560-582`). eval
  still exits if it finds no checkpoint (`acorn/core/eval_stage.py:83-85`).
- Our own subclasses work with the CLI only if registered:
  `setattr(acorn.stages.edge_classifier, "HNLGNN", HNLGNN)` before calling `train_stage.train(...)`.
- Recommended runner (`src/hnl/train/`): call acorn's entry points from Python, so that seeding and run metadata
  (git hash, config hash, env lock, GPU type, job id: brief §9) are added without changing acorn's code path:
  ```python
  from pytorch_lightning import seed_everything
  seed_everything(seed, workers=True)          # before the module is built inside train()
  from acorn.core.train_stage import train; train("ml_train.yaml")           # == acorn train
  from acorn.core.infer_stage import infer; infer("ml_infer.yaml", checkpoint=ckpt)   # == acorn infer -c
  ```
  Run `infer` from a scratch working directory, because of `./lightning_logs`. Alternatively use
  `Trainer(accelerator="gpu", devices=1, logger=False).predict(model)` after
  `model = Cls.load_from_checkpoint(ckpt)` and `model._hparams = {**model._hparams, **prefixed_overrides}`
  (checked on CPU: `synth/s7_predict_trainer.py`).
- Keep acorn's `get_trainer`, or the same `DDPStrategy`, for training. A default-strategy `Trainer` does not shuffle
  the training events (§4.5).

### 4.2 Checkpoints

- `find_latest_checkpoint(stage_dir, ["best*.ckpt", "*.ckpt"])` searches recursively and returns the newest file by
  `ctime` (`acorn/core/core_utils.py:78-88`). That is the newest file, not the best metric. infer and eval use it
  whenever `-c` is missing (`acorn/core/infer_stage.py:78-84`, `acorn/core/eval_stage.py:76-82`).
- Training resumes from the newest `*.ckpt` in `default_root_dir` (or `-r DIR`) and ignores `-c` when that directory
  holds checkpoints (`acorn/core/core_utils.py:182-198`).
- So: always pass `-c` with an explicit path, use one fresh `stage_dir` per (arm, seed, stage), and run one training
  per Slurm job.
- Choose by monitored metric: the `best-*` file. The brief's pre-registration (§5) says "checkpoint (best val loss)";
  see open question 3.
- `--load_only_model_parameters True` loads weights only (`acorn/core/train_stage.py:155-160`). A plain `-c` restores
  the full training state (`:162`).
- Size estimate, 16 B/parameter with AdamW-amsgrad state: metric learning ≈ 51 MB, IGNN ≈ 4 MB, filter ≈ 11 MB per
  checkpoint. Each run keeps best + last.

### 4.3 Inference outputs

- Metric learning: `stage_dir/<split>/event<ID>.pyg.gz` for every split in `data_split`.
- EC (filter, GNN): the same path. `skip_existing: true` resumes an interrupted run (`edge_classifier_stage.py:519-527`).
- CC: `<split>/<prefix>event<ID>.pyg.gz` + `<split>_tracks/<prefix>event<ID>.txt` (hit ids per track, sorted by
  r²+z², label −1 dropped; `track_building_stage.py:317-359`, `connected_components.py:129-142`).
- Infer only the splits the next step needs. The merged checkpoint `data_split` otherwise re-infers train, val and
  test.

### 4.4 Resume and pre-timeout signals

The brief requires a checkpoint every ~30 min and on a pre-timeout signal (§9). acorn's `ModelCheckpoint` saves only
when the monitored metric is logged (each validation), plus `last`, so measure the epoch time in the smoke test. Two
patterns:

- **A (acorn-native; `scontrol requeue` is allowed, DECISIONS.md 2026-10-02).**
  - Setup: `#SBATCH --requeue`, `#SBATCH --signal=USR1@300`, launch with `srun` (the job name must not be `bash` or
    `interactive`), and run from a per-run working directory.
  - On SIGUSR1, Lightning saves `default_root_dir/hpc_ckpt_<n>.ckpt` and runs `scontrol requeue`
    (`site-packages/pytorch_lightning/trainer/connectors/signal_connector.py:45-106`).
  - The requeued job keeps its id, so acorn resumes from `./$SLURM_JOB_ID/*.ckpt` (`acorn/core/core_utils.py:45-54`,
    `:195-198`). If `default_root_dir` is None, Lightning resumes "hpc" from the working directory
    (`site-packages/pytorch_lightning/trainer/connectors/checkpoint_connector.py:121-122`).
  - SLURM detection requires `SLURM_NTASKS` and a job name that is not `bash` / `interactive`
    (`site-packages/lightning_fabric/plugins/environments/slurm.py:103-113`, `:227-232`).
  - `hpc_ckpt` files are never deleted. Example: `examples/Example_3/PM_submit_job_requeue.sh:12-14`.
- **B (brief §9 default: follow-up job).**
  - The runner installs its own USR1 handler that only sets a flag. A callback then calls
    `trainer.save_checkpoint(...)` at the next batch end and sets `trainer.should_stop = True`.
  - Disable Lightning's requeue, either with `SLURMEnvironment(auto_requeue=False)` or by registering the handler
    first (`site-packages/pytorch_lightning/trainer/connectors/signal_connector.py:62`).
  - The next job (`--dependency=afterany:<id>`) runs `train(cfg, checkpoint=<that ckpt>)`.

Unverified: whether subMIT batch jobs set `SLURM_JOB_QOS`, which acorn needs for `./$SLURM_JOB_ID`, and how Lightning
2.6 resumes in the middle of an epoch.

### 4.5 Seeds and determinism

- acorn never seeds training. The only seeds are `TrackBuildingStage.setup("test")` → `torch.manual_seed(0)`
  (`track_building_stage.py:84`) and the readers. acorn's DataLoaders use `batch_size=1` and no shuffle.
- Under `DDPStrategy` (always used by `get_trainer`), Lightning injects
  `DistributedSampler(shuffle=True, seed=int(PL_GLOBAL_SEED or 0))` for training only and reshuffles every epoch
  (`site-packages/pytorch_lightning/trainer/connectors/data_connector.py:169-174`, `:230-243`, `:478-494`;
  `site-packages/pytorch_lightning/trainer/connectors/accelerator_connector.py:554-568`). Checked in
  `synth/s1_sampler.py`: epoch 0 `[4,0,7,3,2,5,1,6]`, epoch 1 `[5,4,2,6,7,3,1,0]`. Validation is unshuffled. A
  default-strategy Trainer gives a sequential order.
- `seed_everything(seed, workers=True)` sets `PL_GLOBAL_SEED` (`site-packages/lightning_fabric/utilities/seed.py:59`).
  Called before the module is built, it fixes:
  - the weight initialisation;
  - the metric-learning query permutation and random pairs;
  - the Filter's subsampling;
  - the training order.
- FRNN and GPU scatter are not bitwise reproducible, so expect seed-to-seed spread only.

## 5. Truth and metrics: acorn vs ours

- **True edges.** Ours are the npz `layerless_true_edges` (definitions §2), which we store as `track_edges`.
  - acorn's `edge_y` comes from `graph_intersection(edge_index, track_edges)` in `get_truth`
    (`metric_learning.py:273-287`; `acorn/stages/graph_construction/models/utils.py:110-183`).
  - With `undirected: true`, `edge_y` = "this unordered pair is a chain edge", which is exactly our label. The
    module-wise grouping of the brief's pitfall 9 comes from acorn's readers and does not apply. Pairs that skip a
    hit are negatives in both.
  - `track_to_edge_map >= 0` marks true edges present in the graph. Its pooled mean over val is "acorn's own `edge_y`
    efficiency".
- **acorn metric-learning metrics** (`metric_learning.py:413-439`): `total_eff`, `total_pur`, `signal_eff`,
  `signal_pur`, `f1`. They are per-event means on the `r_train` / `knn_val` graph, not our pooled efficiency on the
  `r_infer` graph.
- **acorn GNN metrics** (`edge_classifier_stage.py:402-477`): `target_eff`, `target_pur`, `total_pur`, `pur`, `auc`,
  `total_auc`, pooled per epoch at `sigmoid > edge_cut`, with targets = `edge_y & (edge_weights > 0)`.
- **acorn track metrics** (`track_building_stage.py:172-295`, `acorn/stages/track_building/utils.py:32-231`):
  - The default `matching_style` is `ATLAS` (purity only).
  - `two_way` at 0.5 becomes 0.5+1e-6, which is a strict double majority (`:194-196`, `:226-229`) and equals ours.
  - Hits with label −1 form a pseudo-track `-1` that can match particles and enters the fake rate (`:49-50`).
  - Hits on no true edge are pooled as particle 0. That happens to our single-hit particles, since our samples have
    no noise hits.
  - `n_true_hits` is counted from `track_edges` endpoints.
  - `min_track_length` (default 5) gates both the efficiency numerator and the fake-rate denominator
    (`track_building_stage.py:205-210`).
  - Duplicates = extra matched rows / reconstructed particles (`track_building_stage.py:222-228`).
  - `target_tracks` bounds are inclusive, and a scalar means `>=` (`acorn/stages/track_building/utils.py:88-109`).
  - The plots merge particles across events by `particle_id` (`track_building_stage.py:251-262`).
  - Track efficiency by category, |d0|, r_v, first-hit radius or pixel hits: not found in acorn.
- **Ours** (definitions §5, `src/hnl/eval/matching.py`): call `match_event(pid, track)` per event, with
  `pid = hit_particle_id` and `track = hit_track_labels`; −1 means no track. Targets: pT > 1 GeV, abs(eta) < 4 (both
  strict) and ≥ 5 hits (`is_target` in the particles table). Fake rate over tracks with ≥ 5 hits. Duplicates by single
  majority without a length cut.
- **We compute ourselves:**
  - Edge efficiency and purity in unique unordered pairs on the inference graph, also per layer pair
    (`src/hnl/graphs/ceiling.py:evaluate_graph`).
  - The graph-stage ceiling on acorn graphs (CC on the true edges in the graph).
  - The edge false-positive rate at s* (from `scores`, `y`).
  - The in-memory CC scan for s*.
  - DM efficiency per category and bin (truth fits in `derived/truth_fits`), fake rate, duplicates, pair efficiency
    and merge rate.
  - The fixed-FR* working points, the bootstrap and the 3-seed means (`docs/preregistration.md`).
  - acorn-native numbers only as a labelled comparison.

## 6. File count and I/O

### 6.1 What acorn writes, and how big it is

- File count:
  - Feature store: 1 file per event.
  - Metric-learning inference, filter inference, GNN inference: 1 file per event per inferred split.
  - CC: 2 files per event (1 with `save_tracks: false`, 0 in memory).
  - Training: 2 checkpoints plus a few log files.
  - `subdir_size` only moves files into `<int(id)//n>/` subdirectories. It does not reduce the count, and
    `EventDataset` cannot read sharded input.
- Measured sizes (gzip level 1, real events, `synth/s4_sizes.py`):

| Object | Size |
|---|---|
| Feature store, minimal fields above | 20-83 KiB for N = 756-3423 (about 37 KiB at N = 1.47k); +~45 % with `hit_y`, `hit_hid`, `hit_geoid`, `hit_particle_id` |
| Metric-learning output | feature store + 13-27 KiB fixed (`hit_R`, `truth_map`, `batch`) + 1.1-1.7 B per edge on sorted geometric graphs, lower at larger E (E = 2k-240k). The data map measured ~8 B per edge on random edges. |
| GNN output | metric-learning output + 3.6 B per edge (`scores`; `edge_weights` compress away) |

  The paper's graph keeps 99.9 % of true edges at 5.7 % purity, so E ≈ 20-35k per event. At E = 24k an HNL event
  (N = 1470) is 84 KiB after metric learning and 170 KiB after the GNN (measured on the proxy graph).
- Budget: `hnl_lrt` holds 35,240 files (`ceph.dir.rfiles` while this page was written; CephFS counters lag) against
  the 80k trigger.
  - One run (arm, seed) infers about 27k metric-learning events (arm (ii): train 15,999, all val 2,999, test 3,000,
    external 4,999) and about 11k GNN events. With CC that is about 60k per-event files.
  - The byte budget is about 172 GB free on /ceph. One run's packed outputs are about 4 GB at E ≈ 25k, so all
    3 × 3 runs come to ≈ 40 GB.

### 6.2 Options

1. **Node-local staging (recommended for the baseline; acorn's I/O code is unchanged).**
   - Inside each job, put per-event files under `/tmp/$USER/$SLURM_JOB_ID` (1.1 TB free on `/tmp`, measured on
     submit20-1). `env.sh` points `$TMPDIR` to /work, so use the explicit path, and delete it in a `trap ... EXIT`.
   - Rebuild the feature store from the parquet shards in the job, and unpack the previous stage's archive.
   - Training jobs: `stage_dir` on `/ceph/.../hnl_lrt/runs/<run_id>/<stage>` (checkpoints and logs only).
   - Inference jobs: `-c` from /ceph, `stage_dir` local, then pack each `<split>/` into one `tar` on /ceph
     (`tar -C $LOCAL/<stage> -cf $RUN/<stage>/<split>.tar <split>`; the contents are already gzip). Unpack it with
     `tar -xf` into the next job's local `input_dir`.
   - The per-event files of a run then never reach /ceph, which then holds about 20 files per run.
2. **Sharded containers through subclasses (zero per-event files).** acorn is not edited. Hook points:
   - Metric-learning input: `GraphConstructionStage.dataset_class` (`graph_construction_stage.py:68`), built as
     `cls(input_dir, data_name, num_events, use_csv=..., hparams=...)` (`:99-109`).
     - Subclass `EventDataset` and override `find_evt_ids` (`:319-373`, called in `__init__` at `:267`) and `get`
       (`:272-301`).
     - Keep `preprocess_graph` (`:303-312`) and `unscale_features` (`:415-432`, used by `predict_step` at
       `metric_learning.py:485-487`).
     - Set `self.dataset_class` after `super().__init__` in a `MetricLearning` subclass.
   - Metric-learning output: override `MetricLearning.save_graph(event, datatype)` (`metric_learning.py:491-504`).
     Buffer `event.cpu()` and write one shard per split in `on_predict_end`.
   - EC input: `self.dataset_resolver.register(MyGraphDataset)` (`edge_classifier_stage.py:86-92`) in the subclass
     `__init__`, and `dataset_class: MyGraphDataset` in the YAML. `make(...)` passes `input_dir`, `data_name`,
     `num_events`, `stage`, `hparams`, `preprocess`, `transform` (`:158-167`; checked in `synth/s6_resolver.py`).
     Override `__init__`'s file listing (`:642-645`) and `get` (`:650-670`), and keep `preprocess_event`
     (`:672-685`).
   - EC output: override `save_edge_scores(event, dataset)` (`edge_classifier_stage.py:535-558`).
   - CC input: `TrackBuildingStage.dataset_class` (`track_building_stage.py:58`), built as
     `cls(input_dir, data_name, num, stage, hparams)` (`:96-98`).
   - CC output: override `save_graph` / `save_tracks` (`:317-359`), or use in-memory `_build_event`.
   - CLI resolution: `setattr(acorn.stages.<stage>, Name, cls)`.
3. **Per-event files on /ceph, tarred or deleted right after evaluation.** This is allowed as long as the final state
   fits the budget, but one run alone (~60k files) nearly fills the remaining ~45k headroom. Do not use it.

Do not use `ACORN_SAVE_PYG_UNCOMPRESSED` (it about doubles the bytes). Do not put per-event files on `/work` either:
it is a shared filer with a 100 GB quota, and the brief keeps big data off it.

## 7. Open questions and risks

1. **Weighting (user decision).** The options are natural (N; the brief's "all tracks" and the paper's "all tracks
   used for training") or the acorn recipe (R; non-target true edges masked, targets pT ≥ 1 GeV and nhits ≥ 3, written
   in GeV). Use the same choice for metric learning and the GNN.
2. **Filter.** Whether to use it at all: the graphs are small (~1.4k hits, E ≈ 20-35k expected). If it is used:
   whether to keep BatchNorm (batch statistics at evaluation), and `input_cut` chosen on val.
3. **Checkpoint choice for metric learning.** The recipe monitors `f1`/`max`. The pre-registration draft says "best val
   loss". The GNN default is already `val_loss`/`min`.
4. Whether each GNN seed retrains metric learning (3 × 3 metric-learning trainings), or the seeds share one
   metric-learning model per arm. This affects GPU-hours and storage.
5. Arm (iii) composition ("ttbar + HNL 50/50, equal event counts"), and which val events each arm monitors.
   `data_split` cannot sample at random.
6. The trained graph size E, GPU step time and memory are unknown. Run the brief §9 smoke test first, and check the
   epoch length against the ~30-min checkpoint rule.
7. Slurm: whether `SLURM_JOB_QOS` is set (it decides `./$SLURM_JOB_ID`); Lightning mid-epoch resume; and
   `DDPStrategy` on one A30 under `srun`. None of these was tested.
8. The node-local `/tmp` usage policy on the GPU nodes is undocumented. Clean up at exit.
9. Baseline properties kept for control of variables (no big impact): raw phi with the ±π seam (0.5-0.6 % of true
   edges straddle it, as measured in the metric-learning map), the FRNN first-k cap, and TF32 in the EC fit. Report the
   cap-binding fraction from the (r, knn) scan.
10. `hnl.graphs.ceiling.cc_labels` gives isolated hits their own component, while acorn's CC gives them −1. Under our
    single-majority duplicate definition, a singleton counts as a duplicate (`synth/s3_match.py`: duplicate rate 1.0
    vs 0.0 for the same event). Align them before comparing the ceiling with the baseline.
11. Not consulted: the main project's exact acorn configs. "Recipe = Example_3" is inferred from the brief's parameter
    counts (3.17M with 12 inputs, 0.72M, 264k).

## Appendix A. Contradictions between the maps, resolved from source

| Topic | Maps | Source |
|---|---|---|
| Training event order | GNN and run maps: sorted every epoch. Metric-learning map: shuffled. | Shuffled with acorn's `DDPStrategy`; sequential only with a non-DDP Trainer (§4.5; `synth/s1_sampler.py`). |
| `hit_R` units | r²+z² vs scaled | Computed from the scaled `hit_r`, `hit_z` (`graph_construction_stage.py:303-312`, `:396-413`; `acorn/stages/track_building/utils.py:329-337`) and never unscaled. Saved value 0.0093 for r = 31.6 mm (`synth/s5_chain.py`). |
| Filter → GNN | Run map: "drop `input_cut`"; README:64 "comment out" | `input_cut` must be set for the filter to prune (`edge_classifier_stage.py:726-738`). |
| Example_3 metric-learning features | data map: (r, phi, z) | Example_3 metric learning uses 12 features (`examples/Example_3/metric_learning_train.yaml:40-41`). The 3-feature template is Example_2 (`examples/Example_2/metric_learning_train.yaml:42-43`). Example_3's GNN uses (r, phi, z). |
| Feature-store size | 18-56 KiB / ~64 kB / ~68 kB | 20-83 KiB, depending on N and the stored fields (`synth/s4_sizes.py`). |
| Bytes per edge | 2.4-4.7 / 4.5-9.5 / ~8 B | 1.1-1.7 B on sorted geometric graphs; scores add 3.6 B (`synth/s4_sizes.py`). The ~8 B figure is for random edges, the worst case. |
| "Single value" conditions | example comments: `==` | Only bools are compared with `==`; other scalars raise in weighting and `hard_cuts`; in `target_tracks` a scalar means `>=` (`acorn/utils/mapping_utils.py:42-51`; `acorn/stages/track_building/utils.py:106-107`). |
| Naming scheme | Metric-learning map proposed `variable_with_prefix: true` | Both work. The default scheme is used by Example_2/3, so it is chosen here. |
