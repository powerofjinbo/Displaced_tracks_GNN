# acorn pitfalls at the pinned commit (brief §4.5)

acorn `99d153418981cba9018e70b476fc7ea0f17bacd4` + `patches/acorn/0001-0007`. Verified 2026-10-03 by the
`acorn-pitfalls` workflow: one agent per group read the pinned and patched source, wrote toy tests that drive the
real acorn code, and ran them on the patched install and on the unpatched pin (`ACORN_PIN_SRC`); an adversarial
reviewer per group then re-read the code, re-ran the tests and tried to refute each claim. Every claim held; the
reviewers' corrections are folded in below. Raw results: `/work/submit/jinboz1/hnl/tmp/acorn_pitfalls_result.json`.

Tests: `tests/acorn_pitfalls/` (151 CPU tests pass on the patched install; 46 fail on the unpatched pin, i.e. they
reproduce the bugs the patches fix; 18 GPU-marked tests run through `slurm/gpu_tests.sbatch`).

| id | pitfall | at pin | patch | after the patches |
|---|---|---|---|---|
| p01 | weighting condition of unmatched length silently ignored (`map_value_to_edges` all-True) | yes | 0004 | raises; residual silent cases below |
| p02 | `within` / `not_within` broken in `graph_construction/utils.py` | yes | 0003 | fixed (now identical to `mapping_utils`) |
| p03 | FRNN keeps the first K neighbours within r (grid order), not the K nearest | yes | none | unchanged: choose (r, K) so the cap never binds |
| p04 | InteractionGNN doubles edges internally (`undirected_message_passing`, default True) | yes | none | characterised; count unique pairs |
| p05 | CC `on_true_graph_edges` read `graph.y` (None) -> one giant component | yes | 0002 | fixed; we compute ceilings ourselves |
| p06 | `load_reconstruction_df` used node indices as hit ids | yes | 0001 | fixed; converter sets hit_id = row |
| p07 | a feature named `y` becomes `edge_y` (hit y coordinate lost) | yes | none | characterised; name it `hit_y` |
| p08 | inference picks the newest `best*.ckpt` and inherits training hard_cuts etc.; acorn eval never loads weights | yes | none | always pass checkpoint and every key; own evaluation |
| p09 | acorn truth is module-wise consecutive (skip pairs negative; directed unless `undirected: true`) | yes | none | use npz chain truth, `undirected: true` |
| p10 | nested event ids in file names | yes | 0005 | fixed except DoubleMetricLearning / HierarchicalGNN |
| p11 | readers split randomly / by listing order | yes | 0006 | TrackMLReader only; we write splits ourselves |
| p12 | Filter catches OOM and retries | yes | 0007 | fixed chunks; new crash with batchnorm + 1-row chunk |

## Rules for our code (all stages)

Converter (npz -> acorn PyG events):
1. Prefixed names only, typed by length: per hit (N) `hit_id` (= npz row index, `torch.arange(N)`), `hit_x`, `hit_y`,
   `hit_z`, `hit_r`, `hit_phi`, `hit_eta`, `hit_module_id`, `hit_volume_id`, `hit_layer_id`, `hit_particle_id`,
   `hit_hid` (npz hid), optionally `hit_R` = r^2 + z^2; per true edge (T) `track_edges` (= npz layerless_true_edges,
   oriented outward in r^2 + z^2, ties by index) and per-edge copies of particle features `track_particle_id`,
   `track_particle_pt`, `track_particle_nhits`, `track_particle_pdgId`, `track_particle_prompt`, ...; keep the
   original chain under a neutral key (e.g. `truth_chain_edges`). Never use the bare names y, x, pos, pt, radius, eta,
   weight, labels, scores or truth_map (p07). All tensors are torch, never numpy (p06). Flags are torch.bool.
2. `event_id` is a flat 9-digit string (`f"{index:09d}"`), files `event{id}-graph.pyg(.gz)` (p10).
3. trainset / valset / testset are written from `splits/*.txt`; never from an acorn reader (p11).
4. No node-level hard cuts in the baselines (p06).

Configs:
5. `undirected: true` in every metric-learning train AND infer config and in the GNN config (p09, p04); equal node
   scales for r and z or a precomputed `hit_R` (orientation by `rearrange_by_distance`).
6. `undirected_message_passing: true` written explicitly in GNN train and infer configs (the infer config overrides
   the checkpoint silently) (p04). The baseline InteractionGNN ignores `edge_features`: do not configure them.
7. Weighting: first weight 0 for all true edges, then positive rules for the wanted edges only, no later weight <= 0
   overrides (build_signal_edges ORs positive rules while handle_weighting lets the last rule win) (p01). Condition
   forms: bool, `[lo, hi]`, `['in', [...]]`, `['not_in', [...]]`, `['within'/'not_within', [lo, hi]]` on float
   features only; never `is` / `is_not` (they return Python scalars); NaN-free condition features (p01, p02).
8. Metric learning: set `r_train`, `knn`, `knn_val`, `knn_infer`, `r_infer` explicitly; `test_step` hard-codes K=1000
   (p03). Assert `FRNN_AVAILABLE` at the start of every graph-building job; never build acorn graphs on CPU.
9. Filter: `eval_chunk_size` explicit; `batchnorm: false` (or `track_running_stats: true`); no GNNFilter (p12).
10. Do not use DoubleMetricLearning, HierarchicalGNN, the *WithPyG GNN variants (not constructible with PyG 2.7.0)
    or HeteroGraphDataset (p04, p10).

Running:
11. Never let acorn choose a checkpoint: pass the absolute checkpoint path and record its sha256 and hyper_parameters.
    Every inference config sets stage_dir (fresh, no *.ckpt inside), input_dir, data_split (exact counts),
    hard_cuts ({}), skip_existing false, input_cut, undirected, devices; assert architecture keys equal the
    checkpoint's (p08). Assert the loaded ids equal the split list (EventDataset can silently return fewer) (p11).
12. All metrics are computed in `src/hnl` from saved scores against the npz chain truth (acorn eval does not load
    weights; its edge_y is module-wise truth): DM matching, efficiencies, perfect-classifier ceilings (p05, p08, p09).
    acorn's edge_y efficiency is reported separately for comparison.

FRNN (p03):
13. Choose (r, K) jointly on val: for each r, exact per-node counts n_i of other hits within r (cdist per event);
    the cap can bind only where n_i >= K. Prefer K > max n_i over all val events (no binding at all; then FRNN and
    torch_cluster agree on GPU). Report the binding fraction with every graph. Graph-ceiling studies use exact radius
    graphs (cdist / topk), not FRNN.

Skip-layer truth (§6A), later:
14. Define skips by chain rank, not layer difference (27.6 % of true edges are same-layer overlaps). Augment
    `track_edges` and every `track_*` tensor before metric learning; never overwrite `edge_y` alone (p09).

## Details per pitfall

- **p01** (`graph_construction/utils.py` 172-175 at the pin): an unmatched condition length returned an all-True
  mask; 0004 raises `ValueError`. Live route: `build_signal_edges -> get_weight_mask -> map_value_to_edges`, from
  `append_signal_edges` and `log_metrics` of MetricLearning and its subclasses (Transformer, GravNet). Silent cases
  that remain: a length that coincides with E, E/2, T, T/2 or N; `is`/`is_not` scalars; int64 edge_y; length-1
  broadcast; and override-style configs, where signal pairs and the signal_eff / f1 denominator include masked true
  edges (toy: 8 vs 3 edges, signal_eff 0.375 vs 1.0; f1 is the example configs' monitored metric).
- **p02** (213-218 at the pin): `"within" <= tensor` raised TypeError; 0003 makes the code identical to
  `mapping_utils.py` 34-39. Only positive-weight rules with edge_y true reached it. acorn intervals are closed: our
  half-open bins are implemented in our own code.
- **p03**: `MinK::add` (FRNN `mink.cuh` 55-61) keeps the first K candidates with d^2 <= r^2 in increasing linear
  cell order (x-major over the first 3 coordinates; atomicAdd order inside a cell); the V1 kernel (D > 8 or K > 32,
  our emb_dim 12) returns them unsorted; cell size max(r/2, bbox/64). Same capped search in DoubleMetricLearning
  (knn_val 500, test K 1000, knn_infer 500), HGNN (k_max 50) and GravNet (torch_cluster). torch_cluster keeps the
  first max_num_neighbors in index order (GPU) or KD-tree order (CPU). P(n_i >= K) is a conservative bound on the
  fraction of nodes that lose a real neighbour.
- **p04** (`interaction_gnn.py` 200-202): copy i is src->dst, copy E+i dst->src; logit_i = (l_i + l_{E+i}) / 2, so
  scores do not depend on stored direction; edge latents are cat(x[start], x[end]) and stored edge features are never
  read. Only InteractionGNN and InteractionGNNWithPyG double; IGNN2 variants and RecurrentInteractionGNN do not.
- **p05** (`connected_components.py` 80-81 at the pin): `edge_index[:, None]` keeps all edges under PyG 2.7.0; 0002
  selects `edge_y` (bool). Without edge truth it now raises instead of returning one component.
- **p06** (`track_building/utils.py` 41-43): fixed by 0001 (`hit_id[track_edges]`); also used by
  cc_and_all_simple_path. numpy `track_particle_id` would be mis-paired by `repeat(2)` tiling: always torch.
- **p07**: `variable_name_prefix_map` has 84 one-to-one entries (66 hit keys k -> hit_k, 9 edge keys incl.
  y -> edge_y and scores -> edge_scores, 9 track keys), applied to graphs on load/save and to configs (node_features,
  edge_features, weighting, target_tracks, hard_cuts, feature_sets, module_columns). acorn's legacy example reader
  configs list `y` and therefore write no y coordinate.
- **p08**: `lightning_infer` and eval take `max(getctime)` over `rglob('best*.ckpt')` in the whole stage_dir (else any
  `*.ckpt`) and merge `{**ckpt_hparams, **config}`; `acorn eval` never loads weights (`EdgeClassifierStage.evaluate`
  builds `cls(config)` at random init) and ignores a YAML `dataset:` (only `--dataset` works); training resume also
  picks the newest ckpt in `--checkpoint_resume_dir`.
- **p09**: `EventReader._build_true_tracks` links every hit of one module group to every hit of the next (chain over
  the modules hit, not layers). `graph_intersection` is directed unless `undirected=True`; the GNN keeps saved edge_y.
  About 0.45 adjacent-layer same-particle negatives per true edge on HNL events.
- **p10**: 0005 adds `flatten_event_id` to MetricLearning, every EdgeClassifierStage model, TrackBuildingStage and
  Walkthrough; DoubleMetricLearning (breaks with ids nested once) and HierarchicalGNN (twice) still unwrap one level.
- **p11**: 0006 fixes TrackMLReader only; AthenaReader reseeds the global RNG with 42; rerunning a reader into an
  existing stage_dir merges two splits. EventDataset can return fewer than num_events without error.
- **p12**: 0007 evaluates Filter in fixed `torch.split` chunks (bitwise equal on CPU). New: batchnorm with batch
  statistics in eval mode makes logits depend on chunk boundaries and crashes on a 1-row last chunk. GNNFilter still
  catches OOM with a sticky `cached_n`; metric-learning `get_distances` swallows any RuntimeError.

## Data facts found on the way (not acorn bugs)

- 27.6 % of npz true edges join two hits on the same (volume, layer) (overlapping modules); 5 % point inward in r.
- Some particles carry outlier hits far from their trajectory (W muons: about 4 % of their hits are outliers and about 6-7 % of their true edges touch one;
  e.g. a second first-layer pixel hit 0.39 rad away in phi). They inflate any window calibrated on all true edges;
  see DECISIONS.md and the M2 window study.
