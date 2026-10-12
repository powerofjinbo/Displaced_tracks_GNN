"""Canto-L edge classifier on TrackML sectors (docs/trackml_plan.md sections 6.3-6.4).

The network is unchanged: the L1 SymmetricTripletIGNN with the region one-hot and inner-hit truncation augmentation
(configs/stage1/triplet_symregion_trunc.yaml: hidden 128, 8 iterations, triplet top-k 4, 347,777 parameters). Only the
data path differs, because a TrackML event does not fit one forward on an A30:

Event store. Full-event union graphs on node-local disk, one acorn-style file per event,
<input_dir>/<trainset|valset|testset>/event<eid>.pyg.gz (written at job start by `hnl.trackml.graphs stage`; never on
/ceph): r, phi, z (float32, unscaled), hit_particle_id, edge_index (2, E) int32 unordered pairs a < b (base graph from
the trained embedding united with the stored learned-support additions), y (bool per pair: a true edge), event_id.

TrackMLSectorDataset (the model's dataset class) cuts sector graphs on the fly (hnl.trackml.sectors):
  trainset  `windows_per_epoch` random windows per virtual epoch (default 4,000): dataset index g = epoch * W + i (the
            epoch-tagged sampler of hnl.models.augment); event = cycle c = g // N_events of seeded permutations
            (default_rng([seed, c]).permutation), so every training event is visited once per cycle; window centre
            phi ~ U[-pi, pi) from default_rng([seed, g]); core 22.5 deg, halos +-11.25 deg (sectors.window_spec).
            Truncation augmentation (hnl.models.augment.truncation_mask, p = trunc_p, m ~ U{1..n-5}) is decided on the
            full event, deterministic per (seed, g, event), before the window is cut (a truncated particle is a
            late-starting particle in every window that sees it).
  valset /  every INFERENCE sector (8 phi sectors, 45 deg cores, +-22.5 deg halos) of every event, in order
  testset   (index = event * 8 + sector); no augmentation.
  Every item is a PyG Data in acorn's prefixed, preprocessed form: hit_r / hit_phi / hit_z scaled by node_scales,
  hit_particle_id, edge_index (local), edge_y, edge_weights = 1 on the edges the sector owns and 0 on the context
  edges (acorn's loss ignores weight-0 edges, so the loss is the mean over owned edges: the plan's "loss on core
  edges only"), edge_owned, hit_global / edge_global (positions in the event's lists, for stitching), event_id, sector.
TrackMLCanto (configs/trackml/classifier_train.yaml): SymmetricTripletIGNN with that dataset, the epoch-tagged
  training sampler, edge metrics on owned edges only; checkpoint on the lowest valmon loss (owned edges).

Scoring of whole events (8 sector forwards, stitched): hnl.trackml.infer.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from hnl.models.augment import TRUNC_P, _EpochTaggedSampler, truncation_mask  # noqa: F401 (sampler: tests)
from hnl.train.resume_sampler import epoch_completed, resumable_train_loader
from hnl.models.invariant_gnn import SymmetricTripletIGNN
from hnl.models.variants import _register
from hnl.trackml import sectors as S

WINDOWS_PER_EPOCH = 4000


# ------------------------------------------------------------------------------------------------ event store
def save_event_graph(path: str, eid: str, r, phi, z, pid, pairs, y, track_edges, rows=None) -> int:
    """One full-event union graph to <path> (acorn's save_pyg: torch.save + gzip level 1); rows: the hits' rows in the
    full event (truth-filtered conditions). Returns bytes written."""
    from acorn.utils.loading_utils import save_pyg
    g = Data(r=torch.as_tensor(np.asarray(r, dtype=np.float32)), phi=torch.as_tensor(np.asarray(phi, dtype=np.float32)),
             z=torch.as_tensor(np.asarray(z, dtype=np.float32)),
             hit_particle_id=torch.as_tensor(np.asarray(pid, dtype=np.int64)),
             edge_index=torch.as_tensor(np.asarray(pairs, dtype=np.int32)),
             y=torch.as_tensor(np.asarray(y, dtype=bool)),
             track_edges=torch.as_tensor(np.asarray(track_edges, dtype=np.int32)))
    if rows is not None:
        g.rows = torch.as_tensor(np.asarray(rows, dtype=np.int64))
    g.event_id = str(eid)
    g.num_nodes = int(len(r))
    base = path[:-3] if path.endswith(".gz") else path
    save_pyg(g, base)
    return os.path.getsize(base + ".gz")


def load_event_graph(path: str) -> dict:
    from acorn.utils.loading_utils import load_pyg
    g = load_pyg(path[:-3] if path.endswith(".gz") else path, map_location=torch.device("cpu"), weights_only=False)
    return {"eid": str(g.event_id), "r": g.r.numpy(), "phi": g.phi.numpy(), "z": g.z.numpy(),
            "pid": g.hit_particle_id.numpy(), "pairs": g.edge_index.numpy().astype(np.int64), "y": g.y.numpy(),
            "track_edges": g.track_edges.numpy().astype(np.int64),
            "rows": g.rows.numpy() if "rows" in g else np.arange(int(g.num_nodes))}


# ------------------------------------------------------------------------------------------------ sector items
def truncate_full(ev: dict, remove: np.ndarray) -> dict:
    """The event dict without the removed hits and the edges touching them (nodes renumbered)."""
    keep = ~remove
    new = np.cumsum(keep) - 1
    a, b = ev["pairs"]
    ek = keep[a] & keep[b]
    out = dict(ev)
    for k in ("r", "phi", "z", "pid"):
        out[k] = ev[k][keep]
    out["pairs"] = new[ev["pairs"][:, ek]]
    out["y"] = ev["y"][ek]
    out["edge_ids"] = ev.get("edge_ids", np.arange(len(a)))[ek]
    out["hit_ids"] = ev.get("hit_ids", np.arange(len(keep)))[keep]
    if "rows" in ev:
        out["rows"] = ev["rows"][keep]
    te = ev["track_edges"]
    out["track_edges"] = new[te[:, keep[te[0]] & keep[te[1]]]]
    return out


def sector_item(ev: dict, spec: S.SectorSpec, s: int, scales=(1000.0, 3.14, 1000.0), eta=None) -> tuple[Data, S.Sector]:
    """The acorn-form Data of sector s of an event dict (r, phi, z, pid, pairs, y; optional hit_ids / edge_ids into
    the untruncated event) and its Sector."""
    if eta is None:
        eta = S.hit_eta(ev["r"], ev["z"])
    sec = S.split(ev["phi"], eta, ev["pairs"], spec, sectors=[s])[0]
    h, e = sec.hits, sec.edges
    hit_ids = ev.get("hit_ids", None)
    edge_ids = ev.get("edge_ids", None)
    d = Data(hit_r=torch.from_numpy(ev["r"][h].astype(np.float32)) / scales[0],
             hit_phi=torch.from_numpy(ev["phi"][h].astype(np.float32)) / scales[1],
             hit_z=torch.from_numpy(ev["z"][h].astype(np.float32)) / scales[2],
             hit_particle_id=torch.from_numpy(ev["pid"][h].astype(np.int64)),
             edge_index=torch.from_numpy(sec.local_edges.astype(np.int64)),
             edge_y=torch.from_numpy(ev["y"][e].astype(bool)),
             edge_weights=torch.from_numpy(sec.owned.astype(np.float32)),
             edge_owned=torch.from_numpy(sec.owned.astype(bool)),
             hit_global=torch.from_numpy((h if hit_ids is None else hit_ids[h]).astype(np.int64)),
             edge_global=torch.from_numpy((e if edge_ids is None else edge_ids[e]).astype(np.int64)))
    d.event_id = ev["eid"]
    d.sector = int(s)
    d.num_nodes = int(len(h))
    return d, sec


def cycle_event(g: int, n_events: int, seed: int) -> int:
    """Position of the event used at global training step g: cycle c = g // n of seeded permutations."""
    c, k = divmod(int(g), int(n_events))
    return int(np.random.default_rng([int(seed), c]).permutation(n_events)[k])


def window_center(g: int, seed: int) -> float:
    return float(np.random.default_rng([int(seed), 7, int(g)]).uniform(-np.pi, np.pi))


class TrackMLSectorDataset(Dataset):
    """See the module docstring. Constructor arguments as acorn's GraphDataset (made by the stage's resolver)."""

    def __init__(self, input_dir, data_name=None, num_events=None, stage="fit", hparams=None, transform=None,
                 pre_transform=None, pre_filter=None, preprocess=True):
        from acorn.utils.loading_utils import load_datafiles_in_dir
        super().__init__(input_dir, transform, pre_transform, pre_filter)
        self.input_dir, self.data_name, self.stage = input_dir, data_name, stage
        self.hparams = hparams or {}
        self.input_paths = sorted(load_datafiles_in_dir(input_dir, data_name, num_events))
        self.n_events = len(self.input_paths)
        self.seed = int(self.hparams.get("window_seed", self.hparams.get("trunc_seed", torch.initial_seed())))
        self.windows = int(self.hparams.get("windows_per_epoch", WINDOWS_PER_EPOCH))
        self.trunc_p = float(self.hparams.get("trunc_p", TRUNC_P))
        tw = self.hparams.get("train_window", {}) or {}
        self.window_core = float(tw.get("core_width_deg", 22.5)) * S.DEG
        self.window_halo = float(tw.get("halo_phi_deg", 11.25)) * S.DEG
        inf = self.hparams.get("inference_sectors") or {}
        self.spec = S.SectorSpec.from_config(inf) if inf else S.INFERENCE
        feats = [f.removeprefix("hit_") for f in self.hparams.get("node_features", ["r", "phi", "z"])]
        if feats != ["r", "phi", "z"]:
            raise ValueError(f"node_features {feats}: the sector items carry r, phi, z")
        self.scales = tuple(float(s) for s in self.hparams.get("node_scales", [1000.0, 3.14, 1000.0]))
        self._cache = (None, None)

    def len(self):
        if self.data_name == "trainset":
            return self.windows
        return self.n_events * self.spec.n_sectors

    def __getitem__(self, idx):
        # epoch-tagged training indices run past len(): bypass PyG's index mapping (as augment.TruncationGraphDataset)
        if isinstance(idx, (int, np.integer)):
            data = self.get(int(idx))
            return data if self.transform is None else self.transform(data)
        return super().__getitem__(idx)

    def _event(self, k: int) -> dict:
        if self._cache[0] != k:
            self._cache = (k, load_event_graph(self.input_paths[k]))
        return self._cache[1]

    def item(self, idx: int) -> tuple[Data, S.Sector]:
        if self.data_name == "trainset":
            g = int(idx)
            k = cycle_event(g, self.n_events, self.seed)
            ev = self._event(k)
            if self.trunc_p > 0:
                eid = int(ev["eid"])
                rng = np.random.default_rng([self.seed, g, eid])
                R = np.hypot(ev["r"].astype(np.float64), ev["z"].astype(np.float64))
                remove = truncation_mask(ev["pid"], R, rng, self.trunc_p)
                if remove.any():
                    ev = truncate_full(ev, remove)
            spec = S.window_spec(window_center(g, self.seed), self.window_core, self.window_halo)
            return sector_item(ev, spec, 0, self.scales)
        k, s = divmod(int(idx), self.spec.n_sectors)
        return sector_item(self._event(k), self.spec, s, self.scales)

    def get(self, idx):
        return self.item(idx)[0]


# ------------------------------------------------------------------------------------------------ model
class TrackMLCanto(SymmetricTripletIGNN):
    """L1 SymmetricTripletIGNN (detector_input region) trained on TrackML windows with truncation augmentation."""

    def __init__(self, hparams):
        hparams = dict(hparams)
        hparams["dataset_class"] = "TrackMLSectorDataset"
        super().__init__(hparams)
        _register(self, TrackMLSectorDataset)

    def train_dataloader(self):
        # the order of hnl.models.augment's epoch-tagged loader under the runner's DDPStrategy (Lightning's
        # DistributedSamplerWrapper, shuffled per epoch), resumable at the next batch after a mid-epoch checkpoint
        # (hnl.train.resume_sampler; S4 runs in one-hour chunks, DECISIONS 2026-10-08)
        if self.trainset is None:
            return None
        num_workers = self.hparams.get("num_workers", [1, 1, 1])[0]
        return resumable_train_loader(self.trainset, self, num_workers, tagged=True)

    def test_data(self, stage):
        for ds in (self.trainset, self.valset, self.testset):
            if ds is None:
                continue
            d = ds.get(0)
            for k in ("hit_r", "hit_phi", "hit_z", "edge_index", "edge_y", "edge_weights", "edge_owned"):
                if k not in d:
                    raise KeyError(f"{ds.data_name}: sector item lacks {k}")

    def log_metrics(self, batch, output, all_truth, target_truth, loss):
        m = batch.edge_owned.bool()
        return super().log_metrics(batch, output[m], all_truth[m], target_truth[m], loss)

    def on_validation_start(self):
        """Inference sectors are larger than the training windows (G0: up to 11 GB allocated against 14 GB, but 22 GB
        reserved for one sector): the allocator's cache of training-shaped blocks goes back to the driver before
        validation and that of the validation after it. S4 chunk 6727230 ran out of memory at its first validation
        after 1,897 training windows with 13.3 GB reserved but unallocated (DECISIONS 2026-10-08)."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_validation_end(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_train_epoch_end(self):
        """Keep the weights after the listed virtual epochs (default the first: gates G2 / G3 use it, plan 6.3 / 6.7)
        as <stage_dir>/artifacts/epochs/virtual_epoch<e>.ckpt (acorn_runner keeps only every 5th epoch). Only after a
        completed epoch: a chunk's pre-timeout stop in the middle of one also runs this hook (S4 in one-hour chunks)."""
        e = int(self.current_epoch)
        if not epoch_completed(self.trainer):
            return
        if e in [int(x) for x in self.hparams.get("keep_epochs", [0])] and self.hparams.get("stage_dir"):
            path = os.path.join(self.hparams["stage_dir"], "artifacts", "epochs", f"virtual_epoch{e}.ckpt")
            if not os.path.exists(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                self.trainer.save_checkpoint(path)


def model_hparams(config: dict) -> dict:
    """A config dict in the prefixed form the model expects (as acorn_runner does before constructing it)."""
    import copy
    from acorn.utils.loading_utils import add_variable_name_prefix_in_config
    return add_variable_name_prefix_in_config(copy.deepcopy(config))
