"""TrackML events for the Canto pipeline: event sets, hit arrays, evaluation conditions, the metric-learning graph and
the learned-triplet-support additions (docs/trackml_plan.md sections 2-3, 6.0-6.3, 6.5).

Event sets. `event_sets(split, train_scope)` reads a fixed split file: configs/trackml/split_v1.json (accuracy phase,
train_1; keys dev, val, valmon, test, train_train1, train_all) or configs/trackml/split_tp_v1.json (throughput phase,
written by `hnl.trackml.split make-tp` when training_part01 is first streamed; keys dev, val, valmon, test, train).
D1 (plan section 1.4, decided by the user 2026-10-05): the throughput phase; training = the split's train events of
training_part01 and training_part02 (D2, train_1-sized), selected here by the archives the tables were converted from;
the accuracy-phase train_1 test events are the cross-phase test set T2. Calibration events (truth kinematics stored):
the converter's choice (hnl.trackml.convert.load_split for train_1).

Hits. `load_events` returns per event the stored hit columns (hnl.trackml.convert.read_table: x, y, z, volume, layer,
module, pid, rank, cell features ncells, charge, span0, span1, weight) and the derived ones (r, phi as float32 like our
tables; layer_id = volume * 1000 + layer as hnl.trackbuild.mutex_ws.hit_layers; the true edges = consecutive hits of a
particle along rank, noise excluded), plus the event's particles table. Graph node order = row order = hit_id - 1.
Metric-learning inputs (`node_input`): r, phi, z (S1) and, for X1 (plan D3), the log cluster features of
CLUSTER_FEATURES.

Conditions (plan section 3.5): "full" (all hits; every choice is made here), "nonoise" (noise hits removed,
truth-based: Ju et al.'s training condition), "1gev" (only hits of particles with pT > 1 GeV, noise removed:
"TrackML-1GeV" / reduced TrackML). `subset` keeps the original rows (Event.rows) so that results map back to hit ids.

Graph (plan sections 6.1-6.2). The metric-learning graph: every unordered pair within r* in the 12-D embedding with
FRNN's per-query cap K (the K nearest including the point itself), as hnl.train.acorn_runner infer builds it: on CUDA
through acorn's build_edges (FRNN), on CPU through hnl.graphs.helix_union.radius_pairs (cKDTree; it reproduces the
stored FRNN graphs, helix_union --check-base). The union graph adds the learned triplet support of
hnl.graphs.learned_triplet_filter on the candidates within r_w (no cap), with the selection evaluated in chunks of
middle hits (`support_select`): a triplet's two legs meet at its middle hit and the mutual top-k ranks compare only
triplets with the same middle hit, so the chunked result equals LT.select pair for pair, ties included (the
triplets of one middle hit keep their relative order); tests/ and `check_support` compare the two. Field (plan section
6.5): with a field map B(r, |z|) the feature log_pt_turn uses the field at the middle hit instead of 2 T.

Event ids. TrackML event e has the acorn event id 9 * 10^8 + e (string f"9{e:08d}"): the sample-code convention of
hnl.data.acorn_store with code 9 for TrackML, so TrackML and HNL ids never collide.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from hnl.overlay import grant as GRANT
from hnl.trackml.convert import DERIVED, event_index, read_table
from hnl.trackml.split import EXT, REPO

TRACKML_CODE = 9
# Throughput-phase tables (hnl.trackml.throughput.DERIVED_TP). Event ids do not overlap between the phases: accuracy
# 1000-9999 (derived/v1: T2, the smoke and dev events 1000-1099), throughput from 21000 (training_part01 / 02).
DERIVED_TP = os.path.join(EXT, "derived", "tp_v1")
TP_FIRST_ID = 21000
# Pileup overlay tables (docs/pileup_overlay_plan.md section 5.4, decision D9; written by hnl.overlay.writer): overlay
# ids 50,000,000 + 1,000,000 k + host id, above every TrackML id.
DERIVED_OVERLAY = os.path.join(EXT, "derived", "overlay_v1")
OVERLAY_FIRST_ID = 50_000_000
# Module-frame resolutions of TrackML (plan section 6.5, configs/trackml/res_v1.json, measured on the 200 calibration
# events): they replace the ttbar table in the hit sigmas of hnl.graphs.helix_union (legs, candidate triplets and the
# features of the learned triplet support) in every process that imports this module, i.e. every TrackML stage. HNL
# processes never import it and keep hnl.geometry.resolution.RES.
RES_TML_PATH = os.path.join(REPO, "configs", "trackml", "res_v1.json")


def use_trackml_resolution(path: str = RES_TML_PATH) -> dict:
    """Set hnl.graphs.helix_union.FINE / COARSE (read at call time by hit_sigmas) to the TrackML table; returns it."""
    from hnl.graphs import helix_union as HU
    with open(path) as fh:
        res = {int(v): (float(s_u), float(s_v)) for v, (s_u, s_v) in json.load(fh)["RES"].items()}
    if set(res) != set(HU.FINE):
        raise ValueError(f"TrackML RES volumes {sorted(res)} differ from {sorted(HU.FINE)}")
    for v, (s_u, s_v) in res.items():
        HU.FINE[v], HU.COARSE[v] = s_u, s_v
    return res


RES_TML = use_trackml_resolution()
SPLIT_ACCURACY = os.path.join(REPO, "configs", "trackml", "split_v1.json")
SPLIT_THROUGHPUT = os.path.join(REPO, "configs", "trackml", "split_tp_v1.json")
CONDITIONS = ("full", "nonoise", "1gev")
HIT_COLUMNS = ["event", "row", "x", "y", "z", "r", "phi", "volume", "layer", "module", "pid", "rank", "ncells",
               "charge", "span0", "span1", "weight"]
# X1 (plan D3 and section 7): the stored cluster values as metric-learning inputs. The cell count and the cluster extents
# are >= 1 and heavy-tailed (pixel clusters up to ~80 cells); the charge sum is the sum of the cell values (strips:
# binary cells, charge = ncells; pixels: fractional values, ~0.03-3 per cluster) and a few cells carry anomalous values
# (716,689 in train_1 event 1532, hnl.trackml.convert). Each enters as its natural log (the charge clipped at
# CHARGE_FLOOR first), so every input stays O(1) and an anomalous cell cannot dominate the first layer; acorn's
# node_scales for these features are 1. Feature-store attribute hit_<name>; the Embedder reads the same transform.
CHARGE_FLOOR = 1e-3
CLUSTER_FEATURES = ("log_ncells", "log_charge", "log_span0", "log_span1")


def acorn_eid(event: int) -> str:
    return f"{TRACKML_CODE}{int(event):08d}"


def event_from_eid(eid) -> int:
    e = int(eid)
    if e // 100_000_000 != TRACKML_CODE:
        raise ValueError(f"{eid} is not a TrackML acorn event id")
    return e % 100_000_000


# ------------------------------------------------------------------------------------------------ event sets
def event_sets(split: str = SPLIT_THROUGHPUT, train_archives=None, root: str = DERIVED) -> dict:
    """{dev, val, valmon, test, train} event lists (sorted) of a split file. train: the accuracy file's train_train1
    (D2 = train_1; train_archives ignored) or the throughput file's train events stored from train_archives
    (default training_part01, training_part02: D2)."""
    if not os.path.exists(split):
        raise FileNotFoundError(f"split file {split} does not exist (throughput phase: written by "
                                f"`python -m hnl.trackml.split make-tp` when training_part01 is first streamed)")
    with open(split) as fh:
        s = json.load(fh)
    out = {k: sorted(int(e) for e in s[k]) for k in ("dev", "val", "valmon", "test")}
    if "train_train1" in s:
        out["train"] = sorted(int(e) for e in s["train_train1"])
        out["phase"] = "accuracy"
    else:
        arch = list(train_archives or ("training_part01", "training_part02"))
        idx = event_index(root, "hits")
        out["train"] = sorted(int(e) for e in s["train"] if idx.get(int(e)) in arch)
        out["phase"] = "throughput"
        out["train_archives"] = arch
    held = set(out["val"]) | set(out["test"])
    if held & set(out["train"]):
        raise RuntimeError("train overlaps val / test")
    out["split"] = split
    return out


# ------------------------------------------------------------------------------------------------ events
@dataclass
class Event:
    event: int
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    r: np.ndarray
    phi: np.ndarray
    volume: np.ndarray
    layer: np.ndarray
    module: np.ndarray
    pid: np.ndarray
    rank: np.ndarray
    ncells: np.ndarray
    charge: np.ndarray
    weight: np.ndarray
    true_edges: np.ndarray                   # (2, T) rows, src = earlier hit along the particle
    particles: pd.DataFrame | None = None
    rows: np.ndarray | None = None           # rows in the full event (None: this is the full event)
    condition: str = "full"
    meta: dict = field(default_factory=dict)
    span0: np.ndarray | None = None          # cluster extent in cells along ch0 / ch1 (X1 inputs)
    span1: np.ndarray | None = None

    @property
    def n(self) -> int:
        return int(len(self.r))

    @property
    def eid(self) -> str:
        return acorn_eid(self.event)

    @property
    def layer_id(self) -> np.ndarray:
        return self.volume.astype(np.int64) * 1000 + self.layer.astype(np.int64)

    @property
    def noise(self) -> np.ndarray:
        return self.pid == 0

    @property
    def R(self) -> np.ndarray:
        return np.hypot(self.r.astype(np.float64), self.z.astype(np.float64))

    @property
    def eta(self) -> np.ndarray:
        from hnl.trackml.sectors import hit_eta
        return hit_eta(self.r, self.z)

    @property
    def full_rows(self) -> np.ndarray:
        return np.arange(self.n) if self.rows is None else self.rows


def _true_edges(h: pd.DataFrame) -> np.ndarray:
    s = h[h.pid.to_numpy() != 0]
    pid, rank, row = s.pid.to_numpy(), s["rank"].to_numpy(), s.row.to_numpy()
    o = np.lexsort((rank, pid))
    pid, row = pid[o], row[o]
    nxt = pid[1:] == pid[:-1]
    return np.stack([row[:-1][nxt], row[1:][nxt]]).astype(np.int64)


def _event_from(h: pd.DataFrame, p: pd.DataFrame | None, e: int) -> Event:
    h = h.sort_values("row")
    if not np.array_equal(h.row.to_numpy(), np.arange(len(h))):
        raise RuntimeError(f"event {e}: rows are not 0..N-1")
    g = lambda c, dt=None: h[c].to_numpy() if dt is None else h[c].to_numpy().astype(dt)
    opt = lambda c: g(c) if c in h.columns else None
    return Event(event=int(e), x=g("x"), y=g("y"), z=g("z"), r=g("r"), phi=g("phi"), volume=g("volume", np.int64),
                 layer=g("layer", np.int64), module=g("module", np.int64), pid=g("pid", np.int64),
                 rank=g("rank", np.int64), ncells=g("ncells"), charge=g("charge"), weight=g("weight"),
                 true_edges=_true_edges(h), particles=p, span0=opt("span0"), span1=opt("span1"))


def table_root(event: int) -> str:
    """The tables holding an event: derived/overlay_v1 for overlay ids (>= 50,000,000), derived/tp_v1 for
    throughput-phase ids (>= 21000), else derived/v1."""
    return DERIVED_OVERLAY if int(event) >= OVERLAY_FIRST_ID else DERIVED_TP if int(event) >= TP_FIRST_ID else DERIVED


def iter_events(events, chunk: int = 25, root: str | None = None, particles: bool = True):
    """Events in the given order, read chunk by chunk (one parquet read per table, chunk and table root). root None:
    each event from the tables of its phase (`table_root`)."""
    events = [int(e) for e in events]
    for k in range(0, len(events), chunk):
        part = events[k:k + chunk]
        hg, pg = {}, {}
        for r in sorted({root or table_root(e) for e in part}):
            sub = [e for e in part if (root or table_root(e)) == r]
            hg.update(dict(tuple(read_table("hits", sub, HIT_COLUMNS, r).groupby("event"))))
            if particles:
                pg.update(dict(tuple(read_table("particles", sub, root=r).groupby("event"))))
        for e in part:
            if e not in hg:
                raise KeyError(f"event {e} not found in the tables {root or table_root(e)}")
            yield _event_from(hg[e], pg.get(e), e)


def load_events(events, root: str | None = None, particles: bool = True) -> dict[int, Event]:
    return {ev.event: ev for ev in iter_events(events, root=root, particles=particles)}


def condition_mask(ev: Event, condition: str) -> np.ndarray:
    """Hits kept in an evaluation condition (plan section 3.5; truth-based for nonoise and 1gev)."""
    if condition == "full":
        return np.ones(ev.n, dtype=bool)
    if condition == "nonoise":
        return ~ev.noise
    if condition == "1gev":
        if ev.particles is None:
            raise ValueError("condition 1gev needs the particles table")
        p = ev.particles
        fast = p.particle_id.to_numpy()[p.pt.to_numpy() > 1.0]
        return (~ev.noise) & np.isin(ev.pid, fast)
    raise ValueError(f"condition {condition!r} not in {CONDITIONS}")


def subset(ev: Event, keep: np.ndarray, condition: str | None = None) -> Event:
    """The event restricted to the hits in `keep` (bool per hit), renumbered; true edges with both hits kept (the
    conditions drop whole particles or noise, so no chain is broken); rows = original rows."""
    keep = np.asarray(keep, dtype=bool)
    new = np.cumsum(keep) - 1
    te = ev.true_edges[:, keep[ev.true_edges[0]] & keep[ev.true_edges[1]]]
    arr = {k: getattr(ev, k)[keep] for k in ("x", "y", "z", "r", "phi", "volume", "layer", "module", "pid", "rank",
                                             "ncells", "charge", "weight")}
    arr.update({k: getattr(ev, k)[keep] for k in ("span0", "span1") if getattr(ev, k) is not None})
    return replace(ev, **arr, true_edges=new[te], rows=ev.full_rows[keep], condition=condition or ev.condition)


def with_condition(ev: Event, condition: str) -> Event:
    return ev if condition == "full" else subset(ev, condition_mask(ev, condition), condition)


# ------------------------------------------------------------------------------------------------ node inputs
def cluster_feature(ev: Event, name: str) -> np.ndarray:
    """An X1 input (CLUSTER_FEATURES) of every hit, float32: log(ncells), log(max(charge, CHARGE_FLOOR)), log(span0),
    log(span1), computed in float64 from the stored values."""
    if name not in CLUSTER_FEATURES:
        raise KeyError(f"{name!r} is not a cluster feature {CLUSTER_FEATURES}")
    src = name.removeprefix("log_")
    v = getattr(ev, src)
    if v is None:
        raise ValueError(f"event {ev.event}: {src} was not read (hnl.trackml.events.HIT_COLUMNS)")
    v = np.asarray(v, dtype=np.float64)
    if src == "charge":
        v = np.maximum(v, CHARGE_FLOOR)
    elif v.size and v.min() < 1:
        raise ValueError(f"event {ev.event}: {src} below 1")
    return np.log(v).astype(np.float32)


def node_input(ev: Event, name: str) -> np.ndarray:
    """A metric-learning node feature of every hit by its acorn name without the hit_ prefix: an X1 cluster feature or
    an Event column (r, phi, z, ...)."""
    return cluster_feature(ev, name) if name in CLUSTER_FEATURES else np.asarray(getattr(ev, name))


# ------------------------------------------------------------------------------------------------ embedding
class Embedder:
    """acorn MetricLearning's network (make_mlp + F.normalize) from a checkpoint, on CPU or CUDA. Inputs are the
    checkpoint's node_features (`node_input`: Event columns or the X1 cluster features) divided by node_scales in
    float32, as acorn's dataset scales them."""

    def __init__(self, ckpt: str | None = None, device: str = "cpu", hparams: dict | None = None, seed: int = 0):
        import torch
        from acorn.stages.graph_construction.models.utils import make_mlp
        if ckpt is not None:
            sd = torch.load(ckpt, map_location="cpu", weights_only=False)
            hp = sd["hyper_parameters"]
        else:                                    # an untrained network (smoke tests): hparams from a config
            hp = hparams
            torch.manual_seed(seed)
        self.ckpt = ckpt
        self.features = [f.removeprefix("hit_") for f in hp["node_features"]]
        self.scales = [float(s) for s in hp["node_scales"]]
        self.net = make_mlp(len(self.features), [hp["emb_hidden"]] * hp["nb_layer"] + [hp["emb_dim"]],
                            hidden_activation=hp["activation"], output_activation=None, layer_norm=True)
        if ckpt is not None:
            self.net.load_state_dict({k[len("network."):]: v for k, v in sd["state_dict"].items()
                                      if k.startswith("network.")})
        self.net.eval().to(device)
        self.device = device

    def inputs(self, ev: Event):
        """The network's input matrix (N, F) on the CPU: scaled node features, NaN -> 0 (acorn's get_input_data)."""
        import torch
        cols = [torch.tensor(node_input(ev, f), dtype=torch.float32) / s for f, s in zip(self.features, self.scales)]
        x = torch.stack(cols, dim=-1)
        x[x != x] = 0
        return x

    def __call__(self, ev: Event, chunk: int = 65536):
        import torch
        # choke point (addendum H): an embedding network runs on an overlay event only under a grant of kind model or
        # plumb (hnl.overlay.grant; a no-op without a git call for TrackML ids)
        GRANT.check([ev.event], ("model", "plumb"))
        x = self.inputs(ev).to(self.device)
        with torch.inference_mode():
            return torch.cat([torch.nn.functional.normalize(self.net(x[i:i + chunk])) for i in range(0, len(x), chunk)])


def radius_graph(emb, r: float, k_max: int | None = None) -> np.ndarray:
    """Unique unordered pairs (2, E), a < b, within r in the embedding; k_max: FRNN's per-query cap (the k_max nearest
    including the point itself). CUDA tensors: acorn build_edges (FRNN), as hnl.train.acorn_runner infer; CPU:
    hnl.graphs.helix_union.radius_pairs."""
    import torch
    from hnl.graphs.ceiling import unordered_unique
    if isinstance(emb, torch.Tensor) and emb.is_cuda:
        from acorn.stages.graph_construction.models.utils import FRNN_AVAILABLE, build_edges
        if not FRNN_AVAILABLE:
            raise RuntimeError("FRNN not available on this node")
        e = build_edges(query=emb, database=emb, indices=None, r_max=float(r), k_max=int(k_max or 1000),
                        backend="FRNN")
        a, b = torch.minimum(e[0], e[1]), torch.maximum(e[0], e[1])
        code = torch.unique(a * emb.shape[0] + b)
        return np.stack([(code // emb.shape[0]).cpu().numpy(), (code % emb.shape[0]).cpu().numpy()]).astype(np.int64)
    from hnl.graphs.helix_union import radius_pairs
    x = emb.cpu().numpy() if isinstance(emb, torch.Tensor) else np.asarray(emb)
    return unordered_unique(radius_pairs(x, float(r), int(k_max) if k_max else None))


def true_pair_mask(n: int, pairs: np.ndarray, true_edges: np.ndarray) -> np.ndarray:
    """For unordered pairs (a < b): is the pair a true edge (consecutive hits of one particle)."""
    from hnl.graphs.ceiling import unordered_unique
    t = unordered_unique(true_edges)
    return np.isin(pairs[0].astype(np.int64) * n + pairs[1], t[0].astype(np.int64) * n + t[1])


# ------------------------------------------------------------------------------------------------ field map
class FieldMap:
    """B_eff(r, |z|) [T] from configs/trackml/field_map_v1.json (plan section 6.5: median per 50 mm x 100 mm bin of
    calibration-event truth triplets, bilinear interpolation, empty bins filled from the nearest filled one)."""

    def __init__(self, path: str):
        with open(path) as fh:
            d = json.load(fh)
        self.r_edges = np.asarray(d["r_edges"], dtype=np.float64)
        self.z_edges = np.asarray(d["absz_edges"], dtype=np.float64)
        self.B = np.asarray(d["B"], dtype=np.float64)          # (n_r, n_z) bin medians, filled
        self.meta = d.get("provenance", {})

    def __call__(self, r, absz) -> np.ndarray:
        from scipy.interpolate import RegularGridInterpolator
        rc = 0.5 * (self.r_edges[1:] + self.r_edges[:-1])
        zc = 0.5 * (self.z_edges[1:] + self.z_edges[:-1])
        f = RegularGridInterpolator((rc, zc), self.B, bounds_error=False, fill_value=None)
        pts = np.stack([np.clip(r, rc[0], rc[-1]), np.clip(absz, zc[0], zc[-1])], 1)
        return f(pts)


def field_log_pt_turn(lg: dict, e_in: np.ndarray, e_out: np.ndarray, B_mid: np.ndarray) -> np.ndarray:
    """Feature log_pt_turn of hnl.graphs.learned_triplet_filter.triplet_matrix with the field B_mid [T] at the middle
    hit instead of 2 T (the same expression with MM_PER_GEV = 1000 / (0.3 B))."""
    L, ang = lg["L"], lg["ang"]
    turn = np.abs(np.remainder(ang[e_out] - ang[e_in] + np.pi, 2 * np.pi) - np.pi)
    half = 0.5 * (L[e_in] + L[e_out])
    mm_per_gev = 1000.0 / (0.3 * np.maximum(np.asarray(B_mid, dtype=np.float64), 1e-3))
    pt = half / (mm_per_gev * np.maximum(turn, 1e-6))
    return np.log10(np.clip(pt, 1e-2, 1e4)).astype(np.float32)


# ------------------------------------------------------------------------------------------------ triplet support
def support_legs(ev: Event, cand: np.ndarray):
    """Oriented legs of candidate pairs as hnl.graphs.learned_triplet_filter.legs_and_triplets: (src, dst, leg table,
    same-layer flags, usable legs) in float64 from the stored float32 coordinates."""
    from hnl.graphs import helix_union as HU
    r, phi, z = (np.asarray(x, dtype=np.float64) for x in (ev.r, ev.phi, ev.z))
    xyz = np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)
    R = np.hypot(r, z)
    a, b = np.asarray(cand[0], dtype=np.int64), np.asarray(cand[1], dtype=np.int64)
    fwd = (R[a] < R[b]) | ((R[a] == R[b]) & (a < b))
    src, dst = np.where(fwd, a, b), np.where(fwd, b, a)
    lid = ev.layer_id
    same_layer = lid[src] == lid[dst]
    lg = HU.legs(xyz, src, dst, ev.volume)
    use = (~same_layer) & (lg["L"] > 1e-3)
    return src, dst, lg, same_layer, use


def middle_hit_chunks(n: int, src, dst, use, max_triplets: float) -> list[np.ndarray]:
    """Groups of middle hits (contiguous hit ranges) whose upper bound of triplets, sum of in-legs x out-legs over the
    usable legs, stays below max_triplets (a single hit above it forms its own group)."""
    din = np.bincount(dst[use], minlength=n).astype(np.float64)
    dout = np.bincount(src[use], minlength=n).astype(np.float64)
    ub = din * dout
    groups, start, acc = [], 0, 0.0
    for v in np.flatnonzero(ub > 0):
        if acc > 0 and acc + ub[v] > max_triplets:
            groups.append((start, int(v)))
            start, acc = int(v), 0.0
        acc += ub[v]
    groups.append((start, n))
    return [np.arange(a, b) for a, b in groups if b > a]


def count_triplets(ev: Event, cand: np.ndarray, max_triplets: float = 2e7) -> int:
    """Number of candidate triplets (hnl.graphs.helix_union.candidate_triplets) of the candidate pairs, by chunks."""
    from hnl.graphs import helix_union as HU
    src, dst, lg, same_layer, use = support_legs(ev, cand)
    n, tot = ev.n, 0
    if not len(src):
        return 0
    for mids in middle_hit_chunks(n, src, dst, use, max_triplets):
        inm = np.zeros(n, dtype=bool)
        inm[mids] = True
        use_c = use & (inm[dst] | inm[src])
        if use_c.any():
            e_in, _ = HU.candidate_triplets(src, dst, lg, use_c)
            tot += int(inm[dst[e_in]].sum())
    return tot


def support_select_multi(ev: Event, cand: np.ndarray, model, ops, field_map: FieldMap | None = None,
                         max_triplets: float = 2e7, triplet_data=None) -> tuple[dict, dict]:
    """The learned triplet support at several operating points ops = [(tau, k), ...] in one pass over the triplets:
    {(tau, k): kept unordered pairs (2, E')} and statistics. For each point: the legs of the triplets with p > tau among
    the k most probable of both their legs, plus same-layer pairs within SAME_LAYER_MAX whose hits are both on kept legs
    (hnl.graphs.learned_triplet_filter.select), evaluated in chunks of middle hits (exact: see the module docstring).
    triplet_data(e_in, e_out, X, p, src, dst) is called per chunk (e.g. to collect training rows)."""
    from hnl.graphs import helix_union as HU
    from hnl.graphs import learned_triplet_filter as LT
    from hnl.graphs.ceiling import unordered_unique
    ops = [(float(t), int(k)) for t, k in ops]
    src, dst, lg, same_layer, use = support_legs(ev, cand)
    n = ev.n
    keep = {op: np.zeros(len(src), dtype=bool) for op in ops}
    st = {"n_cand": int(len(src)), "n_trip_cand": 0, "n_chunks": 0,
          **{f"n_trip_kept_tau{t:g}_k{k}": 0 for t, k in ops}}
    if len(src):
        B = None
        if field_map is not None:
            B = field_map(np.asarray(ev.r, dtype=np.float64), np.abs(np.asarray(ev.z, dtype=np.float64)))
        for mids in middle_hit_chunks(n, src, dst, use, max_triplets):
            inm = np.zeros(n, dtype=bool)
            inm[mids] = True
            use_c = use & (inm[dst] | inm[src])
            if not use_c.any():
                continue
            e_in, e_out = HU.candidate_triplets(src, dst, lg, use_c)
            t = inm[dst[e_in]]
            e_in, e_out = e_in[t], e_out[t]
            st["n_chunks"] += 1
            if not len(e_in):
                continue
            X = LT.triplet_matrix(lg, e_in, e_out)
            if B is not None:
                X[:, LT.FEATURES.index("log_pt_turn")] = field_log_pt_turn(lg, e_in, e_out, B[dst[e_in]])
            p = model.predict_proba(X)[:, 1] if model is not None else np.zeros(len(e_in))
            if triplet_data is not None:
                triplet_data(e_in, e_out, X, p, src, dst)
            st["n_trip_cand"] += int(len(e_in))
            for tau, k in ops:
                good = p > tau
                ei, eo, pg = e_in[good], e_out[good], p[good]
                m = (HU.rank_within(ei, -pg) < k) & (HU.rank_within(eo, -pg) < k)
                keep[(tau, k)][ei[m]] = True
                keep[(tau, k)][eo[m]] = True
                st[f"n_trip_kept_tau{tau:g}_k{k}"] += int(m.sum())
    near = np.hypot(lg["L"], lg["dz"]) < HU.SAME_LAYER_MAX
    out = {}
    for op in ops:
        kp = keep[op]
        on = np.zeros(n, dtype=bool)
        on[src[kp]] = True
        on[dst[kp]] = True
        sl = same_layer & ~kp & near & on[src] & on[dst]
        out[op] = unordered_unique(np.stack([src[kp | sl], dst[kp | sl]]))
    return out, st


def support_select(ev: Event, cand: np.ndarray, model, tau: float, k: int, field_map: FieldMap | None = None,
                   max_triplets: float = 2e7, return_stats: bool = False):
    """Kept unordered pairs (2, E') of the learned triplet support at one operating point (support_select_multi)."""
    out, st = support_select_multi(ev, cand, model, [(tau, k)], field_map, max_triplets)
    kept = out[(float(tau), int(k))]
    return (kept, st) if return_stats else kept


def support_select_reference(ev: Event, cand: np.ndarray, model, tau: float, k: int) -> np.ndarray:
    """The unchunked hnl.graphs.learned_triplet_filter path (legs_and_triplets + select), for exactness checks."""
    from hnl.graphs import learned_triplet_filter as LT
    s, d, lg, sl, ei, eo, X = LT.legs_and_triplets(ev.r, ev.phi, ev.z, ev.volume, ev.layer_id, cand)
    p = model.predict_proba(X)[:, 1] if len(X) else np.zeros(0)
    return LT.select(s, d, lg, sl, ei, eo, p, tau, k, ev.n)


def union_additions(ev: Event, emb, base: np.ndarray, r_wide: float, model, tau: float, k: int,
                    field_map: FieldMap | None = None, max_triplets: float = 2e7) -> tuple[np.ndarray, dict]:
    """Pairs (2, A) the learned support adds to the base graph (union minus base, a < b, sorted), and statistics."""
    from hnl.graphs.ceiling import unordered_unique
    wide = radius_graph(emb, r_wide, None)
    kept, st = support_select(ev, wide, model, tau, k, field_map, max_triplets, return_stats=True)
    n = ev.n
    union = unordered_unique(np.concatenate([base, kept], axis=1))
    added = union[:, ~np.isin(union[0] * n + union[1], base[0].astype(np.int64) * n + base[1])]
    st.update({"n_wide": int(wide.shape[1]), "n_kept": int(kept.shape[1]), "n_base": int(base.shape[1]),
               "n_union": int(union.shape[1]), "n_added": int(added.shape[1])})
    return added, st


def union_graph(base: np.ndarray, added: np.ndarray | None) -> np.ndarray:
    from hnl.graphs.ceiling import unordered_unique
    if added is None or added.shape[1] == 0:
        return base
    return unordered_unique(np.concatenate([base, added], axis=1))


# ------------------------------------------------------------------------------------------------ additions store
ADDITION_SCHEMA = None


def write_additions(path: str, additions: dict[int, np.ndarray], meta: dict) -> dict:
    """Union additions of several events to one zstd parquet file (one row group per event; columns event, a, b as
    int32 with a < b sorted) and a JSON sidecar with the meta data and per-event counts. Atomic (tmp + rename)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema = pa.schema([("event", pa.int32()), ("a", pa.int32()), ("b", pa.int32())])
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with pq.ParquetWriter(tmp, schema, compression="zstd", compression_level=9,
                          column_encoding={"a": "DELTA_BINARY_PACKED", "b": "DELTA_BINARY_PACKED"},
                          use_dictionary=["event"]) as w:
        for e in sorted(additions):
            p = np.asarray(additions[e], dtype=np.int64)
            t = pa.Table.from_arrays([pa.array(np.full(p.shape[1], e, dtype=np.int32)),
                                      pa.array(p[0].astype(np.int32)), pa.array(p[1].astype(np.int32))], schema=schema)
            w.write_table(t, row_group_size=max(t.num_rows, 1))
    os.replace(tmp, path)
    side = {**meta, "file": os.path.basename(path), "bytes": os.path.getsize(path),
            "events": {str(e): int(additions[e].shape[1]) for e in sorted(additions)}}
    with open(path + ".json", "w") as fh:
        json.dump(side, fh, indent=1)
    return side


def read_additions(directory: str, events) -> dict[int, np.ndarray]:
    """Union additions of the given events from the parquet files of a directory (KeyError for a missing event)."""
    import pyarrow.parquet as pq
    want = {int(e) for e in events}
    out: dict[int, np.ndarray] = {}
    for f in sorted(os.listdir(directory)):
        if not f.endswith(".parquet"):
            continue
        with open(os.path.join(directory, f + ".json")) as fh:
            have = {int(e) for e in json.load(fh)["events"]}
        sel = sorted(want & have)
        if not sel:
            continue
        t = pq.read_table(os.path.join(directory, f), filters=[("event", "in", sel)]).to_pandas()
        for e in sel:
            g = t[t.event == e]
            out[e] = np.stack([g.a.to_numpy(np.int64), g.b.to_numpy(np.int64)])
    missing = sorted(want - set(out))
    if missing:
        raise KeyError(f"no union additions for events {missing[:10]}")
    return out
