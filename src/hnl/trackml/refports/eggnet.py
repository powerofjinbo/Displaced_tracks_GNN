# Copied unchanged from the verified port (workflow wf_9ddbb9d7-334, 2026-10-11): spec agent + adversarial
# check against the authors' unmodified code on hand-counted and randomised events. See the docstring for commits.
"""Faithful numpy port of the EggNet track-level evaluation.

Reference code (github.com/exatrkx/EggNet @ e2bd117c653ba527635906c527b97f865b39f1c0):
  eggnet/utils/cluster.py:23-112      cluster_and_match  (candidate tracks, matching, counts)
  eggnet/core/eval_stage.py:30-87     pooled sums over events, eff/dup/fak formulas
  eggnet/utils/mapping.py:77-141      target-particle mask (inclusive bounds, pid != 0)
  eggnet/datasets/utils/utils.py:129-202  hard cuts (pT>1 GeV model) + nhits recount
  configs/trackML/eval_trackml.yaml   target_tracks: pt [1, inf], nhits [3, inf]; eps 0.1
The same metric code (identical logic) is in the acorn branch used before the EggNet repo existed:
  gitlab.cern.ch/gnn4itkteam/acorn @ a603207 (jay_gnn_metric_learning),
  acorn/stages/node_encoding/node_encoding_stage.py:735-863.

Inputs per event
  hit_pid   int64 [N]  particle id per hit, 0 = noise
  labels    int64 [N]  track id per hit (ours: every hit has an id; singletons allowed)
  particles DataFrame with particle_id, pt [GeV], nhits (hits of the particle in the full event)

Output: dict of integer counts (pool by summing over events, then apply FORMULAS).

port_checked.py = port.py plus two changes found in the adversarial check (counts are otherwise identical):
  1. hard_cut_min_nodes (default 20 = utils.py:152 default; no EggNet/acorn TrackML config sets min_nodes):
     if fewer hits than this survive the hard cut, handle_hard_node_cuts returns UNDER without
     masking (utils.py:81-82), the binary-search loop (utils.py:155-197) never succeeds because no
     phi cut is configured, and the event is evaluated UNCUT (noise and low-pT hits kept), with
     hit_particle_nhits still recounted over the uncut event (utils.py:199-202). port.py always cut.
     Irrelevant for real TrackML pT>1 events (~1e4 hits); matters only for tiny events.
  2. Every non-zero hit_pid must have a row in `particles` (true by construction in the authors'
     feature store, a left merge with particles.csv). port.py silently gave such hits NaN pt, which
     makes them unmatchable and non-target; port_checked raises instead.
"""
import numpy as np
import pandas as pd

COUNT_KEYS = (
    "n_particles",                 # target particles (pid!=0, pt>=1, nhits>=3) with >=1 hit in the event
    "n_matched_particles",         # distinct particle ids (ANY particle) that some track matches
    "n_matched_tracks",            # tracks with a >50% majority particle (any particle)
    "n_matched_target_particles",  # distinct target particles matched by >=1 track
    "n_matched_target_tracks",     # tracks whose >50% majority particle is a target particle
    "n_tracks",                    # all candidate tracks (DBSCAN label >= 0)
)

# Published metrics = these formulas on counts pooled (summed) over all evaluated events
# (eval_stage.py:66 sums, eval_stage.py:83-87 ratios).
FORMULAS = {
    "efficiency": "n_matched_target_particles / n_particles",
    "duplication_rate": "(n_matched_target_tracks - n_matched_target_particles) / n_matched_target_particles",
    "fake_rate": "(n_tracks - n_matched_tracks) / n_matched_particles",
}


def metrics(counts):
    """Apply the EggNet formulas (eval_stage.py:83-87) to (pooled) counts."""
    c = counts
    return {
        "efficiency": c["n_matched_target_particles"] / c["n_particles"],
        "duplication_rate": (c["n_matched_target_tracks"] - c["n_matched_target_particles"])
        / c["n_matched_target_particles"],
        "fake_rate": (c["n_tracks"] - c["n_matched_tracks"]) / c["n_matched_particles"],
    }


def pool(count_dicts):
    """Sum per-event counts (eval_stage.py:66)."""
    out = {k: 0 for k in COUNT_KEYS}
    for d in count_dicts:
        for k in COUNT_KEYS:
            out[k] += int(d[k])
    return out


def _f32(x):
    # mapping.py:89-102 compares event[key].float() (float32) against the bounds
    return np.asarray(x, dtype=np.float64).astype(np.float32)


def port(hit_pid, labels, particles, *, pt_min=1.0, nhits_min=3, min_track_hits=3,
         hard_cut_pt=None, noise_pt_nan=True, hard_cut_min_nodes=20):
    """EggNet cluster_and_match counts for one event.

    pt_min, nhits_min : target_tracks of configs/trackML/eval_trackml.yaml (inclusive, >=).
    min_track_hits    : stand-in for DBSCAN(min_samples=3) (cluster.py:61). Our groups with fewer
                        hits get label -1, i.e. they are treated as DBSCAN-unclustered hits and are
                        not candidates. Use 1 to feed our labels literally (every group is a track).
    hard_cut_pt       : None = full event (eggnet_trackml_full.yaml, arXiv:2506.03415).
                        1.0  = emulate hard_cuts hit_particle_pt [1, inf] (eggnet_trackml.yaml,
                        arXiv:2407.13925 Table 1): drop noise and hits of pt<1 particles before
                        evaluation, recount nhits in the cut event, then apply min_track_hits.
    noise_pt_nan      : True = noise hits carry hit_particle_pt = NaN (acorn left-merge with
                        particles.csv); torch.unique then puts every noise hit in its own group, so
                        a noise-majority track is never matched (counts as fake). False = finite
                        noise pt; a noise-majority track then matches "particle 0".
    hard_cut_min_nodes: only with hard_cut_pt. If fewer hits pass the cut, the authors' event stays
                        uncut (utils.py:81-82, :152-197) but nhits is still recounted (:199-202).
                        Default 20 = authors' default; 0 = always cut (port.py behaviour).
    """
    hit_pid = np.asarray(hit_pid, dtype=np.int64).copy()
    labels = np.asarray(labels, dtype=np.int64).copy()
    if hit_pid.shape != labels.shape:
        raise ValueError("hit_pid and labels must have the same shape")

    # hit_particle_pt / hit_particle_nhits: per-particle values broadcast to hits (left merge).
    p = particles.drop_duplicates("particle_id").set_index("particle_id")
    p.index = p.index.astype(np.int64)
    noise = hit_pid == 0
    missing = ~noise & ~np.isin(hit_pid, p.index.to_numpy())
    if missing.any():
        raise ValueError(f"{int(missing.sum())} hits have a non-zero pid absent from `particles` "
                         f"(e.g. {np.unique(hit_pid[missing])[:5].tolist()}); the authors' store "
                         "always has a particle row for every non-noise hit")
    pt_hit = p["pt"].reindex(hit_pid).to_numpy(dtype=np.float64, copy=True)
    nhits_hit = p["nhits"].reindex(hit_pid).to_numpy(dtype=np.float64, copy=True)
    pt_hit[noise] = np.nan if noise_pt_nan else 0.0
    nhits_hit[noise] = -1.0  # acorn data_reading_stage: particle_nhits = -1 for noise

    if hard_cut_pt is not None:
        # utils.py:129-197 -> mapping.py:100-102: keep hits with hard_cut_pt <= pt.float() <= inf
        with np.errstate(invalid="ignore"):
            keep = _f32(pt_hit) >= np.float32(hard_cut_pt)
        if keep.sum() < hard_cut_min_nodes:
            # utils.py:81-82 returns UNDER without masking; the loop exits with the event uncut
            keep = np.ones_like(keep)
        hit_pid, labels, pt_hit = hit_pid[keep], labels[keep], pt_hit[keep]
        noise = noise[keep]
        # utils.py:199-202: hit_particle_nhits recounted in the (cut or uncut) event
        _, inv, cnt = np.unique(hit_pid, return_inverse=True, return_counts=True)
        nhits_hit = cnt[inv].astype(np.float64)

    n_hits = hit_pid.size
    out = {k: 0 for k in COUNT_KEYS}
    if n_hits == 0:
        return out

    # Candidate tracks: DBSCAN labels >= 0 (cluster.py:61, :100). Groups below min_track_hits -> -1.
    _, inv, cnt = np.unique(labels, return_inverse=True, return_counts=True)
    lab = labels.copy()
    lab[(cnt[inv] < min_track_hits) | (labels < 0)] = -1
    # hit_track_length (cluster.py:62-65), recomputed on the candidate labels
    _, inv, cnt = np.unique(lab, return_inverse=True, return_counts=True)
    track_len = cnt[inv]

    # Target hits (mapping.py:128-141): pid != 0 and pt, nhits within inclusive float32 bounds
    with np.errstate(invalid="ignore"):
        target = (hit_pid != 0) & (_f32(pt_hit) >= np.float32(pt_min)) & (_f32(nhits_hit) >= np.float32(nhits_min))

    # n_particles: unique (pid, pt) over target hits (cluster.py:40-43, :95)
    out["n_particles"] = int(np.unique(hit_pid[target]).size)

    # Group key of torch.unique(hit_track_info, dim=1) (cluster.py:66-78):
    # (label, pid, track_length, pt). length is a function of label and pt of pid, so the key is
    # (label, pid) -- except NaN pt, where every hit is its own group (NaN != NaN in torch.unique).
    nan_pt = np.isnan(pt_hit)
    hit_index = np.arange(n_hits, dtype=np.int64)
    df = pd.DataFrame({
        "lab": lab,
        "pid": hit_pid,
        "len": track_len,
        "nan_uid": np.where(nan_pt, hit_index, -1),
        "target": target,
    })

    def _matched(frame):
        g = frame.groupby(["lab", "pid", "len", "nan_uid"], sort=False).size().reset_index(name="n")
        sel = (g["lab"] >= 0) & (g["n"] / g["len"] > 0.5)  # strict > 0.5 of the TRACK's hits
        return g.loc[sel, "pid"].to_numpy()

    matched_pid = _matched(df)                        # cluster.py:76-81
    matched_target_pid = _matched(df[df["target"]])   # cluster.py:82-92 (length = full track length)

    out["n_matched_particles"] = int(np.unique(matched_pid).size)          # cluster.py:96
    out["n_matched_tracks"] = int(matched_pid.size)                        # cluster.py:97
    out["n_matched_target_particles"] = int(np.unique(matched_target_pid).size)  # cluster.py:93, :98
    out["n_matched_target_tracks"] = int(matched_target_pid.size)          # cluster.py:99
    out["n_tracks"] = int(np.unique(lab[lab >= 0]).size)                   # cluster.py:100
    return out
