# Copied unchanged from the verified port (workflow wf_9ddbb9d7-334, 2026-10-11): spec agent + adversarial
# check against the authors' unmodified code on hand-counted and randomised events. See the docstring for commits.
"""Faithful port of the track metrics of Liu et al., "Hierarchical Graph Neural Networks for
Particle Track Reconstruction", arXiv:2303.01640, JPCS 3206 (2026) 012084, Tables 2-3 (Eff, FR).

Reference code (both read at the cloned commits):
  REPO = github.com/ryanliu30/HierarchicalGNN @ a2563afab739efb470770daa13ef634c84e02b50
         Modules/tracking_utils.py:18-83  (eval_metrics; file unchanged since c3c0f9f, 2022-11-05)
  FORK = github.com/ryanliu30/Tracking-ML-Exa.TrkX @ f107781c124e83455e63a70ddaf73d44b8e002a1
         Pipelines/Common_Tracking_Example/LightningModules/TrackML_ACAT/tracking_utils.py:11-69
The two eval_metrics differ in one place only: FORK:15-16 turns majority_cut == 0.5 into 0.5 + 1e-12,
i.e. a strict ">50%" majority on both sides; REPO keeps ">= 50%".
Default variant = "fork": the paper text and the ACAT-2022 slides say "more than 50%", the FORK holds the
TrackML-full configs (2000/100/100 = 2200 events), and Truth-CC on the authors' TrackML-1GeV events
gives Eff 98.72% with the strict rule (Table 3: 98.72%) but >= 99.2% with the REPO ">=" rule
(truthcc_1gev.py).

Callers (all pass primary=False, pt_cut=1.0, nhits_cut=5, majority_cut=0.5):
  REPO Modules/BipartiteClassification/bipartite_classification_base.py:253-267 (BC-HGNN)
  REPO Modules/EdgeClassifier/edge_classifier_base.py:160-181 (EC-GNN / connected components)
  REPO Modules/GNNEmbedding/embedding_base.py:267-285 (E-GNN, E-HGNN / HDBSCAN)
Per-event values are logged with LightningModule.log_dict and averaged by PyTorch Lightning 1.6
over the test events, weighted by the batch size PL infers from the PyG batch (= number of hits of
the input event, see the "batch size we found is 84695" warnings in FORK
notebooks/TrackML_ACAT/training_script.ipynb). aggregate() below reproduces that.

check_liu (adversarial re-check) notes:
  * port_checked.py = port.py + (a) pur_den kept equal to n_cand - n_pairs_bad - n_matched_nonrec on
    the second early exit (REPO L59-60), which affects only pooled FR; (b) optional unassigned_label;
    (c) aggregate(empty='skip') for the E-GNN caller. All per-event (track_eff, track_pur) values are
    identical to port.py and to the authors' eval_metrics.
  * FORK notebooks/TrackML_ACAT/track_eval.ipynb cell 4 (Truth-CC) calls eval_metrics with
    majority_cut=0.9, not 0.5. On the authors' 1GeV events 0.9 gives Truth-CC Eff ~91.5% (test
    split), far from Table 3's 98.72%, while strict 0.5 gives 98.72% exactly. So Table 3 used 0.5
    (strict), and the 0.9 is a later notebook edit (cell 7 output 0.879 / 0.493 is from that run).

Inputs follow our per-event tables:
  hit_pid   int64 [n_hits]  particle id per hit, 0 = noise
  labels    int64 [n_hits]  our track id per hit (every hit has one; singletons/doublets allowed)
  particles DataFrame with particle_id, pt (initial momentum, GeV), nhits, ...
  hit_pt    optional float [n_hits]: hit-level truth pT = hypot(tpx, tpy). The reference defines the
            particle pT as the MIN over the particle's hits of the hit-level pT (REPO L41). If hit_pt
            is None, particles.pt (initial pT) is used as a proxy.
"""
import numpy as np
import pandas as pd

VARIANTS = ("repo", "fork")


def effective_majority(variant, majority_cut=0.5):
    if variant == "repo":                       # REPO L18: used as given
        return majority_cut
    if variant == "fork":                       # FORK L13-16
        if majority_cut < 0.5:
            raise UserWarning("majority_cut < 0.5")
        return majority_cut + 1e-12 if majority_cut == 0.5 else majority_cut
    raise ValueError(f"variant must be one of {VARIANTS}")


def port(hit_pid, labels, particles=None, variant="fork", pt_cut=1.0, nhits_cut=5,
         majority_cut=0.5, hit_pt=None, unassigned_label=None):
    """Return integer counts for one event (pool them over events, or feed a list to aggregate()).

    unassigned_label: optional label value (e.g. -1) meaning "hit not in any track". Such hits are
    left out of the bipartite graph, as HDBSCAN noise is (REPO embedding_base.py:270), but stay in
    the particle rows (nhits, noise count). Default None = every label value is a candidate id,
    which is the input contract (unassigned hits carry unique singleton ids)."""
    hit_pid = np.asarray(hit_pid).astype(np.int64)
    labels = np.asarray(labels).astype(np.int64)
    if hit_pid.shape != labels.shape:
        raise ValueError("hit_pid and labels must have the same length")
    m = effective_majority(variant, majority_cut)
    in_graph = np.ones(len(labels), bool) if unassigned_label is None else labels != unassigned_label

    # ---- particle rows: REPO L34, unique over ALL hits of the evaluated event (noise pid 0 is a row)
    upid, pinv, nhits = np.unique(hit_pid, return_inverse=True, return_counts=True)
    n_rows = len(upid)
    is_noise_row = upid == 0

    # ---- particle pT: REPO L41 min over hits of event.pt; noise pT set to 0 by the caller
    #      (bipartite_classification_base.py:258, edge_classifier_base.py:172, embedding_base.py:276)
    n_pt_missing = 0
    if hit_pt is not None:
        hp = np.asarray(hit_pt, dtype=np.float64).copy()
        hp[hit_pid == 0] = 0.0
        pt = np.full(n_rows, np.inf)
        np.minimum.at(pt, pinv, hp)
    else:
        if particles is None:
            raise ValueError("need particles (or hit_pt) for pT")
        ptab = particles.drop_duplicates("particle_id").set_index("particle_id")["pt"]
        pt = ptab.reindex(upid).to_numpy(dtype=np.float64).copy()
        pt[is_noise_row] = 0.0
        n_pt_missing = int(np.isnan(pt[~is_noise_row]).sum())   # NaN -> fails pt > cut (non-rec)

    n_nhits_mismatch = -1
    if particles is not None and "nhits" in particles.columns:
        tab_nh = (particles.drop_duplicates("particle_id").set_index("particle_id")["nhits"]
                  .reindex(upid[~is_noise_row]).to_numpy())
        n_nhits_mismatch = int((tab_nh != nhits[~is_noise_row]).sum())

    # ---- reconstructable: REPO L62-63  (pt STRICTLY > pt_cut, nhits >= nhits_cut, no eta/primary cut)
    with np.errstate(invalid="ignore"):
        rec_row = (pt > pt_cut) & (nhits >= nhits_cut)
    n_rec = int(rec_row.sum())                     # noise row has pt 0 -> never reconstructable

    # ---- candidates: REPO L29-32, keep label groups with size >= nhits_cut*majority_cut (= 3 hits)
    ulab, linv, lcnt = np.unique(labels[in_graph], return_inverse=True, return_counts=True)
    keep_lab = lcnt >= nhits_cut * m
    n_cand = int(keep_lab.sum())
    newidx = np.full(len(ulab), -1, dtype=np.int64)
    newidx[keep_lab] = np.arange(n_cand)           # ascending relabel by original label value
    keep_hit = keep_lab[linv]
    hc = newidx[linv[keep_hit]]
    hr = pinv[in_graph][keep_hit]

    out = dict(
        n_hits_event=int(len(hit_pid)),
        n_noise_hits=int((hit_pid == 0).sum()),
        n_particles=int((~is_noise_row).sum()),
        n_labels=int(len(ulab)),
        n_labels_dropped_lt3=int(len(ulab) - n_cand),
        n_cand=n_cand,
        n_rec=n_rec,
        n_pairs_raw=0, n_pairs_bad=0, n_pairs_bad_noise=0, n_pairs_bad_small=0,
        n_matched=0, n_matched_rec=0, n_matched_nonrec=0,
        n_cand_matched=0, n_cand_multimatched=0,
        pur_den=n_cand,
        ref_early_exit=1,
        n_pt_missing=n_pt_missing, n_nhits_mismatch=n_nhits_mismatch,
    )
    out.update(_paper_text_counts(upid, nhits, pt, hr, hc, n_cand, nhits_cut))
    if n_cand == 0:   # reference raises on .max() of an empty tensor (L45): BC caller logs zeros,
        #             E-GNN caller logs {} (event skipped), EC caller has no try (crash)
        return out

    # ---- shared-hit matrix C[p, c]: REPO L45 (column sums = candidate sizes incl. noise hits)
    key = hr * n_cand + hc
    uk, kc = np.unique(key, return_counts=True)
    r = uk // n_cand
    c = uk % n_cand
    k = kc.astype(np.float64)
    size = np.bincount(hc, minlength=n_cand).astype(np.float64)

    # ---- matching: REPO L48-49
    #   C >= m*|cand|  and  C >= m*nhits  and  C*h_c == max_c'(C*h_c')   (argmax over candidates,
    #   ties -> highest relabelled index because h = linspace(1, 1+1e-12) increases)
    h = np.linspace(1, 1 + 1e-12, n_cand)
    hashed = k * h[c]
    rowmax = np.zeros(n_rows)
    np.maximum.at(rowmax, r, hashed)
    match = (k >= m * size[c]) & (k >= m * nhits[r]) & (hashed == rowmax[r])
    r, c, k = r[match], c[match], k[match]
    out["n_pairs_raw"] = int(len(r))
    if len(r) == 0:                                 # REPO L52-53 default_response
        return out

    # ---- drop pairs with <= 2 shared hits or matched to noise: REPO L56-57
    good = (k > m * nhits_cut) & (upid[r] != 0)
    out["n_pairs_bad"] = int((~good).sum())
    out["n_pairs_bad_noise"] = int((upid[r] == 0).sum())
    out["n_pairs_bad_small"] = int(((upid[r] != 0) & ~(k > m * nhits_cut)).sum())
    r, c = r[good], c[good]
    # FIX (check_liu): keep the documented identity pur_den = n_cand - n_pairs_bad - n_matched_nonrec
    # on this early-exit path too. port.py left pur_den = n_cand here, so in POOLED mode the dropped
    # pairs' candidates (noise-matched, <=2 shared hits) counted as fakes only in events where every
    # raw pair was dropped. The per-event reference value (0, 0) is unchanged.
    out["pur_den"] = int(n_cand - out["n_pairs_bad"])
    if len(r) == 0:                                 # REPO L59-60 default_response
        return out
    out["ref_early_exit"] = 0

    mask = rec_row[r]                               # REPO L62
    out["n_matched"] = int(len(r))
    out["n_matched_rec"] = int(mask.sum())
    out["n_matched_nonrec"] = int((~mask).sum())
    cu, cc = np.unique(c, return_counts=True)
    out["n_cand_matched"] = int(len(cu))
    out["n_cand_multimatched"] = int((cc > 1).sum())  # only possible with >=50% (exact 50/50 cand)
    # REPO L75 denominator
    out["pur_den"] = int(n_cand - out["n_pairs_bad"] - out["n_matched_nonrec"])
    return out


def _paper_text_counts(upid, nhits, pt, hr, hc, n_cand, nhits_cut):
    """Literal reading of the paper text (p.5), for comparison only: strict >50% on both sides,
    reconstructable = more than 5 hits and pT >= 1 GeV, FR = 1 - N(matched)/N(candidates>=3 hits)."""
    real = upid != 0
    with np.errstate(invalid="ignore"):
        rec_text = real & (nhits > nhits_cut) & (pt >= 1.0)
    res = dict(text_n_rec=int(rec_text.sum()), text_n_matched=0, text_n_matched_rec=0)
    if n_cand == 0:
        return res
    key = hr * n_cand + hc
    uk, kc = np.unique(key, return_counts=True)
    r, c = uk // n_cand, uk % n_cand
    size = np.bincount(hc, minlength=n_cand)
    ok = (2 * kc > size[c]) & (2 * kc > nhits[r]) & real[r]
    res["text_n_matched"] = int(ok.sum())
    res["text_n_matched_rec"] = int((ok & rec_text[r]).sum())
    return res


# Published metric -> counts.  "as published" = PL 1.6 epoch aggregation of per-event ratios.
FORMULAS = {
    "Eff (Tables 2-3, as published)":
        "sum_e w_e*r_e / sum_e w_e,  r_e = n_matched_rec/n_rec (0 if ref_early_exit),  w_e = n_hits_event",
    "FR (Tables 2-3, as published)":
        "1 - sum_e w_e*q_e / sum_e w_e,  q_e = n_matched_rec/pur_den (0 if ref_early_exit),  "
        "pur_den = n_cand - n_pairs_bad - n_matched_nonrec",
    "Eff (pooled counts)": "sum n_matched_rec / sum n_rec",
    "FR (pooled counts)": "1 - sum n_matched_rec / sum pur_den",
    "Eff (paper text, literal; not what the code computes)": "sum text_n_matched_rec / sum text_n_rec",
    "FR (paper text, literal; not what the code computes)": "1 - sum text_n_matched / sum n_cand",
}


def per_event_ratios(cnt):
    """Exactly the per-event track_eff / track_pur the reference returns (REPO L73, L75, L52-60)."""
    if cnt["ref_early_exit"]:
        return 0.0, 0.0
    with np.errstate(all="ignore"):              # same 0/0 -> nan, x/0 -> inf as the reference
        eff = float(np.float64(cnt["n_matched_rec"]) / np.float64(cnt["n_rec"]))
        pur = float(np.float64(cnt["n_matched_rec"]) / np.float64(cnt["pur_den"]))
    return eff, pur


def aggregate(counts_list, mode="pl_weighted", empty="zero"):
    """mode: 'pl_weighted' (as published), 'event_mean', or 'pooled'. Returns dict(eff, fr).
    empty: what an event with no >=3-hit candidate contributes to the per-event modes (the reference
    raises there). 'zero' = BC-HGNN caller (bipartite_classification_base.py:261-274 logs zeros);
    'skip' = E-GNN/E-HGNN caller (embedding_base.py:279-287 logs {} so PL never sees the event)."""
    if empty == "skip" and mode != "pooled":
        counts_list = [c for c in counts_list if c["n_cand"] > 0]
    if mode == "pooled":
        num = sum(c["n_matched_rec"] for c in counts_list)
        return dict(eff=num / sum(c["n_rec"] for c in counts_list),
                    fr=1 - num / sum(c["pur_den"] for c in counts_list))
    ratios = np.array([per_event_ratios(c) for c in counts_list], dtype=float)
    if mode == "event_mean":
        w = np.ones(len(counts_list))
    elif mode == "pl_weighted":
        w = np.array([c["n_hits_event"] for c in counts_list], dtype=float)
    else:
        raise ValueError(mode)
    eff = float((w * ratios[:, 0]).sum() / w.sum())
    pur = float((w * ratios[:, 1]).sum() / w.sum())
    return dict(eff=eff, fr=1 - pur)
