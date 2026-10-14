"""Dense-jet sanity check of the displaced ttbar targets of paper Sec. IX (NUMBERS TS16-TS18), validation only.

Population: the ttbar validation targets (docs/definitions.md section 4) with 1 <= |d0| < 100 mm ("displaced"),
against the ttbar targets with |d0| < 1 mm ("prompt") of the same events. Model and working point as
scripts/diagnostics/ttbar_loss.py: Canto-L (ttbar) seed 0 at the frozen LM working point, classifier stage at s0 = 0.2.

1. Shared and merged hits. Every hit row carries one particle id. Counted: hits of two particles at the same
   coordinates; hits of one particle twice on one module; for each target hit, the distance to the nearest hit of
   another particle on the same module (the same ACTS geometry id, i.e. the same sensitive surface).
2. Truth labels of hits on shared modules (label-exchange test). For every hit, a reference line through two other
   hits of its particle in the transverse plane: the nearest hits before and after it along the chain of true edges
   that are not on its module, or the two nearest on one side at either end of the chain. For two hits i (particle A)
   and j (particle B) on one module, with d_A, d_B the transverse distances to the reference lines of A and B,
   cur = d_A(i) + d_B(j) and swp = d_A(j) + d_B(i); the labels are exchanged when swp < 0.5 cur and
   cur - swp > 1 mm. The test is blind below about 2 mm of separation. Control: the same test for hits of two
   particles on different modules of one layer, at most 50 mm apart. A hit of an exchanged pair is "exchanged".
   d0_clean: |d0| of the truth-hit fit (docs/definitions.md section 3) to the target's hits that are not exchanged;
   d0_all_fit: the same fit to all its hits (reproduces the stored d0).
3. Graph and classifier stages per target (hnl.eval.oblique_val.classify, the definition behind N:TS16-TS18 and the
   graph-stage ceiling) and per true edge: in the graph, score above s0, and whether an end of the edge is an
   exchanged hit.
Controls: the same tests and stages for the HNL daughters and W muons of the HNL val events.

    python -m hnl.eval.dense_jet_check --out results/dense_jet_check_v1 [--workers N] [--limit N]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import shutil
import tarfile
import tempfile

import numpy as np
import pandas as pd

RUNS = "/ceph/submit/data/user/j/jinboz1/hnl_lrt/runs"
RUN = f"{RUNS}/stage1_final_learned/l1tl/armi_s0"
WP = "mwslmerge 0.05 (trackbuild_val_lmerge)"
S0 = 0.2
DERIVED_EV = os.path.join(os.environ.get("HNL_DATA", "/ceph/submit/data/user/j/jinboz1/hnl_lrt"), "derived", "events",
                          "v1")
CLOSE = (0.1, 0.5, 1.0, 2.0)
SEP_EDGES = [0, 1, 2, 5, 10, 20, 50, 1e5]
PT_EDGES = [1, 2, 5, 10, 1e9]
DENS_EDGES = [-1, 0, 1, 3, 100]
D0_EDGES = [0, 0.1, 1, 3, 10, 30, 100, 1e9]


def chain_orders(pid: np.ndarray, src: np.ndarray, dst: np.ndarray) -> dict:
    """Row order head to tail of every particle (each particle's hits form one chain)."""
    nxt = dict(zip(src.tolist(), dst.tolist()))
    has_prev = set(dst.tolist())
    out = {}
    for p in np.unique(pid):
        rows = np.flatnonzero(pid == p)
        heads = [r for r in rows.tolist() if r not in has_prev]
        if len(heads) != 1:
            raise ValueError(f"particle {p}: {len(heads)} chain heads")
        order, c = [], heads[0]
        while c is not None:
            order.append(c)
            c = nxt.get(c)
        if len(order) != len(rows):
            raise ValueError(f"particle {p}: chain covers {len(order)} of {len(rows)} hits")
        out[int(p)] = np.asarray(order, dtype=np.int64)
    return out


def reference_lines(geoid: np.ndarray, orders: dict) -> tuple[np.ndarray, np.ndarray]:
    """Two other hits of the same particle, not on the hit's module, that define its reference line (-1 if none)."""
    n = len(geoid)
    p1, p2 = np.full(n, -1, np.int64), np.full(n, -1, np.int64)
    for o in orders.values():
        if len(o) < 3:
            continue
        g = geoid[o]
        for k, i in enumerate(o):
            prev = [o[m] for m in range(k - 1, -1, -1) if g[m] != g[k]]
            nxt = [o[m] for m in range(k + 1, len(o)) if g[m] != g[k]]
            if prev and nxt:
                p1[i], p2[i] = prev[0], nxt[0]
            elif len(nxt) >= 2:
                p1[i], p2[i] = nxt[0], nxt[1]
            elif len(prev) >= 2:
                p1[i], p2[i] = prev[0], prev[1]
    return p1, p2


def dline(x, y, a, b, h):
    """Transverse distance of hits h from the lines through hits a and b (arrays)."""
    ux, uy = x[b] - x[a], y[b] - y[a]
    return np.abs(ux * (y[h] - y[a]) - uy * (x[h] - x[a])) / np.maximum(np.hypot(ux, uy), 1e-9)


def pair_test(x, y, pid, p1, p2, i, j):
    ok = (pid[i] != pid[j]) & (p1[i] >= 0) & (p1[j] >= 0)
    i, j = i[ok], j[ok]
    cur = dline(x, y, p1[i], p2[i], i) + dline(x, y, p1[j], p2[j], j)
    swp = dline(x, y, p1[i], p2[i], j) + dline(x, y, p1[j], p2[j], i)
    sep = np.hypot(x[i] - x[j], y[i] - y[j])
    return i, j, sep, (swp < 0.5 * cur) & (cur - swp > 1.0)


def group_pairs(key: np.ndarray):
    """All index pairs (i < j) of rows with equal key."""
    o = np.argsort(key, kind="stable")
    k = key[o]
    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    ends = np.r_[starts[1:], len(k)]
    ii, jj = [], []
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        a, b = np.triu_indices(e - s, 1)
        ii.append(o[s + a])
        jj.append(o[s + b])
    if not ii:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return np.concatenate(ii), np.concatenate(jj)


def refit_d0(x, y, z, vol, lists, q, pt, pdg):
    """|d0| of the truth-hit fit of docs/definitions.md section 3 (hnl.truth.fit_events: Huber fits, fixed R for
    muons, free curvature otherwise, field_sign -1) on the given row lists (chain order); NaN below 3 hits."""
    from hnl.truth import helix as hx
    from hnl.truth.fit_events import _pad_like, hit_scales
    out = np.full(len(lists), np.nan)
    k = [i for i, r in enumerate(lists) if len(r) >= 3]
    if not k:
        return out
    rows = np.concatenate([lists[i] for i in k])
    toh = np.concatenate([np.full(len(lists[i]), j) for j, i in enumerate(k)])
    p = hx.pad_tracks(toh, np.c_[x[rows], y[rows], z[rows]], rows=rows)
    q, pt, pdg = np.asarray(q)[k], np.asarray(pt)[k], np.asarray(pdg)[k]
    R0 = hx.MM_PER_GEV * np.maximum(pt, 1e-6)
    sigma = (-1 * np.sign(q)).astype(np.int64)
    sigma[sigma == 0] = 1
    scale_t = _pad_like(p, hit_scales(vol)[0])
    f0 = hx.fit_circle(p, R0, sigma, scale=scale_t)
    pg0 = hx.perigee(f0.xc, f0.yc, f0.R, f0.sigma)
    k0 = sigma / R0
    fix = hx.fit_perigee(p, pg0.d0, pg0.phi0, k0, free_kappa=False, scale=scale_t)
    free = hx.fit_perigee(p, fix.d0, fix.phi0, k0, free_kappa=True, scale=scale_t)
    out[k] = np.abs(np.where(np.abs(pdg) == 13, fix.d0, free.d0))
    return out


def neighbours(t: pd.DataFrame, nb: pd.DataFrame, dr: float = 0.05) -> np.ndarray:
    """Charged particles with pT > 0.5 GeV and >= 3 hits within dR of each target (table eta, phi), as
    scripts/diagnostics/ttbar_dense.py."""
    out = np.zeros(len(t), dtype=int)
    g = dict(tuple(nb.groupby("event_id")))
    for i, (e, pid, eta, phi) in enumerate(zip(t.event_id, t.particle_id, t.eta, t.phi)):
        o = g.get(e)
        if o is None:
            continue
        dphi = np.angle(np.exp(1j * (o.phi.to_numpy() - phi)))
        r = np.hypot(o.eta.to_numpy() - eta, dphi)
        out[i] = int(((r < dr) & (o.particle_id.to_numpy() != pid)).sum())
    return out


def event_hits(sample: str, events: np.ndarray):
    from hnl.data.convert import read_table
    h = read_table(DERIVED_EV, "hits", sample, events=events,
                   columns=["event", "row", "hid", "x", "y", "z", "volume", "layer", "geoid", "pid"])
    e = read_table(DERIVED_EV, "edges", sample, events=events)
    return dict(tuple(h.groupby("event"))), dict(tuple(e.groupby("event")))


def work_event(task: dict) -> dict:
    h, e, tg = task["hits"].sort_values("row"), task["edges"], task["targets"]
    n = len(h)
    if not (h.row.to_numpy() == np.arange(n)).all():
        raise ValueError(f"event {task['event']}: rows not 0..n-1")
    if h.hid.nunique() != n or (h.pid.to_numpy() == 0).any():
        raise ValueError(f"event {task['event']}: hid not unique or a hit without particle")
    x, y, z = h.x.to_numpy(), h.y.to_numpy(), h.z.to_numpy()
    pid, geo = h.pid.to_numpy(), h.geoid.to_numpy()
    vl = h.volume.to_numpy().astype(np.int64) * 1000 + h.layer.to_numpy()
    orders = chain_orders(pid, e.src.to_numpy(), e.dst.to_numpy())
    p1, p2 = reference_lines(geo, orders)
    # 1. same module: identical positions, nearest other-particle hit, repeated (particle, module)
    i, j = group_pairs(geo)
    other = pid[i] != pid[j]
    d3 = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 + (z[i] - z[j]) ** 2)
    dmin = np.full(n, np.inf)
    np.minimum.at(dmin, i[other], d3[other])
    np.minimum.at(dmin, j[other], d3[other])
    n_same = int((other & (d3 < 1e-4)).sum())
    rep = int((~other).sum())
    shared = np.zeros(n, bool)
    shared[i[other]] = True
    shared[j[other]] = True
    # 2. label-exchange test on shared modules, and the different-module control
    si, sj, ssep, sx = pair_test(x, y, pid, p1, p2, i, j)
    exch = np.zeros(n, bool)
    exch[si[sx]] = True
    exch[sj[sx]] = True
    ci, cj = group_pairs(vl)
    keep = (geo[ci] != geo[cj]) & (np.hypot(x[ci] - x[cj], y[ci] - y[cj]) <= 50.0)
    _, _, csep, cx = pair_test(x, y, pid, p1, p2, ci[keep], cj[keep])
    bins = np.digitize(ssep, SEP_EDGES) - 1
    cbins = np.digitize(csep, SEP_EDGES) - 1
    nb_ = len(SEP_EDGES) - 1
    pairs = {"same_n": np.bincount(bins, minlength=nb_)[:nb_], "same_x": np.bincount(bins[sx], minlength=nb_)[:nb_],
             "ctrl_n": np.bincount(cbins, minlength=nb_)[:nb_], "ctrl_x": np.bincount(cbins[cx], minlength=nb_)[:nb_]}
    rec = []
    for t in tg.itertuples():
        o = orders[int(t.particle_id)]
        r = {"event_id": t.event_id, "particle_id": t.particle_id, "n_shared": int(shared[o].sum()),
             "n_exch": int(exch[o].sum()), "n_exch_inner": int(exch[o[:3]].sum()),
             "dmin_mm": float(dmin[o].min())}
        for dc in CLOSE:
            r[f"n_close_{dc}"] = int((dmin[o] < dc).sum())
        rec.append(r)
    vol = h.volume.to_numpy()
    lists = [orders[int(p)] for p in tg.particle_id]
    d_all = refit_d0(x, y, z, vol, lists, tg.q, tg.pt, tg.pdg)
    d_cl = refit_d0(x, y, z, vol, [o[~exch[o]] for o in lists], tg.q, tg.pt, tg.pdg)
    for r, a_, c_ in zip(rec, d_all, d_cl):
        r["d0_all_fit"], r["d0_clean"] = a_, c_
    out = {"event": task["event"], "rows": rec, "n_hits": n, "n_same": n_same, "rep": rep, "pairs": pairs,
           "n_shared_hits": int(shared.sum()), "n_exch_hits": int(exch.sum()), "edges": None}
    path = task.get("graph")
    if path is None:
        return out
    from acorn.utils.loading_utils import load_pyg
    from hnl.eval.oblique_val import classify
    from hnl.eval.trackbuild_val import load_scored
    from hnl.graphs import ceiling as C
    cl = pd.DataFrame(classify(path, tg.particle_id.to_numpy(np.int64), S0)).set_index("particle_id")
    for r in rec:
        s = cl.loc[r["particle_id"]]
        r.update(stage=s.stage, n_true_chain=int(s.n_true_chain), n_chain_in_graph=int(s.n_chain_in_graph),
                 n_chain_above_cut=int(s.n_chain_above_cut))
    g = load_pyg(path[:-3], weights_only=False)
    if g.num_nodes != n or not (g.hit_particle_id.numpy() == pid).all():
        raise ValueError(f"event {task['event']}: graph nodes do not match the hit rows")
    if np.abs(g.z.numpy() - z).max() > 1e-3:
        raise ValueError(f"event {task['event']}: graph z differs from the hit table")
    _, _, _, _, _, _, a, b, sc = load_scored(path)
    code = a.astype(np.int64) * n + b
    so = np.argsort(code)
    code, sc = code[so], sc[so]
    te = C.unordered_unique(g.track_edges.numpy())
    ce = te[0] * n + te[1]
    pos = np.clip(np.searchsorted(code, ce), 0, max(len(code) - 1, 0))
    ing = code[pos] == ce if len(code) else np.zeros(len(ce), bool)
    above = ing & (sc[pos] > S0)
    epid = pid[te[0]]
    k = np.isin(epid, tg.particle_id.to_numpy())
    out["edges"] = pd.DataFrame({"event_id": task["event_id"], "particle_id": epid[k], "in_graph": ing[k],
                                 "above": above[k], "exch_edge": exch[te[0][k]] | exch[te[1][k]],
                                 "shared_edge": shared[te[0][k]] | shared[te[1][k]]})
    return out


def run_sample(sample: str, events: np.ndarray, targets: pd.DataFrame, graphs: dict | None, workers: int):
    hits, edges = event_hits(sample, events)
    tasks = []
    for ev in events:
        ev = int(ev)
        tg = targets[targets.event == ev]
        if not len(tg):
            continue
        tasks.append({"event": ev, "event_id": int(tg.event_id.iloc[0]), "hits": hits[ev], "edges": edges[ev],
                      "targets": tg[["event_id", "particle_id", "q", "pt", "pdg"]],
                      "graph": None if graphs is None else graphs[int(tg.event_id.iloc[0])]})
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(work_event, tasks, chunksize=2))
    rows = pd.DataFrame([r for o in res for r in o["rows"]])
    ev = pd.DataFrame([{"event": o["event"], "n_hits": o["n_hits"], "n_same": o["n_same"], "rep": o["rep"],
                        "n_shared_hits": o["n_shared_hits"], "n_exch_hits": o["n_exch_hits"]} for o in res])
    pairs = {k: sum(o["pairs"][k] for o in res) for k in res[0]["pairs"]}
    ed = [o["edges"] for o in res if o["edges"] is not None]
    return rows, ev, pairs, (pd.concat(ed, ignore_index=True) if ed else None)


def extract_graphs(run: str, event_ids: np.ndarray, tmp: str) -> dict:
    want = {f"event{int(e)}.pyg.gz": int(e) for e in event_ids}
    got = {}
    with tarfile.open(os.path.join(run, "gnn_scores", "valset.tar")) as tar:
        for m in tar:
            nm = os.path.basename(m.name)
            if nm in want:
                m.name = nm
                tar.extract(m, tmp, filter="data")
                got[want[nm]] = os.path.join(tmp, nm)
    if len(got) != len(want):
        raise SystemExit(f"{run}: {len(got)} of {len(want)} val graphs found")
    return got


def truth(sample_sets: dict) -> pd.DataFrame:
    from hnl.data.convert import read_table
    from hnl.eval import bootstrap_tables as BT
    from hnl.eval.particles import particle_table
    p = particle_table(BT.expected_events(sample_sets))
    extra = []
    for s in p["sample"].unique():
        e = np.unique(p.loc[p["sample"] == s, "event"])
        x = read_table(DERIVED_EV, "particles", s, events=e, columns=["event", "particle_id", "vsec", "gen", "process"])
        x["sample"] = s
        extra.append(x)
    p = p.merge(pd.concat(extra), on=["sample", "event", "particle_id"], how="left")
    p["phi"] = np.arctan2(p.py, p.px)
    p["ad0"] = p.d0.abs()
    return p


def pct(k, n):
    k, n = float(k), float(n)
    if n == 0:
        return "-"
    f = k / n
    return f"{100 * f:.1f} +- {100 * np.sqrt(f * (1 - f) / n):.1f} % ({int(k)}/{int(n)})"


def summarize(d, edges, ev, pairs) -> str:
    tt = d[d["sample"] == "ttbar"]
    disp, prm = tt[tt.pop_ == "displaced"], tt[tt.pop_ == "prompt"]
    hn = d[d["sample"] != "ttbar"]
    L = ["# Dense-jet sanity check (ttbar validation; hnl.eval.dense_jet_check)", "",
         f"Canto-L (ttbar) seed 0, frozen working point {WP}; classifier stage at s0 = {S0}. Uncertainties: "
         "binomial, per particle, hit pair or edge (particles of one jet are not independent).", "",
         "## 1. Shared and merged hits", ""]
    for s, e in ev.items():
        L.append(f"- {s}: {int(e.n_hits.sum())} hits in {len(e)} val events; every hit carries one particle id, none "
                 f"without a particle. Hits of two particles at the same coordinates (< 1e-4 mm): "
                 f"{int(e.n_same.sum())}. A particle with two hits on one module: {int(e.rep.sum())} pairs. Hits on a "
                 f"module that also holds a hit of another particle: {pct(e.n_shared_hits.sum(), e.n_hits.sum())}.")
    L += ["", "Target hits with a hit of another particle on the same module within d (3D):", "",
          "| population | hits | " + " | ".join(f"< {c} mm" for c in CLOSE) + " |", "|---|---|" + "---|" * len(CLOSE)]

    def close_row(lab, s):
        nh = s.n_hits.sum()
        return f"| {lab} | {int(nh)} | " + " | ".join(pct(s[f'n_close_{c}'].sum(), nh) for c in CLOSE) + " |"
    for lab, s in [("ttbar prompt", prm), ("ttbar displaced", disp), ("ttbar displaced, found", disp[disp.dm_matched]),
                   ("ttbar displaced, lost at the classifier", disp[disp.where_ == "classifier"]),
                   ("HNL daughters", hn[hn.category == "hnl"]), ("W muons (HNL events)", hn[hn.category == "w"])]:
        L.append(close_row(lab, s))
    for pb in disp.pt_bin.cat.categories:
        for db in disp.dens.cat.categories:
            s = disp[(disp.pt_bin == pb) & (disp.dens == db)]
            if len(s) >= 20:
                L.append(close_row(f"ttbar displaced, pT {pb}, neighbours {db}", s))
    L += ["", "## 2. Exchanged labels", "",
          "Pairs of hits of two particles: fraction whose labels are exchanged, by transverse separation.", "",
          "| sample | pairs | " + " | ".join(f"{a}-{b} mm" for a, b in zip(SEP_EDGES[:-1], SEP_EDGES[1:])) + " |",
          "|---|---|" + "---|" * (len(SEP_EDGES) - 1)]
    for s, p in pairs.items():
        L.append(f"| {s}, same module | {int(p['same_n'].sum())} | " +
                 " | ".join(pct(a, b) for a, b in zip(p["same_x"], p["same_n"])) + " |")
        L.append(f"| {s}, different modules of one layer (control) | {int(p['ctrl_n'].sum())} | " +
                 " | ".join(pct(a, b) for a, b in zip(p["ctrl_x"], p["ctrl_n"])) + " |")
    L += ["", "Targets with at least one exchanged hit; exchanged hits per hit; and |d0| of the truth-hit fit without "
          "the exchanged hits (d0_clean) below 1 mm:", "",
          "| population | targets | with an exchanged hit | exchanged hits / hits | d0_clean < 1 mm |",
          "|---|---|---|---|---|"]

    def exch_row(lab, s):
        return (f"| {lab} | {len(s)} | {pct((s.n_exch > 0).sum(), len(s))} | {pct(s.n_exch.sum(), s.n_hits.sum())} | "
                f"{pct((s.d0_clean < 1).sum(), s.d0_clean.notna().sum())} |")
    for b in tt.d0_bin_.cat.categories:
        L.append(exch_row(f"ttbar targets, |d0| {b} mm", tt[(tt.d0_bin_ == b) & (tt.category != "w")]))
    L.append(exch_row("ttbar W muons", tt[tt.category == "w"]))
    L.append(exch_row("HNL daughters, |d0| 1-100 mm", hn[(hn.category == "hnl") & (hn.ad0 >= 1) & (hn.ad0 < 100)]))
    L.append(exch_row("HNL daughters, all", hn[hn.category == "hnl"]))
    L.append(exch_row("W muons (HNL events)", hn[hn.category == "w"]))
    for lab, s in [("ttbar displaced, generator particles (gen 0)", disp[disp.gen == 0]),
                   ("ttbar displaced, simulation secondaries (gen >= 1, conversions)", disp[disp.gen >= 1])]:
        L.append(exch_row(lab, s))
    ok = tt[tt.d0_all_fit.notna()]
    L += ["", f"Check of the refit: on all {len(ok)} ttbar targets the fit to all hits reproduces the stored |d0| "
          f"within 1e-3 mm for {pct((np.abs(ok.d0_all_fit - ok.ad0) < 1e-3).sum(), len(ok))}.", ""]
    L += ["Displaced ttbar targets by pT and charged neighbours within dR < 0.05: fraction with an exchanged hit; "
          "efficiency and classifier-stage loss without and with one:", "",
          "| pT (GeV) | neighbours | targets | with an exchanged hit | eff., none | eff., exchanged | clf. loss, none | "
          "clf. loss, exchanged |", "|---|---|---|---|---|---|---|---|"]

    def eff_rows(s0_, lab_pt, lab_d):
        a, b = s0_[s0_.n_exch == 0], s0_[s0_.n_exch > 0]
        return (f"| {lab_pt} | {lab_d} | {len(s0_)} | {pct(len(b), len(s0_))} | {pct(a.dm_matched.sum(), len(a))} | "
                f"{pct(b.dm_matched.sum(), len(b))} | {pct((a.where_ == 'classifier').sum(), len(a))} | "
                f"{pct((b.where_ == 'classifier').sum(), len(b))} |")
    for popn, s_ in [("displaced", disp), ("prompt", prm)]:
        if popn == "prompt":
            L += ["", "Same for the prompt ttbar targets (|d0| < 1 mm):", "",
                  "| pT (GeV) | neighbours | targets | with an exchanged hit | eff., none | eff., exchanged | "
                  "clf. loss, none | clf. loss, exchanged |", "|---|---|---|---|---|---|---|---|"]
        for pb in s_.pt_bin.cat.categories:
            for db in list(s_.dens.cat.categories) + ["all"]:
                s = s_[s_.pt_bin == pb] if db == "all" else s_[(s_.pt_bin == pb) & (s_.dens == db)]
                if len(s) >= 10:
                    L.append(eff_rows(s, pb, db))
        for db in s_.dens.cat.categories:
            s = s_[s_.dens == db]
            if len(s) >= 10:
                L.append(eff_rows(s, "all", db))
        L.append(eff_rows(s_, "all", "all"))
    L += ["", "## 3. Graph-level ceiling and edge efficiency", "",
          "Stage of each target. Graph: the connected components of its true edges in the graph do not double-majority "
          "match it (1 - this is the graph-level ceiling); classifier: those above s0 do not; builder: they do, but the "
          "working point does not.", "",
          "| population | targets | graph ceiling | lost at classifier | lost in builder | found |",
          "|---|---|---|---|---|---|"]

    def stage_row(lab, s):
        n = len(s)
        return (f"| {lab} | {n} | {pct(n - (s.where_ == 'graph').sum(), n)} | "
                f"{pct((s.where_ == 'classifier').sum(), n)} | {pct((s.where_ == 'builder').sum(), n)} | "
                f"{pct(s.dm_matched.sum(), n)} |")
    for lab, s in [("prompt", prm), ("displaced", disp), ("displaced, no exchanged hit", disp[disp.n_exch == 0]),
                   ("displaced, exchanged hit", disp[disp.n_exch > 0]),
                   ("displaced, d0_clean >= 1 mm", disp[disp.d0_clean >= 1])]:
        L.append(stage_row(lab, s))
    hd = hn[hn.category == "hnl"]
    hd1 = hd[(hd.ad0 >= 1) & (hd.ad0 < 100)]
    for lab, s in [("HNL daughters", hd), ("HNL daughters, no exchanged hit", hd[hd.n_exch == 0]),
                   ("HNL daughters, exchanged hit", hd[hd.n_exch > 0]), ("HNL daughters, |d0| 1-100 mm", hd1),
                   ("HNL daughters, |d0| 1-100 mm, no exchanged hit", hd1[hd1.n_exch == 0]),
                   ("HNL daughters, |d0| 1-100 mm, exchanged hit", hd1[hd1.n_exch > 0]),
                   ("W muons (HNL events), no exchanged hit", hn[(hn.category == "w") & (hn.n_exch == 0)]),
                   ("W muons (HNL events), exchanged hit", hn[(hn.category == "w") & (hn.n_exch > 0)])]:
        L.append(stage_row(lab, s))
    for pb in disp.pt_bin.cat.categories:
        L.append(stage_row(f"displaced, pT {pb}", disp[disp.pt_bin == pb]))
    for db in disp.dens.cat.categories:
        L.append(stage_row(f"displaced, neighbours {db}", disp[disp.dens == db]))
    L += ["", "True edges of the targets (unique unordered hit pairs): in the graph and above s0.", "",
          "| population | edges | in graph | above s0 |", "|---|---|---|---|"]
    ed = edges.merge(tt[["event_id", "particle_id", "pop_", "pt_bin", "dens", "n_exch"]],
                     on=["event_id", "particle_id"], how="inner")

    def edge_row(lab, s):
        return f"| {lab} | {len(s)} | {pct(s.in_graph.sum(), len(s))} | {pct(s.above.sum(), len(s))} |"
    for popn in ("prompt", "displaced"):
        s = ed[ed.pop_ == popn]
        L.append(edge_row(f"{popn}, all", s))
        L.append(edge_row(f"{popn}, an end is an exchanged hit", s[s.exch_edge]))
        L.append(edge_row(f"{popn}, no end exchanged", s[~s.exch_edge]))
        L.append(edge_row(f"{popn}, no end exchanged, no end on a shared module", s[~s.exch_edge & ~s.shared_edge]))
        for pb in s.pt_bin.cat.categories:
            for db in list(s.dens.cat.categories) + ["all"]:
                q = s[~s.exch_edge & (s.pt_bin == pb)] if db == "all" else s[~s.exch_edge & (s.pt_bin == pb) &
                                                                             (s.dens == db)]
                if len(q) >= 20:
                    L.append(edge_row(f"{popn}, no end exchanged, pT {pb}, neighbours {db}", q))
    return "\n".join(L) + "\n"


def main(argv=None):
    from hnl.eval import bootstrap_tables as BT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    ap.add_argument("--limit", type=int, default=None, help="first N events of each sample (testing)")
    a = ap.parse_args(argv)
    pt_ = truth({"t": (("ttbar", "val"),)})
    ph = truth({"h": (("hnl10_100", "val"), ("hnl15_200", "val"))})
    nb = pd.concat([pt_, ph])
    nb = nb[(nb.pt > 0.5) & (nb.n_hits >= 3) & (nb.q != 0)][["event_id", "particle_id", "eta", "phi"]]
    tt = pt_[pt_.is_target].copy()
    hn = ph[ph.is_target & ph.category.isin(["hnl", "w"])].copy()
    if a.limit:
        tt = tt[tt.event.isin(np.sort(tt.event.unique())[:a.limit])]
        hn = hn[hn.event_id.isin(np.sort(hn.event_id.unique())[:a.limit])]
    pp, _, _ = BT.val_tables(RUN, WP)
    tt = tt.merge(pp[["event_id", "particle_id", "dm_matched"]], on=["event_id", "particle_id"], how="left")
    hn = hn.merge(pp[["event_id", "particle_id", "dm_matched"]], on=["event_id", "particle_id"], how="left")
    for lab, t in (("ttbar", tt), ("HNL", hn)):
        if t.dm_matched.isna().any():
            raise SystemExit(f"{int(t.dm_matched.isna().sum())} {lab} targets without a val row")
        t["dm_matched"] = t.dm_matched.astype(bool)
    parts, evs, prs = [], [], []
    tmp = tempfile.mkdtemp(prefix="djc_", dir=os.environ.get("TMPDIR", "/tmp"))
    try:
        graphs = extract_graphs(RUN, np.unique(np.r_[tt.event_id, hn.event_id]), tmp)
        r_tt, ev_tt, pr_tt, edges = run_sample("ttbar", np.unique(tt.event), tt, graphs, a.workers)
        for s in ("hnl10_100", "hnl15_200"):
            hs = hn[hn["sample"] == s]
            if not len(hs):
                continue
            r, e, p, _ = run_sample(s, np.unique(hs.event), hs, graphs, a.workers)
            parts.append(r)
            evs.append(e)
            prs.append(p)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    r_h = pd.concat(parts, ignore_index=True)
    pr_h = {k: sum(p[k] for p in prs) for k in prs[0]}
    d = pd.concat([tt.merge(r_tt, on=["event_id", "particle_id"]), hn.merge(r_h, on=["event_id", "particle_id"])],
                  ignore_index=True)
    d["n_dr05"] = neighbours(d, nb)
    d["pop_"] = np.where(d.ad0 < 1, "prompt", np.where(d.ad0 < 100, "displaced", "far"))
    d["dm_matched"] = d.dm_matched.astype("boolean")
    d["where_"] = np.where(d.dm_matched.fillna(False).astype(bool), "found",
                           np.where(d.stage == "pass", "builder", d.stage))
    d["pt_bin"] = pd.cut(d.pt, PT_EDGES, right=False)
    d["dens"] = pd.cut(d.n_dr05, DENS_EDGES)
    d["d0_bin_"] = pd.cut(d.ad0, D0_EDGES, right=False)
    dd = d[(d["sample"] == "ttbar") & (d.pop_ == "displaced")]
    print("displaced ttbar targets", len(dd), dd.where_.value_counts().to_dict())
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    keep = ["sample", "event", "event_id", "particle_id", "category", "pdg", "q", "pt", "eta", "n_hits", "n_pixel_hits",
            "r_first", "d0", "vsec", "gen", "process", "n_dr05", "dm_matched", "stage", "where_", "n_true_chain",
            "n_chain_in_graph", "n_chain_above_cut", "n_shared", "n_exch", "n_exch_inner", "d0_all_fit", "d0_clean",
            "dmin_mm"] + [f"n_close_{c}" for c in CLOSE]
    d[keep].to_csv(a.out + ".csv", index=False)
    edges.to_csv(a.out + "_edges.csv.gz", index=False)
    pr = []
    for s, p in (("ttbar", pr_tt), ("HNL", pr_h)):
        for k in range(len(SEP_EDGES) - 1):
            pr.append({"sample": s, "sep_lo_mm": SEP_EDGES[k], "sep_hi_mm": SEP_EDGES[k + 1],
                       **{key: int(v[k]) for key, v in p.items()}})
    pd.DataFrame(pr).to_csv(a.out + "_pairs.csv", index=False)
    md = summarize(d, edges, {"ttbar": ev_tt, "HNL": pd.concat(evs, ignore_index=True)},
                   {"ttbar": pr_tt, "HNL": pr_h})
    with open(a.out + ".md", "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
