"""Preregistration addendum G, step T1 (docs/preregistration_new_sample.md section 9), truth only: committed before any
model reads an event of the generated samples (docs/new_samples_spec.md).

Per sample (generic_iso, generic_dense, hss15_100, hss40_300; hnl.data.npz.GENERATED, needs HNL_GENERATED=1):
  * sha256 of every file of the sample and of its truth sidecars (sha256sum format, paths relative to generated/);
  * the event checks of docs/definitions.md section 1 and brief 2.3: the format invariants of hnl.data.scans
    .event_checks, and the corruption rule (any hit more than 50 mm outside its module's padded envelope; envelopes of
    the Zenodo scan derived/scans/v1, the same detector); an event is dropped if a format invariant fails or the rule
    selects it;
  * a per-particle truth table (every particle with hits): table values (pT, eta: final state, as in docs/definitions.md
    section 4), hits, pixel hits, the production vertex and initial momentum of the truth sidecar, exact |d0| (helix
    through the production vertex with the initial momentum in the 2 T field, about the z axis), the production radius,
    and for the generic samples the gun record (LLP index, prongs, mass, exact decay radius r_v).
Then (summary):
  * the split lists of addendum G section 3 (iso, dense) and the "all" lists of the h -> s s samples;
  * the train events of section 5.2 for T3 (first 7,999 of the two components' train splits, ordered by event
    number, iso before dense at equal number);
  * the target counts per split: P_G (isolated component), the same population on the dense component, all
    displaced-source targets, the ttbar prompt reference, and the h -> s s population P_H.

Populations (fixed here, before any model output):
  target     pT > 1 GeV, |eta| < 4 (table values), >= 5 hits (docs/definitions.md section 4)
  P_G        targets that are gun particles (displaced source: barcode secondary vertex >= 1000, listed in the
             gun_truth sidecars) with exact |d0| >= 10 mm or r_v >= 100 mm, isolated component
  dense      the same definition on the dense component (addendum G section 7)
  prompt     targets of the ttbar collision (not gun particles) with exact |d0| < 1 mm
  P_H        h -> s s samples: generator-level targets (barcode generation 0, sub-particle 0) with exact |d0| >= 10 mm
             or production radius >= 100 mm. Pythia decays K_S and Lambda in these samples and the files carry no
             ancestry, so P_H includes their daughters. Flags for the breakdown: electron / muon / hadron; s -> ee
             daughter (e+ e- of one generator vertex with invariant mass >= half the scalar mass); V0 daughter (pi+ pi-
             of one generator vertex within 5 MeV of the K_S mass, or p pi- / pbar pi+ within 5 MeV of the Lambda mass,
             initial momenta; both daughters must have hits).

    python -m hnl.eval.addendum_g_t1 scan --sample generic_iso [--workers 16]     (sbatch, one per sample)
    python -m hnl.eval.addendum_g_t1 summary
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import glob
import hashlib
import json
import os
import subprocess

import numpy as np
import pandas as pd

from hnl.data import npz
from hnl.data.scans import CORRUPT_EXCESS_MM, ENVELOPE_PAD_MM, event_checks
from hnl.geometry.ids import PIXEL_VOLUMES, decode_barcode, decode_geometry_id

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DERIVED = os.path.join(npz.DATA_ROOT, "derived")
ENVELOPES = os.path.join(DERIVED, "scans", "v1", "module_envelopes.parquet")
OUT_DATA = os.path.join(DERIVED, "generated_v1", "t1")
OUT_REPO = os.path.join(REPO, "results", "addendum_g_t1")
SPLITS_DIR = os.path.join(REPO, "splits")
COMPONENT = {"generic_iso": "iso", "generic_dense": "dense"}
HSS_MASS = {"hss15_100": 15.0, "hss40_300": 40.0}
SALT = "canto-generic-v1"
N_TRAIN_G = 7999
M_KS, M_LAMBDA, M_PI, M_P = 0.497611, 1.115683, 0.13957039, 0.93827209
FORMAT_TRUE = ("hid_is_perm", "pid_in_table", "particle_ids_unique", "edges_eq_hits_minus_particles",
               "edge_rows_in_range")
FORMAT_ZERO = ("n_edges_diff_pid", "n_edges_same_module", "n_bad_geoid_bits", "n_unexpected_volume", "n_nonfinite_x",
               "n_particles_not_chain")


def _sample(key: str) -> npz.Sample:
    if key not in npz.SAMPLES:
        raise SystemExit(f"{key} is not registered: run with HNL_GENERATED=1")
    return npz.SAMPLES[key]


def _git() -> str:
    return subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def split_of(component: str, event: int) -> str:
    """Addendum G section 3: h = first 8 hex digits of sha256("canto-generic-v1:<component>:<id>"), h mod 10."""
    h = int(hashlib.sha256(f"{SALT}:{component}:{event}".encode()).hexdigest()[:8], 16) % 10
    return "train" if h <= 5 else "val" if h == 6 else "test"


def d0_exact(x, y, px, py, q, b_tesla: float = 2.0) -> np.ndarray:
    """|d0| about the z axis of the helix through (x, y) with transverse momentum (px, py) and charge q in Bz."""
    pt = np.hypot(px, py)
    R = pt / (0.299792458 * b_tesla) * 1000.0
    with np.errstate(invalid="ignore", divide="ignore"):
        ux, uy = px / pt, py / pt
    s = np.sign(q)
    cx, cy = x + s * R * uy, y - s * R * ux
    return np.abs(np.hypot(cx, cy) - R)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _natural(name: str):
    stem = os.path.basename(name).split(".")[0]
    tail = stem.split("_")[-1]
    return (os.path.dirname(name), stem.rstrip("0123456789"), int(tail) if tail.isdigit() else -1, name)


# ------------------------------------------------------------------------------------------------- per event
_ENV = None


def _envelopes():
    global _ENV
    if _ENV is None:
        e = pd.read_parquet(ENVELOPES).sort_values("geoid").reset_index(drop=True)
        _ENV = e
    return _ENV


def envelope_excess(ev: dict) -> tuple[float, int, int]:
    """(largest excess in mm over the padded module envelope, hits outside, hits on modules unseen in the Zenodo scan);
    hnl.data.scans.envelope_flags with the Zenodo envelopes."""
    env = _envelopes()
    geo = ev["layers"]
    g = env.geoid.to_numpy()
    pos = np.clip(np.searchsorted(g, geo), 0, len(g) - 1)
    seen = g[pos] == geo
    unseen = ~seen | np.isnan(env.r_lo.to_numpy()[pos])
    r = ev["x"][:, 0].astype(np.float64)
    phi = ev["x"][:, 1].astype(np.float64)
    z = ev["x"][:, 2].astype(np.float64)
    dphi = np.remainder(phi - env.phi_c.to_numpy()[pos] + np.pi, 2 * np.pi) - np.pi
    excess = np.zeros(len(r))
    pad = ENVELOPE_PAD_MM
    for name, val, pad_v, scale in (("r", r, pad, 1.0), ("z", z, pad, 1.0),
                                    ("dphi", dphi, pad / np.maximum(r, 1.0), r)):
        lo = env[f"{name}_lo"].to_numpy()[pos]
        hi = env[f"{name}_hi"].to_numpy()[pos]
        ex = np.maximum(lo - pad_v - val, val - hi - pad_v) * scale
        np.maximum(excess, np.nan_to_num(ex, nan=0.0), out=excess)
    excess[unseen] = 0.0
    bad = (excess > 0) | unseen
    return float(excess.max()) if len(excess) else 0.0, int(bad.sum()), int(unseen.sum())


def scan_event(args):
    key, i = args
    ev = npz.load_event(key, i, check=True)
    st, _ = event_checks(ev)
    mx, nbad, nunseen = envelope_excess(ev)
    fmt_ok = all(bool(st[k]) for k in FORMAT_TRUE) and all(int(st[k]) == 0 for k in FORMAT_ZERO)
    row = {"event": i, "n_hits": st["n_hits"], "n_particles": st["n_particles"],
           "n_particles_with_hits": st["n_particles_with_hits"], "format_ok": fmt_ok,
           "env_max_excess_mm": mx, "n_env_bad": nbad, "n_env_unseen_module": nunseen,
           "n_r_low": st["n_r_low"], "n_r_high": st["n_r_high"], "n_absz_high": st["n_absz_high"],
           "n_dup_hit_hash_in_event": st["n_dup_hit_hash_in_event"],
           **{k: st[k] for k in FORMAT_TRUE + FORMAT_ZERO}}
    row["corrupted"] = mx > CORRUPT_EXCESS_MM
    row["dropped"] = (not fmt_ok) or row["corrupted"]
    # per-particle table values and hit counts
    P = ev["particles"]
    ids = npz.particle_ids(P)
    pid = ev["pid"]
    upid, cnt = np.unique(pid, return_counts=True)
    vol = decode_geometry_id(ev["layers"])["volume"]
    upix, cpix = np.unique(pid[np.isin(vol, PIXEL_VOLUMES)], return_counts=True)
    rows = np.searchsorted(ids, upid, sorter=np.argsort(ids))
    order = np.argsort(ids)
    tab = P[order[rows]]
    parts = pd.DataFrame({
        "event": np.int32(i), "particle_id": upid.astype(np.int64),
        "pdg": tab[:, npz.COL["particle_type"]].astype(np.int64), "q": tab[:, npz.COL["q"]].astype(np.float32),
        "pt": tab[:, npz.COL["pt"]], "eta": tab[:, npz.COL["eta"]],
        "process": tab[:, npz.COL["process"]].astype(np.int16),
        "n_hits": cnt.astype(np.int32),
        "n_pixel_hits": pd.Series(cpix, index=upix).reindex(upid).fillna(0).astype(np.int32).to_numpy(),
    })
    return row, parts


def load_sidecar(sample: npz.Sample, stem: str) -> dict:
    out = {}
    for f in sorted(glob.glob(os.path.join(sample.root, sample.dirname + "_truth", f"{stem}_*.npz"))):
        z = np.load(f)
        for k in z.files:
            if z[k].ndim:
                out.setdefault(k, []).append(z[k])
    return {k: np.concatenate(v) for k, v in out.items()}


def vertex_tags(t: pd.DataFrame, m_s: float | None) -> pd.DataFrame:
    """V0 and s -> ee flags per particle from its generator vertex (event, vprim, vsec) among particles with hits."""
    g = t[(t.gen == 0) & (t["sub"] == 0) & (t.vsec > 0)].copy()
    g["E"] = np.sqrt(g.p_px ** 2 + g.p_py ** 2 + g.p_pz ** 2 + g.mass ** 2)
    key = ["event", "vprim", "vsec"]
    n = g.groupby(key).particle_id.transform("size")
    two = g[n == 2].sort_values(key + ["particle_id"])
    a, b = two.iloc[0::2], two.iloc[1::2]
    assert (a[key].to_numpy() == b[key].to_numpy()).all()
    m2 = (a.E.to_numpy() + b.E.to_numpy()) ** 2 - sum((a[c].to_numpy() + b[c].to_numpy()) ** 2
                                                       for c in ("p_px", "p_py", "p_pz"))
    m = np.sqrt(np.maximum(m2, 0.0))
    pa, pb = a.pdg.to_numpy(), b.pdg.to_numpy()
    ks = ((pa + pb) == 0) & (np.abs(pa) == 211) & (np.abs(m - M_KS) < 0.005)
    lam = (((pa == 2212) & (pb == -211)) | ((pa == -211) & (pb == 2212)) | ((pa == -2212) & (pb == 211))
           | ((pa == 211) & (pb == -2212))) & (np.abs(m - M_LAMBDA) < 0.005)
    v0 = ks | lam
    see = np.zeros(len(a), bool) if m_s is None else ((pa + pb) == 0) & (np.abs(pa) == 11) & (m >= 0.5 * m_s)
    out = pd.DataFrame({"particle_id": np.r_[a.particle_id.to_numpy(), b.particle_id.to_numpy()],
                        "event": np.r_[a.event.to_numpy(), b.event.to_numpy()],
                        "v0_daughter": np.r_[v0, v0], "s_ee_daughter": np.r_[see, see],
                        "v0_kind": np.r_[np.where(ks, "K_S", np.where(lam, "Lambda", "")),
                                         np.where(ks, "K_S", np.where(lam, "Lambda", ""))]})
    return out


def cmd_scan(a) -> None:
    s = _sample(a.sample)
    os.makedirs(OUT_DATA, exist_ok=True)
    os.makedirs(OUT_REPO, exist_ok=True)
    # checksums of every file of the sample and its sidecars
    files = sorted(glob.glob(os.path.join(s.root, s.dirname, "*")) +
                   glob.glob(os.path.join(s.root, s.dirname + "_truth", "*")), key=_natural)
    with cf.ProcessPoolExecutor(a.workers) as ex:
        sums = list(ex.map(sha256_file, files, chunksize=64))
    with open(os.path.join(OUT_REPO, f"{s.dirname}.sha256"), "w") as f:
        for h, p in zip(sums, files):
            f.write(f"{h}  {os.path.relpath(p, s.root)}\n")
    print(f"[t1] {a.sample}: {len(files)} files checksummed", flush=True)
    # event checks and per-particle table values
    with cf.ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(scan_event, [(a.sample, i) for i in range(s.n_events)], chunksize=20))
    ev = pd.DataFrame([r[0] for r in res])
    parts = pd.concat([r[1] for r in res], ignore_index=True)
    print(f"[t1] {a.sample}: {len(ev)} events, {int(ev.dropped.sum())} dropped, {len(parts)} particles with hits",
          flush=True)
    # initial state from the truth sidecar
    tr = load_sidecar(s, "truth")
    side = pd.DataFrame({"event": tr["p_event"].astype(np.int32), "particle_id": tr["p_particle_id"].astype(np.int64),
                         "p_pdg": tr["p_particle_type"], "p_q": tr["p_q"],
                         "p_vx": tr["p_vx"], "p_vy": tr["p_vy"], "p_vz": tr["p_vz"],
                         "p_px": tr["p_px"], "p_py": tr["p_py"], "p_pz": tr["p_pz"]})
    if side.duplicated(["event", "particle_id"]).any():
        raise SystemExit("duplicate (event, particle) in the truth sidecar")
    t = parts.merge(side, on=["event", "particle_id"], how="left", validate="one_to_one")
    t["has_initial"] = t.p_px.notna()
    bc = decode_barcode(t.particle_id.to_numpy())
    for k, v in bc.items():
        t[k] = v.astype(np.int32)
    t["r_prod"] = np.hypot(t.p_vx, t.p_vy)
    t["d0"] = d0_exact(t.p_vx.to_numpy(), t.p_vy.to_numpy(), t.p_px.to_numpy(), t.p_py.to_numpy(), t.q.to_numpy())
    t["is_target"] = (t.pt > 1.0) & (np.abs(t.eta) < 4.0) & (t.n_hits >= 5)
    if a.sample in COMPONENT:
        g = load_sidecar(s, "gun_truth")
        gun = pd.DataFrame({"event": g["event"].astype(np.int32), "particle_id": g["particle_id"].astype(np.int64),
                            "llp": g["llp"], "n_prongs": g["n_prongs"], "llp_mass": g["llp_mass"],
                            "llp_betagamma": g["llp_betagamma"], "r_v": g["r_v"], "gun_pdg": g["pdg"],
                            "gun_q": g["q"], "gun_px": g["px"], "gun_py": g["py"], "gun_pz": g["pz"],
                            "gun_vx": g["vx"], "gun_vy": g["vy"]})
        gb = decode_barcode(gun.particle_id.to_numpy())
        if not ((gb["vsec"] >= 1000) & (gb["gen"] == 0) & (gb["sub"] == 0)).all():
            raise SystemExit("a gun particle without secondary vertex >= 1000, generation 0")
        t = t.merge(gun, on=["event", "particle_id"], how="left", validate="one_to_one")
        t["gun"] = t.llp.notna()
        bc_gun = (t.vsec >= 1000) & (t.gen == 0) & (t["sub"] == 0)
        mism = int((bc_gun != t.gun).sum())
        gun_hits = t[t.gun]
        chk = {
            "gun_particles": len(gun), "gun_particles_with_hits": int(t.gun.sum()),
            "barcode_vs_sidecar_mismatch": mism,
            "gun_pdg_mismatch": int((gun_hits.gun_pdg != gun_hits.pdg).sum()),
            "gun_initial_vs_sidecar_max_dp": float(np.nanmax(np.abs(gun_hits[["gun_px", "gun_py", "gun_pz"]].to_numpy()
                                                                    - gun_hits[["p_px", "p_py", "p_pz"]].to_numpy()))),
            "gun_vertex_vs_sidecar_max_dr": float(np.nanmax(np.hypot(gun_hits.gun_vx - gun_hits.p_vx,
                                                                     gun_hits.gun_vy - gun_hits.p_vy))),
        }
        t["d0_gun"] = np.where(t.gun, d0_exact(t.gun_vx.to_numpy(), t.gun_vy.to_numpy(), t.gun_px.to_numpy(),
                                               t.gun_py.to_numpy(), t.gun_q.to_numpy()), np.nan)
        chk["gun_d0_vs_sidecar_max_diff_mm"] = float(np.nanmax(np.abs(t.d0_gun - t.d0)[t.gun]))
        t = t.drop(columns=["gun_pdg", "gun_q", "gun_px", "gun_py", "gun_pz", "gun_vx", "gun_vy", "d0_gun"])
        t["in_pop"] = t.is_target & t.gun & ((t.d0 >= 10.0) | (t.r_v >= 100.0))
        t["prompt"] = t.is_target & ~t.gun & (t.d0 < 1.0)
    else:
        t["mass"] = np.abs(npz_mass(t.pdg.to_numpy()))
        tags = vertex_tags(t, HSS_MASS[a.sample])
        t = t.drop(columns=["mass"]).merge(tags, on=["event", "particle_id"], how="left", validate="one_to_one")
        t["v0_daughter"] = t.v0_daughter.fillna(False).astype(bool)
        t["s_ee_daughter"] = t.s_ee_daughter.fillna(False).astype(bool)
        t["v0_kind"] = t.v0_kind.fillna("")
        t["generator"] = (t.gen == 0) & (t["sub"] == 0)
        t["in_pop"] = t.is_target & t.generator & ((t.d0 >= 10.0) | (t.r_prod >= 100.0))
        t["prompt"] = t.is_target & t.generator & (t.d0 < 1.0) & (t.r_prod < 1.0)
        chk = {}
    chk.update({"particles_with_hits": len(t), "without_initial_state": int((~t.has_initial).sum()),
                "sidecar_pdg_mismatch": int((t.p_pdg.notna() & (t.p_pdg != t.pdg)).sum())})
    t = t.drop(columns=["p_pdg", "p_q"])
    tpath = os.path.join(OUT_DATA, f"truth_{a.sample}.parquet")
    epath = os.path.join(OUT_DATA, f"events_{a.sample}.parquet")
    t.to_parquet(tpath, index=False)
    ev.to_parquet(epath, index=False)
    meta = {"sample": a.sample, "dir": s.dirname, "n_files": len(files), "n_events": len(ev),
            "dropped": ev.loc[ev.dropped, "event"].tolist(), "checks": chk,
            "truth_table": tpath, "truth_table_sha256": sha256_file(tpath),
            "events_table": epath, "events_table_sha256": sha256_file(epath),
            "git": _git(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    with open(os.path.join(OUT_REPO, f"scan_{a.sample}.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))


def npz_mass(pdg: np.ndarray) -> np.ndarray:
    """Masses (GeV) of the stable charged species in the files; 0 for anything else (only used for V0 / ee tags)."""
    m = {11: 0.000511, 13: 0.10566, 211: M_PI, 321: 0.493677, 2212: M_P}
    a = np.abs(pdg)
    return np.array([m.get(int(x), 0.0) for x in a])


# ------------------------------------------------------------------------------------------------- summary
def write_split(path: str, header: str, events) -> None:
    body = header + "".join(f"{int(i)}\n" for i in events)
    if os.path.exists(path) and open(path).read() != body:
        raise SystemExit(f"{path} exists with different content")
    with open(path, "w") as f:
        f.write(body)


def cmd_summary(a) -> None:
    metas = {k: json.load(open(os.path.join(OUT_REPO, f"scan_{k}.json"))) for k in npz.GENERATED}
    rows, splits = [], {}
    for k, s in npz.GENERATED.items():
        ev = pd.read_parquet(metas[k]["events_table"])
        t = pd.read_parquet(metas[k]["truth_table"])
        keep = ev.loc[~ev.dropped, "event"].to_numpy()
        if k in COMPONENT:
            comp = COMPONENT[k]
            lab = np.array([split_of(comp, int(i)) for i in range(s.n_events)])
            for sp in ("train", "val", "test"):
                idx = np.setdiff1d(np.flatnonzero(lab == sp), ev.loc[ev.dropped, "event"].to_numpy())
                splits[(k, sp)] = idx
                write_split(os.path.join(SPLITS_DIR, f"{k}_{sp}.txt"),
                            f"# {s.dirname} {sp}: {len(idx)} events; preregistration addendum G section 3: first 8 hex "
                            f"digits of sha256('{SALT}:{comp}:<event>') mod 10 (0-5 train, 6 val, 7-9 test); dropped "
                            f"events removed (results/addendum_g_t1_v1.md)\n", idx)
        else:
            splits[(k, "all")] = keep
            write_split(os.path.join(SPLITS_DIR, f"{k}_all.txt"),
                        f"# {s.dirname} all: {len(keep)} events; evaluation only (docs/new_samples_spec.md section 6); "
                        f"dropped events removed (results/addendum_g_t1_v1.md)\n", keep)
        for (kk, sp), idx in splits.items():
            if kk != k:
                continue
            tt = t[t.event.isin(idx)]
            r = {"sample": k, "split": sp, "events": len(idx), "targets": int(tt.is_target.sum()),
                 "population": int(tt.in_pop.sum()), "prompt": int(tt.prompt.sum())}
            if k in COMPONENT:
                r["gun_particles_with_hits"] = int(tt.gun.sum())
                r["gun_targets"] = int((tt.gun & tt.is_target).sum())
                r["population_rv_ge_100"] = int((tt.in_pop & (tt.r_v >= 100)).sum())
                r["population_rv_lt_100"] = int((tt.in_pop & (tt.r_v < 100)).sum())
                r["population_d0_ge_300"] = int((tt.in_pop & (tt.d0 >= 300)).sum())
            else:
                p = tt[tt.in_pop]
                r["pop_electrons"] = int((np.abs(p.pdg) == 11).sum())
                r["pop_muons"] = int((np.abs(p.pdg) == 13).sum())
                r["pop_hadrons"] = int((~np.isin(np.abs(p.pdg), (11, 13))).sum())
                r["pop_s_ee"] = int(p.s_ee_daughter.sum())
                r["pop_v0_tagged"] = int(p.v0_daughter.sum())
            rows.append(r)
    # train events of section 5.2 (T3): both components' train splits by (event number, component), first 7,999
    tr = sorted([(int(i), 0, "generic_iso") for i in splits[("generic_iso", "train")]] +
                [(int(i), 1, "generic_dense") for i in splits[("generic_dense", "train")]])
    sel = tr[:N_TRAIN_G]
    path = os.path.join(SPLITS_DIR, "generic_train_g7999.txt")
    body = (f"# addendum G section 5.2 (T3): the first {len(sel)} of the {len(tr)} train events of generic_iso and "
            f"generic_dense, ordered by event number, iso before dense at equal number; columns: sample event\n"
            + "".join(f"{smp} {i}\n" for i, _, smp in sel))
    if os.path.exists(path) and open(path).read() != body:
        raise SystemExit(f"{path} exists with different content")
    with open(path, "w") as f:
        f.write(body)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(REPO, "results", "addendum_g_t1_v1.csv"), index=False)
    n_iso = sum(1 for _, _, smp in sel if smp == "generic_iso")
    md = markdown(tab, metas, len(sel), n_iso, len(tr))
    with open(os.path.join(REPO, "results", "addendum_g_t1_v1.md"), "w") as f:
        f.write(md)
    print(md)


def markdown(tab: pd.DataFrame, metas: dict, n_sel: int, n_iso: int, n_tr: int) -> str:
    L = ["# Addendum G, step T1: samples, splits and populations (truth only)", "",
         f"Written {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by `hnl.eval.addendum_g_t1` "
         f"(git {_git()[:7]}), before any model has read an event of these samples. Definitions in the module "
         "docstring; addendum G: `docs/preregistration_new_sample.md`; samples: `docs/new_samples_spec.md`.", "",
         "## Files and events", "",
         "| sample | directory | files (npz + sidecars) | events | dropped | checksums |", "|---|---|---|---|---|---|"]
    for k, m in metas.items():
        L.append(f"| {k} | generated/{m['dir']} | {m['n_files']} | {m['n_events']} | "
                 f"{len(m['dropped'])}{' (' + ', '.join(map(str, m['dropped'])) + ')' if m['dropped'] else ''} | "
                 f"`results/addendum_g_t1/{m['dir']}.sha256` |")
    L += ["", "Event checks: the format invariants of `hnl.data.scans.event_checks` (hid a permutation, every pid in the "
          "table, unique ids, E = hits - particles with hits, edges within one particle and between modules, one chain "
          "per particle, valid geometry ids, finite positions) and the corruption rule of docs/definitions.md section 1 "
          "(a hit more than 50 mm outside its module's padded envelope, envelopes from the Zenodo scan "
          "derived/scans/v1).", "", "| sample | largest envelope excess (mm) | hits outside the padded envelope | hits on "
          "modules unseen in Zenodo | truth checks |", "|---|---|---|---|---|"]
    for k, m in metas.items():
        ev = pd.read_parquet(m["events_table"])
        L.append(f"| {k} | {ev.env_max_excess_mm.max():.2f} | {int(ev.n_env_bad.sum())} | "
                 f"{int(ev.n_env_unseen_module.sum())} | "
                 + ", ".join(f"{a}: {b if not isinstance(b, float) else f'{b:.3g}'}" for a, b in m["checks"].items())
                 + " |")
    L += ["", "Truth tables (one row per particle with hits): " + "; ".join(
        f"`{os.path.relpath(m['truth_table'], npz.DATA_ROOT)}` sha256 `{m['truth_table_sha256'][:16]}`"
        for m in metas.values()) + ".", "",
        "## Splits and populations", "",
        "Split lists: `splits/generic_iso_{train,val,test}.txt`, `splits/generic_dense_{train,val,test}.txt` (addendum "
        "G section 3), `splits/hss15_100_all.txt`, `splits/hss40_300_all.txt`. Training events for T3 (section 5.2): "
        f"`splits/generic_train_g7999.txt`, the first {n_sel} of the {n_tr} train events of both components ordered by "
        f"event number, iso before dense at equal number ({n_iso} iso, {n_sel - n_iso} dense).", "",
        "| sample | split | events | targets | population (P_G / dense / P_H) | prompt reference | gun targets | "
        "pop. r_v >= 100 mm | pop. r_v < 100 mm | pop. abs(d0) >= 300 mm |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in tab[tab["sample"].isin(list(COMPONENT))].iterrows():
        L.append(f"| {r['sample']} | {r['split']} | {r['events']} | {r['targets']} | {r['population']} | {r['prompt']} | "
                 f"{int(r['gun_targets'])} | {int(r['population_rv_ge_100'])} | {int(r['population_rv_lt_100'])} | "
                 f"{int(r['population_d0_ge_300'])} |")
    L += ["", "| sample | events | targets | P_H | electrons | muons | hadrons | s -> ee daughters | V0-tagged | prompt |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in tab[~tab["sample"].isin(list(COMPONENT))].iterrows():
        L.append(f"| {r['sample']} | {r['events']} | {r['targets']} | {r['population']} | {int(r['pop_electrons'])} | "
                 f"{int(r['pop_muons'])} | {int(r['pop_hadrons'])} | {int(r['pop_s_ee'])} | {int(r['pop_v0_tagged'])} | "
                 f"{r['prompt']} |")
    t = tab.set_index(["sample", "split"])
    n_pg = int(t.loc[("generic_iso", "test"), "population"])
    L += ["", f"P_G on the test split: {n_pg} targets ({'at least' if n_pg >= 1000 else 'fewer than'} 1,000: the claims "
          f"are {'not ' if n_pg >= 1000 else ''}labelled underpowered, addendum G section 6).", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--sample", required=True)
    s.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    sub.add_parser("summary")
    a = ap.parse_args(argv)
    {"scan": cmd_scan, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
