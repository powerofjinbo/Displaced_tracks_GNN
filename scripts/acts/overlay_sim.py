"""Gun simulation of the TrackML pileup overlay (docs/pileup_overlay_plan.md sections 2.1 step 2, 4.1, 4.2) and the
truth gates F1 and containment of step H-P (section 10). Runs in the project environment; the Fatras step calls the
ACTS v15 container (hnl.trackml.overlay_sim). No model is run and no model output is read.

    sim IN OUT --build DIR --seed S [--first N --events N --threads T]
        gun particle CSVs (event<N>-particles.csv, ACTS barcodes) -> ActsExampleFatrasGeneric of the given build with
        the TrackML-like RZ field map, pMin 0.1 GeV, bremsstrahlung off (configs/trackml/overlay_sim_v1.json) ->
        event<N>-hits.csv, -particles_initial.csv, -particles_final.csv in OUT, plus fatras.log and
        sim_summary.json. Exits with 4 if any particle failed. The build directory is an argument: build-v15 now,
        the patched build of plan section 3 once it exists.
    field-probe OUT --events N --per-event K --seed S
        the particle mix of the field study's probe (field_material/acts_probe.py gen: mu, pi, e in shares
        0.4 / 0.4 / 0.2, p ~ logU(0.5, 20) GeV, eta ~ U(-3, 3), vertex (0, 0, N(0, 55 mm))); with K >= 50 the first
        50 particles of every event equal those of the earlier probe (job 6812220) for the same seed
    gun-probe OUT --per-host K --seed S
        K gun events per tp_v1 val host in the layout of plan section 7: 8 isolated and 2 dense LLPs of spec v1
        (scripts/acts/gun.py) at the host's hard-scatter vertex (truth particles with vertex field 1), azimuths
        redrawn until the LLP directions are Delta R >= 0.8 apart; barcodes vertex 1, secondary vertex 1001-1010.
        Reads only the truth particles table of the val hosts. Writes gun_probe_truth.csv and hosts.json beside
        the particle CSVs.
    gates --field-probe DIR --gun-probe DIR (--acts-detectors CSV | --module-map CSV) --out STEM [--trackml-segments PQ]
        F1 (no particle failures; mean bending residual against the loop field within 0.1 MeV per |z| band and z
        side) on the field probe, the same numbers on the gun probe, the model-free B_eff per z side, |z| and r band
        against TrackML truth of the calib training events, the containment of every gun-probe hit in its TrackML
        module, and checks that pMin and bremsstrahlung acted. Writes STEM.json and STEM.md.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gun as G  # noqa: E402  (spec v1 gun content: draw_llp, nbody, boost, barcode)

from hnl.trackml import overlay_sim as OS  # noqa: E402

COLS = G.COLS
PROBE_MASS = {11: 0.000510999, 13: 0.105658, 211: 0.139570}
VAL_HOSTS = "val"


# ------------------------------------------------------------------------------------------------ probes
def field_probe(out: str, events: int, per_event: int, seed: int) -> None:
    os.makedirs(out, exist_ok=True)
    for ev in range(events):
        rng = np.random.RandomState((seed * 100003 + ev) % (2**32))
        zv = rng.normal(0, 55.0)
        rows = []
        for i in range(per_event):
            a = abs(int(rng.choice([13, 211, 11], p=[0.4, 0.4, 0.2])))
            q = int(rng.choice([-1, 1]))
            pdg = a * (q if a not in (11, 13) else -q)
            p = float(np.exp(rng.uniform(np.log(0.5), np.log(20.0))))
            eta = rng.uniform(-3, 3)
            phi = rng.uniform(-np.pi, np.pi)
            th = 2 * np.arctan(np.exp(-eta))
            rows.append([(1 << 52) | ((i + 1) << 24), pdg, 0, 0.0, 0.0, zv, 0.0, p * np.sin(th) * np.cos(phi),
                         p * np.sin(th) * np.sin(phi), p * np.cos(th), PROBE_MASS[a], q])
        df = pd.DataFrame(rows, columns=COLS)
        df["particle_id"] = df.particle_id.astype(np.uint64)
        df.to_csv(os.path.join(out, f"event{ev:09d}-particles.csv"), index=False, float_format="%.10g")


def _dphi(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def free_azimuths(eta: float, placed: list, dr_min: float) -> list[tuple[float, float]]:
    """Azimuth intervals in [0, 2 pi) at which a direction (eta, phi) is at least dr_min from every placed
    (eta, phi): the complement of the union of the arcs phi_k +- sqrt(dr_min^2 - (eta - eta_k)^2)."""
    two_pi = 2 * np.pi
    arcs = []
    for e, f in placed:
        de = abs(eta - e)
        if de < dr_min:
            h = np.sqrt(dr_min * dr_min - de * de)
            a = (f - h) % two_pi
            if a + 2 * h >= two_pi:
                arcs += [(a, two_pi), (0.0, a + 2 * h - two_pi)]
            else:
                arcs.append((a, a + 2 * h))
    free, x = [], 0.0
    for a, b in sorted(arcs):
        if a > x:
            free.append((x, a))
        x = max(x, b)
    if x < two_pi:
        free.append((x, two_pi))
    return [(a, b) for a, b in free if b - a > 1e-12]


def probe_gun_event(vertex, rng, n_iso: int = 8, n_dense: int = 2, dr_min: float = 0.8, max_directions: int = 1000):
    """One gun-only event in the layout of plan section 7 at `vertex` (x, y, z, t). The per-LLP content is that of
    scripts/acts/gun.py make_event. If the drawn direction lies within dr_min of an LLP placed before, the azimuth
    is redrawn uniformly over the allowed azimuths at that eta (the same distribution as redrawing until it passes);
    if no azimuth is allowed, eta and r_v are drawn again. Returns the particle rows (ACTS barcodes), one truth row
    per particle and the numbers of LLPs whose first azimuth was rejected and of eta redraws."""
    pv = np.asarray(vertex, np.float64)
    rows, truth, placed = [], [], []
    redraws = [0, 0]
    for k, comp in enumerate(["iso"] * n_iso + ["dense"] * n_dense, 1):
        ds, qs, m, bg = G.draw_llp(comp, rng)
        for _ in range(max_directions):
            eta, phi = rng.uniform(-3, 3), rng.uniform(0, 2 * np.pi)
            while True:
                rv = rng.uniform(1.0, 800.0)
                zv = pv[2] + rv * np.sinh(eta)
                if abs(zv) <= 2900.0:
                    break
                eta = rng.uniform(-3, 3)
            if not any(np.hypot(eta - e, _dphi(phi, f)) < dr_min for e, f in placed):
                break
            free = free_azimuths(eta, placed, dr_min)
            if free:
                redraws[0] += 1
                x = rng.uniform(0, sum(b - a for a, b in free))
                for a, b in free:
                    if x <= b - a:
                        phi = a + x
                        break
                    x -= b - a
                else:
                    phi = free[-1][1]
                break
            redraws[1] += 1
        else:
            raise RuntimeError("no direction satisfies the Delta R rule")
        placed.append((eta, phi))
        u = np.array([np.cos(phi) / np.cosh(eta), np.sin(phi) / np.cosh(eta), np.tanh(eta)])
        beta = bg / np.sqrt(1 + bg * bg)
        length = rv * np.cosh(eta)
        xv = np.array([pv[0] + rv * np.cos(phi), pv[1] + rv * np.sin(phi), zv, pv[3] + length / (beta * G.C_MM_PER_NS)])
        p4 = G.boost(G.nbody(m, [G.MASS[d] for d in ds], rng), beta * u)
        for i, (d, q) in enumerate(zip(ds, qs), 1):
            pid = G.barcode(1, G.VSEC0 + k, i)
            rows.append([pid, G.signed_pdg(d, q), 0, *xv, *p4[i - 1, 1:], G.MASS[d], q])
            truth.append({"particle_id": pid, "llp": k, "component": comp, "n_prongs": len(ds), "llp_mass": m,
                          "llp_betagamma": bg, "llp_eta": eta, "llp_phi": phi, "r_v": rv, "vx": xv[0], "vy": xv[1],
                          "vz": xv[2], "pdg": G.signed_pdg(d, q), "q": q, "px": p4[i - 1, 1], "py": p4[i - 1, 2],
                          "pz": p4[i - 1, 3]})
    df = pd.DataFrame(rows, columns=COLS)
    df["particle_id"] = df.particle_id.astype(np.uint64)
    return df, truth, redraws


def host_vertices(hosts: list[int]) -> pd.DataFrame:
    """Hard-scatter vertex of each host: median production vertex of its truth particles with vertex field 1 and
    r_v < 1 mm (tp_v1 particles table, truth only)."""
    from hnl.trackml.convert import read_table
    from hnl.trackml.split import EXT
    p = read_table("particles", hosts, columns=["event", "vx", "vy", "vz", "vprim", "rv"],
                   root=os.path.join(EXT, "derived", "tp_v1"))
    hs = p[(p.vprim == 1) & (p.rv < 1.0)].groupby("event").agg(vx=("vx", "median"), vy=("vy", "median"),
                                                               vz=("vz", "median"), n=("vx", "size"))
    missing = sorted(set(hosts) - set(hs.index))
    if missing:
        raise ValueError(f"hosts without a vertex-1 particle: {missing[:5]}")
    return hs.loc[hosts].reset_index()


def gun_probe(out: str, per_host: int, seed: int, hosts_spec: str = VAL_HOSTS) -> None:
    from hnl.trackml.config import event_set
    if hosts_spec != VAL_HOSTS:
        raise ValueError("gun-probe hosts are the tp_v1 val events only (development hosts of plan section 6)")
    hosts = sorted(event_set(hosts_spec))
    hv = host_vertices(hosts)
    os.makedirs(out, exist_ok=True)
    truth, redraws, ev = [], [0, 0], 0
    for _, h in hv.iterrows():
        for j in range(per_host):
            rng = np.random.default_rng([seed, ev])
            df, t, n = probe_gun_event((h.vx, h.vy, h.vz, 0.0), rng)
            df.to_csv(os.path.join(out, f"event{ev:09d}-particles.csv"), index=False, float_format="%.10g")
            truth += [{"event": ev, "host": int(h.event), **r} for r in t]
            redraws = [a + b for a, b in zip(redraws, n)]
            ev += 1
    pd.DataFrame(truth).to_csv(os.path.join(out, "gun_probe_truth.csv"), index=False, float_format="%.8g")
    with open(os.path.join(out, "hosts.json"), "w") as fh:
        json.dump({"hosts": hosts_spec, "n_hosts": len(hosts), "per_host": per_host, "seed": seed, "events": ev,
                   "first_azimuth_rejected_per_llp": redraws[0] / (ev * 10),
                   "eta_redraws_per_llp": redraws[1] / (ev * 10),
                   "host_vertex": hv.round(4).to_dict("list")}, fh)
    print(f"gun probe: {ev} events, {len(truth)} particles, {redraws[0] / (ev * 10):.3f} azimuth redraws and "
          f"{redraws[1] / (ev * 10):.4f} eta redraws per LLP")


# ------------------------------------------------------------------------------------------------ simulation
def run_sim(inp: str, out: str, build: str, seed: int, first: int | None, events: int | None, threads: int,
            settings_path: str, field_map: str | None, allow_failures: bool) -> int:
    st = OS.load_settings(settings_path)
    fmap = os.path.abspath(field_map or st["field_map"]["path"])
    md5 = OS.file_md5(fmap)
    if field_map is None and md5 != st["field_map"]["md5"]:
        raise ValueError(f"field map md5 {md5} differs from {settings_path}")
    skip, lo, n = OS.event_range(inp, first, events)
    os.makedirs(out, exist_ok=True)
    if os.listdir(out):
        raise FileExistsError(f"{out} is not empty")
    args = OS.fatras_args(st, os.path.abspath(inp), os.path.abspath(out), seed, threads, skip, n, fmap)
    cmd = OS.container_command(st, build, args, [inp, out, fmap])
    t0 = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    dt = time.time() - t0
    with open(os.path.join(out, "fatras.log"), "w") as fh:
        fh.write(proc.stdout)
    log = OS.parse_fatras_log(proc.stdout)
    exe = os.path.join(build, "bin", st["executable"])
    summary = {"input_dir": os.path.abspath(inp), "output_dir": os.path.abspath(out), "build": os.path.abspath(build),
               "executable_md5": OS.file_md5(exe), "settings": st["version"], "fatras": st["fatras"],
               "field_map": fmap, "field_map_md5": md5, "seed": seed, "first_event": lo, "events": n,
               "threads": threads, "args": args, "returncode": proc.returncode, "wall_s": round(dt, 1), **log}
    with open(os.path.join(out, "sim_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({k: summary[k] for k in ("returncode", "events_processed", "particle_failures", "error_lines",
                                              "fatal_lines", "wall_s")}))
    if proc.returncode != 0:
        return proc.returncode
    if log["events_processed"] != n:
        print(f"processed {log['events_processed']} of {n} events", file=sys.stderr)
        return 5
    if log["particle_failures"] and not allow_failures:
        return 4
    return 0


# ------------------------------------------------------------------------------------------------ gates
def sim_checks(inp: str, hits: pd.DataFrame, init: pd.DataFrame, final: pd.DataFrame, pmin: float) -> dict:
    """Did the settings act? Input particles above pMin all simulated, none below; no secondaries (no brem photons,
    no decays); the largest fractional energy change of electrons per crossing (bremsstrahlung would give a few
    per cent per crossing, plan section 4.2)."""
    ins = []
    for e in sorted(init.event.unique()):
        ins.append(OS.read_csv_event(inp, int(e), "particles").assign(event=e))
    i = pd.concat(ins, ignore_index=True)
    i["p"] = np.sqrt(i.px ** 2 + i.py ** 2 + i.pz ** 2)
    j = i.merge(init[["event", "particle_id"]].assign(sim=True), on=["event", "particle_id"], how="outer",
                indicator=True)
    sim = j._merge != "left_only"
    charged = j.q.fillna(0) != 0
    above, below = charged & (j.p > pmin), charged & (j.p <= pmin)
    gen = ((init.particle_id.to_numpy().astype(np.uint64) >> np.uint64(16)) & np.uint64(0xFF)).astype(int)
    ele = hits.merge(init[["event", "particle_id", "particle_type"]], on=["event", "particle_id"])
    ele = ele[ele.particle_type.abs() == 11]
    frac = (-ele.deltae / ele.te).to_numpy() if len(ele) else np.zeros(0)
    q = (lambda x: round(float(np.quantile(frac, x)), 6) if len(frac) else None)
    outside = (np.hypot(final.vx, final.vy) > OS.WORLD_R_MM) | (final.vz.abs() > OS.WORLD_ABSZ_MM)
    return {"input_particles": int(len(i)), "input_charged_above_pmin": int(above.sum()),
            "final_outside_world": int(outside.sum()),
            "simulated": int(len(init)), "final": int(len(final)),
            "charged_above_pmin_not_simulated": int((above & ~sim).sum()),
            "charged_at_or_below_pmin_simulated": int((below & sim).sum()),
            "simulated_not_in_input": int((j._merge == "right_only").sum()), "secondaries": int((gen > 0).sum()),
            "simulated_with_p_below_0.5_GeV": int((sim & (j.p < 0.5)).sum()),
            "electron_crossings": int(len(ele)),
            "electron_frac_energy_loss_median": q(0.5), "electron_frac_energy_loss_p99": q(0.99),
            "electron_max_frac_energy_loss": round(float(frac.max()), 6) if len(frac) else None,
            "electron_crossings_frac_loss_above_1pct": int((frac > 0.01).sum())}


def trackml_beff(path: str) -> tuple[dict, dict]:
    """B_eff table of the TrackML truth segments (field_material/segments_calib200.parquet: calib training events)."""
    import pyarrow.parquet as pq
    cols = ["tx0", "ty0", "tz0", "tx1", "ty1", "tz1", "tpx0", "tpy0", "tpz0", "tpx1", "tpy1", "tpz1", "q", "pdg", "rv",
            "event"]
    t = pq.read_table(path, columns=cols).to_pandas()
    x0 = t[["tx0", "ty0", "tz0"]].to_numpy(np.float64)
    x1 = t[["tx1", "ty1", "tz1"]].to_numpy(np.float64)
    p0 = t[["tpx0", "tpy0", "tpz0"]].to_numpy(np.float64)
    p1 = t[["tpx1", "tpy1", "tpz1"]].to_numpy(np.float64)
    q = t.q.to_numpy(np.float64)
    m = OS.beff_select(x0, x1, p0, p1, q, t.pdg.to_numpy(), t.rv.to_numpy())
    ev = sorted(t.event.unique().tolist())
    return OS.beff_table(x0[m], x1[m], p0[m], p1[m], q[m]), {"events": len(ev), "first": ev[0], "last": ev[-1]}


def probe_block(d: str, field, pmin: float) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    with open(os.path.join(d, "sim_summary.json")) as fh:
        sim = json.load(fh)
    hits, init, final = OS.read_sim_events(d)
    seg = OS.segments(hits, init)
    f1 = OS.f1_tables(seg, field)
    f1["particle_failures"] = sim["particle_failures"]
    f1["gate_pass"] = bool(f1["pass"] and sim["particle_failures"] == 0 and sim["events_processed"] == sim["events"])
    checks = sim_checks(sim["input_dir"], hits, init, final, pmin)
    keep = ["build", "executable_md5", "field_map", "field_map_md5", "seed", "first_event", "events",
            "events_processed", "particle_failures", "error_lines", "fatal_lines", "wall_s", "args"]
    return {"sim": {k: sim[k] for k in keep}, "checks": checks, "f1": f1}, seg, hits


def run_gates(a) -> int:
    st = OS.load_settings(a.settings)
    fit = st["field_map"]["loop_fit"]
    tml_path = st["trackml_detectors"]
    for path, key in ((fit, st["field_map"]["loop_fit_md5"]), (tml_path, st["trackml_detectors_md5"])):
        if OS.file_md5(path) != key:
            raise ValueError(f"{path}: md5 differs from {a.settings}")
    field = OS.NodeField.from_loop_fit(fit)
    # the reference is the loop model itself: its node table, and the delivered map, against the fitted loops
    d = np.load(fit)
    rr, zz = np.meshgrid(field.r, field.z, indexing="ij")
    inner = (rr <= 1150) & (np.abs(zz) <= 3100)
    br, bz = OS.loop_model(rr[inner], zz[inner], d["currents"], d["loops"])
    mr, mz, mbr, mbz = OS.read_rz_map(st["field_map"]["path"])
    MR, MZ = np.meshgrid(mr, mz, indexing="ij")
    mi = (MR <= 1150) & (np.abs(MZ) <= 3100)
    br2, bz2 = OS.loop_model(MR[mi], MZ[mi], d["currents"], d["loops"])
    ref = {"loop_fit": fit, "loop_fit_md5": st["field_map"]["loop_fit_md5"],
           "nodes_vs_loop_model_max_abs_T": round(float(max(np.abs(field.br[inner] - br).max(),
                                                            np.abs(field.bz[inner] - bz).max())), 7),
           "map_vs_loop_model_max_abs_T": round(float(max(np.abs(mbr[mi] - br2).max(),
                                                          np.abs(mbz[mi] - bz2).max())), 7),
           "map_grid": {"r_mm": [float(mr[0]), float(mr[-1]), len(mr)], "z_mm": [float(mz[0]), float(mz[-1]), len(mz)]},
           "closure": OS.reference_closure(field)}
    pmin = float(st["fatras"]["pmin_gev"])
    fp, fseg, _ = probe_block(a.field_probe, field, pmin)
    gp, gseg, ghits = probe_block(a.gun_probe, field, pmin)
    gp["hosts"] = None
    if a.gun_probe_input:
        with open(os.path.join(a.gun_probe_input, "hosts.json")) as fh:
            gp["hosts"] = {k: v for k, v in json.load(fh).items() if k != "host_vertex"}
    # B_eff against TrackML truth (descriptive)
    beff = None
    if a.trackml_segments:
        tml_tab, info = trackml_beff(a.trackml_segments)
        s = fseg[(fseg.q != 0)]
        x0, x1, p0, p1, q = OS.segment_arrays(s)
        m = OS.beff_select(x0, x1, p0, p1, q, s.pdg.to_numpy(), s.rv.to_numpy())
        sim_tab = OS.beff_table(x0[m], x1[m], p0[m], p1[m], q[m])
        rows = OS.beff_compare(sim_tab, tml_tab)
        diffs = [abs(r[f"diff{sd}"]) for r in rows for sd in ("+z", "-z") if r[f"diff{sd}"] is not None]
        beff = {"trackml_segments": a.trackml_segments, "trackml_events": info, "rows": rows,
                "max_abs_diff_T": round(max(diffs), 4) if diffs else None}
    # containment of every gun-probe hit
    tml = pd.read_csv(tml_path)
    if a.module_map:
        mmap = OS.read_module_map(a.module_map)
        source = {"table": os.path.abspath(a.module_map), "md5": OS.file_md5(a.module_map)}
    elif a.acts_detectors:
        mmap = OS.build_module_map(pd.read_csv(a.acts_detectors), tml)
        source = {"built_from": os.path.abspath(a.acts_detectors), "md5": OS.file_md5(a.acts_detectors)}
    else:
        raise ValueError("give --module-map or --acts-detectors")
    flags = OS.containment(ghits, mmap, tml)
    contain = OS.containment_summary(flags, [ghits.event.to_numpy(), ghits.particle_id.to_numpy()])
    contain["module_map"] = {**source, "modules": int(len(mmap)),
                             "agreement": [OS.map_agreement(mmap, p, ac, tc) for p, ac, tc in a.compare_map]}
    if "d_centre_mm" in mmap:
        contain["module_map"]["max_centre_distance_mm"] = round(float(mmap.d_centre_mm.max()), 6)
        contain["module_map"]["modules_with_centre_distance_above_tol"] = int(
            (mmap.d_centre_mm > OS.CONTAIN_TOL_MM).sum())
    fcont = OS.containment(OS.read_sim_events(a.field_probe)[0], mmap, tml)
    contain["field_probe_hits"] = {k: v for k, v in OS.containment_summary(fcont).items() if k != "per_volume"}
    out = {"step": "H-P", "plan": "docs/pileup_overlay_plan.md section 10 (F1, containment)",
           "settings": st["version"], "fatras": st["fatras"], "field_reference": ref,
           "field_probe": fp, "gun_probe": gp, "beff_vs_trackml": beff, "containment": contain,
           "gates": {"F1": {"rule": "no particle failures; |mean bending residual| <= 0.1 MeV in every |z| band, "
                                    "separately for +z and -z (field probe)",
                            "pass": fp["f1"]["gate_pass"], "max_abs_mean_MeV": fp["f1"]["max_abs_mean_MeV"],
                            "bands_pass": fp["f1"]["pass"], "nsub": fp["f1"]["nsub"],
                            "scoring_closure_max_abs_mean_MeV": ref["closure"][f"nsub{fp['f1']['nsub']}"][
                                "max_abs_mean_MeV"],
                            "particle_failures": fp["sim"]["particle_failures"],
                            "gun_probe_particle_failures": gp["sim"]["particle_failures"],
                            "gun_probe_bands_pass": gp["f1"]["pass"],
                            "gun_probe_same_rule_pass": gp["f1"]["gate_pass"],
                            "gun_probe_max_abs_mean_MeV": gp["f1"]["max_abs_mean_MeV"]},
                     "containment": {"rule": f"every gun hit on its TrackML module plane and inside its bounds "
                                             f"(tolerance {OS.CONTAIN_TOL_MM * 1000:.0f} um); every mapped id in "
                                             "TrackML detectors.csv",
                                     "pass": contain["pass"], "hits": contain["hits"], "failed": contain["failed"]}}}
    with open(a.out + ".json", "w") as fh:
        json.dump(out, fh, indent=1)
    with open(a.out + ".md", "w") as fh:
        fh.write(markdown(out))
    print(json.dumps(out["gates"], indent=1))
    return 0


def _fmt(v, nd=3):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def markdown(o: dict) -> str:
    fp, gp, c = o["field_probe"], o["gun_probe"], o["containment"]
    g, ref = o["gates"], o["field_reference"]
    ok = (lambda v: "pass" if v else "FAIL")
    esc = (lambda t: t.replace("|", "\\|"))
    L = [f"# H-P field and containment gates ({os.path.basename(fp['sim']['build'])})", "",
         "Generated by `scripts/acts/overlay_sim.py gates` (plan section 10, F1 and containment). Truth only: no model "
         "ran and no model output was read.", "",
         f"Build: `{fp['sim']['build']}` (executable md5 {fp['sim']['executable_md5']}). Settings "
         f"`{o['settings']}`: pMin {o['fatras']['pmin_gev']} GeV, bremsstrahlung "
         f"{'on' if o['fatras']['em_radiation'] else 'off'}, field map md5 {fp['sim']['field_map_md5']}.", "",
         "## Gates", "",
         "| Gate | Rule | Result | Value |", "|---|---|---|---|",
         f"| F1 (field probe) | {esc(g['F1']['rule'])} | {ok(g['F1']['pass'])} | "
         f"{g['F1']['particle_failures']} particle "
         f"failures; bands {ok(g['F1']['bands_pass'])}, max \\|mean\\| {_fmt(g['F1']['max_abs_mean_MeV'])} MeV |",
         f"| F1, same rule on the gun probe | as above | {ok(g['F1']['gun_probe_same_rule_pass'])} | "
         f"{g['F1']['gun_probe_particle_failures']} particle failures; bands {ok(g['F1']['gun_probe_bands_pass'])}, "
         f"max \\|mean\\| {_fmt(g['F1']['gun_probe_max_abs_mean_MeV'])} MeV |",
         f"| Containment (gun probe) | {esc(g['containment']['rule'])} | {ok(g['containment']['pass'])} | "
         f"{g['containment']['failed']} of {g['containment']['hits']} gun hits fail |", "",
         f"The kick of the loop field is integrated with {g['F1']['nsub']} samples along each segment. Scored the same "
         "way, material-free RK4 tracks through the same field give a largest band mean of "
         f"{_fmt(g['F1']['scoring_closure_max_abs_mean_MeV'], 4)} MeV (with the 6 samples of the earlier probe: "
         f"{_fmt(ref['closure']['nsub6']['max_abs_mean_MeV'], 4)} MeV). The delivered map equals the loop model within "
         f"{ref['map_vs_loop_model_max_abs_T']} T at its nodes inside r = 1150 mm and |z| = 3100 mm.", ""]
    for name, b in (("Field probe", fp), ("Gun probe", gp)):
        s, k = b["sim"], b["checks"]
        L += [f"## {name}: mean bending residual against the loop field [MeV]", "",
              f"{s['events']} events, {k['simulated']} particles simulated, {s['particle_failures']} particle "
              f"failures ({s['error_lines']} ERROR lines), {k['final_outside_world']} simulated particles end outside "
              f"the detector world; {b['f1']['segments']} segments.", "",
              "| \\|z\\| [mm] | +z n | +z mean | +z sem | -z n | -z mean | -z sem | -z mean, 6 samples |",
              "|---|---|---|---|---|---|---|---|"]
        rows = {(r["side"], r["absz_mm"]): r for r in b["f1"]["bands"]}
        six = {(r["side"], r["absz_mm"]): r for r in b["f1"]["bands_nsub6"]}
        for lo, hi in OS.Z_BANDS:
            key = f"{lo}-{hi}"
            p, m = rows[("+z", key)], rows[("-z", key)]
            L.append(f"| {key} | {p['n']} | {_fmt(p['mean_MeV'])} | {_fmt(p['sem_MeV'])} | {m['n']} | "
                     f"{_fmt(m['mean_MeV'])} | {_fmt(m['sem_MeV'])} | {_fmt(six[('-z', key)]['mean_MeV'])} |")
        L += ["", "By detector region (descriptive):", "", "| Region | side | n | mean | sem |",
              "|---|---|---|---|---|"]
        for r in b["f1"]["regions"]:
            L.append(f"| {r['region']} | {r['side']} | {r['n']} | {_fmt(r['mean_MeV'])} | {_fmt(r['sem_MeV'])} |")
        L += ["", f"Settings checks: {k['charged_above_pmin_not_simulated']} charged particles above pMin not "
                  f"simulated (the failures), {k['charged_at_or_below_pmin_simulated']} at or below it simulated, "
                  f"{k['simulated_with_p_below_0.5_GeV']} simulated particles with p < 0.5 GeV, "
                  f"{k['secondaries']} secondaries. Electron crossings {k['electron_crossings']}: fractional energy "
                  f"loss median {_fmt(k['electron_frac_energy_loss_median'], 5)}, p99 "
                  f"{_fmt(k['electron_frac_energy_loss_p99'], 5)}, "
                  f"largest {_fmt(k['electron_max_frac_energy_loss'], 5)}, "
                  f"{k['electron_crossings_frac_loss_above_1pct']} above 1 %.", ""]
    if o["beff_vs_trackml"]:
        bt = o["beff_vs_trackml"]
        L += ["## B_eff of pointing segments, field probe against TrackML truth [T] (descriptive)", "",
              f"TrackML: calib training events {bt['trackml_events']['first']}-{bt['trackml_events']['last']} "
              f"({bt['trackml_events']['events']} events). Median per bin with at least 200 segments.", "",
              "| \\|z\\| [mm] | r [mm] | +z sim | +z TrackML | -z sim | -z TrackML | asym sim | asym TrackML |",
              "|---|---|---|---|---|---|---|---|"]
        for r in bt["rows"]:
            L.append(f"| {r['absz_mm']} | {r['r_mm']} | {_fmt(r['sim+z'])} | {_fmt(r['trackml+z'])} | "
                     f"{_fmt(r['sim-z'])} | {_fmt(r['trackml-z'])} | {_fmt(r['sim_asym'])} | "
                     f"{_fmt(r['trackml_asym'])} |")
        L.append("")
    mm = c["module_map"]
    src = f"table `{mm['table']}`" if "table" in mm else f"nearest centres from `{mm['built_from']}`"
    L += ["## Containment of the gun-probe hits", "",
          f"Module map: {src}, {mm['modules']} modules"
          + (f"; largest centre distance {_fmt(mm['max_centre_distance_mm'], 4)} mm, "
             f"{mm['modules_with_centre_distance_above_tol']} modules above the tolerance"
             if "max_centre_distance_mm" in mm else "") + ".", "",
          "| Volume | hits | off plane | outside bounds | failed | max \\|w\\| [mm] | max out of bounds [mm] |",
          "|---|---|---|---|---|---|---|"]
    for v, b in c["per_volume"].items():
        L.append(f"| {v} | {b['hits']} | {b['off_plane']} | {b['outside_bounds']} | {b['failed']} | "
                 f"{_fmt(b['max_abs_w_mm'], 4)} | {_fmt(b['max_out_of_bounds_mm'], 4)} |")
    L += ["", f"All: {c['hits']} hits, {c['failed']} failed ({c['unmapped']} unmapped, {c['not_in_trackml']} not in "
              f"TrackML); {c.get('particles_with_failed_hit', '-')} of {c.get('particles', '-')} gun particles have a "
              f"failed hit. Field-probe hits: {c['field_probe_hits']['failed']} of {c['field_probe_hits']['hits']} "
              "fail. A negative out-of-bounds value is the smallest distance of a hit inside the module edge.", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sim")
    s.add_argument("input_dir")
    s.add_argument("output_dir")
    s.add_argument("--build", required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--first", type=int, default=None)
    s.add_argument("--events", type=int, default=None)
    s.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    s.add_argument("--settings", default=OS.SETTINGS_PATH)
    s.add_argument("--field-map", default=None, help="override the map of the settings (its md5 is recorded)")
    s.add_argument("--allow-failures", action="store_true")
    f = sub.add_parser("field-probe")
    f.add_argument("out")
    f.add_argument("--events", type=int, default=1000)
    f.add_argument("--per-event", type=int, default=100)
    f.add_argument("--seed", type=int, default=777)
    g = sub.add_parser("gun-probe")
    g.add_argument("out")
    g.add_argument("--per-host", type=int, default=4)
    g.add_argument("--seed", type=int, required=True)
    t = sub.add_parser("gates")
    t.add_argument("--field-probe", required=True, help="Fatras output directory of the field probe")
    t.add_argument("--gun-probe", required=True, help="Fatras output directory of the gun probe")
    t.add_argument("--gun-probe-input", default=None, help="the gun-probe particle directory (for hosts.json)")
    t.add_argument("--acts-detectors", default=None, help="detectors.csv dumped from the same build (the module "
                   "map is built from it by nearest centres)")
    t.add_argument("--module-map", default=None, help="a module-id table (volume_id, layer_id, acts_module_id, "
                   "trackml_module_id), e.g. configs/trackml/module_map_acts_zmirror_v1.csv")
    t.add_argument("--trackml-segments", default=None)
    t.add_argument("--compare-map", nargs=3, action="append", default=[],
                   metavar=("CSV", "ACTS_COL", "TRACKML_COL"), help="a stored module map to compare with")
    t.add_argument("--settings", default=OS.SETTINGS_PATH)
    t.add_argument("--out", required=True, help="output stem (STEM.json, STEM.md)")
    r = sub.add_parser("report", help="rewrite STEM.md from STEM.json")
    r.add_argument("stem")
    a = ap.parse_args(argv)
    if a.cmd == "report":
        with open(a.stem + ".json") as fh:
            o = json.load(fh)
        with open(a.stem + ".md", "w") as fh:
            fh.write(markdown(o))
        return 0
    if a.cmd == "sim":
        return run_sim(a.input_dir, a.output_dir, a.build, a.seed, a.first, a.events, a.threads, a.settings,
                       a.field_map, a.allow_failures)
    if a.cmd == "field-probe":
        field_probe(a.out, a.events, a.per_event, a.seed)
        return 0
    if a.cmd == "gun-probe":
        gun_probe(a.out, a.per_host, a.seed)
        return 0
    return run_gates(a)


if __name__ == "__main__":
    sys.exit(main())
