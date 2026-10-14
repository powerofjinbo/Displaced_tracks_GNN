"""Module geometry of the z-mirror patched ACTS v15 GenericDetector against the TrackML detectors.csv (gate G0), and the
ACTS -> TrackML module-id table of the overlay (docs/pileup_overlay_plan.md, sections 3 and 10).

ACTS side: the per-surface dump of scripts/acts/dump_generic_modules.cpp (centre, rotation, bounds, thickness, pitch;
slurm/acts/dump_generic_modules.sbatch), or a CsvTrackingGeometryWriter detectors.csv, which has no thickness or pitch.
TrackML side: external/trackml/detectors.csv (Zenodo 4730167). Both describe a module by its centre c, the rotation
R = [u v w] (columns in global coordinates) and the outline: module_minhu at v = -hv, module_maxhu at v = +hv, and
module_hv. ACTS TrapezoidBounds use the same convention (halfLengthXnegY at y = -hy).

Gate G0 (section 10): for every module the centre, the plane and the four corners agree within 4 um, and the id table
is a bijection within every (volume, layer).
- centre: |c_acts - c_trackml|.
- plane: the largest distance of either module's corners from the other module's plane.
- corners: the largest distance between corresponding corners (same local position). A frame turned by 180 degrees
  about w leaves a rectangle in place but moves its corners, so the unordered corner-set distance is reported as well.
The table pairs every ACTS module with the TrackML module of nearest centre in the same (volume, layer).

    python -m hnl.overlay.modules g0 --acts DUMP.csv --out DIR [--writer-csv detectors.csv] [--tol-um 4]
                                     [--map-out configs/trackml/module_map_acts_zmirror_v1.csv --patch P --build B]
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from hnl.geometry.ids import decode_geometry_id

TRACKML_DETECTORS = "/ceph/submit/data/user/j/jinboz1/hnl_lrt/external/trackml/detectors.csv"
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODULE_MAP = os.path.join(_REPO, "configs", "trackml", "module_map_acts_zmirror_v1.csv")
ROT = ["rot_xu", "rot_xv", "rot_xw", "rot_yu", "rot_yv", "rot_yw", "rot_zu", "rot_zv", "rot_zw"]
KEY = ["volume_id", "layer_id", "module_id"]
MAP_COLUMNS = ["volume_id", "layer_id", "acts_module_id", "trackml_module_id"]
G0_TOL_MM = 0.004
N_MODULES = 18728                     # sensitive modules of TrackML and of the GenericDetector (plan section 3)
BARREL_VOLUMES = (8, 13, 17)
BOUNDS_RECTANGLE, BOUNDS_TRAPEZOID = 6, 7   # Acts::SurfaceBounds::BoundsType in v15


def module_key(volume, layer, module) -> np.ndarray:
    """(volume, layer, module) packed as an ACTS v15 sensitive geometry id (volume << 56 | layer << 36 | module)."""
    v, l, m = (np.asarray(x, dtype=np.int64) for x in (volume, layer, module))
    if (v < 0).any() or (v > 0xFF).any() or (l < 0).any() or (l > 0xFFF).any() or (m < 0).any() or (m > 0x0FFFFFFF).any():
        raise ValueError("volume, layer or module out of the geometry-id range")
    return (v << 56) | (l << 36) | m


def read_trackml(path: str = TRACKML_DETECTORS) -> pd.DataFrame:
    d = pd.read_csv(path)
    missing = set(KEY + ["cx", "cy", "cz"] + ROT + ["module_minhu", "module_maxhu", "module_hv"]) - set(d.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return d.reset_index(drop=True)


def acts_table(d: pd.DataFrame) -> pd.DataFrame:
    """Sensitive rows of an ACTS dump with the outline in TrackML terms (module_minhu, module_maxhu, module_hv)."""
    d = d[d.module_id > 0].copy()
    if "module_minhu" in d.columns:                 # dump_generic_modules layout
        bad = ~d.bounds.isin(["rectangle", "trapezoid"])
        if bad.any():
            raise ValueError(f"{int(bad.sum())} sensitive surfaces with bounds other than centred rectangles or trapezoids")
        return d.reset_index(drop=True)
    bt = d.bounds_type.to_numpy()                   # CsvTrackingGeometryWriter layout
    if not np.isin(bt, (BOUNDS_RECTANGLE, BOUNDS_TRAPEZOID)).all():
        raise ValueError("sensitive surfaces with bounds other than rectangles or trapezoids")
    p = [d[f"bound_param{i}"].to_numpy(float) for i in range(4)]
    rect = bt == BOUNDS_RECTANGLE
    if (rect & ~(np.isclose(p[0], -p[2]) & np.isclose(p[1], -p[3]))).any():
        raise ValueError("off-centre rectangle bounds")
    d["module_minhu"] = np.where(rect, p[2], p[0])  # rectangle: minX, minY, maxX, maxY; trapezoid: hxNegY, hxPosY, hy
    d["module_maxhu"] = np.where(rect, p[2], p[1])
    d["module_hv"] = np.where(rect, p[3], p[2])
    return d.reset_index(drop=True)


def read_acts(path: str) -> pd.DataFrame:
    return acts_table(pd.read_csv(path))


def rotations(d: pd.DataFrame) -> np.ndarray:
    """(n, 3, 3): rows x, y, z; columns u, v, w."""
    return d[ROT].to_numpy(float).reshape(-1, 3, 3)


def centres(d: pd.DataFrame) -> np.ndarray:
    return d[["cx", "cy", "cz"]].to_numpy(float)


def corners(d: pd.DataFrame) -> np.ndarray:
    """(n, 4, 3) global corners at local (-minhu, -hv), (minhu, -hv), (maxhu, hv), (-maxhu, hv)."""
    mn, mx, hv = (d[c].to_numpy(float) for c in ("module_minhu", "module_maxhu", "module_hv"))
    lu = np.stack([-mn, mn, mx, -mx], axis=1)
    lv = np.stack([-hv, -hv, hv, hv], axis=1)
    R = rotations(d)
    return centres(d)[:, None, :] + lu[:, :, None] * R[:, None, :, 0] + lv[:, :, None] * R[:, None, :, 1]


def match_modules(trackml: pd.DataFrame, acts: pd.DataFrame) -> pd.DataFrame:
    """Pair every ACTS module with the TrackML module of nearest centre in its (volume, layer). If two ACTS modules
    pick the same TrackML module, the layer is assigned by the Hungarian method on the centre distance instead."""
    rows = []
    tgroups = {k: g for k, g in trackml.groupby(["volume_id", "layer_id"])}
    for (v, l), a in acts.groupby(["volume_id", "layer_id"]):
        t = tgroups.get((v, l))
        n = len(a)
        if t is None or len(t) == 0:
            rows.append(pd.DataFrame({"volume_id": v, "layer_id": l, "acts_module_id": a.module_id.to_numpy(),
                                      "trackml_module_id": -1, "d_centre_mm": np.inf, "d_second_mm": np.inf,
                                      "hungarian": False}))
            continue
        Pa, Pt = centres(a), centres(t)
        k = min(2, len(t))
        dist, idx = cKDTree(Pt).query(Pa, k=k)
        dist, idx = dist.reshape(n, k), idx.reshape(n, k)
        pick = idx[:, 0]
        hung = len(np.unique(pick)) != len(pick)
        if hung:
            C = np.linalg.norm(Pa[:, None, :] - Pt[None, :, :], axis=2)
            ri, ci = linear_sum_assignment(C)
            pick = np.full(n, -1)
            pick[ri] = ci
        ok = pick >= 0
        tid = np.where(ok, t.module_id.to_numpy()[np.where(ok, pick, 0)], -1)
        dc = np.where(ok, np.linalg.norm(Pa - Pt[np.where(ok, pick, 0)], axis=1), np.inf)
        rows.append(pd.DataFrame({"volume_id": v, "layer_id": l, "acts_module_id": a.module_id.to_numpy(),
                                  "trackml_module_id": tid, "d_centre_mm": dc,
                                  "d_second_mm": dist[:, 1] if k == 2 else np.inf, "hungarian": hung}))
    m = pd.concat(rows, ignore_index=True)
    return m.astype({"volume_id": np.int64, "layer_id": np.int64, "acts_module_id": np.int64,
                     "trackml_module_id": np.int64})


def bijection_report(match: pd.DataFrame, trackml: pd.DataFrame, acts: pd.DataFrame) -> dict:
    """Whether the table is a bijection between the ACTS and the TrackML modules within every (volume, layer)."""
    lay = ["volume_id", "layer_id"]
    unmatched_acts = int((match.trackml_module_id < 0).sum())
    good = match[match.trackml_module_id >= 0]
    dup_t = int(good.duplicated(lay + ["trackml_module_id"]).sum())
    dup_a = int(match.duplicated(lay + ["acts_module_id"]).sum())
    tk = set(module_key(trackml.volume_id, trackml.layer_id, trackml.module_id).tolist())
    used = set(module_key(good.volume_id, good.layer_id, good.trackml_module_id).tolist())
    ak = set(module_key(acts.volume_id, acts.layer_id, acts.module_id).tolist())
    mk = set(module_key(match.volume_id, match.layer_id, match.acts_module_id).tolist())
    rep = {"n_acts": int(len(acts)), "n_trackml": int(len(trackml)), "n_rows": int(len(match)),
           "unmatched_acts": unmatched_acts, "duplicate_trackml_targets": dup_t, "duplicate_acts_rows": dup_a,
           "trackml_modules_not_reached": len(tk - used), "acts_modules_missing_from_table": len(ak - mk),
           "layers_needing_hungarian": sorted(f"{v}_{l}" for (v, l), g in match.groupby(lay) if g.hungarian.any()),
           "barrel_identity": bool((good[good.volume_id.isin(BARREL_VOLUMES)].eval("acts_module_id == trackml_module_id")).all()),
           "same_id_frac_endcap": float(good[~good.volume_id.isin(BARREL_VOLUMES)].eval(
               "acts_module_id == trackml_module_id").mean()) if (~good.volume_id.isin(BARREL_VOLUMES)).any() else None}
    rep["bijective"] = bool(unmatched_acts == 0 and dup_t == 0 and dup_a == 0 and len(tk - used) == 0
                            and len(ak - mk) == 0 and len(acts) == len(trackml) == len(match))
    return rep


def compare(trackml: pd.DataFrame, acts: pd.DataFrame, match: pd.DataFrame) -> pd.DataFrame:
    """Per matched module: centre, plane and corner distances [mm], frame dots, bounds and (if dumped) thickness and
    pitch differences."""
    m = match[match.trackml_module_id >= 0].reset_index(drop=True)
    t = trackml.set_index(module_key(trackml.volume_id, trackml.layer_id, trackml.module_id)).loc[
        module_key(m.volume_id, m.layer_id, m.trackml_module_id)].reset_index(drop=True)
    a = acts.set_index(module_key(acts.volume_id, acts.layer_id, acts.module_id)).loc[
        module_key(m.volume_id, m.layer_id, m.acts_module_id)].reset_index(drop=True)
    Ct, Ca = centres(t), centres(a)
    Rt, Ra = rotations(t), rotations(a)
    Kt, Ka = corners(t), corners(a)
    out = m[["volume_id", "layer_id", "acts_module_id", "trackml_module_id"]].copy()
    out["d_centre_mm"] = np.linalg.norm(Ca - Ct, axis=1)
    # plane: corners of each module against the other module's plane
    pa = np.abs(np.einsum("nkj,nj->nk", Ka - Ct[:, None, :], Rt[:, :, 2])).max(axis=1)
    pt = np.abs(np.einsum("nkj,nj->nk", Kt - Ca[:, None, :], Ra[:, :, 2])).max(axis=1)
    out["d_plane_mm"] = np.maximum(pa, pt)
    out["d_corner_mm"] = np.linalg.norm(Ka - Kt, axis=2).max(axis=1)
    pair = np.linalg.norm(Ka[:, :, None, :] - Kt[:, None, :, :], axis=3)      # (n, 4 acts, 4 trackml)
    out["d_corner_set_mm"] = np.maximum(pair.min(axis=2).max(axis=1), pair.min(axis=1).max(axis=1))
    for k, ax in enumerate("uvw"):
        out[f"dot_{ax}"] = np.einsum("ni,ni->n", Rt[:, :, k], Ra[:, :, k])
    out["rot_max_element_diff"] = np.abs(Rt - Ra).reshape(-1, 9).max(axis=1)
    for c in ("module_minhu", "module_maxhu", "module_hv"):
        out[f"d_{c[7:]}_mm"] = a[c].to_numpy(float) - t[c].to_numpy(float)
    for c in ("module_t", "pitch_u", "pitch_v"):
        if c in a.columns and c in t.columns:
            out[f"d_{c}_mm"] = a[c].to_numpy(float) - t[c].to_numpy(float)
    return out


def g0_summary(cmp: pd.DataFrame, bij: dict, tol_mm: float = G0_TOL_MM) -> dict:
    """Gate G0: every module within tol_mm in centre, plane and corners, and a bijective table of N_MODULES rows."""
    ok_c, ok_p, ok_k = (cmp[c] <= tol_mm for c in ("d_centre_mm", "d_plane_mm", "d_corner_mm"))
    per_vol = {}
    for v, g in cmp.groupby("volume_id"):
        e = {"n": int(len(g))}
        for c in ("d_centre_mm", "d_plane_mm", "d_corner_mm", "d_corner_set_mm"):
            e[c.replace("_mm", "_um_max")] = float(1e3 * g[c].max())
            e[c.replace("d_", "n_fail_").replace("_mm", "")] = int((g[c] > tol_mm).sum())
        for ax in "uvw":
            e[f"dot_{ax}_min"] = float(g[f"dot_{ax}"].min())
        e["rot_max_element_diff"] = float(g.rot_max_element_diff.max())
        for c in [c for c in g.columns if c.startswith("d_") and c.endswith("_mm") and c not in
                  ("d_centre_mm", "d_plane_mm", "d_corner_mm", "d_corner_set_mm")]:
            e[c.replace("_mm", "_absmax_mm")] = float(g[c].abs().max())
        per_vol[int(v)] = e
    other = {}
    for c in ("d_minhu_mm", "d_maxhu_mm", "d_hv_mm", "d_module_t_mm", "d_pitch_u_mm", "d_pitch_v_mm"):
        if c in cmp.columns:
            other[c.replace("_mm", "_absmax_mm")] = float(cmp[c].abs().max())
    res = {
        "tolerance_mm": tol_mm,
        "n_compared": int(len(cmp)),
        "n_expected": N_MODULES,
        "max_um": {c.replace("_mm", ""): float(1e3 * cmp[c].max()) for c in
                   ("d_centre_mm", "d_plane_mm", "d_corner_mm", "d_corner_set_mm")},
        "n_fail": {"centre": int((~ok_c).sum()), "plane": int((~ok_p).sum()), "corner": int((~ok_k).sum()),
                   "any": int((~(ok_c & ok_p & ok_k)).sum())},
        "bijection": bij,
        "other_absmax": other,
        "per_volume": per_vol,
    }
    res["pass"] = bool(res["n_fail"]["any"] == 0 and bij["bijective"] and len(cmp) == N_MODULES)
    return res


def writer_cross_check(dump: pd.DataFrame, writer: pd.DataFrame) -> dict:
    """The dump against the CsvTrackingGeometryWriter detectors.csv of the same build: same modules, same numbers."""
    a = dump.set_index(module_key(dump.volume_id, dump.layer_id, dump.module_id)).sort_index()
    w = writer.set_index(module_key(writer.volume_id, writer.layer_id, writer.module_id)).sort_index()
    same = bool(a.index.equals(w.index))
    res = {"same_modules": same, "n_dump": int(len(a)), "n_writer": int(len(w))}
    if same:
        cols = ["cx", "cy", "cz"] + ROT + ["module_minhu", "module_maxhu", "module_hv"]
        res["max_abs_diff"] = float(np.abs(a[cols].to_numpy(float) - w[cols].to_numpy(float)).max())
    return res


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_module_map(match: pd.DataFrame, path: str) -> str:
    """Write the id table (sorted by volume, layer, ACTS module) and return its sha256."""
    t = match[["volume_id", "layer_id", "acts_module_id", "trackml_module_id"]]
    if (t.trackml_module_id < 0).any():
        raise ValueError("unmatched ACTS modules; refusing to write a partial table")
    t = t.sort_values(["volume_id", "layer_id", "acts_module_id"])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    t.to_csv(path, index=False, lineterminator="\n")
    return file_sha256(path)


class ModuleMap:
    """ACTS (volume, layer, sensitive) of the patched GenericDetector -> TrackML module id."""

    def __init__(self, table: pd.DataFrame):
        missing = set(MAP_COLUMNS) - set(table.columns)
        if missing:
            raise ValueError(f"module map: missing columns {sorted(missing)}")
        keys = module_key(table.volume_id, table.layer_id, table.acts_module_id)
        order = np.argsort(keys, kind="stable")
        self._keys = keys[order]
        if (np.diff(self._keys) == 0).any():
            raise ValueError("module map: repeated ACTS module")
        tkeys = module_key(table.volume_id, table.layer_id, table.trackml_module_id)
        if len(np.unique(tkeys)) != len(tkeys):
            raise ValueError("module map: repeated TrackML module (not a bijection)")
        self._trackml = table.trackml_module_id.to_numpy(np.int64)[order]
        self.table = table.reset_index(drop=True)

    @classmethod
    def from_csv(cls, path: str = MODULE_MAP, sha256: str | None = None) -> "ModuleMap":
        """Read the table. The sha256 is taken from the sidecar <path minus .csv>.json unless given."""
        if sha256 is None:
            side = os.path.splitext(path)[0] + ".json"
            if os.path.exists(side):
                with open(side) as f:
                    sha256 = json.load(f)["sha256"]
        if sha256 is not None and file_sha256(path) != sha256:
            raise ValueError(f"{path}: sha256 differs from the recorded {sha256}")
        return cls(pd.read_csv(path))

    def __len__(self) -> int:
        return len(self._keys)

    def trackml_module(self, volume, layer, acts_module) -> np.ndarray:
        """TrackML module ids; KeyError for an ACTS module not in the table."""
        k = module_key(volume, layer, acts_module)
        i = np.searchsorted(self._keys, k)
        i = np.minimum(i, len(self._keys) - 1)
        bad = self._keys[i] != k
        if np.any(bad):
            raise KeyError(f"{int(np.sum(bad))} ACTS modules not in the table, e.g. key {int(np.asarray(k)[bad].ravel()[0])}")
        return self._trackml[i]

    def from_geometry_id(self, geometry_id) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ACTS sensitive geometry ids (as in the Fatras sim-hit CSV) -> TrackML (volume, layer, module)."""
        d = decode_geometry_id(np.asarray(geometry_id, dtype=np.uint64).astype(np.int64))
        if (d["boundary"] != 0).any() or (d["approach"] != 0).any() or (d["sensitive"] == 0).any():
            raise ValueError("geometry ids that are not sensitive surfaces")
        return d["volume"], d["layer"], self.trackml_module(d["volume"], d["layer"], d["sensitive"])


def cmd_g0(a):
    for k in ("acts", "trackml", "writer_csv", "map_out", "patch", "build"):
        if getattr(a, k):
            setattr(a, k, os.path.abspath(getattr(a, k)))
    tml = read_trackml(a.trackml)
    acts = read_acts(a.acts)
    match = match_modules(tml, acts)
    bij = bijection_report(match, tml, acts)
    cmp = compare(tml, acts, match)
    res = {"inputs": {"trackml": a.trackml, "trackml_sha256": file_sha256(a.trackml), "acts": a.acts,
                      "acts_sha256": file_sha256(a.acts)},
           "match_d_centre_um_max": float(1e3 * match.d_centre_mm.max()),
           "match_margin_mm_min": float(match.d_second_mm.min()),
           "g0": g0_summary(cmp, bij, a.tol_um * 1e-3)}
    if a.writer_csv:
        res["writer_cross_check"] = writer_cross_check(acts, read_acts(a.writer_csv))
        res["inputs"]["writer_csv"] = a.writer_csv
    os.makedirs(a.out, exist_ok=True)
    cmp.to_csv(os.path.join(a.out, "g0_modules.csv"), index=False)
    match.to_csv(os.path.join(a.out, "match.csv"), index=False)
    if a.map_out:
        if not bij["bijective"]:
            raise SystemExit("the table is not a bijection; no map written")
        sha = write_module_map(match, a.map_out)
        side = {
            "file": os.path.basename(a.map_out), "sha256": sha, "rows": int(len(match)),
            "columns": MAP_COLUMNS,
            "meaning": ("ACTS v15 GenericDetector with the z-mirror patch (sensitive id = acts_module_id) -> TrackML "
                        "detectors.csv module id, within the same (volume, layer); nearest module centre"),
            "patch": a.patch, "patch_sha256": file_sha256(a.patch) if a.patch else None, "build": a.build,
            "acts_dump": a.acts, "acts_dump_sha256": res["inputs"]["acts_sha256"],
            "trackml_detectors": a.trackml, "trackml_detectors_sha256": res["inputs"]["trackml_sha256"],
            "g0_pass": res["g0"]["pass"], "g0_tolerance_mm": res["g0"]["tolerance_mm"], "g0_max_um": res["g0"]["max_um"],
            "bijective": bij["bijective"], "barrel_identity": bij["barrel_identity"],
            "endcap_same_id_frac": bij["same_id_frac_endcap"],
            "slurm_job": os.environ.get("SLURM_JOB_ID"), "created": datetime.date.today().isoformat(),
            "producer": "python -m hnl.overlay.modules g0 (src/hnl/overlay/modules.py)",
        }
        with open(os.path.splitext(a.map_out)[0] + ".json", "w") as f:
            json.dump(side, f, indent=1)
            f.write("\n")
        res["map"] = {"path": a.map_out, "sha256": sha}
    with open(os.path.join(a.out, "g0.json"), "w") as f:
        json.dump(res, f, indent=1)
    g = res["g0"]
    print(json.dumps({"pass": g["pass"], "max_um": g["max_um"], "n_fail": g["n_fail"], "bijective": bij["bijective"],
                      "other_absmax": g["other_absmax"], "writer_cross_check": res.get("writer_cross_check")}, indent=1))
    for v, e in g["per_volume"].items():
        print(v, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in e.items()})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("g0", help="gate G0 and the module-id table")
    g.add_argument("--acts", required=True, help="dump_generic_modules CSV (or a CsvTrackingGeometryWriter detectors.csv)")
    g.add_argument("--trackml", default=TRACKML_DETECTORS)
    g.add_argument("--writer-csv", default=None, help="CsvTrackingGeometryWriter detectors.csv of the same build")
    g.add_argument("--out", required=True)
    g.add_argument("--tol-um", type=float, default=G0_TOL_MM * 1e3)
    g.add_argument("--map-out", default=None)
    g.add_argument("--patch", default=None)
    g.add_argument("--build", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "g0":
        cmd_g0(a)


if __name__ == "__main__":
    main()
