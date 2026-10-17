"""Statistics and output readers shared by the addendum-H evaluations (hnl.overlay.v1, hnl.overlay.h1;
docs/preregistration_pileup.md sections 6 and 9.2).

Paired event bootstrap. The index table is that of hnl.trackml.evaluate.bootstrap: default_rng(seed).integers(0, n,
size=(B, n)) over the n events of the set in increasing overlay id, B = 500, seed 12345. Every resampling sums the
per-event counts of all quantities on the same resampled events, so the efficiencies of a difference are paired. sigma
is the standard deviation of the resampled statistic with ddof = 1 (pandas' default, which evaluate.bootstrap uses;
resamplings whose statistic is undefined, an empty denominator, are skipped as pandas skips NaN).
`check_index_table` runs evaluate.bootstrap itself on the same counts and requires its bootstrap standard deviation of
one efficiency to equal the one of this index table, so the two cannot drift apart.

Model outputs. The frozen chain (hnl.trackml.trackbuild grid) writes per_target.parquet (event, config, particle_id,
dm, sm_any, k_sm_ge5: one row per target with hits) and configs.json. `wp_config` finds a working point of
configs/trackml/selection_v1.json (WP-A: mwslmerge 0.3; WP-S: mwslmerge 0.1; WP-R-P3: cc 0.8) by its family and
parameters, and `read_per_target` returns (event, particle_id, dm) at it.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from hnl.trackml.split import REPO

B, SEED = 500, 12345
SELECTION = os.path.join(REPO, "configs", "trackml", "selection_v1.json")


# ------------------------------------------------------------------------------------------------ bootstrap
def index_table(n: int, B: int = B, seed: int = SEED) -> np.ndarray:
    """The resampled event indices of hnl.trackml.evaluate.bootstrap (n events, B resamplings)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(B, n))


def std1(x) -> float:
    """Standard deviation with ddof = 1 over the finite values."""
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    return float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")


def bootstrap_stat(counts: np.ndarray, stat, B: int = B, seed: int = SEED) -> tuple[float, float, np.ndarray]:
    """counts (n events, k quantities) -> (stat of the summed counts, ddof-1 std over the B resamplings, the B values).
    stat maps an array (..., k) of summed counts to values (...)."""
    c = np.asarray(counts, np.float64)
    if c.ndim != 2 or not len(c):
        raise ValueError("counts must be (n events >= 1, k)")
    idx = index_table(len(c), B, seed)
    sums = c[idx].sum(axis=1)                                   # (B, k)
    with np.errstate(divide="ignore", invalid="ignore"):
        base = float(stat(c.sum(axis=0)))
        boot = np.asarray(stat(sums), np.float64)
    return base, std1(boot), boot


def pct(m, n):
    """100 m / n (NaN for n = 0)."""
    m, n = np.asarray(m, np.float64), np.asarray(n, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(n > 0, 100.0 * m / np.where(n > 0, n, 1.0), np.nan)


def paired_delta(n, m_a, m_b, B: int = B, seed: int = SEED) -> dict:
    """Delta = eps_a - eps_b [points] on the same targets (per-event counts n, m_a, m_b), its bootstrap sigma and both
    efficiencies, unrounded."""
    c = np.stack([np.asarray(n), np.asarray(m_a), np.asarray(m_b)], 1)
    stat = lambda s: pct(s[..., 1], s[..., 0]) - pct(s[..., 2], s[..., 0])          # noqa: E731
    d, sd, boot = bootstrap_stat(c, stat, B, seed)
    return {"n": int(c[:, 0].sum()), "eps_a": float(pct(c[:, 1].sum(), c[:, 0].sum())),
            "eps_b": float(pct(c[:, 2].sum(), c[:, 0].sum())), "delta": d, "sigma": sd,
            "n_events": int(len(c)), "B": B, "seed": seed}


def check_index_table(n, m, B: int = B, seed: int = SEED) -> dict:
    """hnl.trackml.evaluate.bootstrap on the counts (m_target = m, n_target = n per event) against this module's index
    table: the bootstrap standard deviations of the efficiency m / n must agree (to rounding)."""
    from hnl.trackml.evaluate import bootstrap
    frame = pd.DataFrame({"m_target": np.asarray(m, np.int64), "n_target": np.asarray(n, np.int64),
                          "dup_sum_targets": 0, "m_target_sm3": 0})       # the other ratios over n_target in pooled()
    ref = float(bootstrap(frame, B, seed).loc["ours_eff_target", "boot_std"])
    c = np.stack([np.asarray(n), np.asarray(m)], 1)
    _, mine, _ = bootstrap_stat(c, lambda s: s[..., 1] / s[..., 0], B, seed)
    same = bool(np.isclose(ref, mine, rtol=1e-9, atol=1e-15) or (np.isnan(ref) and np.isnan(mine)))
    return {"evaluate_bootstrap_std": ref, "index_table_std": mine, "equal": same}


# ------------------------------------------------------------------------------------------------ outputs
def wp_params(wp: str = "WP-A", selection: str = SELECTION) -> tuple:
    """(family, *params) of a frozen working point: WP-A, WP-S or WP-R-P1..P4."""
    with open(selection) as fh:
        sel = json.load(fh)["selection"]
    d = sel["WP-R"][wp.split("-")[-1]] if wp.startswith("WP-R-") else sel[wp]
    return (d["family"], *[float(x) for x in str(d["params"]).split(",") if x != ""])


def wp_config(grid_dir: str, wp: str = "WP-A", selection: str = SELECTION) -> int:
    """Index of a working point in a grid's configs.json (exactly one configuration with its family and parameters)."""
    want = wp_params(wp, selection)
    with open(os.path.join(grid_dir, "configs.json")) as fh:
        cfgs = json.load(fh)
    hit = [i for i, c in enumerate(cfgs) if c[0] == want[0] and len(c) == len(want)
           and all(float(x) == y for x, y in zip(c[1:], want[1:]))]
    if len(hit) != 1:
        raise LookupError(f"{grid_dir}: {len(hit)} configurations equal {wp} {want}")
    return hit[0]


def read_per_target(grid_dir: str, wp: str = "WP-A", selection: str = SELECTION) -> pd.DataFrame:
    """(event, particle_id, dm) of a grid at a working point; every (event, particle_id) once."""
    i = wp_config(grid_dir, wp, selection)
    t = pd.read_parquet(os.path.join(grid_dir, "per_target.parquet"), columns=["event", "config", "particle_id", "dm"],
                        filters=[("config", "==", i)])
    t = t[["event", "particle_id", "dm"]].astype({"event": np.int64, "particle_id": np.int64, "dm": bool})
    if t.duplicated(["event", "particle_id"]).any():
        raise ValueError(f"{grid_dir}: a target appears twice at {wp}")
    return t.reset_index(drop=True)


def attach_dm(rows: pd.DataFrame, out: pd.DataFrame, event_col: str, pid_col: str, name: str,
              required) -> pd.Series:
    """The dm flag of each row from an output table (event, particle_id, dm). Rows where `required` is True must be
    present in the outputs (a target with hits always has a row); others without a row are False."""
    key = pd.MultiIndex.from_arrays([out.event.to_numpy(np.int64), out.particle_id.to_numpy(np.int64)])
    s = pd.Series(out.dm.to_numpy(bool), index=key)
    got = s.reindex(pd.MultiIndex.from_arrays([rows[event_col].to_numpy(np.int64), rows[pid_col].to_numpy(np.int64)]))
    miss = got.isna().to_numpy() & np.asarray(required, bool)
    if miss.any():
        raise LookupError(f"{name}: {int(miss.sum())} targets without an output row (first: "
                          f"{rows[miss][[event_col, pid_col]].head(3).to_dict('records')})")
    return pd.Series(got.fillna(False).to_numpy(bool), index=rows.index)
