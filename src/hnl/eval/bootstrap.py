"""Event bootstrap of the preregistered statistics (docs/preregistration.md §5).

One fixed table of resampled events per test set (B = 500, seed 12345), written once to
derived/bootstrap/v1/<set>.npz and reused by every model and seed, so that differences are paired. In the joint test
set (hnl10_100 and hnl15_200 share their underlying events by index) the resampled unit is the event index: both
samples' event i enter a replicate together. A pooled number over several sets sums numerators and denominators over
the sets, each set with its own resampling.

A model's number is the mean over its training seeds, with total uncertainty
    sigma^2 = sigma^2_boot(seed mean) + s^2_seeds / n_seeds;
a difference of two models uses the paired bootstrap of the difference of seed means plus both seed terms.
"""

from __future__ import annotations

import os

import numpy as np

from hnl.data.splits import read_split

B = 500
SEED = 12345
SETS = {
    "hnl_joint_test": (("hnl10_100", "test"), ("hnl15_200", "test")),
    "ttbar_test": (("ttbar", "test"),),
    "hnl15_100_external": (("hnl15_100", "external"),),
}
DEFAULT_DIR = os.path.join(os.environ.get("HNL_DATA", "/ceph/submit/data/user/j/jinboz1/hnl_lrt"), "derived",
                           "bootstrap", "v1")


def set_units(name: str) -> np.ndarray:
    """Sorted resampling units (event indices) of a test set: the union over its samples' split lists."""
    return np.unique(np.concatenate([read_split(s, sp) for s, sp in SETS[name]]))


def make_weights(n_units: int, b: int, rng: np.random.Generator) -> np.ndarray:
    """(b, n_units) multiplicities of each unit in each replicate (n_units draws with replacement)."""
    idx = rng.integers(0, n_units, size=(b, n_units))
    w = np.zeros((b, n_units), dtype=np.int32)
    for k in range(b):
        w[k] = np.bincount(idx[k], minlength=n_units)
    return w


class Bootstrap:
    """The fixed replicate tables. Creates them on first use (refuses to overwrite a table with different units)."""

    def __init__(self, directory: str = DEFAULT_DIR, b: int = B, seed: int = SEED):
        self.units, self.w = {}, {}
        children = np.random.SeedSequence(seed).spawn(len(SETS))
        for (name, _), ss in zip(SETS.items(), children):
            units = set_units(name)
            path = os.path.join(directory, f"{name}.npz")
            if os.path.exists(path):
                z = np.load(path)
                if not np.array_equal(z["units"], units) or z["w"].shape[0] != b:
                    raise RuntimeError(f"{path} does not match the current split / B")
                w = z["w"]
            else:
                w = make_weights(len(units), b, np.random.default_rng(ss))
                os.makedirs(directory, exist_ok=True)
                np.savez_compressed(path + ".tmp.npz", units=units, w=w, seed=seed, b=b)
                os.replace(path + ".tmp.npz", path)
            self.units[name], self.w[name] = units, w

    def per_unit(self, name: str, event_index: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Sum `values` (one per row, rows identified by their event index in the set) into the set's units."""
        units = self.units[name]
        pos = np.searchsorted(units, event_index)
        if np.any(pos >= len(units)) or np.any(units[np.minimum(pos, len(units) - 1)] != event_index):
            raise ValueError(f"event index outside test set {name}")
        return np.bincount(pos, weights=np.asarray(values, dtype=np.float64), minlength=len(units))

    def ratio(self, parts: list[tuple[str, np.ndarray, np.ndarray]]) -> tuple[float, np.ndarray]:
        """parts: (set name, numerator per unit, denominator per unit). Returns (point estimate, B replicates);
        NaN (point and replicates) for an empty population."""
        b = next(iter(self.w.values())).shape[0]
        num = den = 0.0
        rnum, rden = np.zeros(b), np.zeros(b)
        for s, n, d in parts:
            num += float(n.sum())
            den += float(d.sum())
            rnum += self.w[s].astype(np.float64) @ n
            rden += self.w[s].astype(np.float64) @ d
        with np.errstate(invalid="ignore", divide="ignore"):
            return (float(num / den) if den else np.nan), rnum / rden


def seed_mean(points: list[float], reps: list[np.ndarray]) -> dict:
    """A model's number (mean over seeds) and its total uncertainty."""
    p = np.asarray(points, dtype=float)
    r = np.mean(np.stack(reps), axis=0)
    s_boot = float(np.nanstd(r, ddof=1))
    s_seed = float(np.std(p, ddof=1)) if len(p) > 1 else 0.0
    return {"mean": float(p.mean()), "sigma_boot": s_boot, "s_seeds": s_seed, "n_seeds": len(p),
            "sigma": float(np.sqrt(s_boot ** 2 + s_seed ** 2 / len(p)))}


def linear_combination(terms: list[tuple[float, list[float], list[np.ndarray]]], const: float = 0.0) -> dict:
    """const + sum_k c_k * (seed mean of model k), paired over the bootstrap replicates; each model's seed spread
    enters as c_k^2 s_k^2 / n_k. Example: difference new - base = [(1, new...), (-1, base...)]."""
    val = const
    rep = np.full(len(terms[0][2][0]), const, dtype=float)
    seed_var = 0.0
    for c, points, reps in terms:
        p = np.asarray(points, dtype=float)
        val += c * p.mean()
        rep = rep + c * np.mean(np.stack(reps), axis=0)
        if len(p) > 1:
            seed_var += c ** 2 * np.var(p, ddof=1) / len(p)
    s_boot = float(np.nanstd(rep, ddof=1))
    sigma = float(np.sqrt(s_boot ** 2 + seed_var))
    return {"value": float(val), "sigma_boot": s_boot, "sigma_seeds": float(np.sqrt(seed_var)), "sigma": sigma,
            "z": float(val / sigma) if sigma > 0 else np.nan}


def primary_claim(new: tuple[list[float], list[np.ndarray]], base: tuple[list[float], list[np.ndarray]]) -> dict:
    """Preregistration §6 criteria 1-2 and the halving significance, from efficiencies (points, replicates)."""
    diff = linear_combination([(1.0, *new), (-1.0, *base)])
    # halving: 0.5 (1 - e_base) - (1 - e_new) = e_new - 0.5 e_base - 0.5
    halving = linear_combination([(1.0, *new), (-0.5, *base)], const=-0.5)
    e_new, e_base = float(np.mean(new[0])), float(np.mean(base[0]))
    return {"eff_new": e_new, "eff_base": e_base, "ineff_ratio": (1 - e_new) / (1 - e_base) if e_base < 1 else np.nan,
            "criterion1_halved": (1 - e_new) <= 0.5 * (1 - e_base), "difference": diff,
            "criterion2_3sigma": diff["z"] >= 3.0, "halving_margin": halving}
