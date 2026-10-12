"""Claims on the test sets (hnl.eval.claims): logic paths with stubbed test tables and bootstrap."""

import numpy as np
import pandas as pd
import pytest

from hnl.eval import claims as CL


class _BS:
    """Stand-in bootstrap: replicates = point + small noise (deterministic)."""
    def __init__(self, b=50):
        self.b = b


class _RT:
    def __init__(self, eff, fr, run):
        self.eff, self.fr, self.run, self.cut, self.cut_e = eff, fr, run, 0.0, None


def _stub(monkeypatch, effs):
    """effs: model name -> dict(pop -> list of per-seed efficiencies, 'fr' -> list)."""
    rng = np.random.default_rng(0)

    def efficiency(bs, runs, info, mask_fn, sets):
        pop = mask_fn.__name__
        pts = [rt.eff[pop] for rt in runs]
        return pts, [p + 0.002 * rng.standard_normal(bs.b) for p in pts], 100

    def event_ratio(bs, runs, num, den, sets, frame="pe", **k):
        pts = [rt.fr for rt in runs]
        return pts, [p + 0.0005 * rng.standard_normal(bs.b) for p in pts]

    def masks():
        def mk(name):
            f = lambda t: None
            f.__name__ = name
            return f
        return {k: mk(k) for k in ("primary", "w_muons", "ttbar_prompt")}
    monkeypatch.setattr(CL.TS, "efficiency", efficiency)
    monkeypatch.setattr(CL.TS, "event_ratio", event_ratio)
    monkeypatch.setattr(CL.TS, "masks", masks)
    monkeypatch.setattr(CL.BS, "Bootstrap", _BS)
    monkeypatch.setattr(CL, "particle_table", lambda ev: pd.DataFrame())

    def load_models(spec):
        out = {}
        for name, seeds in spec["models"].items():
            out[name] = []
            for k, _ in enumerate(seeds):
                e = {pop: v[k] for pop, v in effs[name].items() if pop != "fr"}
                rt = _RT(e, effs[name]["fr"][k], f"{name}_s{k}")
                rt.pp = pd.DataFrame({"event_id": [1]})
                out[name].append(rt)
        return out
    monkeypatch.setattr(CL, "load_models", load_models)

    def dd_seeds(bs, files):
        pts = [float(f) for f in files]
        return pts, [p + 0.003 * rng.standard_normal(bs.b) for p in pts]
    monkeypatch.setattr(CL, "dd_seeds", dd_seeds)


def test_claims_logic(monkeypatch):
    effs = {
        "ref": {"primary": [0.960, 0.941, 0.950], "w_muons": [0.987] * 3, "ttbar_prompt": [0.86] * 3, "fr": [0.027] * 3},
        "arm_i": {"primary": [0.936, 0.905, 0.923], "w_muons": [0.987] * 3, "ttbar_prompt": [0.87] * 3,
                  "fr": [0.030] * 3},
        "Q": {"primary": [0.989, 0.985, 0.987], "w_muons": [0.997] * 3, "ttbar_prompt": [0.95] * 3, "fr": [0.014] * 3},
        "P": {"primary": [0.973, 0.971, 0.970], "w_muons": [0.998] * 3, "ttbar_prompt": [0.96] * 3, "fr": [0.017] * 3},
        "P_blind": {"primary": [0.970, 0.968, 0.969], "w_muons": [0.998] * 3, "ttbar_prompt": [0.96] * 3,
                    "fr": [0.016] * 3},
    }
    _stub(monkeypatch, effs)
    spec = {"models": {k: [{}] * 3 for k in effs},
            "dd": {"P": ["0.005", "0.01", "-0.002"], "arm_i": ["-0.054", "-0.102", "-0.07"]},
            "claims": {"primary": {"new": "Q", "base": "ref"}, "S1": {"new": "P", "arm_i": "arm_i", "ref": "ref"},
                       "S1b": {"new": "P", "new_llp_blind": "P_blind", "arm_i": "arm_i"}}}
    res = CL.evaluate(spec)
    assert res["primary"]["criterion1_halved"] and res["primary"]["criterion3_prompt_guard"]["pass"]
    assert res["primary"]["difference"]["value"] == pytest.approx(0.987 - 0.9503333, abs=1e-6)
    assert res["S1"]["criterion3_prompt_guard"]["pass"] and res["S1"]["vs_arm_i"]["value"] > 0.04
    assert res["S1b"]["dD_difference"]["value"] == pytest.approx(0.0043333 - (-0.0753333), abs=1e-6)
    assert res["S1b"]["fake_rate_llp_blind"] < res["S1b"]["fake_rate_arm_i"]
    for k in ("primary", "S1", "S1b"):
        assert isinstance(res[k]["holds"], bool)
