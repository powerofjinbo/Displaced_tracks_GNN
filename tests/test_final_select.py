import numpy as np
import pandas as pd

from hnl.eval.final_select import choose


def _rows(pipeline, family, eff, dw=0.0, dp=0.0, seeds=(0, 1, 2), fr=0.02):
    return [{"pipeline": pipeline, "seed": s, "family": family, "config": f"{family},0.1", "eff_primary": eff,
             "eff_w_muons": 0.99, "eff_ttbar_prompt": 0.95, "fake_rate_hnl": fr,
             "d_eff_w_muons_vs_arm_i": dw, "d_eff_ttbar_prompt_vs_arm_i": dp, "d_eff_primary_vs_arm_i": 0.05,
             "d_eff_w_muons_vs_ref": dw, "d_eff_ttbar_prompt_vs_ref": dp, "d_eff_primary_vs_ref": 0.04,
             "llp_blind_config": f"{family},0.2", "n_primary": 100_000} for s in seeds]


def test_choose_applies_guard_r6_and_r8():
    tab = pd.DataFrame(_rows("base", "mwslayer", 0.980) + _rows("base", "mwslayermerge", 0.985)
                       + _rows("union", "mwslayer", 0.996, dp=-0.003)          # best, but fails the prompt guard
                       + _rows("union", "mws2passmerge", 0.994)
                       + _rows("union", "mwslmerge", 0.999, seeds=(0, 1)))      # not all seeds: never chosen
    m = choose(tab).set_index(["pipeline", "family"])
    assert m.loc[("base", "mwslayermerge"), "R6"] and not m.loc[("base", "mwslayer"), "R6"]
    assert m.loc[("union", "mws2passmerge"), "R6"]
    assert not m.loc[("union", "mwslayer"), "R6"] and not m.loc[("union", "mwslayer"), "guard_arm_i"]
    assert not m.loc[("union", "mwslmerge"), "R6"]
    assert m.R8.sum() == 1 and m.loc[("union", "mws2passmerge"), "R8"]
    assert np.isclose(m.loc[("union", "mws2passmerge"), "eff_primary"], 0.994)


def test_choose_ties_prefer_lower_fake_rate():
    tab = pd.DataFrame(_rows("p", "a", 0.99, fr=0.03) + _rows("p", "b", 0.99, fr=0.02))
    m = choose(tab).set_index(["pipeline", "family"])
    assert m.loc[("p", "b"), "R6"] and m.loc[("p", "b"), "R8"] and not m.loc[("p", "a"), "R6"]


def test_claims_spec_uses_frozen_cuts(tmp_path):
    import hashlib
    import json
    from hnl.eval.final_select import claims_spec
    wp = {"FR_star": 0.03, "e_star": 0.01, "reference_arm": "ii",
          "cuts_at_FR_star": {"i": {"0": 0.41, "1": 0.42}, "ii": {"0": 0.51, "1": 0.52}},
          "cuts_at_e_star": {"i": {"0": 0.61, "1": 0.62}, "ii": {"0": 0.71, "1": 0.72}}}
    p = tmp_path / "wp.json"
    p.write_text(json.dumps(wp))
    sel = {"working_points": str(p), "working_points_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
           "reference_arm": "ii", "created_utc": "x", "candidate": {"pipeline": "u", "family": "f"},
           "pipelines": {"u": {"seeds": {"0": {"run": "/r/u0", "cut_e": 0.3}, "1": {"run": "/r/u1", "cut_e": 0.4}}}}}
    s = claims_spec(sel, "/b")
    assert [m["cut"] for m in s["models"]["arm_i"]] == [0.41, 0.42]
    assert [m["cut_e"] for m in s["models"]["ref"]] == [0.71, 0.72] and s["models"]["ref"][0]["run"] == "/b/armii_s0"
    assert s["models"]["new"][1] == {"run": "/r/u1", "cut": 0, "sub": "trackbuild_test", "cut_e": 0.4,
                                     "sub_e": "eval_test"}
    assert s["dd"]["arm_i"] == ["/b/armi_s0/edge_tpr_test_events.parquet", "/b/armi_s1/edge_tpr_test_events.parquet"]
    assert s["claims"]["primary"] == {"new": "new", "base": "ref"}
    p.write_text(json.dumps({**wp, "FR_star": 0.04}))
    import pytest
    with pytest.raises(SystemExit, match="changed"):
        claims_spec(sel, "/b")
