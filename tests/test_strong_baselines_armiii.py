"""Pure helpers of hnl.eval.strong_baselines_armiii (feedback round 2, task E): the val-best row of an arm, the
difference pairs and their blocks, and the grid-edge flags behind round 1's lower-edge rule."""

import pandas as pd
import pytest

from hnl.eval import strong_baselines_armiii as E


def _tab(rows):
    return pd.DataFrame(rows, columns=["label", "eff_primary", "fake_rate_hnl"])


def test_val_best_highest_primary_then_lower_fake_rate():
    t = _tab([("acorn (HNL) + CC, best cut <= FR*", 0.956, 0.029),
              ("acorn (HNL) + walkthrough + merging (Canto builder family)", 0.9857, 0.0154),
              ("acorn (HNL) + MWS-layer + merging (Canto builder)", 0.9857, 0.0143),
              ("Canto-L (frozen: learned merging)", 0.991, 0.010)])
    assert E.val_best(t, "acorn (HNL)") == "acorn (HNL) + MWS-layer + merging (Canto builder)"


def test_val_best_does_not_mix_arms():
    t = _tab([("acorn (ttbar+HNL) + MWS-layer + merging (Canto builder)", 0.99, 0.014),
              ("acorn (ttbar) + MWS-layer + merging (Canto builder)", 0.9677, 0.0166),
              ("acorn (ttbar) + CC, preregistered cut (= best cut)", 0.9231, 0.0297)])
    assert E.val_best(t, "acorn (ttbar)") == "acorn (ttbar) + MWS-layer + merging (Canto builder)"
    assert E.val_best(t, "acorn (ttbar+HNL)") == "acorn (ttbar+HNL) + MWS-layer + merging (Canto builder)"
    with pytest.raises(ValueError):
        E.val_best(t, "acorn (HNL)")


def test_pairs_blocks_and_duplicates():
    hnl_best = "acorn (HNL) + walkthrough + merging (Canto builder family)"
    ttbar_best = "acorn (ttbar) + MWS-layer + merging (Canto builder)"
    iii = [f"{E.ALAB} + {E.BLAB[k]}" for k in ("walkmerge", "mwsm")]
    prs = E.pairs(iii, hnl_best, ttbar_best)
    keys = [(a, b) for a, b, _ in prs]
    assert len(keys) == len(set(keys))                                   # every pair once
    blk = {(a, b): k.split("; ") for a, b, k in prs}
    for x in iii:
        assert blk[(E.NEW, x)] == [E.BLOCKS[0]]
    # walkthrough + merging of arm (iii) against acorn (HNL)'s val-best row (also walkthrough + merging): both blocks
    assert blk[(iii[0], hnl_best)] == [E.BLOCKS[1], E.BLOCKS[3]]
    assert blk[(iii[1], ttbar_best)] == [E.BLOCKS[2], E.BLOCKS[4]]
    assert blk[(iii[1], "acorn (HNL) + MWS-layer + merging (Canto builder)")] == [E.BLOCKS[3]]
    assert blk[(iii[0], "acorn (ttbar) + walkthrough + merging (Canto builder family)")] == [E.BLOCKS[4]]
    assert blk[(E.NEW, hnl_best)] == [E.BLOCKS[5]] and blk[(E.NEW, ttbar_best)] == [E.BLOCKS[5]]
    # no same-builder pair for a builder that arm (iii) does not have in the list
    assert not any(b.startswith("acorn (HNL) + acorn's CC") for _, b in keys)


def test_pairs_cc_rows_same_builder():
    iii = [f"{E.ALAB} + CC, best cut <= FR*", f"{E.ALAB} + CC, preregistered cut rule"]
    prs = {(a, b): k for a, b, k in E.pairs(iii, "acorn (HNL) + x", "acorn (ttbar) + y")}
    assert prs[(iii[0], "acorn (HNL) + CC, best cut <= FR*")] == E.BLOCKS[3]
    assert prs[(iii[1], "acorn (HNL) + CC, preregistered cut (claims' reference)")] == E.BLOCKS[3]
    # acorn (ttbar) has one CC row (its preregistered cut is its best cut): both arm (iii) CC rows are compared with it
    assert prs[(iii[0], "acorn (ttbar) + CC, preregistered cut (= best cut)")] == E.BLOCKS[4]
    assert prs[(iii[1], "acorn (ttbar) + CC, preregistered cut (= best cut)")] == E.BLOCKS[4]


ACW = ([["acwalk", 0.01, m, 0.6] for m in (0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6)]
       + [["acwalk", 0.01, m, a] for m in (0.05, 0.1, 0.2) for a in (0.4, 0.8)]
       + [["acwalk", c, c, 0.6] for c in (0.1, 0.3)]
       + [["acwalk", c, c, 0.6] for c in (0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.0)]
       + [["acwalkg", 0.01, 0.02, 0.6], ["acwalkg", 0.5, 0.9, 0.9]])


def test_edge_flags_acwalk_grid():
    assert len([c for c in ACW if c[0] == "acwalk"]) == 27
    assert E.edge_flags(ACW, "acwalk", [0.01, 0.15, 0.6]) == []                     # interior in every parameter
    assert E.edge_flags(ACW, "acwalk", [0.0, 0.0, 0.6]) == [("score_cut_cc", 0.0, "lowest"), ("walk min", 0.0, "lowest")]
    assert E.edge_flags(ACW, "acwalk", [0.01, 0.6, 0.6]) == [("walk min", 0.6, "highest")]
    assert E.edge_flags(ACW, "acwalk", [0.01, 0.2, 0.8]) == [("walk add", 0.8, "highest")]
    # the other family's configurations do not enter (acwalkg here has walk min up to 0.9)
    assert E.edge_flags(ACW, "acwalkg", [0.5, 0.9, 0.9]) == [("score_cut_cc", 0.5, "highest"),
                                                             ("walk min", 0.9, "highest"),
                                                             ("walk add", 0.9, "highest")]


def test_edge_flags_fixed_parameter_is_not_an_edge():
    cfgs = [["walkmerge", m, a, 1.0] for m in (0.1, 0.2, 0.3) for a in (0.5, 0.7)] + [["mwslayermerge", s] for s in
                                                                                     (0.1, 0.2, 0.3)]
    assert E.edge_flags(cfgs, "walkmerge", [0.2, 0.5, 1.0]) == [("walk add", 0.5, "lowest")]
    assert E.edge_flags(cfgs, "mwslayermerge", [0.1]) == [("s0", 0.1, "lowest")]
    assert E.edge_flags(cfgs, "mwslayermerge", [0.2]) == []


def test_lower_edge_rule():
    assert E.lower_edge_rule("acwalk", []) == "interior (no extension needed)"
    assert E.lower_edge_rule("acwalkg", [("walk add", 0.8, "highest")]) == "interior (no extension needed)"
    assert "no lower value exists" in E.lower_edge_rule("acwalk", [("walk min", 0.0, "lowest")])
    assert "extend the grid downwards" in E.lower_edge_rule("acwalkg", [("walk min", 0.02, "lowest")])
    assert E.lower_edge_rule("walkmerge", [("walk min", 0.1, "lowest")]) == ""
    assert E.lower_edge_rule("mwslayermerge", [("s0", 0.1, "lowest")]) == ""


def _spec_files(tmp_path, entry):
    import json
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"models": [{"label": "row", "entries": [entry]}]}))
    sub = tmp_path / "out"
    sub.mkdir()
    return spec, sub


def test_check_spec_of_output(tmp_path):
    import json
    e = {"run": "/runs/armiii_s0", "sub": str(tmp_path / "out"), "cut": 0, "config": "walkmerge,0.2,0.5,1.0"}
    spec, sub = _spec_files(tmp_path, e)
    prov = {"spec": str(spec), "spec_sha256": E._sha256(str(spec))}
    (sub / "provenance.json").write_text(json.dumps(prov))
    E._check_spec_of_output(e, "row")                                  # same entry, spec unchanged: passes
    with pytest.raises(SystemExit):                                    # another configuration than the job's spec
        E._check_spec_of_output({**e, "config": "walkmerge,0.1,0.5,1.0"}, "row")
    with pytest.raises(SystemExit):                                    # another row label
        E._check_spec_of_output(e, "other row")
    spec.write_text(spec.read_text() + " ")                            # the spec changed after the job: checksum
    with pytest.raises(SystemExit):
        E._check_spec_of_output(e, "row")
