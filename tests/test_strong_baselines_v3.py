"""Pure helpers of hnl.eval.strong_baselines_v3 (feedback round 3, task A2): the training-graph draw of the learned
merging, the edge check and the edge-extension rule, the configuration lists, the full-menu tie rule and the
difference pairs."""

import os

import numpy as np
import pandas as pd
import pytest

from hnl.eval import strong_baselines_v3 as S


def test_pick_graphs_is_the_score_split_sample_draw():
    names = [f"event{c}{i:08d}.pyg.gz" for c in (1, 2, 4) for i in range(0, 900, 7)]
    rng_names = sorted(names)
    ref = [rng_names[k] for k in np.sort(np.random.default_rng(0).choice(len(rng_names), 50, replace=False))]
    assert S.pick_graphs(names[::-1], 50) == ref                    # order of the listing does not matter
    assert S.pick_graphs(names, 10_000) == sorted(names)            # all graphs when n >= their number
    mixed = S.pick_graphs(names, 120)
    assert {x[5] for x in mixed} == {"1", "2", "4"}                 # a random draw covers every sample of a mix


def test_next_values_linear_and_125():
    assert S.next_values([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], "lowest") == [0.05, 0.02]       # linear step reaches 0
    assert S.next_values([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], "highest") == [0.7, 0.8]
    assert S.next_values([0.2, 0.3, 0.4, 0.5], "lowest") == [0.1, 0.05]
    assert S.next_values([0.02, 0.05, 0.1], "highest") == [0.2, 0.5]                     # 1-2-5 grid
    assert S.next_values([0.02, 0.05, 0.1], "lowest") == [0.01, 0.005]
    assert S.next_values([0.5, 0.7], "lowest") == [0.3, 0.1]
    assert S.next_values([0.5, 0.7], "highest") == [0.9, 1.0]                            # stops at the range end
    taus = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
    assert S.next_values(taus, "lowest") == [0.002, 0.001]
    assert S.next_values(taus, "highest") == [1.0]
    assert S.next_values([0.0001, 0.0002], "lowest", 3) == [0.0]                         # 0 ends the range
    assert S.next_values([0.0, 0.01], "lowest") == []
    assert S.next_values([0.5, 1.0], "highest") == []


def test_constrain_family_ranges():
    assert S.constrain("walkmerge", ("walkmerge", 0.4, 0.3, 1.0)) == ("walkmerge", 0.4, 0.4, 1.0)
    assert S.constrain("acwalk", ("acwalk", 0.01, 0.005, 0.2)) == ("acwalk", 0.01, 0.01, 0.2)
    acw = ([("acwalk", 0.01, m, 0.6) for m in (0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6)]
           + [("acwalk", 0.01, m, a) for m in (0.05, 0.1, 0.2) for a in (0.4, 0.8)])
    assert S.extension(acw, "acwalk", [("walk add", 0.4, "lowest", 2)]) == [
        ("acwalk", 0.01, 0.05, 0.2), ("acwalk", 0.01, 0.1, 0.2), ("acwalk", 0.01, 0.2, 0.2)]
    assert S.extension(acw + [("acwalk", 0.01, m, 0.2) for m in (0.05, 0.1, 0.2)], "acwalk",
                       [("walk add", 0.2, "lowest", 2)]) == []                  # the computable range ends
    assert S.constrain("acwalk", ("acwalk", 0.01, 0.2, 0.05)) == ("acwalk", 0.01, 0.2, 0.2)
    assert S.constrain("acwalk", ("acwalk", 0.01, 0.1, 0.05)) is None          # walk add 0.1: not computable
    assert S.constrain("acwalkg", ("acwalkg", 0.01, 0.05, 0.1)) is None
    assert S.constrain("mws2passmerge", ("mws2passmerge", 0.2, 0.2)) is None
    assert S.constrain("mws2passmerge", ("mws2passmerge", 0.3, 0.2)) == ("mws2passmerge", 0.3, 0.2)
    assert S.constrain("mwslayermerge", ("mwslayermerge", 0.0)) is None
    assert S.constrain("mwslmerge", ("mwslmerge", 0.0)) == ("mwslmerge", 0.0)
    assert S.constrain("mwslmerge", ("mwslmerge", 1.2)) is None


def test_edge_flags_global_range_and_fixed_parameters():
    cfgs = [["walkmerge", m, a, 1.0] for m in (0.1, 0.2, 0.3, 0.4, 0.5) for a in (0.5, 0.7) if a >= m]
    fl = S.edge_flags(cfgs, "walkmerge", [0.2, 0.5, 1.0])
    assert fl == [("walk add", 0.5, "lowest", 1)]                   # rms 1.0: one value, fixed, not an edge
    assert S.edge_flags(cfgs, "walkmerge", [0.3, 0.7, 1.0]) == [("walk add", 0.7, "highest", 1)]
    assert S.edge_flags(cfgs, "walkmerge", [0.5, 0.7, 1.0]) == [("walk min", 0.5, "highest", 0),
                                                                  ("walk add", 0.7, "highest", 1)]


def test_extension_slices_and_ranges():
    cfgs = [["walkmerge", m, a, 1.0] for m in (0.1, 0.2, 0.3, 0.4, 0.5) for a in (0.5, 0.7) if a >= m]
    low = S.extension(cfgs, "walkmerge", [("walk add", 0.5, "lowest", 1)])
    assert set(low) == {("walkmerge", 0.1, 0.3, 1.0), ("walkmerge", 0.2, 0.3, 1.0), ("walkmerge", 0.3, 0.3, 1.0),
                        ("walkmerge", 0.4, 0.4, 1.0), ("walkmerge", 0.1, 0.1, 1.0), ("walkmerge", 0.2, 0.2, 1.0)}
    assert len(low) == len(set(low))
    high = S.extension(cfgs, "walkmerge", [("walk add", 0.7, "highest", 1)])
    assert len(high) == 10 and {c[2] for c in high} == {0.9, 1.0}
    two = [["mws2passmerge", a, b] for a in (0.2, 0.3, 0.4, 0.5) for b in (0.02, 0.05, 0.1)]
    ext = S.extension(two, "mws2passmerge", [("s1", 0.2, "lowest", 0), ("s2", 0.1, "highest", 1)])
    assert set(ext) == {("mws2passmerge", 0.1, 0.02), ("mws2passmerge", 0.1, 0.05), ("mws2passmerge", 0.05, 0.02),
                        ("mws2passmerge", 0.3, 0.2), ("mws2passmerge", 0.4, 0.2), ("mws2passmerge", 0.5, 0.2)}
    # nothing new past the range end: an empty extension
    grid = [["mwslmerge", t] for t in (0.0, 0.001, 0.002)]
    assert S.extension(grid, "mwslmerge", [("tau", 0.0, "lowest", 0)]) == []
    # configurations already in the grid are not repeated
    grid2 = cfgs + [list(c) for c in low]
    assert S.extension(grid2, "walkmerge", [("walk add", 0.1, "lowest", 1)]) == []


def test_config_text_roundtrip():
    cfgs = [("acwalk", 0.01, 0.05, 0.2), ("mwslayermerge", 0.02), ("mws2passmerge", 0.3, 0.2)]
    assert S.parse_configs(S.config_text(cfgs)) == cfgs
    with pytest.raises(ValueError):
        S.parse_configs("mwslayermerge,0.05;mwslayermerge,0.05")


def test_menu_tie_rule():
    t = pd.DataFrame({"key": ["mws2pm", "walkmerge", "mwsm", "cc", "lmerge"],
                      "eff_primary": [0.98723, 0.98573, 0.98422, 0.96319, 0.98000],
                      "fake_rate_hnl": [0.01453, 0.01531, 0.01395, 0.02904, 0.01000],
                      "n_primary": [1331.0] * 5})
    r = S.menu_tie(t, 0.029368)
    se = np.sqrt(0.98723 * (1 - 0.98723) / 1331)
    assert r.tied.tolist() == [True, True, bool(0.98422 >= 0.98723 - se), False, False]
    assert r.loc[r.chosen, "key"].item() == "mwsm"                  # tied, lowest fake rate
    t2 = t.assign(fake_rate_hnl=[0.01453, 0.01531, 0.01395, 0.02904, 0.0300])  # above FR*: not eligible
    t2.loc[4, "eff_primary"] = 0.999
    r2 = S.menu_tie(t2, 0.029368)
    assert not r2.eligible[4] and r2.loc[r2.chosen, "key"].item() == "mwsm"
    with pytest.raises(ValueError):
        S.menu_tie(t.assign(fake_rate_hnl=0.05), 0.029368)


def test_pairs_blocks():
    labs = [S.NEW] + [S.family_label(a, k) for a in ("i", "ii", "iii") for k, *_ in S.FAMILIES]
    labs += [S.family_label(a, "best") for a in ("i", "ii", "iii")] + [S.CC_ROWS["i"][0][0], S.CANTO_B_CC]
    labs += [S.canto_label("B", k) for k in S.CANTO_FAMILIES]
    prs = S.pairs(labs)
    keys = [(a, b) for a, b, _ in prs]
    assert len(keys) == len(set(keys))
    blocks = {(a, b): k for a, b, k in prs}
    assert blocks[(S.NEW, S.family_label("ii", "best"))] == "Canto-L minus acorn"
    assert blocks[(S.family_label("iii", "best"), S.family_label("ii", "best"))] == "val-best against val-best"
    assert blocks[(S.CANTO_B_CC, S.CC_ROWS["i"][0][0])].startswith("decomposition, classifier first")
    assert blocks[(S.CANTO_B_FROZEN, S.CANTO_B_CC)].startswith("decomposition, classifier first")
    assert blocks[(S.family_label("i", "mwsm"), S.CC_ROWS["i"][0][0])].startswith("decomposition, builder first")
    assert blocks[(S.canto_label("B", "mwsm"), S.family_label("i", "mwsm"))].startswith("decomposition, builder first")
    # every acorn row has its Canto-L difference, nothing compares Canto-L with itself
    assert all((S.NEW, x) in blocks for x in labs if x.startswith("acorn ("))
    assert not any(a == b for a, b in keys)


def test_labels_keep_rounds_1_2():
    assert S.family_label("ii", "walkmerge") == "acorn (HNL) + walkthrough + merging (Canto builder family)"
    assert S.family_label("i", "mwsm") == "acorn (ttbar) + MWS-layer + merging (Canto builder)"
    assert S.family_label("iii", "acwalkg") == ("acorn (ttbar+HNL) + acorn's CC + walkthrough, score-guided "
                                               "(CCandWalk)")
    assert S.canto_label("L", "lmerge") == S.NEW


def test_ext_parts_order(tmp_path):
    run = tmp_path / "armi_s0"
    for g in ("trackbuild_val_fb3_ext10_o", "trackbuild_val_fb3_ext2_w", "trackbuild_val_fb3_ext1_w",
              "trackbuild_val_fb3_ext1_o", "trackbuild_val_fb3_ext3_x"):
        (run / g).mkdir(parents=True)
        if g != "trackbuild_val_fb3_ext3_x":                         # no configs.json: an unfinished part is ignored
            (run / g / "configs.json").write_text("[]")
    assert S.ext_parts(str(run)) == ["trackbuild_val_fb3_ext1_o", "trackbuild_val_fb3_ext1_w",
                                     "trackbuild_val_fb3_ext2_w", "trackbuild_val_fb3_ext10_o"]
    assert os.path.basename(S.merge_model(str(run))) == "model.joblib"
