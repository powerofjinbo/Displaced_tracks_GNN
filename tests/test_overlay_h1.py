"""hnl.overlay.h1 and hnl.overlay.hstats: gun targets and P_P by vertex field, regions and soft siblings, the pairing of
the two conditions, Delta and its paired event bootstrap (the index table of hnl.trackml.evaluate.bootstrap, B = 500,
seed 12345, ddof = 1), the outcome rule with delta = 1.0 on the unrounded values, the underpowered label, the forward
restriction, the reported numbers and the readers of the frozen grid outputs. Synthetic truth and truth-made
placeholder outputs only."""

import json

import numpy as np
import pandas as pd
import pytest

from hnl.overlay import h1 as H
from hnl.overlay import hstats as HS
from hnl.trackml import evaluate as EVAL


def gpid(llp, k, host_vertex=None):
    return ((1000 + llp) << 52) | (k << 36) if host_vertex is None else (host_vertex << 52) | (k << 36)


def _truth(n_events=12, seed=5):
    """Per event: LLPs 1-8 isolated with two prongs (one displaced target each, the other a target or a soft sibling),
    LLPs 9-10 dense, and host particles (vertex 1, 3) that must never enter."""
    rng = np.random.default_rng(seed)
    P, Hh = [], []
    for i in range(n_events):
        ev = 52_023_000 + i
        for llp in range(1, 11):
            for k, (pt, d0, rv) in enumerate(((5.0, 30.0, 200.0), (0.7 if llp % 4 == 0 else 3.0, 40.0, 200.0)), 1):
                pid = gpid(llp, k)
                nh = 6 if pt > 1 else (2 if llp % 8 == 0 else 0)
                eta = rng.uniform(-2.5, 2.5)
                P.append({"event": ev, "particle_id": pid, "pt": pt, "eta": eta, "d0": d0 * (-1) ** k, "rv": rv,
                          "n_hits": nh, "is_target": pt > 1 and nh >= 5, "r_first": 250.0})
                for j in range(nh):
                    Hh.append({"event": ev, "pid": pid, "z": (1500.0 if (j == nh - 1 and llp in (2, 9)) else 10.0)
                               * np.sign(eta)})
        for k in range(3):
            P.append({"event": ev, "particle_id": gpid(0, k + 1, host_vertex=1 + 2 * (k % 2)), "pt": 4.0, "eta": 0.1,
                      "d0": 50.0, "rv": 300.0, "n_hits": 8, "is_target": True, "r_first": 320.0})
    return pd.DataFrame(P), pd.DataFrame(Hh)


@pytest.fixture(scope="module")
def g():
    p, h = _truth()
    return H.gun_targets(p, h)


def test_gun_targets_and_flags(g):
    assert len(g) == 12 * 20 and ((g.particle_id.to_numpy() >> 52) >= 1001).all()   # host particles never enter
    assert g.groupby("event").pp.sum().eq(8 + 6).all()                       # prong 2 is soft for LLPs 4 and 8
    assert g.groupby("event").pd.sum().eq(4).all()
    sib = g[g.pp & g.soft_sibling]
    assert set(sib.llp) == {8} and ((sib.particle_id.to_numpy() & ((1 << 52) - 1)) == 1 << 36).all()  # prong 1
    assert set(g[g.pp & (g.region != "central")].llp) == {2}
    fwd = g[g.llp == 2]
    assert (fwd.region == np.where(fwd.eta > 0, "fwd+", "fwd-")).all()
    r = H.restrict(g, ("central",))
    assert r.pp.sum() == g.pp.sum() - 12 * 2 and H.restrict(g, None) is g
    with pytest.raises(ValueError):
        H.restrict(g, ("forward",))


def _outs(g, miss_full=(), miss_1gev=()):
    t = g[g.target]
    key = list(zip(t.event, t.particle_id))
    return {"full": pd.DataFrame({"event": t.event, "particle_id": t.particle_id,
                                  "dm": [k not in set(miss_full) for k in key]}),
            "1gev": pd.DataFrame({"event": t.event, "particle_id": t.particle_id,
                                  "dm": [k not in set(miss_1gev) for k in key]})}


def test_delta_sigma_and_index_table(g):
    events = sorted(g.event.unique())
    pp = g[g.pp]
    lost_full = list(zip(pp.event, pp.particle_id))[::7]                   # lost in C-full only
    t = H.pair_conditions(g, _outs(g, miss_full=lost_full))
    r = H.h1(t, events)
    n = len(pp)
    assert r["n_pp"] == n == 168 and r["eps_1gev"] == 100.0
    assert r["delta"] == pytest.approx(100 * (n - len(lost_full)) / n - 100.0)
    # the paired event bootstrap by hand: evaluate.bootstrap's index table over the events in increasing id
    c = H.per_event(t, events, ["dm_full", "dm_1gev"])
    idx = np.random.default_rng(12345).integers(0, len(events), size=(500, len(events)))
    s = c[idx].sum(1)
    boot = 100 * s[:, 1] / s[:, 0] - 100 * s[:, 2] / s[:, 0]
    assert r["sigma"] == pytest.approx(np.std(boot, ddof=1), rel=1e-12) and r["ddof"] == 1
    assert all(x["equal"] for x in r["index_table_check"].values())
    # evaluate.bootstrap itself on the C-full counts gives the same standard deviation of eps(C-full)
    ref = EVAL.bootstrap(pd.DataFrame({"m_target": c[:, 1].astype(int), "n_target": c[:, 0].astype(int),
                                       "dup_sum_targets": 0, "m_target_sm3": 0}), 500, 12345)
    assert ref.loc["ours_eff_target", "boot_std"] == pytest.approx(np.std(s[:, 1] / s[:, 0], ddof=1), rel=1e-9)
    assert r["underpowered"] and "underpowered" in r["label"]


def test_outcome_rule_on_unrounded_values():
    assert H.outcome(-0.25, 0.25) == "holds"
    assert H.outcome(-0.5, 0.25) == "does not hold, inconclusive"             # -1.0 is not > -1.0
    assert H.outcome(-0.5 + 1e-12, 0.25) == "holds"
    assert H.outcome(-3.4, 0.55) == "does not hold, resolved below the margin"
    assert H.outcome(-1.5, 0.25) == "does not hold, inconclusive"             # -1.0 is not < -1.0
    assert H.outcome(-1.5 - 1e-12, 0.25) == "does not hold, resolved below the margin"
    assert H.outcome(0.5, 0.1) == "holds" and H.MARGIN == 1.0 and H.MIN_PP == 1000


def test_pairing_requires_the_same_targets(g):
    outs = _outs(g)
    outs["1gev"] = outs["1gev"].iloc[1:]
    with pytest.raises(LookupError, match="1gev"):
        H.pair_conditions(g, outs)
    outs = _outs(g)
    outs["full"] = pd.concat([outs["full"], outs["full"].iloc[:1]])
    with pytest.raises(ValueError, match="twice"):
        H.pair_conditions(g, outs)


def test_report_flags_restriction_and_soft_siblings(g):
    events = sorted(g.event.unique())
    pp = g[g.pp]
    # the soft-sibling targets (LLP 8, prong 1) are lost in C-full only
    sib = pp[pp.soft_sibling]
    out = H.report(g, _outs(g, miss_full=list(zip(sib.event, sib.particle_id))), events, restriction=("central",),
                   flags=("V2",), lost_share=0.03)
    h = out["h1"]
    assert h["n_pp"] == 168 - 24 and "flags: V2" in h["label"] and h["flags"] == ["V2"]
    assert out["h1_all_pp"]["n_pp"] == 168 and out["restriction"] == ["central"]
    ws = out["without_soft_siblings"]
    assert ws["pp_with_soft_sibling"] == 12 and ws["delta"] == 0.0 and h["delta"] < 0
    assert set(out["bins"]) == set(H.BINS) and out["pp_by_region"]["central"] == 144
    b = out["nav_loss_bound"]
    assert b["bound"] == pytest.approx(0.03 * b["max_abs_bin_deviation"])
    e = out["efficiencies"]
    assert e["pp"]["n"] == 168 and e["pd"]["n"] == 48 and e["target"]["n"] == len(g[g.target])
    d = H.h1(H.pair_conditions(g, _outs(g)), events, descriptive=True)
    assert d["outcome"].startswith("not decided")
    with pytest.raises(ValueError, match="C-full and C-1GeV"):
        H.report(g, {"full": _outs(g)["full"]}, events)


def test_nav_loss_bound():
    """Decision D5: only bins with at least 20 P_P targets enter the max; the others are listed."""
    bins = {"x": [{"bin": "a", "n": 20, "delta": -1.0}, {"bin": "b", "n": 0, "delta": float("nan")},
                  {"bin": "c", "n": 19, "delta": -9.0}, {"bin": "d", "n": 300, "delta": -3.5}],
            "y": [{"bin": "e", "n": 3, "delta": -40.0}]}
    r = H.nav_loss_bound(0.029, bins, -2.0)
    assert H.NAV_MIN_PP == 20 and r["min_pp_per_bin"] == 20 and r["bins_used"] == 2
    assert r["max_abs_bin_deviation"] == 1.5 and r["bound"] == pytest.approx(0.029 * 1.5)
    assert [(b["variable"], b["bin"], b["n"]) for b in r["bins_below_min"]] == [("x", "c", 19), ("y", "e", 3)]
    r = H.nav_loss_bound(0.029, {"x": [{"bin": "c", "n": 19, "delta": -9.0}]}, -2.0)
    assert np.isnan(r["bound"]) and np.isnan(r["max_abs_bin_deviation"]) and len(r["bins_below_min"]) == 1


def test_placeholder_outputs_and_set_events(g, tmp_path):
    outs = H.placeholder_outputs(g)
    assert set(outs) == {"full", "1gev", "hs"} and all(len(x) == g.target.sum() for x in outs.values())
    events = sorted(g.event.unique())
    r = H.report(g, outs, events)
    assert set(r["efficiencies"]["pp"]) == {"n", "full", "1gev", "hs"}
    split = {"sets": {"pilot": {"hosts": [21289, 21332]}}, "hval": [21300, 21117], "htest": [23001, 23005],
             "htest_spares": [24007, 24100], "htest_dropped": [{"host": 23005, "replacement": 24007, "reason": "x"}]}
    assert H.set_events("pilot", split) == [59_021_289, 59_021_332]
    assert H.set_events("hval", split) == [50_021_117, 50_021_300]
    assert H.set_events("htest", split) == [52_023_001, 52_024_007]
    with pytest.raises(LookupError):
        H.set_events("htest", {"htest": None})
    with pytest.raises(ValueError):                                       # not the next spare in hash order
        H.set_events("htest", {**split, "htest_dropped": [{"host": 23005, "replacement": 24100, "reason": "x"}]})


def test_inputs_come_from_the_sealed_directory_or_grids_only(tmp_path):
    """H-val and the pilot read C-full from their sealed directory only; H-test has none; C-1GeV is required."""
    g = {"1gev": str(tmp_path)}
    with pytest.raises(PermissionError, match="sealed directory"):
        H.load_outputs("hval", "WP-A", {**g, "full": str(tmp_path)}, str(tmp_path), "ab" * 32)
    with pytest.raises(PermissionError, match="sealed directory"):
        H.load_outputs("pilot", "WP-R-P3", {**g, "full": str(tmp_path)})
    with pytest.raises(ValueError, match="--sealed"):
        H.load_outputs("hval", "WP-A", g)
    with pytest.raises(ValueError, match="--expect"):                    # the pilot names its scratch manifest
        H.load_outputs("pilot", "WP-R-P3", g, str(tmp_path))
    with pytest.raises(ValueError, match="no sealed directory"):
        H.load_outputs("htest", "WP-A", {**g, "full": str(tmp_path)}, str(tmp_path), "ab" * 32)
    with pytest.raises(ValueError, match="C-1GeV"):
        H.load_outputs("htest", "WP-A", {"full": str(tmp_path)})
    with pytest.raises(KeyError):
        H.load_outputs("closure", "WP-A", g)
    # a sealed H-val directory outside the production runs root is refused before anything in it is read
    with pytest.raises(PermissionError):
        H.load_outputs("hval", "WP-A", g, str(tmp_path), "ab" * 32)


def _grid(tmp_path, cfgs, rows):
    d = tmp_path / "grid"
    d.mkdir()
    (d / "configs.json").write_text(json.dumps(cfgs))
    pd.DataFrame(rows).to_parquet(d / "per_target.parquet", index=False)
    return str(d)


def test_grid_readers(tmp_path):
    assert HS.wp_params("WP-A") == ("mwslmerge", 0.3) and HS.wp_params("WP-R-P3") == ("cc", 0.8)
    cfgs = [["cc", 0.8], ["mwslmerge", 0.1], ["mwslmerge", 0.3]]
    rows = {"event": [5, 5, 5, 6], "config": [2, 1, 2, 2], "particle_id": [11, 11, 12, 11],
            "dm": [True, False, False, True], "sm_any": True, "k_sm_ge5": 1}
    d = _grid(tmp_path, cfgs, rows)
    assert HS.wp_config(d, "WP-A") == 2 and HS.wp_config(d, "WP-S") == 1 and HS.wp_config(d, "WP-R-P3") == 0
    t = HS.read_per_target(d, "WP-A")
    assert t.values.tolist() == [[5, 11, True], [5, 12, False], [6, 11, True]]
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(LookupError):
        HS.wp_config(_grid(other, [["cc", 0.6], ["mwslmerge", 0.3], ["mwslmerge", 0.3]], rows), "WP-A")


def test_bootstrap_helpers():
    assert np.array_equal(HS.index_table(7), np.random.default_rng(12345).integers(0, 7, size=(500, 7)))
    assert np.isnan(HS.std1([1.0])) and HS.std1([1.0, np.nan, 3.0]) == pytest.approx(np.std([1, 3], ddof=1))
    r = HS.paired_delta([10, 10], [9, 10], [10, 10])
    assert r["delta"] == pytest.approx(-5.0) and r["eps_a"] == 95.0 and r["n"] == 20
    chk = HS.check_index_table(np.array([10, 12, 9, 11]), np.array([9, 12, 7, 11]))
    assert chk["equal"]
    with pytest.raises(ValueError):
        HS.bootstrap_stat(np.zeros((0, 2)), lambda s: s[..., 0])


def _record_repo(tmp_path, record=None, name="h_t1_restriction_v1.json"):
    """A git repository with a committed restriction record under results/pileup_hp/ (as the H-T1 code would write)."""
    import os
    import subprocess
    r = str(tmp_path / "rrepo")
    os.makedirs(os.path.join(r, "results", "pileup_hp"))
    rec = {"restriction": None, "flags": [], "lost_share": 0.03} if record is None else record
    p = os.path.join(r, "results", "pileup_hp", name)
    json.dump(rec, open(p, "w"))
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(["git", "init", "-q", r], check=True)
    subprocess.run(["git", "-C", r, "add", "-A"], check=True, env=env)
    subprocess.run(["git", "-C", r, "commit", "-qm", "record"], check=True, env=env)
    return r, p


def test_decisive_run_is_refused_once_made(tmp_path, monkeypatch):
    """Decision D3: on H-val and H-test the run of a working point is refused once its consumed marker exists, its
    pending record exists, or its output exists without either (hnl.overlay.blind.write_once, once_refusal), before
    any input is read; other working points have their own marker. Every grid of the set, the grants (made under the
    launch token), the fixed output path, the committed restriction record and the launch record are checked first."""
    import os
    from hnl.overlay import blind as B
    from hnl.overlay import grant as G
    if not os.path.exists(H.SPLIT_H):
        pytest.skip("needs configs/trackml/split_h_v1.json")
    runs = str(tmp_path / "runs")
    t = str(tmp_path)
    tok = "ab" * 16
    ev = H.set_events("hval")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)                  # grants and checks outside Slurm alike
    monkeypatch.setenv(G.ENV, os.pathsep.join(G.write("hval", c, "eval", ev, t + "/grants", launch=tok)
                                              for c in ("full", "1gev", "hs")))
    monkeypatch.setenv("HNL_H_LAUNCH", tok)
    monkeypatch.setattr(B, "runs_root", lambda repo=B.REPO: runs)
    r, rec = _record_repo(tmp_path, name="h_t05_restriction_v1.json")
    monkeypatch.setattr(H, "REPO", r)
    monkeypatch.setattr(H, "RESULTS_DIR", os.path.join(t, "res"))
    out = H.decisive_out("hval", "WP-A")
    assert out == os.path.join(t, "res", "h1_hval_WP-A_full-1gev-hs_v1.json")
    base = ["run", "--set", "hval", "--sealed", t + "/sealed", "--grid-1gev", t + "/g", "--grid-hs", t + "/g",
            "--restriction-record", rec]
    args = base + ["--out-json", out]
    with pytest.raises(PermissionError, match="launch record"):
        H.main(args)                                                   # not launched (no record under runs)
    monkeypatch.setattr(B, "check_launch", lambda *a, **k: [])
    os.makedirs(os.path.join(runs, "launch"), exist_ok=True)              # the launch record pins the D7 record
    json.dump({"token": tok, "pins": {"results/pileup_hp/h_t05_restriction_v1.json": B.sha256_file(rec)}},
              open(B.launch_path("hval", runs), "w"))
    B.write_once(out, "{}", B.consumed_path("hval", "WP-A", runs), {"set": "hval", "wp": "WP-A"})
    with pytest.raises(PermissionError, match="written once already"):
        H.main(args)
    with pytest.raises(PermissionError, match="written once already"):
        H.main(args + ["--wp", "WP-A"])
    monkeypatch.setattr(H, "load_truth", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("inputs read")))
    wps = H.decisive_out("hval", "WP-S")
    with pytest.raises(RuntimeError, match="inputs read"):              # WP-S has no marker yet: the run goes on
        H.main(base + ["--out-json", wps, "--wp", "WP-S"])
    # an output in place without its marker (a run stopped between the two): refused, and the marker is recorded
    open(wps, "w").write("{}")
    with pytest.raises(PermissionError, match="stopped before its marker"):
        H.main(base + ["--out-json", wps, "--wp", "WP-S"])
    assert os.path.exists(B.consumed_path("hval", "WP-S", runs))
    # a pending record without output or marker (a run stopped while placing its output): consumed
    p2 = B.pending_path(B.consumed_path("hval", "WP-R-P2", runs))
    json.dump({"output": H.decisive_out("hval", "WP-R-P2")}, open(p2, "w"))
    with pytest.raises(PermissionError, match="began placing its output"):
        H.main(base + ["--out-json", H.decisive_out("hval", "WP-R-P2"), "--wp", "WP-R-P2"])
    with pytest.raises(SystemExit):                                     # without C-hs: refused before everything
        H.main(["run", "--set", "hval", "--sealed", t + "/sealed", "--grid-1gev", t + "/g",
                "--out-json", H.decisive_out("hval", "WP-R-P3"), "--wp", "WP-R-P3"])
    assert not os.path.exists(B.consumed_path("hval", "WP-R-P3", runs))


def test_decisive_runs_write_one_path_from_a_committed_record(tmp_path, monkeypatch):
    """The crash rule and decision D7: on H-val and H-test the output path is fixed by set and working point (any other
    --out-json is refused), and the P_P restriction, flag gates, lost share and descriptive state come only from a
    committed record under results/pileup_hp/ (pinned by the set's launch record: on H-val an h_t05_ record, on H-test
    an H-T1 record), and the free arguments are refused whenever given, even empty or zero; free arguments stay for the
    pilot. All of it is refused before the launch, the marker or any input."""
    import os
    import subprocess
    from hnl.overlay import blind as B
    from hnl.overlay import grant as G
    if not os.path.exists(H.SPLIT_H):
        pytest.skip("needs configs/trackml/split_h_v1.json")
    t = str(tmp_path)
    runs = t + "/runs"
    tok = "ab" * 16
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setenv("HNL_H_LAUNCH", tok)
    monkeypatch.setattr(B, "runs_root", lambda repo=B.REPO: runs)
    monkeypatch.setattr(H, "RESULTS_DIR", os.path.join(t, "res"))
    r, rec = _record_repo(tmp_path, name="h_t05_restriction_v1.json")
    monkeypatch.setattr(H, "REPO", r)
    launch_seen = []
    monkeypatch.setattr(G, "launch_problems", lambda *a, **k: [])           # the grants pass; h1's own launch check:
    monkeypatch.setattr(B, "check_launch", lambda *a, **k: launch_seen.append(a) or ["not launched (stand-in)"])
    ev = H.set_events("hval")
    monkeypatch.setenv(G.ENV, os.pathsep.join(G.write("hval", c, "eval", ev, t + "/grants", launch=tok)
                                              for c in ("full", "1gev", "hs")))
    base = ["run", "--set", "hval", "--sealed", t + "/sealed", "--grid-1gev", t + "/g", "--grid-hs", t + "/g"]
    good = ["--out-json", H.decisive_out("hval", "WP-A"), "--restriction-record", rec]
    cases = [(["--out-json", t + "/elsewhere.json", "--restriction-record", rec], "writes .*h1_hval_WP-A_full-1gev"),
             (["--out-json", H.decisive_out("hval", "WP-S"), "--restriction-record", rec], "writes .*WP-A.* only"),
             (good + ["--restriction", "central"], "--restriction given on the command line"),
             (good + ["--flags", "V2"], "--flags given"), (good + ["--lost-share", "0.1"], "--lost-share given"),
             (good + ["--descriptive"], "--descriptive given"),
             (good + ["--lost-share", "0"], "--lost-share given"), (good + ["--lost-share", "0.0"], "--lost-share given"),
             (good + ["--flags", ""], "--flags given"), (good + ["--restriction", ""], "--restriction given"),
             (["--out-json", H.decisive_out("hval", "WP-A")], "--restriction-record .* is required")]
    for extra, msg in cases:
        with pytest.raises(PermissionError, match=msg):
            H.main(base + extra)
    assert not launch_seen                                             # refused before the launch check
    # the record: tracked under results/pileup_hp/, committed as is, with valid values
    loose = tmp_path / "loose.json"
    loose.write_text(json.dumps({"restriction": None, "flags": [], "lost_share": None}))
    with pytest.raises(PermissionError, match="tracked file under results/pileup_hp/"):
        H.main(base + ["--out-json", H.decisive_out("hval", "WP-A"), "--restriction-record", str(loose)])
    with open(rec, "a") as fh:
        fh.write("\n")
    with pytest.raises(PermissionError, match="not committed as is"):
        H.main(base + good)
    subprocess.run(["git", "-C", r, "checkout", "--", "."], check=True)
    untracked = os.path.join(r, "results", "pileup_hp", "h_t05_other.json")
    json.dump({"restriction": None, "flags": [], "lost_share": None}, open(untracked, "w"))
    with pytest.raises(PermissionError, match="not committed as is"):
        H.main(base + ["--out-json", H.decisive_out("hval", "WP-A"), "--restriction-record", untracked])
    assert H.restriction_record(rec)["lost_share"] == 0.03
    for bad, msg in (({"flags": []}, "no restriction"), ({"restriction": ["north"], "flags": [], "lost_share": None},
                                                         "restriction"),
                     ({"restriction": None, "flags": "V2", "lost_share": None}, "flags"),
                     ({"restriction": None, "flags": [], "lost_share": 2.0}, "lost_share"),
                     ({"restriction": None, "flags": [], "lost_share": None, "descriptive": "no"}, "descriptive")):
        rr, pp = _record_repo(tmp_path / f"b{len(os.listdir(t))}", bad)
        with pytest.raises(ValueError, match=msg):
            H.restriction_record(pp, rr)
    with pytest.raises(PermissionError, match="launch record"):        # valid: on to the launch check
        H.main(base + good)
    assert launch_seen
    # H-val: the record must be pinned by the H-val launch record (an h_t05_ record committed before the launch)
    monkeypatch.setattr(B, "check_launch", lambda *a, **k: [])
    os.makedirs(os.path.join(runs, "launch"), exist_ok=True)
    json.dump({"token": tok, "pins": {"configs/trackml/split_h_v1.json": "0" * 64}},
              open(B.launch_path("hval", runs), "w"))
    with pytest.raises(PermissionError, match="not pinned by the H-val launch record"):
        H.main(base + good)
    json.dump({"token": tok, "pins": {"results/pileup_hp/h_t05_restriction_v1.json": "0" * 64}},
              open(B.launch_path("hval", runs), "w"))
    with pytest.raises(PermissionError, match="not pinned by the H-val launch record"):     # another content
        H.main(base + good)
    json.dump({"token": tok, "pins": {"results/pileup_hp/h_t05_restriction_v1.json": B.sha256_file(rec)}},
              open(B.launch_path("hval", runs), "w"))
    monkeypatch.setattr(H, "load_truth", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("inputs read")))
    with pytest.raises(RuntimeError, match="inputs read"):              # pinned: on to the inputs
        H.main(base + good)
    # H-test: the record must be pinned by the launch record (an H-T1 record named in R15)
    r, rec = _record_repo(tmp_path / "t1")
    monkeypatch.setattr(H, "REPO", r)
    monkeypatch.setattr(H, "set_events", lambda s: [52_300_001])
    monkeypatch.setenv(G.ENV, os.pathsep.join(G.write("htest", c, "eval", [52_300_001], t + "/grants", launch=tok)
                                              for c in ("full", "1gev", "hs")))
    monkeypatch.setattr(B, "check_launch", lambda *a, **k: [])
    os.makedirs(os.path.join(runs, "launch"), exist_ok=True)
    json.dump({"token": tok, "pins": {"configs/trackml/split_h_v1.json": "0" * 64}},
              open(B.launch_path("htest", runs), "w"))
    ht = ["run", "--set", "htest", "--grid-full", t + "/g", "--grid-1gev", t + "/g", "--grid-hs", t + "/g",
          "--out-json", H.decisive_out("htest", "WP-A"), "--restriction-record", rec]
    with pytest.raises(PermissionError, match="not pinned by the H-test launch record"):
        H.main(ht)
    json.dump({"token": tok, "pins": {"results/pileup_hp/h_t1_restriction_v1.json": B.sha256_file(rec)}},
              open(B.launch_path("htest", runs), "w"))
    monkeypatch.setattr(H, "load_truth", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("inputs read")))
    with pytest.raises(RuntimeError, match="inputs read"):              # pinned: on to the inputs
        H.main(ht)
    assert not os.path.exists(os.path.join(t, "res"))
    # the job script: the fixed path and no free RESTRICTION / FLAGS / LOST_SHARE on hval and htest
    text = open(os.path.join(B.REPO, "slurm", "overlay", "h1.sbatch")).read()
    assert "python -I -m hnl.overlay.h1 out-path --set \"$SET\" --wp \"$WPN\"" in text
    assert "--restriction-record \"$RESTRICTION_RECORD\"" in text and "${RESTRICTION:+" not in text
    assert H.main(["out-path", "--set", "htest", "--wp", "WP-S"]) is None


def test_decisive_runs_read_a_record_of_their_own_prefix(tmp_path, monkeypatch):
    """Decision D7 (closes the verify residual of the addendum H commit): on H-val the record is an H-val record
    results/pileup_hp/h_t05_*, on H-test an H-T1 record results/pileup_hp/h_t1_* (hnl.overlay.h1.record_prefix), even
    when the launch record pins a record of the other prefix (R15 may name any file under results/pileup_hp/). The
    refusal comes before the launch check, the marker and any input."""
    import os
    import subprocess
    from hnl.overlay import blind as B
    from hnl.overlay import grant as G
    if not os.path.exists(H.SPLIT_H):
        pytest.skip("needs configs/trackml/split_h_v1.json")
    assert H.record_prefix("hval") == B.HT05_RECORDS == "results/pileup_hp/h_t05_"
    assert H.record_prefix("htest") == B.HT1_RECORDS == "results/pileup_hp/h_t1_"
    with pytest.raises(PermissionError, match="hval and htest only"):
        H.record_prefix("pilot")
    t = str(tmp_path)
    runs = t + "/runs"
    tok = "cd" * 16
    r, ht1 = _record_repo(tmp_path)                                      # results/pileup_hp/h_t1_restriction_v1.json
    ht05 = os.path.join(r, "results", "pileup_hp", "h_t05_restriction_v1.json")
    other = os.path.join(r, "results", "pileup_hp", "v2_hval_v1.json")  # a tracked file of neither prefix
    for p in (ht05, other):
        json.dump({"restriction": None, "flags": [], "lost_share": 0.02}, open(p, "w"))
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(["git", "-C", r, "add", "-A"], check=True, env=env)
    subprocess.run(["git", "-C", r, "commit", "-qm", "records"], check=True, env=env)
    assert H.restriction_record(ht05, r, "hval")["lost_share"] == 0.02
    assert H.restriction_record(ht1, r, "htest")["lost_share"] == 0.03
    assert H.restriction_record(other, r)["lost_share"] == 0.02           # no set: the prefix is not asked
    for path, s, msg in ((ht1, "hval", "h_t05_"), (other, "hval", "h_t05_"), (ht05, "htest", "h_t1_"),
                         (other, "htest", "h_t1_"), (ht05, "pilot", "hval and htest only")):
        with pytest.raises(PermissionError, match=msg):
            H.restriction_record(path, r, s)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setenv("HNL_H_LAUNCH", tok)
    monkeypatch.setattr(B, "runs_root", lambda repo=B.REPO: runs)
    monkeypatch.setattr(H, "RESULTS_DIR", os.path.join(t, "res"))
    monkeypatch.setattr(H, "REPO", r)
    monkeypatch.setattr(G, "launch_problems", lambda *a, **k: [])
    launch_seen = []
    monkeypatch.setattr(B, "check_launch", lambda *a, **k: launch_seen.append(a) or [])
    monkeypatch.setattr(H, "load_truth", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("inputs read")))
    os.makedirs(os.path.join(runs, "launch"), exist_ok=True)
    pins = {os.path.relpath(p, r): B.sha256_file(p) for p in (ht05, ht1, other)}   # every record pinned
    for s in ("hval", "htest"):
        json.dump({"token": tok, "pins": pins}, open(B.launch_path(s, runs), "w"))
    # H-val
    ev = H.set_events("hval")
    monkeypatch.setenv(G.ENV, os.pathsep.join(G.write("hval", c, "eval", ev, t + "/grants", launch=tok)
                                              for c in ("full", "1gev", "hs")))
    hv = ["run", "--set", "hval", "--sealed", t + "/sealed", "--grid-1gev", t + "/g", "--grid-hs", t + "/g",
          "--out-json", H.decisive_out("hval", "WP-A")]
    for rec in (ht1, other):
        with pytest.raises(PermissionError, match="reads a record results/pileup_hp/h_t05_"):
            H.main(hv + ["--restriction-record", rec])
    assert not launch_seen
    with pytest.raises(RuntimeError, match="inputs read"):              # its own prefix, pinned: on to the inputs
        H.main(hv + ["--restriction-record", ht05])
    assert launch_seen
    # H-test
    launch_seen.clear()
    monkeypatch.setattr(H, "set_events", lambda s: [52_300_001])
    monkeypatch.setenv(G.ENV, os.pathsep.join(G.write("htest", c, "eval", [52_300_001], t + "/grants", launch=tok)
                                              for c in ("full", "1gev", "hs")))
    ht = ["run", "--set", "htest", "--grid-full", t + "/g", "--grid-1gev", t + "/g", "--grid-hs", t + "/g",
          "--out-json", H.decisive_out("htest", "WP-A")]
    for rec in (ht05, other):
        with pytest.raises(PermissionError, match="reads a record results/pileup_hp/h_t1_"):
            H.main(ht + ["--restriction-record", rec])
    assert not launch_seen
    with pytest.raises(RuntimeError, match="inputs read"):
        H.main(ht + ["--restriction-record", ht1])
    assert launch_seen and not os.path.exists(os.path.join(t, "res"))


def test_hval_and_htest_grids_pass_the_shared_provenance_check(tmp_path, monkeypatch):
    """hnl.overlay.h1 reads H-val and H-test grids only after hnl.overlay.tracks.grid_provenance (the check of the
    closure grids of hnl.overlay.v1); the pilot's placeholder grids after grid_record only."""
    import json
    import os
    from hnl.overlay import blind as B
    from hnl.overlay import tracks as TR
    if not os.path.exists(H.SPLIT_H):
        pytest.skip("needs configs/trackml/split_h_v1.json")
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    (sealed / "blind.json").write_text(json.dumps({"set": "hval", "k": 0, "scores": {"placeholder": False}}))
    monkeypatch.setattr(B, "sealed_per_target", lambda *a, **k: pd.DataFrame(
        {"event": [1], "particle_id": [1], "dm": [True]}))
    monkeypatch.setattr(B, "manifest_sha256", lambda d: "0" * 64)
    seen = []

    def prov(d, set_name, cond, root=None, events=None, split=None, selection=None):
        seen.append((set_name, cond, root))
        raise RuntimeError("checked")
    monkeypatch.setattr(TR, "grid_provenance", prov)
    monkeypatch.setattr(TR, "grid_record", lambda *a, **k: (_ for _ in ()).throw(AssertionError("grid_record")))
    with pytest.raises(RuntimeError, match="checked"):
        H.load_outputs("hval", "WP-A", {"1gev": str(tmp_path), "hs": str(tmp_path)}, str(sealed))
    assert seen == [("hval", "1gev", None)]
    # the shared check itself on an H-val grid: frozen classifier, condition, merge model, production root
    sel = json.load(open(B.SELECTION))["inputs"]
    rec = {"set": "hval", "k": 0, "condition": "hs", "events": B.set_ids("hval"), "root": B.DERIVED_OVERLAY,
           "scores_provenance": {"placeholder": False, "ckpt_sha256": sel["classifier_checkpoint"]["sha256"],
                                 "condition": "hs"},
           "configs": TR.wp_configs(), "merge_model": sel["merge_model"]["path"]}
    monkeypatch.undo()
    for change, ok in (({}, True), ({"merge_model": "/x.joblib"}, False), ({"root": "/tmp/x"}, False),
                       ({"scores_provenance": {**rec["scores_provenance"], "condition": "1gev"}}, False),
                       ({"scores_provenance": {**rec["scores_provenance"], "ckpt_sha256": "0" * 64}}, False)):
        d = tmp_path / f"g{len(list(tmp_path.iterdir()))}"
        d.mkdir()
        (d / "grid.json").write_text(json.dumps({**rec, **change}))
        if ok:
            assert TR.grid_provenance(str(d), "hval", "hs")["condition"] == "hs"
        else:
            with pytest.raises(ValueError):
                TR.grid_provenance(str(d), "hval", "hs")
    with pytest.raises(PermissionError):
        TR.grid_provenance(str(d), "hval", "hs", root=str(tmp_path))        # H-val tables: production root only
