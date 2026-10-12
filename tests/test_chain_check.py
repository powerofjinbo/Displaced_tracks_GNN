"""Guards before evaluations: chain_check (checkpoint identity, working-point shape and val chain) and baseline_val's
completeness / HOLD behaviour."""

import hashlib
import json
import os

import numpy as np
import pandas as pd
import pytest

from hnl.train import chain_check as CC


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _fake_run(root, name, ml_bytes=b"ml", gnn_bytes=b"gnn"):
    run = os.path.join(root, name)
    for st, rec, data in (("ml", "ml_graphs", ml_bytes), ("gnn", "gnn_scores", gnn_bytes)):
        os.makedirs(os.path.join(run, st, "artifacts"), exist_ok=True)
        os.makedirs(os.path.join(run, rec), exist_ok=True)
        ck = os.path.join(run, st, "artifacts", "best.ckpt")
        open(ck, "wb").write(data)
        json.dump({"state": "done", "best_ckpt": ck, "end_utc": "2026-10-03T00:00:00"},
                  open(os.path.join(run, st, "run_1.json"), "w"))
        json.dump({"ckpt": ck, "ckpt_sha256": _sha(ck)}, open(os.path.join(run, rec, "infer_2.json"), "w"))
    return run


def _wp(root, runs):
    arms, seeds = ["i", "ii", "iii"], ["0", "1", "2"]
    return {"s_star": {a: 0.4 for a in arms}, "cuts_at_FR_star": {a: {s: 0.4 for s in seeds} for a in arms},
            "cuts_at_e_star": {a: {s: 0.4 for s in seeds} for a in arms}, "missing_runs": [], "runs": runs,
            "reference_arm": "ii", "unreachable_at_FR_star": [],
            "val_chain": {r: CC.val_chain(os.path.join(root, r)) for r in runs}}


def test_check_run_and_stale_checkpoint(tmp_path):
    run = _fake_run(str(tmp_path), "armi_s0")
    assert CC.check_run(run, ["ml", "gnn"]) == []
    open(os.path.join(run, "gnn", "artifacts", "best.ckpt"), "wb").write(b"retrained")   # changed after val inference
    assert any("sha256" in e for e in CC.check_run(run, ["gnn"]))


def test_check_wp_shape_and_val_chain(tmp_path):
    root = str(tmp_path)
    runs = [f"arm{a}_s{s}" for a in ("i", "ii", "iii") for s in (0, 1, 2)]
    for r in runs:
        _fake_run(root, r)
    wp = _wp(root, runs)
    p = os.path.join(root, "wp.json")
    json.dump(wp, open(p, "w"))
    assert CC.check_wp(p, root) == []
    one_arm = dict(wp, s_star={"i": 0.4})
    json.dump(one_arm, open(p, "w"))
    assert CC.check_wp(p)
    json.dump(dict(wp, unreachable_at_FR_star=[["ii", 1]]), open(p, "w"))
    assert any("reference arm" in e for e in CC.check_wp(p))
    # reviewed by hand (DECISIONS.md): accepted for that seed only
    json.dump(dict(wp, unreachable_at_FR_star=[["ii", 1]], accepted_unreachable_reference=[["ii", "1"]]), open(p, "w"))
    assert not any("reference arm" in e for e in CC.check_wp(p))
    json.dump(dict(wp, unreachable_at_FR_star=[["ii", 1], ["ii", 2]], accepted_unreachable_reference=[["ii", "1"]]),
              open(p, "w"))
    assert any("reference arm" in e for e in CC.check_wp(p))
    json.dump(dict(wp, unreachable_at_FR_star=[["i", 0]]), open(p, "w"))          # other arms: flagged, no error
    assert not any("reference arm" in e for e in CC.check_wp(p))
    # a run re-inferred with another model after the working points were made
    json.dump(wp, open(p, "w"))
    _fake_run(root, "armi_s1", gnn_bytes=b"other model")
    assert any("not the one behind the working points" in e for e in CC.check_wp(p, root))


def test_main_needs_something_to_check():
    with pytest.raises(SystemExit):
        CC.main([])


def test_baseline_val_guards(tmp_path, monkeypatch):
    from hnl.eval import baseline_val as BV
    root = str(tmp_path)
    for r in ("armi_s0", "armii_s0"):
        _fake_run(root, r)
        open(os.path.join(root, r, "eval.done"), "w").close()
    cuts = np.round(np.arange(0.01, 0.991, 0.01), 4)

    def fake_curves(pp, pe, info):
        tag = pp
        fr = (0.020 if tag == "i" else 0.010) * (1 - cuts)
        eff = 0.95 - 0.5 * (cuts - 0.4) ** 2
        return pd.DataFrame({"eff_all_targets": eff, "eff_primary": eff - (0.2 if tag == "i" else 0.05),
                             "fake_rate_hnl": fr, "edge_fpr_hnl": 0.1 * (1 - cuts) ** 2}, index=cuts)
    monkeypatch.setattr(BV, "load_run", lambda run: ("i" if "armi_" in run else "ii", pd.DataFrame({"event_id": [1]})))
    monkeypatch.setattr(BV, "curves", fake_curves)
    monkeypatch.setattr(BV, "particle_table", lambda ev: pd.DataFrame())
    out = os.path.join(root, "out")
    with pytest.raises(SystemExit, match="without eval.done"):
        BV.main(["--runs", root, "--out", out, "--arms", "i,ii", "--seeds", "0,1"])
    BV.main(["--runs", root, "--out", out, "--arms", "i,ii", "--seeds", "0,1", "--allow-missing"])
    wp = json.load(open(os.path.join(out, "working_points.json")))
    assert set(wp["missing_runs"]) == {"armi_s1", "armii_s1"} and set(wp["val_chain"]) == {"armi_s0", "armii_s0"}
