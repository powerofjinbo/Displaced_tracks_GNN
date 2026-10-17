"""Track building of overlay events in the conditions of addendum H (hnl.overlay.tracks). Request rules on any machine;
the grid itself on the stored pilot set (k = 9) with truth-made placeholder scores only. No model runs and no model
output of an overlay event is read."""

import json
import os

import numpy as np
import pandas as pd
import pytest

from hnl.overlay import blind as B
from hnl.overlay import hstats as HS
from hnl.overlay import tracks as TR

needs_split = pytest.mark.skipif(not os.path.exists(B.SPLIT_H), reason="needs configs/trackml/split_h_v1.json")
PILOT_META = os.path.join(B.DERIVED_OVERLAY, "meta_pilot.json")
needs_pilot = pytest.mark.skipif(not (os.path.exists(PILOT_META) and os.path.exists(B.SPLIT_H)),
                                 reason="needs the stored pilot overlay tables")


def test_working_points_of_section_7():
    cfgs = TR.wp_configs()
    assert [c["name"] for c in cfgs] == ["WP-A", "WP-S", "WP-R-P3"]
    assert [B.config_tuple(c) for c in cfgs] == [("mwslmerge", 0.3), ("mwslmerge", 0.1), ("cc", 0.8)]
    assert TR._fmt(1.0) == "1" and TR._fmt(0.15) == "0.15"


@needs_split
def test_request_rules(tmp_path):
    out = str(tmp_path / "g")
    with pytest.raises(PermissionError, match="hnl.overlay.blind only"):
        TR.check_request("hval", "full", out)
    with pytest.raises(PermissionError, match="overlay_version_h_v1.json"):
        TR.check_request("hval", "1gev", out, repo=str(tmp_path))         # no committed record in that repository
    with pytest.raises(PermissionError, match="C-full and C-1GeV only"):
        TR.check_request("closure", "hs", out)
    with pytest.raises(PermissionError, match="production runs root"):
        TR.check_request("pilot", "hs", os.path.join(B.RUNS_H, "x"))
    with pytest.raises(PermissionError, match="read from"):
        TR.check_request("closure", "full", out, root=str(tmp_path))     # production sets: production tables only
    with pytest.raises(ValueError):
        TR.check_request("pilot", "nonoise", out)
    assert TR.check_request("closure", "1gev", out) == (1, None)
    assert TR.check_request("pilot", "hs", out, root=str(tmp_path)) == (9, str(tmp_path))
    assert TR.check_request("pilot", "full", out, root=B.DERIVED_OVERLAY) == (9, None)


@needs_split
def test_overlay_root_rules(tmp_path):
    assert B.overlay_root(0) is None and B.overlay_root(2, B.DERIVED_OVERLAY) is None
    assert B.overlay_root(9, str(tmp_path)) == str(tmp_path)
    for k in (0, 1, 2):
        with pytest.raises(PermissionError):
            B.overlay_root(k, str(tmp_path))


def _side(tmp_path, cond, placeholder=True):
    s = tmp_path / f"scores_{cond}"
    s.mkdir()
    (s / "scores_1.parquet").write_bytes(b"")
    side = {"events": {str(e): 1 for e in B.set_ids("pilot")}, "condition": cond,
            "placeholder": "truth" if placeholder else None, "ckpt_sha256": None}
    (s / "scores_1.parquet.json").write_text(json.dumps(side))
    return str(s)


@needs_split
def test_provenance_by_condition(tmp_path):
    s = _side(tmp_path, "hs")
    assert B.scores_provenance(s, 9, B.set_ids("pilot"), condition="hs")["condition"] == "hs"
    with pytest.raises(ValueError, match="not full"):
        B.scores_provenance(s, 9, B.set_ids("pilot"))
    with pytest.raises(ValueError):
        B.scores_provenance(s, 2, B.set_ids("pilot"), condition="hs")    # a placeholder never passes for k != 9


@needs_pilot
def test_pilot_grid_in_hs_and_1gev_from_placeholder_scores(tmp_path, monkeypatch):
    """Placeholder scores in C-hs and C-1GeV, the dry builders, blocks and merge, and the readers of hnl.overlay.h1.
    The grid runs under dry grants of its condition only (hnl.overlay.grant)."""
    from hnl.overlay import grant as G
    from hnl.overlay.conditions import hs_particle
    from hnl.trackml import events as E
    ids = B.set_ids("pilot")
    sc0 = str(tmp_path / "scores_none")
    monkeypatch.delenv(G.ENV, raising=False)
    with pytest.raises(PermissionError, match="no grant"):
        TR.grid("pilot", "hs", sc0, str(tmp_path / "x"))
    gr = {c: G.write("pilot", c, "dry", ids, str(tmp_path / "grants"), out=str(tmp_path)) for c in ("hs", "1gev")}
    monkeypatch.setenv(G.ENV, gr["1gev"])
    with pytest.raises(PermissionError, match="condition 1gev, not hs"):
        TR.grid("pilot", "hs", sc0, str(tmp_path / "x"))
    assert not os.path.exists(tmp_path / "x")
    monkeypatch.setenv(G.ENV, os.pathsep.join(gr.values()))
    recs = {}
    for cond in ("hs", "1gev"):
        sc = str(tmp_path / f"scores_{cond}")
        info = B.placeholder_scores("pilot", sc, cond)
        assert info["events"] == 5 and info["condition"] == cond
        with pytest.raises(PermissionError, match="mwslmerge"):
            TR.grid("pilot", cond, sc, str(tmp_path / "x"), configs=[{"name": "WP-A", "family": "mwslmerge",
                                                                       "params": "0.3"}])
        out = str(tmp_path / f"grid_{cond}")
        recs[cond] = TR.grid("pilot", cond, sc, out, workers=2)
        assert recs[cond]["scores_provenance"]["placeholder"] and recs[cond]["n_events"] == 5
        with pytest.raises(SystemExit):
            TR.grid("pilot", cond, sc, out)                               # no overwrite
        assert TR.grid_record(out, "pilot", cond)["condition"] == cond
        with pytest.raises(ValueError):
            TR.grid_record(out, "pilot", "full" if cond == "hs" else "hs")
        t = HS.read_per_target(out, "WP-R-P3")
        assert len(t) and set(t.event) == set(ids)
    pt = pd.read_parquet(os.path.join(str(tmp_path / "grid_hs"), "per_target.parquet"))
    assert hs_particle(pt.particle_id.to_numpy()).all()                  # C-hs: hard scatter and gun targets only
    assert B.is_gun(pt.particle_id.to_numpy()).any()
    pe = pd.read_parquet(os.path.join(str(tmp_path / "grid_1gev"), "per_event.parquet"))
    assert "trackml_score" not in pe                                      # no TrackML score outside C-full
    # C-hs keeps exactly the hits of the hard scatter and the gun
    ev = E.load_events(ids[:1])[ids[0]]
    from hnl.overlay.conditions import with_condition
    evh = with_condition(ev, "hs")
    assert evh.n == int(hs_particle(ev.pid).sum()) and evh.n < ev.n
    # blocks pooled by merge give the unsplit grid
    sc = str(tmp_path / "scores_hs")
    for b in (0, 1):
        TR.grid("pilot", "hs", sc, str(tmp_path / "blocked"), block=b, n_blocks=2)
    TR.merge(str(tmp_path / "blocked"))
    a = pd.read_parquet(str(tmp_path / "grid_hs" / "per_target.parquet")).sort_values(["config", "event", "particle_id"])
    b = pd.read_parquet(str(tmp_path / "blocked" / "per_target.parquet")).sort_values(["config", "event", "particle_id"])
    assert np.array_equal(a.to_numpy(), b.to_numpy())
    assert TR.grid_record(str(tmp_path / "blocked"), "pilot", "hs")["n_blocks"] == 2
