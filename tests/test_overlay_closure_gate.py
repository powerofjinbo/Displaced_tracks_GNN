"""hnl.overlay.closure_gate: the closure truth gate of addendum H section 9.2 (hits per crossed layer per region and
|eta| band, cells with at least 2,000 crossed layers on each side, |eta| < 3, tolerance 0.05) and its per-cell report,
on synthetic per-track rows."""

import numpy as np
import pandas as pd

from hnl.overlay import closure_gate as G


def _rows(region, eta, n, layers, hpl_native, hpl_closure, target_share=1.0, sim_hpl=None):
    out = []
    for i in range(n):
        for src, hpl in (("native", hpl_native), ("closure", hpl_closure), ("sim", sim_hpl or hpl_native)):
            out.append({"source": src, "event": 51_021_289, "native": i, "region": region, "abs_eta": eta,
                        "hits": int(round(hpl * layers)), "layers": layers,
                        "closure_target": i < target_share * n})
    return out


def test_cells_and_gate():
    rows = pd.DataFrame(_rows("pixel barrel", 0.5, 600, 4, 1.25, 1.25, target_share=0.9)
                        + _rows("short-strip barrel", 1.5, 500, 5, 1.2, 1.4)
                        + _rows("pixel endcap", 3.5, 800, 5, 2.0, 1.6)
                        + _rows("long-strip endcap", 2.5, 100, 5, 1.0, 1.4))
    cells = {(c["region"], c["abs_eta"]): c for c in G.cells(rows)}
    c = cells[("pixel barrel", "[0.0, 1.0)")]
    assert (c["tracks_native"], c["layers_native"], c["hits_native"]) == (600, 2400, 3000)
    assert c["hits_per_layer_native"] == 1.25 and c["hits_per_track_closure"] == 5.0
    assert c["layers_per_track_sim"] == 4.0 and abs(c["closure_target_share"] - 0.9) < 1e-12
    g = G.gate(list(cells.values()))
    by = {(c["region"], c["abs_eta"]): c for c in g["cells"]}
    assert by[("pixel barrel", "[0.0, 1.0)")]["counted"] and by[("pixel barrel", "[0.0, 1.0)")]["pass"]
    assert by[("short-strip barrel", "[1.0, 2.0)")]["pass"] is False                       # 1.4 against 1.2
    assert not by[("pixel endcap", "[3.0, 4.0)")]["in_rule"] and by[("pixel endcap", "[3.0, 4.0)")]["pass"] is None
    assert not by[("long-strip endcap", "[2.0, 3.0)")]["counted"]                          # 500 crossed layers
    assert g["counted"] == 2 and not g["pass"]
    assert g["failed"] == [{"region": "short-strip barrel", "abs_eta": "[1.0, 2.0)",
                            "diff": by[("short-strip barrel", "[1.0, 2.0)")]["diff"]}]
    assert abs(g["failed"][0]["diff"] - 0.2) < 1e-12


def test_gate_tolerance_edge_and_hits_per_layer_layout():
    # the rows of hnl.overlay.closure_checks.hits_per_layer (no band_lo) are read through their band label
    cell = {"region": "short-strip endcap", "abs_eta": "[2.0, 3.0)", "layers_native": 2000, "layers_closure": 2000,
            "hits_per_layer_native": 1.30, "hits_per_layer_closure": 1.349}
    assert G.gate([cell])["pass"]
    assert not G.gate([{**cell, "hits_per_layer_closure": 1.351}])["pass"]
    assert not G.gate([{**cell, "hits_per_layer_closure": 1.249}])["pass"]
    assert G.gate([{**cell, "layers_closure": 1999}])["no_counted_cell"]
    assert np.isclose(G.gate([cell])["cells"][0]["diff"], 0.049)


def test_outcome_not_informative_and_few_cells():
    """D5: no counted cell is "not informative" (neither a pass nor a fail); one or two counted cells are reported as
    "few cells"; three or more are not."""
    cell = {"region": "short-strip barrel", "abs_eta": "[0.0, 1.0)", "layers_native": 2000, "layers_closure": 2000,
            "hits_per_layer_native": 1.30, "hits_per_layer_closure": 1.32}
    small = {**cell, "layers_native": 30, "layers_closure": 30}
    fwd = {**cell, "abs_eta": "[3.0, 4.0)"}                                    # reported without the rule
    g = G.gate([small, fwd])
    assert g["counted"] == 0 and g["outcome"] == "not informative" and g["pass"] is None
    assert g["no_counted_cell"] and not g["few_cells"] and g["failed"] == []
    assert G.gate([])["outcome"] == "not informative"
    g = G.gate([cell, small])
    assert g["outcome"] == "pass" and g["pass"] is True and g["few_cells"] and g["counted"] == 1
    g = G.gate([cell, {**cell, "region": "pixel barrel"}, {**cell, "region": "long-strip barrel"}])
    assert g["outcome"] == "pass" and not g["few_cells"] and g["counted"] == 3
    g = G.gate([cell, {**cell, "hits_per_layer_closure": 1.40}])
    assert g["outcome"] == "fail" and g["pass"] is False and g["few_cells"]
    assert g["rule"]["not_informative"].startswith("no counted cell")


def test_pilot_closure_cells_are_not_informative():
    """The pilot closure set (five hosts) has no cell with 2,000 crossed layers: the gate is not informative there."""
    rows = pd.DataFrame(_rows("pixel barrel", 0.5, 60, 4, 1.25, 1.25) + _rows("short-strip barrel", 1.5, 50, 5, 1.2,
                                                                                 1.21))
    g = G.gate(G.cells(rows))
    assert g["counted"] == 0 and g["outcome"] == "not informative" and g["pass"] is None
    assert all(c["pass"] is None and c["diff"] is not None for c in g["cells"])
