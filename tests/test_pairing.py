"""Two-track vertices from tracks built from the true hits of HNL daughters (real val events)."""

import numpy as np
import pytest

from hnl.vertex import pairing as VP


@pytest.mark.data
def test_truth_tracks_give_daughter_vertices():
    from hnl.data.splits import read_split
    from hnl.graphs.window_study import load_events
    ev = read_split("hnl10_100", "val")[:200]
    hits, edges, parts = load_events("hnl10_100", ev)
    found, total, close = 0, 0, 0
    for e, h in hits.groupby("event"):
        h = h.sort_values("row")
        pe = parts[(parts.event == e) & (parts.category == "hnl") & parts.is_target]
        if len(pe) != 2 or not np.isfinite(pe.rv.iloc[0]):
            continue
        total += 1
        pid = h.pid.to_numpy()
        # tracks = particles (truth labels), every particle with hits
        _, labels = np.unique(pid, return_inverse=True)
        fit = VP.fit_tracks(h.r.to_numpy(), h.phi.to_numpy(), h.z.to_numpy(), h.volume.to_numpy(), labels)
        pairs, radii = VP.find_vertices(fit)
        da = labels[np.flatnonzero(pid == pe.particle_id.iloc[0])[0]]
        db = labels[np.flatnonzero(pid == pe.particle_id.iloc[1])[0]]
        hit = [k for k, (x, y) in enumerate(pairs) if {x, y} == {da, db}]
        if hit:
            found += 1
            close += abs(radii[hit[0]] - pe.rv.iloc[0]) < max(5.0, 0.05 * pe.rv.iloc[0])
    assert total > 30
    assert found / total > 0.7, (found, total)          # truth tracks: ~0.78 with the default cuts (val)
    assert close / max(found, 1) > 0.9, (close, found)
