"""hnl.trackml.evaluate variant J' (eval_reco_trkx.py with min_hits = 4, plan section 4.2): noise hits and the hits of
particles with n_hits <= 4 are dropped before matching."""

import numpy as np
import pandas as pd

from hnl.trackml.evaluate import JP_MIN_HITS, event_counts, pooled


def test_jprime_drops_short_particles_and_noise():
    assert JP_MIN_HITS == 4
    # particle 1 (6 hits, pT 2), particle 5 (4 hits, pT 0.8: selected, not reconstructable), noise 0
    pid = np.array([1] * 6 + [5] * 4 + [0] * 2)
    t = np.array([True, False])
    parts = pd.DataFrame({"particle_id": [1, 5], "pt": [2.0, 0.8], "eta": [0.1, 0.2], "n_hits": [6, 4],
                          "is_target": t, "is_prompt_target": t, "is_displaced_target": [False, False],
                          "is_far_displaced_target": [False, False], "is_hard_scatter": [False, False],
                          "is_target_pix3": t, "category": ["primary", "primary"]})
    lab = np.array([0] * 6 + [1] * 4 + [2] * 2)
    c, _ = event_counts(pid, lab, parts)
    assert c["m_P1_selected_pt05"] == 2                  # J: the 4-hit particle's track matches
    assert c["n_tracks_jprime"] == 1                     # J': only the 6-hit track keeps hits
    assert c["mJp_P1_selected_pt05"] == 1 and c["mJp_P1_reconstructable_pt05"] == 1
    assert c["tracks_jprime_dm_P1_selected_pt05"] == 1
    p = pooled(pd.DataFrame([c]))
    assert np.isclose(p["P1_Jprime_eps_phys_pt05"], 1 / 2) and np.isclose(p["P1_Jprime_purity_pt05"], 1.0)
