"""TrackML stages use the TrackML resolution table (configs/trackml/res_v1.json) in hnl.graphs.helix_union."""

import numpy as np

from hnl.graphs import helix_union as HU
from hnl.trackml import events as E


def test_trackml_resolution_applied():
    for v, (s_u, s_v) in E.RES_TML.items():
        assert HU.FINE[v] == s_u and HU.COARSE[v] == s_v
    fine, sz, sr = HU.hit_sigmas(np.array([8, 17, 18]))
    assert fine[0] == E.RES_TML[8][0] and sz[1] == E.RES_TML[17][1] and sr[2] == E.RES_TML[18][1]
