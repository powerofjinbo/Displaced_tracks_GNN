"""hnl.trackml.sectors: core partition, halos, ownership, sector graphs, score and track stitching.

    pytest src/hnl/trackml/tests
"""

import numpy as np
import pytest

from hnl.trackml import sectors as S


def _event(seed=0, n=3000, e=20000):
    rng = np.random.default_rng(seed)
    phi = rng.uniform(-np.pi, np.pi, n)
    r = rng.uniform(30, 1000, n)
    z = rng.uniform(-2000, 2000, n)
    eta = S.hit_eta(r, z)
    a = rng.integers(0, n, e)
    # mostly short edges in phi, some long ones (longer than any halo) and some across phi = +-pi
    b = np.empty(e, dtype=np.int64)
    near = np.argsort(phi)
    pos = np.empty(n, dtype=np.int64)
    pos[near] = np.arange(n)
    step = rng.integers(-40, 41, e)
    b[:] = near[(pos[a] + step) % n]
    far = rng.random(e) < 0.02
    b[far] = rng.integers(0, n, far.sum())
    keep = a != b
    pairs = np.unique(np.sort(np.stack([a[keep], b[keep]]), axis=0), axis=1)
    return phi, eta, pairs


@pytest.mark.parametrize("spec", [S.INFERENCE, S.INFERENCE_G2_FALLBACK, S.SectorSpec(n_phi=5, halo_phi=0.1, phi0=0.3),
                                  S.SectorSpec(n_phi=4, halo_phi=0.2, eta_edges=(-1.0, 0.0, 1.5), halo_eta=0.3),
                                  S.window_spec(2.9)])
def test_partition_halos_and_stitching(spec):
    phi, eta, pairs = _event()
    n, E = len(phi), pairs.shape[1]
    core = S.core_sector(phi, eta, spec)
    assert core.min() >= 0 and core.max() < spec.n_sectors
    secs = S.split(phi, eta, pairs, spec)
    S.check_partition(secs, E)
    owner = S.edge_owner(phi, eta, pairs[0], pairs[1], spec)
    for s in secs:
        assert np.all(np.diff(s.hits) > 0) and np.all(np.diff(s.edges) > 0)
        # local edges point to the right global hits
        assert np.array_equal(s.hits[s.local_edges[0]], pairs[0, s.edges])
        assert np.array_equal(s.hits[s.local_edges[1]], pairs[1, s.edges])
        # induced subgraph: every edge with both ends among the sector hits is in it
        m = np.zeros(n, dtype=bool)
        m[s.hits] = True
        assert np.array_equal(np.flatnonzero(m[pairs[0]] & m[pairs[1]]), s.edges)
        # owned edges are exactly the edges whose midpoint lies in this core
        assert np.array_equal(s.edges[s.owned], np.flatnonzero(owner == s.index))
        # core and halo hits are in; hits outside core + halo only as ends of owned edges
        inh = S.in_sector(phi, eta, spec, s.index) | (core == s.index)
        assert inh[s.hits].sum() == s.n_core + s.n_halo
        assert set(np.flatnonzero(inh)) <= set(s.hits)
        assert s.n_core == int((core == s.index).sum())
    ids = np.arange(E, dtype=np.float32)
    st = S.stitch_scores(E, secs, [ids[s.edges] for s in secs])
    assert np.array_equal(st, ids)
    with pytest.raises(ValueError):
        S.stitch_scores(E, secs[1:], [ids[s.edges] for s in secs[1:]])


def test_membership_and_halo_width():
    spec = S.INFERENCE                                   # 45 deg cores, +-22.5 deg halos
    phi = np.radians(np.array([-179.0, -150.0, -136.0, -134.0, -112.0, -91.0, 179.0, 0.0]))
    eta = np.zeros_like(phi)
    m = S.membership(phi, eta, spec)
    # sector 0 core [-180, -135), halo to -112.5 and (wrapping) to 157.5
    assert m[:, 0].tolist() == [True, True, True, True, False, False, True, False]
    assert S.core_sector(phi, eta, spec).tolist() == [0, 0, 0, 1, 1, 1, 7, 4]
    assert (m.sum(1) >= 1).all()


def test_owner_is_short_arc_midpoint_across_pi():
    spec = S.SectorSpec(n_phi=8, halo_phi=0.0)
    phi = np.radians(np.array([179.0, -177.0, 100.0, 110.0]))
    eta = np.zeros(4)
    own = S.edge_owner(phi, eta, np.array([0, 2]), np.array([1, 3]), spec)
    # midpoint of 179 and -177 deg is -179 deg (sector 0), not 1 deg; 100 / 110 -> 105 deg (sector 6)
    assert own.tolist() == [0, 6]


def test_window_spec_core():
    w = S.window_spec(np.radians(30.0))
    b = S.sector_bounds(w, 0)
    assert np.isclose(np.degrees(b["phi_center"]), 30.0) and np.isclose(np.degrees(2 * b["phi_half"]), 22.5)
    assert np.isclose(np.degrees(w.halo_phi), 11.25) and w.n_phi == 16
    with pytest.raises(ValueError):
        S.window_spec(0.0, core_width=np.radians(25.0))


def test_candidate_triplets_count():
    rng = np.random.default_rng(3)
    e = np.unique(np.sort(rng.integers(0, 50, (2, 300)), axis=0), axis=1)
    e = e[:, e[0] != e[1]]
    brute = 0
    for j in range(50):
        d = int(((e[0] == j) | (e[1] == j)).sum())
        brute += d * (d - 1) // 2
    assert S.candidate_triplets(50, e) == brute


def _tracks_event(seed=1, n_tracks=60, hits_per=10):
    """Toy tracks: each a run of hits in phi (some cross sector boundaries), plus noise hits."""
    rng = np.random.default_rng(seed)
    phi, eta, pid = [], [], []
    for t in range(n_tracks):
        p0 = rng.uniform(-np.pi, np.pi)
        span = rng.uniform(0.01, 0.6)
        phi.append(S.wrap(p0 + np.linspace(0, span, hits_per)))
        eta.append(np.full(hits_per, rng.uniform(-2, 2)))
        pid.append(np.full(hits_per, t + 1))
    phi.append(rng.uniform(-np.pi, np.pi, 100))
    eta.append(rng.uniform(-2, 2, 100))
    pid.append(np.zeros(100, dtype=int))
    phi, eta, pid = (np.concatenate(x) for x in (phi, eta, pid))
    # chain edges of every track plus random noise edges
    edges = []
    for t in range(n_tracks):
        h = np.flatnonzero(pid == t + 1)
        edges.append(np.stack([h[:-1], h[1:]]))
    edges = np.concatenate(edges, axis=1)
    return phi, eta, pid, edges


def _oracle(pid):
    return np.where(pid != 0, np.unique(pid, return_inverse=True)[1], -1)


@pytest.mark.parametrize("spec", [S.INFERENCE, S.SectorSpec(n_phi=16, halo_phi=np.radians(5.0))])
def test_stitch_tracks_oracle(spec):
    from hnl.eval.matching import FastMatch
    phi, eta, pid, edges = _tracks_event()
    secs = S.split(phi, eta, edges, spec)
    labs = []
    for s in secs:
        lp = pid[s.hits]
        labs.append(np.where(lp != 0, np.unique(lp, return_inverse=True)[1], -1))
    out = S.stitch_tracks(len(phi), secs, labs, phi, eta, spec)
    fm = FastMatch(pid)
    got = fm.match(out)["dm_matched"]
    ref = fm.match(_oracle(pid))["dm_matched"]
    real = fm.particles != 0
    # every toy track is recovered (tracks longer than the halo are seen in part by two sectors and merged)
    assert got[real].all() and ref[real].all()


def test_stitch_tracks_rules():
    spec = S.SectorSpec(n_phi=2, halo_phi=np.radians(30.0), phi0=0.0)   # cores [0, 180) and [180, 360)
    deg = np.radians
    # track A: 6 hits in core 0 near the boundary at 180 deg (seen whole by both sectors through the halo)
    # track B: 8 hits from 160 to 205 deg (seen in part by each sector beyond the halo? no: inside both halos)
    phi = deg(np.r_[np.linspace(150, 170, 6), np.linspace(160, 205, 8), [175.0, 185.0, 186.0]])
    eta = np.zeros(len(phi))
    pid = np.r_[np.full(6, 1), np.full(8, 2), [3, 3, 3]]
    secs = S.split(phi, eta, np.zeros((2, 0), dtype=np.int64), spec)
    labs = [np.unique(pid[s.hits], return_inverse=True)[1] for s in secs]
    out = S.stitch_tracks(len(phi), secs, labs, phi, eta, spec)
    for p in (1, 2, 3):
        lab = out[pid == p]
        assert (lab >= 0).all() and len(np.unique(lab)) == 1
    assert len(np.unique(out)) == 3
    # two different tracks claiming one hit: the hit goes to the track owned by its core sector
    phi2 = deg(np.r_[[170.0, 172.0, 174.0], [185.0, 187.0, 189.0], 179.0])
    labs2 = []
    secs2 = S.split(phi2, np.zeros(7), np.zeros((2, 0), dtype=np.int64), spec)
    for s in secs2:
        g = s.hits
        lab = np.full(len(g), -1)
        if s.index == 0:            # sector 0 puts the shared hit (179 deg, core 0) on track X
            lab[np.isin(g, [0, 1, 2, 6])] = 0
            lab[np.isin(g, [3, 4, 5])] = 1
        else:                       # sector 1 puts it on track Y
            lab[np.isin(g, [0, 1, 2])] = 0
            lab[np.isin(g, [3, 4, 5, 6])] = 1
        labs2.append(lab)
    out2 = S.stitch_tracks(7, secs2, labs2, phi2, np.zeros(7), spec)
    assert out2[6] == out2[0] and out2[6] != out2[3]


def test_full_event_layout_is_the_event_graph():
    """FULL_EVENT (S5 scoring, hnl.trackml.infer --sectors full): one sector with every hit and edge in the event's
    order, all edges owned, so the forward is G2's full-event forward and stitching returns its scores unchanged."""
    from hnl.trackml.infer import sector_layout
    assert sector_layout({}, "full") is S.FULL_EVENT
    phi, eta, pairs = _event(seed=3)
    (sec,) = S.split(phi, eta, pairs, S.FULL_EVENT)
    assert np.array_equal(sec.hits, np.arange(len(phi))) and np.array_equal(sec.edges, np.arange(pairs.shape[1]))
    assert np.array_equal(sec.local_edges, pairs) and sec.owned.all()
    assert sec.n_core == len(phi) and sec.n_halo == 0 and sec.n_extra == 0
    sc = np.random.default_rng(0).random(pairs.shape[1]).astype(np.float32)
    assert np.array_equal(S.stitch_scores(pairs.shape[1], [sec], [sc]), sc)
