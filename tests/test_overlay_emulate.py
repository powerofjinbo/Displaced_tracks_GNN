"""hnl.overlay.emulate: hole counting, the per-hit drop rate, stop and hazard bookkeeping against an expected layer
sequence, the fallback of sparse hazard cells, and the application to sim hits (stops, inefficiency, charge factor,
reproducibility). Synthetic inputs only."""

import json

import numpy as np
import pandas as pd
import pytest

from hnl.overlay import emulate as E


def test_interior_holes_counts_only_between_first_and_last():
    pres = np.array([[1, 0, 1, 1, 0],          # interior positions 1, 2: one hole at 1
                     [0, 1, 1, 0, 1],          # interior 2, 3: hole at 3
                     [1, 0, 0, 0, 0]], bool)   # one hit only: not counted
    ni, nh = E.interior_holes(pres)
    assert ni.tolist() == [0, 1, 2, 1, 0]
    assert nh.tolist() == [0, 1, 0, 1, 0]


def test_barrel_and_disk_presence_and_rates():
    # track 0: all 10 barrel layers but (13, 4); track 1: first layer only; track 2: pixel disks 9 / 2, 4, 8
    rows = [(0, v, l) for v, l in E.BARREL_LAYERS if (v, l) != (13, 4)] + [(1, 8, 2)]
    rows += [(2, 9, 2), (2, 9, 4), (2, 9, 8)]
    t, v, l = (np.array(x) for x in zip(*rows))
    central = np.array([True, True, False])
    forward = np.array([False, False, True])
    r = E.hole_rates(t, v, l, 3, central, forward)
    assert r["holes"][1304] == [1, 1] and r["holes"][1302] == [1, 0] and r["holes"][802] == [0, 0]
    assert r["holes"][906] == [1, 1] and r["holes"][904] == [1, 0] and r["holes"][702] == [0, 0]
    assert E.last_barrel_index(r["barrel_presence"]).tolist() == [9, 0, -1]


@pytest.mark.parametrize("h,g,f2", [(0.01, 0.0, 0.0), (0.01, 0.0, 0.3), (0.012, 0.006, 0.2), (0.005, 0.0, 0.9)])
def test_per_hit_rate_inverts_the_layer_model(h, g, f2):
    eps = E.per_hit_rate(h, g, f2)
    back = g + (1 - g) * (eps * (1 - f2) + eps ** 2 * f2)
    assert back == pytest.approx(h, abs=1e-12)
    assert E.per_hit_rate(0.004, 0.006, 0.1) == 0.0


def test_stop_table_and_hazards_with_spurious_correction():
    # expected sequence of 4 layers for every track; observed: track 0 complete, track 1 stops after pos 1,
    # track 2 misses only the last layer (looks like a stop at pos 2), track 3 has a hole at pos 1
    codes = [802, 804, 806, 808]
    ft, fc, fo = [], [], []
    for t in range(4):
        for i, c in enumerate(codes):
            ft.append(t), fc.append(c), fo.append(i)
    seq = E.crossing_sequence(np.array(ft), np.array(fc), np.array(fo))
    assert seq.groupby("track").pos.max().tolist() == [3, 3, 3, 3]
    obs = {0: codes, 1: codes[:2], 2: codes[:3], 3: [802, 806, 808]}
    ot = np.array([t for t, cs in obs.items() for _ in cs])
    oc = np.array([c for cs in obs.values() for c in cs])
    st = E.stop_table(seq, ot, oc)
    k = st.groupby("track").k.first().to_dict()
    assert k == {0: 3, 1: 1, 2: 2, 3: 3}
    stops = st[st.stop].set_index("track").code.to_dict()
    assert stops == {1: 804, 2: 806}
    exp = st[st.exposure].groupby("code").size().to_dict()
    assert exp == {802: 4, 804: 4, 806: 3}
    miss = E.miss_rates(st)
    assert miss[804] == pytest.approx(1 / 3)          # interior crossings at 804: tracks 0, 2, 3; track 3 misses
    assert miss[802] == 0.0
    miss_last = pd.Series({808: 0.5})
    cnt = E.hazard_counts(st, miss_last, species=np.full(4, 2), pbin=np.zeros(4, np.int64))
    c806 = cnt[cnt.code == 806].iloc[0]
    # pen population at 806 (pos 2 observed): tracks 0, 2, 3 -> 3 x miss(808) = 1.5 expected spurious stops
    assert c806.stop == 1 and c806.spurious == pytest.approx(1.5) and c806.exposure == 3


def _lam(k, n):
    """lambda with 1 - exp(-lambda) = k / n (unit path factor)."""
    return -np.log(1 - k / n)


def test_solve_lambda_matches_the_exact_hazard():
    rng = np.random.default_rng(0)
    s = 1 + rng.exponential(2.0, 20000)
    for lam in (0.001, 0.01, 0.03):
        K = np.sum(1 - np.exp(-lam * s))
        got = E.solve_lambda(K, s.sum(), (s ** 2).sum(), (s ** 3).sum())
        assert got == pytest.approx(lam, rel=0.02)
    assert E.solve_lambda(-3.0, 10.0, 10.0, 10.0) == 0.0


def test_resolve_hazards_falls_back_to_coarser_cells():
    exp = np.array([3000, 100, 50, 10], float)
    cnt = pd.DataFrame({"sp": [2, 2, 2, 3], "pb": [0, 1, 0, 0], "code": [802, 802, 804, 802], "exposure": exp,
                        "S1": exp, "S2": exp, "S3": exp, "stop": [30, 5, 1, 1], "spurious": [0.0, 0.0, 0.0, 0.0]})
    hz = E.resolve_hazards(cnt, min_exposure=1000)
    pi = hz["pi"]
    tol = dict(rel=2e-3)
    assert pi["802"]["level"][0] == 0 and pi["802"]["lam"][0] == pytest.approx(_lam(30, 3000), **tol)
    assert pi["802"]["level"][1] == 1 and pi["802"]["lam"][1] == pytest.approx(_lam(35, 3100), **tol)
    # pi at 804 (50 exposures, also 50 over all p): species, p bin, volume (volume 8 in p bin 0: 3050 exposures)
    assert pi["804"]["level"][0] == 2 and pi["804"]["lam"][0] == pytest.approx(_lam(31, 3050), **tol)
    # p bin 1 at 804: (pi, 1, volume 8) has 100 exposures, so species and volume pooled over p (3150)
    assert pi["804"]["level"][1] == 3 and pi["804"]["lam"][1] == pytest.approx(_lam(36, 3150), **tol)
    # K at 802 (10 exposures): hadrons pooled in volume 8
    assert hz["K"]["802"]["level"][0] == 4 and hz["K"]["802"]["lam"][0] == pytest.approx(_lam(37, 3160), **tol)


def test_path_factor():
    # barrel: radial track 1, at 60 degrees to the radial direction 2; disk: along z 1, at eta = 1 cosh/sinh ratio
    pf = E.path_factor([8, 8, 9, 9], [30, 30, 50, 50], [0, 0, 0, 0], [1, 0.5, 0, 1.0],
                       [0, 0, 0, 0], [0, np.sqrt(3) / 2, 1, np.sinh(1.0)])
    assert pf[:3] == pytest.approx([1, 2, 1]) and pf[3] == pytest.approx(np.cosh(1.0) / np.sinh(1.0))
    assert E.path_factor([8], [30], [0], [0], [1], [0])[0] == E.S_MAX


def _tables(eps=0.0, hazard=0.0) -> E.Tables:
    codes = [c for c in range(700, 1900)]
    model = {"p_edges": [0.0, 2.0, 4.0, 8.0, 1e9],
             "eps_per_hit": {str(c): eps for c in codes},
             "hazard": {sp: {str(c): {"lam": [-np.log(1 - hazard)] * 4, "level": [0] * 4, "exposure": [9999] * 4}
                             for c in codes} for sp in ("pi", "K", "p")}}
    return E.Tables({"model": model})


def _event(n_particles=200, pdg=211):
    rows = []
    for p in range(1, n_particles + 1):
        for i, (v, l) in enumerate(E.BARREL_LAYERS):
            rows.append((p, v, l, i))
    h = pd.DataFrame(rows, columns=["particle_id", "volume", "layer", "order"]).sample(frac=1, random_state=1)
    h["path_factor"] = 1.0
    parts = pd.DataFrame({"particle_id": np.arange(1, n_particles + 1), "pdg": pdg, "p": 3.0})
    return h.reset_index(drop=True), parts


def test_emulate_identity_without_effects():
    h, p = _event()
    r = E.emulate(h, p, _tables(), np.random.default_rng(0))
    assert r["keep"].all() and not r["stopped"].any() and not r["dropped"].any()


def test_emulate_stops_truncate_after_the_stop_layer():
    h, p = _event(n_particles=300)
    r = E.emulate(h, p, _tables(hazard=0.2), np.random.default_rng(1))
    d = h.assign(keep=r["keep"]).sort_values(["particle_id", "order"])
    for _, g in d.groupby("particle_id"):
        k = g.keep.to_numpy()
        assert k[0]                                      # the first crossing is never removed by a stop
        n = k.sum()
        assert k[:n].all() and not k[n:].any()           # kept hits are a prefix
    reach = d.groupby("particle_id").keep.last().mean()
    assert reach == pytest.approx(0.8 ** 9, abs=0.06)
    # leptons are never stopped
    h2, p2 = _event(pdg=13)
    assert E.emulate(h2, p2, _tables(hazard=0.5), np.random.default_rng(2))["keep"].all()


def test_emulate_inefficiency_rate():
    h, p = _event(n_particles=2000)
    r = E.emulate(h, p, _tables(eps=0.05), np.random.default_rng(3))
    assert r["dropped"].mean() == pytest.approx(0.05, abs=0.006)
    assert np.array_equal(r["keep"], ~r["dropped"])


def test_emulate_is_reproducible_and_independent_of_row_order():
    h, p = _event(n_particles=100)
    t = _tables(eps=0.1, hazard=0.1)
    a = E.emulate(h, p, t, np.random.default_rng(7))
    b = E.emulate(h, p, t, np.random.default_rng(7))
    assert np.array_equal(a["keep"], b["keep"]) and np.array_equal(a["stopped"], b["stopped"])
    perm = np.random.default_rng(0).permutation(len(h))
    c = E.emulate(h.iloc[perm].reset_index(drop=True), p, t, np.random.default_rng(7))
    assert np.array_equal(a["keep"][perm], c["keep"])


def test_emulate_rejects_unknown_particles():
    h, p = _event(n_particles=3)
    with pytest.raises(KeyError):
        E.emulate(h, p.iloc[:2], _tables(), np.random.default_rng(0))


def test_emulation_hash_changes_with_the_model(tmp_path):
    t = _tables()
    a = tmp_path / "a.json"
    a.write_text(json.dumps(t.d))
    h1 = E.emulation_hash(str(a))
    t.d["model"]["eps_per_hit"]["802"] = 0.5
    a.write_text(json.dumps(t.d))
    assert E.emulation_hash(str(a)) != h1


def test_p_bin_edges():
    assert E.p_bin([0.5, 1.99, 2.0, 7.9, 8.0, 500.0]).tolist() == [0, 0, 1, 2, 3, 3]


def test_module_overrides_flag_regions_without_hits():
    # 400 tracks crossing 802 then 804; module 7 of 802 never has an observed hit, module 3 misses 1 %
    rows, obs = [], []
    for t in range(400):
        m = 7 if t < 200 else 3
        rows += [(t, 802, 0, m), (t, 804, 1, 1)]
        if m == 3 and t % 100 != 0:
            obs.append((t, 802))
        obs.append((t, 804))
    t_, c_, o_, m_ = (np.array(x) for x in zip(*rows))
    seq = E.crossing_sequence(t_, c_, o_, module=m_)
    ot, oc = (np.array(x) for x in zip(*obs))
    st = E.stop_table(seq, ot, oc)
    miss = E.miss_rates(st)
    mo = E.module_overrides(st, miss, min_n=100, min_excess=0.05, min_z=5)
    assert mo[["code", "module"]].values.tolist() == [[802, 7]] and mo.miss.iloc[0] == 1.0


def test_emulate_applies_module_overrides():
    h, p = _event(n_particles=500)
    h["module"] = np.where((h.volume == 8) & (h.layer == 2), np.where(h.particle_id % 2 == 0, 5, 6), 1)
    t = _tables(eps=0.0)
    t.d["model"]["eps_module"] = {"802:5": 1.0}
    t = E.Tables(t.d)
    r = E.emulate(h, p, t, np.random.default_rng(4))
    m5 = (h.module == 5).to_numpy() & (h.layer == 2).to_numpy() & (h.volume == 8).to_numpy()
    assert not r["keep"][m5].any() and r["keep"][~m5].all()
