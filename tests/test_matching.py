"""Double-majority matching and track metrics on toy events (docs/definitions.md §5)."""

import numpy as np
import pandas as pd

from hnl.eval import matching as M


def test_dm_basic():
    # particle 1: 6 hits, particle 2: 4 hits, particle 3: 2 hits
    pid = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3])
    # track 0: 5 hits of p1 + 1 of p2 -> DM p1 (5/6 of track, 5/6 of p1)
    # track 1: 2 hits of p2 + 1 of p1 -> track majority p2 (2/3) but p2 shares 2/4 = 50 % -> not > 50 %: no DM
    # track 2: the 2 hits of p3 -> DM p3, but |T| = 2 < 5 so it never counts as fake either way
    track = np.array([0, 0, 0, 0, 0, 1, 0, 1, 1, -1, 2, 2])
    part, trk = M.match_event(pid, track)
    assert part.loc[1, "dm_matched"] and part.loc[1, "dm_track"] == 0
    assert not part.loc[2, "dm_matched"]
    assert part.loc[3, "dm_matched"]
    assert trk.loc[0, "dm_particle"] == 1 and not trk.loc[0, "is_fake"]
    assert trk.loc[1, "dm_particle"] == -1 and not trk.loc[1, "is_fake"]     # only 3 hits: not counted
    assert part.loc[2, "n_sm_tracks"] == 0                                   # track 1 (3 hits) is below the size cut
    part3, _ = M.match_event(pid, track, min_track_hits=3)
    assert part3.loc[2, "n_sm_tracks"] == 1                                  # with a 3-hit cut it is SM-matched to p2


def test_strict_majority_at_exactly_half():
    pid = np.array([7] * 4 + [8] * 4)
    track = np.array([0, 0, -1, -1, 0, 0, -1, -1])                          # 2 + 2 hits: 50 % each, no majority
    part, trk = M.match_event(pid, track)
    assert not part.dm_matched.any()


def test_fake_and_duplicates():
    pid = np.array([1] * 10 + [2] * 10 + [3] * 6)
    # p1 (10 hits) split into tracks of 6 and 4 hits: DM to the 6-hit one; the 4-hit piece counts as a duplicate only
    # if the duplicate size cut (default 5 hits) is lowered to 4
    # track 2 mixes 3 hits of p2 + 3 of p3: no majority -> fake (|T| = 6 >= 5)
    track = np.array([0] * 6 + [1] * 4 + [2] * 3 + [3] * 7 + [2] * 3 + [4] * 3)
    part, trk = M.match_event(pid, track, min_track_hits=4)
    assert part.loc[1, "dm_matched"] and part.loc[1, "n_sm_tracks"] == 2
    assert trk.loc[2, "is_fake"]
    assert part.loc[2, "dm_matched"]                                         # track 3: 7 of p2's 10 hits
    s = M.summarize(part, trk)
    assert s["n_tracks_ge5"] == 3 and abs(s["fake_rate"] - 1 / 3) < 1e-12
    assert abs(s["duplicate_rate"] - 1 / 3) < 1e-12
    part5, _ = M.match_event(pid, track)                                     # default 5-hit cut: no duplicate
    assert part5.loc[1, "n_sm_tracks"] == 1


def test_pair_merge():
    pid = np.array([1] * 6 + [2] * 6)
    merged = np.array([0] * 6 + [0] * 3 + [1] * 3)                          # track 0 holds all of p1, half of p2
    split = np.array([0] * 6 + [1] * 6)
    assert M.pair_merged(pid, merged, 1, 2)
    assert not M.pair_merged(pid, split, 1, 2)


def test_truth_tracks_are_fully_efficient():
    rng = np.random.default_rng(0)
    pid = rng.integers(0, 50, 500)
    part, trk = M.match_event(pid, pid.copy())
    assert part.dm_matched.all() and not trk.is_fake.any()


def test_fast_match_equals_reference():
    """FastMatch reproduces match_event / pair_merged on random labellings (incl. unassigned hits, ties)."""
    rng = np.random.default_rng(7)
    for trial in range(300):
        n = int(rng.integers(1, 60))
        pid = rng.integers(1, int(rng.integers(2, 12)), size=n) * 1000 + 3
        track = rng.integers(-1, int(rng.integers(1, 10)), size=n)
        if trial % 7 == 0:
            track[:] = -1
        fm = M.FastMatch(pid)
        m = fm.match(track)
        if (track >= 0).any():
            part, trk = M.match_event(pid, track)
            part = part.reindex(fm.particles)
            assert np.array_equal(part.dm_matched.to_numpy(), m["dm_matched"])
            assert np.array_equal(part.n_sm_tracks.to_numpy(), m["n_sm_tracks"])
            assert int((trk.n_track >= 5).sum()) == m["n_tracks_ge5"]
            assert int(trk.is_fake.sum()) == m["n_fake"]
            # connectivity: DM without purity; implied by DM
            best = part.best_shared.to_numpy()
            n_p = pd.Series(pid).value_counts().reindex(fm.particles).to_numpy()
            assert np.array_equal(m["connected"], 2 * best > n_p)
            assert (m["connected"] >= m["dm_matched"]).all()
            for pa in fm.particles[:3]:
                for pb in fm.particles[-3:]:
                    assert M.pair_merged(pid, track, int(pa), int(pb)) == fm.merged(m, int(pa), int(pb))
        else:
            assert not m["dm_matched"].any() and m["n_tracks_ge5"] == 0 and m["n_fake"] == 0
