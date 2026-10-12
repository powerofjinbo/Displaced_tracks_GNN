"""Track-to-particle matching and track metrics (docs/definitions.md §5).

Inputs per event: `pid` (particle id of every hit, int64) and `track` (reconstructed track label of every hit, -1 for
hits on no track). A track T and a particle P DM-match ("double majority") if |T n P| > |T| / 2 and |T n P| > |P| / 2,
where |T| counts the hits of T and |P| all hits of P in the event. Single-majority (SM): |T n P| > |T| / 2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def overlap_table(pid: np.ndarray, track: np.ndarray) -> pd.DataFrame:
    """One row per (track, particle) pair sharing hits: n_shared, n_track (|T|), n_particle (|P|)."""
    pid = np.asarray(pid)
    track = np.asarray(track)
    n_particle = pd.Series(pid).value_counts()
    on = track >= 0
    df = pd.DataFrame({"track": track[on], "particle_id": pid[on]})
    n_track = df.groupby("track").size().rename("n_track")
    pairs = df.groupby(["track", "particle_id"]).size().rename("n_shared").reset_index()
    pairs = pairs.join(n_track, on="track")
    pairs["n_particle"] = n_particle.reindex(pairs.particle_id).to_numpy()
    pairs["dm"] = (2 * pairs.n_shared > pairs.n_track) & (2 * pairs.n_shared > pairs.n_particle)
    pairs["sm"] = 2 * pairs.n_shared > pairs.n_track
    return pairs


def match_event(pid: np.ndarray, track: np.ndarray, min_track_hits: int = 5):
    """Returns (per-particle frame indexed by particle_id: dm_matched, dm_track, n_sm_tracks (single-majority matches
    among tracks with >= min_track_hits hits), best_shared;
    per-track frame indexed by track: n_track, dm_particle (or -1), is_fake (|T| >= min_track_hits and no DM match))."""
    pairs = overlap_table(pid, track)
    dm = pairs[pairs.dm]
    # a particle DM-matches at most one track and a track at most one particle (strict majorities)
    assert not dm.particle_id.duplicated().any() and not dm.track.duplicated().any()
    up = np.unique(np.asarray(pid))
    part = pd.DataFrame(index=pd.Index(up, name="particle_id"))
    part["dm_matched"] = part.index.isin(dm.particle_id)
    part["dm_track"] = dm.set_index("particle_id").track.reindex(part.index).fillna(-1).astype(np.int64)
    big = pairs.sm & (pairs.n_track >= min_track_hits)          # duplicates are counted among tracks with >= 5 hits
    part["n_sm_tracks"] = pairs[big].groupby("particle_id").size().reindex(part.index).fillna(0).astype(int)
    part["best_shared"] = pairs.groupby("particle_id").n_shared.max().reindex(part.index).fillna(0).astype(int)
    trk = pairs.groupby("track").n_track.first().to_frame()
    trk["dm_particle"] = dm.set_index("track").particle_id.reindex(trk.index).fillna(-1).astype(np.int64)
    trk["is_fake"] = (trk.n_track >= min_track_hits) & (trk.dm_particle < 0)
    return part, trk


def pair_merged(pid: np.ndarray, track: np.ndarray, pa: int, pb: int) -> bool:
    """True if one track holds at least half of the hits of each of the two particles."""
    pairs = overlap_table(pid, track)
    a = pairs[(pairs.particle_id == pa) & (2 * pairs.n_shared >= pairs.n_particle)].track
    b = pairs[(pairs.particle_id == pb) & (2 * pairs.n_shared >= pairs.n_particle)].track
    return bool(set(a) & set(b))


class FastMatch:
    """numpy version of match_event / pair_merged for many track labellings of one event (same results; tested
    against the pandas reference in tests/test_matching.py). Particles are np.unique(pid)."""

    def __init__(self, pid: np.ndarray):
        self.particles, self.pidx = np.unique(np.asarray(pid), return_inverse=True)
        self.n_particle = np.bincount(self.pidx)
        self.P = len(self.particles)

    def match(self, track: np.ndarray, min_track_hits: int = 5) -> dict:
        """dm_matched, n_sm_tracks (per particle, order of self.particles), n_tracks_ge5, n_fake, and the DM / >= half
        tables needed by merged()."""
        track = np.asarray(track)
        on = track >= 0
        t, p = track[on], self.pidx[on]
        if len(t) == 0:
            z = np.zeros(self.P, dtype=bool)
            return {"dm_matched": z, "n_sm_tracks": np.zeros(self.P, dtype=np.int64), "n_tracks_ge5": 0, "n_fake": 0,
                    "pair_t": t, "pair_p": p, "pair_shared": t, "connected": z}
        tu, tinv = np.unique(t, return_inverse=True)
        n_track = np.bincount(tinv)
        key, n_shared = np.unique(tinv.astype(np.int64) * self.P + p, return_counts=True)
        pt, pp = key // self.P, key % self.P
        dm = (2 * n_shared > n_track[pt]) & (2 * n_shared > self.n_particle[pp])
        sm = 2 * n_shared > n_track[pt]
        dm_matched = np.zeros(self.P, dtype=bool)
        dm_matched[pp[dm]] = True
        big = sm & (n_track[pt] >= min_track_hits)
        n_sm = np.bincount(pp[big], minlength=self.P)
        track_dm = np.zeros(len(tu), dtype=bool)
        track_dm[pt[dm]] = True
        ge5 = n_track >= min_track_hits
        # connectivity: some track holds more than half of the particle's hits (DM without the purity condition, i.e.
        # what an oracle that removes foreign hits from the same tracks would match)
        best = np.zeros(self.P, dtype=np.int64)
        np.maximum.at(best, pp, n_shared)
        return {"dm_matched": dm_matched, "n_sm_tracks": n_sm, "n_tracks_ge5": int(ge5.sum()),
                "n_fake": int((ge5 & ~track_dm).sum()), "pair_t": pt, "pair_p": pp, "pair_shared": n_shared,
                "connected": 2 * best > self.n_particle}

    def merged(self, m: dict, pa: int, pb: int) -> bool:
        """pair_merged on the labelling of match() result m."""
        ia = np.searchsorted(self.particles, pa)
        ib = np.searchsorted(self.particles, pb)
        if ia >= self.P or ib >= self.P or self.particles[ia] != pa or self.particles[ib] != pb:
            return False
        half = 2 * m["pair_shared"] >= self.n_particle[m["pair_p"]]
        ta = m["pair_t"][half & (m["pair_p"] == ia)]
        tb = m["pair_t"][half & (m["pair_p"] == ib)]
        return bool(np.intersect1d(ta, tb).size)


def summarize(particles: pd.DataFrame, tracks: pd.DataFrame) -> dict:
    """particles: rows of targets with columns dm_matched, n_sm_tracks (any category filter applied beforehand);
    tracks: all reconstructed tracks with n_track, is_fake."""
    n = len(particles)
    k = int(particles.dm_matched.sum())
    eff = k / n if n else np.nan
    big = tracks[tracks.n_track >= 5]
    return {
        "n_targets": n, "n_matched": k, "efficiency": eff,
        "efficiency_binomial_err": float(np.sqrt(eff * (1 - eff) / n)) if n else np.nan,
        "duplicate_rate": float((particles.n_sm_tracks - 1).clip(lower=0).sum() / n) if n else np.nan,
        "n_tracks_ge5": int(len(big)), "fake_rate": float(big.is_fake.mean()) if len(big) else np.nan,
    }
