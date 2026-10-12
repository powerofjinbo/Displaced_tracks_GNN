"""Fragment merging on a split helix and on two different helices."""

import numpy as np

from hnl.trackbuild.merge_fragments import merge_fragments


def _helix(R, x0, y0, t0, dt, n, z0=0.0, cot=0.2):
    t = t0 + dt * np.arange(n)
    x, y = x0 + R * np.cos(t), y0 + R * np.sin(t)
    s = R * np.abs(t - t0)
    return np.hypot(x, y), np.arctan2(y, x), z0 + cot * s


def test_split_helix_is_merged_and_other_track_kept():
    # a 1.5 GeV prompt-like helix through 8 barrel radii, split 4 + 4; plus an unrelated helix
    R = 1.5 * 1666.7
    r, phi, z = _helix(R, 0.0, -R, np.pi / 2 - 0.0, 0.0, 1)
    # build hits at the generic-detector barrel radii by solving for t on the circle x^2 + y^2 = rho^2
    radii = np.array([32, 72, 116, 172, 260, 360, 500, 660], dtype=float)
    t = np.arcsin(radii / (2 * R))                      # circle through the origin, centre (0, -R)... chord geometry
    x, y = R * np.sin(2 * t), R * (1 - np.cos(2 * t)) * -1 + 0.0
    hr, hphi = np.hypot(x, y), np.arctan2(y, x)
    hz = 0.3 * R * 2 * t
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # second helix rotated by 1 rad, different curvature
    R2 = 5.0 * 1666.7
    t2 = np.arcsin(radii / (2 * R2))
    x2, y2 = R2 * np.sin(2 * t2), -R2 * (1 - np.cos(2 * t2))
    hr2, hphi2 = np.hypot(x2, y2), np.arctan2(y2, x2) + 1.0
    hz2 = -0.5 * R2 * 2 * t2
    r_all = np.r_[hr, hr2]
    phi_all = np.r_[hphi, hphi2]
    z_all = np.r_[hz, hz2]
    lab = np.r_[labels, np.full(8, 2)]
    out = merge_fragments(r_all, phi_all, z_all, lab)
    assert len(set(out[:8])) == 1                        # the two fragments are one track again
    assert out[8] != out[0] and len(set(out[8:])) == 1


def test_incompatible_close_fragments_not_merged():
    radii = np.array([32, 72, 116, 172, 260, 360, 500, 660], dtype=float)

    def hits(pt, charge, phi0, cot):
        R = pt * 1666.7
        t = np.arcsin(radii / (2 * R))
        x, y = R * np.sin(2 * t), -charge * R * (1 - np.cos(2 * t))
        return np.hypot(x, y), np.arctan2(y, x) + phi0, cot * R * 2 * t
    r1, p1, z1 = hits(1.2, +1, 0.0, 0.2)
    r2, p2, z2 = hits(1.2, -1, 0.0, 0.2)              # opposite charge: same start, bends the other way
    # inner half of track 1 and outer half of track 2 as two fragments
    r = np.r_[r1[:4], r2[4:]]
    phi = np.r_[p1[:4], p2[4:]]
    z = np.r_[z1[:4], z2[4:]]
    out = merge_fragments(r, phi, z, np.array([0, 0, 0, 0, 1, 1, 1, 1]))
    assert out[0] != out[4]


def test_interleaved_halves_are_merged_in_any_mode():
    # one helix with two hits per layer (overlapping modules); the hits alternate between two "half tracks"
    radii = np.repeat(np.array([32, 72, 116, 172, 260, 360], dtype=float), 2) + np.tile([0.0, 3.0], 6)
    R = 2.0 * 1666.7
    t = np.arcsin(radii / (2 * R))
    x, y = R * np.sin(2 * t), -R * (1 - np.cos(2 * t))
    r, phi, z = np.hypot(x, y), np.arctan2(y, x), 0.4 * R * 2 * t
    lab = np.tile([0, 1], 6)
    seq = merge_fragments(r, phi, z, lab)                       # sequential mode: interleaved halves are not candidates
    assert seq[0] != seq[1]
    out = merge_fragments(r, phi, z, lab, mode="any")
    assert len(set(out)) == 1


def test_no_merge_without_lever_arm():
    # two different particles' 2-hit fragments on one endcap disk region (all hits within a few mm in R): a tiny
    # circle fits any 4 such hits, so the union must be rejected (review finding: no lever arm, no merge)
    r = np.array([34.0, 34.5, 33.2, 33.9])
    phi = np.array([0.100, 0.102, 0.150, 0.151])
    z = np.array([1498.0, 1500.0, 1501.0, 1503.0])
    lab = np.array([0, 0, 1, 1])
    for mode in ("sequential", "any"):
        out = merge_fragments(r, phi, z, lab, mode=mode, max_dphi_dir=10.0, max_dcot=10.0)
        assert out[0] != out[2], mode
    # the lever-arm cut alone decides: with no minimum span the same union is accepted in "any" mode
    out = merge_fragments(r, phi, z, lab, mode="any", max_dphi_dir=10.0, max_dcot=10.0, min_span_mm=0.0)
    assert out[0] == out[2]
