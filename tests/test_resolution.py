"""Module-frame hit resolution: axes, projected sigmas, multiple scattering."""

import numpy as np

from hnl.geometry import resolution as RS


def test_module_axes_barrel_and_endcap_strips():
    pc = np.array([0.3, 0.3])
    u, v = RS.axes(np.array([13, 14]), pc)
    tang = np.array([-np.sin(0.3), np.cos(0.3), 0.0])
    assert np.allclose(u[0], tang) and np.allclose(v[0], [0, 0, 1])                  # barrel strip: coarse along z
    assert np.allclose(u[1], tang) and np.allclose(v[1], [np.cos(0.3), np.sin(0.3), 0])  # endcap strip: coarse along r


def test_projected_sigmas():
    su, sv = RS.RES[14]
    pc = np.array([0.3])
    radial = np.array([[np.cos(0.3), np.sin(0.3)]])
    tang = np.array([[-np.sin(0.3), np.cos(0.3)]])
    floor = RS.MIN_PROJ * su
    # endcap strip: a radial track's circle normal is the tangent: fine; a track crossing tangentially: coarse
    assert np.isclose(RS.sigma_transverse(np.array([14]), pc, tang)[0], np.hypot(su, floor))
    assert np.isclose(RS.sigma_transverse(np.array([14]), pc, radial)[0], np.sqrt(sv**2 + floor**2))
    # z of an endcap hit: exact disk z, the radial r error moves the arc length: |cot| s_v for a radial track
    assert np.isclose(RS.sigma_z(np.array([14]), pc, radial, 2.0)[0], np.sqrt((2.0 * sv) ** 2 + floor**2))
    # barrel strip: the z sigma is the coarse axis; a radial track's transverse sigma is the fine axis
    sub, svb = RS.RES[13]
    assert np.isclose(RS.sigma_z(np.array([13]), pc, radial, 2.0)[0], np.sqrt(svb**2 + (RS.MIN_PROJ * sub) ** 2))
    assert np.isclose(RS.sigma_transverse(np.array([13]), pc, tang)[0], np.hypot(sub, RS.MIN_PROJ * sub))


def test_highland_and_crossing():
    assert np.isclose(RS.highland(1.0, 0.01), 0.0136 * 0.1 * (1 + 0.038 * np.log(0.01)))
    # endcap disk crossed at 45 degrees: x / X0 grows by sqrt(2)
    x = RS.crossing_x_x0(np.array([14]), np.array([0.0]), np.array([[1.0, 0.0]]), 1.0)[0]
    assert np.isclose(x, RS.X_X0[14] * np.sqrt(2))
    assert np.isclose(RS.momentum_gev(1000.0 / 0.6, 0.0), 1.0)
