"""Module for testing preprocessing module."""


import numpy as np
import pytest

from .. import base
from ..preprocessing import transforms


def _test_data():
    rng = np.random.RandomState(seed=42)
    x = rng.normal(size=10)
    return x


@pytest.mark.parametrize(
    "gamma",
    [(-0.1), (0.0), (0.1), (0.2)],
)
def test_w_gamma(gamma):
    u = _test_data()
    u_gamma = transforms.H_gamma(u, gamma=gamma)
    w_u_gamma = transforms.W_gamma(u_gamma, gamma=gamma, k=0)
    np.testing.assert_allclose(u, w_u_gamma)


@pytest.mark.parametrize(
    "delta",
    [(-0.1), (0.0), (0.1), (0.2)],
)
def test_w_delta(delta):
    u = _test_data()
    u_delta = transforms.G_delta(u, delta=delta)
    w_u_delta = transforms.W_delta(u_delta, delta=delta)
    np.testing.assert_allclose(u, w_u_delta)


@pytest.mark.parametrize(
    "loc,scale,delta,eps",
    [
        (0.0, 1.0, 0.0, 1e-6),
        (0.0, 1.0, 0.0001, 1e-2),
        (0.4, 2.0, 0.0, 1e-6),
        (
            0.4,
            2.0,
            0.0001,
            1e-2,
        ),  # small deviation from delta = 0 results in small deviation only
    ],
)
def test_identity_transform(loc, scale, delta, eps):
    x = _test_data()
    np_result = transforms.normalize_by_tau(
        y=x,
        tau=base.Tau(
            loc=loc,
            scale=scale,
            lambertw_params=base.LambertWParams(delta=delta),
        ),
    )
    np.testing.assert_allclose(np_result, x, atol=eps)


@pytest.mark.parametrize(
    "loc,scale,delta",
    [(0.0, 1.0, 0.5), (0.4, 2.0, 0.1), (0.4, 2.0, 0.001)],
)
def test_np_transform_inverse_equality(loc, scale, delta):
    x = _test_data()
    tau_tmp = base.Tau(
        loc=loc,
        scale=scale,
        lambertw_params=base.LambertWParams(delta=delta),
    )
    y = transforms.inverse_normalize_by_tau(x, tau_tmp)
    y_reverse = transforms.normalize_by_tau(y, tau_tmp)
    np.testing.assert_allclose(x, y_reverse)
