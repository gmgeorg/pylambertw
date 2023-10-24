"""Module for testing utils module."""


import numpy as np
import pytest
import torchlambertw.distributions as lwd

from pylambertw.utils import distributions as ud


@pytest.mark.parametrize(
    "dist_name",
    [
        ("Normal"),
        ("LogNormal"),
        ("Cauchy"),
        ("Laplace"),
        ("Uniform"),
        ("Exponential"),
        ("Weibull"),
        ("StudentT"),
    ],
)
def test_estimate_params(dist_name):
    rng = np.random.RandomState(42)
    x = rng.normal(100)
    params = ud.estimate_params(x, dist_name)
    constr = lwd.get_distribution_constructor(dist_name)
    param_names = lwd.get_distribution_args(constr)
    assert set(params.keys()) == set(param_names)
