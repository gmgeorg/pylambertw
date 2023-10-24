"""Module for testing mle module."""


import numpy as np
import pytest

import pylambertw.mle


def test_mle_works():
    rng = np.random.RandomState(42)
    y = rng.standard_cauchy(size=1000)
    clf = pylambertw.mle.MLE(distribution_name="Normal", max_iter=10)
    clf.fit(y)

    x = clf.transform(y)
    assert pylambertw.utils.moments.kurtosis(x) == pytest.approx(3.0, 0.5)


@pytest.mark.parametrize(
    "dist_name,lambertw_type",
    [("Normal", "h"), ("Gamma", "h"), ("Gamma", "s")],  # ("Normal", "s"),
)
def test_mle_works_for_distr_type(dist_name, lambertw_type):
    rng = np.random.RandomState(42)
    y = rng.standard_cauchy(size=1000) ** 2
    clf = pylambertw.mle.MLE(
        distribution_name=dist_name, max_iter=10, lambertw_type=lambertw_type
    )
    clf.fit(y)

    clf.transform(y)
