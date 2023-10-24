"""Module for testing igmm module."""


import numpy as np
import pytest

import pylambertw.igmm


def test_igmm_works():
    rng = np.random.RandomState(42)
    y = rng.standard_cauchy(size=1000)
    clf = pylambertw.igmm.IGMM()
    clf.fit(y)

    x = clf.transform(y)
    assert pylambertw.utils.moments.kurtosis(x) == pytest.approx(3.0, 0.01)
