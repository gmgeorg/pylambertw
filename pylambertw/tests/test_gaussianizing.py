"""Test gaussianizing module"""


import numpy as np
import pytest

import pylambertw.utils.moments
from pylambertw.preprocessing import gaussianizing as g


@pytest.mark.parametrize(
    "method,lambertw_type,tol",
    [("igmm", "h", 0.001), ("mle", "h", 2)],
)
def test_gaussianizing_works(method, lambertw_type, tol):
    rng = np.random.RandomState(42)
    y = rng.standard_cauchy(size=1000)
    clf = g.Gaussianizer(method=method, lambertw_type=lambertw_type)
    clf.fit(y)

    x = clf.transform(y)
    assert pylambertw.utils.moments.kurtosis(x) == pytest.approx(3.0, tol)
