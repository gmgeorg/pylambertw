"""Module for iterative generalized methods of moments (IGMM) estimation."""

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
import sklearn

from . import base
from .preprocessing import transforms
from .utils import distributions, moments


def gamma_taylor(y: np.ndarray, skewness_y: Optional[float] = None) -> float:
    """Computes taylor approximation for the 'gamma' parameter."""
    y = y.ravel()
    if skewness_y is None:
        skewness_y = moments.skewness(y)

    if not isinstance(skewness_y, (int, float)):
        raise ValueError(f"skewness_y must be an int/float. Got {type(skewness_y)}.")

    # See Eq(4.5) of Goerg (2011)
    return skewness_y / 6.0


def delta_taylor(
    y: np.ndarray, kurtosis_y: Optional[float] = None, distribution_name: str = "normal"
) -> float:
    """Computes the taylor approximation of the 'delta' parameter given univariate data."""
    y = y.ravel()
    if kurtosis_y is None:
        kurtosis_y = moments.kurtosis(y)

    if not isinstance(kurtosis_y, (int, float)):
        raise ValueError(f"kurtosis_y must be an int/float. Got {type(kurtosis_y)}.")

    if kurtosis_y <= 0.0:
        raise ValueError(
            f"kurtosis_y must be a positive numeric value. Got {kurtosis_y}"
        )

    if distribution_name == "normal":
        if 66 * kurtosis_y - 162 > 0:
            delta_hat = max(0, 1 / 66 * (np.sqrt(66 * kurtosis_y - 162) - 6))
            delta_hat = min(delta_hat, 2)
        else:
            delta_hat = 0.0
    else:
        raise NotImplementedError(
            "Other distribution than 'normal' is not supported for the Taylor approximation."
        )

    return float(delta_hat)


def delta_gmm(
    z: np.ndarray,
    type: str = "h",
    kurtosis_x: float = 3.0,
    skewness_x: float = 0.0,
    delta_init: Optional[float] = None,
    tol: float = np.finfo(float).eps ** 0.25,
    not_negative: bool = False,
    lower: float = -1.0,
    upper: float = 5.0,
):
    """Computes an estimate of delta (tail parameter) per Taylor approximation of the kurtosis."""
    assert isinstance(kurtosis_x, (int, float))
    assert isinstance(skewness_x, (int, float))
    assert len(delta_init) <= 2 if delta_init is not None else True
    assert tol > 0
    assert lower < upper

    delta_init = delta_init or delta_taylor(z)

    def _obj_fct(delta: float):
        if not_negative:
            # convert delta to > 0
            delta = np.exp(delta)
        u_g = transforms.W_delta(z, delta=delta)
        if np.any(np.isinf(u_g)):
            return kurtosis_x**2

        empirical_kurtosis = moments.kurtosis(u_g)
        # for delta -> Inf, u.g can become (numerically) a constant vector
        # thus kurtosis(u.g) = NA.  In this case set empirical.kurtosis
        # to a very large value and continue.
        if np.isnan(empirical_kurtosis):
            empirical_kurtosis = 1e6

            error_msg = f"""
            Kurtosis estimate was NA. Setting to large value ({empirical_kurtosis})
            for optimization to continue.\n Double-check results (in particular the 'delta'
            estimate)
            """

            warnings.warn(error_msg)
        return (empirical_kurtosis - kurtosis_x) ** 2

    if not_negative:
        delta_init = np.log(delta_init + 0.001)

    delta_estimate: base.DeltaEstimate = None
    if not_negative:
        res = scipy.optimize.minimize(
            _obj_fct, delta_init, method="BFGS", tol=tol, options={"disp": False}
        )
        delta_estimate = base.DeltaEstimate(
            delta=res.x[0],
            n_iterations=res.nit,
            method="gmm",
            converged=res.success,
            optimizer_result=res,
        )
    else:
        res = scipy.optimize.minimize_scalar(
            _obj_fct, bounds=(lower, upper), method="bounded", options={"xatol": tol}
        )
        delta_estimate = base.DeltaEstimate(
            delta=res.x,
            n_iterations=res.nfev,
            method="gmm",
            converged=res.success,
            optimizer_result=res,
        )

    delta_hat = delta_estimate.delta
    if not_negative:
        delta_hat = np.exp(delta_hat)
        if np.abs(delta_hat - 1) < 1e-7:
            delta_hat = np.round(delta_hat, 6)

    delta_hat = np.minimum(np.maximum(delta_hat, lower), upper)
    delta_estimate.delta = delta_hat
    return delta_estimate


class IGMM(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Computes the IGMM for multivariate (column-wise) Lambert W x F distributions."""

    def __init__(
        self,
        lambertw_type: str = "h",
        skewness_x: float = 0.0,
        kurtosis_x: float = 3.0,
        max_iter: int = 100,
        lr: float = 0.01,
        not_negative: bool = True,
        location_family: bool = True,
        lower: float = 0.0,
        upper: float = 3.0,
        tolerance: float = 1e-6,
        verbose: int = 0,
    ):
        assert max_iter > 0
        assert verbose >= 0

        self.lambertw_type = base.LambertWType(lambertw_type)
        assert (
            self.lambertw_type == base.LambertWType.H
        ), "Only 'h' is supported in IGMM for now."
        self.skewness_x = skewness_x
        self.kurtosis_x = kurtosis_x
        self.max_iter = max_iter
        self.lr = lr
        self.verbose = verbose
        self.tolerance = tolerance

        self.location_family = location_family
        self.not_negative = not_negative
        self.lower = lower
        self.upper = upper
        self.total_iter = 0
        # estimated parameters
        self.tau = None
        self.tau_init = {}
        self.tau_trace = None

    def _initialize_params(self, data: np.ndarray):
        """Initializes parameters."""
        params_data = distributions.estimate_params(data, "Normal")

        z_init = (data - params_data["loc"]) / params_data["scale"]
        lambertw_params_init = base.LambertWParams(
            delta=delta_gmm(z_init, not_negative=self.not_negative).delta,
        )

        u_init = transforms.W_delta(
            z_init,
            delta=lambertw_params_init.delta,
        )
        params_init = distributions.estimate_params(u_init, "Normal")

        tau_init = base.Tau(
            loc=params_init["loc"],
            scale=params_init["scale"],
            lambertw_params=lambertw_params_init,
        )
        self.tau_init = tau_init

    def fit(self, data: np.ndarray):
        """Trains the IGMM of a Lambert W x F distribution based on methods of moments."""
        if len(data.shape) == 1:
            if isinstance(data, pd.Series):
                data = data.to_frame()
            else:
                data = data[:, np.newaxis]
        if data.shape[1] > 1:
            raise NotImplementedError(
                "IGMM only works for for univariate data. Use Gaussianizer() for > 1 columns."
            )
        self._initialize_params(data)

        tau_trace = np.zeros(shape=(self.max_iter + 1, 3))
        tau_trace[0,] = np.array([
            self.tau_init.loc,
            self.tau_init.scale,
            self.tau_init.lambertw_params.delta,
        ]).reshape(1, -1)

        for kk in range(self.max_iter):
            current = tau_trace[kk, :]
            if self.verbose:
                if (kk) % self.verbose == 0:
                    print(f"Epoch [{kk}/{self.max_iter}], Params: {current}")

            tau_tmp = base.Tau(
                loc=current[0],
                scale=current[1],
                lambertw_params=base.LambertWParams(delta=current[2]),
            )
            zz = (data - tau_tmp.loc) / tau_tmp.scale

            delta_estimate = delta_gmm(
                zz,
                delta_init=tau_tmp.lambertw_params.delta,
                kurtosis_x=self.kurtosis_x,
                tol=self.tolerance,
                not_negative=self.not_negative,
                lower=self.lower,
                upper=self.upper,
            )
            delta_hat = delta_estimate.delta

            uu = transforms.W_delta(zz, delta=delta_hat)
            xx = uu * tau_tmp.scale + tau_tmp.loc
            tau_trace[kk + 1,] = (np.mean(xx), np.std(xx), delta_hat)
            if not self.location_family:
                tau_trace[kk + 1, 0] = 0.0

            self.total_iter += delta_estimate.n_iterations
            tau_diff = tau_trace[kk + 1] - tau_trace[kk]
            if np.linalg.norm(tau_diff) < self.tolerance:
                break

        self.tau_trace = tau_trace[: (kk + 1)]
        self.tau_trace = pd.DataFrame(
            self.tau_trace,
            index=range(self.tau_trace.shape[0]),
            columns=["loc", "scale", "delta"],
        )

        self.tau = base.Tau(
            lambertw_params=base.LambertWParams(
                delta=tau_trace[kk, 2],
            ),
            loc=tau_trace[kk, 0],
            scale=tau_trace[kk, 1],
        )
        if self.verbose:
            print("IGMM\n----\n", self.tau)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transforms data y to the data based on IGMM estimate tau."""
        return transforms.normalize_by_tau(data, tau=self.tau)
