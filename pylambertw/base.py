"""Base class holding core class and result definitions."""


import dataclasses
import enum
from typing import Any, Dict, Union

import numpy as np
import torchlambertw.distributions as lwd


@dataclasses.dataclass
class DeltaEstimate:
    """Class for keeping Lambert W x F parameters."""

    delta: float
    method: str
    n_iterations: int
    converged: bool
    optimizer_result: Any


_EPS = np.finfo(np.float32).eps

_FLOAT_OR_ARRAY = Union[float, np.ndarray]


class LambertWType(enum.Enum):
    """Which type of Lambert W x F transformations."""

    # Skewed Lambert W x F
    S = "s"
    # Heavy-tailed Lambert W x F
    H = "h"
    # double heavy-tailed Lambert W x F
    HH = "hh"


def _is_eq_value(x: _FLOAT_OR_ARRAY, value: float) -> bool:
    """Returns True if 'x' is equal to value; False otherwise."""
    l1_norm = np.sum(np.abs(x - value))
    return l1_norm < _EPS


def _to_one_dim_array(x: _FLOAT_OR_ARRAY) -> np.ndarray:
    if isinstance(x, float):
        return np.array([x])
    if isinstance(x, np.ndarray):
        if len(x.shape) == 0:
            return np.array([x])
        if len(x.shape) == 1:
            return x

    raise ValueError("Could not convert successfully to 1-dim array.")


def _check_params(
    gamma: _FLOAT_OR_ARRAY, delta: _FLOAT_OR_ARRAY, alpha: _FLOAT_OR_ARRAY
) -> None:
    """checks that parameters are correctly set."""
    if not _is_eq_value(gamma, 0.0):
        assert _is_eq_value(delta, 0.0)
        assert _is_eq_value(alpha, 1.0)

    if not _is_eq_value(delta, 0.0):
        assert _is_eq_value(gamma, 0.0)

    assert np.all(alpha > 0.0)
    assert len(delta.shape) == 1


@dataclasses.dataclass
class LambertWParams:
    """Class for keeping Lambert W x F parameters."""

    gamma: np.ndarray = 0.0
    delta: np.ndarray = 0.0
    alpha: np.ndarray = 1.0

    def __post_init__(self):
        self.gamma = _to_one_dim_array(self.gamma)
        self.delta = _to_one_dim_array(self.delta)
        self.alpha = _to_one_dim_array(self.alpha)
        _check_params(self.gamma, self.delta, self.alpha)

    @property
    def lambertw_type(self):
        if not _is_eq_value(self.gamma, 0.0):
            return LambertWType.S

        if len(self.delta) == 1:
            return LambertWType.H

        if len(self.delta) == 2:
            return LambertWType.HH

        raise ValueError(
            "Lambert W Parameters gamma, delta, alpha do not uniquely identify the type."
        )

    def to_numpy(self):
        return np.concatenate([self.gamma, self.delta, self.alpha])

    def __repr__(self):
        out = ""
        if self.lambertw_type == LambertWType.S:
            out = f"gamma: {self.gamma}"
        if self.lambertw_type == LambertWType.H:
            out = f"delta: {self.delta}; alpha: {self.alpha}"

        return out


@dataclasses.dataclass
class Tau:
    """Class for keeping Lambert W x F parameters for transforming data.

    Uses loc/scale for both mean_variance and location_scale families.
    """

    loc: np.ndarray
    scale: np.ndarray
    lambertw_params: LambertWParams

    def __post_init__(self):
        self.loc = _to_one_dim_array(self.loc)
        self.scale = _to_one_dim_array(self.scale)

    def to_numpy(self):
        return np.concatenate([self.loc, self.scale, self.lambertw_params.to_numpy()])

    def __repr__(self):
        out = f"Lambert W x F (type: '{self.lambertw_params.lambertw_type.value}')"
        out += "\n\t" + f"loc={self.loc}; scale={self.scale}"
        out += "\n\t" + str(self.lambertw_params)
        return out


@dataclasses.dataclass
class Theta:
    """Class for keeping Lambert W x F parameters."""

    beta: Dict[
        str,
        _FLOAT_OR_ARRAY,
    ]
    distribution_name: str
    lambertw_params: LambertWParams

    def __repr__(self):
        out = f"Lambert W x {self.distribution_name}"
        out += f" (type: '{self.lambertw_params.lambertw_type.value}')"
        out += "\n\t" + "; ".join([k + "=" + str(v) for k, v in self.beta.items()])
        out += "\n\t" + str(self.lambertw_params)
        return out

    @property
    def tau(self):
        """Converts Theta (distribution dependent) to Tau (transformation only)."""

        distr_constr = lwd.get_distribution_constructor(self.distribution_name)
        distr = distr_constr(**self.beta)

        return Tau(
            loc=distr.mean.numpy()
            if lwd.is_location_family(self.distribution_name)
            else 0.0,
            scale=distr.stddev.numpy(),
            lambertw_params=self.lambertw_params,
        )
