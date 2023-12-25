"""Utlitiies for torch distributions."""

from typing import Callable, Dict

import numpy as np
import scipy.special
import torch
import torchlambertw.distributions as lwd


def torch_softplus(x: torch.tensor) -> torch.tensor:
    """softplus = log(exp(x) + 1)."""
    return torch.log(torch.exp(x) + 1)


def softplus_inverse(x: np.ndarray) -> np.ndarray:
    """Softplus inverse."""
    return np.log(np.exp(x) - 1.0)


def torch_linear(x: torch.tensor) -> torch.tensor:
    """Linear activation function."""
    return x


def linear_inverse(x):
    """y = f(x) It returns the input as it is."""
    return x


def torch_sigmoid(x: torch.tensor) -> torch.tensor:
    """Sigmoid function"""
    return 1.0 / (1.0 + torch.exp(-x))


def get_params_activations(distribution_name: str) -> Dict[str, Callable]:
    """Get activation functions for each distribution parameters."""
    assert isinstance(distribution_name, str)
    distr_constr = lwd.utils.get_distribution_constructor(distribution_name)
    param_names = lwd.utils.get_distribution_args(distr_constr)

    act_fns = {p: (torch_linear, linear_inverse) for p in param_names}

    for s in [
        "scale",
        "rate",
        "concentration",
        "concentration1",
        "concentration0",
        "df",
        "df1",
        "df2",
        "alpha",
    ]:
        if s in act_fns:
            act_fns[s] = (torch_softplus, softplus_inverse)

    for s in ["probs"]:
        if s in act_fns:
            act_fns[s] = (torch_sigmoid, scipy.special.logit)
    return act_fns


def estimate_params(x: np.ndarray, distribution_name: str) -> Dict[str, float]:
    """Estimates parameters for distribution. Default to 0 (logit)."""
    params_activation_fns = get_params_activations(distribution_name)

    # Default to 0.0 on before activation applied.
    init_params = {
        k: float(params_activation_fns[k][0](torch.tensor(0.0)).numpy())
        for k in params_activation_fns.keys()
    }
    if distribution_name == "Normal":
        # Robust estimate
        init_params["loc"] = (np.median(x) + np.mean(x)) / 2.0
        init_params["scale"] = 1.48 * np.mean(np.abs(x - init_params["loc"]))

    if distribution_name == "Exponential":
        init_params["rate"] = 1.0 / np.mean(x)

    if distribution_name in ["Laplace", "Cauchy"]:
        init_params["loc"] = np.median(x)
        init_params["scale"] = np.mean(np.abs(x - init_params["loc"]))

    if distribution_name == "LogNormal":
        init_params = estimate_params(np.log(x), "Normal")

    if distribution_name == "Uniform":
        init_params["low"] = np.min(x)
        init_params["high"] = np.max(x)

    if distribution_name == "StudentT":
        init_params.update(estimate_params(x, "Laplace"))
        # TODO: update
        init_params["df"] = 5

    if distribution_name == "Beta":
        init_params["concentration1"] = 1.0
        init_params["concentration0"] = 1.0

    if distribution_name == "Gamma":
        init_params["rate"] = np.mean(x) / np.var(x)
        init_params["concentration"] = np.mean(x) ** 2 / np.var(x)

    if distribution_name == "Weibull":
        init_params["scale"] = np.mean(x)
        init_params["concentration"] = 1.0

    return init_params
