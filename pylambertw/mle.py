"""Module for maximum likelihood estimation of univariate data for Lambert W x F distributions.


In particular Lambert W x Normal distribution with ability to gaussianize data.
"""
import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd
import sklearn
import torch
import torchlambertw.distributions as lwd
from torch.optim import lr_scheduler

from . import base, igmm
from .preprocessing import transforms
from .utils import distributions as ud

_EPS = 1e-4
_LOGIT_PREFIX = "logit_"

try:
    import torchlambertw.distributions
except ImportError:
    warnings.warn("The 'torchlambertw' module could not be imported.")
else:
    pass


def _dict_torch_to_series(dict_torch: Dict[str, torch.Tensor]) -> pd.Series:
    return pd.Series({k: float(v.detach().numpy()) for k, v in dict_torch.items()})


class MLE(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Computes the MLE for univariate Lambert W x F distributions."""

    def __init__(
        self,
        distribution_name: str = "Normal",
        distribution_constructor: Optional[torch.distributions.Distribution] = None,
        max_iter: int = 100,
        lambertw_type: str = "h",
        use_mean_variance: bool = True,
        lr: float = 0.01,
        tol: float = 1e-6,
        verbose: int = 0,
    ):
        """Initializes the class."""
        self.distribution_name = distribution_name
        self.distribution_constructor = (
            distribution_constructor
            or lwd.utils.get_distribution_constructor(self.distribution_name)
        )
        self.lambertw_type = base.LambertWType(lambertw_type)
        self.max_iter = max_iter
        self.lr = lr
        self.verbose = verbose
        self.use_mean_variance = use_mean_variance
        # Trained parameters
        self.tol = tol
        self.theta = None
        self.theta_init = {}
        self.theta_optim = {}
        self.theta_trace = []
        self.losses = []
        self.param_activation_fns = ud.get_params_activations(self.distribution_name)
        self.igmm = None

    def _initialize_params(self, data: np.ndarray):
        """Initialize parameter estimates."""

        if self.lambertw_type == base.LambertWType.H:
            self.igmm = igmm.IGMM(
                lambertw_type=self.lambertw_type,
                location_family=lwd.utils.is_location_family(self.distribution_name),
            )
            self.igmm.fit(data)
            x_init = self.igmm.transform(data)

            lambertw_params_init = self.igmm.tau.lambertw_params
        else:
            if lwd.utils.is_location_family(self.distribution_name):
                # Default to Normal distriubtion for location family.
                params_data = ud.estimate_params(data, "Normal")
                loc_init = params_data["loc"]
                scale_init = params_data["scale"]
            else:
                # If it's not location family, default to Exponential.
                params_data = ud.estimate_params(data, "Exponential")
                loc_init = 0.0
                scale_init = 1.0 / params_data["rate"]

            z_init = (data - loc_init) / scale_init
            if lwd.utils.is_location_family(self.distribution_name):
                gamma_init = igmm.gamma_taylor(z_init)
            else:
                gamma_init = 0.01

            lambertw_params_init = base.LambertWParams(gamma=gamma_init)
            u_init = transforms.W_gamma(z_init, gamma=lambertw_params_init.gamma, k=0)

            x_init = u_init * scale_init + loc_init

        beta_init = ud.estimate_params(x_init, distribution_name=self.distribution_name)
        theta_init = base.Theta(
            beta=beta_init,
            lambertw_params=lambertw_params_init,
            distribution_name=self.distribution_name,
        )
        self.theta_init = theta_init
        if self.lambertw_type == base.LambertWType.H:
            self.theta_optim["log_delta"] = torch.tensor(
                np.log(theta_init.lambertw_params.delta + _EPS), requires_grad=True
            )
        elif self.lambertw_type == base.LambertWType.S:
            self.theta_optim["gamma"] = torch.tensor(
                theta_init.lambertw_params.gamma, requires_grad=True
            )

        for param_name, val in self.param_activation_fns.items():
            param_name_logit = _LOGIT_PREFIX + param_name
            self.theta_optim[param_name_logit] = torch.tensor(
                val[1](theta_init.beta[param_name]), requires_grad=True
            )

    def fit(self, data: np.ndarray):
        """Trains the MLE of a Lambert W distribution based on torch likelihood optimization."""
        self._initialize_params(data)
        init_params = list(self.theta_optim.values())
        optimizer = torch.optim.NAdam(init_params, lr=self.lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        tr_data = torch.tensor(data)
        for epoch in range(self.max_iter):
            optimizer.zero_grad()  # Clear gradients
            beta_args = {}
            for k, v in self.param_activation_fns.items():
                beta_args[k] = v[0](self.theta_optim[_LOGIT_PREFIX + k])

            if self.lambertw_type == base.LambertWType.H:
                distr = torchlambertw.distributions.TailLambertWDistribution(
                    base_distribution=self.distribution_constructor,
                    base_dist_args=beta_args,
                    tailweight=torch.exp(self.theta_optim["log_delta"]),
                    use_mean_variance=self.use_mean_variance,
                )

            elif self.lambertw_type == base.LambertWType.S:
                distr = torchlambertw.distributions.SkewLambertWDistribution(
                    base_distribution=self.distribution_constructor,
                    base_dist_args=beta_args,
                    skewweight=self.theta_optim["gamma"],
                    use_mean_variance=self.use_mean_variance,
                )

            loglik = distr.log_prob(tr_data).sum()

            loss = -loglik  # Negative log likelihood as the loss
            loss.backward()  # Backpropagate gradients
            optimizer.step()  # Update parameters
            scheduler.step()
            self.losses.append(loss.item())
            self.theta_trace.append(_dict_torch_to_series(self.theta_optim))
            if epoch > 2:
                if np.abs(self.losses[-1] - self.losses[-2]) < self.tol:
                    break
            if self.verbose:
                if (epoch + 1) % self.verbose == 0:
                    print(f"Epoch [{epoch+1}/{self.max_iter}], Loss: {loss.item()}")

        self.theta_trace = pd.concat(self.theta_trace, axis=1).T
        beta_estimate = {}
        for k, v in self.param_activation_fns.items():
            logit_name = _LOGIT_PREFIX + k
            beta_estimate[k] = float(
                v[0](self.theta_optim[logit_name]).detach().numpy()
            )
            self.theta_trace[k] = self.theta_trace[logit_name].apply(
                lambda x: v[0](torch.tensor(x)).numpy()
            )
            self.theta_trace = self.theta_trace.drop(logit_name, axis=1)

        if self.lambertw_type == base.LambertWType.H:
            self.theta_trace["delta"] = np.exp(self.theta_trace["log_delta"])
            self.theta_trace = self.theta_trace.drop("log_delta", axis=1)
            lambertw_params = base.LambertWParams(
                delta=np.exp(self.theta_optim["log_delta"].detach().numpy())
            )
        else:
            lambertw_params = base.LambertWParams(
                gamma=self.theta_optim["gamma"].detach().numpy()
            )
        self.theta = base.Theta(
            lambertw_params=lambertw_params,
            beta=beta_estimate,
            distribution_name=self.distribution_name,
        )

        if self.verbose:
            print("MLE\n---\n", self.theta)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transforms data y to the data based on IGMM estimate tau."""
        return transforms.normalize_by_tau(data, tau=self.theta.tau)
