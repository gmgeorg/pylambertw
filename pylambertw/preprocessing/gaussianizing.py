"""Module for gaussianizing data using Lambert W x F transforms."""
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
import sklearn
import tqdm

from pylambertw.preprocessing import transforms

from .. import base, igmm, mle


class Gaussianizer(sklearn.base.TransformerMixin):
    """Module for column-wise Gaussianization using Lambert W x Normal transforms."""

    def __init__(
        self,
        lambertw_type: Union[str, base.LambertWType],
        method: str = "igmm",
        method_kwargs: Optional[dict] = None,
    ):
        """Initializes the class."""
        if isinstance(lambertw_type, str):
            lambertw_type = base.LambertWType(lambertw_type)
        self.lambertw_type = lambertw_type
        self.method = method

        if self.method not in ["mle", "igmm"]:
            raise NotImplementedError(f"method={method} is not implemented.")

        self.method_kwargs = method_kwargs or {}
        self.estimators: List[Any] = []
        self.columns: List[str] = []

    def fit(self, data: np.ndarray):
        """Trains a Gaussianizer for every column of 'data'."""

        self.columns = None
        if isinstance(data, pd.DataFrame):
            self.columns = data.columns
            data = data.values

        if len(data.shape) == 1:
            data = data[:, np.newaxis]

        n_cols = data.shape[1]
        for i in tqdm.tqdm(range(n_cols), total=n_cols):
            if self.method == "mle":
                estimate_clf = mle.MLE(distribution_name="Normal", **self.method_kwargs)
            elif self.method == "igmm":
                estimate_clf = igmm.IGMM(lambertw_type=self.lambertw_type)
            else:
                raise NotImplementedError(f"method={self.method} is not implemented")

            estimate_clf.fit(data[:, i])
            self.estimators.append(estimate_clf)
            del estimate_clf
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transforms data to a gaussianized version."""
        is_df = False
        index = None
        if isinstance(data, pd.DataFrame):
            is_df = True
            index = data.index
            data = data[self.columns]
            data = data.values

        is_univariate_input = False
        if len(data.shape) == 1:
            is_univariate_input = True
            data = data[:, np.newaxis]

        result = np.zeros_like(data)
        n_cols = len(self.estimators)
        for i in range(n_cols):
            if self.method == "igmm":
                tau_tmp = self.estimators[i].tau
            elif self.method == "mle":
                tau_tmp = base.Tau(
                    loc=self.estimators[i].theta.beta["loc"],
                    scale=self.estimators[i].theta.beta["scale"],
                    lambertw_params=self.estimators[i].theta.lambertw_params,
                )
            else:
                raise NotImplementedError(f"method={self.method} is not implemented.")
            result[:, i] = transforms.normalize_by_tau(y=data[:, i], tau=tau_tmp)

        if is_df:
            result = pd.DataFrame(result, index=index, columns=self.columns)
        else:
            if is_univariate_input:
                result = result.ravel()
        return result

    @property
    def params(self) -> pd.DataFrame:
        if self.method == "mle":
            arr = np.vstack([v.theta.to_numpy() for v in self.estimators])
        if self.method == "igmm":
            arr = np.vstack([v.tau.to_numpy() for v in self.estimators])

        params_df = pd.DataFrame(
            arr, index=self.columns, columns=["loc", "scale", "gamma", "delta", "alpha"]
        )
        # Check for constant columns
        constant_columns = [
            col for col in params_df.columns if params_df[col].nunique() == 1
        ]

        # Drop constant columns
        params_df = params_df.drop(columns=constant_columns)

        return params_df
