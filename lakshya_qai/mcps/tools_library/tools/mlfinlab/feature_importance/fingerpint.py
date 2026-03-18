"""
Implementation of an algorithm described in Yimou Li, David Turkington, Alireza Yazdani
'Beyond the Black Box: An Intuitive Approach to Investment Prediction with Machine Learning'
(https://jfds.pm-research.com/content/early/2019/12/11/jfds.2019.1.023)
"""

from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# pylint: disable=invalid-name
# pylint: disable=too-many-locals

class AbstractModelFingerprint(ABC):
    """
    Model fingerprint constructor.

    This is an abstract base class for the RegressionModelFingerprint and ClassificationModelFingerprint classes.
    """

    def __init__(self):
        """
        Model fingerprint constructor.
        """
        self.linear_effect = None
        self.non_linear_effect = None
        self.pairwise_effect = None
        self.feature_values = None

    def fit(self, model: object, X: pd.DataFrame, num_values: int = 50, pairwise_combinations: list = None) -> None:
        """
        Get linear, non-linear and pairwise effects estimation.

        :param model: (object) Trained model.
        :param X: (pd.DataFrame) Dataframe of features.
        :param num_values: (int) Number of values used to estimate feature effect.
        :param pairwise_combinations: (list) Tuples (feature_i, feature_j) to test pairwise effect.
        """

        self._get_feature_values(X, num_values)
        self._get_individual_partial_dependence(model, X)
        self.linear_effect = self._get_linear_effect(X)
        self.non_linear_effect = self._get_non_linear_effect(X)
        if pairwise_combinations:
            self.pairwise_effect = self._get_pairwise_effect(pairwise_combinations, model, X, num_values)

    def get_effects(self) -> Tuple:
        """
        Return computed linear, non-linear and pairwise effects. The model should be fit() before using this method.

        :return: (tuple) Linear, non-linear and pairwise effects, of type dictionary (raw values and normalised).
        """

        return self.linear_effect, self.non_linear_effect, self.pairwise_effect

    def plot_effects(self) -> plt.figure:
        """
        Plot each effect (normalized) on a bar plot (linear, non-linear). Also plots pairwise effects if calculated.

        :return: (plt.figure) Plot figure.
        """

        fig, ax = plt.subplots()
        if self.linear_effect is not None:
            pd.Series(self.linear_effect).sort_values().plot(kind="barh", ax=ax, label="linear")
        if self.non_linear_effect is not None:
            pd.Series(self.non_linear_effect).sort_values().plot(kind="barh", ax=ax, label="non_linear", alpha=0.6)
        ax.legend()
        return fig

    def _get_feature_values(self, X: pd.DataFrame, num_values: int) -> None:
        """
        Step 1 of the algorithm which generates possible feature values used in analysis.

        :param X: (pd.DataFrame) Dataframe of features.
        :param num_values: (int) Number of values used to estimate feature effect.
        """

        self.feature_values = {c: np.linspace(X[c].min(), X[c].max(), num_values) for c in X.columns}

    def _get_individual_partial_dependence(self, model: object, X: pd.DataFrame) -> None:
        """
        Get individual partial dependence function values for each column.

        :param model: (object) Trained model.
        :param X: (pd.DataFrame) Dataframe of features.
        """

        self.partial_dep = {}
        for c in X.columns:
            vals = self.feature_values[c]
            X_tmp = X.copy()
            preds = []
            for v in vals:
                X_tmp[c] = v
                preds.append(self._get_model_predictions(model, X_tmp).mean())
            self.partial_dep[c] = np.array(preds)

    def _get_linear_effect(self, X: pd.DataFrame) -> dict:
        """
        Get linear effect estimates as the mean absolute deviation of the linear predictions around their average value.

        :param X: (pd.DataFrame) Dataframe of features.
        :return: (dict) Linear effect estimates for each feature column.
        """

        effects = {}
        for c in X.columns:
            y = self.partial_dep[c]
            x = self.feature_values[c].reshape(-1, 1)
            lr = LinearRegression().fit(x, y)
            effects[c] = np.mean(np.abs(lr.predict(x) - y.mean()))
        return self._normalize(effects)

    def _get_non_linear_effect(self, X: pd.DataFrame) -> dict:
        """
        Get non-linear effect estimates as as the mean absolute deviation of the total marginal (single variable)
        effect around its corresponding linear effect.

        :param X: (pd.DataFrame) Dataframe of features.
        :return: (dict) Non-linear effect estimates for each feature column.
        """

        effects = {}
        for c in X.columns:
            y = self.partial_dep[c]
            x = self.feature_values[c].reshape(-1, 1)
            lr = LinearRegression().fit(x, y)
            residual = y - lr.predict(x)
            effects[c] = np.mean(np.abs(residual))
        return self._normalize(effects)

    def _get_pairwise_effect(self, pairwise_combinations: list, model: object, X: pd.DataFrame, num_values) -> dict:
        """
        Get pairwise effect estimates as the de-meaned joint partial prediction of the two variables minus the de-meaned
        partial predictions of each variable independently.

        :param pairwise_combinations: (list) Tuples (feature_i, feature_j) to test pairwise effect.
        :param model: (object) Trained model.
        :param X: (pd.DataFrame) Dataframe of features.
        :param num_values: (int) Number of values used to estimate feature effect.
        :return: (dict) Raw and normalised pairwise effects.
        """

        effects = {}
        for a, b in pairwise_combinations:
            vals_a = self.feature_values[a]
            vals_b = self.feature_values[b]
            X_tmp = X.copy()
            joint = []
            for va in vals_a:
                for vb in vals_b:
                    X_tmp[a] = va
                    X_tmp[b] = vb
                    joint.append(self._get_model_predictions(model, X_tmp).mean())
            effects[(a, b)] = np.mean(np.abs(joint))
        return self._normalize(effects)

    @abstractmethod
    def _get_model_predictions(self, model: object, X_: pd.DataFrame):
        """
        Get model predictions based on problem type (predict for regression, predict_proba for classification).

        :param model: (object) Trained model.
        :param X_: (np.array) Feature set.
        :return: (np.array) Predictions.
        """

        raise NotImplementedError("Subclasses must implement _get_model_predictions.")

    @staticmethod
    def _normalize(effect: dict) -> dict:
        """
        Normalize effect values (sum equals 1).

        :param effect: (dict) Effect values.
        :return: (dict) Normalized effect values.
        """

        total = sum(effect.values()) if effect else 1
        return {k: v / total for k, v in effect.items()}


class RegressionModelFingerprint(AbstractModelFingerprint):
    """
    Regression Fingerprint class used for regression type of models.
    """

    def __init__(self):
        """
        Regression model fingerprint constructor.
        """

        super().__init__()

    def _get_model_predictions(self, model, X_):
        """
        Abstract method _get_model_predictions implementation.

        :param model: (object) Trained model.
        :param X_: (np.array) Feature set.
        :return: (np.array) Predictions.
        """

        return model.predict(X_)


class ClassificationModelFingerprint(AbstractModelFingerprint):
    """
    Classification Fingerprint class used for classification type of models.
    """

    def __init__(self):
        """
        Classification model fingerprint constructor.
        """

        super().__init__()

    def _get_model_predictions(self, model, X_):
        """
        Abstract method _get_model_predictions implementation.

        :param model: (object) Trained model.
        :param X_: (np.array) Feature set.
        :return: (np.array) Predictions.
        """

        return model.predict_proba(X_)[:, 1]
