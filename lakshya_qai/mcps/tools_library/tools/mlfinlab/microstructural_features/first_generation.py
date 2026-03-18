"""
First generation features (Roll Measure/Impact, Corwin-Schultz spread estimator)
"""

import numpy as np
import pandas as pd


def get_roll_measure(close_prices: pd.Series, window: int = 20) -> pd.Series:
    """
    Advances in Financial Machine Learning, page 282.

    Get Roll Measure

    Roll Measure gives the estimate of effective bid-ask spread
    without using quote-data.

    :param close_prices: (pd.Series) Close prices
    :param window: (int) Estimation window
    :return: (pd.Series) Roll measure
    """

    diff = close_prices.diff()
    cov = diff.rolling(window).cov(diff.shift(1))
    return 2 * np.sqrt(np.abs(cov))


def get_roll_impact(close_prices: pd.Series, dollar_volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Get Roll Impact.

    Derivate from Roll Measure which takes into account dollar volume traded.

    :param close_prices: (pd.Series) Close prices
    :param dollar_volume: (pd.Series) Dollar volume series
    :param window: (int) Estimation window
    :return: (pd.Series) Roll impact
    """

    roll = get_roll_measure(close_prices, window)
    return roll / dollar_volume.rolling(window).mean()


# Corwin-Schultz algorithm
def _get_beta(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """
    Advances in Financial Machine Learning, Snippet 19.1, page 285.

    Get beta estimate from Corwin-Schultz algorithm

    :param high: (pd.Series) High prices
    :param low: (pd.Series) Low prices
    :param window: (int) Estimation window
    :return: (pd.Series) Beta estimates
    """

    beta = (np.log(high / low) ** 2).rolling(window).sum()
    return beta


def _get_gamma(high: pd.Series, low: pd.Series) -> pd.Series:
    """
    Advances in Financial Machine Learning, Snippet 19.1, page 285.

    Get gamma estimate from Corwin-Schultz algorithm.

    :param high: (pd.Series) High prices
    :param low: (pd.Series) Low prices
    :return: (pd.Series) Gamma estimates
    """

    return (np.log(high / low) ** 2)


def _get_alpha(beta: pd.Series, gamma: pd.Series) -> pd.Series:
    """
    Advances in Financial Machine Learning, Snippet 19.1, page 285.

    Get alpha from Corwin-Schultz algorithm.

    :param beta: (pd.Series) Beta estimates
    :param gamma: (pd.Series) Gamma estimates
    :return: (pd.Series) Alphas
    """

    return (np.sqrt(2 * beta) - np.sqrt(gamma)) / (3 - 2 * np.sqrt(2))


def get_corwin_schultz_estimator(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """
    Advances in Financial Machine Learning, Snippet 19.1, page 285.

    Get Corwin-Schultz spread estimator using high-low prices

    :param high: (pd.Series) High prices
    :param low: (pd.Series) Low prices
    :param window: (int) Estimation window
    :return: (pd.Series) Corwin-Schultz spread estimators
    """
    # Note: S<0 iif alpha<0

    beta = _get_beta(high, low, window)
    gamma = _get_gamma(high, low).rolling(2).sum()
    alpha = _get_alpha(beta, gamma)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return spread


def get_bekker_parkinson_vol(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """
    Advances in Financial Machine Learning, Snippet 19.2, page 286.

    Get Bekker-Parkinson volatility from gamma and beta in Corwin-Schultz algorithm.

    :param high: (pd.Series) High prices
    :param low: (pd.Series) Low prices
    :param window: (int) Estimation window
    :return: (pd.Series) Bekker-Parkinson volatility estimates
    """
    # pylint: disable=invalid-name

    beta = _get_beta(high, low, window)
    gamma = _get_gamma(high, low).rolling(2).sum()
    return np.sqrt((beta - gamma) / (2 * np.log(2)))
