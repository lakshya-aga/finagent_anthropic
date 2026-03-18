"""
Various volatility estimators
"""
import pandas as pd
import numpy as np


# pylint: disable=redefined-builtin

def get_daily_vol(close, lookback=100):
    """
    Advances in Financial Machine Learning, Snippet 3.1, page 44.

    Daily Volatility Estimates

    Computes the daily volatility at intraday estimation points.

    In practice we want to set profit taking and stop-loss limits that are a function of the risks involved
    in a bet. Otherwise, sometimes we will be aiming too high (tao ≫ sigma_t_i,0), and sometimes too low
    (tao ≪ sigma_t_i,0 ), considering the prevailing volatility. Snippet 3.1 computes the daily volatility
    at intraday estimation points, applying a span of lookback days to an exponentially weighted moving
    standard deviation.

    See the pandas documentation for details on the pandas.Series.ewm function.
    Note: This function is used to compute dynamic thresholds for profit taking and stop loss limits.

    :param close: (pd.Series) Closing prices
    :param lookback: (int) Lookback period to compute volatility
    :return: (pd.Series) Daily volatility value
    """
    close = close.dropna()
    if close.empty:
        return close
    idx = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    idx = idx[idx > 0]
    prev_idx = pd.Series(close.index[idx - 1], index=close.index[close.shape[0] - idx.shape[0]:])
    daily_ret = close.loc[prev_idx.index] / close.loc[prev_idx.values].values - 1.0
    return daily_ret.ewm(span=lookback).std()


def get_parksinson_vol(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """
    Parkinson volatility estimator

    :param high: (pd.Series): High prices
    :param low: (pd.Series): Low prices
    :param window: (int): Window used for estimation
    :return: (pd.Series): Parkinson volatility
    """
    rs = np.log(high / low) ** 2
    return np.sqrt(rs.rolling(window).mean() / (4.0 * np.log(2.0)))


def get_garman_class_vol(open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
                         window: int = 20) -> pd.Series:
    """
    Garman-Class volatility estimator

    :param open: (pd.Series): Open prices
    :param high: (pd.Series): High prices
    :param low: (pd.Series): Low prices
    :param close: (pd.Series): Close prices
    :param window: (int): Window used for estimation
    :return: (pd.Series): Garman-Class volatility
    """
    log_hl = np.log(high / low)
    log_co = np.log(close / open)
    rs = 0.5 * log_hl ** 2 - (2.0 * np.log(2.0) - 1.0) * log_co ** 2
    return np.sqrt(rs.rolling(window).mean())


def get_yang_zhang_vol(open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
                       window: int = 20) -> pd.Series:
    """

    Yang-Zhang volatility estimator

    :param open: (pd.Series): Open prices
    :param high: (pd.Series): High prices
    :param low: (pd.Series): Low prices
    :param close: (pd.Series): Close prices
    :param window: (int): Window used for estimation
    :return: (pd.Series): Yang-Zhang volatility
    """
    log_ho = np.log(high / open)
    log_lo = np.log(low / open)
    log_co = np.log(close / open)
    log_oc = np.log(open / close.shift(1))
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    k = 0.34 / (1.34 + (window + 1.0) / (window - 1.0))
    sigma_o = log_oc.rolling(window).var()
    sigma_c = log_co.rolling(window).var()
    sigma_rs = rs.rolling(window).mean()
    return np.sqrt(sigma_o + k * sigma_c + (1.0 - k) * sigma_rs)
