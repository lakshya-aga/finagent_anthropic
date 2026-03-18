"""
Explosiveness tests: SADF
"""

from typing import Union, Tuple
import pandas as pd
import numpy as np
from mlfinlab.util.multiprocess import mp_pandas_obj


# pylint: disable=invalid-name

def _get_sadf_at_t(X: pd.DataFrame, y: pd.DataFrame, min_length: int, model: str, phi: float) -> float:
    """
    Advances in Financial Machine Learning, Snippet 17.2, page 258.

    SADF's Inner Loop (get SADF value at t)

    :param X: (pd.DataFrame) Lagged values, constants, trend coefficients
    :param y: (pd.DataFrame) Y values (either y or y.diff())
    :param min_length: (int) Minimum number of samples needed for estimation
    :param model: (str) Either 'linear', 'quadratic', 'sm_poly_1', 'sm_poly_2', 'sm_exp', 'sm_power'
    :param phi: (float) Coefficient to penalize large sample lengths when computing SMT, in [0, 1]
    :return: (float) SADF statistics for y.index[-1]
    """
    max_adf = -np.inf
    t = X.index[-1]
    x_all = X.loc[:t].values
    y_all = y.loc[:t].values
    for start in range(0, len(y_all) - min_length + 1):
        x = x_all[start:]
        yv = y_all[start:]
        if len(yv) < min_length:
            continue
        betas, var = get_betas(pd.DataFrame(x), pd.DataFrame(yv))
        if var[0] == 0:
            continue
        adf = betas[0] / np.sqrt(var[0])
        adf = adf / (len(yv) ** phi) if phi > 0 else adf
        if adf > max_adf:
            max_adf = adf
    return max_adf


def _get_y_x(series: pd.Series, model: str, lags: Union[int, list],
             add_const: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Advances in Financial Machine Learning, Snippet 17.2, page 258-259.

    Preparing The Datasets

    :param series: (pd.Series) Series to prepare for test statistics generation (for example log prices)
    :param model: (str) Either 'linear', 'quadratic', 'sm_poly_1', 'sm_poly_2', 'sm_exp', 'sm_power'
    :param lags: (int or list) Either number of lags to use or array of specified lags
    :param add_const: (bool) Flag to add constant
    :return: (pd.DataFrame, pd.DataFrame) Prepared y and X for SADF generation
    """
    series = series.dropna()
    y = series.diff().dropna()
    x = series.shift(1).loc[y.index].to_frame("y_lag1")
    if isinstance(lags, int) and lags > 0:
        lagged = _lag_df(y.to_frame("dy"), lags).dropna()
        x = x.loc[lagged.index].join(lagged)
        y = y.loc[lagged.index]
    elif isinstance(lags, list) and len(lags) > 0:
        lagged = _lag_df(y.to_frame("dy"), lags).dropna()
        x = x.loc[lagged.index].join(lagged)
        y = y.loc[lagged.index]
    if model in ("linear", "sm_poly_1", "sm_exp", "sm_power"):
        trend = np.arange(len(x)) + 1
        x["trend"] = trend
    if model in ("quadratic", "sm_poly_2"):
        trend = np.arange(len(x)) + 1
        x["trend"] = trend
        x["trend2"] = trend ** 2
    if add_const:
        x["const"] = 1.0
    return y.to_frame("y"), x


def _lag_df(df: pd.DataFrame, lags: Union[int, list]) -> pd.DataFrame:
    """
    Advances in Financial Machine Learning, Snipet 17.3, page 259.

    Apply Lags to DataFrame

    :param df: (int or list) Either number of lags to use or array of specified lags
    :param lags: (int or list) Lag(s) to use
    :return: (pd.DataFrame) Dataframe with lags
    """
    if isinstance(lags, int):
        lags = list(range(1, lags + 1))
    frames = []
    for lag in lags:
        shifted = df.shift(lag)
        shifted.columns = [f"{c}_lag{lag}" for c in df.columns]
        frames.append(shifted)
    return pd.concat(frames, axis=1)


def get_betas(X: pd.DataFrame, y: pd.DataFrame) -> Tuple[np.array, np.array]:
    """
    Advances in Financial Machine Learning, Snippet 17.4, page 259.

    Fitting The ADF Specification (get beta estimate and estimate variance)

    :param X: (pd.DataFrame) Features(factors)
    :param y: (pd.DataFrame) Outcomes
    :return: (np.array, np.array) Betas and variances of estimates
    """
    x = X.values
    yv = y.values
    betas, _, _, _ = np.linalg.lstsq(x, yv, rcond=None)
    resid = yv - x.dot(betas)
    n, k = x.shape
    s2 = (resid.T @ resid) / max(n - k, 1)
    cov = s2[0, 0] * np.linalg.pinv(x.T @ x)
    return betas.flatten(), np.diag(cov)


def _sadf_outer_loop(X: pd.DataFrame, y: pd.DataFrame, min_length: int, model: str, phi: float,
                     molecule: list) -> pd.Series:
    """
    This function gets SADF for t times from molecule

    :param X: (pd.DataFrame) Features(factors)
    :param y: (pd.DataFrame) Outcomes
    :param min_length: (int) Minimum number of observations
    :param model: (str) Either 'linear', 'quadratic', 'sm_poly_1', 'sm_poly_2', 'sm_exp', 'sm_power'
    :param phi: (float) Coefficient to penalize large sample lengths when computing SMT, in [0, 1]
    :param molecule: (list) Indices to get SADF
    :return: (pd.Series) SADF statistics
    """
    stats = {}
    for t in molecule:
        stats[t] = _get_sadf_at_t(X.loc[:t], y.loc[:t], min_length, model, phi)
    return pd.Series(stats)

def get_sadf(series: pd.Series, model: str, lags: Union[int, list], min_length: int, add_const: bool = False,
             phi: float = 0, num_threads: int = 8, verbose: bool = True) -> pd.Series:
    """
    Advances in Financial Machine Learning, p. 258-259.

    Multithread implementation of SADF

    SADF fits the ADF regression at each end point t with backwards expanding start points. For the estimation
    of SADF(t), the right side of the window is fixed at t. SADF recursively expands the beginning of the sample
    up to t - min_length, and returns the sup of this set.

    When doing with sub- or super-martingale test, the variance of beta of a weak long-run bubble may be smaller than
    one of a strong short-run bubble, hence biasing the method towards long-run bubbles. To correct for this bias,
    ADF statistic in samples with large lengths can be penalized with the coefficient phi in [0, 1] such that:

    ADF_penalized = ADF / (sample_length ^ phi)

    :param series: (pd.Series) Series for which SADF statistics are generated
    :param model: (str) Either 'linear', 'quadratic', 'sm_poly_1', 'sm_poly_2', 'sm_exp', 'sm_power'
    :param lags: (int or list) Either number of lags to use or array of specified lags
    :param min_length: (int) Minimum number of observations needed for estimation
    :param add_const: (bool) Flag to add constant
    :param phi: (float) Coefficient to penalize large sample lengths when computing SMT, in [0, 1]
    :param num_threads: (int) Number of cores to use
    :param verbose: (bool) Flag to report progress on asynch jobs
    :return: (pd.Series) SADF statistics
    """
    y, x = _get_y_x(series, model, lags, add_const)
    out = mp_pandas_obj(_sadf_outer_loop, ('molecule', x.index[min_length - 1:]),
                        num_threads, X=x, y=y, min_length=min_length, model=model, phi=phi, verbose=verbose)
    return out.sort_index()
