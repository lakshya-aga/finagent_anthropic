"""
Second generation models features: Kyle lambda, Amihud Lambda, Hasbrouck lambda (bar and trade based)
"""

from typing import List
import numpy as np
import pandas as pd

from mlfinlab.structural_breaks.sadf import get_betas

# pylint: disable=invalid-name
def get_bar_based_kyle_lambda(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Advances in Financial Machine Learning, p. 286-288.

    Get Kyle lambda from bars data

    :param close: (pd.Series) Close prices
    :param volume: (pd.Series) Bar volume
    :param window: (int) Rolling window used for estimation
    :return: (pd.Series) Kyle lambdas
    """

    ret = close.diff()
    res = ret.rolling(window).apply(lambda x: get_trades_based_kyle_lambda(x, volume.loc[x.index], np.sign(x))[0], raw=False)
    return res


def get_bar_based_amihud_lambda(close: pd.Series, dollar_volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Advances in Financial Machine Learning, p.288-289.

    Get Amihud lambda from bars data

    :param close: (pd.Series) Close prices
    :param dollar_volume: (pd.Series) Dollar volumes
    :param window: (int) rolling window used for estimation
    :return: (pd.Series) of Amihud lambda
    """

    ret = close.pct_change().abs()
    return (ret / dollar_volume).rolling(window).mean()

def get_bar_based_hasbrouck_lambda(close: pd.Series, dollar_volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Advances in Financial Machine Learning, p.289-290.

    Get Hasbrouck lambda from bars data

    :param close: (pd.Series) Close prices
    :param dollar_volume: (pd.Series) Dollar volumes
    :param window: (int) Rolling window used for estimation
    :return: (pd.Series) Hasbrouck lambda
    """

    ret = close.pct_change()
    res = ret.rolling(window).apply(lambda x: get_trades_based_hasbrouck_lambda(x, dollar_volume.loc[x.index], np.sign(x))[0], raw=False)
    return res


def get_trades_based_kyle_lambda(price_diff: list, volume: list, aggressor_flags: list) -> List[float]:
    """
    Advances in Financial Machine Learning, p.286-288.

    Get Kyle lambda from trades data

    :param price_diff: (list) Price diffs
    :param volume: (list) Trades sizes
    :param aggressor_flags: (list) Trade directions [-1, 1]  (tick rule or aggressor side can be used to define)
    :return: (list) Kyle lambda for a bar and t-value
    """

    y = np.array(price_diff)
    x = np.array(volume) * np.array(aggressor_flags)
    X = pd.DataFrame({"x": x})
    betas, var = get_betas(X, pd.DataFrame(y, columns=["y"]))
    t_val = betas[0] / np.sqrt(var[0]) if var[0] != 0 else 0.0
    return [betas[0], t_val]


def get_trades_based_amihud_lambda(log_ret: list, dollar_volume: list) -> List[float]:
    """
    Advances in Financial Machine Learning, p.288-289.

    Get Amihud lambda from trades data

    :param log_ret: (list) Log returns
    :param dollar_volume: (list) Dollar volumes (price * size)
    :return: (float) Amihud lambda for a bar
    """

    log_ret = np.abs(np.array(log_ret))
    dollar_volume = np.array(dollar_volume)
    return float(np.mean(log_ret / dollar_volume))


def get_trades_based_hasbrouck_lambda(log_ret: list, dollar_volume: list, aggressor_flags: list) -> List[float]:
    """
    Advances in Financial Machine Learning, p.289-290.

    Get Hasbrouck lambda from trades data

    :param log_ret: (list) Log returns
    :param dollar_volume: (list) Dollar volumes (price * size)
    :param aggressor_flags: (list) Trade directions [-1, 1]  (tick rule or aggressor side can be used to define)
    :return: (list) Hasbrouck lambda for a bar and t value
    """

    y = np.array(log_ret)
    x = np.array(dollar_volume) * np.array(aggressor_flags)
    X = pd.DataFrame({"x": x})
    betas, var = get_betas(X, pd.DataFrame(y, columns=["y"]))
    t_val = betas[0] / np.sqrt(var[0]) if var[0] != 0 else 0.0
    return [betas[0], t_val]
