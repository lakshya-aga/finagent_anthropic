"""
The module implementing various functions loading tick, dollar, stock data sets which can be used as
sandbox data.
"""

import os

import numpy as np
import pandas as pd

from mlfinlab.labeling.labeling import get_events, add_vertical_barrier, get_bins
from mlfinlab.util.volatility import get_daily_vol
from mlfinlab.filters.filters import cusum_filter


def load_stock_prices() -> pd.DataFrame:
    """
    Loads stock prices data sets consisting of
    EEM, EWG, TIP, EWJ, EFA, IEF, EWQ, EWU, XLB, XLE, XLF, LQD, XLK, XLU, EPP, FXI, VGK, VPL, SPY, TLT, BND, CSJ,
    DIA starting from 2008 till 2016.

    :return: (pd.DataFrame) The stock_prices data frame.
    """

    data_path = os.path.join(os.path.dirname(__file__), "data", "stock_prices.csv")
    return pd.read_csv(data_path, index_col=0, parse_dates=True)


def load_tick_sample() -> pd.DataFrame:
    """
    Loads E-Mini S&P 500 futures tick data sample.

    :return: (pd.DataFrame) Frame with tick data sample.
    """

    data_path = os.path.join(os.path.dirname(__file__), "data", "tick_sample.csv")
    return pd.read_csv(data_path, index_col=0, parse_dates=True)


def load_dollar_bar_sample() -> pd.DataFrame:
    """
    Loads E-Mini S&P 500 futures dollar bars data sample.

    :return: (pd.DataFrame) Frame with dollar bar data sample.
    """

    data_path = os.path.join(os.path.dirname(__file__), "data", "dollar_bars.csv")
    return pd.read_csv(data_path, index_col=0, parse_dates=True)


def generate_multi_asset_data_set(start_date: pd.Timestamp = pd.Timestamp(2008, 1, 1),
                                  end_date: pd.Timestamp = pd.Timestamp(2020, 1, 1)) -> tuple:
    # pylint: disable=invalid-name
    """
    Generates multi-asset dataset from stock prices labelled by triple-barrier method.

    :param start_date: (pd.Timestamp) Dataset start date.
    :param end_date: (pd.Timestamp) Dataset end date.
    :return: (tuple) Tuple of dictionaries (asset: data) for X, y, cont contract used to label the dataset.
    """

    prices = load_stock_prices().loc[start_date:end_date]
    close = prices['close'] if isinstance(prices.columns, pd.MultiIndex) else prices
    daily_vol = get_daily_vol(close)
    t_events = cusum_filter(close, daily_vol)
    t1 = add_vertical_barrier(t_events, close, num_days=1)
    events = get_events(close, t_events, pt_sl=[1, 1], target=daily_vol, min_ret=0, num_threads=1,
                        vertical_barrier_times=t1)
    bins = get_bins(events, close)
    X = {col: close[[col]] for col in close.columns} if isinstance(close, pd.DataFrame) else {"asset": close}
    y = {col: bins['bin'] for col in X} if isinstance(close, pd.DataFrame) else {"asset": bins['bin']}
    return X, y, events
