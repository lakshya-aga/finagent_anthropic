"""
This module contains class for ETF trick generation and futures roll function, described in Marcos Lopez de Prado's
book 'Advances in Financial Machine Learning' ETF trick class can generate ETF trick series either from .csv files
or from in memory pandas DataFrames
"""

import warnings
import pandas as pd
import numpy as np


class ETFTrick:
    """
    Contains logic of vectorised ETF trick implementation. Can used for both memory data frames (pd.DataFrame) and
    csv files. All data frames, files should be processed in a specific format, described in examples
    """

    def __init__(self, open_df, close_df, alloc_df, costs_df, rates_df=None, index_col=0):
        """
        Constructor

        Creates class object, for csv based files reads the first data chunk.

        :param open_df: (pd.DataFrame or string): open prices data frame or path to csv file,
         corresponds to o(t) from the book
        :param close_df: (pd.DataFrame or string): close prices data frame or path to csv file, corresponds to p(t)
        :param alloc_df: (pd.DataFrame or string): asset allocations data frame or path to csv file (in # of contracts),
         corresponds to w(t)
        :param costs_df: (pd.DataFrame or string): rebalance, carry and dividend costs of holding/rebalancing the
         position, corresponds to d(t)
        :param rates_df: (pd.DataFrame or string): dollar value of one point move of contract includes exchange rate,
         futures contracts multiplies). Corresponds to phi(t)
         For example, 1$ in VIX index, equals 1000$ in VIX futures contract value.
         If None then trivial (all values equal 1.0) is generated
        :param index_col: (int): positional index of index column. Used for to determine index column in csv files
        """

        self.open_df = open_df
        self.close_df = close_df
        self.alloc_df = alloc_df
        self.costs_df = costs_df
        self.rates_df = rates_df
        self.index_col = index_col
        self.cache = None

    def _append_previous_rows(self, cache):
        """
        Uses latest two rows from cache to append into current data. Used for csv based ETF trick, when the next
        batch is loaded and we need to recalculate K value which corresponds to previous batch.

        :param cache: (dict): dictionary which pd.DataFrames with latest 2 rows of open, close, alloc, costs, rates
        :return: (pd.DataFrame): data frame with close price differences (updates self.data_dict)
        """

        if cache is None:
            return
        for key, df in cache.items():
            if df is None:
                continue
            self.data_dict[key] = pd.concat([df, self.data_dict[key]])

    def generate_trick_components(self, cache=None):
        """
        Calculates all etf trick operations which can be vectorised. Outputs multilevel pandas data frame.

        Generated components:
        'w': alloc_df
        'h_t': h_t/K value from ETF trick algorithm from the book. Which K to use is based on previous values and
            cannot be vectorised.
        'close_open': close_df - open_df
        'price_diff': close price differences
        'costs': costs_df
        'rate': rates_df

        :param cache: (dict of pd.DataFrames): dictionary which contains latest 2 rows of open, close, rates, alloc,
            costs, rates data
        :return: (pd.DataFrame): pandas data frame with columns in a format: component_1/asset_name_1,
            component_1/asset_name_2, ..., component_6/asset_name_n
        """

        self.data_dict = {
            "open": self.open_df,
            "close": self.close_df,
            "alloc": self.alloc_df,
            "costs": self.costs_df,
            "rates": self.rates_df if self.rates_df is not None else pd.DataFrame(
                1.0, index=self.close_df.index, columns=self.close_df.columns)
        }
        self._append_previous_rows(cache)
        comps = pd.concat({
            "w": self.data_dict["alloc"],
            "close_open": self.data_dict["close"] - self.data_dict["open"],
            "price_diff": self.data_dict["close"].diff(),
            "costs": self.data_dict["costs"],
            "rate": self.data_dict["rates"]
        }, axis=1)
        return comps

    def _update_cache(self):
        """
        Updates cache (two previous rows) when new data batch is read into the memory. Cache is used to
        recalculate ETF trick value which corresponds to previous batch last row. That is why we need 2 previous rows
        for close price difference calculation

        :return: (dict): dictionary with open, close, alloc, costs and rates last 2 rows
        """

        self.cache = {k: v.tail(2) for k, v in self.data_dict.items()}
        return self.cache

    def _chunk_loop(self, data_df):
        """
        Single ETF trick iteration for currently stored(with needed components) data set in memory (data_df).
        For in-memory data set would yield complete ETF trick series, for csv based
        would generate ETF trick series for current batch.

        :param data_df: The data set on which to apply the ETF trick.
        :return: (pd.Series): pandas Series with ETF trick values
        """

        w = data_df["w"]
        close_open = data_df["close_open"]
        costs = data_df["costs"]
        rate = data_df["rate"]
        pnl = (w * close_open * rate).sum(axis=1) - costs.sum(axis=1)
        etf = 1 + pnl.cumsum()
        return etf

    def _index_check(self):
        """
        Internal check for all price, rates and allocations data frames have the same index
        """

        idx = self.open_df.index
        assert self.close_df.index.equals(idx)
        assert self.alloc_df.index.equals(idx)
        assert self.costs_df.index.equals(idx)
        if self.rates_df is not None:
            assert self.rates_df.index.equals(idx)

    def _get_batch_from_csv(self, batch_size):
        """
        Reads the next batch of data sets from csv files and puts them in class variable data_dict

        :param batch_size: number of rows to read
        """

        self.open_df = pd.read_csv(self.open_df, index_col=self.index_col, parse_dates=True, nrows=batch_size)
        self.close_df = pd.read_csv(self.close_df, index_col=self.index_col, parse_dates=True, nrows=batch_size)
        self.alloc_df = pd.read_csv(self.alloc_df, index_col=self.index_col, parse_dates=True, nrows=batch_size)
        self.costs_df = pd.read_csv(self.costs_df, index_col=self.index_col, parse_dates=True, nrows=batch_size)
        if self.rates_df is not None and isinstance(self.rates_df, str):
            self.rates_df = pd.read_csv(self.rates_df, index_col=self.index_col, parse_dates=True, nrows=batch_size)

    def _rewind_etf_trick(self, alloc_df, etf_series):
        """
        ETF trick uses next open price information, when we process csv file in batches the last row in batch will have
        next open price value as nan, that is why when new batch comes, we need to rewind ETF trick values one step
        back, recalculate ETF trick value for the last row from previous batch using open price from latest batch
        received. This function rewinds values needed for ETF trick calculation recalculate

        :param alloc_df: (pd.DataFrame): data frame with allocations vectors
        :param etf_series (pd.Series): current computed ETF trick series
        """

        if len(etf_series) > 0:
            etf_series.iloc[-1] = etf_series.iloc[-2] if len(etf_series) > 1 else etf_series.iloc[-1]

    def _csv_file_etf_series(self, batch_size):
        """
        Csv based ETF trick series generation

        :param: batch_size: (int): Size of the batch that you would like to make use of
        :return: (pd.Series): pandas Series with ETF trick values starting from 1.0
        """

        self._get_batch_from_csv(batch_size)
        data_df = self.generate_trick_components(cache=self.cache)
        etf = self._chunk_loop(data_df)
        self._update_cache()
        return etf

    def _in_memory_etf_series(self):
        """
        In-memory based ETF trick series generation.

        :return: (pd.Series): pandas Series with ETF trick values starting from 1.0
        """

        self._index_check()
        data_df = self.generate_trick_components()
        return self._chunk_loop(data_df)

    def get_etf_series(self, batch_size=1e5):
        """
        External method which defines which etf trick method to use.

        :param: batch_size: Size of the batch that you would like to make use of
        :return: (pd.Series): pandas Series with ETF trick values starting from 1.0
        """

        if isinstance(self.open_df, str):
            return self._csv_file_etf_series(int(batch_size))
        return self._in_memory_etf_series()

    def reset(self):
        """
        Re-inits class object. This methods can be used to reset file iterators for multiple get_etf_trick() calls.
        """

        self.cache = None


def get_futures_roll_series(data_df, open_col, close_col, sec_col, current_sec_col, roll_backward=False,
                            method='absolute'):
    """
    Function for generating rolling futures series from data frame of multiple futures.

    :param data_df: (pd.DataFrame): pandas DataFrame containing price info, security name and current active futures
     column
    :param open_col: (string): open prices column name
    :param close_col: (string): close prices column name
    :param sec_col: (string): security name column name
    :param current_sec_col: (string): current active security column name. When value in this column changes it means
     rolling
    :param roll_backward: (boolean): True for subtracting final gap value from all values
    :param method: (string): what returns user wants to preserve, 'absolute' or 'relative'
    :return (pd.Series): futures roll close price series
    """

    df = data_df.copy()
    out = []
    adj = 0.0
    prev_sec = df[current_sec_col].iloc[0]
    prev_close = df[close_col].iloc[0]
    for _, row in df.iterrows():
        if row[current_sec_col] != prev_sec:
            gap = row[open_col] - prev_close
            if method == 'absolute':
                adj -= gap if roll_backward else gap
            prev_sec = row[current_sec_col]
        out.append(row[close_col] + adj)
        prev_close = row[close_col]
    return pd.Series(out, index=df.index)
