"""
Inter-bar feature generator which uses trades data and bars index to calculate inter-bar features
"""

import pandas as pd
import numpy as np
from mlfinlab.microstructural_features.entropy import get_shannon_entropy, get_plug_in_entropy, get_lempel_ziv_entropy, \
    get_konto_entropy
from mlfinlab.microstructural_features.encoding import encode_array
from mlfinlab.microstructural_features.second_generation import get_trades_based_kyle_lambda, \
    get_trades_based_amihud_lambda, get_trades_based_hasbrouck_lambda
from mlfinlab.microstructural_features.misc import get_avg_tick_size, vwap
from mlfinlab.microstructural_features.encoding import encode_tick_rule_array
from mlfinlab.util.misc import crop_data_frame_in_batches


# pylint: disable=too-many-instance-attributes

class MicrostructuralFeaturesGenerator:
    """
    Class which is used to generate inter-bar features when bars are already compressed.

    :param trades_input: (str or pd.DataFrame) Path to the csv file or Pandas DataFrame containing raw tick data
                                               in the format[date_time, price, volume]
    :param tick_num_series: (pd.Series) Series of tick number where bar was formed.
    :param batch_size: (int) Number of rows to read in from the csv, per batch.
    :param volume_encoding: (dict) Dictionary of encoding scheme for trades size used to calculate entropy on encoded messages
    :param pct_encoding: (dict) Dictionary of encoding scheme for log returns used to calculate entropy on encoded messages

    """

    def __init__(self, trades_input: (str, pd.DataFrame), tick_num_series: pd.Series, batch_size: int = 2e7,
                 volume_encoding: dict = None, pct_encoding: dict = None):
        """
        Constructor

        :param trades_input: (str or pd.DataFrame) Path to the csv file or Pandas DataFrame containing raw tick data
                                                   in the format[date_time, price, volume]
        :param tick_num_series: (pd.Series) Series of tick number where bar was formed.
        :param batch_size: (int) Number of rows to read in from the csv, per batch.
        :param volume_encoding: (dict) Dictionary of encoding scheme for trades size used to calculate entropy on encoded messages
        :param pct_encoding: (dict) Dictionary of encoding scheme for log returns used to calculate entropy on encoded messages
        """


        self.trades_input = trades_input
        self.tick_num_series = tick_num_series
        self.batch_size = int(batch_size)
        self.volume_encoding = volume_encoding
        self.pct_encoding = pct_encoding
        self.prev_price = None
        self.prev_tick = 0
        self.tick_num = 0
        self._reset_cache()

    def get_features(self, verbose=True, to_csv=False, output_path=None):
        """
        Reads a csv file of ticks or pd.DataFrame in batches and then constructs corresponding microstructural intra-bar features:
        average tick size, tick rule sum, VWAP, Kyle lambda, Amihud lambda, Hasbrouck lambda, tick/volume/pct Shannon, Lempel-Ziv,
        Plug-in entropies if corresponding mapping dictionaries are provided (self.volume_encoding, self.pct_encoding).
        The csv file must have only 3 columns: date_time, price, & volume.

        :param verbose: (bool) Flag whether to print message on each processed batch or not
        :param to_csv: (bool) Flag for writing the results of bars generation to local csv file, or to in-memory DataFrame
        :param output_path: (bool) Path to results file, if to_csv = True
        :return: (DataFrame or None) Microstructural features for bar index
        """

        list_bars = []
        cols = ["date_time", "avg_tick_size", "tick_rule_sum", "vwap", "kyle_lambda", "kyle_t",
                "amihud_lambda", "hasbrouck_lambda", "hasbrouck_t",
                "tick_entropy", "vol_entropy", "pct_entropy", "tick_lz", "vol_lz", "pct_lz",
                "tick_plug", "vol_plug", "pct_plug", "tick_konto", "vol_konto", "pct_konto"]
        if to_csv and output_path is None:
            raise ValueError("output_path must be provided when to_csv=True")
        first_write = True
        tick_set = set(self.tick_num_series.values)
        if isinstance(self.trades_input, pd.DataFrame):
            batches = crop_data_frame_in_batches(self.trades_input, self.batch_size)
        else:
            batches = pd.read_csv(self.trades_input, chunksize=self.batch_size)
        for i, batch in enumerate(batches):
            if i == 0:
                self._assert_csv(batch.head(1))
            list_bars.extend(self._extract_bars(batch.values, tick_set))
            if verbose:
                print(f"Processed batch {i + 1}")
            if to_csv and list_bars:
                pd.DataFrame(list_bars, columns=cols).to_csv(output_path, mode="w" if first_write else "a",
                                                            header=first_write, index=False)
                first_write = False
                list_bars = []
        if to_csv:
            return None
        return pd.DataFrame(list_bars, columns=cols)

    def _reset_cache(self):
        """
        Reset price_diff, trade_size, tick_rule, log_ret arrays to empty when bar is formed and features are
        calculated

        :return: None
        """

        self.price_diff = []
        self.trade_size = []
        self.tick_rule = []
        self.log_ret = []
        self.dollar_volume = []

    def _extract_bars(self, data, tick_set):
        """
        For loop which calculates features for formed bars using trades data

        :param data: (tuple) Contains 3 columns - date_time, price, and volume.
        """

        list_bars = []
        for date_time, price, volume in data:
            self.tick_num += 1
            signed_tick = self._apply_tick_rule(price)
            self.tick_rule.append(signed_tick)
            self.trade_size.append(volume)
            self.dollar_volume.append(price * volume)
            self.price_diff.append(self._get_price_diff(price))
            self.log_ret.append(self._get_log_ret(price))
            if self.tick_num in tick_set:
                self._get_bar_features(pd.Timestamp(date_time), list_bars)
        return list_bars

    def _get_bar_features(self, date_time: pd.Timestamp, list_bars: list) -> list:
        """
        Calculate inter-bar features: lambdas, entropies, avg_tick_size, vwap

        :param date_time: (pd.Timestamp) When bar was formed
        :param list_bars: (list) Previously formed bars
        :return: (list) Inter-bar features
        """

        avg_tick = get_avg_tick_size(self.trade_size)
        tick_sum = np.sum(self.tick_rule)
        vwap_val = vwap(self.dollar_volume, self.trade_size)
        kyle_lambda, kyle_t = get_trades_based_kyle_lambda(self.price_diff, self.trade_size, self.tick_rule)
        amihud_lambda = get_trades_based_amihud_lambda(self.log_ret, self.dollar_volume)
        hasb_lambda, hasb_t = get_trades_based_hasbrouck_lambda(self.log_ret, self.dollar_volume, self.tick_rule)
        tick_msg = encode_tick_rule_array(self.tick_rule)
        tick_ent = get_shannon_entropy(tick_msg)
        tick_lz = get_lempel_ziv_entropy(tick_msg)
        tick_plug = get_plug_in_entropy(tick_msg)
        tick_konto = get_konto_entropy(tick_msg)
        vol_ent = pct_ent = vol_lz = pct_lz = vol_plug = pct_plug = vol_konto = pct_konto = np.nan
        if self.volume_encoding is not None:
            vol_msg = encode_array(self.trade_size, self.volume_encoding)
            vol_ent = get_shannon_entropy(vol_msg)
            vol_lz = get_lempel_ziv_entropy(vol_msg)
            vol_plug = get_plug_in_entropy(vol_msg)
            vol_konto = get_konto_entropy(vol_msg)
        if self.pct_encoding is not None:
            pct_msg = encode_array(self.log_ret, self.pct_encoding)
            pct_ent = get_shannon_entropy(pct_msg)
            pct_lz = get_lempel_ziv_entropy(pct_msg)
            pct_plug = get_plug_in_entropy(pct_msg)
            pct_konto = get_konto_entropy(pct_msg)
        list_bars.append([date_time, avg_tick, tick_sum, vwap_val, kyle_lambda, kyle_t,
                          amihud_lambda, hasb_lambda, hasb_t,
                          tick_ent, vol_ent, pct_ent, tick_lz, vol_lz, pct_lz,
                          tick_plug, vol_plug, pct_plug, tick_konto, vol_konto, pct_konto])
        self._reset_cache()
        return list_bars[-1]

    def _apply_tick_rule(self, price: float) -> int:
        """
        Advances in Financial Machine Learning, page 29.

        Applies the tick rule

        :param price: (float) Price at time t
        :return: (int) The signed tick
        """

        if self.prev_price is None:
            tick = 0
        elif price > self.prev_price:
            tick = 1
        elif price < self.prev_price:
            tick = -1
        else:
            tick = self.prev_tick
        self.prev_price = price
        self.prev_tick = tick
        return tick

    def _get_price_diff(self, price: float) -> float:
        """
        Get price difference between ticks

        :param price: (float) Price at time t
        :return: (float) Price difference
        """

        if self.prev_price is None:
            return 0.0
        return price - self.prev_price

    def _get_log_ret(self, price: float) -> float:
        """
        Get log return between ticks

        :param price: (float) Price at time t
        :return: (float) Log return
        """

        if self.prev_price is None or self.prev_price == 0:
            return 0.0
        return np.log(price / self.prev_price)

    @staticmethod
    def _assert_csv(test_batch):
        """
        Tests that the csv file read has the format: date_time, price, and volume.
        If not then the user needs to create such a file. This format is in place to remove any unwanted overhead.

        :param test_batch: (pd.DataFrame) the first row of the dataset.
        :return: (None)
        """

        assert test_batch.shape[1] == 3, 'Must have only 3 columns in csv: date_time, price, & volume.'
