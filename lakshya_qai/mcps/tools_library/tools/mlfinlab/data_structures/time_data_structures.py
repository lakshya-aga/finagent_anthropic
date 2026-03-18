"""
Advances in Financial Machine Learning, Marcos Lopez de Prado
Chapter 2: Financial Data Structures

Time bars generation logic
"""

# Imports
from typing import Union, Iterable, Optional
import numpy as np
import pandas as pd

from mlfinlab.data_structures.base_bars import BaseBars


# pylint: disable=too-many-instance-attributes
class TimeBars(BaseBars):
    """
    Contains all of the logic to construct the time bars. This class shouldn't be used directly.
    Use get_time_bars instead
    """

    def __init__(self, resolution: str, num_units: int, batch_size: int = 20000000):
        """
        Constructor

        :param resolution: (str) Type of bar resolution: ['D', 'H', 'MIN', 'S']
        :param num_units: (int) Number of days, minutes, etc.
        :param batch_size: (int) Number of rows to read in from the csv, per batch
        """
        super().__init__(metric="time", batch_size=batch_size)
        self.resolution = resolution.upper()
        self.num_units = num_units
        if self.resolution == "D":
            self.delta = pd.Timedelta(days=num_units)
        elif self.resolution == "H":
            self.delta = pd.Timedelta(hours=num_units)
        elif self.resolution == "MIN":
            self.delta = pd.Timedelta(minutes=num_units)
        else:
            self.delta = pd.Timedelta(seconds=num_units)
        self.next_bar_time = None

    def _reset_cache(self):
        """
        Implementation of abstract method _reset_cache for time bars
        """
        self.open_price = None
        self.high_price = None
        self.low_price = None
        self.cum_ticks = 0
        self.cum_dollar_value = 0.0
        self.cum_volume = 0.0
        self.cum_buy_volume = 0.0

    def _extract_bars(self, data: Union[list, tuple, np.ndarray]) -> list:
        """
        For loop which compiles time bars.
        We did investigate the use of trying to solve this in a vectorised manner but found that a For loop worked well.

        :param data: (tuple) Contains 3 columns - date_time, price, and volume.
        :return: (list) Extracted bars
        """
        list_bars = []
        rows = data.values if hasattr(data, "values") else data
        for date_time, price, volume in rows:
            dt = pd.Timestamp(date_time)
            if self.next_bar_time is None:
                self.next_bar_time = dt + self.delta
            if self.open_price is None:
                self.open_price = price
                self.high_price = price
                self.low_price = price
            signed_tick = self._apply_tick_rule(price)
            self.cum_ticks += 1
            self.cum_volume += volume
            self.cum_dollar_value += price * volume
            if signed_tick == 1:
                self.cum_buy_volume += volume
            self._update_high_low(price)
            if dt >= self.next_bar_time:
                self._create_bars(date_time, price, self.high_price, self.low_price, list_bars)
                self.next_bar_time = dt + self.delta
        return list_bars


def get_time_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], resolution: str = 'D', num_units: int = 1, batch_size: int = 20000000,
                  verbose: bool = True, to_csv: bool = False, output_path: Optional[str] = None):
    """
    Creates Time Bars: date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value.

    :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing raw tick data
                            in the format[date_time, price, volume]
    :param resolution: (str) Resolution type ('D', 'H', 'MIN', 'S')
    :param num_units: (int) Number of resolution units (3 days for example, 2 hours)
    :param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: (int) Print out batch numbers (True or False)
    :param to_csv: (bool) Save bars to csv after every batch run (True or False)
    :param output_path: (str) Path to csv file, if to_csv is True
    :return: (pd.DataFrame) Dataframe of time bars, if to_csv=True return None
    """
    bars = TimeBars(resolution=resolution, num_units=num_units, batch_size=batch_size)
    return bars.batch_run(file_path_or_df, verbose=verbose, to_csv=to_csv, output_path=output_path)
