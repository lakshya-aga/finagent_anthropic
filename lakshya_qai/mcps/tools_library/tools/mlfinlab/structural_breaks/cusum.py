"""
Implementation of Chu-Stinchcombe-White test
"""

import pandas as pd
import numpy as np
from mlfinlab.util import mp_pandas_obj


def _get_values_diff(test_type, series, index, ind):
    """
    Gets the difference between two values given a test type.
    :param test_type: (str) Type of the test ['one_sided', 'two_sided']
    :param series: (pd.Series) Series of values
    :param index: (pd.Index) primary index
    :param ind: (pd.Index) secondary index
    :return: (float) Difference between 2 values
    """
    vals = series.loc[index] - series.loc[ind]
    if test_type == 'two_sided':
        return vals.abs()
    return vals


def _get_s_n_for_t(series: pd.Series, test_type: str, molecule: list) -> pd.Series:
    """
    Get maximum S_n_t value for each value from molecule for Chu-Stinchcombe-White test

    :param series: (pd.Series) Series to get statistics for
    :param test_type: (str): Two-sided or one-sided test
    :param molecule: (list) Indices to get test statistics for
    :return: (pd.Series) Statistics
    """
    stats = {}
    for t in molecule:
        idx = series.index.get_loc(t)
        if idx == 0:
            stats[t] = 0.0
            continue
        prev_index = series.index[:idx]
        diff = _get_values_diff(test_type, series, t, prev_index)
        denom = series.loc[prev_index].std()
        if denom == 0 or len(prev_index) == 0:
            stats[t] = 0.0
            continue
        scale = np.sqrt(np.arange(1, idx + 1))
        s_vals = diff.values / (denom * scale)
        stats[t] = np.max(s_vals) if test_type == 'one_sided' else np.max(np.abs(s_vals))
    return pd.Series(stats)


def get_chu_stinchcombe_white_statistics(series: pd.Series, test_type: str = 'one_sided',
                                         num_threads: int = 8, verbose: bool = True) -> pd.Series:
    """
    Multithread Chu-Stinchcombe-White test implementation, p.251

    :param series: (pd.Series) Series to get statistics for
    :param test_type: (str): Two-sided or one-sided test
    :param num_threads: (int) Number of cores
    :param verbose: (bool) Flag to report progress on asynch jobs
    :return: (pd.Series) Statistics
    """
    series = series.dropna()
    out = mp_pandas_obj(_get_s_n_for_t, ('molecule', series.index), num_threads,
                        series=series, test_type=test_type, verbose=verbose)
    return out.sort_index()
