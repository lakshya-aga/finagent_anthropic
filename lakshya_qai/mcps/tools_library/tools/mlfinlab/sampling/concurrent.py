"""
Logic regarding concurrent labels from chapter 4.
"""

import pandas as pd

from mlfinlab.util.multiprocess import mp_pandas_obj


def num_concurrent_events(close_series_index, label_endtime, molecule):
    """
    Advances in Financial Machine Learning, Snippet 4.1, page 60.

    Estimating the Uniqueness of a Label

    This function uses close series prices and label endtime (when the first barrier is touched) to compute the number
    of concurrent events per bar.

    :param close_series_index: (pd.Series) Close prices index
    :param label_endtime: (pd.Series) Label endtime series (t1 for triple barrier events)
    :param molecule: (an array) A set of datetime index values for processing
    :return: (pd.Series) Number concurrent labels for each datetime index
    """

    label_endtime = label_endtime.fillna(close_series_index[-1])
    counts = pd.Series(0, index=close_series_index)
    for t0 in molecule:
        t1 = label_endtime.loc[t0]
        counts.loc[t0:t1] += 1
    return counts


def _get_average_uniqueness(label_endtime, num_conc_events, molecule):
    """
    Advances in Financial Machine Learning, Snippet 4.2, page 62.

    Estimating the Average Uniqueness of a Label

    This function uses close series prices and label endtime (when the first barrier is touched) to compute the number
    of concurrent events per bar.

    :param label_endtime: (pd.Series) Label endtime series (t1 for triple barrier events)
    :param num_conc_events: (pd.Series) Number of concurrent labels (output from num_concurrent_events function).
    :param molecule: (an array) A set of datetime index values for processing.
    :return: (pd.Series) Average uniqueness over event's lifespan.
    """

    out = pd.Series(index=molecule, dtype=float)
    for t0 in molecule:
        t1 = label_endtime.loc[t0]
        out.loc[t0] = (1.0 / num_conc_events.loc[t0:t1]).mean()
    return out


def get_av_uniqueness_from_triple_barrier(triple_barrier_events, close_series, num_threads, verbose=True):
    """
    This function is the orchestrator to derive average sample uniqueness from a dataset labeled by the triple barrier
    method.

    :param triple_barrier_events: (pd.DataFrame) Events from labeling.get_events()
    :param close_series: (pd.Series) Close prices.
    :param num_threads: (int) The number of threads concurrently used by the function.
    :param verbose: (bool) Flag to report progress on asynch jobs
    :return: (pd.Series) Average uniqueness over event's lifespan for each index in triple_barrier_events
    """

    label_endtime = triple_barrier_events['t1'].fillna(close_series.index[-1])
    num_conc = mp_pandas_obj(num_concurrent_events, ('molecule', label_endtime.index),
                             num_threads, close_series_index=close_series.index, label_endtime=label_endtime,
                             verbose=verbose)
    av_unique = mp_pandas_obj(_get_average_uniqueness, ('molecule', label_endtime.index),
                              num_threads, label_endtime=label_endtime, num_conc_events=num_conc, verbose=verbose)
    return av_unique
