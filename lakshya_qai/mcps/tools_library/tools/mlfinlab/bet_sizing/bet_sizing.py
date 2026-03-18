"""
This module contains functionality for determining bet sizes for investments based on machine learning predictions.
These implementations are based on bet sizing approaches described in Chapter 10.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, moment

from mlfinlab.bet_sizing.ch10_snippets import get_signal, avg_active_signals, discrete_signal
from mlfinlab.bet_sizing.ch10_snippets import get_w, get_target_pos, limit_price, bet_size
from mlfinlab.bet_sizing.ef3m import M2N, raw_moment, most_likely_parameters


def bet_size_probability(events, prob, num_classes, pred=None, step_size=0.0, average_active=False, num_threads=1):
    """
    Calculates the bet size using the predicted probability. Note that if 'average_active' is True, the returned
    pandas.Series will be twice the length of the original since the average is calculated at each bet's open and close.

    :param events: (pandas.DataFrame) Contains at least the column 't1', the expiry datetime of the product, with
     a datetime index, the datetime the position was taken.
    :param prob: (pandas.Series) The predicted probability.
    :param num_classes: (int) The number of predicted bet sides.
    :param pred: (pd.Series) The predicted bet side. Default value is None which will return a relative bet size
     (i.e. without multiplying by the side).
    :param step_size: (float) The step size at which the bet size is discretized, default is 0.0 which imposes no
     discretization.
    :param average_active: (bool) Option to average the size of active bets, default value is False.
    :param num_threads: (int) The number of processing threads to utilize for multiprocessing, default value is 1.
    :return: (pandas.Series) The bet size, with the time index.
    """

    signal = get_signal(prob, num_classes, pred)
    if step_size > 0:
        signal = discrete_signal(signal, step_size)
    if average_active:
        df = pd.DataFrame({'signal': signal, 't1': events['t1']})
        return avg_active_signals(df, num_threads=num_threads)
    return signal


def bet_size_dynamic(current_pos, max_pos, market_price, forecast_price, cal_divergence=10, cal_bet_size=0.95,
                     func='sigmoid'):
    """
    Calculates the bet sizes, target position, and limit price as the market price and forecast price fluctuate.
    The current position, maximum position, market price, and forecast price can be passed as separate pandas.Series
    (with a common index), as individual numbers, or a combination thereof. If any one of the aforementioned arguments
    is a pandas.Series, the other arguments will be broadcast to a pandas.Series of the same length and index.

    :param current_pos: (pandas.Series, int) Current position.
    :param max_pos: (pandas.Series, int) Maximum position
    :param market_price: (pandas.Series, float) Market price.
    :param forecast_price: (pandas.Series, float) Forecast price.
    :param cal_divergence: (float) The divergence to use in calibration.
    :param cal_bet_size: (float) The bet size to use in calibration.
    :param func: (string) Function to use for dynamic calculation. Valid options are: 'sigmoid', 'power'.
    :return: (pandas.DataFrame) Bet size (bet_size), target position (t_pos), and limit price (l_p).
    """

    df = confirm_and_cast_to_df({
        'current_pos': current_pos,
        'max_pos': max_pos,
        'market_price': market_price,
        'forecast_price': forecast_price
    })
    w = get_w(cal_divergence, cal_bet_size, func)
    price_div = df['forecast_price'] - df['market_price']
    bet = bet_size(w, price_div, func)
    t_pos = df['max_pos'] * bet
    l_p = limit_price(t_pos, df['current_pos'], df['forecast_price'], w, df['max_pos'], func)
    return pd.DataFrame({'bet_size': bet, 't_pos': t_pos, 'l_p': l_p})


def bet_size_budget(events_t1, sides):
    """
    Calculates a bet size from the bet sides and start and end times. These sequences are used to determine the
    number of concurrent long and short bets, and the resulting strategy-independent bet sizes are the difference
    between the average long and short bets at any given time. This strategy is based on the section 10.2
    in "Advances in Financial Machine Learning". This creates a linear bet sizing scheme that is aligned to the
    expected number of concurrent bets in the dataset.

    :param events_t1: (pandas.Series) The end datetime of the position with the start datetime as the index.
    :param sides: (pandas.Series) The side of the bet with the start datetime as index. Index must match the
     'events_t1' argument exactly. Bet sides less than zero are interpretted as short, bet sides greater than zero
     are interpretted as long.
    :return: (pandas.DataFrame) The 'events_t1' and 'sides' arguments as columns, with the number of concurrent
     active long and short bets, as well as the bet size, in additional columns.
    """

    df = get_concurrent_sides(events_t1, sides)
    df['bet_size'] = (df['long'] - df['short']) / (df['long'] + df['short'])
    return df


def bet_size_reserve(events_t1, sides, fit_runs=100, epsilon=1e-5, factor=5, variant=2, max_iter=10_000,
                     num_workers=1, return_parameters=False):
    """
    Calculates the bet size from bet sides and start and end times. These sequences are used to determine the number
    of concurrent long and short bets, and the difference between the two at each time step, c_t. A mixture of two
    Gaussian distributions is fit to the distribution of c_t, which is then used to determine the bet size. This
    strategy results in a sigmoid-shaped bet sizing response aligned to the expected number of concurrent long
    and short bets in the dataset.

    Note that this function creates a <mlfinlab.bet_sizing.ef3m.M2N> object and makes use of the parallel fitting
    functionality. As such, this function accepts and passes fitting parameters to the
    mlfinlab.bet_sizing.ef3m.M2N.mp_fit() method.

    :param events_t1: (pandas.Series) The end datetime of the position with the start datetime as the index.
    :param sides: (pandas.Series) The side of the bet with the start datetime as index. Index must match the
     'events_t1' argument exactly. Bet sides less than zero are interpretted as short, bet sides greater than zero
     are interpretted as long.
    :param fit_runs: (int) Number of runs to execute when trying to fit the distribution.
    :param epsilon: (float) Error tolerance.
    :param factor: (float) Lambda factor from equations.
    :param variant: (int) Which algorithm variant to use, 1 or 2.
    :param max_iter: (int) Maximum number of iterations after which to terminate loop.
    :param num_workers: (int) Number of CPU cores to use for multiprocessing execution, set to -1 to use all
     CPU cores. Default is 1.
    :param return_parameters: (bool) If True, function also returns a dictionary of the fited mixture parameters.
    :return: (pandas.DataFrame) The 'events_t1' and 'sides' arguments as columns, with the number of concurrent
     active long, short bets, the difference between long and short, and the bet size in additional columns.
     Also returns the mixture parameters if 'return_parameters' is set to True.
    """

    df = get_concurrent_sides(events_t1, sides)
    c_t = df['long'] - df['short']
    moments = [c_t.mean(), moment(c_t, 2), moment(c_t, 3), moment(c_t, 4), moment(c_t, 5)]
    model = M2N(moments, epsilon=epsilon, factor=factor, n_runs=fit_runs, variant=variant, max_iter=max_iter,
                num_workers=num_workers)
    fit = model.mp_fit()
    params = most_likely_parameters(fit)
    df['bet_size'] = c_t.apply(lambda x: single_bet_size_mixed(x, list(params.values())))
    return (df, params) if return_parameters else df


def confirm_and_cast_to_df(d_vars):
    """
    Accepts either pandas.Series (with a common index) or integer/float values, casts all non-pandas.Series values
    to Series, and returns a pandas.DataFrame for further calculations. This is a helper function to the
    'bet_size_dynamic' function.

    :param d_vars: (dict) A dictionary where the values are either pandas.Series or single int/float values.
     All pandas.Series passed are assumed to have the same index. The keys of the dictionary will be used for column
     names in the returned pandas.DataFrame.
    :return: (pandas.DataFrame) The values from the input dictionary in pandas.DataFrame format, with dictionary
     keys as column names.
    """

    idx = None
    for v in d_vars.values():
        if isinstance(v, pd.Series):
            idx = v.index
            break
    out = {}
    for k, v in d_vars.items():
        if isinstance(v, pd.Series):
            out[k] = v
        else:
            out[k] = pd.Series(v, index=idx)
    return pd.DataFrame(out)


def get_concurrent_sides(events_t1, sides):
    """
    Given the side of the position along with its start and end timestamps, this function returns two pandas.Series
    indicating the number of concurrent long and short bets at each timestamp.

    :param events_t1: (pandas.Series) The end datetime of the position with the start datetime as the index.
    :param sides: (pandas.Series) The side of the bet with the start datetime as index. Index must match the
     'events_t1' argument exactly. Bet sides less than zero are interpretted as short, bet sides greater than zero
     are interpretted as long.
    :return: (pandas.DataFrame) The 'events_t1' and 'sides' arguments as columns, with two additional columns
     indicating the number of concurrent active long and active short bets at each timestamp.
    """

    df = pd.DataFrame({'t1': events_t1, 'side': sides})
    idx = df.index.union(df['t1']).dropna().sort_values()
    long = pd.Series(0, index=idx)
    short = pd.Series(0, index=idx)
    for t0, row in df.iterrows():
        t1 = row['t1']
        if pd.isna(t1):
            t1 = idx[-1]
        if row['side'] > 0:
            long.loc[t0:t1] += 1
        elif row['side'] < 0:
            short.loc[t0:t1] += 1
    res = pd.DataFrame({'t1': events_t1, 'side': sides})
    res['long'] = long.reindex(df.index, method='ffill').fillna(0).values
    res['short'] = short.reindex(df.index, method='ffill').fillna(0).values
    return res


def cdf_mixture(x_val, parameters):
    """
    The cumulative distribution function of a mixture of 2 normal distributions, evaluated at x_val.

    :param x_val: (float) Value at which to evaluate the CDF.
    :param parameters: (list) The parameters of the mixture, [mu_1, mu_2, sigma_1, sigma_2, p_1]
    :return: (float) CDF of the mixture.
    """

    mu1, mu2, s1, s2, p1 = parameters
    return p1 * norm.cdf(x_val, mu1, s1) + (1 - p1) * norm.cdf(x_val, mu2, s2)


def single_bet_size_mixed(c_t, parameters):
    """
    Returns the single bet size based on the description provided in question 10.4(c), provided the difference in
    concurrent long and short positions, c_t, and the fitted parameters of the mixture of two Gaussain distributions.

    :param c_t: (int) The difference in the number of concurrent long bets minus short bets.
    :param parameters: (list) The parameters of the mixture, [mu_1, mu_2, sigma_1, sigma_2, p_1]
    :return: (float) Bet size.
    """

    return 2 * cdf_mixture(c_t, parameters) - 1
