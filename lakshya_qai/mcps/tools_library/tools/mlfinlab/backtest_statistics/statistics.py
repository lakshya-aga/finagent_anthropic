"""
Implements statistics related to:
- flattening and flips
- average period of position holding
- concentration of bets
- drawdowns
- various Sharpe ratios
- minimum track record length
"""
import warnings
import pandas as pd
import scipy.stats as ss
import numpy as np


def timing_of_flattening_and_flips(target_positions: pd.Series) -> pd.DatetimeIndex:
    """
    Advances in Financial Machine Learning, Snippet 14.1, page 197

    Derives the timestamps of flattening or flipping trades from a pandas series
    of target positions. Can be used for position changes analysis, such as
    frequency and balance of position changes.

    Flattenings - times when open position is bing closed (final target position is 0).
    Flips - times when positive position is reversed to negative and vice versa.

    :param target_positions: (pd.Series) Target position series with timestamps as indices
    :return: (pd.DatetimeIndex) Timestamps of trades flattening, flipping and last bet
    """
    pos = target_positions.fillna(0)
    prev = pos.shift(1).fillna(0)
    flatten = (prev != 0) & (pos == 0)
    flip = (prev * pos) < 0
    events = pos.index[flatten | flip]
    if len(pos) > 0:
        events = events.union(pd.DatetimeIndex([pos.index[-1]]))
    return events


def average_holding_period(target_positions: pd.Series) -> float:
    """
    Advances in Financial Machine Learning, Snippet 14.2, page 197

    Estimates the average holding period (in days) of a strategy, given a pandas series
    of target positions using average entry time pairing algorithm.

    Idea of an algorithm:

    * entry_time = (previous_time * weight_of_previous_position + time_since_beginning_of_trade * increase_in_position )
      / weight_of_current_position
    * holding_period ['holding_time' = time a position was held, 'weight' = weight of position closed]
    * res = weighted average time a trade was held

    :param target_positions: (pd.Series) Target position series with timestamps as indices
    :return: (float) Estimated average holding period, NaN if zero or unpredicted
    """
    pos = target_positions.fillna(0)
    if pos.abs().sum() == 0:
        return np.nan
    entry_time = None
    entry_pos = 0.0
    weighted_holding = []
    for t, p in pos.items():
        if entry_time is None:
            if p != 0:
                entry_time = t
                entry_pos = p
            continue
        change = p - entry_pos
        if entry_pos == 0 and p != 0:
            entry_time = t
            entry_pos = p
            continue
        if entry_pos != 0 and p == 0:
            holding = (t - entry_time).total_seconds() / (60 * 60 * 24)
            weighted_holding.append((holding, abs(entry_pos)))
            entry_time = None
            entry_pos = 0.0
            continue
        if entry_pos != 0 and p != 0 and np.sign(entry_pos) != np.sign(p):
            holding = (t - entry_time).total_seconds() / (60 * 60 * 24)
            weighted_holding.append((holding, abs(entry_pos)))
            entry_time = t
            entry_pos = p
            continue
        if entry_pos != 0 and abs(p) < abs(entry_pos):
            holding = (t - entry_time).total_seconds() / (60 * 60 * 24)
            closed = abs(entry_pos) - abs(p)
            weighted_holding.append((holding, closed))
            entry_pos = p
            entry_time = t
            continue
        if entry_pos != 0 and abs(p) > abs(entry_pos):
            avg_ns = (entry_time.value * abs(entry_pos) + t.value * abs(change)) / abs(p)
            entry_time = pd.Timestamp(int(avg_ns))
            entry_pos = p
    if len(weighted_holding) == 0:
        return np.nan
    total_w = sum(w for _, w in weighted_holding)
    return sum(h * w for h, w in weighted_holding) / total_w if total_w else np.nan

def bets_concentration(returns: pd.Series) -> float:
    """
    Advances in Financial Machine Learning, Snippet 14.3, page 201

    Derives the concentration of returns from given pd.Series of returns.

    Algorithm is based on Herfindahl-Hirschman Index where return weights
    are taken as an input.

    :param returns: (pd.Series) Returns from bets
    :return: (float) Concentration of returns (nan if less than 3 returns)
    """
    rets = returns.dropna()
    if len(rets) < 3:
        return np.nan
    w = rets.abs()
    if w.sum() == 0:
        return 0.0
    w = w / w.sum()
    return (w ** 2).sum()


def all_bets_concentration(returns: pd.Series, frequency: str = 'M') -> tuple:
    """
    Advances in Financial Machine Learning, Snippet 14.3, page 201

    Given a pd.Series of returns, derives concentration of positive returns, negative returns
    and concentration of bets grouped by time intervals (daily, monthly etc.).
    If after time grouping less than 3 observations, returns nan.

    Properties or results:

    * low positive_concentration ⇒ no right fat-tail of returns (desirable)
    * low negative_concentration ⇒ no left fat-tail of returns (desirable)
    * low time_concentration ⇒ bets are not concentrated in time, or are evenly concentrated (desirable)
    * positive_concentration == 0 ⇔ returns are uniform
    * positive_concentration == 1 ⇔ only one non-zero return exists

    :param returns: (pd.Series) Returns from bets
    :param frequency: (str) Desired time grouping frequency from pd.Grouper
    :return: (tuple of floats) Concentration of positive, negative and time grouped concentrations
    """
    rets = returns.dropna()
    pos = bets_concentration(rets[rets > 0])
    neg = bets_concentration(-rets[rets < 0])
    by_time = bets_concentration(rets.groupby(pd.Grouper(freq=frequency)).sum())
    return pos, neg, by_time

def drawdown_and_time_under_water(returns: pd.Series, dollars: bool = False) -> tuple:
    """
    Advances in Financial Machine Learning, Snippet 14.4, page 201

    Calculates drawdowns and time under water for pd.Series of either relative price of a
    portfolio or dollar price of a portfolio.

    Intuitively, a drawdown is the maximum loss suffered by an investment between two consecutive high-watermarks.
    The time under water is the time elapsed between an high watermark and the moment the PnL (profit and loss)
    exceeds the previous maximum PnL. We also append the Time under water series with period from the last
    high-watermark to the last return observed.

    Return details:

    * Drawdown series index is the time of a high watermark and the value of a
      drawdown after it.
    * Time under water index is the time of a high watermark and how much time
      passed till the next high watermark in years. Also includes time between
      the last high watermark and last observation in returns as the last element.

    :param returns: (pd.Series) Returns from bets
    :param dollars: (bool) Flag if given dollar performance and not returns.
                    If dollars, then drawdowns are in dollars, else as a %.
    :return: (tuple of pd.Series) Series of drawdowns and time under water
    """
    series = returns.dropna().copy()
    if not dollars:
        series = (1.0 + series).cumprod()
    hwm = series.cummax()
    if dollars:
        dd = hwm - series
    else:
        dd = 1.0 - series / hwm
    hwm_ts = hwm[hwm == series].index
    dd_series = dd.loc[hwm_ts]
    tuw = []
    for i in range(len(hwm_ts) - 1):
        delta = (hwm_ts[i + 1] - hwm_ts[i]).days / 365.25
        tuw.append(delta)
    if len(hwm_ts) > 0:
        delta = (series.index[-1] - hwm_ts[-1]).days / 365.25
        tuw.append(delta)
    tuw_series = pd.Series(tuw, index=hwm_ts)
    return dd_series, tuw_series


def sharpe_ratio(returns: pd.Series, entries_per_year: int = 252, risk_free_rate: float = 0) -> float:
    """
    Calculates annualized Sharpe ratio for pd.Series of normal or log returns.

    Risk_free_rate should be given for the same period the returns are given.
    For example, if the input returns are observed in 3 months, the risk-free
    rate given should be the 3-month risk-free rate.

    :param returns: (pd.Series) Returns - normal or log
    :param entries_per_year: (int) Times returns are recorded per year (252 by default)
    :param risk_free_rate: (float) Risk-free rate (0 by default)
    :return: (float) Annualized Sharpe ratio
    """
    rets = returns.dropna() - risk_free_rate
    vol = rets.std()
    if vol == 0 or len(rets) == 0:
        return np.nan
    return np.sqrt(entries_per_year) * rets.mean() / vol


def information_ratio(returns: pd.Series, benchmark: float = 0, entries_per_year: int = 252) -> float:
    """
    Calculates annualized information ratio for pd.Series of normal or log returns.

    Benchmark should be provided as a return for the same time period as that between
    input returns. For example, for the daily observations it should be the
    benchmark of daily returns.

    It is the annualized ratio between the average excess return and the tracking error.
    The excess return is measured as the portfolio’s return in excess of the benchmark’s
    return. The tracking error is estimated as the standard deviation of the excess returns.

    :param returns: (pd.Series) Returns - normal or log
    :param benchmark: (float) Benchmark for performance comparison (0 by default)
    :param entries_per_year: (int) Times returns are recorded per year (252 by default)
    :return: (float) Annualized information ratio
    """
    excess = returns.dropna() - benchmark
    te = excess.std()
    if te == 0 or len(excess) == 0:
        return np.nan
    return np.sqrt(entries_per_year) * excess.mean() / te


def probabilistic_sharpe_ratio(observed_sr: float, benchmark_sr: float, number_of_returns: int,
                               skewness_of_returns: float = 0, kurtosis_of_returns: float = 3) -> float:
    """
    Calculates the probabilistic Sharpe ratio (PSR) that provides an adjusted estimate of SR,
    by removing the inflationary effect caused by short series with skewed and/or
    fat-tailed returns.

    Given a user-defined benchmark Sharpe ratio and an observed Sharpe ratio,
    PSR estimates the probability that SR ̂is greater than a hypothetical SR.
    - It should exceed 0.95, for the standard significance level of 5%.
    - It can be computed on absolute or relative returns.

    :param observed_sr: (float) Sharpe ratio that is observed
    :param benchmark_sr: (float) Sharpe ratio to which observed_SR is tested against
    :param number_of_returns: (int) Times returns are recorded for observed_SR
    :param skewness_of_returns: (float) Skewness of returns (0 by default)
    :param kurtosis_of_returns: (float) Kurtosis of returns (3 by default)
    :return: (float) Probabilistic Sharpe ratio
    """
    if number_of_returns <= 1:
        return np.nan
    denom = np.sqrt((1 - skewness_of_returns * observed_sr +
                     (kurtosis_of_returns - 1) * observed_sr ** 2 / 4.0) / (number_of_returns - 1))
    return ss.norm.cdf((observed_sr - benchmark_sr) / denom)


def deflated_sharpe_ratio(observed_sr: float, sr_estimates: list, number_of_returns: int,
                          skewness_of_returns: float = 0, kurtosis_of_returns: float = 3,
                          estimates_param: bool = False, benchmark_out: bool = False) -> float:
    """
    Calculates the deflated Sharpe ratio (DSR) - a PSR where the rejection threshold is
    adjusted to reflect the multiplicity of trials. DSR is estimated as PSR[SR∗], where
    the benchmark Sharpe ratio, SR∗, is no longer user-defined, but calculated from
    SR estimate trails.

    DSR corrects SR for inflationary effects caused by non-Normal returns, track record
    length, and multiple testing/selection bias.
    - It should exceed 0.95, for the standard significance level of 5%.
    - It can be computed on absolute or relative returns.

    Function allows the calculated SR benchmark output and usage of only
    standard deviation and number of SR trails instead of full list of trails.

    :param observed_sr: (float) Sharpe ratio that is being tested
    :param sr_estimates: (list) Sharpe ratios estimates trials list or
        properties list: [Standard deviation of estimates, Number of estimates]
        if estimates_param flag is set to True.
    :param  number_of_returns: (int) Times returns are recorded for observed_SR
    :param skewness_of_returns: (float) Skewness of returns (0 by default)
    :param kurtosis_of_returns: (float) Kurtosis of returns (3 by default)
    :param estimates_param: (bool) Flag to use properties of estimates instead of full list
    :param benchmark_out: (bool) Flag to output the calculated benchmark instead of DSR
    :return: (float) Deflated Sharpe ratio or Benchmark SR (if benchmark_out)
    """
    if estimates_param:
        std_est, n_est = sr_estimates
    else:
        est = np.array(sr_estimates, dtype=float)
        std_est = est.std(ddof=1) if len(est) > 1 else 0.0
        n_est = len(est)
    if n_est <= 1:
        sr_star = 0.0
    else:
        sr_star = std_est * ss.norm.ppf(1 - 1.0 / n_est)
    if benchmark_out:
        return sr_star
    return probabilistic_sharpe_ratio(observed_sr, sr_star, number_of_returns,
                                      skewness_of_returns, kurtosis_of_returns)


def minimum_track_record_length(observed_sr: float, benchmark_sr: float,
                                skewness_of_returns: float = 0,
                                kurtosis_of_returns: float = 3,
                                alpha: float = 0.05) -> float:
    """
    Calculates the minimum track record length (MinTRL) - "How long should a track
    record be in order to have statistical confidence that its Sharpe ratio is above
    a given threshold?”

    If a track record is shorter than MinTRL, we do not  have  enough  confidence
    that  the  observed Sharpe ratio ̂is above the designated Sharpe ratio threshold.

    MinTRLis expressed in terms of number of observations, not annual or calendar terms.

    :param observed_sr: (float) Sharpe ratio that is being tested
    :param benchmark_sr: (float) Sharpe ratio to which observed_SR is tested against
    :param  number_of_returns: (int) Times returns are recorded for observed_SR
    :param skewness_of_returns: (float) Skewness of returns (0 by default)
    :param kurtosis_of_returns: (float) Kurtosis of returns (3 by default)
    :param alpha: (float) Desired significance level (0.05 by default)
    :return: (float) Minimum number of track records
    """
    if observed_sr <= benchmark_sr:
        return np.inf
    z = ss.norm.ppf(1 - alpha)
    denom = (observed_sr - benchmark_sr) ** 2
    num = 1 - skewness_of_returns * observed_sr + (kurtosis_of_returns - 1) * observed_sr ** 2 / 4.0
    return 1 + num * (z ** 2) / denom
