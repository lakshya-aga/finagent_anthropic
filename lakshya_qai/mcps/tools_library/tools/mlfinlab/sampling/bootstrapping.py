"""
Logic regarding sequential bootstrapping from chapter 4.
"""

import pandas as pd
import numpy as np
from numba import jit, prange


def get_ind_matrix(samples_info_sets, price_bars):
    """
    Advances in Financial Machine Learning, Snippet 4.3, page 65.

    Build an Indicator Matrix

    Get indicator matrix. The book implementation uses bar_index as input, however there is no explanation
    how to form it. We decided that using triple_barrier_events and price bars by analogy with concurrency
    is the best option.

    :param samples_info_sets: (pd.Series): Triple barrier events(t1) from labeling.get_events
    :param price_bars: (pd.DataFrame): Price bars which were used to form triple barrier events
    :return: (np.array) Indicator binary matrix indicating what (price) bars influence the label for each observation
    """
    t0 = samples_info_sets.index
    t1 = samples_info_sets.values
    bar_index = price_bars.index
    ind_mat = np.zeros((len(bar_index), len(t0)))
    for j, (start, end) in enumerate(zip(t0, t1)):
        mask = (bar_index >= start) & (bar_index <= end)
        ind_mat[mask, j] = 1
    return ind_mat


def get_ind_mat_average_uniqueness(ind_mat):
    """
    Advances in Financial Machine Learning, Snippet 4.4. page 65.

    Compute Average Uniqueness

    Average uniqueness from indicator matrix

    :param ind_mat: (np.matrix) Indicator binary matrix
    :return: (float) Average uniqueness
    """
    uniq = get_ind_mat_label_uniqueness(ind_mat)
    return np.mean(np.nanmean(uniq, axis=0))


def get_ind_mat_label_uniqueness(ind_mat):
    """
    Advances in Financial Machine Learning, An adaption of Snippet 4.4. page 65.

    Returns the indicator matrix element uniqueness.

    :param ind_mat: (np.matrix) Indicator binary matrix
    :return: (np.matrix) Element uniqueness
    """
    concurrency = ind_mat.sum(axis=1).reshape(-1, 1)
    return np.divide(ind_mat, concurrency, out=np.zeros_like(ind_mat, dtype=float), where=concurrency != 0)


@jit(parallel=True, nopython=True)
def _bootstrap_loop_run(ind_mat, prev_concurrency):  # pragma: no cover
    """
    Part of Sequential Bootstrapping for-loop. Using previously accumulated concurrency array, loops through all samples
    and generates averages uniqueness array of label based on previously accumulated concurrency

    :param ind_mat (np.array): Indicator matrix from get_ind_matrix function
    :param prev_concurrency (np.array): Accumulated concurrency from previous iterations of sequential bootstrapping
    :return: (np.array): Label average uniqueness based on prev_concurrency
    """
    n_labels = ind_mat.shape[1]
    avg_u = np.empty(n_labels, dtype=np.float64)
    for i in prange(n_labels):
        idx = ind_mat[:, i] > 0
        if idx.sum() == 0:
            avg_u[i] = 0.0
            continue
        c = prev_concurrency[idx] + ind_mat[idx, i]
        u = ind_mat[idx, i] / c
        avg_u[i] = u.mean()
    return avg_u


def seq_bootstrap(ind_mat, sample_length=None, warmup_samples=None, compare=False, verbose=False,
                  random_state=np.random.RandomState()):
    """
    Advances in Financial Machine Learning, Snippet 4.5, Snippet 4.6, page 65.

    Return Sample from Sequential Bootstrap

    Generate a sample via sequential bootstrap.
    Note: Moved from pd.DataFrame to np.matrix for performance increase

    :param ind_mat: (pd.DataFrame) Indicator matrix from triple barrier events
    :param sample_length: (int) Length of bootstrapped sample
    :param warmup_samples: (list) List of previously drawn samples
    :param compare: (boolean) Flag to print standard bootstrap uniqueness vs sequential bootstrap uniqueness
    :param verbose: (boolean) Flag to print updated probabilities on each step
    :param random_state: (np.random.RandomState) Random state
    :return: (array) Bootstrapped samples indexes
    """
    ind_mat = np.asarray(ind_mat, dtype=float)
    if sample_length is None:
        sample_length = ind_mat.shape[1]
    phi = []
    if warmup_samples is None:
        warmup_samples = []
    prev_concurrency = ind_mat[:, warmup_samples].sum(axis=1) if len(warmup_samples) > 0 else np.zeros(ind_mat.shape[0])
    for _ in range(sample_length):
        avg_u = _bootstrap_loop_run(ind_mat, prev_concurrency)
        prob = avg_u / avg_u.sum() if avg_u.sum() > 0 else np.ones_like(avg_u) / len(avg_u)
        choice = random_state.choice(ind_mat.shape[1], p=prob)
        phi.append(choice)
        prev_concurrency += ind_mat[:, choice]
        if verbose:
            print(f"Chosen {choice}, avg uniqueness {avg_u[choice]:.4f}")
    if compare:
        std_u = get_ind_mat_average_uniqueness(ind_mat)
        seq_u = get_ind_mat_average_uniqueness(ind_mat[:, phi])
        print(f"Standard uniqueness: {std_u:.4f}, Sequential uniqueness: {seq_u:.4f}")
    return phi
