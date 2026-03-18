"""
Entropy calculation module (Shannon, Lempel-Ziv, Plug-In, Konto)
"""

import math
from typing import Union

import numpy as np
from numba import njit


def get_shannon_entropy(message: str) -> float:
    """
    Advances in Financial Machine Learning, page 263-264.

    Get Shannon entropy from message

    :param message: (str) Encoded message
    :return: (float) Shannon entropy
    """

    if len(message) == 0:
        return 0.0
    _, counts = np.unique(list(message), return_counts=True)
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def get_lempel_ziv_entropy(message: str) -> float:
    """
    Advances in Financial Machine Learning, Snippet 18.2, page 266.

    Get Lempel-Ziv entropy estimate

    :param message: (str) Encoded message
    :return: (float) Lempel-Ziv entropy
    """

    i = 0
    n = len(message)
    c = 0
    while i < n:
        l = 1
        while message[i:i + l] in message[:i] and i + l <= n:
            l += 1
        c += 1
        i += l
    return c / n if n > 0 else 0.0


def _prob_mass_function(message: str, word_length: int) -> dict:
    """
    Advances in Financial Machine Learning, Snippet 18.1, page 266.

    Compute probability mass function for a one-dim discete rv

    :param message: (str or array) Encoded message
    :param word_length: (int) Approximate word length
    :return: (dict) Dict of pmf for each word from message
    """

    if word_length <= 0:
        return {}
    counts = {}
    for i in range(0, len(message) - word_length + 1):
        w = message[i:i + word_length]
        counts[w] = counts.get(w, 0) + 1
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def get_plug_in_entropy(message: str, word_length: int = None) -> float:
    """
    Advances in Financial Machine Learning, Snippet 18.1, page 265.

    Get Plug-in entropy estimator

    :param message: (str or array) Encoded message
    :param word_length: (int) Approximate word length
    :return: (float) Plug-in entropy
    """

    if word_length is None:
        word_length = max(1, int(math.log(len(message), 2)) if len(message) > 1 else 1)
    pmf = _prob_mass_function(message, word_length)
    if len(pmf) == 0:
        return 0.0
    probs = np.array(list(pmf.values()))
    return float(-(probs * np.log2(probs)).sum())


@njit()
def _match_length(message: str, start_index: int, window: int) -> Union[int, str]:    # pragma: no cover
    """
    Advances in Financial Machine Learning, Snippet 18.3, page 267.

    Function That Computes the Length of the Longest Match

    :param message: (str or array) Encoded message
    :param start_index: (int) Start index for search
    :param window: (int) Window length
    :return: (int, str) Match length and matched string
    """

    n = len(message)
    max_len = 0
    max_str = ""
    start = max(0, start_index - window) if window > 0 else 0
    for i in range(start, start_index):
        l = 0
        while start_index + l < n and message[i + l] == message[start_index + l]:
            l += 1
            if i + l >= start_index:
                break
        if l > max_len:
            max_len = l
            max_str = message[start_index:start_index + l]
    return max_len, max_str


def get_konto_entropy(message: str, window: int = 0) -> float:
    """
    Advances in Financial Machine Learning, Snippet 18.4, page 268.

    Implementations of Algorithms Discussed in Gao et al.[2008]

    Get Kontoyiannis entropy

    :param message: (str or array) Encoded message
    :param window: (int) Expanding window length, can be negative
    :return: (float) Kontoyiannis entropy
    """

    n = len(message)
    if n == 0:
        return 0.0
    sum_log = 0.0
    for i in range(1, n):
        w = window if window > 0 else i
        l, _ = _match_length(message, i, w)
        sum_log += math.log2((l + 1) if l > 0 else 1)
    return sum_log / n
