"""
Various functions for message encoding (quantile)
"""
import numpy as np


def encode_tick_rule_array(tick_rule_array: list) -> str:
    """
    Encode array of tick signs (-1, 1, 0)

    :param tick_rule_array: (list) Tick rules
    :return: (str) Encoded message
    """

    mapping = {-1: 'a', 0: 'b', 1: 'c'}
    return ''.join(mapping.get(x, 'b') for x in tick_rule_array)


def _get_ascii_table() -> list:
    """
    Get all ASCII symbols

    :return: (list) ASCII symbols
    """

    return [chr(i) for i in range(33, 127)]


def quantile_mapping(array: list, num_letters: int = 26) -> dict:
    """
    Generate dictionary of quantile-letters based on values from array and dictionary length (num_letters).

    :param array: (list) Values to split on quantiles
    :param num_letters: (int) Number of letters(quantiles) to encode
    :return: (dict) Dict of quantile-symbol
    """

    letters = _get_ascii_table()[:num_letters]
    qs = np.quantile(array, np.linspace(0, 1, num_letters))
    return {q: letters[i] for i, q in enumerate(qs)}


def sigma_mapping(array: list, step: float = 0.01) -> dict:
    """
    Generate dictionary of sigma encoded letters based on values from array and discretization step.

    :param array: (list) Values to split on quantiles
    :param step: (float) Discretization step (sigma)
    :return: (dict) Dict of value-symbol
    """

    letters = _get_ascii_table()
    arr = np.asarray(array)
    bins = np.arange(arr.min(), arr.max() + step, step)
    return {b: letters[i % len(letters)] for i, b in enumerate(bins)}


def _find_nearest(array: list, value: float) -> float:
    """
    Find the nearest element from array to value.

    :param array: (list) Values
    :param value: (float) Value for which the nearest element needs to be found
    :return: (float) The nearest to the value element in array
    """

    array = np.asarray(list(array))
    return array[(np.abs(array - value)).argmin()]


def _get_letter_from_encoding(value: float, encoding_dict: dict) -> str:
    """
    Get letter for float/int value from encoding dict.

    :param value: (float/int) Value to use
    :param encoding_dict: (dict) Used dictionary
    :return: (str) Letter from encoding dict
    """

    key = _find_nearest(list(encoding_dict.keys()), value)
    return encoding_dict[key]


def encode_array(array: list, encoding_dict: dict) -> str:
    """
    Encode array with strings using encoding dict, in case of multiple occurrences of the minimum values,
    the indices corresponding to the first occurrence are returned

    :param array: (list) Values to encode
    :param encoding_dict: (dict) Dict of quantile-symbol
    :return: (str) Encoded message
    """

    return ''.join(_get_letter_from_encoding(v, encoding_dict) for v in array)
