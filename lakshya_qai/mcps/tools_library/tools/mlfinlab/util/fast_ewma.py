"""
This module contains an implementation of an exponentially weighted moving average based on sample size.
The inspiration and context for this code was from a blog post by writen by Maksim Ivanov:
https://towardsdatascience.com/financial-machine-learning-part-0-bars-745897d4e4ba
"""

import os
import numpy as np

# Opt-in only: avoids numba import hangs on some Python/conda stacks.
_NUMBA_OK = False
if os.environ.get("FIN_KIT_USE_NUMBA", "0") == "1":
    try:
        from numba import jit, float64, int64
        _NUMBA_OK = True
    except Exception:  # pragma: no cover
        _NUMBA_OK = False


def _ewma_impl(arr_in, window):
    alpha = 2.0 / (window + 1.0)
    n = arr_in.shape[0]
    out = np.empty(n, dtype=np.float64)
    num = 0.0
    den = 0.0
    for i in range(n):
        num = arr_in[i] + (1.0 - alpha) * num
        den = 1.0 + (1.0 - alpha) * den
        out[i] = num / den
    return out


if _NUMBA_OK:
    @jit((float64[:], int64), nopython=False, nogil=True)
    def ewma(arr_in, window):  # pragma: no cover
        """Numba-accelerated EWMA when numba is available."""
        return _ewma_impl(arr_in, window)
else:
    def ewma(arr_in, window):  # pragma: no cover
        """Pure-numpy EWMA fallback for environments without stable numba."""
        return _ewma_impl(arr_in, window)
