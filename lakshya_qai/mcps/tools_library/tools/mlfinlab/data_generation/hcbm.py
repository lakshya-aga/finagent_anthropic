"""
Implementation of the Hierarchical Correlation Block Model (HCBM) matrix.
"Clustering financial time series: How long is enough?" by Marti, G., Andler, S., Nielsen, F. and Donnat, P.
https://www.ijcai.org/Proceedings/16/Papers/367.pdf
"""
import numpy as np
import pandas as pd
from statsmodels.sandbox.distributions.multivariate import multivariate_t_rvs


def _hcbm_mat_helper(mat, n_low=0, n_high=214, rho_low=0.1, rho_high=0.9, blocks=4, depth=4):
    """
    Helper function for `generate_hcmb_mat` that recursively places rho values to HCBM matrix
    given as an input.

    By using a uniform distribution we select the start and end locations of the blocks in the
    matrix. For each block, we recurse depth times and repeat splitting up the sub-matrix into
    blocks. Each depth level has a unique correlation (rho) values generated from a uniform
    distributions, and bounded by `rho_low` and `rho_high`. This function works as a
    side-effect to the `mat` parameter.

    It is reproduced with modifications from the following paper:
    `Marti, G., Andler, S., Nielsen, F. and Donnat, P., 2016.
    Clustering financial time series: How long is enough?. arXiv preprint arXiv:1603.04017.
    <https://www.ijcai.org/Proceedings/16/Papers/367.pdf>`_

    :param mat: (np.array) Parent HCBM matrix.
    :param n_low: (int) Start location of HCMB matrix to work on.
    :param n_high: (int) End location of HCMB matrix to work on.
    :param rho_low: (float) Lower correlation bound of the matrix. Must be greater or equal
    to 0.
    :param rho_high: (float) Upper correlation bound of the matrix. Must be less or equal to 1.
    :param blocks: (int) Maximum number of blocks to generate per level of depth.
    :param depth: (int) Depth of recursion for generating new blocks.
    """

    if depth <= 0 or n_high - n_low <= 1:
        return
    size = n_high - n_low
    n_blocks = np.random.randint(1, blocks + 1)
    splits = np.sort(np.random.choice(range(n_low + 1, n_high), n_blocks - 1, replace=False)) if size > 2 else []
    bounds = [n_low] + list(splits) + [n_high]
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        rho = np.random.uniform(rho_low, rho_high)
        mat[a:b, a:b] = rho
        _hcbm_mat_helper(mat, a, b, rho_low, rho_high, blocks, depth - 1)


def generate_hcmb_mat(t_samples, n_size, rho_low=0.1, rho_high=0.9, blocks=4, depth=4, permute=False):
    """
    Generates a Hierarchical Correlation Block Model (HCBM) matrix  of correlation values.

    By using a uniform distribution we select the start and end locations of the blocks in the
    matrix. For each block, we recurse depth times and repeat splitting up the sub-matrix into
    blocks. Each depth level has a unique correlation (rho) values generated from a uniform
    distributions, and bounded by `rho_low` and `rho_high`.

    It is reproduced with modifications from the following paper:
    `Marti, G., Andler, S., Nielsen, F. and Donnat, P., 2016.
    Clustering financial time series: How long is enough?. arXiv preprint arXiv:1603.04017.
    <https://www.ijcai.org/Proceedings/16/Papers/367.pdf>`_

    :param t_samples: (int) Number of HCBM matrices to generate.
    :param n_size: (int) Size of HCBM matrix.
    :param rho_low: (float) Lower correlation bound of the matrix. Must be greater or equal to 0.
    :param rho_high: (float) Upper correlation bound of the matrix. Must be less or equal to 1.
    :param blocks: (int) Number of blocks to generate per level of depth.
    :param depth: (int) Depth of recursion for generating new blocks.
    :param permute: (bool) Whether to permute the final HCBM matrix.
    :return: (np.array) Generated HCBM matrix of shape (t_samples, n_size, n_size).
    """

    mats = []
    for _ in range(t_samples):
        mat = np.eye(n_size)
        _hcbm_mat_helper(mat, 0, n_size, rho_low, rho_high, blocks, depth)
        mat = (mat + mat.T) / 2
        np.fill_diagonal(mat, 1)
        if permute:
            p = np.random.permutation(n_size)
            mat = mat[p][:, p]
        mats.append(mat)
    return np.array(mats)


def time_series_from_dist(corr, t_samples=1000, dist="normal", deg_free=3):
    """
    Generates a time series from a given correlation matrix.

    It uses multivariate sampling from distributions to create the time series. It supports
    normal and student-t distributions. This method relies and acts as a wrapper for the
    `np.random.multivariate_normal` and
    `statsmodels.sandbox.distributions.multivariate.multivariate_t_rvs` modules.
    `<https://numpy.org/doc/stable/reference/random/generated/numpy.random.multivariate_normal.html>`_
    `<https://www.statsmodels.org/stable/sandbox.html?highlight=sandbox#module-statsmodels.sandbox>`_

    It is reproduced with modifications from the following paper:
    `Marti, G., Andler, S., Nielsen, F. and Donnat, P., 2016.
    Clustering financial time series: How long is enough?. arXiv preprint arXiv:1603.04017.
    <https://www.ijcai.org/Proceedings/16/Papers/367.pdf>`_

    :param corr: (np.array) Correlation matrix.
    :param t_samples: (int) Number of samples in the time series.
    :param dist: (str) Type of distributions to use.
        Can take the values ["normal", "student"].
    :param deg_free: (int) Degrees of freedom. Only used for student-t distribution.
    :return: (pd.DataFrame) The resulting time series of shape (len(corr), t_samples).
    """

    if dist == "student":
        data = multivariate_t_rvs(np.zeros(len(corr)), corr, df=deg_free, n=t_samples)
    else:
        data = np.random.multivariate_normal(np.zeros(len(corr)), corr, size=t_samples)
    return pd.DataFrame(data)
