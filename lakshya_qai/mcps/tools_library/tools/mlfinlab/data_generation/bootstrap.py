"""
Implementation of generating bootstrapped matrices from
"Bootstrap validation of links of a minimum spanning tree" by F. Musciotto,
L. Marotta, S. Miccichè, and R. N. Mantegna https://arxiv.org/pdf/1802.03395.pdf.
"""

import numpy as np
import pandas as pd


def row_bootstrap(mat, n_samples=1, size=None):
    """
    Uses the Row Bootstrap method to generate a new matrix of size equal or smaller than the given matrix.

    It samples with replacement a random row from the given matrix. If the required bootstrapped
    columns' size is less than the columns of the original matrix, it randomly samples contiguous
    columns of the required size. It cannot generate a matrix greater than the original.

    It is inspired by the following paper:
    `Musciotto, F., Marotta, L., Miccichè, S. and Mantegna, R.N., 2018. Bootstrap validation of
    links of a minimum spanning tree. Physica A: Statistical Mechanics and its Applications,
    512, pp.1032-1043. <https://arxiv.org/pdf/1802.03395.pdf>`_.

    :param mat: (pd.DataFrame/np.array) Matrix to sample from.
    :param n_samples: (int) Number of matrices to generate.
    :param size: (tuple) Size of the bootstrapped matrix.
    :return: (np.array) The generated bootstrapped matrices. Has shape (n_samples, size[0], size[1]).
    """

    arr = np.asarray(mat)
    n_rows, n_cols = arr.shape
    if size is None:
        size = (n_rows, n_cols)
    out = np.zeros((n_samples, size[0], size[1]))
    for i in range(n_samples):
        rows = np.random.choice(n_rows, size[0], replace=True)
        cols = np.arange(n_cols)
        if size[1] < n_cols:
            start = np.random.randint(0, n_cols - size[1] + 1)
            cols = cols[start:start + size[1]]
        out[i] = arr[np.ix_(rows, cols)]
    return out


def pair_bootstrap(mat, n_samples=1, size=None):
    """
    Uses the Pair Bootstrap method to generate a new correlation matrix of returns.

    It generates a correlation matrix based on the number of columns of the returns matrix given. It
    samples with replacement a pair of columns from the original matrix, the rows of the pairs generate
    a new row-bootstrapped matrix. The correlation value of the pair of assets is calculated and
    its value is used to fill the corresponding value in the generated correlation matrix.

    It is inspired by the following paper:
    `Musciotto, F., Marotta, L., Miccichè, S. and Mantegna, R.N., 2018. Bootstrap validation of
    links of a minimum spanning tree. Physica A: Statistical Mechanics and its Applications,
    512, pp.1032-1043. <https://arxiv.org/pdf/1802.03395.pdf>`_.

    :param mat: (pd.DataFrame/np.array) Returns matrix to sample from.
    :param n_samples: (int) Number of matrices to generate.
    :param size: (int) Size of the bootstrapped correlation matrix.
    :return: (np.array) The generated bootstrapped correlation matrices. Has shape (n_samples, mat.shape[1], mat.shape[1]).
    """

    arr = np.asarray(mat)
    n_rows, n_cols = arr.shape
    if size is None:
        size = n_cols
    out = np.zeros((n_samples, size, size))
    for i in range(n_samples):
        rows = np.random.choice(n_rows, n_rows, replace=True)
        sample = arr[rows, :size]
        out[i] = np.corrcoef(sample, rowvar=False)
    return out


def block_bootstrap(mat, n_samples=1, size=None, block_size=None):
    """
    Uses the Block Bootstrap method to generate a new matrix of size equal to or smaller than the given matrix.

    It divides the original matrix into blocks of the given size. It samples with replacement random
    blocks to populate the bootstrapped matrix. It cannot generate a matrix greater than the original.

    It is inspired by the following paper:
    `Künsch, H.R., 1989. The jackknife and the bootstrap for general stationary observations.
    Annals of Statistics, 17(3), pp.1217-1241. <https://projecteuclid.org/euclid.aos/1176347265>`_.

    :param mat: (pd.DataFrame/np.array) Matrix to sample from.
    :param n_samples: (int) Number of matrices to generate.
    :param size: (tuple) Size of the bootstrapped matrix.
    :param block_size: (tuple) Size of the blocks.
    :return: (np.array) The generated bootstrapped matrices. Has shape (n_samples, size[0], size[1]).
    """

    arr = np.asarray(mat)
    n_rows, n_cols = arr.shape
    if size is None:
        size = (n_rows, n_cols)
    if block_size is None:
        block_size = (min(10, size[0]), min(10, size[1]))
    out = np.zeros((n_samples, size[0], size[1]))
    for i in range(n_samples):
        r = 0
        while r < size[0]:
            c = 0
            while c < size[1]:
                r0 = np.random.randint(0, n_rows - block_size[0] + 1)
                c0 = np.random.randint(0, n_cols - block_size[1] + 1)
                br = min(block_size[0], size[0] - r)
                bc = min(block_size[1], size[1] - c)
                out[i, r:r + br, c:c + bc] = arr[r0:r0 + br, c0:c0 + bc]
                c += bc
            r += br
    return out
