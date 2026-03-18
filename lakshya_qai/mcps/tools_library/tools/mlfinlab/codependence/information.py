"""
Implementations of mutual information (I) and variation of information (VI) codependence measures from Cornell
lecture slides: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994&download=yes
"""
import numpy as np
import scipy.stats as ss
from sklearn.metrics import mutual_info_score


# pylint: disable=invalid-name

def get_optimal_number_of_bins(num_obs: int, corr_coef: float = None) -> int:
    """
    Calculates optimal number of bins for discretization based on number of observations
    and correlation coefficient (univariate case).

    Algorithms used in this function were originally proposed in the works of Hacine-Gharbi et al. (2012)
    and Hacine-Gharbi and Ravier (2018). They are described in the Cornell lecture notes:
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994&download=yes (p.26)

    :param num_obs: (int) Number of observations.
    :param corr_coef: (float) Correlation coefficient, used to estimate the number of bins for univariate case.
    :return: (int) Optimal number of bins.
    """
    if corr_coef is None:
        return int(np.round(np.sqrt(num_obs)))
    corr_coef = min(max(corr_coef, -0.999999), 0.999999)
    return int(np.round(np.sqrt(num_obs / (1 - corr_coef ** 2))))


def get_mutual_info(x: np.array, y: np.array, n_bins: int = None, normalize: bool = False,
                    estimator: str = 'standard') -> float:
    """
    Returns mutual information (MI) between two vectors.

    This function uses the discretization with the optimal bins algorithm proposed in the works of
    Hacine-Gharbi et al. (2012) and Hacine-Gharbi and Ravier (2018).

    Read Cornell lecture notes for more information about the mutual information:
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994&download=yes.

    This function supports multiple ways the mutual information can be estimated:

    1. ``standard`` - the standard way of estimation - binning observations according to a given
       number of bins and applying the MI formula.
    2. ``standard_copula`` - estimating the copula (as a normalized ranking of the observations) and
       applying the standard mutual information estimator on it.
    3. ``copula_entropy`` - estimating the copula (as a normalized ranking of the observations) and
       calculating its entropy. Then MI estimator = (-1) * copula entropy.

    The last two estimators' implementation is taken from the blog post by Dr. Gautier Marti.
    Read this blog post for more information about the differences in the estimators:
    https://gmarti.gitlab.io/qfin/2020/07/01/mutual-information-is-copula-entropy.html

    :param x: (np.array) X vector.
    :param y: (np.array) Y vector.
    :param n_bins: (int) Number of bins for discretization, if None the optimal number will be calculated.
                         (None by default)
    :param normalize: (bool) Flag used to normalize the result to [0, 1]. (False by default)
    :param estimator: (str) Estimator to be used for calculation. [``standard``, ``standard_copula``, ``copula_entropy``]
                            (``standard`` by default)
    :return: (float) Mutual information score.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if n_bins is None:
        n_bins = get_optimal_number_of_bins(len(x), np.corrcoef(x, y)[0, 1])
    if estimator in ('standard_copula', 'copula_entropy'):
        x = ss.rankdata(x) / (len(x) + 1.0)
        y = ss.rankdata(y) / (len(y) + 1.0)
    x_bin = np.floor(x * n_bins).astype(int) if estimator != 'standard' else np.digitize(x, np.histogram_bin_edges(x, n_bins)) - 1
    y_bin = np.floor(y * n_bins).astype(int) if estimator != 'standard' else np.digitize(y, np.histogram_bin_edges(y, n_bins)) - 1
    mi = mutual_info_score(x_bin, y_bin)
    if estimator == 'copula_entropy':
        mi = -mi
    if normalize:
        hx = mutual_info_score(x_bin, x_bin)
        hy = mutual_info_score(y_bin, y_bin)
        mi = mi / np.sqrt(hx * hy) if hx * hy > 0 else 0.0
    return mi


def variation_of_information_score(x: np.array, y: np.array, n_bins: int = None, normalize: bool = False) -> float:
    """
    Returns variantion of information (VI) between two vectors.

    This function uses the discretization using optimal bins algorithm proposed in the works of
    Hacine-Gharbi et al. (2012) and Hacine-Gharbi and Ravier (2018).

    Read Cornell lecture notes for more information about the variation of information:
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994&download=yes.

    :param x: (np.array) X vector.
    :param y: (np.array) Y vector.
    :param n_bins: (int) Number of bins for discretization, if None the optimal number will be calculated.
                         (None by default)
    :param normalize: (bool) True to normalize the result to [0, 1]. (False by default)
    :return: (float) Variation of information score.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if n_bins is None:
        n_bins = get_optimal_number_of_bins(len(x), np.corrcoef(x, y)[0, 1])
    x_bin = np.digitize(x, np.histogram_bin_edges(x, n_bins)) - 1
    y_bin = np.digitize(y, np.histogram_bin_edges(y, n_bins)) - 1
    mi = mutual_info_score(x_bin, y_bin)
    hx = mutual_info_score(x_bin, x_bin)
    hy = mutual_info_score(y_bin, y_bin)
    vi = hx + hy - 2 * mi
    if normalize:
        denom = hx + hy
        vi = vi / denom if denom > 0 else 0.0
    return vi
