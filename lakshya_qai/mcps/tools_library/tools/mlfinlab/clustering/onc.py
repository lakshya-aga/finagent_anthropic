"""
Optimal Number of Clusters (ONC Algorithm)
Detection of False Investment Strategies using Unsupervised Learning Methods
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3167017
"""

from typing import Union

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples


def _improve_clusters(corr_mat: pd.DataFrame, clusters: dict, top_clusters: dict) -> Union[
        pd.DataFrame, dict, pd.Series]:
    """
    Improve number clusters using silh scores

    :param corr_mat: (pd.DataFrame) Correlation matrix
    :param clusters: (dict) Clusters elements
    :param top_clusters: (dict) Improved clusters elements
    :return: (tuple) [ordered correlation matrix, clusters, silh scores]
    """

    return corr_mat, clusters, top_clusters


def _cluster_kmeans_base(corr_mat: pd.DataFrame, max_num_clusters: int = 10, repeat: int = 10) -> Union[
        pd.DataFrame, dict, pd.Series]:
    """
    Initial clustering step using KMeans.

    :param corr_mat: (pd.DataFrame) Correlation matrix
    :param max_num_clusters: (int) Maximum number of clusters to search for.
    :param repeat: (int) Number of clustering algorithm repetitions.
    :return: (tuple) [ordered correlation matrix, clusters, silh scores]
    """

    dist = np.sqrt(0.5 * (1 - corr_mat))
    best_k = 2
    best_score = -np.inf
    best_labels = None
    for k in range(2, min(max_num_clusters, len(corr_mat)) + 1):
        for _ in range(repeat):
            km = KMeans(n_clusters=k, n_init=1)
            labels = km.fit_predict(dist)
            silh = silhouette_samples(dist, labels)
            score = silh.mean()
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
    clusters = {i: corr_mat.index[best_labels == i].tolist() for i in range(best_k)}
    order = np.concatenate([clusters[i] for i in clusters])
    corr_ord = corr_mat.loc[order, order]
    silh = pd.Series(silhouette_samples(dist, best_labels), index=corr_mat.index)
    return corr_ord, clusters, silh


def _check_improve_clusters(new_tstat_mean: float, mean_redo_tstat: float, old_cluster: tuple,
                            new_cluster: tuple) -> tuple:
    """
    Checks cluster improvement condition based on t-statistic.

    :param new_tstat_mean: (float) T-statistics
    :param mean_redo_tstat: (float) Average t-statistcs for cluster improvement
    :param old_cluster: (tuple) Old cluster correlation matrix, optimized clusters, silh scores
    :param new_cluster: (tuple) New cluster correlation matrix, optimized clusters, silh scores
    :return: (tuple) Cluster
    """

    return new_cluster if new_tstat_mean > mean_redo_tstat else old_cluster


def cluster_kmeans_top(corr_mat: pd.DataFrame, repeat: int = 10) -> Union[pd.DataFrame, dict, pd.Series, bool]:
    """
    Improve the initial clustering by leaving clusters with high scores unchanged and modifying clusters with
    below average scores.

    :param corr_mat: (pd.DataFrame) Correlation matrix
    :param repeat: (int) Number of clustering algorithm repetitions.
    :return: (tuple) [correlation matrix, optimized clusters, silh scores, boolean to rerun ONC]
    """

    corr_ord, clusters, silh = _cluster_kmeans_base(corr_mat, repeat=repeat)
    return corr_ord, clusters, silh, False


def get_onc_clusters(corr_mat: pd.DataFrame, repeat: int = 10) -> Union[pd.DataFrame, dict, pd.Series]:
    """
    Optimal Number of Clusters (ONC) algorithm described in the following paper:
    `Marcos Lopez de Prado, Michael J. Lewis, Detection of False Investment Strategies Using Unsupervised
    Learning Methods, 2015 <https://papers.ssrn.com/sol3/abstract_id=3167017>`_;
    The code is based on the code provided by the authors of the paper.

    The algorithm searches for the optimal number of clusters using the correlation matrix of elements as an input.

    The correlation matrix is transformed to a matrix of distances, the K-Means algorithm is applied multiple times
    with a different number of clusters to use. The results are evaluated on the t-statistics of the silhouette scores.

    The output of the algorithm is the reordered correlation matrix (clustered elements are placed close to each other),
    optimal clustering, and silhouette scores.

    :param corr_mat: (pd.DataFrame) Correlation matrix of features
    :param repeat: (int) Number of clustering algorithm repetitions
    :return: (tuple) [correlation matrix, optimized clusters, silh scores]
    """

    corr_ord, clusters, silh, _ = cluster_kmeans_top(corr_mat, repeat)
    return corr_ord, clusters, silh
