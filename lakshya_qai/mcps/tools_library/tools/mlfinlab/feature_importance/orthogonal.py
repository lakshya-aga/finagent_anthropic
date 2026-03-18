"""
Module which implements feature PCA compression and PCA analysis of feature importance.
"""

import pandas as pd
import numpy as np
from scipy.stats import weightedtau, kendalltau, spearmanr, pearsonr


def _get_eigen_vector(dot_matrix, variance_thresh, num_features=None):
    """
    Advances in Financial Machine Learning, Snippet 8.5, page 119.

    Computation of Orthogonal Features

    Gets eigen values and eigen vector from matrix which explain % variance_thresh of total variance.

    :param dot_matrix: (np.array): Matrix for which eigen values/vectors should be computed.
    :param variance_thresh: (float): Percentage % of overall variance which compressed vectors should explain.
    :param num_features: (int) Manually set number of features, overrides variance_thresh. (None by default)
    :return: (pd.Series, pd.DataFrame): Eigenvalues, Eigenvectors.
    """

    eigen_vals, eigen_vecs = np.linalg.eigh(dot_matrix)
    idx = eigen_vals.argsort()[::-1]
    eigen_vals, eigen_vecs = eigen_vals[idx], eigen_vecs[:, idx]
    if num_features is None:
        cum = np.cumsum(eigen_vals) / eigen_vals.sum()
        num_features = np.searchsorted(cum, variance_thresh) + 1
    eigen_vals = pd.Series(eigen_vals[:num_features])
    eigen_vecs = pd.DataFrame(eigen_vecs[:, :num_features], index=dot_matrix.index)
    return eigen_vals, eigen_vecs


def _standardize_df(data_frame):
    """
    Helper function which divides df by std and extracts mean.

    :param data_frame: (pd.DataFrame): Dataframe to standardize
    :return: (pd.DataFrame): Standardized dataframe
    """

    return (data_frame - data_frame.mean()) / data_frame.std()


def get_orthogonal_features(feature_df, variance_thresh=.95, num_features=None):
    """
    Advances in Financial Machine Learning, Snippet 8.5, page 119.

    Computation of Orthogonal Features.

    Gets PCA orthogonal features.

    :param feature_df: (pd.DataFrame): Dataframe of features.
    :param variance_thresh: (float): Percentage % of overall variance which compressed vectors should explain.
    :param num_features: (int) Manually set number of features, overrides variance_thresh. (None by default)
    :return: (pd.DataFrame): Compressed PCA features which explain %variance_thresh of variance.
    """

    std = _standardize_df(feature_df)
    dot = pd.DataFrame(np.dot(std.T, std), index=feature_df.columns, columns=feature_df.columns)
    _, evec = _get_eigen_vector(dot, variance_thresh, num_features)
    return pd.DataFrame(std.values @ evec.values, index=feature_df.index)


def get_pca_rank_weighted_kendall_tau(feature_imp, pca_rank):
    """
    Advances in Financial Machine Learning, Snippet 8.6, page 121.

    Computes Weighted Kendall's Tau Between Feature Importance and Inverse PCA Ranking.

    :param feature_imp: (np.array): Feature mean importance.
    :param pca_rank: (np.array): PCA based feature importance rank.
    :return: (float): Weighted Kendall Tau of feature importance and inverse PCA rank with p_value.
    """

    return weightedtau(feature_imp, -pca_rank)


def feature_pca_analysis(feature_df, feature_importance, variance_thresh=0.95):
    """
    Performs correlation analysis between feature importance (MDI for example, supervised) and PCA eigenvalues
    (unsupervised).

    High correlation means that probably the pattern identified by the ML algorithm is not entirely overfit.

    :param feature_df: (pd.DataFrame): Features dataframe.
    :param feature_importance: (pd.DataFrame): Individual MDI feature importance.
    :param variance_thresh: (float): Percentage % of overall variance which compressed vectors should explain in PCA compression.
    :return: (dict): Dictionary with kendall, spearman, pearson and weighted_kendall correlations and p_values.
    """

    std = _standardize_df(feature_df)
    dot = pd.DataFrame(np.dot(std.T, std), index=feature_df.columns, columns=feature_df.columns)
    e_val, _ = _get_eigen_vector(dot, variance_thresh)
    pca_rank = np.arange(len(e_val))
    imp = feature_importance.mean(axis=1).values
    return {
        "kendall": kendalltau(imp, -pca_rank),
        "spearman": spearmanr(imp, -pca_rank),
        "pearson": pearsonr(imp, -pca_rank),
        "weighted_kendall": get_pca_rank_weighted_kendall_tau(imp, pca_rank)
    }
