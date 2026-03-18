"""
Implements the following classes from Chapter 12 of AFML:

- Combinatorial Purged Cross-Validation class.
- Stacked Combinatorial Purged Cross-Validation class.
"""
# pylint: disable=too-many-locals, arguments-differ, invalid-name, unused-argument

from itertools import combinations
from typing import List

import pandas as pd
import numpy as np
from scipy.special import comb
from sklearn.model_selection import KFold

from mlfinlab.cross_validation.cross_validation import ml_get_train_times


def _get_number_of_backtest_paths(n_train_splits: int, n_test_splits: int) -> int:
    """
    Number of combinatorial paths for CPCV(N,K).

    :param n_train_splits: (int) Number of train splits.
    :param n_test_splits: (int) Number of test splits.
    :return: (int) Number of backtest paths for CPCV(N,k).
    """

    return int(comb(n_train_splits, n_test_splits, exact=True))


class CombinatorialPurgedKFold(KFold):
    """
    Advances in Financial Machine Learning, Chapter 12.

    Implements Combinatorial Purged Cross Validation (CPCV).

    The train is purged of observations overlapping test-label intervals.
    Test set is assumed contiguous (shuffle=False), w/o training samples in between.
    """

    def __init__(self,
                 n_splits: int = 3,
                 n_test_splits: int = 2,
                 samples_info_sets: pd.Series = None,
                 pct_embargo: float = 0.):
        """
        Initialize.

        :param n_splits: (int) The number of splits. Default to 3
        :param samples_info_sets: (pd.Series) The information range on which each record is constructed from
            *samples_info_sets.index*: Time when the information extraction started.
            *samples_info_sets.value*: Time when the information extraction ended.
        :param pct_embargo: (float) Percent that determines the embargo size.
        """

        super().__init__(n_splits=n_splits, shuffle=False)
        self.n_test_splits = n_test_splits
        self.samples_info_sets = samples_info_sets
        self.pct_embargo = pct_embargo
        self.backtest_paths = [[] for _ in range(_get_number_of_backtest_paths(n_splits, n_test_splits))]

    def _generate_combinatorial_test_ranges(self, splits_indices: dict) -> List:
        """
        Using start and end indices of test splits from KFolds and number of test_splits (self.n_test_splits),
        generates combinatorial test ranges splits.

        :param splits_indices: (dict) Test fold integer index: [start test index, end test index].
        :return: (list) Combinatorial test splits ([start index, end index]).
        """

        splits = list(splits_indices.items())
        combs = combinations(splits, self.n_test_splits)
        return [[v for _, v in comb] for comb in combs]

    def _fill_backtest_paths(self, train_indices: list, test_splits: list):
        """
        Using start and end indices of test splits and purged/embargoed train indices from CPCV, find backtest path and
        place in the path where these indices should be used.

        :param test_splits: (list) List of lists with first element corresponding to test start index and second - test end.
        """

        for path in self.backtest_paths:
            if len(path) == 0:
                path.append((train_indices, test_splits))
                return
        self.backtest_paths.append([(train_indices, test_splits)])

    def split(self,
              X: pd.DataFrame,
              y: pd.Series = None,
              groups=None) -> tuple:
        """
        The main method to call for the PurgedKFold class.

        :param X: (pd.DataFrame) Samples dataset that is to be split.
        :param y: (pd.Series) Sample labels series.
        :param groups: (array-like), with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        :return: (tuple) [train list of sample indices, and test list of sample indices].
        """

        if self.samples_info_sets is None:
            raise ValueError("samples_info_sets is required")
        indices = np.arange(X.shape[0])
        embargo = int(len(indices) * self.pct_embargo)
        splits = list(super().split(X, y, groups))
        splits_idx = {i: [test_idx[0], test_idx[-1]] for i, (_, test_idx) in enumerate(splits)}
        test_ranges = self._generate_combinatorial_test_ranges(splits_idx)
        for test_splits in test_ranges:
            test_idx = np.concatenate([indices[t0:t1 + 1] for t0, t1 in test_splits])
            test_times = self.samples_info_sets.iloc[test_idx]
            train_times = ml_get_train_times(self.samples_info_sets, test_times)
            train_idx = np.where(self.samples_info_sets.index.isin(train_times.index))[0]
            if embargo > 0:
                for _, t1 in test_splits:
                    embargo_idx = indices[t1 + 1:t1 + embargo + 1]
                    train_idx = np.setdiff1d(train_idx, embargo_idx)
            self._fill_backtest_paths(train_idx, test_splits)
            yield train_idx, test_idx


class StackedCombinatorialPurgedKFold(KFold):
    """
    Advances in Financial Machine Learning, Chapter 12.

    Implements Stacked Combinatorial Purged Cross Validation (CPCV). It implements CPCV for multiasset dataset.

    The train is purged of observations overlapping test-label intervals.
    Test set is assumed contiguous (shuffle=False), w/o training samples in between.
    """

    def __init__(self,
                 n_splits: int = 3,
                 n_test_splits: int = 2,
                 samples_info_sets_dict: dict = None,
                 pct_embargo: float = 0.):
        """
        Initialize.

        :param n_splits: (int) The number of splits. Default to 3
        :param samples_info_sets_dict: (dict) Dictionary of samples info sets.
                                        ASSET_1: SAMPLE_INFO_SETS, ASSET_2:...

            *samples_info_sets.index*: Time when the information extraction started.
            *samples_info_sets.value*: Time when the information extraction ended.
        :param pct_embargo: (float) Percent that determines the embargo size.
        """

        super().__init__(n_splits=n_splits, shuffle=False)
        self.n_test_splits = n_test_splits
        self.samples_info_sets_dict = samples_info_sets_dict
        self.pct_embargo = pct_embargo
        self.backtest_paths = {}

    def _fill_backtest_paths(self, asset, train_indices: list, test_splits: list):
        """
        Using start and end indices of test splits and purged/embargoed train indices from CPCV, find backtest path and
        place in the path where these indices should be used.

        :param asset: (str) Asset for which backtest paths are filled.
        :param train_indices: (list) List of lists with first element corresponding to train start index, second - test end.
        :param test_splits: (list) List of lists with first element corresponding to test start index and second - test end.
        """

        if asset not in self.backtest_paths:
            self.backtest_paths[asset] = [[] for _ in range(_get_number_of_backtest_paths(self.n_splits, self.n_test_splits))]
        for path in self.backtest_paths[asset]:
            if len(path) == 0:
                path.append((train_indices, test_splits))
                return
        self.backtest_paths[asset].append([(train_indices, test_splits)])

    def _generate_combinatorial_test_ranges(self, splits_indices: dict) -> List:
        """
        Using start and end indices of test splits from KFolds and number of test_splits (self.n_test_splits),
        generates combinatorial test ranges splits.

        :param splits_indices: (dict) Test fold integer index: [start test index, end test index].
        :return: (list) Combinatorial test splits ([start index, end index]).
        """

        splits = list(splits_indices.items())
        combs = combinations(splits, self.n_test_splits)
        return [[v for _, v in comb] for comb in combs]

    def split(self,
              X_dict: dict,
              y_dict: dict = None,
              groups=None) -> tuple:
        """
        The main method to call for the PurgedKFold class.

        :param X_dict: (dict) Dictionary of asset : X_{asset}.
        :param y_dict: (dict) Dictionary of asset : y_{asset}.
        :param groups: (array-like), with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        :return: (tuple) [train list of sample indices, and test list of sample indices].
        """

        splits_out = {}
        for asset, X in X_dict.items():
            cv = CombinatorialPurgedKFold(n_splits=self.n_splits, n_test_splits=self.n_test_splits,
                                          samples_info_sets=self.samples_info_sets_dict[asset],
                                          pct_embargo=self.pct_embargo)
            splits_out[asset] = list(cv.split(X, None))
            self.backtest_paths[asset] = cv.backtest_paths
        return splits_out
