# Copyright 2019, Anonymous Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt

"""
Module which implements feature importance algorithms as described in Chapter 8 of Advances in Financial Machine
Learning and Clustered Feature Importance algorithms as described in Chapter 6 Section 6.5.2 of Machine Learning for
Asset Managers.

And feature importance algorithms multi-asset data set (stacked feature importance approach).
"""

# pylint: disable=invalid-name, invalid-unary-operand-type, comparison-with-callable
# pylint: disable=too-many-locals, too-many-branches, unsubscriptable-object

from typing import Callable
import pandas as pd
import numpy as np

from sklearn.metrics import log_loss
from sklearn.model_selection import BaseCrossValidator
import matplotlib.pyplot as plt

from mlfinlab.cross_validation.cross_validation import ml_cross_val_score, stacked_dataset_from_dict


def mean_decrease_impurity(model, feature_names, clustered_subsets=None):
    """
    Advances in Financial Machine Learning, Snippet 8.2, page 115.

    MDI Feature importance

    Mean decrease impurity (MDI) is a fast, explanatory-importance (in-sample, IS) method specific to tree-based
    classifiers, like RF. At each node of each decision tree, the selected feature splits the subset it received in
    such a way that impurity is decreased. Therefore, we can derive for each decision tree how much of the overall
    impurity decrease can be assigned to each feature. And given that we have a forest of trees, we can average those
    values across all estimators and rank the features accordingly.

    Tip:
    Masking effects take place when some features are systematically ignored by tree-based classifiers in favor of
    others. In order to avoid them, set max_features=int(1) when using sklearn’s RF class. In this way, only one random
    feature is considered per level.

    Notes:

    * MDI cannot be generalized to other non-tree based classifiers
    * The procedure is obviously in-sample.
    * Every feature will have some importance, even if they have no predictive power whatsoever.
    * MDI has the nice property that feature importances add up to 1, and every feature importance is bounded between 0 and 1.
    * method does not address substitution effects in the presence of correlated features. MDI dilutes the importance of
      substitute features, because of their interchangeability: The importance of two identical features will be halved,
      as they are randomly chosen with equal probability.
    * Sklearn’s RandomForest class implements MDI as the default feature importance score. This choice is likely
      motivated by the ability to compute MDI on the fly, with minimum computational cost.

    Clustered Feature Importance( Machine Learning for Asset Manager snippet 6.4 page 86) :
    Clustered MDI  is the  modified version of MDI (Mean Decreased Impurity). It  is robust to substitution effect that
    takes place when two or more explanatory variables share a substantial amount of information (predictive power).CFI
    algorithm described by Dr Marcos Lopez de Prado  in Clustered Feature  Importance section of book Machine Learning
    for Asset Manager. Here  instead of  taking the importance  of  every feature, we consider the importance of every
    feature subsets, thus every feature receive the importance of subset it belongs to.

    :param model: (object): Trained tree based classifier.
    :param feature_names: (list): Array of feature names.
    :param clustered_subsets: (list) Feature clusters for Clustered Feature Importance (CFI). Default None will not apply CFI.
        Structure of the input must be a list of list/s i.e. a list containing the clusters/subsets of feature
        name/s inside a list. E.g- [['I_0','I_1','R_0','R_1'],['N_1','N_2'],['R_3']]
    :return: (pd.DataFrame): Mean and standard deviation feature importance.
    """

    imp = np.array([est.feature_importances_ for est in model.estimators_])
    imp = pd.DataFrame(imp, columns=feature_names)
    if clustered_subsets is None:
        return pd.DataFrame({'mean': imp.mean(), 'std': imp.std()})
    cl_imp = {}
    for cluster in clustered_subsets:
        cl_imp[tuple(cluster)] = imp[cluster].sum(axis=1)
    cl_df = pd.DataFrame(cl_imp)
    return pd.DataFrame({'mean': cl_df.mean(), 'std': cl_df.std()})


def _mean_decrease_accuracy_round(model, X, y, cv_gen, clustered_subsets=None, sample_weight_train=None,
                                  sample_weight_score=None, scoring=log_loss, require_proba=True, random_state=42):
    """
    Implements one round of MDA Feature importance.
    Advances in Financial Machine Learning, Snippet 8.3, page 116-117.

    MDA Feature Importance

    Mean decrease accuracy (MDA) is a slow, predictive-importance (out-of-sample, OOS) method. First, it fits a
    classifier; second, it derives its performance OOS according to some performance score (accuracy, negative log-loss,
    etc.); third, it permutates each column of the features matrix (X), one column at a time, deriving the performance
    OOS after each column’s permutation. The importance of a feature is a function of the loss in performance caused by
    its column’s permutation. Some relevant considerations include:

    * This method can be applied to any classifier, not only tree-based classifiers.
    * MDA is not limited to accuracy as the sole performance score. For example, in the context of meta-labeling
      applications, we may prefer to score a classifier with F1 rather than accuracy. That is one reason a better
      descriptive name would have been “permutation importance.” When the scoring function does not correspond to a
      metric space, MDA results should be used as a ranking.
    * Like MDI, the procedure is also susceptible to substitution effects in the presence of correlated features.
      Given two identical features, MDA always considers one to be redundant to the other. Unfortunately, MDA will make
      both features appear to be outright irrelevant, even if they are critical.
    * Unlike MDI, it is possible that MDA concludes that all features are unimportant. That is because MDA is based on
      OOS performance.
    * The CV must be purged and embargoed.

    Clustered Feature Importance( Machine Learning for Asset Manager snippet 6.5 page 87) :
    Clustered MDA is the modified version of MDA (Mean Decreased Accuracy). It is robust to substitution effect that takes
    place when two or more explanatory variables share a substantial amount of information (predictive power).CFI algorithm
    described by Dr Marcos Lopez de Prado  in Clustered Feature  Importance (Presentation Slides)
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3517595. Instead of shuffling (permutating) all variables
    individually (like in MDA), we shuffle all variables in cluster together. Next, we follow all the  rest of the
    steps as in MDA. It can used by simply specifying the clustered_subsets argument.

    :param model: (sklearn.Classifier): Any sklearn classifier.
    :param X: (pd.DataFrame): Train set features.
    :param y: (pd.DataFrame, np.array): Train set labels.
    :param cv_gen: (cross_validation.PurgedKFold): Cross-validation object.
    :param clustered_subsets: (list) Feature clusters for Clustered Feature Importance (CFI). Default None will not apply CFI.
        Structure of the input must be a list of list/s i.e. a list containing the clusters/subsets of feature
        name/s inside a list. E.g- [['I_0','I_1','R_0','R_1'],['N_1','N_2'],['R_3']]
    :param sample_weight_train: (np.array) Sample weights used to train the model for each record in the dataset.
    :param sample_weight_score: (np.array) Sample weights used to evaluate the model quality.
    :param scoring: (function): Scoring function used to determine importance.
    :param require_proba: (bool): Boolean flag indicating that scoring function expects probabilities as input.
    :param random_state: (int) Random seed for shuffling the features.
    :return: (pd.DataFrame): Mean and standard deviation of feature importance.
    """

    np.random.seed(random_state)
    score = ml_cross_val_score(model, X, y, cv_gen, sample_weight_train, sample_weight_score, scoring, require_proba)
    imp = pd.Series(index=X.columns, dtype=float)
    for col in X.columns:
        X_ = X.copy()
        X_[col] = np.random.permutation(X_[col].values)
        score_ = ml_cross_val_score(model, X_, y, cv_gen, sample_weight_train, sample_weight_score, scoring, require_proba)
        imp[col] = score_.mean() - score.mean()
    return pd.DataFrame({'mean': imp, 'std': 0})


def mean_decrease_accuracy(model, X, y, cv_gen, clustered_subsets=None, sample_weight_train=None,
                           sample_weight_score=None, scoring=log_loss, require_proba=True, n_repeat=1):
    """
    Advances in Financial Machine Learning, Snippet 8.3, page 116-117.

    MDA Feature Importance (averaged over different random seeds `n_repeat` times)

    Mean decrease accuracy (MDA) is a slow, predictive-importance (out-of-sample, OOS) method. First, it fits a
    classifier; second, it derives its performance OOS according to some performance score (accuracy, negative log-loss,
    etc.); third, it permutates each column of the features matrix (X), one column at a time, deriving the performance
    OOS after each column’s permutation. The importance of a feature is a function of the loss in performance caused by
    its column’s permutation. Some relevant considerations include:

    * This method can be applied to any classifier, not only tree-based classifiers.
    * MDA is not limited to accuracy as the sole performance score. For example, in the context of meta-labeling
      applications, we may prefer to score a classifier with F1 rather than accuracy. That is one reason a better
      descriptive name would have been “permutation importance.” When the scoring function does not correspond to a
      metric space, MDA results should be used as a ranking.
    * Like MDI, the procedure is also susceptible to substitution effects in the presence of correlated features.
      Given two identical features, MDA always considers one to be redundant to the other. Unfortunately, MDA will make
      both features appear to be outright irrelevant, even if they are critical.
    * Unlike MDI, it is possible that MDA concludes that all features are unimportant. That is because MDA is based on
      OOS performance.
    * The CV must be purged and embargoed.

    Clustered Feature Importance( Machine Learning for Asset Manager snippet 6.5 page 87) :
    Clustered MDA is the modified version of MDA (Mean Decreased Accuracy). It is robust to substitution effect that takes
    place when two or more explanatory variables share a substantial amount of information (predictive power).CFI algorithm
    described by Dr Marcos Lopez de Prado  in Clustered Feature  Importance (Presentation Slides)
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3517595. Instead of shuffling (permutating) all variables
    individually (like in MDA), we shuffle all variables in cluster together. Next, we follow all the  rest of the
    steps as in MDA. It can used by simply specifying the clustered_subsets argument.

    :param model: (sklearn.Classifier): Any sklearn classifier.
    :param X: (pd.DataFrame): Train set features.
    :param y: (pd.DataFrame, np.array): Train set labels.
    :param cv_gen: (cross_validation.PurgedKFold): Cross-validation object.
    :param clustered_subsets: (list) Feature clusters for Clustered Feature Importance (CFI). Default None will not apply CFI.
        Structure of the input must be a list of list/s i.e. a list containing the clusters/subsets of feature
        name/s inside a list. E.g- [['I_0','I_1','R_0','R_1'],['N_1','N_2'],['R_3']]
    :param sample_weight_train: (np.array) Sample weights used to train the model for each record in the dataset.
    :param sample_weight_score: (np.array) Sample weights used to evaluate the model quality.
    :param scoring: (function): Scoring function used to determine importance.
    :param require_proba: (bool): Boolean flag indicating that scoring function expects probabilities as input.
    :param n_repeat: (int) Number of times to repeat MDA feature importance with different random seeds.
    :return: (pd.DataFrame): Mean and standard deviation of feature importance.
    """

    res = []
    for i in range(n_repeat):
        res.append(_mean_decrease_accuracy_round(model, X, y, cv_gen, clustered_subsets,
                                                  sample_weight_train, sample_weight_score,
                                                  scoring, require_proba, random_state=i))
    mean = pd.concat([r['mean'] for r in res], axis=1).mean(axis=1)
    std = pd.concat([r['mean'] for r in res], axis=1).std(axis=1)
    return pd.DataFrame({'mean': mean, 'std': std})


def single_feature_importance(clf, X, y, cv_gen, sample_weight_train=None, sample_weight_score=None, scoring=log_loss,
                              require_proba=True):
    """
    Advances in Financial Machine Learning, Snippet 8.4, page 118.

    Implementation of SFI

    Substitution effects can lead us to discard important features that happen to be redundant. This is not generally a
    problem in the context of prediction, but it could lead us to wrong conclusions when we are trying to understand,
    improve, or simplify a model. For this reason, the following single feature importance method can be a good
    complement to MDI and MDA.

    Single feature importance (SFI) is a cross-section predictive-importance (out-of- sample) method. It computes the
    OOS performance score of each feature in isolation. A few considerations:

    * This method can be applied to any classifier, not only tree-based classifiers.
    * SFI is not limited to accuracy as the sole performance score.
    * Unlike MDI and MDA, no substitution effects take place, since only one feature is taken into consideration at a time.
    * Like MDA, it can conclude that all features are unimportant, because performance is evaluated via OOS CV.

    The main limitation of SFI is that a classifier with two features can perform better than the bagging of two
    single-feature classifiers. For example, (1) feature B may be useful only in combination with feature A;
    or (2) feature B may be useful in explaining the splits from feature A, even if feature B alone is inaccurate.
    In other words, joint effects and hierarchical importance are lost in SFI. One alternative would be to compute the
    OOS performance score from subsets of features, but that calculation will become intractable as more features are
    considered.

    :param clf: (sklearn.Classifier): Any sklearn classifier.
    :param X: (pd.DataFrame): Train set features.
    :param y: (pd.DataFrame, np.array): Train set labels.
    :param cv_gen: (cross_validation.PurgedKFold): Cross-validation object.
    :param sample_weight_train: (np.array) Sample weights used to train the model for each record in the dataset.
    :param sample_weight_score: (np.array) Sample weights used to evaluate the model quality.
    :param scoring: (function): Scoring function used to determine importance.
    :param require_proba: (bool): Boolean flag indicating that scoring function expects probabilities as input.
    :return: (pd.DataFrame): Mean and standard deviation of feature importance.
    """

    imp = {}
    for col in X.columns:
        score = ml_cross_val_score(clf, X[[col]], y, cv_gen, sample_weight_train, sample_weight_score,
                                   scoring, require_proba)
        imp[col] = score.mean()
    return pd.DataFrame({'mean': pd.Series(imp), 'std': 0})


def plot_feature_importance(importance_df, oob_score, oos_score, save_fig=False, output_path=None):
    """
    Advances in Financial Machine Learning, Snippet 8.10, page 124.

    Feature importance plotting function.

    Plot feature importance.

    :param importance_df: (pd.DataFrame): Mean and standard deviation feature importance.
    :param oob_score: (float): Out-of-bag score.
    :param oos_score: (float): Out-of-sample (or cross-validation) score.
    :param save_fig: (bool): Boolean flag to save figure to a file.
    :param output_path: (str): If save_fig is True, path where figure should be saved.
    """

    fig, ax = plt.subplots()
    importance_df['mean'].sort_values().plot(kind='barh', xerr=importance_df['std'], ax=ax)
    ax.set_title(f"OOB: {oob_score:.3f}, OOS: {oos_score:.3f}")
    if save_fig and output_path:
        fig.savefig(output_path)
    return fig


def _stacked_mean_decrease_accuracy_round(model: object, X_dict: dict, y_dict: dict, cv_gen: BaseCrossValidator,
                                          clustered_subsets=None, sample_weight_train_dict: dict = None,
                                          sample_weight_score_dict=None,
                                          scoring: Callable[[np.array, np.array], float] = log_loss,
                                          require_proba: bool = True, random_state: int = 42):
    """
    Implements one round of Stacked MDA Feature importance.
    Advances in Financial Machine Learning, Snippet 8.3, page 116-117.

    MDA Feature Importance

    Mean decrease accuracy (MDA) is a slow, predictive-importance (out-of-sample, OOS) method. First, it fits a
    classifier; second, it derives its performance OOS according to some performance score (accuracy, negative log-loss,
    etc.); third, it permutates each column of the features matrix (X), one column at a time, deriving the performance
    OOS after each column’s permutation. The importance of a feature is a function of the loss in performance caused by
    its column’s permutation. Some relevant considerations include:

    * This method can be applied to any classifier, not only tree-based classifiers.
    * MDA is not limited to accuracy as the sole performance score. For example, in the context of meta-labeling
      applications, we may prefer to score a classifier with F1 rather than accuracy. That is one reason a better
      descriptive name would have been “permutation importance.” When the scoring function does not correspond to a
      metric space, MDA results should be used as a ranking.
    * Like MDI, the procedure is also susceptible to substitution effects in the presence of correlated features.
      Given two identical features, MDA always considers one to be redundant to the other. Unfortunately, MDA will make
      both features appear to be outright irrelevant, even if they are critical.
    * Unlike MDI, it is possible that MDA concludes that all features are unimportant. That is because MDA is based on
      OOS performance.
    * The CV must be purged and embargoed.

    Clustered Feature Importance( Machine Learning for Asset Manager snippet 6.5 page 87) :
    Clustered MDA is the modified version of MDA (Mean Decreased Accuracy). It is robust to substitution effect that takes
    place when two or more explanatory variables share a substantial amount of information (predictive power).CFI algorithm
    described by Dr Marcos Lopez de Prado  in Clustered Feature  Importance (Presentation Slides)
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3517595. Instead of shuffling (permutating) all variables
    individually (like in MDA), we shuffle all variables in cluster together. Next, we follow all the  rest of the
    steps as in MDA. It can used by simply specifying the clustered_subsets argument.

    :param model: (sklearn.Classifier): Any sklearn classifier.
    :param X_dict: (dict) Dictionary of asset : X_{asset}.
    :param y_dict: (dict) Dictionary of asset : y_{asset}
    :param cv_gen: (cross_validation.PurgedKFold): Cross-validation object.
    :param clustered_subsets: (list) Feature clusters for Clustered Feature Importance (CFI). Default None will not apply CFI.
                              Structure of the input must be a list of list/s i.e. a list containing the clusters/subsets of feature
                              name/s inside a list. E.g- [['I_0','I_1','R_0','R_1'],['N_1','N_2'],['R_3']]
    :param sample_weight_train_dict: (dict) Dictionary of asset: sample_weights_train_{asset}
    :param sample_weight_score_dict: (dict) Dictionary of asset: sample_weights_score_{asset}
    :param scoring: (function): Scoring function used to determine importance.
    :param require_proba: (bool): Boolean flag indicating that scoring function expects probabilities as input.
    :param random_state: (int) Random seed for shuffling the features.
    :return: (pd.DataFrame): Mean and standard deviation of feature importance.
    """

    np.random.seed(random_state)
    # Use provided CV generator on stacked dicts
    scores = []
    for train, test in cv_gen.split(X_dict, y_dict):
        X_train, y_train, w_train, _ = stacked_dataset_from_dict(X_dict, y_dict, sample_weight_train_dict,
                                                                 sample_weight_score_dict, train)
        X_test, y_test, _, w_score = stacked_dataset_from_dict(X_dict, y_dict, sample_weight_train_dict,
                                                               sample_weight_score_dict, test)
        clf = model
        clf.fit(X_train, y_train, sample_weight=w_train)
        base_pred = clf.predict_proba(X_test) if require_proba else clf.predict(X_test)
        base_score = scoring(y_test, base_pred, sample_weight=w_score)
        imp = pd.Series(index=X_train.columns, dtype=float)
        for col in X_train.columns:
            X_test_perm = X_test.copy()
            X_test_perm[col] = np.random.permutation(X_test_perm[col].values)
            pred = clf.predict_proba(X_test_perm) if require_proba else clf.predict(X_test_perm)
            imp[col] = scoring(y_test, pred, sample_weight=w_score) - base_score
        scores.append(imp)
    imp_df = pd.concat(scores, axis=1)
    return pd.DataFrame({'mean': imp_df.mean(axis=1), 'std': imp_df.std(axis=1)})


def stacked_mean_decrease_accuracy(model: object, X_dict: dict, y_dict: dict, cv_gen: BaseCrossValidator,
                                   clustered_subsets=None, sample_weight_train_dict: dict = None,
                                   sample_weight_score_dict=None,
                                   scoring: Callable[[np.array, np.array], float] = log_loss,
                                   require_proba: bool = True, n_repeat: int = 1):
    """
    Advances in Financial Machine Learning, Snippet 8.3, page 116-117.

    Stacked MDA Feature Importance for multi-asset dataset (averaged over different random seeds `n_repeat` times)

    Mean decrease accuracy (MDA) is a slow, predictive-importance (out-of-sample, OOS) method. First, it fits a
    classifier; second, it derives its performance OOS according to some performance score (accuracy, negative log-loss,
    etc.); third, it permutates each column of the features matrix (X), one column at a time, deriving the performance
    OOS after each column’s permutation. The importance of a feature is a function of the loss in performance caused by
    its column’s permutation. Some relevant considerations include:

    * This method can be applied to any classifier, not only tree-based classifiers.
    * MDA is not limited to accuracy as the sole performance score. For example, in the context of meta-labeling
      applications, we may prefer to score a classifier with F1 rather than accuracy. That is one reason a better
      descriptive name would have been “permutation importance.” When the scoring function does not correspond to a
      metric space, MDA results should be used as a ranking.
    * Like MDI, the procedure is also susceptible to substitution effects in the presence of correlated features.
      Given two identical features, MDA always considers one to be redundant to the other. Unfortunately, MDA will make
      both features appear to be outright irrelevant, even if they are critical.
    * Unlike MDI, it is possible that MDA concludes that all features are unimportant. That is because MDA is based on
      OOS performance.
    * The CV must be purged and embargoed.

    Clustered Feature Importance( Machine Learning for Asset Manager snippet 6.5 page 87) :
    Clustered MDA is the modified version of MDA (Mean Decreased Accuracy). It is robust to substitution effect that takes
    place when two or more explanatory variables share a substantial amount of information (predictive power).CFI algorithm
    described by Dr Marcos Lopez de Prado  in Clustered Feature  Importance (Presentation Slides)
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3517595. Instead of shuffling (permutating) all variables
    individually (like in MDA), we shuffle all variables in cluster together. Next, we follow all the  rest of the
    steps as in MDA. It can used by simply specifying the clustered_subsets argument.

    :param model: (sklearn.Classifier): Any sklearn classifier.
    :param X_dict: (dict) Dictionary of asset : X_{asset}.
    :param y_dict: (dict) Dictionary of asset : y_{asset}
    :param cv_gen: (cross_validation.StackedPurgedKFold): Cross-validation object.
    :param clustered_subsets: (list) Feature clusters for Clustered Feature Importance (CFI). Default None will not apply CFI.
                              Structure of the input must be a list of list/s i.e. a list containing the clusters/subsets of feature
                              name/s inside a list. E.g- [['I_0','I_1','R_0','R_1'],['N_1','N_2'],['R_3']]
    :param sample_weight_train_dict: (dict) Dictionary of asset: sample_weights_train_{asset}
    :param sample_weight_score_dict: (dict) Dictionary of asset: sample_weights_score_{asset}
    :param scoring: (function): Scoring function used to determine importance.
    :param require_proba: (bool): Boolean flag indicating that scoring function expects probabilities as input.
    :param n_repeat: (int) Number of times to repeat MDA feature importance with different random seeds.
    :return: (pd.DataFrame): Mean and standard deviation of feature importance.
    """

    res = []
    for i in range(n_repeat):
        res.append(_stacked_mean_decrease_accuracy_round(model, X_dict, y_dict, cv_gen, clustered_subsets,
                                                         sample_weight_train_dict, sample_weight_score_dict,
                                                         scoring, require_proba, random_state=i))
    mean = pd.concat([r['mean'] for r in res], axis=1).mean(axis=1)
    std = pd.concat([r['mean'] for r in res], axis=1).std(axis=1)
    return pd.DataFrame({'mean': mean, 'std': std})
