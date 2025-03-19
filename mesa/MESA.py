"""
 # @ Author: Chaorong Chen
 # @ Create Time: 2022-06-14 17:00:56
 # @ Modified by: Chaorong Chen
 # @ Modified time: 2024-12-19 16:26:11
 # @ Description: MESA
 """

import sys
import time
import pandas as pd
from sklearn.base import clone
from joblib import Parallel, delayed
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from boruta import BorutaPy
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import GenericUnivariateSelect, VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from collections.abc import Sequence


def disp_mesa(txt):
    print("@%s \t%s" % (time.asctime(), txt), file=sys.stderr)


def wilcoxon(X, y):
    """
    Score function for feature selection using Wilcoxon rank-sum test.

    Args:
        X: dataframe or array of shape (n_features, n_samples)
        y: array-like of shape (n_samples,)

    Returns:
        p-values of Wilcoxon rank-sum test for each feature
    """
    return -mannwhitneyu(X[y == 0], X[y == 1])[1]


class BorutaSelector(BorutaPy):
    """
    BorutaSelector is a feature selection class that extends BorutaPy to select the top n features based on their ranking.
    Parameters
    ----------
    n : int, optional (default=10)
        The number of top features to select.
    **kwargs :
        Additional keyword arguments to pass to the BorutaPy constructor.
    Methods
    -------
    fit(X, y)
        Fits the Boruta feature selection algorithm on the provided data.
    transform(X)
        Transforms the data to contain only the selected top n features.
    get_support()
        Returns the indices of the selected top n features.
    """

    def __init__(self, n=10, **kwargs):
        super().__init__(**kwargs)
        self.n = n

    def fit(self, X, y):
        super().fit(X, y)
        self.indices = np.argsort(self.ranking_)[: self.n]
        return self

    def transform(self, X):
        try:
            self.ranking_
        except AttributeError:
            raise ValueError("You need to call the fit(X, y) method first.")
        try:
            return X.iloc[:, self.indices]
        except:
            return X[:, self.indices]

    def get_support(self):
        return self.indices


class missing_value_processing:
    def __init__(self, ratio=0.9, imputer=SimpleImputer(strategy="mean")):
        self.ratio = ratio
        self.imputer = imputer

    def fit(self, X, y=None):
        if self.ratio > 0:
            self.indices = np.where(
                pd.DataFrame(X).count(axis="rows") >= X.shape[0] * self.ratio
            )[0]
            self.imputer = clone(self.imputer).fit(
                pd.DataFrame(X).iloc[:, self.indices]
            )
            return self
        else:
            raise ValueError("The ratio of valid values should be greater than 0.")

    def transform(self, X):
        if self.ratio > 0:
            return pd.DataFrame(
                self.imputer.transform(pd.DataFrame(X).iloc[:, self.indices]),
                index=X.index,
                columns=X.columns[self.indices],
            )
        else:
            raise ValueError("The ratio of valid values should be greater than 0.")

    def get_support(self):
        return self.indices


class MESA_modality:
    """
    A class used to represent the MESA modality.

    Attributes
    ----------
    random_state : int
        Random seed for reproducibility.
    boruta_estimator : estimator object
        The estimator used for the Boruta feature selection.
    top_n : int
        Number of top features to select using Boruta.
    variance_threshold : float
        Threshold for variance threshold feature selection.
    normalization : bool
        Whether to apply normalization to the data.
    missing : float
        Threshold for missing values.
    classifier : estimator object
        The classifier used for prediction.
    selector : selector object
        The selector used for univariate feature selection.

    Methods
    -------
    fit(X, y)
        Fits the pipeline and classifier to the data.
    transform(X)
        Transforms the data using the fitted pipeline.
    predict(X)
        Predicts the class labels for the input data.
    predict_proba(X)
        Predicts class probabilities for the input data.
    transform_predict(X)
        Transforms the data and then predicts the class labels.
    transform_predict_proba(X)
        Transforms the data and then predicts class probabilities.
    get_support(step=None)
        Gets the indices of the selected features.
    get_params(deep=True)
        Gets the parameters of the MESA_modality instance.
    """

    def __init__(
        self,
        random_state=0,
        boruta_estimator=RandomForestClassifier(random_state=0, n_jobs=-1),
        top_n=100,
        variance_threshold=0,
        normalization=False,
        missing=0.1,
        classifier=RandomForestClassifier(random_state=0, n_jobs=-1),
        selector=GenericUnivariateSelect(
            score_func=wilcoxon, mode="k_best", param=2000
        ),
        **kwargs
    ):
        self.random_state = random_state
        self.boruta_estimator = boruta_estimator
        self.top_n = top_n
        self.variance_threshold = variance_threshold
        self.normalization = normalization
        self.missing = missing
        self.classifier = classifier
        self.selector = selector
        for key, value in kwargs.items():
            setattr(self, key, value)
        pass

    def fit(self, X, y):
        pipeline_steps = [
            missing_value_processing(ratio=1 - self.missing),
            VarianceThreshold(self.variance_threshold),
            self.selector,
            BorutaSelector(
                estimator=self.boruta_estimator,
                random_state=self.random_state,
                verbose=0,
                n_estimators="auto",
                n=self.top_n,
            ),
        ]
        if self.normalization:
            pipeline_steps.insert(1, Normalizer())
        self.pipeline = make_pipeline(*pipeline_steps).fit(X, y)
        self.classifier = self.classifier.fit(self.pipeline.transform(X), y)
        return self

    def transform(self, X):
        return self.pipeline.transform(X)

    def predict(self, X):
        return self.classifier.predict(X)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    def transform_predict(self, X):
        return self.classifier.predict(self.pipeline.transform(X))

    def transform_predict_proba(self, X):
        return self.classifier.predict_proba(self.pipeline.transform(X))

    def get_support(self, step=None):
        if step == None:
            return self.pipeline[0].get_support()[
                self.pipeline[-2].get_support(indices=True)[self.pipeline[-1].indices]
            ]
        else:
            return self.pipeline[step].get_support(indices=True)

    def get_params(self, deep=True):
        return {
            "random_state": self.random_state,
            "boruta_estimator": self.boruta_estimator,
            "top_n": self.top_n,
            "variance_threshold": self.variance_threshold,
            "normalization": self.normalization,
            "missing": self.missing,
            "classifier": self.classifier,
            "selector": self.selector,
        }


class MESA:
    """

    Parameters
    ----------
    meta_estimator : estimator object
        The meta-estimator to be used for stacking the base estimators.
    random_state : int, default=0
        The seed used by the random number generator.
    cv : cross-validation generator, default=StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        The cross-validation splitting strategy.
    **kwargs : additional keyword arguments
        Additional parameters to set as attributes of the class.

    Methods
    -------
    fit(modalities, X_list, y)
        Fit the model to the training data.
    predict(X_list_test)
        Predict the class labels for the provided data.
    predict_proba(X_list_test)
        Predict class probabilities for the provided data.

    Attributes
    ----------
    meta_estimator : estimator object
        The meta-estimator used for stacking.
    random_state : int
        The seed used by the random number generator.
    cv : cross-validation generator
        The cross-validation splitting strategy.
    modalities : list
        List of modalities (base estimators).
    base_estimators : list
        List of fitted base estimators.
    splits : list
        List of train-test indices for cross-validation.
    """

    def __init__(
        self,
        meta_estimator,
        random_state=0,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
        **kwargs
    ):
        self.meta_estimator = meta_estimator
        self.random_state = random_state
        self.cv = cv
        for key, value in kwargs.items():
            setattr(self, key, value)
        pass

    def _internal_cv(self, X, y, base_estimator, train_index, test_index):
        X_train, X_test = X[train_index, :], X[test_index, :]
        return base_estimator.fit(X_train, np.array(y)[train_index]).predict_proba(
            X_test
        )

    def _base_fit(self, X, y, base_estimator):
        def _internal_cv(train_index, test_index):
            X_train, X_test = X[train_index, :], X[test_index, :]
            return (
                clone(base_estimator)
                .fit(X_train, np.array(y)[train_index])
                .predict_proba(X_test)
            )

        base_probability = np.vstack(
            Parallel(n_jobs=-1, verbose=0)(
                delayed(_internal_cv)(train_index, test_index)
                for train_index, test_index in self.splits
            )
        )
        return base_probability

    def fit(self, modalities, X_list, y):
        # add check parameters
        self.modalities = modalities
        self.splits = [
            (train_index, test_index)
            for train_index, test_index in self.cv.split(X_list[0], y)
        ]
        y_stacking = np.hstack(
            [np.array(y)[test_index] for train_index, test_index in self.splits]
        )
        base_probability = np.hstack(
            [
                self._base_fit(m.transform(X), y, clone(m.classifier))  ########
                for m, X in zip(modalities, X_list)
            ]
        )
        # self.base_estimators = [m.classifier for m in modalities]
        self.meta_estimator.fit(base_probability, y_stacking)
        return self

    def predict(self, X_list_test):
        base_probability_test = np.hstack(
            [m.transform_predict_proba(X) for m, X in zip(self.modalities, X_list_test)]
        )
        return self.meta_estimator.predict(base_probability_test)

    def predict_proba(self, X_list_test):
        base_probability_test = np.hstack(
            [m.transform_predict_proba(X) for m, X in zip(self.modalities, X_list_test)]
        )
        return self.meta_estimator.predict_proba(base_probability_test)


# Code for missing value imputation and dataset splitting


def cv_preprocessing(X, train_index, test_index, ratio=1, normalization=False):
    """
    Parameters
    ----------
    X : dataframe of shape (n_features, n_samples)
        Input samples.
    train_index : list/array/tuple of
        The training set indices for the LOO split.
    test_index : list/array/tuple of
        The testing set indices for the LOO split.
    ratio : float, default = 1
        The threshold for feature filtering. Only features have valid values for > (ratio*samples) are kept and then imputed.
    normalization: boolean, default = False
        If scale dataset witt normalizer during preprocessing
    Returns
    ----------
    X_train_cleaned : dataframe of shape (n_train_samples, n_features)
        Cleaned, missing-value-imputed training set.
    X_test_cleaned :dataframe of shape (n_test_samples, n_features)
        Cleaned, missing-value-imputed testing datasets.
    """
    X_temp = X
    X_train_temp, X_test_temp = X_temp.iloc[train_index, :], X_temp.iloc[test_index, :]
    X_train_seleted = np.where(
        X_train_temp.count(axis="rows") >= X_train_temp.shape[0] * ratio
    )[0]
    imputer = SimpleImputer(strategy="mean")
    if normalization:
        scaler = Normalizer()
        X_train_cleaned = pd.DataFrame(
            scaler.fit_transform(
                imputer.fit_transform(X_train_temp.iloc[:, X_train_seleted].values)
            )
        )
        X_test_cleaned = pd.DataFrame(
            scaler.transform(
                imputer.transform(X_test_temp.iloc[:, X_train_seleted].values)
            )
        )
    else:
        X_train_cleaned = pd.DataFrame(
            imputer.fit_transform(X_train_temp.iloc[:, X_train_seleted].values)
        )
        X_test_cleaned = pd.DataFrame(
            imputer.transform(X_test_temp.iloc[:, X_train_seleted].values)
        )
    X_train_cleaned.index, X_test_cleaned.index = (
        X_temp.index[train_index],
        X_temp.index[test_index],
    )  # put Sample ID back
    X_train_cleaned.columns, X_test_cleaned.columns = (
        X.columns[X_train_seleted],
        X.columns[X_train_seleted],
    )
    return X_train_cleaned, X_test_cleaned


class MESA_CV:
    """
    A class used to perform cross-validation for the MESA model.

    Attributes
    ----------
    random_state : int
        Random seed for reproducibility.
    cv : StratifiedKFold
        Cross-validation splitting strategy.
    selector : GenericUnivariateSelect
        Feature selection method.
    boruta_est : RandomForestClassifier
        Estimator used for Boruta feature selection.
    classifier : RandomForestClassifier
        Classifier used for training.
    variance_threshold : float
        Threshold for variance-based feature selection.
    top_n : int
        Number of top features to select.
    kwargs : dict
        Additional keyword arguments.

    Methods
    -------
    _cv_iter(X, y, train_index, test_index, missing_ratio, normalization, variance_threshold, proba=True)
        Perform a single iteration of cross-validation for a single modality.
    _cv_iter_mesa(X, y, train_index, test_index, missing_ratio, normalization, variance_threshold, proba=True)
        Perform a single iteration of cross-validation for multiple modalities.
    fit(X, y)
        Fit the model using cross-validation.
    get_performance()
        Calculate the performance of the model using ROC AUC score.
    """

    def __init__(
        self,
        random_state=0,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
        selector=GenericUnivariateSelect(
            score_func=wilcoxon, mode="k_best", param=2000
        ),
        boruta_est=RandomForestClassifier(random_state=0, n_jobs=-1),
        classifier=RandomForestClassifier(random_state=0, n_jobs=-1),
        normalization=False,
        variance_threshold=0,
        top_n=100,
        missing=0.1,
        **kwargs  # meta_estimator=RandomForestClassifier(random_state=0, n_jobs=-1),
    ):
        # self.meta_estimator = meta_estimator
        self.random_state = random_state
        self.cv = cv
        self.seletor = selector
        self.top_n = top_n
        self.kwargs = kwargs
        self.boruta_est = boruta_est
        self.classifier = classifier
        self.missing = missing
        self.normalization = normalization
        self.variance_threshold = (
            variance_threshold  # todo: consider situation when have multiple modalities
        )
        for key, value in kwargs.items():
            setattr(self, key, value)
        pass

    def _cv_iter(
        self,
        X,
        y,
        train_index,
        test_index,
        missing_ratio,
        normalization,
        variance_threshold,
        selector,
        proba=True,
    ):
        X_train, X_test = cv_preprocessing(
            X, train_index, test_index, 1 - missing_ratio, normalization
        )
        # X_train, X_test = X_train.values, X_test.values
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]

        modality = MESA_modality(
            selector=selector,
            random_state=self.random_state,
            top_n=self.top_n,
            missing=0,
            classifier=self.classifier,
            boruta_estimator=self.boruta_est,
            normalization=False,
            variance_threshold=variance_threshold,
        )
        if proba:
            y_pred = modality.fit(X_train, y_train).transform_predict_proba(X_test)
        else:
            y_pred = modality.fit(X_train, y_train).transform_predict(X_test)
        return y_pred, y_test, modality.get_support()

    def _cv_iter_mesa(
        self,
        X,
        y,
        train_index,
        test_index,
        missing_ratio,
        normalization,
        variance_threshold,
        selector,
        proba=True,
    ):
        temp = [
            cv_preprocessing(
                X_, train_index, test_index, 1 - missing_ratio, normalization
            )
            for X_ in X
        ]
        X_train = [_[0] for _ in temp]
        X_test = [_[1] for _ in temp]
        del temp
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]

        modalities = [
            MESA_modality(
                selector=clone(selector),
                random_state=self.random_state,
                top_n=self.top_n,
                missing=0,
                classifier=clone(self.classifier),
                boruta_estimator=self.boruta_est,
                normalization=False,
                variance_threshold=variance_threshold,
            ).fit(X_train_, y_train)
            for X_train_ in X_train
        ]
        mesa = MESA(
            meta_estimator=self.meta_estimator, random_state=self.random_state
        ).fit(
            modalities, X_train, y_train
        )  # ValueError: X has 95986 features, but GenericUnivariateSelect is expecting 25545 features as input.

        if proba:
            y_pred = mesa.predict_proba(X_test)
        else:
            y_pred = mesa.predict(X_test)
        return y_pred, y_test

    def fit(self, X, y):
        slctr = clone(self.seletor)
        if (
            isinstance(X, Sequence) and not isinstance(X, str) and len(X) > 1
        ):  # multiple modalities
            disp_mesa("Mutiple modalities input")
            self.cv_result = Parallel(n_jobs=-1)(
                delayed(self._cv_iter_mesa)(
                    X,
                    y,
                    train_index,
                    test_index,
                    self.missing,
                    self.normalization,
                    self.variance_threshold,
                    slctr,
                )
                for train_index, test_index in self.cv.split(
                    X[0], y
                )  # check if all X_ is have the same sample index
            )
        elif isinstance(X, (pd.DataFrame, np.ndarray)):  # single modality
            disp_mesa("Single modality input")
            self.cv_result = Parallel(n_jobs=-1)(
                delayed(self._cv_iter)(
                    X,
                    y,
                    train_index,
                    test_index,
                    self.missing,
                    self.normalization,
                    self.variance_threshold,
                    slctr,
                )
                for train_index, test_index in self.cv.split(X, y)
            )
        else:
            raise ValueError(
                "X should be a list of modality matrixs or a single modality matrix"
            )
        return self

    def get_performance(self):
        y_pred = [_[0][:, 1] for _ in self.cv_result]
        y_true = [_[1] for _ in self.cv_result]
        return np.array(
            [roc_auc_score(y_true[_], y_pred[_]) for _ in range(len(y_true))]
        ).mean()
