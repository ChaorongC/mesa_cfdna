"""
 # @ Author: Chaorong Chen
 # @ Create Time: 2022-06-14 17:00:56
 # @ Modified by: Chaorong Chen
 # @ Modified time: 2025-05-23 01:38:13
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
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from boruta import BorutaPy
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import GenericUnivariateSelect, VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from collections.abc import Sequence
from sklearn.linear_model import LogisticRegression

def disp_mesa(txt):
    """
    Display a timestamped message to stderr for MESA logging.
    
    Parameters
    ----------
    txt : str
        The message text to display.
    """
    print("@%s \t%s" % (time.asctime(), txt), file=sys.stderr)


def wilcoxon(X, y):
    """
    Score function for feature selection using Wilcoxon rank-sum test.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix where each row is a sample and each column is a feature.
    y : array-like of shape (n_samples,)
        Binary target labels (0 or 1).

    Returns
    -------
    ndarray of shape (n_features,)
        Negative p-values from Wilcoxon rank-sum test for each feature.
        Lower values indicate more significant differences between classes.
    """
    return -mannwhitneyu(X[y == 0], X[y == 1])[1]


class BorutaSelector(BorutaPy):
    """
    Feature selector that extends BorutaPy to select the top n features based on ranking.
    
    This class wraps the Boruta feature selection algorithm and provides functionality
    to select a specific number of top-ranked features rather than all confirmed features.
    
    Parameters
    ----------
    n : int, default=10
        The number of top features to select based on Boruta ranking.
    **kwargs : dict
        Additional keyword arguments passed to the BorutaPy constructor.
        
    Attributes
    ----------
    n : int
        Number of features to select.
    indices : ndarray
        Indices of the selected features after fitting.
        
    Methods
    -------
    fit(X, y)
        Fit the Boruta algorithm and identify top n features.
    transform(X)
        Transform data to contain only the selected features.
    get_support()
        Get indices of selected features.
        
    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> selector = BorutaSelector(n=5, estimator=RandomForestClassifier())
    >>> selector.fit(X_train, y_train)
    >>> X_selected = selector.transform(X_test)
    """

    def __init__(self, n=10, **kwargs):
        super().__init__(**kwargs)
        self.n = n

    def fit(self, X, y):
        """
        Fit the Boruta feature selection algorithm and select top n features.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        super().fit(X, y)
        self.indices = np.argsort(self.ranking_)[: self.n]
        return self

    def transform(self, X):
        """
        Transform data to contain only the selected top n features.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.
            
        Returns
        -------
        X_transformed : array-like of shape (n_samples, n)
            Transformed data with only selected features.
            
        Raises
        ------
        ValueError
            If fit has not been called yet.
        """
        try:
            self.ranking_
        except AttributeError:
            raise ValueError("You need to call the fit(X, y) method first.")
        try:
            return X.iloc[:, self.indices]
        except:
            return X[:, self.indices]

    def get_support(self):
        """
        Get indices of the selected features.
        
        Returns
        -------
        indices : ndarray
            Indices of the selected top n features.
        """
        return self.indices


class missing_value_processing:
    """
    Preprocessing class for handling missing values with ratio-based filtering.
    
    This class filters out features with too many missing values and imputes
    the remaining missing values using a specified imputation strategy.
    
    Parameters
    ----------
    ratio : float, default=0.9
        Minimum ratio of non-missing values required to keep a feature.
        Features with less than this ratio of valid values are removed.
    imputer : sklearn imputer, default=SimpleImputer(strategy="mean")
        Imputation strategy for remaining missing values.
        
    Attributes
    ----------
    ratio : float
        The minimum valid value ratio threshold.
    imputer : sklearn imputer
        The fitted imputer for missing value replacement.
    indices : ndarray
        Indices of features that pass the missing value threshold.
        
    Methods
    -------
    fit(X, y=None)
        Identify features meeting the valid ratio threshold and fit imputer.
    transform(X)
        Transform data by removing high-missing features and imputing values.
    get_support()
        Get indices of features that passed the missing value filter.
    """

    def __init__(self, ratio=0.9, imputer=SimpleImputer(strategy="mean")):
        self.ratio = ratio
        self.imputer = imputer

    def fit(self, X, y=None):
        """
        Fit the missing value processor by identifying valid features and fitting imputer.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), optional
            Target values (ignored, present for API consistency).
            
        Returns
        -------
        self : object
            Returns the instance itself.
            
        Raises
        ------
        ValueError
            If ratio is not greater than 0.
        """
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
        """
        Transform data by removing high-missing features and imputing remaining values.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.
            
        Returns
        -------
        X_transformed : pandas.DataFrame
            Transformed data with missing values handled.
            
        Raises
        ------
        ValueError
            If ratio is not greater than 0.
        """
        if self.ratio > 0:
            return pd.DataFrame(
                self.imputer.transform(pd.DataFrame(X).iloc[:, self.indices]),
                index=X.index,
                columns=X.columns[self.indices],
            )
        else:
            raise ValueError("The ratio of valid values should be greater than 0.")

    def get_support(self):
        """
        Get indices of features that passed the missing value filter.
        
        Returns
        -------
        indices : ndarray
            Indices of features with sufficient non-missing values.
        """
        return self.indices


class MESA_modality:
    """
    A comprehensive modality class for feature selection and classification in MESA.
    
    This class implements a complete pipeline including missing value handling,
    variance filtering, univariate feature selection, Boruta feature selection,
    normalization, and classification.

    Parameters
    ----------
    random_state : int, default=0
        Random seed for reproducibility across all components.
    boruta_estimator : estimator object, default=RandomForestClassifier(random_state=0, n_jobs=-1)
        The base estimator used for Boruta feature selection.
    top_n : int, default=100
        Number of top features to select using Boruta algorithm.
    variance_threshold : float, default=0
        Threshold for variance-based feature filtering. Features with variance
        below this threshold are removed.
    normalization : bool, default=False
        Whether to apply L2 normalization to the data.
    missing : float, default=0.1
        Maximum proportion of missing values allowed per feature.
        Features exceeding this are removed.
    classifier : estimator object, default=RandomForestClassifier(random_state=0, n_jobs=-1)
        The final classifier for prediction.
    selector : int or selector object, default=GenericUnivariateSelect(score_func=wilcoxon, mode="k_best", param=2000)
        Univariate feature selector. If int, creates GenericUnivariateSelect with that many features.
    **kwargs : dict
        Additional parameters to set as attributes.

    Attributes
    ----------
    pipeline : sklearn.pipeline.Pipeline
        The fitted preprocessing pipeline.
    classifier : estimator object
        The fitted final classifier.

    Methods
    -------
    fit(X, y)
        Fit the complete pipeline and classifier to training data.
    transform(X)
        Apply only the preprocessing pipeline to data.
    predict(X)
        Predict class labels for preprocessed data.
    predict_proba(X)
        Predict class probabilities for preprocessed data.
    transform_predict(X)
        Apply preprocessing pipeline and predict class labels.
    transform_predict_proba(X)
        Apply preprocessing pipeline and predict class probabilities.
    get_support(step=None)
        Get indices of features selected by pipeline components.
    get_params(deep=True)
        Get parameters of the MESA_modality instance.
        
    Examples
    --------
    >>> modality = MESA_modality(top_n=50, missing=0.2)
    >>> modality.fit(X_train, y_train)
    >>> predictions = modality.transform_predict_proba(X_test)
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
        if isinstance(selector, int):
            self.selector = GenericUnivariateSelect(
                score_func=wilcoxon, mode="k_best", param=selector
            )
        else:
            self.selector = selector
        for key, value in kwargs.items():
            setattr(self, key, value)

    def fit(self, X, y):
        """
        Fit the complete preprocessing pipeline and classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Training target values.
            
        Returns
        -------
        self : object
            Returns the fitted instance.
        """
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
        """
        Apply the preprocessing pipeline to data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.
            
        Returns
        -------
        X_transformed : array-like
            Preprocessed data.
        """
        return self.pipeline.transform(X)

    def predict(self, X):
        """
        Predict class labels for preprocessed data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Preprocessed feature matrix.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        return self.classifier.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities for preprocessed data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Preprocessed feature matrix.
            
        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        return self.classifier.predict_proba(X)

    def transform_predict(self, X):
        """
        Apply preprocessing pipeline and predict class labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Raw feature matrix.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        return self.classifier.predict(self.pipeline.transform(X))

    def transform_predict_proba(self, X):
        """
        Apply preprocessing pipeline and predict class probabilities.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Raw feature matrix.
            
        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        return self.classifier.predict_proba(self.pipeline.transform(X))

    def get_support(self, step=None):
        """
        Get indices of features selected by pipeline components.
        
        Parameters
        ----------
        step : int, optional
            Specific pipeline step to get support from. If None, returns
            final selected feature indices.
            
        Returns
        -------
        support : ndarray
            Indices of selected features.
        """
        if step == None:
            return self.pipeline[0].get_support()[
                self.pipeline[-2].get_support(indices=True)[self.pipeline[-1].indices]
            ]
        else:
            return self.pipeline[step].get_support(indices=True)

    def get_params(self, deep=True):
        """
        Get parameters of the MESA_modality instance.
        
        Parameters
        ----------
        deep : bool, default=True
            If True, return parameters for this estimator and contained objects.
            
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
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
    Multi-modality Ensemble with Stacking Architecture for integrating multiple data modalities.
    
    MESA implements a stacking ensemble approach where multiple modalities (data types)
    are processed independently by base modality estimators, and their outputs are
    combined using a meta-estimator for final prediction.

    Parameters
    ----------
    modalities : list of estimators
        List of base modality estimators, each responsible for processing one data modality.
    meta_estimator : estimator object, default=LogisticRegression()
        The meta-estimator used for combining outputs from base modalities.
    random_state : int, default=0
        Random seed for reproducibility.
    cv : cross-validation generator, default=RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
        Cross-validation strategy for generating meta-features.
    **kwargs : dict
        Additional parameters to set as attributes.

    Attributes
    ----------
    modalities : list
        List of fitted base modality estimators.
    meta_estimator : estimator
        Fitted meta-estimator for final predictions.
    splits : list
        Cross-validation train/test index pairs used for meta-feature generation.

    Methods
    -------
    fit(X_list, y)
        Fit all modality estimators and the meta-estimator.
    predict(X_list_test)
        Predict class labels using the ensemble.
    predict_proba(X_list_test)
        Predict class probabilities using the ensemble.
    get_support(step=None)
        Get feature support information from all modalities.
        
    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> modalities = [MESA_modality(), MESA_modality()]
    >>> meta_est = LogisticRegression()
    >>> mesa = MESA(modalities, meta_est)
    >>> mesa.fit([X1_train, X2_train], y_train)
    >>> predictions = mesa.predict([X1_test, X2_test])
    """

    def __init__(
        self,
        modalities,
        meta_estimator=LogisticRegression(),
        random_state=0,
        cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0),
        **kwargs
    ):
        self.meta_estimator = meta_estimator
        self.random_state = random_state
        self.modalities = modalities
        self.cv = cv
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _base_fit(self, X, y, base_estimator):
        """
        Generate meta-features using cross-validation for a single modality.
        
        Parameters
        ----------
        X : array-like
            Transformed feature matrix for one modality.
        y : array-like
            Target values.
        base_estimator : estimator
            Base estimator for this modality.
            
        Returns
        -------
        base_probability : ndarray
            Cross-validated probability predictions for meta-learning.
        """
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

    def fit(self, X_list, y):
        """
        Fit all modality estimators and the meta-estimator.
        
        Parameters
        ----------
        X_list : list of array-like
            List of feature matrices, one for each modality.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : object
            Returns the fitted instance.
        """
        # add check parameters
        self.modalities = [clone(m).fit(X, y) for m, X in zip(self.modalities, X_list)]
        self.splits = [
            (train_index, test_index)
            for train_index, test_index in self.cv.split(X_list[0], y)
        ]
        y_stacking = np.hstack(
            [np.array(y)[test_index] for train_index, test_index in self.splits]
        )
        base_probability = np.hstack(
            [
                self._base_fit(m.transform(X), y, clone(m.classifier))
                for m, X in zip(self.modalities, X_list)
            ]
        )
        self.meta_estimator.fit(base_probability, y_stacking)
        return self

    def predict(self, X_list_test):
        """
        Predict class labels using the fitted ensemble.
        
        Parameters
        ----------
        X_list_test : list of array-like
            List of test feature matrices, one for each modality.
            
        Returns
        -------
        y_pred : ndarray
            Predicted class labels.
        """
        base_probability_test = np.hstack(
            [m.transform_predict_proba(X) for m, X in zip(self.modalities, X_list_test)]
        )
        return self.meta_estimator.predict(base_probability_test)

    def predict_proba(self, X_list_test):
        """
        Predict class probabilities using the fitted ensemble.
        
        Parameters
        ----------
        X_list_test : list of array-like
            List of test feature matrices, one for each modality.
            
        Returns
        -------
        y_proba : ndarray
            Predicted class probabilities.
        """
        base_probability_test = np.hstack(
            [m.transform_predict_proba(X) for m, X in zip(self.modalities, X_list_test)]
        )
        return self.meta_estimator.predict_proba(base_probability_test)

    def get_support(self, step=None):
        """
        Get feature support information from all modalities.
        
        Parameters
        ----------
        step : int, optional
            Specific pipeline step to get support from. If None, returns
            final selected features from all modalities.
            
        Returns
        -------
        support : list
            List of feature support arrays, one for each modality.
        """
        if step == None:
            return [
                m.pipeline[0].get_support()[
                    m.pipeline[-2].get_support(indices=True)[m.pipeline[-1].indices]
                ]
                for m in self.modalities
            ]
        else:
            return [m.pipeline[step].get_support(indices=True) for m in self.modalities]


class MESA_CV:
    """
    Cross-validation wrapper for MESA models with performance evaluation.
    
    This class provides cross-validation functionality for both single modality
    and multi-modality MESA models, with built-in performance metrics calculation.

    Parameters
    ----------
    modality : estimator object
        The modality estimator to evaluate. Can be MESA_modality for single modality
        or MESA for multi-modality evaluation.
    random_state : int, default=0
        Random seed for reproducibility.
    cv : cross-validation generator, default=StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        Cross-validation splitting strategy.
    **kwargs : dict
        Additional parameters to set as attributes.

    Attributes
    ----------
    cv_result : list
        Results from cross-validation iterations, stored as (y_pred, y_true) tuples.
    modality : estimator
        The modality estimator being evaluated.

    Methods
    -------
    fit(X, y)
        Perform cross-validation on the provided data.
    get_performance()
        Calculate and return the mean ROC AUC score across CV folds.
        
    Examples
    --------
    >>> modality = MESA_modality()
    >>> cv_eval = MESA_CV(modality, cv=StratifiedKFold(n_splits=10))
    >>> cv_eval.fit(X, y)
    >>> auc_score = cv_eval.get_performance()
    """

    def __init__(
        self,
        modality,
        random_state=0,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
        **kwargs
    ):
        self.random_state = random_state
        self.cv = cv
        self.modality = modality
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _cv_iter(
        self,
        X,
        y,
        train_index,
        test_index,
        proba=True,
        return_feature_in=False,
        mesa=False
    ):
        """
        Perform a single iteration of cross-validation.
        
        Parameters
        ----------
        X : array-like or list of array-like
            Feature data. Single array for single modality, list for multi-modality.
        y : array-like
            Target values.
        train_index : array-like
            Indices for training samples.
        test_index : array-like
            Indices for test samples.
        proba : bool, default=True
            Whether to return probabilities (True) or class predictions (False).
        return_feature_in : bool, default=False
            Whether to return feature selection information.
        mesa : bool, default=False
            Whether this is multi-modality (True) or single modality (False).
            
        Returns
        -------
        y_pred : ndarray
            Predictions for test samples.
        y_test : ndarray
            True labels for test samples.
        support : ndarray, optional
            Feature selection support (only if return_feature_in=True).
        """
        if mesa:
            X_train = []
            X_test = []
            for X_temp in X:
                X_train.append(X_temp.iloc[train_index, :])
                X_test.append(X_temp.iloc[test_index, :])
        else:
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]

        modality = clone(self.modality)
        if proba:
            y_pred = modality.fit(X_train, y_train).transform_predict_proba(X_test)
        else:
            y_pred = modality.fit(X_train, y_train).transform_predict(X_test)
        
        if return_feature_in:
            return y_pred, y_test, modality.get_support()
        else:
            return y_pred, y_test

    def fit(self, X, y):
        """
        Perform cross-validation on the provided data.
        
        Parameters
        ----------
        X : array-like or list of array-like
            Feature data. Can be:
            - Single DataFrame/array for single modality evaluation
            - List of DataFrames/arrays for multi-modality evaluation
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : object
            Returns the fitted instance with cv_result attribute populated.
            
        Raises
        ------
        ValueError
            If X format doesn't match the modality type.
        """
        if (
            isinstance(X, Sequence) 
            and not isinstance(X, str) 
            and len(X) > 1
        ) and (isinstance(self.modality, MESA)):  # multiple modalities
            disp_mesa("Multiple modalities input")
            self.cv_result = Parallel(n_jobs=-1)(
                delayed(self._cv_iter)(
                    X,
                    y,
                    train_index,
                    test_index,
                    mesa=True
                )
                for train_index, test_index in self.cv.split(X[0], y)
            )
        elif isinstance(X, (pd.DataFrame, np.ndarray)):  # single modality
            disp_mesa("Single modality input")
            self.cv_result = Parallel(n_jobs=-1)(
                delayed(self._cv_iter)(
                    X,
                    y,
                    train_index,
                    test_index,
                    mesa=False
                )
                for train_index, test_index in self.cv.split(X, y)
            )
        else:
            raise ValueError(
                "X should be a list of modality matrices or a single modality matrix"
            )
        return self

    def get_performance(self):
        """
        Calculate the mean ROC AUC score across all cross-validation folds.
        
        Returns
        -------
        performance : float
            Mean ROC AUC score across all CV folds.
            
        Raises
        ------
        ValueError
            If fit() has not been called yet.
            
        Notes
        -----
        For LeaveOneOut CV, returns single AUC score. For other CV strategies,
        returns the mean AUC across all folds.
        """
        if self.cv_result == None:
            raise ValueError("You need to call the fit(X, y) method first.")
        
        # Import LeaveOneOut here to avoid circular imports
        from sklearn.model_selection import LeaveOneOut
        
        if isinstance(self.cv, LeaveOneOut):
            y_pred = [_[0][:, 1][0] for _ in self.cv_result]
            y_true = [_[1][0] for _ in self.cv_result]
            return roc_auc_score(y_true, y_pred)
        else:
            y_pred = [_[0][:, 1] for _ in self.cv_result]
            y_true = [_[1] for _ in self.cv_result]
            return np.array(
                [roc_auc_score(y_true[_], y_pred[_]) for _ in range(len(y_true))]
            ).mean()
