"""
 # @ Author: Chaorong Chen
 # @ Create Time: 2022-06-14 17:00:56
 # @ Modified by: Chaorong Chen
 # @ Description: MESA
 """

import sys
import time
from collections.abc import Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.feature_selection import GenericUnivariateSelect, VarianceThreshold, f_regression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from ._selection import (
    BorutaSelector,
    RedundancyPruner,
    get_support_indices,
    missing_value_processing,
    wilcoxon,
)
from ._task_utils import (
    CLASSIFICATION,
    REGRESSION,
    default_boruta_estimator,
    default_cv,
    default_meta_estimator,
    default_metric,
    default_predictor,
    default_selector,
    ensure_estimator_matches_task,
    score_metric,
    slice_rows,
    stacking_output,
    validate_task,
)


def disp_mesa(txt):
    """
    Display a timestamped message to stderr for MESA logging.

    Parameters
    ----------
    txt : str
        The message text to display.
    """
    print("@%s \t%s" % (time.asctime(), txt), file=sys.stderr)


class MESA_modality:
    """
    A comprehensive modality class for feature selection and prediction in MESA.

    This class implements a complete pipeline including missing value handling,
    variance filtering, univariate feature selection, Boruta feature selection,
    normalization, and task-aware prediction.
    """

    def __init__(
        self,
        task=CLASSIFICATION,
        random_state=0,
        boruta_estimator=None,
        top_n=100,
        variance_threshold=0,
        normalization=False,
        missing=0.1,
        redundancy_pruning=None,
        redundancy_threshold=0.95,
        redundancy_method="pearson",
        redundancy_estimator=None,
        redundancy_cv=3,
        redundancy_metric=None,
        predictor=None,
        classifier=None,
        selector=None,
        **kwargs,
    ):
        self.task = validate_task(task)
        self.random_state = random_state
        self.boruta_estimator = boruta_estimator
        self.top_n = top_n
        self.variance_threshold = variance_threshold
        self.normalization = normalization
        self.missing = missing
        self.redundancy_pruning = redundancy_pruning
        self.redundancy_threshold = redundancy_threshold
        self.redundancy_method = redundancy_method
        self.redundancy_estimator = redundancy_estimator
        self.redundancy_cv = redundancy_cv
        self.redundancy_metric = redundancy_metric
        self.predictor = predictor
        self.classifier = classifier
        self.selector = selector
        for key, value in kwargs.items():
            setattr(self, key, value)

    def fit(self, X, y):
        """Fit the complete preprocessing pipeline and predictor."""
        selector = self._resolved_selector()
        boruta_estimator = self._resolved_boruta_estimator()
        predictor = self._resolved_predictor()

        ensure_estimator_matches_task(boruta_estimator, self.task, "boruta_estimator")
        ensure_estimator_matches_task(
            self.redundancy_estimator, self.task, "redundancy_estimator"
        )
        ensure_estimator_matches_task(predictor, self.task, "predictor")

        pipeline_steps = [
            missing_value_processing(ratio=1 - self.missing),
            VarianceThreshold(self.variance_threshold),
            clone(selector),
            RedundancyPruner(
                mode=self.redundancy_pruning,
                threshold=self.redundancy_threshold,
                method=self.redundancy_method,
                task=self.task,
                estimator=self.redundancy_estimator,
                cv=self.redundancy_cv,
                metric=self.redundancy_metric,
            ),
            BorutaSelector(
                estimator=clone(boruta_estimator),
                random_state=self.random_state,
                verbose=0,
                n_estimators="auto",
                n=self.top_n,
            ),
        ]
        if self.normalization:
            pipeline_steps.insert(1, Normalizer())

        self.pipeline = make_pipeline(*pipeline_steps).fit(X, y)
        self.predictor_ = clone(predictor).fit(self.pipeline.transform(X), y)
        self.classifier_ = self.predictor_
        return self

    def transform(self, X):
        """Apply the preprocessing pipeline to data."""
        return self.pipeline.transform(X)

    def predict(self, X):
        """Predict values for preprocessed data."""
        return self.predictor_.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities for preprocessed data."""
        if self.task == REGRESSION:
            raise ValueError("predict_proba is only available when task='classification'.")
        return self.predictor_.predict_proba(X)

    def transform_predict(self, X):
        """Apply preprocessing pipeline and predict values."""
        return self.predictor_.predict(self.pipeline.transform(X))

    def transform_predict_proba(self, X):
        """Apply preprocessing pipeline and predict class probabilities."""
        if self.task == REGRESSION:
            raise ValueError(
                "transform_predict_proba is only available when task='classification'."
            )
        return self.predictor_.predict_proba(self.pipeline.transform(X))

    def get_support(self, step=None):
        """Get indices of features selected by pipeline components."""
        if step is not None:
            return get_support_indices(self.pipeline[step])

        support = np.arange(len(self.pipeline[0].get_support()))
        for pipeline_step in self.pipeline:
            if hasattr(pipeline_step, "get_support"):
                support = support[get_support_indices(pipeline_step)]
        return support

    def get_params(self, deep=True):
        """Get parameters of the MESA_modality instance."""
        return {
            "task": self.task,
            "random_state": self.random_state,
            "boruta_estimator": self.boruta_estimator,
            "top_n": self.top_n,
            "variance_threshold": self.variance_threshold,
            "normalization": self.normalization,
            "missing": self.missing,
            "redundancy_pruning": self.redundancy_pruning,
            "redundancy_threshold": self.redundancy_threshold,
            "redundancy_method": self.redundancy_method,
            "redundancy_estimator": self.redundancy_estimator,
            "redundancy_cv": self.redundancy_cv,
            "redundancy_metric": self.redundancy_metric,
            "predictor": self.predictor,
            "classifier": self.classifier,
            "selector": self.selector,
        }

    def _resolved_selector(self):
        if isinstance(self.selector, int):
            return GenericUnivariateSelect(
                score_func=wilcoxon if self.task == CLASSIFICATION else f_regression,
                mode="k_best",
                param=self.selector,
            )
        if self.selector is None:
            return default_selector(self.task, wilcoxon)
        return self.selector

    def _resolved_boruta_estimator(self):
        if self.boruta_estimator is None:
            return default_boruta_estimator(self.task, self.random_state)
        return self.boruta_estimator

    def _resolved_predictor(self):
        if self.predictor is not None:
            return self.predictor
        if self.classifier is not None:
            return self.classifier
        return default_predictor(self.task, self.random_state)


class MESA:
    """Multi-modality ensemble with stacking architecture for integrating multiple data modalities."""

    def __init__(
        self,
        modalities,
        task=CLASSIFICATION,
        meta_estimator=None,
        random_state=0,
        cv=None,
        **kwargs,
    ):
        self.task = validate_task(task)
        self.meta_estimator = (
            default_meta_estimator(self.task) if meta_estimator is None else meta_estimator
        )
        self.random_state = random_state
        self.modalities = modalities
        self.cv = default_cv(self.task, self.random_state, repeated=True) if cv is None else cv
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _base_fit(self, X, y, base_estimator):
        """Generate meta-features using cross-validation for a single modality."""

        def _internal_cv(train_index, test_index):
            X_train, X_test = X[train_index, :], X[test_index, :]
            fitted = clone(base_estimator).fit(X_train, np.array(y)[train_index])
            return stacking_output(fitted, X_test, self.task)

        return np.vstack(
            Parallel(n_jobs=-1, verbose=0)(
                delayed(_internal_cv)(train_index, test_index)
                for train_index, test_index in self.splits
            )
        )

    def fit(self, X_list, y):
        """Fit all modality estimators and the meta-estimator."""
        ensure_estimator_matches_task(self.meta_estimator, self.task, "meta_estimator")
        for modality in self.modalities:
            if getattr(modality, "task", self.task) != self.task:
                raise ValueError("All modalities should use the same task as MESA.")

        self.modalities = [clone(m).fit(X, y) for m, X in zip(self.modalities, X_list)]
        self.splits = [
            (train_index, test_index)
            for train_index, test_index in self.cv.split(X_list[0], y)
        ]
        y_stacking = np.hstack(
            [np.array(y)[test_index] for train_index, test_index in self.splits]
        )
        base_prediction = np.hstack(
            [self._base_fit(m.transform(X), y, m.predictor_) for m, X in zip(self.modalities, X_list)]
        )
        self.meta_estimator_ = clone(self.meta_estimator).fit(base_prediction, y_stacking)
        return self

    def predict(self, X_list_test):
        """Predict labels or continuous values using the ensemble."""
        base_prediction_test = np.hstack(
            [
                stacking_output(m.predictor_, m.transform(X), self.task)
                for m, X in zip(self.modalities, X_list_test)
            ]
        )
        return self.meta_estimator_.predict(base_prediction_test)

    def predict_proba(self, X_list_test):
        """Predict class probabilities using the fitted ensemble."""
        if self.task == REGRESSION:
            raise ValueError("predict_proba is only available when task='classification'.")
        base_prediction_test = np.hstack(
            [
                stacking_output(m.predictor_, m.transform(X), self.task)
                for m, X in zip(self.modalities, X_list_test)
            ]
        )
        return self.meta_estimator_.predict_proba(base_prediction_test)

    def get_support(self, step=None):
        """Get feature support information from all modalities."""
        if step is not None:
            return [get_support_indices(m.pipeline[step]) for m in self.modalities]
        return [m.get_support() for m in self.modalities]


class MESA_CV:
    """Cross-validation wrapper for MESA models with performance evaluation."""

    def __init__(
        self,
        modality,
        task=CLASSIFICATION,
        random_state=0,
        cv=None,
        performance_metric=None,
        **kwargs,
    ):
        self.task = validate_task(task)
        self.random_state = random_state
        self.cv = default_cv(self.task, self.random_state, repeated=False) if cv is None else cv
        self.modality = modality
        self.performance_metric = performance_metric
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
        mesa=False,
    ):
        """Perform a single iteration of cross-validation."""
        if mesa:
            X_train = [slice_rows(X_temp, train_index) for X_temp in X]
            X_test = [slice_rows(X_temp, test_index) for X_temp in X]
        else:
            X_train, X_test = slice_rows(X, train_index), slice_rows(X, test_index)

        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
        modality = clone(self.modality)
        fitted = modality.fit(X_train, y_train)

        if mesa:
            y_pred = fitted.predict_proba(X_test) if self.task == CLASSIFICATION and proba else fitted.predict(X_test)
        elif self.task == CLASSIFICATION and proba:
            y_pred = fitted.transform_predict_proba(X_test)
        else:
            y_pred = fitted.transform_predict(X_test)

        if return_feature_in:
            return y_pred, y_test, fitted.get_support()
        return y_pred, y_test

    def fit(self, X, y):
        """Perform cross-validation on the provided data."""
        if getattr(self.modality, "task", self.task) != self.task:
            raise ValueError("modality task should match MESA_CV task.")

        if (
            isinstance(X, Sequence)
            and not isinstance(X, str)
            and len(X) > 1
            and isinstance(self.modality, MESA)
        ):
            disp_mesa("Multiple modalities input")
            self.cv_result = Parallel(n_jobs=-1)(
                delayed(self._cv_iter)(X, y, train_index, test_index, mesa=True)
                for train_index, test_index in self.cv.split(X[0], y)
            )
        elif isinstance(X, (pd.DataFrame, np.ndarray)):
            disp_mesa("Single modality input")
            self.cv_result = Parallel(n_jobs=-1)(
                delayed(self._cv_iter)(X, y, train_index, test_index, mesa=False)
                for train_index, test_index in self.cv.split(X, y)
            )
        else:
            raise ValueError("X should be a list of modality matrices or a single modality matrix")
        return self

    def get_performance(self, metric=None):
        """Calculate the mean performance score across all cross-validation folds."""
        if self.cv_result is None:
            raise ValueError("You need to call the fit(X, y) method first.")

        from sklearn.model_selection import LeaveOneOut

        metric = metric or self.performance_metric or default_metric(self.task)
        if isinstance(self.cv, LeaveOneOut):
            y_pred = [self._fold_prediction(_[0])[0] for _ in self.cv_result]
            y_true = [_[1][0] for _ in self.cv_result]
            return score_metric(y_true, y_pred, self.task, metric)

        y_pred = [self._fold_prediction(_[0]) for _ in self.cv_result]
        y_true = [_[1] for _ in self.cv_result]
        return np.array(
            [score_metric(y_true[i], y_pred[i], self.task, metric) for i in range(len(y_true))]
        ).mean()

    def _fold_prediction(self, y_pred):
        if self.task == CLASSIFICATION:
            return np.asarray(y_pred)[:, 1]
        return np.asarray(y_pred)
