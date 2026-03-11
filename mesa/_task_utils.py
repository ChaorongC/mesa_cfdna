import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.base import is_classifier, is_regressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import GenericUnivariateSelect, f_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import (
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)

CLASSIFICATION = "classification"
REGRESSION = "regression"
SUPPORTED_TASKS = {CLASSIFICATION, REGRESSION}
CLASSIFICATION_METRICS = {"roc_auc"}
REGRESSION_METRICS = {
    "r2",
    "neg_mean_squared_error",
    "neg_root_mean_squared_error",
    "pearson",
    "spearman",
}


def validate_task(task):
    if task not in SUPPORTED_TASKS:
        raise ValueError("task should be one of: 'classification', 'regression'.")
    return task


def default_selector(task, classification_score_func):
    if task == CLASSIFICATION:
        return GenericUnivariateSelect(
            score_func=classification_score_func,
            mode="k_best",
            param=2000,
        )
    return GenericUnivariateSelect(score_func=f_regression, mode="k_best", param=2000)


def default_boruta_estimator(task, random_state):
    if task == CLASSIFICATION:
        return RandomForestClassifier(random_state=random_state, n_jobs=-1)
    return RandomForestRegressor(random_state=random_state, n_jobs=-1)


def default_predictor(task, random_state):
    if task == CLASSIFICATION:
        return RandomForestClassifier(random_state=random_state, n_jobs=-1)
    return RandomForestRegressor(random_state=random_state, n_jobs=-1)


def default_meta_estimator(task):
    if task == CLASSIFICATION:
        return LogisticRegression()
    return LinearRegression()


def default_cv(task, random_state, repeated=False):
    if task == CLASSIFICATION:
        if repeated:
            return RepeatedStratifiedKFold(
                n_splits=5,
                n_repeats=10,
                random_state=random_state,
            )
        return StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    if repeated:
        return RepeatedKFold(n_splits=5, n_repeats=10, random_state=random_state)
    return KFold(n_splits=5, shuffle=True, random_state=random_state)


def default_metric(task):
    if task == CLASSIFICATION:
        return "roc_auc"
    return "r2"


def validate_metric(task, metric):
    if metric is None:
        return default_metric(task)
    allowed = CLASSIFICATION_METRICS if task == CLASSIFICATION else REGRESSION_METRICS
    if metric not in allowed:
        raise ValueError(
            f"Unsupported metric '{metric}' for task '{task}'. Allowed: {', '.join(sorted(allowed))}."
        )
    return metric


def score_metric(y_true, y_pred, task, metric=None):
    metric = validate_metric(task, metric)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if metric == "roc_auc":
        return roc_auc_score(y_true, y_pred)
    if metric == "r2":
        return r2_score(y_true, y_pred)
    if metric == "neg_mean_squared_error":
        return -mean_squared_error(y_true, y_pred)
    if metric == "neg_root_mean_squared_error":
        return -mean_squared_error(y_true, y_pred) ** 0.5
    if metric == "pearson":
        score = pearsonr(y_true, y_pred)[0]
        return -np.inf if np.isnan(score) else score
    if metric == "spearman":
        score = spearmanr(y_true, y_pred)[0]
        return -np.inf if np.isnan(score) else score
    raise ValueError(f"Unsupported metric '{metric}'.")


def prediction_vector(estimator, X, task):
    if task == CLASSIFICATION:
        if hasattr(estimator, "predict_proba"):
            return estimator.predict_proba(X)[:, 1]
        if hasattr(estimator, "decision_function"):
            return estimator.decision_function(X)
        return estimator.predict(X)
    return estimator.predict(X)


def stacking_output(estimator, X, task):
    if task == CLASSIFICATION:
        if hasattr(estimator, "predict_proba"):
            return estimator.predict_proba(X)
        pred = prediction_vector(estimator, X, task)
        return np.asarray(pred).reshape(-1, 1)
    return np.asarray(estimator.predict(X)).reshape(-1, 1)


def ensure_estimator_matches_task(estimator, task, role):
    if estimator is None:
        return
    if task == CLASSIFICATION and is_regressor(estimator):
        raise ValueError(f"{role} should be a classifier when task='classification'.")
    if task == REGRESSION and is_classifier(estimator):
        raise ValueError(f"{role} should be a regressor when task='regression'.")


def slice_rows(X, row_index):
    try:
        return X.iloc[row_index, :]
    except AttributeError:
        return X[row_index, :]
