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
    """Validate the task label used across the MESA API.

    Parameters
    ----------
    task : str
        Requested learning task. Supported values are ``"classification"``
        and ``"regression"``.

    Returns
    -------
    str
        The validated task string.

    Raises
    ------
    ValueError
        If ``task`` is not one of the supported values.
    """
    if task not in SUPPORTED_TASKS:
        raise ValueError("task should be one of: 'classification', 'regression'.")
    return task


def default_selector(task, classification_score_func):
    """Build the default first-stage univariate selector for a task.

    Parameters
    ----------
    task : str
        Learning task used to choose the score function.
    classification_score_func : callable
        Score function used for classification, typically ``wilcoxon``.

    Returns
    -------
    sklearn.feature_selection.GenericUnivariateSelect
        A selector configured with task-appropriate scoring.
    """
    if task == CLASSIFICATION:
        return GenericUnivariateSelect(
            score_func=classification_score_func,
            mode="k_best",
            param=2000,
        )
    return GenericUnivariateSelect(score_func=f_regression, mode="k_best", param=2000)


def default_boruta_estimator(task, random_state):
    """Return the default estimator used internally by Boruta.

    Parameters
    ----------
    task : str
        Learning task that determines classifier vs regressor behavior.
    random_state : int
        Random seed passed to the random forest estimator.

    Returns
    -------
    sklearn.base.BaseEstimator
        A random-forest classifier for classification or random-forest
        regressor for regression.
    """
    if task == CLASSIFICATION:
        return RandomForestClassifier(random_state=random_state, n_jobs=-1)
    return RandomForestRegressor(random_state=random_state, n_jobs=-1)


def default_predictor(task, random_state):
    """Return the default final predictor for a modality.

    Parameters
    ----------
    task : str
        Learning task that determines classifier vs regressor behavior.
    random_state : int
        Random seed passed to the random forest estimator.

    Returns
    -------
    sklearn.base.BaseEstimator
        A task-appropriate random forest model.
    """
    if task == CLASSIFICATION:
        return RandomForestClassifier(random_state=random_state, n_jobs=-1)
    return RandomForestRegressor(random_state=random_state, n_jobs=-1)


def default_meta_estimator(task):
    """Return the default second-level estimator used by ``MESA``.

    Parameters
    ----------
    task : str
        Learning task that determines classifier vs regressor behavior.

    Returns
    -------
    sklearn.base.BaseEstimator
        Logistic regression for classification or linear regression for
        regression.
    """
    if task == CLASSIFICATION:
        return LogisticRegression()
    return LinearRegression()


def default_cv(task, random_state, repeated=False):
    """Construct the default cross-validation splitter for a task.

    Parameters
    ----------
    task : str
        Learning task used to choose stratified vs non-stratified splitting.
    random_state : int
        Random seed for shuffle-enabled splitters.
    repeated : bool, default=False
        Whether to use repeated K-fold style CV rather than a single K-fold
        pass.

    Returns
    -------
    sklearn.model_selection.BaseCrossValidator
        A task-appropriate CV splitter.
    """
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
    """Return the default evaluation metric for a task.

    Parameters
    ----------
    task : str
        Learning task used to choose the metric.

    Returns
    -------
    str
        ``"roc_auc"`` for classification or ``"r2"`` for regression.
    """
    if task == CLASSIFICATION:
        return "roc_auc"
    return "r2"


def validate_metric(task, metric):
    """Validate a metric name against the allowed set for a task.

    Parameters
    ----------
    task : str
        Learning task used to choose the valid metric set.
    metric : str or None
        Metric requested by the caller. ``None`` resolves to the task default.

    Returns
    -------
    str
        The validated metric name.
    """
    if metric is None:
        return default_metric(task)
    allowed = CLASSIFICATION_METRICS if task == CLASSIFICATION else REGRESSION_METRICS
    if metric not in allowed:
        raise ValueError(
            f"Unsupported metric '{metric}' for task '{task}'. Allowed: {', '.join(sorted(allowed))}."
        )
    return metric


def score_metric(y_true, y_pred, task, metric=None):
    """Score predictions using a task-aware metric.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels or continuous outcomes.
    y_pred : array-like
        Predicted probabilities, scores, or continuous values.
    task : str
        Learning task used to interpret the prediction vector.
    metric : str or None, default=None
        Metric name. If ``None``, the task default metric is used.

    Returns
    -------
    float
        Scalar performance score.
    """
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
    """Extract a one-dimensional prediction vector from an estimator.

    Parameters
    ----------
    estimator : sklearn-compatible estimator
        Fitted estimator used to generate predictions.
    X : array-like
        Input design matrix.
    task : str
        Learning task used to choose probability, decision, or direct
        prediction output.

    Returns
    -------
    numpy.ndarray
        One-dimensional prediction vector.
    """
    if task == CLASSIFICATION:
        if hasattr(estimator, "predict_proba"):
            return estimator.predict_proba(X)[:, 1]
        if hasattr(estimator, "decision_function"):
            return estimator.decision_function(X)
        return estimator.predict(X)
    return estimator.predict(X)


def stacking_output(estimator, X, task):
    """Generate modality-level outputs used as stacking features.

    Parameters
    ----------
    estimator : sklearn-compatible estimator
        Fitted modality-level estimator.
    X : array-like
        Input matrix for a single modality.
    task : str
        Learning task used to choose probability-style or scalar outputs.

    Returns
    -------
    numpy.ndarray
        Two-dimensional array suitable for concatenation into meta-features.
    """
    if task == CLASSIFICATION:
        if hasattr(estimator, "predict_proba"):
            return estimator.predict_proba(X)
        pred = prediction_vector(estimator, X, task)
        return np.asarray(pred).reshape(-1, 1)
    return np.asarray(estimator.predict(X)).reshape(-1, 1)


def ensure_estimator_matches_task(estimator, task, role):
    """Validate that an estimator is compatible with the requested task.

    Parameters
    ----------
    estimator : sklearn-compatible estimator or None
        Estimator to validate. ``None`` is accepted and ignored.
    task : str
        Learning task expected by the surrounding API.
    role : str
        Human-readable estimator role used in error messages.
    """
    if estimator is None:
        return
    if task == CLASSIFICATION and is_regressor(estimator):
        raise ValueError(f"{role} should be a classifier when task='classification'.")
    if task == REGRESSION and is_classifier(estimator):
        raise ValueError(f"{role} should be a regressor when task='regression'.")


def slice_rows(X, row_index):
    """Slice rows from pandas or NumPy inputs using a shared helper path.

    Parameters
    ----------
    X : pandas.DataFrame or numpy.ndarray
        Matrix-like input object.
    row_index : array-like
        Row indices to retain.

    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        Row-subsetted object of the same general type as ``X``.
    """
    try:
        return X.iloc[row_index, :]
    except AttributeError:
        return X[row_index, :]
