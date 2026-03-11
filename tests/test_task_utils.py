import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, RepeatedKFold, RepeatedStratifiedKFold, StratifiedKFold

from mesa._selection import wilcoxon
from mesa._task_utils import (
    CLASSIFICATION,
    REGRESSION,
    default_boruta_estimator,
    default_cv,
    default_meta_estimator,
    default_metric,
    default_predictor,
    default_selector,
    ensure_estimator_matches_task,
    prediction_vector,
    score_metric,
    slice_rows,
    stacking_output,
    validate_metric,
    validate_task,
)


def test_validate_task_and_metric():
    assert validate_task(CLASSIFICATION) == CLASSIFICATION
    assert validate_task(REGRESSION) == REGRESSION
    with pytest.raises(ValueError):
        validate_task("unknown")

    assert validate_metric(CLASSIFICATION, None) == "roc_auc"
    assert validate_metric(REGRESSION, None) == "r2"
    with pytest.raises(ValueError):
        validate_metric(CLASSIFICATION, "r2")


def test_default_estimators_and_cv():
    assert isinstance(default_boruta_estimator(CLASSIFICATION, 0), RandomForestClassifier)
    assert isinstance(default_boruta_estimator(REGRESSION, 0), RandomForestRegressor)
    assert isinstance(default_predictor(CLASSIFICATION, 0), RandomForestClassifier)
    assert isinstance(default_predictor(REGRESSION, 0), RandomForestRegressor)
    assert isinstance(default_meta_estimator(CLASSIFICATION), LogisticRegression)
    assert isinstance(default_meta_estimator(REGRESSION), LinearRegression)
    assert isinstance(default_cv(CLASSIFICATION, 0), StratifiedKFold)
    assert isinstance(default_cv(CLASSIFICATION, 0, repeated=True), RepeatedStratifiedKFold)
    assert isinstance(default_cv(REGRESSION, 0), KFold)
    assert isinstance(default_cv(REGRESSION, 0, repeated=True), RepeatedKFold)
    assert default_metric(CLASSIFICATION) == "roc_auc"
    assert default_metric(REGRESSION) == "r2"


def test_default_selector_by_task():
    cls_selector = default_selector(CLASSIFICATION, wilcoxon)
    reg_selector = default_selector(REGRESSION, wilcoxon)
    assert cls_selector.score_func is wilcoxon
    assert reg_selector.score_func.__name__ == "f_regression"


def test_score_metric_variants():
    y_true_cls = np.array([0, 1, 0, 1])
    y_pred_cls = np.array([0.1, 0.9, 0.2, 0.8])
    assert score_metric(y_true_cls, y_pred_cls, CLASSIFICATION) > 0.9

    y_true_reg = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred_reg = np.array([1.1, 1.9, 3.1, 3.9])
    assert score_metric(y_true_reg, y_pred_reg, REGRESSION, "r2") > 0.9
    assert np.isfinite(score_metric(y_true_reg, y_pred_reg, REGRESSION, "neg_mean_squared_error"))
    assert np.isfinite(score_metric(y_true_reg, y_pred_reg, REGRESSION, "neg_root_mean_squared_error"))
    assert np.isfinite(score_metric(y_true_reg, y_pred_reg, REGRESSION, "pearson"))
    assert np.isfinite(score_metric(y_true_reg, y_pred_reg, REGRESSION, "spearman"))


def test_prediction_and_stacking_helpers():
    Xc, yc = make_classification(n_samples=40, n_features=6, n_informative=4, random_state=0)
    clf = LogisticRegression(max_iter=1000).fit(Xc, yc)
    pred_vec_cls = prediction_vector(clf, Xc[:5], CLASSIFICATION)
    stack_cls = stacking_output(clf, Xc[:5], CLASSIFICATION)
    assert pred_vec_cls.shape == (5,)
    assert stack_cls.shape == (5, 2)

    Xr, yr = make_regression(n_samples=40, n_features=6, n_informative=4, random_state=0)
    reg = LinearRegression().fit(Xr, yr)
    pred_vec_reg = prediction_vector(reg, Xr[:5], REGRESSION)
    stack_reg = stacking_output(reg, Xr[:5], REGRESSION)
    assert pred_vec_reg.shape == (5,)
    assert stack_reg.shape == (5, 1)


def test_ensure_estimator_matches_task_and_slice_rows():
    ensure_estimator_matches_task(LogisticRegression(), CLASSIFICATION, "predictor")
    ensure_estimator_matches_task(LinearRegression(), REGRESSION, "predictor")
    with pytest.raises(ValueError):
        ensure_estimator_matches_task(LinearRegression(), CLASSIFICATION, "predictor")
    with pytest.raises(ValueError):
        ensure_estimator_matches_task(LogisticRegression(), REGRESSION, "predictor")

    X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    sliced_df = slice_rows(X, [0, 2])
    sliced_np = slice_rows(X.to_numpy(), [0, 2])
    assert sliced_df.shape == (2, 2)
    assert sliced_np.shape == (2, 2)
