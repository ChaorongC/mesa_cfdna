import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from mesa._selection import (
    BorutaSelector,
    RedundancyPruner,
    get_support_indices,
    missing_value_processing,
    wilcoxon,
)
from mesa._task_utils import CLASSIFICATION, REGRESSION


def test_wilcoxon_returns_vector():
    X = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.2, 0.1, 0.4],
            [1.0, 1.1, 1.2],
            [1.1, 1.0, 1.3],
        ]
    )
    y = np.array([0, 0, 1, 1])
    score = wilcoxon(X, y)
    assert np.asarray(score).shape == (3,)


def test_missing_value_processing_fit_transform_and_errors():
    X = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0],
            "b": [1.0, 2.0, 3.0],
            "c": [np.nan, np.nan, 1.0],
        }
    )
    proc = missing_value_processing(ratio=0.5)
    proc.fit(X)
    transformed = proc.transform(X)
    assert transformed.shape[1] == 2
    assert len(proc.get_support()) == 2

    with pytest.raises(ValueError):
        missing_value_processing(ratio=0).fit(X)


def test_redundancy_pruner_classification_and_regression():
    rng = np.random.RandomState(0)
    base = rng.randn(40)
    X = pd.DataFrame(
        {
            "f1": base,
            "f2": base * 0.999 + rng.randn(40) * 1e-3,
            "f3": rng.randn(40),
            "f4": rng.randn(40),
        }
    )
    yc = np.array([0, 1] * 20)
    yr = base + rng.randn(40) * 0.1

    cls_pruner = RedundancyPruner(mode="score", threshold=0.95, task=CLASSIFICATION)
    cls_pruner.fit(X, yc)
    assert len(cls_pruner.get_support(indices=True)) < X.shape[1]
    assert cls_pruner.transform(X).shape[1] == len(cls_pruner.get_support(indices=True))

    reg_pruner = RedundancyPruner(
        mode="model",
        threshold=0.95,
        task=REGRESSION,
        estimator=RandomForestRegressor(n_estimators=10, random_state=0),
        cv=3,
    )
    reg_pruner.fit(X, yr)
    assert len(reg_pruner.get_support(indices=True)) < X.shape[1]


def test_redundancy_pruner_validation_errors():
    X = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    y = np.array([0, 1, 0])
    with pytest.raises(ValueError):
        RedundancyPruner(mode="bad").fit(X, y)
    with pytest.raises(ValueError):
        RedundancyPruner(mode="score", threshold=1.5).fit(X, y)
    with pytest.raises(ValueError):
        RedundancyPruner(mode="score").fit(X, None)


def test_boruta_selector_and_support_index_helper():
    X = pd.DataFrame(np.random.RandomState(0).randn(40, 6))
    y = np.array([0, 1] * 20)
    selector = BorutaSelector(
        n=3,
        estimator=RandomForestClassifier(n_estimators=20, random_state=0),
        random_state=0,
        verbose=0,
        n_estimators="auto",
    )
    selector.fit(X, y)
    transformed = selector.transform(X)
    assert transformed.shape[1] == 3
    assert len(selector.get_support()) == 3

    class StepWithMask:
        def get_support(self):
            return [0, 2]

    class StepWithIndices:
        def get_support(self, indices=True):
            return np.array([1, 3])

    assert np.array_equal(get_support_indices(StepWithMask()), np.array([0, 2]))
    assert np.array_equal(get_support_indices(StepWithIndices()), np.array([1, 3]))
