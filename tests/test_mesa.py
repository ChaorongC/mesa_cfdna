import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold

from mesa import MESA, MESA_CV, MESA_modality


class ProbabilityColumnModality:
    """Minimal cloneable modality whose first input column is P(positive)."""

    def __init__(self, positive_label=1, invalid=None, task="classification"):
        self.positive_label = positive_label
        self.invalid = invalid
        self.task = task

    def get_params(self, deep=True):
        return {
            "positive_label": self.positive_label,
            "invalid": self.invalid,
            "task": self.task,
        }

    def fit(self, X, y):
        self.classes_ = np.unique(np.asarray(y))
        self.predictor_ = self
        return self

    def transform_predict_proba(self, X):
        score = np.asarray(X, dtype=float)[:, 0].copy()
        if self.invalid == "nan":
            score[0] = np.nan
        elif self.invalid == "high":
            score[0] = 1.1
        positive = np.flatnonzero(self.classes_ == self.positive_label)
        if positive.size != 1:
            raise ValueError("positive_label should identify exactly one class")
        proba = np.column_stack([1.0 - score, 1.0 - score])
        proba[:, positive[0]] = score
        if self.invalid == "sum":
            proba[0] = [0.4, 0.4]
        return proba

    def get_support(self, step=None):
        return np.asarray([0])


def test_mesa_modality_classification_and_regression_defaults():
    cls = MESA_modality(task="classification")
    reg = MESA_modality(task="regression")
    assert cls._resolved_predictor().__class__.__name__ == "RandomForestClassifier"
    assert cls._resolved_boruta_estimator().__class__.__name__ == "RandomForestClassifier"
    assert reg._resolved_predictor().__class__.__name__ == "RandomForestRegressor"
    assert reg._resolved_boruta_estimator().__class__.__name__ == "RandomForestRegressor"


def test_mesa_modality_classification_flow():
    X, y = make_classification(n_samples=80, n_features=14, n_informative=7, random_state=0)
    X = pd.DataFrame(X)
    model = MESA_modality(
        task="classification",
        top_n=5,
        selector=10,
        classifier=RandomForestClassifier(n_estimators=20, random_state=0),
        boruta_estimator=RandomForestClassifier(n_estimators=20, random_state=0),
        random_state=0,
    )
    model.fit(X, y)
    pred = model.predict(model.transform(X.iloc[:8]))
    proba = model.transform_predict_proba(X.iloc[:8])
    assert pred.shape == (8,)
    assert proba.shape == (8, 2)
    assert len(model.get_support()) == 5


def test_mesa_modality_regression_flow_and_probability_error():
    X, y = make_regression(n_samples=80, n_features=14, n_informative=7, noise=0.5, random_state=0)
    X = pd.DataFrame(X)
    model = MESA_modality(
        task="regression",
        top_n=5,
        selector=10,
        predictor=RandomForestRegressor(n_estimators=20, random_state=0),
        boruta_estimator=RandomForestRegressor(n_estimators=20, random_state=0),
        random_state=0,
    )
    model.fit(X, y)
    pred = model.transform_predict(X.iloc[:8])
    assert pred.shape == (8,)
    with pytest.raises(ValueError):
        model.transform_predict_proba(X.iloc[:8])


def test_mesa_ensemble_classification_and_regression():
    X1, y_cls = make_classification(n_samples=70, n_features=12, n_informative=6, random_state=0)
    X2, _ = make_classification(n_samples=70, n_features=10, n_informative=5, random_state=1)
    X1 = pd.DataFrame(X1)
    X2 = pd.DataFrame(X2)

    cls_modalities = [
        MESA_modality(
            task="classification",
            top_n=4,
            selector=8,
            classifier=RandomForestClassifier(n_estimators=20, random_state=0),
            boruta_estimator=RandomForestClassifier(n_estimators=20, random_state=0),
            random_state=0,
        ),
        MESA_modality(
            task="classification",
            top_n=3,
            selector=6,
            classifier=LogisticRegression(max_iter=1000),
            boruta_estimator=RandomForestClassifier(n_estimators=20, random_state=0),
            random_state=0,
        ),
    ]
    mesa_cls = MESA(task="classification", modalities=cls_modalities)
    mesa_cls.fit([X1, X2], y_cls)
    proba = mesa_cls.predict_proba([X1.iloc[:10], X2.iloc[:10]])
    assert proba.shape == (10, 2)
    assert len(mesa_cls.get_support()) == 2

    Xr1, y_reg = make_regression(n_samples=70, n_features=12, n_informative=6, noise=1.0, random_state=0)
    Xr2, _ = make_regression(n_samples=70, n_features=10, n_informative=5, noise=1.0, random_state=1)
    Xr1 = pd.DataFrame(Xr1)
    Xr2 = pd.DataFrame(Xr2)
    reg_modalities = [
        MESA_modality(
            task="regression",
            top_n=4,
            selector=8,
            predictor=LinearRegression(),
            boruta_estimator=RandomForestRegressor(n_estimators=20, random_state=0),
            random_state=0,
        ),
        MESA_modality(
            task="regression",
            top_n=3,
            selector=6,
            predictor=LinearRegression(),
            boruta_estimator=RandomForestRegressor(n_estimators=20, random_state=0),
            random_state=0,
        ),
    ]
    mesa_reg = MESA(task="regression", modalities=reg_modalities)
    mesa_reg.fit([Xr1, Xr2], y_reg)
    pred = mesa_reg.predict([Xr1.iloc[:10], Xr2.iloc[:10]])
    assert pred.shape == (10,)
    with pytest.raises(ValueError):
        mesa_reg.predict_proba([Xr1.iloc[:10], Xr2.iloc[:10]])


def test_mesa_control_anchor_rank_blend_classification():
    X1, y = make_classification(
        n_samples=72,
        n_features=12,
        n_informative=6,
        random_state=10,
    )
    X2, _ = make_classification(
        n_samples=72,
        n_features=10,
        n_informative=5,
        random_state=11,
    )
    X1 = pd.DataFrame(X1)
    X2 = pd.DataFrame(X2)

    modalities = [
        MESA_modality(
            task="classification",
            top_n=4,
            selector=8,
            classifier=RandomForestClassifier(n_estimators=20, random_state=0),
            boruta_estimator=RandomForestClassifier(n_estimators=20, random_state=0),
            random_state=0,
        ),
        MESA_modality(
            task="classification",
            top_n=3,
            selector=6,
            classifier=LogisticRegression(max_iter=1000),
            boruta_estimator=RandomForestClassifier(n_estimators=20, random_state=0),
            random_state=0,
        ),
    ]
    model = MESA(
        task="classification",
        modalities=modalities,
        integration_method="control_anchor_rank_blend",
        integration_weights=[0.6, 0.4],
        random_state=0,
    )

    model.fit([X1, X2], y)
    proba = model.predict_proba([X1.iloc[:9], X2.iloc[:9]])
    pred = model.predict([X1.iloc[:9], X2.iloc[:9]])

    assert proba.shape == (9, 2)
    assert pred.shape == (9,)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.all((proba[:, 1] >= 0.0) & (proba[:, 1] <= 1.0))
    assert len(model.control_anchor_scores_) == 2
    assert model.integration_weights_.tolist() == [0.6, 0.4]
    expected_controls = sum(
        int(np.sum(np.asarray(y)[test_index] == 0))
        for _, test_index in model.splits
    )
    assert all(anchor.shape == (expected_controls,) for anchor in model.control_anchor_scores_)


def test_mesa_probability_blend_equal_and_weighted_probabilities():
    X1 = pd.DataFrame({"probability": [0.1, 0.8, 0.6, 0.4]})
    X2 = pd.DataFrame({"probability": [0.3, 0.4, 0.2, 0.9]})
    y = np.asarray([0, 1, 1, 0])
    modalities = [ProbabilityColumnModality(), ProbabilityColumnModality()]

    equal = MESA(modalities=modalities, integration_method="probability_blend")
    equal.fit([X1, X2], y)
    equal_proba = equal.predict_proba([X1, X2])

    expected_equal = np.asarray([0.2, 0.6, 0.4, 0.65])
    np.testing.assert_allclose(equal_proba[:, 1], expected_equal)
    np.testing.assert_allclose(equal_proba.sum(axis=1), 1.0)
    assert equal.predict([X1, X2]).tolist() == [0, 1, 0, 1]
    assert equal.integration_weights_.tolist() == [0.5, 0.5]
    assert not hasattr(equal, "splits")
    assert not hasattr(equal, "meta_estimator_")

    weighted = MESA(
        modalities=modalities,
        integration_method="probability_blend",
        integration_weights=[0.75, 0.25],
    )
    weighted.fit([X1, X2], y)
    expected_weighted = 0.75 * X1.iloc[:, 0] + 0.25 * X2.iloc[:, 0]
    np.testing.assert_allclose(
        weighted.predict_proba([X1, X2])[:, 1],
        expected_weighted,
    )


def test_mesa_probability_blend_respects_control_label_and_class_order():
    X1 = pd.DataFrame({"probability": [0.2, 0.8, 0.7, 0.3]})
    X2 = pd.DataFrame({"probability": [0.4, 0.6, 0.9, 0.1]})
    y = np.asarray(["control", "case", "case", "control"])
    modalities = [
        ProbabilityColumnModality(positive_label="case"),
        ProbabilityColumnModality(positive_label="case"),
    ]
    model = MESA(
        modalities=modalities,
        integration_method="probability_blend",
        control_label="control",
    )

    model.fit([X1, X2], y)
    np.testing.assert_allclose(
        model.predict_proba([X1, X2])[:, 1],
        [0.3, 0.7, 0.8, 0.2],
    )
    assert model.classes_.tolist() == ["control", "case"]
    assert model.predict([X1, X2]).tolist() == ["control", "case", "case", "control"]


@pytest.mark.parametrize(
    "weights, message",
    [
        ([1.0], "match the number"),
        ([1.1, -0.1], "non-negative"),
        ([np.nan, np.nan], "finite"),
        ([0.4, 0.4], "sum to 1"),
    ],
)
def test_mesa_probability_blend_rejects_invalid_weights(weights, message):
    X = pd.DataFrame({"probability": [0.2, 0.8, 0.3, 0.7]})
    model = MESA(
        modalities=[ProbabilityColumnModality(), ProbabilityColumnModality()],
        integration_method="probability_blend",
        integration_weights=weights,
    )
    with pytest.raises(ValueError, match=message):
        model.fit([X, X], np.asarray([0, 1, 0, 1]))


def test_mesa_probability_blend_rejects_incomplete_or_invalid_inputs():
    X = pd.DataFrame({"probability": [0.2, 0.8, 0.3, 0.7]})
    y = np.asarray([0, 1, 0, 1])
    modalities = [ProbabilityColumnModality(), ProbabilityColumnModality()]

    with pytest.raises(ValueError, match="one matrix per modality"):
        MESA(modalities=modalities, integration_method="probability_blend").fit([X], y)
    with pytest.raises(ValueError, match="align with y"):
        MESA(modalities=modalities, integration_method="probability_blend").fit(
            [X, X.iloc[:-1]], y
        )

    model = MESA(modalities=modalities, integration_method="probability_blend").fit(
        [X, X], y
    )
    with pytest.raises(ValueError, match="one matrix per modality"):
        model.predict_proba([X])
    with pytest.raises(ValueError, match="same samples"):
        model.predict_proba([X, X.iloc[:-1]])

    for invalid, message in [
        ("nan", "finite"),
        ("high", "between 0 and 1"),
        ("sum", "sum to 1"),
    ]:
        invalid_model = MESA(
            modalities=[ProbabilityColumnModality(invalid=invalid)],
            integration_method="probability_blend",
        ).fit([X], y)
        with pytest.raises(ValueError, match=message):
            invalid_model.predict_proba([X])


def test_mesa_probability_blend_rejects_multiclass_and_regression():
    X = pd.DataFrame({"probability": [0.2, 0.8, 0.3, 0.7, 0.4, 0.6]})
    multiclass = MESA(
        modalities=[ProbabilityColumnModality(positive_label=2)],
        integration_method="probability_blend",
    )
    with pytest.raises(ValueError, match="exactly two classes"):
        multiclass.fit([X], np.asarray([0, 1, 2, 0, 1, 2]))

    regression = MESA(
        task="regression",
        modalities=[MESA_modality(task="regression")],
        integration_method="probability_blend",
    )
    with pytest.raises(ValueError, match="probability_blend"):
        regression.fit([X], np.linspace(0.0, 1.0, len(X)))


def test_mesa_probability_blend_is_cloneable_and_runs_in_mesa_cv():
    X1 = pd.DataFrame({"probability": [0.1, 0.8, 0.2, 0.7, 0.3, 0.9]})
    X2 = pd.DataFrame({"probability": [0.2, 0.7, 0.3, 0.6, 0.4, 0.8]})
    y = np.asarray([0, 1, 0, 1, 0, 1])
    model = MESA(
        modalities=[ProbabilityColumnModality(), ProbabilityColumnModality()],
        integration_method="probability_blend",
        integration_weights=[0.6, 0.4],
        custom_flag="preserved",
    )

    cloned = clone(model)
    assert cloned.integration_method == "probability_blend"
    assert cloned.integration_weights == [0.6, 0.4]
    assert cloned.custom_flag == "preserved"

    evaluator = MESA_CV(
        modality=model,
        task="classification",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=0),
    ).fit([X1, X2], y)
    assert np.isfinite(evaluator.get_performance())


def test_mesa_control_anchor_rank_blend_respects_control_label():
    X1, y_numeric = make_classification(
        n_samples=60,
        n_features=10,
        n_informative=5,
        random_state=12,
    )
    X2, _ = make_classification(
        n_samples=60,
        n_features=8,
        n_informative=4,
        random_state=13,
    )
    y = np.where(y_numeric == 0, "control", "case")
    modalities = [
        MESA_modality(
            task="classification",
            top_n=3,
            selector=6,
            classifier=RandomForestClassifier(n_estimators=20, random_state=0),
            boruta_estimator=RandomForestClassifier(n_estimators=20, random_state=0),
        ),
        MESA_modality(
            task="classification",
            top_n=3,
            selector=6,
            classifier=RandomForestClassifier(n_estimators=20, random_state=1),
            boruta_estimator=RandomForestClassifier(n_estimators=20, random_state=1),
        ),
    ]
    model = MESA(
        task="classification",
        modalities=modalities,
        integration_method="control_anchor_rank_blend",
        control_label="control",
        random_state=0,
    )

    model.fit([pd.DataFrame(X1), pd.DataFrame(X2)], y)
    proba = model.predict_proba([pd.DataFrame(X1).iloc[:6], pd.DataFrame(X2).iloc[:6]])

    assert model.classes_.tolist() == ["control", "case"]
    assert proba.shape == (6, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_control_anchor_rank_blend_uses_non_control_probability_column():
    class EstimatorWithStringClasses:
        classes_ = np.asarray(["case", "control"])

        def predict_proba(self, X):
            return np.tile(np.asarray([[0.8, 0.2]]), (len(X), 1))

    model = MESA(
        modalities=[MESA_modality()],
        integration_method="control_anchor_rank_blend",
        control_label="control",
    )
    model.classes_ = np.asarray(["control", "case"])

    scores = model._positive_class_scores(
        EstimatorWithStringClasses(),
        pd.DataFrame(np.zeros((3, 2))),
    )

    assert np.allclose(scores, [0.8, 0.8, 0.8])


def test_mesa_control_anchor_rank_blend_rejects_regression():
    Xr1, y = make_regression(
        n_samples=50,
        n_features=10,
        n_informative=5,
        random_state=2,
    )
    Xr2, _ = make_regression(
        n_samples=50,
        n_features=8,
        n_informative=4,
        random_state=3,
    )
    modalities = [
        MESA_modality(
            task="regression",
            top_n=3,
            selector=6,
            predictor=LinearRegression(),
            boruta_estimator=RandomForestRegressor(n_estimators=20, random_state=0),
        ),
        MESA_modality(
            task="regression",
            top_n=3,
            selector=6,
            predictor=LinearRegression(),
            boruta_estimator=RandomForestRegressor(n_estimators=20, random_state=0),
        ),
    ]
    model = MESA(
        task="regression",
        modalities=modalities,
        integration_method="control_anchor_rank_blend",
    )

    with pytest.raises(ValueError, match="control_anchor_rank_blend"):
        model.fit([pd.DataFrame(Xr1), pd.DataFrame(Xr2)], y)


def test_mesa_rejects_unknown_integration_method():
    with pytest.raises(ValueError, match="integration_method"):
        MESA(
            modalities=[MESA_modality()],
            integration_method="not_a_method",
        ).fit([pd.DataFrame(np.random.randn(20, 4))], np.array([0, 1] * 10))


def test_mesa_cv_classification_and_regression_metrics():
    Xc, yc = make_classification(n_samples=60, n_features=10, n_informative=5, random_state=0)
    Xc = pd.DataFrame(Xc)
    cv_cls = MESA_CV(
        modality=MESA_modality(
            task="classification",
            top_n=4,
            selector=8,
            classifier=RandomForestClassifier(n_estimators=20, random_state=0),
            boruta_estimator=RandomForestClassifier(n_estimators=20, random_state=0),
            random_state=0,
        ),
        task="classification",
    )
    cv_cls.fit(Xc, yc)
    assert np.isfinite(cv_cls.get_performance())

    Xr, yr = make_regression(n_samples=60, n_features=10, n_informative=5, noise=1.0, random_state=0)
    Xr = pd.DataFrame(Xr)
    cv_reg = MESA_CV(
        modality=MESA_modality(
            task="regression",
            top_n=4,
            selector=8,
            predictor=LinearRegression(),
            boruta_estimator=RandomForestRegressor(n_estimators=20, random_state=0),
            random_state=0,
        ),
        task="regression",
    )
    cv_reg.fit(Xr, yr)
    assert np.isfinite(cv_reg.get_performance())
    assert np.isfinite(cv_reg.get_performance(metric="neg_root_mean_squared_error"))
    assert np.isfinite(cv_reg.get_performance(metric="pearson"))


def test_task_mismatch_errors():
    with pytest.raises(ValueError):
        MESA(
            task="regression",
            modalities=[MESA_modality(task="classification")],
        ).fit([pd.DataFrame(np.random.randn(20, 4))], np.random.randn(20))

    with pytest.raises(ValueError):
        MESA_CV(modality=MESA_modality(task="classification"), task="regression").fit(
            pd.DataFrame(np.random.randn(20, 4)),
            np.random.randn(20),
        )
