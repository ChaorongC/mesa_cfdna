import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from mesa import MESA, MESA_CV, MESA_modality


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
