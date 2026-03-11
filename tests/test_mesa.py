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
