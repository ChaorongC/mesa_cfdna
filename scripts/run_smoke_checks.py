#!/usr/bin/env python3

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mesa import MESA, MESA_CV, MESA_modality


def run_classification_checks():
    X1, y = make_classification(
        n_samples=80,
        n_features=14,
        n_informative=7,
        random_state=0,
    )
    X2, _ = make_classification(
        n_samples=80,
        n_features=10,
        n_informative=5,
        random_state=1,
    )
    X1 = pd.DataFrame(X1)
    X2 = pd.DataFrame(X2)

    modality = MESA_modality(
        task="classification",
        top_n=5,
        selector=10,
        predictor=RandomForestClassifier(n_estimators=20, random_state=0),
        boruta_estimator=RandomForestClassifier(n_estimators=20, random_state=0),
        random_state=0,
    )
    modality.fit(X1, y)
    proba = modality.transform_predict_proba(X1.iloc[:12])
    auc = roc_auc_score(y[:12], proba[:, 1])

    mesa = MESA(
        task="classification",
        modalities=[
            MESA_modality(
                task="classification",
                top_n=5,
                selector=10,
                predictor=RandomForestClassifier(n_estimators=20, random_state=0),
                boruta_estimator=RandomForestClassifier(n_estimators=20, random_state=0),
                random_state=0,
            ),
            MESA_modality(
                task="classification",
                top_n=4,
                selector=8,
                predictor=LogisticRegression(max_iter=1000),
                boruta_estimator=RandomForestClassifier(n_estimators=20, random_state=0),
                random_state=0,
            ),
        ],
    )
    mesa.fit([X1, X2], y)
    mesa_proba = mesa.predict_proba([X1.iloc[:12], X2.iloc[:12]])

    cv_eval = MESA_CV(
        modality=MESA_modality(
            task="classification",
            top_n=5,
            selector=10,
            predictor=RandomForestClassifier(n_estimators=20, random_state=0),
            boruta_estimator=RandomForestClassifier(n_estimators=20, random_state=0),
            random_state=0,
        ),
        task="classification",
    )
    cv_eval.fit(X1, y)

    return {
        "single_modality_auc": auc,
        "single_modality_proba_shape": tuple(proba.shape),
        "ensemble_proba_shape": tuple(mesa_proba.shape),
        "cv_auc": cv_eval.get_performance(),
    }


def run_regression_checks():
    X1, y = make_regression(
        n_samples=90,
        n_features=16,
        n_informative=8,
        noise=0.5,
        random_state=0,
    )
    X2, _ = make_regression(
        n_samples=90,
        n_features=12,
        n_informative=6,
        noise=1.0,
        random_state=1,
    )
    X1 = pd.DataFrame(X1)
    X2 = pd.DataFrame(X2)

    modality = MESA_modality(
        task="regression",
        top_n=6,
        selector=12,
        predictor=RandomForestRegressor(n_estimators=20, random_state=0),
        boruta_estimator=RandomForestRegressor(n_estimators=20, random_state=0),
        random_state=0,
    )
    modality.fit(X1, y)
    pred = modality.transform_predict(X1.iloc[:12])
    r2 = r2_score(y[:12], pred)

    mesa = MESA(
        task="regression",
        modalities=[
            MESA_modality(
                task="regression",
                top_n=5,
                selector=10,
                predictor=LinearRegression(),
                boruta_estimator=RandomForestRegressor(n_estimators=20, random_state=0),
                random_state=0,
            ),
            MESA_modality(
                task="regression",
                top_n=4,
                selector=8,
                predictor=LinearRegression(),
                boruta_estimator=RandomForestRegressor(n_estimators=20, random_state=0),
                random_state=0,
            ),
        ],
    )
    mesa.fit([X1, X2], y)
    mesa_pred = mesa.predict([X1.iloc[:12], X2.iloc[:12]])

    cv_eval = MESA_CV(
        modality=MESA_modality(
            task="regression",
            top_n=5,
            selector=10,
            predictor=LinearRegression(),
            boruta_estimator=RandomForestRegressor(n_estimators=20, random_state=0),
            random_state=0,
        ),
        task="regression",
    )
    cv_eval.fit(X1, y)

    return {
        "single_modality_r2": r2,
        "single_modality_pred_shape": tuple(pred.shape),
        "ensemble_pred_shape": tuple(mesa_pred.shape),
        "cv_r2": cv_eval.get_performance(),
        "cv_rmse": cv_eval.get_performance(metric="neg_root_mean_squared_error"),
    }


def main():
    classification = run_classification_checks()
    regression = run_regression_checks()

    print("classification", classification)
    print("regression", regression)


if __name__ == "__main__":
    main()
