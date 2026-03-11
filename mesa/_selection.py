import numpy as np
import pandas as pd
from boruta import BorutaPy
from scipy.stats import mannwhitneyu
from sklearn.base import clone
from sklearn.feature_selection import f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold

from ._task_utils import (
    CLASSIFICATION,
    REGRESSION,
    ensure_estimator_matches_task,
    prediction_vector,
    score_metric,
    validate_task,
)


def wilcoxon(X, y):
    """Score function for feature selection using Wilcoxon rank-sum test."""
    return -mannwhitneyu(X[y == 0], X[y == 1])[1]


class BorutaSelector(BorutaPy):
    """Select the top-n Boruta-ranked features."""

    def __init__(self, n=10, **kwargs):
        super().__init__(**kwargs)
        self.n = n

    def fit(self, X, y):
        super().fit(X, y)
        self.indices = np.argsort(self.ranking_)[: self.n]
        return self

    def transform(self, X):
        try:
            self.ranking_
        except AttributeError as exc:
            raise ValueError("You need to call the fit(X, y) method first.") from exc
        try:
            return X.iloc[:, self.indices]
        except AttributeError:
            return X[:, self.indices]

    def get_support(self):
        return self.indices


class missing_value_processing:
    """Filter columns by missingness and impute remaining values."""

    def __init__(self, ratio=0.9, imputer=SimpleImputer(strategy="mean")):
        self.ratio = ratio
        self.imputer = imputer

    def fit(self, X, y=None):
        if self.ratio <= 0:
            raise ValueError("The ratio of valid values should be greater than 0.")
        self.indices = np.where(
            pd.DataFrame(X).count(axis="rows") >= X.shape[0] * self.ratio
        )[0]
        self.imputer = clone(self.imputer).fit(pd.DataFrame(X).iloc[:, self.indices])
        return self

    def transform(self, X):
        if self.ratio <= 0:
            raise ValueError("The ratio of valid values should be greater than 0.")
        return pd.DataFrame(
            self.imputer.transform(pd.DataFrame(X).iloc[:, self.indices]),
            index=X.index,
            columns=X.columns[self.indices],
        )

    def get_support(self):
        return self.indices


class RedundancyPruner:
    """Remove highly correlated features by keeping one representative per block."""

    def __init__(
        self,
        mode=None,
        threshold=0.95,
        method="pearson",
        task=CLASSIFICATION,
        estimator=None,
        cv=3,
        metric=None,
    ):
        self.mode = mode
        self.threshold = threshold
        self.method = method
        self.task = task
        self.estimator = estimator
        self.cv = cv
        self.metric = metric

    def _score_features(self, X_df, y):
        if self.task == REGRESSION:
            scores, _ = f_regression(X_df, y)
            return np.nan_to_num(scores, nan=-np.inf, neginf=-np.inf)

        scores = np.full(X_df.shape[1], -np.inf, dtype=float)
        y_arr = np.asarray(y)
        for i, column in enumerate(X_df.columns):
            try:
                x0 = X_df.loc[y_arr == 0, column]
                x1 = X_df.loc[y_arr == 1, column]
                scores[i] = -mannwhitneyu(x0, x1)[1]
            except Exception:
                scores[i] = -np.inf
        return scores

    def _model_feature_score(self, feature, y):
        X_feature = np.asarray(feature).reshape(-1, 1)
        y_arr = np.asarray(y)
        if self.estimator is None:
            estimator = (
                LogisticRegression(random_state=0)
                if self.task == CLASSIFICATION
                else LinearRegression()
            )
        else:
            estimator = self.estimator
        ensure_estimator_matches_task(estimator, self.task, "redundancy_estimator")

        if isinstance(self.cv, int):
            cv = (
                StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=0)
                if self.task == CLASSIFICATION
                else KFold(n_splits=self.cv, shuffle=True, random_state=0)
            )
        else:
            cv = self.cv

        fold_scores = []
        for train_index, test_index in cv.split(X_feature, y_arr):
            X_train = X_feature[train_index]
            X_test = X_feature[test_index]
            y_train = y_arr[train_index]
            y_test = y_arr[test_index]
            try:
                fitted = clone(estimator).fit(X_train, y_train)
                pred = prediction_vector(fitted, X_test, self.task)
                fold_scores.append(score_metric(y_test, pred, self.task, self.metric))
            except Exception:
                fold_scores.append(-np.inf)
        return np.mean(fold_scores) if fold_scores else -np.inf

    def _model_scores(self, X_df, y):
        return np.array(
            [self._model_feature_score(X_df.iloc[:, i], y) for i in range(X_df.shape[1])],
            dtype=float,
        )

    def fit(self, X, y=None):
        self.task = validate_task(self.task)
        X_df = pd.DataFrame(X)
        n_features = X_df.shape[1]
        self.n_features_in_ = n_features

        if n_features == 0:
            self.indices = np.array([], dtype=int)
            return self
        if self.mode is None:
            self.indices = np.arange(n_features)
            return self
        if y is None:
            raise ValueError("Redundancy pruning requires y during fit.")
        if self.mode not in {"score", "model"}:
            raise ValueError("Redundancy pruning mode should be one of: None, 'score', 'model'.")
        if not 0 < self.threshold < 1:
            raise ValueError("Correlation pruning threshold should be between 0 and 1.")

        corr = X_df.corr(method=self.method).abs().fillna(0)
        feature_scores = (
            self._score_features(X_df, y)
            if self.mode == "score"
            else self._model_scores(X_df, y)
        )

        ranked_indices = np.argsort(np.nan_to_num(feature_scores, nan=-np.inf))[::-1]
        selected = []
        discarded = np.zeros(n_features, dtype=bool)
        for i in ranked_indices:
            if discarded[i]:
                continue
            selected.append(i)
            correlated = corr.iloc[i, :] >= self.threshold
            discarded[np.where(correlated.to_numpy())[0]] = True

        self.feature_scores_ = feature_scores
        self.indices = np.array(sorted(selected), dtype=int)
        return self

    def transform(self, X):
        if not hasattr(self, "indices"):
            raise ValueError("You need to call the fit(X, y) method first.")
        try:
            return X.iloc[:, self.indices]
        except AttributeError:
            return X[:, self.indices]

    def get_support(self, indices=False):
        if not hasattr(self, "indices"):
            raise ValueError("You need to call the fit(X, y) method first.")
        if indices:
            return self.indices
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.indices] = True
        return mask


def get_support_indices(step):
    """Return selected feature indices from heterogeneous selector APIs."""
    try:
        return step.get_support(indices=True)
    except TypeError:
        support = step.get_support()
        return np.asarray(support, dtype=int)
