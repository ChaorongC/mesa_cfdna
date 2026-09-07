import numpy as np
import pandas as pd
import pytest

from mesa._selection import DataFrameNormalizer, DataFrameVarianceThreshold


def test_dataframe_normalizer_preserves_index_columns_and_values():
    X = pd.DataFrame(
        {
            "a": [3.0, 0.0, 1.0],
            "b": [4.0, 5.0, 2.0],
            "c": [0.0, 12.0, 2.0],
        },
        index=["sample_1", "sample_2", "sample_3"],
    )

    normalizer = DataFrameNormalizer(norm="l2").fit(X)
    transformed = normalizer.transform(X)

    assert isinstance(transformed, pd.DataFrame)
    assert transformed.index.equals(X.index)
    assert transformed.columns.equals(X.columns)

    expected = X.to_numpy(dtype=float)
    expected = expected / np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(transformed.to_numpy(), expected)


def test_dataframe_normalizer_aligns_to_training_columns_and_ignores_extras():
    X_train = pd.DataFrame(
        {
            "a": [3.0, 0.0],
            "b": [4.0, 5.0],
        },
        index=["train_1", "train_2"],
    )
    X_test = pd.DataFrame(
        {
            "extra": [100.0, 200.0],
            "b": [8.0, 0.0],
            "a": [6.0, 5.0],
        },
        index=["test_1", "test_2"],
    )

    normalizer = DataFrameNormalizer().fit(X_train)
    transformed = normalizer.transform(X_test)

    assert transformed.columns.tolist() == ["a", "b"]
    assert transformed.index.equals(X_test.index)

    expected = X_test[["a", "b"]].to_numpy(dtype=float)
    expected = expected / np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(transformed.to_numpy(), expected)


def test_dataframe_normalizer_rejects_missing_training_columns():
    X = pd.DataFrame(
        {
            "a": [1.0, 2.0],
            "b": [3.0, 4.0],
        },
        index=["s1", "s2"],
    )

    normalizer = DataFrameNormalizer().fit(X)

    with pytest.raises(ValueError, match="columns used during fitting are missing"):
        normalizer.transform(X[["a"]])


def test_dataframe_normalizer_rejects_duplicate_feature_names():
    X = pd.DataFrame(
        [[1.0, 2.0], [2.0, 3.0]],
        columns=["duplicate", "duplicate"],
    )

    with pytest.raises(ValueError, match="X columns must be unique"):
        DataFrameNormalizer().fit(X)


def test_dataframe_normalizer_retains_numpy_behavior():
    X = np.asarray(
        [
            [3.0, 4.0, 0.0],
            [0.0, 5.0, 12.0],
        ]
    )

    normalizer = DataFrameNormalizer().fit(X)
    transformed = normalizer.transform(X)

    assert isinstance(transformed, np.ndarray)
    np.testing.assert_allclose(np.linalg.norm(transformed, axis=1), 1.0)


def test_normalizer_then_variance_threshold_preserves_sample_ids():
    X = pd.DataFrame(
        {
            "variable_a": [1.0, 2.0, 3.0, 4.0],
            "constant": [7.0, 7.0, 7.0, 7.0],
            "variable_b": [2.0, 4.0, 8.0, 16.0],
        },
        index=["s1", "s2", "s3", "s4"],
    )

    normalized = DataFrameNormalizer().fit_transform(X)
    filtered = DataFrameVarianceThreshold().fit_transform(normalized)

    assert isinstance(normalized, pd.DataFrame)
    assert isinstance(filtered, pd.DataFrame)
    assert normalized.index.equals(X.index)
    assert filtered.index.equals(X.index)
    assert filtered.columns.isin(X.columns).all()
