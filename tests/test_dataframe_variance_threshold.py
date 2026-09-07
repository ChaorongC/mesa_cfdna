import numpy as np
import pandas as pd
import pytest

from mesa._selection import DataFrameVarianceThreshold


def test_dataframe_variance_threshold_preserves_index_names_and_training_order():
    X = pd.DataFrame(
        {
            "variable_a": [1.0, 2.0, 3.0, 4.0],
            "constant": [7.0, 7.0, 7.0, 7.0],
            "variable_b": [2.0, 4.0, 8.0, 16.0],
        },
        index=["sample_1", "sample_2", "sample_3", "sample_4"],
    )

    selector = DataFrameVarianceThreshold(threshold=0).fit(X)

    # Deliberately reorder the incoming columns. The transformer should align
    # them to the training feature order before applying the learned mask.
    transformed = selector.transform(X[["variable_b", "constant", "variable_a"]])

    assert isinstance(transformed, pd.DataFrame)
    assert transformed.index.equals(X.index)
    assert transformed.columns.tolist() == ["variable_a", "variable_b"]
    np.testing.assert_allclose(
        transformed.to_numpy(),
        X[["variable_a", "variable_b"]].to_numpy(),
    )
    np.testing.assert_array_equal(selector.get_support(indices=True), [0, 2])
    assert selector.get_feature_names_out().tolist() == ["variable_a", "variable_b"]


def test_dataframe_variance_threshold_rejects_missing_training_columns():
    X = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [3.0, 4.0, 5.0],
        },
        index=["s1", "s2", "s3"],
    )

    selector = DataFrameVarianceThreshold().fit(X)

    with pytest.raises(ValueError, match="columns used during fitting are missing"):
        selector.transform(X[["a"]])


def test_dataframe_variance_threshold_rejects_duplicate_feature_names():
    X = pd.DataFrame(
        [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
        columns=["duplicate", "duplicate"],
    )

    with pytest.raises(ValueError, match="X columns must be unique"):
        DataFrameVarianceThreshold().fit(X)


def test_dataframe_variance_threshold_retains_numpy_behavior():
    X = np.asarray(
        [
            [1.0, 5.0, 2.0],
            [2.0, 5.0, 4.0],
            [3.0, 5.0, 8.0],
        ]
    )

    selector = DataFrameVarianceThreshold().fit(X)
    transformed = selector.transform(X)

    assert isinstance(transformed, np.ndarray)
    np.testing.assert_allclose(transformed, X[:, [0, 2]])
