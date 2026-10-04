import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    FeatureEngineer,
    LogTransformation,
    MinMaxScaling,
    OneHotEncoding,
    StandardScaling,
)


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "area": [0.0, 9.0, 99.0],
            "price": [1.0, 2.0, 3.0],
            "hood": ["a", "b", "a"],
        },
        index=[10, 20, 30],
    )


def test_log_transformation(df):
    out = FeatureEngineer(LogTransformation(["area"])).apply_feature_engineering(df)
    assert out["area"].tolist() == pytest.approx(np.log1p([0.0, 9.0, 99.0]).tolist())
    assert df["area"].tolist() == [0.0, 9.0, 99.0]


def test_standard_scaling(df):
    out = StandardScaling(["price"]).apply_transformation(df)
    assert out["price"].mean() == pytest.approx(0.0)


def test_minmax_scaling(df):
    out = MinMaxScaling(["price"]).apply_transformation(df)
    assert out["price"].tolist() == [0.0, 0.5, 1.0]


def test_one_hot_encoding(df):
    # Regression: OneHotEncoder(sparse=False) raises TypeError on scikit-learn >= 1.4.
    out = OneHotEncoding(["hood"]).apply_transformation(df)
    assert "hood" not in out.columns
    assert out["hood_b"].tolist() == [0.0, 1.0, 0.0]
    # Rows stay aligned with the original index and no NaN rows are introduced.
    assert list(out.index) == [10, 20, 30]
    assert out.isna().sum().sum() == 0
