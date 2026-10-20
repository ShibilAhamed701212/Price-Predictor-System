import numpy as np
import pandas as pd
import pytest

from src.model_building import build_pipeline


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 60
    return pd.DataFrame(
        {
            "Order": np.arange(n),
            "PID": rng.integers(5e8, 9e8, n),
            "Gr Liv Area": rng.normal(7, 0.3, n),
            "Neighborhood": rng.choice(["A", "B"], n),
        }
    )


def test_identifier_columns_are_not_features(df):
    pipe = build_pipeline(df).fit(df, df["Gr Liv Area"] * 2)
    features = list(pipe.named_steps["preprocessor"].get_feature_names_out())
    assert not any("Order" in f or "PID" in f for f in features)
    assert "num__Gr Liv Area" in features
    assert "cat__Neighborhood_A" in features


def test_predicts_without_identifier_columns(df):
    pipe = build_pipeline(df).fit(df, df["Gr Liv Area"] * 2)
    preds = pipe.predict(df.drop(columns=["Order", "PID"]))
    assert preds == pytest.approx((df["Gr Liv Area"] * 2).to_numpy())
