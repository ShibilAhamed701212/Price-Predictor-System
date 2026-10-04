import numpy as np
import pandas as pd
import pytest

from src.data_splitter import DataSplitter, SimpleTrainTestSplitStrategy
from src.model_building import LinearRegressionStrategy, ModelBuilder
from src.model_evaluator import ModelEvaluator, RegressionModelEvaluationStrategy


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=200)
    x2 = rng.normal(size=200)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": 3 * x1 - 2 * x2 + 1})


def test_split_is_reproducible(df):
    splitter = DataSplitter(SimpleTrainTestSplitStrategy(test_size=0.2, random_state=42))
    X_train, X_test, y_train, y_test = splitter.split(df, "target")
    assert len(X_train) == 160 and len(X_test) == 40
    assert "target" not in X_train.columns
    again = splitter.split(df, "target")
    assert X_test.index.equals(again[1].index)


def test_build_and_evaluate_linear_model(df):
    X_train, X_test, y_train, y_test = DataSplitter(SimpleTrainTestSplitStrategy()).split(
        df, "target"
    )
    model = ModelBuilder(LinearRegressionStrategy()).build_model(X_train, y_train)
    metrics = ModelEvaluator(RegressionModelEvaluationStrategy()).evaluate(model, X_test, y_test)
    assert metrics["R-Squared"] == pytest.approx(1.0)
    assert metrics["Mean Squared Error"] == pytest.approx(0.0, abs=1e-12)


def test_build_rejects_wrong_types(df):
    with pytest.raises(TypeError):
        LinearRegressionStrategy().build_and_train_model(df.values, df["target"])
