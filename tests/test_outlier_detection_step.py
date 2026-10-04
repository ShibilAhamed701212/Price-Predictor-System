import pandas as pd
import pytest

pytest.importorskip("zenml")

from steps.outlier_detection_step import outlier_detection_step  # noqa: E402


def test_outliers_are_detected_on_the_target_column_only():
    # Regression: the step removed rows with a Z-score above 3 in *any* numeric column,
    # which dropped every rare-but-valid value (e.g. a basement half bath).
    df = pd.DataFrame(
        {
            "SalePrice": [10.0] * 20 + [1000.0, 10.0],
            "Bsmt Half Bath": [0] * 21 + [1],
            "Neighborhood": ["a"] * 22,
        }
    )

    out = outlier_detection_step.entrypoint(df, column_name="SalePrice")

    assert len(out) == 21
    assert out["SalePrice"].max() == 10.0
    assert out["Bsmt Half Bath"].max() == 1  # rare value in another column is kept
    assert "Neighborhood" not in out.columns  # only numeric columns are passed on


def test_missing_column_raises():
    with pytest.raises(ValueError):
        outlier_detection_step.entrypoint(pd.DataFrame({"x": [1.0]}), column_name="SalePrice")
