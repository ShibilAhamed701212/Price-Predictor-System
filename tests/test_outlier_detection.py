import pandas as pd

from src.outlier_detection import IQROutlierDetection, OutlierDetector, ZScoreOutlierDetection


def _df():
    values = [10.0] * 20 + [1000.0]
    return pd.DataFrame({"x": values, "y": range(21)})


def test_zscore_removes_extreme_row():
    out = OutlierDetector(ZScoreOutlierDetection(threshold=3)).handle_outliers(_df())
    assert len(out) == 20
    assert out["x"].max() == 10.0


def test_iqr_detects_extreme_value():
    mask = IQROutlierDetection().detect_outliers(_df())
    assert mask["x"].sum() == 1


def test_cap_keeps_all_rows():
    out = OutlierDetector(ZScoreOutlierDetection()).handle_outliers(_df(), method="cap")
    assert len(out) == 21
    assert out["x"].max() < 1000.0


def test_unknown_method_returns_input():
    df = _df()
    assert OutlierDetector(ZScoreOutlierDetection()).handle_outliers(df, method="?") is df
