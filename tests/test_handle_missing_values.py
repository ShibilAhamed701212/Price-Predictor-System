import numpy as np
import pandas as pd
import pytest

from src.handle_missing_values import (
    DropMissingValuesStrategy,
    FillMissingValuesStrategy,
    MissingValueHandler,
)


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "num": [1.0, np.nan, 3.0, 3.0],
            "cat": ["x", None, "x", "y"],
        }
    )


def test_mean_fills_numeric_only(df):
    out = FillMissingValuesStrategy("mean").handle(df)
    assert out["num"].tolist() == pytest.approx([1.0, 7 / 3, 3.0, 3.0])
    assert out["cat"].isna().sum() == 1
    assert df["num"].isna().sum() == 1  # input is not modified


def test_median(df):
    out = FillMissingValuesStrategy("median").handle(df)
    assert out.loc[1, "num"] == 3.0


def test_mode_fills_every_column(df):
    # Regression: the mode branch used chained `fillna(inplace=True)`, which silently
    # did nothing under pandas copy-on-write.
    out = FillMissingValuesStrategy("mode").handle(df)
    assert out.isna().sum().sum() == 0
    assert out.loc[1, "num"] == 3.0
    assert out.loc[1, "cat"] == "x"


def test_mode_skips_all_nan_column():
    df = pd.DataFrame({"empty": [np.nan, np.nan], "num": [1.0, np.nan]})
    out = FillMissingValuesStrategy("mode").handle(df)
    assert out["empty"].isna().all()
    assert out["num"].tolist() == [1.0, 1.0]


def test_constant(df):
    out = FillMissingValuesStrategy("constant", fill_value=0).handle(df)
    assert out.loc[1, "num"] == 0
    assert out.loc[1, "cat"] == 0


def test_drop_rows(df):
    out = MissingValueHandler(DropMissingValuesStrategy(axis=0)).handle_missing_values(df)
    assert len(out) == 3
