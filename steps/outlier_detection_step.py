import logging

import pandas as pd
from zenml import step

from src.outlier_detection import OutlierDetector, ZScoreOutlierDetection


@step
def outlier_detection_step(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Detects and removes outliers using OutlierDetector."""
    if df is None:
        logging.error("Received a NoneType DataFrame.")
        raise ValueError("Input df must be a non-null pandas DataFrame.")

    if not isinstance(df, pd.DataFrame):
        logging.error(f"Expected pandas DataFrame, got {type(df)} instead.")
        raise ValueError("Input df must be a pandas DataFrame.")

    logging.info(f"Starting outlier detection step with DataFrame of shape: {df.shape}")

    if column_name not in df.columns:
        logging.error(f"Column '{column_name}' does not exist in the DataFrame.")
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

    # Only numeric columns are passed on to the model (categorical columns are dropped).
    df_numeric = df.select_dtypes(include=[int, float])

    # Detect outliers on column_name only. Filtering on every numeric column removed ~30%
    # of the rows, including every house with a basement half bath, which left that
    # feature (almost) constant in training and produced absurd predictions for such homes.
    outlier_detector = OutlierDetector(ZScoreOutlierDetection(threshold=3))
    outliers = outlier_detector.detect_outliers(df_numeric[[column_name]])[column_name]
    df_cleaned = df_numeric[~outliers]
    logging.info(f"Removed {int(outliers.sum())} outlier rows based on '{column_name}'.")
    return df_cleaned
