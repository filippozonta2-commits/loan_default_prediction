"""Feature selection with explicit outcome-leakage controls."""

from __future__ import annotations

import pandas as pd

TARGET_COLUMN = "default"

LEAKAGE_COLUMNS = [
    "loan_status",
    "out_prncp",
    "out_prncp_inv",
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",
]


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return predictors and target after removing known post-outcome fields."""
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}'.")

    drop_columns = [column for column in LEAKAGE_COLUMNS if column in frame.columns]
    features = frame.drop(columns=drop_columns + [TARGET_COLUMN])
    target = frame[TARGET_COLUMN].astype(int)
    return features, target
