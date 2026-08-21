"""Dataset loading and target construction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FINAL_STATUS_TO_TARGET = {
    "Fully Paid": 0,
    "Does not meet the credit policy. Status:Fully Paid": 0,
    "Charged Off": 1,
    "Default": 1,
    "Does not meet the credit policy. Status:Charged Off": 1,
}


def load_final_outcomes(path: str | Path) -> pd.DataFrame:
    """Load LendingClub data and retain only loans with final outcomes."""
    frame = pd.read_csv(path, low_memory=False)
    if "loan_status" not in frame.columns:
        raise ValueError("Expected a 'loan_status' column.")

    frame = frame.loc[frame["loan_status"].isin(FINAL_STATUS_TO_TARGET)].copy()
    frame["default"] = frame["loan_status"].map(FINAL_STATUS_TO_TARGET).astype("int8")
    return frame
