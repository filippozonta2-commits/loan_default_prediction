import pandas as pd

from loan_default.features import split_features_target


def test_leakage_columns_are_removed():
    frame = pd.DataFrame(
        {
            "annual_inc": [50_000, 80_000],
            "total_pymnt": [10_000, 20_000],
            "loan_status": ["Fully Paid", "Charged Off"],
            "default": [0, 1],
        }
    )

    X, y = split_features_target(frame)

    assert X.columns.tolist() == ["annual_inc"]
    assert y.tolist() == [0, 1]
