# Loan Default Prediction

End-to-end credit risk modeling project built on the LendingClub `loan.csv` dataset. The project compares logistic regression and XGBoost under class imbalance, with a focus on leakage prevention, threshold-aware evaluation, and model explainability.

> Portfolio project by Filippo Zonta — Quantitative Finance & Data Science.

## Business problem

Predict whether a loan will default using information available at origination. The target is intentionally limited to loans with final outcomes:

- **0 — non-default:** `Fully Paid`, `Does not meet the credit policy. Status:Fully Paid`
- **1 — default:** `Charged Off`, `Default`, `Does not meet the credit policy. Status:Charged Off`

After filtering, the modeling sample contains **256,939 loans** and **74 raw columns**. The target rate is **18.38% default**.

## Key results

| Model / strategy | ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic regression baseline | 0.7477 | 0.3484 | 0.5903 | 0.4382 |
| Logistic regression, no resampling | 0.7069 | 0.5160 | 0.0220 | 0.0426 |
| Random oversampling | 0.7067 | 0.2915 | 0.6637 | 0.4051 |
| Random undersampling | 0.7069 | 0.2918 | 0.6661 | 0.4059 |
| SMOTE | 0.6955 | 0.4526 | 0.0607 | 0.1070 |
| Tuned XGBoost | 0.7106 | 0.3000 | 0.6600 | — |

The experiments came from different notebook stages and preprocessing specifications, so rows should not be interpreted as a strict leaderboard. The repository code provides one consistent pipeline for future reruns.

## Leakage controls

Post-outcome variables are excluded before training, including balances, payments, recoveries, collection fees, and last/next payment information. See `src/loan_default/features.py` for the complete list.

## Repository structure

```text
.
├── configs/
│   └── model.yaml
├── data/
│   ├── raw/            # ignored; place loan.csv here
│   └── processed/      # ignored
├── notebooks/          # original analysis notebooks can be added here
├── src/loan_default/
│   ├── data.py
│   ├── features.py
│   ├── evaluate.py
│   └── train.py
├── tests/
├── .gitignore
├── pyproject.toml
└── requirements.txt
```

## Quick start

1. Download the LendingClub dataset from Kaggle and place it at `data/raw/loan.csv`.
2. Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Train the baseline:

```bash
python -m loan_default.train --data data/raw/loan.csv
```

## Modeling decisions

- Stratified train/test split
- Median imputation and scaling for numeric features
- Most-frequent imputation and one-hot encoding for categorical features
- Class-weighted logistic regression as the transparent baseline
- ROC-AUC plus precision, recall, F1, confusion matrix, and PR-AUC
- Threshold exposed as a business decision rather than treated as a fixed truth

## Limitations

This is an educational portfolio project, not a production underwriting system. Historical LendingClub data may contain selection bias and temporal drift. Performance should be validated out-of-time, probability calibration should be assessed, and fairness/compliance review is required before any real lending use. The raw dataset is not redistributed in this repository.

## Next steps

- Add time-based validation and probability calibration
- Restore the original EDA/modeling notebooks
- Add SHAP explanations on a memory-safe sample
- Compare decisions under explicit false-negative and false-positive costs
- Add a lightweight Streamlit risk-decision demo

## License

MIT.


## Upcoming projects

- [Portfolio Dashboard](projects/portfolio_dashboard/README.md)
- [AML Transaction Monitoring](projects/aml_transaction_monitoring/README.md)
