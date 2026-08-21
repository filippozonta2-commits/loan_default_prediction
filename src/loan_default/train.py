"""Train a leakage-aware logistic-regression baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from loan_default.data import load_final_outcomes
from loan_default.evaluate import evaluate_probabilities
from loan_default.features import split_features_target


def build_pipeline(X, max_iter: int = 1000, class_weight: str = "balanced") -> Pipeline:
    """Build preprocessing and transparent baseline model."""
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()

    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "one_hot",
                            OneHotEncoder(handle_unknown="ignore", min_frequency=20),
                        ),
                    ]
                ),
                categorical,
            ),
        ]
    )

    return Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "model",
                LogisticRegression(
                    max_iter=max_iter,
                    class_weight=class_weight,
                    solver="liblinear",
                ),
            ),
        ]
    )


def train(data_path: Path, config_path: Path, output_dir: Path) -> dict:
    """Train, evaluate, and persist a baseline pipeline."""
    config = yaml.safe_load(config_path.read_text())
    frame = load_final_outcomes(data_path)
    X, y = split_features_target(frame)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y,
    )

    model_config = config["model"]
    pipeline = build_pipeline(
        X_train,
        max_iter=model_config["max_iter"],
        class_weight=model_config["class_weight"],
    )
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    metrics = evaluate_probabilities(
        y_test, probabilities, threshold=config["decision_threshold"]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_dir / "baseline.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(train(args.data, args.config, args.output), indent=2))
