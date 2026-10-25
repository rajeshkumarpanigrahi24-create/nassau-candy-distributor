"""Train, evaluate, and save the delay classifier and cost regressor."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.evaluation import classification_metrics, regression_metrics
from src.preprocessing import (
    MODEL_FEATURES,
    build_pipeline,
    prepare_training_data,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = PROJECT_ROOT / "data" / "shipments.csv"
DEFAULT_MODELS = PROJECT_ROOT / "models"


def train(data_path: Path = DEFAULT_DATA, model_dir: Path = DEFAULT_MODELS, seed: int = 42) -> dict:
    """Fit both models from one deterministic train/test split and persist results."""
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Run `python generate_data.py` first."
        )
    try:
        raw_data = pd.read_csv(data_path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise ValueError(f"Could not read dataset {data_path}: {error}") from error
    features, delay_target, cost_target = prepare_training_data(raw_data)
    if len(features) < 10:
        raise ValueError("At least 10 usable shipment rows are required for training.")
    if delay_target.nunique() < 2:
        raise ValueError("The delay target must contain both on-time and delayed examples.")
    if delay_target.value_counts().min() < 2:
        raise ValueError(
            "At least two examples of each delay class are required for a stratified split."
        )

    indices = range(len(features))
    train_indices, test_indices = train_test_split(
        list(indices),
        test_size=0.2,
        random_state=seed,
        stratify=delay_target,
    )
    x_train, x_test = features.iloc[train_indices], features.iloc[test_indices]

    classifier = build_pipeline("classification", seed)
    classifier.fit(x_train, delay_target.iloc[train_indices])
    delay_prediction = classifier.predict(x_test)

    regressor = build_pipeline("regression", seed)
    regressor.fit(x_train, cost_target.iloc[train_indices])
    cost_prediction = regressor.predict(x_test)

    results = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": data_path.name,
        "rows_used": int(len(features)),
        "train_rows": int(len(train_indices)),
        "test_rows": int(len(test_indices)),
        "random_seed": seed,
        "features": MODEL_FEATURES,
        "classification": classification_metrics(
            delay_target.iloc[test_indices].to_numpy(), delay_prediction
        ),
        "regression": regression_metrics(
            cost_target.iloc[test_indices].to_numpy(), cost_prediction
        ),
        "notice": "Models use synthetic educational data; not actual Nassau Candy records.",
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, model_dir / "delay_classifier.joblib")
    joblib.dump(regressor, model_dir / "shipping_cost_regressor.joblib")
    with (model_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Shipment CSV path.")
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS, help="Model output folder.")
    parser.add_argument("--seed", type=int, default=42, help="Train/test and model seed.")
    args = parser.parse_args()
    try:
        metrics = train(args.data, args.models, args.seed)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Trained on {metrics['rows_used']} rows; test set: {metrics['test_rows']} rows.")
    print(f"Delay classifier: {json.dumps(metrics['classification'], indent=2)}")
    print(f"Shipping cost regressor: {json.dumps(metrics['regression'], indent=2)}")
    print(f"Saved models and metrics to {args.models.resolve()}")


if __name__ == "__main__":
    main()
