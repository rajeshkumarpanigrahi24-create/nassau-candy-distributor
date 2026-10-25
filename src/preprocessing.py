"""Feature definitions and reusable scikit-learn preprocessing pipelines."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CLASSIFICATION_TARGET = "is_delayed"
REGRESSION_TARGET = "shipping_cost"

# These fields are known when a shipment is planned. Actual delivery results,
# delivery_status, and shipping_cost are deliberately absent from predictors.
NUMERIC_FEATURES = [
    "distance_km",
    "shipment_weight_kg",
    "dispatch_hour",
    "expected_delivery_days",
]
CATEGORICAL_FEATURES = [
    "factory_location",
    "customer_location",
    "shipping_route",
    "traffic_level",
    "weather_condition",
    "shipping_mode",
]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def prepare_training_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Validate required fields and construct targets without outcome leakage."""
    required = set(
        MODEL_FEATURES
        + ["shipping_cost", "actual_delivery_days", "expected_delivery_days"]
    )
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    usable = data.copy()
    actual_days = pd.to_numeric(usable["actual_delivery_days"], errors="coerce")
    expected_days = pd.to_numeric(usable["expected_delivery_days"], errors="coerce")
    cost = pd.to_numeric(usable["shipping_cost"], errors="coerce")
    valid_targets = (
        actual_days.notna()
        & expected_days.notna()
        & cost.notna()
        & np.isfinite(actual_days)
        & np.isfinite(expected_days)
        & np.isfinite(cost)
    )
    usable = usable.loc[valid_targets].copy()
    if usable.empty:
        raise ValueError("No rows have the target values required for training.")

    features = usable[MODEL_FEATURES].copy()
    numeric = features[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    categorical = features[CATEGORICAL_FEATURES].astype(object)
    features = pd.concat([numeric, categorical], axis=1)
    classification_target = (actual_days.loc[valid_targets] > expected_days.loc[valid_targets]).astype(int)
    regression_target = cost.loc[valid_targets]

    return (
        features.reset_index(drop=True),
        classification_target.reset_index(drop=True),
        regression_target.reset_index(drop=True),
    )


def build_pipeline(
    task: Literal["classification", "regression"], seed: int = 42
) -> Pipeline:
    """Build imputation, encoding, and model steps as one safe inference pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    if task == "classification":
        estimator = RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    elif task == "regression":
        estimator = RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported task: {task}")
    return Pipeline(steps=[("preprocessing", preprocessing), ("model", estimator)])


def make_prediction_row(values: dict[str, object]) -> pd.DataFrame:
    """Validate and order a single user-entered prediction row."""
    missing = [feature for feature in MODEL_FEATURES if feature not in values]
    if missing:
        raise ValueError(f"Missing prediction inputs: {', '.join(missing)}")
    row = pd.DataFrame([{feature: values[feature] for feature in MODEL_FEATURES}])
    for feature in NUMERIC_FEATURES:
        row[feature] = pd.to_numeric(row[feature], errors="coerce")
    if row[NUMERIC_FEATURES].isna().any().any():
        raise ValueError("Numeric prediction inputs must be valid numbers.")
    return row
