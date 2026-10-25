"""Generate a reproducible synthetic factory-to-customer shipment dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "shipments.csv"


def generate_shipments(rows: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Return synthetic shipment records; no records come from Nassau Candy."""
    if rows < 1:
        raise ValueError("rows must be a positive integer.")

    rng = np.random.default_rng(seed)
    factories = np.array(["Factory A", "Factory B", "Factory C"])
    customer_locations = np.array(
        ["Nassau", "Freeport", "Marsh Harbour", "George Town", "Spanish Wells"]
    )
    factory_locations = rng.choice(factories, size=rows, p=[0.50, 0.30, 0.20])
    customer = rng.choice(
        customer_locations, size=rows, p=[0.32, 0.24, 0.18, 0.15, 0.11]
    )

    distance_by_customer = {
        "Nassau": 35,
        "Freeport": 210,
        "Marsh Harbour": 280,
        "George Town": 390,
        "Spanish Wells": 165,
    }
    factory_adjustment = {"Factory A": 0, "Factory B": 22, "Factory C": 48}
    distance = np.array(
        [
            distance_by_customer[location] + factory_adjustment[factory]
            for location, factory in zip(customer, factory_locations)
        ],
        dtype=float,
    )
    distance += rng.normal(0, 18, size=rows)
    distance = np.maximum(distance, 12).round(1)

    traffic = rng.choice(["Low", "Medium", "High"], size=rows, p=[0.36, 0.44, 0.20])
    weather = rng.choice(
        ["Clear", "Rain", "Storm"], size=rows, p=[0.67, 0.27, 0.06]
    )
    weight = np.clip(rng.lognormal(mean=3.6, sigma=0.72, size=rows), 2, 900).round(1)
    mode = rng.choice(
        ["Truck", "Van", "Freight"], size=rows, p=[0.49, 0.32, 0.19]
    )
    dispatch_hour = rng.integers(5, 20, size=rows)

    speed = np.select(
        [mode == "Van", mode == "Freight"],
        [58.0, 42.0],
        default=52.0,
    )
    traffic_hours = np.select(
        [traffic == "Medium", traffic == "High"], [0.55, 1.35], default=0.0
    )
    weather_hours = np.select(
        [weather == "Rain", weather == "Storm"], [0.7, 2.3], default=0.0
    )
    dispatch_hours_penalty = np.where(
        (dispatch_hour >= 7) & (dispatch_hour <= 9)
        | (dispatch_hour >= 16) & (dispatch_hour <= 18),
        0.45,
        0.0,
    )
    expected_days = np.maximum(
        1, np.ceil(distance / speed / 8 + 0.35).astype(int)
    )

    delay_probability = (
        0.07
        + 0.14 * (traffic == "High")
        + 0.09 * (traffic == "Medium")
        + 0.19 * (weather == "Storm")
        + 0.06 * (weather == "Rain")
        + 0.10 * (mode == "Freight")
        + 0.04 * (dispatch_hour >= 16)
        + 0.08 * (distance > 300)
    )
    delayed = rng.random(rows) < np.clip(delay_probability, 0.03, 0.78)
    duration_days = (
        distance / speed / 8
        + traffic_hours / 8
        + weather_hours / 8
        + dispatch_hours_penalty / 8
        + rng.normal(0.45, 0.24, size=rows)
    )
    duration_days += delayed * rng.uniform(0.7, 2.2, size=rows)
    actual_days = np.maximum(1, np.ceil(duration_days)).astype(int)
    status = np.where(actual_days > expected_days, "Delayed", "On Time")

    mode_base = np.select(
        [mode == "Van", mode == "Freight"], [18.0, 95.0], default=34.0
    )
    cost = (
        mode_base
        + distance * np.select(
            [mode == "Van", mode == "Freight"], [0.19, 0.28], default=0.23
        )
        + weight * np.select(
            [mode == "Van", mode == "Freight"], [0.065, 0.035], default=0.05
        )
        + np.where(traffic == "High", 16, np.where(traffic == "Medium", 6, 0))
        + np.where(weather == "Storm", 24, np.where(weather == "Rain", 8, 0))
        + rng.normal(0, 12, size=rows)
    )
    cost = np.maximum(cost, 15).round(2)

    route = np.char.add(
        np.char.add(factory_locations.astype(str), " to "), customer.astype(str)
    )
    shipments = pd.DataFrame(
        {
            "shipment_id": [f"SHP-{number:05d}" for number in range(1, rows + 1)],
            "factory_location": factory_locations,
            "customer_location": customer,
            "shipping_route": route,
            "distance_km": distance,
            "traffic_level": traffic,
            "weather_condition": weather,
            "shipment_weight_kg": weight,
            "shipping_mode": mode,
            "shipping_cost": cost,
            "dispatch_hour": dispatch_hour,
            "expected_delivery_days": expected_days,
            "actual_delivery_days": actual_days,
            "delivery_status": status,
        }
    )

    # A small, realistic amount of missing input data exercises the imputation pipeline.
    for column, fraction in {
        "traffic_level": 0.012,
        "weather_condition": 0.01,
        "shipment_weight_kg": 0.008,
    }.items():
        missing = rng.choice(rows, size=max(1, int(rows * fraction)), replace=False)
        shipments.loc[missing, column] = np.nan
    return shipments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1500, help="Number of rows to make.")
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path (defaults to data/shipments.csv).",
    )
    args = parser.parse_args()

    try:
        shipments = generate_shipments(args.rows, args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        shipments.to_csv(args.output, index=False)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Generated {len(shipments):,} synthetic shipments at {args.output}")
    print("This educational dataset is not actual Nassau Candy company data.")


if __name__ == "__main__":
    main()
