"""Command-line exploratory analysis for the generated shipment CSV."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "shipments.csv"


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"{DATA_PATH} does not exist. Run `python generate_data.py` from the project folder."
        )
    data = pd.read_csv(DATA_PATH)
    print("Synthetic shipment dataset (not actual Nassau Candy records)")
    print(f"Rows and columns: {data.shape}")
    print("\nMissing values:")
    print(data.isna().sum().sort_values(ascending=False).head(10))
    print("\nSummary statistics:")
    print(data[["distance_km", "shipment_weight_kg", "shipping_cost",
                "expected_delivery_days", "actual_delivery_days"]].describe())

    route_summary = data.groupby("shipping_route").agg(
        on_time_rate=("delivery_status", lambda status: status.eq("On Time").mean()),
        average_cost=("shipping_cost", "mean"),
        average_days=("actual_delivery_days", "mean"),
    )
    route_summary = route_summary.sort_values("on_time_rate", ascending=False)
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(data=route_summary.reset_index(), y="shipping_route",
                x="on_time_rate", ax=axes[0], color="seagreen")
    axes[0].set_title("On-time delivery rate by route")
    axes[0].set_xlabel("On-time rate")
    sns.scatterplot(
        data=data, x="distance_km", y="shipping_cost", hue="shipping_mode",
        alpha=0.65, ax=axes[1],
    )
    axes[1].set_title("Shipping cost versus distance")
    figure.tight_layout()
    output_path = PROJECT_ROOT / "data" / "eda_summary.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"\nSaved charts to {output_path}")


if __name__ == "__main__":
    main()
