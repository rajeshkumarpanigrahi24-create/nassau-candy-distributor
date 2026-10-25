"""Interactive Streamlit dashboard for synthetic shipping-route analysis."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from src.preprocessing import MODEL_FEATURES, make_prediction_row

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "shipments.csv"
MODEL_DIR = PROJECT_ROOT / "models"

st.set_page_config(
    page_title="Shipping Route Efficiency | Nassau Candy Distributor",
    page_icon="🚚",
    layout="wide",
)


@st.cache_data
def load_shipments(path_string: str, modified_time: float) -> pd.DataFrame:
    del modified_time
    return pd.read_csv(path_string)


@st.cache_resource
def load_artifacts(model_dir: str, classifier_modified: float, regressor_modified: float):
    del classifier_modified, regressor_modified
    folder = Path(model_dir)
    classifier_path = folder / "delay_classifier.joblib"
    regressor_path = folder / "shipping_cost_regressor.joblib"
    metrics_path = folder / "metrics.json"
    if not classifier_path.is_file() or not regressor_path.is_file():
        raise FileNotFoundError("Saved models are missing. Run `python train_model.py`.")
    classifier = joblib.load(classifier_path)
    regressor = joblib.load(regressor_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    return classifier, regressor, metrics


def money(value: float) -> str:
    return f"${value:,.2f}"


st.title("Factory-to-Customer Shipping Route Efficiency")
st.caption(
    "Nassau Candy Distributor | Educational demonstration using synthetic shipment data "
    "— not actual Nassau Candy company data."
)

if not DATA_PATH.is_file():
    st.error("Shipment data was not found. In the project folder, run `python generate_data.py`.")
    st.stop()
try:
    shipment_data = load_shipments(str(DATA_PATH), DATA_PATH.stat().st_mtime)
except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
    st.error(f"Could not load shipment data: {error}")
    st.stop()

required_columns = set(
    MODEL_FEATURES
    + [
        "shipment_id",
        "shipping_cost",
        "actual_delivery_days",
        "expected_delivery_days",
        "delivery_status",
    ]
)
missing_columns = sorted(required_columns.difference(shipment_data.columns))
if missing_columns:
    st.error(f"Shipment CSV is missing required columns: {', '.join(missing_columns)}")
    st.stop()

with st.sidebar:
    st.header("Explore shipments")
    routes = sorted(shipment_data["shipping_route"].dropna().unique().tolist())
    selected_routes = st.multiselect("Route", routes, default=routes)
    modes = sorted(shipment_data["shipping_mode"].dropna().unique().tolist())
    selected_modes = st.multiselect("Shipping mode", modes, default=modes)
    statuses = sorted(shipment_data["delivery_status"].dropna().unique().tolist())
    selected_statuses = st.multiselect("Delivery status", statuses, default=statuses)
    traffic_values = sorted(shipment_data["traffic_level"].dropna().unique().tolist())
    selected_traffic = st.multiselect("Traffic level", traffic_values, default=traffic_values)

filtered = shipment_data[
    shipment_data["shipping_route"].isin(selected_routes)
    & shipment_data["shipping_mode"].isin(selected_modes)
    & shipment_data["delivery_status"].isin(selected_statuses)
    & shipment_data["traffic_level"].isin(selected_traffic)
].copy()

if filtered.empty:
    st.warning("No shipments match these filters. Select one or more values in the sidebar.")
    st.stop()

total = len(filtered)
delay_rate = (filtered["delivery_status"] == "Delayed").mean() * 100
on_time_rate = 100 - delay_rate
average_cost = filtered["shipping_cost"].mean()
average_duration = filtered["actual_delivery_days"].mean()
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Shipments", f"{total:,}")
kpi2.metric("On-time delivery", f"{on_time_rate:.1f}%")
kpi3.metric("Average shipping cost", money(average_cost))
kpi4.metric("Average delivery time", f"{average_duration:.2f} days")

st.markdown(
    "### How to read this dashboard\n"
    "Routes are compared using delivery success, average cost, travel distance, and time. "
    "A shipment is labeled **Delayed** when actual delivery days exceed expected days. "
    "The prediction form only uses information available at dispatch; it does not use actual "
    "delivery outcomes."
)

route_summary = (
    filtered.assign(is_on_time=filtered["delivery_status"].eq("On Time"))
    .groupby("shipping_route", as_index=False)
    .agg(
        shipments=("shipment_id", "count"),
        success_rate=("is_on_time", "mean"),
        average_cost=("shipping_cost", "mean"),
        average_days=("actual_delivery_days", "mean"),
        average_distance=("distance_km", "mean"),
    )
)
route_summary["success_rate"] *= 100
route_summary["delay_percentage"] = 100 - route_summary["success_rate"]
route_summary = route_summary.sort_values(
    ["success_rate", "average_cost"], ascending=[False, True]
)

st.subheader("Route efficiency comparison")
st.caption(
    "Higher success rate and lower cost/time are preferable. Route scores are descriptive "
    "averages, not a guarantee for the next shipment."
)
left, right = st.columns(2)
with left:
    st.bar_chart(
        route_summary.set_index("shipping_route")[["success_rate", "delay_percentage"]],
        y_label="Shipments (%)",
    )
with right:
    st.bar_chart(
        route_summary.set_index("shipping_route")[["average_cost"]],
        y_label="Average shipping cost ($)",
    )
st.dataframe(
    route_summary.rename(
        columns={
            "shipping_route": "Route",
            "shipments": "Shipments",
            "success_rate": "On-time success (%)",
            "average_cost": "Average cost ($)",
            "average_days": "Average delivery days",
            "average_distance": "Average distance (km)",
            "delay_percentage": "Delay (%)",
        }
    ).style.format(
        {
            "On-time success (%)": "{:.1f}",
            "Average cost ($)": "${:.2f}",
            "Average delivery days": "{:.2f}",
            "Average distance (km)": "{:.1f}",
            "Delay (%)": "{:.1f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Shipping cost and delivery duration")
chart_left, chart_right = st.columns(2)
with chart_left:
    mode_cost = filtered.groupby("shipping_mode", as_index=False)["shipping_cost"].mean()
    st.bar_chart(mode_cost.set_index("shipping_mode"), y_label="Average cost ($)")
with chart_right:
    duration = filtered.groupby("actual_delivery_days").size().rename("Shipments")
    st.line_chart(duration, y_label="Shipments", x_label="Actual delivery days")

st.subheader("Shipment records")
st.dataframe(filtered, use_container_width=True, hide_index=True)
st.download_button(
    "Download filtered shipment data",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_shipments.csv",
    mime="text/csv",
)

st.subheader("Model performance")
try:
    classifier, regressor, metrics = load_artifacts(
        str(MODEL_DIR),
        (MODEL_DIR / "delay_classifier.joblib").stat().st_mtime
        if (MODEL_DIR / "delay_classifier.joblib").exists()
        else 0,
        (MODEL_DIR / "shipping_cost_regressor.joblib").stat().st_mtime
        if (MODEL_DIR / "shipping_cost_regressor.joblib").exists()
        else 0,
    )
except (OSError, ValueError, joblib.externals.loky.process_executor.TerminatedWorkerError, EOFError) as error:
    st.warning(f"Could not load trained models: {error}")
    st.info("In the project folder, run `python train_model.py`, then refresh this page.")
    st.stop()

classification = metrics.get("classification", {})
regression = metrics.get("regression", {})
metric_columns = st.columns(7)
for column, label, key, format_string in [
    (metric_columns[0], "Accuracy", "accuracy", "{:.1%}"),
    (metric_columns[1], "Precision", "precision", "{:.1%}"),
    (metric_columns[2], "Recall", "recall", "{:.1%}"),
    (metric_columns[3], "F1-score", "f1", "{:.1%}"),
    (metric_columns[4], "Cost MAE", "mae", "${:.2f}"),
    (metric_columns[5], "Cost RMSE", "rmse", "${:.2f}"),
    (metric_columns[6], "Cost R²", "r2", "{:.2f}"),
]:
    value = classification.get(key) if key in classification else regression.get(key)
    column.metric(label, format_string.format(value) if value is not None else "N/A")
st.caption(
    "Metrics come from a held-out test set. Cost MAE/RMSE are in dollars; R² describes "
    "fit on that test set. These metrics are illustrative because the data is synthetic."
)

matrix = classification.get("confusion_matrix")
if matrix and len(matrix) == 2:
    fig, axis = plt.subplots(figsize=(5, 3.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted on time", "Predicted delayed"],
        yticklabels=["Actual on time", "Actual delayed"],
        ax=axis,
    )
    axis.set_title("Delay classifier confusion matrix")
    axis.set_xlabel("Prediction")
    axis.set_ylabel("Actual outcome")
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Try a planned shipment")
st.write(
    "Enter information known at dispatch. The classifier estimates delay risk; the regressor "
    "estimates shipping cost. Actual delivery results are never used as prediction inputs."
)
with st.form("shipment_prediction"):
    form_left, form_right = st.columns(2)
    with form_left:
        factory = st.selectbox(
            "Factory",
            sorted(shipment_data["factory_location"].dropna().unique()),
        )
        customer = st.selectbox(
            "Customer location",
            sorted(shipment_data["customer_location"].dropna().unique()),
        )
        distance = st.number_input("Distance (km)", min_value=1.0, value=100.0, step=5.0)
        weight = st.number_input(
            "Shipment weight (kg)", min_value=0.1, value=50.0, step=5.0
        )
        mode = st.selectbox(
            "Shipping mode", sorted(shipment_data["shipping_mode"].dropna().unique())
        )
    with form_right:
        traffic = st.selectbox(
            "Traffic level", ["Low", "Medium", "High"], index=1
        )
        weather = st.selectbox(
            "Weather condition", ["Clear", "Rain", "Storm"], index=0
        )
        dispatch_hour = st.slider("Dispatch hour (24-hour clock)", 0, 23, 9)
        expected_days = st.number_input(
            "Expected delivery days", min_value=1, value=2, step=1
        )
    submitted = st.form_submit_button("Predict shipment")

if submitted:
    try:
        prediction_values = {
            "factory_location": factory,
            "customer_location": customer,
            "shipping_route": f"{factory} to {customer}",
            "distance_km": distance,
            "traffic_level": traffic,
            "weather_condition": weather,
            "shipment_weight_kg": weight,
            "shipping_mode": mode,
            "dispatch_hour": dispatch_hour,
            "expected_delivery_days": expected_days,
        }
        prediction_row = make_prediction_row(prediction_values)
        delay_probability = float(classifier.predict_proba(prediction_row)[0][1])
        cost_estimate = float(regressor.predict(prediction_row)[0])
        result_left, result_right = st.columns(2)
        result_left.metric("Estimated delay probability", f"{delay_probability:.1%}")
        result_right.metric("Estimated shipping cost", money(cost_estimate))
        if delay_probability >= 0.5:
            st.warning(
                "Higher predicted delay risk. Consider an earlier dispatch, a lower-traffic "
                "departure window, or checking alternate modes."
            )
        else:
            st.success("The model estimates a lower delay risk for these inputs.")
    except (ValueError, TypeError, KeyError) as error:
        st.error(f"Could not make a prediction: {error}")

st.subheader("Data-driven operating recommendations")
if len(route_summary):
    best_service = route_summary.sort_values(
        ["success_rate", "average_cost"], ascending=[False, True]
    ).iloc[0]
    lowest_cost = route_summary.sort_values(
        ["average_cost", "success_rate"], ascending=[True, False]
    ).iloc[0]
    st.markdown(
        f"- **Service reliability:** {best_service['shipping_route']} had the highest "
        f"on-time rate in the filtered sample ({best_service['success_rate']:.1f}%). "
        "Check its volume and cost before moving more shipments.\n"
        f"- **Cost benchmark:** {lowest_cost['shipping_route']} had the lowest mean shipping "
        f"cost ({money(lowest_cost['average_cost'])}); compare service levels before selecting it.\n"
        "- **Delay prevention:** review high-traffic and storm-prone dispatches and consider "
        "earlier departure windows or contingency buffers.\n"
        "- **Improve the evidence:** replace synthetic examples with validated operational "
        "records before using these findings for business decisions."
    )
