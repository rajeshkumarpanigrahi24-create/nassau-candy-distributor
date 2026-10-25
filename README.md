# Factory-to-Customer Shipping Route Efficiency Analysis

**Nassau Candy Distributor — educational project**

This beginner-friendly Python project analyzes factory-to-customer deliveries, compares routes, predicts shipment delay risk, estimates shipping cost, and presents the results in an interactive Streamlit dashboard.

> **Synthetic data notice:** `data/shipments.csv` is generated for this project. It is not actual Nassau Candy company data and must not be treated as evidence about the company's operations. Replace it only with data you have permission to use, and validate the results before making business decisions.

## Project architecture

```text
shipping-route-efficiency/
├── app.py                         # Streamlit dashboard and shipment prediction form
├── generate_data.py               # Reproducibly generates synthetic shipment records
├── train_model.py                 # Trains, evaluates, and saves both ML models
├── requirements.txt               # Python package dependencies
├── README.md                      # Setup, methods, limitations, and usage guide
├── data/
│   ├── shipments.csv              # Generated synthetic data (created by the command)
│   └── eda_summary.png            # Optional EDA chart (created by notebooks/eda.py)
├── models/
│   ├── delay_classifier.joblib    # Saved delay-risk pipeline
│   ├── shipping_cost_regressor.joblib # Saved shipping-cost pipeline
│   └── metrics.json              # Held-out test metrics and confusion matrix
├── notebooks/
│   └── eda.py                     # Script for basic exploratory analysis and charts
└── src/
    ├── __init__.py
    ├── preprocessing.py           # Shared feature lists, imputation, encoding, and pipelines
    └── evaluation.py               # Classification and regression evaluation helpers
```

## Windows PowerShell setup in VS Code

1. Open this project folder in VS Code and open **Terminal → New Terminal** (PowerShell).
2. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` only if appropriate for your machine, or run the environment's executables directly as shown below.
3. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
4. Generate the synthetic data:

   ```powershell
   python generate_data.py
   ```
5. Train and evaluate the saved models:

   ```powershell
   python train_model.py
   ```
6. Start the dashboard:

   ```powershell
   python -m streamlit run app.py
   ```

   Streamlit prints a local URL (normally `http://localhost:8501`). Select the `.venv` interpreter in VS Code with **Python: Select Interpreter**.

Optional commands:

```powershell
python generate_data.py --rows 2500 --seed 17
python train_model.py --seed 17
python notebooks\eda.py
```

Regenerate data before retraining when you change the row count or seed. The app loads existing Joblib models and does not train during page refresh. If the CSV or saved models are missing, the dashboard explains which setup command to run.

## Dataset fields

The CSV includes `shipment_id`, `factory_location`, `customer_location`, `shipping_route`, `distance_km`, `traffic_level`, `weather_condition`, `shipment_weight_kg`, `shipping_mode`, `shipping_cost`, `dispatch_hour`, `expected_delivery_days`, `actual_delivery_days`, and `delivery_status`. A small amount of missing input data is intentional so the training pipeline demonstrates imputation.

## Methodology

1. **Generate and inspect:** `generate_data.py` produces a reproducible synthetic dataset with a fixed default seed. `notebooks/eda.py` prints data quality and numeric summaries, then saves basic route and cost charts.
2. **Prepare:** `src/preprocessing.py` checks required columns, coerces numeric fields, and defines categorical/numeric features. Numeric missing values are median-imputed and scaled; categorical missing values are most-frequent-imputed and one-hot encoded. Unknown categories at prediction time are ignored by the encoder.
3. **Prevent leakage:** both models use only data available when a shipment is planned: route/factory/customer, distance, traffic, weather, shipment weight, shipping mode, dispatch hour, and expected delivery days. The delay label is created from `actual_delivery_days > expected_delivery_days`, but neither actual delivery days nor `delivery_status` is supplied to the classifier. `shipping_cost` is the regression target and is also excluded from its input features.
4. **Split and train:** a reproducible 80/20 train/test split (seed 42 by default) is stratified by delay status. Two Random Forest pipelines are fitted: a balanced Random Forest classifier for delay risk and a Random Forest regressor for shipping-cost estimates.
5. **Evaluate and save:** held-out predictions are evaluated and written to `models/metrics.json`; fitted preprocessing and estimator pipelines are saved with Joblib in `models/`.
6. **Explore and predict:** the Streamlit dashboard filters shipments, compares routes, displays KPIs/charts and model metrics, and accepts a planned shipment to estimate delay probability and cost.

## Algorithms and metrics

- **Delay classification:** `RandomForestClassifier`; accuracy, precision, recall, F1-score, and a 2x2 confusion matrix.
- **Shipping-cost regression:** `RandomForestRegressor`; mean absolute error (MAE), root mean squared error (RMSE), and R-squared (R²). MAE and RMSE are in the same currency units as the synthetic `shipping_cost` column.
- **Route comparison:** shipment count, on-time success percentage, delay percentage, mean shipping cost, mean delivery duration, and mean distance.

Precision answers how often predicted delays are actually delays; recall answers how many actual delays are found. The dashboard shows both because missing a delay and raising a false alert have different operational implications.

## File purpose

- `generate_data.py`: creates the synthetic records and output directory. Optional `--rows`, `--seed`, and `--output` arguments customize the generation.
- `train_model.py`: loads the CSV, validates fields and target classes, creates the shared train/test split, fits both models, evaluates them, and persists artifacts.
- `src/preprocessing.py`: keeps feature selection and transformations consistent between training and live prediction.
- `src/evaluation.py`: calculates requested held-out classification and regression metrics.
- `app.py`: loads the dataset and saved models, displays the filtered analysis, and provides a dispatch-time prediction form.
- `notebooks/eda.py`: provides a simple repeatable exploratory data analysis script (it is a Python script, not an `.ipynb` notebook).
- `requirements.txt`: lists packages needed to generate data, train models, inspect charts, and run Streamlit.

## Limitations and responsible use

- This synthetic data is designed to demonstrate a workflow; it does not establish real route performance or real cost behavior.
- Random train/test splitting is suitable for the exercise, but production evaluation should consider time-based splits and later shipment cohorts.
- A route's averages can be unstable when it has few observations; always inspect shipment count and operational context.
- A predicted probability or cost is an estimate, not a guarantee. Synthetic data patterns and model scores should not drive actual dispatch decisions.
- The demonstration does not model carrier contracts, capacity constraints, inventory, vehicle routing, or changing fuel prices.

## Future enhancements

- Replace synthetic inputs with consented, validated operational records and add schema/data-quality checks.
- Track calibration, subgroup performance, temporal drift, and model versions.
- Add confidence intervals or prediction ranges, cost-versus-service trade-offs, and route-capacity constraints.
- Include fuel, carrier, service-level, holiday, and delivery-attempt data when available.
- Deploy the dashboard with access controls, monitoring, and secure data handling.
