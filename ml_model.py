"""
ml_model.py — Core Machine Learning pipeline for AI Energy Consumption Predictor
─────────────────────────────────────────────────────────────────────────────────
Models    : Linear Regression · Random Forest · ARIMA
Metrics   : MAE · RMSE · R² Score
Extras    : Anomaly Detection · 7-Day Forecast · Plotly Interactive Charts
"""

import warnings
warnings.filterwarnings("ignore")

import io, json
import numpy  as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io            as pio
from plotly.subplots import make_subplots

from sklearn.linear_model  import LinearRegression
from sklearn.ensemble      import RandomForestRegressor
from sklearn.metrics       import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

# ─── CONSTANTS ──────────────────────────────────────────────────────────────
TRAIN_RATIO   = 0.70
FORECAST_DAYS = 7
ANOMALY_STD   = 2.0
MIN_ROWS      = 30     # Minimum required records after cleaning

# ─── COLOUR PALETTE ──────────────────────────────────────────────────────────
COLORS = {
    "actual" : "#60A5FA",   # blue-400
    "lr"     : "#FBBF24",   # amber-400
    "arima"  : "#A78BFA",   # violet-400
    "rf"     : "#34D399",   # emerald-400
    "anomaly": "#F87171",   # red-400
    "forecast": "#22D3EE",  # cyan-400
}

# ════════════════════════════════════════════════════════════════════════════
#  STEP 1 — VALIDATE & LOAD
# ════════════════════════════════════════════════════════════════════════════
def load_and_validate(filepath: str) -> pd.DataFrame:
    """
    Load CSV, check required columns, return raw DataFrame.
    Raises ValueError with user-friendly messages on any validation error.
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise ValueError(f"Cannot read file: {e}")

    # Normalise column names to upper-case for comparison
    df.columns = [c.strip().upper() for c in df.columns]

    if "DATE" not in df.columns:
        raise ValueError("Missing required column: 'Date'. Please ensure your CSV has a Date column.")
    if "KWH" not in df.columns:
        raise ValueError("Missing required column: 'KWH'. Please ensure your CSV has a KWH column.")

    # Rename to standard form
    df.rename(columns={"DATE": "Date", "KWH": "KWH"}, inplace=True)
    return df


# ════════════════════════════════════════════════════════════════════════════
#  STEP 2 — PREPROCESS
# ════════════════════════════════════════════════════════════════════════════
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    • Parse dates (drop un-parseable)
    • Forward-fill then back-fill missing KWH
    • Sort chronologically
    """
    # Parse date column to datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.dropna(subset=["Date"], inplace=True)

    # Handle missing KWH values
    df["KWH"] = pd.to_numeric(df["KWH"], errors="coerce")
    df["KWH"].fillna(method="ffill", inplace=True)
    df["KWH"].fillna(method="bfill", inplace=True)

    # Drop any remaining NaN rows (edge case)
    df.dropna(subset=["KWH"], inplace=True)

    # Sort chronologically — critical for time-series integrity
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    if len(df) < MIN_ROWS:
        raise ValueError(
            f"Dataset too small ({len(df)} usable rows). "
            f"Minimum {MIN_ROWS} records required for reliable predictions."
        )

    return df


# ════════════════════════════════════════════════════════════════════════════
#  STEP 3 — FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based and lag features for ML models.
    ARIMA uses the raw KWH series separately (no feature engineering needed).
    """
    df = df.copy()
    df["DayOfYear"]  = df["Date"].dt.dayofyear
    df["DayOfWeek"]  = df["Date"].dt.dayofweek
    df["Month"]      = df["Date"].dt.month
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["KWH_Lag1"]   = df["KWH"].shift(1)               # Previous day
    df["Rolling7"]   = df["KWH"].shift(1).rolling(7).mean()  # 7-day average
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ════════════════════════════════════════════════════════════════════════════
#  STEP 4 — METRICS HELPER
# ════════════════════════════════════════════════════════════════════════════
def calc_metrics(y_true, y_pred, name: str) -> dict:
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    return {"model": name, "mae": mae, "rmse": rmse, "r2": r2}


# ════════════════════════════════════════════════════════════════════════════
#  STEP 5 — TRAIN MODELS & COMPARE
# ════════════════════════════════════════════════════════════════════════════
FEATURES = ["DayOfYear", "DayOfWeek", "Month", "WeekOfYear", "KWH_Lag1", "Rolling7"]

def train_and_evaluate(df_feat: pd.DataFrame):
    """
    Train LR, ARIMA, RF on the 70% training slice.
    Returns all predictions, metrics list, and fitted objects.
    """
    split = int(len(df_feat) * TRAIN_RATIO)
    train_df = df_feat.iloc[:split]
    test_df  = df_feat.iloc[split:]

    X_train = train_df[FEATURES]
    y_train = train_df["KWH"]
    X_test  = test_df[FEATURES]
    y_test  = test_df["KWH"]

    # Scale features
    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    Xs_test  = scaler.transform(X_test)

    all_metrics = []
    preds       = {}

    # ── Linear Regression ────────────────────────────────────────────────────
    lr = LinearRegression()
    lr.fit(Xs_train, y_train)
    lr_preds = lr.predict(Xs_test)
    m = calc_metrics(y_test, lr_preds, "Linear Regression")
    all_metrics.append(m)
    preds["lr"] = lr_preds.tolist()

    # ── ARIMA(5,1,0) ──────────────────────────────────────────────────────────
    try:
        arima_model = ARIMA(train_df["KWH"].values, order=(5, 1, 0))
        arima_fit   = arima_model.fit()
        arima_fc    = arima_fit.forecast(steps=len(test_df))
        arima_preds = np.array(arima_fc)
    except Exception:
        # Fallback: use training mean repeated
        arima_preds = np.full(len(test_df), y_train.mean())
        arima_fit   = None

    m = calc_metrics(y_test, arima_preds, "ARIMA")
    all_metrics.append(m)
    preds["arima"] = arima_preds.tolist()

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(Xs_train, y_train)
    rf_preds = rf.predict(Xs_test)
    m = calc_metrics(y_test, rf_preds, "Random Forest")
    all_metrics.append(m)
    preds["rf"] = rf_preds.tolist()

    return (train_df, test_df, y_test.values.tolist(),
            preds, all_metrics, lr, rf, arima_fit, scaler)


# ════════════════════════════════════════════════════════════════════════════
#  STEP 6 — ANOMALY DETECTION
# ════════════════════════════════════════════════════════════════════════════
def detect_anomalies(df: pd.DataFrame) -> dict:
    """Threshold = mean ± ANOMALY_STD × std on the complete KWH series."""
    mean = df["KWH"].mean()
    std  = df["KWH"].std()
    upper = mean + ANOMALY_STD * std
    lower = mean - ANOMALY_STD * std
    flags = df[(df["KWH"] > upper) | (df["KWH"] < lower)]
    anomaly_list = []
    for _, row in flags.iterrows():
        anomaly_list.append({
            "date" : str(row["Date"].date()),
            "kwh"  : round(row["KWH"], 2),
            "type" : "HIGH" if row["KWH"] > upper else "LOW"
        })
    return {
        "count"      : len(anomaly_list),
        "upper_limit": round(upper, 2),
        "lower_limit": round(lower, 2),
        "mean"       : round(mean, 2),
        "anomalies"  : anomaly_list,
    }


# ════════════════════════════════════════════════════════════════════════════
#  STEP 7 — 7-DAY FORECAST
# ════════════════════════════════════════════════════════════════════════════
def forecast_7_days(df_feat: pd.DataFrame, best_name: str,
                    lr, rf, arima_fit, scaler) -> list:
    """
    Iteratively predict the next 7 days using the best model.
    For ARIMA, extend the in-sample forecast by 7 more steps.
    For ML models, roll the feature window forward day-by-day.
    """
    last_date = df_feat["Date"].max()
    dates     = pd.date_range(start=last_date + pd.Timedelta(days=1),
                               periods=FORECAST_DAYS, freq="D")
    preds     = []

    if best_name == "ARIMA" and arima_fit is not None:
        try:
            # Extend forecast beyond test horizon
            total_steps = len(df_feat) - int(len(df_feat) * TRAIN_RATIO) + FORECAST_DAYS
            fc = arima_fit.forecast(steps=total_steps)
            preds = [round(float(v), 2) for v in fc[-FORECAST_DAYS:]]
        except Exception:
            pass

    if not preds:
        rolling_buf = list(df_feat["KWH"].iloc[-7:].values)
        prev_kwh    = df_feat["KWH"].iloc[-1]
        for d in dates:
            feat = pd.DataFrame([{
                "DayOfYear" : d.dayofyear,
                "DayOfWeek" : d.dayofweek,
                "Month"     : d.month,
                "WeekOfYear": d.isocalendar()[1],
                "KWH_Lag1"  : prev_kwh,
                "Rolling7"  : float(np.mean(rolling_buf)),
            }])
            feat_s = scaler.transform(feat[FEATURES])
            pred   = rf.predict(feat_s)[0] if best_name == "Random Forest" \
                     else lr.predict(feat_s)[0]
            preds.append(round(float(pred), 2))
            rolling_buf.append(pred)
            rolling_buf.pop(0)
            prev_kwh = pred

    return [{"date": str(d.date()), "kwh": p}
            for d, p in zip(dates, preds)]


# ════════════════════════════════════════════════════════════════════════════
#  STEP 8 — PLOTLY CHARTS (JSON serialisable)
# ════════════════════════════════════════════════════════════════════════════
def build_charts(df_orig: pd.DataFrame, df_feat: pd.DataFrame,
                 test_df: pd.DataFrame, y_test: list,
                 preds: dict, all_metrics: list,
                 forecast: list, anomaly_info: dict,
                 best_name: str) -> dict:
    """
    Returns two Plotly chart dicts (JSON) to be rendered by Plotly.js:
      1. main_chart  — Actual vs Predicted for all 3 models
      2. comp_chart  — Model comparison bar chart (MAE / RMSE / R²)
    """
    test_dates  = [str(d.date()) for d in test_df["Date"]]
    all_dates   = [str(d.date()) for d in df_orig["Date"]]
    fc_dates    = [f["date"] for f in forecast]
    fc_vals     = [f["kwh"]  for f in forecast]

    # ── Chart 1: Actual vs Predicted (3 sub-plots stacked) ──────────────────
    fig1 = make_subplots(
        rows=3, cols=1,
        shared_xaxes=False,
        subplot_titles=(
            "📊 Linear Regression vs Actual",
            "📉 ARIMA vs Actual",
            "🌲 Random Forest vs Actual",
        ),
        vertical_spacing=0.08,
    )
    model_map = [
        ("lr",    "Linear Regression", COLORS["lr"]),
        ("arima", "ARIMA",             COLORS["arima"]),
        ("rf",    "Random Forest",     COLORS["rf"]),
    ]
    for row_idx, (key, label, col) in enumerate(model_map, start=1):
        # Actual line
        fig1.add_trace(go.Scatter(
            x=test_dates, y=y_test,
            name="Actual" if row_idx == 1 else "Actual",
            line=dict(color=COLORS["actual"], width=2),
            showlegend=(row_idx == 1),
        ), row=row_idx, col=1)
        # Predicted line
        fig1.add_trace(go.Scatter(
            x=test_dates, y=preds[key],
            name=label,
            line=dict(color=col, width=2, dash="dash"),
            showlegend=True,
        ), row=row_idx, col=1)

    fig1.update_layout(
        height=700,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E2E8F0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=80, b=40),
    )
    fig1.update_xaxes(gridcolor="rgba(255,255,255,0.05)", showgrid=True)
    fig1.update_yaxes(gridcolor="rgba(255,255,255,0.05)", showgrid=True, title_text="KWH")

    # ── Chart 2: Model Comparison Bar ────────────────────────────────────────
    model_names = [m["model"] for m in all_metrics]
    mae_vals    = [m["mae"]   for m in all_metrics]
    rmse_vals   = [m["rmse"]  for m in all_metrics]
    r2_vals     = [m["r2"]    for m in all_metrics]

    fig2 = make_subplots(
        rows=1, cols=3,
        subplot_titles=("MAE (lower = better)", "RMSE (lower = better)", "R² Score (higher = better)"),
    )
    bar_colors = [COLORS["lr"], COLORS["arima"], COLORS["rf"]]

    for vals, col_idx, ylab in [
        (mae_vals,  1, "MAE"),
        (rmse_vals, 2, "RMSE"),
        (r2_vals,   3, "R²"),
    ]:
        fig2.add_trace(go.Bar(
            x=model_names, y=vals,
            marker_color=bar_colors,
            text=[f"{v:.2f}" for v in vals],
            textposition="outside",
            showlegend=False,
        ), row=1, col=col_idx)
        fig2.update_yaxes(title_text=ylab, row=1, col=col_idx)

    fig2.update_layout(
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E2E8F0"),
        margin=dict(l=40, r=20, t=60, b=80),
    )
    fig2.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    fig2.update_yaxes(gridcolor="rgba(255,255,255,0.05)")

    # ── Chart 3: Anomaly + Full Series ───────────────────────────────────────
    anom_dates = [a["date"] for a in anomaly_info["anomalies"]]
    anom_vals  = [a["kwh"]  for a in anomaly_info["anomalies"]]

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=all_dates, y=df_orig["KWH"].tolist(),
        name="Energy (KWH)", line=dict(color=COLORS["actual"], width=1.8),
    ))
    fig3.add_hline(y=anomaly_info["upper_limit"], line_dash="dash",
                   line_color=COLORS["anomaly"], annotation_text=f"Upper: {anomaly_info['upper_limit']} KWH")
    fig3.add_hline(y=anomaly_info["lower_limit"], line_dash="dash",
                   line_color="#FB923C", annotation_text=f"Lower: {anomaly_info['lower_limit']} KWH")
    fig3.add_hline(y=anomaly_info["mean"], line_dash="dot",
                   line_color="#94A3B8", annotation_text=f"Mean: {anomaly_info['mean']} KWH")
    if anom_dates:
        fig3.add_trace(go.Scatter(
            x=anom_dates, y=anom_vals,
            mode="markers", name=f"Anomalies ({anomaly_info['count']})",
            marker=dict(color=COLORS["anomaly"], size=10, symbol="x",
                        line=dict(color="white", width=1.5)),
        ))
    fig3.update_layout(
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E2E8F0"),
        margin=dict(l=40, r=20, t=40, b=60),
    )
    fig3.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    fig3.update_yaxes(gridcolor="rgba(255,255,255,0.05)", title_text="KWH")

    return {
        "main_chart"    : pio.to_json(fig1),
        "compare_chart" : pio.to_json(fig2),
        "anomaly_chart" : pio.to_json(fig3),
    }


# ════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════
def run_pipeline(filepath: str) -> dict:
    """
    Full pipeline: load → validate → preprocess → train → forecast → charts.
    Returns a single dict with everything the frontend needs.
    """
    # 1. Load & validate
    df_raw = load_and_validate(filepath)

    # 2. Preprocess (clean, sort)
    df_clean = preprocess(df_raw.copy())

    # 3. Feature engineering
    df_feat = engineer_features(df_clean.copy())

    # 4. Train all models
    (train_df, test_df, y_test,
     preds, all_metrics,
     lr, rf, arima_fit, scaler) = train_and_evaluate(df_feat)

    # 5. Best model (lowest RMSE)
    best_metric = min(all_metrics, key=lambda x: x["rmse"])
    best_name   = best_metric["model"]

    # 6. Anomaly detection on clean series
    anomaly_info = detect_anomalies(df_clean)

    # 7. 7-day forecast
    forecast = forecast_7_days(df_feat, best_name, lr, rf, arima_fit, scaler)

    # 8. Build Plotly charts
    charts = build_charts(
        df_clean, df_feat, test_df, y_test,
        preds, all_metrics, forecast,
        anomaly_info, best_name,
    )

    # ── Assemble final response ───────────────────────────────────────────────
    return {
        "status"       : "success",
        "dataset_info" : {
            "total_rows"   : len(df_clean),
            "date_from"    : str(df_clean["Date"].min().date()),
            "date_to"      : str(df_clean["Date"].max().date()),
            "kwh_mean"     : round(df_clean["KWH"].mean(), 2),
            "kwh_min"      : round(df_clean["KWH"].min(), 2),
            "kwh_max"      : round(df_clean["KWH"].max(), 2),
        },
        "models"       : all_metrics,
        "best_model"   : best_name,
        "best_metrics" : best_metric,
        "anomalies"    : anomaly_info,
        "forecast"     : forecast,
        "next_day"     : forecast[0] if forecast else None,
        "charts"       : charts,
    }
