"""
=============================================================================
  CAMPUS ENERGY CONSUMPTION — PREDICTIVE ANALYTICS SYSTEM
  ─────────────────────────────────────────────────────────
  Models   : Linear Regression · ARIMA · Random Forest
  Metrics  : MAE · RMSE · R² Score
  Extras   : Anomaly Detection · 7-Day Forecast · Plots (saved as PNG)
  Device   : CPU-compatible (no GPU required)
=============================================================================
"""

# ─── 1. IMPORTS ─────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")          # Suppress verbose statsmodels output

import os
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # Non-interactive backend → saves PNGs
import matplotlib.pyplot as plt
import matplotlib.dates  as mdates

from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing   import StandardScaler

from statsmodels.tsa.arima.model import ARIMA

# ─── 2. CONFIGURATION ────────────────────────────────────────────────────────
DATASET_PATH   = "campus_energy_data.csv"   # CSV must have Date & KWH columns
TRAIN_RATIO    = 0.70                        # 70% training / 30% testing
FORECAST_DAYS  = 7                           # Days to predict into the future
ANOMALY_STD    = 2.0                         # Threshold: mean ± N × std
PLOT_DIR       = "plots"                     # Output folder for all plots
os.makedirs(PLOT_DIR, exist_ok=True)

print("=" * 70)
print("  CAMPUS ENERGY CONSUMPTION — PREDICTIVE ANALYTICS SYSTEM")
print("=" * 70)

# ─── 3. LOAD DATA ────────────────────────────────────────────────────────────
print("\n[STEP 1] Loading dataset …")
df = pd.read_csv(DATASET_PATH)
print(f"         Raw records loaded : {len(df)}")
print(f"         Columns found      : {list(df.columns)}")

# ─── 4. PREPROCESSING ────────────────────────────────────────────────────────
print("\n[STEP 2] Preprocessing …")

# 4a. Rename columns to standard names (case-insensitive)
df.columns = [c.strip().upper() for c in df.columns]
df.rename(columns={"DATE": "Date", "KWH": "KWH"}, inplace=True)

# 4b. Parse Date column → datetime
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# 4c. Drop rows where Date parsing failed (un-parseable dates)
date_nulls = df["Date"].isna().sum()
if date_nulls:
    print(f"         Dropped {date_nulls} rows with invalid dates.")
df.dropna(subset=["Date"], inplace=True)

# 4d. Handle missing KWH values — forward-fill then back-fill as fallback
missing_kwh = df["KWH"].isna().sum()
print(f"         Missing KWH values : {missing_kwh}  → filled via forward-fill")
df["KWH"].fillna(method="ffill", inplace=True)
df["KWH"].fillna(method="bfill", inplace=True)   # Handles leading NaNs

# 4e. Sort chronologically (essential for time-series integrity)
df.sort_values("Date", inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"         Final records after cleaning : {len(df)}")
print(f"         Date range : {df['Date'].min().date()}  →  {df['Date'].max().date()}")
print(f"         KWH stats  : min={df['KWH'].min():.1f}  max={df['KWH'].max():.1f}  "
      f"mean={df['KWH'].mean():.1f}")

# ─── 5. FEATURE ENGINEERING FOR ML MODELS ───────────────────────────────────
print("\n[STEP 3] Engineering features for ML models …")

# Time-based features capture seasonality and weekly patterns
df["DayOfYear"]  = df["Date"].dt.dayofyear          # 1–365 seasonal signal
df["DayOfWeek"]  = df["Date"].dt.dayofweek          # 0=Mon … 6=Sun (weekend dip)
df["Month"]      = df["Date"].dt.month               # Monthly seasonality
df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)

# Lag feature: previous day's KWH (strong autocorrelation in energy data)
df["KWH_Lag1"]   = df["KWH"].shift(1)

# Rolling 7-day average to smooth short-term noise
df["Rolling7"]   = df["KWH"].shift(1).rolling(window=7).mean()

# Drop the first few rows where lag/rolling are NaN
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"         Usable records after feature creation : {len(df)}")

# ─── 6. TRAIN / TEST SPLIT (CHRONOLOGICAL) ──────────────────────────────────
print("\n[STEP 4] Splitting data — 70% train / 30% test (chronological) …")

split_idx   = int(len(df) * TRAIN_RATIO)
train_df    = df.iloc[:split_idx].copy()
test_df     = df.iloc[split_idx:].copy()

print(f"         Training samples : {len(train_df)}  "
      f"({train_df['Date'].min().date()} → {train_df['Date'].max().date()})")
print(f"         Testing  samples : {len(test_df)}  "
      f"({test_df['Date'].min().date()} → {test_df['Date'].max().date()})")

FEATURES = ["DayOfYear", "DayOfWeek", "Month", "WeekOfYear", "KWH_Lag1", "Rolling7"]
TARGET   = "KWH"

X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_test,  y_test  = test_df[FEATURES],  test_df[TARGET]

# Scale features (helps Linear Regression; harmless for RF)
scaler   = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ─── 7. HELPER: METRIC CALCULATION ──────────────────────────────────────────
def calc_metrics(y_true, y_pred, model_name="Model"):
    """Return dict with MAE, RMSE, R² for a set of predictions."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {"Model": model_name, "MAE": mae, "RMSE": rmse, "R2": r2}

results = []   # Will hold metric dicts for all models

# ─── 8. MODEL A: LINEAR REGRESSION ──────────────────────────────────────────
print("\n[STEP 5] Training Linear Regression …")

lr_model = LinearRegression()
lr_model.fit(X_train_s, y_train)
lr_preds = lr_model.predict(X_test_s)

lr_metrics = calc_metrics(y_test, lr_preds, "Linear Regression")
results.append(lr_metrics)
print(f"         MAE={lr_metrics['MAE']:.2f}  RMSE={lr_metrics['RMSE']:.2f}  "
      f"R²={lr_metrics['R2']:.4f}")

# ─── 9. MODEL B: ARIMA ───────────────────────────────────────────────────────
print("\n[STEP 6] Training ARIMA(5,1,0) …  (this may take a few seconds)")

# ARIMA works on the raw KWH series (univariate time-series)
arima_train_series = train_df[TARGET].values
arima_test_len     = len(test_df)

try:
    arima_model  = ARIMA(arima_train_series, order=(5, 1, 0))
    arima_fit    = arima_model.fit()

    # Forecast exactly as many steps as the test set
    arima_forecast = arima_fit.forecast(steps=arima_test_len)
    arima_preds    = np.array(arima_forecast)

    arima_metrics = calc_metrics(y_test.values, arima_preds, "ARIMA")
    results.append(arima_metrics)
    print(f"         MAE={arima_metrics['MAE']:.2f}  RMSE={arima_metrics['RMSE']:.2f}  "
          f"R²={arima_metrics['R2']:.4f}")
except Exception as e:
    print(f"         ARIMA failed: {e}  — skipping.")
    arima_preds   = np.full(arima_test_len, y_train.mean())
    arima_metrics = calc_metrics(y_test.values, arima_preds, "ARIMA (fallback)")
    results.append(arima_metrics)

# ─── 10. MODEL C: RANDOM FOREST ──────────────────────────────────────────────
print("\n[STEP 7] Training Random Forest (200 trees) …")

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1          # Use all available CPU cores
)
rf_model.fit(X_train_s, y_train)
rf_preds = rf_model.predict(X_test_s)

rf_metrics = calc_metrics(y_test, rf_preds, "Random Forest")
results.append(rf_metrics)
print(f"         MAE={rf_metrics['MAE']:.2f}  RMSE={rf_metrics['RMSE']:.2f}  "
      f"R²={rf_metrics['R2']:.4f}")

# ─── 11. MODEL COMPARISON TABLE ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("  MODEL COMPARISON RESULTS")
print("=" * 70)

metrics_df = pd.DataFrame(results)
print(f"\n{'Model':<22} {'MAE':>10} {'RMSE':>10} {'R² Score':>12}")
print("-" * 56)
for _, row in metrics_df.iterrows():
    print(f"  {row['Model']:<20} {row['MAE']:>10.2f} {row['RMSE']:>10.2f} {row['R2']:>12.4f}")
print("-" * 56)

# Determine best model by lowest RMSE (robust to scale)
best_idx   = metrics_df["RMSE"].idxmin()
best_model = metrics_df.loc[best_idx, "Model"]
print(f"\n  ✅  Best Model : {best_model}  (lowest RMSE = {metrics_df.loc[best_idx,'RMSE']:.2f})")

# ─── 12. ANOMALY DETECTION ───────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  ANOMALY DETECTION  (Threshold = Mean ± {:.0f}×Std)".format(ANOMALY_STD))
print("=" * 70)

# Compute global statistics on the full cleaned series
kwh_mean   = df["KWH"].mean()
kwh_std    = df["KWH"].std()
upper_lim  = kwh_mean + ANOMALY_STD * kwh_std
lower_lim  = kwh_mean - ANOMALY_STD * kwh_std

anomalies  = df[(df["KWH"] > upper_lim) | (df["KWH"] < lower_lim)].copy()

print(f"\n  Normal range : {lower_lim:.1f} KWH  →  {upper_lim:.1f} KWH")
print(f"  Anomalies detected : {len(anomalies)}\n")

if not anomalies.empty:
    print(f"  {'Date':<14} {'KWH':>10}  {'Deviation':>12}")
    print("  " + "-" * 38)
    for _, row in anomalies.iterrows():
        dev = row["KWH"] - kwh_mean
        flag = "↑ HIGH" if row["KWH"] > upper_lim else "↓ LOW"
        print(f"  {str(row['Date'].date()):<14} {row['KWH']:>10.1f}  "
              f"({dev:+.1f})  {flag}")
else:
    print("  No anomalies found — energy usage looks normal throughout.")

# ─── 13. 7-DAY FORECAST ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"  7-DAY FORECAST  (using {best_model})")
print("=" * 70)

last_date       = df["Date"].max()
last_kwh        = df["KWH"].iloc[-1]
rolling7_last   = df["KWH"].iloc[-7:].mean()

forecast_dates  = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                periods=FORECAST_DAYS, freq="D")
forecast_preds  = []

# ── ARIMA 7-day forecast (preferred when ARIMA is best) ─────────────────────
if best_model.startswith("ARIMA"):
    try:
        future_fc = arima_fit.forecast(steps=arima_test_len + FORECAST_DAYS)
        arima_7   = future_fc[-FORECAST_DAYS:]
        forecast_preds = list(arima_7)
    except Exception:
        forecast_preds = []

# ── ML-based iterative forecast (Linear Regression or Random Forest) ─────────
if not forecast_preds:
    # Roll forward one day at a time, updating lag & rolling window
    rolling_buffer = list(df["KWH"].iloc[-7:].values)  # last 7 days as buffer
    prev_kwh       = last_kwh

    for future_date in forecast_dates:
        feat = pd.DataFrame([{
            "DayOfYear"  : future_date.dayofyear,
            "DayOfWeek"  : future_date.dayofweek,
            "Month"      : future_date.month,
            "WeekOfYear" : future_date.isocalendar()[1],
            "KWH_Lag1"   : prev_kwh,
            "Rolling7"   : np.mean(rolling_buffer)
        }])
        feat_s   = scaler.transform(feat[FEATURES])

        if best_model == "Random Forest":
            pred = rf_model.predict(feat_s)[0]
        else:
            pred = lr_model.predict(feat_s)[0]

        forecast_preds.append(pred)
        rolling_buffer.append(pred)
        rolling_buffer.pop(0)        # Keep buffer at 7 elements
        prev_kwh = pred

# ── Print forecast table ──────────────────────────────────────────────────────
print(f"\n  {'Day':<6} {'Date':<14} {'Predicted KWH':>15}")
print("  " + "-" * 36)
for i, (fdate, fpred) in enumerate(zip(forecast_dates, forecast_preds), 1):
    print(f"  Day {i}  {str(fdate.date()):<14} {fpred:>15.2f} KWH")

next_day_pred = forecast_preds[0]

# ─── 14. VISUALIZATION ───────────────────────────────────────────────────────
print("\n[PLOTTING] Generating charts …")

test_dates = test_df["Date"].values

# ── Color palette ────────────────────────────────────────────────────────────
C_ACTUAL = "#2196F3"
C_LR     = "#FF9800"
C_ARIMA  = "#9C27B0"
C_RF     = "#4CAF50"
C_ANOM   = "#F44336"
C_FC     = "#00BCD4"

# ── PLOT 1: All Models vs Actual ─────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=False)
fig.suptitle("Campus Energy Consumption — Model Predictions vs Actual",
             fontsize=15, fontweight="bold", y=1.01)

model_data = [
    ("Linear Regression", lr_preds,    C_LR),
    ("ARIMA",             arima_preds, C_ARIMA),
    ("Random Forest",     rf_preds,    C_RF),
]

for ax, (mname, mpreds, mcol) in zip(axes, model_data):
    ax.plot(test_dates, y_test.values, color=C_ACTUAL, linewidth=2,
            label="Actual", zorder=3)
    ax.plot(test_dates, mpreds, color=mcol, linewidth=1.8, linestyle="--",
            label=mname, alpha=0.9, zorder=2)
    ax.fill_between(test_dates, y_test.values, mpreds,
                    alpha=0.12, color=mcol)
    ax.set_title(mname, fontsize=12, pad=6)
    ax.set_ylabel("Energy (KWH)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

axes[-1].set_xlabel("Date")
plt.tight_layout()
plot1_path = os.path.join(PLOT_DIR, "01_actual_vs_predicted.png")
plt.savefig(plot1_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"         Saved → {plot1_path}")

# ── PLOT 2: Anomaly Detection ─────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(14, 5))
ax2.plot(df["Date"], df["KWH"], color=C_ACTUAL, linewidth=1.4,
         label="Energy (KWH)", zorder=2)
ax2.axhline(upper_lim, color=C_ANOM, linestyle="--", linewidth=1.5,
            label=f"Upper limit ({upper_lim:.0f} KWH)", alpha=0.8)
ax2.axhline(lower_lim, color="#FF5722", linestyle="--", linewidth=1.5,
            label=f"Lower limit ({lower_lim:.0f} KWH)", alpha=0.8)
ax2.axhline(kwh_mean, color="#607D8B", linestyle=":", linewidth=1.2,
            label=f"Mean ({kwh_mean:.0f} KWH)", alpha=0.8)

if not anomalies.empty:
    ax2.scatter(anomalies["Date"], anomalies["KWH"],
                color=C_ANOM, s=70, zorder=5,
                label=f"Anomalies ({len(anomalies)})", edgecolors="black", linewidths=0.5)

ax2.set_title("Anomaly Detection — Campus Energy Consumption",
              fontsize=13, fontweight="bold")
ax2.set_xlabel("Date")
ax2.set_ylabel("Energy (KWH)")
ax2.legend(loc="upper left", fontsize=9)
ax2.grid(True, linestyle=":", alpha=0.5)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax2.xaxis.set_major_locator(mdates.MonthLocator())
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
plt.tight_layout()
plot2_path = os.path.join(PLOT_DIR, "02_anomaly_detection.png")
plt.savefig(plot2_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"         Saved → {plot2_path}")

# ── PLOT 3: 7-Day Forecast ────────────────────────────────────────────────────
recent_n    = 60                             # Show last 60 days for context
recent_df   = df.tail(recent_n)

fig3, ax3 = plt.subplots(figsize=(14, 5))
ax3.plot(recent_df["Date"], recent_df["KWH"],
         color=C_ACTUAL, linewidth=2, label="Historical (last 60 days)")
ax3.plot(forecast_dates, forecast_preds,
         color=C_FC, linewidth=2.2, linestyle="--",
         marker="o", markersize=6, label=f"7-Day Forecast ({best_model})")

# Confidence band (±5% for visual appeal — approximate uncertainty)
fc_arr = np.array(forecast_preds)
ax3.fill_between(forecast_dates,
                 fc_arr * 0.95, fc_arr * 1.05,
                 color=C_FC, alpha=0.20, label="±5% confidence band")

# Vertical separator at forecast start
ax3.axvline(x=last_date, color="#9E9E9E", linestyle=":", linewidth=1.5)
ax3.text(last_date, ax3.get_ylim()[0], " Forecast\n Start",
         fontsize=8, color="#9E9E9E", va="bottom")

ax3.set_title(f"7-Day Energy Forecast — {best_model}",
              fontsize=13, fontweight="bold")
ax3.set_xlabel("Date")
ax3.set_ylabel("Energy (KWH)")
ax3.legend(loc="upper left", fontsize=9)
ax3.grid(True, linestyle=":", alpha=0.5)
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax3.xaxis.set_major_locator(mdates.WeekdayLocator(interval=3))
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right")
plt.tight_layout()
plot3_path = os.path.join(PLOT_DIR, "03_seven_day_forecast.png")
plt.savefig(plot3_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"         Saved → {plot3_path}")

# ── PLOT 4: Model Accuracy Bar Chart ──────────────────────────────────────────
fig4, axes4 = plt.subplots(1, 3, figsize=(13, 5))
fig4.suptitle("Model Performance Comparison", fontsize=14, fontweight="bold")

metric_names  = ["MAE", "RMSE", "R²"]
metric_keys   = ["MAE", "RMSE", "R2"]
colors4       = [C_LR, C_ARIMA, C_RF]
model_labels  = metrics_df["Model"].tolist()

for ax, mkey, xlabel in zip(axes4, metric_keys, metric_names):
    vals = metrics_df[mkey].tolist()
    bars = ax.bar(model_labels, vals, color=colors4, edgecolor="white",
                  linewidth=1.2, zorder=3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title(xlabel, fontsize=11)
    ax.set_ylabel(xlabel)
    ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
    ax.set_xticks(range(len(model_labels)))
    ax.set_xticklabels(model_labels, rotation=15, ha="right", fontsize=8)

plt.tight_layout()
plot4_path = os.path.join(PLOT_DIR, "04_model_comparison.png")
plt.savefig(plot4_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"         Saved → {plot4_path}")

# ─── 15. FINAL OUTPUT SUMMARY ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  ██████  FINAL RESULTS SUMMARY  ██████")
print("=" * 70)

print(f"\n  🏆  Best Model           : {best_model}")
print(f"      → MAE              = {metrics_df.loc[best_idx, 'MAE']:.2f} KWH")
print(f"      → RMSE             = {metrics_df.loc[best_idx, 'RMSE']:.2f} KWH")
print(f"      → R² Score         = {metrics_df.loc[best_idx, 'R2']:.4f}")

print(f"\n  📅  Next Day Prediction  : {str((last_date + pd.Timedelta(days=1)).date())}")
print(f"      → Predicted KWH    = {next_day_pred:.2f} KWH")

print(f"\n  📆  7-Day Forecast Summary :")
for i, (fdate, fpred) in enumerate(zip(forecast_dates, forecast_preds), 1):
    print(f"      Day {i}  ({str(fdate.date())})  →  {fpred:.2f} KWH")

print(f"\n  ⚠️   Anomalies Detected    : {len(anomalies)} spike(s) outside normal range")

print("\n  📊  Plots saved to:")
for p in [plot1_path, plot2_path, plot3_path, plot4_path]:
    print(f"      • {p}")

print("\n" + "=" * 70)
print("  Analysis complete. All outputs saved successfully.")
print("=" * 70 + "\n")
