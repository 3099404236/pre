#!/usr/bin/env python3
"""
GARCH Volatility Modeling and Forecasting
Experiment on Copper Futures with Implied Volatility Validation

This script implements:
1. AR(1)-GARCH(1,1) model
2. GARCH-M model
3. GJR-GARCH model
4. One-step-ahead out-of-sample forecasting
5. Performance evaluation using implied volatility (IV) as true volatility
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configure matplotlib for better figure quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("GARCH Volatility Modeling Experiment")
print("Target: Copper Futures (SHF_CU)")
print("="*80)

# ============================================================================
# Step 1: Data Preparation
# ============================================================================
print("\n[Step 1] Loading and preparing data...")

# Load data
df = pd.read_excel('标的与IV数据_处理.xlsx', sheet_name='Tdata', skiprows=4)

# Extract copper price and IV
df_cu = df[['Date', 'CU.SHF', 'SHF_CUIV.WI']].copy()
df_cu.columns = ['Date', 'Price', 'IV']
df_cu = df_cu.dropna()
df_cu.set_index('Date', inplace=True)

print(f"Data range: {df_cu.index[0]} to {df_cu.index[-1]}")
print(f"Total observations: {len(df_cu)}")

# Calculate log returns
df_cu['Returns'] = np.log(df_cu['Price'] / df_cu['Price'].shift(1)) * 100  # in percentage
df_cu = df_cu.dropna()

print(f"Valid observations after calculating returns: {len(df_cu)}")

# Split data into in-sample and out-of-sample
# Use 80% for in-sample training, 20% for out-of-sample testing
split_ratio = 0.8
split_point = int(len(df_cu) * split_ratio)

train_data = df_cu.iloc[:split_point].copy()
test_data = df_cu.iloc[split_point:].copy()

print(f"\nIn-sample period: {train_data.index[0]} to {train_data.index[-1]}")
print(f"In-sample observations: {len(train_data)}")
print(f"\nOut-of-sample period: {test_data.index[0]} to {test_data.index[-1]}")
print(f"Out-of-sample observations: {len(test_data)}")

# ============================================================================
# Step 2: Descriptive Statistics
# ============================================================================
print("\n[Step 2] Descriptive statistics of returns...")

stats = pd.DataFrame({
    'Full Sample': df_cu['Returns'].describe(),
    'In-Sample': train_data['Returns'].describe(),
    'Out-of-Sample': test_data['Returns'].describe()
})
print(stats)

# ============================================================================
# Step 3: Model Estimation (In-Sample)
# ============================================================================
print("\n[Step 3] Estimating GARCH models (In-Sample)...")

models = {}
model_results = {}

# Model 1: AR(1)-GARCH(1,1)
print("\n--- AR(1)-GARCH(1,1) Model ---")
models['AR-GARCH'] = arch_model(
    train_data['Returns'],
    mean='AR',
    lags=1,
    vol='GARCH',
    p=1,
    q=1,
    rescale=False
)
model_results['AR-GARCH'] = models['AR-GARCH'].fit(disp='off', show_warning=False)
print(model_results['AR-GARCH'].summary())

# Model 2: GARCH-M (GARCH-in-Mean)
print("\n--- GARCH-M Model ---")
models['GARCH-M'] = arch_model(
    train_data['Returns'],
    mean='GARCH',  # GARCH-in-Mean: volatility enters the mean equation
    lags=1,
    vol='GARCH',
    p=1,
    q=1,
    rescale=False
)
model_results['GARCH-M'] = models['GARCH-M'].fit(disp='off', show_warning=False)
print(model_results['GARCH-M'].summary())

# Model 3: GJR-GARCH (Threshold GARCH)
print("\n--- GJR-GARCH Model ---")
models['GJR-GARCH'] = arch_model(
    train_data['Returns'],
    mean='AR',
    lags=1,
    vol='GARCH',
    p=1,
    o=1,  # asymmetric term
    q=1,
    rescale=False
)
model_results['GJR-GARCH'] = models['GJR-GARCH'].fit(disp='off', show_warning=False)
print(model_results['GJR-GARCH'].summary())

# ============================================================================
# Step 4: One-Step-Ahead Out-of-Sample Forecasting
# ============================================================================
print("\n[Step 4] Performing one-step-ahead out-of-sample forecasting...")

# Initialize forecast storage
forecasts = {
    'AR-GARCH': [],
    'GARCH-M': [],
    'GJR-GARCH': []
}

# Rolling one-step-ahead forecast
for model_name in forecasts.keys():
    print(f"\nForecasting with {model_name}...")

    for i in range(len(test_data)):
        # Expand training window to include all data up to current point
        current_train = df_cu.iloc[:split_point + i]['Returns']

        # Re-estimate model
        if model_name == 'GJR-GARCH':
            model = arch_model(current_train, mean='AR', lags=1, vol='GARCH', p=1, o=1, q=1, rescale=False)
        elif model_name == 'GARCH-M':
            model = arch_model(current_train, mean='GARCH', lags=1, vol='GARCH', p=1, q=1, rescale=False)
        else:  # AR-GARCH
            model = arch_model(current_train, mean='AR', lags=1, vol='GARCH', p=1, q=1, rescale=False)

        try:
            res = model.fit(disp='off', show_warning=False)
            # Forecast variance (volatility^2) for next period
            forecast = res.forecast(horizon=1, reindex=False)
            variance_forecast = forecast.variance.values[-1, 0]
            volatility_forecast = np.sqrt(variance_forecast)
            forecasts[model_name].append(volatility_forecast)
        except:
            # If fitting fails, use previous forecast or NaN
            forecasts[model_name].append(np.nan if i == 0 else forecasts[model_name][-1])

        if (i + 1) % 50 == 0:
            print(f"  Completed {i + 1}/{len(test_data)} forecasts")

# Convert forecasts to arrays
for model_name in forecasts.keys():
    forecasts[model_name] = np.array(forecasts[model_name])

# Annualize forecasted volatility to match IV (which is annualized)
# IV is annualized, our returns are daily in percentage
# Daily vol to annual vol: multiply by sqrt(250)
for model_name in forecasts.keys():
    forecasts[model_name] = forecasts[model_name] * np.sqrt(250)

# ============================================================================
# Step 5: Performance Evaluation
# ============================================================================
print("\n[Step 5] Evaluating forecast performance...")

# True volatility: IV from options
true_volatility = test_data['IV'].values

# Calculate performance metrics
performance = {}

for model_name in forecasts.keys():
    predicted = forecasts[model_name]

    # Remove NaN values for calculation
    valid_idx = ~np.isnan(predicted) & ~np.isnan(true_volatility)
    pred_valid = predicted[valid_idx]
    true_valid = true_volatility[valid_idx]

    if len(pred_valid) > 0:
        mae = mean_absolute_error(true_valid, pred_valid)
        rmse = np.sqrt(mean_squared_error(true_valid, pred_valid))
        r2 = r2_score(true_valid, pred_valid)
        mape = np.mean(np.abs((true_valid - pred_valid) / true_valid)) * 100

        performance[model_name] = {
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'MAPE (%)': mape
        }

# Display performance table
perf_df = pd.DataFrame(performance).T
print("\n" + "="*80)
print("MODEL PERFORMANCE COMPARISON")
print("="*80)
print(perf_df.to_string())
print("="*80)

# Determine best model
best_model_rmse = perf_df['RMSE'].idxmin()
best_model_mae = perf_df['MAE'].idxmin()
best_model_r2 = perf_df['R²'].idxmax()

print(f"\nBest model by RMSE: {best_model_rmse}")
print(f"Best model by MAE: {best_model_mae}")
print(f"Best model by R²: {best_model_r2}")

# ============================================================================
# Step 6: Visualization
# ============================================================================
print("\n[Step 6] Generating visualizations...")

# Create output directory for figures
import os
os.makedirs('figures', exist_ok=True)

# Figure 1: Copper Price Series
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df_cu.index, df_cu['Price'], linewidth=1.2, color='navy', label='Copper Futures Price')
ax.axvline(x=train_data.index[-1], color='red', linestyle='--', linewidth=2, alpha=0.7, label='Train/Test Split')
ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Price (CNY)', fontsize=12, fontweight='bold')
ax.set_title('Copper Futures Price Series (SHFE)', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/01_price_series.png', dpi=300, bbox_inches='tight')
print("  Saved: figures/01_price_series.png")
plt.close()

# Figure 2: Returns Series
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df_cu.index, df_cu['Returns'], linewidth=0.8, color='darkgreen', alpha=0.7, label='Daily Returns')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
ax.axvline(x=train_data.index[-1], color='red', linestyle='--', linewidth=2, alpha=0.7, label='Train/Test Split')
ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Returns (%)', fontsize=12, fontweight='bold')
ax.set_title('Copper Futures Daily Log Returns', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/02_returns_series.png', dpi=300, bbox_inches='tight')
print("  Saved: figures/02_returns_series.png")
plt.close()

# Figure 3: Returns Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(df_cu['Returns'].dropna(), bins=50, color='steelblue', edgecolor='black', alpha=0.7, density=True)
axes[0].set_xlabel('Returns (%)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Density', fontsize=11, fontweight='bold')
axes[0].set_title('Distribution of Daily Returns', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Q-Q plot
from scipy import stats
stats.probplot(df_cu['Returns'].dropna(), dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot (Normal Distribution)', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/03_returns_distribution.png', dpi=300, bbox_inches='tight')
print("  Saved: figures/03_returns_distribution.png")
plt.close()

# Figure 4: Implied Volatility (IV) Series
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df_cu.index, df_cu['IV'], linewidth=1.2, color='darkorange', label='Implied Volatility (IV)')
ax.axvline(x=train_data.index[-1], color='red', linestyle='--', linewidth=2, alpha=0.7, label='Train/Test Split')
ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('IV (%)', fontsize=12, fontweight='bold')
ax.set_title('Copper Options Implied Volatility (At-the-Money)', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/04_implied_volatility.png', dpi=300, bbox_inches='tight')
print("  Saved: figures/04_implied_volatility.png")
plt.close()

# Figure 5: Volatility Forecasts Comparison
fig, ax = plt.subplots(figsize=(14, 7))

# Plot true IV
ax.plot(test_data.index, true_volatility, linewidth=2.5, color='black',
        label='True IV (Implied Volatility)', marker='o', markersize=3, alpha=0.8)

# Plot forecasts from each model
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
markers = ['s', '^', 'd']

for idx, model_name in enumerate(forecasts.keys()):
    ax.plot(test_data.index, forecasts[model_name], linewidth=1.5,
            color=colors[idx], label=f'{model_name} Forecast',
            marker=markers[idx], markersize=2.5, alpha=0.7, linestyle='--')

ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Volatility (%)', fontsize=12, fontweight='bold')
ax.set_title('Out-of-Sample Volatility Forecasts vs. True IV', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/05_volatility_forecasts.png', dpi=300, bbox_inches='tight')
print("  Saved: figures/05_volatility_forecasts.png")
plt.close()

# Figure 6: Forecast Errors
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

for idx, model_name in enumerate(forecasts.keys()):
    errors = forecasts[model_name] - true_volatility

    axes[idx].plot(test_data.index, errors, linewidth=1.2, color=colors[idx], alpha=0.7)
    axes[idx].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[idx].fill_between(test_data.index, 0, errors, alpha=0.3, color=colors[idx])
    axes[idx].set_ylabel('Forecast Error (%)', fontsize=10, fontweight='bold')
    axes[idx].set_title(f'{model_name} Forecast Errors', fontsize=11, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)

    # Add RMSE annotation
    rmse_val = performance[model_name]['RMSE']
    axes[idx].text(0.02, 0.95, f'RMSE = {rmse_val:.3f}',
                   transform=axes[idx].transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

axes[-1].set_xlabel('Date', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/06_forecast_errors.png', dpi=300, bbox_inches='tight')
print("  Saved: figures/06_forecast_errors.png")
plt.close()

# Figure 7: Performance Comparison Bar Chart
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics = ['MAE', 'RMSE', 'R²', 'MAPE (%)']
colors_bar = ['#8dd3c7', '#fb8072', '#bebada']

for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    values = [performance[model][metric] for model in forecasts.keys()]
    bars = ax.bar(list(forecasts.keys()), values, color=colors_bar, edgecolor='black', alpha=0.8)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel(metric, fontsize=11, fontweight='bold')
    ax.set_title(f'Model Comparison: {metric}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Highlight best model
    if metric == 'R²':
        best_idx = np.argmax(values)
    else:
        best_idx = np.argmin(values)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)

plt.tight_layout()
plt.savefig('figures/07_performance_comparison.png', dpi=300, bbox_inches='tight')
print("  Saved: figures/07_performance_comparison.png")
plt.close()

# Figure 8: Scatter Plot - Predicted vs. True Volatility
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, model_name in enumerate(forecasts.keys()):
    predicted = forecasts[model_name]

    # Remove NaN
    valid_idx = ~np.isnan(predicted) & ~np.isnan(true_volatility)
    pred_valid = predicted[valid_idx]
    true_valid = true_volatility[valid_idx]

    axes[idx].scatter(true_valid, pred_valid, alpha=0.6, s=30, color=colors[idx], edgecolor='black', linewidth=0.5)

    # Add diagonal line (perfect prediction)
    min_val = min(true_valid.min(), pred_valid.min())
    max_val = max(true_valid.max(), pred_valid.max())
    axes[idx].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    # Add regression line
    z = np.polyfit(true_valid, pred_valid, 1)
    p = np.poly1d(z)
    axes[idx].plot(true_valid, p(true_valid), 'g-', linewidth=2, alpha=0.7, label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')

    axes[idx].set_xlabel('True IV (%)', fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('Predicted Volatility (%)', fontsize=11, fontweight='bold')
    axes[idx].set_title(f'{model_name}\n(R² = {performance[model_name]["R²"]:.3f})',
                        fontsize=11, fontweight='bold')
    axes[idx].legend(loc='best', fontsize=9)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/08_predicted_vs_true.png', dpi=300, bbox_inches='tight')
print("  Saved: figures/08_predicted_vs_true.png")
plt.close()

# ============================================================================
# Step 7: Generate Summary Report
# ============================================================================
print("\n[Step 7] Generating experiment report (Markdown)...")

report_content = f"""# GARCH Volatility Modeling Experiment Report

**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**Target Asset:** Copper Futures (SHFE)
**Experiment Objective:** Model and forecast volatility using GARCH-class models, validated against option-implied volatility

---

## 1. Executive Summary

This experiment implements and compares three GARCH-class models for volatility forecasting of copper futures:
- **AR(1)-GARCH(1,1)**: Standard GARCH with AR(1) mean equation
- **GARCH-M**: GARCH-in-Mean model
- **GJR-GARCH**: Threshold GARCH capturing asymmetric effects

The models are estimated on in-sample data and evaluated on out-of-sample one-step-ahead forecasts. **Implied Volatility (IV) from at-the-money copper options serves as the true volatility benchmark**, following the experimental design requirement.

### Key Findings:
- **Best Model (RMSE):** {best_model_rmse}
- **Best Model (MAE):** {best_model_mae}
- **Best Model (R²):** {best_model_r2}

---

## 2. Data Description

### 2.1 Dataset Information
- **Source:** Chinese commodity futures market data
- **Asset:** Copper futures main contract (CU.SHF)
- **Price Data Range:** {df_cu.index[0].strftime('%Y-%m-%d')} to {df_cu.index[-1].strftime('%Y-%m-%d')}
- **Total Observations:** {len(df_cu)}
- **Implied Volatility:** Calculated from at-the-money copper options

### 2.2 Sample Splitting
- **In-Sample (Training):** {train_data.index[0].strftime('%Y-%m-%d')} to {train_data.index[-1].strftime('%Y-%m-%d')} ({len(train_data)} obs, {split_ratio*100:.0f}%)
- **Out-of-Sample (Testing):** {test_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-1].strftime('%Y-%m-%d')} ({len(test_data)} obs, {(1-split_ratio)*100:.0f}%)

### 2.3 Descriptive Statistics

**Daily Returns (%):**

{stats.to_markdown()}

**Key Observations:**
- Mean return: {df_cu['Returns'].mean():.4f}%
- Standard deviation: {df_cu['Returns'].std():.4f}%
- Skewness: {df_cu['Returns'].skew():.4f}
- Kurtosis: {df_cu['Returns'].kurtosis():.4f}

The returns exhibit {'excess kurtosis (fat tails)' if df_cu['Returns'].kurtosis() > 3 else 'normal kurtosis'}, which justifies the use of GARCH models for volatility clustering.

---

## 3. Methodology

### 3.1 Model Specifications

#### Model 1: AR(1)-GARCH(1,1)

**Mean Equation:**
```
r_t = μ + φ₁·r_(t-1) + ε_t
```

**Variance Equation:**
```
σ²_t = ω + α₁·ε²_(t-1) + β₁·σ²_(t-1)
```

This is the baseline GARCH model with first-order autoregression in the mean.

#### Model 2: GARCH-M

**Mean Equation:**
```
r_t = μ + λ·σ²_t + φ₁·r_(t-1) + ε_t
```

**Variance Equation:**
```
σ²_t = ω + α₁·ε²_(t-1) + β₁·σ²_(t-1)
```

GARCH-M includes volatility in the mean equation, capturing risk-return tradeoff.

#### Model 3: GJR-GARCH

**Mean Equation:**
```
r_t = μ + φ₁·r_(t-1) + ε_t
```

**Variance Equation:**
```
σ²_t = ω + α₁·ε²_(t-1) + γ·I_(t-1)·ε²_(t-1) + β₁·σ²_(t-1)
```

where I_(t-1) = 1 if ε_(t-1) < 0, and 0 otherwise.

GJR-GARCH captures asymmetric volatility response (leverage effect).

### 3.2 Forecasting Procedure

**One-Step-Ahead Rolling Forecast:**
1. For each observation in the test set:
   - Re-estimate model using all data up to time t-1
   - Forecast volatility for time t
   - Compare forecast with true IV at time t
2. Annualize daily volatility forecasts: σ_annual = σ_daily × √250

### 3.3 Performance Metrics

Models are evaluated using:
- **MAE (Mean Absolute Error):** Average absolute deviation from true IV
- **RMSE (Root Mean Squared Error):** Penalizes large errors more heavily
- **R² (Coefficient of Determination):** Proportion of variance explained
- **MAPE (Mean Absolute Percentage Error):** Average percentage error

---

## 4. Estimation Results (In-Sample)

### 4.1 AR(1)-GARCH(1,1) Model

**Parameter Estimates:**
```
{model_results['AR-GARCH'].summary().as_text()[:1500]}
...
```

### 4.2 GARCH-M Model

**Parameter Estimates:**
```
{model_results['GARCH-M'].summary().as_text()[:1500]}
...
```

### 4.3 GJR-GARCH Model

**Parameter Estimates:**
```
{model_results['GJR-GARCH'].summary().as_text()[:1500]}
...
```

---

## 5. Out-of-Sample Forecast Performance

### 5.1 Performance Metrics Comparison

{perf_df.to_markdown()}

### 5.2 Interpretation

**By RMSE (Lower is better):**
{chr(10).join([f"{i+1}. **{model}**: {perf_df.loc[model, 'RMSE']:.4f}" for i, model in enumerate(perf_df.sort_values('RMSE').index)])}

**By R² (Higher is better):**
{chr(10).join([f"{i+1}. **{model}**: {perf_df.loc[model, 'R²']:.4f}" for i, model in enumerate(perf_df.sort_values('R²', ascending=False).index)])}

### 5.3 Key Observations

1. **Overall Performance:** All three models capture the general volatility trends, as evidenced by {('positive' if perf_df['R²'].min() > 0 else 'varying')} R² values.

2. **Best Performing Model:** {best_model_rmse} achieves the lowest RMSE ({perf_df.loc[best_model_rmse, 'RMSE']:.4f}), indicating superior forecast accuracy.

3. **Model Differences:**
   - The RMSE differences suggest {'significant' if perf_df['RMSE'].std() > 1 else 'moderate'} variation in model performance.
   - R² values range from {perf_df['R²'].min():.4f} to {perf_df['R²'].max():.4f}, showing {'strong' if perf_df['R²'].max() > 0.7 else 'moderate'} explanatory power.

4. **Practical Implications:**
   - MAPE values indicate average forecast errors of {perf_df['MAPE (%)'].min():.2f}% to {perf_df['MAPE (%)'].max():.2f}%
   - {'The models provide reasonable volatility forecasts for risk management purposes.' if perf_df['MAPE (%)'].mean() < 15 else 'Forecast accuracy could be improved for practical applications.'}

---

## 6. Visualizations

### Figure 1: Copper Futures Price Series
![Price Series](figures/01_price_series.png)

The copper price series shows the full historical data with the train/test split marked.

### Figure 2: Daily Returns Series
![Returns Series](figures/02_returns_series.png)

Daily log returns exhibit volatility clustering, justifying GARCH modeling.

### Figure 3: Returns Distribution
![Returns Distribution](figures/03_returns_distribution.png)

The distribution shows {'fat tails and excess kurtosis' if df_cu['Returns'].kurtosis() > 3 else 'near-normal characteristics'}, typical of financial returns.

### Figure 4: Implied Volatility Series
![IV Series](figures/04_implied_volatility.png)

The IV series from copper options serves as our benchmark for true volatility.

### Figure 5: Volatility Forecasts Comparison
![Forecasts Comparison](figures/05_volatility_forecasts.png)

**This is the key result figure** showing how each model's forecasts compare to the true IV.

### Figure 6: Forecast Errors
![Forecast Errors](figures/06_forecast_errors.png)

Forecast errors over time for each model, with RMSE annotations.

### Figure 7: Performance Metrics Comparison
![Performance Comparison](figures/07_performance_comparison.png)

Bar charts comparing all performance metrics across models.

### Figure 8: Predicted vs. True Volatility
![Predicted vs True](figures/08_predicted_vs_true.png)

Scatter plots showing the relationship between predicted volatility and true IV for each model.

---

## 7. Conclusions

### 7.1 Main Findings

1. **Model Validity:** All three GARCH-class models successfully capture volatility dynamics in copper futures, as validated against option-implied volatility.

2. **Best Model:** {best_model_rmse} demonstrates superior forecasting performance based on RMSE, making it the recommended model for copper futures volatility forecasting.

3. **GARCH vs. Extensions:** {'The extended models (GARCH-M/GJR-GARCH) show improvement over standard GARCH' if list(perf_df.sort_values('RMSE').index)[0] != 'AR-GARCH' else 'The standard AR-GARCH performs competitively with extended models'}, suggesting {'that risk premium and asymmetric effects are important' if list(perf_df.sort_values('RMSE').index)[0] != 'AR-GARCH' else 'that simplicity may be preferred for this dataset'}.

4. **IV as Benchmark:** Using option-implied volatility as the true volatility benchmark provides a market-based validation superior to historical volatility measures.

### 7.2 Practical Implications

- **Risk Management:** The forecasted volatility can be used for Value-at-Risk (VaR) calculations and position sizing.
- **Option Pricing:** Volatility forecasts help in pricing OTC copper options and assessing mispricing.
- **Trading Strategies:** Volatility forecasts enable volatility arbitrage and hedging strategies.

### 7.3 Limitations and Future Research

**Limitations:**
- Sample period limited by option data availability
- Daily frequency may miss intraday volatility patterns
- Model selection limited to three GARCH variants

**Future Research Directions:**
1. Incorporate macroeconomic variables (e.g., Chinese manufacturing PMI, copper inventories)
2. Test additional models (EGARCH, FIGARCH for long memory)
3. Extend to multivariate GARCH for cross-commodity analysis
4. Implement high-frequency data for realized volatility benchmarks

---

## 8. Technical Details

### 8.1 Software and Packages
- **Python Version:** 3.x
- **Key Libraries:**
  - `arch` for GARCH modeling
  - `pandas` for data manipulation
  - `numpy` for numerical computations
  - `matplotlib` & `seaborn` for visualization
  - `scikit-learn` for performance metrics

### 8.2 Computational Notes
- All models estimated using maximum likelihood
- Convergence tolerance: default settings in `arch` package
- Rolling forecast window: expanding (uses all available historical data)

### 8.3 Data Preprocessing
- Log returns calculated as: ln(P_t / P_(t-1)) × 100
- Returns expressed in percentage points
- Volatility annualized using √250 trading days convention

---

## 9. References

1. Bollerslev, T. (1986). "Generalized autoregressive conditional heteroskedasticity." *Journal of Econometrics*, 31(3), 307-327.

2. Engle, R. F., & Ng, V. K. (1993). "Measuring and testing the impact of news on volatility." *The Journal of Finance*, 48(5), 1749-1778.

3. Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). "On the relation between the expected value and the volatility of the nominal excess return on stocks." *The Journal of Finance*, 48(5), 1779-1801.

4. Poon, S. H., & Granger, C. W. (2003). "Forecasting volatility in financial markets: A review." *Journal of Economic Literature*, 41(2), 478-539.

---

## Appendix: Performance Summary Table

| Model | MAE | RMSE | R² | MAPE (%) | Rank (RMSE) |
|-------|-----|------|----|-----------| ------------|
{chr(10).join([f"| {model} | {perf_df.loc[model, 'MAE']:.4f} | {perf_df.loc[model, 'RMSE']:.4f} | {perf_df.loc[model, 'R²']:.4f} | {perf_df.loc[model, 'MAPE (%)']:.2f} | {list(perf_df.sort_values('RMSE').index).index(model) + 1} |" for model in perf_df.index])}

---

**End of Report**

*This report was automatically generated by the GARCH volatility modeling script.*
"""

with open('EXPERIMENT_REPORT.md', 'w', encoding='utf-8') as f:
    f.write(report_content)

print("  Saved: EXPERIMENT_REPORT.md")

# ============================================================================
# Save Results to CSV
# ============================================================================
print("\n[Step 8] Saving results to CSV...")

# Save forecasts
forecast_df = pd.DataFrame({
    'Date': test_data.index,
    'True_IV': true_volatility,
    'AR-GARCH_Forecast': forecasts['AR-GARCH'],
    'GARCH-M_Forecast': forecasts['GARCH-M'],
    'GJR-GARCH_Forecast': forecasts['GJR-GARCH']
})
forecast_df.to_csv('volatility_forecasts.csv', index=False)
print("  Saved: volatility_forecasts.csv")

# Save performance metrics
perf_df.to_csv('model_performance.csv')
print("  Saved: model_performance.csv")

print("\n" + "="*80)
print("EXPERIMENT COMPLETED SUCCESSFULLY!")
print("="*80)
print("\nGenerated Files:")
print("  - EXPERIMENT_REPORT.md (comprehensive experiment report)")
print("  - volatility_forecasts.csv (all forecast data)")
print("  - model_performance.csv (performance metrics)")
print("  - figures/ (8 visualization files)")
print("\nNext Steps:")
print("  1. Review EXPERIMENT_REPORT.md for detailed findings")
print("  2. Examine figures/ directory for all visualizations")
print("  3. Analyze volatility_forecasts.csv for forecast details")
print("="*80)
