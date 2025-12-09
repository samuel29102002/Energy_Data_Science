# HEMS Optimization Using Data Science

**Course:** ITS8080 Energy Data Science  
**Author:** Samuel Heinrich (252145MV)  
**Institution:** Tallinn University of Technology  
**Date:** December 2024

---

## Overview

This repository implements a complete Home Energy Management System (HEMS) optimization pipeline. The project covers data preprocessing, statistical and machine learning demand forecasting, and optimal battery storage control under time-of-use pricing.

The pipeline processes one year of hourly household data (demand, PV generation, electricity prices) to forecast energy demand and optimize a 10 kWh battery system for cost minimization.

---

## Repository Structure

```
Energy_Data_Science/
├── data/
│   ├── raw/                    # Input datasets
│   └── processed/              # Cleaned and feature-engineered data
├── notebooks/
│   ├── 01-03_visualization     # EDA and data profiling
│   ├── 04_pv_cleaning          # PV data imputation
│   ├── 05_feature_engineering  # Temporal and lag features
│   ├── 06_ts_decomposition     # STL decomposition
│   ├── 07_stats_models_ARMA    # ARIMA/SARIMA models
│   ├── 08_ml_models            # XGBoost regression
│   ├── 09_forecasting_pipeline # 7-day rolling forecast
│   ├── 10_exogenous_models     # Exogenous variable integration
│   └── 11_optim_storage        # LP-based battery optimization
├── src/
│   ├── modeling_ml.py          # XGBoost utilities
│   ├── modeling_stats.py       # ARIMA/SARIMA utilities
│   ├── forecasting.py          # Rolling forecast pipeline
│   ├── plotting.py             # Visualization helpers
│   └── dash_app/               # Interactive dashboard
├── reports/
│   ├── figures/                # Generated plots (PNG)
│   └── tables/                 # Metrics and results (CSV)
└── scripts/                    # Validation and export utilities
```

---

## Results

### Forecasting Performance (7-Day Rolling Evaluation)

| Model | MAE (kW) | RMSE (kW) | nRMSE |
|-------|----------|-----------|-------|
| Naive Baseline | 0.115 | 0.182 | 0.340 |
| Seasonal Naive | 0.162 | 0.253 | 0.745 |
| SARIMA | 0.206 | 0.295 | 1.062 |
| XGBoost | 0.407 | 0.536 | 2.768 |

The Naive baseline outperforms complex models on the 7-day out-of-sample test, indicating low short-term variability in demand patterns.

### Battery Storage Optimization (24-Hour Horizon)

| Scenario | Daily Cost | PV Generation | Grid Import | Grid Export | Self-Consumption |
|----------|------------|---------------|-------------|-------------|------------------|
| PV Low   | 0.64 EUR   | 1.22 kWh      | 11.16 kWh   | 0.00 kWh    | 1.22 kWh         |
| PV High  | -0.05 EUR  | 12.99 kWh     | 0.00 kWh    | 0.95 kWh    | 12.04 kWh        |

Under high PV conditions, the optimized dispatch achieves net profit through export arbitrage and full self-consumption.

---

## Installation

### Prerequisites

- Python 3.10+
- macOS: `brew install libomp` (required for XGBoost)

### Setup

```bash
git clone https://github.com/samuel29102002/Energy_Data_Science.git
cd Energy_Data_Science

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Data

Place the following files in `data/raw/`:
- `train_252145.csv` - Historical demand, PV, and price data
- `forecast.csv` - 7-day forecast evaluation period
- `optimisation.csv` - 24-hour optimization scenarios

---

## Usage

### Run Notebooks

```bash
jupyter lab
```

Execute notebooks in order (01 through 11) to reproduce all results.

### Launch Dashboard

```bash
python src/dash_app/app.py
```

Access at `http://127.0.0.1:8050`

---

## Methods

### Data Preprocessing
- Gap detection and linear/spline imputation for PV data
- Feature engineering: hour, weekday, month, lag features (1h, 24h, 168h)
- STL decomposition for trend/seasonality extraction

### Forecasting
- Statistical: ARIMA(2,1,2), SARIMA(1,1,1)(1,1,1,24)
- Machine Learning: XGBoost with 600 estimators, max_depth=6
- Evaluation: Walk-forward validation with 24h horizon

### Optimization
- Linear programming with CVXPY
- Decision variables: grid import/export, battery charge/discharge
- Constraints: power balance, SOC limits (0-10 kWh), charge/discharge rates

---

## Technical Stack

| Category | Tools |
|----------|-------|
| Data Processing | pandas, numpy, scipy |
| Statistical Models | statsmodels (SARIMAX) |
| Machine Learning | xgboost, scikit-learn |
| Optimization | cvxpy |
| Visualization | matplotlib, seaborn, plotly |
| Dashboard | dash, dash-bootstrap-components |

---

## License

This project was developed for academic purposes as part of the ITS8080 course at TalTech.

---

## Contact

Samuel Heinrich  
GitHub: [@samuel29102002](https://github.com/samuel29102002)
