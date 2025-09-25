# ITS8080 – HEMS Semester Project (2025)

**Course:** Energy Data Science
**Project:** Home Energy Management System (HEMS) – Forecasting, Modeling, and Optimal Control
**Repository:** *Energy_Data_Science*
**Authors:** *Samuel Heinrich*
**Student Code:** 

---

## 📖 Project Overview

This project develops a complete energy data science pipeline for a **Home Energy Management System (HEMS)**.The tasks cover:

1. Data visualization and exploration
2. PV data cleaning and imputation
3. Feature engineering (time-based & weather-related)
4. Time series decomposition
5. ARMA-family statistical modeling
6. Advanced machine learning models
7. Forecasting with rolling evaluation on out-of-sample data
8. Exogenous models using engineered features
9. Optimal 24h storage control under PV_low and PV_high scenarios

The results will be:

- 📄 A **report (≤25 pages)** with figures, tables, and clear explanations
- 📂 This **GitHub repository** with reproducible code and notebooks

---

## 📂 Repository Structure

```
Energy_Data_Science/
  ├─ README.md                # Project overview
  ├─ data/
  │   ├─ raw/                 # train_test.csv, forecast.csv, optimisation.csv
  │   ├─ interim/             # intermediate results
  │   └─ processed/           # cleaned/engineered data
  ├─ notebooks/               # Analysis & modeling in task order
  │   ├─ 01_visualization.ipynb
  │   ├─ 02_cleaning_pv.ipynb
  │   ├─ 03_feature_engineering.ipynb
  │   ├─ 04_ts_decomposition.ipynb
  │   ├─ 05_stats_models_ARMA.ipynb
  │   ├─ 06_ml_models.ipynb
  │   ├─ 07_forecasting_pipeline.ipynb
  │   ├─ 08_exogenous_models.ipynb
  │   └─ 09_optim_storage.ipynb
  ├─ src/                     # Modular reusable code
  │   ├─ config.py
  │   ├─ data_io.py
  │   ├─ cleaning.py
  │   ├─ features.py
  │   ├─ modeling_stats.py
  │   ├─ modeling_ml.py
  │   ├─ forecasting.py
  │   └─ optimize_battery.py
  ├─ reports/
  │   ├─ figures/             # Generated plots
  │   └─ tables/              # Generated result tables
  └─ tests/                   # Unit tests
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/samuel29102002/Energy_Data_Science.git
cd Energy_Data_Science
```

### 2. Create a Python virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Activate the environment later

```bash
source .venv/bin/activate  # On Windows use: .venv\\Scripts\\activate
```

### 4. Run Jupyter Lab

```bash
jupyter lab
```

### 5. Add the course datasets

Place the provided `train_test.csv`, `forecast.csv`, and `optimisation.csv` files inside `data/raw/`. Placeholder files are already tracked so you can overwrite them when the real data arrives.

---

## ▶️ How to Run

### Data Preprocessing & Cleaning

- Run notebooks `01_visualization.ipynb` and `02_cleaning_pv.ipynb` to explore and clean the PV data.
- Outputs are saved in `data/processed/` and `reports/figures/`.

### Feature Engineering & Decomposition

- Use `03_feature_engineering.ipynb` and `04_ts_decomposition.ipynb`.

### Modeling

- Statistical ARMA-family models: `05_stats_models_ARMA.ipynb`
- Machine Learning model: `06_ml_models.ipynb`

### Forecasting

- Rolling 7-day forecasts on `forecast.csv`: `07_forecasting_pipeline.ipynb`
- Includes naive baselines and comparisons.

### Exogenous Models

- Enhanced models with exogenous features: `08_exogenous_models.ipynb`

### Optimization

- Battery storage optimization: `09_optim_storage.ipynb`
- Compares PV_low vs PV_high cases.

---

## 📊 Results

Key outputs include:

- Data cleaning comparison (PV_mod1 with 3 methods)
- Feature importance rankings
- Demand decomposition (trend, seasonality, residual)
- ARMA vs ML forecasting performance (normalized RMSE, MAE)
- Exogenous vs autoregressive models
- Optimal storage control decisions for 24h horizon

---

## 🔗 Links

- 📘 **Final Report PDF:** [Link to report](#)
- 📂 **GitHub Repo:** [Link to my GitHub](https://github.com/samuel29102002/Energy_Data_Science)

---

## ✅ Submission Checklist

- [ ] Report ≤25 pages with labeled figures/tables
- [ ] Three PV cleaning methods compared (PV_mod1)
- [ ] Feature engineering and ranking completed
- [ ] Demand decomposition and typical profiles
- [ ] ARMA-family models with daily walk-forward evaluation
- [ ] ML model + hyperparameter table + comparison
- [ ] Rolling 7-day forecast (24h horizon, 0h lead time)
- [ ] Exogenous models evaluated with MAE & normalized RMSE
- [ ] 24h battery optimization for PV_low and PV_high
- [ ] GitHub repo with nbviewer access
