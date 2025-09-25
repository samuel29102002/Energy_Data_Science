# ITS8080 – HEMS Semester Project (2025)

**Course:** Energy Data Science
**Project:** Home Energy Management System (HEMS) – Forecasting, Modeling, and Optimal Control
**Repository:** *Energy_Data_Science*
**Authors:** *Samuel Heinrich*
**Student Code: 252145MV**

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
  │   ├─ raw/                 # train_252145.csv, forecast.csv, optimisation.csv
  │   ├─ interim/             # intermediate results
  │   └─ processed/           # cleaned/engineered data
  ├─ notebooks/               # Analysis & modeling in task order
  │   ├─ 01_visualization.ipynb
  │   ├─ 02_project_planning.ipynb
  │   ├─ 03_visualization.ipynb
  │   ├─ 04_pv_cleaning.ipynb
  │   ├─ 05_feature_engineering.ipynb
  │   ├─ 06_ts_decomposition.ipynb
  │   ├─ 07_stats_models_ARMA.ipynb
  │   ├─ 08_ml_models.ipynb
  │   ├─ 09_forecasting_pipeline.ipynb
  │   ├─ 10_exogenous_models.ipynb
  │   └─ 11_optim_storage.ipynb
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

### 2. Activate your Anaconda environment

```bash
conda activate energy-ds  # replace with the environment you already created
```

Need a fresh environment?

```bash
conda create -n energy-ds python=3.10
conda activate energy-ds
```

### 3. Install the dependencies

```bash
conda install --file requirements.txt
# or, inside the environment:
pip install -r requirements.txt
```

### 4. (Optional) expose the kernel to Jupyter

```bash
python -m ipykernel install --user --name energy-ds --display-name "Python (energy-ds)"
```

### 5. Run Jupyter Lab

```bash
jupyter lab
```

### 6. Add the course datasets

Place the provided `train_252145.csv`, `forecast.csv`, and `optimisation.csv` files inside `data/raw/`. Placeholder files are already tracked so you can overwrite them when the real data arrives.

---

## ▶️ How to Run

### Initial Assessment & Planning

- Run `01_visualization.ipynb` to profile PV generation, demand, and price for Task 1.
- Use `02_project_planning.ipynb` for the lifecycle diagram (outputs in `reports/figures/`).
- Explore Task 3 visuals in `03_visualization.ipynb`; artefacts land in `reports/figures/` and `reports/tables/`.
- Interactive Plotly exports for the dashboard are stored under `reports/figures/interactive/`.

### PV Cleaning & Feature Prep

- Execute `04_pv_cleaning.ipynb` to audit PV sensor gaps and compare imputation strategies; figures/tables feed the report and dashboard.
- Use `05_feature_engineering.ipynb` and `06_ts_decomposition.ipynb` for derived features and decomposition analysis.

### Modeling

- Statistical ARMA-family models: `07_stats_models_ARMA.ipynb`
- Machine Learning model: `08_ml_models.ipynb`

### Forecasting

- Rolling 7-day forecasts on `forecast.csv`: `09_forecasting_pipeline.ipynb`
- Includes naive baselines and comparisons.

### Exogenous Models

- Enhanced models with exogenous features: `10_exogenous_models.ipynb`

### Optimization

- Battery storage optimization: `11_optim_storage.ipynb`
- Compares PV_low vs PV_high cases.

### Dash App

- Launch the interactive dashboard for exploratory analysis:

```bash
python src/dash_app/app.py
```

- Open the printed URL (default `http://127.0.0.1:8050`). The **Overview** tab tracks key metrics, the **Visualisation Studio** tab provides interactive plot selection (timeseries, distributions, boxplots, heatmaps, profiles), and the **PV Cleaning** tab compares imputation strategies with live overlays and summary tables.

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
