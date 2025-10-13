# Energy Data Science: A Home Energy Management System Case Study

Project Title: Energy_Data_Science  
Author: Your Name (Student ID: Your ID)  
Course: ITS8080 Energy Data Science  
Date: 12 Oct 2025

---

## Abstract

This project develops an end-to-end analytics workflow for a residential Home Energy Management System (HEMS) leveraging a historical dataset of household electricity demand, photovoltaic (PV) generation, and exogenous variables. The methodology spans data exploration, cleaning, feature engineering, time-series decomposition, statistical modeling (ARMA/ARIMA), and machine-learning forecasting, culminating in a rolling forecasting pipeline and a battery storage optimization for cost reduction. Using visual diagnostics and tabular summaries exported from Notebooks 01–11, we quantify seasonality (diurnal/weekly), examine missingness and PV corrections, rank predictive features, and compare baseline/statistical vs. ML performance. Walk-forward evaluation shows consistent improvements over naive benchmarks, while exogenous features further reduce RMSE/MAE. Optimization experiments indicate that a modest battery can lower energy costs and increase PV self-consumption, with sensitivity to PV availability and price signals. The deliverables include a reproducible report and a Dash dashboard for stakeholder-facing insights.

---

## 1. Introduction

### 1.1 Context and problem framing
Smart electrification and distributed energy resources (DERs) are reshaping residential energy systems. Home Energy Management Systems (HEMS) coordinate flexible demand, PV production, and storage subject to price signals and comfort constraints. Robust short-horizon forecasting and data-driven optimization are essential to minimize energy costs and emissions while safeguarding user comfort. This report documents the Energy_Data_Science project, which implements a complete data science workflow: exploration, cleaning, feature engineering, time-series decomposition, statistical and machine learning forecasting, and optimal battery dispatch.

### 1.2 Digitalization and household energy (Task 1)
Digitalization transforms the electricity sector by enabling high-frequency sensing, edge analytics, and automated control across the value chain—from demand response to DER orchestration [1], [2]. In households, improved forecasts drive better scheduling of storage and flexible loads, enabling higher PV self-consumption and reduced peak imports. However, digitalization also raises concerns about data privacy, algorithmic bias, and resilience, which motivate the use of transparent methods, strong data governance, and robust validation.

### 1.3 Initial data familiarization and visual inspection (Task 1)
We performed a visual check of key signals to validate conventions and units and to surface obvious pathologies:

- Demand and PV display clear diurnal cycles; PV is (near) zero overnight and peaks around solar noon. Weekday vs. weekend demand patterns diverge, reflecting occupancy effects (Figures 3.1–3.4).  
- Correlations indicating physical drivers (temperature–demand) are visible in Task 3 correlation plots.  
- Price (if available) typically co-varies with demand and scarcity; while not plotted as a standalone figure here, price can be incorporated as a covariate or a downstream optimization signal.  

These checks establish an interpretable baseline consistent with the broader literature on residential demand and PV variability [2], [8].

### 1.4 Why solar data matters and where it is used (Task 1)
Solar generation data is critical for household- and grid-level decision-making. For private consumers, higher PV self-consumption reduces bills and emissions; for businesses, it informs portfolio hedging, asset sizing, and demand response participation. Accurate PV/demand forecasts improve sizing of storage/inverters, day-ahead commitments, and real-time control [2], [8].

---

## 2. Methodology & Project Planning

### 2.1 Dataset, variables, and units (Task 2)
The working dataset (`data/raw/train_252145.csv`) contains timestamped household demand (kWh), PV generation (kWh), and auxiliary variables such as temperature. Engineered features (calendar fields, lags, moving statistics) are stored in `data/processed/task5_features.parquet`. Curated tables (metrics, KPIs) reside in `reports/tables`, and figures in `reports/figures`.

Target and covariates:
- Demand (target) in kWh; for hourly resolution, values reflect energy consumed within the hour.  
- PV (exogenous) in kWh; constrained to be nonnegative and near zero during night hours.  
- Temperature and calendar indicators (hour, weekday, month) as key exogenous drivers.  
- Market signals (e.g., price) can be used downstream in cost optimization; if available, they can also serve as exogenous predictors.

### 2.2 Data science lifecycle and mapping to tasks (Task 2)
We adopt the CRISP‑DM framework—business/data understanding; preparation; modeling; evaluation; deployment [9]. The project plan (Figure 2.1) maps tasks 01–11 onto these phases:  
Task 1 (context/digitalization) and Task 3 (EDA) support data understanding; Task 4 (cleaning) finalizes preparation; Tasks 5–8 (features/statistics/ML) constitute modeling; Task 9 (pipeline) and Task 10 (exogenous models) are evaluation and refinement; Task 11 (optimization) is deployment into decision support.

- Planning diagram: Figure 2.1 (`reports/figures/task2_lifecycle.png`).  
Figure 2.1 — Project plan and lifecycle (source: project assets).

### 2.3 Effort distribution and risks (Task 2)
Effort concentrated on data preparation and feature engineering (Tasks 4–5), which substantially determine forecasting quality [5]. Model comparison and pipeline validation (Tasks 7–9) required careful walk‑forward evaluation to avoid leakage. Key risks included: (i) missingness and PV anomalies; (ii) covariate shift/seasonal drift; (iii) overfitting in ML. Mitigations: robust cleaning, regularization, out‑of‑sample validation, and diagnostic visualization.

### 2.4 External data sources and assumptions (Task 2)
Where available, weather reanalysis or market price data can enhance forecast accuracy and inform optimization [2], [10]. Assumptions include correct timestamp alignment (local time without DST ambiguity) and representative training periods for the intended operational horizon.

---

## 3. Data Exploration and Visualization (Task 3)

We begin with exploratory data analysis (EDA) to understand magnitudes, trends, and dependencies.

- Demand and PV time series show clear diurnal patterns (Figures 3.1–3.2).  
  - Figure 3.1 — Demand and PV overview (`reports/figures/01_demand_pv_timeseries.png`).  
  - Figure 3.2 — One-week sample detail (`reports/figures/01_demand_pv_daily_sample.png`).
- Typical daily profiles (Figure 3.3) and weekday-hour heatmaps (Figure 3.4) emphasize regularity and behavior shifts across weekdays vs. weekends.  
  - Figure 3.3 — Diurnal demand profile (`reports/figures/02_demand_diurnal_profile.png`).  
  - Figure 3.4 — Weekday×hour heatmap (`reports/figures/02_demand_weekday_heatmap.png`).
- Distributional checks (Task 3 figures in `reports/figures/task3_fig*.png`) confirm right-skewness in PV and moderate spread in demand; correlation heatmaps identify temperature and calendar features as relevant drivers (`reports/figures/task3_fig4_correlation_heatmap.png`).

Ethical visualization standards were applied: consistent axes and units, readable legends, visible uncertainty where applicable, and avoidance of deceptive scaling [3].

---

## 4. Data Cleaning and Preprocessing (Task 4)

Cleaning tackled (i) missingness; (ii) timestamp consistency; (iii) PV corrections (e.g., eliminating spurious negative values, smoothing short null gaps). Figure 4.1 visualizes missingness at column×time granularity, while Table 4.1 summarizes PV data quality metrics.

- Figure 4.1 — Missingness heatmap (`reports/figures/04_missingness_heatmap.png`).
- Table 4.1 — PV gap statistics (`reports/tables/04_pv_gap_stats.csv`).

Missing value mechanisms were assessed (MCAR/MAR/MNAR where plausible for sensors/weather). Strategies included: (a) deletion of rare anomalous rows; (b) univariate interpolation or forward/backward fill for short gaps; (c) multivariate imputation for blocks when supported by correlated covariates. PV-specific checks ensured zero output at night and bounded daytime values. Before/after comparisons are visible in Task 4 overlays (`reports/figures/task4_fig_imputation_overlay.png`) and variability profiles (`reports/figures/task4_fig_daily_profiles.png`).

---

## 5. Feature Engineering (Task 5)

From cleaned data, we built calendar features (hour, weekday, month), lagged demand/PV, rolling statistics (e.g., 24h mean), and interaction terms. Feature distributions were checked for outliers and transform need (e.g., logs for skewness) (see `reports/tables/05_feature_stats.csv`). Figure 5.1 presents feature importance by absolute correlation with Demand (for illustration; model-specific importances are in ML results).

- Table 5.1 — Feature summary and correlation with Demand (`reports/tables/05_feature_stats.csv`).  
- Figure 5.1 — Feature importance proxy (`reports/figures/05_feature_importance.png`).

---

## 6. Time Series Analysis (Task 6)

We applied STL decomposition to Demand to extract trend, seasonal, and residual components. Diurnal seasonality is strong with moderate weekly effects; the seasonal component aligns with Figure 6.1. Typical hourly profiles differ by month and weekday (Task 6 panels in `reports/figures/stats_*`).

- Figure 6.1 — STL seasonality and decomposition panels (`reports/figures/03_demand_seasonality_stl.png`).  
- Additional seasonal panels: `reports/figures/demand_seasonality_strength.png`, `reports/figures/demand_typical_hourly_by_month.png`.

Let $y_t$ denote hourly demand. A seasonal decomposition writes $y_t = T_t + S_t + R_t$, where $T_t$ is trend, $S_t$ seasonal (daily/weekly), and $R_t$ residual. Strong $S_t$ suggests models with seasonal/periodic structure.

---

## 7. Statistical Modelling (Task 7)

We used autocorrelation (ACF) and partial autocorrelation (PACF) diagnostics (Figure 7.1; `reports/figures/stats_acf_pacf.png`) to select ARMA/ARIMA orders. A general ARIMA($p,d,q$) with seasonal components is:  
$\Phi_p(B) (1 - B)^d y_t = \Theta_q(B) \varepsilon_t$, possibly extended with seasonal polynomials. Candidate models were compared on rolling windows using RMSE/MAE.

- Figure 7.1 — ACF/PACF panels.  
- Figure 7.2 — Forecast overlay for best statistical model (`reports/figures/stats_forecast_overlay_best.png`).  
- Figure 7.3 — Walk-forward panels (`reports/figures/stats_walkforward_panels.png`).  
- Figure 7.4 — Metrics bar chart (`reports/figures/stats_metrics_bar.png`).

Statistical baselines outperformed naive persistence but were later surpassed by ML models with richer features (Section 8).

---

## 8. Machine Learning Modelling (Task 8)

We trained supervised learners (e.g., gradient boosting, random forests) on engineered features. Hyperparameters were tuned via cross-validation (grid/random search depending on model). Learning curves and feature importance plots (`reports/figures/ml_learning_curve.png`, `reports/figures/ml_feat_importance.png`) indicate stable generalization with diminishing returns after moderate sample sizes.

- Figure 8.1 — ML feature importance (`reports/figures/ml_feat_importance.png`).  
- Figure 8.2 — Learning curve (`reports/figures/ml_learning_curve.png`).  
- Figure 8.3 — Forecast overlay (`reports/figures/ml_forecast_overlay.png`).  
- Figure 8.4 — Metrics comparison (`reports/figures/ml_metrics_comparison.png`).

A summary of hyperparameters is provided in the ML notebook outputs (Task 8). ML models achieved lower RMSE than statistical models, particularly with exogenous variables (Section 10).

---

## 9. Forecasting Pipeline (Task 9)

We implemented a rolling 7-day forecasting pipeline with refits or updates as appropriate. Baselines included persistence (lag-24) and simple seasonal averages. Diagnostics for the baseline are visualized in Figure 9.1. The pipeline standardizes input processing, feature assembly, model prediction, and metric logging for consistent comparison across models and horizons.

- Figure 9.1 — Baseline diagnostics (`reports/figures/06_baseline_diagnostics.png`).

Results: Aggregated metrics across rolling windows show stable superiority of ML models over statistical baselines, with the best model’s RMSE also reflected on the dashboard (Overview metrics).

---

## 10. Models with Exogenous Inputs (Task 10)

Inclusion of exogenous features (temperature, calendar variables, PV) improved short-horizon forecasts. Table 10.1 summarizes MAE/RMSE by model with/without exogenous inputs; improvements are visible in the dashboard’s model comparison and in Task 10 notebook tables.

- Table 10.1 — Comparative metrics (see `reports/tables/07_kpis.csv` for KPI-style aggregates; model-specific tables in notebooks).

Discussion: Exogenous signals help capture demand sensitivity and daylight-driven PV-dampened imports. Gains are most pronounced during transition seasons when weather variability is higher.

---

## 11. Optimal Control of Energy Storage (Task 11)

We formulated a cost-minimizing optimal control problem for a residential battery subject to power/energy limits and state-of-charge (SoC) dynamics. Given price signals and forecasts, the optimizer schedules charge/discharge to reduce net import costs and increase PV self-consumption.

- Decision variables: hourly charge/discharge and SoC.  
- Constraints: SoC bounds, charge/discharge power limits, energy balance, PV/use limits.  
- Objective: minimize purchase cost minus sale revenues.

Scenarios: PV-low and PV-high cases demonstrate sensitivity of cost savings to renewable availability. The dashboard optimization page reports KPIs (total cost, energy bought/sold, battery cycles) and trajectories (SoC and energy flows). Overall, storage reduced energy costs and curtailed peaks; absolute savings depend on the spread between buy/sell tariffs and forecast accuracy.

---

## 12. Discussion and Critical Analysis

- Household efficiency: Forecast-driven control curtails imports during peak prices and shifts demand to PV hours, increasing self-consumption.  
- Model selection: ML with exogenous inputs typically outperforms ARIMA in this setting; however, statistical models remain valuable for interpretability and robustness on limited data [4], [5].  
- Data quality: Cleaning and PV corrections (Section 4) are pivotal—small biases propagate to forecasts and operations.  
- Limitations: Single-household scope, potential concept drift, measurement error, and limited tariff complexity.  
- Improvements: Probabilistic forecasting, hierarchical reconciliation (appliance-level), and model-based control with explicit uncertainty (stochastic optimization, MPC) [6].  
- Ethics and usability: Visualizations follow clarity and accessibility best practices; the stakeholder dashboard supports transparent decision-making.

---

## 13. Conclusion

This project delivers a complete HEMS analytics chain from raw data to operational optimization. We demonstrate that (i) careful cleaning and feature engineering expose strong seasonality; (ii) ML models with exogenous inputs outperform baselines and ARIMA variants; and (iii) a simple battery optimization informed by forecasts can reduce costs and increase PV self-consumption. Future work should incorporate uncertainty-aware forecasts, real-time adaptation, and richer tariff structures to unlock further savings and resilience.

---

## References

[1] International Energy Agency, “Digitalization and Energy,” IEA, 2017.  
[2] P. Palensky and D. Dietrich, “Demand Side Management: Demand Response, Intelligent Energy Systems, and Smart Loads,” IEEE Trans. Industrial Informatics, vol. 7, no. 3, pp. 381–388, 2011.  
[3] E. R. Tufte, The Visual Display of Quantitative Information, 2nd ed., Graphics Press, 2001.  
[4] G. E. P. Box, G. M. Jenkins, G. C. Reinsel, and G. M. Ljung, Time Series Analysis: Forecasting and Control, 5th ed., Wiley, 2015.  
[5] R. J. Hyndman and G. Athanasopoulos, Forecasting: Principles and Practice, 3rd ed., OTexts, 2021.  
[6] J. Maciejowski, Predictive Control with Constraints, Prentice Hall, 2002.  
[7] T. Hastie, R. Tibshirani, and J. Friedman, The Elements of Statistical Learning, 2nd ed., Springer, 2009.  
[8] M. Islam et al., “Short-Term Solar Power Forecasting: A Review,” Renewable and Sustainable Energy Reviews, 2017.

---

## Appendix: List of Referenced Assets

Figures

- 01_demand_pv_timeseries.png  
- 01_demand_pv_daily_sample.png  
- 02_demand_diurnal_profile.png  
- 02_demand_weekday_heatmap.png  
- 03_demand_seasonality_stl.png  
- 04_missingness_heatmap.png  
- ml_feat_importance.png  
- ml_learning_curve.png  
- ml_forecast_overlay.png  
- ml_metrics_comparison.png  
- stats_acf_pacf.png  
- stats_forecast_overlay_best.png  
- stats_walkforward_panels.png  
- stats_metrics_bar.png  
- task2_lifecycle.png  

Tables

- 04_pv_gap_stats.csv  
- 05_feature_stats.csv  
- 07_kpis.csv  

Notes: All assets are located under `reports/figures` and `reports/tables`. The analysis narrative corresponds to Notebooks 01–11.
