#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end analysis script for the TOP → Revenue assignment.
Outputs:
  - figs/fig1_hexbin.png
  - figs/fig2_spline_uncontrolled.png
  - figs/fig3_marginal_effects.png
  - report/data_profile.json
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.api import add_constant
from statsmodels.regression.linear_model import OLS
from statsmodels.nonparametric.smoothers_lowess import lowess
from patsy import dmatrix

# -------------------------
# Config
# -------------------------
DATA_PATH = os.environ.get("CSV_PATH", "../data/testdata.csv")
OUT_DIR   = os.environ.get("OUT_DIR", "..")
FIG_DIR   = os.path.join(OUT_DIR, "figs")
REP_DIR   = os.path.join(OUT_DIR, "report")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(REP_DIR, exist_ok=True)

# -------------------------
# 1) Load and clean
# -------------------------
raw = pd.read_csv(DATA_PATH)
initial_rows = len(raw)
missing_counts = raw.isna().sum().to_dict()

df = raw.dropna(subset=["revenue", "top"]).copy()
after_drop_rows = len(df)

# Cast categoricals
for col in ["browser", "platform", "site"]:
    if col in df.columns:
        df[col] = df[col].astype("category")

# Winsorize at 1%/99%
winsor_thresholds = {}
for col in ["revenue", "top"]:
    lo, hi = df[col].quantile([0.01, 0.99])
    winsor_thresholds[col] = (float(lo), float(hi))
    df[col] = np.clip(df[col], lo, hi)

# New variable
df["log_rev"] = np.log1p(df["revenue"])

# Rows affected by winsorization (per column)
rows_winsorized = {}
for col in ["revenue", "top"]:
    lo, hi = winsor_thresholds[col]
    rows_winsorized[col] = int(((raw[col] < lo) | (raw[col] > hi)).sum()) if col in raw.columns else 0

# -------------------------
# 2) Quick EDA & Fig 1
# -------------------------
eda_stats = {
    "revenue": {
        "mean": float(df["revenue"].mean()),
        "median": float(df["revenue"].median()),
        "sd": float(df["revenue"].std()),
    },
    "top": {
        "mean": float(df["top"].mean()),
        "median": float(df["top"].median()),
        "sd": float(df["top"].std()),
    },
    "corr_revenue_top": float(df[["revenue","top"]].corr().iloc[0,1])
}

# Hexbin + LOWESS trend
plt.figure(figsize=(7,5), dpi=150)
hb = plt.hexbin(df["top"], df["revenue"], gridsize=40, mincnt=1)
# LOWESS
low = lowess(df["revenue"], df["top"], frac=0.2, return_sorted=True)
plt.plot(low[:,0], low[:,1])
plt.xlabel("Time on Page (TOP)")
plt.ylabel("Revenue")
plt.title("Revenue vs Time on Page (EDA)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig1_hexbin.png"))
plt.close()

# -------------------------
# 3) Modeling
# -------------------------
def fit_ols(y, X):
    Xc = add_constant(X, has_constant="add")
    model = OLS(y, Xc).fit(cov_type="HC1")  # robust SEs
    return model

# M1: revenue ~ top (levels)
m1 = fit_ols(df["revenue"], df[["top"]])

# M2: log1p(revenue) ~ spline(top, df=4)
X_spline = dmatrix("bs(top, df=4, degree=3, include_intercept=False)", {"top": df["top"]}, return_type="dataframe")
m2 = fit_ols(df["log_rev"], X_spline)

# Predictions for M2 for plotting
top_grid = np.linspace(df["top"].min(), df["top"].max(), 200)
Xgrid_spline = dmatrix("bs(top, df=4, degree=3, include_intercept=False)", {"top": top_grid}, return_type="dataframe")
Xgrid_spline_c = add_constant(Xgrid_spline, has_constant="add")
pred2 = m2.get_prediction(Xgrid_spline_c)
pred2_mean = pred2.predicted_mean
pred2_ci = pred2.conf_int()

plt.figure(figsize=(7,5), dpi=150)
plt.plot(top_grid, pred2_mean)
plt.fill_between(top_grid, pred2_ci[:,0], pred2_ci[:,1], alpha=0.2)
plt.xlabel("Time on Page (TOP)")
plt.ylabel("log(1 + Revenue)")
plt.title("Diminishing Returns Check (Spline, Uncontrolled)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig2_spline_uncontrolled.png"))
plt.close()

# M3: log1p(revenue) ~ spline(top) + C(browser)+C(platform)+C(site)
design = dmatrix(
    "bs(top, df=4, degree=3, include_intercept=False) + C(browser) + C(platform) + C(site)",
    df,
    return_type="dataframe"
)

design_info = design.design_info
m3 = fit_ols(df["log_rev"], design)

# Marginal effect curve: compare predictions at top +/- 1SD around mean
mu = df["top"].mean()
sd = df["top"].std()

from patsy import build_design_matrices

def predict_logrev(model, top_values, df_ref):
    # Build a design matrix holding other factors at their reference (first category)
    tmp = df_ref.copy()
    tmp = tmp.iloc[:1].copy()  # single row template
    tmp = pd.concat([tmp]*len(top_values), ignore_index=True)
    tmp["top"] = top_values
    # set reference categories to the first levels
    for col in ["browser","platform","site"]:
        if col in tmp.columns and str(tmp[col].dtype) == "category":
            tmp[col] = tmp[col].cat.categories[0]
    X = build_design_matrices([design_info], tmp)[0]
    Xc = add_constant(X, has_constant="add")
    pred = model.get_prediction(Xc).predicted_mean
    return pred

top_vals = np.linspace(df["top"].quantile(0.05), df["top"].quantile(0.95), 120)
log_pred = predict_logrev(m3, top_vals, df)

plt.figure(figsize=(7,5), dpi=150)
plt.plot(top_vals, np.expm1(log_pred))
plt.xlabel("Time on Page (TOP)")
plt.ylabel("Predicted Revenue (Controlled)")
plt.title("Controlled Marginal Effect of TOP")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig3_marginal_effects.png"))
plt.close()

# Compute effect for summary: % change from mu to mu+1SD
log_mu = predict_logrev(m3, np.array([mu]), df)[0]
log_mu_sd = predict_logrev(m3, np.array([mu+sd]), df)[0]
percent_change_mu_to_mu_sd = float(np.expm1(log_mu_sd - log_mu) * 100.0)

# -------------------------
# 4) Save data profile + key stats
# -------------------------
profile = {
    "initial_rows": initial_rows,
    "rows_after_drop": after_drop_rows,
    "n_columns": raw.shape[1],
    "missing_counts": missing_counts,
    "winsorized_thresholds": winsor_thresholds,
    "rows_winsorized": rows_winsorized,
    "eda_stats": eda_stats,
    "m1_slope_top": float(m1.params.get("top", float("nan"))),
    "m1_r2": float(m1.rsquared),
    "controlled_effect_mu_to_mu_plus_1sd_percent": percent_change_mu_to_mu_sd
}
with open(os.path.join(REP_DIR, "data_profile.json"), "w") as f:
    json.dump(profile, f, indent=2)

print("Analysis complete.")
print("Saved:", os.path.join(FIG_DIR, "fig1_hexbin.png"))
print("Saved:", os.path.join(FIG_DIR, "fig2_spline_uncontrolled.png"))
print("Saved:", os.path.join(FIG_DIR, "fig3_marginal_effects.png"))
print("Saved:", os.path.join(REP_DIR, "data_profile.json"))
