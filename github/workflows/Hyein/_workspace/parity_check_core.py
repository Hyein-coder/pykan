"""Scratch parity validator: drive kan_analysis_core.analyze_model on the
`conditional` model exactly as toy_KAN_analyze would, write to a scratch dir,
and let the caller diff the MODEL-ONLY outputs against the pre-refactor baseline.
Does not touch toy_KAN_analyze.py or any canonical output dir.
"""
import os
import numpy as np
import torch
import joblib
from sklearn.model_selection import train_test_split

from github.workflows.Hyein.toy_KAN_sweep import KANRegressor, FUNCTION_ZOO
import github.workflows.Hyein.kan_analysis_core as core

DATA = "conditional"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "parity_core_out")
os.makedirs(OUT, exist_ok=True)

km = os.path.join(HERE, "..", "analytical_results", DATA, "kan_models")
scaler_X = joblib.load(os.path.join(km, f"{DATA}_scaler_X.pkl"))
scaler_y = joblib.load(os.path.join(km, f"{DATA}_scaler_y.pkl"))
w = KANRegressor(device="cpu")
w.load_model(os.path.join(km, f"{DATA}_best_kan_model"))
model = w.model

cfg = FUNCTION_ZOO[DATA]
target_func, bounds, feat_names = cfg["func"], cfg["bounds"], cfg["names"]
nx = len(bounds)

# regen data the same way toy does (unseeded uniform; model-only CSVs are
# independent of this draw, kan_interval_scores is not — expected).
X_raw = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds],
                          size=(1000, nx))
y_raw = np.apply_along_axis(target_func, 1, X_raw).reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2,
                                                    random_state=42)
X_norm = scaler_X.transform(X_train)
y_norm = scaler_y.transform(y_train)

model, sym_info, msg = core.select_model_reading(model, "as-saved", model_wrapper=w)
print(msg)

res = core.analyze_model(
    model, scaler_X=scaler_X, scaler_y=scaler_y, feat_names=feat_names,
    bounds=bounds, savepath=OUT, X_norm=X_norm, y_norm=y_norm, X_raw=X_train,
    true_func=target_func, device="cpu", tag="", data_name=DATA,
    model_mode="as-saved",
)
print("transition_points_per_input:", res.get("transition_points_per_input"))
print("OUT:", OUT)
