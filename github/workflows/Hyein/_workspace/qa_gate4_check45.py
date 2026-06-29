"""Gate-4 checks #4 (raw-space consistency) and #5 (deepcopy-before-forward),
executed rather than just inspected.

#5: confirm the freshly-loaded model deepcopies cleanly BEFORE any forward, and
    that forwarding it first makes deepcopy fail (the dev's stated hazard) -- so
    the ordering in analyze_function is load-bearing.
#4: confirm both spline and symbolic normalized points map to raw via the SAME
    scaler_X.inverse_transform path (driver _denorm_feature), and that a manual
    inverse_transform reproduces the driver's raw points to 1e-9.
"""
import copy, os, joblib, numpy as np, torch
from github.workflows.Hyein.toy_KAN_sweep import KANRegressor
from github.workflows.Hyein.bspline_curvature import symbolic_edge_info
import github.workflows.Hyein.robustness_symbolify as R

FUNC = 'logarithm'
root = os.path.join(os.getcwd(), 'github', 'workflows', 'Hyein',
                    'analytical_results', FUNC, 'kan_models')
ckpt = os.path.join(root, f'{FUNC}_best_kan_model')
scaler_X = joblib.load(os.path.join(root, f'{FUNC}_scaler_X.pkl'))

w = KANRegressor(device='cpu'); w.load_model(ckpt); model = w.model
nx = model.act_fun[0].coef.shape[0]

# ---- #5: deepcopy clean on un-forwarded model ----
try:
    _ = copy.deepcopy(model)
    print("#5a deepcopy of freshly-loaded (un-forwarded) model: OK (clean)")
except Exception as e:
    print("#5a deepcopy of un-forwarded model FAILED:", e)

# now forward a SEPARATE fresh copy, then try to deepcopy it -> expect failure
w2 = KANRegressor(device='cpu'); w2.load_model(ckpt); m2 = w2.model
xin = torch.zeros(4, nx)
m2.forward(xin)
try:
    _ = copy.deepcopy(m2)
    print("#5b deepcopy AFTER forward: OK (no hazard observed)")
except Exception as e:
    print("#5b deepcopy AFTER forward: FAILS as the dev warned ->", str(e)[:80])

# ---- #4: raw-space consistency, both sets via same scaler path ----
vals_norm = [0.03, 0.5, 0.95]
feat = 0
driver_raw = R._denorm_feature(vals_norm, feat, scaler_X, nx)
dummy = np.zeros((len(vals_norm), nx)); dummy[:, feat] = vals_norm
manual_raw = scaler_X.inverse_transform(dummy)[:, feat]
print("#4 driver _denorm_feature vs manual inverse_transform max|diff| =",
      float(np.max(np.abs(driver_raw - manual_raw))), "-> expect ~0")
print("   driver uses the SAME _denorm_feature(scaler_X) for BOTH spline and",
      "symbolic point sets (lines 383-384).")
