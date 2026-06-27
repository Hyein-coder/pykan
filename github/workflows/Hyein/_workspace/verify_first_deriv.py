"""Quick check: analytical 1st derivative + value vs autograd, on damping_sin."""
import os
import numpy as np
import torch

from github.workflows.Hyein.toy_KAN_sweep import KANRegressor
from github.workflows.Hyein.bspline_curvature import (
    activation_value, activation_first_derivative, edge_curves,
)

data_name = 'damping_sin'
savepath = os.path.join('github', 'workflows', 'Hyein', 'analytical_results',
                        data_name, 'kan_models')
mw = KANRegressor(device='cpu')
mw.load_model(os.path.join(savepath, f'{data_name}_best_kan_model'))
model = mw.model
model = model.to(torch.float64)  # higher precision for the check

l = 0
act = model.act_fun[l]
in_dim, out_dim = act.coef.shape[:2]
k = act.k

max_err_d1 = 0.0
max_err_val = 0.0
for i in range(in_dim):
    knots = act.grid[i, k - 1:-2].detach().cpu().numpy()
    lo, hi = float(knots.min()), float(knots.max())
    xs = np.linspace(lo + 0.05 * (hi - lo), hi - 0.05 * (hi - lo), 200)

    # autograd reference on the full activation
    x = torch.zeros(len(xs), in_dim, dtype=torch.float64)
    x[:, i] = torch.tensor(xs, dtype=torch.float64)
    x = x.detach().requires_grad_(True)
    v = activation_value(model, l, x)            # (b,in,out)
    for j in range(out_dim):
        g1, = torch.autograd.grad(v[:, i, j].sum(), x, retain_graph=True)
        ref_d1 = g1[:, i].detach().cpu().numpy()

        phi, d1, d2 = edge_curves(model, l, i, j, xs)
        max_err_d1 = max(max_err_d1, float(np.max(np.abs(d1 - ref_d1))))

        # value cross-check vs autograd-graph value
        max_err_val = max(max_err_val, float(np.max(np.abs(phi - v[:, i, j].detach().cpu().numpy()))))

print(f"max |phi'_analytical - phi'_autograd| = {max_err_d1:.3e}")
print(f"max |phi_value mismatch|              = {max_err_val:.3e}")
print("FIRST-DERIV PASS" if max_err_d1 < 1e-6 else "FIRST-DERIV FAIL")
