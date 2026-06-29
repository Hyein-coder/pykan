"""Verify dead-edge protection is a REAL (non-vacuous) test on some function:
find an invert-symbolify model that has both symbolified edges AND dead edges
(mask=0 not in symbolic_edge_info), then confirm spline reading flips only the
symbolified ones."""
import os, numpy as np
from github.workflows.Hyein.toy_KAN_sweep import KANRegressor
from github.workflows.Hyein.bspline_curvature import symbolic_edge_info
import github.workflows.Hyein.robustness_symbolify as R

for FUNC in ['log2', 'exponential', 'rosenbrock', 'logarithm']:
    root = os.path.join(os.getcwd(), 'github', 'workflows', 'Hyein',
                        'analytical_results', FUNC, 'kan_models')
    ckpt = os.path.join(root, f'{FUNC}_best_kan_model')
    w = KANRegressor(device='cpu'); w.load_model(ckpt); model = w.model
    sym = symbolic_edge_info(model, 0)
    act0 = model.act_fun[0]
    in_dim, out_dim = act0.coef.shape[:2]
    mb = act0.mask.detach().cpu().numpy()
    dead = [(i, j) for i in range(in_dim) for j in range(out_dim)
            if (i, j) not in sym and mb[i, j] == 0.0]
    spline = R._build_spline_reading(model, sym)
    ma = spline.act_fun[0].mask.detach().cpu().numpy()
    dead_after = [(i, j) for (i, j) in dead if ma[i, j] != 0.0]
    print(f"{FUNC:12s} symbolified={sorted(sym.keys())} dead(mask0,not-sym)={dead} "
          f"-> dead edges revived by spline reading: {dead_after} "
          f"({'OK' if not dead_after else 'FAIL'})")
