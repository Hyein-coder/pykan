"""
grid_k_sweep.py
===============
Retrain KAN from scratch for every (func, grid, k) combination and record
the B-spline transition points found in layer-0 spline coefficients.

Results are written to:
    github/workflows/Hyein/_workspace/grid_k_sweep_results.csv

Usage
-----
    # defaults
    python grid_k_sweep.py

    # custom
    python grid_k_sweep.py --funcs exponential logarithm --grids 3 5 10 --ks 3 4
"""

import argparse
import os
import sys
import traceback

import json

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, r'D:\pykan')
from github.workflows.Hyein.toy_KAN_sweep import FUNCTION_ZOO, KANRegressor
from kan.experiments.analysis import find_indices_sign_revert

# ── configurable defaults ──────────────────────────────────────────────────
DEFAULT_FUNCS = ['exponential', 'logarithm', 'log2', 'conditional', 'rosenbrock']
DEFAULT_GRIDS = [3, 5, 7, 10, 15, 30]
DEFAULT_KS    = [5]
N_SAMPLES     = 1000
STEPS         = 50          # fallback if no *_kan_metrics.json found
SEED          = 42
ANALYTICAL_RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'analytical_results')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '_workspace', 'sweep_models')
OUT_CSV = os.path.join(
    os.path.dirname(__file__), '_workspace', 'grid_k_sweep_results.csv'
)
# ──────────────────────────────────────────────────────────────────────────


def load_best_params(func_name):
    """Load optimised hyperparameters from analytical_results/{func}/*_kan_metrics.json.

    Returns the best_params dict on success, None if the file is missing.
    grid and k are intentionally excluded — those come from the sweep.
    """
    path = os.path.join(
        ANALYTICAL_RESULTS_DIR, func_name, 'kan_models',
        f'{func_name}_kan_metrics.json',
    )
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get('best_params')


def extract_transition_points(model):
    """Return {feat_idx: [tp_norm, ...]} from layer-0 spline coefficients.

    Uses 2nd-derivative sign reversals for depth-1 KANs (single layer),
    1st-derivative sign reversals for depth-2 KANs, mirroring toy_KAN_analyze.py.
    """
    l = 0
    act   = model.act_fun[l]
    ni, no = act.coef.shape[:2]
    coef  = act.coef.tolist()          # (ni, no, n_coef)
    depth = len(model.act_fun)

    tps = {}
    for i in range(ni):
        knots = act.grid[i, model.k - 1:-2].cpu().detach().numpy()
        ips_all = []
        for j in range(no):
            coef_node = coef[i][j]
            slope     = [x - y for x, y in zip(coef_node[1:],  coef_node[:-1])]
            slope_2nd = [(x - y) * 10 for x, y in zip(slope[1:], slope[:-1])]

            if depth == 1:
                idx_rev = find_indices_sign_revert(slope_2nd)
                idx_rev = [ir + 1 for ir in idx_rev]   # offset for 2nd diff
            elif depth == 2:
                idx_rev = find_indices_sign_revert(slope)
            else:
                idx_rev = []

            for ir in idx_rev:
                if 0 <= ir < len(knots):
                    ips_all.append(float(knots[ir]))

        tps[i] = sorted(set(round(v, 5) for v in ips_all))
    return tps


def denorm_tps(tps_norm, feat_idx, nx, scaler_X):
    """Map normalized-space TP values back to raw input space."""
    if not tps_norm:
        return []
    dummy = np.zeros((len(tps_norm), nx))
    dummy[:, feat_idx] = tps_norm
    return scaler_X.inverse_transform(dummy)[:, feat_idx].tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--funcs',    nargs='+', default=DEFAULT_FUNCS,
                        choices=list(FUNCTION_ZOO.keys()))
    parser.add_argument('--grids',    nargs='+', type=int, default=DEFAULT_GRIDS)
    parser.add_argument('--ks',       nargs='+', type=int, default=DEFAULT_KS)
    parser.add_argument('--steps',    type=int,  default=STEPS)
    parser.add_argument('--n_samples',type=int,  default=N_SAMPLES)
    parser.add_argument('--out',      default=OUT_CSV)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Funcs : {args.funcs}")
    print(f"Grids : {args.grids}")
    print(f"k's   : {args.ks}")
    print(f"Steps : {args.steps}  |  N={args.n_samples}\n")

    rows  = []
    total = len(args.funcs) * len(args.grids) * len(args.ks)
    done  = 0

    for func_name in args.funcs:
        cfg        = FUNCTION_ZOO[func_name]
        target_fn  = cfg['func']
        bounds     = cfg['bounds']
        feat_names = cfg['names']
        nx         = len(bounds)

        # ── load optimised base hyperparameters (grid & k will be overridden) ──
        bp = load_best_params(func_name)
        if bp:
            base = dict(
                n_layers      = bp.get('n_layers',      1),
                steps         = bp.get('steps',         args.steps),
                lr            = bp.get('lr',            0.1),
                lamb          = bp.get('lamb',          0.01),
                lamb_coef     = bp.get('lamb_coef',     0.1),
                lamb_entropy  = bp.get('lamb_entropy',  0.1),
                sym_range     = bp.get('sym_range',     10),
            )
            print(f"  [{func_name}] best_params loaded — "
                  f"steps={base['steps']}, lr={base['lr']}, "
                  f"lamb={base['lamb']}, lamb_coef={base['lamb_coef']}, "
                  f"lamb_entropy={base['lamb_entropy']}, "
                  f"sym_range={base['sym_range']}")
        else:
            base = dict(
                n_layers=1, steps=args.steps, lr=0.1,
                lamb=0.01, lamb_coef=0.1, lamb_entropy=0.1, sym_range=10,
            )
            print(f"  [{func_name}] no metrics JSON found — using defaults")

        # ── fixed data for this function (same across all grid/k runs) ──
        rng   = np.random.default_rng(SEED)
        X_raw = rng.uniform(
            low =[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=(args.n_samples, nx),
        )
        y_raw = np.apply_along_axis(target_fn, 1, X_raw).reshape(-1, 1)

        scaler_X = MinMaxScaler(feature_range=(0.1, 0.9))
        scaler_y = MinMaxScaler()
        X_norm = scaler_X.fit_transform(X_raw)
        y_norm = scaler_y.fit_transform(y_raw).ravel()

        # ── save scalers once per function ──
        func_model_dir = os.path.join(MODELS_DIR, func_name)
        os.makedirs(func_model_dir, exist_ok=True)
        joblib.dump(scaler_X, os.path.join(func_model_dir, 'scaler_X.pkl'))
        joblib.dump(scaler_y, os.path.join(func_model_dir, 'scaler_y.pkl'))

        for grid in args.grids:
            for k in args.ks:
                done += 1
                tag = f"[{done:3d}/{total}] {func_name:<14} grid={grid:2d}  k={k}"
                print(tag, end='  ', flush=True)

                try:
                    reg = KANRegressor(
                        n_layers     = base['n_layers'],
                        grid         = grid,          # ← swept
                        k            = k,             # ← swept
                        steps        = base['steps'],
                        lr           = base['lr'],
                        lamb         = base['lamb'],
                        lamb_coef    = base['lamb_coef'],
                        lamb_entropy = base['lamb_entropy'],
                        sym_range    = base['sym_range'],
                        symbolic_enabled=True,   # skip — irrelevant for TPs
                        pruning_enabled=False,    # skip — would alter arch differently per run
                        device=device,
                    )
                    reg.fit(X_norm, y_norm)
                    model = reg.model

                    # ── forward pass on full dataset to populate feature_score ──
                    X_tensor = torch.tensor(X_norm, dtype=torch.float32, device=device)
                    with torch.no_grad():
                        model.forward(X_tensor)
                    scores = model.feature_score.detach().cpu().numpy()

                    # ── R² on the internal test split (random_state=42 always) ──
                    with torch.no_grad():
                        y_pred_n = model(reg.dataset['test_input']).cpu().numpy()
                    y_pred_raw = scaler_y.inverse_transform(y_pred_n).ravel()
                    y_test_raw = scaler_y.inverse_transform(
                        reg.dataset['test_label'].cpu().numpy()
                    ).ravel()
                    test_r2 = float(r2_score(y_test_raw, y_pred_raw))

                    # ── save model checkpoint ──
                    ckpt_path = os.path.join(func_model_dir, f'g{grid}_k{k}')
                    reg.save_model(ckpt_path)

                    # ── transition points ──
                    tps_norm = extract_transition_points(model)

                    for i in range(nx):
                        tp_n = tps_norm.get(i, [])
                        tp_r = denorm_tps(tp_n, i, nx, scaler_X)
                        rows.append({
                            'func':         func_name,
                            'grid':         grid,
                            'k':            k,
                            'feature_idx':  i,
                            'feature_name': feat_names[i],
                            'test_r2':      round(test_r2, 4),
                            'attribution':  round(float(scores[i]), 5),
                            'n_tp':         len(tp_n),
                            'tp_norm':      str(tp_n),
                            'tp_raw':       str(tp_r),
                            'first_tp_norm': tp_n[0] if tp_n else float('nan'),
                            'first_tp_raw':  tp_r[0] if tp_r else float('nan'),
                            'ckpt_path':    ckpt_path,
                        })

                    tp_counts = [len(tps_norm.get(i, [])) for i in range(nx)]
                    print(f"R²={test_r2:.3f}  TPs={tp_counts}")

                except Exception as e:
                    print(f"FAILED — {e}")
                    traceback.print_exc()
                    for i in range(nx):
                        rows.append({
                            'func': func_name, 'grid': grid, 'k': k,
                            'feature_idx': i,
                            'feature_name': feat_names[i] if i < len(feat_names) else f'x{i}',
                            'test_r2': float('nan'), 'attribution': float('nan'),
                            'n_tp': -1, 'tp_norm': str(e), 'tp_raw': str(e),
                            'first_tp_norm': float('nan'), 'first_tp_raw': float('nan'),
                            'ckpt_path': '',
                        })

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nSaved {len(df)} rows → {args.out}")


if __name__ == '__main__':
    main()
