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
DEFAULT_KS    = [0,1,2,3,4,5,6,7,8]
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


# Minimum number of data points a segment must contain for its sectional
# importance rank to be considered reliable (mirrors the >=5 guard in
# sectional_gsa.py). Candidates whose before/after segment is sparser than this
# cannot be confirmed by the rank-change criterion.
MIN_SEG_SAMPLES = 5


def _feature_rank(model, X_sub, feat_idx):
    """Importance rank of ``feat_idx`` over the data subset ``X_sub``.

    Recomputes KAN's own attribution (``model.feature_score``) restricted to the
    given subset by running a forward pass on it, then returns the rank of
    ``feat_idx`` in the descending feature_score ordering (0 = most important).

    Returns ``None`` if the subset is too small for a reliable score.
    """
    if X_sub.shape[0] < MIN_SEG_SAMPLES:
        return None
    with torch.no_grad():
        model.forward(X_sub)
    scores = model.feature_score.detach().cpu().numpy()
    # argsort descending; rank = position of feat_idx in that order.
    order = np.argsort(-scores, kind='stable')
    return int(np.where(order == feat_idx)[0][0])


def _confirm_by_rank_change(model, X_t, feat_idx, knots, cand_idx, device):
    """Filter sign-reversal candidates by the importance-rank-change criterion.

    Walks the sorted candidate knot indices left-to-right. For each candidate at
    knot value ``c``, the *before* segment spans ``[last_confirmed_tp, c)`` and
    the *after* segment spans ``[c, +inf)`` along ``feat_idx`` (i.e. the ranges
    are delimited by the previously confirmed transition points). The candidate
    is confirmed only if feature ``feat_idx``'s importance rank — computed via
    ``model.feature_score`` over each segment — differs between the two. Once a
    candidate is confirmed it becomes the left boundary for the next one.

    Returns the list of confirmed knot indices (a subset of ``cand_idx``).
    """
    x_col = X_t[:, feat_idx]
    lo_val = float('-inf')            # left boundary = last confirmed TP
    confirmed = []
    for ir in cand_idx:
        c_val = float(knots[ir])
        before_mask = (x_col >= lo_val) & (x_col < c_val)
        after_mask  = (x_col >= c_val)

        rank_before = _feature_rank(model, X_t[before_mask], feat_idx)
        rank_after  = _feature_rank(model, X_t[after_mask], feat_idx)

        # Need both segments populated enough to judge a rank change.
        if rank_before is None or rank_after is None:
            continue
        if rank_before != rank_after:
            confirmed.append(ir)
            lo_val = c_val            # advance left boundary to confirmed TP
    return confirmed


def extract_transition_points(model, X=None, device='cpu'):
    """Return {feat_idx: [tp_norm, ...]} from layer-0 spline coefficients.

    Two criteria are applied per feature:

    1. **Sign reversal** — candidate knot indices are those where the spline
       coefficient difference reverses sign. Uses 2nd-derivative sign reversals
       for depth-1 KANs (single layer), 1st-derivative sign reversals for
       depth-2 KANs, mirroring toy_KAN_analyze.py.
    2. **Importance-rank change** — when ``X`` is provided and the model has >=2
       inputs, each candidate is confirmed only if feature ``i``'s importance
       rank (from ``model.feature_score``) changes between the segment before and
       after it; segments are delimited by already-confirmed transition points
       (see ``_confirm_by_rank_change``). With a single input, or when ``X`` is
       None, the rank criterion is skipped and all sign-reversal candidates are
       kept.

    Parameters
    ----------
    model : MultKAN
        Trained KAN whose layer-0 splines are analyzed.
    X : np.ndarray, optional
        ``[N, ni]`` normalized-space input data used to compute sectional
        importance ranks. If None, only the sign-reversal criterion is applied.
    device : str
        Torch device for the forward passes when ``X`` is given.
    """
    l = 0
    act   = model.act_fun[l]
    ni, no = act.coef.shape[:2]
    coef  = act.coef.tolist()          # (ni, no, n_coef)
    depth = len(model.act_fun)

    use_rank = X is not None and ni >= 2
    X_t = (torch.as_tensor(X, dtype=torch.float32, device=device)
           if use_rank else None)

    tps = {}
    for i in range(ni):
        knots = act.grid[i, model.k - 1:-2].cpu().detach().numpy()

        # ── criterion 1: collect sign-reversal candidate knot indices ──
        cand_idx = set()
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
                    cand_idx.add(ir)
        cand_idx = sorted(cand_idx)

        # ── criterion 2: confirm by importance-rank change ──
        if use_rank and cand_idx:
            cand_idx = _confirm_by_rank_change(
                model, X_t, i, knots, cand_idx, device)

        ips_all = [float(knots[ir]) for ir in cand_idx]
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
            )
            print(f"  [{func_name}] best_params loaded — "
                  f"steps={base['steps']}, lr={base['lr']}, "
                  f"lamb={base['lamb']}, lamb_coef={base['lamb_coef']}, "
                  f"lamb_entropy={base['lamb_entropy']}")
        else:
            base = dict(
                n_layers=1, steps=args.steps, lr=0.1,
                lamb=0.01, lamb_coef=0.1, lamb_entropy=0.1,
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
                        symbolic_enabled=False,   # skip — irrelevant for TPs
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

                    # ── transition points (sign reversal + rank change) ──
                    tps_norm = extract_transition_points(
                        model, X=X_norm, device=device)

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
