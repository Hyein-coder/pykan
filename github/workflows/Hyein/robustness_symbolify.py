"""Robustness of KAN ranking-transition points to symbolification.

This standalone driver measures how stable the **location and number** of KAN
ranking-transition points are when an edge activation is swapped between its
learned B-spline form and a symbolic (closed-form) approximation. It does this
by building **two readings of the SAME persisted network** and comparing the
ranking transitions each yields:

  * SPLINE READING  -- every (originally) symbolified edge is forced back onto
    its persisted spline branch (``act_fun[0].mask=1``, ``symbolic_enabled=False``).
  * SYMBOLIC READING -- the symbolic branch is used. For an already-symbolified
    model that is the as-saved state; for a pure-spline model we run
    ``auto_symbolic`` (KANRegressor production defaults) to introduce it.

Both readings are evaluated with ``find_ranking_transitions`` from
``bspline_curvature`` on a SHARED normalized x-grid (so they are directly
comparable), then transitions are denormalized to raw input space and matched
nearest-neighbour per feature.

Two construction directions:
  * "invert-symbolify"   -- model saved symbolified; we reconstruct the spline side.
  * "introduce-symbolify"-- model saved pure-spline; we construct the symbolic side.

This module does NOT modify any pykan library file or any existing Hyein
script. It only reads them and reuses their patterns (``KANRegressor.load_model``,
``toy_KAN_analyze.py`` sections 1-2 load/regen, ``bspline_curvature`` ranking
transitions).

Run in the ``pykan-new`` conda env:
    python github/workflows/Hyein/robustness_symbolify.py [func_name]
"""

import argparse
import copy
import os
import traceback

import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

# --- YAML tuple constructor (mirror toy_KAN_analyze.py header) ----------------
def _tuple_constructor(loader, node):
    return tuple(loader.construct_sequence(node))

yaml.add_constructor('tag:yaml.org,2002:python/tuple', _tuple_constructor,
                     Loader=yaml.SafeLoader)
try:
    yaml.add_constructor('tag:yaml.org,2002:python/tuple', _tuple_constructor,
                         Loader=yaml.Loader)
except AttributeError:
    pass

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from github.workflows.Hyein.toy_KAN_sweep import KANRegressor, FUNCTION_ZOO
from github.workflows.Hyein.bspline_curvature import (
    find_ranking_transitions, symbolic_edge_info, feature_sensitivity,
)

# auto_symbolic defaults = KANRegressor production defaults.
SYM_LIB = ['sin', 'cos', 'x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt',
           'tanh', '1/x', '1/x^2']
SYM_A_RANGE = (-10, 10)
SYM_B_RANGE = (-10, 10)
SYM_R2_THRESHOLD = 0.0
SYM_WEIGHT_SIMPLE = 0.0

DEFAULT_FUNCS = ['conditional', 'damping_sin', 'log2', 'exponential',
                 'logarithm', 'rosenbrock', 'ishigami']

# Status taxonomy ------------------------------------------------------------
STATUS_OK = 'ok'
STATUS_SYMBOLIFY_FAILED = 'symbolify_failed'
STATUS_NO_SYMBOLIFICATION = 'no_symbolification'
STATUS_SYMBOLIC_DEGENERATE = 'symbolic_degenerate'
STATUS_SPLINE_READING_UNAVAILABLE = 'spline_reading_unavailable'
STATUS_LOAD_FAILED = 'load_failed'

SA_RC = {
    'figure.dpi': 150, 'figure.facecolor': 'white', 'figure.autolayout': True,
    'axes.facecolor': 'white', 'axes.edgecolor': '#444444', 'axes.linewidth': 0.8,
    'axes.labelsize': 11, 'axes.labelcolor': 'black', 'axes.grid': False,
    'xtick.labelsize': 9, 'xtick.color': 'black', 'ytick.labelsize': 9,
    'ytick.color': 'black', 'font.family': 'sans-serif', 'font.size': 9,
    'legend.fontsize': 7, 'legend.framealpha': 0.0, 'lines.linewidth': 1.3,
    'savefig.dpi': 150, 'savefig.bbox': 'tight', 'savefig.facecolor': 'white',
}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _shared_x_grid(model, layer=0, n_eval=400):
    """Dense normalized sweep over the layer-0 interior knot range (section 3.6)."""
    act = model.act_fun[layer]
    k = model.k
    knots = act.grid[:, k - 1:-2].detach().cpu().numpy()
    lo, hi = float(np.min(knots)), float(np.max(knots))
    return np.linspace(lo, hi, n_eval)


def _denorm_feature(vals_norm, feat_idx, scaler_X, nx):
    """Denormalize a list of normalized values for ONE feature to raw space.

    Builds a full-width dummy row, sets the feature column, inverse-transforms
    (the §3.6/§3.8 convention). Returns a numpy array (possibly empty).
    """
    vals = list(vals_norm or [])
    if not vals:
        return np.array([], dtype=float)
    dummy = np.zeros((len(vals), nx))
    dummy[:, feat_idx] = vals
    return scaler_X.inverse_transform(dummy)[:, feat_idx]


def _r2_on_test(model, test_input_norm, y_test_norm):
    """R2 of a reading on the held-out test set, in normalized space.

    Forwards a *reading* (spline/symbolic deepcopy) -- never the clean loaded
    model (see deepcopy note in analyze_function). Mirrors the toy_KAN_sweep.py
    convention: r2_score on normalized y. Returns NaN on any failure.
    """
    try:
        with torch.no_grad():
            y_pred = model.forward(test_input_norm).detach().cpu().numpy().reshape(-1)
        return float(r2_score(np.asarray(y_test_norm).reshape(-1), y_pred))
    except Exception as e:
        print(f"      [r2] scoring failed (ignoring): {e}")
        return float('nan')


def _sensitivity_is_degenerate(model, x_grid, layer=0):
    """True if every feature's s_i(x) is ~0 / NaN over the whole grid."""
    in_dim = model.act_fun[layer].coef.shape[0]
    peak = 0.0
    for i in range(in_dim):
        s = feature_sensitivity(model, i, x_grid, layer=layer)
        if np.any(np.isfinite(s)):
            peak = max(peak, float(np.nanmax(np.abs(s))))
    return peak < 1e-9


def _build_spline_reading(model, sym_info):
    """deepcopy with symbolified edges (ALL layers) forced back onto the spline.

    Restores the spline mask (=1) on every symbolified edge in EVERY layer, then
    disables the symbolic branch globally. Scanning all layers (not just layer 0)
    is required for multi-layer models: ``symbolic_enabled=False`` turns OFF the
    symbolic branch in every layer, so any layer left with spline mask=0 would
    output nothing and decapitate the network -- producing a spurious negative
    R2 even when the layer-0 transition analysis is perfectly valid. Pruned edges
    (no symbolic fn, spline mask already 0) are left untouched, so pruning is
    preserved. For a pure-spline model this is a clean deepcopy with symbolic off.

    ``sym_info`` (layer-0 symbolic edges) is kept for backward compatibility but
    is no longer the sole source; symbolic_edge_info is queried per layer.
    """
    m = copy.deepcopy(model)
    for layer in range(len(m.act_fun)):
        for (i, j) in symbolic_edge_info(m, layer):
            m.act_fun[layer].mask.data[i, j] = 1.0
    m.symbolic_enabled = False
    return m


def _build_symbolic_reading(model, sym_info, train_input_norm, refit, dataset,
                            steps, lr, stop_grid):
    """deepcopy that uses the symbolic branch.

    already-symbolified (sym_info non-empty): use as-saved, no auto_symbolic.
    pure-spline (sym_info empty): run auto_symbolic with production defaults;
    optionally a short LBFGS refit. Returns (model_copy, n_edges, refit_done).
    Raises on auto_symbolic failure (caller maps to symbolify_failed).
    """
    m = copy.deepcopy(model)
    refit_done = False
    if sym_info:
        # As-saved symbolic reading.
        n_edges = len(symbolic_edge_info(m, 0))
        return m, n_edges, refit_done

    # Introduce-symbolify on a pure-spline model.
    m.symbolic_enabled = True
    # auto_symbolic needs cached activations: one forward pass first.
    m.forward(train_input_norm)
    m.auto_symbolic(lib=SYM_LIB, a_range=SYM_A_RANGE, b_range=SYM_B_RANGE,
                    r2_threshold=SYM_R2_THRESHOLD, weight_simple=SYM_WEIGHT_SIMPLE,
                    verbose=0)
    n_edges = len(symbolic_edge_info(m, 0))
    if refit and n_edges > 0:
        # Mirror toy_KAN_sweep.py lines ~205-206 (short LBFGS post-symbolic fit).
        try:
            m.fit(dataset, opt='LBFGS', steps=steps,
                  stop_grid_update_step=stop_grid, lr=lr)
            refit_done = True
        except Exception as e:
            print(f"      [refit] short LBFGS fit failed (ignoring): {e}")
    return m, n_edges, refit_done


def _match_transitions(spline_pts, sym_pts, tol):
    """Nearest-neighbour match two sets of raw-space points (same feature).

    Greedy by ascending pair distance, each point used once, only pairs within
    ``tol`` matched. Returns (matches, added, dropped):
      matches : list of (spline_pt, sym_pt, drift)   drift = sym - spline
      added   : symbolic points with no spline partner (introduced by symbolify)
      dropped : spline points with no symbolic partner (removed by symbolify)
    """
    sp = list(enumerate(spline_pts))
    sy = list(enumerate(sym_pts))
    pairs = []
    for (a, pa) in sp:
        for (b, pb) in sy:
            d = abs(pb - pa)
            if d <= tol:
                pairs.append((d, a, b))
    pairs.sort(key=lambda t: t[0])
    used_s, used_y, matches = set(), set(), []
    for d, a, b in pairs:
        if a in used_s or b in used_y:
            continue
        used_s.add(a)
        used_y.add(b)
        matches.append((spline_pts[a], sym_pts[b], sym_pts[b] - spline_pts[a]))
    added = [sym_pts[b] for (b, _) in enumerate(sym_pts) if b not in used_y]
    dropped = [spline_pts[a] for (a, _) in enumerate(spline_pts) if a not in used_s]
    return matches, added, dropped


# ----------------------------------------------------------------------------
# Per-function analysis
# ----------------------------------------------------------------------------
def analyze_function(func_name, rel_thresh=0.1, refit=False, match_frac=0.05,
                     device=None):
    """Build spline + symbolic readings of one function and compare transitions.

    Returns a dict with keys: ``func``, ``status``, ``reason``, ``direction``,
    ``n_edges_symbolified``, ``per_feature`` (compare records), ``summary`` row,
    ``paths`` (written files). Never raises on per-function failure -- failures
    are captured into ``status``.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    root_dir = os.path.join(os.getcwd(), 'github', 'workflows', 'Hyein',
                            'analytical_results', func_name)
    savepath = os.path.join(root_dir, 'kan_models')
    ckpt_path = os.path.join(savepath, f'{func_name}_best_kan_model')
    scaler_x_path = os.path.join(savepath, f'{func_name}_scaler_X.pkl')
    scaler_y_path = os.path.join(savepath, f'{func_name}_scaler_y.pkl')

    result = {
        'func': func_name, 'status': STATUS_LOAD_FAILED, 'reason': '',
        'direction': '', 'n_edges_symbolified': 0, 'per_feature': [],
        'summary': None, 'paths': {},
    }

    # --- 1. Load model + scalers (toy_KAN_analyze.py §1) ----------------------
    if not (os.path.exists(scaler_x_path) and os.path.exists(scaler_y_path)):
        result['reason'] = 'scalers not found'
        print(f"[{func_name}] status={result['status']}: {result['reason']}")
        return result
    scaler_X = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)
    try:
        wrapper = KANRegressor(device=device)
        wrapper.load_model(ckpt_path)
        model = wrapper.model
    except Exception as e:
        result['reason'] = f'load_model failed: {e}'
        print(f"[{func_name}] status={result['status']}: {result['reason']}")
        return result

    # --- 2. Regenerate train data + one forward pass (toy_KAN_analyze.py §2) --
    config = FUNCTION_ZOO[func_name]
    target_func = config['func']
    bounds = config['bounds']
    feat_names = config['names']
    nx = len(bounds)

    np.random.seed(0)  # reproducible regen; data only used to re-fit/symbolify
    X_raw = np.random.uniform(low=[b[0] for b in bounds],
                              high=[b[1] for b in bounds], size=(1000, nx))
    y_raw = np.apply_along_axis(target_func, 1, X_raw).reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw,
                                                        test_size=0.2,
                                                        random_state=42)
    X_train_norm = scaler_X.transform(X_train)
    y_train_norm = scaler_y.transform(y_train)
    train_input_norm = torch.tensor(X_train_norm, dtype=torch.float32, device=device)
    # Held-out test set (2nd & 4th train_test_split outputs) for R2 scoring.
    X_test_norm = scaler_X.transform(X_test)
    y_test_norm = scaler_y.transform(y_test)
    test_input_norm = torch.tensor(X_test_norm, dtype=torch.float32, device=device)
    dataset = {
        'train_input': train_input_norm,
        'train_label': torch.tensor(y_train_norm, dtype=torch.float32,
                                    device=device).reshape(-1, 1),
        'test_input': train_input_norm,
        'test_label': torch.tensor(y_train_norm, dtype=torch.float32,
                                   device=device).reshape(-1, 1),
    }
    # NOTE: do NOT run model.forward on the loaded `model` here. A forward pass
    # caches non-leaf tensors (spline_preacts etc.) that break copy.deepcopy
    # ("Only Tensors created explicitly by the user support deepcopy"). We
    # deepcopy the two readings from the *clean* loaded model first; each copy
    # runs its own forward (auto_symbolic / find_ranking_transitions) afterward.

    # --- 3. Detect saved state + log construction direction -------------------
    # symbolic_edge_info reads masks only -- no forward needed.
    sym_info_saved = symbolic_edge_info(model, 0)
    already_symbolified = bool(sym_info_saved)
    direction = 'invert-symbolify' if already_symbolified else 'introduce-symbolify'
    result['direction'] = direction
    print(f"[{func_name}] direction={direction}; "
          f"saved symbolic edges={sorted(sym_info_saved.items())}")

    x_grid = _shared_x_grid(model, layer=0, n_eval=400)  # reads grid only

    # --- 4. SPLINE READING ----------------------------------------------------
    r2_spline, r2_symbolic = np.nan, np.nan
    spline_model = _build_spline_reading(model, sym_info_saved)
    spline_degenerate = _sensitivity_is_degenerate(spline_model, x_grid, layer=0)
    spline_trans, spline_info = find_ranking_transitions(
        spline_model, x_grid=x_grid, rel_thresh=rel_thresh, layer=0)
    r2_spline = _r2_on_test(spline_model, test_input_norm, y_test_norm)

    # --- 5. SYMBOLIC READING --------------------------------------------------
    steps = getattr(wrapper, 'steps', 20)
    lr = getattr(wrapper, 'lr', 0.1)
    stop_grid = getattr(wrapper, 'stop_grid_update_step', 20)
    sym_failed = False
    try:
        sym_model, n_edges, refit_done = _build_symbolic_reading(
            model, sym_info_saved, train_input_norm, refit, dataset,
            steps, lr, stop_grid)
    except Exception as e:
        sym_failed = True
        n_edges = 0
        refit_done = False
        sym_model = None
        print(f"[{func_name}] auto_symbolic raised: {e}")
        traceback.print_exc()
    if not sym_failed and sym_model is not None:
        r2_symbolic = _r2_on_test(sym_model, test_input_norm, y_test_norm)

    result['n_edges_symbolified'] = int(n_edges if already_symbolified or not sym_failed else 0)

    # --- 6. Status determination ---------------------------------------------
    sym_trans, sym_info_dict, sym_degenerate = [], {}, False
    if sym_failed:
        status = STATUS_SYMBOLIFY_FAILED
        reason = 'auto_symbolic raised; spline-only row emitted'
    else:
        sym_edge_info = symbolic_edge_info(sym_model, 0)
        sym_info_dict = sym_edge_info
        result['n_edges_symbolified'] = len(sym_edge_info)
        if not already_symbolified and len(sym_edge_info) == 0:
            status = STATUS_NO_SYMBOLIFICATION
            reason = 'auto_symbolic produced 0 symbolified edges; null contrast'
        else:
            sym_degenerate = _sensitivity_is_degenerate(sym_model, x_grid, layer=0)
            sym_trans, _ = find_ranking_transitions(
                sym_model, x_grid=x_grid, rel_thresh=rel_thresh, layer=0)
            if sym_degenerate and not spline_degenerate:
                status = STATUS_SYMBOLIC_DEGENERATE
                reason = 'symbolic-side s_i flat/NaN while spline-side nonzero'
            elif already_symbolified and spline_degenerate:
                status = STATUS_SPLINE_READING_UNAVAILABLE
                reason = ('mask-inverted spline reading is s_i~0 everywhere; '
                          'reporting symbolic-only')
            else:
                status = STATUS_OK
                reason = ''
    result['status'] = status
    result['reason'] = reason
    print(f"[{func_name}] status={status}"
          + (f": {reason}" if reason else "")
          + f"; n_edges_symbolified={result['n_edges_symbolified']}; "
          f"refit={'on' if (refit and not already_symbolified) else 'off'}; "
          f"R2(spline)={r2_spline:.4f}; R2(symbolic)={r2_symbolic:.4f}")

    # --- 7. Per-feature compare in RAW space ----------------------------------
    spline_pts_norm = {i: sorted(t['point'] for t in spline_trans if t['feat_idx'] == i)
                       for i in range(nx)}
    sym_pts_norm = {i: sorted(t['point'] for t in sym_trans if t['feat_idx'] == i)
                    for i in range(nx)}
    spline_dom_norm = {i: sorted(t['point'] for t in spline_trans
                                 if t['feat_idx'] == i and t['dominant'])
                       for i in range(nx)}
    sym_dom_norm = {i: sorted(t['point'] for t in sym_trans
                              if t['feat_idx'] == i and t['dominant'])
                    for i in range(nx)}

    sym_edge_names = ', '.join(sorted(set(str(v) for v in sym_info_dict.values()))) \
        if sym_info_dict else ''

    n_spline_total, n_sym_total = 0, 0
    n_spline_dom, n_sym_dom = 0, 0
    all_drifts, all_drift_fracs = [], []
    n_matched_total, n_added_total, n_dropped_total = 0, 0, 0
    csv_rows = []
    per_feature_S = {}

    for i in range(nx):
        lo_raw, hi_raw = bounds[i]
        width = float(hi_raw - lo_raw)
        tol = match_frac * width

        sp_raw = sorted(_denorm_feature(spline_pts_norm[i], i, scaler_X, nx).tolist())
        sy_raw = sorted(_denorm_feature(sym_pts_norm[i], i, scaler_X, nx).tolist())
        n_spline_total += len(sp_raw)
        n_sym_total += len(sy_raw)
        n_spline_dom += len(spline_dom_norm[i])
        n_sym_dom += len(sym_dom_norm[i])

        # store sensitivity curves for the plot
        per_feature_S[i] = {
            'spline': spline_info['S'][i],
            'sym': (None if sym_failed or status == STATUS_NO_SYMBOLIFICATION
                    else feature_sensitivity(sym_model, i, x_grid, layer=0)),
        }

        matches, added, dropped = _match_transitions(sp_raw, sy_raw, tol)
        n_matched_total += len(matches)
        n_added_total += len(added)
        n_dropped_total += len(dropped)

        for (sp_pt, sy_pt, drift) in matches:
            frac = drift / width if width else np.nan
            all_drifts.append(abs(drift))
            all_drift_fracs.append(abs(frac))
            csv_rows.append({
                'feature': feat_names[i], 'feat_idx': i, 'branch': 'matched',
                'raw_point_spline': sp_pt, 'raw_point_symbolic': sy_pt,
                'matched': 1, 'added': 0, 'dropped': 0,
                'drift': drift, 'drift_frac': frac,
                'sym_edge_names': sym_edge_names, 'status': status,
            })
        for pt in added:
            csv_rows.append({
                'feature': feat_names[i], 'feat_idx': i, 'branch': 'symbolic',
                'raw_point_spline': np.nan, 'raw_point_symbolic': pt,
                'matched': 0, 'added': 1, 'dropped': 0,
                'drift': np.nan, 'drift_frac': np.nan,
                'sym_edge_names': sym_edge_names, 'status': status,
            })
        for pt in dropped:
            csv_rows.append({
                'feature': feat_names[i], 'feat_idx': i, 'branch': 'spline',
                'raw_point_spline': pt, 'raw_point_symbolic': np.nan,
                'matched': 0, 'added': 0, 'dropped': 1,
                'drift': np.nan, 'drift_frac': np.nan,
                'sym_edge_names': sym_edge_names, 'status': status,
            })

        result['per_feature'].append({
            'feat_idx': i, 'name': feat_names[i],
            'spline_raw': sp_raw, 'sym_raw': sy_raw,
            'matched': matches, 'added': added, 'dropped': dropped,
        })

    union_pairs = n_matched_total + n_added_total + n_dropped_total
    match_rate = (n_matched_total / union_pairs) if union_pairs else np.nan
    median_drift = float(np.median(all_drifts)) if all_drifts else np.nan
    max_drift = float(np.max(all_drifts)) if all_drifts else np.nan
    median_drift_frac = float(np.median(all_drift_fracs)) if all_drift_fracs else np.nan
    max_drift_frac = float(np.max(all_drift_fracs)) if all_drift_fracs else np.nan

    result['summary'] = {
        'func': func_name, 'status': status, 'direction': direction,
        'n_edges_symbolified': result['n_edges_symbolified'],
        'n_spline': n_spline_total, 'n_sym': n_sym_total,
        'n_spline_dominant': n_spline_dom, 'n_sym_dominant': n_sym_dom,
        'n_matched': n_matched_total, 'n_added': n_added_total,
        'n_dropped': n_dropped_total, 'count_match_rate': match_rate,
        'median_drift': median_drift, 'max_drift': max_drift,
        'median_drift_frac': median_drift_frac, 'max_drift_frac': max_drift_frac,
        'refit': bool(refit and not already_symbolified),
        'rel_thresh': rel_thresh, 'match_frac': match_frac,
        'r2_spline': r2_spline, 'r2_symbolic': r2_symbolic,
        'reason': reason,
    }

    # --- 8. CSV ---------------------------------------------------------------
    csv_path = os.path.join(savepath, f'{func_name}_symbolify_robustness.csv')
    pd.DataFrame(csv_rows if csv_rows else [{
        'feature': '', 'feat_idx': -1, 'branch': 'none',
        'raw_point_spline': np.nan, 'raw_point_symbolic': np.nan,
        'matched': 0, 'added': 0, 'dropped': 0, 'drift': np.nan,
        'drift_frac': np.nan, 'sym_edge_names': sym_edge_names, 'status': status,
    }]).to_csv(csv_path, index=False)
    result['paths']['csv'] = csv_path

    # --- 9. Figure: per-feature s_i(x), spline (solid) vs symbolic (dashed) ---
    fig_paths = _plot_robustness(
        func_name, x_grid, per_feature_S, spline_info['tau'],
        spline_pts_norm, sym_pts_norm, feat_names, nx, savepath,
        direction, status)
    result['paths'].update(fig_paths)

    return result


def _plot_robustness(func_name, x_grid, per_feature_S, tau, spline_pts_norm,
                     sym_pts_norm, feat_names, nx, savepath, direction, status):
    """Per-feature s_i(x): spline (solid) vs symbolic (dashed), τ lines, vlines."""
    paths = {}
    with plt.rc_context(SA_RC):
        ncols = min(nx, 3)
        nrows = (nx + ncols - 1) // ncols
        fig, axs = plt.subplots(nrows, ncols, squeeze=False,
                                figsize=(4.2 * ncols, 3.2 * nrows))
        axs_flat = axs.flatten()
        spline_color, sym_color = '#1f77b4', '#d62728'
        for i in range(nx):
            ax = axs_flat[i]
            S = per_feature_S[i]
            ax.plot(x_grid, S['spline'], color=spline_color, lw=1.4, ls='-',
                    label=r"spline $|\phi'|$")
            if S['sym'] is not None:
                ax.plot(x_grid, S['sym'], color=sym_color, lw=1.4, ls='--',
                        label=r"symbolic $|\phi'|$")
            ax.axhline(tau, color='black', ls='--', lw=0.9, alpha=0.7,
                       label=r"$\tau$")

            first = True
            for p in spline_pts_norm[i]:
                ax.axvline(p, color=spline_color, ls='-', alpha=0.7, lw=1.0,
                           label='spline transition' if first else '_')
                first = False
            first = True
            for p in sym_pts_norm[i]:
                ax.axvline(p, color=sym_color, ls=':', alpha=0.8, lw=1.2,
                           label='symbolic transition' if first else '_')
                first = False

            ax.set_xlabel(f"normalized {feat_names[i]}")
            ax.set_ylabel(r"$s_i=\sum_j|\phi'_{ij}|$")
            ax.legend(loc='best', fontsize=6)
        for k in range(nx, len(axs_flat)):
            axs_flat[k].set_visible(False)
        fig.suptitle(f"{func_name} — symbolify robustness "
                     f"[{direction}; status={status}]",
                     fontsize=11, fontweight='bold')
        base = os.path.join(savepath, f"{func_name}_symbolify_robustness")
        for ext, kw in [('.png', dict(dpi=300)), ('.svg', dict(format='svg')),
                        ('.eps', dict(format='eps'))]:
            fig.savefig(base + ext, **kw)
            paths[ext.lstrip('.')] = base + ext
        plt.close(fig)
    return paths


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Robustness of KAN ranking transitions to symbolification.")
    parser.add_argument('func_name', type=str, nargs='?', default=None,
                        help='Optional single function; default = full set.')
    parser.add_argument('--rel-thresh', type=float, default=0.1,
                        help='tau = rel_thresh * max_i max_x s_i (default 0.1).')
    parser.add_argument('--refit', action='store_true',
                        help='Short LBFGS refit after introduce-symbolify '
                             '(pure-spline models only). Default off.')
    parser.add_argument('--match-frac', type=float, default=0.05,
                        help='Match tolerance as fraction of domain width '
                             '(default 0.05).')
    args = parser.parse_args()

    funcs = [args.func_name] if args.func_name else DEFAULT_FUNCS

    summary_rows = []
    status_table = []
    for fn in funcs:
        print("\n" + "=" * 60)
        print(f"FUNCTION: {fn}")
        print("=" * 60)
        try:
            res = analyze_function(fn, rel_thresh=args.rel_thresh,
                                   refit=args.refit, match_frac=args.match_frac)
        except Exception as e:
            print(f"[{fn}] UNEXPECTED failure: {e}")
            traceback.print_exc()
            res = {'func': fn, 'status': 'unexpected_error', 'summary': None}
        if res.get('summary'):
            summary_rows.append(res['summary'])
        status_table.append((fn, res.get('status', 'unknown'),
                             res.get('direction', ''),
                             res.get('n_edges_symbolified', 0)))

    # Combined summary CSV for the downstream aggregation agent.
    workspace = os.path.join(os.getcwd(), 'github', 'workflows', 'Hyein', '_workspace')
    os.makedirs(workspace, exist_ok=True)
    summary_csv = os.path.join(workspace, 'robustness_symbolify_summary.csv')
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
        print(f"\n[summary] Combined summary written: {summary_csv}")

    print("\n" + "=" * 60)
    print("STATUS TABLE (func | status | direction | n_edges_symbolified)")
    print("=" * 60)
    for fn, st, di, ne in status_table:
        print(f"  {fn:<14} | {st:<26} | {di:<18} | {ne}")


if __name__ == '__main__':
    main()
