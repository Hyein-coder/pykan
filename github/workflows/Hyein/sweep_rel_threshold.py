"""Sweep the ranking-transition threshold ``rel_thresh`` to choose a principled value.

The KAN ranking-transition detector (``find_ranking_transitions``) flags a
"down" transition where a feature's bottom-layer local sensitivity
``s_i(x) = sum_j |phi'_{i,j}(x)|`` drops below ``tau = rel_thresh * max_i max_x s_i``.
A single fixed ``rel_thresh`` (0.1) is hard to justify, so this standalone driver
sweeps a grid of ``rel_thresh`` values for the ``conditional`` function and scores,
per threshold, whether the true kink is detected cleanly.

``conditional``: ``f = 2*x0 + x1  (x0<0)  else  x1``. x0 (feat_idx 0) becomes
inert at x0=0 (normalized 0.5 / raw 0). A good detector finds **exactly one clean
x0 down-transition** there. ``tau`` must sit above the dead segment (x0>0, s0~0)
and below the active segment (x0<0, slope~2); too low pushes the crossing toward
x0>>0, too high crosses early inside the active region -- so ``loc_err`` varies
with ``rel_thresh`` and the sweep finds the value that best localizes the kink.

Load-only: reuses ``find_ranking_transitions`` and the saved KAN; no retraining,
no data regeneration. Modifies no existing script.

Run in the ``pykan-new`` conda env:
    PYTHONPATH=. PYTHONUTF8=1 python -m github.workflows.Hyein.sweep_rel_threshold
"""
import argparse
import os
import traceback

import joblib
import numpy as np
import pandas as pd
import yaml

# --- YAML tuple constructor (mirror robustness_symbolify.py header) -----------
# The saved KAN config YAML stores python/tuple tags; register a constructor
# before any checkpoint load or yaml.load chokes.
def _tuple_constructor(loader, node):
    return tuple(loader.construct_sequence(node))

yaml.add_constructor('tag:yaml.org,2002:python/tuple', _tuple_constructor,
                     Loader=yaml.SafeLoader)
try:
    yaml.add_constructor('tag:yaml.org,2002:python/tuple', _tuple_constructor,
                         Loader=yaml.Loader)
except AttributeError:
    pass

from github.workflows.Hyein.toy_KAN_sweep import KANRegressor, FUNCTION_ZOO
from github.workflows.Hyein.bspline_curvature import find_ranking_transitions

# Default rel_thresh grid (skill spec).
DEFAULT_REL_THRESHS = [0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30,
                       0.40, 0.50]

# True kink for conditional: x0=0 -> normalized 0.5 (raw 0).
TRUE_KINK_NORM = 0.5

RESULTS_ROOT = os.path.join('github', 'workflows', 'Hyein', 'analytical_results')
DEFAULT_OUT = os.path.join('github', 'workflows', 'Hyein', 'figures_for_paper')

FEAT_IDX = 0  # x0 is the conditional feature.


# ----------------------------------------------------------------------------
# Load helpers
# ----------------------------------------------------------------------------
def _load_model(name):
    """Load the persisted KAN for ``name`` (mirror aggregate_ranking_transitions)."""
    savepath = os.path.join(RESULTS_ROOT, name, 'kan_models')
    ckpt = os.path.join(savepath, f'{name}_best_kan_model')
    mw = KANRegressor(device='cpu')
    mw.load_model(ckpt)
    return mw.model


def _load_scaler_X(name):
    """Load the input scaler for raw denormalization."""
    savepath = os.path.join(RESULTS_ROOT, name, 'kan_models')
    return joblib.load(os.path.join(savepath, f'{name}_scaler_X.pkl'))


def _denorm_feature(vals_norm, feat_idx, scaler_X, nx):
    """Denormalize normalized values of ONE feature to raw space.

    Builds a full-width dummy row, sets the feature column, inverse-transforms
    (mirror robustness_symbolify._denorm_feature). Returns a numpy array.
    """
    vals = list(vals_norm or [])
    if not vals:
        return np.array([], dtype=float)
    dummy = np.zeros((len(vals), nx))
    dummy[:, feat_idx] = vals
    return scaler_X.inverse_transform(dummy)[:, feat_idx]


# ----------------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------------
def sweep_function(name, rel_threshs, match_tol, layer=0):
    """Sweep ``rel_thresh`` for one function; return one metrics dict per value.

    For each rel_thresh: call find_ranking_transitions WITHOUT an explicit x_grid
    (so it uses the default data-range knots), collect feat-0 down/up points
    (normalized -> raw), and score detection of the true kink at normalized 0.5.

    ``match_tol`` is treated as an absolute normalized tolerance around 0.5
    (the data range maps to roughly [0.1, 0.9], so this is a sensible band).
    """
    model = _load_model(name)
    scaler_X = _load_scaler_X(name)
    nx = model.act_fun[layer].coef.shape[0]  # number of input features

    rows = []
    for rt in rel_threshs:
        transitions, info = find_ranking_transitions(
            model, rel_thresh=rt, layer=layer)
        tau = float(info['tau'])

        down_norm = [t['point'] for t in transitions
                     if t['feat_idx'] == FEAT_IDX and t['direction'] == 'down']
        up_norm = [t['point'] for t in transitions
                   if t['feat_idx'] == FEAT_IDX and t['direction'] == 'up']

        down_raw = _denorm_feature(down_norm, FEAT_IDX, scaler_X, nx)
        up_raw = _denorm_feature(up_norm, FEAT_IDX, scaler_X, nx)

        n_down = len(down_norm)
        n_up = len(up_norm)

        # loc_err = nearest x0 down-transition to the true kink (normalized).
        if n_down > 0:
            errs = np.abs(np.array(down_norm) - TRUE_KINK_NORM)
            loc_err = float(np.min(errs))
        else:
            loc_err = float('nan')

        # detected = some down point within match_tol (normalized) of the kink.
        detected = bool(n_down > 0 and loc_err <= match_tol)
        clean = bool(n_down == 1)  # exactly one x0 down-transition, no spurious extras

        rows.append({
            'func': name,
            'rel_thresh': float(rt),
            'feat_idx': FEAT_IDX,
            'n_down': n_down,
            'n_up': n_up,
            'points_norm': ';'.join(f'{v:.4f}' for v in down_norm),
            'points_raw': ';'.join(f'{v:.4f}' for v in down_raw),
            'tau': tau,
            'detected': detected,
            'loc_err': loc_err,
            'clean': clean,
        })
    return rows


def recommend(rows, match_tol):
    """Pick a recommended rel_thresh from the sweep rows.

    Among (detected & clean) rows pick min loc_err; tie-break by largest
    index-distance to the nearest non-(detected&clean) rel_thresh (robustness:
    sits in the middle of a good band). If none qualify, report none + the
    closest case (min loc_err among detected, else fewest n_down).
    Returns a dict suitable for the recommendation CSV.
    """
    func = rows[0]['func'] if rows else ''
    rel_threshs = [r['rel_thresh'] for r in rows]
    good_mask = [bool(r['detected'] and r['clean']) for r in rows]
    good_idx = [i for i, g in enumerate(good_mask) if g]

    if good_idx:
        # band = span of qualifying rel_threshs.
        band_lo = min(rel_threshs[i] for i in good_idx)
        band_hi = max(rel_threshs[i] for i in good_idx)
        best_loc = min(rows[i]['loc_err'] for i in good_idx)
        # candidates with min loc_err (within tiny epsilon).
        cand = [i for i in good_idx
                if abs(rows[i]['loc_err'] - best_loc) <= 1e-12]

        def robustness(i):
            # index-distance to nearest non-(detected&clean) index; larger = deeper
            # inside a clean band. If all rows are good, fall back to the span.
            bad = [j for j in range(len(rows)) if not good_mask[j]]
            if not bad:
                return len(rows)
            return min(abs(i - j) for j in bad)

        best_i = max(cand, key=robustness)
        return {
            'func': func,
            'recommended_rel_thresh': rows[best_i]['rel_thresh'],
            'loc_err': rows[best_i]['loc_err'],
            'band_lo': band_lo,
            'band_hi': band_hi,
            'note': (f'min loc_err in detected&clean band '
                     f'[{band_lo:g},{band_hi:g}]; tie-break by robustness'),
        }

    # No detected&clean row: be explicit, report the closest case.
    detected_idx = [i for i, r in enumerate(rows) if r['detected']]
    if detected_idx:
        best_i = min(detected_idx, key=lambda i: rows[i]['loc_err'])
        note = ('NO detected&clean rel_thresh; closest = detected but not clean '
                f"(n_down={rows[best_i]['n_down']})")
    elif rows:
        # nobody detected: fewest down-transitions (closest to a clean single hit).
        best_i = min(range(len(rows)),
                     key=lambda i: (rows[i]['n_down'], rows[i]['loc_err']
                                    if rows[i]['n_down'] else np.inf))
        note = ('NO detected rel_thresh; closest = fewest n_down '
                f"({rows[best_i]['n_down']})")
    else:
        return {'func': func, 'recommended_rel_thresh': float('nan'),
                'loc_err': float('nan'), 'band_lo': float('nan'),
                'band_hi': float('nan'), 'note': 'no rows'}

    return {
        'func': func,
        'recommended_rel_thresh': float('nan'),
        'loc_err': rows[best_i]['loc_err'],
        'band_lo': float('nan'),
        'band_hi': float('nan'),
        'note': note,
    }


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--funcs', nargs='+', default=['conditional'],
                    choices=list(FUNCTION_ZOO.keys()),
                    help='functions to sweep (currently scoped to conditional)')
    ap.add_argument('--rel-threshs', type=float, nargs='+',
                    default=DEFAULT_REL_THRESHS,
                    help='rel_thresh grid to sweep')
    ap.add_argument('--match-tol', type=float, default=0.05,
                    help='normalized tolerance around the true kink (0.5)')
    ap.add_argument('--layer', type=int, default=0,
                    help='KAN layer for the ranking-transition detector')
    ap.add_argument('--out', default=DEFAULT_OUT,
                    help='output directory for the CSVs')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    all_rows = []
    recs = []
    for name in args.funcs:
        try:
            rows = sweep_function(name, args.rel_threshs, args.match_tol,
                                  layer=args.layer)
            all_rows.extend(rows)
            rec = recommend(rows, args.match_tol)
            recs.append(rec)

            # Console report.
            print(f"\n=== {name} ===")
            for r in rows:
                print(f"  rel_thresh={r['rel_thresh']:.2f}  n_down={r['n_down']}  "
                      f"n_up={r['n_up']}  tau={r['tau']:.4f}  "
                      f"detected={r['detected']}  clean={r['clean']}  "
                      f"loc_err={r['loc_err']:.4f}  pts_norm=[{r['points_norm']}]")
            if not np.isnan(rec['recommended_rel_thresh']):
                print(f"  --> RECOMMENDED rel_thresh = "
                      f"{rec['recommended_rel_thresh']:g}  "
                      f"(loc_err={rec['loc_err']:.4f}, "
                      f"band=[{rec['band_lo']:g},{rec['band_hi']:g}])")
            else:
                print(f"  --> NO recommendation. {rec['note']}")
        except Exception as e:  # one missing model must not kill the run.
            traceback.print_exc()
            print(f"  {name}: FAILED ({e})")
            recs.append({'func': name, 'recommended_rel_thresh': float('nan'),
                         'loc_err': float('nan'), 'band_lo': float('nan'),
                         'band_hi': float('nan'), 'note': f'failed: {e}'})

    # Tidy sweep CSV.
    sweep_csv = os.path.join(args.out, 'rel_threshold_sweep.csv')
    cols = ['func', 'rel_thresh', 'feat_idx', 'n_down', 'n_up', 'points_norm',
            'points_raw', 'tau', 'detected', 'loc_err', 'clean']
    pd.DataFrame(all_rows, columns=cols).to_csv(sweep_csv, index=False)
    print(f"\nWrote {sweep_csv}")

    # Recommendation CSV.
    rec_csv = os.path.join(args.out, 'rel_threshold_recommendation.csv')
    rec_cols = ['func', 'recommended_rel_thresh', 'loc_err', 'band_lo',
                'band_hi', 'note']
    pd.DataFrame(recs, columns=rec_cols).to_csv(rec_csv, index=False)
    print(f"Wrote {rec_csv}")


if __name__ == '__main__':
    main()
