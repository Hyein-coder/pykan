"""
sweep_spline_inflection.py
==========================
Inflection-point analysis on every model trained by grid_k_sweep.py.

For each saved checkpoint in `_workspace/sweep_models/<func>/g{grid}_k{k}`, this
reproduces the `fig_spline` figure from toy_KAN_analyze.py (the Layer-0 spline
coefficient / slope / 2nd-slope panel with inflection points marked) and saves
it. Optionally also dumps the detected inflection points to a summary CSV.

Input
-----
    _workspace/sweep_models/<func>/scaler_X.pkl
    _workspace/sweep_models/<func>/g{grid}_k{k}{_config.yml,_state,_cache_data}

Output
------
    _workspace/sweep_spline/<func>/<func>_g{grid}_k{k}_spline_L0.png
    _workspace/sweep_spline/inflection_points.csv   (--csv, default on)

Usage
-----
    python sweep_spline_inflection.py
    python sweep_spline_inflection.py --funcs exponential log2 --formats png svg
    python sweep_spline_inflection.py --grids 3 5 7 --ks 2 3
"""

import argparse
import os
import re
import sys

import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, r'D:\pykan')

# Windows consoles default to cp949 here; keep emoji/unicode prints from crashing.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# YAML python/tuple constructor — same fix as toy_KAN_analyze.py, needed by KAN ckpt configs
def _tuple_constructor(loader, node):
    return tuple(loader.construct_sequence(node))


yaml.add_constructor('tag:yaml.org,2002:python/tuple', _tuple_constructor, Loader=yaml.SafeLoader)
try:
    yaml.add_constructor('tag:yaml.org,2002:python/tuple', _tuple_constructor, Loader=yaml.Loader)
except AttributeError:
    pass

from github.workflows.Hyein.toy_KAN_sweep import FUNCTION_ZOO
from kan.custom_multkan_ddp import KAN
from kan.experiments.analysis import find_indices_sign_revert

# ── paths ──────────────────────────────────────────────────────────────────────
HERE       = os.path.dirname(__file__)
MODELS_DIR = os.path.join(HERE, '_workspace', 'sweep_models')
OUT_DIR    = os.path.join(HERE, '_workspace', 'sweep_spline')
RESULTS_CSV = os.path.join(HERE, '_workspace', 'grid_k_sweep_results.csv')

CKPT_RE = re.compile(r'^g(\d+)_k(\d+)_config\.yml$')


def discover_checkpoints(func_dir):
    """Return [(grid, k, stem_path), ...] sorted by (grid, k)."""
    out = []
    for fn in os.listdir(func_dir):
        m = CKPT_RE.match(fn)
        if m:
            grid, k = int(m.group(1)), int(m.group(2))
            stem = os.path.join(func_dir, f'g{grid}_k{k}')
            out.append((grid, k, stem))
    return sorted(out, key=lambda t: (t[0], t[1]))


def load_r2_lookup():
    """(func, grid, k) -> test_r2, from the sweep results CSV (best-effort)."""
    if not os.path.exists(RESULTS_CSV):
        return {}
    df = pd.read_csv(RESULTS_CSV)
    lut = {}
    for _, r in df.drop_duplicates(['func', 'grid', 'k']).iterrows():
        lut[(r['func'], int(r['grid']), int(r['k']))] = r['test_r2']
    return lut


def draw_spline_figure(model, feat_names, device):
    """Replicate fig_spline from toy_KAN_analyze.py (lines 227-296), Layer 0.

    Returns (fig, inflection_points_per_input). The model must already have had a
    forward pass run so spline_preacts/postacts and feature_score are populated.
    """
    l = 0
    act = model.act_fun[l]
    ni, no = act.coef.shape[:2]
    coef = act.coef.tolist()
    depth = len(model.act_fun)

    scores_tot = model.feature_score.detach().cpu().numpy()
    sort_order_act = np.argsort(scores_tot)[::-1]
    feat_colors = [plt.get_cmap('RdYlBu')(x) for x in np.linspace(0.1, 0.9, ni)]

    inflection_points_per_input = [None] * ni

    fig_spline, axs_spline = plt.subplots(nrows=no, ncols=ni, squeeze=False,
                                          figsize=(4 * ni, 3 * no),
                                          constrained_layout=True)

    for col_pos, i in enumerate(sort_order_act):
        knot_points_actual = act.grid[i, model.k - 1:-2].cpu().detach().numpy()
        feature_inflections_all = []
        for j in range(no):
            ax2 = axs_spline[j, col_pos]
            coef_node = coef[i][j]
            knot_indices = np.arange(len(coef_node))

            slope = [x - y for x, y in zip(coef_node[1:], coef_node[:-1])]
            slope_2nd = [(x - y) * 10 for x, y in zip(slope[1:], slope[:-1])]

            ax2.plot(knot_indices, coef_node, marker='o',
                     color=feat_colors[col_pos], label='Coefficients')

            slope_indices = knot_indices[:-1] + 0.5
            ax2.bar(slope_indices, slope, width=0.3, align='center',
                    hatch='///', edgecolor='dimgray', facecolor='none', label='Slope')

            if depth == 1:
                ax2.bar(slope_indices[1:] - 0.3, slope_2nd, width=0.3, align='center',
                        hatch='xx', edgecolor='steelblue', facecolor='none', label='2nd Slope')

            ax2.set_xticks(knot_indices)
            ax2.set_xticklabels([f"{val:.2f}" for val in knot_points_actual],
                                rotation=45, fontsize=9)

            if depth == 1:
                idx_revert = find_indices_sign_revert(slope_2nd)
                idx_revert = [ir + 1 for ir in idx_revert]
            elif depth == 2:
                idx_revert = find_indices_sign_revert(slope)
            else:
                idx_revert = []

            if idx_revert:
                first_vline = True
                for ir in idx_revert:
                    inflection_val = knot_points_actual[ir]
                    feature_inflections_all.append(inflection_val)
                    label_to_add = "Inflection" if first_vline else "_"
                    ax2.axvline(x=ir, color='green', linestyle='--', alpha=0.7,
                                label=label_to_add)
                    first_vline = False

            ax2.set_xlabel(f"{feat_names[i]}")
            ax2.set_ylabel(f"node ({l+1}, {j})")
            ax2.axhline(0, color='dimgray', linestyle='--', alpha=0.4)
            ax2.legend(loc='best')

        feature_inflections = sorted(set(feature_inflections_all))
        inflection_points_per_input[i] = feature_inflections

    return fig_spline, inflection_points_per_input


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models-dir', default=MODELS_DIR)
    parser.add_argument('--out', default=OUT_DIR)
    parser.add_argument('--funcs', nargs='*', default=None,
                        help='subset of function names (default: all dirs found)')
    parser.add_argument('--grids', nargs='*', type=int, default=None,
                        help='subset of grid values (default: all found)')
    parser.add_argument('--ks', nargs='*', type=int, default=None,
                        help='subset of k values (default: all found)')
    parser.add_argument('--formats', nargs='*', default=['png'],
                        help='figure formats to save (png svg eps)')
    parser.add_argument('--n-samples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-csv', action='store_true',
                        help='skip writing the inflection_points.csv summary')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out, exist_ok=True)
    r2_lut = load_r2_lookup()

    funcs = args.funcs or sorted(
        d for d in os.listdir(args.models_dir)
        if os.path.isdir(os.path.join(args.models_dir, d))
    )

    summary_rows = []
    n_done = n_fail = 0

    for func in funcs:
        func_dir = os.path.join(args.models_dir, func)
        if func not in FUNCTION_ZOO:
            print(f'⚠ {func}: not in FUNCTION_ZOO, skipping')
            continue
        scaler_path = os.path.join(func_dir, 'scaler_X.pkl')
        if not os.path.exists(scaler_path):
            print(f'⚠ {func}: scaler_X.pkl missing, skipping')
            continue

        cfg        = FUNCTION_ZOO[func]
        bounds     = cfg['bounds']
        feat_names = cfg['names']
        nx         = len(bounds)
        scaler_X   = joblib.load(scaler_path)

        # fixed input sample, normalised exactly as during training
        rng   = np.random.default_rng(args.seed)
        X_raw = rng.uniform(low =[b[0] for b in bounds],
                            high=[b[1] for b in bounds],
                            size=(args.n_samples, nx))
        X_norm   = scaler_X.transform(X_raw)
        X_tensor = torch.tensor(X_norm, dtype=torch.float32, device=device)

        func_out = os.path.join(args.out, func)
        os.makedirs(func_out, exist_ok=True)

        ckpts = discover_checkpoints(func_dir)
        if args.grids is not None:
            ckpts = [c for c in ckpts if c[0] in args.grids]
        if args.ks is not None:
            ckpts = [c for c in ckpts if c[1] in args.ks]

        print(f'\n=== {func} === ({len(ckpts)} models)')
        for grid, k, stem in ckpts:
            tag = f'  g{grid:<3d} k{k}'
            if not os.path.exists(stem + '_config.yml'):
                print(f'{tag}  → config missing, skip')
                continue
            try:
                model = KAN.loadckpt(path=stem)
                if hasattr(model, 'to'):
                    model.to(device)
                with torch.no_grad():
                    model.forward(X_tensor)

                fig, infl = draw_spline_figure(model, feat_names, device)
                r2 = r2_lut.get((func, grid, k))
                r2_txt = f'  R²={r2:.3f}' if r2 is not None and not pd.isna(r2) else ''
                fig.suptitle(f'{func}  ·  grid={grid}  k={k}{r2_txt}',
                             fontsize=11, fontweight='bold')

                base = os.path.join(func_out, f'{func}_g{grid}_k{k}_spline_L0')
                for ext in args.formats:
                    fig.savefig(f'{base}.{ext}',
                                dpi=300 if ext == 'png' else None,
                                format=None if ext == 'png' else ext)
                plt.close(fig)

                for fi in range(nx):
                    summary_rows.append({
                        'func': func, 'grid': grid, 'k': k,
                        'feature_idx': fi, 'feature_name': feat_names[fi],
                        'test_r2': r2,
                        'n_inflection': len(infl[fi]) if infl[fi] else 0,
                        'inflection_points_norm': str(infl[fi] or []),
                    })
                n_done += 1
                print(f'{tag}  → saved  (infl/feat={[len(p or []) for p in infl]}){r2_txt}')
            except Exception as exc:
                n_fail += 1
                print(f'{tag}  → FAILED: {exc}')

    if summary_rows and not args.no_csv:
        out_csv = os.path.join(args.out, 'inflection_points.csv')
        pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
        print(f'\n📑 Inflection summary → {out_csv}')

    print(f'\nDone. {n_done} figures saved, {n_fail} failed. Output: {args.out}')


if __name__ == '__main__':
    main()
