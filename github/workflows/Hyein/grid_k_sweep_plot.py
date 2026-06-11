"""
grid_k_sweep_plot.py
====================
Visualise transition-point robustness results from grid_k_sweep.py.

Input:   _workspace/grid_k_sweep_results.csv
Output:  _workspace/sweep_plots/
    tp_vs_grid.png/svg       — TP location vs grid, one line per k
    tp_count_heatmap.png/svg — n_tp found per (grid, k) cell
    r2_heatmap.png/svg       — test R² per (grid, k) cell
    tp_std_bar.png/svg       — std of TP across grids (robustness score)

Usage
-----
    python grid_k_sweep_plot.py
    python grid_k_sweep_plot.py --csv path/to/results.csv --out path/to/plots/
"""

import argparse
import ast
import os
import sys

import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, r'D:\pykan')

try:
    from github.workflows.Hyein.toy_KAN_sweep import FUNCTION_ZOO, KANRegressor
    import joblib
    import torch
    _HAS_ZOO = True
except Exception:
    _HAS_ZOO = False

# ── paths ─────────────────────────────────────────────────────────────────────
HERE         = os.path.dirname(__file__)
IN_CSV       = os.path.join(HERE, '_workspace', 'grid_k_sweep_results.csv')
INTERVAL_CSV = os.path.join(HERE, '_workspace', 'grid_k_sweep_results_interval.csv')
MODELS_DIR   = os.path.join(HERE, '_workspace', 'sweep_models')
OUT_DIR      = os.path.join(HERE, '_workspace', 'sweep_plots')

# in-process cache: func → DataFrame of interval attribution
_interval_cache: dict = {}

# ── plot style ────────────────────────────────────────────────────────────────
RC = {
    'figure.dpi': 150, 'figure.facecolor': 'white', 'figure.autolayout': True,
    'axes.facecolor': 'white', 'axes.edgecolor': '#444444', 'axes.linewidth': 0.8,
    'axes.labelsize': 10, 'axes.labelcolor': 'black',
    'axes.grid': True,
    'xtick.labelsize': 9, 'xtick.color': 'black',
    'ytick.labelsize': 9, 'ytick.color': 'black',
    'font.family': 'sans-serif', 'font.size': 9,
    'lines.linewidth': 1.4, 'lines.markersize': 5,
    'legend.fontsize': 7, 'legend.framealpha': 0.0,
    'savefig.dpi': 150, 'savefig.bbox': 'tight', 'savefig.facecolor': 'white',
}


# ── fixed per-feature colours (same as toy_KAN_analyze AGSM panels) ──────────
FEAT_COLORS = ['#1f77b4', '#d62728']   # blue = feat-0, red = feat-1

# ── per-feature colormaps for k-value lines in tp_by_grid_* plots ─────────────
# feat-0: dark-blue → light-blue  |  feat-1: dark-red → light-red
CMAP_FEAT0 = LinearSegmentedColormap.from_list('blue_gradient', ['#08519C', '#9ECAE1'])
CMAP_FEAT1 = LinearSegmentedColormap.from_list('red_gradient',  ['#A50F15', '#FCAE91'])
FEAT_CMAPS = [CMAP_FEAT0, CMAP_FEAT1]

# line styles for k values (4 max)
K_STYLES = {2: 'solid', 3: 'dashed', 4: 'dotted', 5: 'dashdot'}


# ── helpers ───────────────────────────────────────────────────────────────────
def save(fig, stem, out_dir):
    for ext in ['png', 'svg']:
        fig.savefig(os.path.join(out_dir, f'{stem}.{ext}'))
    plt.close(fig)
    print(f'  Saved {stem}.png/svg')


def feat_order(func_df):
    """Feature indices sorted by mean attribution (highest first)."""
    return (
        func_df.groupby(['feature_idx', 'feature_name'])['attribution']
        .mean()
        .sort_values(ascending=False)
        .index.tolist()          # list of (feature_idx, feature_name)
    )


# ── AGSM-style comparison plots ───────────────────────────────────────────────
def _parse_tps(val):
    """Parse tp_raw / tp_norm string → list of floats."""
    try:
        return [float(v) for v in ast.literal_eval(str(val))]
    except Exception:
        return []


def _feat_bounds(func, fidx):
    """Raw-space (lo, hi) for feature fidx, from FUNCTION_ZOO if available."""
    if _HAS_ZOO and func in FUNCTION_ZOO:
        b = FUNCTION_ZOO[func]['bounds']
        if fidx < len(b):
            return float(b[fidx][0]), float(b[fidx][1])
    return None, None


def _get_interval_data(df, func):
    """Return per-interval attribution DataFrame for func (cached).

    Loads from INTERVAL_CSV if already computed, otherwise runs forward passes
    on the saved model checkpoints and writes the result to INTERVAL_CSV.
    Returns None when models or ZOO data are unavailable.
    """
    global _interval_cache
    if func in _interval_cache:
        return _interval_cache[func]

    # ── load from on-disk cache (only if all slicing features are present) ────
    nx_expected = df[df['func'] == func]['feature_idx'].nunique()
    if os.path.exists(INTERVAL_CSV):
        cached = pd.read_csv(INTERVAL_CSV)
        func_cached = cached[cached['func'] == func]
        if (not func_cached.empty and
                func_cached['slicing_feat_idx'].nunique() >= nx_expected):
            result = func_cached.reset_index(drop=True)
            _interval_cache[func] = result
            return result

    if not _HAS_ZOO:
        return None

    # ── compute from saved checkpoints ───────────────────────────────────────
    fdf = df[df['func'] == func]
    feat_map = (fdf[['feature_idx', 'feature_name']]
                .drop_duplicates()
                .sort_values('feature_idx'))
    nx          = len(feat_map)
    feat_names  = feat_map['feature_name'].tolist()

    scaler_path = os.path.join(MODELS_DIR, func, 'scaler_X.pkl')
    if not os.path.exists(scaler_path):
        print(f'  [interval] scaler not found for {func}, skipping')
        return None

    scaler_X = joblib.load(scaler_path)

    cfg = FUNCTION_ZOO.get(func)
    if cfg is None:
        return None
    rng   = np.random.default_rng(42)
    X_raw = rng.uniform(low =[b[0] for b in cfg['bounds']],
                        high=[b[1] for b in cfg['bounds']],
                        size=(1000, nx))
    X_norm   = scaler_X.transform(X_raw)
    X_tensor = torch.tensor(X_norm, dtype=torch.float32)

    rows = []
    models_seen = fdf[['grid', 'k', 'ckpt_path']].drop_duplicates(['grid', 'k'])
    print(f'  [interval] computing for {func} '
          f'({len(models_seen)} models × {nx} slicing features) …')

    for _, mrow in models_seen.iterrows():
        grid  = int(mrow['grid'])
        k     = int(mrow['k'])
        ckpt  = str(mrow['ckpt_path'])
        if not ckpt or not os.path.exists(ckpt + '_config.yml'):
            continue
        try:
            from kan.custom_multkan_ddp import KAN as _KAN
            model = _KAN.loadckpt(path=ckpt)

            interval_edges = np.linspace(0.1, 0.9, grid + 1)
            for sf in range(nx):
                for ii in range(grid):
                    lo_n = float(interval_edges[ii])
                    hi_n = float(interval_edges[ii + 1])
                    mask = ((X_norm[:, sf] >= lo_n) & (X_norm[:, sf] <= hi_n))
                    if mask.sum() < 2:
                        attrs = {f'attr_{i}': float('nan') for i in range(nx)}
                    else:
                        Xi = X_tensor[mask]
                        with torch.no_grad():
                            model.forward(Xi)
                        sc  = model.feature_score.detach().cpu().numpy()
                        std = torch.std(Xi, dim=0).detach().cpu().numpy()
                        sn  = sc / (std + 1e-6)
                        attrs = {f'attr_{i}': round(float(sn[i]), 5)
                                 for i in range(nx)}
                    dummy = np.zeros((2, nx))
                    dummy[0, sf] = lo_n
                    dummy[1, sf] = hi_n
                    raw = scaler_X.inverse_transform(dummy)[:, sf]
                    rows.append({
                        'func': func, 'grid': grid, 'k': k,
                        'slicing_feat_idx':  sf,
                        'slicing_feat_name': feat_names[sf],
                        'interval_idx': ii,
                        'lo_norm': round(lo_n, 4), 'hi_norm': round(hi_n, 4),
                        'lo_raw':  round(float(raw[0]), 4),
                        'hi_raw':  round(float(raw[1]), 4),
                        **attrs,
                    })
        except Exception as exc:
            print(f'  [interval] failed g{grid} k{k}: {exc}')

    if not rows:
        return None

    new_df = pd.DataFrame(rows)

    # append to on-disk cache
    if os.path.exists(INTERVAL_CSV):
        combined = pd.concat([pd.read_csv(INTERVAL_CSV), new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(INTERVAL_CSV, index=False)
    print(f'  [interval] cached {len(new_df)} rows → {INTERVAL_CSV}')

    _interval_cache[func] = new_df
    return new_df


def plot_tp_comparison(df, funcs, out_dir, mode='by_grid'):
    """AGSM-modes-style TP comparison figure.

    mode='by_grid'  → rows = grid values, cols = features,
                       TP lines coloured by linestyle per k;
                       each panel also shows per-interval grouped attribution
                       bars (all features, sliced by that column's feature)
                       on a secondary y-axis, averaged over k.
    mode='by_k'     → rows = k values, cols = features,
                       lines coloured by grid (sequential alpha, 2 feature colours)
    """
    if mode == 'by_grid':
        row_vals  = sorted(df['grid'].unique())
        vary_vals = sorted(df['k'].unique())
        row_key   = 'grid'
        vary_key  = 'k'
        vary_lw   = {v: 1.6 for v in vary_vals}
    else:
        row_vals   = sorted(df['k'].unique())
        vary_vals  = sorted(df['grid'].unique())
        row_key    = 'k'
        vary_key   = 'grid'
        vary_alpha = {v: 0.3 + 0.7 * i / max(len(vary_vals) - 1, 1)
                      for i, v in enumerate(vary_vals)}
        vary_lw    = {v: 0.8 + 1.0 * i / max(len(vary_vals) - 1, 1)
                      for i, v in enumerate(vary_vals)}

    for func in funcs:
        fdf    = df[df['func'] == func]
        feats  = feat_order(fdf)          # [(fidx, fname), ...]
        n_rows = len(row_vals)
        n_cols = len(feats)

        # per-interval attribution: compute or load from cache
        if mode == 'by_grid':
            fdf_int = _get_interval_data(df, func)
        else:
            fdf_int = None

        # per-k colors per feature column (sampled from feature colormap)
        if mode == 'by_grid':
            n_vary = max(len(vary_vals) - 1, 1)
            k_feat_colors = {
                col_i: {
                    v: FEAT_CMAPS[col_i % len(FEAT_CMAPS)](i / n_vary)
                    for i, v in enumerate(vary_vals)
                }
                for col_i in range(len(feats))
            }
        attr_cols = (sorted([c for c in fdf_int.columns if c.startswith('attr_')],
                             key=lambda c: int(c.split('_')[1]))
                     if fdf_int is not None else [])

        # shared y-limit per column: max attribution across all grid rows
        # so bars are directly comparable up and down each column
        col_ymax = {}
        if fdf_int is not None and attr_cols:
            for col_i, (fidx, _) in enumerate(feats):
                vals = fdf_int[fdf_int['slicing_feat_idx'] == fidx][attr_cols]
                col_ymax[col_i] = float(vals.max().max()) * 1.1

        with plt.rc_context(RC):
            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(4 * n_cols, 2.8 * n_rows),
                squeeze=False,
            )

            for row_i, row_val in enumerate(row_vals):
                for col_i, (fidx, fname) in enumerate(feats):
                    ax       = axes[row_i, col_i]
                    feat_col = FEAT_COLORS[col_i % len(FEAT_COLORS)]
                    lo, hi   = _feat_bounds(func, fidx)

                    # fall back to data range if ZOO not available
                    sub = fdf[fdf['feature_idx'] == fidx]
                    if lo is None:
                        valid = sub['first_tp_raw'].dropna()
                        lo    = valid.min() - 0.05 * (valid.max() - valid.min() + 1e-6)
                        hi    = valid.max() + 0.05 * (valid.max() - valid.min() + 1e-6)

                    ax.set_xlim(lo, hi)
                    ax.set_ylim(0, 1)
                    ax.set_yticks([])
                    ax.set_xlabel(fname, fontsize=9)
                    ax.set_title(f'{row_key}={row_val}', fontsize=9)

                    # ── per-interval attribution bars (sliced by this column's feature) ──
                    if fdf_int is not None and attr_cols:
                        panel_int = fdf_int[
                            (fdf_int['grid']             == row_val) &
                            (fdf_int['slicing_feat_idx'] == fidx)
                        ]
                        if not panel_int.empty:
                            # average across k values
                            agg = (panel_int.groupby('interval_idx')
                                   .agg(lo_raw=('lo_raw', 'first'),
                                        hi_raw=('hi_raw', 'first'),
                                        **{c: (c, 'mean') for c in attr_cols})
                                   .reset_index())

                            ax_b = ax.twinx()
                            ax_b.set_ylim(0, col_ymax.get(col_i, 1.0))
                            ax_b.tick_params(axis='y', labelsize=6, pad=1)
                            ax_b.set_ylabel('Attribution', fontsize=6, labelpad=2)

                            n_feats_bar = len(attr_cols)
                            for _, irow in agg.iterrows():
                                int_w   = irow['hi_raw'] - irow['lo_raw']
                                mid     = (irow['lo_raw'] + irow['hi_raw']) / 2.0
                                sub_w   = int_w * 0.8 / n_feats_bar
                                for bi, ac in enumerate(attr_cols):
                                    fi      = int(ac.split('_')[1])
                                    bcolor  = FEAT_COLORS[fi % len(FEAT_COLORS)]
                                    offset  = (bi - (n_feats_bar - 1) / 2.0) * sub_w
                                    val     = irow[ac]
                                    if not np.isnan(val):
                                        ax_b.bar(mid + offset, val, sub_w * 0.9,
                                                 color=bcolor, alpha=0.30,
                                                 edgecolor='none', zorder=1)

                    rdf = fdf[
                        (fdf[row_key]       == row_val) &
                        (fdf['feature_idx'] == fidx)
                    ]

                    for _, row_data in rdf.iterrows():
                        vary_val = row_data[vary_key]
                        tps      = _parse_tps(row_data['tp_raw'])
                        if not tps:
                            continue

                        if mode == 'by_grid':
                            color = k_feat_colors[col_i][vary_val]
                            lw    = vary_lw[vary_val]
                            ls    = K_STYLES.get(int(vary_val), 'solid')
                        else:
                            color = feat_col
                            lw    = vary_lw[vary_val]
                            ls    = 'solid'
                            ax.set_alpha(1.0)

                        alpha = vary_alpha[vary_val] if mode == 'by_k' else 0.85

                        first = True
                        for tp in tps:
                            ax.axvline(
                                x=tp,
                                color=color,
                                linestyle=ls,
                                linewidth=lw,
                                alpha=alpha,
                                zorder=3,
                                label=f'{vary_key}={int(vary_val)}' if first else '_',
                            )
                            first = False

                    # ── legend (first column only to avoid clutter) ──
                    if col_i == 0:
                        if mode == 'by_grid':
                            handles = [
                                mlines.Line2D([], [], color=k_feat_colors[col_i][v],
                                              linewidth=1.4,
                                              linestyle=K_STYLES.get(int(v), 'solid'),
                                              label=f'k={v}')
                                for v in vary_vals
                            ]
                        else:
                            handles = [
                                mlines.Line2D([], [], color=feat_col,
                                              linewidth=vary_lw[v],
                                              alpha=vary_alpha[v],
                                              label=f'grid={v}')
                                for v in vary_vals
                            ]
                        ax.legend(handles=handles, loc='best', fontsize=7)

            vary_desc = (f'colour+style by {vary_key}'
                     if mode == 'by_grid' else f'coloured by {vary_key}')
            fig.suptitle(
                f'{func}  —  transition points organised by {row_key}  ({vary_desc})',
                fontsize=10, fontweight='bold',
            )
            save(fig, f'tp_{mode}_{func}', out_dir)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default=IN_CSV)
    parser.add_argument('--out', default=OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = df[df['n_tp'] >= 0].copy()   # drop failed rows

    funcs   = list(df['func'].unique())
    k_vals  = sorted(df['k'].unique())
    grids   = sorted(df['grid'].unique())
    n_funcs = len(funcs)

    # colour per k value
    k_colors = {k: cm.tab10(i / max(len(k_vals) - 1, 1))
                for i, k in enumerate(k_vals)}

    max_feats = max(len(feat_order(df[df['func'] == f])) for f in funcs)

    print(f'Functions : {funcs}')
    print(f'Grids     : {grids}')
    print(f'k values  : {k_vals}\n')

    # ── Figure 1: TP location vs grid ────────────────────────────────────────
    # rows = funcs, cols = features (sorted by attribution)
    with plt.rc_context(RC):
        fig, axes = plt.subplots(
            n_funcs, max_feats,
            figsize=(4 * max_feats, 3 * n_funcs),
            squeeze=False,
        )

        for row, func in enumerate(funcs):
            fdf   = df[df['func'] == func]
            feats = feat_order(fdf)

            for col, (fidx, fname) in enumerate(feats):
                ax    = axes[row, col]
                fdata = fdf[fdf['feature_idx'] == fidx]

                for k in k_vals:
                    kdf = fdata[fdata['k'] == k].sort_values('grid')
                    ax.plot(
                        kdf['grid'], kdf['first_tp_raw'],
                        marker='o', color=k_colors[k], label=f'k={k}',
                    )

                ax.set_xlabel('grid')
                ax.set_ylabel('first TP (raw space)')
                ax.set_title(f'{func}  ·  {fname}', fontsize=9)
                ax.set_xticks(grids)
                ax.legend(loc='best')

            for col in range(len(feats), max_feats):
                axes[row, col].set_visible(False)

        fig.suptitle('Transition point location vs grid', fontsize=11,
                     fontweight='bold')
        save(fig, 'tp_vs_grid', args.out)

    # ── Figure 2: TP count heatmap (grid × k) ────────────────────────────────
    with plt.rc_context(RC):
        fig, axes = plt.subplots(
            n_funcs, max_feats,
            figsize=(3.5 * max_feats, 3 * n_funcs),
            squeeze=False,
        )

        for row, func in enumerate(funcs):
            fdf   = df[df['func'] == func]
            feats = feat_order(fdf)

            for col, (fidx, fname) in enumerate(feats):
                ax    = axes[row, col]
                fdata = fdf[fdf['feature_idx'] == fidx]

                pivot = fdata.pivot_table(
                    index='k', columns='grid', values='n_tp', aggfunc='mean'
                )
                im = ax.imshow(
                    pivot.values, aspect='auto', cmap='YlOrRd',
                    vmin=0,
                )
                ax.set_xticks(range(len(pivot.columns)))
                ax.set_xticklabels(pivot.columns.astype(int))
                ax.set_yticks(range(len(pivot.index)))
                ax.set_yticklabels(pivot.index.astype(int))
                ax.set_xlabel('grid')
                ax.set_ylabel('k')
                ax.set_title(f'{func}  ·  {fname}', fontsize=9)
                fig.colorbar(im, ax=ax, label='n_tp')

                # annotate cells
                for r in range(pivot.shape[0]):
                    for c in range(pivot.shape[1]):
                        val = pivot.values[r, c]
                        if not np.isnan(val):
                            ax.text(c, r, f'{val:.0f}', ha='center',
                                    va='center', fontsize=8, color='black')

            for col in range(len(feats), max_feats):
                axes[row, col].set_visible(False)

        fig.suptitle('Number of transition points found (grid × k)',
                     fontsize=11, fontweight='bold')
        save(fig, 'tp_count_heatmap', args.out)

    # ── Figure 3: test R² heatmap (grid × k, one panel per func) ────────────
    # R² is per model (same for all features), so average across features
    with plt.rc_context(RC):
        fig, axes = plt.subplots(
            1, n_funcs,
            figsize=(3.5 * n_funcs, 3.5),
            squeeze=False,
        )

        for col, func in enumerate(funcs):
            ax  = axes[0, col]
            fdf = df[df['func'] == func]

            pivot = fdf.pivot_table(
                index='k', columns='grid', values='test_r2', aggfunc='mean'
            )
            im = ax.imshow(
                pivot.values, aspect='auto', cmap='RdYlGn',
                vmin=0, vmax=1,
            )
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns.astype(int))
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index.astype(int))
            ax.set_xlabel('grid')
            ax.set_ylabel('k')
            ax.set_title(func, fontsize=9)
            fig.colorbar(im, ax=ax, label='test R²')

            for r in range(pivot.shape[0]):
                for c in range(pivot.shape[1]):
                    val = pivot.values[r, c]
                    if not np.isnan(val):
                        ax.text(c, r, f'{val:.2f}', ha='center',
                                va='center', fontsize=7, color='black')

        fig.suptitle('Test R² across grid × k', fontsize=11, fontweight='bold')
        save(fig, 'r2_heatmap', args.out)

    # ── Figure 4: TP stability — std of first_tp_raw across grids ────────────
    # For each (func, feature, k): std of first_tp_raw over all grid values.
    # Lower std = more robust.
    with plt.rc_context(RC):
        fig, axes = plt.subplots(
            n_funcs, max_feats,
            figsize=(4 * max_feats, 3 * n_funcs),
            squeeze=False,
        )

        bar_width = 0.8 / max(len(k_vals), 1)
        x_base    = np.arange(len(k_vals))

        for row, func in enumerate(funcs):
            fdf   = df[df['func'] == func]
            feats = feat_order(fdf)

            for col, (fidx, fname) in enumerate(feats):
                ax    = axes[row, col]
                fdata = fdf[fdf['feature_idx'] == fidx]

                stds = []
                for k in k_vals:
                    vals = fdata[fdata['k'] == k]['first_tp_raw'].dropna()
                    stds.append(vals.std() if len(vals) > 1 else 0.0)

                bars = ax.bar(
                    x_base, stds,
                    color=[k_colors[k] for k in k_vals],
                    width=bar_width * len(k_vals) * 0.8,
                    edgecolor='black', linewidth=0.5,
                )
                ax.bar_label(bars, fmt='%.3f', fontsize=7, padding=2)
                ax.set_xticks(x_base)
                ax.set_xticklabels([f'k={k}' for k in k_vals])
                ax.set_ylabel('std of first TP (raw)')
                ax.set_title(f'{func}  ·  {fname}', fontsize=9)

            for col in range(len(feats), max_feats):
                axes[row, col].set_visible(False)

        fig.suptitle('Transition point stability across grids\n'
                     '(lower = more robust)', fontsize=11, fontweight='bold')
        save(fig, 'tp_std_bar', args.out)

    # ── Figures 5 & 6: AGSM-style TP comparison (one file per func) ─────────
    print('\nDrawing TP comparison plots …')
    plot_tp_comparison(df, funcs, args.out, mode='by_grid')
    plot_tp_comparison(df, funcs, args.out, mode='by_k')

    print(f'\nAll plots saved to: {args.out}')


if __name__ == '__main__':
    main()
