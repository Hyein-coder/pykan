"""Aggregate the ranking-transition (small |phi'|) analysis for the meta_executor
toy cases into a single multi-panel figure.

Each panel shows, for one fitted KAN, the per-feature bottom-layer local
sensitivity s_i(x) = sum_j |phi'_{i,j}(x)| on the shared normalized input axis,
the small threshold tau = rel_thresh * max sensitivity, and the ranking
transitions (red = the dominant feature's |phi'| dropping below tau).

Reuses bspline_curvature.{feature_sensitivity, find_ranking_transitions}; needs
only the model parameters (no data / forward pass).

Run:  PYTHONUTF8=1 PYTHONPATH=D:/pykan python -m github.workflows.Hyein.aggregate_ranking_transitions
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from github.workflows.Hyein.toy_KAN_sweep import KANRegressor, FUNCTION_ZOO
from github.workflows.Hyein.bspline_curvature import find_ranking_transitions

# The 7 cases run through meta_executor.py (target_data).
NAMES = ['conditional', 'damping_sin', 'log2', 'exponential',
         'logarithm', 'rosenbrock', 'ishigami']
REL_THRESH = 0.1
RESULTS_ROOT = os.path.join('github', 'workflows', 'Hyein', 'analytical_results')
OUT_DIR = os.path.join('github', 'workflows', 'Hyein', 'figures_for_paper')


def _load_model(name):
    savepath = os.path.join(RESULTS_ROOT, name, 'kan_models')
    ckpt = os.path.join(savepath, f'{name}_best_kan_model')
    mw = KANRegressor(device='cpu')
    mw.load_model(ckpt)
    return mw.model


def _panel(ax, name):
    model = _load_model(name)
    feat_names = FUNCTION_ZOO[name]['names']
    ni = model.act_fun[0].coef.shape[0]
    transitions, info = find_ranking_transitions(model, rel_thresh=REL_THRESH)
    xg, S, tau = info['x_grid'], info['S'], info['tau']

    colors = [plt.get_cmap('tab10')(c % 10) for c in range(ni)]
    for i in range(ni):
        ax.plot(xg, S[i], color=colors[i], lw=1.2, label=str(feat_names[i]))
    ax.axhline(tau, color='black', ls='--', lw=0.9, alpha=0.7,
               label=rf"$\tau={REL_THRESH:g}\cdot$max")

    # All τ-crossings, every feature: solid = down (→negligible), dotted = up.
    first_d, first_u = True, True
    for t in transitions:
        c = colors[t['feat_idx']]
        if t['direction'] == 'down':
            ax.axvline(t['point'], color=c, ls='-', lw=1.2, alpha=0.85,
                       label='transition (down)' if first_d else '_')
            first_d = False
        else:
            ax.axvline(t['point'], color=c, ls=':', lw=1.0, alpha=0.7,
                       label='transition (up)' if first_u else '_')
            first_u = False

    ax.set_title(f"{name}  (n={ni}, {len(transitions)} tr.)", fontsize=10)
    ax.set_xlabel("normalized input")
    ax.set_ylabel(r"$\sum_j|\phi'_{ij}|$")
    ax.legend(loc='best', fontsize=6)
    return transitions


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ncols = 3
    nrows = (len(NAMES) + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.3 * nrows),
                            constrained_layout=True)
    axs = np.atleast_1d(axs).flatten()

    summary = {}
    for idx, name in enumerate(NAMES):
        try:
            summary[name] = _panel(axs[idx], name)
        except Exception as e:  # keep the grid intact if one model is missing
            axs[idx].text(0.5, 0.5, f"{name}\nfailed:\n{e}", ha='center',
                          va='center', fontsize=8, transform=axs[idx].transAxes)
            axs[idx].set_axis_off()
            summary[name] = None

    for k in range(len(NAMES), len(axs)):
        axs[k].set_visible(False)

    fig.suptitle(r"Ranking transition (small $|\phi'|$) — meta_executor toy cases",
                 fontsize=13, fontweight='bold')
    for ext in ['.png', '.svg', '.eps']:
        fig.savefig(os.path.join(OUT_DIR, f"ranking_transition_all{ext}"))
    plt.close(fig)

    print("Saved:", os.path.join(OUT_DIR, "ranking_transition_all.(png/svg/eps)"))
    for name, trans in summary.items():
        if trans is None:
            print(f"  {name}: FAILED")
            continue
        down = [round(t['point'], 3) for t in trans if t['direction'] == 'down']
        up = [round(t['point'], 3) for t in trans if t['direction'] == 'up']
        print(f"  {name}: {len(trans)} tr.  down={down}  up={up}")


if __name__ == '__main__':
    main()
