"""MLP (sklearn ``MLPRegressor``) gradient-based local-sensitivity analysis driver.

The MLP analog of ``toy_KAN_analyze.py``: load a tuned MLP + its scalers, compute
the gradient-based local-sensitivity trajectory ``s_i(x) = |df/dx_i|`` (finite
differences) in the normalized [0.1, 0.9] space, detect ranking transitions the
same τ-crossing way KAN does, and run the gradient-AGSM baseline. Outputs go to
``analytical_results/<func>/nn_analysis/``.

Run:
    PYTHONPATH=. PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
        python -m github.workflows.Hyein.toy_NN_analyze conditional
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from github.workflows.Hyein.toy_KAN_sweep import FUNCTION_ZOO
from github.workflows.Hyein.kan_analysis_core import SA_RC, denorm, _step_over_edges
from github.workflows.Hyein.sectional_gsa import (
    compute_gradient_agsm, find_agsm_transition_points,
)
from github.workflows.Hyein.nn_sensitivity import (
    mlp_predict_norm, mlp_batch_func,
    mlp_feature_sensitivity, mlp_ranking_transitions,
)

HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------
# FIGURE 1: MLP local-sensitivity trajectory + ranking transitions
# ----------------------------------------------------------------------------
def plot_nn_ranking_transition(x_grid, S_mid, S_mean, S_std, tau, transitions,
                               feat_names, scaler_X, nx, savepath, data_name):
    """Per-feature MLP s_i(x): midpoint line + MC mean±std band + τ + transitions.

    Mirrors ``kan_analysis_core.plot_ranking_transition_figure`` style (SA_RC, no
    title, no grid, transparent legend, RdYlBu per-feature colors) but on a raw
    x-axis (each feature denormalized through its own column) and with both
    marginalization curves overlaid.
    """
    feat_colors = [plt.get_cmap('RdYlBu')(c) for c in np.linspace(0.1, 0.9, nx)]
    with plt.rc_context(SA_RC):
        fig, ax = plt.subplots(figsize=(7, 4))
        for i in range(nx):
            color = feat_colors[i]
            x_raw = denorm(x_grid, i, scaler_X, nx)  # per-feature raw x-axis
            # midpoint: clean 1-D curve (others fixed at 0.5)
            ax.plot(x_raw, S_mid[i], color=color, lw=1.6, ls='-',
                    label=rf"$|\partial f/\partial x|$ {feat_names[i]} (midpoint)")
            # MC mean + ±std band (others marginalized ~ U(0.1,0.9))
            ax.plot(x_raw, S_mean[i], color=color, lw=1.0, ls='--',
                    label=rf"{feat_names[i]} (MC mean)")
            ax.fill_between(x_raw, S_mean[i] - S_std[i], S_mean[i] + S_std[i],
                            color=color, alpha=0.18, lw=0,
                            label=rf"{feat_names[i]} (MC $\pm\sigma$)")
        ax.axhline(tau, color='black', ls='--', lw=1.0, alpha=0.7, label=r"$\tau$")

        # τ-crossing vlines: down solid / up dotted, colored by feature.
        first_d, first_u = True, True
        for t in transitions:
            i = t['feat_idx']
            x_raw_pt = denorm([t['point']], i, scaler_X, nx)[0]
            if t['direction'] == 'down':
                ax.axvline(x_raw_pt, color=feat_colors[i], ls='-', alpha=0.85, lw=1.3,
                           label='transition (down)' if first_d else '_')
                first_d = False
            else:
                ax.axvline(x_raw_pt, color=feat_colors[i], ls=':', alpha=0.7, lw=1.1,
                           label='transition (up)' if first_u else '_')
                first_u = False

        ax.set_xlabel("Feature value (raw)")
        ax.set_ylabel(r"local sensitivity  $|\partial f/\partial x_i|$")
        ax.legend(loc='best', fontsize=7)
        for ext in ['.png', '.svg', '.eps']:
            fig.savefig(os.path.join(savepath, f"{data_name}_nn_ranking_transition{ext}"))
        plt.close(fig)
    print(f"📊 MLP ranking-transition figure saved: {data_name}_nn_ranking_transition.(png/svg/eps)")


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MLP gradient-based sensitivity analysis.")
    parser.add_argument("func_name", type=str, nargs='?', default="conditional",
                        choices=FUNCTION_ZOO.keys())
    args = parser.parse_args()
    data_name = args.func_name

    # --- Load MLP + scalers -------------------------------------------------
    nn_dir = os.path.join(HERE, "analytical_results", data_name, "nn_models")
    mlp = joblib.load(os.path.join(nn_dir, f"{data_name}_best_mlp_model.pkl"))
    scaler_X = joblib.load(os.path.join(nn_dir, f"{data_name}_mlp_scaler_X.pkl"))
    scaler_y = joblib.load(os.path.join(nn_dir, f"{data_name}_mlp_scaler_y.pkl"))

    cfg = FUNCTION_ZOO[data_name]
    func, bounds, names = cfg["func"], cfg["bounds"], cfg["names"]
    nx = len(bounds)

    # --- Regenerate data exactly like the KAN side (same seed/conventions) --
    np.random.seed(42)
    X_raw = np.random.uniform(low=[b[0] for b in bounds],
                              high=[b[1] for b in bounds], size=(1000, nx))
    y_raw = np.apply_along_axis(func, 1, X_raw).reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42)
    X_train_norm = scaler_X.transform(X_train)

    predict_norm = mlp_predict_norm(mlp)

    # --- Sensitivity sweep (normalized [0.1, 0.9]) --------------------------
    x_grid = np.linspace(0.1, 0.9, 400)

    # Ranking transitions via MC marginalization (the noisy/entangled view).
    transition_points_per_input, transitions, info = mlp_ranking_transitions(
        predict_norm, x_grid, nx, rel_thresh=0.2, mode='mc')
    tau = info['tau']
    S_mean = info['S']            # MC mean per feature
    S_std = info['S_std']         # MC std per feature

    # Midpoint sensitivity per feature (clean overlay curve).
    S_mid = {i: mlp_feature_sensitivity(predict_norm, x_grid, i, nx, mode='midpoint')
             for i in range(nx)}

    # --- Output dir ---------------------------------------------------------
    savepath = os.path.join(HERE, "analytical_results", data_name, "nn_analysis")
    os.makedirs(savepath, exist_ok=True)

    # --- FIGURE 1 -----------------------------------------------------------
    try:
        plot_nn_ranking_transition(
            x_grid, S_mid, S_mean, S_std, tau, transitions,
            names, scaler_X, nx, savepath, data_name)
    except Exception as e:
        import traceback
        print(f"⚠️ FIGURE 1 (ranking transition) failed: {e}")
        traceback.print_exc()

    # --- Ranking-transition CSV ---------------------------------------------
    try:
        rows = []
        for i in range(nx):
            pts_norm = transition_points_per_input[i]
            pts_raw = list(denorm(pts_norm, i, scaler_X, nx)) if pts_norm else []
            n_down = sum(1 for t in transitions if t['feat_idx'] == i and t['direction'] == 'down')
            n_up = sum(1 for t in transitions if t['feat_idx'] == i and t['direction'] == 'up')
            rows.append({
                'feat_idx': i, 'feat_name': names[i],
                'points_norm': [round(p, 4) for p in pts_norm],
                'points_raw': [round(float(p), 4) for p in pts_raw],
                'n_down': n_down, 'n_up': n_up, 'tau': tau,
            })
        pd.DataFrame(rows).to_csv(
            os.path.join(savepath, f"{data_name}_nn_ranking_transition.csv"), index=False)
        print(f"📄 Saved: {data_name}_nn_ranking_transition.csv")
    except Exception as e:
        import traceback
        print(f"⚠️ Ranking-transition CSV failed: {e}")
        traceback.print_exc()

    # --- Global ranking of features (mean midpoint sensitivity over the grid)
    global_s = np.array([np.nanmean(S_mid[i]) for i in range(nx)])
    top2 = np.argsort(global_s)[::-1][:2].tolist()  # nx=2 -> both, dominant first

    # --- gradient-AGSM (raw space) ------------------------------------------
    try:
        batch_func = mlp_batch_func(mlp, scaler_X, scaler_y)
        i_idx = top2[0]
        j_idx = top2[1] if len(top2) > 1 else top2[0]

        sc, sh, sa, r = compute_gradient_agsm(
            func_batch=batch_func, bounds=bounds, feat_names=names,
            top2_idx=top2, n_sections=10, n_samples_per_section=512,
            seed=42, section_mode='equal',
        )
        tps = find_agsm_transition_points(
            sc[i_idx], sa[i_idx], sc[j_idx], sa[j_idx],
            names[i_idx], names[j_idx]) if i_idx != j_idx else []

        # CSV
        rows = []
        for feat_idx in top2:
            for k, (center, s_hat, s_a_v, r_v) in enumerate(zip(
                    sc[feat_idx], sh[feat_idx], sa[feat_idx], r[feat_idx])):
                rows.append({'Feature': names[feat_idx], 'Feature_idx': feat_idx,
                             'Section_k': k, 'Section_center': center,
                             'S_hat': s_hat, 'S_a': s_a_v, 'R': r_v})
        pd.DataFrame(rows).to_csv(
            os.path.join(savepath, f"{data_name}_nn_agsm_sectional.csv"), index=False)

        # Step figure: AGSM S_a per feature + AGSM transition vlines.
        with plt.rc_context(SA_RC):
            fig, ax = plt.subplots(figsize=(7, 4))
            colors = ['#1f77b4', '#d62728']
            for n, feat_idx in enumerate(top2):
                centers = np.asarray(sc[feat_idx], dtype=float)
                lo_f, hi_f = bounds[feat_idx]
                edges = np.linspace(lo_f, hi_f, len(centers) + 1)  # 'equal' mode edges
                _step_over_edges(ax, edges, sa[feat_idx],
                                 color=colors[n % len(colors)], label=names[feat_idx])
            first_tp = True
            for tp in tps:
                ax.axvline(tp['point'], color='orange', ls=':', alpha=0.8, lw=1.2,
                           label='AGSM transition' if first_tp else '_')
                first_tp = False
            ax.set_xlabel("Feature value (raw)")
            ax.set_ylabel(r"$S^a_{l,[k]}$ (normalized)")
            ax.legend(loc='best', fontsize=7)
            for ext in ['.png', '.svg', '.eps']:
                fig.savefig(os.path.join(savepath, f"{data_name}_nn_agsm{ext}"))
            plt.close(fig)

        print(f"📐 gradient-AGSM saved: {data_name}_nn_agsm.(png/svg/eps) + _sectional.csv")
        print(f"📐 AGSM transitions (raw): {[round(t['point'], 3) for t in tps]}")
    except Exception as e:
        import traceback
        print(f"⚠️ gradient-AGSM failed: {e}")
        traceback.print_exc()

    # --- Report -------------------------------------------------------------
    print("\n================ SUMMARY ================")
    print(f"func={data_name}  nx={nx}  top2(by mean midpoint s)={top2}  tau={tau:.4g}")
    for i in range(nx):
        pts_norm = transition_points_per_input[i]
        pts_raw = list(denorm(pts_norm, i, scaler_X, nx)) if pts_norm else []
        print(f"  feat {i} ({names[i]}): transitions norm={[round(p,4) for p in pts_norm]} "
              f"raw={[round(float(p),4) for p in pts_raw]}")
    # MC band width (mean of the std over the grid, per feature)
    for i in range(nx):
        print(f"  feat {i} MC std magnitude (mean over grid) = {np.nanmean(S_std[i]):.5g}  "
              f"(midpoint mean s = {np.nanmean(S_mid[i]):.5g})")


if __name__ == "__main__":
    main()
