import argparse
import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import yaml  # <--- [NEW] Import YAML

# ==========================================
# [FIX] Register Python Tuple for YAML Loading
# ==========================================
# This fixes the "could not determine a constructor for tag:yaml.org,2002:python/tuple" error
def tuple_constructor(loader, node):
    return tuple(loader.construct_sequence(node))

yaml.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor, Loader=yaml.SafeLoader)
# Depending on PyYAML version/method used by KAN, we might need to register it for the default Loader too
try:
    yaml.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor, Loader=yaml.Loader)
except AttributeError:
    pass # yaml.Loader might not exist in some setups, safe to ignore if SafeLoader is used

# ==========================================
# Import your wrapper and function ZOO
from SALib.sample import sobol as saltelli
from github.workflows.Hyein.toy_KAN_sweep import KANRegressor, FUNCTION_ZOO
from kan.experiments.analysis import find_indices_sign_revert
from github.workflows.Hyein.bspline_curvature import (
    find_inflection_points, edge_curves, symbolic_edge_info,
    feature_sensitivity, find_ranking_transitions, data_range_knots,
)
# Reuse the spline/symbolic reading builders from the robustness driver so the
# three model-reading modes here match that pipeline exactly (no duplication).
from github.workflows.Hyein.robustness_symbolify import (
    _build_spline_reading, _build_symbolic_reading,
)


SA_RC = {
    'figure.dpi': 150,
    'figure.facecolor': 'white',
    'figure.autolayout': True,
    'axes.facecolor': 'white',
    'axes.edgecolor': '#444444',
    'axes.linewidth': 0.8,
    'axes.spines.top': True,
    'axes.spines.right': True,
    'axes.labelsize': 12,
    'axes.labelcolor': 'black',
    'axes.titlelocation': 'center',
    'axes.grid': False,
    'xtick.labelsize': 12,
    'xtick.color': 'black',
    'xtick.direction': 'out',
    'ytick.labelsize': 10,
    'ytick.color': 'black',
    'ytick.direction': 'out',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue LT Pro', 'Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'font.size': 10,
    'font.weight': '300',
    'axes.labelweight': '500',
    'text.color': 'black',
    'patch.edgecolor': 'black',
    'patch.linewidth': 0.7,
    'patch.force_edgecolor': True,
    'legend.fontsize': 8,
    'legend.title_fontsize': 9,
    'legend.framealpha': 0.0,
    'legend.edgecolor': '#444444',
    'lines.linewidth': 0.7,
    'lines.markersize': 3,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
}


def _step_over_edges(ax, edges, vals, **kwargs):
    """Plot a piecewise-constant step whose transitions fall exactly on ``edges``
    (the section boundary knots), not at midpoints between section centers.

    ``edges`` has one more element than ``vals``. NaN sections render as a gap at
    that section only. Use this for every sectional step plot so the steps line
    up with the section boundaries / grid knots.
    """
    vals = np.asarray(vals, dtype=float)
    edges = np.asarray(edges, dtype=float)
    ax.step(edges, np.append(vals, vals[-1]), where='post', **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Tune KAN for Analytical Functions.")
    parser.add_argument("func_name", type=str, nargs='?', default="rosenbrock",
                        choices=FUNCTION_ZOO.keys(),
                        help="Choose a function from the ZOO.")
    parser.add_argument("--model-mode", type=str, default="as-saved",
                        choices=["as-saved", "spline", "symbolic"],
                        help="Which reading of the saved KAN to analyze: "
                             "'as-saved' (on-disk mix of spline/symbolic edges), "
                             "'spline' (force every symbolified edge back onto its "
                             "spline branch; symbolic branch off), or 'symbolic' "
                             "(use the symbolic branch; run auto_symbolic if the "
                             "model was saved as pure spline). Non-default modes "
                             "write into a kan_models/<mode> subfolder.")
    parser.add_argument("--refit", action="store_true",
                        help="With --model-mode symbolic on a pure-spline model, "
                             "run a short LBFGS refit after auto_symbolic "
                             "(mirrors the training pipeline). Ignored for "
                             "already-symbolified models. Default off.")

    args = parser.parse_args()
    data_name = args.func_name
    model_mode = args.model_mode
    refit_symbolic = args.refit
    plt.rcParams.update(SA_RC)
    # ==========================================
    # 1. Setup Paths & Load Model/Scalers
    # ==========================================
    root_dir = os.path.join(os.getcwd(), 'github', 'workflows', 'Hyein', 'analytical_results', data_name)
    savepath = os.path.join(root_dir, "kan_models")

    ckpt_path = os.path.join(savepath, f'{data_name}_best_kan_model')
    scaler_x_path = os.path.join(savepath, f'{data_name}_scaler_X.pkl')
    scaler_y_path = os.path.join(savepath, f'{data_name}_scaler_y.pkl')

    print(f"📂 Loading results from: {savepath}")

    # A. Load Scalers
    if not os.path.exists(scaler_x_path) or not os.path.exists(scaler_y_path):
        print("❌ Error: Scalers not found.")
        return

    scaler_X = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)

    # B. Initialize Wrapper & Load Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_wrapper = KANRegressor(device=device)

    try:
        model_wrapper.load_model(ckpt_path)
        model = model_wrapper.model  # Access the actual MultKAN object
        print("✅ KAN Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # ==========================================
    # 2. Regenerate Data
    # ==========================================
    print("\n🎲 Regenerating Train data for analysis...")
    config = FUNCTION_ZOO[data_name]
    target_func = config["func"]
    bounds = config["bounds"]
    feat_names = config["names"]
    nx = len(bounds)

    # CONVENTION: knots, inflection points, and ranking-transition points are kept
    # in NORMALIZED space throughout the script (the space the spline grid lives
    # in). They are denormalized to RAW input values only for PLOTTING, via this
    # single helper. (The one exception is the §3.6 ranking-transition figure,
    # which overlays all features on a shared normalized axis on purpose.)
    def denorm(vals, feat_idx):
        """Normalized value(s) for one feature → raw input space (for plotting)."""
        arr = np.atleast_1d(np.asarray(vals, dtype=float))
        if arr.size == 0:
            return arr
        dummy = np.zeros((arr.size, nx))
        dummy[:, feat_idx] = arr
        return scaler_X.inverse_transform(dummy)[:, feat_idx]

    X_raw = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(1000, nx))
    y_raw = np.apply_along_axis(target_func, 1, X_raw).reshape(-1, 1)
    # noise = np.random.normal(0, np.std(y_raw) * 0.05, size=y_raw.shape)
    # y_raw = y_raw + noise

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

    # Normalize Inputs (Critical for range analysis 0.1 ~ 0.9)
    X_train_norm = scaler_X.transform(X_train)

    # Create dataset dict (needed for forward pass logic sometimes)
    dataset = {
        'train_input': torch.tensor(X_train_norm, dtype=torch.float32, device=device),
        'train_label': torch.tensor(y_train, dtype=torch.float32, device=device).reshape(-1, 1)
        # Label scaling optional here
    }

    # ==========================================
    # 2.1 Select model reading mode (as-saved / spline / symbolic)
    # ==========================================
    # IMPORTANT: do this BEFORE any forward pass on the loaded `model`. The spline/
    # symbolic readings deepcopy it, and a forward pass caches non-leaf tensors that
    # break copy.deepcopy. Each reading runs its own forward afterwards.
    sym_info_saved = symbolic_edge_info(model, 0)
    if model_mode == 'as-saved':
        print(f"🧩 Model reading: as-saved (symbolic edges on disk: "
              f"{sorted(sym_info_saved.items()) if sym_info_saved else 'none'}).")
    elif model_mode == 'spline':
        model = _build_spline_reading(model, sym_info_saved)
        model_wrapper.model = model
        print(f"🧩 Model reading: strictly SPLINE — symbolic branch off; "
              f"{len(sym_info_saved)} symbolified edge(s) reverted to spline.")
    elif model_mode == 'symbolic':
        steps = getattr(model_wrapper, 'steps', 20)
        lr = getattr(model_wrapper, 'lr', 0.1)
        stop_grid = getattr(model_wrapper, 'stop_grid_update_step', 20)
        # Fit-ready dataset (normalized labels, test=train) only needed for the
        # optional LBFGS refit when introducing symbolify on a pure-spline model.
        fit_dataset = None
        if refit_symbolic and not sym_info_saved:
            y_train_norm_t = torch.tensor(scaler_y.transform(y_train),
                                          dtype=torch.float32, device=device).reshape(-1, 1)
            fit_dataset = {
                'train_input': dataset['train_input'], 'train_label': y_train_norm_t,
                'test_input': dataset['train_input'], 'test_label': y_train_norm_t,
            }
        model, n_edges, refit_done = _build_symbolic_reading(
            model, sym_info_saved, dataset['train_input'], refit_symbolic,
            fit_dataset, steps, lr, stop_grid)
        model_wrapper.model = model
        kind = 'as-saved symbolic' if sym_info_saved else 'introduced via auto_symbolic'
        extra = '' if sym_info_saved else f"; LBFGS refit={'done' if refit_done else 'off'}"
        print(f"🧩 Model reading: strictly SYMBOLIC — {kind}; "
              f"{n_edges} symbolified edge(s){extra}.")

    # Non-default modes write into a kan_models/<mode> subfolder so the canonical
    # as-saved outputs are never clobbered. (Gallery copies are tagged separately.)
    if model_mode != 'as-saved':
        savepath = os.path.join(savepath, model_mode)
        os.makedirs(savepath, exist_ok=True)
        print(f"📁 Outputs for this run → {savepath}")

    # ==========================================
    # 2.5 [NEW] Plot Input vs Output (Ground Truth vs Prediction)
    # ==========================================

    pred_y_norm = model(dataset['train_input']).detach().cpu().numpy()
    try:
        pred_y = scaler_y.inverse_transform(pred_y_norm)
    except ValueError:
        # Fallback if dimensions mismatch or scaler wasn't fitted on 2D
        pred_y = pred_y_norm

    n_features = X_train.shape[1]
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols

    fig_io, axs_io = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), constrained_layout=True)
    axs_io = axs_io.flatten()

    for i in range(n_features):
        ax = axs_io[i]

        # Plot Ground Truth (Gray)
        # X_train is the raw input (before normalization), y_train is raw output
        ax.scatter(X_train[:, i], y_train, alpha=0.5, c='gray', s=15, label='Ground Truth')

        # Plot Prediction (Red)
        ax.scatter(X_train[:, i], pred_y, alpha=0.5, c='red', s=15, label='Prediction')

        feature_label = feat_names[i] if feat_names and i < len(feat_names) else f"Feature {i}"
        ax.set_xlabel(feature_label)
        ax.set_ylabel("Output y")
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_features, len(axs_io)):
        axs_io[i].axis('off')

    # Save & Show
    plot_path_io = os.path.join(savepath, f"{data_name}_input_vs_output.png")
    plt.savefig(plot_path_io, dpi=300)
    # plt.show()

    # Run forward pass once to populate internals (splines, activations)
    model.forward(dataset['train_input'])
    scores_tot = model.feature_score.detach().cpu().numpy()  # Global scores
    #
    # fig_tot, ax_tot = plt.subplots()
    #
    # positions = range(len(scores_tot))
    # bars = ax_tot.bar(positions, scores_tot, color='skyblue', edgecolor='black')
    # ax_tot.bar_label(bars, fmt='%.2f', padding=3)
    # ax_tot.set_xticks(list(positions))  # Set positions first
    # ax_tot.set_xticklabels(feat_names, rotation=15, ha='center')  # Then set text labels
    # ax_tot.set_ylabel("Global Attribution Score")
    # ax_tot.set_title(f"Feature Importance: {data_name}")
    #
    # # Save & Show
    # plot_path_tot = os.path.join(savepath, f"{data_name}_scores_global.png")
    # plt.tight_layout()
    # plt.savefig(plot_path_tot, dpi=300)
    # plt.show()

    with plt.rc_context({'figure.autolayout': False}):
        model.plot()
        plt.savefig(os.path.join(savepath, f"{data_name}_model.png"))
        plt.close()

    # ==========================================
    # 3. Inflection Point Analysis (Layer 0)
    # ==========================================
    print("\n🔍 Analyzing Inflection Points in Layer 0...")
    l = 0
    act = model.act_fun[l]
    ni, no = act.coef.shape[:2]
    coef = act.coef.tolist()
    depth = len(model.act_fun)
    # Pre-allocate indexed by original feature; downstream code uses inflection_points_per_input[mask_idx]
    inflection_points_per_input = [None] * ni
    sort_order_act = np.argsort(scores_tot)[::-1]
    feat_colors = [plt.get_cmap('RdYlBu')(x) for x in np.linspace(0.1, 0.9, ni)]

    fig_eval, axs_eval = plt.subplots(nrows=no, ncols=ni, squeeze=False,
                                      figsize=(4 * ni, 3 * no),
                                      constrained_layout=True)
    fig_spline, axs_spline = plt.subplots(nrows=no, ncols=ni, squeeze=False,
                                          figsize=(4 * ni, 3 * no),
                                          constrained_layout=True)

    for col_pos, i in enumerate(sort_order_act):
        knot_points_actual = act.grid[i, model.k - 1:-2].cpu().detach().numpy()
        # Sweep only the data-range knots (drop the extrapolation padding), so
        # symbolified sqrt/log edges are never evaluated outside their domain.
        sweep_knots = data_range_knots(act, i).cpu().detach().numpy()
        x_sweep = np.linspace(float(sweep_knots.min()),
                              float(sweep_knots.max()), 400)
        x_sweep_raw = denorm(x_sweep, i)  # raw x for plotting (sweep stays norm)
        feature_inflections_all = []
        for j in range(no):
            ax = axs_eval[j, col_pos]
            ax2 = axs_spline[j, col_pos]
            ax3 = ax2.twinx()

            inputs = model.spline_preacts[l][:, j, i].cpu().detach().numpy()
            outputs = model.spline_postacts[l][:, j, i].cpu().detach().numpy()
            coef_node = coef[i][j]
            knot_indices = np.arange(len(coef_node))

            rank = np.argsort(inputs)
            ax.plot(denorm(inputs, i)[rank], outputs[rank], marker='o',
                    color=feat_colors[col_pos], label='Activation')

            # --- coef-based detector (kept as fallback / comparison only) ---
            slope = [x - y for x, y in zip(coef_node[1:], coef_node[:-1])]
            slope_2nd = [(x - y) * 10 for x, y in zip(slope[1:], slope[:-1])]

            ax2.plot(knot_indices, coef_node, marker='o',
                     color=feat_colors[col_pos], label='Coefficients')

            slope_indices = knot_indices[:-1] + 0.5
            ax3.bar(slope_indices, slope, width=0.3, align='center',
                    hatch='///', edgecolor='dimgray', facecolor='none', label='Slope')

            if depth == 1:
                ax3.bar(slope_indices[1:] - 0.3, slope_2nd, width=0.3, align='center',
                        hatch='xx', edgecolor='steelblue', facecolor='none', label='2nd Slope')

            ax2.set_xticks(knot_indices)
            ax2.set_xticklabels([f"{val:.2f}" for val in denorm(knot_points_actual, i)], rotation=45, fontsize=9)

            if depth == 1:
                idx_revert = find_indices_sign_revert(slope_2nd)
                idx_revert = [ir + 1 for ir in idx_revert]
            elif depth == 2:
                idx_revert = find_indices_sign_revert(slope)
            else:
                idx_revert = []

            # --- analytic detector (source of truth for inflection_points_per_input) ---
            ips_ij = find_inflection_points(model, l, i, j_list=[j])
            feature_inflections_all.extend(ips_ij)

            # analytic phi' / phi'' overlay on the activation plot
            _, dphi, d2phi = edge_curves(model, l, i, j, x_sweep)
            ax_d = ax.twinx()
            ax_d.plot(x_sweep_raw, dphi, color='#1f77b4', lw=0.9, ls='--', label=r"$\phi'$")
            ax_d.plot(x_sweep_raw, d2phi, color='#d62728', lw=0.9, ls=':', label=r"$\phi''$")
            ax_d.axhline(0, color='gray', lw=0.5, alpha=0.5)
            ax_d.set_ylabel(r"$\phi'\,,\ \phi''$")

            # coef-based inflection vlines (green dashed) — fallback comparison
            first_c = True
            for ir in idx_revert:
                lab = 'coef-based' if first_c else '_'
                ax2.axvline(x=ir, color='green', linestyle='--', alpha=0.6, label=lab)
                if ir < len(knot_points_actual):
                    ax.axvline(x=denorm([knot_points_actual[ir]], i)[0], color='green',
                               linestyle='--', alpha=0.6, label=lab)
                first_c = False

            # analytic inflection vlines (purple solid) — the actual detection
            if ips_ij:
                frac_idx = np.interp(ips_ij, knot_points_actual,
                                     np.arange(len(knot_points_actual)))
                first_a = True
                for xinf, fi in zip(ips_ij, frac_idx):
                    lab = 'analytic' if first_a else '_'
                    ax.axvline(x=denorm([xinf], i)[0], color='purple', linestyle='-',
                               alpha=0.7, label=lab)
                    ax2.axvline(x=fi, color='purple', linestyle='-', alpha=0.7, label=lab)
                    first_a = False

            ax.set_xlabel(f"{feat_names[i]}")
            ax.set_ylabel(f"node ({l+1}, {j})")
            ax2.set_xlabel(f"{feat_names[i]}")
            ax2.set_ylabel(f"$c_i$ at node ({l+1}, {j})")
            ax3.set_ylabel(f"$\Delta c_i$ & $\Delta^2 c_i$")
            ax3.axhline(0, color='dimgray', linestyle='--', alpha=0.4)
            h_ax, l_ax = ax.get_legend_handles_labels()
            h_d, l_d = ax_d.get_legend_handles_labels()
            ax.legend(h_ax + h_d, l_ax + l_d, loc='best', fontsize=7)
            handles2, labels2 = ax2.get_legend_handles_labels()
            handles3, labels3 = ax3.get_legend_handles_labels()
            ax3.legend(handles2 + handles3, labels2 + labels3, loc='best', fontsize=7)

        feature_inflections = sorted(set(feature_inflections_all))
        inflection_points_per_input[i] = feature_inflections

    fig_eval.savefig(os.path.join(savepath, f"{data_name}_activations_values_L{l}.png"), dpi=300)
    fig_eval.savefig(os.path.join(savepath, f"{data_name}_activations_values_L{l}.svg"), format='svg')
    fig_eval.savefig(os.path.join(savepath, f"{data_name}_activations_values_L{l}.eps"), format='eps')
    fig_spline.savefig(os.path.join(savepath, f"{data_name}_activations_L{l}.png"), dpi=300)
    fig_spline.savefig(os.path.join(savepath, f"{data_name}_activations_L{l}.svg"), format='svg')
    fig_spline.savefig(os.path.join(savepath, f"{data_name}_activations_L{l}.eps"), format='eps')
    plt.close(fig_eval)
    plt.close(fig_spline)
    print(f"📊 Activation analysis saved to: {savepath}")

    # ==========================================
    # 3.6 Ranking transition by small 1st-derivative |φ'|
    # ==========================================
    # Per-feature bottom-layer local sensitivity s_i(x) = Σ_j |φ'_{i,j}(x)| (exact,
    # feature-separable). A ranking transition = where the dominant feature's s_i
    # drops below a small threshold τ (it becomes locally negligible).
    print("\n🏁 Computing ranking transitions (small |φ'|)...")
    # Transition points used by all downstream analysis (§3.5/3.7/3.8/3.9/4):
    # the RANKING-TRANSITION points (τ-crossings of |φ'_i|), replacing the
    # inflection points. Falls back to inflection points if §3.6 fails.
    transition_points_per_input = []
    try:
        rel_thresh = 0.1  # τ = rel_thresh · max_i max_x |φ'_i|  (tunable / exploratory)
        l0 = 0
        knots_all = data_range_knots(act).cpu().detach().numpy()
        x_grid_rt = np.linspace(float(knots_all.min()), float(knots_all.max()), 400)
        transitions, info = find_ranking_transitions(
            model, x_grid=x_grid_rt, rel_thresh=rel_thresh, layer=l0)
        tau = info['tau']
        S = info['S']
        # Per-feature ranking-transition points (all τ-crossings) -> downstream source.
        transition_points_per_input = [
            sorted(t['point'] for t in transitions if t['feat_idx'] == i)
            for i in range(ni)
        ]

        feat_colors_rt = [plt.get_cmap('tab10')(c) for c in range(ni)]
        with plt.rc_context({'figure.autolayout': True}):
            fig_rt, ax_rt = plt.subplots(figsize=(7, 4))
            for i in range(ni):
                ax_rt.plot(x_grid_rt, S[i], color=feat_colors_rt[i], lw=1.4,
                           label=rf"$|\phi'|$ {feat_names[i]}")
            ax_rt.axhline(tau, color='black', ls='--', lw=1.0, alpha=0.7,
                          label=rf"$\tau={rel_thresh:g}\cdot$max")

            # All τ-crossings, every feature: solid = down (→negligible),
            # dotted = up (→active); colored by the crossing feature.
            first_d, first_u = True, True
            for t in transitions:
                c = feat_colors_rt[t['feat_idx']]
                if t['direction'] == 'down':
                    ax_rt.axvline(t['point'], color=c, ls='-', alpha=0.85, lw=1.3,
                                  label='transition (down)' if first_d else '_')
                    first_d = False
                else:
                    ax_rt.axvline(t['point'], color=c, ls=':', alpha=0.7, lw=1.1,
                                  label='transition (up)' if first_u else '_')
                    first_u = False

            # overlay KAN inflection points (normalized) for comparison
            first_inf = True
            for i in range(ni):
                for ip in (inflection_points_per_input[i] or []):
                    ax_rt.axvline(ip, color='green', ls='--', alpha=0.5, lw=0.9,
                                  label='KAN inflection' if first_inf else '_')
                    first_inf = False

            ax_rt.set_xlabel("normalized input value")
            ax_rt.set_ylabel(r"local sensitivity  $\sum_j|\phi'_{ij}|$")
            ax_rt.set_title(rf"{data_name} — ranking transition (small $|\phi'|$)")
            ax_rt.legend(loc='best', fontsize=7)
            for ext in ['.png', '.svg', '.eps']:
                fig_rt.savefig(os.path.join(savepath, f"{data_name}_ranking_transition{ext}"))
            plt.close(fig_rt)

        pd.DataFrame(transitions).to_csv(
            os.path.join(savepath, f"{data_name}_ranking_transition.csv"), index=False)
        down_pts = [round(t['point'], 3) for t in transitions if t['direction'] == 'down']
        up_pts = [round(t['point'], 3) for t in transitions if t['direction'] == 'up']
        print(f"🏁 τ = {tau:.4g}; down-crossings (→negligible) = {down_pts}; "
              f"up-crossings (→active) = {up_pts}")
        print(f"🏁 Saved: {data_name}_ranking_transition.(png/svg/eps/csv)")
    except Exception as e:
        import traceback
        print(f"⚠️ Ranking-transition section 3.6 failed: {e}")
        traceback.print_exc()

    # ==========================================
    # 3.5 Attribution Trajectory across Grid Intervals
    # ==========================================
    print("\n📈 Computing Attribution Trajectory across grid intervals...")

    sort_order_global = sort_order_act  # same ordering; feat_colors already defined in section 3
    rank_of_feat = {int(orig): rank for rank, orig in enumerate(sort_order_global)}

    n_cols_traj = min(ni, 3)
    n_rows_traj = (ni + n_cols_traj - 1) // n_cols_traj

    fig_traj, axs_traj = plt.subplots(n_rows_traj, n_cols_traj, squeeze=False,
                                      figsize=(4 * n_cols_traj, 3 * n_rows_traj),
                                      constrained_layout=True)
    axs_traj_flat = axs_traj.flatten()

    for col_pos, split_feat_idx in enumerate(sort_order_global):
        split_feat_idx = int(split_feat_idx)
        ax = axs_traj_flat[col_pos]
        knots = act.grid[split_feat_idx, model.k - 1:-2].cpu().detach().numpy()

        interval_scores = []
        interval_centers = []

        for lb, ub in zip(knots[:-1], knots[1:]):
            mask = (dataset['train_input'][:, split_feat_idx] > lb) & \
                   (dataset['train_input'][:, split_feat_idx] <= ub)
            if torch.any(mask):
                x_slice = dataset['train_input'][mask, :]
                x_std = torch.std(x_slice, dim=0).detach().cpu().numpy()
                model.forward(x_slice)
                score = model.feature_score.detach().cpu().numpy().copy()
                interval_scores.append(score / (x_std + 1e-6))
                interval_centers.append(float((lb + ub) / 2))

        if len(interval_scores) < 2:
            ax.set_visible(False)
            continue

        scores_arr = np.array(interval_scores)  # (n_valid_intervals, ni)
        # interval centers are normalized; denormalize to raw for the x-axis.
        x_pos = denorm(np.array(interval_centers), split_feat_idx)

        # --- Primary axis: line plots per feature ---
        for orig_idx in sort_order_global:
            orig_idx = int(orig_idx)
            rank = rank_of_feat[orig_idx]
            ax.plot(x_pos, scores_arr[:, orig_idx], marker='o',
                    color=feat_colors[rank],
                    label=f"x{rank}: {feat_names[orig_idx]}")

        for ip in (transition_points_per_input[split_feat_idx] or []):
            ax.axvline(x=denorm([ip], split_feat_idx)[0], color='purple',
                       linestyle='-', alpha=0.7, linewidth=1.2)

        ax.set_xlabel(f"{feat_names[split_feat_idx]}")
        ax.set_ylabel("Normalized Attribution Score")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.2)

        # --- Secondary axis: relative importance as bar plot ---
        if ni >= 2:
            g1_idx = int(sort_order_global[0])
            g2_idx = int(sort_order_global[1])
            log_ratio = np.log10(
                (scores_arr[:, g1_idx] + 1e-9) / (scores_arr[:, g2_idx] + 1e-9)
            )
            ax2 = ax.twinx()
            ax2.bar(x_pos, log_ratio, width=(x_pos[1] - x_pos[0]) * 0.4 if len(x_pos) > 1 else 0.05,
                    color='dimgray', alpha=0.25, zorder=1,
                    label=r'$\mathcal{R}(x_0,x_1)$')
            ax2.axhline(0, color='dimgray', linestyle='--', alpha=0.4)
            ax2.set_ylabel(r'Relative Importance  $\mathcal{R}(x_0,x_1)$')
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc='best')
        else:
            ax.legend(loc='best')

    for k in range(ni, len(axs_traj_flat)):
        axs_traj_flat[k].set_visible(False)

    traj_base = os.path.join(savepath, f"{data_name}_attribution_trajectory")
    fig_traj.savefig(traj_base + ".png", dpi=300)
    fig_traj.savefig(traj_base + ".svg", format='svg')
    fig_traj.savefig(traj_base + ".eps", format='eps')
    plt.close(fig_traj)
    print(f"📊 Attribution trajectory saved to: {traj_base}.png/svg/eps")

    # ==========================================
    # 3.7 Contour Analysis (Analytic Function + KAN Inflection Points)
    # ==========================================
    if nx >= 2:
        print("\n🗺️ Generating Contour Analysis...")

        top_2_idx = np.argsort(scores_tot)[-2:][::-1]
        f1_idx, f2_idx = int(top_2_idx[0]), int(top_2_idx[1])
        f1_name, f2_name = feat_names[f1_idx], feat_names[f2_idx]

        grid_res = 50
        x1_min, x1_max = bounds[f1_idx]
        x2_min, x2_max = bounds[f2_idx]

        x1_lin = np.linspace(x1_min, x1_max, grid_res)
        x2_lin = np.linspace(x2_min, x2_max, grid_res)
        X1_mesh, X2_mesh = np.meshgrid(x1_lin, x2_lin)
        grid_coords = np.stack([X1_mesh.ravel(), X2_mesh.ravel()], axis=-1)

        # Fix all other features at the midpoint of their bounds
        mean_raw = np.array([(b[0] + b[1]) / 2 for b in bounds])
        grid_input = np.tile(mean_raw, (grid_res ** 2, 1))
        grid_input[:, f1_idx] = grid_coords[:, 0]
        grid_input[:, f2_idx] = grid_coords[:, 1]

        Z = np.apply_along_axis(target_func, 1, grid_input).reshape(grid_res, grid_res)

        # Denormalize ranking-transition points from [0.1, 0.9] → raw space
        def get_denorm_ips(feat_idx):
            raw_ips = transition_points_per_input[feat_idx] or []
            valid_ips = [ip for ip in raw_ips if 0.05 < ip < 0.95]
            if not valid_ips:
                return []
            dummy = np.zeros((len(valid_ips), nx))
            dummy[:, feat_idx] = valid_ips
            return scaler_X.inverse_transform(dummy)[:, feat_idx]

        f1_ips = get_denorm_ips(f1_idx)
        f2_ips = get_denorm_ips(f2_idx)

        fig_c, ax_c = plt.subplots(figsize=(4, 3))
        cp = ax_c.contourf(X1_mesh, X2_mesh, Z, levels=30, cmap='RdYlBu_r', alpha=0.8)
        cbar = fig_c.colorbar(cp, ax=ax_c)
        cbar.set_label("y")

        for ip in f1_ips:
            ax_c.axvline(x=ip, color='green', linestyle='--', alpha=0.5)
        for ip in f2_ips:
            ax_c.axhline(y=ip, color='green', linestyle='--', alpha=0.5)

        ax_c.set_xlabel(f1_name)
        ax_c.set_ylabel(f2_name)
        ax_c.set_xlim([x1_min, x1_max])
        ax_c.set_ylim([x2_min, x2_max])

        contour_base = os.path.join(savepath, f"{data_name}_contour_mean_fixed")
        fig_c.savefig(contour_base + ".png")
        fig_c.savefig(contour_base + ".svg", format='svg')
        fig_c.savefig(contour_base + ".eps", format='eps')
        plt.close(fig_c)
        print(f"📊 Contour saved to: {contour_base}.png/svg/eps")

    # ==========================================
    # 3.8 Sectional GSA (AGSM) Comparison
    # ==========================================
    print("\n📐 Computing Sectional GSA (AGSM) [equal + quantile + kan]...")
    try:
        from github.workflows.Hyein.sectional_gsa import (
            compute_gradient_agsm, find_agsm_transition_points,
            plot_agsm_vs_kan, make_batch_func, compute_global_from_sectional,
            make_sections,
        )

        n_sections_agsm = len(act.grid[0, model.k - 1:-2]) - 1
        top2_idx_agsm = np.argsort(scores_tot)[::-1][:2].tolist()
        i_idx, j_idx = top2_idx_agsm[0], top2_idx_agsm[1]

        def _kan_batch_func(X_raw):
            X_np = np.atleast_2d(np.asarray(X_raw, dtype=float))
            X_norm = scaler_X.transform(X_np)
            X_tensor = torch.tensor(X_norm, dtype=torch.float32, device=device)
            with torch.no_grad():
                y_pred = model(X_tensor).cpu().numpy()
            try:
                y_inv = scaler_y.inverse_transform(y_pred)
            except Exception:
                y_inv = y_pred
            return y_inv.ravel()

        batch_func = _kan_batch_func

        def denorm_ips(ips_norm, feat_idx):
            valid = [ip for ip in (ips_norm or []) if 0.05 < ip < 0.95]
            if not valid:
                return []
            dummy = np.zeros((len(valid), nx))
            dummy[:, feat_idx] = valid
            return scaler_X.inverse_transform(dummy)[:, feat_idx].tolist()

        def denorm_knots(knots_norm, feat_idx):
            """Denormalize KAN grid knots (one feature) to raw input space."""
            dummy = np.zeros((len(knots_norm), nx))
            dummy[:, feat_idx] = knots_norm
            return scaler_X.inverse_transform(dummy)[:, feat_idx]

        # Ranking-transition points (raw space) per top-2 feature — used as the
        # KAN-side transition markers in the AGSM comparison (replaces inflection).
        kan_ips_raw_agsm = {idx: denorm_ips(transition_points_per_input[idx], idx)
                            for idx in top2_idx_agsm}

        agsm_results = {}
        for mode in ['equal', 'quantile', 'kan']:
            kwargs = dict(
                func_batch=batch_func,
                bounds=bounds,
                feat_names=feat_names,
                top2_idx=top2_idx_agsm,
                n_sections=n_sections_agsm,
                n_samples_per_section=512,
                seed=42,
                section_mode=mode,
            )
            if mode == 'quantile':
                kwargs['data_X'] = X_train  # raw space training data
            if mode == 'kan':
                # Use the KAN model's actual grid knot positions (raw space)
                # as section boundaries, per investigated feature.
                kan_knots_dict = {}
                for feat_idx in top2_idx_agsm:
                    knots_norm = act.grid[feat_idx, model.k - 1:-2].cpu().detach().numpy()
                    kan_knots_dict[feat_idx] = denorm_knots(knots_norm, feat_idx)
                kwargs['kan_knots_raw'] = kan_knots_dict

            sc, sh, sa, r = compute_gradient_agsm(**kwargs)
            tps = find_agsm_transition_points(
                sc[i_idx], sa[i_idx], sc[j_idx], sa[j_idx],
                feat_names[i_idx], feat_names[j_idx],
            )
            # True section edges (boundary knots) for this mode, per feature.
            # Reuses make_sections with the same args so the edges match exactly
            # what compute_gradient_agsm sectioned on (grid knots for 'kan' mode).
            section_edges_mode = {}
            for feat_idx in top2_idx_agsm:
                lo_f, hi_f = bounds[feat_idx]
                edges_f, _ = make_sections(
                    lo_f, hi_f, n_sections_agsm, mode=mode,
                    data_col=(X_train[:, feat_idx] if mode == 'quantile' else None),
                    kan_knots_raw=(kan_knots_dict[feat_idx] if mode == 'kan' else None),
                )
                section_edges_mode[feat_idx] = edges_f

            agsm_results[mode] = {
                'section_centers': sc, 'S_hat': sh, 'S_a': sa, 'R': r, 'tps': tps,
                'edges': section_edges_mode,
            }

            # Save CSV
            rows = []
            for feat_idx in top2_idx_agsm:
                for k, (center, s_hat, s_a_v, r_v) in enumerate(zip(
                    sc[feat_idx], sh[feat_idx], sa[feat_idx], r[feat_idx]
                )):
                    rows.append({'Feature': feat_names[feat_idx], 'Feature_idx': feat_idx,
                                 'Section_k': k, 'Section_center': center,
                                 'S_hat': s_hat, 'S_a': s_a_v, 'R': r_v})
            pd.DataFrame(rows).to_csv(
                os.path.join(savepath, f"{data_name}_agsm_sectional_{mode}.csv"), index=False
            )

            # KAN attribution per section (sections of i_idx, for comparison with AGSM S_a)
            kan_attr = {}
            for feat_idx in top2_idx_agsm:
                # Mask over the TRUE section edges (boundary knots), not midpoints
                # between centers, so attribution intervals match the sections.
                edges_raw = np.asarray(section_edges_mode[feat_idx], dtype=float)
                n_eff = len(edges_raw) - 1
                dummy = np.zeros((n_eff + 1, nx))
                dummy[:, feat_idx] = edges_raw
                edges_norm = scaler_X.transform(dummy)[:, feat_idx]

                attr_sections = []
                for k in range(n_eff):
                    lo_n = min(edges_norm[k], edges_norm[k + 1])
                    hi_n = max(edges_norm[k], edges_norm[k + 1])
                    x_col = dataset['train_input'][:, feat_idx]
                    mask_k = (x_col >= lo_n) & (x_col < hi_n)
                    if torch.any(mask_k) and mask_k.sum().item() >= 5:
                        x_slice = dataset['train_input'][mask_k]
                        x_std = torch.std(x_slice, dim=0).detach().cpu().numpy()
                        model.forward(x_slice)
                        score = model.feature_score.detach().cpu().numpy().copy()
                        attr_sections.append(score / (x_std + 1e-6))
                    else:
                        attr_sections.append(np.full(nx, np.nan))
                kan_attr[feat_idx] = np.array(attr_sections)  # (n_eff, nx)
            agsm_results[mode]['kan_attr'] = kan_attr

        # Combined plot: 2 rows (one per mode)
        SA_RC_AGSM = {
            'figure.dpi': 150, 'figure.facecolor': 'white', 'figure.autolayout': True,
            'axes.facecolor': 'white', 'axes.edgecolor': '#444444', 'axes.linewidth': 0.8,
            'axes.labelsize': 11, 'axes.labelcolor': 'black', 'axes.grid': False,
            'xtick.labelsize': 9, 'xtick.color': 'black', 'xtick.direction': 'out',
            'ytick.labelsize': 9, 'ytick.color': 'black', 'ytick.direction': 'out',
            'font.family': 'sans-serif', 'font.size': 9, 'font.weight': '300',
            'axes.labelweight': '500', 'text.color': 'black',
            'legend.fontsize': 7, 'legend.framealpha': 0.0, 'lines.linewidth': 1.2,
            'savefig.dpi': 150, 'savefig.bbox': 'tight', 'savefig.facecolor': 'white',
        }

        feat_colors_agsm = ['#1f77b4', '#d62728']
        mode_labels = {'equal': 'Equal-distance', 'quantile': 'Quantile',
                       'kan': 'KAN-grid'}
        plot_modes = ['equal', 'quantile', 'kan']

        with plt.rc_context(SA_RC_AGSM):
            fig, axes = plt.subplots(len(plot_modes), 2, figsize=(10, 7.5),
                                     sharex=False)
            for row, mode in enumerate(plot_modes):
                ax_agsm = axes[row, 0]
                ax_attr = axes[row, 1]
                res = agsm_results[mode]
                sc_plot = res['section_centers']
                sa_plot = res['S_a']
                tps_plot = res['tps']
                edges_mode = res['edges']
                n_eff = len(sc_plot[i_idx])

                # Left column: AGSM S_a (steps transition at the section edges)
                for color, feat_idx in zip(feat_colors_agsm, top2_idx_agsm):
                    _step_over_edges(ax_agsm, edges_mode[feat_idx], sa_plot[feat_idx],
                                     color=color, label=feat_names[feat_idx])

                first_inflect = True
                for feat_idx in top2_idx_agsm:
                    for ip in kan_ips_raw_agsm.get(feat_idx, []):
                        ax_agsm.axvline(x=ip, color='green', linestyle='--', alpha=0.7,
                                        linewidth=1.0,
                                        label='ranking transition' if first_inflect else '_')
                        first_inflect = False

                first_tp = True
                for tp in tps_plot:
                    ax_agsm.axvline(x=tp['point'], color='orange', linestyle=':', alpha=0.8,
                                    linewidth=1.2,
                                    label='AGSM transition' if first_tp else '_')
                    first_tp = False

                ax_agsm.set_xlabel(feat_names[i_idx])
                ax_agsm.set_ylabel(r'$S^a_{l,[k]}$')
                ax_agsm.set_title(f'{mode_labels[mode]} (N={n_eff}) — AGSM')
                ax_agsm.legend(loc='best')

                # Right column: KAN attribution
                kan_attr_plot = res.get('kan_attr', {})
                if kan_attr_plot and i_idx in kan_attr_plot:
                    attr_mat = kan_attr_plot[i_idx]  # (n_eff, nx)
                    for color, feat_idx in zip(feat_colors_agsm, top2_idx_agsm):
                        _step_over_edges(ax_attr, edges_mode[i_idx], attr_mat[:, feat_idx],
                                         color=color, label=feat_names[feat_idx])

                    first_inflect = True
                    for feat_idx in top2_idx_agsm:
                        for ip in kan_ips_raw_agsm.get(feat_idx, []):
                            ax_attr.axvline(x=ip, color='green', linestyle='--', alpha=0.7,
                                            linewidth=1.0,
                                            label='ranking transition' if first_inflect else '_')
                            first_inflect = False

                ax_attr.set_xlabel(feat_names[i_idx])
                ax_attr.set_ylabel('KAN Attribution')
                ax_attr.set_title(f'{mode_labels[mode]} (N={n_eff}) — KAN attr')
                ax_attr.legend(loc='best')

            fig.suptitle(data_name, fontsize=11, fontweight='bold')
            for ext in ['.png', '.svg', '.eps']:
                fig.savefig(os.path.join(savepath, f"{data_name}_agsm_modes{ext}"))
            plt.close(fig)

        print(f"📐 AGSM (equal + quantile + kan) saved: {savepath}")
        for mode, res in agsm_results.items():
            tp_vals = [f"{t['point']:.3f}" for t in res['tps']]
            print(f"   {mode}: AGSM transitions = {tp_vals}")

    except Exception as e:
        import traceback
        print(f"⚠️ AGSM section 3.8 failed: {e}")
        traceback.print_exc()

    # ==========================================
    # 3.9 Curvature-Based Inflection + Per-Interval Dual Measure (New KAN analysis)
    # ==========================================
    # Replaces the coefficient finite-difference inflection detector (section 3)
    # with the ANALYTICAL 2nd derivative of the full learned activation, then
    # compares AGSM S_a and KAN attribution over the inflection-segmented domain.
    print("\n🧭 Computing curvature-based inflection points (analytical 2nd derivative)...")
    try:
        from github.workflows.Hyein.sectional_gsa import (
            compute_gradient_agsm, find_agsm_transition_points,
        )

        if nx < 2:
            raise RuntimeError("curvature section 3.9 needs >=2 inputs; skipping.")

        l = 0
        act = model.act_fun[l]
        top2_curv = np.argsort(scores_tot)[::-1][:2].tolist()
        ci_idx, cj_idx = int(top2_curv[0]), int(top2_curv[1])

        # Symbolified edges route through model.symbolic_fun (spline disabled);
        # the curvature/derivatives below use the symbolic function for those edges.
        sym_info = symbolic_edge_info(model, l)
        if sym_info:
            print("⚠️ Symbolic edges detected (spline branch disabled) — curvature "
                  "uses the symbolic function for these edges:")
            for (ei, ej), nm in sorted(sym_info.items()):
                print(f"     edge ({feat_names[ei]} -> node {ej}): {nm}")

        # --- 1. Ranking-transition points (normalized space) used as the KAN-side
        #        transition points for the dual measure (replaces inflection). ---
        curv_ips_norm = {idx: (transition_points_per_input[idx] or [])
                         for idx in top2_curv}

        # --- 1b. Figure: activation phi(x) with its analytical phi'(x), phi''(x) ---
        # Plots the learned activation and its exact 1st/2nd derivatives per edge,
        # with green vlines at the phi'' zero-crossings (detected inflections).
        # Curves are evaluated over the NORMALIZED sweep (spline space) but the
        # x-axis is denormalized to RAW input values for display.
        no_l = act.coef.shape[1]
        with plt.rc_context({'figure.autolayout': True}):
            fig_d, axs_d = plt.subplots(no_l, len(top2_curv), squeeze=False,
                                        figsize=(5 * len(top2_curv), 3 * no_l))
            for col, i_feat in enumerate(top2_curv):
                knots_i = data_range_knots(act, i_feat).cpu().detach().numpy()
                x_sweep = np.linspace(float(knots_i.min()), float(knots_i.max()), 400)
                x_sweep_raw = denorm(x_sweep, i_feat)  # raw x for display
                for j in range(no_l):
                    ax = axs_d[j, col]
                    ax2 = ax.twinx()
                    phi, dphi, d2phi = edge_curves(model, l, i_feat, j, x_sweep)
                    ln0 = ax.plot(x_sweep_raw, phi, color='#222222', lw=1.6, label=r'$\phi(x)$')
                    ln1 = ax2.plot(x_sweep_raw, dphi, color='#1f77b4', lw=1.0, ls='--',
                                   label=r"$\phi'(x)$")
                    ln2 = ax2.plot(x_sweep_raw, d2phi, color='#d62728', lw=1.0, ls=':',
                                   label=r"$\phi''(x)$")
                    ax2.axhline(0, color='gray', lw=0.6, alpha=0.6)
                    extra_handles = []
                    # green dashed: this edge's phi'' zero-crossings (inflection points)
                    first = True
                    for ip in find_inflection_points(model, l, i_feat, j_list=[j]):
                        h = ax.axvline(denorm([ip], i_feat)[0], color='green', ls='--',
                                       alpha=0.6, lw=1.0,
                                       label='Inflection' if first else '_')
                        if first:
                            extra_handles.append(h)
                        first = False
                    # orange solid: this FEATURE's ranking-transition points (where the
                    # local sensitivity |phi'| drops below tau). Per-feature, so the same
                    # vline is drawn on every edge (node j) of input i_feat.
                    first_t = True
                    for tp in (curv_ips_norm[i_feat] or []):
                        h = ax.axvline(denorm([tp], i_feat)[0], color='darkorange', ls='-',
                                       alpha=0.85, lw=1.3,
                                       label='ranking transition' if first_t else '_')
                        if first_t:
                            extra_handles.append(h)
                        first_t = False
                    ax.set_xlabel(f"{feat_names[i_feat]}")
                    ax.set_ylabel(r'$\phi$')
                    ax2.set_ylabel(r"$\phi'\,,\ \phi''$")
                    sym_tag = (f"  [symbolic: {sym_info[(i_feat, j)]}]"
                               if (i_feat, j) in sym_info else "")
                    ax.set_title(f"edge ({feat_names[i_feat]} -> node {j}){sym_tag}")
                    lns = ln0 + ln1 + ln2 + extra_handles
                    ax.legend(lns, [ln.get_label() for ln in lns], loc='best', fontsize=7)
            fig_d.suptitle(f"{data_name} — activation & analytical derivatives (L0)",
                           fontsize=11, fontweight='bold')
            for ext in ['.png', '.svg', '.eps']:
                fig_d.savefig(os.path.join(savepath,
                              f"{data_name}_activation_derivatives_L0{ext}"))
            plt.close(fig_d)
        print(f"🧭 Activation+derivatives figure saved: "
              f"{data_name}_activation_derivatives_L0.(png/svg/eps)")

        # --- 2. Denormalize to raw space (analysis band filter, as in 3.7/3.8) ---
        def _denorm_feat(vals_norm, feat_idx):
            vals = [v for v in (vals_norm or []) if 0.05 < v < 0.95]
            if not vals:
                return []
            dummy = np.zeros((len(vals), nx))
            dummy[:, feat_idx] = vals
            return scaler_X.inverse_transform(dummy)[:, feat_idx].tolist()

        curv_ips_raw = {idx: _denorm_feat(curv_ips_norm[idx], idx) for idx in top2_curv}

        # --- 3. Single source of truth: inflection-based section edges (raw) ---
        # custom_edges[feat] = [lo, <interior inflections>, hi]. The interior of
        # this array is reused verbatim for (a) AGSM custom_edges, (b) KAN
        # attribution interval masks, and (c) the plotted vlines -> fully traceable.
        custom_edges = {}
        for idx in top2_curv:
            lo_raw, hi_raw = bounds[idx]
            interior = sorted(v for v in curv_ips_raw[idx] if lo_raw < v < hi_raw)
            custom_edges[idx] = np.array([lo_raw] + interior + [hi_raw], dtype=float)

        # --- 4. KAN surrogate batch func (raw space), as in section 3.8 ---
        def _kan_batch_func_curv(X_raw):
            X_np = np.atleast_2d(np.asarray(X_raw, dtype=float))
            X_norm = scaler_X.transform(X_np)
            X_tensor = torch.tensor(X_norm, dtype=torch.float32, device=device)
            with torch.no_grad():
                y_pred = model(X_tensor).cpu().numpy()
            try:
                y_inv = scaler_y.inverse_transform(y_pred)
            except Exception:
                y_inv = y_pred
            return y_inv.ravel()

        # --- 5. AGSM S_a over the inflection-segmented intervals ---
        sc, sh, sa, r = compute_gradient_agsm(
            func_batch=_kan_batch_func_curv, bounds=bounds, feat_names=feat_names,
            top2_idx=top2_curv, n_sections=10, n_samples_per_section=512, seed=42,
            section_mode='custom', custom_edges=custom_edges,
        )
        curv_tps = find_agsm_transition_points(
            sc[ci_idx], sa[ci_idx], sc[cj_idx], sa[cj_idx],
            feat_names[ci_idx], feat_names[cj_idx],
        )

        # --- 6. Per-interval KAN attribution over the SAME intervals (3.8 pattern) ---
        curv_kan_attr = {}
        for feat_idx in top2_curv:
            edges_raw = custom_edges[feat_idx]
            n_eff = len(edges_raw) - 1
            dummy = np.zeros((len(edges_raw), nx))
            dummy[:, feat_idx] = edges_raw
            edges_norm = scaler_X.transform(dummy)[:, feat_idx]
            attr_sections = []
            for kk in range(n_eff):
                lo_n = min(edges_norm[kk], edges_norm[kk + 1])
                hi_n = max(edges_norm[kk], edges_norm[kk + 1])
                x_col = dataset['train_input'][:, feat_idx]
                mask_k = (x_col >= lo_n) & (x_col < hi_n)
                if torch.any(mask_k) and mask_k.sum().item() >= 5:
                    x_slice = dataset['train_input'][mask_k]
                    x_std = torch.std(x_slice, dim=0).detach().cpu().numpy()
                    model.forward(x_slice)
                    score = model.feature_score.detach().cpu().numpy().copy()
                    attr_sections.append(score / (x_std + 1e-6))
                else:
                    attr_sections.append(np.full(nx, np.nan))
            curv_kan_attr[feat_idx] = np.array(attr_sections)  # (n_eff, nx)

        # --- 7. CSV ---
        rows = []
        for feat_idx in top2_curv:
            for kk, (center, s_hat, s_a_v, r_v) in enumerate(zip(
                    sc[feat_idx], sh[feat_idx], sa[feat_idx], r[feat_idx])):
                rows.append({'Feature': feat_names[feat_idx], 'Feature_idx': feat_idx,
                             'Section_k': kk, 'Section_center': center,
                             'S_hat': s_hat, 'S_a': s_a_v, 'R': r_v})
        pd.DataFrame(rows).to_csv(
            os.path.join(savepath, f"{data_name}_curvature_inflection.csv"), index=False)

        # --- 8. Dual-panel figure: AGSM S_a (left) | KAN attribution (right) ---
        # vlines come from custom_edges[ci_idx] interior (the same array driving
        # AGSM sectioning and the ci_idx attribution masks).
        ip_lines = list(custom_edges[ci_idx][1:-1])
        feat_colors_curv = ['#1f77b4', '#d62728']

        def _seg_edges(feat_idx, centers):
            """Section edges for piecewise-constant plotting (the inflection edges)."""
            ce = np.asarray(custom_edges[feat_idx], dtype=float)
            centers = np.asarray(centers, dtype=float)
            if ce.size == centers.size + 1:
                return ce
            lo, hi = float(bounds[feat_idx][0]), float(bounds[feat_idx][1])
            e = np.empty(centers.size + 1)
            e[0], e[-1] = lo, hi
            if centers.size > 1:
                e[1:-1] = 0.5 * (centers[:-1] + centers[1:])
            return e

        with plt.rc_context({'figure.autolayout': True}):
            fig_cv, (ax_l, ax_rt) = plt.subplots(1, 2, figsize=(10, 3.4))

            for color, feat_idx in zip(feat_colors_curv, top2_curv):
                _step_over_edges(ax_l, _seg_edges(feat_idx, sc[feat_idx]),
                                 sa[feat_idx], color=color, label=feat_names[feat_idx])
            first = True
            for ip in ip_lines:
                ax_l.axvline(ip, color='green', linestyle='--', alpha=0.7, linewidth=1.0,
                             label='ranking transition' if first else '_')
                first = False
            first = True
            for tp in curv_tps:
                ax_l.axvline(tp['point'], color='orange', linestyle=':', alpha=0.8,
                             linewidth=1.2, label='AGSM transition' if first else '_')
                first = False
            ax_l.set_xlabel(feat_names[ci_idx])
            ax_l.set_ylabel(r'$S^a_{l,[k]}$')
            ax_l.set_title('AGSM (transition-segmented)')
            ax_l.legend(loc='best')

            attr_mat = curv_kan_attr[ci_idx]
            edges_i = _seg_edges(ci_idx, sc[ci_idx])
            for color, feat_idx in zip(feat_colors_curv, top2_curv):
                # Piecewise-constant over the SAME inflection edges, so the steps
                # change exactly at the inflection vlines.
                _step_over_edges(ax_rt, edges_i, attr_mat[:, feat_idx],
                                 color=color, label=feat_names[feat_idx])
            first = True
            for ip in ip_lines:
                ax_rt.axvline(ip, color='green', linestyle='--', alpha=0.7, linewidth=1.0,
                              label='ranking transition' if first else '_')
                first = False
            ax_rt.set_xlabel(feat_names[ci_idx])
            ax_rt.set_ylabel('KAN Attribution')
            ax_rt.set_title('KAN attribution (transition-segmented)')
            ax_rt.legend(loc='best')

            fig_cv.suptitle(f"{data_name} — ranking-transition dual measure", fontsize=11, fontweight='bold')
            for ext in ['.png', '.svg', '.eps']:
                fig_cv.savefig(os.path.join(savepath, f"{data_name}_curvature_inflection{ext}"))
            plt.close(fig_cv)

        print(f"🧭 Ranking-transition pts (normalized): {curv_ips_norm}")
        print(f"🧭 Ranking-transition pts (raw): {curv_ips_raw}")
        print(f"🧭 AGSM transitions: {[round(t['point'], 3) for t in curv_tps]}")
        print(f"🧭 Saved: {data_name}_curvature_inflection.(png/svg/eps/csv)")

    except Exception as e:
        import traceback
        print(f"⚠️ Curvature section 3.9 failed: {e}")
        traceback.print_exc()

    # ==========================================
    # 4. Range-Based Attribution Scoring (Iterative Search)
    # ==========================================

    # Sort features by global score (Highest -> Lowest)
    sorted_feat_indices = np.argsort(scores_tot)[::-1]
    n_features = scores_tot.shape[0]

    def _build_interval_masks(feat_idx):
        """Build per-interval masks/labels for splitting on feat_idx's transition points.

        Returns (masks, labels, split_points) or None if the feature has no valid
        transition points (in 0.1~0.9) that yield >=2 active intervals.
        """
        raw_ips = transition_points_per_input[feat_idx]
        valid_ips = [ip for ip in raw_ips if ip is not None and 0.1 < ip < 0.9]
        unique_ips = sorted(list(set([round(ip, 3) for ip in valid_ips])))
        if len(unique_ips) == 0:
            return None
        # Intervals: [0.1, ip1, ip2, ..., 0.9]
        mask_interval = [0.1] + unique_ips + [0.9]
        x_mask_data = dataset['train_input'][:, feat_idx]
        candidate_masks = [((x_mask_data > lb) & (x_mask_data <= ub))
                           for lb, ub in zip(mask_interval[:-1], mask_interval[1:])]
        non_empty_count = sum([1 for m in candidate_masks if torch.any(m)])
        if non_empty_count < 2:
            return None
        # Labels show RAW (un-normalized) x boundaries: inverse-transform the
        # normalized interval edges through scaler_X. Masking itself stays in
        # normalized space, where the data and the 0.1/0.9 bounds live.
        dummy = np.zeros((len(mask_interval), nx))
        dummy[:, feat_idx] = mask_interval
        mask_interval_raw = scaler_X.inverse_transform(dummy)[:, feat_idx]
        labels = [f'{lb:.3g} < x{feat_idx} <= {ub:.3g}'
                  for lb, ub in zip(mask_interval_raw[:-1], mask_interval_raw[1:])]
        return candidate_masks, labels, mask_interval

    def _compute_interval_scores(masks, labels=None):
        """Forward-pass KAN attribution per interval, normalized by per-slice input std."""
        scores_interval_norm = []
        for i, mask in enumerate(masks):
            if torch.any(mask):
                x_tensor_masked = dataset['train_input'][mask, :]
                # Standard deviation of input in this slice (used for normalization)
                x_std = torch.std(x_tensor_masked, dim=0).detach().cpu().numpy()
                # Forward pass on masked data to get local attribution
                model.forward(x_tensor_masked)
                score_masked = model.feature_score.detach().cpu().numpy()
                scores_interval_norm.append(score_masked / (x_std + 1e-6))
                if labels is not None:
                    print(f"   Interval {labels[i]}: {mask.sum().item()} samples")
            else:
                scores_interval_norm.append(np.zeros(scores_tot.shape))
                if labels is not None:
                    print(f"   Interval {labels[i]}: 0 samples (Skipping)")
        return scores_interval_norm

    def _plot_interval_scores(scores_interval_norm, labels, feat_idx):
        """Grouped bar chart of per-feature attribution across feat_idx's intervals."""
        width = 0.2
        n_intervals = len(scores_interval_norm)
        fig, ax = plt.subplots(figsize=(max(8, n_intervals * 2), 5))
        x_positions = np.arange(n_intervals)
        max_score = max([max(s) for s in scores_interval_norm]) if scores_interval_norm else 1.0

        for fi in range(n_features):
            feat_scores = [s[fi] for s in scores_interval_norm]
            offset = (fi - n_features / 2) * width + width / 2
            ax.bar(x_positions + offset, feat_scores, width, label=f"{feat_names[fi]}")

        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=15, ha='center', fontsize=9)
        ax.set_ylabel("Normalized Attribution Score")
        ax.set_title(f"Feature Importance per Range (sliced by {feat_names[feat_idx]})")
        ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
        ax.set_ylim(0, max_score * 1.2)
        plt.tight_layout()
        fname = f"{data_name}_scores_interval_x{feat_idx}.png"
        plot_path_score = os.path.join(savepath, fname)
        plt.savefig(plot_path_score)
        # Also drop a copy into a shared gallery folder so every function's
        # score-interval plots can be browsed together in one place. Anchor on the
        # project root (savepath may be redirected to a kan_models/<mode> subfolder)
        # and tag non-default modes so spline/symbolic copies sit beside the
        # as-saved one instead of clobbering it.
        gallery_dir = os.path.join(os.getcwd(), 'github', 'workflows', 'Hyein',
                                   'figures_for_paper', 'scores_interval_all')
        os.makedirs(gallery_dir, exist_ok=True)
        mode_tag = '' if model_mode == 'as-saved' else f'_{model_mode}'
        gallery_path = os.path.join(
            gallery_dir, f"{data_name}{mode_tag}_scores_interval_x{feat_idx}.png")
        plt.savefig(gallery_path)
        plt.close(fig)
        print(f"📊 Range-based score plot saved to: {plot_path_score}")
        print(f"🖼️  Gallery copy saved to: {gallery_path}")
        return plot_path_score

    # Draw a per-interval score plot for EVERY feature that has transition points.
    # The first qualifying feature (highest global score) is kept as the "selected"
    # feature for the downstream split-data / NN-training pipeline.
    print("\n🔍 Plotting per-interval scores for every feature with transition points...")

    selected_mask_idx = None
    selected_split_points = None
    masks = []
    labels = []
    scores_interval_norm = []

    for feat_idx in sorted_feat_indices:
        feat_name = feat_names[feat_idx]
        built = _build_interval_masks(feat_idx)
        if built is None:
            print(f"   Feature {feat_idx} ({feat_name}): skipping "
                  f"(no valid transition points / <2 active intervals).")
            continue

        f_masks, f_labels, f_split = built
        print(f"   Feature {feat_idx} ({feat_name}): {len(f_labels)} intervals ✅")
        f_scores = _compute_interval_scores(f_masks, f_labels)
        _plot_interval_scores(f_scores, f_labels, feat_idx)

        # Keep the first (highest-scoring) qualifying feature for downstream use.
        if selected_mask_idx is None:
            selected_mask_idx = feat_idx
            selected_split_points = f_split
            masks = f_masks
            labels = f_labels
            scores_interval_norm = f_scores

    # Fallback: If no feature provided a valid split, pick the top feature (to prevent crash)
    if selected_mask_idx is None:
        print("⚠️ Warning: No feature provided a valid split. Defaulting to top feature.")
        selected_mask_idx = sorted_feat_indices[0]
        x_mask_data = dataset['train_input'][:, selected_mask_idx]
        masks = [(x_mask_data > -np.inf)]  # Dummy mask (all data)
        labels = ["All Range"]
        scores_interval_norm = _compute_interval_scores(masks, labels)
        _plot_interval_scores(scores_interval_norm, labels, selected_mask_idx)

    print(f"\n✂️ Selected Feature {selected_mask_idx} ({feat_names[selected_mask_idx]}) "
          f"for downstream split-data pipeline.")

    # ==========================================
    # 4.5 [NEW] Save Range Split Data for NN Training
    # ==========================================
    split_data_savepath = os.path.join(savepath, f"{data_name}_range_split_data.pkl")

    split_data = {
        'dataset': dataset,
        'masks': masks,
        'labels': labels,
        'selected_mask_idx': selected_mask_idx,
        'selected_mask_name': feat_names[selected_mask_idx],
        'split_points': selected_split_points,  # interval boundaries in [0.1, 0.9] space
        'inflection_points_per_input': inflection_points_per_input,  # per-feature inflection points (original)
        'transition_points_per_input': transition_points_per_input,  # ranking-transition points (used downstream)
        'feature_names': feat_names,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }

    joblib.dump(split_data, split_data_savepath)

    # Save KAN interval scores as CSV for cross-method comparison
    kan_scores_df = pd.DataFrame(scores_interval_norm, columns=feat_names)
    kan_scores_df.insert(0, 'Interval_Label', labels)
    kan_scores_df.to_csv(os.path.join(savepath, f"{data_name}_kan_interval_scores.csv"), index=False)

    # ==========================================
    # 5. Plot Range-Based Scores
    # ==========================================
    # Per-interval score plots are now drawn above for EVERY feature with transition
    # points (see `_plot_interval_scores` in the loop). Nothing to do here.

    # ==========================================
    # 6. Attribution Scoring on Saltelli Dataset
    # (mirrors toy_KAN_sweep.py lines 339-373)
    # ==========================================
    print("\n🧂 [6] Computing Attribution Scores on Saltelli Dataset...")

    problem = {
        'num_vars': nx,
        'names': feat_names,
        'bounds': bounds
    }
    X_saltelli_raw = saltelli.sample(problem, 512, calc_second_order=True, seed=42)
    X_saltelli_norm = scaler_X.transform(X_saltelli_raw)
    X_saltelli_tensor = torch.tensor(X_saltelli_norm, dtype=torch.float32, device=device)

    model.forward(X_saltelli_tensor)
    scores_saltelli = model.feature_score.detach().cpu().numpy()

    if len(scores_saltelli.shape) > 1:
        scores_saltelli = scores_saltelli.flatten()

    fig_s, ax_s = plt.subplots()
    positions = range(len(scores_saltelli))
    bars = ax_s.bar(positions, scores_saltelli, color='skyblue', edgecolor='black')
    ax_s.bar_label(bars, fmt='%.2f', padding=3)
    ax_s.set_xticks(list(positions))
    ax_s.set_xticklabels(feat_names, rotation=15, ha='center')
    ax_s.set_ylabel("Global Attribution Score")
    ax_s.set_title(f"Feature Importance (Saltelli): {data_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, f"{data_name}_scores_global_saltelli.png"), dpi=300)
    # plt.show()

    df_scores_saltelli = pd.DataFrame({
        'Feature': feat_names,
        'Global_Attribution_Score': scores_saltelli
    }).sort_values(by='Global_Attribution_Score', ascending=False)
    df_scores_saltelli.to_csv(
        os.path.join(savepath, f"{data_name}_global_attribution_scores_saltelli.csv"), index=False
    )
    print(f"📊 Saltelli attribution scores saved to: {savepath}")


if __name__ == "__main__":
    main()