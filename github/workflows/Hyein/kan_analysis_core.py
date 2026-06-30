"""Shared KAN post-training analysis core.

This module is the single source of the post-training analysis that
``toy_KAN_analyze.py`` (reference), ``grid_k_sweep.py`` and
``material_KAN_analyze.py`` all call, so the SAME logic (ranking-transition
detection, per-interval AGSM + KAN attribution, scores_interval, activation/φ′
plots, Saltelli) runs everywhere.

Design rules
------------
* Build ON the low-level predefined functions, never duplicate them:
  ``bspline_curvature`` (edge_curves / symbolic_edge_info / feature_sensitivity /
  find_ranking_transitions / data_range_knots) and ``sectional_gsa``
  (compute_gradient_agsm / find_agsm_transition_points / make_sections /
  make_batch_func / compute_global_from_sectional). ``robustness_symbolify``
  supplies the spline/symbolic reading builders.
* Space convention: inflection / ranking-transition points and knots are kept in
  NORMALIZED space internally, denormalized to RAW only for plotting via the
  single ``denorm`` helper. The §3.6 ranking figure is the one exception: it
  overlays features on a shared normalized axis on purpose.
* ``model-only`` sections are transferable across the three consumers;
  ``true_func``-gated bits (§2.5 ground-truth overlay, §3.7 analytic contour
  surface) degrade gracefully when ``true_func is None``.
* ``tag`` prefixes every output filename ("" for toy → filenames unchanged).

Toy is the reference whose outputs must stay byte-identical (parity).
"""

import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from SALib.sample import sobol as saltelli

from github.workflows.Hyein.bspline_curvature import (
    edge_curves, symbolic_edge_info,
    feature_sensitivity, find_ranking_transitions, data_range_knots,
)
from github.workflows.Hyein.robustness_symbolify import (
    _build_spline_reading, _build_symbolic_reading,
)


# ==========================================================================
# Shared plot style + step helper (moved verbatim from toy_KAN_analyze.py)
# ==========================================================================
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


# ==========================================================================
# Single denorm helper (collapses toy's 5 variants: §denorm, get_denorm_pts,
# denorm_pts, denorm_knots, _denorm_feat). ``band`` selects the analysis-band
# filter that some callers applied before denormalizing:
#   band=None        -> no filter (plain value -> raw, used for sweeps/knots)
#   band=(0.05,0.95) -> keep 0.05 < v < 0.95 (§3.7/§3.8/§3.9 transition markers)
#   band=(0.1, 0.9)  -> keep 0.1  < v < 0.9  (kept available for callers)
# ==========================================================================
def denorm(vals, feat_idx, scaler_X, nx, band=None):
    """Normalized value(s) for one feature → raw input space (for plotting).

    With ``band=(lo, hi)`` the values are first filtered to ``lo < v < hi`` and a
    Python list is returned (matching toy's transition-marker helpers); with
    ``band=None`` every value is mapped and a numpy array is returned (matching
    toy's plain ``denorm`` used for sweeps and knot ticks).
    """
    if band is None:
        arr = np.atleast_1d(np.asarray(vals, dtype=float))
        if arr.size == 0:
            return arr
        dummy = np.zeros((arr.size, nx))
        dummy[:, feat_idx] = arr
        return scaler_X.inverse_transform(dummy)[:, feat_idx]
    lo, hi = band
    valid = [v for v in (vals or []) if lo < v < hi]
    if not valid:
        return []
    dummy = np.zeros((len(valid), nx))
    dummy[:, feat_idx] = valid
    return scaler_X.inverse_transform(dummy)[:, feat_idx].tolist()


# ==========================================================================
# §3.8 / §3.9 shared KAN surrogate batch func (single copy; toy had two
# identical inner defs `_kan_batch_func` / `_kan_batch_func_dual`).
# ==========================================================================
def kan_batch_func(model, scaler_X, scaler_y, device):
    """Return a raw-space surrogate ``f(X_raw[m,n]) -> y[m]`` of the KAN model."""
    def _batch(X_raw):
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
    return _batch


# ==========================================================================
# §2.1 model-reading-mode selection (as-saved / spline / symbolic) + two-phase
# escalation. MODEL-ONLY: only the refit-label source is parameterized
# (``train_label_norm`` — toy passes regenerated y; material passes real labels).
# ==========================================================================
def select_model_reading(model, mode, *, model_wrapper=None, refit=False,
                         refit_steps=None, train_input_norm=None,
                         train_label_norm=None, device='cpu'):
    """Return ``(model, sym_info_saved, msg)`` for the requested reading mode.

    ``train_input_norm`` / ``train_label_norm`` are torch tensors on ``device``
    (the normalized training inputs/labels); they drive the symbolic-mode LBFGS
    refit + R² scorer. The label source is the caller's responsibility.
    """
    sym_info_saved = symbolic_edge_info(model, 0)
    msg = ''
    if mode == 'as-saved':
        msg = (f"🧩 Model reading: as-saved (symbolic edges on disk: "
               f"{sorted(sym_info_saved.items()) if sym_info_saved else 'none'}).")
    elif mode == 'spline':
        model = _build_spline_reading(model, sym_info_saved)
        if model_wrapper is not None:
            model_wrapper.model = model
        msg = (f"🧩 Model reading: strictly SPLINE — symbolic branch off; "
               f"{len(sym_info_saved)} symbolified edge(s) reverted to spline.")
    elif mode == 'symbolic':
        lr = getattr(model_wrapper, 'lr', 0.1)
        stop_grid = getattr(model_wrapper, 'stop_grid_update_step', 20)
        # Two-phase accuracy escalation, both stopping early at R²≥0.8:
        #   Phase 1 — sweep auto_symbolic's a/b search range (±10→±20→±50→±100→±200).
        #   Phase 2 — if still under target, escalate LBFGS refit steps at the best
        #             a/b range (100→200→…→cap). The cap is --refit-steps (else 500).
        ab_ranges = [10, 20, 50, 100, 200]
        cap = refit_steps if refit_steps is not None else 500
        refit_steps_schedule = sorted({s for s in [100, 200, 300, 500, 1000] if s < cap}
                                      | {cap})
        fit_dataset = {
            'train_input': train_input_norm, 'train_label': train_label_norm,
            'test_input': train_input_norm, 'test_label': train_label_norm,
        }
        y_true_norm = train_label_norm.detach().cpu().numpy().reshape(-1)

        def _sym_r2(cand):
            """R2 of a candidate symbolic reading vs the (normalized) training labels."""
            with torch.no_grad():
                yp = cand.forward(train_input_norm).detach().cpu().numpy().reshape(-1)
            return float(r2_score(y_true_norm, yp))

        # Only the introduce-symbolify (pure-spline) path runs auto_symbolic +
        # escalation; already-saved symbolic models are used as-is.
        model, n_edges, refit_done = _build_symbolic_reading(
            model, sym_info_saved, train_input_norm, refit,
            fit_dataset, refit_steps_schedule[0], lr, stop_grid,
            ab_ranges=ab_ranges, r2_target=0.8, r2_scorer=_sym_r2,
            refit_steps_schedule=refit_steps_schedule)
        if model_wrapper is not None:
            model_wrapper.model = model
        final_r2 = _sym_r2(model)
        kind = 'as-saved symbolic' if sym_info_saved else 'introduced via auto_symbolic'
        extra = '' if sym_info_saved else f"; LBFGS refit={'done' if refit_done else 'off'}"
        msg = (f"🧩 Model reading: strictly SYMBOLIC — {kind}; "
               f"{n_edges} symbolified edge(s){extra}; R²={final_r2:.4f}.")
    return model, sym_info_saved, msg


# ==========================================================================
# §3 ranking transitions (single source). rel_thresh=0.2 preserved.
# ==========================================================================
def compute_ranking_transitions(model, scaler_X, nx, rel_thresh=0.2, layer=0):
    """Compute ranking-transition points (1st-derivative |φ'| τ-crossings).

    Returns ``(transition_points_per_input, transitions, info, x_grid)`` where
    ``transition_points_per_input[i]`` is the sorted normalized points for input
    ``i`` — the single source of ranking-transition points for all downstream
    sections.
    """
    act = model.act_fun[layer]
    ni = act.coef.shape[0]
    transition_points_per_input = [[] for _ in range(ni)]
    transitions, info, x_grid_rt = [], None, None
    try:
        knots_all = data_range_knots(act).cpu().detach().numpy()
        x_grid_rt = np.linspace(float(knots_all.min()), float(knots_all.max()), 400)
        transitions, info = find_ranking_transitions(
            model, x_grid=x_grid_rt, rel_thresh=rel_thresh, layer=layer)
        transition_points_per_input = [
            sorted(t['point'] for t in transitions if t['feat_idx'] == i)
            for i in range(ni)
        ]
    except Exception as e:
        import traceback
        print(f"⚠️ Ranking-transition computation failed: {e}")
        traceback.print_exc()
    return transition_points_per_input, transitions, info, x_grid_rt


# ==========================================================================
# §3 activation / φ′ + spline-coef figures. MODEL-ONLY.
# ==========================================================================
def plot_activations(model, scaler_X, nx, feat_names, scores_tot,
                     transition_points_per_input, savepath, data_name,
                     tag='', layer=0):
    """§3: activation values + analytic φ' overlay, and spline coefficients."""
    l = layer
    act = model.act_fun[l]
    ni, no = act.coef.shape[:2]
    coef = act.coef.tolist()
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
        x_sweep_raw = denorm(x_sweep, i, scaler_X, nx)  # raw x (sweep stays norm)
        for j in range(no):
            ax = axs_eval[j, col_pos]
            ax2 = axs_spline[j, col_pos]
            ax3 = ax2.twinx()

            inputs = model.spline_preacts[l][:, j, i].cpu().detach().numpy()
            outputs = model.spline_postacts[l][:, j, i].cpu().detach().numpy()
            coef_node = coef[i][j]
            knot_indices = np.arange(len(coef_node))

            rank = np.argsort(inputs)
            ax.plot(denorm(inputs, i, scaler_X, nx)[rank], outputs[rank], marker='o',
                    color=feat_colors[col_pos], label='Activation')

            # coefficient first-difference (slope), shown as bars on ax3
            slope = [x - y for x, y in zip(coef_node[1:], coef_node[:-1])]

            ax2.plot(knot_indices, coef_node, marker='o',
                     color=feat_colors[col_pos], label='Coefficients')

            slope_indices = knot_indices[:-1] + 0.5
            ax3.bar(slope_indices, slope, width=0.3, align='center',
                    hatch='///', edgecolor='dimgray', facecolor='none', label='Slope')

            ax2.set_xticks(knot_indices)
            ax2.set_xticklabels([f"{val:.2f}" for val in denorm(knot_points_actual, i, scaler_X, nx)], rotation=45, fontsize=9)

            # analytic phi' overlay on the activation plot
            _, dphi, _ = edge_curves(model, l, i, j, x_sweep)
            ax_d = ax.twinx()
            ax_d.plot(x_sweep_raw, dphi, color='#1f77b4', lw=0.9, ls='--', label=r"$\phi'$")
            ax_d.axhline(0, color='gray', lw=0.5, alpha=0.5)
            ax_d.set_ylabel(r"$\phi'$")

            # ranking-transition vlines (per feature: where s_i=Σ_j|φ'| drops below τ)
            first_t = True
            for tp in (transition_points_per_input[i] or []):
                ax.axvline(x=denorm([tp], i, scaler_X, nx)[0], color='darkorange', linestyle='-',
                           alpha=0.85, lw=1.3,
                           label='ranking transition' if first_t else '_')
                first_t = False

            ax.set_xlabel(f"{feat_names[i]}")
            ax.set_ylabel(f"node ({l+1}, {j})")
            ax2.set_xlabel(f"{feat_names[i]}")
            ax2.set_ylabel(f"$c_i$ at node ({l+1}, {j})")
            ax3.set_ylabel(r"$\Delta c_i$")
            ax3.axhline(0, color='dimgray', linestyle='--', alpha=0.4)
            h_ax, l_ax = ax.get_legend_handles_labels()
            h_d, l_d = ax_d.get_legend_handles_labels()
            ax.legend(h_ax + h_d, l_ax + l_d, loc='best', fontsize=7)
            handles2, labels2 = ax2.get_legend_handles_labels()
            handles3, labels3 = ax3.get_legend_handles_labels()
            ax3.legend(handles2 + handles3, labels2 + labels3, loc='best', fontsize=7)

    pre = f"{tag}{data_name}"
    fig_eval.savefig(os.path.join(savepath, f"{pre}_activations_values_L{l}.png"), dpi=300)
    fig_eval.savefig(os.path.join(savepath, f"{pre}_activations_values_L{l}.svg"), format='svg')
    fig_eval.savefig(os.path.join(savepath, f"{pre}_activations_values_L{l}.eps"), format='eps')
    fig_spline.savefig(os.path.join(savepath, f"{pre}_activations_L{l}.png"), dpi=300)
    fig_spline.savefig(os.path.join(savepath, f"{pre}_activations_L{l}.svg"), format='svg')
    fig_spline.savefig(os.path.join(savepath, f"{pre}_activations_L{l}.eps"), format='eps')
    plt.close(fig_eval)
    plt.close(fig_spline)
    print(f"📊 Activation analysis saved to: {savepath}")


# ==========================================================================
# §3.6 ranking-transition figure (shared NORMALIZED x-axis). MODEL-ONLY.
# ==========================================================================
def plot_ranking_transition_figure(model, feat_names, transitions, info, x_grid_rt,
                                    rel_thresh, savepath, data_name, tag='', layer=0):
    """§3.6: per-feature local sensitivity Σ_j|φ'| with τ-crossings + CSV."""
    print("\n🏁 Plotting ranking transitions (small |φ'|)...")
    if info is None:
        return
    act = model.act_fun[layer]
    ni = act.coef.shape[0]
    pre = f"{tag}{data_name}"
    try:
        tau = info['tau']
        S = info['S']
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

            ax_rt.set_xlabel("normalized input value")
            ax_rt.set_ylabel(r"local sensitivity  $\sum_j|\phi'_{ij}|$")
            ax_rt.set_title(rf"{data_name} — ranking transition (small $|\phi'|$)")
            ax_rt.legend(loc='best', fontsize=7)
            for ext in ['.png', '.svg', '.eps']:
                fig_rt.savefig(os.path.join(savepath, f"{pre}_ranking_transition{ext}"))
            plt.close(fig_rt)

        pd.DataFrame(transitions).to_csv(
            os.path.join(savepath, f"{pre}_ranking_transition.csv"), index=False)
        down_pts = [round(t['point'], 3) for t in transitions if t['direction'] == 'down']
        up_pts = [round(t['point'], 3) for t in transitions if t['direction'] == 'up']
        print(f"🏁 τ = {tau:.4g}; down-crossings (→negligible) = {down_pts}; "
              f"up-crossings (→active) = {up_pts}")
        print(f"🏁 Saved: {pre}_ranking_transition.(png/svg/eps/csv)")
    except Exception as e:
        import traceback
        print(f"⚠️ Ranking-transition figure (§3.6) failed: {e}")
        traceback.print_exc()


# ==========================================================================
# §3.5 attribution trajectory across grid intervals. MODEL-ONLY.
# ==========================================================================
def plot_attribution_trajectory(model, dataset, scaler_X, nx, feat_names, scores_tot,
                                transition_points_per_input, savepath, data_name,
                                tag='', layer=0):
    """§3.5: per-grid-interval KAN attribution trajectory + relative-importance bars."""
    print("\n📈 Computing Attribution Trajectory across grid intervals...")
    l = layer
    act = model.act_fun[l]
    ni = act.coef.shape[0]
    sort_order_global = np.argsort(scores_tot)[::-1]
    feat_colors = [plt.get_cmap('RdYlBu')(x) for x in np.linspace(0.1, 0.9, ni)]
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
        x_pos = denorm(np.array(interval_centers), split_feat_idx, scaler_X, nx)

        # --- Primary axis: line plots per feature ---
        for orig_idx in sort_order_global:
            orig_idx = int(orig_idx)
            rank = rank_of_feat[orig_idx]
            ax.plot(x_pos, scores_arr[:, orig_idx], marker='o',
                    color=feat_colors[rank],
                    label=f"x{rank}: {feat_names[orig_idx]}")

        for ip in (transition_points_per_input[split_feat_idx] or []):
            ax.axvline(x=denorm([ip], split_feat_idx, scaler_X, nx)[0], color='purple',
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

    traj_base = os.path.join(savepath, f"{tag}{data_name}_attribution_trajectory")
    fig_traj.savefig(traj_base + ".png", dpi=300)
    fig_traj.savefig(traj_base + ".svg", format='svg')
    fig_traj.savefig(traj_base + ".eps", format='eps')
    plt.close(fig_traj)
    print(f"📊 Attribution trajectory saved to: {traj_base}.png/svg/eps")


# ==========================================================================
# §2.5 input-vs-output (TRUE_FUNC-GATED via y_true). MODEL-ONLY otherwise.
# ==========================================================================
def plot_input_vs_output(model, dataset, scaler_y, X_train, feat_names,
                         savepath, data_name, tag='', y_true=None):
    """§2.5: per-feature scatter of prediction (and ground truth if y_true given)."""
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

        # Plot Ground Truth (Gray) — only when the true labels are available.
        if y_true is not None:
            ax.scatter(X_train[:, i], y_true, alpha=0.5, c='gray', s=15, label='Ground Truth')

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
    plot_path_io = os.path.join(savepath, f"{tag}{data_name}_input_vs_output.png")
    plt.savefig(plot_path_io, dpi=300)
    plt.close(fig_io)


# ==========================================================================
# §3.7 contour (TRUE_FUNC-GATED). Analytic surface if true_func else KAN-forward.
# ==========================================================================
def plot_contour(model, scaler_X, scaler_y, nx, bounds, feat_names, scores_tot,
                 transition_points_per_input, savepath, data_name,
                 tag='', true_func=None, device='cpu'):
    """§3.7: 2-D contour of the response with ranking-transition guide lines.

    Uses the analytic ``true_func`` surface when given, else a KAN-forward
    surface (material §9 style).
    """
    if nx < 2:
        return
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

    if true_func is not None:
        Z = np.apply_along_axis(true_func, 1, grid_input).reshape(grid_res, grid_res)
    else:
        # KAN-forward surface (no analytic ground truth available).
        batch = kan_batch_func(model, scaler_X, scaler_y, device)
        Z = batch(grid_input).reshape(grid_res, grid_res)

    # Denormalize ranking-transition points from [0.1, 0.9] → raw space (band filter).
    f1_pts = denorm(transition_points_per_input[f1_idx], f1_idx, scaler_X, nx, band=(0.05, 0.95))
    f2_pts = denorm(transition_points_per_input[f2_idx], f2_idx, scaler_X, nx, band=(0.05, 0.95))

    fig_c, ax_c = plt.subplots(figsize=(4, 3))
    cp = ax_c.contourf(X1_mesh, X2_mesh, Z, levels=30, cmap='RdYlBu_r', alpha=0.8)
    cbar = fig_c.colorbar(cp, ax=ax_c)
    cbar.set_label("y")

    for ip in f1_pts:
        ax_c.axvline(x=ip, color='green', linestyle='--', alpha=0.5)
    for ip in f2_pts:
        ax_c.axhline(y=ip, color='green', linestyle='--', alpha=0.5)

    ax_c.set_xlabel(f1_name)
    ax_c.set_ylabel(f2_name)
    ax_c.set_xlim([x1_min, x1_max])
    ax_c.set_ylim([x2_min, x2_max])

    contour_base = os.path.join(savepath, f"{tag}{data_name}_contour_mean_fixed")
    fig_c.savefig(contour_base + ".png")
    fig_c.savefig(contour_base + ".svg", format='svg')
    fig_c.savefig(contour_base + ".eps", format='eps')
    plt.close(fig_c)
    print(f"📊 Contour saved to: {contour_base}.png/svg/eps")


# ==========================================================================
# §3.8 AGSM dual-measure: equal / quantile / kan. MODEL-ONLY.
# ==========================================================================
def agsm_dual_measure(model, dataset, scaler_X, scaler_y, nx, bounds, feat_names,
                      scores_tot, transition_points_per_input, savepath, data_name,
                      X_train=None, tag='', device='cpu', layer=0):
    """§3.8: AGSM sectional sensitivity (equal/quantile/kan) vs KAN attribution."""
    print("\n📐 Computing Sectional GSA (AGSM) [equal + quantile + kan]...")
    try:
        from github.workflows.Hyein.sectional_gsa import (
            compute_gradient_agsm, find_agsm_transition_points,
            make_sections,
        )

        act = model.act_fun[layer]
        n_sections_agsm = len(act.grid[0, model.k - 1:-2]) - 1
        top2_idx_agsm = np.argsort(scores_tot)[::-1][:2].tolist()
        i_idx, j_idx = top2_idx_agsm[0], top2_idx_agsm[1]

        batch_func = kan_batch_func(model, scaler_X, scaler_y, device)

        def denorm_pts(pts_norm, feat_idx):
            return denorm(pts_norm, feat_idx, scaler_X, nx, band=(0.05, 0.95))

        def denorm_knots(knots_norm, feat_idx):
            """Denormalize KAN grid knots (one feature) to raw input space."""
            return denorm(knots_norm, feat_idx, scaler_X, nx)

        # Ranking-transition points (raw space) per top-2 feature — used as the
        # KAN-side transition markers in the AGSM comparison.
        kan_tr_raw_agsm = {idx: denorm_pts(transition_points_per_input[idx], idx)
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
            kan_knots_dict = None
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
                os.path.join(savepath, f"{tag}{data_name}_agsm_sectional_{mode}.csv"), index=False
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

        # Combined plot: rows (one per mode)
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
                    for ip in kan_tr_raw_agsm.get(feat_idx, []):
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
                        for ip in kan_tr_raw_agsm.get(feat_idx, []):
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
                fig.savefig(os.path.join(savepath, f"{tag}{data_name}_agsm_modes{ext}"))
            plt.close(fig)

        print(f"📐 AGSM (equal + quantile + kan) saved: {savepath}")
        for mode, res in agsm_results.items():
            tp_vals = [f"{t['point']:.3f}" for t in res['tps']]
            print(f"   {mode}: AGSM transitions = {tp_vals}")

    except Exception as e:
        import traceback
        print(f"⚠️ AGSM section 3.8 failed: {e}")
        traceback.print_exc()


# ==========================================================================
# §3.9 transition-segmented dual measure. MODEL-ONLY.
# ==========================================================================
def transition_segmented_dual_measure(model, dataset, scaler_X, scaler_y, nx, bounds,
                                       feat_names, scores_tot,
                                       transition_points_per_input, savepath, data_name,
                                       tag='', device='cpu', layer=0):
    """§3.9: AGSM S_a vs KAN attribution over ranking-transition–segmented intervals."""
    print("\n🧭 Computing ranking-transition–segmented dual measure (AGSM + attribution)...")
    try:
        from github.workflows.Hyein.sectional_gsa import (
            compute_gradient_agsm, find_agsm_transition_points,
        )

        if nx < 2:
            raise RuntimeError("section 3.9 needs >=2 inputs; skipping.")

        l = layer
        act = model.act_fun[l]
        top2_dual = np.argsort(scores_tot)[::-1][:2].tolist()
        ci_idx, cj_idx = int(top2_dual[0]), int(top2_dual[1])

        # Symbolified edges route through model.symbolic_fun (spline disabled);
        # the derivatives below use the symbolic function for those edges.
        sym_info = symbolic_edge_info(model, l)
        if sym_info:
            print("⚠️ Symbolic edges detected (spline branch disabled) — φ/φ' "
                  "use the symbolic function for these edges:")
            for (ei, ej), nm in sorted(sym_info.items()):
                print(f"     edge ({feat_names[ei]} -> node {ej}): {nm}")

        # --- 1. Ranking-transition points (normalized space) used as the KAN-side
        #        transition points for the dual measure. ---
        tr_pts_norm = {idx: (transition_points_per_input[idx] or [])
                         for idx in top2_dual}

        # --- 1b. Figure: activation phi(x) with its analytical phi'(x) ---
        no_l = act.coef.shape[1]
        with plt.rc_context({'figure.autolayout': True}):
            fig_d, axs_d = plt.subplots(no_l, len(top2_dual), squeeze=False,
                                        figsize=(5 * len(top2_dual), 3 * no_l))
            for col, i_feat in enumerate(top2_dual):
                knots_i = data_range_knots(act, i_feat).cpu().detach().numpy()
                x_sweep = np.linspace(float(knots_i.min()), float(knots_i.max()), 400)
                x_sweep_raw = denorm(x_sweep, i_feat, scaler_X, nx)  # raw x for display
                for j in range(no_l):
                    ax = axs_d[j, col]
                    ax2 = ax.twinx()
                    phi, dphi, _ = edge_curves(model, l, i_feat, j, x_sweep)
                    ln0 = ax.plot(x_sweep_raw, phi, color='#222222', lw=1.6, label=r'$\phi(x)$')
                    ln1 = ax2.plot(x_sweep_raw, dphi, color='#1f77b4', lw=1.0, ls='--',
                                   label=r"$\phi'(x)$")
                    ax2.axhline(0, color='gray', lw=0.6, alpha=0.6)
                    extra_handles = []
                    # orange solid: this FEATURE's ranking-transition points (where the
                    # local sensitivity |phi'| drops below tau). Per-feature, so the same
                    # vline is drawn on every edge (node j) of input i_feat.
                    first_t = True
                    for tp in (tr_pts_norm[i_feat] or []):
                        h = ax.axvline(denorm([tp], i_feat, scaler_X, nx)[0], color='darkorange', ls='-',
                                       alpha=0.85, lw=1.3,
                                       label='ranking transition' if first_t else '_')
                        if first_t:
                            extra_handles.append(h)
                        first_t = False
                    ax.set_xlabel(f"{feat_names[i_feat]}")
                    ax.set_ylabel(r'$\phi$')
                    ax2.set_ylabel(r"$\phi'$")
                    sym_tag = (f"  [symbolic: {sym_info[(i_feat, j)]}]"
                               if (i_feat, j) in sym_info else "")
                    ax.set_title(f"edge ({feat_names[i_feat]} -> node {j}){sym_tag}")
                    lns = ln0 + ln1 + extra_handles
                    ax.legend(lns, [ln.get_label() for ln in lns], loc='best', fontsize=7)
            fig_d.suptitle(f"{data_name} — activation & first derivative (L0)",
                           fontsize=11, fontweight='bold')
            for ext in ['.png', '.svg', '.eps']:
                fig_d.savefig(os.path.join(savepath,
                              f"{tag}{data_name}_activation_derivatives_L0{ext}"))
            plt.close(fig_d)
        print(f"🧭 Activation+derivatives figure saved: "
              f"{tag}{data_name}_activation_derivatives_L0.(png/svg/eps)")

        # --- 2. Denormalize to raw space (analysis band filter, as in 3.7/3.8) ---
        def _denorm_feat(vals_norm, feat_idx):
            return denorm(vals_norm, feat_idx, scaler_X, nx, band=(0.05, 0.95))

        tr_pts_raw = {idx: _denorm_feat(tr_pts_norm[idx], idx) for idx in top2_dual}

        # --- 3. Single source of truth: ranking-transition section edges (raw) ---
        # custom_edges[feat] = [lo, <interior ranking transitions>, hi]. The interior
        # of this array is reused verbatim for (a) AGSM custom_edges, (b) KAN
        # attribution interval masks, and (c) the plotted vlines -> fully traceable.
        custom_edges = {}
        for idx in top2_dual:
            lo_raw, hi_raw = bounds[idx]
            interior = sorted(v for v in tr_pts_raw[idx] if lo_raw < v < hi_raw)
            custom_edges[idx] = np.array([lo_raw] + interior + [hi_raw], dtype=float)

        # --- 4. KAN surrogate batch func (raw space), as in section 3.8 ---
        _kan_batch_func_dual = kan_batch_func(model, scaler_X, scaler_y, device)

        # --- 5. AGSM S_a over the ranking-transition–segmented intervals ---
        sc, sh, sa, r = compute_gradient_agsm(
            func_batch=_kan_batch_func_dual, bounds=bounds, feat_names=feat_names,
            top2_idx=top2_dual, n_sections=10, n_samples_per_section=512, seed=42,
            section_mode='custom', custom_edges=custom_edges,
        )
        dual_agsm_tps = find_agsm_transition_points(
            sc[ci_idx], sa[ci_idx], sc[cj_idx], sa[cj_idx],
            feat_names[ci_idx], feat_names[cj_idx],
        )

        # --- 6. Per-interval KAN attribution over the SAME intervals (3.8 pattern) ---
        dual_kan_attr = {}
        for feat_idx in top2_dual:
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
            dual_kan_attr[feat_idx] = np.array(attr_sections)  # (n_eff, nx)

        # --- 7. CSV ---
        rows = []
        for feat_idx in top2_dual:
            for kk, (center, s_hat, s_a_v, r_v) in enumerate(zip(
                    sc[feat_idx], sh[feat_idx], sa[feat_idx], r[feat_idx])):
                rows.append({'Feature': feat_names[feat_idx], 'Feature_idx': feat_idx,
                             'Section_k': kk, 'Section_center': center,
                             'S_hat': s_hat, 'S_a': s_a_v, 'R': r_v})
        pd.DataFrame(rows).to_csv(
            os.path.join(savepath, f"{tag}{data_name}_transition_dual_measure.csv"), index=False)

        # --- 8. Dual-panel figure: AGSM S_a (left) | KAN attribution (right) ---
        # vlines come from custom_edges[ci_idx] interior (the same array driving
        # AGSM sectioning and the ci_idx attribution masks).
        ip_lines = list(custom_edges[ci_idx][1:-1])
        feat_colors_dual = ['#1f77b4', '#d62728']

        def _seg_edges(feat_idx, centers):
            """Section edges for piecewise-constant plotting (ranking-transition edges)."""
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

            for color, feat_idx in zip(feat_colors_dual, top2_dual):
                _step_over_edges(ax_l, _seg_edges(feat_idx, sc[feat_idx]),
                                 sa[feat_idx], color=color, label=feat_names[feat_idx])
            first = True
            for ip in ip_lines:
                ax_l.axvline(ip, color='green', linestyle='--', alpha=0.7, linewidth=1.0,
                             label='ranking transition' if first else '_')
                first = False
            first = True
            for tp in dual_agsm_tps:
                ax_l.axvline(tp['point'], color='orange', linestyle=':', alpha=0.8,
                             linewidth=1.2, label='AGSM transition' if first else '_')
                first = False
            ax_l.set_xlabel(feat_names[ci_idx])
            ax_l.set_ylabel(r'$S^a_{l,[k]}$')
            ax_l.set_title('AGSM (transition-segmented)')
            ax_l.legend(loc='best')

            attr_mat = dual_kan_attr[ci_idx]
            edges_i = _seg_edges(ci_idx, sc[ci_idx])
            for color, feat_idx in zip(feat_colors_dual, top2_dual):
                # Piecewise-constant over the SAME ranking-transition edges, so the
                # steps change exactly at the transition vlines.
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
                fig_cv.savefig(os.path.join(savepath, f"{tag}{data_name}_transition_dual_measure{ext}"))
            plt.close(fig_cv)

        print(f"🧭 Ranking-transition pts (normalized): {tr_pts_norm}")
        print(f"🧭 Ranking-transition pts (raw): {tr_pts_raw}")
        print(f"🧭 AGSM transitions: {[round(t['point'], 3) for t in dual_agsm_tps]}")
        print(f"🧭 Saved: {tag}{data_name}_transition_dual_measure.(png/svg/eps/csv)")

    except Exception as e:
        import traceback
        print(f"⚠️ Curvature section 3.9 failed: {e}")
        traceback.print_exc()


# ==========================================================================
# §4 / §4.5 scores_interval (+ gallery + split-data). MODEL-ONLY.
# ==========================================================================
def scores_interval(model, dataset, scaler_X, scaler_y, nx, feat_names, scores_tot,
                    transition_points_per_input, savepath, data_name,
                    tag='', model_mode='as-saved'):
    """§4/§4.5: per-interval KAN attribution bars per feature + split-data pkl + CSV."""
    n_features = scores_tot.shape[0]
    sorted_feat_indices = np.argsort(scores_tot)[::-1]

    def _build_interval_masks(feat_idx):
        """Build per-interval masks/labels for splitting on feat_idx's transition points.

        Returns (masks, labels, split_points) or None if the feature has no valid
        transition points (in 0.1~0.9) that yield >=2 active intervals.
        """
        raw_pts = transition_points_per_input[feat_idx]
        valid_pts = [ip for ip in raw_pts if ip is not None and 0.1 < ip < 0.9]
        unique_pts = sorted(list(set([round(ip, 3) for ip in valid_pts])))
        if len(unique_pts) == 0:
            return None
        # Intervals: [0.1, ip1, ip2, ..., 0.9]
        mask_interval = [0.1] + unique_pts + [0.9]
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
        fname = f"{tag}{data_name}_scores_interval_x{feat_idx}.png"
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
            gallery_dir, f"{tag}{data_name}{mode_tag}_scores_interval_x{feat_idx}.png")
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

    # --- 4.5 Save Range Split Data for NN Training ---
    import joblib
    split_data_savepath = os.path.join(savepath, f"{tag}{data_name}_range_split_data.pkl")

    split_data = {
        'dataset': dataset,
        'masks': masks,
        'labels': labels,
        'selected_mask_idx': selected_mask_idx,
        'selected_mask_name': feat_names[selected_mask_idx],
        'split_points': selected_split_points,  # interval boundaries in [0.1, 0.9] space
        'transition_points_per_input': transition_points_per_input,  # ranking-transition points (used downstream)
        'feature_names': feat_names,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }

    joblib.dump(split_data, split_data_savepath)

    # Save KAN interval scores as CSV for cross-method comparison
    kan_scores_df = pd.DataFrame(scores_interval_norm, columns=feat_names)
    kan_scores_df.insert(0, 'Interval_Label', labels)
    kan_scores_df.to_csv(os.path.join(savepath, f"{tag}{data_name}_kan_interval_scores.csv"), index=False)


# ==========================================================================
# §6 Saltelli global attribution. MODEL-ONLY.
# ==========================================================================
def saltelli_global(model, scaler_X, nx, bounds, feat_names, savepath, data_name,
                    tag='', device='cpu'):
    """§6: KAN global attribution scores on a Saltelli (Sobol) sample."""
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
    plt.savefig(os.path.join(savepath, f"{tag}{data_name}_scores_global_saltelli.png"), dpi=300)

    df_scores_saltelli = pd.DataFrame({
        'Feature': feat_names,
        'Global_Attribution_Score': scores_saltelli
    }).sort_values(by='Global_Attribution_Score', ascending=False)
    df_scores_saltelli.to_csv(
        os.path.join(savepath, f"{tag}{data_name}_global_attribution_scores_saltelli.csv"), index=False
    )
    print(f"📊 Saltelli attribution scores saved to: {savepath}")


# ==========================================================================
# Entry point: analyze_model. Runs the selected sections, each in its own
# try/except (mirroring toy's per-section guards), returns a results dict.
# ==========================================================================
DEFAULT_SECTIONS = (
    'input_vs_output',      # §2.5  (true_func-gated overlay)
    'ranking_transitions',  # §3    (compute) + §3.6 (figure)
    'activations',          # §3
    'attribution_trajectory',  # §3.5
    'contour',              # §3.7  (true_func-gated surface)
    'agsm_dual_measure',    # §3.8
    'transition_segmented_dual_measure',  # §3.9
    'scores_interval',      # §4 / §4.5
    'saltelli',             # §6
)


def analyze_model(model, *, scaler_X, scaler_y, feat_names, bounds, savepath,
                  X_norm, y_norm=None, X_raw=None, true_func=None, device='cpu',
                  tag='', data_name=None, model_mode='as-saved',
                  rel_thresh=0.2, sections=DEFAULT_SECTIONS):
    """Run the chosen analysis sections on a loaded KAN model.

    Parameters
    ----------
    model : MultKAN
        The (already mode-selected) KAN to analyze. A forward pass is run here.
    scaler_X, scaler_y : fitted sklearn scalers.
    feat_names : list[str]
    bounds : list of [lo, hi] (raw space), one per input.
    savepath : str  output directory (caller handles the kan_models/<mode> redirect).
    X_norm : np.ndarray [N, n]  normalized training inputs.
    y_norm : np.ndarray | None  normalized training labels (only used for plots).
    X_raw : np.ndarray | None   raw training inputs (needed for §3.8 quantile mode).
    true_func : callable | None per-row analytic function (gates §2.5 / §3.7).
    tag : str  filename prefix ("" for toy → unchanged filenames).
    data_name : str  base filename stem (defaults to tag-less if None — required).
    sections : iterable of section names to run (subset of DEFAULT_SECTIONS).

    Returns
    -------
    dict with at least ``transition_points_per_input`` and ``scores_tot``.
    """
    if data_name is None:
        raise ValueError("analyze_model requires data_name (filename stem).")
    nx = len(bounds)
    sections = set(sections)
    plt.rcParams.update(SA_RC)

    X_train = np.asarray(X_raw) if X_raw is not None else None
    # y_true (raw) for the §2.5 ground-truth overlay: regenerate via true_func if
    # available and we have raw inputs; else None (prediction-only scatter).
    y_true = None
    if true_func is not None and X_train is not None:
        y_true = np.apply_along_axis(true_func, 1, X_train).reshape(-1, 1)

    dataset = {
        'train_input': torch.tensor(np.asarray(X_norm), dtype=torch.float32, device=device),
    }
    if y_norm is not None:
        dataset['train_label'] = torch.tensor(np.asarray(y_norm), dtype=torch.float32,
                                               device=device).reshape(-1, 1)

    results = {}

    # §2.5 input vs output (true_func-gated overlay) — runs before the heavy
    # forward passes, mirroring toy's ordering.
    if 'input_vs_output' in sections and X_train is not None:
        try:
            plot_input_vs_output(model, dataset, scaler_y, X_train, feat_names,
                                 savepath, data_name, tag=tag, y_true=y_true)
        except Exception as e:
            import traceback
            print(f"⚠️ §2.5 input-vs-output failed: {e}")
            traceback.print_exc()

    # Run forward pass once to populate internals (splines, activations) and the
    # global feature score, exactly as toy does before §3.
    with plt.rc_context({'figure.autolayout': False}):
        try:
            model.plot()
            plt.savefig(os.path.join(savepath, f"{tag}{data_name}_model.png"))
            plt.close()
        except Exception:
            plt.close()
    model.forward(dataset['train_input'])
    scores_tot = model.feature_score.detach().cpu().numpy()  # Global scores
    results['scores_tot'] = scores_tot

    # §3 ranking transitions (single source) — needed by most downstream sections.
    transition_points_per_input, transitions, info, x_grid_rt = \
        compute_ranking_transitions(model, scaler_X, nx, rel_thresh=rel_thresh)
    results['transition_points_per_input'] = transition_points_per_input
    results['transitions'] = transitions
    results['info'] = info

    print("\n🔍 Analyzing activations & ranking transitions in Layer 0...")

    if 'activations' in sections:
        try:
            plot_activations(model, scaler_X, nx, feat_names, scores_tot,
                             transition_points_per_input, savepath, data_name, tag=tag)
        except Exception as e:
            import traceback
            print(f"⚠️ §3 activations failed: {e}")
            traceback.print_exc()

    if 'ranking_transitions' in sections:
        # §3.6 figure (the §3 computation already ran above unconditionally).
        plot_ranking_transition_figure(model, feat_names, transitions, info, x_grid_rt,
                                        rel_thresh, savepath, data_name, tag=tag)

    if 'attribution_trajectory' in sections:
        try:
            plot_attribution_trajectory(model, dataset, scaler_X, nx, feat_names, scores_tot,
                                        transition_points_per_input, savepath, data_name, tag=tag)
        except Exception as e:
            import traceback
            print(f"⚠️ §3.5 attribution trajectory failed: {e}")
            traceback.print_exc()

    if 'contour' in sections:
        try:
            plot_contour(model, scaler_X, scaler_y, nx, bounds, feat_names, scores_tot,
                         transition_points_per_input, savepath, data_name,
                         tag=tag, true_func=true_func, device=device)
        except Exception as e:
            import traceback
            print(f"⚠️ §3.7 contour failed: {e}")
            traceback.print_exc()

    if 'agsm_dual_measure' in sections:
        agsm_dual_measure(model, dataset, scaler_X, scaler_y, nx, bounds, feat_names,
                          scores_tot, transition_points_per_input, savepath, data_name,
                          X_train=X_train, tag=tag, device=device)

    if 'transition_segmented_dual_measure' in sections:
        transition_segmented_dual_measure(model, dataset, scaler_X, scaler_y, nx, bounds,
                                           feat_names, scores_tot,
                                           transition_points_per_input, savepath, data_name,
                                           tag=tag, device=device)

    if 'scores_interval' in sections:
        try:
            scores_interval(model, dataset, scaler_X, scaler_y, nx, feat_names, scores_tot,
                            transition_points_per_input, savepath, data_name,
                            tag=tag, model_mode=model_mode)
        except Exception as e:
            import traceback
            print(f"⚠️ §4 scores_interval failed: {e}")
            traceback.print_exc()

    if 'saltelli' in sections:
        try:
            saltelli_global(model, scaler_X, nx, bounds, feat_names, savepath, data_name,
                            tag=tag, device=device)
        except Exception as e:
            import traceback
            print(f"⚠️ §6 Saltelli failed: {e}")
            traceback.print_exc()

    return results
