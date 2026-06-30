"""KAN-vs-MLP local-sensitivity comparison (Phase 3 of the MLP baseline).

Loads a trained KAN and a trained sklearn ``MLPRegressor`` for the same toy
function, computes each model's **local sensitivity trajectory** ``s_i(x)`` and
its **ranking-transition points** (τ-crossings) on the SAME normalized sweep
(``x_grid``) and SAME ``rel_thresh``, and compares them:

  1. Overlay figure ``kan_vs_mlp_sensitivity_{func}.{png,svg,eps}`` — per input
     feature, each method's ``s_i`` normalized to its OWN max (so shapes live in
     [0,1] and the τ threshold becomes a single horizontal line at
     ``rel_thresh``). KAN ``s_i`` = smooth solid line; MLP midpoint ``s_i`` =
     dashed line + MC ±σ band (same own-max normalization). Vertical transition
     lines for each method (denormalized to raw x). x-axis is raw.

  2. Transition-alignment figure ``kan_vs_mlp_transition_alignment_{func}.{png,
     svg,eps}`` + ``.csv`` — per feature, both methods' transition points are
     denormalized to raw and matched with
     ``robustness_symbolify._match_transitions`` at ``tol = 0.05 * domain width``.
     Summary panel mirrors ``build_symbolify_robustness_paper.make_figure``
     (per-feature count bars + |drift| panel). CSV one row per transition point
     (matched / added / dropped) + a true-kink reference row.

Reuses (does NOT duplicate): ``kan_analysis_core`` (compute_ranking_transitions,
denorm, SA_RC), ``nn_sensitivity`` (mlp_predict_norm, mlp_feature_sensitivity,
mlp_ranking_transitions), ``robustness_symbolify._match_transitions``,
``toy_KAN_sweep`` (KANRegressor, FUNCTION_ZOO), ``bspline_curvature.data_range_knots``.

Run from D:\\pykan, pykan-new env, PYTHONPATH=. PYTHONUTF8=1 PYTHONIOENCODING=utf-8:
    python -m github.workflows.Hyein.compare_kan_vs_mlp conditional
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from github.workflows.Hyein.kan_analysis_core import (
    compute_ranking_transitions, denorm, SA_RC,
)
from github.workflows.Hyein.nn_sensitivity import (
    mlp_predict_norm, mlp_feature_sensitivity, mlp_ranking_transitions,
)
from github.workflows.Hyein.robustness_symbolify import _match_transitions
from github.workflows.Hyein.toy_KAN_sweep import KANRegressor, FUNCTION_ZOO

# ----------------------------------------------------------------------------
# Paths / style
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
AR = os.path.join(HERE, "analytical_results")
FIG = os.path.join(HERE, "figures_for_paper")

# Method colors / styles (KAN = blue solid, MLP = orange dashed).
C_KAN = "#3B6FB6"
C_MLP = "#E08214"
EXTS = (".png", ".svg", ".eps")

# True kink reference per function (raw x location where the response changes).
# Only the conditional function (x[0] < 0) has an analytic kink, at x0 = 0.
TRUE_KINK = {"conditional": {"feat_idx": 0, "x_raw": 0.0}}


def _save(fig, base):
    for ext in EXTS:
        fig.savefig(base + ext)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Load both models for one function
# ----------------------------------------------------------------------------
def _load_models(func_name, device="cpu"):
    kan_dir = os.path.join(AR, func_name, "kan_models")
    nn_dir = os.path.join(AR, func_name, "nn_models")

    kan_reg = KANRegressor(device=device).load_model(
        os.path.join(kan_dir, f"{func_name}_best_kan_model"))
    kan_model = kan_reg.model
    scaler_X_kan = joblib.load(os.path.join(kan_dir, f"{func_name}_scaler_X.pkl"))
    scaler_y_kan = joblib.load(os.path.join(kan_dir, f"{func_name}_scaler_y.pkl"))

    mlp = joblib.load(os.path.join(nn_dir, f"{func_name}_best_mlp_model.pkl"))
    scaler_X_mlp = joblib.load(os.path.join(nn_dir, f"{func_name}_mlp_scaler_X.pkl"))
    scaler_y_mlp = joblib.load(os.path.join(nn_dir, f"{func_name}_mlp_scaler_y.pkl"))

    return {
        "kan_model": kan_model, "scaler_X_kan": scaler_X_kan,
        "scaler_y_kan": scaler_y_kan, "mlp": mlp,
        "scaler_X_mlp": scaler_X_mlp, "scaler_y_mlp": scaler_y_mlp,
    }


# ----------------------------------------------------------------------------
# Compute KAN + MLP sensitivities/transitions on the SAME grid & rel_thresh
# ----------------------------------------------------------------------------
def compute_comparison(func_name, rel_thresh=0.2, device="cpu"):
    cfg = FUNCTION_ZOO[func_name]
    bounds = cfg["bounds"]
    feat_names = cfg["names"]
    nx = len(bounds)

    M = _load_models(func_name, device=device)
    kan_model = M["kan_model"]
    scaler_X_kan = M["scaler_X_kan"]
    scaler_X_mlp = M["scaler_X_mlp"]

    # --- KAN: s_i (normalized) + transitions; x_grid is the normalized sweep. ---
    kan_tpp, kan_trans, kan_info, x_grid = compute_ranking_transitions(
        kan_model, scaler_X_kan, nx, rel_thresh=rel_thresh)
    if x_grid is None or kan_info is None:
        raise RuntimeError("KAN ranking-transition computation returned no grid/info.")
    kan_S = kan_info["S"]          # {feat: s_i array (normalized)}
    kan_tau = kan_info["tau"]

    # --- MLP: use THE SAME x_grid (comparability). ---
    predict_norm = mlp_predict_norm(M["mlp"])
    mlp_tpp, mlp_trans, mlp_info = mlp_ranking_transitions(
        predict_norm, x_grid, nx, rel_thresh=rel_thresh, mode="mc")
    mlp_S = mlp_info["S"]          # MC mean per feat (normalized space)
    mlp_S_std = mlp_info["S_std"]  # MC std band per feat
    mlp_tau = mlp_info["tau"]

    # Clean midpoint trajectory per feature (others fixed at 0.5).
    mlp_S_mid = {i: mlp_feature_sensitivity(predict_norm, x_grid, i, nx,
                                            mode="midpoint")
                 for i in range(nx)}

    return {
        "func": func_name, "bounds": bounds, "feat_names": feat_names, "nx": nx,
        "x_grid": x_grid, "rel_thresh": rel_thresh,
        "scaler_X_kan": scaler_X_kan, "scaler_X_mlp": scaler_X_mlp,
        "kan_S": kan_S, "kan_tau": kan_tau, "kan_tpp": kan_tpp,
        "mlp_S": mlp_S, "mlp_S_std": mlp_S_std, "mlp_S_mid": mlp_S_mid,
        "mlp_tau": mlp_tau, "mlp_tpp": mlp_tpp,
    }


def _own_max_norm(arr):
    """Normalize an array to its own max (so the τ=rel_thresh line is universal)."""
    arr = np.asarray(arr, dtype=float)
    mx = float(np.nanmax(arr)) if np.any(np.isfinite(arr)) else 0.0
    if mx <= 0:
        return arr * 0.0, 0.0
    return arr / mx, mx


# ----------------------------------------------------------------------------
# Figure 1: overlay s_i(x) (own-max normalized) + transition vlines
# ----------------------------------------------------------------------------
def make_overlay_figure(R):
    func = R["func"]
    nx = R["nx"]
    feat_names = R["feat_names"]
    x_grid = R["x_grid"]
    rel_thresh = R["rel_thresh"]
    bounds = R["bounds"]

    with plt.rc_context(SA_RC):
        fig, axes = plt.subplots(1, nx, figsize=(4.2 * nx, 3.2), squeeze=False)
        axes = axes[0]
        for i in range(nx):
            ax = axes[i]
            # Raw x for this feature's sweep (each method uses its own scaler).
            x_raw_kan = denorm(x_grid, i, R["scaler_X_kan"], nx)
            x_raw_mlp = denorm(x_grid, i, R["scaler_X_mlp"], nx)

            # --- KAN s_i, normalized to its own max -> smooth solid line. ---
            kan_n, _ = _own_max_norm(R["kan_S"][i])
            ax.plot(x_raw_kan, kan_n, color=C_KAN, lw=1.6, ls="-",
                    label="KAN $s_i$ (own-max)")

            # --- MLP midpoint s_i, own-max normalized -> dashed line. The MC band
            #     is scaled by the SAME max so band & line share a scale. ---
            mlp_mid_n, mid_max = _own_max_norm(R["mlp_S_mid"][i])
            ax.plot(x_raw_mlp, mlp_mid_n, color=C_MLP, lw=1.4, ls="--",
                    label="MLP $s_i$ midpoint (own-max)")
            # MC mean ± std band, normalized by the MC mean's own max.
            mc_mean = np.asarray(R["mlp_S"][i], dtype=float)
            mc_std = np.asarray(R["mlp_S_std"][i], dtype=float)
            mc_max = float(np.nanmax(mc_mean)) if np.any(np.isfinite(mc_mean)) else 0.0
            if mc_max > 0:
                lo = (mc_mean - mc_std) / mc_max
                hi = (mc_mean + mc_std) / mc_max
                ax.fill_between(x_raw_mlp, lo, hi, color=C_MLP, alpha=0.18,
                                lw=0, label="MLP MC $\\pm\\sigma$ band")

            # --- τ line at rel_thresh on the own-max scale. ---
            ax.axhline(rel_thresh, color="black", ls=":", lw=1.0, alpha=0.7,
                       label=f"$\\tau={rel_thresh:g}$ (rel.)")

            # --- Transition vlines (raw x). ---
            first_k = True
            for tp in (R["kan_tpp"][i] or []):
                xr = denorm([tp], i, R["scaler_X_kan"], nx)[0]
                ax.axvline(xr, color=C_KAN, ls="-", lw=1.2, alpha=0.85,
                           label="KAN transition" if first_k else "_")
                first_k = False
            first_m = True
            for tp in (R["mlp_tpp"][i] or []):
                xr = denorm([tp], i, R["scaler_X_mlp"], nx)[0]
                ax.axvline(xr, color=C_MLP, ls="--", lw=1.2, alpha=0.85,
                           label="MLP transition" if first_m else "_")
                first_m = False

            # --- True kink reference (as a legend entry, NOT a text box). ---
            tk = TRUE_KINK.get(func)
            if tk is not None and tk["feat_idx"] == i:
                ax.axvline(tk["x_raw"], color="#444444", ls="-.", lw=1.1, alpha=0.7,
                           label=f"true kink ($x_0={tk['x_raw']:g}$)")

            ax.set_xlabel(feat_names[i])
            ax.set_ylabel("local sensitivity $s_i$ (own-max norm.)")
            ax.set_xlim(bounds[i][0], bounds[i][1])
            ax.set_ylim(0, 1.08)
            ax.legend(loc="best")

        base = os.path.join(FIG, f"kan_vs_mlp_sensitivity_{func}")
        _save(fig, base)
    print(f"[fig1] overlay saved: {base}.(png/svg/eps)")
    return base


# ----------------------------------------------------------------------------
# Transition alignment (per feature) + CSV
# ----------------------------------------------------------------------------
def align_transitions(R):
    func = R["func"]
    nx = R["nx"]
    feat_names = R["feat_names"]
    bounds = R["bounds"]

    rows = []          # CSV rows
    per_feat = []      # summary records for the figure
    for i in range(nx):
        lo, hi = float(bounds[i][0]), float(bounds[i][1])
        width = hi - lo
        tol = 0.05 * width

        kan_pts_raw = sorted(denorm(R["kan_tpp"][i] or [], i, R["scaler_X_kan"], nx))
        mlp_pts_raw = sorted(denorm(R["mlp_tpp"][i] or [], i, R["scaler_X_mlp"], nx))
        kan_pts_raw = [float(v) for v in kan_pts_raw]
        mlp_pts_raw = [float(v) for v in mlp_pts_raw]

        # _match_transitions(spline, sym, tol): here spline<-KAN, sym<-MLP, so
        # drift = mlp - kan, added = MLP-only, dropped = KAN-only.
        matches, added, dropped = _match_transitions(kan_pts_raw, mlp_pts_raw, tol)

        drifts = [d for (_, _, d) in matches]
        abs_drifts = [abs(d) for d in drifts]
        per_feat.append({
            "feat_idx": i, "feat_name": feat_names[i], "width": width,
            "n_kan": len(kan_pts_raw), "n_mlp": len(mlp_pts_raw),
            "matched": len(matches), "added": len(added), "dropped": len(dropped),
            "median_drift_pct": (100.0 * np.median(abs_drifts) / width) if abs_drifts else np.nan,
            "max_drift_pct": (100.0 * np.max(abs_drifts) / width) if abs_drifts else np.nan,
        })

        for (kp, mp, d) in matches:
            rows.append({"feature": feat_names[i], "feat_idx": i,
                         "kan_point_raw": kp, "mlp_point_raw": mp,
                         "status": "matched", "drift": d,
                         "drift_frac": d / width})
        for ap in added:
            rows.append({"feature": feat_names[i], "feat_idx": i,
                         "kan_point_raw": np.nan, "mlp_point_raw": ap,
                         "status": "added", "drift": np.nan, "drift_frac": np.nan})
        for dp in dropped:
            rows.append({"feature": feat_names[i], "feat_idx": i,
                         "kan_point_raw": dp, "mlp_point_raw": np.nan,
                         "status": "dropped", "drift": np.nan, "drift_frac": np.nan})

    # True-kink reference row (annotation row in the CSV, not a text box).
    tk = TRUE_KINK.get(func)
    if tk is not None:
        rows.append({"feature": feat_names[tk["feat_idx"]], "feat_idx": tk["feat_idx"],
                     "kan_point_raw": np.nan, "mlp_point_raw": np.nan,
                     "status": "true_kink_reference", "drift": np.nan,
                     "drift_frac": np.nan, "x_raw": tk["x_raw"]})

    df = pd.DataFrame(rows)
    csv_path = os.path.join(FIG, f"kan_vs_mlp_transition_alignment_{func}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[align] CSV saved: {csv_path}")
    return df, per_feat, csv_path


# ----------------------------------------------------------------------------
# Figure 2: alignment summary (count bars + drift panel), make_figure style
# ----------------------------------------------------------------------------
def make_alignment_figure(R, per_feat):
    func = R["func"]
    labels = [pf["feat_name"] for pf in per_feat]
    x = np.arange(len(per_feat))
    w = 0.38

    n_kan = [pf["n_kan"] for pf in per_feat]
    n_mlp = [pf["n_mlp"] for pf in per_feat]
    med = np.array([pf["median_drift_pct"] for pf in per_feat], dtype=float)
    mx = np.array([pf["max_drift_pct"] for pf in per_feat], dtype=float)
    has_match = np.array([pf["matched"] > 0 for pf in per_feat])

    C_DRIFT = "#2C8C5A"
    C_FLAG = "#C0392B"

    with plt.rc_context(SA_RC):
        fig, (ax, axd) = plt.subplots(
            2, 1, figsize=(max(5.5, 2.4 * len(per_feat)), 6.0), sharex=True,
            gridspec_kw={"height_ratios": [2.4, 1.0], "hspace": 0.12})

        b1 = ax.bar(x - w / 2, n_kan, w, label="KAN ($n_{KAN}$)",
                    color=C_KAN, edgecolor="#222", linewidth=0.5)
        b2 = ax.bar(x + w / 2, n_mlp, w, label="MLP ($n_{MLP}$)",
                    color=C_MLP, edgecolor="#222", linewidth=0.5)
        ax.bar_label(b1, padding=2, fontsize=8)
        ax.bar_label(b2, padding=2, fontsize=8)
        ax.set_ylabel("# ranking transitions")
        ax.set_ylim(0, max(max(n_kan, default=0), max(n_mlp, default=0), 1) * 1.3)
        ax.legend(loc="upper right")

        ymax = ax.get_ylim()[1]
        for i, pf in enumerate(per_feat):
            txt = f"m={pf['matched']} +{pf['added']} -{pf['dropped']}"
            ax.annotate(txt, (i, -ymax * 0.015), ha="center", va="top",
                        fontsize=7, color="#555", annotation_clip=False)

        # Bottom: matched-transition |drift| as % of domain width.
        for i in range(len(per_feat)):
            if has_match[i] and np.isfinite(mx[i]):
                axd.plot([x[i], x[i]], [med[i], mx[i]], color=C_DRIFT, lw=1.2,
                         zorder=2, alpha=0.6)
                axd.plot(x[i], mx[i], marker="_", color=C_DRIFT, ms=12, mew=1.4, zorder=3)
                axd.plot(x[i], med[i], marker="o", color=C_DRIFT, ms=7,
                         mec="#1c5c39", zorder=4)
                axd.annotate(f"{med[i]:.2f}%", (x[i], med[i]), xytext=(7, 0),
                             textcoords="offset points", va="center",
                             fontsize=7, color="#1c5c39")
            else:
                axd.annotate("no match", (x[i], 0.0), ha="center", va="bottom",
                             fontsize=7, color=C_FLAG, weight="600")
                axd.plot(x[i], 0.0, marker="x", color=C_FLAG, ms=8, mew=1.6, zorder=4)

        axd.set_ylabel("|drift|\n(% domain width)")
        axd.set_xlabel("input feature")
        finite_mx = mx[np.isfinite(mx)]
        top = (max(finite_mx.max() if finite_mx.size else 1.0, 1.0)) * 1.55
        axd.set_ylim(0, top)
        axd.axhline(5.0, ls="--", lw=0.9, color="#888", zorder=1)
        axd.set_xticks(x)
        axd.set_xticklabels(labels, rotation=20, ha="right")

        leg = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=C_DRIFT,
                   markersize=7, label="median |drift| (matched)"),
            Line2D([0], [0], marker="_", color=C_DRIFT, markersize=12, lw=0,
                   label="max |drift|"),
            Line2D([0], [0], ls="--", color="#888", label="5% domain width"),
            Line2D([0], [0], marker="x", color=C_FLAG, lw=0, markersize=8,
                   label="no matched transition"),
        ]
        axd.legend(handles=leg, loc="upper center", ncol=2, fontsize=7)

        base = os.path.join(FIG, f"kan_vs_mlp_transition_alignment_{func}")
        _save(fig, base)
    print(f"[fig2] alignment summary saved: {base}.(png/svg/eps)")
    return base


# ----------------------------------------------------------------------------
# Console comparison report
# ----------------------------------------------------------------------------
def print_report(R, per_feat):
    func = R["func"]
    nx = R["nx"]
    tk = TRUE_KINK.get(func)
    print("\n================ KAN vs MLP comparison ================")
    print(f"function = {func} | rel_thresh = {R['rel_thresh']} | "
          f"x_grid: n={R['x_grid'].size}, [{R['x_grid'].min():.4f}, {R['x_grid'].max():.4f}]")
    for i in range(nx):
        name = R["feat_names"][i]
        kan_raw = sorted(denorm(R["kan_tpp"][i] or [], i, R["scaler_X_kan"], nx))
        mlp_raw = sorted(denorm(R["mlp_tpp"][i] or [], i, R["scaler_X_mlp"], nx))
        # MC band mean-σ (noise proxy): mean ratio of σ to mean over the sweep.
        mean = np.asarray(R["mlp_S"][i], dtype=float)
        std = np.asarray(R["mlp_S_std"][i], dtype=float)
        m = np.nanmean(mean) if mean.size else np.nan
        s = np.nanmean(std) if std.size else np.nan
        rel_noise = (s / m) if (m and np.isfinite(m) and m > 0) else np.nan
        pf = per_feat[i]
        print(f"\n  [{name}]")
        print(f"    KAN transitions (raw): {[round(v, 4) for v in kan_raw]}")
        print(f"    MLP transitions (raw): {[round(v, 4) for v in mlp_raw]}")
        print(f"    matched={pf['matched']} added(MLP-only)={pf['added']} "
              f"dropped(KAN-only)={pf['dropped']} | "
              f"median|drift|={pf['median_drift_pct']:.3f}% "
              f"max|drift|={pf['max_drift_pct']:.3f}% of domain")
        print(f"    MLP MC band noise proxy: mean(s)={m:.4g}, mean(sigma)={s:.4g}, "
              f"sigma/mean={rel_noise:.3f}")
        if tk is not None and tk["feat_idx"] == i:
            xk = tk["x_raw"]
            dk = min((abs(v - xk) for v in kan_raw), default=np.nan)
            dm = min((abs(v - xk) for v in mlp_raw), default=np.nan)
            print(f"    true kink x0={xk:g}: nearest KAN={dk:.4f}, nearest MLP={dm:.4f}")
    print("=======================================================\n")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compare KAN vs MLP local sensitivity + transition points.")
    parser.add_argument("func_name", type=str, nargs="?", default="conditional",
                        choices=list(FUNCTION_ZOO.keys()),
                        help="Function from the ZOO (default: conditional).")
    parser.add_argument("--rel_thresh", type=float, default=0.2,
                        help="Relative tau threshold (default 0.2, same for both).")
    args = parser.parse_args()

    os.makedirs(FIG, exist_ok=True)

    R = compute_comparison(args.func_name, rel_thresh=args.rel_thresh)
    make_overlay_figure(R)
    df, per_feat, _ = align_transitions(R)
    make_alignment_figure(R, per_feat)
    print_report(R, per_feat)


if __name__ == "__main__":
    main()
