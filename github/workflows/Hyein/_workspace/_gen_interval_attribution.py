"""Per-transition-point-set attribution figures for the KAN-vs-AGSM project.

For each dataset (exponential, logarithm, log2, rosenbrock) TWO 1x4-panel
figures are produced. The 4 panels of BOTH figures correspond to the 4
transition-point SETS that define the interval splits:

    Panel 1: KAN transition points
    Panel 2: AGSM equal-distance transition points
    Panel 3: AGSM quantile transition points
    Panel 4: AGSM kan-grid transition points

Only the interval boundaries differ between panels; the measured quantity is
fixed within a figure.

Series 1 -- file stem ``{name}_attr_by_transition_KANscore``:
    bar height = KAN normalized attribution score, the SAME quantity computed
    in toy_KAN_analyze.py (forward-pass each NORMALIZED-space slice, read
    ``model.feature_score``, divide by the per-feature in-slice std). Bars
    grouped by ALL features. Splits operate in KAN-normalized space.

Series 2 -- file stem ``{name}_attr_by_transition_Sa``:
    bar height = AGSM ``S_a`` (cross-input comparable), recomputed over the
    subdomains defined by that panel's transition-point set, with the analytic
    function in RAW input space. Bars grouped by the top-2 features.

Grouped-bar style mirrors toy_KAN_analyze.py lines 773-808 (SA_RC rc dict,
FEAT_COLORS, width-0.2 offset pattern). Outputs (per dataset, per series, x3
formats) under figures_for_paper/:
    {name}_attr_by_transition_KANscore.{png,svg,eps}
    {name}_attr_by_transition_Sa.{png,svg,eps}

Run (from D:\\pykan):
    set PYTHONIOENCODING=utf-8 && set PYTHONUTF8=1 && \
    C:\\Users\\user\\miniconda3\\envs\\pykan-new\\python.exe -m \
    github.workflows.Hyein._workspace._gen_interval_attribution
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# Allow `python -m ...` from D:\pykan; also be importable directly.
_PROJ = r"D:\pykan"
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

from github.workflows.Hyein.toy_KAN_sweep import KANRegressor, FUNCTION_ZOO
from github.workflows.Hyein.sectional_gsa import (
    compute_gradient_agsm, find_agsm_transition_points, make_batch_func,
)

BASE = r"D:\pykan\github\workflows\Hyein"
RESULTS = os.path.join(BASE, "analytical_results")
FIGDIR = os.path.join(BASE, "figures_for_paper")

DATASETS = ["exponential", "logarithm", "log2", "rosenbrock"]
AGSM_MODES = ["equal", "quantile", "kan"]

# 4 transition-point-set panels (shared by both series).
PANEL_KEYS = ["kan", "equal", "quantile", "kan_grid"]
PANEL_TITLES = {
    "kan": "KAN transition pts",
    "equal": "AGSM equal",
    "quantile": "AGSM quantile",
    "kan_grid": "AGSM kan-grid",
}
# Map a panel key onto the AGSM CSV mode it draws its transition points from.
PANEL_AGSM_MODE = {"equal": "equal", "quantile": "quantile", "kan_grid": "kan"}

WIDTH = 0.2

# Drop transition points that land within this fraction of the domain width of
# an outer edge (otherwise a crossing on the boundary spawns a degenerate
# zero-width interval, e.g. log2 kan-grid at the lower bound).
_EDGE_MARGIN_FRAC = 0.01

SA_RC = {
    "figure.dpi": 150, "figure.facecolor": "white",
    "axes.facecolor": "white", "axes.edgecolor": "#444444", "axes.linewidth": 0.8,
    "axes.labelsize": 11, "axes.labelcolor": "black", "axes.grid": False,
    "xtick.labelsize": 9, "xtick.color": "black", "xtick.direction": "out",
    "ytick.labelsize": 9, "ytick.color": "black", "ytick.direction": "out",
    "font.family": "sans-serif", "font.size": 9, "font.weight": "300",
    "axes.labelweight": "500", "text.color": "black",
    "legend.fontsize": 6.5, "legend.framealpha": 0.0, "lines.linewidth": 1.2,
    "savefig.dpi": 150, "savefig.bbox": "tight", "savefig.facecolor": "white",
}

FEAT_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def kan_dir(name):
    return os.path.join(RESULTS, name, "kan_models")


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_split_data(name):
    return joblib.load(os.path.join(kan_dir(name), f"{name}_range_split_data.pkl"))


def load_kan_model(name):
    """Load the trained KAN exactly as toy_KAN_analyze.py does."""
    ckpt_path = os.path.join(kan_dir(name), f"{name}_best_kan_model")
    wrapper = KANRegressor(device=_DEVICE)
    wrapper.load_model(ckpt_path)
    return wrapper.model


def agsm_feature_order(name, mode):
    """Return (order, feat_names) from an AGSM CSV, CSV row order = top-1 first."""
    df = pd.read_csv(os.path.join(kan_dir(name), f"{name}_agsm_sectional_{mode}.csv"))
    order, fnames = [], []
    for fidx, g in df.groupby("Feature_idx", sort=False):
        order.append(int(fidx))
        fnames.append(str(g["Feature"].iloc[0]))
    return order, fnames, df


def agsm_transition_points_raw(name, mode):
    """Recompute the raw-space AGSM S_a-curve crossings for a mode.

    Mirrors _gen_agsm_modes_comparison.py: read the per-mode CSV, build the
    top-2 S_a curves, run find_agsm_transition_points. Returns
    (investigated_feat_idx, top2_order, [raw crossing x-values]).
    """
    _, _, df = agsm_feature_order(name, mode)
    curves, order = {}, []
    for fidx, g in df.groupby("Feature_idx", sort=False):
        g = g.sort_values("Section_k")
        curves[int(fidx)] = (
            g["Section_center"].to_numpy(float),
            g["S_a"].to_numpy(float),
            str(g["Feature"].iloc[0]),
        )
        order.append(int(fidx))
    i_idx, j_idx = order[0], order[1]
    tps = find_agsm_transition_points(
        curves[i_idx][0], curves[i_idx][1],
        curves[j_idx][0], curves[j_idx][1],
        curves[i_idx][2], curves[j_idx][2],
    )
    return i_idx, order, [t["point"] for t in tps]


# ----------------------------------------------------------------------------
# Space conversion helpers (dummy-array trick)
# ----------------------------------------------------------------------------
def to_norm(scaler_X, nx, feat_idx, raw_vals):
    raw_vals = np.atleast_1d(np.asarray(raw_vals, float))
    if raw_vals.size == 0:
        return np.array([])
    dummy = np.zeros((raw_vals.size, nx))
    dummy[:, feat_idx] = raw_vals
    return scaler_X.transform(dummy)[:, feat_idx]


def to_raw(scaler_X, nx, feat_idx, norm_vals):
    norm_vals = np.atleast_1d(np.asarray(norm_vals, float))
    if norm_vals.size == 0:
        return np.array([])
    dummy = np.zeros((norm_vals.size, nx))
    dummy[:, feat_idx] = norm_vals
    return scaler_X.inverse_transform(dummy)[:, feat_idx]


# ----------------------------------------------------------------------------
# Series 1: KAN normalized attribution score per interval
# ----------------------------------------------------------------------------
def kan_scores_for_edges(model, train_input, nx, feat_idx, edges_norm):
    """For split edges (normalized) along feat_idx, KAN score[n_int, nx].

    Reproduces the masking + forward-pass + std-normalization in
    toy_KAN_analyze.py lines ~726-739. Empty intervals -> zeros.
    """
    col = train_input[:, feat_idx]
    edges = list(edges_norm)
    n_int = len(edges) - 1
    out = np.zeros((n_int, nx))
    for i in range(n_int):
        lb, ub = edges[i], edges[i + 1]
        if i == 0:
            mask = (col >= lb) & (col <= ub)
        else:
            mask = (col > lb) & (col <= ub)
        if torch.any(mask):
            x_slice = train_input[mask, :]
            x_std = torch.std(x_slice, dim=0).detach().cpu().numpy()
            model.forward(x_slice)
            score = model.feature_score.detach().cpu().numpy().copy()
            out[i] = score / (x_std + 1e-6)
        # else: leave zeros
    return out


def series1_panels(name):
    """Build the 4 KAN-score panels for one dataset.

    Returns (feat_names, list of panel dicts), each panel:
        {key, title, labels, values[n_int, nx], feat_idx (investigated), space}
    plus a per-panel record of the transition points (norm + raw).
    """
    pkl = load_split_data(name)
    scaler_X = pkl["scaler_X"]
    feat_names = pkl["feature_names"]
    nx = len(feat_names)
    train_input = pkl["dataset"]["train_input"]
    model = load_kan_model(name)

    panels = []
    tp_record = {}

    for key in PANEL_KEYS:
        if key == "kan":
            inv_idx = pkl["selected_mask_idx"]
            # split_points already normalized [0.1..0.9] incl. ends.
            edges_norm = sorted(float(s) for s in pkl["split_points"])
            interior_norm = [e for e in edges_norm[1:-1]]
            interior_raw = to_raw(scaler_X, nx, inv_idx, interior_norm).tolist() \
                if interior_norm else []
        else:
            mode = PANEL_AGSM_MODE[key]
            inv_idx, _, raw_tps = agsm_transition_points_raw(name, mode)
            # raw TPs -> normalized; keep strictly inside (0.1, 0.9).
            norm_tps = to_norm(scaler_X, nx, inv_idx, raw_tps) if raw_tps else np.array([])
            lo_n, hi_n = 0.1, 0.9
            margin = _EDGE_MARGIN_FRAC * (hi_n - lo_n)
            keep_norm = sorted(float(v) for v in norm_tps
                               if lo_n + margin < v < hi_n - margin)
            interior_norm = keep_norm
            interior_raw = to_raw(scaler_X, nx, inv_idx, interior_norm).tolist() \
                if interior_norm else []
            edges_norm = [0.1] + interior_norm + [0.9]

        values = kan_scores_for_edges(model, train_input, nx, inv_idx, edges_norm)

        feat_sym = f"x{inv_idx}"  # match toy_KAN_analyze.py label convention
        labels = [f"{edges_norm[i]:.2f} < {feat_sym} <= {edges_norm[i + 1]:.2f}"
                  for i in range(len(edges_norm) - 1)]

        panels.append({
            "key": key, "title": PANEL_TITLES[key], "labels": labels,
            "values": values, "inv_idx": inv_idx,
        })
        tp_record[key] = {
            "inv_idx": inv_idx,
            "interior_norm": [round(v, 4) for v in interior_norm],
            "interior_raw": [round(v, 4) for v in interior_raw],
            "n_intervals": len(labels),
        }

    return feat_names, panels, tp_record


# ----------------------------------------------------------------------------
# Series 2: AGSM S_a per interval (raw space, analytic function)
# ----------------------------------------------------------------------------
def series2_panels(name):
    """Build the 4 S_a panels for one dataset (raw-space AGSM)."""
    pkl = load_split_data(name)
    scaler_X = pkl["scaler_X"]
    feat_names_pkl = pkl["feature_names"]
    nx = len(feat_names_pkl)

    zoo = FUNCTION_ZOO[name]
    func_batch = make_batch_func(zoo["func"])
    bounds = zoo["bounds"]
    feat_names = zoo["names"]

    # Top-2 features = the two Feature_idx in the AGSM CSV (chosen by KAN score).
    top2, _, _ = agsm_feature_order(name, "equal")
    top2 = top2[:2]
    i_idx, j_idx = top2[0], top2[1]

    # Verify the two top features share bounds; note if they differ.
    bounds_i = [float(bounds[i_idx][0]), float(bounds[i_idx][1])]
    bounds_j = [float(bounds[j_idx][0]), float(bounds[j_idx][1])]
    same_bounds = np.allclose(bounds_i, bounds_j)
    lo, hi = bounds_i  # apply i's bounds as the outer edges to both top-2 feats

    panels = []
    tp_record = {}

    for key in PANEL_KEYS:
        if key == "kan":
            # KAN transition points are normalized -> inverse_transform to raw,
            # along the KAN-selected feature, then clip to [lo, hi].
            inv_idx = pkl["selected_mask_idx"]
            interior_norm = sorted(float(s) for s in pkl["split_points"])[1:-1]
            raw_all = to_raw(scaler_X, nx, inv_idx, interior_norm) \
                if interior_norm else np.array([])
            margin = _EDGE_MARGIN_FRAC * (hi - lo)
            interior_raw = sorted(float(v) for v in raw_all
                                  if lo + margin < v < hi - margin)
        else:
            mode = PANEL_AGSM_MODE[key]
            _, _, raw_tps = agsm_transition_points_raw(name, mode)
            margin = _EDGE_MARGIN_FRAC * (hi - lo)
            interior_raw = sorted(float(v) for v in raw_tps
                                  if lo + margin < v < hi - margin)

        edges = np.array([lo] + interior_raw + [hi], dtype=float)
        custom_edges = {i_idx: edges, j_idx: edges}

        sc, sh, sa, r = compute_gradient_agsm(
            func_batch=func_batch, bounds=bounds, feat_names=feat_names,
            top2_idx=top2, n_sections=len(edges) - 1, n_samples_per_section=512,
            seed=42, section_mode="custom", custom_edges=custom_edges,
        )
        # Build values[n_int, 2] in top-2 order; both share the same edges.
        n_int = len(edges) - 1
        values = np.zeros((n_int, 2))
        for col, fidx in enumerate(top2):
            arr = np.nan_to_num(np.asarray(sa[fidx], float), nan=0.0)
            # n_eff should equal n_int (shared custom edges); guard anyway.
            m = min(len(arr), n_int)
            values[:m, col] = arr[:m]

        feat_sym = f"x{i_idx}"
        labels = [f"{edges[k]:.2f} < {feat_sym} <= {edges[k + 1]:.2f}"
                  for k in range(n_int)]

        panels.append({
            "key": key, "title": PANEL_TITLES[key], "labels": labels,
            "values": values, "inv_idx": i_idx,
        })
        tp_record[key] = {
            "inv_idx": i_idx,
            "interior_raw": [round(v, 4) for v in interior_raw],
            "n_intervals": n_int,
        }

    top2_names = [feat_names[i_idx], feat_names[j_idx]]
    return top2_names, panels, tp_record, same_bounds, (bounds_i, bounds_j)


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def grouped_bars(ax, labels, feat_names, values, ylabel, title, xlabel):
    n_intervals = len(labels)
    n_features = values.shape[1]
    x_positions = np.arange(n_intervals)
    max_score = float(np.nanmax(values)) if values.size else 1.0

    for feat_idx in range(n_features):
        feat_scores = values[:, feat_idx]
        offset = (feat_idx - n_features / 2) * WIDTH + WIDTH / 2
        color = FEAT_COLORS[feat_idx % len(FEAT_COLORS)]
        ax.bar(x_positions + offset, feat_scores, WIDTH,
               label=feat_names[feat_idx], color=color)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=20, ha="center", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylim(0, max_score * 1.2 if max_score > 0 else 1.0)
    ax.legend(loc="upper right")


def build_figure(name, series, feat_names, panels, ylabel, xlabel, suptitle):
    with plt.rc_context(SA_RC):
        fig, axes = plt.subplots(1, 4, figsize=(22, 5), squeeze=False)
        axes = axes[0]
        for ax, panel in zip(axes, panels):
            grouped_bars(ax, panel["labels"], feat_names, panel["values"],
                         ylabel=ylabel, title=panel["title"], xlabel=xlabel)
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        os.makedirs(FIGDIR, exist_ok=True)
        stem = os.path.join(FIGDIR, f"{name}_attr_by_transition_{series}")
        out_paths = []
        for ext in ("png", "svg", "eps"):
            p = f"{stem}.{ext}"
            fig.savefig(p)
            out_paths.append(p)
        plt.close(fig)
    return out_paths


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def main():
    all_paths = []
    notes = []

    for name in DATASETS:
        # ---- Series 1 (KAN score) ----
        feat_names1, panels1, tp1 = series1_panels(name)
        paths1 = build_figure(
            name, "KANscore", feat_names1, panels1,
            ylabel="Normalized Attribution Score",
            xlabel="Interval (normalized)",
            suptitle=f"{name}  |  Series 1: KAN attribution score by transition-point set",
        )
        all_paths.extend(paths1)
        for p in paths1:
            print(f"Saved: {p}")

        # Sanity: Panel 1 (KAN TPs) vs existing kan_interval_scores.csv.
        kan_csv = pd.read_csv(
            os.path.join(kan_dir(name), f"{name}_kan_interval_scores.csv"))
        ref = kan_csv[feat_names1].to_numpy(float)
        got = panels1[0]["values"]
        if ref.shape == got.shape and np.allclose(ref, got, rtol=1e-3, atol=1e-3):
            sanity = "MATCH"
        else:
            sanity = f"DIFFER (ref{ref.shape} got{got.shape}, maxabs="
            try:
                sanity += f"{np.nanmax(np.abs(ref - got)):.4g})"
            except Exception:
                sanity += "n/a)"

        # ---- Series 2 (S_a) ----
        top2_names, panels2, tp2, same_bounds, bnds = series2_panels(name)
        paths2 = build_figure(
            name, "Sa", top2_names, panels2,
            ylabel=r"$S^a_{l,[k]}$",
            xlabel="Interval (raw)",
            suptitle=f"{name}  |  Series 2: AGSM $S_a$ by transition-point set",
        )
        all_paths.extend(paths2)
        for p in paths2:
            print(f"Saved: {p}")

        # Record notes.
        single_int = [k for k in PANEL_KEYS
                      if tp2[k]["n_intervals"] == 1 or tp1[k]["n_intervals"] == 1]
        notes.append({
            "name": name, "tp1": tp1, "tp2": tp2,
            "sanity": sanity, "same_bounds": same_bounds, "bounds": bnds,
            "collapsed": single_int,
        })

    # ---- Report ----
    print("\n" + "=" * 78)
    print("PER-DATASET TRANSITION POINTS AND NOTES")
    print("=" * 78)
    for n in notes:
        print(f"\n### {n['name']}")
        print(f"  Panel-1 sanity (KAN score vs kan_interval_scores.csv): {n['sanity']}")
        print(f"  top-2 bounds equal: {n['same_bounds']}  "
              f"(i={n['bounds'][0]}, j={n['bounds'][1]})")
        print("  Series 1 (normalized split space):")
        for k in PANEL_KEYS:
            r = n["tp1"][k]
            print(f"    {k:<9} inv=x{r['inv_idx']} n_int={r['n_intervals']} "
                  f"interior_norm={r['interior_norm']} interior_raw={r['interior_raw']}")
        print("  Series 2 (raw split space):")
        for k in PANEL_KEYS:
            r = n["tp2"][k]
            print(f"    {k:<9} inv=x{r['inv_idx']} n_int={r['n_intervals']} "
                  f"interior_raw={r['interior_raw']}")
        if n["collapsed"]:
            print(f"  NOTE: single-interval panels (no interior TP): {n['collapsed']}")

    print(f"\nTotal files written: {len(all_paths)}")
    missing = [p for p in all_paths if not os.path.exists(p)]
    print("All output files verified present." if not missing
          else "MISSING:\n" + "\n".join("  " + p for p in missing))


if __name__ == "__main__":
    main()
