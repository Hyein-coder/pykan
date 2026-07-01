"""
Compare how the shared CO2RR features affect an ECONOMICS metric (MSP) versus a
SUSTAINABILITY metric (LCA).

Both CO2RRMSP and CO2RRLCA are trained on the *identical* 8-feature input space
(current density, voltage, Faradaic efficiency, CO conversion, crossover rate,
capture energy, electricity price, membrane cost), differing only in the target.
This makes them directly comparable feature-by-feature.

Outputs (in material_kan_models/CO2RR_MSP_vs_LCA/):
  1. <tag>_global_attribution_compare.png/svg  - grouped bars, importance SHARE per metric
  2. <tag>_sensitivity_trajectories.png/svg    - |df/dx_i| swept over each feature, twin-axis overlay
  3. <tag>_activation_shapes.png/svg           - learned phi_i(x_i), mean-centred overlay
  4. <tag>_compare_summary.csv                 - score / rank / share / rank-shift per feature

Run (from repo root, pykan-new env):
  python github/workflows/Hyein/co2rr_msp_vs_lca_compare.py
  python github/workflows/Hyein/co2rr_msp_vs_lca_compare.py --econ CO2RRMSP --sust CO2RRLCA
"""
import argparse
import os

import joblib
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from github.workflows.Hyein.toy_KAN_sweep import KANRegressor


# ------------------------------------------------------------------ helpers
def load_case(data_name, root_dir, device):
    """Load a trained CO2RR KAN plus the saved (normalized) analysis design.

    Returns a dict with the model and everything needed to score / sweep it in
    the same input space it was trained on.
    """
    savepath = os.path.join(root_dir, "material_kan_models", data_name)
    ckpt_path = os.path.join(savepath, f"{data_name}_best_kan_model")
    split_path = os.path.join(savepath, f"{data_name}_range_split_data.pkl")

    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Range-split data not found: {split_path}\n"
                                f"Run material_KAN_analyze.py {data_name} first.")

    split = joblib.load(split_path)

    wrapper = KANRegressor(device=device)
    wrapper.load_model(ckpt_path)
    model = wrapper.model
    print(f"  loaded {data_name}")

    saltelli = torch.as_tensor(np.asarray(split["saltelli_input"]),
                               dtype=torch.float32, device=device)

    # Global attribution over the shared Saltelli design (reproduces the
    # *_global_attribution_scores.csv produced by material_KAN_analyze.py).
    # spline_pre/postacts are overwritten by every forward() call, so we
    # snapshot the layer-0 activations HERE (right after the Saltelli forward)
    # before any later gradient sweeps clobber them.
    model.forward(saltelli)
    scores = model.feature_score.detach().cpu().numpy().flatten()
    pre0 = model.spline_preacts[0].detach().cpu().numpy()    # (N, n_out, n_in)
    post0 = model.spline_postacts[0].detach().cpu().numpy()  # (N, n_out, n_in)

    return {
        "name": data_name,
        "model": model,
        "scores": scores,
        "saltelli": saltelli,
        "pre0": pre0,
        "post0": post0,
        "scaler_X": split["scaler_X"],
        "scaler_y": split["scaler_y"],
        "feat_names": list(split["feature_names"]),
        "train_input": split["dataset"]["train_input"].to(device),
    }


def real_ranges(scaler_X, scaler_y, n_features):
    """Denormalization spans: real-unit width per feature and for the target.

    Works for any linear scaler by inverse-transforming the [0]^p and [1]^p
    corners (avoids depending on MinMaxScaler-specific attributes).
    """
    lo = scaler_X.inverse_transform(np.zeros((1, n_features)))[0]
    hi = scaler_X.inverse_transform(np.ones((1, n_features)))[0]
    x_range = hi - lo
    y_lo = scaler_y.inverse_transform(np.zeros((1, 1)))[0, 0]
    y_hi = scaler_y.inverse_transform(np.ones((1, 1)))[0, 0]
    y_range = y_hi - y_lo
    return x_range, y_range


def sensitivity_along(case, feat_idx, grid_n=120, x_lo=0.05, x_hi=0.95):
    """|d(target)/d(x_i)| in REAL units, sweeping x_i with others held at the
    (normalized) training median. Returns (x_real, s_real)."""
    model = case["model"]
    device = case["saltelli"].device
    n_features = case["saltelli"].shape[1]

    base = torch.median(case["train_input"], dim=0).values  # typical operating point
    xs_norm = torch.linspace(x_lo, x_hi, grid_n, device=device)

    X = base.repeat(grid_n, 1).clone()
    X[:, feat_idx] = xs_norm
    X.requires_grad_(True)

    y = model(X).sum()
    grad = torch.autograd.grad(y, X)[0][:, feat_idx]  # d y_norm / d x_norm

    x_range, y_range = real_ranges(case["scaler_X"], case["scaler_y"], n_features)
    # chain rule: d y_real / d x_real = grad_norm * y_range / x_range_i
    s_real = grad.detach().cpu().numpy() * y_range / (x_range[feat_idx] + 1e-12)

    x_real = case["scaler_X"].inverse_transform(
        _fill_col(xs_norm.detach().cpu().numpy(), feat_idx, base.cpu().numpy(), n_features)
    )[:, feat_idx]
    return x_real, np.abs(s_real)


def activation_shape(case, feat_idx):
    """Mean-centred learned edge activation phi_i(x_i) at layer 0 (summed over
    outputs = net contribution of feature i). Returns (x_real_sorted, phi)."""
    pre = case["pre0"][:, :, feat_idx]    # (N, n_out) -- snapshot on Saltelli design
    post = case["post0"][:, :, feat_idx]  # (N, n_out)
    x_norm = pre[:, 0]                 # preact == input to the edge (same across outputs)
    phi = post.sum(axis=1)            # total contribution of this feature
    phi = phi - phi.mean()

    n_features = case["saltelli"].shape[1]
    x_real = case["scaler_X"].inverse_transform(
        _fill_col(x_norm, feat_idx, np.zeros(n_features), n_features)
    )[:, feat_idx]

    order = np.argsort(x_real)
    return x_real[order], phi[order]


def _fill_col(col_vals, feat_idx, base_vec, n_features):
    """Build an (N, n_features) array with base_vec broadcast and column feat_idx
    replaced by col_vals -- so inverse_transform recovers the real x_i axis."""
    arr = np.tile(base_vec, (len(col_vals), 1)).astype(float)
    arr[:, feat_idx] = col_vals
    return arr


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Compare CO2RR economics (MSP) vs sustainability (LCA) feature effects.")
    ap.add_argument("--econ", default="CO2RRMSP", help="economics dataset (default CO2RRMSP)")
    ap.add_argument("--sust", default="CO2RRLCA", help="sustainability dataset (default CO2RRLCA)")
    args = ap.parse_args()

    root_dir = os.path.join(os.getcwd(), "github", "workflows", "Hyein")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading models ({device})...")
    econ = load_case(args.econ, root_dir, device)
    sust = load_case(args.sust, root_dir, device)

    if econ["feat_names"] != sust["feat_names"]:
        raise ValueError("Feature sets differ between the two datasets; cannot compare.")
    feat_names = econ["feat_names"]
    n_features = len(feat_names)

    # Pretty short labels for compact axes.
    pretty = {
        "Current density (A/cm2)": "Current density",
        "Voltage (V)": "Voltage",
        "Faradaic efficiency": "Faradaic eff.",
        "CO conversion": "CO conversion",
        "Crossover rate": "Crossover rate",
        "Capture energy (MJ/kgCO2)": "Capture energy",
        "Electricity price (USD/kWh)": "Elec. price",
        "Membrane cost (USD/m2)": "Membrane cost",
    }
    short = [pretty.get(f, f) for f in feat_names]

    out_dir = os.path.join(root_dir, "material_kan_models", f"{args.econ}_vs_{args.sust}")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{args.econ}_vs_{args.sust}"

    ECON_COLOR = "#c0392b"  # economics = red
    SUST_COLOR = "#27ae60"  # sustainability = green
    econ_lbl = f"{args.econ} (economics)"
    sust_lbl = f"{args.sust} (sustainability)"

    # -------------------------------------------------- 1. global attribution
    econ_share = econ["scores"] / (econ["scores"].sum() + 1e-12)
    sust_share = sust["scores"] / (sust["scores"].sum() + 1e-12)

    # order features by combined importance (most informative on the left)
    order = np.argsort(econ_share + sust_share)[::-1]
    x = np.arange(n_features)
    w = 0.4

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - w / 2, econ_share[order], w, color=ECON_COLOR, edgecolor="black",
           linewidth=0.6, label=econ_lbl)
    ax.bar(x + w / 2, sust_share[order], w, color=SUST_COLOR, edgecolor="black",
           linewidth=0.6, label=sust_lbl)
    ax.set_xticks(x)
    ax.set_xticklabels([short[i] for i in order], rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Attribution share")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{tag}_global_attribution_compare.png"), dpi=300)
    fig.savefig(os.path.join(out_dir, f"{tag}_global_attribution_compare.svg"), format="svg")
    plt.close(fig)
    print("  [1/4] global attribution comparison")

    # -------------------------------------------------- 2. sensitivity trajectories
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows),
                            constrained_layout=True)
    axs = np.atleast_1d(axs).flatten()
    lines, labels = [], []
    for pos, i in enumerate(order):
        ax = axs[pos]
        xe, se = sensitivity_along(econ, i)
        xs_, ss = sensitivity_along(sust, i)

        le, = ax.plot(xe, se, color=ECON_COLOR, lw=1.8)
        ax.set_ylabel(f"|d {args.econ}/dx|", color=ECON_COLOR, fontsize=8)
        ax.tick_params(axis="y", labelcolor=ECON_COLOR, labelsize=7)

        ax2 = ax.twinx()
        ls, = ax2.plot(xs_, ss, color=SUST_COLOR, lw=1.8)
        ax2.set_ylabel(f"|d {args.sust}/dx|", color=SUST_COLOR, fontsize=8)
        ax2.tick_params(axis="y", labelcolor=SUST_COLOR, labelsize=7)

        ax.set_xlabel(short[i], fontsize=9)
        ax.grid(alpha=0.25)
        if pos == 0:
            lines, labels = [le, ls], [econ_lbl, sust_lbl]

    for j in range(n_features, len(axs)):
        axs[j].axis("off")
    fig.legend(lines, labels, loc="lower center", ncol=2, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, -0.04))
    fig.savefig(os.path.join(out_dir, f"{tag}_sensitivity_trajectories.png"), dpi=300,
                bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"{tag}_sensitivity_trajectories.svg"), format="svg",
                bbox_inches="tight")
    plt.close(fig)
    print("  [2/4] sensitivity trajectories")

    # -------------------------------------------------- 3. activation shapes
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows),
                            constrained_layout=True)
    axs = np.atleast_1d(axs).flatten()
    lines, labels = [], []
    for pos, i in enumerate(order):
        ax = axs[pos]
        xe, pe = activation_shape(econ, i)
        xs_, ps = activation_shape(sust, i)
        le, = ax.plot(xe, pe, color=ECON_COLOR, lw=1.8)
        ax.set_ylabel(f"{args.econ} phi", color=ECON_COLOR, fontsize=8)
        ax.tick_params(axis="y", labelcolor=ECON_COLOR, labelsize=7)
        ax2 = ax.twinx()
        ls, = ax2.plot(xs_, ps, color=SUST_COLOR, lw=1.8)
        ax2.set_ylabel(f"{args.sust} phi", color=SUST_COLOR, fontsize=8)
        ax2.tick_params(axis="y", labelcolor=SUST_COLOR, labelsize=7)
        ax.set_xlabel(short[i], fontsize=9)
        ax.axhline(0, color="gray", lw=0.6, ls=":")
        ax.grid(alpha=0.25)
        if pos == 0:
            lines, labels = [le, ls], [econ_lbl, sust_lbl]
    for j in range(n_features, len(axs)):
        axs[j].axis("off")
    fig.legend(lines, labels, loc="lower center", ncol=2, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, -0.04))
    fig.savefig(os.path.join(out_dir, f"{tag}_activation_shapes.png"), dpi=300,
                bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"{tag}_activation_shapes.svg"), format="svg",
                bbox_inches="tight")
    plt.close(fig)
    print("  [3/4] activation shapes")

    # -------------------------------------------------- 4. summary CSV
    econ_rank = (-econ["scores"]).argsort().argsort() + 1  # 1 = most important
    sust_rank = (-sust["scores"]).argsort().argsort() + 1
    df = pd.DataFrame({
        "Feature": feat_names,
        f"{args.econ}_score": econ["scores"],
        f"{args.econ}_share": econ_share,
        f"{args.econ}_rank": econ_rank,
        f"{args.sust}_score": sust["scores"],
        f"{args.sust}_share": sust_share,
        f"{args.sust}_rank": sust_rank,
        "rank_shift": sust_rank - econ_rank,  # +: more important for economics
    }).sort_values(f"{args.econ}_rank")
    csv_path = os.path.join(out_dir, f"{tag}_compare_summary.csv")
    df.to_csv(csv_path, index=False)
    print("  [4/4] summary CSV")

    print(f"\nDone. Outputs -> {out_dir}")
    print("\nGlobal attribution share (top features):")
    with pd.option_context("display.width", 140, "display.max_columns", None):
        print(df[["Feature", f"{args.econ}_share", f"{args.econ}_rank",
                  f"{args.sust}_share", f"{args.sust}_rank", "rank_shift"]]
              .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
