"""Generate a paper-quality 4x3 comparison figure of Sectional GSA (AGSM)
across the three sectioning modes (equal-distance / quantile / kan) for all
four analytic datasets.

Rows  = datasets (exponential, logarithm, log2, rosenbrock)
Cols  = sectioning mode (equal / quantile / kan)

Each panel: step plot of S_a for the top-2 features (read from the per-mode
CSVs), KAN inflection points (green dashed, denormalized from the pkl), and
AGSM transition points (orange dotted, recomputed from the S_a curves).

For the exponential dataset the analytical true transition x0 = +ln(2)/2 is
drawn as a black dashed line.

Outputs:
    figures_for_paper/VS_sectional_gsa_modes.png
    figures_for_paper/VS_sectional_gsa_modes.svg
"""

import os

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reuse the project transition-point detector for consistency with the pipeline.
from github.workflows.Hyein.sectional_gsa import find_agsm_transition_points

BASE = r"D:\pykan\github\workflows\Hyein"
RESULTS = os.path.join(BASE, "analytical_results")
FIGDIR = os.path.join(BASE, "figures_for_paper")

DATASETS = ["exponential", "logarithm", "log2", "rosenbrock"]
MODES = ["equal", "quantile", "kan"]
MODE_LABELS = {"equal": "Equal-distance", "quantile": "Quantile", "kan": "KAN-grid"}

# Analytical true transition points (raw x value), where known.
TRUE_TRANSITION = {
    "exponential": np.log(2) / 2.0,  # ~ +0.3466
}

FEAT_COLORS = ["#1f77b4", "#d62728"]

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


def kan_dir(name):
    return os.path.join(RESULTS, name, "kan_models")


def load_mode_csv(name, mode):
    path = os.path.join(kan_dir(name), f"{name}_agsm_sectional_{mode}.csv")
    return pd.read_csv(path)


def load_kan_inflections(name):
    """Return {feat_idx: [raw inflection x-values]} denormalized from the pkl."""
    pkl = joblib.load(os.path.join(kan_dir(name), f"{name}_range_split_data.pkl"))
    scaler_X = pkl["scaler_X"]
    ips = pkl["inflection_points_per_input"]
    nx = len(pkl["feature_names"])

    out = {}
    for feat_idx in range(nx):
        valid = [ip for ip in (ips[feat_idx] or []) if 0.05 < ip < 0.95]
        if not valid:
            out[feat_idx] = []
            continue
        dummy = np.zeros((len(valid), nx))
        dummy[:, feat_idx] = valid
        out[feat_idx] = scaler_X.inverse_transform(dummy)[:, feat_idx].tolist()
    return out


def per_feature_curves(df):
    """Split a mode CSV into ordered {feat_idx: (centers, S_a, feat_name)}.

    Preserves CSV row order (top-1 feature first) so panel/legend ordering
    matches the pipeline.
    """
    curves = {}
    order = []
    for feat_idx, g in df.groupby("Feature_idx", sort=False):
        g = g.sort_values("Section_k")
        curves[int(feat_idx)] = (
            g["Section_center"].to_numpy(float),
            g["S_a"].to_numpy(float),
            str(g["Feature"].iloc[0]),
        )
        order.append(int(feat_idx))
    return curves, order


def main():
    # Pre-load everything and keep a per-dataset/per-mode transition record.
    summary_rows = []
    cache = {}

    for name in DATASETS:
        kan_ips = load_kan_inflections(name)
        cache[name] = {"kan_ips": kan_ips, "modes": {}}
        for mode in MODES:
            df = load_mode_csv(name, mode)
            curves, order = per_feature_curves(df)
            i_idx, j_idx = order[0], order[1]
            ci, sai, _ = curves[i_idx]
            cj, saj, _ = curves[j_idx]
            tps = find_agsm_transition_points(
                ci, sai, cj, saj,
                curves[i_idx][2], curves[j_idx][2],
            )
            cache[name]["modes"][mode] = {
                "curves": curves, "order": order, "tps": tps,
            }

            # KAN inflection(s) for the top-1 feature (the sectioned x-axis feature).
            kan_for_top = kan_ips.get(i_idx, [])
            summary_rows.append({
                "dataset": name,
                "mode": mode,
                "agsm": [round(t["point"], 3) for t in tps],
                "kan": [round(v, 3) for v in kan_for_top],
                "true": (round(TRUE_TRANSITION[name], 3)
                         if name in TRUE_TRANSITION else None),
            })

    # ---- Build the 4x2 figure ----
    with plt.rc_context(SA_RC):
        fig, axes = plt.subplots(len(DATASETS), len(MODES), sharey=True,
                                 figsize=(13, 11), squeeze=False)
        for row, name in enumerate(DATASETS):
            kan_ips = cache[name]["kan_ips"]
            for col, mode in enumerate(MODES):
                ax = axes[row][col]
                m = cache[name]["modes"][mode]
                curves, order, tps = m["curves"], m["order"], m["tps"]
                i_idx = order[0]

                for color, feat_idx in zip(FEAT_COLORS, order):
                    centers, sa, fname = curves[feat_idx]
                    ax.step(centers, sa, where="mid", color=color, label=fname)

                # KAN inflection points (green dashed) for both sectioned feats.
                first_kan = True
                for feat_idx in order:
                    for ip in kan_ips.get(feat_idx, []):
                        ax.axvline(x=ip, color="green", linestyle="--", alpha=0.7,
                                   linewidth=1.0,
                                   label="KAN inflection" if first_kan else "_")
                        first_kan = False

                # AGSM transition points (orange dotted).
                first_tp = True
                for tp in tps:
                    ax.axvline(x=tp["point"], color="orange", linestyle=":",
                               alpha=0.85, linewidth=1.2,
                               label="AGSM transition" if first_tp else "_")
                    first_tp = False

                # Analytical true transition (black dashed) where known.
                if name in TRUE_TRANSITION:
                    ax.axvline(x=TRUE_TRANSITION[name], color="black",
                               linestyle="--", alpha=0.9, linewidth=1.2,
                               label="True transition")

                ax.set_ylabel(r"$S^a_{l,[k]}$")
                if row == 0:
                    ax.set_title(MODE_LABELS[mode], fontsize=11, fontweight="bold")
                if col == 0:
                    # Dataset label on the left of each row.
                    ax.text(-0.30, 0.5, name, transform=ax.transAxes,
                            rotation=90, va="center", ha="center",
                            fontsize=11, fontweight="bold")
                ax.set_xlabel(curves[i_idx][2])
                ax.legend(loc="best")

        fig.suptitle("Sectional GSA (AGSM): equal-distance / quantile / KAN-grid sectioning",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0.02, 0, 1, 0.98])

        os.makedirs(FIGDIR, exist_ok=True)
        png = os.path.join(FIGDIR, "VS_sectional_gsa_modes.png")
        svg = os.path.join(FIGDIR, "VS_sectional_gsa_modes.svg")
        fig.savefig(png)
        fig.savefig(svg)
        plt.close(fig)
        print(f"Saved: {png}")
        print(f"Saved: {svg}")

    # ---- Print comparison table ----
    print()
    header = (f"{'Dataset':<12} | {'Mode':<9} | {'AGSM transition':<22} | "
              f"{'KAN inflection':<22} | {'True transition'}")
    print(header)
    print("-" * len(header))
    for r in summary_rows:
        agsm = ", ".join(f"{v:.3f}" for v in r["agsm"]) if r["agsm"] else "none"
        kan = ", ".join(f"{v:.3f}" for v in r["kan"]) if r["kan"] else "none"
        true = f"{r['true']:.3f}" if r["true"] is not None else "N/A"
        print(f"{r['dataset']:<12} | {r['mode']:<9} | {agsm:<22} | "
              f"{kan:<22} | {true}")


if __name__ == "__main__":
    main()
