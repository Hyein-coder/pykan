"""Paper-quality summary figure for the rel-threshold (tau-frac) sweep.

Standalone figure-builder for Phase 5 of the curvature-inflection harness.
It is READ-ONLY: it reads two already-produced, gate-verified CSVs and writes
one multi-format figure. It does NOT retrain, re-sweep, or modify any existing
analysis script. It follows the project plotting guide (ProjectSummary.md):
the shared SA_RC rcParams, RdYlBu palette, transparent legend, no grid,
no titles, y-headroom, and png+svg+eps output.

Inputs (in figures_for_paper/):
  * rel_threshold_sweep.csv          -- per (func, rel_thresh) sweep rows.
      columns: func, rel_thresh, feat_idx, n_down, n_up, points_norm,
               points_raw, tau, detected, loc_err, clean
  * rel_threshold_recommendation.csv -- per-func chosen threshold + clean band.
      columns: func, recommended_rel_thresh, loc_err, band_lo, band_hi, note

Output (in figures_for_paper/):
  * rel_threshold_sweep.{png,svg,eps}   (png at dpi=300)

For each func in the sweep CSV one panel is drawn over a categorical
x = rel_thresh axis:
  * bars (left axis)  : location error |transition - true x0 kink| (normalized);
                        the recommended threshold's bar is a contrasting colour
                        with its own legend entry.
  * line (right axis) : # x0 down-transitions (n_down), drawn IN FRONT of the
                        bars, with a dotted "clean == 1" reference.
No shaded band, no in-plot text box, no title -- per the project plotting guide.

Run in the pykan-new conda env:
    set PYTHONPATH=.
    python github/workflows/Hyein/plot_rel_threshold_sweep.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless save, mirrors the other paper builders
import matplotlib.pyplot as plt

# --- paths -------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures_for_paper")
SWEEP_CSV = os.path.join(FIG, "rel_threshold_sweep.csv")
RECO_CSV = os.path.join(FIG, "rel_threshold_recommendation.csv")
OUT_STEM = os.path.join(FIG, "rel_threshold_sweep")

# --- project standard plot style (ProjectSummary.md "Plotting Style (SA_RC)") -
SA_RC = {
    # Figure
    "figure.dpi": 150, "figure.facecolor": "white", "figure.autolayout": True,
    # Axes
    "axes.facecolor": "white", "axes.edgecolor": "#444444", "axes.linewidth": 0.8,
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.labelsize": 12, "axes.labelcolor": "black", "axes.labelweight": "500",
    "axes.titlelocation": "center", "axes.grid": False,
    # Ticks
    "xtick.labelsize": 12, "xtick.color": "black", "xtick.direction": "out",
    "ytick.labelsize": 10, "ytick.color": "black", "ytick.direction": "out",
    # Font
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue LT Pro", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10, "font.weight": "300", "text.color": "black",
    # Patches (bars, legend swatches)
    "patch.edgecolor": "black", "patch.linewidth": 0.7, "patch.force_edgecolor": True,
    # Legend
    "legend.fontsize": 8, "legend.title_fontsize": 9, "legend.framealpha": 0.0,
    "legend.edgecolor": "#444444",
    # Lines
    "lines.linewidth": 0.7, "lines.markersize": 3,
    # Saving
    "savefig.dpi": 150, "savefig.bbox": "tight", "savefig.facecolor": "white",
}

# RdYlBu anchors (project palette): blue for the bulk loc_err bars, red to
# highlight the recommended threshold's bar; near-black for the foreground line.
_RDYLBU = plt.get_cmap("RdYlBu")
C_BAR = _RDYLBU(0.85)       # blue -- location-error bars
C_BAR_RECO = _RDYLBU(0.10)  # red  -- recommended-threshold bar
C_NDOWN = "#222222"         # near-black -- transition-count line (front)


def _panel(ax, df_f, reco):
    """Draw the sweep panel for a single function onto `ax`.

    Bars (left axis)  : loc_err (normalized |transition - true kink|); the
                        recommended threshold's bar is a contrasting colour with
                        its own legend entry.
    Line (right axis) : n_down (x0 down-transitions), drawn IN FRONT of the bars,
                        with a dotted clean-target (== 1) reference.
    No in-plot text box.
    """
    df_f = df_f.sort_values("rel_thresh").reset_index(drop=True)
    rel = df_f["rel_thresh"].to_numpy(dtype=float)
    pos = np.arange(len(rel))                      # categorical positions
    n_down = df_f["n_down"].to_numpy(dtype=float)
    le = df_f["loc_err"].to_numpy(dtype=float)     # NaN where not detected

    rec_tau = float(reco["recommended_rel_thresh"])
    rec_idx = int(np.argmin(np.abs(rel - rec_tau)))

    # --- bars (left axis): bulk (blue) + recommended (red), separate legends ---
    finite = np.isfinite(le)
    other = finite & (pos != rec_idx)
    ax.bar(pos[other], le[other], color=C_BAR, width=0.7,
           label=r"location error $|\Delta|$ (norm)")
    if finite[rec_idx]:
        ax.bar([pos[rec_idx]], [le[rec_idx]], color=C_BAR_RECO, width=0.7,
               label=f"recommended (rel_thresh = {rec_tau:g})")
    ax.set_ylabel(r"Location error $|\Delta|$ (normalized)")
    ax.set_xlabel(r"$\epsilon_{threshold}$")
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{v:g}" for v in rel])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2)         # y headroom (project guide)

    # --- transition-count line (right axis), drawn IN FRONT of the bars --------
    ax2 = ax.twinx()
    ax2.plot(pos, n_down, "o-", color=C_NDOWN, lw=1.5, ms=5, zorder=5,
             label=r"# $x_0$ down-transitions")
    ax2.axhline(1.0, color=C_NDOWN, ls=":", lw=1.0, zorder=4,
                label="clean target (n_down = 1)")
    ax2.set_ylabel(r"# $x_0$ down-transitions")
    nd_max = int(np.nanmax(n_down)) if np.isfinite(n_down).any() else 1
    ax2.set_ylim(0, (nd_max + 1) * 1.2)
    ax2.set_yticks(range(0, nd_max + 2))

    # Composite the line axis ABOVE the bar axis (line in front), keeping the
    # bars visible (transparent line-axis patch).
    ax2.set_zorder(ax.get_zorder() + 1)
    ax.patch.set_visible(True)
    ax2.patch.set_visible(False)

    # --- combined legend on the top (line) axis so nothing hides it -----------
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper center")


def main():
    sweep = pd.read_csv(SWEEP_CSV)
    reco = pd.read_csv(RECO_CSV)

    funcs = list(sweep["func"].drop_duplicates())
    n = len(funcs)

    with plt.rc_context(SA_RC):
        fig, axes = plt.subplots(n, 1, figsize=(7.2, 4.4 * n), squeeze=False)
        for i, func in enumerate(funcs):
            df_f = sweep[sweep["func"] == func]
            reco_f = reco[reco["func"] == func]
            if reco_f.empty:
                raise ValueError(f"no recommendation row for func={func}")
            _panel(axes[i, 0], df_f, reco_f.iloc[0])

        for ext in ("png", "svg", "eps"):
            dpi = 300 if ext == "png" else None
            out = f"{OUT_STEM}.{ext}"
            fig.savefig(out, dpi=dpi)
            print(f"wrote {out}")
        plt.close(fig)

    print(f"=== rel_threshold sweep figure built for funcs: {funcs} ===")


if __name__ == "__main__":
    main()
