"""Aggregate the symbolification-robustness study into a paper figure + table.

NEW aggregation code (does not modify pykan or existing Hyein scripts).

Reads:
  - _workspace/robustness_symbolify_summary.csv  (one row per function)
  - analytical_results/{func}/kan_models/{func}_symbolify_robustness.csv  (paired transitions)

Writes -> figures_for_paper/:
  - symbolify_robustness_summary.{png,svg,eps}
  - symbolify_robustness_summary.csv

Run from D:\\pykan with PYTHONPATH=. in the pykan-new conda env.
"""
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.join(HERE, "_workspace")
AR = os.path.join(HERE, "analytical_results")
FIG = os.path.join(HERE, "figures_for_paper")
os.makedirs(FIG, exist_ok=True)

SUMMARY_CSV = os.path.join(WS, "robustness_symbolify_summary.csv")

# Project standard plot style (matches sectional_gsa.SA_RC)
SA_RC = {
    "figure.dpi": 150, "figure.facecolor": "white",
    "axes.facecolor": "white", "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8, "axes.spines.top": True, "axes.spines.right": True,
    "axes.labelsize": 12, "axes.labelcolor": "black", "axes.grid": False,
    "xtick.labelsize": 10, "xtick.color": "black", "xtick.direction": "out",
    "ytick.labelsize": 10, "ytick.color": "black", "ytick.direction": "out",
    "font.family": "sans-serif", "font.size": 10, "font.weight": "300",
    "axes.labelweight": "500", "text.color": "black",
    "legend.fontsize": 8, "legend.framealpha": 0.0,
    "lines.linewidth": 1.2, "savefig.bbox": "tight",
}


# ----------------------------------------------------------------------------
# Build the tidy paper table (self-contained, derived from per-func + summary)
# ----------------------------------------------------------------------------
def build_table():
    summ = pd.read_csv(SUMMARY_CSV)
    rows = []
    for _, s in summ.iterrows():
        func = s["func"]
        pf_path = os.path.join(AR, func, "kan_models", f"{func}_symbolify_robustness.csv")
        pf = pd.read_csv(pf_path)

        # Derive matched/added/dropped straight from the per-function detail so
        # the table is self-contained and verifiable against the raw pairing.
        matched = int((pf["matched"] == 1).sum())
        added = int((pf["added"] == 1).sum())
        dropped = int((pf["dropped"] == 1).sum())

        # Location drift for matched transitions only.
        m = pf[pf["matched"] == 1]
        if len(m) > 0:
            abs_drift = m["drift"].abs()
            abs_frac = m["drift_frac"].abs()
            median_drift_raw = float(abs_drift.median())
            max_drift_raw = float(abs_drift.max())
            median_drift_pct = float(abs_frac.median()) * 100.0
            max_drift_pct = float(abs_frac.max()) * 100.0
            # Recover domain width from drift / drift_frac (constant per func).
            dw = abs_drift / abs_frac.replace(0, np.nan)
            domain_width = float(dw.dropna().median()) if dw.notna().any() else np.nan
        else:
            median_drift_raw = max_drift_raw = np.nan
            median_drift_pct = max_drift_pct = np.nan
            domain_width = np.nan

        n_spline = int(s["n_spline"])
        n_sym = int(s["n_sym"])
        rows.append({
            "func": func,
            "direction": s["direction"],
            "status": s["status"],
            "n_edges_symbolified": int(s["n_edges_symbolified"]),
            "n_spline": n_spline,
            "n_sym": n_sym,
            "delta_count": n_sym - n_spline,
            "matched": matched,
            "added": added,
            "dropped": dropped,
            "count_match_rate": float(s["count_match_rate"]),
            "domain_width": domain_width,
            "median_drift_raw": median_drift_raw,
            "max_drift_raw": max_drift_raw,
            "median_drift_pct": median_drift_pct,
            "max_drift_pct": max_drift_pct,
        })
    df = pd.DataFrame(rows)
    return df


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
def make_figure(df):
    funcs = df["func"].tolist()
    n = len(funcs)
    x = np.arange(n)
    w = 0.38

    C_SPLINE = "#3B6FB6"   # spline reading
    C_SYM = "#E08214"      # symbolic reading
    C_DRIFT = "#2C8C5A"    # drift line
    C_FLAG = "#C0392B"     # non-ok flag

    with plt.rc_context(SA_RC):
        fig, (ax, axd) = plt.subplots(
            2, 1, figsize=(8.2, 6.2), sharex=True,
            gridspec_kw={"height_ratios": [2.4, 1.0], "hspace": 0.12},
        )

        # ---- Top: grouped bars n_spline vs n_sym ----
        b1 = ax.bar(x - w / 2, df["n_spline"], w, label="Spline reading ($n_{spline}$)",
                    color=C_SPLINE, edgecolor="#222", linewidth=0.5)
        b2 = ax.bar(x + w / 2, df["n_sym"], w, label="Symbolic reading ($n_{sym}$)",
                    color=C_SYM, edgecolor="#222", linewidth=0.5)
        ax.bar_label(b1, padding=2, fontsize=8)
        ax.bar_label(b2, padding=2, fontsize=8)

        ax.set_ylabel("# ranking transitions")
        ax.set_ylim(0, max(df["n_spline"].max(), df["n_sym"].max()) * 1.22)
        ax.legend(loc="upper right", ncol=1)
        ax.set_title("Robustness of KAN ranking transitions under symbolification",
                     fontsize=12, weight="600", pad=8)

        # Annotate each function: direction + status (+ flag non-ok) + matched
        ymax = ax.get_ylim()[1]
        for i, (_, r) in enumerate(df.iterrows()):
            dirn = "introduce" if str(r["direction"]).startswith("introduce") else "invert"
            ok = str(r["status"]).lower() == "ok"
            txt = f"{dirn}\nm={r['matched']} +{r['added']} -{r['dropped']}"
            ax.annotate(txt, (i, -ymax * 0.015), ha="center", va="top",
                        fontsize=6.5, color="#555",
                        annotation_clip=False)
            if not ok:
                ax.annotate("STATUS!=ok", (i, ymax * 0.80), ha="center",
                            fontsize=7, color=C_FLAG, weight="700")

        # ---- Bottom: location drift (% of domain width) for matched ----
        med = df["median_drift_pct"].values
        mx = df["max_drift_pct"].values
        has_match = df["matched"].values > 0

        for i in range(n):
            if has_match[i]:
                # vertical range median..max
                axd.plot([x[i], x[i]], [med[i], mx[i]], color=C_DRIFT,
                         lw=1.2, zorder=2, alpha=0.6)
                axd.plot(x[i], mx[i], marker="_", color=C_DRIFT, ms=12,
                         mew=1.4, zorder=3)
                axd.plot(x[i], med[i], marker="o", color=C_DRIFT, ms=7,
                         mec="#1c5c39", zorder=4)
                axd.annotate(f"{med[i]:.2f}%", (x[i], med[i]), xytext=(7, 0),
                             textcoords="offset points", va="center",
                             fontsize=7, color="#1c5c39")
            else:
                # No matched transitions -> drift undefined (count changed)
                axd.annotate("no match\n(count changed)", (x[i], 0.0),
                             ha="center", va="bottom", fontsize=6.5,
                             color=C_FLAG, weight="600")
                axd.plot(x[i], 0.0, marker="x", color=C_FLAG, ms=8, mew=1.6,
                         zorder=4)

        axd.set_ylabel("Location drift\n(% domain width)")
        axd.set_xlabel("Toy function")
        top = (max(mx[np.isfinite(mx)].max() if np.isfinite(mx).any() else 1.0, 1.0)) * 1.55
        axd.set_ylim(0, top)
        axd.axhline(5.0, ls="--", lw=0.9, color="#888", zorder=1)
        axd.annotate("5% domain width", (0.0, 5.0), xytext=(2, 2),
                     textcoords="offset points", ha="left", va="bottom",
                     fontsize=6.5, color="#888")

        axd.set_xticks(x)
        axd.set_xticklabels(funcs, rotation=20, ha="right")

        # drift legend
        leg = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=C_DRIFT,
                   markersize=7, label="median |drift| (matched)"),
            Line2D([0], [0], marker="_", color=C_DRIFT, markersize=12, lw=0,
                   label="max |drift|"),
            Line2D([0], [0], marker="x", color=C_FLAG, lw=0, markersize=8,
                   label="no matched transition"),
        ]
        axd.legend(handles=leg, loc="upper center", ncol=3, fontsize=7,
                   bbox_to_anchor=(0.5, 1.02))

        for ext in ("png", "svg", "eps"):
            dpi = 300 if ext == "png" else None
            fig.savefig(os.path.join(FIG, f"symbolify_robustness_summary.{ext}"),
                        dpi=dpi)
        plt.close(fig)


def main():
    df = build_table()
    out_csv = os.path.join(FIG, "symbolify_robustness_summary.csv")
    df.to_csv(out_csv, index=False)
    make_figure(df)

    # Printed interpretation summary -------------------------------------
    print("=== symbolify robustness paper aggregation ===")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(df.to_string(index=False))
    print()
    robust, nonrobust = [], []
    for _, r in df.iterrows():
        count_pres = (r["delta_count"] == 0)
        small_drift = (r["matched"] > 0 and r["median_drift_pct"] < 5.0)
        if count_pres and small_drift:
            robust.append(r["func"])
        else:
            nonrobust.append(r["func"])
    print("Count-preserved AND small matched drift (<5% dw):", robust if robust else "(none)")
    print("Count CHANGED under symbolify (non-robust):", nonrobust)
    print(f"\nWrote: {out_csv}")
    print("Wrote: symbolify_robustness_summary.{png,svg,eps}")
    return df


if __name__ == "__main__":
    main()
