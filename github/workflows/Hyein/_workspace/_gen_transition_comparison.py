"""Build a comprehensive transition-point comparison CSV across all analytic
datasets and all three AGSM section modes (equal / quantile / kan).

One row per (dataset, mode). For each row:
  - AGSM transitions: recomputed from the per-mode CSV via
    ``find_agsm_transition_points`` (up to 2 reported).
  - KAN inflection points on the top feature: denormalized from the pkl.
  - True transition: known only for exponential (= +ln(2)/2).
  - Comparison metrics (abs diff, relative error vs domain width, errors vs
    the true value where known).

Output:
    figures_for_paper/agsm_transition_comparison.csv
"""

import os

import joblib
import numpy as np
import pandas as pd

from github.workflows.Hyein.sectional_gsa import find_agsm_transition_points

BASE = r"D:\pykan\github\workflows\Hyein"
RESULTS = os.path.join(BASE, "analytical_results")
FIGDIR = os.path.join(BASE, "figures_for_paper")

DATASETS = ["exponential", "logarithm", "log2", "rosenbrock"]
MODES = ["equal", "quantile", "kan"]

# Analytical true transition (raw x value) where known.
TRUE_TRANSITION = {
    "exponential": np.log(2.0) / 2.0,  # ~ +0.34657
}


def kan_dir(name):
    return os.path.join(RESULTS, name, "kan_models")


def load_mode_csv(name, mode):
    path = os.path.join(kan_dir(name), f"{name}_agsm_sectional_{mode}.csv")
    return pd.read_csv(path)


def per_feature_curves(df):
    """Return ordered {feat_idx: (centers, S_a, name)} preserving CSV row order."""
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


def load_pkl(name):
    return joblib.load(os.path.join(kan_dir(name), f"{name}_range_split_data.pkl"))


def kan_inflections_raw(pkl, feat_idx):
    """Denormalized KAN inflection x-values on feature ``feat_idx`` (raw space)."""
    scaler_X = pkl["scaler_X"]
    ips = pkl["inflection_points_per_input"]
    nx = len(pkl["feature_names"])
    valid = [ip for ip in (ips[feat_idx] or []) if 0.05 < ip < 0.95]
    if not valid:
        return []
    dummy = np.zeros((len(valid), nx))
    dummy[:, feat_idx] = valid
    return scaler_X.inverse_transform(dummy)[:, feat_idx].tolist()


def main():
    rows = []
    for name in DATASETS:
        pkl = load_pkl(name)
        scaler_X = pkl["scaler_X"]
        true_t = TRUE_TRANSITION.get(name, np.nan)

        for mode in MODES:
            df = load_mode_csv(name, mode)
            curves, order = per_feature_curves(df)
            i_idx, j_idx = order[0], order[1]
            ci, sai, name_i = curves[i_idx]
            cj, saj, name_j = curves[j_idx]
            n_sections = len(ci)

            tps = find_agsm_transition_points(ci, sai, cj, saj, name_i, name_j)
            agsm = sorted(t["point"] for t in tps)
            agsm_1 = agsm[0] if len(agsm) >= 1 else np.nan
            agsm_2 = agsm[1] if len(agsm) >= 2 else np.nan

            # KAN inflections on the top (sectioned) feature.
            kan_ips = sorted(kan_inflections_raw(pkl, i_idx))
            kan_1 = kan_ips[0] if len(kan_ips) >= 1 else np.nan
            kan_2 = kan_ips[1] if len(kan_ips) >= 2 else np.nan

            # Domain width of the top feature (raw space).
            lo = float(scaler_X.data_min_[i_idx])
            hi = float(scaler_X.data_max_[i_idx])
            width = hi - lo

            # |AGSM_transition_1 - nearest KAN inflection|.
            if np.isfinite(agsm_1) and len(kan_ips):
                nearest = kan_ips[int(np.argmin(np.abs(np.asarray(kan_ips) - agsm_1)))]
                abs_diff_1 = abs(agsm_1 - nearest)
                rel_err_1 = abs_diff_1 / width if width else np.nan
            else:
                abs_diff_1 = np.nan
                rel_err_1 = np.nan

            # Errors vs the analytical true value (exponential only).
            if np.isfinite(true_t) and np.isfinite(agsm_1):
                abs_diff_true_agsm = abs(agsm_1 - true_t)
            else:
                abs_diff_true_agsm = np.nan
            if np.isfinite(true_t) and len(kan_ips):
                kan_near_true = kan_ips[int(np.argmin(np.abs(np.asarray(kan_ips) - true_t)))]
                abs_diff_true_kan = abs(kan_near_true - true_t)
            else:
                abs_diff_true_kan = np.nan

            rows.append({
                "Dataset": name,
                "Section_mode": mode,
                "N_sections": n_sections,
                "Feature_top": name_i,
                "Feature_2nd": name_j,
                "AGSM_transition_1": agsm_1,
                "AGSM_transition_2": agsm_2,
                "KAN_inflection_top_1": kan_1,
                "KAN_inflection_top_2": kan_2,
                "True_transition": true_t,
                "Abs_diff_1": abs_diff_1,
                "Rel_error_1": rel_err_1,
                "Abs_diff_true_AGSM": abs_diff_true_agsm,
                "Abs_diff_true_KAN": abs_diff_true_kan,
            })

    cols = [
        "Dataset", "Section_mode", "N_sections", "Feature_top", "Feature_2nd",
        "AGSM_transition_1", "AGSM_transition_2",
        "KAN_inflection_top_1", "KAN_inflection_top_2",
        "True_transition", "Abs_diff_1", "Rel_error_1",
        "Abs_diff_true_AGSM", "Abs_diff_true_KAN",
    ]
    out = pd.DataFrame(rows, columns=cols)
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, "agsm_transition_comparison.csv")
    out.to_csv(path, index=False)
    print("Wrote", path)
    print()
    with pd.option_context("display.max_columns", None,
                           "display.width", 200,
                           "display.float_format", lambda v: f"{v:.4f}"):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
