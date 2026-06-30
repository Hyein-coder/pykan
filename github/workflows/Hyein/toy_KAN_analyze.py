import argparse
import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
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
from github.workflows.Hyein.bspline_curvature import (
    edge_curves, symbolic_edge_info,
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
    parser.add_argument("--refit-steps", type=int, default=None,
                        help="LBFGS step count for the post-symbolic refit (symbolic "
                             "mode with --refit). Higher = more accurate symbolic fit, "
                             "helps reach the R²≥0.8 target. Default: the loaded "
                             "model's own training step count.")

    args = parser.parse_args()
    data_name = args.func_name
    model_mode = args.model_mode
    refit_symbolic = args.refit
    refit_steps = args.refit_steps
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

    # CONVENTION: knots and ranking-transition points are kept in NORMALIZED space
    # throughout the script (the space the spline grid lives in). They are
    # denormalized to RAW input values only for PLOTTING, via this
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
    # 2.1 Model reading mode + shared analysis (kan_analysis_core)
    # ==========================================
    # All post-training analysis now lives in kan_analysis_core.analyze_model so
    # toy_KAN_analyze / grid_k_sweep / material_KAN_analyze share ONE implementation.
    # Toy is the reference — its outputs must stay parity-identical.
    from github.workflows.Hyein.kan_analysis_core import (
        select_model_reading, analyze_model,
    )

    train_input_norm = dataset['train_input']
    train_label_norm = torch.tensor(scaler_y.transform(y_train), dtype=torch.float32,
                                    device=device).reshape(-1, 1)
    model, sym_info_saved, mode_msg = select_model_reading(
        model, model_mode, model_wrapper=model_wrapper, refit=refit_symbolic,
        refit_steps=refit_steps, train_input_norm=train_input_norm,
        train_label_norm=train_label_norm, device=device)
    print(mode_msg)

    # Non-default modes write into a kan_models/<mode> subfolder so the canonical
    # as-saved outputs are preserved.
    if model_mode != 'as-saved':
        savepath = os.path.join(savepath, model_mode)
        os.makedirs(savepath, exist_ok=True)
        print(f"\U0001F4C1 Outputs for this run -> {savepath}")

    analyze_model(
        model, scaler_X=scaler_X, scaler_y=scaler_y, feat_names=feat_names,
        bounds=bounds, savepath=savepath, X_norm=X_train_norm,
        y_norm=scaler_y.transform(y_train), X_raw=X_train, true_func=target_func,
        device=device, tag='', data_name=data_name, model_mode=model_mode,
    )


if __name__ == "__main__":
    main()
