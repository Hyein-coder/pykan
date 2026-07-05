"""
material_3d_surfaces.py
=======================

Draw a 3D KAN prediction surface for EVERY 2-variable combination of the input
features vs. the output y, for any material dataset in the project.

For each feature pair (f1, f2) a grid is swept over their observed range while
all *other* features are held fixed at their training-set mean (the "Fixed at
Mean" mode already used in material_KAN_analyze.py, section 9). The KAN
prediction over that grid is drawn as a 3D surface (z = output y), with the
real training points overlaid for context.

Usage
-----
    conda activate pykan-new
    python -m github.workflows.Hyein.material_3d_surfaces AgNP
    python -m github.workflows.Hyein.material_3d_surfaces AgNP --seed 3 --grid_res 40
    python -m github.workflows.Hyein.material_3d_surfaces P3HT --eps

The loading logic (paths, scalers, outlier removal, train/test split) is copied
verbatim from material_KAN_analyze.py so the surfaces are consistent with the
rest of the material analysis pipeline. Works for any dataset that has a saved
KAN model + scalers under material_kan_models/<data_name>[ _seed_<seed>].
"""

import argparse
import os
import itertools

import joblib
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from kan.custom_processing import remove_outliers_iqr
from github.workflows.Hyein.toy_KAN_sweep import KANRegressor


# ==========================================
# Register Python Tuple for YAML Loading (same as material_KAN_analyze.py)
# ==========================================
def tuple_constructor(loader, node):
    return tuple(loader.construct_sequence(node))


yaml.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor, Loader=yaml.SafeLoader)
try:
    yaml.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor, Loader=yaml.Loader)
except AttributeError:
    pass


# ==========================================
# SA_RC plotting style (project house style — see ProjectSummary.md)
# No grid, no titles, transparent legend, RdYlBu palette.
# ==========================================
SA_RC = {
    'figure.dpi': 150,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#444444',
    'axes.linewidth': 0.8,
    'axes.labelsize': 12,
    'axes.labelcolor': 'black',
    'axes.grid': False,
    'xtick.labelsize': 10,
    'xtick.color': 'black',
    'ytick.labelsize': 10,
    'ytick.color': 'black',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue LT Pro', 'Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'font.size': 10,
    'font.weight': '300',
    'axes.labelweight': '500',
    'text.color': 'black',
    'legend.fontsize': 8,
    'legend.framealpha': 0.0,
    'legend.edgecolor': '#444444',
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
}

# pretty output-name lookup (mirrors material_KAN_analyze.py)
PRETTY_OUTPUT_NAME = {
    "loss": "Spectrum fitness",
    "minimum_selling_price": "Minimum selling price",
    "NPV (USD)": "Net present value",
}


def load_material_model(data_name, rand_seed=None):
    """Load model, scalers and (denormalized) training data for a material dataset.

    Returns a dict with everything the surface plotting needs. Loading logic is
    identical to material_KAN_analyze.py so results stay consistent.
    """
    root_dir = os.path.join(os.getcwd(), 'github', 'workflows', 'Hyein')
    filepath = os.path.join(root_dir, "data", f"{data_name}.csv")

    if rand_seed is not None:
        savepath = os.path.join(root_dir, "material_kan_models", data_name + f"_seed_{rand_seed}")
    else:
        savepath = os.path.join(root_dir, "material_kan_models", data_name)
        rand_seed = 42

    ckpt_path = os.path.join(savepath, f'{data_name}_best_kan_model')
    scaler_x_path = os.path.join(savepath, f'{data_name}_mlp_scaler_X.pkl')
    scaler_y_path = os.path.join(savepath, f'{data_name}_mlp_scaler_y.pkl')

    if not os.path.exists(scaler_x_path) or not os.path.exists(scaler_y_path):
        raise FileNotFoundError(f"Scalers not found under {savepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at {filepath}")

    scaler_X = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_wrapper = KANRegressor(device=device)
    model_wrapper.load_model(ckpt_path)
    model = model_wrapper.model
    print("✅ KAN Model loaded successfully!")

    # --- Regenerate data exactly like material_KAN_analyze.py ---
    filedata = pd.read_csv(filepath)
    name_X = filedata.columns[:-1].tolist()
    name_y = filedata.columns[-1]
    df_in = filedata[name_X]
    df_out = filedata[[name_y]]

    df_in_final, df_out_final = remove_outliers_iqr(df_in, df_out)
    print(f"# of data after removing outliers: {len(df_in_final)} "
          f"({len(df_in) - len(df_in_final)} removed)")

    X = df_in_final[name_X].values
    y = df_out_final[name_y].values.reshape(-1, 1)

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_seed)
    X_train_denorm, X_val, y_train_denorm, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=rand_seed)

    # material_KAN_analyze.py re-fits scaler_X on the training split — mirror it
    # so transform()/inverse_transform() match the rest of the pipeline.
    scaler_X.fit(X_train_denorm)
    scaler_y.fit(y_train_denorm)

    output_name = PRETTY_OUTPUT_NAME.get(name_y, name_y)

    return {
        'model': model,
        'device': device,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'X_train_denorm': X_train_denorm,
        'y_train_denorm': y_train_denorm,
        'feat_names': name_X,
        'output_name': output_name,
        'savepath': savepath,
    }


def predict_surface(model, scaler_X, scaler_y, device,
                    f1_idx, f2_idx, x1_lin, x2_lin, fixed_denorm):
    """Predict the (denormalized) KAN output over an (f1, f2) grid.

    All features except f1, f2 are held at `fixed_denorm` (a per-feature vector,
    typically the training mean). Returns X1_mesh, X2_mesh, Z (all grid_res x grid_res).
    """
    n_features = fixed_denorm.shape[0]
    X1_mesh, X2_mesh = np.meshgrid(x1_lin, x2_lin)
    grid_coords = np.stack([X1_mesh.ravel(), X2_mesh.ravel()], axis=-1)

    grid_input_denorm = np.tile(fixed_denorm, (grid_coords.shape[0], 1))
    grid_input_denorm[:, f1_idx] = grid_coords[:, 0]
    grid_input_denorm[:, f2_idx] = grid_coords[:, 1]

    with torch.no_grad():
        in_norm = torch.tensor(scaler_X.transform(grid_input_denorm),
                               dtype=torch.float32, device=device)
        Z = scaler_y.inverse_transform(model(in_norm).cpu().numpy()).reshape(X1_mesh.shape)

    return X1_mesh, X2_mesh, Z


def _style_3d_axes(ax, f1_name, f2_name, output_name):
    ax.set_xlabel(f1_name, labelpad=8)
    ax.set_ylabel(f2_name, labelpad=8)
    ax.set_zlabel(output_name, labelpad=8)
    # keep panes light and unobtrusive (house style: no busy grid/box)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_edgecolor('#dddddd')
        pane.set_alpha(0.15)
    ax.tick_params(labelsize=8)


def main():
    parser = argparse.ArgumentParser(
        description="3D KAN prediction surfaces for every 2-variable feature combination.")
    parser.add_argument("data_name", type=str, nargs='?', default="AgNP",
                        help="Dataset name (default: AgNP). Any dataset with a saved "
                             "KAN model under material_kan_models/ works.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed of the saved model dir (default: None -> base dir, seed 42 for split).")
    parser.add_argument("--grid_res", type=int, default=40,
                        help="Surface grid resolution per axis (default: 40).")
    parser.add_argument("--no_scatter", action="store_true",
                        help="Do not overlay the real training data points.")
    parser.add_argument("--eps", action="store_true",
                        help="Also save .eps (vector; can be large for surfaces).")
    args = parser.parse_args()

    plt.rcParams.update(SA_RC)

    print(f"DATASET: {args.data_name}")
    ctx = load_material_model(args.data_name, args.seed)
    model = ctx['model']
    scaler_X, scaler_y = ctx['scaler_X'], ctx['scaler_y']
    device = ctx['device']
    X_train_denorm = ctx['X_train_denorm']
    y_train_denorm = ctx['y_train_denorm']
    feat_names = ctx['feat_names']
    output_name = ctx['output_name']
    savepath = ctx['savepath']

    n_features = X_train_denorm.shape[1]
    grid_res = args.grid_res

    # Feature axis ranges: use the analysis domain [0.1, 0.9] in normalized space
    # (consistent with material_KAN_analyze.py section 9), denormalized per feature.
    x_max_denorm = scaler_X.inverse_transform([[0.9] * n_features])[0]
    x_min_denorm = scaler_X.inverse_transform([[0.1] * n_features])[0]
    lin_per_feat = [np.linspace(x_min_denorm[i], x_max_denorm[i], grid_res)
                    for i in range(n_features)]

    # Other features fixed at training mean (denormalized)
    fixed_denorm = np.mean(X_train_denorm, axis=0)

    pairs = list(itertools.combinations(range(n_features), 2))
    print(f"🧮 {n_features} features -> {len(pairs)} pairwise 3D surfaces.")

    out_dir = os.path.join(savepath, "surfaces_3d")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Combined grid of all pairs ----
    n_cols = min(3, len(pairs))
    n_rows = (len(pairs) + n_cols - 1) // n_cols
    fig_all = plt.figure(figsize=(5.2 * n_cols, 4.4 * n_rows))

    for p, (f1_idx, f2_idx) in enumerate(pairs):
        f1_name, f2_name = feat_names[f1_idx], feat_names[f2_idx]
        X1, X2, Z = predict_surface(model, scaler_X, scaler_y, device,
                                    f1_idx, f2_idx,
                                    lin_per_feat[f1_idx], lin_per_feat[f2_idx],
                                    fixed_denorm)

        # ---- Individual figure ----
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X1, X2, Z, cmap='RdYlBu_r', alpha=0.9,
                               rstride=1, cstride=1, linewidth=0, antialiased=True)
        if not args.no_scatter:
            ax.scatter(X_train_denorm[:, f1_idx], X_train_denorm[:, f2_idx],
                       y_train_denorm.ravel(), c='black', s=6, alpha=0.25,
                       depthshade=True, label='Training data')
        cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.12)
        cbar.set_label(output_name, rotation=270, labelpad=16)
        _style_3d_axes(ax, f1_name, f2_name, output_name)

        # sanitize only the leaf filename (keep the directory path intact)
        leaf = f"{args.data_name}_surface_{f1_name}__{f2_name}"
        leaf = "".join(c if (c.isalnum() or c in "._-") else "_" for c in leaf)
        base = os.path.join(out_dir, leaf)
        fig.savefig(base + ".png", dpi=300)
        fig.savefig(base + ".svg", format='svg')
        if args.eps:
            fig.savefig(base + ".eps", format='eps')
        plt.close(fig)

        # ---- Subplot on the combined figure ----
        ax_a = fig_all.add_subplot(n_rows, n_cols, p + 1, projection='3d')
        ax_a.plot_surface(X1, X2, Z, cmap='RdYlBu_r', alpha=0.9,
                          rstride=1, cstride=1, linewidth=0, antialiased=True)
        if not args.no_scatter:
            ax_a.scatter(X_train_denorm[:, f1_idx], X_train_denorm[:, f2_idx],
                         y_train_denorm.ravel(), c='black', s=4, alpha=0.2, depthshade=True)
        _style_3d_axes(ax_a, f1_name, f2_name, output_name)
        print(f"   ✅ {f1_name} × {f2_name}")

    fig_all.tight_layout()
    combined_base = os.path.join(out_dir, f"{args.data_name}_surfaces_3d_all")
    fig_all.savefig(combined_base + ".png", dpi=200)
    fig_all.savefig(combined_base + ".svg", format='svg')
    if args.eps:
        fig_all.savefig(combined_base + ".eps", format='eps')
    plt.close(fig_all)

    print(f"\n📊 Saved {len(pairs)} individual surfaces + combined grid to:\n   {out_dir}")


if __name__ == "__main__":
    main()
