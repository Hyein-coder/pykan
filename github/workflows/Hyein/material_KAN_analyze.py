import argparse
import os
import json
import joblib
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import yaml  # <--- [NEW] Import YAML
from kan.custom_processing import remove_outliers_iqr

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
from github.workflows.Hyein.toy_KAN_sweep import KANRegressor
from kan.experiments.analysis import find_index_sign_revert


def main():
    parser = argparse.ArgumentParser(description="Run SHAP and Sobol analysis for a specific dataset.")
    parser.add_argument("data_name", type=str, nargs='?', default="P3HT",
                        help="The name of the dataset (default: P3HT)")

    args = parser.parse_args()
    data_name = args.data_name
    # ==========================================
    # 1. Setup Paths & Load Model/Scalers
    # ==========================================
    root_dir = os.path.join(os.getcwd(), 'github', 'workflows', 'Hyein')
    filepath = os.path.join(root_dir, "data", f"{data_name}.csv")
    savepath = os.path.join(root_dir, "material_kan_models", data_name)

    ckpt_path = os.path.join(savepath, f'{data_name}_best_kan_model')
    scaler_x_path = os.path.join(savepath, f'{data_name}_mlp_scaler_X.pkl')
    scaler_y_path = os.path.join(savepath, f'{data_name}_mlp_scaler_y.pkl')

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
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"❌ Error: Data file not found at {filepath}")
        return

    filedata = pd.read_csv(filepath)
    name_X = filedata.columns[:-1].tolist()
    name_y = filedata.columns[-1]
    df_in = filedata[name_X]
    df_out = filedata[[name_y]]
    print(f"TARGET: {name_y}")

    df_in_final, df_out_final = remove_outliers_iqr(df_in, df_out)

    removed_count = len(df_in) - len(df_in_final)
    print(f"# of data after removing outliers: {len(df_in_final)} ({removed_count} removed)")

    X = df_in_final[name_X].values
    y = df_out_final[name_y].values.reshape(-1, 1)

    X_temp_denorm, X_test_denorm, y_temp_denorm, y_test_denorm = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_denorm, X_val_denorm, y_train_denorm, y_val_denorm = train_test_split(X_temp_denorm, y_temp_denorm,
                                                                                  test_size=0.2, random_state=42)
    print(f"Train/Validation/Test : {len(X_train_denorm)} / {len(X_val_denorm)} / {len(X_test_denorm)}")

    feat_names = name_X

    X_train_norm = scaler_X.fit_transform(X_train_denorm)
    y_train_norm = scaler_y.fit_transform(y_train_denorm)
    X_test_norm = scaler_X.fit_transform(X_test_denorm)
    y_test_norm = scaler_y.fit_transform(y_test_denorm)

    # Create dataset dict (needed for forward pass logic sometimes)
    dataset = {
        'train_input': torch.tensor(X_train_norm, dtype=torch.float32, device=device),
        'train_label': torch.tensor(y_train_denorm, dtype=torch.float32, device=device).reshape(-1, 1),
        'test_input': torch.tensor(X_test_norm, dtype=torch.float32, device=device),
        'test_label': torch.tensor(y_test_denorm, dtype=torch.float32, device=device).reshape(-1, 1)
        # Label scaling optional here
    }

    # Run forward pass once to populate internals (splines, activations)
    model.forward(dataset['train_input'])
    scores_tot = model.feature_score.detach().cpu().numpy()  # Global scores

    # ==========================================
    # 3. Inflection Point Analysis (Layer 0)
    # ==========================================
    print("\n🔍 Analyzing Inflection Points in Layer 0...")

    l = 0  # Analyze Layer 0
    act = model.act_fun[l]
    ni, no = act.coef.shape[:2]
    coef = act.coef.tolist()
    depth = len(model.act_fun)

    inflection_points_per_input = []  # Store list of inflection points for each input feature

    fig, axs = plt.subplots(nrows=no, ncols=ni, squeeze=False,
                            figsize=(max(2.5 * ni, 6), max(2.5 * no, 3.5)),
                            constrained_layout=True)

    for i in range(ni):  # For each input feature
        feature_inflections = []
        for j in range(no):  # For each output node of the layer
            ax = axs[j, i]

            # 1. Get Data
            inputs = model.spline_preacts[l][:, j, i].cpu().detach().numpy()
            outputs = model.spline_postacts[l][:, j, i].cpu().detach().numpy()

            coef_node = coef[i][j]
            num_knot = act.grid.shape[1]
            spline_radius = int((num_knot - len(coef_node)) / 2)

            # 2. Plot Activations
            rank = np.argsort(inputs)
            ax.plot(inputs[rank], outputs[rank], marker='o', ms=2, lw=1, label='Activations')

            # 3. Plot Coefficients & Slope
            ax2 = ax.twinx()
            # Plot coefficients (control points)
            ax2.scatter(act.grid[i, spline_radius:-spline_radius].cpu(), coef_node,
                        s=20, color='white', edgecolor='k', label='Coefficients')

            # Calculate Slope
            slope = [x - y for x, y in zip(coef_node[1:], coef_node[:-1])]
            slope_2nd = [(x - y)*10 for x, y in zip(slope[1:], slope[:-1])]
            bar_width = (act.grid[i, 1:] - act.grid[i, :-1]).mean().item() / 2  # Approx width

            # Plot Slope
            ax2.bar(act.grid[i, spline_radius:-(spline_radius + 1)].cpu(), slope,
                    width=bar_width, align='center', color='r', alpha=0.3, label='Slope')
            if depth == 1:
                ax2.bar(act.grid[i, spline_radius:-(spline_radius + 2)], slope_2nd,
                        width=bar_width, align='edge', color='g', label='2nd Slope')

            ax.set_title(f'in {i} -> out {j}', fontsize=9)

            # 4. Find Inflection
            if depth == 1:
                idx_revert = find_index_sign_revert(slope_2nd)
            elif depth == 2:
                idx_revert = find_index_sign_revert(slope)
            else:
                print("Depth > 2 not supported yet.")
                idx_revert = None

            if idx_revert is not None:
                inflection_val = act.grid[i, spline_radius + idx_revert].item()
                feature_inflections.append(inflection_val)
                # Mark on plot
                ax.axvline(x=inflection_val, color='g', linestyle='--', alpha=0.7)

        inflection_points_per_input.append(feature_inflections)

    plt.suptitle(f"Layer {l} Activation Analysis: {data_name}", fontsize=12)
    plot_path_act = os.path.join(savepath, f"{data_name}_activations_L{l}.png")
    plt.savefig(plot_path_act)
    plt.show()
    print(f"📊 Activation analysis saved to: {plot_path_act}")

    # ==========================================
    # 4. Range-Based Attribution Scoring
    # ==========================================
    # We will pick the most significant input feature (highest global score) to slice
    # Or you can set `mask_idx` manually.

    # Let's pick the feature with the highest global score to analyze
    mask_idx = np.argmax(scores_tot)
    print(f"\n✂️ Slicing data based on Feature {mask_idx} (Highest Importance)...")

    # Get valid inflection points for this feature (within 0.1~0.9 range)
    raw_ips = inflection_points_per_input[mask_idx]
    valid_ips = [ip for ip in raw_ips if ip is not None and 0.1 < ip < 0.9]

    # Remove duplicates and sort
    unique_ips = sorted(list(set([round(ip, 3) for ip in valid_ips])))

    # Define Intervals: [0.1, ip1, ip2, ..., 0.9]
    mask_interval = [0.1] + unique_ips + [0.9]
    print(f"   Intervals defined: {mask_interval}")

    # Create Masks
    x_mask = dataset['train_input'][:, mask_idx]
    masks = [((x_mask > lb) & (x_mask <= ub)) for lb, ub in zip(mask_interval[:-1], mask_interval[1:])]
    labels = [f'{lb:.2f} < x{mask_idx} <= {ub:.2f}' for lb, ub in zip(mask_interval[:-1], mask_interval[1:])]

    scores_interval_norm = []

    # Compute Scores per Interval
    for i, mask in enumerate(masks):
        if torch.any(mask):
            x_tensor_masked = dataset['train_input'][mask, :]

            # Standard deviation of input in this slice (used for normalization)
            x_std = torch.std(x_tensor_masked, dim=0).detach().cpu().numpy()

            # Forward pass on masked data to get local attribution
            model.forward(x_tensor_masked)
            score_masked = model.feature_score.detach().cpu().numpy()

            # Normalize score
            score_norm = score_masked / (x_std + 1e-6)
            scores_interval_norm.append(score_norm)
            print(f"   Interval {labels[i]}: {mask.sum().item()} samples")
        else:
            scores_interval_norm.append(np.zeros(scores_tot.shape))
            print(f"   Interval {labels[i]}: 0 samples (Skipping)")

    # ==========================================
    # 5. Plot Range-Based Scores
    # ==========================================
    width = 0.08
    n_features = scores_tot.shape[0]
    n_intervals = len(scores_interval_norm)

    fig, ax = plt.subplots(figsize=(max(8, n_intervals * 2), 5))

    # X-axis: Intervals
    x_positions = np.arange(n_intervals)

    # We want to show bars for EACH feature within each interval group
    # But usually, we want to see how feature importance changes across intervals.
    # Let's group by Interval on X-axis.

    max_score = max([max(s) for s in scores_interval_norm]) if scores_interval_norm else 1.0

    for feat_idx in range(n_features):
        # Extract score of this feature across all intervals
        feat_scores = [s[feat_idx] for s in scores_interval_norm]

        # Offset bars
        offset = (feat_idx - n_features / 2) * width + width / 2
        bars = ax.bar(x_positions + offset, feat_scores, width, label=f"{feat_names[feat_idx]}")
        # ax.bar_label(bars, fmt='%.2f', fontsize=7, padding=3)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=15, ha='center', fontsize=9)
    ax.set_ylabel("Normalized Attribution Score")
    ax.set_title(f"Feature Importance per Range (sliced by {feat_names[mask_idx]})")
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
    ax.set_ylim(0, max_score * 1.2)
    plt.tight_layout()

    plot_path_score = os.path.join(savepath, f"{data_name}_scores_interval_x{mask_idx}.png")
    plt.savefig(plot_path_score)
    plt.show()
    print(f"📊 Range-based score plot saved to: {plot_path_score}")


    model.forward(dataset['train_input'])
    scores_tot = model.feature_score.detach().cpu().numpy()  # Global scores

    # ==========================================
    # 6. Plot Global Scores
    # ==========================================
    fig_tot, ax_tot = plt.subplots(figsize=(5,5))

    positions = range(len(scores_tot))
    bars = ax_tot.bar(positions, scores_tot, color='skyblue', edgecolor='black')
    ax_tot.bar_label(bars, fmt='%.2f', padding=3)
    ax_tot.set_xticks(list(positions))  # Set positions first
    ax_tot.set_xticklabels(feat_names, rotation=15, ha='center')  # Then set text labels
    ax_tot.set_ylabel("Global Attribution Score")
    ax_tot.set_title(f"Feature Importance: {data_name}")

    # Save & Show
    plot_path_tot = os.path.join(savepath, f"{data_name}_scores_global.png")
    plt.tight_layout()
    plt.savefig(plot_path_tot, dpi=300)
    plt.show()

    # Ensure scores_tot is a flat 1D array
    if len(scores_tot.shape) > 1:
        scores_tot = scores_tot.flatten()

    df_scores = pd.DataFrame({
        'Feature': feat_names,
        'Global_Attribution_Score': scores_tot
    })

    # Optional: Sort by importance
    df_scores = df_scores.sort_values(by='Global_Attribution_Score', ascending=False)

    # 3. Save to CSV
    score_csv_path = os.path.join(savepath, f'{data_name}_global_attribution_scores.csv')
    df_scores.to_csv(score_csv_path, index=False)

    # ==========================================
    # 7. Parity Plot
    # ==========================================

    # y_pred_test_norm = model.predict(dataset['test_input'])
    # r2_test = r2_score(y_test_norm, y_pred_test_norm)
    #
    # plt.figure(figsize=(6, 6))
    # y_pred_test = scaler_y.inverse_transform(y_pred_test_norm.reshape(1, -1))
    # plt.scatter(y_test_denorm, y_pred_test, alpha=0.6, edgecolors='k', s=30, label='Test Data')
    #
    # # Perfect fit line
    # min_val = min(y_test_denorm.min(), y_pred_test.min())
    # max_val = max(y_test_denorm.max(), y_pred_test.max())
    # plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')
    #
    # plt.title(f"Parity Plot: {data_name} (R2={r2_test:.4f})")
    # plt.xlabel("Actual Value")
    # plt.ylabel("Predicted Value")
    # plt.legend()
    # plt.grid(True, linestyle='--', alpha=0.5)
    #
    # plot_path = os.path.join(savepath, f"{data_name}_parity_plot.png")
    # plt.savefig(plot_path, dpi=300)
    # plt.show()
    # print(f"📊 Parity plot saved to: {plot_path}")

if __name__ == "__main__":
    main()