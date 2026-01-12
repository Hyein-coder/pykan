import argparse
import os
import json
import joblib
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import itertools  # <--- [IMPORTANT] Added for multi-feature combinations
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import yaml
from kan.custom_processing import remove_outliers_iqr


# ==========================================
# [FIX] Register Python Tuple for YAML Loading
# ==========================================
def tuple_constructor(loader, node):
    return tuple(loader.construct_sequence(node))


yaml.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor, Loader=yaml.SafeLoader)
try:
    yaml.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor, Loader=yaml.Loader)
except AttributeError:
    pass

# ==========================================
# Import your wrapper and function ZOO
from github.workflows.Hyein.toy_KAN_sweep import KANRegressor
from kan.experiments.analysis import find_indices_sign_revert


def main():
    parser = argparse.ArgumentParser(description="Run SHAP and Sobol analysis for a specific dataset.")
    parser.add_argument("data_name", type=str, nargs='?', default="CO2RRNPV",
                        help="The name of the dataset")

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

    print(f"DATASET: {data_name}")

    if not os.path.exists(scaler_x_path) or not os.path.exists(scaler_y_path):
        print("❌ Error: Scalers not found.")
        return

    scaler_X = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_wrapper = KANRegressor(device=device)

    try:
        model_wrapper.load_model(ckpt_path)
        model = model_wrapper.model
        print("✅ KAN Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # ==========================================
    # 2. Regenerate Data
    # ==========================================
    if not os.path.exists(filepath):
        print(f"❌ Error: Data file not found at {filepath}")
        return

    filedata = pd.read_csv(filepath)
    name_X = filedata.columns[:-1].tolist()
    name_y = filedata.columns[-1]
    df_in = filedata[name_X]
    df_out = filedata[[name_y]]

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

    dataset = {
        'train_input': torch.tensor(X_train_norm, dtype=torch.float32, device=device),
        'train_label': torch.tensor(y_train_denorm, dtype=torch.float32, device=device).reshape(-1, 1),
        'test_input': torch.tensor(X_test_norm, dtype=torch.float32, device=device),
        'test_label': torch.tensor(y_test_denorm, dtype=torch.float32, device=device).reshape(-1, 1)
    }

    # ==========================================
    # 2.5 Plot Input vs Output
    # ==========================================
    pred_y_norm = model(dataset['train_input']).detach().cpu().numpy()
    try:
        pred_y = scaler_y.inverse_transform(pred_y_norm)
    except ValueError:
        pred_y = pred_y_norm

    n_features = X_train_denorm.shape[1]
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols

    fig_io, axs_io = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), constrained_layout=True)
    axs_io = axs_io.flatten()

    for i in range(n_features):
        ax = axs_io[i]
        ax.scatter(X_train_denorm[:, i], y_train_denorm, alpha=0.5, c='gray', s=15, label='Ground Truth')
        ax.scatter(X_train_denorm[:, i], pred_y, alpha=0.5, c='red', s=15, label='Prediction')
        feature_label = feat_names[i] if feat_names and i < len(feat_names) else f"Feature {i}"
        ax.set_xlabel(feature_label)
        ax.set_ylabel("Output y")
        ax.set_title(f"{feature_label} vs Output")
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    for i in range(n_features, len(axs_io)):
        axs_io[i].axis('off')
    plt.suptitle(f"Input vs Output Analysis: {data_name}", fontsize=14)
    plt.savefig(os.path.join(savepath, f"{data_name}_input_vs_output.png"), dpi=300)
    plt.show()

    model.forward(dataset['train_input'])
    scores_tot = model.feature_score.detach().cpu().numpy()
    model.plot()
    plt.savefig(os.path.join(savepath, f"{data_name}_model.png"))

    # ==========================================
    # 3. Inflection Point Analysis (Layer 0)
    # ==========================================
    print("\n🔍 Analyzing Inflection Points in Layer 0...")
    l = 0
    act = model.act_fun[l]
    ni, no = act.coef.shape[:2]
    coef = act.coef.tolist()
    depth = len(model.act_fun)
    inflection_points_per_input = []

    fig, axs = plt.subplots(nrows=no, ncols=ni, squeeze=False, figsize=(max(2.5 * ni, 6), max(2.5 * no, 3.5)),
                            constrained_layout=True)

    for i in range(ni):
        feature_inflections_all = []
        for j in range(no):
            ax = axs[j, i]
            inputs = model.spline_preacts[l][:, j, i].cpu().detach().numpy()
            outputs = model.spline_postacts[l][:, j, i].cpu().detach().numpy()
            coef_node = coef[i][j]
            num_knot = act.grid.shape[1]
            spline_radius = int((num_knot - len(coef_node)) / 2)

            rank = np.argsort(inputs)
            ax.plot(inputs[rank], outputs[rank], marker='o', ms=2, lw=1)
            ax2 = ax.twinx()

            slope = [x - y for x, y in zip(coef_node[1:], coef_node[:-1])]
            slope_2nd = [(x - y)*10 for x, y in zip(slope[1:], slope[:-1])]
            bar_width = (act.grid[i, 1:] - act.grid[i, :-1]).mean().item() / 2  # Approx width

            # Plot Slope
            ax2.bar(act.grid[i, spline_radius:-(spline_radius + 1)].cpu(), slope,
                    width=bar_width, align='center', color='r', alpha=0.3, label='Slope')
            if depth == 1:
                ax2.bar(act.grid[i, spline_radius+1:-(spline_radius + 1)] + bar_width/3, slope_2nd,
                        width=bar_width, align='edge', color='g', alpha=0.3, label='2nd Slope')

            if depth == 1:
                idx_revert = find_indices_sign_revert(slope_2nd)
                idx_revert = [ir + 1 for ir in idx_revert]
            elif depth == 2:
                idx_revert = find_indices_sign_revert(slope)
            else:
                idx_revert = None

            if idx_revert:
                for ir in idx_revert:
                    inflection_val = act.grid[i, spline_radius + ir].item()
                    feature_inflections_all.append(inflection_val)
                    ax.axvline(x=inflection_val, color='g', linestyle='--', alpha=0.7)

            ax.set_title(f'in {i} -> out {j}', fontsize=9)

        feature_inflections = sorted(set(feature_inflections_all))
        inflection_points_per_input.append(feature_inflections)

    plt.savefig(os.path.join(savepath, f"{data_name}_activations_L{l}.png"))
    plt.show()

    # ==========================================
    # 4. Multi-Feature Range-Based Slicing
    # ==========================================

    # Settings
    NUM_FEATURES_TO_COMBINE = 2
    MIN_NON_EMPTY_INTERVALS = 2

    # Sort features by global score (Highest -> Lowest)
    sorted_feat_indices = np.argsort(scores_tot)[::-1]

    selected_features_data = []  # List of dicts: {'index', 'masks', 'labels'}

    print(f"\n🔍 Searching for top {NUM_FEATURES_TO_COMBINE} features that split data into valid ranges...")

    for mask_idx in sorted_feat_indices:
        if len(selected_features_data) >= NUM_FEATURES_TO_COMBINE:
            break

        feat_name = feat_names[mask_idx]
        raw_ips = inflection_points_per_input[mask_idx]
        valid_ips = [ip for ip in raw_ips if ip is not None and 0.1 < ip < 0.9]
        unique_ips = sorted(list(set([round(ip, 3) for ip in valid_ips])))

        if len(unique_ips) == 0:
            continue

        mask_interval = [0.1] + unique_ips + [0.9]
        x_mask_data = dataset['train_input'][:, mask_idx]

        current_feat_masks = []
        current_feat_labels = []

        for lb, ub in zip(mask_interval[:-1], mask_interval[1:]):
            m = ((x_mask_data > lb) & (x_mask_data <= ub))
            current_feat_masks.append(m)
            current_feat_labels.append(f'{lb:.2f}<x{mask_idx}<{ub:.2f}')

        non_empty_count = sum([1 for m in current_feat_masks if torch.any(m)])

        if non_empty_count >= MIN_NON_EMPTY_INTERVALS:
            print(f"   ✅ Selected Feature {mask_idx} ({feat_name}): Found {non_empty_count} active intervals")
            selected_features_data.append({
                'index': mask_idx,
                'masks': current_feat_masks,
                'labels': current_feat_labels
            })

    # Combine Masks (Cartesian Product)
    final_masks = []
    final_labels = []

    if not selected_features_data:
        print("⚠️ Warning: No valid splitting features found. Defaulting to top feature (All Range).")
        fallback_idx = sorted_feat_indices[0]
        x_mask_data = dataset['train_input'][:, fallback_idx]
        final_masks = [(x_mask_data > -np.inf)]
        final_labels = [f"All Range"]
        selected_features_indices = [fallback_idx]
    else:
        print(f"\n🔗 Combining intervals from {len(selected_features_data)} features...")
        selected_features_indices = [d['index'] for d in selected_features_data]

        lists_of_masks = [d['masks'] for d in selected_features_data]
        lists_of_labels = [d['labels'] for d in selected_features_data]

        for combined_mask_tuple, combined_label_tuple in zip(itertools.product(*lists_of_masks),
                                                             itertools.product(*lists_of_labels)):

            combined_m = combined_mask_tuple[0]
            for m in combined_mask_tuple[1:]:
                combined_m = combined_m & m

            combined_l = " & ".join(combined_label_tuple)

            if torch.any(combined_m):
                final_masks.append(combined_m)
                final_labels.append(combined_l)

        print(f"   -> Generated {len(final_masks)} non-empty combined regions.")

    # Calculate scores for final masks
    print(f"\n✂️ Scoring {len(final_masks)} regions...")
    scores_interval_norm = []

    for i, mask in enumerate(final_masks):
        x_tensor_masked = dataset['train_input'][mask, :]
        x_std = torch.std(x_tensor_masked, dim=0).detach().cpu().numpy()

        model.forward(x_tensor_masked)
        score_masked = model.feature_score.detach().cpu().numpy()
        score_norm = score_masked / (x_std + 1e-6)
        scores_interval_norm.append(score_norm)
        print(f"   Region {i + 1}: {mask.sum().item()} samples")

    # ==========================================
    # 4.5 Save Range Split Data
    # ==========================================
    split_data_savepath = os.path.join(savepath, f"{data_name}_range_split_data.pkl")
    print(f"\n💾 Saving range split data to: {split_data_savepath}")

    split_data = {
        'dataset': dataset,
        'masks': final_masks,  # UPDATED: Now saving the combined masks
        'labels': final_labels,  # UPDATED: Combined labels
        'selected_features_indices': selected_features_indices,  # UPDATED: List of indices
        'feature_names': feat_names,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }

    joblib.dump(split_data, split_data_savepath)

    # ==========================================
    # 5. Plot Range-Based Scores
    # ==========================================
    width = 0.08
    n_features_plot = scores_tot.shape[0]
    n_intervals = len(scores_interval_norm)

    # Dynamic figure size based on number of intervals
    fig, ax = plt.subplots(figsize=(max(10, n_intervals * 1.5), 6))
    x_positions = np.arange(n_intervals)
    max_score = max([max(s) for s in scores_interval_norm]) if scores_interval_norm else 1.0

    for feat_idx in range(n_features_plot):
        feat_scores = [s[feat_idx] for s in scores_interval_norm]
        offset = (feat_idx - n_features_plot / 2) * width + width / 2
        ax.bar(x_positions + offset, feat_scores, width, label=f"x{feat_idx}: {feat_names[feat_idx]}")

    ax.set_xticks(x_positions)
    # Truncate labels if too long for the plot
    short_labels = [l if len(l) < 30 else l[:15] + "..." + l[-10:] for l in final_labels]
    ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)

    ax.set_ylabel("Normalized Attribution Score")
    # Join indices for the filename/title
    indices_str = "_".join(map(str, selected_features_indices))
    ax.set_title(f"Feature Importance per Range (Features {indices_str})")
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize='small')
    ax.set_ylim(0, max_score * 1.2)
    plt.tight_layout()

    plot_path_score = os.path.join(savepath, f"{data_name}_scores_interval_combined.png")
    plt.savefig(plot_path_score)
    plt.show()
    print(f"📊 Range-based score plot saved to: {plot_path_score}")

    # ==========================================
    # 6. Global Scores
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

    # ==========================================
    # 8. Plot Input vs Output (Colored by Combined Range)
    # ==========================================
    print("\n📈 Plotting Input vs Output (Original Data Colored by Range)...")

    pred_y_norm = model(dataset['train_input']).detach().cpu().numpy()
    try:
        pred_y = scaler_y.inverse_transform(pred_y_norm)
    except ValueError:
        pred_y = pred_y_norm

    cmap = plt.get_cmap('tab10')
    # Use final_masks here
    colors = [cmap(k % 10) for k in range(len(final_masks))]

    n_features = X_train_denorm.shape[1]
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols

    fig_io, axs_io = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), constrained_layout=True)
    axs_io = axs_io.flatten()

    for i in range(n_features):
        ax = axs_io[i]
        # Background: Predictions
        ax.scatter(X_train_denorm[:, i], pred_y, alpha=0.15, c='gray', s=20, label='Prediction')

        # Foreground: Original Data Colored by Slice
        for m_idx, (mask, label) in enumerate(zip(final_masks, final_labels)):
            if torch.is_tensor(mask):
                mask_np = mask.cpu().numpy().astype(bool)
            else:
                mask_np = mask

            if mask_np.sum() > 0:
                x_seg = X_train_denorm[mask_np, i]
                y_seg_original = y_train_denorm[mask_np]

                # Truncate label for legend
                lbl_short = label if len(label) < 20 else label[:8] + "..." + label[-8:]
                ax.scatter(x_seg, y_seg_original, alpha=0.9, s=25,
                           color=colors[m_idx], label=lbl_short)

        feature_label = f"x{i}: {feat_names[i]}" if feat_names and i < len(feat_names) else f"Feature {i}"
        ax.set_xlabel(feature_label)
        ax.set_ylabel("Output y")

        if i == 0:
            ax.legend(loc='upper left', fontsize=6, markerscale=1.5)
        ax.grid(True, alpha=0.3)

    for i in range(n_features, len(axs_io)):
        axs_io[i].axis('off')

    plt.suptitle(f"Input vs Output (Colored by {indices_str}): {data_name}", fontsize=14)
    plot_path_io = os.path.join(savepath, f"{data_name}_in_out_range_combined.png")
    plt.savefig(plot_path_io, dpi=300)
    plt.show()
    print(f"📊 Colored Input-Output plots saved to: {plot_path_io}")


if __name__ == "__main__":
    main()