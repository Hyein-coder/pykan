import argparse
import os
os.environ["OMP_NUM_THREADS"] = "1"
import joblib
import torch
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from SALib.sample import sobol as sample_sobol
from SALib.analyze import sobol as analyze_sobol
from sklearn.metrics import pairwise_distances_argmin_min

# Assuming this custom module exists in your environment
from kan.custom_processing import remove_outliers_iqr
from github.workflows.Hyein.toy_NN_SHAP_Sobol import plot_custom_bars

def main():
    # ==========================================
    # 0. Argument Parsing
    # ==========================================
    parser = argparse.ArgumentParser(description="Run SHAP and Sobol analysis for a specific dataset.")
    parser.add_argument("data_name", type=str, nargs='?', default="AgNP",
                        help="The name of the dataset (default: P3HT)")

    # Optional: Add flags for specific settings if needed
    # parser.add_argument("--nsamples", type=int, default=100, help="Number of samples for SHAP")

    args = parser.parse_args()
    data_name = args.data_name

    print(f"🚀 Starting analysis for: {data_name}")

    # ==========================================
    # 1. Data Loading & Preprocessing
    # ==========================================
    root_dir = os.path.join(os.getcwd(), 'github', 'workflows', 'Hyein')
    filepath = os.path.join(root_dir, "data", f"{data_name}.csv")
    savepath = os.path.join(root_dir, "material_nn_models", data_name)

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
    # X_train_denorm, X_val_denorm, y_train_denorm, y_val_denorm = train_test_split(X_temp_denorm, y_temp_denorm,
    #                                                                               test_size=0.2, random_state=42)

    # print(f"Train/Validation/Test : {len(X_train_denorm)} / {len(X_val_denorm)} / {len(X_test_denorm)}")

    feature_names = name_X

    # ==========================================
    # 2. Load Models & Scalers
    # ==========================================
    model_path = os.path.join(savepath, f'{data_name}_best_mlp_model.pkl')

    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        return

    loaded_model = joblib.load(model_path)
    scaler_X = joblib.load(os.path.join(savepath, f'{data_name}_mlp_scaler_X.pkl'))
    scaler_y = joblib.load(os.path.join(savepath, f'{data_name}_mlp_scaler_y.pkl'))

    # Apply scaling
    X_temp_norm = scaler_X.transform(X_temp_denorm)
    X_test_norm = scaler_X.transform(X_test_denorm)

    # ==========================================
    # 3. SHAP Analysis
    # ==========================================
    num_data = len(X_temp_norm)
    num_shap_sample = 1000

    if num_shap_sample < num_data:
        X_train_summary = shap.kmeans(X_temp_norm, num_shap_sample)
    else:
        X_train_summary = X_temp_norm

    explainer = shap.KernelExplainer(loaded_model.predict, X_train_summary)

    shap_values = explainer.shap_values(X_test_norm)

    # [NEW] Save SHAP Values to CSV
    # 1. Raw SHAP values (useful for reproducing dot plots)
    shap_raw_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_raw_path = os.path.join(savepath, f'{data_name}_shap_values_raw.csv')
    shap_raw_df.to_csv(shap_raw_path, index=False)

    # 2. SHAP Summary/Importance (Mean Absolute Value - mirrors the Bar Plot)
    shap_summary_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Abs_SHAP': np.abs(shap_values).mean(axis=0)
    })

    shap_summary_path = os.path.join(savepath, f'{data_name}_shap_importance.csv')
    shap_summary_df.to_csv(shap_summary_path, index=False)

    # Plot 1: Bar Plot
    plot_custom_bars(
        names=shap_summary_df['Feature'],
        values=shap_summary_df['Mean_Abs_SHAP'],
        title=f"SHAP Global Importance (MLP) - {data_name}",
        ylabel="mean(|SHAP value|)",
        savepath=os.path.join(savepath, f"{data_name}_mlp_shap_bar_plot.png"),
        color='thistle'
    )

    # Plot 2: Dot Plot
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_test_norm,
        feature_names=feature_names,
        show=True
    )
    plt.savefig(os.path.join(savepath, f'{data_name}_shap_dot_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ==========================================
    # 4. SALib (Sobol) Analysis
    # ==========================================
    n_features = X_temp_norm.shape[1]

    # [Dynamic Bounds Adjustment]
    # We use the actual feature_range from the loaded scaler.
    # If scaler was (-1, 1), it uses that. If (0, 1), it uses that.
    scaler_min, scaler_max = scaler_X.feature_range
    print(f"ℹ️  Sobol Bounds set to scaler range: [{scaler_min}, {scaler_max}]")

    problem = {
        'num_vars': n_features,
        'names': feature_names,
        'bounds': [[scaler_min, scaler_max]] * n_features
    }

    N = 1024
    X_sobol = sample_sobol.sample(problem, N, calc_second_order=True)

    print(f"⏳ Running Sobol analysis on {X_sobol.shape[0]} samples...")
    Y_sobol = loaded_model.predict(X_sobol)

    Si = analyze_sobol.analyze(problem, Y_sobol, calc_second_order=True)

    # Results
    print("\n[Sobol Analysis Result]")
    results_df = pd.DataFrame({
        'Feature': feature_names,
        'Total_Effect (ST)': Si['ST'],
        'First_Order (S1)': Si['S1']
    })

    print(results_df.sort_values(by='First_Order (S1)', ascending=False))

    # [NEW] Save Sobol Indices to CSV
    sobol_csv_path = os.path.join(savepath, f"{data_name}_sobol_indices.csv")
    results_df.to_csv(sobol_csv_path, index=False)

    # Plot
    plot_custom_bars(
        names=results_df['Feature'],
        values=results_df['First_Order (S1)'],
        title=f"Sobol Sensitivity (MLP) - {data_name}",
        ylabel="First Order Index (S1)",
        savepath=os.path.join(savepath, f"{data_name}_mlp_sobol_plot.png"),
        color='bisque'
    )

if __name__ == "__main__":
    main()
