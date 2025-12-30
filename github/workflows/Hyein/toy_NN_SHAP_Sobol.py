import argparse
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from SALib.sample import sobol as saltelli
from SALib.analyze import sobol
from github.workflows.Hyein.toy_analytic_SHAP_Sobol import FUNCTION_ZOO


def main():
    # ==========================================
    # 2. Argument Parsing & Setup
    # ==========================================
    parser = argparse.ArgumentParser(description="Analyze MLP models trained on analytical functions.")
    parser.add_argument("func_name", type=str, nargs='?', default="original",
                        choices=FUNCTION_ZOO.keys(),
                        help="Choose a function: " + ", ".join(FUNCTION_ZOO.keys()))

    args = parser.parse_args()
    case_name = args.func_name

    print(f"🚀 Running MLP Analysis for case: '{case_name}'")

    # Paths
    # Note: Adjust 'root_dir' if your saved models are elsewhere.
    # Currently pointing to: github/workflows/Hyein/analytical_results/<case>/nn_models
    root_dir = os.path.join(os.getcwd(), 'github', 'workflows', 'Hyein', "analytical_results", case_name)
    model_dir = os.path.join(root_dir, "nn_models")

    # Check if model directory exists
    if not os.path.exists(model_dir):
        print(f"❌ Error: Model directory not found at {model_dir}")
        print("   Please run the training script (tune_analytical_mlp.py) first.")
        return

    # ==========================================
    # 3. Load Model and Scalers
    # ==========================================
    try:
        model_path = os.path.join(model_dir, f'{case_name}_best_mlp_model.pkl')
        scaler_X_path = os.path.join(model_dir, f'{case_name}_mlp_scaler_X.pkl')
        scaler_y_path = os.path.join(model_dir, f'{case_name}_mlp_scaler_y.pkl')

        mlp_model = joblib.load(model_path)
        scaler_X = joblib.load(scaler_X_path)
        scaler_y = joblib.load(scaler_y_path)
        print(f"✅ Loaded model and scalers from {model_dir}")
    except FileNotFoundError as e:
        print(f"❌ Error loading files: {e}")
        return

    # ==========================================
    # 4. Define Prediction Wrapper
    # ==========================================
    # This wrapper handles the translation between "Physical World" (SHAP/Sobol) and "Neural Net World" (Normalized)
    def mlp_wrapper(X_raw):
        # 1. Scale Input: Physical -> Normalized
        X_norm = scaler_X.transform(X_raw)

        # 2. Predict
        y_pred_norm = mlp_model.predict(X_norm)

        # 3. Inverse Scale Output: Normalized -> Physical
        # Reshape is needed because scaler expects 2D array
        y_pred_raw = scaler_y.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()

        return y_pred_raw

    # Load Config
    config = FUNCTION_ZOO[case_name]
    bounds = config["bounds"]
    feature_names = config["names"]
    n_features = len(bounds)

    # Output Directory for Analysis
    analysis_savepath = os.path.join(root_dir, "nn_analysis")
    os.makedirs(analysis_savepath, exist_ok=True)

    # ==========================================
    # 5. Sobol Analysis
    # ==========================================
    print("\n[1/2] 🎲 Running Sobol Analysis on MLP...")

    problem = {
        'num_vars': n_features,
        'names': feature_names,
        'bounds': bounds
    }

    # Generate samples (Physical Domain)
    X_sobol = saltelli.sample(problem, 1024, calc_second_order=True)

    # Predict using Wrapper
    Y_sobol = mlp_wrapper(X_sobol)

    # Analyze
    Si = sobol.analyze(problem, Y_sobol, calc_second_order=True)

    # Save & Plot
    results_df = pd.DataFrame({
        'Feature': feature_names,
        'Total_Effect (ST)': Si['ST'],
        'First_Order (S1)': Si['S1']
    }).sort_values(by='Total_Effect (ST)', ascending=False)

    print(results_df)
    results_df.to_csv(os.path.join(analysis_savepath, "mlp_sobol_indices.csv"), index=False)

    plt.figure(figsize=(8, 5))
    plt.title(f"Sobol Sensitivity (MLP Model) - {case_name}")
    plt.barh(results_df['Feature'][::-1], results_df['Total_Effect (ST)'][::-1],
             color='lightcoral')  # Different color to distinguish
    plt.xlabel("Total Effect Index")
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_savepath, "mlp_sobol_plot.png"), dpi=150)
    plt.close()

    # ==========================================
    # 6. SHAP Analysis
    # ==========================================
    print("\n[2/2] 🔍 Running SHAP Analysis on MLP...")

    # 1. Background (Physical Domain)
    X_bg = np.random.uniform(
        low=[b[0] for b in bounds],
        high=[b[1] for b in bounds],
        size=(100, n_features)
    )

    # 2. Test Data (Physical Domain)
    X_test = np.random.uniform(
        low=[b[0] for b in bounds],
        high=[b[1] for b in bounds],
        size=(500, n_features)
    )

    # Use the wrapper for the explainer
    explainer = shap.KernelExplainer(mlp_wrapper, X_bg)
    shap_values = explainer.shap_values(X_test)

    # Save Raw SHAP
    pd.DataFrame(shap_values, columns=feature_names).to_csv(
        os.path.join(analysis_savepath, "mlp_shap_raw.csv"), index=False
    )

    # Calculate Mean Absolute SHAP
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Abs_SHAP': mean_abs_shap
    }).sort_values(by='Mean_Abs_SHAP', ascending=False)

    shap_importance_df.to_csv(os.path.join(analysis_savepath, "mlp_shap_mean_abs.csv"), index=False)

    # Plot 1: Dot Plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.savefig(os.path.join(analysis_savepath, "mlp_shap_dot_plot.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Bar Plot
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.savefig(os.path.join(analysis_savepath, "mlp_shap_bar_plot.png"), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Analysis Complete! Results saved in: {analysis_savepath}")


if __name__ == "__main__":
    main()