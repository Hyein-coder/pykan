import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from SALib.sample import sobol as saltelli
from SALib.analyze import sobol
from github.workflows.Hyein.toy_7_log_sum_factory import LOG_SUM_ZOO

# ==========================================
# 1. Define Your Functions Here
# ==========================================
# Format:
# 'name': {
#     'func': lambda x: ...,  (input x is a numpy array [x0, x1, ...])
#     'bounds': [[min, max], [min, max], ...],
#     'feature_names': ['x0', 'x1', ...]
# }

STANDARD_ZOO = {
    # Toy 3
    "original": {
        "func": lambda x: np.sin(2 * x[0]) + 5 * x[1],
        "bounds": [[-np.pi, np.pi], [-1, 1]],
        "names": ["Angle (x0)", "Linear (x1)"]
    },

    # Toy 5
    "mult_periodic": {
        "func": lambda x: x[1] * np.sin(2 * x[0]),
        "bounds": [[-np.pi, np.pi], [-1, 1]],
        "names": ["Angle (x0)", "Multiplier (x1)"]
    },

    # Toy 4
    "exponential": {
        "func": lambda x: np.exp(-2 * x[0]) + x[1],
        "bounds": [[-1, 1], [-1, 1]],
        "names": ["Exponent (x0)", "Linear (x1)"]
    },

    # Toy 6
    "logarithm": {
        "func": lambda x: np.log(20 * (x[0] + 1.2)) + x[1],
        "bounds": [[-1, 1], [-1, 1]],
        "names": ["Log (x0)", "Linear (x1)"]
    },

    # Toy 8
    "convolution": {
        "func": lambda x: x[0] ** 2 / (x[1] + 1.08) / 1.8,
        "bounds": [[-1, 1], [-1, 1]],
        "names": ["Convex (x0)", "Denominator (x1)"]
    }
}
FUNCTION_ZOO = {**STANDARD_ZOO, **LOG_SUM_ZOO}

def main():
    # ==========================================
    # 2. Argument Parsing
    # ==========================================
    parser = argparse.ArgumentParser(description="Analyze analytical functions.")
    parser.add_argument("func_name", type=str, nargs='?', default="original",
                        choices=FUNCTION_ZOO.keys(),
                        help="Choose a function: " + ", ".join(FUNCTION_ZOO.keys()))

    args = parser.parse_args()
    case_name = args.func_name

    print(f"🚀 Running Analysis for case: '{case_name}'")

    # Load config
    config = FUNCTION_ZOO[case_name]
    # Wrapper to handle batch inputs (N, D) -> (N, )
    # We use np.apply_along_axis to allow the lambda to work on rows
    model_func = lambda X: np.apply_along_axis(config["func"], 1, X)

    bounds = config["bounds"]
    feature_names = config["names"]
    n_features = len(bounds)

    # Save Path
    savepath = os.path.join(os.getcwd(), 'github', 'workflows', 'Hyein', "analytical_results", case_name)
    os.makedirs(savepath, exist_ok=True)

    # ==========================================
    # 3. Sobol Analysis
    # ==========================================
    print("\n[1/2] 🎲 Running Sobol Analysis...")

    problem = {
        'num_vars': n_features,
        'names': feature_names,
        'bounds': bounds
    }

    # Generate samples
    X_sobol = saltelli.sample(problem, 1024, calc_second_order=True)
    Y_sobol = model_func(X_sobol)

    # Analyze
    Si = sobol.analyze(problem, Y_sobol, calc_second_order=True)

    # Save & Plot
    results_df = pd.DataFrame({
        'Feature': feature_names,
        'Total_Effect (ST)': Si['ST'],
        'First_Order (S1)': Si['S1']
    }).sort_values(by='Total_Effect (ST)', ascending=False)

    print(results_df)
    results_df.to_csv(os.path.join(savepath, "sobol_indices.csv"), index=False)

    plt.figure(figsize=(8, 5))
    plt.title(f"Sobol Sensitivity - {case_name}")
    plt.barh(results_df['Feature'][::-1], results_df['Total_Effect (ST)'][::-1], color='skyblue')
    plt.xlabel("Total Effect Index")
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, "sobol_plot.png"), dpi=150)
    plt.close()

    # ==========================================
    # 4. SHAP Analysis
    # ==========================================
    print("\n[2/2] 🔍 Running SHAP Analysis...")

    # 1. Background (Random uniform within bounds)
    # Using 100 background samples is usually enough for analytical functions
    X_bg = np.random.uniform(
        low=[b[0] for b in bounds],
        high=[b[1] for b in bounds],
        size=(100, n_features)
    )

    # 2. Test Grid (Create a meshgrid to visualize the domain well)
    # We dynamically create a grid for N dimensions
    # For simplicity, we just take 1000 random points for SHAP to get a global view
    X_test = np.random.uniform(
        low=[b[0] for b in bounds],
        high=[b[1] for b in bounds],
        size=(500, n_features)
    )

    explainer = shap.KernelExplainer(model_func, X_bg)
    shap_values = explainer.shap_values(X_test)

    # Save
    pd.DataFrame(shap_values, columns=feature_names).to_csv(
        os.path.join(savepath, "shap_raw.csv"), index=False
    )

    # 2. [NEW] Calculate and Save Mean Absolute SHAP (Bar Plot Data)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Abs_SHAP': mean_abs_shap
    }).sort_values(by='Mean_Abs_SHAP', ascending=False)

    shap_importance_df.to_csv(os.path.join(savepath, "shap_mean_abs.csv"), index=False)

    # Plot Summary
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.savefig(os.path.join(savepath, "shap_dot_plot.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # [NEW] Plot 2: Bar Plot (Mean Absolute Values)
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names,
        plot_type="bar",  # <--- This creates the bar plot
        show=False
    )
    plt.savefig(os.path.join(savepath, "shap_bar_plot.png"), dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
