import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from SALib.sample import sobol as saltelli
from SALib.analyze import sobol
from github.workflows.Hyein.toy_7_log_sum_factory import LOG_SUM_ZOO
from github.workflows.Hyein.toy_8_convex_factory import CONVEX_ZOO

def plot_custom_bars(names, values, title, ylabel, savepath, color="skyblue", show=True):
    """
    Helper function to draw vertical bar plots in the specific requested style:
    - Vertical bars
    - Skyblue color with black edge
    - Values printed on top
    - Rotated x-axis labels
    """
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 6))

    # Create Vertical Bars
    bars = ax.bar(names, values, color=color, edgecolor='black', width=0.7)

    # Add number labels on top of bars
    # padding=3 adds a little space between the bar and the text
    ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=10)

    # Formatting
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)

    # Rotate x-axis labels slightly for readability
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha='center', fontsize=10)

    # Adjust Y-limit to make room for labels
    if len(values) > 0:
        ax.set_ylim(0, max(values) * 1.15)

    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()
    # print(f"   📊 Plot saved: {savepath}")

STANDARD_ZOO = {
    # Toy 3
    "original": {
        "func": lambda x: np.sin(2 * x[0]) + 5 * x[1],
        "bounds": [[-np.pi, np.pi], [-1, 1]],
        "names": ["Angle (x0)", "Linear (x1)"],
        "mask_idx": None,
        "mask_division": []
    },

    # Toy 5
    "mult_periodic": {
        "func": lambda x: x[1] * np.sin(2 * x[0]),
        "bounds": [[-np.pi, np.pi], [-1, 1]],
        "names": ["Angle (x0)", "Multiplier (x1)"],
        "mask_idx": None,
        "mask_division": []
    },

    # Toy 4
    "exponential": {
        "func": lambda x: np.exp(-2 * x[0]) + x[1],
        "bounds": [[-1, 1], [-1, 1]],
        "names": ["Exponent (x0)", "Linear (x1)"],
        "mask_idx": 0,
        "mask_division": [0.4]
    },

    # Toy 6
    "logarithm": {
        "func": lambda x: np.log(20 * (x[0] + 1.2)) + x[1],
        "bounds": [[-1, 1], [-1, 1]],
        "names": ["Log (x0)", "Linear (x1)"],
        "mask_idx": 0,
        "mask_division": [-0.8]
    },

    # Toy 8
    "convolution": {
        "func": lambda x: x[0] ** 2 / (x[1] + 1.08) / 1.8,
        "bounds": [[-1, 1], [-1, 1]],
        "names": ["Convex (x0)", "Denominator (x1)"],
        "mask_idx": None,
        "mask_division": []
    }
}
FUNCTION_ZOO = {**STANDARD_ZOO, **LOG_SUM_ZOO, **CONVEX_ZOO}

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

    # Plot (Custom Style)
    plot_custom_bars(
        names=results_df['Feature'],
        values=results_df['Total_Effect (ST)'],
        title=f"Sobol Sensitivity: {case_name}",
        ylabel="Total Effect Index (ST)",
        savepath=os.path.join(savepath, "sobol_plot.png"),
        color='bisque'
    )

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
    })

    shap_importance_df.to_csv(os.path.join(savepath, "shap_mean_abs.csv"), index=False)

    # Plot Summary
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.savefig(os.path.join(savepath, "shap_dot_plot.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # [NEW] Plot 2: Bar Plot (Mean Absolute Values)
    # Plot (Custom Style)
    plot_custom_bars(
        names=shap_importance_df['Feature'],
        values=shap_importance_df['Mean_Abs_SHAP'],
        title=f"SHAP Importance: {case_name}",
        ylabel="mean(|SHAP value|)",
        savepath=os.path.join(savepath, "shap_bar_plot.png"),
        color='thistle'
    )

if __name__ == "__main__":
    main()
