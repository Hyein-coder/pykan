import argparse
import os
import joblib
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

# Import KAN
from kan import create_dataset
from kan.custom import MultKAN

# Import shared factory
# (Make sure this path matches your actual project structure)
from github.workflows.Hyein.toy_analytic_SHAP_Sobol import FUNCTION_ZOO


# ==========================================
# 2. Scikit-Learn Wrapper for KAN
# ==========================================
class KANRegressor(BaseEstimator, RegressorMixin):
    """
    A wrapper to make MultKAN compatible with Scikit-learn's RandomizedSearchCV.
    Now includes Symbolic Regression capabilities.
    """

    def __init__(self,
                 hidden_layer=5,
                 grid=3,
                 k=3,
                 lamb=0.01,
                 lamb_coef=0.1,  # [NEW] Coefficient regularization
                 lamb_entropy=0.1,  # [NEW] Entropy regularization
                 lr=0.1,  # [NEW] Learning rate
                 steps=20,
                 # Symbolic Regression Parameters (Defaults)
                 symbolic_enabled=True,
                 sym_lib=None,  # List of functions: ['x', 'x^2', 'sin', ...]
                 sym_weight_simple=0.0,
                 sym_r2_threshold=0.0,  # 0.0 means stricter filtering usually not applied aggressively
                 sym_a_range=(-10, 10),
                 sym_b_range=(-10, 10),
                 device='cpu'):

        self.hidden_layer = hidden_layer
        self.grid = grid
        self.k = k
        self.lamb = lamb
        self.lamb_coef = lamb_coef
        self.lamb_entropy = lamb_entropy
        self.lr = lr
        self.steps = steps

        # Symbolic params
        self.symbolic_enabled = symbolic_enabled
        lib = ['sin', 'cos', 'x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh', '1/x', '1/x^2']
        self.sym_lib = sym_lib if sym_lib is not None else lib
        self.sym_weight_simple = sym_weight_simple
        self.sym_r2_threshold = sym_r2_threshold
        self.sym_a_range = sym_a_range
        self.sym_b_range = sym_b_range

        self.device = device
        self.model = None
        self._n_features = None

    def fit(self, X, y):
        self._n_features = X.shape[1]

        # Determine width
        if self.hidden_layer > 0:
            width = [self._n_features, self.hidden_layer, 1]
        else:
            width = [self._n_features, 1]

        # Initialize KAN
        self.model = MultKAN(width=width, grid=self.grid, k=self.k,
                             seed=42, device=self.device)

        # Create dataset dictionary
        dataset = {
            'train_input': torch.tensor(X, dtype=torch.float32, device=self.device),
            'train_label': torch.tensor(y, dtype=torch.float32, device=self.device).reshape(-1, 1),
            'test_input': torch.tensor(X, dtype=torch.float32, device=self.device),
            # Using Train as dummy test for internal loop
            'test_label': torch.tensor(y, dtype=torch.float32, device=self.device).reshape(-1, 1)
        }
        self.dataset = dataset

        # 1. Initialize with BASE Grid (Grid=3)
        start_grid = 3
        self.model = MultKAN(width=width, grid=start_grid, k=self.k,
                             grid_range=(0.1, 0.9), seed=42, device=self.device)

        # 2. Phase 1: Train Coarse Model (Numeric)
        # Added new regularization parameters and LR
        self.model.fit(dataset, opt='LBFGS', steps=self.steps,
                       lamb=self.lamb, lamb_coef=self.lamb_coef, lamb_entropy=self.lamb_entropy,
                       lr=self.lr)

        # 3. Phase 2: Grid Extension (if needed)
        if self.grid > start_grid:
            self.model = self.model.refine(self.grid)
            self.model.fit(dataset, opt='LBFGS', steps=self.steps,
                           lamb=self.lamb, lamb_coef=self.lamb_coef, lamb_entropy=self.lamb_entropy,
                           lr=self.lr)

        if self.symbolic_enabled:
            try:
                # Attempt to lock activation functions to symbolic formulas
                self.model.auto_symbolic(lib=self.sym_lib,
                                         weight_simple=self.sym_weight_simple,
                                         r2_threshold=self.sym_r2_threshold,
                                         verbose=0,  # Turn off verbose for grid search
                                         a_range=self.sym_a_range,
                                         b_range=self.sym_b_range)

                # Re-fit the coefficients of the chosen symbolic functions
                self.model.fit(dataset, opt='LBFGS', steps=self.steps, lr=self.lr)
            except Exception as e:
                print(f"   ⚠️ Symbolic failed (ignoring): {e}")

        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("You must train the model before predicting!")

        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            # If symbolic was successful, this forward pass uses the symbolic formulas
            prediction = self.model.forward(X_tensor)

        return prediction.detach().cpu().numpy().flatten()

    def save_model(self, path):
        self.model.saveckpt(path=path)

    def load_model(self, ckpt_path):
        """
        Loads the model using KAN.loadckpt from custom_multkan_ddp.
        This automatically restores architecture (width, grid, etc.).
        """
        from kan.custom_multkan_ddp import KAN

        print(f"Loading KAN model from: {ckpt_path}")
        # The .pth extension is usually handled automatically by loadckpt,
        # but we pass the base path as requested.
        self.model = KAN.loadckpt(path=ckpt_path)

        # Ensure the loaded model is on the correct device
        # (KAN.loadckpt might load to CPU by default)
        if hasattr(self.model, 'to'):
            self.model.to(self.device)

        return self


# ==========================================
# 3. Main Tuning Script
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Tune KAN for Analytical Functions.")
    parser.add_argument("func_name", type=str, nargs='?', default="original",
                        choices=FUNCTION_ZOO.keys(),
                        help="Choose a function from the ZOO.")

    args = parser.parse_args()
    data_name = args.func_name

    print(f"🚀 Starting KAN Tuning for: '{data_name}'")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   - Device: {device}")

    # Output Directory
    root_dir = os.path.join(os.getcwd(), 'github', 'workflows', 'Hyein', 'analytical_results', data_name)
    savepath = os.path.join(root_dir, "kan_models")
    os.makedirs(savepath, exist_ok=True)

    # ==========================================
    # 4. Universal Data Generation
    # ==========================================
    config = FUNCTION_ZOO[data_name]
    target_func = config["func"]
    bounds = config["bounds"]
    feat_names = config["names"]
    nx = len(bounds)

    print(f"🎲 Generating synthetic data ({nx}D)...")

    X_raw = np.random.uniform(
        low=[b[0] for b in bounds],
        high=[b[1] for b in bounds],
        size=(1000, nx)
    )

    y_raw = np.apply_along_axis(target_func, 1, X_raw).reshape(-1, 1)

    # Add 5% Noise
    # noise = np.random.normal(0, np.std(y_raw) * 0.05, size=y_raw.shape)
    # y_raw = y_raw + noise

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)
    print(f"   - Train: {len(X_train)}, Test: {len(X_test)}")

    scaler_X = MinMaxScaler(feature_range=(0.1, 0.9))
    scaler_y = MinMaxScaler(feature_range=(0.1, 0.9))

    X_train_norm = scaler_X.fit_transform(X_train)
    y_train_norm = scaler_y.fit_transform(y_train)

    X_test_norm = scaler_X.transform(X_test)
    y_test_norm = scaler_y.transform(y_test)

    # ==========================================
    # 5. Hyperparameter Tuning
    # ==========================================
    param_distributions = {
        'hidden_layer': [0, nx],
        'grid': [3, 5, 10],
        'k': [3],
        'steps': [20, 50],
        'lamb': [0.001, 0.01, 0.1],
        'lamb_coef': [0.01, 0.1, 1.0],  # Penalize large coefficients (sparsity)
        'lamb_entropy': [0.1, 2.0, 10.0],  # Penalize complexity (for symbolic)
        'lr': [0.05, 0.1, 0.5]  # Learning rate for LBFGS
    }

    # Pass default symbolic options here if you want to override defaults
    # For now, we rely on the class defaults or you can set fixed values
    kan_wrapper = KANRegressor(device=device, symbolic_enabled=True)

    search = RandomizedSearchCV(
        estimator=kan_wrapper,
        param_distributions=param_distributions,
        n_iter=15,  # Increased slightly to cover new params
        cv=3,
        scoring='r2',
        n_jobs=1,  # IMPORTANT: Keep 1 for CUDA safety
        verbose=1,
        random_state=42
    )

    print("\n🏎️  Starting Randomized Hyperparameter Search (KAN with Symbolic)...")
    search.fit(X_train_norm, y_train_norm.ravel())

    # ==========================================
    # 6. Results & Saving
    # ==========================================
    best_estimator = search.best_estimator_
    best_kan_model = best_estimator.model

    print("\n" + "=" * 40)
    print(f"🏆 Best Parameters: {search.best_params_}")
    print("=" * 40)

    y_pred_test_norm = best_estimator.predict(X_test_norm)
    r2_test = r2_score(y_test_norm, y_pred_test_norm)

    print(f"📊 [Test Set]       R2 Score: {r2_test:.4f}")

    # Output Symbolic Formula
    try:
        formula = best_kan_model.symbolic_formula()[0][0]
        print(f"🧮 Symbolic Formula: {formula}")
    except:
        formula = "Symbolic conversion failed or not enabled"

    metrics_data = {
        "dataset": data_name,
        "test_r2": r2_test,
        "best_params": search.best_params_,
        "symbolic_formula": str(formula)
    }

    json_path = os.path.join(savepath, f'{data_name}_kan_metrics.json')
    with open(json_path, "w") as f:
        json.dump(metrics_data, f, indent=4)

    # Save Models
    best_kan_model.saveckpt(path=os.path.join(savepath, f'{data_name}_best_kan_model'))
    joblib.dump(scaler_X, os.path.join(savepath, f'{data_name}_mlp_scaler_X.pkl'))
    joblib.dump(scaler_y, os.path.join(savepath, f'{data_name}_mlp_scaler_y.pkl'))

    print(f"💾 Saved model and scalers to {savepath}")

    # Draw Parity Plot
    plt.figure(figsize=(6, 6))
    y_pred_test = scaler_y.inverse_transform(y_pred_test_norm.reshape(1, -1))
    plt.scatter(y_test, y_pred_test, alpha=0.6, edgecolors='k', s=30, label='Test Data')

    # Perfect fit line
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')

    plt.title(f"Parity Plot: {data_name} (R2={r2_test:.4f})")
    plt.xlabel("Actual Value")
    plt.ylabel("Predicted Value")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plot_path = os.path.join(savepath, f"{data_name}_parity_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.show()
    print(f"📊 Parity plot saved to: {plot_path}")

    best_kan_model.forward(best_estimator.dataset['train_input'])
    scores_tot = best_kan_model.feature_score.detach().cpu().numpy()  # Global scores

    fig_tot, ax_tot = plt.subplots()

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

if __name__ == "__main__":
    main()