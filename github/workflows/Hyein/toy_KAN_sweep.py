import argparse
import os
import joblib
import json
import torch
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

# Import KAN
from kan import create_dataset
from kan.custom import MultKAN

# Import shared factory
from github.workflows.Hyein.toy_analytic_SHAP_Sobol import FUNCTION_ZOO


# ==========================================
# 2. Scikit-Learn Wrapper for KAN
# ==========================================
class KANRegressor(BaseEstimator, RegressorMixin):
    """
    A wrapper to make MultKAN compatible with Scikit-learn's RandomizedSearchCV.
    """

    def __init__(self,
                 hidden_layer=5,
                 grid=3,
                 k=3,
                 lamb=0.01,
                 steps=20,
                 device='cpu'):
        self.hidden_layer = hidden_layer
        self.grid = grid
        self.k = k
        self.lamb = lamb
        self.steps = steps
        self.device = device
        self.model = None
        self._n_features = None

    def fit(self, X, y):
        self._n_features = X.shape[1]

        # Determine width: [Input, Hidden, Output] or [Input, Output]
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
            'test_input': torch.tensor(X, dtype=torch.float32, device=self.device),  # Dummy
            'test_label': torch.tensor(y, dtype=torch.float32, device=self.device).reshape(-1, 1)
        }

        # 3. Initialize with BASE Grid (Grid=3)
        # Even if target is 10, we start small.
        start_grid = 3
        self.model = MultKAN(width=width, grid=start_grid, k=self.k,
                             grid_range=(0.1, 0.9), seed=42, device=self.device)

        # 4. Phase 1: Train Coarse Model
        # print(f"   -> Phase 1: Training grid={start_grid}")
        self.model.fit(dataset, opt='LBFGS', steps=self.steps, lamb=self.lamb)

        # 5. Phase 2: Grid Extension (if needed)
        if self.grid > start_grid:
            # print(f"   -> Phase 2: Extending grid {start_grid} -> {self.grid}")
            self.model = self.model.refine(self.grid)

            # Train again (Fine-tuning)
            self.model.fit(dataset, opt='LBFGS', steps=self.steps, lamb=self.lamb)

        return self

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

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("You must train the model before predicting!")

        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            prediction = self.model.forward(X_tensor)

        return prediction.detach().cpu().numpy().flatten()


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
    nx = len(bounds)

    print(f"🎲 Generating synthetic data ({nx}D)...")

    # Generate X within specific bounds for each feature
    # bounds is list of lists: [[min, max], [min, max], ...]
    X_raw = np.random.uniform(
        low=[b[0] for b in bounds],
        high=[b[1] for b in bounds],
        size=(1000, nx)
    )

    # Generate y (Apply function row-by-row)
    # This works for both STANDARD_ZOO (lambdas) and LOG_SUM_ZOO (wrappers)
    y_raw = np.apply_along_axis(target_func, 1, X_raw).reshape(-1, 1)

    # Add 5% Noise
    noise = np.random.normal(0, np.std(y_raw) * 0.05, size=y_raw.shape)
    y_raw = y_raw + noise

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
        'hidden_layer': [0, nx],  # 0=Symbolic (shallow), nx=Deeper
        'grid': [3, 5, 10],  # Granularity
        'k': [3],  # Cubic spline
        'lamb': [0.001, 0.01, 0.1],  # Regularization
        'steps': [20, 50]  # Training steps
    }

    kan_wrapper = KANRegressor(device=device)

    search = RandomizedSearchCV(
        estimator=kan_wrapper,
        param_distributions=param_distributions,
        n_iter=10,
        cv=3,
        scoring='r2',
        n_jobs=1,  # IMPORTANT: Keep 1 for CUDA safety
        verbose=1,
        random_state=42
    )

    print("\n🏎️  Starting Randomized Hyperparameter Search (KAN)...")
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

    # Save Metrics
    metrics_data = {
        "dataset": data_name,
        "test_r2": r2_test,
        "best_params": search.best_params_
    }

    json_path = os.path.join(savepath, f'{data_name}_kan_metrics.json')
    with open(json_path, "w") as f:
        json.dump(metrics_data, f, indent=4)

    # Save Models
    best_kan_model.saveckpt(path=os.path.join(savepath, f'{data_name}_best_kan_model'))
    joblib.dump(scaler_X, os.path.join(savepath, f'{data_name}_mlp_scaler_X.pkl'))
    joblib.dump(scaler_y, os.path.join(savepath, f'{data_name}_mlp_scaler_y.pkl'))

if __name__ == "__main__":
    main()