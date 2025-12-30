import os
import json
import numpy as np
import torch
from github.workflows.Hyein.toy_KAN_sweep import KANRegressor  # Import the wrapper class


def load_and_predict(data_name):
    # 1. Define Paths
    root_dir = os.path.join(os.getcwd(), 'github', 'workflows', 'Hyein', 'analytical_results', data_name)
    savepath = os.path.join(root_dir, "kan_models")

    # Path to the saved model (without .pth extension usually, depending on saveckpt)
    # in tune_kan.py we saved it as: os.path.join(savepath, f'{data_name}_best_kan_model')
    ckpt_path = os.path.join(savepath, f'{data_name}_best_kan_model')

    # 2. Initialize Wrapper
    # We only need to specify the device. Architecture params (grid, hidden)
    # are irrelevant here because loadckpt overwrites self.model.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = KANRegressor(device=device)

    # 3. Load the Model
    try:
        model.load_model(ckpt_path)
        print("✅ Model loaded successfully using custom_multkan_ddp!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 4. Make a Prediction (Sanity Check)
    # We need to know input dimension for dummy data.
    # Parsing from name (e.g., log_sum_5d -> 5)
    if "log_sum" in data_name:
        n_features = int(data_name.split('_')[2][:-1])
    elif data_name == "convolution":
        n_features = 2
    else:
        n_features = 2

    print(f"🔮 Testing prediction with {n_features}D random input...")
    X_new = np.random.uniform(-1, 1, size=(5, n_features))
    preds = model.predict(X_new)

    print("   Predictions:", preds)


if __name__ == "__main__":
    # Example usage
    load_and_predict("original")