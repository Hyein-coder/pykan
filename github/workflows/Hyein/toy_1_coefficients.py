#%%
import pandas as pd
from kan.custom_utils import (plot_data_per_interval, plot_spline_coefficients, plot_activation_functions,
                              plot_activation_and_spline_coefficients, get_masks)
import matplotlib.pyplot as plt
import os
import datetime

root_dir = os.path.join(os.getcwd(), 'github', 'workflows', 'Hyein')
save_dir = os.path.join(root_dir, "custom_figures")
# save_dir = "D:\pykan\github\workflows\Hyein\custom_figures"
time_stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')

# file_name = [
#     r"D:\pykan\github\workflows\Hyein\multkan_sweep_autosave\20250930_150005_auto_10sin(x1)+5x2^2.xlsx",
#     r"D:\pykan\github\workflows\Hyein\multkan_sweep_autosave\20250930_150232_auto_10sin(x1)+10x2^2.xlsx",
#     r"D:\pykan\github\workflows\Hyein\multkan_sweep_autosave\20250930_150542_auto_10sin(x1)+20x2^2.xlsx",
#     r"D:\pykan\github\workflows\Hyein\multkan_sweep_autosave\20251001_092047_auto_10sin(x1)+40x2^2.xlsx",
# ]
# x_coeff = [5, 10, 20, 40]
# ground_truth = lambda xsc, x1, x2: 10 * np.sin(x1) + xsc * x2**2

file_name = [
    "20251001_102135_auto_10sin(x1)+5x2.xlsx",
    # r"D:\pykan\github\workflows\Hyein\multkan_sweep_autosave\20251001_103807_auto_10sin(x1)+10x2.xlsx",
    # r"D:\pykan\github\workflows\Hyein\multkan_sweep_autosave\20251001_104111_auto_10sin(x1)+20x2.xlsx",
]
x_coeff = [5]
ground_truth = lambda xc, x1, x2: 10 * np.sin(x1) + xc * x2

file_data = []
for f in file_name:
    df = pd.read_excel(os.path.join(root_dir, 'multkan_sweep_autosave', f), sheet_name='best_avg_by_params')
    file_data.append(df)

#%%
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from kan.experiments.multkan_hparam_sweep import evaluate_params
import numpy as np
from kan.utils import ex_round

sym_res = []
for xc, d_opt in zip(x_coeff, file_data):
    save_tag = f'periodic_{time_stamp}_{xc}x2^2'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"This script is running on {device}.")

    x1_grid = np.linspace(-np.pi * 2, np.pi * 2, 60)
    x2_grid = np.linspace(-1, 1, 30)

    x1, x2= np.meshgrid(x1_grid, x2_grid)
    X = np.stack((x1.flatten(), x2.flatten()), axis=1)

    # xsc = x_square_coeff[0]
    # d_opt = file_data[0]
    d_opt = d_opt.iloc[0]
    d_opt = d_opt.to_dict()
    params = {k: v for k, v in d_opt.items() if "param_" in k}
    params = {key.replace('param_', ''): value for key, value in params.items()}

    # y = 10 * np.sin(x1) + xsc * x2**2
    # y = 10 * np.sin(x1) + xc * x2
    y = ground_truth(xc, x1, x2)

    y = y.flatten().reshape(-1, 1)

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)  # 0.2 × 0.8 = 0.16 (전체의 16%)

    scaler_X = MinMaxScaler(feature_range=(0.1, 0.9))
    scaler_y = MinMaxScaler(feature_range=(0.1, 0.9))
    X_train_norm = scaler_X.fit_transform(X_train)
    y_train_norm = scaler_y.fit_transform(y_train)
    X_val_norm = scaler_X.transform(X_val)
    X_test_norm = scaler_X.transform(X_test)
    y_val_norm = scaler_y.transform(y_val)
    y_test_norm = scaler_y.transform(y_test)

    params['symbolic'] = False

    res, model, fit_kwargs, dataset = evaluate_params(
        X_train_norm, y_train_norm, X_val_norm, y_val_norm, params, X_test_norm, y_test_norm,
        0, scaler_y, device.type,
        special_tag=save_tag,
        special_dir=save_dir,
    )

    lib = ['sin', 'cos', 'x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh', '1/x', '1/x^2']
    sym_weight_simple = params.get('sym_weight_simple', 0.8)
    sym_r2_threshold = params.get('sym_r2_threshold', 0.)

    model.auto_symbolic(lib=lib, weight_simple=sym_weight_simple, r2_threshold=sym_r2_threshold)
    model.fit(dataset, **fit_kwargs)
    model.plot()
    plt.show()
    sym_fun = ex_round(model.symbolic_formula()[0][0], 4)
    sym_res.append(sym_fun)

    X_norm = scaler_X.transform(X)
    y_norm = scaler_y.transform(y)
    name_X = [f'x{idx}' for idx in range(X_norm.shape[1])]
    name_y = ['y']

    mask_idx = 0
    mask_interval = [-np.pi, -np.pi/2, np.pi/2, np.pi]

    fig_x1, ax_x1 = plot_data_per_interval(X, y, name_X, name_y, mask_idx, mask_interval)
    plt.savefig(os.path.join(save_dir, f"{save_tag}_data_colored.png"))
    plt.show()

    # Plot learned activation functions (splines) per edge after training/pruning
    plot_activation_and_spline_coefficients(model, save_tag=save_tag, x=dataset, layers=None)

    # Compute attribution score
    scores_tot = model.node_scores[0].detach().cpu().numpy()
    fig, ax = plt.subplots()
    ax.bar(list(range(scores_tot.shape[0])), scores_tot.tolist())
    plt.savefig(os.path.join(save_dir, f"{save_tag}_scores_L0.png"))
    plt.show()

    masks = get_masks(X, mask_idx, mask_interval)
    scores_interval = []
    for mask in masks:
        if np.any(mask):
            x_masked = X[mask, :]   # 이게 아니라 fabricated, 임의의 input을 주게 되면 어떨까?
            x_norm_masked = scaler_X.transform(x_masked)
            x_tensor_masked = torch.tensor(x_norm_masked, dtype=torch.float32, device=device)
            model.forward(x_tensor_masked)
            scores_interval.append(model.node_scores[0].detach().cpu().numpy().copy())
        else:
            scores_interval.append(np.zeros(scores_tot.shape))

    xticks = np.arange(len(masks))
    xticklabels = [f'{lb:.2f} < x{mask_idx} <= {ub:.2f}' for lb, ub in zip(mask_interval[:-1], mask_interval[1:])]
    width = 0.25
    fig, ax = plt.subplots()
    for idx in range(scores_tot.shape[0]):
        ax.bar(xticks + idx * width, [s[idx] for s in scores_interval], width, label=f"x{idx}")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=10, ha='center', fontsize=8)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{save_tag}_scores_L0_interval.png"))
    plt.show()

with open(os.path.join(save_dir, f"{save_tag}_sym_res.txt"), 'w') as f:
    for sym in sym_res:
        f.write(f"{sym}\n")
