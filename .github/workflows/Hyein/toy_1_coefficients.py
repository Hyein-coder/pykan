#%%
import pandas as pd
from kan.custom_utils import plot_data_per_interval, plot_spline_coefficients, plot_activation_functions
import matplotlib.pyplot as plt
import os
import datetime

save_dir = "D:\pykan\.github\workflows\Hyein\custom_figures"
time_stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')

file_name = [
    r"D:\pykan\.github\workflows\Hyein\multkan_sweep_autosave\20250930_150005_auto_10sin(x1)+5x2^2.xlsx",
    # r"D:\pykan\.github\workflows\Hyein\multkan_sweep_autosave\20250930_150542_auto_10sin(x1)+20x2^2.xlsx",
]
x_square_coeff = [5]

file_data = []
for f in file_name:
    df = pd.read_excel(f, sheet_name='best_avg_by_params')
    file_data.append(df)

#%%
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from kan.experiments.multkan_hparam_sweep import evaluate_params
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"This script is running on {device}.")

x1_grid = np.linspace(-np.pi, np.pi, 30)
x2_grid = np.linspace(-1, 1, 30)

x1, x2= np.meshgrid(x1_grid, x2_grid)
X = np.stack((x1.flatten(), x2.flatten()), axis=1)

xsc = x_square_coeff[0]

d_opt = file_data[0].iloc[0]
d_opt = d_opt.to_dict()
params = {k: v for k, v in d_opt.items() if "param_" in k}
params = {key.replace('param_', ''): value for key, value in params.items()}

y = 10 * np.sin(x1) + xsc * x2**2

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

X_train_tensor = torch.tensor(X_train_norm, dtype=torch.float32, device=device)
X_val_tensor = torch.tensor(X_val_norm, dtype=torch.float32, device=device)
X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32, device=device)
y_train_tensor = torch.tensor(y_train_norm, dtype=torch.float32, device=device)
y_val_tensor = torch.tensor(y_val_norm, dtype=torch.float32, device=device)
y_test_tensor = torch.tensor(y_test_norm, dtype=torch.float32, device=device)

dataset = {'train_input': X_train_tensor,'train_label': y_train_tensor,
            'val_input': X_val_tensor, 'val_label': y_val_tensor,
            'test_input': X_test_tensor,'test_label': y_test_tensor }

res, model = evaluate_params(X_train, y_train, X_val, y_val, params,
                             X_test, y_test, 0, scaler_y, device)

#%%
lib = ['sin', 'cos', 'x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh', '1/x', '1/x^2']
sym_weight_simple = params.get('sym_weight_simple', 0.8)
sym_r2_threshold = params.get('sym_r2_threshold', 0.)
fit_kwargs = {
    'opt': params.get('opt', 'LBFGS'),
    'steps': params.get('steps', 50),
    'lamb': params.get('lamb', 0.0),
    'lamb_l1': params.get('lamb_l1', 1.0),
    'lamb_entropy': params.get('lamb_entropy', 2.0),
    'lamb_coef': params.get('lamb_coef', 0.0),
    'lamb_coefdiff': params.get('lamb_coefdiff', 0.0),
    'update_grid': params.get('update_grid', True),
    'lr': params.get('lr', 1.0),
    'batch': params.get('batch', -1),
    'log': params.get('log', 1),
}

# model.auto_symbolic(lib=lib, weight_simple=sym_weight_simple, r2_threshold=sym_r2_threshold)
model.fit(dataset, **fit_kwargs)    # ===> 여기서 에러가 남 ㅂㄷㅂㄷ
model.plot()
plt.show()

#%%
save_tag = f'periodic_{time_stamp}_{xsc}x2^2'

X_norm = scaler_X.transform(X)
y_norm = scaler_y.transform(y)
name_X = [f'x{idx}' for idx in range(X_norm.shape[1])]
name_y = ['y']

fig_x1, ax_x1 = plot_data_per_interval(X_norm, y_norm, name_X, name_y, 1, [0, 0.4])
plt.savefig(os.path.join(save_dir, f"{save_tag}_data_colored.png"))
plt.show()

# Plot learned activation functions (splines) per edge after training/pruning
plot_activation_functions(model, save_tag=save_tag, x=dataset, layers=None)
plot_spline_coefficients(model, save_tag=save_tag)

scores = model.node_scores[0]
# print(scores)
fig, ax = plt.subplots()
ax.bar(list(range(scores.shape[0])), scores.tolist())
plt.savefig(os.path.join(save_dir, f"{save_tag}_scores_L0.png"))
plt.show()
