import pandas as pd
import numpy as np
import torch
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from kan import KAN
from kan.custom import MultKAN
from kan.custom_utils import remove_outliers_iqr, evaluate_model_performance, plot_activation_functions
import datetime
save_tag = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
save_dir = os.path.join(os.getcwd(), '.github', 'workflows', 'Hyein', 'custom_figures')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"This script is running on {device}.")

x1_grid = np.linspace(-np.pi, np.pi, 15)
x2_grid = np.linspace(0, 1, 15)
x3_grid = np.linspace(-1, 1, 10)
# x1, x2, x3 = np.meshgrid(x1_grid, x2_grid, x3_grid)
# X = np.stack((x1.flatten(), x2.flatten(), x3.flatten()), axis=1)
# y = np.exp(-x1) + x2 - x3**2
# y = 5 * np.exp(np.sin(x1)) + 3 * x2 - x3

x1, x2= np.meshgrid(x1_grid, x2_grid)
X = np.stack((x1.flatten(), x2.flatten()), axis=1)
# y = 10 * np.abs(x1) + 5*x2**2
y = np.sin(2*x1) + 5 * x2

y = y.flatten().reshape(-1, 1)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)  # 0.2 × 0.8 = 0.16 (전체의 16%)

print(f"전체 데이터셋 크기: {len(X)}")
print(f"훈련셋 크기: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"검증셋 크기: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
print(f"테스트셋 크기: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

scaler_X = MinMaxScaler(feature_range=(0.1, 0.9))
scaler_y = MinMaxScaler(feature_range=(0.1, 0.9))
X_train_norm = scaler_X.fit_transform(X_train) # 훈련 데이터로 스케일러 학습 및 변환 (fit_transform)
y_train_norm = scaler_y.fit_transform(y_train)

X_val_norm = scaler_X.transform(X_val)
y_val_norm = scaler_y.transform(y_val)
X_test_norm = scaler_X.transform(X_test)
y_test_norm = scaler_y.transform(y_test)

X_train_tensor = torch.tensor(X_train_norm, dtype=torch.float32, device=device)
X_val_tensor = torch.tensor(X_val_norm, dtype=torch.float32, device=device)
X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32, device=device)
y_train_tensor = torch.tensor(y_train_norm, dtype=torch.float32, device=device)
y_val_tensor = torch.tensor(y_val_norm, dtype=torch.float32, device=device)
y_test_tensor = torch.tensor(y_test_norm, dtype=torch.float32, device=device)

dataset = {'train_input': X_train_tensor,'train_label': y_train_tensor,
            'val_input': X_val_tensor, 'val_label': y_val_tensor,
            'test_input': X_test_tensor,'test_label': y_test_tensor }

for key, value in dataset.items():
    print(f"{key}: {value.shape}")

#
import matplotlib.pyplot as plt

X_norm = scaler_X.transform(X)
y_norm = scaler_y.transform(y)

nx = X_norm.shape[1]
fig, axs = plt.subplots(nrows=1, ncols=nx, figsize=(15, 3))
for idx_x in range(nx):
    ax = axs[idx_x]
    ax.scatter(X_norm[:, idx_x], y_norm, color='black')
plt.savefig(os.path.join(save_dir, f"data_{save_tag}.png"))
plt.show()
#%%
model = MultKAN(width=[nx, 3, 1], mult_arity=0, grid=30, k=3, seed=0, device=device)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"학습가능 파라미터 수: {num_params:,}")

for name, p in model.named_parameters():
    if p.requires_grad:
        print(f"{name:40s} {p.shape} {p.numel():5d}")

# KAN 학습
model.fit(dataset, opt="LBFGS", steps=50, lamb=0.001, lamb_entropy=5)
model.plot()
plt.show()

#%
model = model.prune(node_th=1e-2, edge_th=3e-2)  # 더 자르고 싶으면 값을 높이고, 덜 자르고 변수를 많이 있게 하고 싶으면 값을 낮추기
model.plot()
plt.show()

#
from kan.utils import ex_round
lib = ['x', 'x^2', 'tanh', 'sin', '1/x', '1/x^2']
# lib = ['x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh', 'sin', '1/x', '1/x^2']
model.auto_symbolic(lib=lib)
# model.plot()

model.fit(dataset, opt="LBFGS", steps=50)
model.plot()
plt.show()
formula = ex_round(model.symbolic_formula()[0][0], 4)
print("formula=", formula)
print(model.node_scores)

#%%
val_pred, val_actual, val_metrics = evaluate_model_performance(model, dataset, scaler_y, display=True)
# test_pred, test_actual, test_metrics = evaluate_model_performance(model, dataset, scaler_y, "test")

plt.figure(figsize=(4, 4))  # 도화지 그리기~

plt.scatter(val_actual, val_pred, facecolors='none', edgecolors='r', s=15, label='Model Predictions')

# 제일 작은 값, 제일 큰 값 설정
min_val = min(val_actual.min(), val_pred.min())
max_val = max(val_actual.max(), val_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Fit (y=x)')  # y = x 선긋기

# 그래프 제목과 축 레이블 설정
plt.xlabel("Actual" , fontsize=9)   # Actual 다음에 우리가 보고자 하는 output predicting 변수가 뜸
plt.ylabel("Predicted", fontsize=9) #
plt.title(f'Test Set: Actual vs. Predicted (R² = {val_metrics["r2"]:.4f})', fontsize=11)
plt.legend()
plt.grid(True)  # 격자 on
plt.axis('equal') # x, y축 스케일을 동일하게 설정
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f"validation_{save_tag}.png"))
plt.show()
#%%
from kan.custom_utils import plot_data_per_interval, plot_spline_coefficients

X_norm = scaler_X.transform(X)
y_norm = scaler_y.transform(y)
name_X = [f'x{idx}' for idx in range(X_norm.shape[1])]
name_y = ['y']

fig_x1, ax_x1 = plot_data_per_interval(X_norm, y_norm, name_X, name_y, 0, [0, 0.3, 0.6])
plt.show()

# Plot learned activation functions (splines) per edge after training/pruning
plot_activation_functions(model, x=dataset, layers=None)
plot_spline_coefficients(model)

scores = model.node_scores[0]
# print(scores)
fig, ax = plt.subplots()
ax.bar(list(range(scores.shape[0])), scores.tolist())
plt.show()