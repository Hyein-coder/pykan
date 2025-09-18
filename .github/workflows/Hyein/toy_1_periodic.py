import pandas as pd
import numpy as np
import torch
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from kan import KAN
from kan.custom_utils import remove_outliers_iqr, evaluate_model_performance

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"This script is running on {device}.")

x1_grid = np.linspace(-np.pi, np.pi, 15)
x2_grid = np.linspace(-1, 1, 15)
x3_grid = np.linspace(-1, 1, 10)
x1, x2, x3 = np.meshgrid(x1_grid, x2_grid, x3_grid)
y = np.exp(-x1) + x2 - x3**2
# y = 5 * np.exp(np.sin(x1)) + 3 * x2 - x3
# y = 10 * np.abs(x1) + 5*x2**2
# y = 10 * np.sin(x1) + 5 * x2**2

X = np.stack((x1.flatten(), x2.flatten(), x3.flatten()), axis=1)
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
plt.show()
#%%
model = KAN(width=[nx, 6, 1], grid=3, k=3, seed=42, device=device)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"학습가능 파라미터 수: {num_params:,}")

for name, p in model.named_parameters():
    if p.requires_grad:
        print(f"{name:40s} {p.shape} {p.numel():5d}")

# KAN 학습
model.fit(dataset, opt="LBFGS", steps=50, lamb=0.001)
model.plot()
plt.show()

#%
model = model.prune(node_th=1e-2, edge_th=3e-2)  # 더 자르고 싶으면 값을 높이고, 덜 자르고 변수를 많이 있게 하고 싶으면 값을 낮추기

#학습
# model.fit(dataset, opt="LBFGS", steps=50)  # update_grid 가 False일때랑 True일때의 차이는?

model = model.refine(30)
model.fit(dataset, opt="LBFGS", steps=50)
model.plot()
plt.show()

#
from kan.utils import ex_round
lib = ['x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh', 'sin', '1/x', '1/x^2']
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
plt.show()
#%%
X_norm = scaler_X.transform(X)
y_norm = scaler_y.transform(y)

nx = X_norm.shape[1]
fig, axs = plt.subplots(nrows=1, ncols=nx, figsize=(15, 3))
for idx_x in range(nx):
    ax = axs[idx_x]
    ax.scatter(X_norm[:, idx_x], y_norm, color='black')
    # ax.set_title(name_X[idx_x], fontsize=8)

# X_norm: (N, D), y_norm: (N,) 또는 (N, 1)
x2 = X_norm[:, 0]
y_vals = y_norm.ravel()  # y가 (N,1)이어도 (N,)으로 평탄화

# 4개 구간 마스크 정의
mask_knots = [0, 0.3, 0.6, 1.0]
masks = [ ((x2 > mask_knots[i]) & (x2 <= mask_knots[i+1])) for i in range(len(mask_knots)-1)]

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
labels = [f'{mask_knots[i]} < x2 <= {mask_knots[i+1]}' for i in range(len(mask_knots)-1)]

# plt.figure(figsize=(7, 5))
for mask, c, lab in zip(masks, colors, labels):
    if np.any(mask):  # 해당 구간 데이터가 있을 때만 그림
        for idx_x in range(nx):
            ax = axs[idx_x]
            ax.scatter(X_norm[mask, idx_x], y_vals[mask], s=20, color=c, alpha=0.75, edgecolor='none', label=lab)
            # ax.set_title(name_X[idx_x], fontsize=8)
axs[-1].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
plt.show()

for act_fun in model.act_fun:
    ni, no = act_fun.coef.shape[:2]
    coef = act_fun.coef.tolist()
    fig, axs = plt.subplots(nrows=no, ncols=ni, figsize=(15, 3), squeeze=False)
    for idx_in, coef_in in enumerate(coef):
        for idx_out, coef_node in enumerate(coef_in):
            ax = axs[idx_out, idx_in]
            ax.scatter(list(range(len(coef_node))), coef_node)
    plt.show()