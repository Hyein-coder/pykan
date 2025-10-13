import torch
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from kan.custom_utils import remove_outliers_iqr
import pandas as pd

from kan.experiments.multkan_hparam_sweep import evaluate_params
from kan.custom_utils import plot_data_per_interval, plot_activation_and_spline_coefficients, get_masks
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"This script is running on {device}.")
save_tag = "CO2RR_MSP"

params = {
    "grid": 3,
    "grid_range": "[0, 1]",
    "lamb": 0.001,
    "lamb_coef": 0.1,
    "lamb_coefdiff": 0.1,
    "lamb_entropy": 0.01,
    "lr": 0.1,
    "prune": True,
    "pruning_th": 0.01,
    "update_grid": True,
    "width": "[[8, 0], [2, 0], [2, 0], [1, 0]]",
}
params['grid'] = 30

dir_current = os.getcwd()
filepath = os.path.join(dir_current, "github\workflows\TaeWoong", "25.01.14_CO2RR_GSA.xlsx")

xls = pd.ExcelFile(filepath)
df_in  = pd.read_excel(xls, sheet_name='Input')
df_out = pd.read_excel(xls, sheet_name='Output')

df_in_final, df_out_final = remove_outliers_iqr(df_in, df_out)

removed_count = len(df_in) - len(df_in_final)  # 몇 개 지웠는지 세기
print(f"이상치 제거 후 데이터 수: {len(df_in_final)} 개 ({removed_count} 개 제거됨)")
print("--- 이상치 제거 완료 ---\n")

name_X = [
    "Current density (mA/cm2)",
    "Faradaic efficiency (%)",
    "CO coversion",
    "Voltage (V)",
    "Electricity cost ($/kWh)",
    "Membrain cost ($/m2)",
    "Catpure energy (GJ/ton)",
    "Crossover rate"
]
name_y = "MSP ($/kgCO)" # Required energy_total (MJ/kgCO) # MSP ($/kgCO)
X = df_in_final[name_X].values
y = df_out_final[name_y].values.reshape(-1, 1)

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)  # 0.2 × 0.8 = 0.16 (전체의 16%)

print(f"전체 데이터셋 크기: {len(X)}")
print(f"훈련셋 크기: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"검증셋 크기: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
print(f"테스트셋 크기: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# 1. MinMaxScaler 객체 생성 --- 범위를 0.1~0.9로 재설정
scaler_X = MinMaxScaler(feature_range=(0.1, 0.9))
scaler_y = MinMaxScaler(feature_range=(0.1, 0.9))

X_train_norm = scaler_X.fit_transform(X_train) # 훈련 데이터로 스케일러 학습 및 변환 (fit_transform)
y_train_norm = scaler_y.fit_transform(y_train) # X_train의 각 변수(컬럼)별로 최소값은 0, 최대값은 1이 되도록 변환됩니다.

X_val_norm = scaler_X.transform(X_val)
X_test_norm = scaler_X.transform(X_test)

y_val_norm = scaler_y.transform(y_val)
y_test_norm = scaler_y.transform(y_test)

res, model, fit_kwargs, dataset = evaluate_params(
    X_train_norm, y_train_norm, X_val_norm, y_val_norm, params, X_test_norm, y_test_norm,
    0, scaler_y, device.type,
    special_tag=save_tag,
    special_dir='D:/pykan/github/workflows/Hyein/custom_figures'
)

model.plot()
plt.show()

#%%
from kan.utils import ex_round

lib = ['sin', 'cos', 'x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh', '1/x', '1/x^2']
sym_weight_simple = params.get('sym_weight_simple', 0.8)
sym_r2_threshold = params.get('sym_r2_threshold', 0.)

model.auto_symbolic(lib=lib, weight_simple=sym_weight_simple, r2_threshold=sym_r2_threshold)
model.fit(dataset, **fit_kwargs)
model.plot()
plt.show()
sym_fun = ex_round(model.symbolic_formula()[0][0], 4)

X_norm = scaler_X.transform(X)
y_norm = scaler_y.transform(y)

plot_activation_and_spline_coefficients(model, save_tag=save_tag, x=dataset, layers=None)
