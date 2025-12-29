import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from kan.custom_processing import remove_outliers_iqr
import torch
import os

root_dir = os.path.join(os.getcwd(), 'github', 'workflows', 'Hyein')
filepath = os.path.join(root_dir, "data", "CrossedBarrel.csv")
filedata = pd.read_csv(filepath)
name_X = filedata.columns[:-1].tolist()
name_y = filedata.columns[-1]
df_in = filedata[name_X]
df_out = filedata[[name_y]]
print(f"TARGET: {name_y}")

df_in_final, df_out_final = remove_outliers_iqr(df_in, df_out)

removed_count = len(df_in) - len(df_in_final)
print(f"# of data after removing outliers: {len(df_in_final)} 개 ({removed_count} 개 제거됨)")

X = df_in_final[name_X].values
y = df_out_final[name_y].values.reshape(-1, 1)

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2,
                                                  random_state=42)
print(f"Train set: {len(X_train)} ({len(X_train) / len(X) * 100:.1f}%)")
print(f"Validation set: {len(X_val)} ({len(X_val) / len(X) * 100:.1f}%)")
print(f"Test set: {len(X_test)} ({len(X_test) / len(X) * 100:.1f}%)")

scaler_X = MinMaxScaler(feature_range=(0.1, 0.9))
scaler_y = MinMaxScaler(feature_range=(0.1, 0.9))
X_train_norm = scaler_X.fit_transform(X_train)
y_train_norm = scaler_y.fit_transform(y_train)
X_val_norm = scaler_X.transform(X_val)
X_test_norm = scaler_X.transform(X_test)
y_val_norm = scaler_y.transform(y_val)
y_test_norm = scaler_y.transform(y_test)

# 3. 모델 선언 (기본 설정)
# n_estimators: 나무의 개수 (보통 100~1000)
# learning_rate: 학습률 (보통 0.01~0.1)
# max_depth: 나무의 깊이 (너무 깊으면 과적합, 보통 3~6)
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    n_jobs=-1,
    random_state=42,
    early_stopping_rounds=50  # <--- 이 설정이 최신 버전에선 여기로 왔습니다
)

# 학습 (Fit)
print("학습을 시작합니다...")
model.fit(
    X_train_norm, y_train_norm,
    eval_set=[(X_test_norm, y_test_norm)],
    verbose=False
)

# 결과 확인
y_pred_norm = model.predict(X_test_norm)
r2 = r2_score(y_test_norm, y_pred_norm)
mse = mean_squared_error(y_test_norm, y_pred_norm)

print("-" * 30)
print(f"✅ XGBoost R2 Score: {r2:.4f}")
print(f"📉 MSE (Mean Squared Error): {mse:.4f}")
print("-" * 30)

# 6. (추가) 중요 변수 확인하기
# 어떤 변수가 예측에 가장 큰 영향을 줬는지 봅니다.
# KAN 모델링 시 힌트가 될 수 있습니다.
print("Feature Importances:", model.feature_importances_)

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SALib.sample import saltelli
from SALib.analyze import sobol

# ==========================================
# 1. SALib 문제 정의 (Problem Definition)
# ==========================================
# 모델이 학습된 입력 변수의 개수와 범위를 정의합니다.
# 주의: 스케일링된 데이터로 학습했다면, 범위도 그에 맞춰야 합니다.
# 예: MinMaxScaler(-1, 1)을 썼다면 bounds는 [-1, 1] 이어야 합니다.

# 변수 개수 (X_train의 컬럼 수)
n_features = X_train.shape[1]

# 변수 이름 (없으면 그냥 x0, x1... 으로 생성)
feature_names = [f"Feature {i}" for i in range(n_features)]
# 만약 pandas 컬럼 이름이 있다면: feature_names = list(X.columns)

problem = {
    'num_vars': n_features,
    'names': feature_names,
    'bounds': [[-1, 1]] * n_features  # 모든 변수의 범위가 -1 ~ 1 이라고 가정 (스케일링 맞춤)
}

# ==========================================
# 2. 샘플 데이터 생성 (Sample Generation)
# ==========================================
# N은 샘플링 개수입니다. 클수록 정확하지만 계산 시간이 늘어납니다. (보통 1024 이상 권장)
# 총 실행 횟수 = N * (2 * D + 2)  (D는 변수 개수)
N = 1024
X_sobol = saltelli.sample(problem, N, calc_second_order=True)

print(f"생성된 샘플 개수: {X_sobol.shape[0]}개")

# ==========================================
# 3. 모델 예측 실행 (Run Model)
# ==========================================
# 생성된 샘플(X_sobol)을 XGBoost 모델에 넣고 예측값(Y)을 구합니다.
# XGBoost predict는 numpy array를 잘 받으므로 바로 넣으면 됩니다.

Y_sobol = model.predict(X_sobol)

# ==========================================
# 4. Sobol 분석 수행 (Analyze)
# ==========================================
# calc_second_order=True면 변수 간의 상호작용(Interaction)까지 분석합니다.
Si = sobol.analyze(problem, Y_sobol, calc_second_order=True)

# ==========================================
# 5. 결과 확인 및 시각화
# ==========================================

# 텍스트로 출력
print("\n[Sobol Analysis Result]")
total_si = Si['ST'] # Total Effect Index (총 영향력)
first_si = Si['S1'] # First Order Index (단독 영향력)

results_df = pd.DataFrame({
    'Feature': feature_names,
    'Total_Effect (ST)': total_si,
    'First_Order (S1)': first_si
}).sort_values(by='Total_Effect (ST)', ascending=False)

print(results_df)

# 막대 그래프 그리기 (상위 10개만)
plt.figure(figsize=(10, 6))
plt.title("Feature Sensitivity (Total Effect Index)")
plt.barh(results_df['Feature'][:10][::-1], results_df['Total_Effect (ST)'][:10][::-1])
plt.xlabel("Total Effect Index (ST)")
plt.show()