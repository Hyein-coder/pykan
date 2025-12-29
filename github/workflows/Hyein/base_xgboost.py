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
# fit() 안에는 이제 early_stopping_rounds를 쓰지 않습니다.
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)], # 검증 데이터는 여전히 여기에 필요합니다
    verbose=False
)

# 결과 확인
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("-" * 30)
print(f"✅ XGBoost R2 Score: {r2:.4f}")
print(f"📉 MSE (Mean Squared Error): {mse:.4f}")
print("-" * 30)

# 6. (추가) 중요 변수 확인하기
# 어떤 변수가 예측에 가장 큰 영향을 줬는지 봅니다.
# KAN 모델링 시 힌트가 될 수 있습니다.
print("Feature Importances:", model.feature_importances_)