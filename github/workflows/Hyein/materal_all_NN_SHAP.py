#%%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from kan.custom_processing import remove_outliers_iqr
import torch
import os
import pandas as pd

data_name = "Perovskite"

root_dir = os.path.join(os.getcwd(), 'github', 'workflows', 'Hyein')
filepath = os.path.join(root_dir, "data", f"{data_name}.csv")
savepath = os.path.join(root_dir, "nn_models")

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

X_temp_denorm, X_test_denorm, y_temp_denorm, y_test_denorm = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_denorm, X_val_denorm, y_train_denorm, y_val_denorm = train_test_split(X_temp_denorm, y_temp_denorm, test_size=0.2,
                                                  random_state=42)
print(f"Train set: {len(X_train_denorm)} ({len(X_train_denorm) / len(X) * 100:.1f}%)")
print(f"Validation set: {len(X_val_denorm)} ({len(X_val_denorm) / len(X) * 100:.1f}%)")
print(f"Test set: {len(X_test_denorm)} ({len(X_test_denorm) / len(X) * 100:.1f}%)")

# scaler_X = MinMaxScaler(feature_range=(0.1, 0.9))
# scaler_y = MinMaxScaler(feature_range=(0.1, 0.9))
# X_train_norm = scaler_X.fit_transform(X_train_denorm)
# y_train_norm = scaler_y.fit_transform(y_train_denorm)
# X_val_norm = scaler_X.transform(X_val_denorm)
# X_test_norm = scaler_X.transform(X_test_denorm)
# y_val_norm = scaler_y.transform(y_val_denorm)
# y_test_norm = scaler_y.transform(y_test_denorm)

feature_names = name_X

#%%
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. 저장된 모델과 스케일러 불러오기
# ==========================================
loaded_model = joblib.load(os.path.join(savepath, f'{data_name}_best_mlp_model.pkl'))
scaler_X = joblib.load(os.path.join(savepath, f'{data_name}_mlp_scaler_X.pkl'))
scaler_y = joblib.load(os.path.join(savepath, f'{data_name}_mlp_scaler_y.pkl'))

X_train_norm = scaler_X.transform(X_train_denorm)
X_val_norm = scaler_X.transform(X_val_denorm)
X_test_norm = scaler_X.transform(X_test_denorm)
y_train_norm = scaler_y.transform(y_train_denorm)
y_val_norm = scaler_y.transform(y_val_denorm)
y_test_norm = scaler_y.transform(y_test_denorm)

num_data = len(X_train_denorm)
# num_shap_sample = min(num_data, 100)
num_shap_sample = 100
# ==========================================
# 2. SHAP 분석을 위한 데이터 준비
# ==========================================
# 주의: MLP는 계산이 오래 걸리므로, 전체 데이터 대신 '요약(Summary)'을 사용합니다.
# X_train_norm: 앞서 학습에 썼던 스케일링된 학습 데이터 (numpy array)

# 배경 데이터(Background) 설정: 학습 데이터의 일부(예: 100개)를 샘플링
# kmeans를 쓰면 데이터를 대표하는 50~100개의 점을 찾아줍니다. (속도 향상 핵심)
if num_shap_sample < num_data:
    X_train_summary = shap.kmeans(X_train_norm, num_shap_sample)
else:
    X_train_summary = X_train_norm

# ==========================================
# 3. Explainer 생성 (KernelExplainer 사용)
# ==========================================
# link="identity": 회귀(Regression) 문제이므로 identity 사용
explainer = shap.KernelExplainer(loaded_model.predict, X_train_summary)

# ==========================================
# 4. SHAP 값 계산 (Test 데이터 중 일부만)
# ==========================================
# X_test_norm: 스케일링된 테스트 데이터
# 전체를 다 넣으면 너무 오래 걸릴 수 있으니, 50~100개만 먼저 테스트해보세요.
nsamples = 100
print(f"⏳ SHAP 값 계산 중... (데이터 {nsamples}개 기준)")

shap_values = explainer.shap_values(X_test_norm[:nsamples])

# ==========================================
# 1. 요약 막대 그래프 (Bar Plot) 저장
# ==========================================
plt.figure() # 새로운 도화지 생성

# show=False 옵션이 중요합니다! (바로 화면에 뿌리지 않고 메모리에 잡아둠)
shap.summary_plot(
    shap_values,
    X_test_norm[:nsamples],
    feature_names=feature_names if 'feature_names' in locals() else None,
    plot_type="bar",
    show=False
)
# 파일로 저장 (dpi=300은 고화질 인쇄용 해상도)
plt.savefig(os.path.join(savepath, f'{data_name}_shap_bar_plot.png'), dpi=300, bbox_inches='tight')
plt.show()
plt.close() # 메모리 해제

# ==========================================
# 2. 상세 점 그래프 (Dot Plot) 저장
# ==========================================
plt.figure() # 새로운 도화지 생성

shap.summary_plot(
    shap_values,
    X_test_norm[:nsamples],
    feature_names=feature_names if 'feature_names' in locals() else None,
    show=False
)
# 파일로 저장
plt.savefig(os.path.join(savepath, f'{data_name}_shap_dot_plot.png'), dpi=300, bbox_inches='tight')
plt.show()
plt.close()
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
n_features = X_train_norm.shape[1]

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

Y_sobol = loaded_model.predict(X_sobol)

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
plt.savefig(os.path.join(savepath, f"{data_name}_sobol_analysis.png"))
plt.show()

