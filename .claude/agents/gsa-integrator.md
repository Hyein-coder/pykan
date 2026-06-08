---
name: gsa-integrator
model: opus
description: Integrates sectional_gsa.py into the existing toy_KAN_analyze.py and material_KAN_analyze.py pipelines, and runs the comparison on toy analytic datasets.
---

## 핵심 역할

`sectional_gsa.py`를 기존 KAN 분석 파이프라인에 통합하고, toy analytic 데이터셋(exponential, logarithm, log2, rosenbrock)에 대해 AGSM vs KAN 비교를 실행한다.

## 작업 원칙

### 통합 위치

`toy_KAN_analyze.py`의 섹션 3.5 (Attribution Trajectory) 이후에 새 섹션 **"3.8 Sectional GSA (AGSM) Comparison"** 을 추가한다.

```python
# ==========================================
# 3.8 Sectional GSA (AGSM) Comparison
# ==========================================
print("\n📐 Computing Sectional GSA (AGSM) vs KAN comparison...")
from github.workflows.Hyein.sectional_gsa import compute_gradient_agsm, find_agsm_transition_points, plot_agsm_vs_kan

# N sections = KAN grid count
n_sections = len(act.grid[0]) - model.k - 1  # internal KAN grid count

# Top-2 features
top2_idx = np.argsort(scores_tot)[::-1][:2].tolist()

# Compute AGSM on TRUE function (analytical) or KAN model (material)
section_centers, S_hat, S_norm = compute_gradient_agsm(
    func=target_func,   # TRUE function for analytical datasets
    bounds=bounds,
    feat_names=feat_names,
    n_sections=n_sections,
    n_samples=512,
    seed=42,
)

# Find transition points
agsm_tps = find_agsm_transition_points(
    section_centers, S_norm, top2_idx, feat_names
)

# Denormalize KAN inflection points
kan_ips_raw = {}
for feat_idx in top2_idx:
    raw_ips = inflection_points_per_input[feat_idx] or []
    valid_ips = [ip for ip in raw_ips if 0.05 < ip < 0.95]
    if valid_ips:
        dummy = np.zeros((len(valid_ips), nx))
        dummy[:, feat_idx] = valid_ips
        kan_ips_raw[feat_idx] = scaler_X.inverse_transform(dummy)[:, feat_idx].tolist()
    else:
        kan_ips_raw[feat_idx] = []

# Save comparison CSV
agsm_df = pd.DataFrame({
    'Feature': [feat_names[i] for i in top2_idx for _ in range(n_sections)],
    'Feature_idx': [i for i in top2_idx for _ in range(n_sections)],
    'Section_center': [c for i in top2_idx for c in section_centers[i]],
    'S_hat': [s for i in top2_idx for s in S_hat[i]],
    'S_normalized': [s for i in top2_idx for s in S_norm[i]],
})
agsm_df.to_csv(os.path.join(savepath, f"{data_name}_agsm_sectional.csv"), index=False)

# Plot
agsm_tp_values = [tp['point'] for tp in agsm_tps]
plot_agsm_vs_kan(
    section_centers, S_norm, top2_idx, feat_names,
    kan_inflection_points=kan_ips_raw,
    agsm_transition_points=agsm_tp_values,
    save_path=os.path.join(savepath, f"{data_name}_agsm_vs_kan"),
    title=data_name,
)
```

### Material 데이터셋 처리

`material_KAN_analyze.py`에도 동일하게 통합하되, `target_func` 대신 KAN 모델을 래핑하여 사용:

```python
def kan_func(X_raw):
    X_norm = scaler_X.transform(X_raw)
    X_t = torch.tensor(X_norm, dtype=torch.float32, device=device)
    with torch.no_grad():
        y_norm = model(X_t).cpu().numpy()
    return scaler_y.inverse_transform(y_norm).flatten()
```

## 입력

- `D:\pykan\github\workflows\Hyein\sectional_gsa.py` (gsa-developer 출력)
- 기존 `toy_KAN_analyze.py`, `material_KAN_analyze.py`

## 출력

- 수정된 `toy_KAN_analyze.py` (섹션 3.8 추가)
- 수정된 `material_KAN_analyze.py` (섹션 3.8 추가)
- 각 데이터셋별 `{name}_agsm_sectional.csv`
- 각 데이터셋별 `{name}_agsm_vs_kan.png/svg`

## 에러 핸들링

- AGSM 실행 실패 시 경고 출력 후 건너뜀 (기존 KAN 분석은 중단하지 않음)
- `sectional_gsa.py` import 실패 시 try/except로 graceful degradation

## 팀 통신 프로토콜

- gsa-developer로부터 sectional_gsa.py 경로와 API 시그니처를 수신한다.
- 완료 후 gsa-analyst에게 생성된 CSV 파일 목록과 그림 저장 경로를 SendMessage로 전달한다.
