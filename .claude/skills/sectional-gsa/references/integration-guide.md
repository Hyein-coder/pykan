# Sectional GSA — 기존 코드 통합 가이드

## toy_KAN_analyze.py 통합 포인트

### 삽입 위치
섹션 3.5 (Attribution Trajectory) 이후, 섹션 4 (Range-Based Attribution Scoring) 이전.

### 구간 수 추출
```python
# KAN grid 수 = 내부 knot 수
n_grid = len(act.grid[0]) - model.k - 1
# 또는 모델 config에서:
# n_grid = config.get('grid', 10)
```

### 정규화 전 KAN inflection point 추출
inflection_points_per_input[feat_idx]는 정규화된 [0.1, 0.9] 공간의 값이다.
raw space로 변환:
```python
def denorm_inflection(ips_norm, feat_idx, nx, scaler_X):
    valid = [ip for ip in (ips_norm or []) if 0.05 < ip < 0.95]
    if not valid:
        return []
    dummy = np.zeros((len(valid), nx))
    dummy[:, feat_idx] = valid
    return scaler_X.inverse_transform(dummy)[:, feat_idx].tolist()
```

### Analytical Function 래핑
FUNCTION_ZOO의 func는 single-row array를 받음 → numpy 배치 처리로 래핑 필요:
```python
def make_batch_func(single_func):
    return lambda X: np.apply_along_axis(single_func, 1, X).flatten()

batch_func = make_batch_func(target_func)
```

### 삽입 코드 스니펫
```python
# ==========================================
# 3.8 Sectional GSA (AGSM) Comparison
# ==========================================
try:
    from github.workflows.Hyein.sectional_gsa import (
        compute_gradient_agsm, find_agsm_transition_points,
        plot_agsm_vs_kan, make_sections
    )
    
    n_sections = len(act.grid[0]) - model.k - 1
    top2_idx = np.argsort(scores_tot)[::-1][:2].tolist()
    
    batch_func = lambda X: np.apply_along_axis(target_func, 1, X).flatten()
    
    section_centers, S_hat, S_norm = compute_gradient_agsm(
        func=batch_func,
        bounds=bounds,
        feat_names=feat_names,
        top2_idx=top2_idx,
        n_sections=n_sections,
        n_samples=512,
        seed=42,
        section_mode='equal',
    )
    
    kan_ips_raw = {
        idx: denorm_inflection(inflection_points_per_input[idx], idx, nx, scaler_X)
        for idx in top2_idx
    }
    
    agsm_tps = find_agsm_transition_points(
        section_centers[top2_idx[0]], S_norm[top2_idx[0]],
        section_centers[top2_idx[1]], S_norm[top2_idx[1]],
        feat_names[top2_idx[0]], feat_names[top2_idx[1]],
    )
    
    # Save CSV
    rows = []
    for feat_idx in top2_idx:
        for k, (center, s_hat, s_norm) in enumerate(
            zip(section_centers[feat_idx], S_hat[feat_idx], S_norm[feat_idx])
        ):
            rows.append({'Feature': feat_names[feat_idx], 'Section_k': k,
                         'Section_center': center, 'S_hat': s_hat, 'S_normalized': s_norm})
    pd.DataFrame(rows).to_csv(
        os.path.join(savepath, f"{data_name}_agsm_sectional.csv"), index=False
    )
    
    # Plot
    plot_agsm_vs_kan(
        section_centers=section_centers,
        S_normalized=S_norm,
        top2_idx=top2_idx,
        feat_names=feat_names,
        kan_inflection_points=kan_ips_raw,
        agsm_transition_points=[tp['point'] for tp in agsm_tps],
        save_path=os.path.join(savepath, f"{data_name}_agsm_vs_kan"),
        title=data_name,
    )
    print(f"✅ AGSM comparison saved to: {savepath}")
    
except Exception as e:
    print(f"⚠️ AGSM comparison failed: {e}")
```

## material_KAN_analyze.py 통합

### KAN 모델을 func으로 래핑
```python
def make_kan_func(model, scaler_X, scaler_y, device):
    def kan_func(X_raw):
        X_norm = scaler_X.transform(X_raw)
        X_t = torch.tensor(X_norm, dtype=torch.float32, device=device)
        with torch.no_grad():
            y_norm = model(X_t).cpu().numpy()
        return scaler_y.inverse_transform(y_norm).flatten()
    return kan_func
```

### Bounds 추출
```python
# Material 데이터셋의 bounds는 scaler_X에서 추출
bounds = [[scaler_X.data_min_[i], scaler_X.data_max_[i]] for i in range(n_features)]
```

## plot_conventional.ipynb 업데이트

`SOURCES` dict에 AGSM 추가:
```python
SOURCES['AGSM'] = ("{name_data}_agsm_global.csv", 'S_global')
```

전체 민감도 합산 (AGSM global = sum of S_hat over all sections):
- `sectional_gsa.py`에 `compute_global_from_sectional(S_hat, section_widths)` 함수 추가
- 결과를 `{name_data}_agsm_global.csv`로 저장

## 파일 저장 규칙

```
analytical_results/{name}/kan_models/
├── {name}_agsm_sectional.csv    # section-by-section S values
├── {name}_agsm_vs_kan.png/svg   # comparison plot
└── {name}_agsm_global.csv       # global scores (for bar plot)
```
