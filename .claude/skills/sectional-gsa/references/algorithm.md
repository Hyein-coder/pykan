# Sectional GSA (AGSM) — 알고리즘 상세

## 전체 알고리즘 의사코드

```
Input:
  func: callable (X [N,d]) -> y [N]
  bounds: [[lo_0, hi_0], ..., [lo_d-1, hi_d-1]]
  feat_names: [str * d]
  top2_idx: [i, j]  (top-2 global feature indices)
  n_sections: int   (= KAN grid count)
  n_samples: int    (MC samples per section, default 512)
  section_mode: 'equal' | 'quantile' | 'kan'

Compute global S_hat for normalization denominator:
  For each feature l in {0, ..., d-1}:
    S_hat_global[l] = approximate_global_sensitivity(func, bounds, l, n_samples)

For each target feature l in top2_idx:
  Compute section edges for feature l:
    if mode == 'equal':   edges = linspace(lo_l, hi_l, n_sections+1)
    if mode == 'quantile': edges = quantiles of training data column l
    if mode == 'kan':     edges = KAN grid knots for feature l (denormalized)

  For each section k in {0, ..., n_sections-1}:
    a_k, b_k = edges[k], edges[k+1]
    
    Sample M points:
      X[:, l] ~ U(a_k, b_k)
      X[:, j] ~ U(lo_j, hi_j) for j != l
    
    Compute |df/dx_l| at all M points (numerical or autograd)
    
    S_hat[l][k] = (b_k - a_k) / (hi_l - lo_l) * mean(|df/dx_l|)
  
  section_centers[l] = midpoints of edges

Normalization denominator:
  denom = sum over all l, all k of S_hat[l][k]

S_normalized[l][k] = S_hat[l][k] / denom

Find transition points for pair (i, j):
  If len(top2_idx) == 2 and domains are compatible:
    For each section k:
      Plot S_normalized[i][k] vs S_normalized[j][k] at section_centers[i][k]
    Transition = interpolated crossing where S_norm[i][k] == S_norm[j][k]
```

## 수치 미분

```python
def numerical_gradient_magnitude(func, X, feat_idx, eps=1e-5):
    X_plus = X.copy(); X_plus[:, feat_idx] += eps
    X_minus = X.copy(); X_minus[:, feat_idx] -= eps
    return np.abs((func(X_plus) - func(X_minus)) / (2 * eps))
```

함수가 numpy 배열을 받지 않는 경우 (element-wise):
```python
def wrap_func(func):
    def wrapped(X):
        return np.apply_along_axis(func, 1, X)
    return wrapped
```

## KAN Autograd Gradient

```python
def kan_gradient_magnitude(model, X_raw, feat_idx, scaler_X, device):
    X_norm = scaler_X.transform(X_raw)
    X_t = torch.tensor(X_norm, dtype=torch.float32, device=device, requires_grad=True)
    y = model(X_t)
    y.sum().backward()
    return X_t.grad[:, feat_idx].abs().detach().cpu().numpy()
```

## 전환점 탐지

```python
def find_agsm_transition_points(section_centers_i, S_i, section_centers_j, S_j,
                                 feat_name_i, feat_name_j):
    """
    Find crossing points of S_i and S_j curves.
    Assumes section_centers_i == section_centers_j (same domain bounds).
    
    Returns list of dicts:
        {'point': float, 'from_feat': str, 'to_feat': str}
    """
    transitions = []
    diff = S_i - S_j  # positive: feature i dominates
    for k in range(len(diff) - 1):
        if diff[k] * diff[k+1] < 0:  # sign change -> crossing
            # Linear interpolation for crossing
            x1, x2 = section_centers_i[k], section_centers_i[k+1]
            d1, d2 = diff[k], diff[k+1]
            crossing = x1 - d1 * (x2 - x1) / (d2 - d1)
            if diff[k] > 0:
                transitions.append({'point': crossing, 'from': feat_name_i, 'to': feat_name_j})
            else:
                transitions.append({'point': crossing, 'from': feat_name_j, 'to': feat_name_i})
    return transitions
```

## 정규화 고려사항

두 feature의 도메인이 다를 때:
- 구간 너비 비율 `(b_k - a_k) / (hi_l - lo_l)` 를 곱하면 도메인 크기 차이를 보정
- 동일 도메인 (대부분의 analytical function)에서는 자동으로 약분됨

Section이 단일 점 또는 매우 좁을 때:
- 최소 5개 이상의 MC 샘플이 구간 내에 있어야 신뢰할 수 있음
- 샘플 부족 구간은 NaN으로 마킹 후 시각화에서 제외

## KAN 전환점 vs AGSM 전환점 비교 지표

```python
def compute_alignment_metrics(kan_ips, agsm_tps, domain_width):
    """
    kan_ips: list of float (raw space)
    agsm_tps: list of dicts {'point': float}
    domain_width: hi - lo of the sectioned feature
    """
    results = []
    for kip in kan_ips:
        nearest = min(agsm_tps, key=lambda t: abs(t['point'] - kip), default=None)
        if nearest is not None:
            diff = abs(kip - nearest['point'])
            rel_err = diff / domain_width
            results.append({
                'KAN_inflection': kip,
                'AGSM_transition': nearest['point'],
                'abs_diff': diff,
                'rel_error': rel_err,
            })
    return results
```
