---
name: sectional-gsa
description: >
  Implement Sectional GSA (AGSM, Pannier 2015) as a baseline for KAN transition-point comparison.
  Computes gradient-based sectional sensitivity measures S^a_{l,[k]} per feature per section,
  identifies transition points where dominant feature changes, and compares with KAN inflection points.
  Use whenever asked to: implement Sectional GSA, run AGSM analysis, compare KAN inflection points
  with sensitivity-based transition points, add AGSM to the conventional comparison plot,
  or generate sectional sensitivity figures.
---

## 목적과 비교 논거

Sectional GSA (AGSM)는 KAN이 찾는 전환점(inflection point)이 얼마나 타당한지를 검증하는 베이스라인이다.

**KAN 방식**: 스플라인 계수의 2차 미분 부호 역전(`find_indices_sign_revert`)으로 전환점 탐지.
**AGSM 방식**: 구간별 편미분 절댓값 평균으로 민감도 측정 → 지배 변수가 바뀌는 지점 = 전환점.
**기대 결과**: 두 방법의 전환점이 일치 또는 KAN이 더 정밀함을 보임으로써 KAN의 강점을 강조.

## 수학 공식

전체 민감도 (Pannier 2015 Eq. 2):
$$\hat{S}_i = G_i = \frac{1}{V(H)}\int_H\left|\frac{\partial f}{\partial x_i}(x)\right|dx, \quad S_i = \frac{\hat{S}_i}{\sum_j \hat{S}_j}$$

구간 민감도 (AGSM):
$$\hat{S}^a_{l,[k]} = \frac{1}{V(H)} \int_{C_{l,[k]}} \left|\frac{\partial f}{\partial x_l}(x)\right|dx$$

$C_{l,[k]}$: $x_l \in A_{l,[k]}$이고 $x_j \in A_j$ (j≠l) 인 서브 공간

정규화:
$$S^a_{l,[k]} = \frac{\hat{S}^a_{l,[k]}}{\sum_{j=1}^n \sum_{m=1}^{N_j} \hat{S}^a_{j,[m]}}$$

MC 근사:
$$\hat{S}^a_{l,[k]} \approx \frac{(b_k - a_k)}{(b_l - a_l)} \cdot \frac{1}{M}\sum_{m=1}^M \left|\frac{\partial f}{\partial x_l}(x^{(m)})\right|$$

where $x_l^{(m)} \sim U(a_k, b_k)$, $x_j^{(m)} \sim U(A_j)$ for $j \neq l$.

## 구현 가이드라인 (사용자 지정)

- top-2 rank features에 대해 subdomains 생성
- N 구간수 = KAN의 grid 수와 동일
- 구간 옵션: 1) Equal-distance (기본) 2) Quantiles from data 3) KAN grid points 사용
- top-2 feature $x_i$, $x_j$ 각각에 대해 $S^a_{l,[k]}$ 계산 ($l = i$ and $l = j$)

## 구현 파일 위치

- 신규 모듈: `D:\pykan\github\workflows\Hyein\sectional_gsa.py`
- 통합 위치 (toy): `toy_KAN_analyze.py` 섹션 3.8 추가
- 통합 위치 (material): `material_KAN_analyze.py` 섹션 3.8 추가

## 기존 코드 패턴 재사용

수치 미분 대신 KAN 모델에는 autograd 활용:
```python
# KAN model gradient
X_t = torch.tensor(X_norm, dtype=torch.float32, device=device, requires_grad=True)
y = model(X_t)
y.sum().backward()
grad = X_t.grad[:, feat_idx].abs().detach().cpu().numpy()
```

## 상세 참조 문서

- `references/algorithm.md` — 전체 알고리즘 의사코드 + 수식 유도
- `references/integration-guide.md` — toy_KAN_analyze.py / material_KAN_analyze.py 통합 상세
