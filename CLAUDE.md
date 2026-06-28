# KAN Project — Hyein

## 하네스: Sectional GSA

**목표:** Sectional GSA (AGSM, Pannier 2015)를 KAN 전환점 비교를 위한 베이스라인으로 구현하고 결과를 논문용 그림으로 정리한다.

**트리거:** Sectional GSA 구현, AGSM 분석, KAN 전환점 비교, Pannier 2015 구현, gradient 기반 구간 민감도 관련 작업 요청 시 `sectional-gsa-orchestrator` 스킬을 사용하라. 단순 질문은 직접 응답 가능.

## 하네스: Curvature Inflection (New KAN analysis)

**목표:** 학습된 활성함수의 해석적 2차 미분으로 변곡점을 찾고, 그 구간마다 AGSM `S_a`와 KAN attribution을 함께 계산하여 무의미한 전환점 탐지를 줄인다.

**트리거:** New KAN analysis 구현, 해석적/B-spline 2차 미분, 곡률 기반 변곡점 탐지(계수 유한차분 대체), 변곡점 구간별 AGSM+attribution, curvature plot 수정/재실행 관련 요청 시 `curvature-inflection-orchestrator` 스킬을 사용하라. 단순 질문은 직접 응답 가능.

**변경 이력:**
| 날짜 | 변경 내용 | 대상 | 사유 |
|------|----------|------|------|
| 2026-06-05 | 초기 구성 | 전체 | Sectional GSA 베이스라인 구현 요청 |
| 2026-06-27 | curvature-inflection 스킬·오케스트레이터 + curvature-watcher 에이전트 추가 | skills/curvature-inflection(-orchestrator), agents/curvature-watcher, bspline_curvature.py, toy_KAN_analyze.py §3.9 | New KAN analysis(해석적 곡률 변곡점 + 구간별 dual measure) 구현 요청 |
| 2026-06-27 | 활성함수 φ와 해석적 1차·2차 미분(φ', φ'') 시각화 추가 | bspline_curvature.py(value/1st-deriv/edge_curves), toy_KAN_analyze.py §3.9 | "전체 함수와 해석적 도함수를 분석 스크립트에 그려라" 요청 |
| 2026-06-27 | symbolic-aware 곡률: symbolify된 엣지(spline mask=0)의 학습함수를 symbolic_fun에서 sympy 해석적 미분으로 반영 + 경고/라벨 | bspline_curvature.py(_symbolic_edge_deriv/_symbolic_branch/symbolic_edge_info), toy_KAN_analyze.py §3.9 | exponential/logarithm 등 단층 모델에서 활성·도함수가 0으로 나오는 버그 |
| 2026-06-27 | §3 변곡점 탐지를 계수 유한차분 → 해석적(find_inflection_points)으로 교체(단일 소스), 모든 그림이 해석적 변곡점 사용; 계수기반은 비교용 fallback 유지 | toy_KAN_analyze.py §3(+§3.5/3.7/3.8/3.9/4 연동) | 모든 그림이 해석적 변곡점을 표시하도록 요청 |
| 2026-06-27 | step 함수 그림 수정: 범례 KAN inflection 중복 제거, step 경계를 구간 knot(edge)에 정렬(where='post'), §3.8 attribution 마스킹도 실제 edge 사용 | toy_KAN_analyze.py §3.8/§3.9(_step_over_edges), sectional_gsa.py plot_agsm_vs_kan | agsm_modes 범례 중복·curvature_inflection 불연속·step edge가 grid knot에 안 맞음 |
| 2026-06-28 | 모델 전체 입력 미분(층 가로지르는 chain rule, ∂f/∂x·∂²f/∂x²)을 추가했다가 **폐기·되돌림** — 코드/문서 모두 엣지(바닥층) 단위로 환원 | bspline_curvature.py, toy_KAN_analyze.py §3/§3.6, skills/curvature-inflection(-orchestrator), CLAUDE.md | chain rule가 KAN 바닥층의 feature separation 이점을 무력화하므로 사용자가 폐기 결정 |
| 2026-06-28 | 랭킹 전환(작은 1차 미분) 분석 추가: 바닥층 `s_i=Σ_j|φ'_{ij}|` 랭킹, 지배 feature가 τ 아래로 떨어지는 전환점 + 그림 | bspline_curvature.py(feature_sensitivity/find_ranking_transitions), toy_KAN_analyze.py §3.6, skills/curvature-inflection | "작은 1차 미분으로 랭킹 전환을 찾자" 요청 |
