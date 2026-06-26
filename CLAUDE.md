# KAN Project — Hyein

## 하네스: Sectional GSA

**목표:** Sectional GSA (AGSM, Pannier 2015)를 KAN 전환점 비교를 위한 베이스라인으로 구현하고 결과를 논문용 그림으로 정리한다.

**트리거:** Sectional GSA 구현, AGSM 분석, KAN 전환점 비교, Pannier 2015 구현, gradient 기반 구간 민감도 관련 작업 요청 시 `sectional-gsa-orchestrator` 스킬을 사용하라. 단순 질문은 직접 응답 가능.

**변경 이력:**
| 날짜 | 변경 내용 | 대상 | 사유 |
|------|----------|------|------|
| 2026-06-05 | 초기 구성 | 전체 | Sectional GSA 베이스라인 구현 요청 |

## 하네스: Adaptive Knot Placement

**목표:** 곡률 기반 적응형 B-spline knot 배치(Li et al., CAD 2005)를 KAN grid update에 구현하고, uniform/quantile grid와 비교한다.

**트리거:** adaptive knot 배치, 곡률 grid(`grid_mode='curvature'`), knot equidistribution, fidelity tier 업그레이드(Menger 곡률·변곡점·가변 knot 개수), adaptive vs uniform/quantile 비교 그림 관련 작업 요청 시 `adaptive-knots-orchestrator` 스킬을 사용하라. 알고리즘 질문은 `adaptive-knots` 스킬 참조로 직접 응답 가능.

**변경 이력:**
| 날짜 | 변경 내용 | 대상 | 사유 |
|------|----------|------|------|
| 2026-06-26 | 초기 구성 + minimal tier 구현 | kan/KANLayer.py, kan/MultKAN.py, 신규 에이전트·스킬 | 곡률 equidistribution knot 배치 구현 요청 |
