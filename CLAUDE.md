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
