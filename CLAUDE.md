# KAN Project — Hyein

KAN 전환점 분석용 하네스 2개. 작업 요청이 트리거에 맞으면 해당 오케스트레이터 스킬을 쓰고, 단순 질문은 직접 응답한다.
각 변경 이력 항목 형식: **날짜** · 변경 내용 *(대상 파일)* — 사유.

---

## 하네스 1 — Sectional GSA

**목표** Sectional GSA (AGSM, Pannier 2015)를 KAN 전환점 비교 베이스라인으로 구현하고 결과를 논문용 그림으로 정리한다.

**트리거 → `sectional-gsa-orchestrator`** Sectional GSA 구현 · AGSM 분석 · KAN 전환점 비교 · Pannier 2015 구현 · gradient 기반 구간 민감도.

**변경 이력**
- **2026-06-05** · 초기 구성 *(전체)* — Sectional GSA 베이스라인 구현 요청.

---

## 하네스 2 — Curvature Inflection (New KAN analysis)

**목표** 학습된 활성함수의 해석적 2차 미분으로 변곡점을 찾고, 구간마다 AGSM `S_a`와 KAN attribution을 함께 계산하여 무의미한 전환점 탐지를 줄인다.

**트리거 → `curvature-inflection-orchestrator`** New KAN analysis 구현 · 해석적/B-spline 2차 미분 · 곡률 기반 변곡점 탐지(계수 유한차분 대체) · 변곡점 구간별 AGSM+attribution · symbolification robustness · curvature plot 수정/재실행.

**변경 이력**

- **2026-06-27**
  - curvature-inflection 스킬·오케스트레이터 + curvature-watcher 에이전트 추가 *(skills/curvature-inflection(-orchestrator), agents/curvature-watcher, bspline_curvature.py, toy_KAN_analyze.py §3.9)* — New KAN analysis(해석적 곡률 변곡점 + 구간별 dual measure) 구현 요청.
  - 활성함수 φ와 해석적 1·2차 미분(φ', φ'') 시각화 추가 *(bspline_curvature.py: value/1st-deriv/edge_curves, §3.9)* — "전체 함수와 해석적 도함수를 분석 스크립트에 그려라" 요청.
  - symbolic-aware 곡률: symbolify된 엣지(spline mask=0)의 학습함수를 symbolic_fun에서 sympy 해석적 미분으로 반영 + 경고/라벨 *(bspline_curvature.py: _symbolic_edge_deriv/_symbolic_branch/symbolic_edge_info, §3.9)* — exponential/logarithm 등 단층 모델에서 활성·도함수가 0으로 나오는 버그.
  - §3 변곡점 탐지를 계수 유한차분 → 해석적(find_inflection_points)으로 교체(단일 소스); 계수기반은 비교용 fallback 유지 *(toy_KAN_analyze.py §3 + §3.5/3.7/3.8/3.9/4 연동)* — 모든 그림이 해석적 변곡점을 표시하도록 요청.
  - step 함수 그림 수정: 범례 KAN inflection 중복 제거, step 경계를 구간 knot(edge)에 정렬(where='post'), §3.8 attribution 마스킹도 실제 edge 사용 *(toy_KAN_analyze.py §3.8/§3.9 _step_over_edges, sectional_gsa.py plot_agsm_vs_kan)* — agsm_modes 범례 중복·curvature_inflection 불연속·step edge가 grid knot에 안 맞음.

- **2026-06-28**
  - 모델 전체 입력 미분(층 가로지르는 chain rule, ∂f/∂x·∂²f/∂x²) 추가 후 **폐기·되돌림** — 코드/문서 모두 엣지(바닥층) 단위로 환원 *(bspline_curvature.py, §3/§3.6, skills/curvature-inflection(-orchestrator), CLAUDE.md)* — chain rule가 KAN 바닥층의 feature separation 이점을 무력화하므로 사용자가 폐기 결정.
  - 랭킹 전환(작은 1차 미분) 분석 추가: 바닥층 `s_i=Σ_j|φ'_{ij}|` 랭킹, 지배 feature가 τ 아래로 떨어지는 전환점 + 그림 *(bspline_curvature.py: feature_sensitivity/find_ranking_transitions, §3.6, skills/curvature-inflection)* — "작은 1차 미분으로 랭킹 전환을 찾자" 요청.
  - symbolification robustness 테스트 추가(랭킹 전환점의 위치·개수가 symbolify에 강건한지): 같은 학습 모델의 spline reading(마스크 역전) vs symbolic reading 대조 드라이버 + 집계 그림; watcher를 교정 루프로 격상(symbolify 실패 분류표) *(robustness_symbolify.py 신규, skills/symbolification-robustness 신규, curvature-inflection-orchestrator Phase 4, agents/curvature-watcher 게이트 4·5)* — symbolification robustness 요청 + watcher가 실패 시 교정하도록.

- **2026-06-29**
  - §4 range score 그림을 전환점 있는 **모든** feature에 대해 출력(이전엔 첫 유효 feature 1개만): `_build_interval_masks`/`_compute_interval_scores`/`_plot_interval_scores` 헬퍼로 분리, 루프에서 feature별 `{data}_scores_interval_x#.png` 저장. 첫 유효 feature는 downstream split-data용으로 유지. 잔존 `mask_idx` 오라벨 수정 *(toy_KAN_analyze.py §4/§5)* — "전환점 탐지된 모든 feature에 대해 그려라" 요청.
  - log2 등에서 symbolify된 sqrt/log/거듭제곱 엣지의 RuntimeWarning(invalid sqrt/log/power) 해결: ① sweep을 패딩 제외 **데이터 구간 knot**(`grid[i,k:k+G+1]`)으로 클램프(`data_range_knots` 헬퍼) ② `_symbolic_branch`가 off-target 입력열을 0으로 채워 폐기하는 구조 탓에 sqrt(0-0.0425) 등 도메인 밖 평가가 발생 → 평가부(`_evaluated`)를 `np.errstate(invalid/divide='ignore')`로 감쌈(값은 nan_to_num으로 이미 정화·폐기됨) *(bspline_curvature.py: data_range_knots/find_inflection_points/find_ranking_transitions/verify_against_autograd/_evaluated, toy_KAN_analyze.py §3/§3.6/§3.9 sweep)* — log2 RuntimeWarning 조사 요청. 결과(전환점 위치·개수)는 불변 확인.
  - scores_interval 그림 갤러리를 `figures_for_paper/scores_interval_all/`로 이동(이전 `analytical_results/scores_interval_all/`); 갤러리 경로를 `os.getcwd()` 기준으로 고정 *(toy_KAN_analyze.py `_plot_interval_scores`)* — 사용자가 폴더 이동 후 반영 요청.
  - scores_interval 구간 라벨을 정규화값 → **raw 입력값**으로 표시(`scaler_X.inverse_transform`, masking은 정규화 공간 유지). CSV `Interval_Label`도 raw *(toy_KAN_analyze.py `_build_interval_masks`)* — "x 범위를 raw로 표기" 요청.
  - 곡률/2차 미분/변곡점 제거(toy_KAN_analyze.py만): `inflection_points_per_input`·`find_inflection_points`·계수 2차차분(`slope_2nd`)·φ'' 곡선·변곡점 vline(green/purple) 전부 삭제. 활성함수 그림(activations_values, activation_derivatives)은 φ·φ'·**ranking transition(orange)** 만 남겨 슬림화. ranking transition 계산을 §3 앞으로 끌어올려(`transition_points_per_input` 단일 소스) §3 그림에서도 표시, §3.6은 그 결과로 그림만 그림. `bspline_curvature.py`의 곡률/2차도함수 함수는 다른 스크립트(sweep_spline_inflection.py·material_KAN_analyze.py·toy_KAN_analyze_multi.py·QA)가 쓰므로 **보존**. dual-measure 그림(ranking transition 기반)은 유지하고 파일명을 `curvature_inflection` → **`transition_dual_measure`**로 리네임(구 산출물 56개 삭제) *(toy_KAN_analyze.py §3/§3.6/§3.9, split_data pkl, import)* — "곡률 더 이상 불필요" 요청. conditional symbolic·damping_sin 검증.
  - normalized/raw 공간 규약 통일: knots·inflection·ranking-transition 점은 **정규화 공간**으로 내부 보관(소스 오브 트루스), **플롯에서만** raw로 역변환(단일 `denorm(vals, feat_idx)` 헬퍼). per-feature 플롯(§3 activations/activations_values, §3.5 trajectory, §3.9 activation_derivatives)을 raw 축으로 전환; 이미 raw였던 §3.7 contour·AGSM·dual-measure·scores_interval은 유지. **예외**: §3.6 ranking_transition은 여러 feature를 한 축에 겹쳐 비교하므로(피처별 raw bound 상이) 정규화 축 유지 *(toy_KAN_analyze.py: denorm 헬퍼 + §3/§3.5/§3.9 plot sites)* — 플롯마다 normalized/raw가 섞여 전환점 위치가 그림마다 달라 보이는 문제. conditional symbolic로 raw 축·vline 위치 검증.
  - 모델 reading 모드 CLI 옵션 추가: `--model-mode {as-saved,spline,symbolic}`(+`--refit`). spline=symbolify 엣지를 spline 가지로 역전(symbolic off), symbolic=symbolic 가지 사용(순수 spline 저장 모델은 auto_symbolic, `--refit` 시 짧은 LBFGS). `robustness_symbolify._build_spline_reading`/`_build_symbolic_reading` 재사용(중복 없음). 비-default 모드는 `kan_models/<mode>/` 서브폴더 + 갤러리 파일명 mode 태그로 as-saved 산출물 보존 *(toy_KAN_analyze.py §2.1 + argparse + import)* — 저장모델 그대로/순수 spline/순수 symbolic 중 선택 요청. log2(invert)·damping_sin(introduce) 3모드+refit 검증 완료.
