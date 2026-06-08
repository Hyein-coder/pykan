---
name: sectional-gsa-orchestrator
description: >
  Orchestrates the full Sectional GSA (AGSM) implementation pipeline for the KAN project.
  Triggers when asked to: implement Sectional GSA baseline, compare KAN transition points
  with AGSM, add AGSM comparison to the paper, run sectional sensitivity analysis,
  implement Pannier 2015, compute gradient-based sectional measures, or update the KAN
  conventional-method comparison to include AGSM. Also handles re-runs, partial updates,
  and "fix the AGSM plot" type requests.
---

## 실행 모드: 에이전트 팀 (파이프라인)

4명의 팀원이 순차적으로 협업한다. 각 에이전트는 이전 에이전트의 산출물을 받아 다음 단계를 수행한다.

## Phase 0: 컨텍스트 확인

실행 전 기존 산출물을 확인하여 실행 모드를 결정한다:

```
_workspace/ 존재 여부 확인
├── 없음 → 초기 실행 (Phase 1부터)
├── 있고 sectional_gsa.py도 있음 →
│   사용자가 "fix plot" 요청: gsa-analyst만 재실행
│   사용자가 "new dataset" 요청: gsa-integrator + gsa-analyst 재실행
│   사용자가 "fix implementation" 요청: gsa-developer + gsa-integrator + gsa-analyst 재실행
└── sectional_gsa.py 없음 → gsa-developer부터 재실행
```

## Phase 1: 논문 분석 (gsa-researcher)

**담당**: gsa-researcher 에이전트
**소요 시간**: ~5분
**입력**: PDF + 사용자 노트 경로
**출력**: `_workspace/01_gsa_algorithm_spec.md`

PDF 경로: `D:\ObsidianVault\KAN\VibeCoding\References\KAN\RESS_2015_Pannier_Sectional global sensitivity measures.pdf`

사용자 노트 경로: `D:\ObsidianVault\KAN\VibeCoding\Projects\KAN\Sectional GSA as a baseline.md`

**핵심 추출 항목:**
- AGSM 수식: $\hat{S}^a_{l,[k]} = \frac{1}{V(H)} \int_{C_{l,[k]}} |\partial f/\partial x_l| dx$
- 정규화: $S^a_{l,[k]} = \hat{S}^a_{l,[k]} / (\sum_{j,m} \hat{S}^a_{j,[m]})$
- 구현 가이드라인 (사용자 노트): top-2 features, N = KAN grid count, 3가지 section 모드

## Phase 2: 구현 (gsa-developer)

**담당**: gsa-developer 에이전트
**소요 시간**: ~10분
**입력**: `_workspace/01_gsa_algorithm_spec.md`
**출력**: `sectional_gsa.py`

필수 함수:
- `compute_gradient_agsm(func, bounds, feat_names, top2_idx, n_sections, ...)` → (section_centers dict, S_hat dict, S_norm dict)
- `make_sections(bounds_l, n_sections, mode, ...)` → (edges, centers)
- `find_agsm_transition_points(...)` → list of transition dicts
- `plot_agsm_vs_kan(...)` → saves PNG/SVG/EPS

## Phase 3: 통합 및 실행 (gsa-integrator)

**담당**: gsa-integrator 에이전트
**소요 시간**: ~15분
**입력**: `sectional_gsa.py` 경로 + API 시그니처
**출력**: 수정된 `toy_KAN_analyze.py`, CSV 파일들, 비교 그림들

통합 위치: `toy_KAN_analyze.py` 섹션 3.5 이후, 섹션 4 이전

**실행 대상 데이터셋 (우선순위):**
1. `exponential` (가장 명확한 전환점 예상: $e^{-2x_0}$ vs $x_1$)
2. `logarithm`
3. `log2`
4. `rosenbrock`

## Phase 4: 분석 및 시각화 (gsa-analyst)

**담당**: gsa-analyst 에이전트
**소요 시간**: ~10분
**입력**: CSV 파일들 + KAN inflection point 데이터
**출력**: 논문용 비교 그림들, 수치 지표 CSV

최종 저장 위치: `figures_for_paper/`

## 데이터 전달 프로토콜

- **파일 기반**: 모든 중간 산출물은 `_workspace/`에 저장
- **메시지 기반**: 각 에이전트는 완료 후 다음 에이전트에게 파일 경로를 SendMessage로 전달
- `_workspace/` 디렉토리: `D:\pykan\github\workflows\Hyein\_workspace\`

## 에러 핸들링

- PDF 읽기 실패: 사용자 노트만으로 Phase 1 진행 가능 (노트에 완전한 수식 포함)
- AGSM 실행 실패: 기존 KAN 분석 파이프라인은 영향받지 않음 (try/except로 감쌈)
- 전환점 미발견: 해당 데이터셋에서 지배 변수 변화 없음으로 기록하고 계속 진행

## 테스트 시나리오

**정상 흐름**:
1. "exponential" 함수에서 KAN이 x0 ≈ -0.35 (raw)에서 전환점 발견
2. AGSM이 같은 구간에서 S_norm[0] == S_norm[1] 교차점 발견
3. 두 전환점의 절대 차이 < 5% of domain width

**에러 흐름**:
1. PDF 읽기 실패 → 사용자 노트 기반으로 Phase 1 완료 → 나머지 정상 진행
2. n_sections이 KAN grid보다 작을 때 → 최소 5 사용

## 후속 작업 지원

다음 표현이 포함된 요청에서 이 스킬을 재사용:
- "AGSM 다시 실행", "sectional GSA 수정", "전환점 비교 업데이트"
- "AGSM plot 수정", "새 데이터셋에 AGSM 추가"
- "material 데이터셋에 AGSM 적용"
- "AGSM 결과로 논문 그림 업데이트"
