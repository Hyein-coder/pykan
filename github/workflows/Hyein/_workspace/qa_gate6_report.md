# QA GATE 6 — rel-threshold sweep (`sweep_rel_threshold.py`)

**Verdict: PASS** (all 6 checks). Independent adversarial gate; verified by execution in `pykan-new`.

Target: `github/workflows/Hyein/sweep_rel_threshold.py`
Outputs: `figures_for_paper/rel_threshold_sweep.csv`, `figures_for_paper/rel_threshold_recommendation.csv`

---

## Checklist

### 1. x_grid convention — PASS
- Script calls `find_ranking_transitions(model, rel_thresh=rt, layer=layer)` (lines 113-114) with **no** explicit `x_grid`.
- Default branch in `bspline_curvature.find_ranking_transitions` (line 640-643): `x_grid is None` → `data_range_knots(act)` → `grid[..., k:k+G+1]` (data-range knots).
- Does NOT replicate the old `robustness_symbolify._shared_x_grid` convention `grid[:, k-1:-2]` (which is passed explicitly there). Correct new convention. PASS.

### 2. Numeric reproduction (core) — PASS
Independent throwaway `_workspace/gate6_recompute.py` (does not import the sweep), loaded conditional model via `KANRegressor(device='cpu').load_model(...)`, called `find_ranking_transitions(model, rel_thresh=rt, layer=0)` directly:

| rel_thresh | recomputed n_down | recomputed loc_err | CSV loc_err | recomputed tau | CSV tau |
|---|---|---|---|---|---|
| 0.10 | 1 | 0.019048 | 0.0190476... | 0.177894 | 0.1778941... |
| 0.20 | 1 | 0.001003 | 0.0010025... | 0.355788 | 0.3557882... |

Matches the expected (n_down=1, loc_err≈0.0190 at 0.10; n_down=1, loc_err≈0.0010 at 0.20). Down-points: 0.519 (0.10), 0.499 (0.20). PASS.

Additionally re-ran the full script to scratchpad: **both CSVs reproduce byte-identically** to the committed copies (`diff` clean).

### 3. feat index + normalized/raw — PASS
- `FUNCTION_ZOO['conditional']`: `func = x0*2 + x1 if x0<0 else x1`, names `["Conditional (x0)", "Linear (x1)"]`, bounds `[[-1,1],[-1,1]]`. feat 0 = conditional input. `FEAT_IDX = 0`. PASS.
- `points_norm` from `t['point']` (normalized x). `points_raw` via `_denorm_feature` → `scaler_X.inverse_transform` (line 91). PASS.
- True-kink uses `TRUE_KINK_NORM = 0.5` (raw 0; x0=0 midpoint of [-1,1]). At rt=0.20, points_norm=0.4990 → points_raw=-0.0026 ≈ raw 0, consistent. PASS.

### 4. Recommendation logic — PASS
- detected&clean rows: rel_thresh ∈ {0.06, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40} — contiguous run, 0.50 (the only non-good row) sits at the end.
- band [min,max] = [0.06, 0.40]. Matches rec CSV.
- min loc_err among detected&clean = 0.001003 at rel_thresh=0.20 → recommended 0.20. Matches rec CSV.
- No-match path is explicit: `recommend()` (lines 197-222) sets `recommended_rel_thresh=NaN` and a descriptive `note` ("NO detected&clean..." / "NO detected...") rather than silently picking. PASS.

### 5. No existing script modified — PASS (with noted pre-existing change)
`git status` / `git diff --stat`:
- **New files only**: `sweep_rel_threshold.py`, `figures_for_paper/rel_threshold_sweep.csv`, `figures_for_paper/rel_threshold_recommendation.csv`. (CLAUDE.md also modified — changelog, not a target script.)
- **Pre-existing, NOT from this task**: `toy_KAN_sweep.py` — its `main()` hyperparameter-search block (steps/sym_range/n_iter/symbolic_enabled). `sweep_rel_threshold.py` imports ONLY `KANRegressor, FUNCTION_ZOO` from it (line 46) and never calls `main()`. Not a task failure; noted.
- `bspline_curvature.py`, `toy_KAN_analyze.py`, `robustness_symbolify.py`: unchanged.

### 6. Robustness — PASS
Per-function `try/except` (lines 249-275): a failing/missing model logs traceback, appends a NaN recommendation row with `note='failed: ...'`, and the loop continues. PASS.

---

## Reproduction commands
```
PYTHONPATH=. PYTHONUTF8=1 python github/workflows/Hyein/_workspace/gate6_recompute.py
PYTHONPATH=. PYTHONUTF8=1 python -m github.workflows.Hyein.sweep_rel_threshold --out <tmp>
```
(python = `C:/Users/user/miniconda3/envs/pykan-new/python.exe`, cwd `D:\pykan`)
