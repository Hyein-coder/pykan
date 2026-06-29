# robustness_symbolify.py — implementation notes

Driver: `github/workflows/Hyein/robustness_symbolify.py`
Run in `pykan-new`. Invoke from the repo root with the package importable, e.g.
`PYTHONPATH=. python github/workflows/Hyein/robustness_symbolify.py [func_name]`
(the script uses `from github.workflows.Hyein...` absolute imports, so `D:\pykan`
must be on `sys.path`).

## What it measures
How robust the **location and number** of KAN ranking-transition points
(`s_i(x)=Σ_j|φ'_{i,j}(x)|` crossing τ; `bspline_curvature.find_ranking_transitions`)
are to **symbolification**. It builds two readings of the **same persisted
network** — never retrains the whole model — and compares the transitions each
yields, in raw input space.

## The two construction directions
Detected per function via `symbolic_edge_info(model, 0)` (non-empty ⇒ already
symbolified). The source of truth is the on-disk masks, not the sweep code.

- **introduce-symbolify** (pure spline saved: `conditional`, `damping_sin`,
  `ishigami`): the symbolic reading is *constructed* in-memory by
  `auto_symbolic` with KANRegressor production defaults
  (`lib`=12-fn library, `a_range=b_range=(-10,10)`, `r2_threshold=0`,
  `weight_simple=0`). The spline reading is the as-saved model with the symbolic
  branch disabled (a no-op restore).
- **invert-symbolify** (already symbolified: `log2`, `exponential`,
  `logarithm`, `rosenbrock`): the symbolic reading is the as-saved state (we do
  NOT re-run auto_symbolic). The spline reading is *reconstructed* by restoring
  only the symbolified edges onto their persisted spline branch
  (`act_fun[0].mask[i,j]=1` for edges in `symbolic_edge_info`) and setting
  `symbolic_enabled=False`. Genuinely dead/pruned edges (mask=0, not in
  `symbolic_edge_info`) are left untouched.

## Why the deepcopy happens before any forward pass
A `model.forward(...)` caches non-leaf tensors (`spline_preacts`, etc.) that
break `copy.deepcopy` ("Only Tensors created explicitly by the user … support
deepcopy"). So we deepcopy both readings from the **freshly loaded, un-forwarded
model**. `symbolic_edge_info` and the shared-grid construction read masks/grid
only (no forward). The symbolic copy runs its own forward inside
`auto_symbolic`; `find_ranking_transitions` runs its own forwards on each copy.

## The refit asymmetry
Saved-symbolic models went through production refit (short LBFGS after
auto_symbolic). The in-memory introduce-symbolify path is by default **no-refit**
— it isolates the pure symbolify substitution. `--refit` reproduces the
production path (mirrors `toy_KAN_sweep.py` ~205-206) but is asymmetric: refit
perturbs even the non-symbolified spline edges, so if drift becomes
refit-dominated, report the no-refit run as primary. `--refit` only affects
introduce-symbolify (pure-spline) functions; invert-symbolify functions ignore it.

## Comparability
Both readings use the **same** `x_grid` (dense linspace over the layer-0 interior
knot range `act.grid[:, k-1:-2]`, 400 pts), `rel_thresh`, and `layer=0`. τ is
computed per reading (`rel_thresh · max_i max_x s_i`), so each reading uses its
own "small" scale — this matches `find_ranking_transitions`' own convention.

## Metrics (raw input space)
Each transition `point` (normalized) → raw via `scaler_X.inverse_transform`
(full-width dummy row, replace the feature column; §3.6/§3.8 convention).
- **number**: `n_spline`, `n_sym` (total, and dominant-only counts).
- **location**: per-feature greedy nearest-neighbour matching of the two point
  sets; tolerance = `match_frac × domain_width` (default 0.05). Yields
  matched/added/dropped and per-pair drift Δx (raw and as fraction of width).
  `added` = symbolic points with no spline partner (introduced); `dropped` =
  spline points with no symbolic partner (removed).
- **stability summary/function**: `count_match_rate` =
  matched / (matched+added+dropped), median & max |drift|, n_edges_symbolified,
  status.

## Status taxonomy (never silently skipped; recorded in CSV + summary)
- `ok` — both readings produced usable s_i; transitions compared.
- `symbolify_failed` — `auto_symbolic` raised (introduce path). Spline-only row
  emitted; symbolic side empty; continue.
- `no_symbolification` — auto_symbolic produced 0 symbolified edges on a
  pure-spline model. Null contrast (symbolic side empty); continue.
- `symbolic_degenerate` — symbolic-side s_i flat/NaN everywhere while spline-side
  is nonzero (peak |s_i| < 1e-9 over the grid).
- `spline_reading_unavailable` — already-symbolified model whose mask-inverted
  spline reading is s_i≈0 everywhere; symbolic-only reported.
Each function is wrapped in try/except so one failure never aborts the set; an
unexpected exception is recorded as `unexpected_error`.

## Outputs
Per function in `analytical_results/{func}/kan_models/`:
- `{func}_symbolify_robustness.{png,svg,eps}` — per-feature s_i(x): spline
  (solid) vs symbolic (dashed), both τ lines, transition vlines colored by branch
  (spline solid / symbolic dotted); title shows direction + status.
- `{func}_symbolify_robustness.csv` — paired transitions: feature, branch
  (matched/spline=dropped/symbolic=added), raw points, matched/added/dropped
  flags, drift, drift_frac, symbolified-edge names, status.
Combined: `_workspace/robustness_symbolify_summary.csv` — one row per function
(the stability summary + status) for the downstream aggregation agent.
Figures saved png(dpi=300)/svg/eps, mirroring `toy_KAN_analyze.py`.

## CLI
`python github/workflows/Hyein/robustness_symbolify.py [func_name]`
(no positional ⇒ full set of 7). Flags: `--rel-thresh` (0.1), `--refit` (off),
`--match-frac` (0.05).

## Verified run (full set, no refit)
All 7 functions ran `status=ok`. Both directions exercised:
introduce-symbolify {conditional(4 edges), damping_sin(4), ishigami(9)},
invert-symbolify {log2(4), exponential(2), logarithm(2), rosenbrock(4)}.
Symbolification visibly reshapes s_i and adds/drops transitions (e.g. logarithm:
6 spline transitions vs 1 symbolic; rosenbrock: 3 spline vs 0 symbolic) — the
intended robustness signal. Near-edge spline wiggle produces spurious τ-crossings
that the symbolic reading removes.
