"""MLP (sklearn ``MLPRegressor``) gradient-based local-sensitivity baseline.

This is the MLP analog of the KAN bottom-layer local sensitivity
``s_i(x) = sum_j |phi'_{ij}(x)|`` (analytic, univariate, edge-separated). The MLP
has NO autograd, so ``|df/dx_i|`` is computed by **central finite differences** on
``mlp.predict``. Unlike KAN edges, ``df/dx_i`` depends on the other inputs (the
network entangles them), so the 1-D sensitivity trajectory must fix/marginalize
the other inputs — we expose both a clean ``midpoint`` mode (others = 0.5) and an
``mc`` mode (others ~ U(0.1,0.9), returning mean ± std band).

Space convention (mirrors the KAN side): sensitivity is computed in the
NORMALIZED [0.1, 0.9] space the MLP was trained on (``mlp.predict(X_norm)``);
raw space is used only for plotting. The separate raw-space ``mlp_batch_func`` is
provided solely to feed ``sectional_gsa.compute_gradient_agsm`` (which works in
raw space).

Reuses (does not duplicate): the τ ranking-transition logic from
``bspline_curvature.find_ranking_transitions`` and ``find_indices_sign_revert``.
"""

import numpy as np

from kan.experiments.analysis import find_indices_sign_revert


# ----------------------------------------------------------------------------
# Predict wrappers
# ----------------------------------------------------------------------------
def mlp_predict_norm(mlp):
    """Return ``f(X_norm[m,n]) -> y_norm[m]`` for a trained ``MLPRegressor``.

    The MLP was trained on normalized inputs/labels, so ``mlp.predict`` already
    maps normalized inputs to normalized outputs; we only flatten the result.
    """
    def predict_norm(X_norm):
        X_norm = np.atleast_2d(np.asarray(X_norm, dtype=float))
        return np.asarray(mlp.predict(X_norm)).ravel()
    return predict_norm


def mlp_batch_func(mlp, scaler_X, scaler_y):
    """Return a raw-space surrogate ``f(X_raw[m,n]) -> y_raw[m]``.

    Mirrors ``kan_analysis_core.kan_batch_func``: normalize inputs, predict
    (normalized output), then inverse-transform the labels back to raw space.
    Used to feed ``sectional_gsa.compute_gradient_agsm`` (raw-space MC integration).
    """
    def batch(X_raw):
        X_raw = np.atleast_2d(np.asarray(X_raw, dtype=float))
        X_norm = scaler_X.transform(X_raw)
        y_norm = np.asarray(mlp.predict(X_norm)).reshape(-1, 1)
        try:
            y_raw = scaler_y.inverse_transform(y_norm)
        except Exception:
            y_raw = y_norm
        return np.asarray(y_raw, dtype=float).ravel()
    return batch


# ----------------------------------------------------------------------------
# Finite-difference local sensitivity |df/dx_i| over a 1-D normalized sweep
# ----------------------------------------------------------------------------
# Default step mirrors sectional_gsa's normalized-space finite difference:
# fd_eps (1e-5) scaled by the normalized domain width (0.9 - 0.1 = 0.8).
_DEFAULT_EPS = 1e-5 * 0.8


def mlp_feature_sensitivity(predict_norm, x_grid, feat_idx, nx, mode,
                            *, x_fixed=0.5, n_mc=64, seed=42, eps=None):
    """Finite-difference ``|df/dx_{feat_idx}|`` along the 1-D normalized ``x_grid``.

    Central difference on ``predict_norm``:
        ``|(f(X+eps) - f(X-eps)) / (2*eps)|``
    with the feat_idx column swept over ``x_grid`` and the OTHER columns either
    fixed at ``x_fixed`` (midpoint) or randomly drawn (mc).

    Parameters
    ----------
    predict_norm : callable
        ``f(X_norm[m,n]) -> y_norm[m]`` (see ``mlp_predict_norm``).
    x_grid : array
        Normalized sweep values for ``feat_idx``.
    feat_idx : int
        Input index being differentiated/swept.
    nx : int
        Number of inputs.
    mode : {'midpoint', 'mc'}
        - 'midpoint' : other inputs fixed at ``x_fixed``; returns ``s`` (array).
        - 'mc'       : other inputs ~ U(0.1, 0.9) over ``n_mc`` seeded draws;
                       returns ``(mean, std)`` arrays over the draws.
    x_fixed : float
        Value for the other inputs in 'midpoint' mode (normalized, default 0.5).
    n_mc : int
        Number of Monte-Carlo draws in 'mc' mode.
    seed : int
        RNG seed for the MC draws.
    eps : float, optional
        Central-difference step (default ``1e-5 * 0.8`` in normalized space).

    Returns
    -------
    midpoint -> ``s`` (array, len(x_grid))
    mc       -> ``(mean, std)`` (arrays, len(x_grid))
    """
    x_grid = np.asarray(x_grid, dtype=float)
    step = float(_DEFAULT_EPS if eps is None else eps)
    m = x_grid.size

    def _grad_at(other_cols):
        """|df/dx_feat_idx| over x_grid, with all other columns = other_cols (len nx)."""
        X = np.tile(np.asarray(other_cols, dtype=float), (m, 1))  # (m, nx)
        X[:, feat_idx] = x_grid
        Xp = X.copy(); Xp[:, feat_idx] += step
        Xm = X.copy(); Xm[:, feat_idx] -= step
        g = (predict_norm(Xp) - predict_norm(Xm)) / (2.0 * step)
        return np.abs(g)

    if mode == 'midpoint':
        other = np.full(nx, float(x_fixed))
        return _grad_at(other)

    if mode == 'mc':
        rng = np.random.RandomState(seed)
        draws = np.empty((n_mc, m), dtype=float)
        for d in range(n_mc):
            # Draw the OTHER inputs ~ U(0.1, 0.9); feat_idx column is overwritten
            # by x_grid inside _grad_at, so its drawn value is irrelevant.
            other = rng.uniform(0.1, 0.9, size=nx)
            draws[d] = _grad_at(other)
        return draws.mean(axis=0), draws.std(axis=0)

    raise ValueError(f"unknown mode {mode!r} (expected 'midpoint' or 'mc')")


# ----------------------------------------------------------------------------
# Ranking transitions (τ-crossings) — τ logic copied from
# bspline_curvature.find_ranking_transitions, S built from MLP sensitivities.
# ----------------------------------------------------------------------------
def mlp_ranking_transitions(predict_norm, x_grid, nx, rel_thresh=0.2, mode='mc'):
    """MLP ranking transitions where a feature's |df/dx_i| drops below τ.

    Same algorithm as ``bspline_curvature.find_ranking_transitions``:
    ``tau = rel_thresh * max_i max_x s_i``; the per-position dominant feature is
    ``argmax_i s_i``; each ``s_i - tau`` sign-revert index is a transition (down if
    falling, up if rising), flagged ``dominant`` if that feature was dominant just
    before and is going down. For ``mode='mc'`` the MC **mean** is used as ``S[i]``.

    Returns ``(transition_points_per_input, transitions, info)`` where
    ``transition_points_per_input[i]`` is the sorted points for input ``i`` and
    ``info = {'x_grid', 'S', 'S_std', 'tau'}`` (``S_std`` is None unless mc).
    """
    x_grid = np.asarray(x_grid, dtype=float)

    S = {}
    S_std = {} if mode == 'mc' else None
    for i in range(nx):
        out = mlp_feature_sensitivity(predict_norm, x_grid, i, nx, mode=mode)
        if mode == 'mc':
            S[i], S_std[i] = out
        else:
            S[i] = out

    stack = np.vstack([S[i] for i in range(nx)])              # (nx, T)
    scale = float(np.nanmax(stack)) if np.any(np.isfinite(stack)) else 0.0
    tau = rel_thresh * scale
    dominant = np.argmax(stack, axis=0)                       # (T,)

    transitions = []
    for i in range(nx):
        diff = S[i] - tau
        for idx in find_indices_sign_revert(list(diff), epsilon=0.0):
            direction = 'down' if diff[idx] < diff[idx - 1] else 'up'
            transitions.append({
                'point': float(x_grid[idx]),
                'feat_idx': i,
                'direction': direction,
                'dominant': bool(dominant[idx - 1] == i and direction == 'down'),
            })
    transitions.sort(key=lambda t: t['point'])

    transition_points_per_input = [
        sorted(t['point'] for t in transitions if t['feat_idx'] == i)
        for i in range(nx)
    ]
    info = {'x_grid': x_grid, 'S': S, 'S_std': S_std, 'tau': tau}
    return transition_points_per_input, transitions, info
