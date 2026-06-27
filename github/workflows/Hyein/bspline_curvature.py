"""Analytical curvature (2nd derivative) of KAN edge activations.

This module computes the **analytical second derivative** of a KAN edge's
learned activation function and uses its sign changes to locate **inflection
points** -- the transition points used by the "New KAN analysis".

Background
----------
The existing pipeline (``toy_KAN_analyze.py`` section 3) detects inflection
points from *finite differences of the spline control coefficients*
(``slope_2nd`` + ``find_indices_sign_revert``). That is a noisy coefficient-space
proxy. Here we instead differentiate the learned function analytically.

A KAN edge ``(input i -> output j)`` of layer ``l`` evaluates

    phi_{ij}(x) = mask[i,j] * ( scale_base[i,j] * base_fun(x)
                                + scale_sp[i,j]  * spline(x; grid[i], coef[i,j]) )

(see ``KANLayer.forward``). Its 2nd derivative is therefore

    phi''_{ij}(x) = mask[i,j] * ( scale_base[i,j] * base_fun''(x)
                                  + scale_sp[i,j]  * spline''(x) )

* The **spline part** is differentiated with the analytical B-spline derivative
  recurrence (de Boor): the derivative of an order-``k`` spline is an order-``k-1``
  spline whose coefficients are ``d_a = k (c_a - c_{a-1}) / (t_{a+k} - t_a)``.
  Applying it twice gives an order-``k-2`` spline (so ``k >= 2`` is required;
  the pykan default ``k=3`` works). We reuse ``kan.spline.B_batch`` to evaluate
  the resulting lower-order basis, exactly mirroring ``coef2curve``.
* The **base part** is ``SiLU`` by default, whose 2nd derivative has a closed
  form; a generic ``base_fun`` falls back to autograd.

Everything operates in the model's **normalized** input space (the space the
spline grid lives in). Callers that compare against raw-space quantities must
denormalize the returned inflection x-values via ``scaler_X`` -- this module
does no space conversion, matching the convention of ``sectional_gsa.py``.

Public API
----------
``eval_spline_second_derivative(grid, coef, k, x_eval)``
    Analytical spline 2nd derivative for a whole layer, shape (batch,in,out).
``activation_second_derivative(model, l, x_eval)``
    Full edge-activation 2nd derivative (spline + base), shape (batch,in,out).
``edge_second_derivative(model, l, i, j, x_sweep)``
    Convenience 1-D sweep of edge (i->j)'s activation 2nd derivative.
``find_inflection_points(model, l, i, ...)``
    Inflection x-values (normalized space) for input ``i`` of layer ``l``.
``autograd_edge_second_derivative(model, l, i, j, x_sweep)``
    Reference autograd 2nd derivative (for verification only).
``verify_against_autograd(model, l, ...)``
    Convenience checker returning the max abs error vs autograd.
"""

import numpy as np
import torch

from kan.spline import B_batch, coef2curve
from kan.experiments.analysis import find_indices_sign_revert

_EPS = 1e-12


# ----------------------------------------------------------------------------
# B-spline derivative recurrence
# ----------------------------------------------------------------------------
def _deriv_coef(grid, coef, order):
    """One step of the B-spline derivative recurrence (order -> order-1).

    For a spline ``S(x) = sum_a coef_a B_{a,order}(x)`` over knot vector ``grid``,
    the derivative is ``S'(x) = sum_a d_a B_{a,order-1}(x)`` with

        d_a = order * (coef_a - coef_{a-1}) / (t_{a+order} - t_a),

    using the boundary convention ``coef_{-1} = coef_{nc} = 0``. The returned
    coefficient tensor has one more entry along the last axis than ``coef``,
    matching the basis count of ``B_batch(x, grid, order-1)``.

    Parameters
    ----------
    grid : torch.Tensor, shape (in_dim, M)
        Extended knot vector per input (pykan layout: ``M = G + 2k + 1``).
    coef : torch.Tensor, shape (in_dim, out_dim, nc)
        Spline coefficients of order ``order`` (``nc = M - order - 1``).
    order : int
        Current spline order of ``coef``.

    Returns
    -------
    torch.Tensor, shape (in_dim, out_dim, nc + 1)
        Coefficients of the order-``(order-1)`` derivative spline.
    """
    in_dim, out_dim, nc = coef.shape
    z = torch.zeros(in_dim, out_dim, 1, dtype=coef.dtype, device=coef.device)
    cpad = torch.cat([z, coef, z], dim=-1)            # (in,out,nc+2): c_{-1}..c_{nc}
    diff = cpad[..., 1:] - cpad[..., :-1]             # (in,out,nc+1): c_a - c_{a-1}

    a = torch.arange(nc + 1, device=grid.device)
    denom = (grid[:, a + order] - grid[:, a]).unsqueeze(1)  # (in,1,nc+1): t_{a+order}-t_a
    out = order * diff / denom
    # Degenerate knot spacing (repeated knots) -> that basis function is null.
    out = torch.where(denom.abs() > _EPS, out, torch.zeros_like(out))
    return out


def spline_second_derivative_coef(grid, coef, k):
    """Coefficients of the analytical 2nd derivative spline (order ``k-2``).

    Returns ``(d2, k2)`` where ``d2`` has shape ``(in_dim, out_dim, M-k+1)`` and
    ``k2 = k - 2``. Evaluate with ``B_batch(x, grid, k2)``.
    """
    if k < 2:
        raise ValueError("Second derivative requires spline order k >= 2; got %d." % k)
    d1 = _deriv_coef(grid, coef, k)        # order k-1
    d2 = _deriv_coef(grid, d1, k - 1)      # order k-2
    return d2, k - 2


def eval_spline_second_derivative(grid, coef, k, x_eval):
    """Analytical 2nd derivative of the spline part, for a whole layer.

    Mirrors ``kan.spline.coef2curve`` but with the order-``k-2`` derivative
    coefficients and basis.

    Parameters
    ----------
    grid : torch.Tensor, shape (in_dim, M)
    coef : torch.Tensor, shape (in_dim, out_dim, M-k-1)
    k : int
        Spline order.
    x_eval : torch.Tensor, shape (batch, in_dim)

    Returns
    -------
    torch.Tensor, shape (batch, in_dim, out_dim)
    """
    d2, k2 = spline_second_derivative_coef(grid, coef, k)
    b = B_batch(x_eval, grid, k=k2)                       # (batch, in, M-k+1)
    return torch.einsum('ijk,jlk->ijl', b, d2.to(b.device))


# ----------------------------------------------------------------------------
# Base function 2nd derivative
# ----------------------------------------------------------------------------
def _base_second_derivative(base_fun, x):
    """Second derivative of the residual base function, evaluated at ``x``.

    Closed form for ``SiLU`` (the pykan default); autograd fallback otherwise.

    SiLU(x) = x * sigmoid(x);  with s = sigmoid(x):
        SiLU''(x) = s (1 - s) [ 2 + x (1 - 2 s) ].
    """
    if isinstance(base_fun, torch.nn.SiLU):
        s = torch.sigmoid(x)
        return s * (1.0 - s) * (2.0 + x * (1.0 - 2.0 * s))

    # Generic elementwise base: differentiate twice with autograd (exact).
    x_ = x.detach().clone().requires_grad_(True)
    y = base_fun(x_)
    g1, = torch.autograd.grad(y.sum(), x_, create_graph=True)
    g2, = torch.autograd.grad(g1.sum(), x_)
    return g2.detach()


# ----------------------------------------------------------------------------
# Full activation 2nd derivative
# ----------------------------------------------------------------------------
def activation_second_derivative(model, l, x_eval):
    """Analytical 2nd derivative of the full edge activations of layer ``l``.

    ``phi''_{ij}(x) = mask[i,j] (scale_base[i,j] base''(x_i)
                                 + scale_sp[i,j] spline''_{ij}(x_i))``.

    Because each edge depends only on its own input column, ``x_eval`` may set
    the column of interest to a sweep and leave the others arbitrary.

    Parameters
    ----------
    model : MultKAN
    l : int
        Layer index.
    x_eval : torch.Tensor, shape (batch, in_dim)

    Returns
    -------
    torch.Tensor, shape (batch, in_dim, out_dim)
    """
    act = model.act_fun[l]
    x_eval = x_eval.to(act.coef.device)

    sp2 = eval_spline_second_derivative(act.grid, act.coef, act.k, x_eval)  # (b,in,out)
    base2 = _base_second_derivative(act.base_fun, x_eval)                   # (b,in)

    y = (act.scale_base[None, :, :] * base2[:, :, None]
         + act.scale_sp[None, :, :] * sp2)
    y = act.mask[None, :, :] * y
    return y


def edge_second_derivative(model, l, i, j, x_sweep):
    """1-D sweep of edge ``(i -> j)``'s activation 2nd derivative (normalized x).

    Parameters
    ----------
    model : MultKAN
    l, i, j : int
        Layer, input, output indices.
    x_sweep : array-like or torch.Tensor, shape (batch,)
        Normalized-space input values for column ``i``.

    Returns
    -------
    np.ndarray, shape (batch,)
    """
    act = model.act_fun[l]
    in_dim = act.coef.shape[0]
    device = act.coef.device
    xs = torch.as_tensor(np.asarray(x_sweep, dtype=float), dtype=act.coef.dtype,
                         device=device)
    x_eval = torch.zeros(xs.shape[0], in_dim, dtype=act.coef.dtype, device=device)
    x_eval[:, i] = xs
    with torch.no_grad():
        d2 = activation_second_derivative(model, l, x_eval)[:, i, j]
    return d2.detach().cpu().numpy()


# ----------------------------------------------------------------------------
# Activation value and first derivative (for visualization)
# ----------------------------------------------------------------------------
def _base_first_derivative(base_fun, x):
    """First derivative of the residual base function at ``x``.

    Closed form for ``SiLU``; autograd fallback otherwise.
    SiLU'(x) = sigmoid(x) [ 1 + x (1 - sigmoid(x)) ].
    """
    if isinstance(base_fun, torch.nn.SiLU):
        s = torch.sigmoid(x)
        return s * (1.0 + x * (1.0 - s))

    x_ = x.detach().clone().requires_grad_(True)
    y = base_fun(x_)
    g1, = torch.autograd.grad(y.sum(), x_)
    return g1.detach()


def eval_spline_first_derivative(grid, coef, k, x_eval):
    """Analytical 1st derivative of the spline part, for a whole layer.

    Mirrors ``coef2curve`` with the order-``k-1`` derivative coefficients/basis.
    Returns shape (batch, in_dim, out_dim).
    """
    if k < 1:
        raise ValueError("First derivative requires spline order k >= 1; got %d." % k)
    d1 = _deriv_coef(grid, coef, k)                       # order k-1
    b = B_batch(x_eval, grid, k=k - 1)                    # (batch, in, M-k)
    return torch.einsum('ijk,jlk->ijl', b, d1.to(b.device))


def activation_value(model, l, x_eval):
    """Full edge-activation value ``phi_{ij}(x)`` of layer ``l``.

    ``phi_{ij}(x) = mask[i,j] (scale_base[i,j] base(x_i) + scale_sp[i,j] spline_{ij}(x_i))``.
    Returns shape (batch, in_dim, out_dim).
    """
    act = model.act_fun[l]
    x_eval = x_eval.to(act.coef.device)
    spl = coef2curve(x_eval, act.grid, act.coef, act.k)              # (b,in,out)
    base = act.base_fun(x_eval)                                      # (b,in)
    y = act.scale_base[None, :, :] * base[:, :, None] + act.scale_sp[None, :, :] * spl
    return act.mask[None, :, :] * y


def activation_first_derivative(model, l, x_eval):
    """Analytical 1st derivative of the full edge activations of layer ``l``.

    ``phi'_{ij}(x) = mask[i,j] (scale_base[i,j] base'(x_i) + scale_sp[i,j] spline'_{ij}(x_i))``.
    Returns shape (batch, in_dim, out_dim).
    """
    act = model.act_fun[l]
    x_eval = x_eval.to(act.coef.device)
    sp1 = eval_spline_first_derivative(act.grid, act.coef, act.k, x_eval)  # (b,in,out)
    base1 = _base_first_derivative(act.base_fun, x_eval)                   # (b,in)
    y = act.scale_base[None, :, :] * base1[:, :, None] + act.scale_sp[None, :, :] * sp1
    return act.mask[None, :, :] * y


def edge_curves(model, l, i, j, x_sweep):
    """Activation and its analytical derivatives on a 1-D normalized sweep.

    Returns ``(phi, dphi, d2phi)`` as numpy arrays of shape (batch,) for edge
    ``(i -> j)`` of layer ``l``, evaluated at normalized inputs ``x_sweep``.
    Convenient for plotting the function alongside ``phi'`` and ``phi''``.
    """
    act = model.act_fun[l]
    in_dim = act.coef.shape[0]
    device = act.coef.device
    xs = torch.as_tensor(np.asarray(x_sweep, dtype=float), dtype=act.coef.dtype,
                         device=device)
    x_eval = torch.zeros(xs.shape[0], in_dim, dtype=act.coef.dtype, device=device)
    x_eval[:, i] = xs
    with torch.no_grad():
        v = activation_value(model, l, x_eval)[:, i, j]
        d1 = activation_first_derivative(model, l, x_eval)[:, i, j]
        d2 = activation_second_derivative(model, l, x_eval)[:, i, j]
    return (v.detach().cpu().numpy(),
            d1.detach().cpu().numpy(),
            d2.detach().cpu().numpy())


# ----------------------------------------------------------------------------
# Inflection point detection
# ----------------------------------------------------------------------------
def find_inflection_points(model, l, i, x_grid=None, n_eval=400,
                           j_list=None, eps=None, rel_eps=1e-2):
    """Inflection x-values for input ``i`` of layer ``l`` (normalized space).

    Evaluates the analytical activation 2nd derivative on a dense sweep over
    input ``i``'s grid range and detects persistent sign changes (curvature
    reversals) with ``find_indices_sign_revert`` -- the same detector used by
    the existing pipeline, so detection semantics match. Inflections are
    collected over all (or selected) output edges ``j`` and de-duplicated.

    Parameters
    ----------
    model : MultKAN
    l, i : int
        Layer and input index.
    x_grid : np.ndarray, optional
        Normalized-space sweep. Defaults to a dense linspace over the interior
        knot range ``act.grid[i, k-1:-2]``.
    n_eval : int
        Number of sweep points when ``x_grid`` is None.
    j_list : sequence of int, optional
        Output edges to scan. Defaults to all outputs.
    eps : float, optional
        Absolute zero-tolerance for ``find_indices_sign_revert``. If None, an
        adaptive value ``rel_eps * max|phi''|`` is used per edge (the raw 2nd
        derivative scale varies widely between functions, so an absolute
        default would not generalize).
    rel_eps : float
        Fraction of the per-edge peak magnitude used when ``eps`` is None.

    Returns
    -------
    list of float
        Sorted, de-duplicated inflection x-values in **normalized** space.
    """
    act = model.act_fun[l]
    k = act.k
    if x_grid is None:
        knots = act.grid[i, k - 1:-2].detach().cpu().numpy()
        lo, hi = float(np.min(knots)), float(np.max(knots))
        x_grid = np.linspace(lo, hi, n_eval)
    x_grid = np.asarray(x_grid, dtype=float)

    no = act.coef.shape[1]
    js = range(no) if j_list is None else j_list

    inflections = []
    for j in js:
        if float(act.mask[i, j].detach().cpu()) == 0.0:
            continue  # pruned edge contributes no activation
        d2 = edge_second_derivative(model, l, i, j, x_grid)
        if not np.any(np.isfinite(d2)):
            continue
        peak = float(np.nanmax(np.abs(d2)))
        ep = eps if eps is not None else max(1e-9, rel_eps * peak)
        idx_revert = find_indices_sign_revert(list(d2), epsilon=ep)
        for ir in idx_revert:
            inflections.append(float(x_grid[ir]))

    return sorted(set(inflections))


# ----------------------------------------------------------------------------
# Verification helpers (used by QA; not needed in the analysis path)
# ----------------------------------------------------------------------------
def autograd_edge_second_derivative(model, l, i, j, x_sweep):
    """Reference 2nd derivative of edge ``(i -> j)`` via autograd double-grad.

    Differentiates the *full* activation (``scale_base*base + scale_sp*spline``,
    spline via ``coef2curve``) twice. Used only to validate the analytical path.
    """
    act = model.act_fun[l]
    in_dim = act.coef.shape[0]
    device = act.coef.device
    xs = torch.as_tensor(np.asarray(x_sweep, dtype=float), dtype=act.coef.dtype,
                         device=device)
    x = torch.zeros(xs.shape[0], in_dim, dtype=act.coef.dtype, device=device)
    x[:, i] = xs
    x = x.detach().requires_grad_(True)

    base = act.base_fun(x)
    yspl = coef2curve(x, act.grid, act.coef, act.k)
    y = (act.scale_base[None, :, :] * base[:, :, None]
         + act.scale_sp[None, :, :] * yspl)
    y = act.mask[None, :, :] * y
    yij = y[:, i, j]

    g1, = torch.autograd.grad(yij.sum(), x, create_graph=True)
    g2, = torch.autograd.grad(g1[:, i].sum(), x)
    return g2[:, i].detach().cpu().numpy()


def verify_against_autograd(model, l, edges=None, n_eval=200, margin=0.05):
    """Max abs error of the analytical 2nd derivative vs autograd, per edge.

    Sweeps each edge over the interior of its grid range (trimmed by ``margin``
    on each side to avoid exact-knot kinks) and compares
    ``activation_second_derivative`` to ``autograd_edge_second_derivative``.

    Returns
    -------
    dict {(i, j): {'max_abs_err': float, 'scale': float, 'rel_err': float}}
    """
    act = model.act_fun[l]
    in_dim, out_dim = act.coef.shape[:2]
    k = act.k
    if edges is None:
        edges = [(i, j) for i in range(in_dim) for j in range(out_dim)]

    report = {}
    for (i, j) in edges:
        knots = act.grid[i, k - 1:-2].detach().cpu().numpy()
        lo, hi = float(np.min(knots)), float(np.max(knots))
        span = hi - lo
        xs = np.linspace(lo + margin * span, hi - margin * span, n_eval)
        ana = edge_second_derivative(model, l, i, j, xs)
        ref = autograd_edge_second_derivative(model, l, i, j, xs)
        finite = np.isfinite(ana) & np.isfinite(ref)
        if not np.any(finite):
            report[(i, j)] = {'max_abs_err': np.nan, 'scale': np.nan, 'rel_err': np.nan}
            continue
        err = float(np.max(np.abs(ana[finite] - ref[finite])))
        scale = float(np.max(np.abs(ref[finite]))) + 1e-12
        report[(i, j)] = {'max_abs_err': err, 'scale': scale, 'rel_err': err / scale}
    return report
