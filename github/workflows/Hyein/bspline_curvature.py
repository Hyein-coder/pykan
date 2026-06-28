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
import sympy
import torch

from kan.spline import B_batch, coef2curve
from kan.experiments.analysis import find_indices_sign_revert
from kan.utils import SYMBOLIC_LIB

_EPS = 1e-12

# Cache of lambdified analytic derivatives, keyed by (id(sym_layer), i, j, order).
_SYM_DERIV_CACHE = {}


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
    return y + _symbolic_branch(model, l, x_eval, order=2)


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
# Symbolic branch (analytic, closed-form derivatives)
# ----------------------------------------------------------------------------
# When an edge is symbolified, pykan disables its spline branch (act_fun mask 0)
# and the learned function lives in model.symbolic_fun as g(x) = c*f(a*x+b)+d
# (Symbolic_KANLayer.forward). Its derivatives are closed form:
#     g'(x)  = c*a   * f'(a*x+b)
#     g''(x) = c*a^2 * f''(a*x+b)
# We obtain f', f'' analytically via sympy (the spline path is analytic too;
# autograd is reserved for verification). funs_sympy / SYMBOLIC_LIB index by
# [out=j][in=i] -- transposed relative to act_fun's [in, out].
def _symbolic_edge_deriv(sym_layer, i, j, order):
    """Numeric callable for the closed-form order-th derivative of edge (i->j).

    Differentiates ``c*f(a*x+b)+d`` symbolically with sympy and lambdifies it.
    Falls back to autograd on the torch fun only if the sympy path fails.
    """
    key = (id(sym_layer), i, j, order)
    cached = _SYM_DERIV_CACHE.get(key)
    if cached is not None:
        return cached

    a = float(sym_layer.affine[j, i, 0]); b = float(sym_layer.affine[j, i, 1])
    c = float(sym_layer.affine[j, i, 2]); d = float(sym_layer.affine[j, i, 3])
    name = sym_layer.funs_name[j][i]

    try:
        f_sympy = (SYMBOLIC_LIB[name][1] if name in SYMBOLIC_LIB
                   else sym_layer.funs_sympy[j][i])
        X = sympy.symbols('x')
        expr = c * f_sympy(a * X + b) + d
        if order > 0:
            expr = sympy.diff(expr, X, order)
        fn = sympy.lambdify(X, expr, 'numpy')

        def _evaluated(x_np, _fn=fn):
            out = np.asarray(_fn(x_np), dtype=float)
            if out.shape != np.shape(x_np):     # constant expr -> scalar; broadcast
                out = np.broadcast_to(out, np.shape(x_np)).copy()
            return out
    except Exception:
        # Last-resort: autograd on the torch fun (still exact, just not symbolic).
        torch_f = sym_layer.funs[j][i]

        def _evaluated(x_np, _f=torch_f, _a=a, _b=b, _c=c, _d=d, _order=order):
            xt = torch.as_tensor(np.asarray(x_np, dtype=float), dtype=torch.float64,
                                 requires_grad=True)
            y = _c * _f(_a * xt + _b) + _d
            if _order == 0:
                return y.detach().cpu().numpy()
            g1, = torch.autograd.grad(y.sum(), xt, create_graph=_order > 1)
            if _order == 2:
                g1, = torch.autograd.grad(g1.sum(), xt)
            return g1.detach().cpu().numpy()

    _SYM_DERIV_CACHE[key] = _evaluated
    return _evaluated


def _symbolic_branch(model, l, x_eval, order=0):
    """Symbolic branch contribution (value or derivative) of layer ``l``.

    Returns (batch, in_dim, out_dim); zeros when the model has no active symbolic
    layer. Only edges with ``symbolic_fun[l].mask[j,i] != 0`` contribute.
    """
    act = model.act_fun[l]
    in_dim, out_dim = act.coef.shape[:2]
    out = torch.zeros(x_eval.shape[0], in_dim, out_dim,
                      dtype=x_eval.dtype, device=x_eval.device)
    if not getattr(model, 'symbolic_enabled', False):
        return out
    sym_list = getattr(model, 'symbolic_fun', None)
    if sym_list is None or l >= len(sym_list):
        return out
    sym = sym_list[l]

    x_np_all = x_eval.detach().cpu().numpy()
    for i in range(in_dim):
        xi_np = x_np_all[:, i]
        for j in range(out_dim):
            if float(sym.mask[j, i].detach().cpu()) == 0.0:
                continue
            vals = _symbolic_edge_deriv(sym, i, j, order)(xi_np)
            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            out[:, i, j] = torch.as_tensor(vals, dtype=x_eval.dtype,
                                           device=x_eval.device)
    return out


def symbolic_edge_info(model, l):
    """Map ``{(i, j): funs_name}`` for active symbolic edges of layer ``l``."""
    info = {}
    if not getattr(model, 'symbolic_enabled', False):
        return info
    sym_list = getattr(model, 'symbolic_fun', None)
    if sym_list is None or l >= len(sym_list):
        return info
    sym = sym_list[l]
    in_dim, out_dim = model.act_fun[l].coef.shape[:2]
    for i in range(in_dim):
        for j in range(out_dim):
            if float(sym.mask[j, i].detach().cpu()) != 0.0:
                info[(i, j)] = sym.funs_name[j][i]
    return info


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
    y = act.mask[None, :, :] * y
    return y + _symbolic_branch(model, l, x_eval, order=0)


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
    y = act.mask[None, :, :] * y
    return y + _symbolic_branch(model, l, x_eval, order=1)


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
    sym_info = symbolic_edge_info(model, l)  # edges active via the symbolic branch

    inflections = []
    for j in js:
        # Skip only edges that are inactive in BOTH branches. Symbolified edges
        # have a zero spline mask but carry the function via symbolic_fun, so the
        # spline mask alone must not gate them out.
        if float(act.mask[i, j].detach().cpu()) == 0.0 and (i, j) not in sym_info:
            continue
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

    # Include the symbolic branch so the reference equals the FULL edge activation.
    if getattr(model, 'symbolic_enabled', False):
        sym_list = getattr(model, 'symbolic_fun', None)
        if sym_list is not None and l < len(sym_list):
            sym = sym_list[l]
            if float(sym.mask[j, i].detach().cpu()) != 0.0:
                a, b, c, d = (sym.affine[j, i, t] for t in range(4))
                yij = yij + (c * sym.funs[j][i](a * x[:, i] + b) + d)

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


# ----------------------------------------------------------------------------
# Ranking transition via small 1st-derivative (bottom-layer, feature-separable)
# ----------------------------------------------------------------------------
# Each layer-0 edge activation is univariate in its own input, so the per-feature
# local sensitivity s_i(x_i) = sum_j |phi'_{i,j}(x_i)| is exact and depends only
# on x_i. A "ranking transition" is where the (currently dominant) feature's
# sensitivity falls below a small threshold tau -> it becomes locally negligible.

def feature_sensitivity(model, i, x_grid, layer=0):
    """Bottom-layer local sensitivity ``s_i(x) = sum_j |phi'_{i,j}(x)|``.

    Depends only on input ``i`` (each edge is univariate). ``x_grid`` is in the
    model's normalized input space. Returns a numpy array of shape (len(x_grid),).
    """
    act = model.act_fun[layer]
    in_dim = act.coef.shape[0]
    device = act.coef.device
    xs = torch.as_tensor(np.asarray(x_grid, dtype=float), dtype=act.coef.dtype,
                         device=device)
    x_eval = torch.zeros(xs.shape[0], in_dim, dtype=act.coef.dtype, device=device)
    x_eval[:, i] = xs
    with torch.no_grad():
        phi1 = activation_first_derivative(model, layer, x_eval)[:, i, :]  # (batch, out)
        s = phi1.abs().sum(dim=1)                                          # sum over output edges
    s = s.detach().cpu().numpy()
    return np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)


def find_ranking_transitions(model, x_grid=None, rel_thresh=0.1, n_eval=400, layer=0):
    """Ranking transitions where a feature's local sensitivity drops below tau.

    For every input feature, ``s_i = feature_sensitivity`` over a shared sweep of
    the layer's grid interior. With ``tau = rel_thresh * max_i max_t s_i`` (a
    cross-feature-comparable "small"), find where each ``s_i`` crosses ``tau``
    (persistent sign change of ``s_i - tau``). The per-position dominant feature
    is ``argmax_i s_i``; its downward crossings are flagged ``dominant=True``.

    Returns ``(transitions, info)`` where transitions is a list of dicts
    ``{'point', 'feat_idx', 'direction' ('down'|'up'), 'dominant'}`` (normalized
    x), and info is ``{'x_grid', 'S' (dict feat->curve), 'tau'}``.
    """
    act = model.act_fun[layer]
    k = act.k
    in_dim = act.coef.shape[0]

    if x_grid is None:
        knots = act.grid[:, k - 1:-2].detach().cpu().numpy()
        lo, hi = float(np.min(knots)), float(np.max(knots))
        x_grid = np.linspace(lo, hi, n_eval)
    x_grid = np.asarray(x_grid, dtype=float)

    S = {i: feature_sensitivity(model, i, x_grid, layer=layer) for i in range(in_dim)}
    stack = np.vstack([S[i] for i in range(in_dim)])          # (in_dim, T)
    scale = float(np.nanmax(stack)) if np.any(np.isfinite(stack)) else 0.0
    tau = rel_thresh * scale
    dominant = np.argmax(stack, axis=0)                       # (T,)

    transitions = []
    for i in range(in_dim):
        diff = S[i] - tau
        for idx in find_indices_sign_revert(list(diff), epsilon=0.0):
            direction = 'down' if diff[idx] < diff[idx - 1] else 'up'
            transitions.append({
                'point': float(x_grid[idx]),
                'feat_idx': i,
                'direction': direction,
                # dominant just before this index and going negligible
                'dominant': bool(dominant[idx - 1] == i and direction == 'down'),
            })
    transitions.sort(key=lambda t: t['point'])
    return transitions, {'x_grid': x_grid, 'S': S, 'tau': tau}
