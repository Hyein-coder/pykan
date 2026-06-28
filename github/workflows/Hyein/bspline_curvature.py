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
# Full-model input derivatives (chain rule across layers)
# ----------------------------------------------------------------------------
# These compose the per-edge analytic derivatives across all layers (summation
# nodes) to get the derivative of the whole KAN output f w.r.t. each input x_i.
# Forward recipe (MultKAN.forward): a^{(l+1)}_j = s^{(l)}_j * sum_i phi_{ij}(acts[l][:,i]) + const,
# with combined diagonal scale s^{(l)}_j = node_scale[l][j] * subnode_scale[l][j].

def _as_model_input(model, x):
    """Coerce x (numpy or tensor) to a float tensor on the model's device."""
    device = model.act_fun[0].coef.device
    dtype = model.act_fun[0].coef.dtype
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x, dtype=float), device=device, dtype=dtype)


def _combined_node_scale(model, l):
    """Combined diagonal node scale s^{(l)} = node_scale[l] * subnode_scale[l].

    Raises for layers with multiplication nodes (this composition assumes pure
    summation nodes, matching the current models).
    """
    w_next = model.width[l + 1]
    n_mult = w_next[1] if isinstance(w_next, (list, tuple)) else 0
    if n_mult and n_mult > 0:
        raise ValueError(
            "model derivatives support summation-only models; layer %d has %d mult node(s)."
            % (l + 1, n_mult))
    ns = model.node_scale[l].detach()
    ss = model.subnode_scale[l].detach()
    return (ns * ss)


def _layer_acts(model, x_eval):
    """Forward x_eval and return the cached per-layer node activations acts[l]."""
    x_t = _as_model_input(model, x_eval)
    with torch.no_grad():
        model(x_t)              # populates model.acts (acts[l] = input to layer l)
    return [a.detach() for a in model.acts]


def  model_directional_derivatives(model, x_eval, feat_idx):
    """Full-model value f, ∂f/∂x_i and ∂²f/∂x_i² at points ``x_eval``.

    Differentiates the whole KAN output w.r.t. input column ``feat_idx`` via
    forward-mode 2nd-order AD along e_i, composing the symbolic-aware per-edge
    analytic derivatives across layers. Scalar output assumed.

    Returns (f, df, d2f) as numpy arrays of shape (batch,).
    """
    if int(model.width_in[-1]) != 1:
        raise ValueError("model_directional_derivatives expects a scalar output "
                         "(width_in[-1]==1); got %s." % (model.width_in[-1],))
    acts = _layer_acts(model, x_eval)
    L = len(model.act_fun)
    device = model.act_fun[0].coef.device
    dtype = model.act_fun[0].coef.dtype
    batch, n0 = acts[0].shape

    d = torch.zeros(batch, n0, device=device, dtype=dtype)
    d[:, feat_idx] = 1.0
    h = torch.zeros(batch, n0, device=device, dtype=dtype)

    with torch.no_grad():
        for l in range(L):
            phi1 = activation_first_derivative(model, l, acts[l])   # (b, n_l, n_{l+1})
            phi2 = activation_second_derivative(model, l, acts[l])  # (b, n_l, n_{l+1})
            s = _combined_node_scale(model, l).to(device=device, dtype=dtype)  # (n_{l+1},)
            d_new = s[None, :] * torch.einsum('bp,bpj->bj', d, phi1)
            h_new = s[None, :] * (torch.einsum('bp,bpj->bj', d * d, phi2)
                                  + torch.einsum('bp,bpj->bj', h, phi1))
            d, h = d_new, h_new

    f = acts[L][:, 0]
    out = []
    for t in (f, d[:, 0], h[:, 0]):
        a = t.detach().cpu().numpy()
        out.append(np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0))
    return out[0], out[1], out[2]


def model_gradient(model, x_eval):
    """Full-model gradient ∂f/∂x for all input features at ``x_eval``.

    Jacobian chain ``D[l+1] = J[l] @ D[l]`` with ``J[l][:,j,i] = s^{(l)}_j φ'_{ij}``.
    Returns numpy array shape (batch, n_inputs). Scalar output assumed.
    """
    if int(model.width_in[-1]) != 1:
        raise ValueError("model_gradient expects a scalar output.")
    acts = _layer_acts(model, x_eval)
    L = len(model.act_fun)
    device = model.act_fun[0].coef.device
    dtype = model.act_fun[0].coef.dtype
    batch, n0 = acts[0].shape

    D = torch.eye(n0, device=device, dtype=dtype)[None].expand(batch, n0, n0).clone()
    with torch.no_grad():
        for l in range(L):
            phi1 = activation_first_derivative(model, l, acts[l])      # (b, n_l, n_{l+1})
            s = _combined_node_scale(model, l).to(device=device, dtype=dtype)
            J = s[None, None, :] * phi1                                # (b, n_l, n_{l+1})
            D = torch.einsum('bij,bik->bjk', J, D)                     # (b, n_{l+1}, n0)
    grad = D[:, 0, :].detach().cpu().numpy()
    return np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)


def model_gradient_autograd(model, x_eval):
    """Reference ∂f/∂x for all features via autograd. Shape (batch, n_inputs)."""
    x = _as_model_input(model, x_eval).clone().requires_grad_(True)
    y = model(x)
    g, = torch.autograd.grad(y.sum(), x, create_graph=False)
    return g.detach().cpu().numpy()


def model_directional_autograd(model, x_eval, feat_idx):
    """Reference (∂f/∂x_i, ∂²f/∂x_i²) via autograd double-grad. Shapes (batch,)."""
    x = _as_model_input(model, x_eval).clone().requires_grad_(True)
    y = model(x)
    g, = torch.autograd.grad(y.sum(), x, create_graph=True)
    gi = g[:, feat_idx]
    h, = torch.autograd.grad(gi.sum(), x)
    return (g[:, feat_idx].detach().cpu().numpy(),
            h[:, feat_idx].detach().cpu().numpy())


def find_model_inflection_points(model, feat_idx, x_grid=None, x_fixed=None,
                                 n_eval=400, eps=None, rel_eps=1e-2):
    """Full-model inflection x-values for input ``feat_idx`` (normalized space).

    Sweeps ``feat_idx`` over its grid interior with the other inputs held at
    ``x_fixed`` (default: per-feature grid-interior midpoint ~ band mean),
    computes ∂²f/∂x_i² via ``model_directional_derivatives``, and detects sign
    changes with ``find_indices_sign_revert``. Returns sorted normalized x-values.
    """
    act0 = model.act_fun[0]
    k = act0.k
    n0 = act0.coef.shape[0]

    def _interior(p):
        return act0.grid[p, k - 1:-2].detach().cpu().numpy()

    if x_grid is None:
        kn = _interior(feat_idx)
        x_grid = np.linspace(float(kn.min()), float(kn.max()), n_eval)
    x_grid = np.asarray(x_grid, dtype=float)

    if x_fixed is None:
        x_fixed = np.array([0.5 * (float(_interior(p).min()) + float(_interior(p).max()))
                            for p in range(n0)], dtype=float)
    x_fixed = np.asarray(x_fixed, dtype=float).ravel()

    X = np.tile(x_fixed, (len(x_grid), 1))
    X[:, feat_idx] = x_grid
    _, _, d2 = model_directional_derivatives(model, X, feat_idx)
    if not np.any(np.isfinite(d2)):
        return []
    peak = float(np.nanmax(np.abs(d2)))
    ep = eps if eps is not None else max(1e-9, rel_eps * peak)
    idx_revert = find_indices_sign_revert(list(d2), epsilon=ep)
    return sorted(set(float(x_grid[i]) for i in idx_revert))


def verify_model_derivatives(model, x_eval, feat_indices=None):
    """Max abs error of analytic full-model derivatives vs autograd.

    Returns {'grad_max_abs_err': float, 'curv': {i: {'df_max_abs_err','d2f_max_abs_err'}}}.
    """
    g_ana = model_gradient(model, x_eval)
    g_ref = model_gradient_autograd(model, x_eval)
    rep = {'grad_max_abs_err': float(np.max(np.abs(g_ana - g_ref))), 'curv': {}}
    feats = range(g_ana.shape[1]) if feat_indices is None else feat_indices
    for i in feats:
        _, df_i, d2f_i = model_directional_derivatives(model, x_eval, i)
        df_ref, d2_ref = model_directional_autograd(model, x_eval, i)
        rep['curv'][int(i)] = {
            'df_max_abs_err': float(np.max(np.abs(df_i - df_ref))),
            'd2f_max_abs_err': float(np.max(np.abs(d2f_i - d2_ref))),
        }
    return rep
