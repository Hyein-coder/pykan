"""Sectional Global Sensitivity Analysis (AGSM).

Standalone implementation of the Argument-value Sectional Global Sensitivity
Measure (AGSM) from:

    Pannier, S. & Graf, W. (2015). "Sectional global sensitivity measures."
    Reliability Engineering and System Safety, 134, 110-117.
    DOI: 10.1016/j.ress.2014.09.009

AGSM segments the domain of a single investigated input ``x_l`` into ``N``
contiguous sections while every other input spans its full range. The
derivative-based global sensitivity measure uses the *absolute* partial
derivative ``|df/dx_l|`` (Pannier Eq. 4); the absolute value is mandatory --
signed derivatives can integrate to zero even when an input clearly matters.

This module is self-contained: it operates purely in *raw* input space. The
caller is responsible for any space conversion (e.g. denormalizing KAN spline
knots from scaler_X space to raw space before passing them in as
``kan_knots_raw``).

Key measures (per investigated input ``l``, per section ``k``):

  S_hat[l][k]   : raw mean of |df/dx_l| over points whose x_l lands in section k
  S_hat_global  : weighted sum of S_hat (== full-domain mean |df/dx_l|)
  S_a[l][k]     : normalized, cross-input comparable AGSM (Eq. 14)
  R[l][k]       : within-input relative measure (sums to 1 over k, Eq. 13)

Dependencies: numpy, matplotlib (pandas/scipy optional, not required here).
torch is NOT used -- KAN gradients are handled by the caller via a batch func.
"""

import warnings

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# Plot style (project standard)
# ----------------------------------------------------------------------------
SA_RC = {
    'figure.figsize': (4, 3), 'figure.dpi': 150, 'figure.facecolor': 'white',
    'figure.autolayout': True, 'axes.facecolor': 'white', 'axes.edgecolor': '#444444',
    'axes.linewidth': 0.8, 'axes.spines.top': True, 'axes.spines.right': True,
    'axes.labelsize': 12, 'axes.labelcolor': 'black', 'axes.grid': False,
    'xtick.labelsize': 10, 'xtick.color': 'black', 'xtick.direction': 'out',
    'ytick.labelsize': 10, 'ytick.color': 'black', 'ytick.direction': 'out',
    'font.family': 'sans-serif', 'font.size': 10, 'font.weight': '300',
    'axes.labelweight': '500', 'text.color': 'black',
    'legend.fontsize': 8, 'legend.framealpha': 0.0,
    'lines.linewidth': 1.2, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
}

_EPS = 1e-12


# ----------------------------------------------------------------------------
# Function wrapping helper
# ----------------------------------------------------------------------------
def make_batch_func(single_func):
    """Wrap a per-row FUNCTION_ZOO callable into a batch callable.

    The ZOO funcs are written for a single sample indexed by position, e.g.
    ``lambda x: np.sin(2*x[0]) + 5*x[1]``, and some use a scalar Python ``if``
    (e.g. ``conditional``). They cannot be vectorized, so we batch them with
    ``np.apply_along_axis``.

    Parameters
    ----------
    single_func : callable
        Maps one row ``x`` (1-D array) to a scalar.

    Returns
    -------
    callable
        ``f(X)`` where ``X`` is ``[m, n]`` raw-space numpy, returning ``[m]``.
    """
    def f(X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.apply_along_axis(single_func, 1, X).astype(float).ravel()
    return f


# ----------------------------------------------------------------------------
# Section construction
# ----------------------------------------------------------------------------
def make_sections(lo, hi, n_sections, mode='equal', data_col=None,
                  kan_knots_raw=None, custom_edges=None):
    """Build section edges (in raw space) and their midpoints.

    Parameters
    ----------
    lo, hi : float
        Lower/upper bound of the investigated input's domain.
    n_sections : int
        Requested number of sections ``N``.
    mode : {'equal', 'quantile', 'kan', 'custom'}
        - 'equal'    : ``np.linspace(lo, hi, n_sections+1)``.
        - 'quantile' : data-driven quantile edges from ``data_col``.
        - 'kan'      : use ``kan_knots_raw`` (already denormalized by caller),
                       ends snapped to ``[lo, hi]``.
        - 'custom'   : use ``custom_edges`` directly (raw space, already
                       includes the outer ends), sorted/clipped to [lo, hi].
    data_col : np.ndarray, optional
        Raw-space values of the investigated input (required for 'quantile').
    kan_knots_raw : np.ndarray, optional
        Raw-space KAN spline knots (required for 'kan').
    custom_edges : np.ndarray, optional
        Raw-space section edges (required for 'custom'). The full edge array
        (interior transition points plus the outer [lo, hi] ends) is expected;
        ends are snapped to [lo, hi] and duplicates removed.

    Returns
    -------
    edges : np.ndarray
        Section boundaries, strictly increasing, ``edges[0]=lo``,
        ``edges[-1]=hi``. Length ``n_eff + 1`` where ``n_eff <= n_sections``
        if degenerate edges were deduplicated.
    centers : np.ndarray
        Section midpoints, length ``n_eff``.
    """
    lo = float(lo)
    hi = float(hi)

    if mode == 'equal':
        edges = np.linspace(lo, hi, n_sections + 1)

    elif mode == 'quantile':
        if data_col is None:
            raise ValueError("mode='quantile' requires data_col.")
        data_col = np.asarray(data_col, dtype=float)
        data_col = data_col[np.isfinite(data_col)]
        if data_col.size == 0:
            warnings.warn("quantile: data_col empty/all-nonfinite; "
                          "falling back to 'equal'.")
            edges = np.linspace(lo, hi, n_sections + 1)
        else:
            edges = np.quantile(data_col, np.linspace(0.0, 1.0, n_sections + 1))
            edges = np.clip(edges, lo, hi)
            # Snap ends so the partition fully tiles [lo, hi].
            edges[0] = lo
            edges[-1] = hi
            edges = np.unique(edges)
            if edges.size < 2:
                warnings.warn("quantile: degenerate edges collapsed; "
                              "falling back to 'equal'.")
                edges = np.linspace(lo, hi, n_sections + 1)
            elif edges.size - 1 < n_sections:
                warnings.warn(
                    "quantile: ties reduced effective sections from "
                    "%d to %d." % (n_sections, edges.size - 1))

    elif mode == 'kan':
        if kan_knots_raw is None:
            raise ValueError("mode='kan' requires kan_knots_raw.")
        knots = np.asarray(kan_knots_raw, dtype=float).ravel()
        knots = knots[np.isfinite(knots)]
        if knots.size < 2:
            warnings.warn("kan: insufficient knots; falling back to 'equal'.")
            edges = np.linspace(lo, hi, n_sections + 1)
        else:
            edges = np.sort(knots)
            edges = np.clip(edges, lo, hi)
            edges[0] = lo
            edges[-1] = hi
            edges = np.unique(edges)
            if edges.size < 2:
                warnings.warn("kan: degenerate knots; falling back to 'equal'.")
                edges = np.linspace(lo, hi, n_sections + 1)

    elif mode == 'custom':
        if custom_edges is None:
            raise ValueError("mode='custom' requires custom_edges.")
        edges = np.asarray(custom_edges, dtype=float).ravel()
        edges = edges[np.isfinite(edges)]
        edges = np.sort(edges)
        edges = np.clip(edges, lo, hi)
        if edges.size < 2:
            edges = np.array([lo, hi], dtype=float)
        edges[0] = lo
        edges[-1] = hi
        edges = np.unique(edges)
        if edges.size < 2:
            warnings.warn("custom: degenerate edges; falling back to 'equal'.")
            edges = np.linspace(lo, hi, n_sections + 1)

    else:
        raise ValueError(
            "Unknown mode %r; expected 'equal'|'quantile'|'kan'|'custom'."
            % (mode,))

    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


# ----------------------------------------------------------------------------
# Numerical gradient (absolute)
# ----------------------------------------------------------------------------
def numerical_gradient_abs(func_batch, X, feat_idx, eps=1e-5):
    """Absolute partial derivative ``|df/dx_{feat_idx}|`` via central diff.

    Parameters
    ----------
    func_batch : callable
        ``func_batch(X)`` maps ``[N, d]`` -> ``[N]``.
    X : np.ndarray
        ``[N, d]`` evaluation points (raw space).
    feat_idx : int
        Column along which to differentiate.
    eps : float
        Absolute finite-difference step.

    Returns
    -------
    np.ndarray
        ``[N]`` array of ``|df/dx_{feat_idx}|``. Non-finite entries are set to
        ``np.nan`` so the caller can mask them.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    step = float(eps)
    Xp = X.copy()
    Xm = X.copy()
    Xp[:, feat_idx] += step
    Xm[:, feat_idx] -= step

    fp = np.asarray(func_batch(Xp), dtype=float).ravel()
    fm = np.asarray(func_batch(Xm), dtype=float).ravel()

    grad = (fp - fm) / (2.0 * step)
    grad = np.abs(grad)
    grad[~np.isfinite(grad)] = np.nan
    return grad


def _section_mean(grad, in_k):
    """Mean of finite gradients in a section; NaN if too few valid samples."""
    g = grad[in_k]
    g = g[np.isfinite(g)]
    if g.size < 5:
        return np.nan
    return float(np.mean(g))


# ----------------------------------------------------------------------------
# Main AGSM computation
# ----------------------------------------------------------------------------
def compute_gradient_agsm(func_batch, bounds, feat_names, top2_idx,
                          n_sections=10, n_samples_per_section=512, seed=42,
                          section_mode='equal', data_X=None,
                          kan_knots_raw=None, custom_edges=None, fd_eps=1e-5):
    """Compute Sectional GSA (AGSM) for the given target inputs.

    Parameters
    ----------
    func_batch : callable
        Batch func ``[m, n] -> [m]`` in raw input space (see ``make_batch_func``).
    bounds : sequence of [lo, hi]
        Per-input raw-space bounds, length ``n``.
    feat_names : sequence of str
        Per-input names, length ``n``.
    top2_idx : sequence of int
        Indices of the inputs to section (typically the top-2 ranked).
    n_sections : int
        Number of sections ``N`` per investigated input.
    n_samples_per_section : int
        MC samples drawn per section (total samples per input scale with N).
    seed : int
        RNG seed for reproducibility.
    section_mode : {'equal', 'quantile', 'kan'}
        Section construction mode (see ``make_sections``).
    data_X : np.ndarray, optional
        ``[M, n]`` raw-space data matrix (required for ``section_mode='quantile'``).
    kan_knots_raw : dict, optional
        ``{feat_idx: np.ndarray}`` of raw-space KAN knots
        (required for ``section_mode='kan'``).
    custom_edges : dict, optional
        ``{feat_idx: np.ndarray}`` of raw-space section edges (including the
        outer ends), required for ``section_mode='custom'``.
    fd_eps : float
        Base finite-difference step; scaled per-input by its width.

    Returns
    -------
    section_centers : dict {feat_idx: np.ndarray [n_eff]}
    S_hat           : dict {feat_idx: np.ndarray [n_eff]}  raw mean |df/dx_l| per section
    S_a             : dict {feat_idx: np.ndarray [n_eff]}  normalized cross-comparable
    R               : dict {feat_idx: np.ndarray [n_eff]}  within-feature (sums to 1)

    Notes
    -----
    Also computed internally and consistent with the spec:
      - Global ``S_hat_global[l]`` = sum_k S_hat[l][k] * width[l][k] / (hi-lo)
        == full-domain mean |df/dx_l| (the weighted sum).
      - ``total_denom`` = sum over all targets/sections of the width-weighted
        S_hat, used to normalize ``S_a`` so it is cross-input comparable.
    """
    rng = np.random.default_rng(seed)
    bounds = [[float(b[0]), float(b[1])] for b in bounds]
    n_inputs = len(bounds)

    section_centers = {}
    section_edges = {}
    S_hat = {}                 # raw per-section mean |df/dx_l|
    S_hat_global = {}          # width-weighted global mean per target
    width_frac = {}            # width[l][k] / (hi-lo)

    # ---- Step 1+2+3: per-target sectioning + per-section raw mean gradient ----
    for l in top2_idx:
        lo, hi = bounds[l]
        full_width = hi - lo

        data_col = data_X[:, l] if data_X is not None else None
        knots_l = kan_knots_raw.get(l) if kan_knots_raw is not None else None
        custom_l = custom_edges.get(l) if custom_edges is not None else None
        edges, centers = make_sections(
            lo, hi, n_sections, mode=section_mode,
            data_col=data_col, kan_knots_raw=knots_l, custom_edges=custom_l)
        n_eff = len(centers)

        widths = np.diff(edges)
        wfrac = widths / full_width if full_width > _EPS else np.zeros_like(widths)

        step = fd_eps * full_width if full_width > _EPS else fd_eps

        s_hat_l = np.full(n_eff, np.nan)
        for k in range(n_eff):
            a_k, b_k = edges[k], edges[k + 1]
            # Sample subinput space C_{l,[k]}: x_l ~ U(a_k, b_k), others full.
            Xs = np.empty((n_samples_per_section, n_inputs))
            for j in range(n_inputs):
                if j == l:
                    Xs[:, j] = rng.uniform(a_k, b_k, size=n_samples_per_section)
                else:
                    Xs[:, j] = rng.uniform(bounds[j][0], bounds[j][1],
                                           size=n_samples_per_section)
            grad = numerical_gradient_abs(func_batch, Xs, l, eps=step)
            valid = np.isfinite(grad)
            if valid.sum() < 5:
                warnings.warn(
                    "Feature %r (%s) section %d has <5 valid samples; "
                    "S_hat set to NaN." % (l, feat_names[l], k))
                s_hat_l[k] = np.nan
            else:
                s_hat_l[k] = float(np.mean(grad[valid]))

        section_centers[l] = centers
        section_edges[l] = edges
        S_hat[l] = s_hat_l
        width_frac[l] = wfrac

        # Global = width-weighted sum of per-section means (== full-domain mean).
        valid_g = np.isfinite(s_hat_l)
        if valid_g.any():
            S_hat_global[l] = float(np.sum(s_hat_l[valid_g] * wfrac[valid_g]))
        else:
            S_hat_global[l] = 0.0

    # ---- total denominator across all targets (width-weighted) ----
    total_denom = 0.0
    for l in top2_idx:
        s_hat_l = S_hat[l]
        wfrac = width_frac[l]
        valid = np.isfinite(s_hat_l)
        total_denom += float(np.sum(s_hat_l[valid] * wfrac[valid]))

    # ---- Step 4: derived measures ----
    S_a = {}
    R = {}
    for l in top2_idx:
        s_hat_l = S_hat[l]
        wfrac = width_frac[l]
        valid = np.isfinite(s_hat_l)
        contrib = np.zeros_like(s_hat_l)
        contrib[valid] = s_hat_l[valid] * wfrac[valid]   # S_hat_a contribution

        glob_l = S_hat_global[l]

        # Cross-input comparable: S_a = contribution / total_denom.
        if total_denom > _EPS:
            sa = contrib / total_denom
        else:
            warnings.warn("Total denominator ~0; S_a set to zeros.")
            sa = np.zeros_like(s_hat_l)

        # Within-input relative: R = contribution / S_hat_global[l] (sums to 1).
        if glob_l > _EPS:
            r = contrib / glob_l
        else:
            warnings.warn(
                "Feature %r (%s) has zero global measure; R/S_a set to zeros."
                % (l, feat_names[l]))
            r = np.zeros_like(s_hat_l)
            sa = np.zeros_like(s_hat_l)

        S_a[l] = sa
        R[l] = r

    return section_centers, S_hat, S_a, R


# ----------------------------------------------------------------------------
# Global from sectional
# ----------------------------------------------------------------------------
def compute_global_from_sectional(S_hat, section_centers, bounds_l):
    """Reconstruct the global sensitivity measure from sectional means.

    ``S_hat_global = sum_k S_hat[k] * width_k / total_width``.

    Parameters
    ----------
    S_hat : np.ndarray
        Per-section raw mean ``|df/dx_l|`` (may contain NaN for empty sections).
    section_centers : np.ndarray
        Section midpoints (used to reconstruct widths together with bounds).
    bounds_l : sequence [lo, hi]
        Raw-space bounds of the investigated input.

    Returns
    -------
    float
        The width-weighted global mean ``|df/dx_l|``.

    Notes
    -----
    Widths are reconstructed from centers by assuming each center is the
    midpoint of its section and that sections tile ``[lo, hi]`` contiguously.
    Edges are recovered as the midpoints between consecutive centers with the
    bounds as the outer edges.
    """
    S_hat = np.asarray(S_hat, dtype=float)
    centers = np.asarray(section_centers, dtype=float)
    lo, hi = float(bounds_l[0]), float(bounds_l[1])
    full_width = hi - lo
    if full_width <= _EPS:
        return 0.0

    n = len(centers)
    if n == 0:
        return 0.0

    # Reconstruct edges: interior edges = midpoints between centers; ends = bounds.
    edges = np.empty(n + 1)
    edges[0] = lo
    edges[-1] = hi
    if n > 1:
        edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    widths = np.diff(edges)
    wfrac = widths / full_width

    valid = np.isfinite(S_hat)
    return float(np.sum(S_hat[valid] * wfrac[valid]))


# ----------------------------------------------------------------------------
# Transition point detection
# ----------------------------------------------------------------------------
def find_agsm_transition_points(section_centers_i, S_a_i,
                                section_centers_j, S_a_j,
                                feat_name_i, feat_name_j):
    """Find x-values where the dominant feature switches (S_a curves cross).

    Both curves should share the same section boundaries (same domain bounds).
    If they do not, ``S_a_j`` is resampled onto ``section_centers_i`` via linear
    interpolation.

    Parameters
    ----------
    section_centers_i, section_centers_j : np.ndarray
        Section midpoints for features i and j.
    S_a_i, S_a_j : np.ndarray
        Cross-input comparable AGSM values for features i and j.
    feat_name_i, feat_name_j : str
        Feature names for reporting.

    Returns
    -------
    list of dict
        Each: ``{'point': float, 'from_feat': str, 'to_feat': str,
        'section_k': int}`` -- the interpolated crossing x-value, the dominant
        feature before/after, and the left section index of the crossing.
    """
    ci = np.asarray(section_centers_i, dtype=float)
    cj = np.asarray(section_centers_j, dtype=float)
    sai = np.asarray(S_a_i, dtype=float).copy()
    saj = np.asarray(S_a_j, dtype=float).copy()

    # Align j onto i's centers if boundaries differ.
    same = (ci.shape == cj.shape) and np.allclose(ci, cj)
    if not same:
        saj = np.interp(ci, cj, saj)
    centers = ci

    # Treat NaN as 0 (empty/zero-effect sections do not dominate).
    sai = np.nan_to_num(sai, nan=0.0, posinf=0.0, neginf=0.0)
    saj = np.nan_to_num(saj, nan=0.0, posinf=0.0, neginf=0.0)

    diff = sai - saj
    transitions = []
    for k in range(len(diff) - 1):
        d0, d1 = diff[k], diff[k + 1]
        if d0 * d1 < 0:  # strict sign change -> a crossing in (centers[k], centers[k+1])
            denom = (d1 - d0)
            if abs(denom) < _EPS:
                cross = 0.5 * (centers[k] + centers[k + 1])
            else:
                cross = centers[k] - d0 * (centers[k + 1] - centers[k]) / denom
            if d0 > 0:
                from_feat, to_feat = feat_name_i, feat_name_j
            else:
                from_feat, to_feat = feat_name_j, feat_name_i
            transitions.append({
                'point': float(cross),
                'from_feat': from_feat,
                'to_feat': to_feat,
                'section_k': int(k),
            })
    return transitions


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def plot_agsm_vs_kan(section_centers, S_a, top2_idx, feat_names,
                     kan_inflection_points, agsm_transition_points,
                     save_path, title=''):
    """Plot AGSM sectional sensitivity vs KAN inflection / AGSM transitions.

    Parameters
    ----------
    section_centers : dict {feat_idx: np.ndarray}
        Section midpoints per investigated input.
    S_a : dict {feat_idx: np.ndarray}
        Cross-input comparable AGSM per investigated input.
    top2_idx : sequence of int
        The (up to two) investigated input indices.
    feat_names : sequence of str
        Per-input names.
    kan_inflection_points : sequence of float or dict
        KAN inflection x-values (raw space). May be a flat list (drawn on every
        panel) or a dict ``{feat_idx: [values]}`` (drawn on the matching panel).
    agsm_transition_points : sequence of dict
        Output of ``find_agsm_transition_points`` (uses each item's 'point').
    save_path : str
        Path stem; ``.png``, ``.svg``, ``.eps`` are appended.
    title : str
        Figure / axis title.

    Notes
    -----
    If the two features share the same domain bounds (typical for analytic
    functions), both curves are drawn on one axis with a shared raw-space
    x-axis. Otherwise two stacked subplots are used (each with its own x-axis).
    """
    idxs = list(top2_idx)

    def _inflect_for(feat):
        if kan_inflection_points is None:
            return []
        if isinstance(kan_inflection_points, dict):
            return list(kan_inflection_points.get(feat, []))
        return list(kan_inflection_points)

    transitions = agsm_transition_points or []

    # Determine whether the features share bounds (compare center spans).
    shared = False
    if len(idxs) == 2:
        c0 = np.asarray(section_centers[idxs[0]], dtype=float)
        c1 = np.asarray(section_centers[idxs[1]], dtype=float)
        if c0.shape == c1.shape and np.allclose(
                [c0.min(), c0.max()], [c1.min(), c1.max()], rtol=1e-3, atol=1e-6):
            shared = True

    colors = ['#1f77b4', '#d62728']

    with plt.rc_context(SA_RC):
        if len(idxs) == 1 or shared:
            fig, ax = plt.subplots()
            axes = [ax]
            for n, feat in enumerate(idxs):
                centers = np.asarray(section_centers[feat], dtype=float)
                vals = np.asarray(S_a[feat], dtype=float)
                ax.plot(centers, vals, drawstyle='steps-mid',
                        color=colors[n % len(colors)],
                        label=str(feat_names[feat]))
            _overlay(ax, _flat_inflect(kan_inflection_points, idxs),
                     [t['point'] for t in transitions])
            ax.set_xlabel('Feature value (raw)')
            ax.set_ylabel(r'$S^{a}_{l,[k]}$ (normalized)')
            if title:
                ax.set_title(title)
            ax.legend()
        else:
            fig, axes = plt.subplots(len(idxs), 1, sharex=False)
            axes = np.atleast_1d(axes)
            for n, feat in enumerate(idxs):
                ax = axes[n]
                centers = np.asarray(section_centers[feat], dtype=float)
                vals = np.asarray(S_a[feat], dtype=float)
                ax.plot(centers, vals, drawstyle='steps-mid',
                        color=colors[n % len(colors)],
                        label=str(feat_names[feat]))
                _overlay(ax, _inflect_for(feat),
                         [t['point'] for t in transitions])
                ax.set_ylabel(r'$S^{a}_{l,[k]}$')
                ax.legend()
            axes[-1].set_xlabel('Feature value (raw)')
            if title:
                axes[0].set_title(title)

        fig.savefig(save_path + '.png')
        fig.savefig(save_path + '.svg')
        fig.savefig(save_path + '.eps')
        plt.close(fig)

    return save_path


def _flat_inflect(kan_inflection_points, idxs):
    """Collect inflection x-values for a shared-axis plot."""
    if kan_inflection_points is None:
        return []
    if isinstance(kan_inflection_points, dict):
        out = []
        for feat in idxs:
            out.extend(kan_inflection_points.get(feat, []))
        return out
    return list(kan_inflection_points)


def _overlay(ax, inflection_pts, transition_pts):
    """Draw KAN inflection (green dashed) and AGSM transition (orange dotted) lines."""
    seen_inflect_label = False
    for x in inflection_pts:
        if not np.isfinite(x):
            continue
        ax.axvline(x, color='green', linestyle='--', linewidth=1.0, alpha=0.8,
                   label=None if seen_inflect_label else 'KAN inflection')
        seen_inflect_label = True
    seen_trans_label = False
    for x in transition_pts:
        if not np.isfinite(x):
            continue
        ax.axvline(x, color='orange', linestyle=':', linewidth=1.2, alpha=0.9,
                   label=None if seen_trans_label else 'AGSM transition')
        seen_trans_label = True
