"""Precision-aware cached stages of the curve pipeline."""

from functools import lru_cache, wraps

from ._hyperelliptic import _hyperelliptic_periods
from ._hyperelliptic.model import (
    _hyperelliptic_coefficients, _hyperelliptic_roots,
)
from ._context import _curve_cache_state
from .differentials import _baker_differentials
from .integration import _integrate_lifted_path_chain
from .jacobian import _canonical_polygon_riemann_constant
from .monodromy import (
    _numerical_ordered_canonical_polygon, _ordered_monodromy_graph,
    _radial_plane_curve_monodromy, _symplectic_reduce_intersection,
)
from .polynomial import _plane_curve_critical_values

# Cached computational stages
# ---------------------------

_MONODROMY_CIRCLE_STEPS = 12
_MONODROMY_MAX_REFINEMENTS = 20


def _curve_stage_cache(maxsize):
    """Cache a curve stage by its key and the context's numerical state.

    Stage keys normally contain prepared curves and, for period integration,
    differential callables.  An unhashable callable bypasses this cache.
    Cached values are private immutable records; public functions assemble
    matrices from them, so a cached result cannot be mutated publicly.
    """
    def decorator(f):
        @lru_cache(maxsize=maxsize)
        def cached(unused_state, ctx, key):
            return f(ctx, key)

        @wraps(f)
        def wrapper(ctx, key):
            try:
                hash(key)
            except TypeError:
                # Callable differential objects are valid public inputs even
                # when they deliberately opt out of hashing.  They cannot be
                # LRU keys, but the numerical stage remains usable.
                return f(ctx, key)
            return cached(_curve_cache_state(ctx), ctx, key)

        wrapper.cache_info = cached.cache_info
        wrapper.cache_clear = cached.cache_clear
        return wrapper
    return decorator


@_curve_stage_cache(32)
def _stage_branch_locus(ctx, curve):
    """Return ``(branch_values, resultant)`` for a prepared plane curve."""
    return _plane_curve_critical_values(ctx, curve)


@_curve_stage_cache(16)
def _stage_hyperelliptic_periods(ctx, key):
    """Return a specialized first- or second-kind half-period bundle.

    The cached matrices are private.  Public orchestration copies them before
    returning a result so callers cannot mutate the cache.
    """
    coefficients, second_kind = key
    return _hyperelliptic_periods(
        ctx, coefficients, second_kind=second_kind)


@_curve_stage_cache(16)
def _stage_hyperelliptic_homology(ctx, coefficients):
    """Validate a Baker-marked model and return its genus."""
    coefficients = _hyperelliptic_coefficients(ctx, coefficients)
    # The ordered roots determine the Baker marking as well as checking that
    # the model is smooth. Period integration remains a separate lazy stage.
    _hyperelliptic_roots(ctx, coefficients)
    return (len(coefficients) - 2) // 2


@_curve_stage_cache(8)
def _stage_monodromy(ctx, curve):
    """Return the guarded radial monodromy system of a prepared curve."""
    branch_values, unused_resultant = _stage_branch_locus(ctx, curve)
    return _radial_plane_curve_monodromy(
        ctx, curve, branch_values,
        circle_steps=_MONODROMY_CIRCLE_STEPS,
        max_refinements=_MONODROMY_MAX_REFINEMENTS)


@_curve_stage_cache(16)
def _stage_baker_differentials(ctx, key):
    """Cache the structured Baker basis for the current numerical state."""
    curve, genus = key
    return _baker_differentials(ctx, curve, genus)


@_curve_stage_cache(8)
def _stage_monodromy_graph(ctx, curve):
    """Return ``(lifted_graph, symplectic_reduction)`` for a curve."""
    monodromy = _stage_monodromy(ctx, curve)
    graph = _ordered_monodromy_graph(monodromy)
    reduction = _symplectic_reduce_intersection(graph.intersection)
    return graph, reduction


@_curve_stage_cache(8)
def _stage_canonical_polygon(ctx, curve):
    """Return the certified numerical canonical polygon for a curve."""
    graph, unused_reduction = _stage_monodromy_graph(ctx, curve)
    return _numerical_ordered_canonical_polygon(
        ctx, graph, _stage_monodromy(ctx, curve))


@_curve_stage_cache(8)
def _stage_canonical_cycles(ctx, curve):
    """Return polygon sides realizing the canonical compact homology basis."""
    return _stage_canonical_polygon(ctx, curve).chains


@_curve_stage_cache(16)
def _stage_cycle_integrals(ctx, key):
    """Integrate differential forms over the canonical cycles of a curve.

    ``key`` is ``(curve, forms, quadrature_order)``.  The result is
    ``(columns, max_sheet_residual)`` with one column of form values per
    canonical cycle.
    """
    curve, forms, quadrature_order = key
    forms = tuple(forms)
    genus = _stage_monodromy(ctx, curve).genus
    branch_values = (_stage_branch_locus(ctx, curve)[0]
                     if quadrature_order == "geometry" else None)
    chains = _stage_canonical_cycles(ctx, curve)
    columns = []
    max_sheet_residual = ctx.zero
    integral_cache = {}
    for chain in chains[:2 * genus]:
        integral = _integrate_lifted_path_chain(
            ctx, curve, chain, forms, quadrature_order=quadrature_order,
            integral_cache=integral_cache, branch_values=branch_values)
        columns.append(integral.values)
        max_sheet_residual = max(
            max_sheet_residual, integral.max_sheet_residual)
    return tuple(columns), max_sheet_residual


@_curve_stage_cache(8)
def _stage_riemann_constant(ctx, key):
    """Return the direct additive Riemann constant at the polygon base."""
    curve, forms, quadrature_order = key
    forms = tuple(forms)
    genus = _stage_monodromy(ctx, curve).genus
    if len(forms) != genus:
        raise ValueError("one holomorphic differential is required per genus")
    columns, unused_residual = _stage_cycle_integrals(
        ctx, (curve, forms, quadrature_order))
    periods = ctx.matrix([
        [columns[column][row] for column in range(2 * genus)]
        for row in range(genus)])
    a_periods = periods[:, :genus]
    raw_tau = a_periods ** -1 * periods[:, genus:]
    tau = (raw_tau + raw_tau.T) / 2
    polygon = _stage_canonical_polygon(ctx, curve)
    return _canonical_polygon_riemann_constant(
        ctx, curve, polygon, forms, a_periods, tau,
        max(12, ctx.dps // 2) if quadrature_order == "geometry"
        else quadrature_order)


def _curve_differential_sequence(differentials, name):
    """Return a validated tuple of differential callables."""
    try:
        forms = tuple(differentials)
    except TypeError:
        raise ValueError(name + " must be a sequence of callables")
    if not forms or any(not callable(value) for value in forms):
        raise ValueError(name + " must be a sequence of callables")
    return forms


def _tau_imaginary_eigenvalues(ctx, tau):
    """Return the eigenvalues of the imaginary part of a period matrix."""
    genus = tau.rows
    imaginary_tau = ctx.matrix([
        [ctx.im(tau[row, column]) for column in range(genus)]
        for row in range(genus)])
    return tuple(ctx.eigsy(imaginary_tau, eigvals_only=True))
