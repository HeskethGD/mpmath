"""Precision-aware cached stages of the curve pipeline."""

from functools import lru_cache, wraps

from ._hyperelliptic import _hyperelliptic_periods
from ._hyperelliptic.model import (
    _hyperelliptic_coefficients, _hyperelliptic_roots,
)
from ._context import _curve_cache_state
from ._records import _GeometricPeriodData
from .geometry import _voronoi_plane_graph
from .continuation import _lift_plane_graph
from .differentials import (
    _baker_basis, _baker_callable, _evaluate_baker_basis,
)
from .integration import (
    _integrate_plane_curve_path, _integrate_geometric_chains,
    _integrate_geometric_loops_iterated, _integrate_geometric_callable_chains,
    _integrate_plane_curve_branch, _pullback_plane_curve_differentials,
)
from .jacobian import (
    _canonical_polygon_riemann_constant, _normalised_differentials,
    _riemann_constant_from_iterated_cycles,
    _finite_geometric_abel_value, _normalise_curve_endpoint,
)
from .monodromy import (
    _graph_cycle_word, _numerical_ordered_canonical_polygon,
    _ordered_monodromy_graph,
    _radial_plane_curve_monodromy, _symplectic_reduce_intersection,
    _geometric_ribbon_graph, _geometric_canonical_polygon,
)
from .polynomial import _plane_curve_critical_values

# Cached computational stages
# ---------------------------

_MONODROMY_CIRCLE_STEPS = 8
_MONODROMY_MAX_REFINEMENTS = 20
_GEOMETRIC_GUARD_BITS = 10


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


@_curve_stage_cache(8)
def _stage_geometric_cover(ctx, curve):
    """Construct and lift a native base-plane graph at the current precision."""
    values, unused_resultant = _stage_branch_locus(ctx, curve)
    geometry = _voronoi_plane_graph(ctx, values)
    return _lift_plane_graph(ctx, curve, geometry)


@_curve_stage_cache(8)
def _stage_geometric_polygon(ctx, curve):
    """Return a covering graph and its canonical based polygon together."""
    cover = _stage_geometric_cover(ctx, curve)
    graph = _geometric_ribbon_graph(ctx, cover)
    polygon = _geometric_canonical_polygon(ctx, cover, graph)
    return graph, polygon


@_curve_stage_cache(8)
def _stage_geometric_periods_working(ctx, curve):
    """Unrounded period data shared by guarded dependent computations."""
    cover = _stage_geometric_cover(ctx, curve)
    graph, polygon = _stage_geometric_polygon(ctx, curve)
    if graph.genus == 0:
        raise ValueError("a genus-zero curve has no first-kind periods")
    basis = _baker_basis(ctx, curve, graph.genus)
    columns, residual = _integrate_geometric_chains(
        ctx, curve, polygon.chains, basis, cover.geometry.branch_values)
    return _GeometricPeriodData(
        graph.genus, basis, columns, residual, cover, graph, polygon, ctx.prec)


@_curve_stage_cache(8)
def _stage_geometric_custom_periods_working(ctx, key):
    """Unrounded g-form periods with order checks, without Baker checks.

    The caller selects the interpretation as first or second kind and is
    responsible for the forms being regular on the integration paths.
    """
    curve, forms = key
    cover = _stage_geometric_cover(ctx, curve)
    graph, polygon = _stage_geometric_polygon(ctx, curve)
    if not graph.genus or len(forms) != graph.genus:
        raise ValueError("differentials must contain one form per positive genus")
    columns, residual = _integrate_geometric_callable_chains(
        ctx, curve, polygon.chains, forms, cover.geometry.branch_values)
    return _GeometricPeriodData(
        graph.genus, None, columns, residual, cover, graph, polygon, ctx.prec)


def _geometric_period_data_working(ctx, curve, forms):
    if forms is None:
        return _stage_geometric_periods_working(ctx, curve)
    return _stage_geometric_custom_periods_working(ctx, (curve, forms))


@_curve_stage_cache(8)
def _stage_geometric_custom_periods(ctx, key):
    with ctx.extraprec(_GEOMETRIC_GUARD_BITS):
        data = _stage_geometric_custom_periods_working(ctx, key)
    return data._replace(
        columns=tuple(tuple(+v for v in column) for column in data.columns),
        max_sheet_residual=+data.max_sheet_residual)


@_curve_stage_cache(8)
def _stage_geometric_periods(ctx, curve):
    """Private native-graph backend for automatic first-kind periods.

    ``curve`` is a prepared polynomial. Columns contain full periods in the
    returned polygon's marking, with numerator order recorded in ``basis``.
    Columns are rounded to the caller's precision; the cover and based words
    retain their guarded working precision for later path operations.

    This entry point does not select or replace the public radial backend.
    In particular, these periods must not be paired with a radial polygon's
    Abel coordinates or Riemann constants. Supplied holomorphic callables
    use the separate custom-period stage with successive-order checks.
    """
    with ctx.extraprec(_GEOMETRIC_GUARD_BITS):
        data = _stage_geometric_periods_working(ctx, curve)
    return data._replace(
        columns=tuple(tuple(+value for value in column) for column in data.columns),
        max_sheet_residual=+data.max_sheet_residual)


@_curve_stage_cache(8)
def _stage_geometric_riemann_constant(ctx, curve):
    """Return immutable (K, based a-loop integrals) in the geometric marking.

    K is rounded to the caller's precision. Loop integrals retain working
    precision and use forms normalized by the unrounded full a-periods.
    The base place is the root of the matching geometric period polygon.
    """
    return _geometric_riemann_constant(ctx, curve)


@_curve_stage_cache(8)
def _stage_geometric_custom_riemann_constant(ctx, key):
    return _geometric_riemann_constant(ctx, *key)


def _geometric_riemann_constant(ctx, curve, forms=None):
    with ctx.extraprec(_GEOMETRIC_GUARD_BITS):
        data = _geometric_period_data_working(ctx, curve, forms)
        genus = data.genus
        periods = ctx.matrix([[column[row] for column in data.columns]
                              for row in range(genus)])
        a_periods = periods[:, :genus]
        raw_tau = a_periods ** -1 * periods[:, genus:]
        tau = (raw_tau + raw_tau.T) / 2
        if forms is None:
            forms = tuple(_baker_callable(ctx, data.basis, i) for i in range(genus))
        normalised = _normalised_differentials(ctx, forms, a_periods)
        polygon = data.polygon.polygon
        cycles = _integrate_geometric_loops_iterated(
            ctx, curve, data.cover, data.graph, polygon,
            polygon.a_loops, normalised)
        value = _riemann_constant_from_iterated_cycles(ctx, tau, cycles)
    return tuple(+entry for entry in value), cycles


def _geometric_abel_value(ctx, curve, place, base_place=None):
    """Private single-place wrapper for geometric Abel integration."""
    return _geometric_abel_divisor(ctx, curve, (place,), base_place)


def _geometric_abel_divisor(ctx, curve, places, base_place=None, forms=None,
                            second_forms=()):
    """Integrate a divisor with shared operation-local edge and rule caches.

    Ownership is checked at caller precision before entering the guarded
    computation. A chart tail runs from the represented place to its affine
    junction, so its pulled-back integral is subtracted from the open value.
    """
    # Charts depend on the stage validators; import here to avoid a cycle.
    from .charts import _validated_chart_coordinate_map

    def endpoint(place, name):
        junction, tail, unused_place = _normalise_curve_endpoint(
            ctx, curve, place, name)
        coordinate_map = (None if tail is None else
                          _validated_chart_coordinate_map(ctx, tail.chart))
        return junction, tail, coordinate_map

    targets = tuple(endpoint(place, "place") for place in places)
    base = None if base_place is None else endpoint(base_place, "base_place")
    with ctx.extraprec(_GEOMETRIC_GUARD_BITS):
        supplied = forms is not None or bool(second_forms)
        data = _geometric_period_data_working(ctx, curve, forms)
        if forms is None:
            forms = tuple(_baker_callable(ctx, data.basis, i)
                          for i in range(data.genus))

        if second_forms and len(second_forms) != data.genus:
            raise ValueError("second_differentials must contain one form per genus")
        forms = forms + tuple(second_forms)
        rules, edges = {}, {}

        def value_at(endpoint):
            junction, tail, coordinate_map = endpoint
            value = _finite_geometric_abel_value(
                ctx, curve, data, junction, forms,
                quadrature_cache=rules, edge_cache=edges,
                check_convergence=supplied)
            if tail is not None:
                pullbacks = _pullback_plane_curve_differentials(forms, coordinate_map)
                local = _integrate_plane_curve_branch(
                    ctx, tail.chart.curve, tail.branch, pullbacks,
                    check_convergence=bool(second_forms))
                value = tuple(a - b for a, b in zip(value, local.values))
            return value

        values = tuple(value_at(target) for target in targets)
        value = tuple(ctx.fsum(item[i] for item in values)
                      for i in range(len(forms)))
        if base is not None and targets:
            origin = value_at(base)
            value = tuple(a - len(targets) * b for a, b in zip(value, origin))
    return tuple(+entry for entry in value)


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
    basis = _baker_basis(ctx, curve, genus)
    forms = tuple(
        _baker_callable(ctx, basis, index)
        for index in range(len(basis[0])))
    return forms, basis


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

    ``key`` is ``(curve, forms, quadrature_order, baker_basis)``.  The
    result is ``(columns, max_sheet_residual)`` with one column of form
    values per canonical cycle.  Ordinary periods are additive, so integrate
    each generator on each required sheet once and apply the polygon's
    integer cycle transformation.  Iterated integrals still use the full
    based paths retained by the polygon.
    """
    curve, forms, quadrature_order, baker_basis = key
    forms = tuple(forms)
    genus = _stage_monodromy(ctx, curve).genus
    branch_values = (_stage_branch_locus(ctx, curve)[0]
                     if quadrature_order == "geometry" else None)
    graph, unused_reduction = _stage_monodromy_graph(ctx, curve)
    polygon = _stage_canonical_polygon(ctx, curve)
    monodromy = _stage_monodromy(ctx, curve)
    identity = tuple(range(curve.y_degree))
    generators = tuple(
        generator for generator in monodromy.ribbon_generators
        if generator.permutation != identity)
    if (tuple(generator.permutation for generator in generators)
            != graph.permutations or
            any(orientation != 1 for orientation in graph.branch_orientations)):
        raise ValueError("ordered graph and numerical generators differ")
    max_sheet_residual = ctx.zero
    generator_integrals = {}
    quadrature_cache = {}
    evaluator = None
    if baker_basis is not None:
        automatic_count = len(baker_basis[0])

        def evaluator(x, y):
            automatic = _evaluate_baker_basis(
                ctx, baker_basis, x, y)
            supplied = tuple(
                differential(x, y)
                for differential in forms[automatic_count:])
            return automatic + supplied

    canonical_rows = polygon.transformation[:2 * genus]
    needed_cycles = {
        index for row in canonical_rows
        for index, coefficient in enumerate(row) if coefficient}
    graph_columns = [None] * len(graph.cycles)
    for cycle_index in sorted(needed_cycles):
        cycle = graph.cycles[cycle_index]
        word = _graph_cycle_word(graph, cycle)
        sheet = word.start_sheet
        pieces = []
        for step in word.steps:
            generator = generators[step.branch_index]
            for unused in range(step.turns):
                cache_key = step.branch_index, sheet
                integral = generator_integrals.get(cache_key)
                if integral is None:
                    integral = _integrate_plane_curve_path(
                        ctx, curve, generator.continuation, forms,
                        sheet=sheet, quadrature_order=quadrature_order,
                        branch_values=branch_values,
                        differential_evaluator=evaluator,
                        quadrature_cache=quadrature_cache)
                    generator_integrals[cache_key] = integral
                    max_sheet_residual = max(
                        max_sheet_residual, integral.max_sheet_residual)
                pieces.append(integral.values)
                sheet = generator.permutation[sheet]
        if sheet != word.start_sheet:
            raise ValueError("numerical graph cycle did not close")
        graph_columns[cycle_index] = tuple(
            ctx.fsum(piece[index] for piece in pieces)
            for index in range(len(forms)))

    columns = tuple(tuple(ctx.fsum(
        coefficient * graph_columns[index][form_index]
        for index, coefficient in enumerate(row) if coefficient)
        for form_index in range(len(forms)))
        for row in canonical_rows)
    return tuple(columns), max_sheet_residual


@_curve_stage_cache(8)
def _stage_riemann_constant(ctx, key):
    """Return the direct additive Riemann constant at the polygon base."""
    curve, forms, quadrature_order, baker_basis = key
    forms = tuple(forms)
    genus = _stage_monodromy(ctx, curve).genus
    if len(forms) != genus:
        raise ValueError("one holomorphic differential is required per genus")
    columns, unused_residual = _stage_cycle_integrals(
        ctx, (curve, forms, quadrature_order, baker_basis))
    periods = ctx.matrix([
        [columns[column][row] for column in range(2 * genus)]
        for row in range(genus)])
    a_periods = periods[:, :genus]
    raw_tau = a_periods ** -1 * periods[:, genus:]
    tau = (raw_tau + raw_tau.T) / 2
    polygon = _stage_canonical_polygon(ctx, curve)
    branch_values = (_stage_branch_locus(ctx, curve)[0]
                     if quadrature_order == "geometry" else None)
    return _canonical_polygon_riemann_constant(
        ctx, curve, polygon, forms, a_periods, tau,
        quadrature_order, branch_values=branch_values)


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
