"""Input normalization and Jacobian-level curve operations."""

from ._context import _curve_cache_state
from ._records import (
    CurvePlace,
    _ClassifiedCurve,
    _CurveChartTail,
    _HyperellipticModel,
    _PlaneCurvePlace,
)
from .continuation import _continue_plane_curve_sheets_adaptive, _point_segment_distance
from .homology import _geometric_sheet_connector
from .integration import _integrate_plane_curve_path
from .polynomial import (
    _evaluate_plane_derivative,
    _evaluate_plane_polynomial,
    _polynomial_multiply,
    _polynomial_trim,
    _prepare_plane_curve,
)


def _classify_hyperelliptic_model(ctx, curve):
    """Recognize a polynomial quadratic cover with constant leading term.

    For ``A*y**2 + B(x)*y + C(x) = 0`` with constant nonzero ``A``, use
    ``z = y + B/(2*A)`` and ``z**2 = B**2/(4*A**2) - C/A``.  The returned
    descriptor contains the ascending coefficients of the right-hand side
    and of the shift added to the caller's ``y`` coordinate.

    This is deliberately a cheap structural classification.  Root
    separation and smoothness are checked lazily by the specialized engine.
    Models with a nonconstant coefficient of ``y**2`` remain on the general
    path because completing their square can change affine and infinite
    places birationally.
    """
    if curve.y_degree != 2:
        return None
    quadratic = [term for term in curve.terms if term[1] == 2]
    if len(quadratic) != 1 or quadratic[0][0] != 0:
        return None
    leading = quadratic[0][2]
    linear_degree = max(
        [term[0] for term in curve.terms if term[1] == 1], default=0)
    constant_degree = max(
        [term[0] for term in curve.terms if term[1] == 0], default=0)
    linear = [ctx.zero] * (linear_degree + 1)
    constant = [ctx.zero] * (constant_degree + 1)
    for x_power, y_power, coefficient in curve.terms:
        if y_power == 1:
            linear[x_power] = coefficient
        elif y_power == 0:
            constant[x_power] = coefficient
    shift = _polynomial_trim(
        ctx, tuple(value / (2 * leading) for value in linear))
    square = _polynomial_multiply(ctx, shift, shift)
    size = max(len(square), len(constant))
    polynomial = tuple(
        (square[index] if index < len(square) else ctx.zero)
        - (constant[index] / leading
           if index < len(constant) else ctx.zero)
        for index in range(size))
    polynomial = _polynomial_trim(ctx, polynomial)
    # The current specialized engine starts in genus one.  Lower-degree
    # quadratic covers continue through the general machinery.
    if len(polynomial) < 4:
        return None
    return _HyperellipticModel(polynomial, shift)


def _normalise_algebraic_curve_input(ctx, curve):
    """Return a prepared curve and its optional specialized model."""
    if isinstance(curve, _ClassifiedCurve):
        return curve.curve, curve.hyperelliptic
    if hasattr(curve, "items"):
        prepared = _prepare_plane_curve(ctx, curve)
    else:
        try:
            values = tuple(curve)
        except TypeError:
            raise ValueError("curve must be coefficients or sparse plane terms")
        if not values:
            raise ValueError("curve must not be empty")
        if all(isinstance(term, (tuple, list)) and len(term) == 3
               for term in values):
            terms = {}
            for x_power, y_power, coefficient in values:
                try:
                    coefficient = ctx.convert(coefficient)
                except (TypeError, ValueError):
                    raise ValueError("polynomial coefficients must be numbers")
                key = (x_power, y_power)
                terms[key] = terms.get(key, ctx.zero) + coefficient
            prepared = _prepare_plane_curve(ctx, terms)
        else:
            coefficients = tuple(ctx.convert(value) for value in values)
            terms = {(0, 2): ctx.one}
            terms.update({
                (degree, 0): -coefficient
                for degree, coefficient in enumerate(coefficients)
                if coefficient
            })
            prepared = _prepare_plane_curve(ctx, terms)
    return prepared, _classify_hyperelliptic_model(ctx, prepared)


def _evaluate_ascending_polynomial(ctx, coefficients, value):
    result = ctx.zero
    for coefficient in reversed(coefficients):
        result = result * value + coefficient
    return result


def _to_hyperelliptic_points(ctx, model, points):
    """Map affine points from the user's equation to ``z**2 = P(x)``."""
    return tuple(
        (x, y + _evaluate_ascending_polynomial(ctx, model.y_shift, x))
        for x, y in points)


def _period_matrix_from_columns(ctx, columns, start, count, genus):
    return ctx.matrix([
        [columns[column][start + row] for column in range(2 * genus)]
        for row in range(count)
    ])


def _normalised_differentials(ctx, differentials, a_periods):
    inverse = a_periods ** -1
    result = []
    for row in range(len(differentials)):
        coefficients = tuple(
            inverse[row, column]
            for column in range(len(differentials)))

        def differential(x, y, coefficients=coefficients):
            return ctx.fsum(
                coefficient * form(x, y)
                for coefficient, form in zip(coefficients, differentials))

        result.append(differential)
    return tuple(result)


def _riemann_constant_from_iterated_cycles(ctx, tau, cycle_integrals):
    """Apply the canonical-polygon formula to normalized based a-loops."""
    genus = tau.rows
    if tau.cols != genus or len(cycle_integrals) != genus:
        raise ValueError("canonical cycles and period matrix disagree")
    value = ctx.matrix(genus, 1)
    for row in range(genus):
        correction = ctx.fsum(
            cycle_integrals[cycle].iterated[row][cycle]
            for cycle in range(genus) if cycle != row)
        value[row] = (1 + tau[row, row]) / 2 + correction
    return value


def _jacobian_characteristic(ctx, value, tau):
    """Express a Jacobian point as the literal pair ``(a, b)``."""
    genus = tau.rows
    lattice = ctx.matrix([
        [ctx.re(1 if row == column else 0)
         for column in range(genus)]
        + [ctx.re(tau[row, column]) for column in range(genus)]
        for row in range(genus)
    ] + [
        [ctx.im(1 if row == column else 0)
         for column in range(genus)]
        + [ctx.im(tau[row, column]) for column in range(genus)]
        for row in range(genus)
    ])
    target = ctx.matrix(
        [ctx.re(entry) for entry in value]
        + [ctx.im(entry) for entry in value])
    coordinates = lattice ** -1 * target
    reduced = []
    snap_tolerance = ctx.power(ctx.eps, ctx.mpf("0.25"))
    for coordinate in coordinates:
        coordinate -= ctx.floor(coordinate)
        half = ctx.nint(2 * coordinate) / 2
        if abs(coordinate - half) <= snap_tolerance:
            coordinate = half % 1
        reduced.append(+coordinate)
    b = tuple(reduced[:genus])
    a = tuple(reduced[genus:])
    return a, b


def _normalise_curve_place(ctx, curve, place, name="place"):
    """Return a validated regular finite place on a prepared curve."""
    if isinstance(place, CurvePlace):
        place = (place.x, place.y)
    try:
        x, y = place
    except (TypeError, ValueError):
        raise ValueError(name + " must be a finite (x, y) pair")
    x = ctx.convert(x)
    y = ctx.convert(y)
    if not ctx.isfinite(x) or not ctx.isfinite(y):
        raise ValueError(name + " must be a finite (x, y) pair")
    scale = max(ctx.one, ctx.fsum(
        abs(coefficient * x ** x_power * y ** y_power)
        for x_power, y_power, coefficient in curve.terms))
    if abs(_evaluate_plane_polynomial(ctx, curve, x, y)) > (
            100 * ctx.sqrt(ctx.eps) * scale):
        raise ValueError(name + " must lie on the curve")
    if abs(_evaluate_plane_derivative(ctx, curve, x, y, "y")) <= (
            100 * ctx.sqrt(ctx.eps)):
        raise ValueError(name + " must be regular for the x projection")
    return _PlaneCurvePlace(x, y)


def _normalise_curve_endpoint(ctx, curve, place, name):
    """Return ``(junction, tail, public_place)`` for any place input.

    A chart-backed place is checked for curve and numerical-context
    ownership and reduced to its affine junction point together with the
    local chart tail reaching the place itself.
    """
    if isinstance(place, CurvePlace) and place.chart is not None:
        tail = place.chart
        if not isinstance(tail, _CurveChartTail):
            raise ValueError(name + " carries an invalid chart description")
        if tail.chart.curve_key != (curve, _curve_cache_state(ctx)):
            raise ValueError(
                name + " was constructed for a different curve or precision")
        junction = _normalise_curve_place(
            ctx, curve, (place.x, place.y), name)
        return junction, tail, CurvePlace(junction.x, junction.y, tail)
    junction = _normalise_curve_place(ctx, curve, place, name)
    return junction, None, CurvePlace(junction.x, junction.y)


def _guarded_open_path(ctx, start, end, branch_points):
    """Choose a simple deterministic polyline avoiding critical values."""
    scale = max(
        [ctx.one, abs(start), abs(end)]
        + [abs(point) for point in branch_points])
    midpoint = (start + end) / 2
    candidates = [(start, end)]
    for direction in (ctx.one, -ctx.one, ctx.j, -ctx.j,
                      1 + ctx.j, 1 - ctx.j, -1 + ctx.j, -1 - ctx.j):
        candidates.append((start, midpoint + direction * scale, end))

    def clearance(path):
        return min(
            _point_segment_distance(ctx, point, left, right)
            for point in branch_points
            for left, right in zip(path, path[1:]))

    return max(candidates, key=clearance)


def _finite_geometric_abel_value(ctx, curve, data, place, differentials,
                                 quadrature_cache=None, edge_cache=None,
                                 check_convergence=False):
    """Integrate from the geometric root to a validated regular finite place.

    The open lift starts with all sheets in the root fibre. A tree connector
    first reaches the sheet whose open lift ends at the requested place.
    Integrals are additive here, so reversed connector edges contribute with
    a minus sign in their original sheet labelling. Supplied caches must
    belong to this same cover, differential basis and numerical state.
    """
    cover, graph = data.cover, data.graph
    root = data.polygon.polygon.root
    start = cover.geometry.vertices[root[0]]
    branch_points = cover.geometry.branch_values
    path = ((start, start) if place.x == start else
            _guarded_open_path(ctx, start, place.x, branch_points))
    continuation = _continue_plane_curve_sheets_adaptive(
        ctx, curve, path, initial_sheets=cover.fibres[root[0]],
        max_refinements=20)
    target_sheet = min(range(curve.y_degree), key=lambda sheet:
                       abs(continuation.fibres[-1][sheet] - place.y))
    if abs(continuation.fibres[-1][target_sheet] - place.y) > (
            100 * ctx.sqrt(ctx.eps) * max(ctx.one, abs(place.y))):
        raise ValueError("place could not be matched to a geometric sheet")
    connector = _geometric_sheet_connector(graph, root, target_sheet)
    rules = {} if quadrature_cache is None else quadrature_cache
    edges = {} if edge_cache is None else edge_cache

    def integrate(lift, sheet):
        return _integrate_plane_curve_path(
            ctx, curve, lift, differentials, sheet=sheet,
            quadrature_order="geometry", branch_values=branch_points,
            quadrature_cache=rules, check_convergence=check_convergence).values

    pieces = [integrate(continuation, target_sheet)]
    for index, orientation in connector:
        edge = graph.edges[index]
        if index not in edges:
            edges[index] = integrate(cover.continuations[edge.base_edge], edge.sheet)
        pieces.append(tuple(orientation * value for value in edges[index]))
    return tuple(ctx.fsum(piece[i] for piece in pieces)
                 for i in range(len(differentials)))
