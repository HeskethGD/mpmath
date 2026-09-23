"""Input normalization and Jacobian-level curve operations."""

from ._context import _curve_cache_state
from ._records import CurvePlace, _CurveChartTail, _PlaneCurvePlace
from .continuation import (
    _concatenate_plane_curve_continuations,
    _continue_plane_curve_sheets_adaptive, _point_segment_distance,
)
from .integration import (
    _integrate_plane_curve_path, _integrate_plane_curve_path_iterated,
)
from .monodromy import (
    _concatenate_continuation_sequence, _shortest_monodromy_word,
)
from .polynomial import (
    _evaluate_plane_derivative, _evaluate_plane_polynomial,
    _ordered_plane_curve_sheets, _prepare_plane_curve,
)

def _normalise_algebraic_curve_input(ctx, curve):
    """Return ``(prepared_curve, hyperelliptic_coefficients_or_none)``."""
    if hasattr(curve, "items"):
        return _prepare_plane_curve(ctx, curve), None
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
        return _prepare_plane_curve(ctx, terms), None
    coefficients = tuple(ctx.convert(value) for value in values)
    terms = {(0, 2): ctx.one}
    terms.update({
        (degree, 0): -coefficient
        for degree, coefficient in enumerate(coefficients)
        if coefficient
    })
    return _prepare_plane_curve(ctx, terms), coefficients


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


def _canonical_polygon_riemann_constant(
        ctx, curve, numerical_polygon, differentials,
        a_periods, tau, quadrature_order):
    """Evaluate the vector of Riemann constants from a canonical polygon.

    This is equation (81) in Deconinck--Patterson, *Computing with plane
    algebraic curves and Riemann surfaces: the algorithms of the Maple
    package "Algcurves"*, Lecture Notes in Mathematics 2013 (2011), and
    equation (6.1) in Patterson, *Algebraic Solitons* (2007).  Their contour
    orientation is converted to the ribbon-boundary orientation retained by
    this module.  The resulting value is mpmath's additive theta shift,
    characterized by ``theta(A(D) + K) = 0``.  The input continuations are
    certified based ``a``-loops of a canonical polygon; theta functions are
    not used to construct or select the result.
    """
    differentials = tuple(differentials)
    genus = len(differentials)
    if (len(numerical_polygon.a_continuations) != genus
            or tau.rows != genus or tau.cols != genus):
        raise ValueError(
            "canonical polygon, differentials and periods disagree")
    normalised = _normalised_differentials(
        ctx, differentials, a_periods)
    cycle_integrals = tuple(
        _integrate_plane_curve_path_iterated(
            ctx, curve, continuation, normalised,
            sheet=numerical_polygon.polygon.root[1],
            quadrature_order=quadrature_order)
        for continuation in numerical_polygon.a_continuations)
    value = ctx.matrix(genus, 1)
    for row in range(genus):
        correction = ctx.fsum(
            cycle_integrals[cycle].iterated[row][cycle]
            for cycle in range(genus) if cycle != row)
        value[row] = (1 + tau[row, row]) / 2 + correction
    return value, cycle_integrals, normalised


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


def _finite_base_abel_value(
        ctx, curve, monodromy, branch_points, base_place,
        normalised_differentials, quadrature_order):
    path = _guarded_open_path(
        ctx, monodromy.base_point, base_place.x, branch_points)
    continuation = _continue_plane_curve_sheets_adaptive(
        ctx, curve, path, initial_sheets=monodromy.base_sheets,
        max_refinements=20)
    target_sheet = min(
        range(curve.y_degree),
        key=lambda index: abs(
            continuation.fibres[-1][index] - base_place.y))
    match_error = abs(
        continuation.fibres[-1][target_sheet] - base_place.y)
    scale = max(ctx.one, abs(base_place.y))
    if match_error > 100 * ctx.sqrt(ctx.eps) * scale:
        raise ValueError("base_place could not be matched to a sheet")
    generators = tuple(monodromy.ribbon_generators)
    word = _shortest_monodromy_word(
        tuple(generator.permutation for generator in generators),
        0, target_sheet)
    connector = _concatenate_continuation_sequence(
        ctx, tuple(generators[index].continuation for index in word))
    if connector is not None:
        continuation = _concatenate_plane_curve_continuations(
            ctx, connector, continuation)
    integral = _integrate_plane_curve_path(
        ctx, curve, continuation, normalised_differentials, sheet=0,
        quadrature_order=quadrature_order)
    return ctx.matrix(integral.values)


def _jacobian_lattice_matrix(ctx, tau):
    genus = tau.rows
    return ctx.matrix([
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


def _reduce_jacobian_point(ctx, value, tau, lattice_inverse):
    genus = tau.rows
    coordinates = lattice_inverse * ctx.matrix(
        [ctx.re(entry) for entry in value]
        + [ctx.im(entry) for entry in value])
    shift = tuple(ctx.nint(entry) for entry in coordinates)
    return value - ctx.matrix([
        shift[row] + ctx.fsum(
            tau[row, column] * shift[genus + column]
            for column in range(genus))
        for row in range(genus)
    ])


def _theta_divisor_samples(
        ctx, curve, monodromy, branch_points, normalised_differentials,
        genus, quadrature_order):
    """Construct deterministic degree ``g-1`` Abel images."""
    if genus == 1:
        return (ctx.matrix([ctx.zero]),)
    count = 2 * genus + 1
    center = monodromy.center
    radius = max(abs(point - center) for point in branch_points)
    point_values = []
    for index in range(count):
        angle = 2 * ctx.pi * (index + ctx.mpf("0.173")) / count
        x = center + ctx.mpf("0.35") * radius * ctx.exp(ctx.j * angle)
        sheets = _ordered_plane_curve_sheets(ctx, curve, x)
        place = _PlaneCurvePlace(x, sheets[index % curve.y_degree])
        point_values.append(_finite_base_abel_value(
            ctx, curve, monodromy, branch_points, place,
            normalised_differentials, quadrature_order))
    samples = []
    for index in range(2 * genus):
        value = ctx.matrix(genus, 1)
        for offset in range(genus - 1):
            value += point_values[(index + offset) % count]
        samples.append(value)
    return tuple(samples)
