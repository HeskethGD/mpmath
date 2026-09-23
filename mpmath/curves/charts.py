"""Local-chart construction and integration for algebraic curves."""

from ._context import _curve_cache_state
from ._records import CurveChart, CurveIntegral, CurvePlace, _CurveChartTail, _PlaneCurve
from ._stages import _curve_differential_sequence
from .continuation import _continue_plane_curve_branch
from .integration import (
    _integrate_plane_curve_branch, _pullback_plane_curve_differentials,
)
from .jacobian import _normalise_algebraic_curve_input
from .polynomial import (
    _blow_up_plane_curve_y, _evaluate_plane_polynomial,
    _finite_plane_curve_sheets, _monomial_plane_curve_chart,
    _prepare_plane_curve, _reciprocal_y_plane_curve,
)

def _identity_chart_coordinates(t, w):
    """Return the identity coordinate map of an affine curve."""
    return t, w, 1


def _prepare_chart_curve(ctx, curve):
    """Prepare a general ``(t, w)`` curve from sparse input or terms.

    Ascending coefficient sequences are deliberately rejected: unlike the
    public curve input they cannot express a general chart curve.
    """
    if hasattr(curve, "items"):
        return _prepare_plane_curve(ctx, curve)
    try:
        values = tuple(curve)
    except TypeError:
        raise ValueError("chart curves must be sparse plane terms")
    if not values:
        raise ValueError("chart curves must not be empty")
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
        return _prepare_plane_curve(ctx, terms)
    raise ValueError("chart curves must be sparse plane terms")


def _curve_chart_source(ctx, source):
    """Return the chart of a raw curve input or an existing chart."""
    if isinstance(source, CurveChart):
        return _validate_chart(ctx, source)
    prepared, unused_hyperelliptic = _normalise_algebraic_curve_input(
        ctx, source)
    return CurveChart(
        (prepared, _curve_cache_state(ctx)), prepared,
        _identity_chart_coordinates)


def _validate_chart(ctx, chart, ambient_curve=None):
    """Return a validated chart record."""
    if not isinstance(chart, CurveChart):
        raise ValueError("chart must be a CurveChart")
    if (not isinstance(chart.curve_key, tuple)
            or len(chart.curve_key) != 2
            or not isinstance(chart.curve, _PlaneCurve)):
        raise ValueError("chart carries invalid ownership data")
    if not callable(chart.coordinate_map):
        raise ValueError("chart coordinate_map must be callable")
    expected_state = _curve_cache_state(ctx)
    if chart.curve_key[1] != expected_state:
        raise ValueError(
            "chart was constructed for a different working precision")
    if (ambient_curve is not None
            and chart.curve_key != (ambient_curve, expected_state)):
        raise ValueError("chart was constructed for a different curve")
    return chart


def _validated_chart_coordinate_map(ctx, chart):
    """Return a coordinate map checking the ambient curve equation."""
    ambient_curve = chart.curve_key[0]
    coordinate_map = chart.coordinate_map

    def validated(t, w):
        x, y, dx_dt = coordinate_map(t, w)
        x = ctx.convert(x)
        y = ctx.convert(y)
        dx_dt = ctx.convert(dx_dt)
        if not all(ctx.isfinite(value) for value in (x, y, dx_dt)):
            raise ValueError(
                "chart coordinate map must be finite away from the place")
        scale = max(ctx.one, ctx.fsum(
            abs(coefficient * x ** x_power * y ** y_power)
            for x_power, y_power, coefficient in ambient_curve.terms))
        if abs(_evaluate_plane_polynomial(
                ctx, ambient_curve, x, y)) > (
                100 * ctx.sqrt(ctx.eps) * scale):
            raise ValueError("chart does not parametrize its ambient curve")
        return x, y, dx_dt

    return validated


# Public chart functions
# ---------------------


def chart(ctx, curve, chart_curve, coordinate_map):
    r"""Return a user-supplied local chart of a plane algebraic curve.

    ``curve`` is the ambient curve specification. ``chart_curve`` gives the local
    curve as a sparse mapping from
    ``(t_power, w_power)`` pairs to coefficients, or a sequence of
    ``(t_power, w_power, coefficient)`` terms.  ``coordinate_map(t, w)``
    must return the ambient triple ``(x, y, dx/dt)``, where ``x`` depends
    on ``t`` alone.  The returned ``CurveChart`` is bound to the ambient
    curve and working precision, and is accepted by the other chart
    methods and by :meth:`AlgebraicCurve.chart_place`.
    """
    ambient, unused_hyperelliptic = _normalise_algebraic_curve_input(
        ctx, curve)
    prepared = _prepare_chart_curve(ctx, chart_curve)
    if not callable(coordinate_map):
        raise ValueError("coordinate_map must be callable")
    return CurveChart(
        (ambient, _curve_cache_state(ctx)), prepared, coordinate_map)


def monomial_chart(ctx, source, x_power, y_power):
    r"""Return the monomial chart ``x = t**x_power, y = t**y_power*w``.

    ``source`` is the curve itself, or another ``CurveChart`` to compose
    with.  Negative powers describe places above infinity.  Repeated
    factors are cleared so the chart curve is a polynomial in ``t`` and
    ``w``; the chart does not claim to normalize a singular chart.
    """
    base = _curve_chart_source(ctx, source)
    curve = _monomial_plane_curve_chart(
        ctx, base.curve, x_power, y_power)
    base_map = base.coordinate_map

    def coordinate_map(t, w, base_map=base_map, x_power=x_power,
                       y_power=y_power):
        x, y, dx_dt = base_map(t ** x_power, t ** y_power * w)
        return x, y, dx_dt * x_power * t ** (x_power - 1)

    return CurveChart(base.curve_key, curve, coordinate_map)


def _curve_chart_reciprocal_y(ctx, source):
    """Return the private reciprocal ``v = 1/w`` chart transform.

    The reciprocal chart reparametrizes the base ``w`` coordinate as
    ``v = 1/w``, so its ambient point at ``(t, v)`` is the base chart's
    point at ``(t, 1/v)``.  For an affine base curve this is the usual
    ``y = 1/v`` chart.
    """
    base = _curve_chart_source(ctx, source)
    curve = _reciprocal_y_plane_curve(ctx, base.curve)
    base_map = base.coordinate_map

    def coordinate_map(t, v, base_map=base_map):
        return base_map(t, 1 / v)

    return CurveChart(base.curve_key, curve, coordinate_map)


def _curve_chart_blow_up(ctx, source, center, power=1):
    """Return the private blow-up chart ``w = center + t**power*u``.

    The substitution separates branches of ``source`` meeting above a
    common ``w`` value at ``t = 0``.  ``source`` may be a curve or another
    chart, and the returned chart composes the coordinate maps.
    """
    base = _curve_chart_source(ctx, source)
    curve = _blow_up_plane_curve_y(
        ctx, base.curve, center, y_power=power)
    center = ctx.convert(center)
    base_map = base.coordinate_map

    def coordinate_map(t, u, base_map=base_map, center=center,
                        power=power):
        return base_map(t, center + t ** power * u)

    return CurveChart(base.curve_key, curve, coordinate_map)


def chart_fibre(ctx, chart, t):
    r"""Return the ordered fibre of chart ``w`` values over ``t``.

    The values are ordered by real and imaginary part, like
    :meth:`AlgebraicCurve.fibre`.  A fibre whose values do not separate
    indicates that the chart does not resolve the requested place and is
    rejected.
    """
    chart = _validate_chart(ctx, chart)
    t = ctx.convert(t)
    if not ctx.isfinite(t):
        raise ValueError("t must be finite")
    values = _finite_plane_curve_sheets(ctx, chart.curve, t)
    values = tuple(sorted(
        values, key=lambda value: (ctx.re(value), ctx.im(value))))
    scale = max([ctx.one] + [abs(value) for value in values])
    separation = min(
        (abs(left - right)
         for index, left in enumerate(values)
         for right in values[index + 1:]),
        default=ctx.inf)
    if separation <= 100 * ctx.sqrt(ctx.eps) * scale:
        raise ValueError(
            "the chart fibre is not simple over t; the chart does not "
            "separate the requested place")
    return values


def chart_place(ctx, curve, chart, seed, cutoff):
    r"""Return the chart-backed place reached by a local branch.

    ``seed`` is the branch value of ``w`` at ``t = 0``, for example from
    :meth:`AlgebraicCurve.chart_fibre`; the branch is continued along the
    straight chart path from ``t = 0`` to ``t = cutoff``.  The returned
    place is represented by its finite affine cutoff point together with a
    chart tail describing the local branch, and is bound to ``curve`` and
    the working precision.  The chart must parametrize ``curve``: the
    cutoff point is checked to lie on the curve.

    >>> from mpmath import algebraic_curve, mp
    >>> mp.dps = 15
    >>> curve = algebraic_curve({(0, 2): 1, (1, 0): 1, (3, 0): -1})
    >>> chart = curve.monomial_chart(-2, -3)
    >>> [mp.nstr(value, 3) for value in curve.chart_fibre(chart, 0)]
    ['(-1.0 + 0.0j)', '(1.0 + 0.0j)']
    >>> place = curve.chart_place(chart, 1, mp.mpf("0.05"))
    >>> mp.nstr(place.x, 6)
    '400.0'
    """
    prepared, unused_hyperelliptic = _normalise_algebraic_curve_input(
        ctx, curve)
    chart = _validate_chart(ctx, chart, prepared)
    cutoff = ctx.convert(cutoff)
    if not ctx.isfinite(cutoff) or not cutoff:
        raise ValueError("cutoff must be finite and nonzero")
    branch = _continue_plane_curve_branch(
        ctx, chart.curve, (0, cutoff), seed)
    coordinate_map = _validated_chart_coordinate_map(ctx, chart)
    x, y, unused_dx_dt = coordinate_map(
        branch.path[-1], branch.values[-1])
    for t, w in zip(branch.path[1:], branch.values[1:]):
        coordinate_map(t, w)
    tail = _CurveChartTail(chart, branch)
    return CurvePlace(x, y, tail)


def chart_integral(ctx, chart, differentials, t_path, seed):
    r"""Integrate ambient differentials along a local chart branch.

    ``differentials`` are ambient ``f(x, y)`` callables returning the
    coefficient of ``dx``; they are pulled back through the chart's
    coordinate map, so a single callable gives a scalar and a sequence
    gives one entry per form.  ``t_path`` is a sequence of finite ``t``
    values along which the branch is continued from ``seed`` at
    ``t_path[0]``.  Closed chart loops therefore compute residues of
    pulled-back forms at the place.
    """
    chart = _validate_chart(ctx, chart)
    if callable(differentials):
        single = True
        forms = (differentials,)
    else:
        single = False
        forms = _curve_differential_sequence(
            differentials, "differentials")
    try:
        t_path = tuple(ctx.convert(value) for value in t_path)
    except TypeError:
        raise ValueError("t_path must be a sequence of chart values")
    if not t_path:
        raise ValueError("t_path must contain at least one point")
    if any(not ctx.isfinite(value) for value in t_path):
        raise ValueError("t_path values must be finite")
    branch = _continue_plane_curve_branch(
        ctx, chart.curve, t_path, seed)
    pullbacks = _pullback_plane_curve_differentials(
        forms, _validated_chart_coordinate_map(ctx, chart))
    integral = _integrate_plane_curve_branch(
        ctx, chart.curve, branch, pullbacks)
    values = integral.values[0] if single else integral.values
    return CurveIntegral(
        values, integral.max_sheet_residual, integral.segments)
