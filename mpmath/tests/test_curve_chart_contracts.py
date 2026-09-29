"""Chart input failures and exact integrals through the public curve API."""

import pytest

from mpmath import CurveChart, mp
from mpmath.curves.jacobian import _normalize_curve_endpoint
from mpmath.curves.polynomial import _prepare_plane_curve


TERMS = {(0, 2): 1, (1, 0): -1}


@pytest.mark.parametrize("specification,message", [
    (None, "sparse plane terms"),
    ((), "must not be empty"),
    ((0, -1, 1), "sparse plane terms"),
    (((0, 2, "invalid"),), "coefficients must be numbers"),
])
def test_chart_rejects_invalid_specification(specification, message):
    curve = mp.algebraic_curve(TERMS)
    with pytest.raises(ValueError, match=message):
        curve.chart(specification, lambda t, w: (t, w, 1))


def test_chart_requires_a_callable_coordinate_map():
    curve = mp.algebraic_curve(TERMS)
    with pytest.raises(ValueError, match="coordinate_map must be callable"):
        curve.chart(TERMS, None)


@pytest.mark.parametrize("kind,message", [
    ("not_chart", "must be a CurveChart"),
    ("ownership", "invalid ownership data"),
    ("coordinate_map", "coordinate_map must be callable"),
])
def test_chart_rejects_invalid_records(kind, message):
    curve = mp.algebraic_curve(TERMS)
    chart = curve.monomial_chart(1, 0)
    invalid = {
        "not_chart": None,
        "ownership": CurveChart(None, chart.curve, chart.coordinate_map),
        "coordinate_map": chart._replace(coordinate_map=None),
    }[kind]
    with pytest.raises(ValueError, match=message):
        curve.chart_fibre(invalid, 1)


def test_chart_fibre_rejects_nonfinite_and_unresolved_places():
    curve = mp.algebraic_curve(TERMS)
    chart = curve.monomial_chart(1, 0)
    with pytest.raises(ValueError, match="t must be finite"):
        curve.chart_fibre(chart, mp.inf)
    with pytest.raises(ValueError, match="does not separate"):
        curve.chart_fibre(chart, 0)


@pytest.mark.parametrize("cutoff", [0, mp.inf, mp.nan])
def test_chart_place_requires_a_finite_nonzero_cutoff(cutoff):
    curve = mp.algebraic_curve(TERMS)
    chart = curve.monomial_chart(2, 1)
    with pytest.raises(ValueError, match="finite and nonzero"):
        curve.chart_place(chart, 1, cutoff)


def test_chart_rejects_nonfinite_coordinate_maps():
    curve = mp.algebraic_curve(TERMS)
    chart = curve.chart({(0, 1): 1, (0, 0): -1},
                        lambda t, w: (mp.inf, w, 1))
    with pytest.raises(ValueError, match="must be finite away from the place"):
        curve.chart_place(chart, 1, mp.mpf('.1'))


@pytest.mark.parametrize("path,message", [
    (None, "sequence of chart values"),
    ((), "at least one point"),
    ((0, mp.inf), "values must be finite"),
])
def test_chart_integral_rejects_invalid_paths(path, message):
    curve = mp.algebraic_curve(TERMS)
    chart = curve.monomial_chart(2, 1)
    with pytest.raises(ValueError, match=message):
        curve.chart_integral(chart, lambda x, y: 1, path, 1)


def test_composed_chart_integrals_match_exact_primitives_and_reverse():
    ctx = mp.clone()
    ctx.dps = 25
    curve = ctx.algebraic_curve(TERMS)
    # x=t^2, y=t on the positive branch; composing t=s^2 gives x=s^4.
    chart = curve.monomial_chart(2, 1)
    composed = curve.monomial_chart(2, 0, source=chart)
    assert composed.coordinate_map(ctx.mpf(2), ctx.one) == (16, 4, 32)
    forms = (lambda x, y: 1, lambda x, y: 2*x)
    forward = curve.chart_integral(composed, forms, (1, 2), 1)
    reverse = curve.chart_integral(composed, forms, (2, 1), 1)
    assert ctx.almosteq(forward.values[0], 15)
    assert ctx.almosteq(forward.values[1], 255)
    assert all(ctx.almosteq(a, -b) for a, b in zip(forward.values, reverse.values))
    scalar = curve.chart_integral(composed, forms[0], (1, 2), 1)
    assert ctx.almosteq(scalar.values, forward.values[0])
    assert scalar.max_sheet_residual < 100*ctx.eps
    assert scalar.segments > 0


def test_chart_place_cannot_cross_precision_contexts():
    ctx = mp.clone()
    ctx.dps = 18
    curve = ctx.algebraic_curve({(0, 2): 1, (3, 0): -1, (1, 0): 1})
    chart = curve.monomial_chart(-2, -3)
    place = curve.chart_place(chart, 1, ctx.mpf('.1'))
    prepared = _prepare_plane_curve(ctx, curve.specification)
    with ctx.workdps(25):
        with pytest.raises(ValueError, match="different curve or precision"):
            _normalize_curve_endpoint(ctx, prepared, place, "target")
