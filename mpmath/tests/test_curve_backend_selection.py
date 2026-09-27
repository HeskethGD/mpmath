"""Focused checks for the experimental per-curve backend choice."""
import pytest
from mpmath import mp
from mpmath.curves import _operations
from mpmath.curves.algebraic_curve import CurvePlace
from mpmath.curves.jacobian import _reduce_jacobian_point, _jacobian_lattice_matrix

TERMS = {(0, 3): 1, (4, 0): -1, (1, 0): 1, (0, 0): -1}


def test_geometric_interface_keeps_one_marking(monkeypatch):
    ctx = mp.clone()
    ctx.dps = 18
    curve = ctx.algebraic_curve(TERMS, _general_backend='geometric')

    def radial_forbidden(*args, **kwargs):
        raise AssertionError('unexpected radial fallback')

    for name in ('_stage_monodromy', '_stage_monodromy_graph',
                 '_stage_cycle_integrals', '_stage_riemann_constant'):
        monkeypatch.setattr(_operations, name, radial_forbidden)
    periods = curve.first_kind_periods()
    assert curve.genus == 3
    for result in (periods, curve.homology, curve.riemann_constant()):
        assert result.marking == 'geometric-polygon'
        assert curve.validate(result).passed
    assert curve.validate(curve.genus_data).passed
    assert ctx.norm(periods.tau-curve.riemann_matrix()) == 0
    original = periods.omega[0, 0]
    periods.omega[0, 0] = 123
    periods = curve.first_kind_periods()
    assert periods.omega[0, 0] == original
    target = curve.fibre(ctx.mpc('.3', '.7'))[0]
    base = curve.fibre(ctx.mpc('.4', '.8'))[-1]
    value = curve.abel_map([target, target], base_place=base)
    assert ctx.norm(value - 2*curve.abel_map(target, base_place=base)) < ctx.mpf('1e-16')
    assert ctx.norm(curve.abel_map([])) == 0
    assert ctx.norm(curve.abel_map(base, base_place=base)) == 0
    reduced = curve.abel_map([target, target], base_place=base, reduce=True)
    assert ctx.norm(reduced-curve.lattice_reduce(value, periods).value) < ctx.mpf('1e-16')
    constant = curve.riemann_constant(base_place=base)
    argument = (2*periods.omega)**-1 * value + constant.value
    argument = _reduce_jacobian_point(
        ctx, argument, periods.tau, _jacobian_lattice_matrix(ctx, periods.tau)**-1)
    assert abs(ctx.rtheta(argument, periods.tau)) < ctx.mpf('1e-12')
    with ctx.workdps(23):
        with pytest.warns(UserWarning, match='context changed'):
            higher = curve.first_kind_periods()
        assert higher.marking == 'geometric-polygon'
        assert curve.homology.marking == higher.marking
        assert ctx.norm(higher.omega-periods.omega) < ctx.mpf('1e-16')
    assert curve.first_kind_periods().omega[0, 0] == original


def test_geometric_unsupported_operations_are_explicit():
    ctx = mp.clone()
    curve = ctx.algebraic_curve(TERMS, _general_backend='geometric')
    forms = (lambda x, y: 1,)
    for operation in (
        lambda: curve.first_kind_periods(forms),
        lambda: curve.riemann_matrix(forms),
        lambda: curve.riemann_constant(forms),
        lambda: curve.abel_map([], forms),
        lambda: curve.second_kind_periods(),
        lambda: curve.second_kind_abel_map([]),
    ):
        with pytest.raises(NotImplementedError, match='automatic first-kind'):
            operation()
    chart_place = CurvePlace(0, 1, object())
    for operation in (
        lambda: curve.abel_map(chart_place),
        lambda: curve.abel_map([], base_place=chart_place),
        lambda: curve.riemann_constant(base_place=chart_place),
    ):
        with pytest.raises(ValueError, match='invalid chart description'):
            operation()
    with pytest.raises(ValueError, match='_general_backend'):
        ctx.algebraic_curve(TERMS, _general_backend='unknown')


def test_hyperelliptic_dispatch_ignores_general_backend(monkeypatch):
    ctx = mp.clone()
    ctx.dps = 18

    def geometric_forbidden(*args, **kwargs):
        raise AssertionError('hyperelliptic curve entered geometric backend')

    for name in ('_stage_geometric_periods', '_stage_geometric_polygon',
                 '_stage_geometric_riemann_constant'):
        monkeypatch.setattr(_operations, name, geometric_forbidden)
    curve = ctx.algebraic_curve({(0, 2): 1, (3, 0): -1, (1, 0): 1},
                               _general_backend='geometric')
    for result in (curve.first_kind_periods(), curve.second_kind_periods(),
                   curve.homology, curve.riemann_constant()):
        assert result.engine == 'hyperelliptic'
        assert result.marking == 'baker'
    default = ctx.algebraic_curve((0, -1, 0, 1))
    point = (ctx.mpf(2), ctx.sqrt(6))
    assert ctx.norm(curve.abel_map(point)-default.abel_map(point)) == 0
    assert curve.first_kind_periods((lambda x, y: 1/y,)).marking == 'canonical-polygon'


def test_radial_default_and_monodromy_diagnostic():
    ctx = mp.clone()
    ctx.dps = 18
    terms = {(0, 3): 1, (3, 0): 1, (0, 0): -1}
    default = ctx.algebraic_curve(terms)
    radial = ctx.algebraic_curve(terms, _general_backend='radial')
    geometric = ctx.algebraic_curve(terms, _general_backend='geometric')
    assert default.first_kind_periods().marking == 'canonical-polygon'
    assert ctx.norm(default.riemann_matrix()-radial.riemann_matrix()) == 0
    assert geometric.first_kind_periods().marking == 'geometric-polygon'
    assert geometric.monodromy == radial.monodromy
    assert geometric.validate(geometric.monodromy).passed


def test_geometric_chart_endpoints_cutoffs_and_theta():
    ctx = mp.clone()
    ctx.dps = 18
    curve = ctx.algebraic_curve({(0, 3): 1, (4, 0): -1, (0, 0): 1},
                               _general_backend='geometric')
    infinity = curve.monomial_chart(-3, -4)
    branch = curve.chart(
        {(0, 3): 1, (0, 0): -4, (3, 0): -6, (6, 0): -4, (9, 0): -1},
        lambda t, w: (1+t**3, t*w, 3*t**2))
    periods = curve.first_kind_periods()
    inverse = (2*periods.omega)**-1
    tau = periods.tau
    lattice_inverse = _jacobian_lattice_matrix(ctx, tau)**-1
    places = []
    for chart, seed in ((infinity, ctx.one), (branch, ctx.root(4, 3))):
        outer = curve.chart_place(chart, seed, ctx.mpf('.3'))
        inner = curve.chart_place(chart, seed, ctx.mpf('.25'))
        first = curve.abel_map(outer)
        second = curve.abel_map(inner)
        # Independently chosen open paths can differ by a period.
        difference = curve.lattice_reduce(first-second, periods).value
        assert ctx.norm(difference) < ctx.mpf('1e-14')
        local = curve.chart_integral(
            chart, periods.differentials, (0, ctx.mpf('.3')), seed)
        junction = curve.abel_map((outer.x, outer.y))
        assert ctx.norm(first-junction+ctx.matrix(local.values)) < ctx.mpf('1e-14')
        places.append(outer)
    constant = curve.riemann_constant()
    # g=3: the divisor consists of one ramification point and infinity.
    argument = inverse*curve.abel_map(places) + constant.value
    reduced = _reduce_jacobian_point(ctx, argument, tau, lattice_inverse)
    assert abs(ctx.rtheta(reduced, tau)) < ctx.mpf('1e-11')
    shifted = curve.riemann_constant(base_place=places[0])
    argument = inverse*curve.abel_map(places, base_place=places[0])+shifted.value
    reduced = _reduce_jacobian_point(ctx, argument, tau, lattice_inverse)
    assert abs(ctx.rtheta(reduced, tau)) < ctx.mpf('1e-11')
    assert ctx.norm(curve.abel_map(places[0], base_place=places[0])) == 0
    other = ctx.algebraic_curve(TERMS, _general_backend='geometric')
    with pytest.raises(ValueError, match='different curve or precision'):
        other.abel_map(places[0])
    with ctx.workdps(23):
        with pytest.warns(UserWarning, match='context changed'):
            with pytest.raises(ValueError, match='different curve or precision'):
                curve.abel_map(places[0])
    assert ctx.dps == 18
