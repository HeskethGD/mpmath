"""Targeted tests of the private geometric backend and its numerical layers."""

from mpmath.curves import _operations

import pytest

from mpmath import mp
import mpmath.curves.continuation as continuation
from mpmath.curves.geometry import _voronoi_plane_graph
from mpmath.curves.polynomial import _prepare_plane_curve
from mpmath.curves.quadrature import _legendre_edge_rule
from mpmath.curves._stages import (
    _stage_geometric_periods,
)


@pytest.mark.parametrize("points", [
    (-1-1j, 1-1j, 1+1j, -1+1j), (-2, -1, 1, 2),
    (-2-1j, 2, 1+3j, -1+2j), (0, 0.001, 1+1j, -1+2j),
])
def test_geometric_cells_and_affine_covariance(points):
    with mp.workdps(30):
        points = tuple(mp.convert(z) for z in points)
        graph = _voronoi_plane_graph(mp, points)
        moved = _voronoi_plane_graph(mp, [1000 + 7*z for z in points])
        assert graph.edges == moved.edges
        assert graph.cells == moved.cells
        assert len(graph.vertices) - len(graph.edges) + len(graph.cells) == 1
        for site, cell in zip(graph.branch_values, graph.cells):
            for a, b in zip(cell, cell[1:] + cell[:1]):
                left, right = graph.vertices[a], graph.vertices[b]
                assert mp.im(mp.conj(right-left)*(site-left)) >= -mp.mpf('1e-25')
        for a, b in zip(graph.vertices, moved.vertices):
            assert abs(1000 + 7*a - b) < mp.mpf('1e-25')


def test_geometric_unresolved_sites_fail_and_context_is_restored():
    with mp.workdps(20):
        with pytest.raises(mp.NoConvergence):
            _voronoi_plane_graph(mp, (0, mp.mpf('1e-12'), 1))
        curve = _prepare_plane_curve(mp, {(0, 2): 1, (1, 0): -1})
        precision = mp.prec
        with pytest.raises(ValueError, match="genus-zero"):
            _stage_geometric_periods(mp, curve)
        assert mp.prec == precision


@pytest.mark.parametrize("order", (3, 8, 24, 64))
def test_edge_rules_integrate_polynomial_moments(order):
    with mp.workdps(30):
        rule = _legendre_edge_rule(mp, order)
        assert all(0 < t < 1 and w > 0 for t, w in rule)
        for degree in (0, 1, order, 2*order-1):
            value = mp.fsum(w*t**degree for t, w in rule)
            assert abs(value - mp.one/(degree+1)) < 100*mp.eps


def test_edge_sampler_random_access_and_local_refinement(monkeypatch):
    with mp.workdps(30):
        curve = _prepare_plane_curve(mp, {(0, 2): 1, (1, 0): -1})
        lift = continuation._continue_plane_curve_sheets_adaptive(mp, curve, (1, 4))
        sampler = continuation._LiftedEdgeSampler(mp, curve, lift)
        original = continuation._newton_plane_curve_sheet
        fail_once = [True]

        def correct(*args, **kwargs):
            result = original(*args, **kwargs)
            if fail_once[0]:
                fail_once[0] = False
                return result[:-1] + (False,)
            return result

        monkeypatch.setattr(continuation, '_newton_plane_curve_sheet', correct)
        for t in map(mp.mpf, ('0.9', '0.1', '0.7', '0', '1')):
            for sheet in range(2):
                x, y, unused_residual = sampler.sample(t, sheet)
                assert abs(y - lift.fibres[0][sheet]*mp.sqrt(x)) < mp.mpf('1e-27')
        assert sampler.refinements == 1
        with mp.extraprec(10):
            with pytest.raises(ValueError, match="precision"):
                sampler.sample(mp.mpf('0.3'), 0)


def test_edge_sampler_unresolved_query_is_bounded(monkeypatch):
    with mp.workdps(20):
        curve = _prepare_plane_curve(mp, {(0, 2): 1, (1, 0): -1})
        lift = continuation._continue_plane_curve_sheets_adaptive(mp, curve, (1, 4))
        sampler = continuation._LiftedEdgeSampler(mp, curve, lift)
        original = continuation._newton_plane_curve_sheet

        def unresolved(*args, **kwargs):
            return original(*args, **kwargs)[:-1] + (False,)

        monkeypatch.setattr(continuation, '_newton_plane_curve_sheet', unresolved)
        with pytest.raises(mp.NoConvergence, match='did not resolve'):
            sampler.sample(mp.mpf('0.3'), 0)
        assert sampler.refinements <= 21


def _matrix(ctx, columns):
    return ctx.matrix([[column[i] for column in columns]
                       for i in range(len(columns[0]))])


def test_geometric_cache_precision_and_clone_context():
    ctx = mp.clone()
    curve = _prepare_plane_curve(ctx, {(0, 2): 1, (3, 0): -1, (1, 0): 1})
    with ctx.workdps(20):
        first = _stage_geometric_periods(ctx, curve)
        assert _stage_geometric_periods(ctx, curve) is first
        assert first.working_precision == ctx.prec + 10
    with ctx.workdps(35):
        second = _stage_geometric_periods(ctx, curve)
        assert second is not first
        assert second.polygon.polygon == first.polygon.polygon
        assert ctx.norm(_matrix(ctx, first.columns)-_matrix(ctx, second.columns)) < ctx.mpf('1e-19')
    with ctx.workdps(20):
        assert _stage_geometric_periods(ctx, curve) is first


def test_geometric_riemann_constant_precision_and_normalization():
    from mpmath.curves._stages import _stage_geometric_riemann_constant

    ctx = mp.clone()
    curve = _prepare_plane_curve(ctx, {(0, 3): 1, (4, 0): -1,
                                       (1, 0): 1, (0, 0): -1})
    results = []
    polygons = []
    for digits in (18, 25):
        with ctx.workdps(digits):
            precision = ctx.prec
            result = _stage_geometric_riemann_constant(ctx, curve)
            assert ctx.prec == precision
            assert _stage_geometric_riemann_constant(ctx, curve) is result
            value, cycles = result
            polygons.append(_stage_geometric_periods(ctx, curve).polygon.polygon)
            assert len(value) == 3
            for j, cycle in enumerate(cycles):
                for i, entry in enumerate(cycle.values):
                    assert abs(entry - (i == j)) < ctx.mpf(10)**(-digits+2)
            results.append(value)
    assert polygons[0] == polygons[1]
    assert ctx.norm(ctx.matrix(results[0])-ctx.matrix(results[1])) < ctx.mpf('1e-16')


@pytest.mark.parametrize('terms,genus', [
    ({(0, 2): 1, (5, 0): -1, (1, 0): 1, (0, 0): -1}, 2),
    ({(0, 3): 1, (4, 0): -1, (1, 0): 1, (0, 0): -1}, 3),
])
def test_geometric_abel_theta_divisor_and_base_change(terms, genus):
    from mpmath.curves._stages import (
        _stage_geometric_riemann_constant,
    )
    from mpmath.curves.polynomial import _ordered_plane_curve_sheets

    with mp.workdps(18):
        curve = _prepare_plane_curve(mp, terms)
        data = _stage_geometric_periods(mp, curve)
        full = _matrix(mp, data.columns)
        inverse_a = full[:, :genus] ** -1
        raw_tau = inverse_a * full[:, genus:]
        tau = (raw_tau + raw_tau.T) / 2
        constant = mp.matrix(_stage_geometric_riemann_constant(mp, curve)[0])
        root = data.polygon.polygon.root
        x = data.cover.geometry.vertices[root[0]]
        base = (x, data.cover.fibres[root[0]][root[1]])
        assert mp.norm(mp.matrix(_geometric_abel_value(mp, curve, base))) < mp.mpf('1e-17')
        # Include a different sheet at precisely the same base x-coordinate.
        other = (x, data.cover.fibres[root[0]][(root[1]+1) % curve.y_degree])
        displacement = inverse_a * mp.matrix(_geometric_abel_value(mp, curve, other))
        reduced = _operations.lattice_reduce(mp, (genus-1)*displacement + constant, tau).value
        assert abs(mp.rtheta(reduced, tau)) < mp.mpf('1e-12')
        point_x = mp.mpc('0.3', '0.7')
        sheets = _ordered_plane_curve_sheets(mp, curve, point_x)
        for sheet in (0, len(sheets)-1):
            point = (point_x, sheets[sheet])
            value = inverse_a * mp.matrix(_geometric_abel_value(mp, curve, point))
            shifted = inverse_a * mp.matrix(_geometric_abel_value(
                mp, curve, point, base_place=other))
            assert mp.norm(shifted - (value-displacement)) < mp.mpf('1e-16')
            # The repeated effective divisor (g-1)*P lies on the theta divisor.
            for argument in ((genus-1)*value + constant,
                             (genus-1)*shifted + constant + (genus-1)*displacement):
                reduced = _operations.lattice_reduce(mp, argument, tau).value
                assert abs(mp.rtheta(reduced, tau)) < mp.mpf('1e-12')


def test_geometric_abel_divisor_shares_only_operation_local_caches(monkeypatch):
    import mpmath.curves.jacobian as jacobian
    from mpmath.curves._stages import _geometric_abel_divisor

    with mp.workdps(18):
        curve = _prepare_plane_curve(mp, {(0, 3): 1, (4, 0): -1,
                                          (1, 0): 1, (0, 0): -1})
        data = _stage_geometric_periods(mp, curve)
        root = data.polygon.polygon.root
        other = (data.cover.geometry.vertices[root[0]],
                 data.cover.fibres[root[0]][(root[1]+1) % curve.y_degree])
        expected = _geometric_abel_value(mp, curve, other)
        cover_ids = {id(lift) for lift in data.cover.continuations}
        original = jacobian._integrate_plane_curve_path
        edge_calls, rules = [], []

        def counted(*args, **kwargs):
            rules.append(kwargs['quadrature_cache'])
            if id(args[2]) in cover_ids:
                edge_calls.append((id(args[2]), kwargs['sheet']))
            return original(*args, **kwargs)

        monkeypatch.setattr(jacobian, '_integrate_plane_curve_path', counted)
        actual = _geometric_abel_divisor(mp, curve, (other, other))
        assert mp.norm(mp.matrix(actual)-2*mp.matrix(expected)) < mp.mpf('1e-16')
        assert edge_calls and len(edge_calls) == len(set(edge_calls))
        assert all(rule is rules[0] for rule in rules)
        count, first_rules = len(edge_calls), rules[0]
        edge_calls.clear()
        rules.clear()
        actual = _geometric_abel_divisor(mp, curve, (other, other), base_place=other)
        assert mp.norm(mp.matrix(actual)) == 0
        assert len(edge_calls) == count
        assert all(rule is rules[0] for rule in rules)
        assert rules[0] is not first_rules


def _geometric_abel_value(ctx, curve, place, base_place=None):
    from mpmath.curves._stages import _geometric_abel_divisor
    return _geometric_abel_divisor(ctx, curve, (place,), base_place=base_place)
