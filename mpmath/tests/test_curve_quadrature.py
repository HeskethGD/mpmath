import pytest

from mpmath import mp
from mpmath.curves.continuation import _continue_plane_curve_sheets
from mpmath.curves.integration import _integrate_plane_curve_path
from mpmath.curves.polynomial import _prepare_plane_curve
from mpmath.curves.quadrature import _geometric_quadrature_order


@pytest.mark.parametrize("dps", (20, 30, 40))
def test_geometry_quadrature_resolves_nearby_branch_value(dps):
    with mp.workdps(dps):
        branch = mp.mpc(1, "0.2")
        curve = _prepare_plane_curve(mp, {
            (0, 2): 1, (1, 0): -1, (0, 0): branch,
        })
        continuation = _continue_plane_curve_sheets(mp, curve, (-1, 1))
        order = _geometric_quadrature_order(mp, -1, 1, (branch,))
        assert order > 12
        assert _geometric_quadrature_order(
            mp, -1, 0, (branch,)) < order
        exact = 2 * (continuation.fibres[-1][0]
                     - continuation.fibres[0][0])
        form = (lambda x, y: 1 / y,)
        coarse = _integrate_plane_curve_path(
            mp, curve, continuation, form, quadrature_order=12)
        local = _integrate_plane_curve_path(
            mp, curve, continuation, form, quadrature_order="geometry",
            branch_values=(branch,))
        assert abs(coarse.values[0] - exact) > mp.mpf("1e-8")
        assert abs(local.values[0] - exact) < 100 * mp.eps


def test_edge_panels_resolve_deep_local_refinement_accurately():
    from mpmath.curves.quadrature import _geometric_edge_panels, _legendre_edge_rule
    with mp.workdps(30):
        branch = mp.mpc('0.137', '0.00001')
        panels = _geometric_edge_panels(mp, -1, 1, (branch,))
        assert panels[0][0] == 0 and panels[-1][1] == 1
        assert all(a[1] == b[0] for a, b in zip(panels, panels[1:]))
        assert min(b-a for a, b, n in panels) < mp.mpf(2)**-12
        assert len(panels) < 50
        rules = {n: _legendre_edge_rule(mp, n) for a, b, n in panels}
        actual = mp.fsum(2*(b-a)*weight/mp.sqrt(-1+2*(a+(b-a)*node)-branch)
                         for a, b, n in panels for node, weight in rules[n])
        exact = 2*(mp.sqrt(1-branch)-mp.sqrt(-1-branch))
        assert abs(actual-exact) < mp.mpf('1e-28')


def test_edge_panel_work_and_representability_limits(monkeypatch):
    import mpmath.curves.quadrature as quadrature
    calls = []

    def high_order(*args):
        calls.append(1)
        return 128

    monkeypatch.setattr(quadrature, '_geometric_quadrature_order', high_order)
    with mp.workdps(20):
        with pytest.raises(mp.NoConvergence, match='panel budget'):
            quadrature._geometric_edge_panels(mp, 0, 1, (1j,), max_panels=3)
        assert len(calls) == 3
        with pytest.raises(mp.NoConvergence, match='midpoint is unresolved'):
            quadrature._geometric_edge_panels(mp, mp.one, mp.one+mp.eps, (1j,))
        with pytest.raises(ValueError, match='positive integer'):
            quadrature._geometric_edge_panels(mp, 0, 1, (1j,), max_panels=0)
