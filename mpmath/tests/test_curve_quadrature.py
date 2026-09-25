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
