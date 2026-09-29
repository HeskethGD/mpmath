import pytest

from mpmath import mp
from mpmath.curves.differentials import (
    _baker_basis, _baker_callable, _evaluate_baker_basis, _newton_polygon,
)
from mpmath.curves.polynomial import _prepare_plane_curve


def test_baker_basis_order_evaluation_and_applicability_checks():
    with mp.workdps(25):
        klein = _prepare_plane_curve(mp, {
            (3, 1): 1, (0, 3): 1, (1, 0): 1,
        })
        forms = tuple(_baker_callable(mp, _baker_basis(mp, klein, 3), i) for i in range(3))
        assert tuple(form.numerator for form in forms) == (
            (0, 0), (1, 0), (0, 1))
        basis = _baker_basis(mp, klein, 3)
        x, y = mp.mpc("0.2", "0.1"), mp.mpc("0.7", "-0.3")
        assert _evaluate_baker_basis(mp, basis, x, y) == tuple(
            form(x, y) for form in forms)
        with pytest.raises(ValueError, match="interior-point count"):
            _baker_basis(mp, klein, 2)

        # The edge polynomial 1-2*t+t**2 has a repeated toric root.
        kovalevskaya = _prepare_plane_curve(mp, {
            (0, 0): 1, (1, 0): -5.2, (2, 0): 5.4,
            (1, 2): -2, (2, 2): 6, (3, 2): -4, (2, 4): 1,
        })
        with pytest.raises(ValueError, match="degenerate edge"):
            _baker_basis(mp, kovalevskaya, 3)


def test_newton_polygon_rejects_one_dimensional_support():
    two_points = _prepare_plane_curve(mp, {(0, 1): 1, (1, 0): 1})
    collinear = _prepare_plane_curve(mp, {
        (0, 1): 1, (1, 1): 1, (2, 1): 1,
    })
    for curve in (two_points, collinear):
        with pytest.raises(ValueError, match="two-dimensional interior"):
            _newton_polygon(curve)


def test_genus_zero_has_no_first_kind_basis():
    curve = _prepare_plane_curve(mp, {(0, 2): 1, (1, 0): -1})
    with pytest.raises(ValueError, match="genus-zero"):
        _baker_basis(mp, curve, 0)
