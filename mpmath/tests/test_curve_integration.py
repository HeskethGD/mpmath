import pytest

import mpmath.curves._stages as curve_stages
import mpmath.curves.jacobian as curve_jacobian
from mpmath import mp
from mpmath.curves.continuation import _continue_plane_curve_sheets
from mpmath.curves.integration import (
    _integrate_lifted_path_chain, _integrate_plane_curve_path,
    _integrate_plane_curve_path_iterated,
)
from mpmath.curves.polynomial import (
    _newton_polynomial_root, _prepare_plane_curve,
)


def test_sparse_fibre_newton_reaches_working_precision():
    with mp.workdps(50):
        coefficients = (mp.mpc(-2, 1), mp.zero, mp.zero, mp.one)
        target = mp.root(-coefficients[0], 3)
        root, residual, derivative, scale, converged = (
            _newton_polynomial_root(
                mp, coefficients, target * mp.mpf("1.01"), 20))
        assert converged
        assert abs(root - target) < mp.mpf("1e-48")
        assert abs(residual) <= 100 * mp.eps * scale
        assert abs(derivative - 3 * root**2) < mp.mpf("1e-48")


def test_generator_period_integrals_match_whole_cycle_paths():
    # The optimized period stage integrates short generator lifts and then
    # applies the polygon transformation.  Compare that result with direct
    # integration of the original long lifted cycles on a genus-three curve.
    with mp.workdps(20):
        curve = _prepare_plane_curve(mp, {
            (0, 3): 1, (4, 0): -1, (3, 0): -2,
            (2, 0): -3, (1, 0): -5, (0, 0): -7,
        })

        def denominator(x, y):
            return 3 * y**2

        forms = (lambda x, y: 1 / denominator(x, y),
                 lambda x, y: x / denominator(x, y),
                 lambda x, y: y / denominator(x, y))
        branches = curve_stages._stage_branch_locus(mp, curve)[0]
        optimized, unused_residual = curve_stages._stage_cycle_integrals(
            mp, (curve, forms, "geometry", None))
        cache = {}
        rules = {}
        original = tuple(_integrate_lifted_path_chain(
            mp, curve, chain, forms, quadrature_order="geometry",
            branch_values=branches, integral_cache=cache,
            quadrature_cache=rules).values
            for chain in curve_stages._stage_canonical_cycles(mp, curve))
        assert max(abs(left - right)
                   for left_column, right_column in zip(optimized, original)
                   for left, right in zip(left_column, right_column)
                   ) < mp.mpf("1e-17")


def test_path_integration_accepts_a_shared_differential_evaluator():
    with mp.workdps(30):
        curve = _prepare_plane_curve(
            mp, {(0, 2): 1, (1, 0): -1})
        continuation = _continue_plane_curve_sheets(mp, curve, (1, 4))
        calls = []

        def unused_callable(x, y):
            raise AssertionError("individual differential was evaluated")

        def evaluate_together(x, y):
            calls.append((x, y))
            return 1 / y, y

        result = _integrate_plane_curve_path(
            mp, curve, continuation,
            (unused_callable, unused_callable), sheet=1,
            quadrature_order=24,
            differential_evaluator=evaluate_together)
        assert len(calls) == 24
        assert abs(result.values[0] - 2) < mp.mpf("1e-22")
        assert abs(result.values[1] - mp.mpf("14") / 3) < mp.mpf("1e-22")


def test_iterated_path_integration_accepts_geometry_quadrature():
    with mp.workdps(30):
        curve = _prepare_plane_curve(
            mp, {(0, 2): 1, (1, 0): -1})
        continuation = _continue_plane_curve_sheets(mp, curve, (1, 4))
        result = _integrate_plane_curve_path_iterated(
            mp, curve, continuation, (lambda x, y: 1 / y,), sheet=1,
            quadrature_order="geometry", branch_values=(0,))
        assert abs(result.values[0] - 2) < mp.mpf("1e-28")
        assert abs(result.iterated[0][0] - 2) < mp.mpf("1e-28")


def test_checked_path_quadrature_detects_an_unmodelled_nearby_pole():
    with mp.workdps(30):
        curve = _prepare_plane_curve(
            mp, {(0, 2): 1, (1, 0): -1})
        continuation = _continue_plane_curve_sheets(mp, curve, (1, 4))
        pole = mp.mpc("2.5", "0.01")
        with pytest.raises(mp.NoConvergence):
            _integrate_plane_curve_path(
                mp, curve, continuation,
                (lambda x, y: 1 / (x - pole),), sheet=1,
                quadrature_order="geometry", branch_values=(0,),
                check_convergence=True)


def test_first_kind_abel_map_uses_geometry_quadrature(monkeypatch):
    with mp.workdps(20):
        curve = mp.algebraic_curve({
            (0, 3): 1, (2, 0): -1, (1, 0): 1,
        })
        target = curve.fibre(2)[0]
        integrate = curve_jacobian._integrate_plane_curve_path
        calls = []

        def recording_integral(*args, **kwargs):
            calls.append((kwargs.get("quadrature_order"),
                          kwargs.get("branch_values")))
            return integrate(*args, **kwargs)

        monkeypatch.setattr(
            curve_jacobian, "_integrate_plane_curve_path",
            recording_integral)
        curve.abel_map(target)
        assert calls
        assert all(order == "geometry" and branches
                   for order, branches in calls)


def test_second_kind_abel_map_checks_geometry_quadrature(monkeypatch):
    with mp.workdps(20):
        curve = mp.algebraic_curve({
            (0, 3): 1, (2, 0): -1, (1, 0): 1,
        })
        target = curve.fibre(2)[0]
        integrate = curve_jacobian._integrate_plane_curve_path
        calls = []

        def recording_integral(*args, **kwargs):
            calls.append((kwargs.get("quadrature_order"),
                          kwargs.get("branch_values"),
                          kwargs.get("check_convergence")))
            return integrate(*args, **kwargs)

        monkeypatch.setattr(
            curve_jacobian, "_integrate_plane_curve_path",
            recording_integral)
        curve.second_kind_abel_map(
            target,
            second_differentials=(lambda x, y: x / (3 * y**2),))
        assert calls
        assert all(order == "geometry" and branches and checked
                   for order, branches, checked in calls)
