from mpmath import mp
from mpmath.curves.continuation import _continue_plane_curve_sheets
from mpmath.curves.integration import _integrate_plane_curve_path
from mpmath.curves.polynomial import _prepare_plane_curve


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
