import pytest

import mpmath.curves.jacobian as curve_jacobian
import mpmath.curves.integration as curve_integration
from mpmath import mp
from mpmath.curves.continuation import _continue_plane_curve_sheets_adaptive
from mpmath.curves.integration import (
    _concatenate_iterated_path_integrals, _gauss_indefinite_matrix,
    _integrate_plane_curve_path,
    _integrate_plane_curve_path_iterated,
    _newton_plane_curve_segment_samples,
    _pullback_plane_curve_differentials, _reverse_iterated_path_integrals,
)
from mpmath.curves._records import _IteratedPathIntegrals
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


def test_path_integration_accepts_a_shared_differential_evaluator():
    with mp.workdps(30):
        curve = _prepare_plane_curve(
            mp, {(0, 2): 1, (1, 0): -1})
        continuation = _continue_plane_curve_sheets_adaptive(mp, curve, (1, 4))
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
        assert len(calls) == 24 * (len(continuation.path)-1)
        assert abs(result.values[0] - 2) < mp.mpf("1e-22")
        assert abs(result.values[1] - mp.mpf("14") / 3) < mp.mpf("1e-22")


def test_iterated_path_integration_accepts_geometry_quadrature():
    with mp.workdps(30):
        curve = _prepare_plane_curve(
            mp, {(0, 2): 1, (1, 0): -1})
        continuation = _continue_plane_curve_sheets_adaptive(mp, curve, (1, 4))
        result = _integrate_plane_curve_path_iterated(
            mp, curve, continuation, (lambda x, y: 1 / y,), sheet=1,
            quadrature_order="geometry", branch_values=(0,))
        assert abs(result.values[0] - 2) < mp.mpf("1e-28")
        assert abs(result.iterated[0][0] - 2) < mp.mpf("1e-28")


def test_checked_path_quadrature_detects_an_unmodelled_nearby_pole():
    with mp.workdps(30):
        curve = _prepare_plane_curve(
            mp, {(0, 2): 1, (1, 0): -1})
        continuation = _continue_plane_curve_sheets_adaptive(mp, curve, (1, 4))
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


def test_checked_chart_tail_checks_each_component_and_bounds_work(monkeypatch):
    from mpmath.curves.continuation import _continue_plane_curve_branch
    from mpmath.curves.integration import _integrate_plane_curve_branch
    ctx = mp.clone()
    ctx.dps = 20
    curve = _prepare_plane_curve(ctx, {(0, 1): 1, (0, 0): -1})
    branch = _continue_plane_curve_branch(ctx, curve, (0, 1), 1)
    original = ctx.gauss_quadrature
    orders = []

    def counted(order, *args, **kwargs):
        orders.append(order)
        return original(order, *args, **kwargs)

    monkeypatch.setattr(ctx, 'gauss_quadrature', counted)
    # A large convergent component must not hide a smaller divergent one.
    forms = (lambda t, w: ctx.mpf('1e100'), lambda t, w: 1/t**2)
    with pytest.raises(ctx.NoConvergence, match='endpoint may be a pole'):
        _integrate_plane_curve_branch(ctx, curve, branch, forms, check_convergence=True)
    assert orders == [40, 56, 72, 88]
    assert ctx.dps == 20


@pytest.mark.parametrize("options,message", [
    ({"sheet": -1}, "index the initial fibre"),
    ({"sheet": 1}, "index the initial fibre"),
    ({"quadrature_order": 1}, "integer at least 2"),
    ({"quadrature_order": "geometry"}, "requires branch values"),
    ({"quadrature_cache": []}, "must be a dictionary"),
])
@pytest.mark.parametrize("integrate", [
    _integrate_plane_curve_path, _integrate_plane_curve_path_iterated,
])
def test_path_quadrature_rejects_invalid_controls(integrate, options, message):
    curve = _prepare_plane_curve(mp, {(0, 1): 1, (1, 0): -1})
    path = _continue_plane_curve_sheets_adaptive(mp, curve, (0, 1))
    with pytest.raises(ValueError, match=message):
        integrate(mp, curve, path, (lambda x, y: 1,), **options)


@pytest.mark.parametrize("forms", [(), (1,)])
@pytest.mark.parametrize("integrate", [
    _integrate_plane_curve_path, _integrate_plane_curve_path_iterated,
])
def test_path_quadrature_requires_callable_forms(integrate, forms):
    curve = _prepare_plane_curve(mp, {(0, 1): 1, (1, 0): -1})
    path = _continue_plane_curve_sheets_adaptive(mp, curve, (0, 1))
    with pytest.raises(ValueError, match="sequence of callables"):
        integrate(mp, curve, path, forms)


def test_path_quadrature_validates_vectorized_evaluation():
    curve = _prepare_plane_curve(mp, {(0, 1): 1, (1, 0): -1})
    path = _continue_plane_curve_sheets_adaptive(mp, curve, (0, 1))
    forms = (lambda x, y: 1,)
    with pytest.raises(ValueError, match="sequence of callables"):
        _integrate_plane_curve_path(mp, curve, path, None)
    with pytest.raises(ValueError, match="differential_evaluator must be callable"):
        _integrate_plane_curve_path(mp, curve, path, forms, differential_evaluator=1)
    with pytest.raises(ValueError, match="wrong number of values"):
        _integrate_plane_curve_path(mp, curve, path, forms,
                                   differential_evaluator=lambda x, y: (1, 2))
    with pytest.raises(ValueError, match="check_convergence must be boolean"):
        _integrate_plane_curve_path(mp, curve, path, forms, check_convergence="yes")


def test_zero_length_paths_have_zero_ordinary_and_iterated_integrals():
    from mpmath.curves.continuation import _continue_plane_curve_branch
    from mpmath.curves.integration import _integrate_plane_curve_branch
    ctx = mp.clone()
    ctx.dps = 20
    curve = _prepare_plane_curve(ctx, {(0, 1): 1, (1, 0): -1})
    path = _continue_plane_curve_sheets_adaptive(ctx, curve, (1, 1))
    branch = _continue_plane_curve_branch(ctx, curve, (1, 1), 1)
    forms = (lambda x, y: 1,)
    for integrate in (_integrate_plane_curve_path, _integrate_plane_curve_path_iterated):
        result = integrate(ctx, curve, path, forms)
        assert result.values == (0,)
        assert result.max_sheet_residual == 0
    assert result.iterated == ((0,),)
    assert _integrate_plane_curve_branch(ctx, curve, branch, forms).values == (0,)


@pytest.mark.parametrize("options,message", [
    ({"quadrature_order": 1}, "integer at least 2"),
    ({"check_convergence": "yes"}, "must be boolean"),
])
def test_chart_quadrature_rejects_invalid_controls(options, message):
    from mpmath.curves.continuation import _continue_plane_curve_branch
    from mpmath.curves.integration import _integrate_plane_curve_branch
    curve = _prepare_plane_curve(mp, {(0, 1): 1, (1, 0): -1})
    branch = _continue_plane_curve_branch(mp, curve, (0, 1), 0)
    with pytest.raises(ValueError, match=message):
        _integrate_plane_curve_branch(mp, curve, branch, (lambda x, y: 1,), **options)


@pytest.mark.parametrize("forms", [None, (), (1,)])
def test_chart_quadrature_requires_callable_forms(forms):
    from mpmath.curves.continuation import _continue_plane_curve_branch
    from mpmath.curves.integration import _integrate_plane_curve_branch
    curve = _prepare_plane_curve(mp, {(0, 1): 1, (1, 0): -1})
    branch = _continue_plane_curve_branch(mp, curve, (0, 1), 0)
    with pytest.raises(ValueError, match="sequence of callables"):
        _integrate_plane_curve_branch(mp, curve, branch, forms)


def test_iterated_integrals_obey_chen_concatenation_and_reversal():
    ctx = mp.clone()
    ctx.dps = 25
    curve = _prepare_plane_curve(ctx, {(0, 1): 1, (1, 0): -1})
    forms = (lambda x, y: 1, lambda x, y: x)
    paths = tuple(_continue_plane_curve_sheets_adaptive(ctx, curve, endpoints)
                  for endpoints in ((0, 1), (1, 2), (0, 2)))
    left, right, whole = tuple(
        _integrate_plane_curve_path_iterated(ctx, curve, path, forms,
                                             quadrature_order=12)
        for path in paths)
    composed = _concatenate_iterated_path_integrals(ctx, left, right)
    for a, b in zip(composed.values, whole.values):
        assert abs(a - b) < 100 * ctx.eps
    for a_row, b_row in zip(composed.iterated, whole.iterated):
        assert all(abs(a - b) < 100 * ctx.eps for a, b in zip(a_row, b_row))
    reversed_left = _reverse_iterated_path_integrals(ctx, left)
    round_trip = _concatenate_iterated_path_integrals(ctx, left, reversed_left)
    assert all(abs(value) < 100 * ctx.eps for value in round_trip.values)
    assert all(abs(value) < 100 * ctx.eps
               for row in round_trip.iterated for value in row)


def test_iterated_integrals_reject_incompatible_dimensions():
    valid = _IteratedPathIntegrals((1,), ((1,),), 0, 1)
    bad = _IteratedPathIntegrals((1, 2), ((1, 2),), 0, 1)
    with pytest.raises(ValueError, match="incompatible sizes"):
        _concatenate_iterated_path_integrals(mp, valid, bad)
    with pytest.raises(ValueError, match="incompatible size"):
        _reverse_iterated_path_integrals(mp, bad)
    with pytest.raises(ValueError, match="equal length"):
        _gauss_indefinite_matrix(mp, (mp.mpf('.5'),), ())


def test_chart_pullback_applies_dx_dt_and_validates_inputs():
    pullback = _pullback_plane_curve_differentials(
        (lambda x, y: x + y,), lambda t, u: (t*t, u, 2*t))
    assert pullback[0](3, 4) == 78
    with pytest.raises(ValueError, match="sequence of callables"):
        _pullback_plane_curve_differentials((), lambda t, u: (t, u, 1))
    with pytest.raises(ValueError, match="coordinate_map must be callable"):
        _pullback_plane_curve_differentials((lambda x, y: x,), None)


def test_chart_quadrature_rejects_inconsistent_branch_data():
    from mpmath.curves.continuation import _continue_plane_curve_branch
    from mpmath.curves.integration import _integrate_plane_curve_branch
    curve = _prepare_plane_curve(mp, {(0, 1): 1, (1, 0): -1})
    branch = _continue_plane_curve_branch(mp, curve, (0, 1), 0)
    inconsistent = branch._replace(values=branch.values[:-1])
    with pytest.raises(ValueError, match="path and values are inconsistent"):
        _integrate_plane_curve_branch(
            mp, curve, inconsistent, (lambda x, y: x,))


@pytest.mark.parametrize("broken,message", [
    ("path", "path and fibres are inconsistent"),
    ("fibre", "fibres have the wrong degree"),
])
def test_path_quadrature_rejects_inconsistent_continuation(broken, message):
    curve = _prepare_plane_curve(mp, {(0, 2): 1, (1, 0): -1})
    continuation = _continue_plane_curve_sheets_adaptive(mp, curve, (1, 2))
    if broken == "path":
        continuation = continuation._replace(path=continuation.path[:-1])
    else:
        continuation = continuation._replace(
            fibres=((continuation.fibres[0][0],),) + continuation.fibres[1:])
    with pytest.raises(ValueError, match=message):
        _integrate_plane_curve_path(
            mp, curve, continuation, (lambda x, y: 1/y,))


def test_segment_newton_declines_unsafe_predictions_and_inconsistent_endpoints():
    ctx = mp.clone()
    ctx.dps = 18
    curve = _prepare_plane_curve(ctx, {(0, 2): 1, (1, 0): -1})
    sample = _newton_plane_curve_segment_samples
    half = (ctx.mpf('.5'),)
    assert sample(ctx, curve, 0, 1, (0, 0), (-1, 1), half, 1) is None
    assert sample(ctx, curve, 1, 16, (-1, 1), (-4, 4), half, 1) is None
    assert sample(ctx, curve, 1, 2, (-1, 1), (-ctx.sqrt(2), 2), half, 1) is None
    assert sample(ctx, curve, 1, 100, (-1, 1), (-10, 10),
                  (ctx.mpf('.01'),), 1) is None
    # A nearly duplicate endpoint label is unsafe even if Newton reaches
    # the expected value to within its ordinary residual tolerance.
    near = ctx.sqrt(2)
    assert sample(ctx, curve, 1, 2, (-1, 1),
                  (near, near + ctx.mpf('1e-12')), half, 1) is None


def test_riemann_constant_rejects_incompatible_cycle_count():
    with pytest.raises(ValueError, match="canonical cycles and period matrix disagree"):
        curve_jacobian._riemann_constant_from_iterated_cycles(mp, mp.eye(1), ())


def test_chart_quadrature_reports_a_stalled_node_correction(monkeypatch):
    from mpmath.curves.continuation import _continue_plane_curve_branch
    ctx = mp.clone()
    curve = _prepare_plane_curve(ctx, {(0, 2): 1, (1, 0): -1})
    branch = _continue_plane_curve_branch(ctx, curve, (1, 2), 1)
    derivative = curve_integration._evaluate_plane_derivative

    def stalled_derivative(context, prepared, x, y, variable):
        if variable == "y":
            return context.zero
        return derivative(context, prepared, x, y, variable)

    monkeypatch.setattr(
        curve_integration, "_evaluate_plane_derivative", stalled_derivative)
    with pytest.raises(ctx.NoConvergence, match="node did not resolve"):
        curve_integration._integrate_plane_curve_branch(
            ctx, curve, branch, (lambda t, u: 1,), quadrature_order=4)
