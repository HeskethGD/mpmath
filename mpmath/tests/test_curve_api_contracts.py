"""Public curve contracts that do not depend on theta or Kleinian functions."""

import warnings

import pytest

import mpmath.curves._operations as curve_operations
from mpmath import CurvePlace, mp
from mpmath.curves import (
    CurveBranchLocus, CurveGenus, CurveHomology, CurveMonodromy,
    CurveRiemannConstant,
)
from mpmath.curves.polynomial import _prepare_plane_curve
from mpmath.curves._stages import _stage_geometric_periods
from mpmath.curves.jacobian import _finite_geometric_abel_value


@pytest.mark.parametrize("specification,message", [
    (None, "coefficients or sparse plane terms"),
    ((), "must not be empty"),
    ({}, "must be nonzero"),
    ({(0, 2): 0}, "must be nonzero"),
    ({(2, 0): 1}, "must depend on y"),
    ({(0, -1): 1}, "nonnegative integers"),
    ({"y": 1}, "nonnegative integers"),
    ({(0, 2): "invalid"}, "coefficients must be numbers"),
    (((0, 2, "invalid"),), "coefficients must be numbers"),
    ({(0, 2): "inf"}, "coefficients must be finite"),
    ({(0, 2): "nan"}, "coefficients must be finite"),
])
def test_curve_constructor_rejects_invalid_specification(specification, message):
    with pytest.raises(ValueError, match=message):
        mp.algebraic_curve(specification)


def test_curve_materializes_input_and_returns_a_copy_of_the_specification():
    ctx = mp.clone()
    source = {(0, 2): 1, (3, 0): -1, (1, 0): 1}
    expected = dict(source)
    curve = ctx.algebraic_curve(source)
    source[(3, 0)] = -2
    exported = curve.specification
    exported[(3, 0)] = -3
    assert curve.specification == expected
    assert repr(curve) == (
        f"AlgebraicCurve(x_degree=3, y_degree=2, ctx.prec={ctx.prec})")

    terms = ((i, j, value) for (i, j), value in expected.items())
    from_terms = ctx.algebraic_curve(terms)
    assert tuple(terms) == ()
    assert from_terms.specification == tuple(
        (i, j, value) for (i, j), value in expected.items())
    assert from_terms.branch_locus == curve.branch_locus
    with ctx.workdps(25), pytest.warns(UserWarning, match="context changed"):
        assert curve.branch_locus.branch_values == (-1, 0, 1)


@pytest.mark.parametrize("attribute,value", [
    ("rounding", "d"), ("trap_complex", True),
])
def test_curve_context_state_changes_warn_once_and_restore(attribute, value):
    ctx = mp.clone()
    curve = ctx.algebraic_curve({(0, 2): 1, (1, 0): -1})
    original = curve.branch_locus
    previous = getattr(ctx, attribute)
    setattr(ctx, attribute, value)
    with pytest.warns(UserWarning, match="context changed"):
        assert curve.branch_locus == original
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert curve.branch_locus == original
        setattr(ctx, attribute, previous)
        assert curve.branch_locus == original
        setattr(ctx, attribute, value)
        assert curve.branch_locus == original
    assert caught == []


def test_curve_recomputes_exact_inputs_without_inventing_coefficient_precision():
    ctx = mp.clone()
    ctx.dps = 15
    rounded = ctx.mpf("0.1")
    exact = ctx.algebraic_curve({(0, 1): 10, (0, 0): -1})
    inexact = ctx.algebraic_curve({(0, 1): 1, (0, 0): -rounded})
    with ctx.workdps(40):
        with pytest.warns(UserWarning, match="context changed"):
            exact_y = exact.fibre(0)[0].y
        with pytest.warns(UserWarning, match="context changed"):
            inexact_y = inexact.fibre(0)[0].y
        assert abs(exact_y - ctx.mpf("0.1")) < ctx.eps
        assert inexact_y == rounded
        assert abs(inexact_y - exact_y) > ctx.mpf("1e-20")


def test_period_results_preserve_basis_and_do_not_share_mutable_matrices():
    ctx = mp.clone()
    ctx.dps = 18
    curve = ctx.algebraic_curve((0, -1, 0, 1))
    first = curve.first_kind_periods()
    expected = tuple(+getattr(first, name) for name in ("omega", "omega_prime", "tau"))
    for name in ("omega", "omega_prime", "tau"):
        getattr(first, name)[0, 0] = 0
    # A supplied basis selects a different marking without changing the
    # default basis or contaminating its cached matrices.
    forms = (lambda x, y: 2 / y,)
    supplied = curve.first_kind_periods(forms)
    assert supplied.differentials == forms
    assert supplied.marking == "geometric-polygon"
    again = curve.first_kind_periods()
    assert again.marking == "baker"
    for name, value in zip(("omega", "omega_prime", "tau"), expected):
        assert getattr(again, name) == value
    second = curve.second_kind_periods()
    saved = tuple(+getattr(second, name) for name in ("eta", "eta_prime", "kappa"))
    for name in ("eta", "eta_prime", "kappa"):
        getattr(second, name)[0, 0] = 0
    second_again = curve.second_kind_periods()
    for name, value in zip(("eta", "eta_prime", "kappa"), saved):
        assert getattr(second_again, name) == value
    assert curve.first_kind_periods().marking == "baker"
    constant = curve.riemann_constant()
    constant.value[0] = 0
    assert ctx.almosteq(curve.riemann_constant().value[0], (1 + expected[2][0, 0]) / 2)


def test_curve_validation_reports_failed_topological_checks():
    curve = mp.algebraic_curve({(0, 2): 1, (1, 0): -1})
    cases = [
        (CurveBranchLocus(2, (0, 0), (0, 1)), "branch_values_distinct"),
        (CurveBranchLocus(2, (0,), (1,)), "resultant_nonconstant"),
        (CurveGenus(1, 2, 2), "riemann_hurwitz_balance"),
        (CurveMonodromy(1, (-1, 1), (0,), ((0, 1),), (0, 1),
                       2, 0, False, True, 1), "monodromy_transitive"),
        (CurveMonodromy(1, (-1, 1), (0,), ((1, 0),), (0, 1),
                       2, 0, True, False, 1), "monodromy_product_identity"),
        (CurveHomology(1, 2, 0, 2, 0, ((0, 1), (1, 0)),
                       ((1, 0), (0, 1)), "general", "geometric-polygon"),
         "intersection_form_antisymmetric"),
        (CurveHomology(1, 2, 0, 2, 0, ((0, 0), (0, 0)),
                       ((1, 0), (0, 1)), "general", "geometric-polygon"),
         "intersection_form_rank"),
    ]
    for result, name in cases:
        report = curve.validate(result)
        assert not report.passed
        assert report.maximum_residual is None
        assert not next(check.passed for check in report.checks if check.name == name)
    with pytest.raises(TypeError, match="curve result record"):
        curve.validate(mp.eye(1))


@pytest.mark.parametrize("characteristic", [None, ((), ()), ((mp.inf,), (0,))])
def test_riemann_constant_validation_rejects_invalid_characteristics(characteristic):
    curve = mp.algebraic_curve((0, -1, 0, 1))
    result = CurveRiemannConstant(
        mp.matrix([mp.mpc('.5', '.5')]), characteristic, None, None,
        "hyperelliptic", "baker")
    report = curve.validate(result)
    assert not report.passed
    assert not next(check.passed for check in report.checks
                    if check.name == "characteristic_finite")


@pytest.mark.parametrize("value", [mp.matrix([[0, 1]]), mp.matrix([mp.inf])])
def test_riemann_constant_validation_rejects_invalid_values(value):
    curve = mp.algebraic_curve((0, -1, 0, 1))
    result = CurveRiemannConstant(
        value, ((0,), (0,)), None, None, "hyperelliptic", "baker")
    assert not curve.validate(result).passed


def test_curve_validation_uses_recorded_residual_and_current_precision():
    ctx = mp.clone()
    ctx.dps = 15
    curve = ctx.algebraic_curve((0, -1, 0, 1))
    residual = ctx.mpf("1e-8")
    result = CurveRiemannConstant(
        ctx.matrix([ctx.mpc('.5', '.5')]), ((ctx.mpf('.5'),), (ctx.mpf('.5'),)),
        None, residual, "general", "geometric-polygon")
    assert curve.validate(result).passed
    with ctx.workdps(40), pytest.warns(UserWarning, match="context changed"):
        report = curve.validate(result)
    assert not report.passed
    assert report.maximum_residual == residual
    assert not next(check.passed for check in report.checks
                    if check.name == "max_sheet_residual")


@pytest.mark.parametrize("periods", [None, [[1, 2]], [["invalid"]]])
def test_lattice_reduction_rejects_invalid_periods(periods):
    curve = mp.algebraic_curve((0, -1, 0, 1))
    with pytest.raises(ValueError, match="periods must be square"):
        curve.lattice_reduce([0], periods)


@pytest.mark.parametrize("value", [None, [0, 1], [[0, 1]], ["invalid"]])
def test_lattice_reduction_rejects_invalid_vectors(value):
    curve = mp.algebraic_curve((0, -1, 0, 1))
    with pytest.raises(ValueError, match="genus-length column vector"):
        curve.lattice_reduce(value, mp.matrix([[1j]]))


@pytest.mark.parametrize("forms", [0, (), (1,)])
def test_public_integrals_reject_invalid_differential_sequences(forms):
    curve = mp.algebraic_curve({(0, 2): 1, (1, 0): -1})
    path = curve.path((1, 1), (4, 2))
    with pytest.raises(ValueError, match="sequence of callables"):
        curve.integral(forms, path)


def test_public_integral_requires_a_lifted_path():
    curve = mp.algebraic_curve({(0, 2): 1, (1, 0): -1})
    with pytest.raises(TypeError, match="must be a CurvePath"):
        curve.integral(lambda x, y: 1, [(1, 1), (4, 2)])


@pytest.mark.parametrize("place,message", [
    (None, "finite .* pair"),
    ((mp.inf, 1), "finite .* pair"),
    ((0, 0), "regular for the x projection"),
])
def test_public_path_rejects_nonregular_endpoints(place, message):
    curve = mp.algebraic_curve({(0, 2): 1, (1, 0): -1})
    with pytest.raises(ValueError, match=message):
        curve.path(place, (1, 1))


def test_hyperelliptic_supplied_second_kind_forms_require_first_kind_basis():
    curve = mp.algebraic_curve((0, -1, 0, 1))
    forms = (lambda x, y: x/y,)
    with pytest.raises(ValueError, match="require a supplied first-kind basis"):
        curve.second_kind_periods(second_differentials=forms)
    with pytest.raises(ValueError, match="require a supplied first-kind basis"):
        curve.second_kind_abel_map([], second_differentials=forms)


def test_hyperelliptic_chart_endpoints_require_supplied_forms():
    curve = mp.algebraic_curve((0, -1, 0, 1))
    chart = curve.monomial_chart(-2, -3)
    place = curve.chart_place(chart, 1, mp.mpf('.1'))
    for target in (place, [place]):
        with pytest.raises(ValueError, match="chart-backed places require the general pipeline"):
            curve.abel_map(target)
    with pytest.raises(ValueError, match="base_place must be one affine point"):
        curve.abel_map([], base_place=[])


def test_hyperelliptic_second_kind_unreduced_result_and_scalar_target():
    ctx = mp.clone()
    ctx.dps = 18
    curve = ctx.algebraic_curve((0, -1, 0, 1))
    target = (ctx.mpf(2), ctx.sqrt(6))
    result = curve.second_kind_abel_map(target)
    assert result.reduction_shift is None
    assert result.engine == "hyperelliptic"
    assert all(ctx.isfinite(entry) for entry in result.value)
    with pytest.raises(ValueError, match="affine point"):
        curve.abel_map(2)


def test_geometric_periods_reject_nonpositive_imaginary_part(monkeypatch):
    ctx = mp.clone()
    ctx.dps = 18
    curve = _prepare_plane_curve(ctx, {(0, 2): 1, (3, 0): -1, (1, 0): 1})
    monkeypatch.setattr(
        curve_operations, "_tau_imaginary_eigenvalues",
        lambda context, tau: (-context.one,))
    with pytest.raises(ValueError, match="not positive definite"):
        curve_operations._geometric_first_kind_periods(ctx, curve)


def test_geometric_abel_open_path_rejects_an_unmatched_place():
    ctx = mp.clone()
    ctx.dps = 18
    curve = _prepare_plane_curve(ctx, {(0, 2): 1, (3, 0): -1, (1, 0): 1})
    data = _stage_geometric_periods(ctx, curve)
    with pytest.raises(ValueError, match="could not be matched"):
        _finite_geometric_abel_value(ctx, curve, data, CurvePlace(2, 0), ())
