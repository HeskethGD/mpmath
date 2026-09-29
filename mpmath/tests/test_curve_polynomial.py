"""Exact algebraic identities for the polynomial primitives used by curves."""

import pytest

import mpmath.curves.polynomial as curve_polynomial
from mpmath import mp
from mpmath.curves.polynomial import (
    _evaluate_plane_derivative, _finite_plane_curve_sheets,
    _minimum_cost_assignment, _monomial_plane_curve_chart,
    _newton_polynomial_root, _plane_curve_sheets,
    _plane_polynomial_y_coefficients, _prepare_plane_curve,
    _plane_polynomial_y_coefficients_with_x_derivative,
    _plane_curve_critical_values,
    _polynomial_add, _polynomial_derivative, _polynomial_determinant,
    _polynomial_divmod, _polynomial_exact_quotient, _polynomial_multiply,
    _polynomial_squarefree_part,
)


@pytest.mark.parametrize("dividend,divisor", [
    ((1,), (1, 1)),
    ((1, 2, 1), (1, 1)),
    ((0, 0, 1), (-1, 1)),
    ((1, 0, 0, 1), (2,)),
])
def test_polynomial_division_reconstructs_the_dividend(dividend, divisor):
    dividend, divisor = tuple(map(mp.mpf, dividend)), tuple(map(mp.mpf, divisor))
    quotient, remainder = _polynomial_divmod(mp, dividend, divisor)
    assert _polynomial_add(mp, _polynomial_multiply(mp, quotient, divisor), remainder) == dividend
    assert remainder == (0,) or len(remainder) < len(divisor)


def test_polynomial_constant_and_division_failures():
    assert _polynomial_derivative(mp, (7,)) == (0,)
    assert _polynomial_squarefree_part(mp, (7,)) == (1,)
    with pytest.raises(ZeroDivisionError, match="division by zero"):
        _polynomial_divmod(mp, (1,), (0,))
    with pytest.raises(ValueError, match="division was not exact"):
        _polynomial_exact_quotient(mp, (1, 0, 1), (0, 1))


@pytest.mark.parametrize("matrix", [
    (((0,), (1,)), ((1,), (0, 1))),
    (((0,), (1,)), ((0,), (0, 1))),
    (((1, 1), (0, 1)), ((0, 2), (1, -1))),
    (((1,), (0, 1), (1, 1)), ((0, 1), (1,), (0,)), ((1,), (0,), (0, 1))),
])
def test_polynomial_determinant_agrees_with_numeric_determinants(matrix):
    determinant = _polynomial_determinant(mp, matrix)
    for x in (-2, 0, 1, 3):
        values = mp.matrix([[mp.polyval(entry, x) for entry in row] for row in matrix])
        assert mp.almosteq(mp.polyval(determinant, x), mp.det(values))


@pytest.mark.parametrize("matrix", [(), (((1,), (2,)),)])
def test_polynomial_determinant_requires_a_square_matrix(matrix):
    with pytest.raises(ValueError, match="square matrix"):
        _polynomial_determinant(mp, matrix)


@pytest.mark.parametrize("coefficients,message", [
    ([(0, 1)], "must map"),
    ({(0, -1): 1}, "nonnegative integers"),
    ({(0, 1): object()}, "must be numbers"),
    ({(0, 1): mp.inf}, "must be finite"),
    ({(0, 1): 0}, "must be nonzero"),
    ({(1, 0): 1}, "depend on y"),
])
def test_plane_polynomial_rejects_invalid_support(coefficients, message):
    with pytest.raises(ValueError, match=message):
        _prepare_plane_curve(mp, coefficients)


def test_plane_polynomial_projection_guards_and_finite_fibres():
    curve = _prepare_plane_curve(mp, {(1, 2): 1, (0, 1): 1, (0, 0): -1})
    with pytest.raises(ValueError, match="degree drops"):
        _plane_polynomial_y_coefficients(mp, curve, 0)
    with pytest.raises(ValueError, match="degree drops"):
        _plane_polynomial_y_coefficients_with_x_derivative(mp, curve, 0)
    assert _finite_plane_curve_sheets(mp, curve, 0) == (mp.one,)
    disappearing = _prepare_plane_curve(mp, {(1, 2): 1, (1, 0): 1})
    assert _finite_plane_curve_sheets(mp, disappearing, 0) == ()
    with pytest.raises(ValueError, match="must be finite"):
        _finite_plane_curve_sheets(mp, curve, mp.inf)
    with pytest.raises(ValueError, match="must be finite"):
        _plane_curve_sheets(mp, curve, mp.inf)
    with pytest.raises(ValueError, match="variable must be"):
        _evaluate_plane_derivative(mp, curve, 1, 1, "z")
    with pytest.raises(ValueError, match="chart powers"):
        _monomial_plane_curve_chart(mp, curve, 0, 1)
    with pytest.raises(ValueError, match="equal sizes"):
        _minimum_cost_assignment(mp, (0,), (0, 1))


def test_newton_reports_stationary_failed_predictions():
    # Both polynomials have zero derivative at the proposed starting value,
    # but a nonzero residual, so neither Newton step can be taken.
    for coefficients in ((1, 0, 1), (1, 0, 1, 1)):
        candidate, residual, derivative, scale, converged = (
            _newton_polynomial_root(mp, coefficients, 0, 4))
        assert candidate == 0 and residual == 1 and derivative == 0
        assert scale >= 1 and not converged


def test_projection_without_finite_critical_values_is_rejected():
    # y**2 + 1 has a constant y-resultant and no finite branch value in x.
    curve = _prepare_plane_curve(mp, {(0, 2): 1, (0, 0): 1})
    with pytest.raises(ValueError, match="no finite critical polynomial"):
        _plane_curve_critical_values(mp, curve)


def test_critical_value_resolution_retries_a_failed_precision_stage(monkeypatch):
    ctx = mp.clone()
    curve = _prepare_plane_curve(ctx, {(0, 2): 1, (1, 0): -1})
    resultant = curve_polynomial._plane_curve_resultant_y
    attempts = []

    def fail_first(context, prepared):
        attempts.append(context.prec)
        if len(attempts) == 1:
            raise context.NoConvergence("resultant did not resolve")
        return resultant(context, prepared)

    monkeypatch.setattr(curve_polynomial, "_plane_curve_resultant_y", fail_first)
    critical, unused_resultant = _plane_curve_critical_values(ctx, curve)
    assert critical == (0,)
    assert len(attempts) == 2 and attempts[1] > attempts[0]


def test_critical_value_resolution_reports_exhausted_retries(monkeypatch):
    ctx = mp.clone()
    curve = _prepare_plane_curve(ctx, {(0, 2): 1, (1, 0): -1})
    attempts = []

    def fail(context, prepared):
        attempts.append(context.prec)
        raise context.NoConvergence("resultant did not resolve")

    monkeypatch.setattr(curve_polynomial, "_plane_curve_resultant_y", fail)
    with pytest.raises(ValueError, match="failed to resolve finite critical values"):
        _plane_curve_critical_values(ctx, curve)
    assert len(attempts) == 3


def test_squarefree_part_rejects_an_inexact_numerical_gcd(monkeypatch):
    monkeypatch.setattr(
        curve_polynomial, "_polynomial_gcd", lambda ctx, left, right: (0, 1))
    with pytest.raises(ValueError, match="failed to form squarefree"):
        _polynomial_squarefree_part(mp, (1, 0, 1))
