"""Orchestration for the specialized hyperelliptic implementation."""

from .integration import (
    _branch_target_integrals, _hyperelliptic_intervals,
    _second_kind_interval,
)
from .jacobian import (
    _abel_lattice_shift, _branch_abel_values, _branch_second_kind_values,
    _first_kind_periods, _second_kind_periods, _symmetrize_period_matrix,
    _validate_legendre_relation,
)
from .model import (
    _admissible_branch_vertex, _evaluate_polynomial,
    _normalise_abel_targets, _prepare_hyperelliptic_curve,
    _target_branch_index,
)


def _hyperelliptic_periods(ctx, coefficients, method="auto",
                           second_kind=False):
    """Compute Baker-marked hyperelliptic period matrices."""
    target_eps = +ctx.eps
    quadrature_guard = 20
    cancellation_guard = 40 if second_kind else 0
    with ctx.extraprec(quadrature_guard + cancellation_guard):
        curve_data = _prepare_hyperelliptic_curve(
            ctx, coefficients, method)
        (coefficients, roots, unused_root_tolerance, use_real_method,
         genus, even_degree) = curve_data
        monomial_count = 2 * genus + 1 if second_kind else genus
        intervals, b_sign = _hyperelliptic_intervals(
            ctx, coefficients, roots, use_real_method, monomial_count)
        omega, omega_prime, tau, inverse_omega = _first_kind_periods(
            ctx, intervals, genus, even_degree, b_sign, target_eps)
        if second_kind:
            second_coefficients = coefficients
            if not even_degree:
                second_coefficients += (ctx.zero,)
            eta, eta_prime = _second_kind_periods(
                ctx, second_coefficients, intervals, genus, even_degree,
                b_sign)
            kappa = eta * inverse_omega
            _symmetrize_period_matrix(
                ctx, kappa, target_eps, "kappa matrix")
            _validate_legendre_relation(
                ctx, omega, omega_prime, eta, eta_prime, target_eps)
    if second_kind:
        return (+omega, +omega_prime, +eta, +eta_prime,
                +tau, +kappa)
    return +omega, +omega_prime, +tau


def _hyperelliptic_abel_map(
        ctx, coefficients, target, method="auto", reduce=False,
        second_kind=False, _return_shift=False):
    """Evaluate Baker-marked first- and optionally second-kind integrals."""
    target_eps = +ctx.eps
    quadrature_guard = 20
    cancellation_guard = 40 if second_kind else 0
    with ctx.extraprec(quadrature_guard + cancellation_guard):
        targets = _normalise_abel_targets(ctx, target)
        curve_data = _prepare_hyperelliptic_curve(
            ctx, coefficients, method)
        (coefficients, roots, unused_root_tolerance, use_real_method,
         genus, even_degree) = curve_data
        for x, y in targets:
            curve_value = _evaluate_polynomial(ctx, coefficients, x)
            evaluation_scale = max(
                ctx.one, abs(y ** 2),
                ctx.fsum(abs(coefficient * x ** degree)
                         for degree, coefficient in enumerate(coefficients)))
            tolerance = 100 * target_eps * evaluation_scale
            if not ctx.almosteq(
                    y ** 2, curve_value,
                    rel_eps=100 * target_eps, abs_eps=tolerance):
                raise ValueError("target point must satisfy y**2 = P(x)")

        second_coefficients = coefficients
        if second_kind and not even_degree:
            second_coefficients += (ctx.zero,)
        monomial_count = 2 * genus + 1 if second_kind else genus
        intervals, b_sign = _hyperelliptic_intervals(
            ctx, coefficients, roots, use_real_method, monomial_count)
        branch_values = _branch_abel_values(
            ctx, roots, intervals, coefficients[-1], genus, even_degree)
        if second_kind:
            second_branch_values = _branch_second_kind_values(
                ctx, second_coefficients, roots, intervals, genus,
                even_degree)
        result = ctx.zeros(genus, 1)
        if second_kind:
            second_result = ctx.zeros(genus, 1)
        for x, y in targets:
            branch_index = _target_branch_index(
                ctx, x, y, roots, target_eps)
            if branch_index is None:
                branch_index = _admissible_branch_vertex(
                    ctx, x, roots, target_eps)
                final_monomials = _branch_target_integrals(
                    ctx, roots, coefficients[-1], branch_index, x, y,
                    monomial_count, target_eps)
            else:
                final_monomials = (ctx.zero,) * monomial_count
            for row in range(genus):
                result[row] += (
                    branch_values[branch_index][row]
                    + final_monomials[row])
                if second_kind:
                    second_result[row] += (
                        second_branch_values[branch_index][row]
                        + _second_kind_interval(
                            ctx, second_coefficients, final_monomials, row,
                            genus))

        if reduce:
            omega, omega_prime, unused_tau, unused_inverse = (
                _first_kind_periods(
                    ctx, intervals, genus, even_degree, b_sign, target_eps))
            periods, lattice_shift = _abel_lattice_shift(
                ctx, result, omega, omega_prime, target_eps)
            result -= periods * lattice_shift
            if second_kind:
                eta, eta_prime = _second_kind_periods(
                    ctx, second_coefficients, intervals, genus, even_degree,
                    b_sign)
                second_periods = ctx.matrix(genus, 2 * genus)
                second_periods[:, :genus] = 2 * eta
                second_periods[:, genus:] = 2 * eta_prime
                second_result += second_periods * lattice_shift
    if second_kind and _return_shift:
        shift = (tuple(int(value) for value in lattice_shift)
                 if reduce else None)
        return +result, +second_result, shift
    if second_kind:
        return +result, +second_result
    return +result
