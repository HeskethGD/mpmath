"""Period and Jacobian assembly for Baker-marked curves."""

from .integration import (
    _infinity_branch_integrals, _infinity_second_kind_integrals,
    _second_kind_interval,
)


def _matrix_tuple(matrix):
    """Return a matrix as an immutable tuple of row tuples."""
    return tuple(tuple(matrix[row, column] for column in range(matrix.cols))
                 for row in range(matrix.rows))

def _first_kind_periods(ctx, intervals, genus, even_degree, b_sign,
                        target_eps):
    """Construct first-kind half-periods from adjacent branch integrals."""
    cycle_offset = 1 if even_degree else 0
    omega = ctx.matrix(genus)
    omega_prime = ctx.matrix(genus)
    for row in range(genus):
        for column in range(genus):
            omega[row, column] = intervals[2 * column + cycle_offset][row]
            omega_prime[row, column] = b_sign * ctx.fsum(
                intervals[2 * edge + 1 + cycle_offset][row]
                for edge in range(column, genus))
    inverse_omega = ctx.inverse(omega)
    tau = inverse_omega * omega_prime
    _symmetrize_period_matrix(ctx, tau, target_eps, "period matrix")
    ctx._rtheta_tau_data(_matrix_tuple(tau))
    return omega, omega_prime, tau, inverse_omega



def _branch_abel_values(ctx, roots, intervals, leading, genus, even_degree):
    """Return deterministic Abel images of all finite branch points."""
    values = [None] * len(roots)
    if even_degree:
        values[0] = (ctx.zero,) * genus
        for index, interval in enumerate(intervals):
            values[index + 1] = tuple(
                values[index][power] + interval[power]
                for power in range(genus))
    else:
        values[-1] = _infinity_branch_integrals(
            ctx, roots, leading, genus)
        for index in range(len(intervals) - 1, -1, -1):
            values[index] = tuple(
                values[index + 1][power] - intervals[index][power]
                for power in range(genus))
    return tuple(values)


def _branch_second_kind_values(ctx, coefficients, roots, intervals, genus,
                               even_degree):
    """Return compatible second-kind integrals at every branch point."""
    second_intervals = tuple(tuple(
        _second_kind_interval(ctx, coefficients, monomials, row, genus)
        for row in range(genus)) for monomials in intervals)
    values = [None] * len(roots)
    if even_degree:
        values[0] = (ctx.zero,) * genus
        for index, interval in enumerate(second_intervals):
            values[index + 1] = tuple(
                values[index][row] + interval[row]
                for row in range(genus))
    else:
        values[-1] = _infinity_second_kind_integrals(
            ctx, coefficients, roots, genus)
        for index in range(len(second_intervals) - 1, -1, -1):
            values[index] = tuple(
                values[index + 1][row] - second_intervals[index][row]
                for row in range(genus))
    return tuple(values)



def _abel_lattice_shift(ctx, value, omega, omega_prime, target_eps):
    """Resolve an Abelian vector into the full period lattice."""
    genus = omega.rows
    periods = ctx.matrix(genus, 2 * genus)
    periods[:, :genus] = 2 * omega
    periods[:, genus:] = 2 * omega_prime
    real_periods = ctx.matrix(2 * genus)
    right_hand_side = ctx.matrix(2 * genus, 1)
    for row in range(genus):
        right_hand_side[row] = ctx.re(value[row])
        right_hand_side[genus + row] = ctx.im(value[row])
        for column in range(2 * genus):
            real_periods[row, column] = ctx.re(periods[row, column])
            real_periods[genus + row, column] = ctx.im(
                periods[row, column])
    try:
        coordinates = ctx.lu_solve(real_periods, right_hand_side)
    except (ValueError, ZeroDivisionError):
        raise ValueError("failed to resolve the full period lattice")
    reconstructed = periods * coordinates
    scale = max(ctx.one, ctx.norm(value), ctx.norm(periods) * ctx.norm(
        coordinates))
    if ctx.norm(reconstructed - value) > 100 * target_eps * scale:
        raise ValueError("failed to resolve the full period lattice")
    lattice_shift = ctx.matrix([
        ctx.floor(coordinate + ctx.convert(0.5))
        for coordinate in coordinates
    ])
    return periods, lattice_shift



def _second_kind_periods(ctx, coefficients, intervals, genus, even_degree,
                         b_sign):
    """Construct canonical second-kind half-period matrices."""
    cycle_offset = 1 if even_degree else 0
    eta = ctx.matrix(genus)
    eta_prime = ctx.matrix(genus)
    for row in range(genus):
        second_intervals = [
            _second_kind_interval(
                ctx, coefficients, values, row, genus)
            for values in intervals
        ]
        for column in range(genus):
            eta[row, column] = -second_intervals[
                2 * column + cycle_offset]
            eta_prime[row, column] = -b_sign * ctx.fsum(
                second_intervals[2 * edge + 1 + cycle_offset]
                for edge in range(column, genus))
    return eta, eta_prime


def _symmetrize_period_matrix(ctx, matrix, target_eps, description):
    """Check numerical symmetry and average only rounding-level differences."""
    for row in range(matrix.rows):
        for column in range(row):
            difference = abs(matrix[row, column] - matrix[column, row])
            scale = max(ctx.one, abs(matrix[row, column]),
                        abs(matrix[column, row]))
            if difference > 100 * target_eps * scale:
                raise ValueError(
                    f"failed to compute a symmetric {description}")
            entry = (matrix[row, column] + matrix[column, row]) / 2
            matrix[row, column] = matrix[column, row] = entry


def _validate_legendre_relation(ctx, omega, omega_prime, eta, eta_prime,
                                target_eps):
    """Check the generalized Legendre relation for half-period matrices."""
    genus = omega.rows
    periods = ctx.matrix(2 * genus)
    periods[:genus, :genus] = omega
    periods[:genus, genus:] = omega_prime
    periods[genus:, :genus] = eta
    periods[genus:, genus:] = eta_prime
    symplectic = ctx.zeros(2 * genus)
    for index in range(genus):
        symplectic[index, genus + index] = -1
        symplectic[genus + index, index] = 1
    expected = -ctx.pi * ctx.j * symplectic / 2
    residual = ctx.norm(periods * symplectic * periods.T - expected)
    scale = max(ctx.one, ctx.norm(periods) ** 2, ctx.norm(expected))
    if residual > 100 * target_eps * scale:
        raise ValueError("failed to satisfy the generalized Legendre relation")


def _hyperelliptic_characteristic(ctx, genus):
    """Return the Riemann characteristic for the selected canonical basis."""
    half = ctx.convert(0.5)
    a = (half,) * genus
    b = tuple(half if (genus - index) & 1 else ctx.zero
              for index in range(genus))
    return a, b

