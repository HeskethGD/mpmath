# Numerical period data for hyperelliptic curves
# ------------------------------------------------
#
# The first implementation covers odd-degree real models
#
#     y**2 = P(x),  deg(P) = 2*g + 1,
#
# with distinct real branch points.  It uses the classical Baker cycle
# arrangement written explicitly, for example, in Bernatska, "Computation of
# P-Functions on Plane Algebraic Curves", J. Exp. Math. 2 (2026), Section 3.
# Endpoint singularities are removed by a cosine parametrization; mpmath's
# existing adaptive quadrature then integrates smooth functions.
#
# The associated second-kind differentials are equation (1.3) of Buchstaber,
# Enolskii and Leykin, "Hyperelliptic Kleinian Functions and Applications".
# BEL write full periods as 2*omega and attach a minus sign to their eta
# integrals.  Here omega and eta are full periods and follow Bernatska's sign
# convention, for which kappa = eta*omega**-1 and the Legendre constant is
# +2*pi*i.

from .functions import defun
from .riemann_theta import _matrix_tuple


def _real_hyperelliptic_coefficients(ctx, coefficients):
    """Validate real polynomial coefficients in ascending order."""
    try:
        coefficients = tuple(ctx.convert(value) for value in coefficients)
    except (TypeError, ValueError):
        raise ValueError("coefficients must be a sequence of real numbers")
    if not coefficients or not coefficients[-1]:
        raise ValueError("the leading coefficient must be nonzero")
    if any(not ctx.isfinite(value) or ctx.im(value)
           for value in coefficients):
        raise ValueError("coefficients must be finite real numbers")
    degree = len(coefficients) - 1
    if degree < 3 or not degree & 1:
        raise ValueError("the polynomial degree must be odd and at least 3")
    return tuple(ctx.re(value) for value in coefficients)


def _real_hyperelliptic_roots(ctx, coefficients):
    """Return distinct real roots in increasing order."""
    roots = ctx.polyroots(coefficients, maxsteps=200, error=False)
    scale = max([ctx.one] + [abs(root) for root in roots])
    tolerance = ctx.sqrt(ctx.eps) * scale
    if any(abs(ctx.im(root)) > tolerance for root in roots):
        raise ValueError("the polynomial must have only real roots")
    roots = sorted(ctx.re(root) for root in roots)
    if any(abs(right - left) <= tolerance
           for left, right in zip(roots, roots[1:])):
        raise ValueError("the polynomial must have distinct roots")
    return tuple(roots)


def _real_branch_integrals(ctx, roots, leading, interval, count):
    """Integrate monomials x**k dx/y between two adjacent roots."""
    left = roots[interval]
    right = roots[interval + 1]
    midpoint = (left + right) / 2
    half_width = (right - left) / 2
    excluded = interval, interval + 1
    # Continuing sqrt(P(x)) from the right of every branch point fixes the
    # sheet.  Moving left across one real root multiplies it by i.
    phase = ctx.j ** (len(roots) - interval - 1)
    root_leading = ctx.sqrt(leading)

    def integrand(angle, power):
        x = midpoint - half_width * ctx.cos(angle)
        remaining = ctx.fprod(
            x - root for index, root in enumerate(roots)
            if index not in excluded)
        return x ** power / (
            root_leading * phase * ctx.sqrt(abs(remaining)))

    return tuple(ctx.quad(lambda angle, k=k: integrand(angle, k),
                          [0, ctx.pi])
                 for k in range(count))


def _second_kind_interval(ctx, coefficients, monomials, row, genus):
    """Combine monomial integrals into one canonical second-kind integral."""
    # BEL (1.3), with j = row + 1 and a zero x**(2*g+2) coefficient for the
    # present odd-degree model.
    j = row + 1
    return ctx.fsum(
        (power + 1 - j) * coefficients[power + 1 + j]
        * monomials[power] / 4
        for power in range(j, 2 * genus + 2 - j))


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
    """Check the generalized Legendre relation for full period matrices."""
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
    expected = 2 * ctx.pi * ctx.j * symplectic
    residual = ctx.norm(periods * symplectic * periods.T - expected)
    scale = max(ctx.one, ctx.norm(periods) ** 2, ctx.norm(expected))
    if residual > 100 * target_eps * scale:
        raise ValueError("failed to satisfy the generalized Legendre relation")


@defun
def hyperelliptic_periods(ctx, coefficients, method="auto",
                          second_kind=False):
    r"""
    Compute periods of a real odd-degree hyperelliptic curve.

    ``coefficients`` gives the coefficients of a polynomial :math:`P` in the
    ascending order used by :func:`~mpmath.polyval`, defining

    .. math::

        y^2 = P(x), \qquad \deg P = 2g+1.

    The polynomial must currently have distinct real roots. The roots are
    ordered automatically; they need not be supplied by the user.

    By default the result is ``(omega, omega_prime, tau)``, where ``omega``
    and ``omega_prime`` are the full first-kind a- and b-period matrices and
    ``tau = omega**-1 * omega_prime``. If ``second_kind=True``, the result is
    ``(omega, omega_prime, eta, eta_prime, tau, kappa)``. Here ``eta`` and
    ``eta_prime`` are the full periods of the associated canonical
    second-kind differentials and ``kappa = eta * omega**-1``.

    The second-kind convention follows Bernatska's sign and the algebraic
    differential basis of Buchstaber, Enolskii and Leykin. With full periods,
    the combined matrix satisfies the generalized Legendre relation with
    constant :math:`2\pi i`.

    See [BEL1997]_, particularly equation (1.3) and Lemma 1.1, and
    [Bernatska2026]_ for the full-period convention used here.

    ``method`` may be ``"auto"`` or ``"real"``. The selector anticipates a
    general complex-branch implementation; both values currently select the
    classical real-branch construction.

    """
    if method not in ("auto", "real"):
        raise ValueError("method must be 'auto' or 'real'")
    # Unary plus freezes this context constant at the caller's precision;
    # otherwise ctx.eps would follow the temporary guard precision below.
    target_eps = +ctx.eps
    quadrature_guard = 20
    # The higher monomials in second-kind differentials can be individually
    # much larger than their period sums. Extra guard bits protect the
    # cancellation; the symmetry and Legendre checks below still decide
    # whether they were sufficient for a particular curve.
    second_kind_cancellation_guard = 40 if second_kind else 0
    with ctx.extraprec(quadrature_guard + second_kind_cancellation_guard):
        coefficients = _real_hyperelliptic_coefficients(ctx, coefficients)
        roots = _real_hyperelliptic_roots(ctx, coefficients)
        genus = (len(coefficients) - 2) // 2
        monomial_count = 2 * genus + 1 if second_kind else genus
        intervals = [
            _real_branch_integrals(
                ctx, roots, coefficients[-1], interval, monomial_count)
            for interval in range(2 * genus)
        ]
        omega = ctx.matrix(genus)
        omega_prime = ctx.matrix(genus)
        for row in range(genus):
            for column in range(genus):
                omega[row, column] = 2 * intervals[2 * column][row]
                omega_prime[row, column] = 2 * ctx.fsum(
                    intervals[2 * edge + 1][row]
                    for edge in range(column, genus))
        inverse_omega = ctx.inverse(omega)
        tau = inverse_omega * omega_prime
        # Symmetry is a sensitive independent check on the paths, signs and
        # numerical integration.  Do not average away a material discrepancy.
        _symmetrize_period_matrix(
            ctx, tau, target_eps, "period matrix")
        ctx._rtheta_tau_data(_matrix_tuple(tau))
        if second_kind:
            # Pad the missing degree 2*g+2 coefficient of the odd model.
            second_coefficients = coefficients + (ctx.zero,)
            eta = ctx.matrix(genus)
            eta_prime = ctx.matrix(genus)
            for row in range(genus):
                second_intervals = [
                    _second_kind_interval(
                        ctx, second_coefficients, values, row, genus)
                    for values in intervals
                ]
                for column in range(genus):
                    eta[row, column] = 2 * second_intervals[2 * column]
                    eta_prime[row, column] = 2 * ctx.fsum(
                        second_intervals[2 * edge + 1]
                        for edge in range(column, genus))
            kappa = eta * inverse_omega
            _symmetrize_period_matrix(
                ctx, kappa, target_eps, "kappa matrix")
            _validate_legendre_relation(
                ctx, omega, omega_prime, eta, eta_prime, target_eps)
    if second_kind:
        return (+omega, +omega_prime, +eta, +eta_prime,
                +tau, +kappa)
    return +omega, +omega_prime, +tau
