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


def _real_branch_integrals(ctx, roots, leading, interval, genus):
    """Integrate all first-kind differentials between adjacent roots."""
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
                 for k in range(genus))


@defun
def hyperelliptic_periods(ctx, coefficients, method="auto"):
    r"""
    Compute first-kind periods of a real odd-degree hyperelliptic curve.

    ``coefficients`` gives the coefficients of a polynomial :math:`P` in the
    ascending order used by :func:`~mpmath.polyval`, defining

    .. math::

        y^2 = P(x), \qquad \deg P = 2g+1.

    The polynomial must currently have distinct real roots. The roots are
    ordered automatically; they need not be supplied by the user. The result
    is ``(omega, omega_prime, tau)``, where ``omega`` and ``omega_prime`` are
    the full first-kind a- and b-period matrices and
    ``tau = omega**-1 * omega_prime``.

    ``method`` may be ``"auto"`` or ``"real"``. The selector anticipates a
    general complex-branch implementation; both values currently select the
    classical real-branch construction.

    """
    if method not in ("auto", "real"):
        raise ValueError("method must be 'auto' or 'real'")
    target_eps = ctx.eps
    with ctx.extraprec(20):
        coefficients = _real_hyperelliptic_coefficients(ctx, coefficients)
        roots = _real_hyperelliptic_roots(ctx, coefficients)
        genus = (len(coefficients) - 2) // 2
        intervals = [
            _real_branch_integrals(
                ctx, roots, coefficients[-1], interval, genus)
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
        tau = ctx.inverse(omega) * omega_prime
        # Symmetry is a sensitive independent check on the paths, signs and
        # numerical integration.  Do not average away a material discrepancy.
        for row in range(genus):
            for column in range(row):
                difference = abs(tau[row, column] - tau[column, row])
                scale = max(ctx.one, abs(tau[row, column]),
                            abs(tau[column, row]))
                if difference > 100 * target_eps * scale:
                    raise ValueError(
                        "failed to compute a symmetric period matrix")
                entry = (tau[row, column] + tau[column, row]) / 2
                tau[row, column] = tau[column, row] = entry
        ctx._rtheta_tau_data(_matrix_tuple(tau))
    return +omega, +omega_prime, +tau
