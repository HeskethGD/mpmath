# Numerical period data for hyperelliptic curves
# ------------------------------------------------
#
# The implementation covers models
#
#     y**2 = P(x),  deg(P) = 2*g + 1 or 2*g + 2,
#
# with distinct branch points. The real-axis and ordered-polygon constructions
# use the classical Baker cycle arrangement written explicitly, for example,
# in Bernatska, "Computation of P-Functions on Plane Algebraic Curves",
# J. Exp. Math. 2 (2026), Section 3.
# In even degree the smallest root is the distinguished branch point e0 and
# the final cut joins the largest root to e0 through infinity; this shifts the
# finite interval indices by one relative to the odd-degree construction.
# Endpoint singularities are removed by a cosine parametrization; mpmath's
# existing adaptive quadrature then integrates smooth functions.
#
# The associated second-kind differentials are equation (1.3) of Buchstaber,
# Enolskii and Leykin, "Hyperelliptic Kleinian Functions and Applications".
# BEL write full periods as 2*omega and attach a minus sign to their eta
# integrals.  Here omega and eta are full periods and follow Bernatska's sign
# convention, for which kappa = eta*omega**-1 and the Legendre constant is
# +2*pi*i.

import itertools

from .functions import defun
from .riemann_theta import _matrix_tuple


def _hyperelliptic_coefficients(ctx, coefficients):
    """Validate polynomial coefficients in ascending order."""
    try:
        coefficients = tuple(ctx.convert(value) for value in coefficients)
    except (TypeError, ValueError):
        raise ValueError("coefficients must be a sequence of numbers")
    if not coefficients or not coefficients[-1]:
        raise ValueError("the leading coefficient must be nonzero")
    if any(not ctx.isfinite(value) for value in coefficients):
        raise ValueError("coefficients must be finite numbers")
    degree = len(coefficients) - 1
    if degree < 3:
        raise ValueError("the polynomial degree must be at least 3")
    return coefficients


def _hyperelliptic_roots(ctx, coefficients):
    """Return distinct roots ordered by real and then imaginary part."""
    roots = ctx.polyroots(coefficients, maxsteps=200, error=False)
    scale = max([ctx.one] + [abs(root) for root in roots])
    tolerance = ctx.sqrt(ctx.eps) * scale
    roots = sorted(roots, key=lambda root: (ctx.re(root), ctx.im(root)))
    if any(abs(right - left) <= tolerance
           for left, right in itertools.pairwise(roots)):
        raise ValueError("the polynomial must have distinct roots")
    return tuple(roots), tolerance


def _real_hyperelliptic_data(ctx, coefficients, roots, tolerance):
    """Return real coefficients and roots, rejecting nonreal data."""
    if any(ctx.im(value) for value in coefficients):
        raise ValueError("coefficients must be finite real numbers")
    if any(abs(ctx.im(root)) > tolerance for root in roots):
        raise ValueError("the polynomial must have only real roots")
    return (tuple(ctx.re(value) for value in coefficients),
            tuple(ctx.re(root) for root in roots))


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


def _complex_branch_data(ctx, roots, leading, interval, count):
    """Integrate along one complex segment and return endpoint branch data."""
    left = roots[interval]
    right = roots[interval + 1]
    midpoint = (left + right) / 2
    half_width = (right - left) / 2
    excluded = interval, interval + 1
    other_roots = tuple(
        root for index, root in enumerate(roots) if index not in excluded)
    references = tuple(midpoint - root for root in other_roots)
    reference_roots = tuple(ctx.sqrt(value) for value in references)
    paired_root = ctx.sqrt(-half_width ** 2)
    root_leading = ctx.sqrt(leading)

    def remaining_root(x):
        # Each quotient traces a segment centred at 1 which cannot cross the
        # negative real axis unless an excluded branch point lies on this
        # segment. This fixes a continuous square root without mutable state.
        return ctx.fprod(
            root * ctx.sqrt((x - branch) / reference)
            for branch, reference, root in zip(
                other_roots, references, reference_roots))

    ratio = half_width / paired_root

    def integrand(angle, power):
        x = midpoint - half_width * ctx.cos(angle)
        return ratio * x ** power / (root_leading * remaining_root(x))

    integrals = tuple(
        ctx.quad(lambda angle, k=k: integrand(angle, k), [0, ctx.pi])
        for k in range(count))
    endpoint_scale = 2 * root_leading * paired_root
    left_coefficient = endpoint_scale * remaining_root(left)
    right_coefficient = endpoint_scale * remaining_root(right)
    return integrals, left_coefficient, right_coefficient


def _complex_branch_integrals(ctx, roots, leading, count):
    """Continue a square-root sheet along the ordered branch-point path."""
    data = [
        _complex_branch_data(ctx, roots, leading, interval, count)
        for interval in range(len(roots) - 1)
    ]
    # This choice agrees with the established real-branch results. In even
    # degree the finite branch point e0 replaces infinity at the path origin.
    multiplier = -ctx.one if len(roots) % 2 == 0 else ctx.one
    multipliers = [multiplier]
    # Root finding and the endpoint comparison both lose sensitivity as
    # branch points approach one another. This matches the distinct-root
    # threshold used above while leaving the sign decision unambiguous.
    sheet_transport_tolerance = 100 * ctx.sqrt(ctx.eps)
    for interval in range(len(data) - 1):
        incoming = roots[interval] - roots[interval + 1]
        outgoing = roots[interval + 2] - roots[interval + 1]
        angle = ctx.arg(outgoing) - ctx.arg(incoming)
        if angle <= 0:
            angle += 2 * ctx.pi
        transport = (ctx.sqrt(abs(outgoing) / abs(incoming))
                     * ctx.exp(ctx.j * angle / 2))
        quotient = data[interval][2] * transport / data[interval + 1][1]
        if abs(quotient - 1) <= sheet_transport_tolerance:
            sign = ctx.one
        elif abs(quotient + 1) <= sheet_transport_tolerance:
            sign = -ctx.one
        else:
            raise ValueError("failed to continue the square-root sheet")
        multiplier *= sign
        multipliers.append(multiplier)
    return [tuple(multiplier * value for value in values)
            for multiplier, (values, unused_left, unused_right)
            in zip(multipliers, data)]


def _second_kind_interval(ctx, coefficients, monomials, row, genus):
    """Combine monomial integrals into one canonical second-kind integral."""
    # BEL (1.3), with j = row + 1. The caller supplies coefficients through
    # degree 2*g+2, padding that coefficient with zero in odd degree.
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


def _hyperelliptic_characteristic(ctx, genus):
    """Return the Riemann characteristic for the selected canonical basis."""
    half = ctx.convert(0.5)
    a = (half,) * genus
    b = tuple(half if (genus - index) & 1 else ctx.zero
              for index in range(genus))
    return a, b


@defun
def hyperelliptic_periods(ctx, coefficients, method="auto",
                          second_kind=False):
    r"""
    Compute periods of a hyperelliptic curve.

    ``coefficients`` gives the coefficients of a polynomial :math:`P` in the
    ascending order used by :func:`~mpmath.polyval`, defining

    .. math::

        y^2 = P(x), \qquad \deg P \in \{2g+1, 2g+2\}.

    The polynomial must have distinct roots. The roots are ordered
    automatically; they need not be supplied by the user.

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

    ``method`` may be ``"auto"``, ``"real"`` or ``"complex"``. The default
    uses the real-axis construction when the coefficients and roots are real,
    and otherwise uses an ordered polygonal path through the complex branch
    points. The latter order is lexicographic by real and imaginary part.

    """
    if method not in ("auto", "real", "complex"):
        raise ValueError("method must be 'auto', 'real' or 'complex'")
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
        coefficients = _hyperelliptic_coefficients(ctx, coefficients)
        roots, root_tolerance = _hyperelliptic_roots(ctx, coefficients)
        real_data = (
            not any(ctx.im(value) for value in coefficients)
            and not any(abs(ctx.im(root)) > root_tolerance for root in roots)
        )
        use_real_method = (
            method == "real" or (method == "auto" and real_data))
        if use_real_method:
            coefficients, roots = _real_hyperelliptic_data(
                ctx, coefficients, roots, root_tolerance)
        genus = (len(coefficients) - 2) // 2
        even_degree = not (len(coefficients) - 1) % 2
        # The first finite a-cycle starts at interval zero in odd degree and
        # interval one in even degree, where e0 is itself a finite root.
        cycle_offset = 1 if even_degree else 0
        monomial_count = 2 * genus + 1 if second_kind else genus
        if use_real_method:
            intervals = [
                _real_branch_integrals(
                    ctx, roots, coefficients[-1], interval, monomial_count)
                for interval in range(2 * genus + cycle_offset)
            ]
            b_sign = ctx.one
        else:
            intervals = _complex_branch_integrals(
                ctx, roots, coefficients[-1], monomial_count)
            b_sign = -ctx.one
        omega = ctx.matrix(genus)
        omega_prime = ctx.matrix(genus)
        for row in range(genus):
            for column in range(genus):
                omega[row, column] = (
                    2 * intervals[2 * column + cycle_offset][row])
                omega_prime[row, column] = 2 * b_sign * ctx.fsum(
                    intervals[2 * edge + 1 + cycle_offset][row]
                    for edge in range(column, genus))
        inverse_omega = ctx.inverse(omega)
        tau = inverse_omega * omega_prime
        # Symmetry is a sensitive independent check on the paths, signs and
        # numerical integration.  Do not average away a material discrepancy.
        _symmetrize_period_matrix(
            ctx, tau, target_eps, "period matrix")
        ctx._rtheta_tau_data(_matrix_tuple(tau))
        if second_kind:
            # BEL's formula includes degree 2*g+2. It is zero in odd degree.
            second_coefficients = coefficients
            if not even_degree:
                second_coefficients += (ctx.zero,)
            eta = ctx.matrix(genus)
            eta_prime = ctx.matrix(genus)
            for row in range(genus):
                second_intervals = [
                    _second_kind_interval(
                        ctx, second_coefficients, values, row, genus)
                    for values in intervals
                ]
                for column in range(genus):
                    eta[row, column] = (
                        2 * second_intervals[2 * column + cycle_offset])
                    eta_prime[row, column] = 2 * b_sign * ctx.fsum(
                        second_intervals[2 * edge + 1 + cycle_offset]
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


@defun
def hyperelliptic_kleinian_data(ctx, coefficients, method="auto"):
    r"""
    Construct the curve-dependent data required by Kleinian functions.

    This is a convenience interface for the hyperelliptic
    curves supported by :func:`~mpmath.hyperelliptic_periods`. It returns
    ``(omega, tau, kappa, characteristic)``, ready for use with
    :func:`~mpmath.kleinian_sigma`, :func:`~mpmath.kleinian_zeta`, and
    :func:`~mpmath.kleinian_p`.

    The characteristic is the vector of Riemann constants for the canonical
    cycle basis used by the period construction. The base point is the
    branch point at infinity in odd degree and the first finite branch point
    in the selected ordering in even degree. In Bernatska's notation it is

    .. math::

        [K] = \sum_{j=1}^{g} [\varepsilon_{2j}].

    It therefore depends on the cycle and base-point conventions and should
    not be combined with period matrices constructed in a different basis.
    The curve-dependent normalization constant of the sigma function is not
    included.

    See [Bernatska2026]_, equations (3.16) and (3.17).

    """
    data = hyperelliptic_periods(
        ctx, coefficients, method=method, second_kind=True)
    omega, unused_omega_prime, unused_eta, unused_eta_prime, tau, kappa = data
    characteristic = _hyperelliptic_characteristic(ctx, omega.rows)
    return omega, tau, kappa, characteristic
