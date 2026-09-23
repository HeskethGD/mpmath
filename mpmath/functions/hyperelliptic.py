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
# Abel maps reuse these adjacent-root integrals as a path backbone. Odd-degree
# curves add one reciprocal-coordinate path from infinity; each affine target
# is joined to an unobstructed branch point by x = e + (x_target-e)*t**2.
# Factorwise square-root continuation along only that last segment fixes a
# reference sheet, which is matched against the y-coordinate supplied by the
# caller. Optional reduction resolves the result in the real full-period
# lattice and selects a centered fundamental-parallelotope representative.
# Incomplete second-kind integrals follow the same paths. At the unique
# odd-degree point at infinity, their Laurent principal parts in the
# reciprocal local parameter are removed before quadrature; even-degree maps
# retain the finite base point e0. Lattice reduction applies one shared cycle
# shift to the first- and second-kind vectors.
#
# The associated second-kind differentials and period conventions are those
# of Buchstaber, Enolskii and Leykin, "Hyperelliptic Kleinian Functions and
# Applications". Thus omega, omega_prime, eta and eta_prime are half-period
# matrices, with 2*eta = -integral_a(dr). The Kleinian exponential matrix is
# kappa = eta*omega**-1 and the Legendre constant is -pi*i/2.

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


def _prepare_hyperelliptic_curve(ctx, coefficients, method):
    """Normalize a curve and select its real or complex path construction."""
    if method not in ("auto", "real", "complex"):
        raise ValueError("method must be 'auto', 'real' or 'complex'")
    coefficients = _hyperelliptic_coefficients(ctx, coefficients)
    roots, root_tolerance = _hyperelliptic_roots(ctx, coefficients)
    real_data = (
        not any(ctx.im(value) for value in coefficients)
        and not any(abs(ctx.im(root)) > root_tolerance for root in roots)
    )
    use_real_method = method == "real" or (method == "auto" and real_data)
    if use_real_method:
        coefficients, roots = _real_hyperelliptic_data(
            ctx, coefficients, roots, root_tolerance)
    genus = (len(coefficients) - 2) // 2
    even_degree = not (len(coefficients) - 1) % 2
    return (coefficients, roots, root_tolerance, use_real_method,
            genus, even_degree)


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


def _hyperelliptic_intervals(ctx, coefficients, roots, use_real_method,
                             count):
    """Return adjacent branch integrals and the b-period orientation."""
    if use_real_method:
        intervals = [
            _real_branch_integrals(
                ctx, roots, coefficients[-1], interval, count)
            for interval in range(len(roots) - 1)
        ]
        b_sign = ctx.one
    else:
        intervals = _complex_branch_integrals(
            ctx, roots, coefficients[-1], count)
        b_sign = -ctx.one
    return intervals, b_sign


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


def _infinity_direction(ctx, roots):
    """Choose a deterministic root-free ray leaving the terminal root."""
    terminal = roots[-1]
    initial = terminal - roots[-2]
    clearance = 100 * ctx.sqrt(ctx.eps)
    count = 2 * len(roots) + 1
    for step in range(count):
        direction = initial * ctx.exp(ctx.j * ctx.pi * step / count)
        blocked = False
        for root in roots[:-1]:
            ratio = (root - terminal) / direction
            if (ctx.re(ratio) > 0
                    and abs(ctx.im(ratio)) <= clearance * max(
                        ctx.one, abs(ratio))):
                blocked = True
                break
        if not blocked:
            return direction
    raise ValueError("failed to choose a root-free path from infinity")


def _infinity_branch_integrals(ctx, roots, leading, count):
    """Integrate first-kind differentials from infinity to the last root."""
    terminal = roots[-1]
    direction = _infinity_direction(ctx, roots)
    root_direction = ctx.sqrt(direction)
    root_leading = ctx.sqrt(leading)
    root_count = len(roots)
    ratios = tuple((terminal - root) / direction for root in roots)
    endpoint_product = ctx.fprod(
        ctx.sqrt(ratio) for ratio in ratios[:-1])

    def integrand(unit, power):
        if not unit:
            exponent = root_count - 3 - 2 * power
            if exponent:
                return ctx.zero
            return (-2 * direction * direction ** power
                    / (root_leading * root_direction ** root_count))
        if unit == 1:
            return (-2 * direction * terminal ** power
                    / (root_leading * root_direction ** root_count
                       * endpoint_product))
        parameter = unit / (1 - unit)
        product = ctx.fprod(
            ctx.sqrt(1 + ratio * parameter ** 2) for ratio in ratios)
        exponent = root_count - 3 - 2 * power
        numerator = (-2 * direction
                     * (terminal * parameter ** 2 + direction) ** power
                     * parameter ** exponent)
        derivative = 1 / (1 - unit) ** 2
        return (numerator * derivative
                / (root_leading * root_direction ** root_count * product))

    return tuple(ctx.quad(
        lambda unit, power=power: integrand(unit, power), [0, 1])
        for power in range(count))


def _inverse_sqrt_product_series(ctx, ratios, order):
    """Expand the inverse square root of a product of linear factors."""
    result = [ctx.one] + [ctx.zero] * order
    for ratio in ratios:
        factor = [ctx.one]
        for degree in range(1, order + 1):
            factor.append(
                -factor[-1] * (2 * degree - 1) * ratio / (2 * degree))
        result = [ctx.fsum(
            result[index] * factor[degree - index]
            for index in range(degree + 1))
            for degree in range(order + 1)]
    return result


def _infinity_second_kind_integrals(ctx, coefficients, roots, genus):
    """Regularize second-kind integrals from infinity to the last root."""
    terminal = roots[-1]
    direction = _infinity_direction(ctx, roots)
    root_direction = ctx.sqrt(direction)
    root_count = len(roots)
    root_leading = ctx.sqrt(coefficients[root_count])
    ratios = tuple((terminal - root) / direction for root in roots)
    denominator = root_leading * root_direction ** root_count
    common_constant = -2 * direction / denominator

    # With x = terminal + direction/t**2, every transformed differential is
    # s**(-pole_order) H(s) dt, s=t**2, where H is analytic at zero. Removing
    # the first ``pole_order`` Taylor coefficients gives the Hadamard finite
    # part at infinity. Keeping |ratio*s| <= 1/16 makes the short Taylor tail
    # used on the first subinterval gain at least four bits per term; the
    # genus-dependent guard covers the product of all root factors.
    pole_order = genus + 1
    ratio_scale = max(abs(ratio) for ratio in ratios)
    series_endpoint = min(
        ctx.one, 1 / (4 * ctx.sqrt(ratio_scale)))
    series_bits_per_term = 4
    series_guard_terms = 12 + genus
    series_order = pole_order + max(
        series_guard_terms,
        (ctx.prec + series_bits_per_term - 1) // series_bits_per_term)
    inverse_root_series = _inverse_sqrt_product_series(
        ctx, ratios, series_order)
    results = []
    for row in range(genus):
        numerator = [ctx.zero] * (2 * genus + 1)
        j = row + 1
        for power in range(j, 2 * genus + 2 - j):
            coefficient = ((power + 1 - j)
                           * coefficients[power + 1 + j] / 4)
            for degree in range(power + 1):
                index = genus - 1 - power + degree + pole_order
                numerator[index] += (
                    coefficient * ctx.binomial(power, degree)
                    * terminal ** degree * direction ** (power - degree))
        analytic_series = [
            common_constant * ctx.fsum(
                numerator[index] * inverse_root_series[degree - index]
                for index in range(min(degree, len(numerator) - 1) + 1))
            for degree in range(series_order + 1)
        ]
        principal_part = ctx.fsum(
            analytic_series[degree]
            * series_endpoint ** (2 * (degree - pole_order) + 1)
            / (2 * (degree - pole_order) + 1)
            for degree in range(pole_order)
        )
        regular_part = ctx.fsum(
            analytic_series[degree]
            * series_endpoint ** (2 * (degree - pole_order) + 1)
            / (2 * (degree - pole_order) + 1)
            for degree in range(pole_order, series_order + 1)
        )

        def integrand(parameter):
            if ctx.isinf(parameter):
                return ctx.zero
            square = parameter ** 2
            x = terminal + direction / square
            product = ctx.fprod(
                ctx.sqrt(1 + ratio * square) for ratio in ratios)
            polynomial = ctx.fsum(
                (power + 1 - j) * coefficients[power + 1 + j]
                * x ** power / 4
                for power in range(j, 2 * genus + 2 - j))
            return (common_constant * parameter ** (root_count - 3)
                    * polynomial / product)

        remainder = ctx.quad(
            integrand, [series_endpoint, ctx.one, ctx.inf])
        results.append(principal_part + regular_part + remainder)
    return tuple(results)


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


def _normalise_abel_targets(ctx, target):
    """Normalize one affine point or a sequence of affine points."""
    try:
        items = tuple(target)
    except TypeError:
        raise ValueError(
            "target must be an affine point (x, y) or a sequence of points")
    if not items:
        return ()

    def point(value):
        try:
            entries = tuple(value)
        except TypeError:
            raise ValueError("each target point must be a pair (x, y)")
        if len(entries) != 2:
            raise ValueError("each target point must be a pair (x, y)")
        try:
            x, y = (ctx.convert(entry) for entry in entries)
        except (TypeError, ValueError):
            raise ValueError("target coordinates must be numbers")
        if not ctx.isfinite(x) or not ctx.isfinite(y):
            raise ValueError("target coordinates must be finite")
        return x, y

    if len(items) == 2:
        try:
            x, y = (ctx.convert(item) for item in items)
        except (TypeError, ValueError):
            pass
        else:
            if not ctx.isfinite(x) or not ctx.isfinite(y):
                raise ValueError("target coordinates must be finite")
            return ((x, y),)
    return tuple(point(value) for value in items)


def _evaluate_polynomial(ctx, coefficients, value):
    """Evaluate an ascending coefficient vector by Horner's rule."""
    result = ctx.zero
    for coefficient in reversed(coefficients):
        result = result * value + coefficient
    return result


def _target_branch_index(ctx, x, y, roots, target_eps):
    """Return the index if an affine target is a branch point."""
    scale = max([ctx.one, abs(x)] + [abs(root) for root in roots])
    tolerance = 100 * target_eps * scale
    for index, root in enumerate(roots):
        if abs(x - root) <= tolerance and abs(y) <= tolerance:
            return index
    return None


def _admissible_branch_vertex(ctx, target, roots, target_eps):
    """Choose the nearest branch point with an unobstructed final segment."""
    path_tolerance = 100 * ctx.sqrt(target_eps)
    candidates = []
    for index, branch in enumerate(roots):
        difference = target - branch
        obstructed = False
        for other_index, other in enumerate(roots):
            if other_index == index:
                continue
            ratio = (other - branch) / difference
            if (0 < ctx.re(ratio) < 1
                    and abs(ctx.im(ratio)) <= path_tolerance * max(
                        ctx.one, abs(ratio))):
                obstructed = True
                break
        if not obstructed:
            candidates.append((abs(difference), index))
    if not candidates:
        raise ValueError("failed to find a branch-point path to target")
    return min(candidates)[1]


def _branch_target_integrals(ctx, roots, leading, branch_index, target,
                             supplied_y, count, target_eps):
    """Integrate from one branch point to a selected affine lift."""
    branch = roots[branch_index]
    difference = target - branch
    other_roots = tuple(
        root for index, root in enumerate(roots) if index != branch_index)
    references = tuple(branch - root for root in other_roots)
    ratios = tuple(difference / reference for reference in references)
    denominator = (ctx.sqrt(leading) * ctx.sqrt(difference)
                   * ctx.fprod(ctx.sqrt(reference)
                               for reference in references))

    def integrand(parameter, power):
        x = branch + difference * parameter ** 2
        continuation = ctx.fprod(
            ctx.sqrt(1 + ratio * parameter ** 2) for ratio in ratios)
        return 2 * difference * x ** power / (denominator * continuation)

    continued_y = denominator * ctx.fprod(
        ctx.sqrt(1 + ratio) for ratio in ratios)
    sheet_tolerance = 1000 * target_eps * max(
        ctx.one, abs(supplied_y), abs(continued_y))
    if abs(continued_y - supplied_y) <= sheet_tolerance:
        sign = ctx.one
    elif abs(continued_y + supplied_y) <= sheet_tolerance:
        sign = -ctx.one
    else:
        raise ValueError("failed to select the target's square-root sheet")
    return tuple(sign * ctx.quad(
        lambda parameter, power=power: integrand(parameter, power), [0, 1])
        for power in range(count))


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


def _second_kind_interval(ctx, coefficients, monomials, row, genus):
    """Combine monomial integrals into one canonical second-kind integral."""
    # BEL (1.3), with j = row + 1. The caller supplies coefficients through
    # degree 2*g+2, padding that coefficient with zero in odd degree.
    j = row + 1
    return ctx.fsum(
        (power + 1 - j) * coefficients[power + 1 + j]
        * monomials[power] / 4
        for power in range(j, 2 * genus + 2 - j))


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
    and ``omega_prime`` are the first-kind a- and b-half-period matrices and
    ``tau = omega**-1 * omega_prime``. Thus the corresponding complete
    periods are ``2*omega`` and ``2*omega_prime``. If ``second_kind=True``,
    the result is ``(omega, omega_prime, eta, eta_prime, tau, kappa)``.
    Here ``eta`` and ``eta_prime`` are the second-kind half-period matrices,
    defined with the classical minus sign, and
    ``kappa = eta * omega**-1``.

    In terms of the canonical first- and second-kind differentials,

    .. math::

        2\omega_{ij}=\oint_{a_j}du_i, \qquad
        2\omega'_{ij}=\oint_{b_j}du_i,

        2\eta_{ij}=-\oint_{a_j}dr_i, \qquad
        2\eta'_{ij}=-\oint_{b_j}dr_i.

    The second-kind differential basis and half-period convention follow
    Buchstaber, Enolskii and Leykin. The generalized Legendre relation is

    .. math::

        \mathcal P J \mathcal P^T=-\frac{\pi i}{2}J, \qquad
        \mathcal P=\begin{pmatrix}\omega&\omega'\\
        \eta&\eta'\end{pmatrix}, \qquad
        J=\begin{pmatrix}0&-I\\I&0\end{pmatrix}.

    See [BEL1997]_, particularly equation (1.3) and Lemma 1.1.

    ``method`` may be ``"auto"``, ``"real"`` or ``"complex"``. The default
    uses the real-axis construction when the coefficients and roots are real,
    and otherwise uses an ordered polygonal path through the complex branch
    points. The latter order is lexicographic by real and imaginary part.

    For the real-axis construction, the square-root sheet is continued from
    the right of all branch points. Moving left across each real root
    multiplies the continued square root by ``j``. Consequently, on successive
    real ovals where :math:`P(x)>0`, the selected value of :math:`y` may have
    alternating signs; it is not the positive principal square root chosen
    independently on every oval.

    """
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
            # BEL's formula includes degree 2*g+2. It is zero in odd degree.
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


@defun
def hyperelliptic_abel_map(ctx, coefficients, target, method="auto",
                           reduce=False, second_kind=False):
    r"""
    Evaluate the Abel map of points on a hyperelliptic curve.

    ``coefficients`` defines the curve :math:`y^2=P(x)` in the same
    ascending order accepted by :func:`~mpmath.hyperelliptic_periods`.
    ``target`` is either one affine point ``(x, y)`` or a sequence of such
    points representing an effective divisor. Repetition represents
    multiplicity, and an empty sequence returns the zero vector. Both
    coordinates are required: away from a branch point, the sign of ``y``
    selects the sheet.

    For a divisor :math:`D=P_1+\cdots+P_n`, this returns

    .. math::

        A(D)=\sum_{j=1}^n\int_{P_0}^{P_j}
        \left(\frac{dx}{y},\frac{x\,dx}{y},\ldots,
        \frac{x^{g-1}dx}{y}\right)^T.

    If ``second_kind=True``, the return value is ``(A, R)``, where ``A`` is
    the same Abel image and

    .. math::

        R(D)=\sum_{j=1}^n\int_{P_0}^{P_j}dr

    uses the canonical second-kind differential vector of [BEL1997]_,
    equation (1.3). At the odd-degree point at infinity these integrals mean
    their finite parts in a reciprocal local parameter. For even degree the
    base point is finite and no regularization is needed.

    The base point :math:`P_0` is the unique point at infinity for an
    odd-degree curve and the first ordered branch point for an even-degree
    curve. Coordinates have the same order as the rows of ``omega`` returned
    by :func:`~mpmath.hyperelliptic_periods`, so the result can be passed
    directly to the Kleinian functions without a Riemann-constant shift.

    ``method`` has the same ``"auto"``, ``"real"`` and ``"complex"``
    choices as :func:`~mpmath.hyperelliptic_periods`. The integration paths
    are deterministic but an Abel map is naturally defined only modulo the
    full period lattice. By default the path-dependent value is returned.
    If ``reduce=True``, full periods are subtracted to put its real lattice
    coordinates in the centered parallelotope :math:`[-1/2,1/2)^{2g}`.
    This is fundamental-cell reduction, not a closest-vector calculation.
    When second-kind values are requested, the same cycle shift is applied
    to both vectors. With the half-period conventions of
    :func:`~mpmath.hyperelliptic_periods`, subtracting
    :math:`2\omega m+2\omega'n` from ``A`` adds
    :math:`2\eta m+2\eta'n` to ``R``.

    **Example**

    Recover a point on a genus-one curve from its Abel image::

        >>> from mpmath import hyperelliptic_abel_map, kleinian_p
        >>> from mpmath import hyperelliptic_data, mp
        >>> coefficients = [0, -4, 0, 4]
        >>> omega, tau, kappa, characteristic = hyperelliptic_data(
        ...     coefficients)
        >>> u = mp.mpf('0.3')
        >>> x = mp.weierp(u, omega1=omega[0, 0],
        ...                  omega2=(omega*tau)[0, 0])
        >>> y = mp.weierpprime(u, omega1=omega[0, 0],
        ...                       omega2=(omega*tau)[0, 0])
        >>> image = hyperelliptic_abel_map(coefficients, (x, y))
        >>> mp.almosteq(kleinian_p(
        ...     image, omega, tau, kappa, (0, 0), characteristic), x)
        True

    """
    target_eps = +ctx.eps
    quadrature_guard = 20
    # Higher monomials and the regularized Laurent terms can cancel in the
    # canonical second-kind combinations, as they do for complete periods.
    second_kind_cancellation_guard = 40 if second_kind else 0
    with ctx.extraprec(quadrature_guard + second_kind_cancellation_guard):
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
    if second_kind:
        return +result, +second_result
    return +result


@defun
def hyperelliptic_data(ctx, coefficients, method="auto"):
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
    in the selected ordering in even degree. In the branch-point notation
    recorded explicitly by Bernatska, :math:`K` is the vector of Riemann
    constants, square brackets denote the corresponding half-integer
    characteristic, and
    :math:`[\varepsilon_k]` is the characteristic of the Abel image of the
    branch point :math:`e_k`. Thus

    .. math::

        [K] = \sum_{j=1}^{g} [\varepsilon_{2j}].

    It therefore depends on the cycle and base-point conventions and should
    not be combined with period matrices constructed in a different basis.
    The returned data are compatible with ``normalization="hyperelliptic"``
    in :func:`~mpmath.kleinian_sigma`, which selects the canonical
    Schur--Weierstrass normalization.

    See [Bernatska2026]_, equations (3.16) and (3.17).

    """
    data = hyperelliptic_periods(
        ctx, coefficients, method=method, second_kind=True)
    omega, unused_omega_prime, unused_eta, unused_eta_prime, tau, kappa = data
    characteristic = _hyperelliptic_characteristic(ctx, omega.rows)
    return omega, tau, kappa, characteristic
