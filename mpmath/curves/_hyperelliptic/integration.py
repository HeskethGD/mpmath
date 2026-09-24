"""Specialized quadrature for Baker-marked hyperelliptic curves."""

from .model import _infinity_direction

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



def _second_kind_interval(ctx, coefficients, monomials, row, genus):
    """Combine monomial integrals into one canonical second-kind integral."""
    # BEL (1.3), with j = row + 1. The caller supplies coefficients through
    # degree 2*g+2, padding that coefficient with zero in odd degree.
    j = row + 1
    return ctx.fsum(
        (power + 1 - j) * coefficients[power + 1 + j]
        * monomials[power] / 4
        for power in range(j, 2 * genus + 2 - j))



