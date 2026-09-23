"""Polynomial representation, evaluation, and sheet solving."""

from math import comb

from ._records import _PlaneCurve

# Curve representation and evaluation
# -----------------------------------

def _polynomial_trim(ctx, coefficients, tolerance=None):
    """Trim an ascending univariate polynomial."""
    coefficients = list(coefficients)
    if tolerance is None:
        while len(coefficients) > 1 and not coefficients[-1]:
            coefficients.pop()
    else:
        scale = max([ctx.one] + [abs(value) for value in coefficients])
        while (len(coefficients) > 1
               and abs(coefficients[-1]) <= tolerance * scale):
            coefficients.pop()
    return tuple(coefficients or (ctx.zero,))


def _polynomial_add(ctx, left, right, right_scale=1):
    size = max(len(left), len(right))
    result = [ctx.zero] * size
    for index in range(size):
        result[index] = (
            (left[index] if index < len(left) else ctx.zero)
            + right_scale * (
                right[index] if index < len(right) else ctx.zero))
    return _polynomial_trim(ctx, result)


def _polynomial_multiply(ctx, left, right):
    result = [ctx.zero] * (len(left) + len(right) - 1)
    for left_degree, left_coefficient in enumerate(left):
        for right_degree, right_coefficient in enumerate(right):
            result[left_degree + right_degree] += (
                left_coefficient * right_coefficient)
    return _polynomial_trim(ctx, result)


def _polynomial_derivative(ctx, coefficients):
    """Differentiate an ascending univariate polynomial."""
    if len(coefficients) <= 1:
        return (ctx.zero,)
    return _polynomial_trim(ctx, tuple(
        degree * coefficients[degree]
        for degree in range(1, len(coefficients))))


def _polynomial_divmod(ctx, dividend, divisor, tolerance=None):
    """Divide ascending polynomials, trimming numerical roundoff."""
    if tolerance is None:
        tolerance = 100 * ctx.eps
    dividend = list(_polynomial_trim(ctx, dividend, tolerance))
    divisor = _polynomial_trim(ctx, divisor, tolerance)
    if len(divisor) == 1 and not divisor[0]:
        raise ZeroDivisionError("polynomial division by zero")
    quotient = [ctx.zero] * max(1, len(dividend) - len(divisor) + 1)
    while len(dividend) >= len(divisor):
        degree = len(dividend) - len(divisor)
        coefficient = dividend[-1] / divisor[-1]
        quotient[degree] += coefficient
        for index, value in enumerate(divisor):
            dividend[index + degree] -= coefficient * value
        dividend = list(_polynomial_trim(ctx, dividend, tolerance))
        if len(dividend) == 1 and abs(dividend[0]) <= tolerance:
            dividend[0] = ctx.zero
            break
    return (_polynomial_trim(ctx, quotient, tolerance),
            _polynomial_trim(ctx, dividend, tolerance))


def _polynomial_exact_quotient(ctx, dividend, divisor):
    tolerance = ctx.sqrt(ctx.eps)
    quotient, remainder = _polynomial_divmod(
        ctx, dividend, divisor, tolerance=tolerance)
    scale = max([ctx.one] + [abs(value) for value in dividend])
    if max(abs(value) for value in remainder) > tolerance * scale:
        raise ValueError("polynomial Bareiss division was not exact")
    return quotient


def _polynomial_monic(ctx, coefficients, tolerance):
    coefficients = _polynomial_trim(ctx, coefficients, tolerance)
    if len(coefficients) == 1 and not coefficients[0]:
        return coefficients
    leading = coefficients[-1]
    return tuple(value / leading for value in coefficients)


def _polynomial_gcd(ctx, left, right):
    """Return a numerical monic GCD of ascending polynomials."""
    tolerance = ctx.sqrt(ctx.eps)
    left = _polynomial_monic(ctx, left, tolerance)
    right = _polynomial_monic(ctx, right, tolerance)
    while not (len(right) == 1 and not right[0]):
        unused, remainder = _polynomial_divmod(
            ctx, left, right, tolerance=tolerance)
        scale = max([ctx.one] + [abs(value) for value in left])
        if max(abs(value) for value in remainder) <= tolerance * scale:
            remainder = (ctx.zero,)
        left, right = right, _polynomial_monic(
            ctx, remainder, tolerance)
    return left


def _polynomial_squarefree_part(ctx, coefficients):
    """Remove repeated factors from a univariate polynomial."""
    tolerance = ctx.sqrt(ctx.eps)
    coefficients = _polynomial_trim(ctx, coefficients, tolerance)
    derivative = _polynomial_derivative(ctx, coefficients)
    divisor = _polynomial_gcd(ctx, coefficients, derivative)
    quotient, remainder = _polynomial_divmod(
        ctx, coefficients, divisor, tolerance=tolerance)
    scale = max([ctx.one] + [abs(value) for value in coefficients])
    if max(abs(value) for value in remainder) > tolerance * scale:
        raise ValueError("failed to form squarefree critical polynomial")
    return _polynomial_monic(ctx, quotient, tolerance)


def _polynomial_determinant(ctx, matrix):
    """Return a polynomial-matrix determinant by Bareiss elimination."""
    matrix = [[tuple(entry) for entry in row] for row in matrix]
    size = len(matrix)
    if not size or any(len(row) != size for row in matrix):
        raise ValueError("polynomial determinant requires a square matrix")
    sign = 1
    previous = (ctx.one,)
    for pivot_index in range(size - 1):
        pivot_row = next((row for row in range(pivot_index, size)
                          if any(matrix[row][pivot_index])), None)
        if pivot_row is None:
            return (ctx.zero,)
        if pivot_row != pivot_index:
            matrix[pivot_index], matrix[pivot_row] = (
                matrix[pivot_row], matrix[pivot_index])
            sign = -sign
        pivot = matrix[pivot_index][pivot_index]
        for row in range(pivot_index + 1, size):
            for column in range(pivot_index + 1, size):
                numerator = _polynomial_add(
                    ctx,
                    _polynomial_multiply(ctx, pivot, matrix[row][column]),
                    _polynomial_multiply(
                        ctx, matrix[row][pivot_index],
                        matrix[pivot_index][column]),
                    right_scale=-1)
                matrix[row][column] = (
                    numerator if pivot_index == 0 else
                    _polynomial_exact_quotient(ctx, numerator, previous))
        previous = pivot
    determinant = matrix[-1][-1]
    if sign < 0:
        determinant = tuple(-value for value in determinant)
    return _polynomial_trim(ctx, determinant)

def _prepare_plane_curve(ctx, coefficients):
    """Return a validated sparse bivariate polynomial.

    ``coefficients`` maps ``(x_power, y_power)`` pairs to numeric
    coefficients.  The representation is deliberately private while the
    numerical algorithms establish which public input forms would be useful.
    """
    try:
        items = coefficients.items()
    except AttributeError:
        raise ValueError(
            "coefficients must map (x_power, y_power) to numbers")

    terms = []
    for powers, coefficient in items:
        if (not isinstance(powers, tuple) or len(powers) != 2
                or any(not isinstance(power, int) or power < 0
                       for power in powers)):
            raise ValueError(
                "polynomial powers must be pairs of nonnegative integers")
        try:
            coefficient = ctx.convert(coefficient)
        except (TypeError, ValueError):
            raise ValueError("polynomial coefficients must be numbers")
        if not ctx.isfinite(coefficient):
            raise ValueError("polynomial coefficients must be finite")
        if coefficient:
            terms.append((powers[0], powers[1], coefficient))

    if not terms:
        raise ValueError("the plane curve polynomial must be nonzero")
    terms.sort(key=lambda term: (term[1], term[0]))
    x_degree = max(term[0] for term in terms)
    y_degree = max(term[1] for term in terms)
    if not y_degree:
        raise ValueError("the plane curve must depend on y")
    return _PlaneCurve(tuple(terms), x_degree, y_degree)


def _plane_curve_resultant_y(ctx, curve):
    """Return Res_y(F,F_y) as ascending coefficients in x."""
    degree = curve.y_degree
    coefficients = [[ctx.zero] * (curve.x_degree + 1)
                    for unused in range(degree + 1)]
    for x_power, y_power, coefficient in curve.terms:
        coefficients[y_power][x_power] = coefficient
    coefficients = [
        _polynomial_trim(ctx, polynomial)
        for polynomial in reversed(coefficients)
    ]
    derivative = [
        tuple((degree - index) * value for value in polynomial)
        for index, polynomial in enumerate(coefficients[:-1])
    ]
    derivative_degree = degree - 1
    size = degree + derivative_degree
    zero = (ctx.zero,)
    matrix = []
    for shift in range(derivative_degree):
        row = [zero] * size
        row[shift:shift + degree + 1] = coefficients
        matrix.append(row)
    for shift in range(degree):
        row = [zero] * size
        row[shift:shift + derivative_degree + 1] = derivative
        matrix.append(row)
    return _polynomial_determinant(ctx, matrix)


def _plane_curve_critical_values(ctx, curve):
    """Return distinct finite candidates from the y-resultant."""
    failure = None
    for extra_precision in (30, 60, 120):
        no_finite_polynomial = False
        try:
            with ctx.extraprec(extra_precision):
                resultant = _plane_curve_resultant_y(ctx, curve)
                if len(resultant) <= 1:
                    no_finite_polynomial = True
                else:
                    squarefree = _polynomial_squarefree_part(ctx, resultant)
                    roots = tuple(ctx.polyroots(
                        squarefree, maxsteps=1000, error=False))
                    scale = max([ctx.one] + [abs(root) for root in roots])
                    tolerance = ctx.sqrt(ctx.eps) * scale
                    distinct = []
                    for root in sorted(roots, key=lambda value: (
                            ctx.re(value), ctx.im(value))):
                        if not distinct or min(
                                abs(root - value) for value in distinct
                                ) > tolerance:
                            distinct.append(root)
        except (ValueError, ZeroDivisionError, ctx.NoConvergence) as exc:
            failure = exc
            continue
        if no_finite_polynomial:
            raise ValueError(
                "the projection has no finite critical polynomial")
        return (tuple(+value for value in distinct),
                tuple(+value for value in resultant))
    raise ValueError("failed to resolve finite critical values") from failure


def _evaluate_plane_polynomial(ctx, curve, x, y):
    """Evaluate a prepared bivariate polynomial at ``(x, y)``."""
    return ctx.fsum(
        coefficient * x ** x_power * y ** y_power
        for x_power, y_power, coefficient in curve.terms)


def _evaluate_plane_derivative(ctx, curve, x, y, variable):
    """Evaluate the first partial derivative with respect to ``x`` or ``y``."""
    if variable == "x":
        return ctx.fsum(
            x_power * coefficient * x ** (x_power - 1) * y ** y_power
            for x_power, y_power, coefficient in curve.terms if x_power)
    if variable == "y":
        return ctx.fsum(
            y_power * coefficient * x ** x_power * y ** (y_power - 1)
            for x_power, y_power, coefficient in curve.terms if y_power)
    raise ValueError("variable must be 'x' or 'y'")


def _plane_polynomial_y_coefficients(ctx, curve, x):
    """Return ascending coefficients of ``F(x, y)`` as a polynomial in y."""
    coefficients = [ctx.zero] * (curve.y_degree + 1)
    for x_power, y_power, coefficient in curve.terms:
        coefficients[y_power] += coefficient * x ** x_power
    if not coefficients[-1]:
        raise ValueError(
            "the projection degree drops at the requested x value")
    return coefficients


def _reciprocal_y_plane_curve(ctx, curve):
    """Return ``y**degree * F(x, 1/y)`` as a prepared plane curve."""
    degree = curve.y_degree
    return _prepare_plane_curve(ctx, {
        (x_power, degree - y_power): coefficient
        for x_power, y_power, coefficient in curve.terms
    })


def _monomial_plane_curve_chart(ctx, curve, x_power, y_power):
    """Return a polynomial chart for ``x=t**x_power, y=t**y_power*w``.

    Negative powers are cleared by dividing by the smallest resulting power
    of ``t``.  The transformation is useful for weighted local charts at zero
    and infinity; it does not claim to normalize a singular chart.
    """
    if (not isinstance(x_power, int) or not x_power
            or not isinstance(y_power, int)):
        raise ValueError("chart powers must be integers and x_power nonzero")
    weighted = [
        (x_power * x_degree + y_power * y_degree,
         y_degree, coefficient)
        for x_degree, y_degree, coefficient in curve.terms
    ]
    minimum = min(term[0] for term in weighted)
    coefficients = {}
    for t_power, w_power, coefficient in weighted:
        powers = t_power - minimum, w_power
        coefficients[powers] = coefficients.get(powers, ctx.zero) + coefficient
    return _prepare_plane_curve(ctx, coefficients)


def _blow_up_plane_curve_y(ctx, curve, center, y_power=1):
    """Return the strict transform for ``y=center+x**y_power*w``."""
    if not isinstance(y_power, int) or y_power < 1:
        raise ValueError("y_power must be a positive integer")
    center = ctx.convert(center)
    coefficients = {}
    for x_degree, degree, coefficient in curve.terms:
        for w_degree in range(degree + 1):
            t_power = x_degree + y_power * w_degree
            value = (coefficient * comb(degree, w_degree)
                     * center ** (degree - w_degree))
            powers = t_power, w_degree
            coefficients[powers] = coefficients.get(powers, ctx.zero) + value
    coefficients = {powers: coefficient
                    for powers, coefficient in coefficients.items()
                    if coefficient}
    minimum = min(x_power for x_power, unused in coefficients)
    return _prepare_plane_curve(ctx, {
        (x_power - minimum, w_power): coefficient
        for (x_power, w_power), coefficient in coefficients.items()
    })


def _finite_plane_curve_sheets(ctx, curve, x):
    """Return finite roots in a fibre where the chart degree may drop."""
    x = ctx.convert(x)
    if not ctx.isfinite(x):
        raise ValueError("the projection point must be finite")
    coefficients = [ctx.zero] * (curve.y_degree + 1)
    for x_power, y_power, coefficient in curve.terms:
        coefficients[y_power] += coefficient * x ** x_power
    while coefficients and not coefficients[-1]:
        coefficients.pop()
    if len(coefficients) < 2:
        return ()
    return tuple(ctx.polyroots(
        coefficients, maxsteps=100, cleanup=False, extraprec=20))


# Root matching and sheet fibres
# ------------------------------

def _minimum_cost_assignment(ctx, references, candidates):
    """Match two equally sized point sets with minimum total displacement."""
    if len(references) != len(candidates):
        raise ValueError("root sets must have equal sizes")
    # Dynamic programming is compact and deterministic for the modest cover
    # degrees targeted initially.  It avoids the factorial behaviour of
    # enumerating every permutation and can later be replaced independently.
    states = {0: (ctx.zero, ())}
    for reference in references:
        next_states = {}
        for mask, (cost, assignment) in states.items():
            for index, candidate in enumerate(candidates):
                bit = 1 << index
                if mask & bit:
                    continue
                next_mask = mask | bit
                next_value = (cost + abs(reference - candidate),
                              assignment + (index,))
                previous = next_states.get(next_mask)
                if previous is None or next_value[0] < previous[0]:
                    next_states[next_mask] = next_value
        states = next_states
    return states[(1 << len(references)) - 1][1]


def _minimum_root_separation(ctx, roots):
    """Return the minimum pairwise distance in a fibre."""
    if len(roots) < 2:
        return ctx.inf
    return min(abs(roots[right] - roots[left])
               for right in range(1, len(roots))
               for left in range(right))


def _plane_curve_sheets(ctx, curve, x, roots_init=None):
    """Return all y-sheets of a prepared curve above a regular x value."""
    x = ctx.convert(x)
    if not ctx.isfinite(x):
        raise ValueError("the projection point must be finite")
    coefficients = _plane_polynomial_y_coefficients(ctx, curve, x)
    if curve.y_degree == 1:
        return (-coefficients[0] / coefficients[1],)
    if curve.y_degree == 2:
        constant, linear, quadratic = coefficients
        discriminant = linear**2 - 4 * quadratic * constant
        root = ctx.sqrt(discriminant)
        denominator = 2 * quadratic
        return ((-linear - root) / denominator,
                (-linear + root) / denominator)
    roots = ctx.polyroots(
        coefficients, maxsteps=100, cleanup=False, extraprec=20,
        roots_init=roots_init)
    return tuple(roots)


def _ordered_plane_curve_sheets(ctx, curve, x):
    """Return a deterministic initial ordering of the sheets above x."""
    roots = _plane_curve_sheets(ctx, curve, x)
    return tuple(sorted(roots, key=lambda root: (ctx.re(root), ctx.im(root))))
