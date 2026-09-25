"""Newton-polygon first-kind differentials for nondegenerate plane curves."""

from math import gcd

from .polynomial import _polynomial_gcd


def _cross(origin, left, right):
    """Return the oriented area of three exponent points."""
    return ((left[0] - origin[0]) * (right[1] - origin[1])
            - (left[1] - origin[1]) * (right[0] - origin[0]))


def _newton_polygon(curve):
    """Return the counterclockwise convex hull of the polynomial support."""
    points = sorted((x, y) for x, y, unused in curve.terms)
    if len(points) < 3:
        raise ValueError("the Newton polygon has no two-dimensional interior")
    lower = []
    for point in points:
        while len(lower) > 1 and _cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(points):
        while len(upper) > 1 and _cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    polygon = tuple(lower[:-1] + upper[:-1])
    if len(polygon) < 3:
        raise ValueError("the Newton polygon has no two-dimensional interior")
    return polygon


def _interior_lattice_points(polygon):
    """Enumerate strictly interior exponent points in Baker basis order."""
    points = []
    for y in range(min(p[1] for p in polygon) + 1,
                   max(p[1] for p in polygon)):
        for x in range(min(p[0] for p in polygon) + 1,
                       max(p[0] for p in polygon)):
            point = (x, y)
            if all(_cross(polygon[i], polygon[(i + 1) % len(polygon)],
                          point) > 0 for i in range(len(polygon))):
                points.append(point)
    return tuple(points)


def _check_newton_edges(ctx, curve, polygon):
    """Reject edges whose face polynomial has a repeated nonzero root."""
    support = {(x, y): coefficient for x, y, coefficient in curve.terms}
    for start, end in zip(polygon, polygon[1:] + polygon[:1]):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = gcd(abs(dx), abs(dy))
        if length == 1:
            continue
        step = (dx // length, dy // length)
        coefficients = tuple(support.get(
            (start[0] + i * step[0], start[1] + i * step[1]),
            ctx.zero) for i in range(length + 1))
        derivative = tuple(i * coefficients[i]
                           for i in range(1, len(coefficients)))
        # Face roots lie in C*, since the endpoint coefficients are nonzero.
        with ctx.extraprec(30):
            divisor = _polynomial_gcd(ctx, coefficients, derivative)
        if len(divisor) > 1:
            raise ValueError(
                "the Newton polygon has a degenerate edge; "
                "supply first-kind differentials")


def _evaluate_baker_basis(ctx, basis, x, y):
    """Evaluate all ``h/F_y`` forms in a structured Baker basis."""
    numerators, denominator_terms = basis
    denominator = ctx.fsum(
        coefficient * x**x_power * y**y_power
        for x_power, y_power, coefficient in denominator_terms)
    return tuple(
        x**x_power * y**y_power / denominator
        for x_power, y_power in numerators)


def _baker_callable(ctx, basis, index):
    """Adapt one structured Baker form to the public callable convention."""
    numerator = basis[0][index]

    def differential(x, y):
        denominator = ctx.fsum(
            coefficient * x**x_power * y**y_power
            for x_power, y_power, coefficient in basis[1])
        return x**numerator[0] * y**numerator[1] / denominator

    # Automatic forms are returned in result records alongside supplied
    # callables.  Retain the useful description without exposing a private
    # implementation class.
    differential.numerator = numerator
    return differential


def _baker_basis(ctx, curve, genus):
    """Construct numerator exponents and common ``F_y`` denominator terms."""
    if genus == 0:
        raise ValueError("a genus-zero curve has no first-kind periods")
    polygon = _newton_polygon(curve)
    _check_newton_edges(ctx, curve, polygon)
    points = _interior_lattice_points(polygon)
    if len(points) != genus:
        raise ValueError(
            "Newton interior-point count differs from the curve genus; "
            "supply first-kind differentials")
    numerators = tuple((x - 1, y - 1) for x, y in points)
    denominator_terms = tuple(
        (x, y - 1, y * coefficient)
        for x, y, coefficient in curve.terms if y)
    return numerators, denominator_terms


def _baker_differentials(ctx, curve, genus):
    """Return callable adapters for a validated structured Baker basis."""
    basis = _baker_basis(ctx, curve, genus)
    return tuple(
        _baker_callable(ctx, basis, index)
        for index in range(len(basis[0])))
