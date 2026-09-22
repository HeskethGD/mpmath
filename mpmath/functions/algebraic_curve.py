"""Numerical building blocks for plane algebraic curves.

The public ``curve_`` functions expose the computational pipeline for a
smooth plane algebraic curve in separate stages: branch locus, monodromy,
genus, homology and period matrices.  Each stage validates its own input,
returns a small immutable result record and reuses cached earlier stages,
so a user requesting one stage never pays for a later one.

The remaining sections cover projection, continuation, lifted paths,
monodromy, homology and numerical period assembly.  They are private while
their numerical contracts are validated.
"""

from collections import namedtuple
from fractions import Fraction
from functools import lru_cache, wraps
from math import comb

from .functions import defun


_PlaneCurve = namedtuple(
    "_PlaneCurve", "terms x_degree y_degree")
_SheetContinuation = namedtuple(
    "_SheetContinuation",
    "sheets permutation max_residual min_separation "
    "max_prediction_correction steps path fibres refinements")
_BranchContinuation = namedtuple(
    "_BranchContinuation",
    "values max_residual min_derivative steps path refinements")
_MonodromyData = namedtuple(
    "_MonodromyData",
    "base_point base_sheets branch_points permutations "
    "infinity_permutation ramification genus continuations")
_BranchGenerator = namedtuple(
    "_BranchGenerator",
    "kind value permutation continuation radius")
_OrderedMonodromyData = namedtuple(
    "_OrderedMonodromyData",
    "base_point base_sheets center branch_points product_generators "
    "ribbon_generators ramification genus minimum_clearance")
_PathIntegrals = namedtuple(
    "_PathIntegrals", "values max_sheet_residual segments")
_IteratedPathIntegrals = namedtuple(
    "_IteratedPathIntegrals",
    "values iterated max_sheet_residual segments")
_LiftedPlaneCurvePath = namedtuple(
    "_LiftedPlaneCurvePath", "continuation sheet start end")
_LiftedPathTerm = namedtuple(
    "_LiftedPathTerm", "coefficient continuation sheet")
_LiftedPathChain = namedtuple(
    "_LiftedPathChain", "terms")
_BoundaryPlace = namedtuple(
    "_BoundaryPlace", "multiplicity x y")
_LiftedGraphEdge = namedtuple(
    "_LiftedGraphEdge", "tail head branch_index sheet")
_LiftedMonodromyGraph = namedtuple(
    "_LiftedMonodromyGraph",
    "degree genus permutations branch_cycles vertices edges rotation "
    "tree_edges chord_edges cycles intersection boundary_components "
    "intersection_rank")
_SymplecticReduction = namedtuple(
    "_SymplecticReduction",
    "transformation form genus radical_rank")
_BranchLoopStep = namedtuple(
    "_BranchLoopStep", "branch_index turns")
_GraphCycleWord = namedtuple(
    "_GraphCycleWord", "start_sheet steps")
_NumericalGraphCycles = namedtuple(
    "_NumericalGraphCycles",
    "branch_continuations words chains")
_PlaneCurvePeriods = namedtuple(
    "_PlaneCurvePeriods",
    "periods a_periods b_periods tau symmetry_residual "
    "imaginary_eigenvalues max_sheet_residual")
CurveBranchLocus = namedtuple(
    "CurveBranchLocus", "degree branch_values resultant")
CurveMonodromy = namedtuple(
    "CurveMonodromy",
    "base_point base_sheets branch_values permutations "
    "infinity_permutation ramification genus transitive product_identity "
    "minimum_clearance")
CurveGenus = namedtuple("CurveGenus", "genus degree ramification")
CurveHomology = namedtuple(
    "CurveHomology",
    "genus cycle_count boundary_components intersection_rank radical_rank "
    "intersection_form transformation")
CurvePeriods = namedtuple(
    "CurvePeriods",
    "genus differentials omega omega_prime tau eta eta_prime kappa "
    "symmetry_residual kappa_symmetry_residual imaginary_eigenvalues "
    "max_sheet_residual")
CurveCheck = namedtuple("CurveCheck", "name value passed")
CurveValidation = namedtuple(
    "CurveValidation", "kind passed maximum_residual checks")
CurvePlace = namedtuple("CurvePlace", "x y")
CurvePath = namedtuple(
    "CurvePath", "start end sheet continuation")
CurveIntegral = namedtuple(
    "CurveIntegral", "values max_sheet_residual segments")
CurveLatticeReduction = namedtuple(
    "CurveLatticeReduction", "value shift")
# Internal place records use the public CurvePlace representation.
_PlaneCurvePlace = CurvePlace


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
    resultant = _plane_curve_resultant_y(ctx, curve)
    if len(resultant) <= 1:
        raise ValueError("the projection has no finite critical polynomial")
    squarefree = _polynomial_squarefree_part(ctx, resultant)
    try:
        roots = tuple(ctx.polyroots(
            squarefree, maxsteps=1000, error=False))
    except Exception as exc:
        raise ValueError("failed to resolve finite critical values") from exc
    scale = max([ctx.one] + [abs(root) for root in roots])
    tolerance = ctx.sqrt(ctx.eps) * scale
    distinct = []
    for root in sorted(roots, key=lambda value: (
            ctx.re(value), ctx.im(value))):
        if not distinct or min(abs(root - value)
                               for value in distinct) > tolerance:
            distinct.append(root)
    return tuple(distinct), resultant


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


# Base-plane paths
# ----------------

def _real_branch_loop_path(ctx, base_point, branch_point, branch_points,
                           circle_steps=24, corridor_height=None):
    """Construct a guarded loop for a set of distinct real branch values.

    The path descends from a real base point to a common lower-half-plane
    corridor, approaches the selected branch point vertically, traverses one
    positive circle, and returns along the stem.  More general complex branch
    configurations will use a separate path-construction strategy.
    """
    base_point = ctx.convert(base_point)
    branch_point = ctx.convert(branch_point)
    branch_points = tuple(ctx.convert(point) for point in branch_points)
    if ctx.im(base_point) or any(ctx.im(point) for point in branch_points):
        raise ValueError("real branch-loop construction requires real points")
    if branch_point not in branch_points:
        raise ValueError("branch_point must occur in branch_points")
    if len(set(branch_points)) != len(branch_points):
        raise ValueError("branch points must be distinct")
    if base_point in branch_points:
        raise ValueError("the base point must be regular")
    if not isinstance(circle_steps, int) or circle_steps < 8:
        raise ValueError("circle_steps must be an integer at least 8")

    spacing = min(abs(branch_point - point)
                  for point in branch_points + (base_point,)
                  if point != branch_point)
    radius = spacing / 4
    extent = max([ctx.one] + [abs(point - base_point)
                              for point in branch_points])
    if corridor_height is None:
        corridor_height = extent / 4
    else:
        corridor_height = ctx.convert(corridor_height)
        if not ctx.isfinite(corridor_height) or corridor_height <= 0:
            raise ValueError("corridor_height must be finite and positive")
    corridor_height = max(corridor_height, 2 * radius)

    corridor_base = base_point - ctx.j * corridor_height
    corridor_target = branch_point - ctx.j * corridor_height
    approach = branch_point - ctx.j * radius
    stem = (base_point, corridor_base, corridor_target, approach)
    circle = tuple(
        branch_point + radius * ctx.exp(
            ctx.j * (-ctx.pi / 2 + 2 * ctx.pi * step / circle_steps))
        for step in range(1, circle_steps + 1))
    return stem + circle + tuple(reversed(stem[:-1]))


def _point_segment_distance(ctx, point, start, end):
    """Return the Euclidean distance from a complex point to a segment."""
    direction = end - start
    if not direction:
        return abs(point - start)
    parameter = ctx.re(
        (point - start) * ctx.conj(direction)) / abs(direction)**2
    parameter = max(ctx.zero, min(ctx.one, parameter))
    return abs(point - (start + parameter * direction))


def _radial_branch_geometry(ctx, branch_points, base_point=None):
    """Choose a guarded exterior base point and radial loop radii."""
    points = tuple(ctx.convert(point) for point in branch_points)
    if not points:
        raise ValueError("at least one finite branch point is required")
    if any(not ctx.isfinite(point) for point in points):
        raise ValueError("branch points must be finite")
    if len(set(points)) != len(points):
        raise ValueError("branch points must be distinct")

    center = ctx.fsum(points) / len(points)
    spread = max(abs(point - center) for point in points)
    if not spread:
        spread = max(ctx.one, abs(center))

    def clearance(candidate):
        if candidate in points or abs(candidate - center) <= spread:
            return ctx.zero
        distances = [abs(candidate - point) for point in points]
        for target_index, target in enumerate(points):
            for other_index, other in enumerate(points):
                if target_index == other_index:
                    continue
                distances.append(_point_segment_distance(
                    ctx, other, candidate, target))
        return min(distances)

    if base_point is None:
        outer_radius = 3 * spread
        candidates = tuple(
            center + outer_radius * ctx.exp(
                2j * ctx.pi * (2 * index + 1) / 32)
            for index in range(16)
        )
        base_point = max(candidates, key=clearance)
    else:
        base_point = ctx.convert(base_point)
        if not ctx.isfinite(base_point):
            raise ValueError("base point must be finite")
    minimum_clearance = clearance(base_point)
    scale = max(spread, abs(base_point - center))
    if minimum_clearance <= ctx.sqrt(ctx.eps) * scale:
        raise ValueError("radial branch paths have insufficient clearance")

    radii = []
    for index, point in enumerate(points):
        distances = [abs(point - base_point)]
        distances.extend(abs(point - other)
                         for other in points if other != point)
        distances.extend(
            _point_segment_distance(ctx, point, base_point, other)
            for other_index, other in enumerate(points)
            if other_index != index)
        radii.append(min(distances) / 5)
    return base_point, center, tuple(radii), minimum_clearance


def _radial_branch_loop_path(
        ctx, base_point, branch_point, radius, circle_steps=24):
    """Return a positive lollipop loop on one guarded radial spoke."""
    if not isinstance(circle_steps, int) or circle_steps < 8:
        raise ValueError("circle_steps must be an integer at least 8")
    radius = ctx.convert(radius)
    if not ctx.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    direction = (base_point - branch_point) / abs(
        base_point - branch_point)
    approach = branch_point + radius * direction
    angle = ctx.arg(direction)
    circle = tuple(
        branch_point + radius * ctx.exp(
            ctx.j * (angle + 2 * ctx.pi * step / circle_steps))
        for step in range(1, circle_steps + 1))
    return (base_point, approach) + circle + (base_point,)


# Sheet continuation
# ------------------

def _predict_plane_curve_sheets(ctx, curve, x, next_x, sheets):
    """Predict sheets at ``next_x`` by implicit differentiation."""
    step = next_x - x
    predictions = []
    threshold = ctx.sqrt(ctx.eps)
    for sheet in sheets:
        derivative_x = _evaluate_plane_derivative(
            ctx, curve, x, sheet, "x")
        derivative_y = _evaluate_plane_derivative(
            ctx, curve, x, sheet, "y")
        scale = max(ctx.one, abs(derivative_x), abs(derivative_y))
        if abs(derivative_y) <= threshold * scale:
            predictions.append(sheet)
        else:
            predictions.append(
                sheet - derivative_x * step / derivative_y)
    return tuple(predictions)


def _closed_path_permutation(ctx, path, final_sheets, initial_sheets):
    """Return endpoint monodromy, or ``None`` for an open base-plane path."""
    scale = max(ctx.one, abs(path[0]), abs(path[-1]))
    if abs(path[-1] - path[0]) > ctx.sqrt(ctx.eps) * scale:
        return None
    return _minimum_cost_assignment(ctx, final_sheets, initial_sheets)


def _continue_plane_curve_sheets(ctx, curve, path, initial_sheets=None):
    """Continue every sheet along a prescribed sequence of x-values.

    The path is assumed to avoid critical values.  Adaptive path refinement is
    intentionally a separate future concern; this primitive reports the
    diagnostics needed to decide when refinement is necessary.
    """
    path = tuple(ctx.convert(point) for point in path)
    if not path:
        raise ValueError(
            "the continuation path must contain at least one point")
    if any(not ctx.isfinite(point) for point in path):
        raise ValueError("continuation path points must be finite")

    if initial_sheets is None:
        initial_sheets = _ordered_plane_curve_sheets(ctx, curve, path[0])
    else:
        initial_sheets = tuple(ctx.convert(sheet) for sheet in initial_sheets)
        if len(initial_sheets) != curve.y_degree:
            raise ValueError("initial_sheets must contain one value per sheet")
    sheets = initial_sheets
    fibres = [sheets]

    max_residual = max(
        abs(_evaluate_plane_polynomial(ctx, curve, path[0], sheet))
        for sheet in sheets)
    min_separation = _minimum_root_separation(ctx, sheets)
    max_prediction_correction = ctx.zero

    for x, next_x in zip(path, path[1:]):
        predictions = _predict_plane_curve_sheets(
            ctx, curve, x, next_x, sheets)
        candidates = _plane_curve_sheets(
            ctx, curve, next_x, roots_init=predictions)
        assignment = _minimum_cost_assignment(ctx, predictions, candidates)
        next_sheets = tuple(candidates[index] for index in assignment)
        max_prediction_correction = max(
            max_prediction_correction,
            max(abs(value - prediction)
                for value, prediction in zip(next_sheets, predictions)))
        max_residual = max(
            max_residual,
            max(abs(_evaluate_plane_polynomial(
                ctx, curve, next_x, sheet)) for sheet in next_sheets))
        min_separation = min(
            min_separation, _minimum_root_separation(ctx, next_sheets))
        sheets = next_sheets
        fibres.append(sheets)

    permutation = _closed_path_permutation(
        ctx, path, sheets, initial_sheets)
    return _SheetContinuation(
        sheets=sheets,
        permutation=permutation,
        max_residual=max_residual,
        min_separation=min_separation,
        max_prediction_correction=max_prediction_correction,
        steps=len(path) - 1,
        path=path,
        fibres=tuple(fibres),
        refinements=0,
    )


def _continue_plane_curve_sheets_adaptive(
        ctx, curve, path, initial_sheets=None, max_refinements=12,
        correction_fraction=None, motion_fraction=None):
    """Continue sheets while bisecting numerically unsafe path segments.

    Acceptance compares prediction error and sheet motion with the local
    minimum sheet separation.  Geometry remains the caller's responsibility:
    subdivision improves resolution but cannot repair a path through a
    critical value.
    """
    path = tuple(ctx.convert(point) for point in path)
    if not path:
        raise ValueError(
            "the continuation path must contain at least one point")
    if any(not ctx.isfinite(point) for point in path):
        raise ValueError("continuation path points must be finite")
    if not isinstance(max_refinements, int) or max_refinements < 0:
        raise ValueError("max_refinements must be a nonnegative integer")
    if correction_fraction is None:
        correction_fraction = ctx.mpf("0.2")
    else:
        correction_fraction = ctx.convert(correction_fraction)
    if motion_fraction is None:
        motion_fraction = ctx.mpf("0.4")
    else:
        motion_fraction = ctx.convert(motion_fraction)
    if correction_fraction <= 0 or motion_fraction <= 0:
        raise ValueError("continuation fractions must be positive")

    if initial_sheets is None:
        initial_sheets = _ordered_plane_curve_sheets(ctx, curve, path[0])
    else:
        initial_sheets = tuple(ctx.convert(sheet) for sheet in initial_sheets)
        if len(initial_sheets) != curve.y_degree:
            raise ValueError("initial_sheets must contain one value per sheet")

    accepted_path = [path[0]]
    accepted_fibres = [initial_sheets]
    diagnostics = {
        "max_residual": max(
            abs(_evaluate_plane_polynomial(ctx, curve, path[0], sheet))
            for sheet in initial_sheets),
        "min_separation": _minimum_root_separation(ctx, initial_sheets),
        "max_prediction_correction": ctx.zero,
        "refinements": 0,
    }

    def advance(left, right, sheets, depth):
        predictions = _predict_plane_curve_sheets(
            ctx, curve, left, right, sheets)
        try:
            candidates = _plane_curve_sheets(
                ctx, curve, right, roots_init=predictions)
        except ctx.NoConvergence:
            candidates = None

        if candidates is not None:
            assignment = _minimum_cost_assignment(
                ctx, predictions, candidates)
            next_sheets = tuple(candidates[index] for index in assignment)
            separation = min(
                _minimum_root_separation(ctx, sheets),
                _minimum_root_separation(ctx, next_sheets))
            correction = max(
                abs(value - prediction)
                for value, prediction in zip(next_sheets, predictions))
            motion = max(abs(value - sheet)
                         for value, sheet in zip(next_sheets, sheets))
            acceptable = (
                correction <= correction_fraction * separation
                and motion <= motion_fraction * separation)
        else:
            acceptable = False

        if not acceptable:
            if depth >= max_refinements:
                raise ctx.NoConvergence(
                    "sheet continuation did not resolve a path segment")
            midpoint = (left + right) / 2
            if midpoint == left or midpoint == right:
                raise ctx.NoConvergence(
                    "sheet continuation path cannot be subdivided further")
            diagnostics["refinements"] += 1
            middle_sheets = advance(left, midpoint, sheets, depth + 1)
            return advance(midpoint, right, middle_sheets, depth + 1)

        residual = max(abs(_evaluate_plane_polynomial(
            ctx, curve, right, sheet)) for sheet in next_sheets)
        diagnostics["max_residual"] = max(
            diagnostics["max_residual"], residual)
        diagnostics["min_separation"] = min(
            diagnostics["min_separation"], separation)
        diagnostics["max_prediction_correction"] = max(
            diagnostics["max_prediction_correction"], correction)
        accepted_path.append(right)
        accepted_fibres.append(next_sheets)
        return next_sheets

    sheets = initial_sheets
    for left, right in zip(path, path[1:]):
        sheets = advance(left, right, sheets, 0)

    permutation = _closed_path_permutation(
        ctx, accepted_path, sheets, initial_sheets)
    return _SheetContinuation(
        sheets=sheets,
        permutation=permutation,
        max_residual=diagnostics["max_residual"],
        min_separation=diagnostics["min_separation"],
        max_prediction_correction=(
            diagnostics["max_prediction_correction"]),
        steps=len(accepted_path) - 1,
        path=tuple(accepted_path),
        fibres=tuple(accepted_fibres),
        refinements=diagnostics["refinements"],
    )


def _continue_plane_curve_branch(
        ctx, curve, path, initial_y, max_refinements=12,
        max_newton_steps=20):
    """Continue one simple branch, including from a local-chart endpoint."""
    path = tuple(ctx.convert(point) for point in path)
    if not path:
        raise ValueError("the branch path must contain at least one point")
    if any(not ctx.isfinite(point) for point in path):
        raise ValueError("branch path points must be finite")
    if not isinstance(max_refinements, int) or max_refinements < 0:
        raise ValueError("max_refinements must be a nonnegative integer")
    if not isinstance(max_newton_steps, int) or max_newton_steps < 1:
        raise ValueError("max_newton_steps must be a positive integer")
    initial_y = ctx.convert(initial_y)

    def polynomial_scale(x, y):
        return max(ctx.one, ctx.fsum(
            abs(coefficient * x ** x_power * y ** y_power)
            for x_power, y_power, coefficient in curve.terms))

    initial_residual = abs(_evaluate_plane_polynomial(
        ctx, curve, path[0], initial_y))
    derivative = abs(_evaluate_plane_derivative(
        ctx, curve, path[0], initial_y, "y"))
    scale = polynomial_scale(path[0], initial_y)
    if initial_residual > 100 * ctx.sqrt(ctx.eps) * scale:
        raise ValueError("initial_y is not on the plane curve")
    if derivative <= 100 * ctx.sqrt(ctx.eps) * scale:
        raise ValueError("initial branch must be simple in the chart")

    accepted_path = [path[0]]
    accepted_values = [initial_y]
    diagnostics = {
        "max_residual": initial_residual,
        "min_derivative": derivative,
        "refinements": 0,
    }

    def advance(left, right, value, depth):
        derivative_x = _evaluate_plane_derivative(
            ctx, curve, left, value, "x")
        derivative_y = _evaluate_plane_derivative(
            ctx, curve, left, value, "y")
        prediction = value - derivative_x * (right - left) / derivative_y
        candidate = prediction
        converged = False
        for unused in range(max_newton_steps):
            residual_value = _evaluate_plane_polynomial(
                ctx, curve, right, candidate)
            candidate_derivative = _evaluate_plane_derivative(
                ctx, curve, right, candidate, "y")
            candidate_scale = polynomial_scale(right, candidate)
            if abs(residual_value) <= 100 * ctx.eps * candidate_scale:
                converged = True
                break
            if abs(candidate_derivative) <= ctx.eps * candidate_scale:
                break
            candidate -= residual_value / candidate_derivative

        correction = abs(candidate - prediction)
        motion = abs(candidate - value)
        acceptable = (converged
                      and correction <= max(ctx.one, motion) / 4)
        if not acceptable:
            if depth >= max_refinements:
                raise ctx.NoConvergence(
                    "single-branch continuation did not resolve a segment")
            midpoint = (left + right) / 2
            diagnostics["refinements"] += 1
            middle = advance(left, midpoint, value, depth + 1)
            return advance(midpoint, right, middle, depth + 1)

        residual = abs(_evaluate_plane_polynomial(
            ctx, curve, right, candidate))
        derivative = abs(_evaluate_plane_derivative(
            ctx, curve, right, candidate, "y"))
        diagnostics["max_residual"] = max(
            diagnostics["max_residual"], residual)
        diagnostics["min_derivative"] = min(
            diagnostics["min_derivative"], derivative)
        accepted_path.append(right)
        accepted_values.append(candidate)
        return candidate

    value = initial_y
    for left, right in zip(path, path[1:]):
        value = advance(left, right, value, 0)
    return _BranchContinuation(
        values=tuple(accepted_values),
        max_residual=diagnostics["max_residual"],
        min_derivative=diagnostics["min_derivative"],
        steps=len(accepted_path) - 1,
        path=tuple(accepted_path),
        refinements=diagnostics["refinements"],
    )


def _reverse_plane_curve_branch(continuation):
    """Reverse a continued single branch."""
    return _BranchContinuation(
        values=tuple(reversed(continuation.values)),
        max_residual=continuation.max_residual,
        min_derivative=continuation.min_derivative,
        steps=continuation.steps,
        path=tuple(reversed(continuation.path)),
        refinements=continuation.refinements,
    )


# Lifted-path composition
# -----------------------

def _reverse_plane_curve_continuation(ctx, continuation):
    """Reverse a continued path while preserving its lifted sheet labels."""
    path = tuple(reversed(continuation.path))
    fibres = tuple(reversed(continuation.fibres))
    permutation = _closed_path_permutation(
        ctx, path, fibres[-1], fibres[0])
    return _SheetContinuation(
        sheets=fibres[-1],
        permutation=permutation,
        max_residual=continuation.max_residual,
        min_separation=continuation.min_separation,
        max_prediction_correction=(
            continuation.max_prediction_correction),
        steps=continuation.steps,
        path=path,
        fibres=fibres,
        refinements=continuation.refinements,
    )


def _concatenate_plane_curve_continuations(ctx, left, right):
    """Join continued paths, relabeling the right fibre at the common point."""
    left_degree = len(left.fibres[0])
    if any(len(fibre) != left_degree
           for fibre in left.fibres + right.fibres):
        raise ValueError("continued paths must have the same degree")
    scale = max(ctx.one, abs(left.path[-1]), abs(right.path[0]))
    if abs(left.path[-1] - right.path[0]) > ctx.sqrt(ctx.eps) * scale:
        raise ValueError("continued paths do not share an endpoint")

    assignment = _minimum_cost_assignment(
        ctx, left.fibres[-1], right.fibres[0])
    right_fibres = tuple(
        tuple(fibre[index] for index in assignment)
        for fibre in right.fibres)
    path = left.path + right.path[1:]
    fibres = left.fibres + right_fibres[1:]
    permutation = _closed_path_permutation(
        ctx, path, fibres[-1], fibres[0])
    return _SheetContinuation(
        sheets=fibres[-1],
        permutation=permutation,
        max_residual=max(left.max_residual, right.max_residual),
        min_separation=min(left.min_separation, right.min_separation),
        max_prediction_correction=max(
            left.max_prediction_correction,
            right.max_prediction_correction),
        steps=left.steps + right.steps,
        path=path,
        fibres=fibres,
        refinements=left.refinements + right.refinements,
    )


def _lift_plane_curve_path(
        ctx, curve, path, start_y, match_tolerance=None,
        max_refinements=12):
    """Continue the lift starting at a specified regular finite place.

    ``start_y`` identifies a point in the fibre over ``path[0]``.  The
    returned sheet index is local to the continuation; callers need not rely
    on the deterministic ordering used internally for the complete fibre.
    """
    continuation = _continue_plane_curve_sheets_adaptive(
        ctx, curve, path, max_refinements=max_refinements)
    start_y = ctx.convert(start_y)
    if not ctx.isfinite(start_y):
        raise ValueError("start_y must be finite")
    start_fibre = continuation.fibres[0]
    sheet = min(range(len(start_fibre)),
                key=lambda index: abs(start_fibre[index] - start_y))
    match_error = abs(start_fibre[sheet] - start_y)
    scale = max(ctx.one, abs(start_y), abs(start_fibre[sheet]))
    if match_tolerance is None:
        match_tolerance = 100 * ctx.sqrt(ctx.eps) * scale
    else:
        match_tolerance = ctx.convert(match_tolerance)
        if (not ctx.isfinite(match_tolerance)
                or match_tolerance < 0):
            raise ValueError("match_tolerance must be finite and nonnegative")
    if match_error > match_tolerance:
        raise ValueError("start_y does not identify a sheet at path[0]")
    return _LiftedPlaneCurvePath(
        continuation=continuation,
        sheet=sheet,
        start=_PlaneCurvePlace(
            continuation.path[0], start_fibre[sheet]),
        end=_PlaneCurvePlace(
            continuation.path[-1], continuation.fibres[-1][sheet]),
    )


# Lifted-path chains
# ------------------

def _prepare_lifted_path_chain(terms):
    """Return a validated integer chain of selected lifted paths."""
    prepared = []
    try:
        terms = tuple(terms)
    except TypeError:
        raise ValueError(
            "chain terms must be (coefficient, path, sheet) tuples")
    for term in terms:
        if not isinstance(term, tuple) or len(term) != 3:
            raise ValueError(
                "chain terms must be (coefficient, path, sheet) tuples")
        coefficient, continuation, sheet = term
        if not isinstance(coefficient, int):
            raise ValueError("chain coefficients must be integers")
        if not coefficient:
            continue
        if not isinstance(sheet, int):
            raise ValueError("chain sheets must be integer indices")
        if (not continuation.fibres
                or not 0 <= sheet < len(continuation.fibres[0])):
            raise ValueError("chain sheet does not index the lifted path")
        prepared.append(_LiftedPathTerm(
            coefficient, continuation, sheet))
    return _LiftedPathChain(tuple(prepared))


def _lifted_path_endpoints(term):
    """Return the initial and final places of one lifted path term."""
    continuation = term.continuation
    sheet = term.sheet
    start = continuation.path[0], continuation.fibres[0][sheet]
    end = continuation.path[-1], continuation.fibres[-1][sheet]
    return start, end


def _same_numerical_place(ctx, left, right):
    """Return whether two numerical place representatives agree."""
    scale = max(ctx.one, abs(left[0]), abs(left[1]),
                abs(right[0]), abs(right[1]))
    tolerance = ctx.sqrt(ctx.eps) * scale
    return (abs(left[0] - right[0]) <= tolerance
            and abs(left[1] - right[1]) <= tolerance)


def _lifted_path_chain_boundary(ctx, chain):
    """Return the nonzero numerical boundary of a lifted-path chain."""
    places = []
    for term in chain.terms:
        start, end = _lifted_path_endpoints(term)
        for place, multiplicity in (
                (start, -term.coefficient),
                (end, term.coefficient)):
            for index, (existing, total) in enumerate(places):
                if _same_numerical_place(ctx, place, existing):
                    places[index] = existing, total + multiplicity
                    break
            else:
                places.append((place, multiplicity))
    return tuple(
        _BoundaryPlace(multiplicity, place[0], place[1])
        for place, multiplicity in places if multiplicity)


def _close_monodromy_lift(ctx, continuation, sheet):
    """Repeat a closed base loop until the selected lifted sheet closes."""
    permutation = continuation.permutation
    if permutation is None:
        raise ValueError(
            "closing a monodromy lift requires a closed base path")
    if not isinstance(sheet, int) or not 0 <= sheet < len(permutation):
        raise ValueError("sheet must index the lifted path")
    result = continuation
    image = permutation[sheet]
    repeats = 1
    while image != sheet:
        if repeats >= len(permutation):
            raise ValueError("invalid monodromy orbit")
        result = _concatenate_plane_curve_continuations(
            ctx, result, continuation)
        image = permutation[image]
        repeats += 1
    return result


# Lifted-path integration
# -----------------------

def _integrate_plane_curve_path(
        ctx, curve, continuation, differentials, sheet=0,
        quadrature_order=None):
    """Integrate coefficients of dx along one continued sheet.

    Each differential is a callable ``differential(x, y)`` returning the
    coefficient of ``dx``.  Intermediate fibres are solved and matched
    against the linearly interpolated endpoint fibres, making quadrature
    independent of mutable root-tracking state.
    """
    try:
        differentials = tuple(differentials)
    except TypeError:
        raise ValueError("differentials must be a sequence of callables")
    if not differentials or any(not callable(value)
                                for value in differentials):
        raise ValueError("differentials must be a sequence of callables")
    if not isinstance(sheet, int) or not 0 <= sheet < curve.y_degree:
        raise ValueError("sheet must index the initial fibre")
    if len(continuation.path) != len(continuation.fibres):
        raise ValueError("continuation path and fibres are inconsistent")
    if any(len(fibre) != curve.y_degree
           for fibre in continuation.fibres):
        raise ValueError("continuation fibres have the wrong degree")
    if quadrature_order is None:
        quadrature_order = max(16, 2 * ctx.dps)
    if not isinstance(quadrature_order, int) or quadrature_order < 2:
        raise ValueError("quadrature_order must be an integer at least 2")
    nodes, weights = ctx.gauss_quadrature(
        quadrature_order, "legendre")
    parameters = tuple((nodes[index] + 1) / 2
                       for index in range(quadrature_order))
    weights = tuple(weights[index] / 2
                    for index in range(quadrature_order))

    values = [ctx.zero] * len(differentials)
    max_sheet_residual = ctx.zero
    for segment in range(len(continuation.path) - 1):
        left_x = continuation.path[segment]
        right_x = continuation.path[segment + 1]
        delta_x = right_x - left_x
        if not delta_x:
            continue
        left_fibre = continuation.fibres[segment]
        right_fibre = continuation.fibres[segment + 1]

        def lifted_point(parameter):
            nonlocal max_sheet_residual
            parameter = ctx.convert(parameter)
            x = left_x + parameter * delta_x
            if not parameter:
                ordered = left_fibre
            elif parameter == 1:
                ordered = right_fibre
            else:
                predictions = tuple(
                    left + parameter * (right - left)
                    for left, right in zip(left_fibre, right_fibre))
                candidates = _plane_curve_sheets(
                    ctx, curve, x, roots_init=predictions)
                assignment = _minimum_cost_assignment(
                    ctx, predictions, candidates)
                ordered = tuple(candidates[index] for index in assignment)
            y = ordered[sheet]
            residual = abs(_evaluate_plane_polynomial(ctx, curve, x, y))
            max_sheet_residual = max(max_sheet_residual, residual)
            return x, y

        contributions = [[] for unused in differentials]
        for parameter, weight in zip(parameters, weights):
            x, y = lifted_point(parameter)
            for index, differential in enumerate(differentials):
                contributions[index].append(
                    weight * differential(x, y))
        for index, terms in enumerate(contributions):
            values[index] += delta_x * ctx.fsum(terms)

    return _PathIntegrals(
        values=tuple(values),
        max_sheet_residual=max_sheet_residual,
        segments=len(continuation.path) - 1,
    )


def _gauss_indefinite_matrix(ctx, parameters):
    """Integrate the Lagrange basis from zero to each Gauss node."""
    parameters = tuple(parameters)
    result = []
    for upper in parameters:
        row = []
        for index, node in enumerate(parameters):
            polynomial = (ctx.one,)
            denominator = ctx.one
            for other_index, other in enumerate(parameters):
                if other_index == index:
                    continue
                polynomial = _polynomial_multiply(
                    ctx, polynomial, (-other, ctx.one))
                denominator *= node - other
            row.append(ctx.fsum(
                coefficient * upper ** (degree + 1)
                / ((degree + 1) * denominator)
                for degree, coefficient in enumerate(polynomial)))
        result.append(tuple(row))
    return tuple(result)


def _integrate_plane_curve_path_iterated(
        ctx, curve, continuation, differentials, sheet=0,
        quadrature_order=None):
    """Integrate forms and their ordered pairwise iterated integrals."""
    differentials = tuple(differentials)
    if not differentials or any(not callable(value)
                                for value in differentials):
        raise ValueError("differentials must be a sequence of callables")
    if not isinstance(sheet, int) or not 0 <= sheet < curve.y_degree:
        raise ValueError("sheet must index the initial fibre")
    if quadrature_order is None:
        quadrature_order = max(12, ctx.dps // 2)
    if not isinstance(quadrature_order, int) or quadrature_order < 2:
        raise ValueError("quadrature_order must be an integer at least 2")

    nodes, weights = ctx.gauss_quadrature(
        quadrature_order, "legendre")
    parameters = tuple((nodes[index] + 1) / 2
                       for index in range(quadrature_order))
    weights = tuple(weights[index] / 2
                    for index in range(quadrature_order))
    indefinite = _gauss_indefinite_matrix(ctx, parameters)
    count = len(differentials)
    values = [ctx.zero] * count
    iterated = [[ctx.zero] * count for unused in range(count)]
    max_sheet_residual = ctx.zero

    for segment in range(len(continuation.path) - 1):
        left_x = continuation.path[segment]
        right_x = continuation.path[segment + 1]
        delta_x = right_x - left_x
        if not delta_x:
            continue
        left_fibre = continuation.fibres[segment]
        right_fibre = continuation.fibres[segment + 1]
        samples = []
        for parameter in parameters:
            x = left_x + parameter * delta_x
            predictions = tuple(
                left + parameter * (right - left)
                for left, right in zip(left_fibre, right_fibre))
            candidates = _plane_curve_sheets(
                ctx, curve, x, roots_init=predictions)
            assignment = _minimum_cost_assignment(
                ctx, predictions, candidates)
            y = candidates[assignment[sheet]]
            max_sheet_residual = max(
                max_sheet_residual,
                abs(_evaluate_plane_polynomial(ctx, curve, x, y)))
            samples.append(tuple(
                differential(x, y) for differential in differentials))

        local_primitives = tuple(tuple(
            delta_x * ctx.fsum(
                indefinite[node_index][sample_index]
                * samples[sample_index][form_index]
                for sample_index in range(quadrature_order))
            for form_index in range(count))
            for node_index in range(quadrature_order))
        for outer in range(count):
            for inner in range(count):
                iterated[outer][inner] += delta_x * ctx.fsum(
                    weights[node_index] * samples[node_index][outer]
                    * (values[inner]
                       + local_primitives[node_index][inner])
                    for node_index in range(quadrature_order))
        for form_index in range(count):
            values[form_index] += delta_x * ctx.fsum(
                weights[node_index] * samples[node_index][form_index]
                for node_index in range(quadrature_order))

    return _IteratedPathIntegrals(
        values=tuple(values),
        iterated=tuple(tuple(row) for row in iterated),
        max_sheet_residual=max_sheet_residual,
        segments=len(continuation.path) - 1,
    )


def _pullback_plane_curve_differentials(differentials, coordinate_map):
    """Pull coefficients of ``dx`` back through a numerical local chart.

    ``coordinate_map(t, u)`` returns ``(x, y, dx_dt)``.  The resulting
    callables are coefficients of ``dt`` on the chart curve.
    """
    differentials = tuple(differentials)
    if not differentials or any(not callable(value)
                                for value in differentials):
        raise ValueError("differentials must be a sequence of callables")
    if not callable(coordinate_map):
        raise ValueError("coordinate_map must be callable")

    def pullback(differential):
        def pulled_back(t, u):
            x, y, dx_dt = coordinate_map(t, u)
            return differential(x, y) * dx_dt
        return pulled_back

    return tuple(pullback(differential) for differential in differentials)


def _integrate_plane_curve_branch(
        ctx, curve, continuation, differentials, quadrature_order=None):
    """Integrate coefficients of ``dt`` along one continued chart branch."""
    try:
        differentials = tuple(differentials)
    except TypeError:
        raise ValueError("differentials must be a sequence of callables")
    if not differentials or any(not callable(value)
                                for value in differentials):
        raise ValueError("differentials must be a sequence of callables")
    if len(continuation.path) != len(continuation.values):
        raise ValueError("branch path and values are inconsistent")
    if quadrature_order is None:
        quadrature_order = max(16, 2 * ctx.dps)
    if not isinstance(quadrature_order, int) or quadrature_order < 2:
        raise ValueError("quadrature_order must be an integer at least 2")
    nodes, weights = ctx.gauss_quadrature(quadrature_order, "legendre")
    parameters = tuple((nodes[index] + 1) / 2
                       for index in range(quadrature_order))
    weights = tuple(weights[index] / 2
                    for index in range(quadrature_order))

    def polynomial_scale(t, u):
        return max(ctx.one, ctx.fsum(
            abs(coefficient * t ** t_power * u ** u_power)
            for t_power, u_power, coefficient in curve.terms))

    def solve(t, initial):
        value = initial
        for unused in range(20):
            residual = _evaluate_plane_polynomial(ctx, curve, t, value)
            if abs(residual) <= 100 * ctx.eps * polynomial_scale(t, value):
                return value, abs(residual)
            derivative = _evaluate_plane_derivative(
                ctx, curve, t, value, "y")
            if not derivative:
                break
            value -= residual / derivative
        raise ctx.NoConvergence(
            "quadrature node did not resolve the chart branch")

    values = [ctx.zero] * len(differentials)
    max_sheet_residual = ctx.zero
    for segment in range(len(continuation.path) - 1):
        left_t = continuation.path[segment]
        right_t = continuation.path[segment + 1]
        delta_t = right_t - left_t
        if not delta_t:
            continue
        left_u = continuation.values[segment]
        right_u = continuation.values[segment + 1]
        contributions = [[] for unused in differentials]
        for parameter, weight in zip(parameters, weights):
            t = left_t + parameter * delta_t
            prediction = left_u + parameter * (right_u - left_u)
            u, residual = solve(t, prediction)
            max_sheet_residual = max(max_sheet_residual, residual)
            for index, differential in enumerate(differentials):
                contributions[index].append(weight * differential(t, u))
        for index, terms in enumerate(contributions):
            values[index] += delta_t * ctx.fsum(terms)
    return _PathIntegrals(
        values=tuple(values),
        max_sheet_residual=max_sheet_residual,
        segments=len(continuation.path) - 1,
    )


def _integrate_lifted_path_chain(
        ctx, curve, chain, differentials, quadrature_order=None):
    """Integrate supplied differentials termwise over a lifted-path chain."""
    try:
        differentials = tuple(differentials)
    except TypeError:
        raise ValueError("differentials must be a sequence of callables")
    if not differentials or any(not callable(value)
                                for value in differentials):
        raise ValueError("differentials must be a sequence of callables")
    values = [ctx.zero] * len(differentials)
    max_sheet_residual = ctx.zero
    segments = 0
    for term in chain.terms:
        integral = _integrate_plane_curve_path(
            ctx, curve, term.continuation, differentials,
            sheet=term.sheet, quadrature_order=quadrature_order)
        for index, value in enumerate(integral.values):
            values[index] += term.coefficient * value
        max_sheet_residual = max(
            max_sheet_residual, integral.max_sheet_residual)
        segments += integral.segments
    return _PathIntegrals(
        values=tuple(values),
        max_sheet_residual=max_sheet_residual,
        segments=segments,
    )


def _assemble_plane_curve_periods(
        ctx, curve, canonical_chains, differentials, genus,
        quadrature_order=None):
    """Integrate canonical chains and assemble a normalized Riemann matrix."""
    canonical_chains = tuple(canonical_chains)
    differentials = tuple(differentials)
    if not isinstance(genus, int) or genus < 1:
        raise ValueError("genus must be a positive integer")
    if len(canonical_chains) < 2 * genus:
        raise ValueError("canonical_chains must contain 2*genus cycles")
    if len(differentials) != genus:
        raise ValueError("one holomorphic differential is required per genus")

    columns = []
    max_sheet_residual = ctx.zero
    for chain in canonical_chains[:2 * genus]:
        integral = _integrate_lifted_path_chain(
            ctx, curve, chain, differentials,
            quadrature_order=quadrature_order)
        columns.append(integral.values)
        max_sheet_residual = max(
            max_sheet_residual, integral.max_sheet_residual)
    periods = ctx.matrix([
        [columns[column][row] for column in range(2 * genus)]
        for row in range(genus)])
    a_periods = periods[:, :genus]
    b_periods = periods[:, genus:]
    tau = a_periods ** -1 * b_periods
    imaginary_tau = ctx.matrix([
        [ctx.im(tau[row, column]) for column in range(genus)]
        for row in range(genus)])
    imaginary_eigenvalues = tuple(ctx.eigsy(
        imaginary_tau, eigvals_only=True))
    return _PlaneCurvePeriods(
        periods=periods,
        a_periods=a_periods,
        b_periods=b_periods,
        tau=tau,
        symmetry_residual=ctx.norm(tau - tau.T),
        imaginary_eigenvalues=imaginary_eigenvalues,
        max_sheet_residual=max_sheet_residual,
    )


# Monodromy invariants
# --------------------

def _compose_permutations(left, right):
    """Return ``left`` after ``right`` for permutation image tuples."""
    if len(left) != len(right):
        raise ValueError("permutations must have equal sizes")
    return tuple(left[right[index]] for index in range(len(left)))


def _inverse_permutation(permutation):
    """Return the inverse of a permutation image tuple."""
    if sorted(permutation) != list(range(len(permutation))):
        raise ValueError("invalid permutation")
    return tuple(permutation.index(index)
                 for index in range(len(permutation)))


def _permutation_cycles(permutation, include_fixed=False):
    """Return the disjoint cycles of a permutation."""
    if sorted(permutation) != list(range(len(permutation))):
        raise ValueError("invalid permutation")
    visited = set()
    cycles = []
    for start in range(len(permutation)):
        if start in visited:
            continue
        cycle = []
        current = start
        while current not in visited:
            visited.add(current)
            cycle.append(current)
            current = permutation[current]
        if include_fixed or len(cycle) > 1:
            cycles.append(tuple(cycle))
    return tuple(cycles)


def _monodromy_orbit(permutations, start=0):
    """Return the sheet orbit generated by a sequence of permutations."""
    orbit = {start}
    pending = [start]
    while pending:
        sheet = pending.pop()
        for permutation in permutations:
            image = permutation[sheet]
            if image not in orbit:
                orbit.add(image)
                pending.append(image)
    return frozenset(orbit)


def _riemann_hurwitz_genus(degree, permutations):
    """Return genus and total ramification for a connected sphere cover."""
    ramification = sum(
        sum(len(cycle) - 1
            for cycle in _permutation_cycles(permutation))
        for permutation in permutations)
    numerator = 2 - 2 * degree + ramification
    if numerator < 0 or numerator % 2:
        raise ValueError("monodromy data do not define a valid genus")
    return numerator // 2, ramification


def _integer_matrix_rank(matrix):
    """Return the exact rational rank of an integer matrix."""
    rows = [list(map(Fraction, row)) for row in matrix]
    if not rows:
        return 0
    row_count = len(rows)
    column_count = len(rows[0])
    rank = 0
    for column in range(column_count):
        pivot = next((row for row in range(rank, row_count)
                      if rows[row][column]), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        pivot_value = rows[rank][column]
        rows[rank] = [value / pivot_value for value in rows[rank]]
        for row in range(row_count):
            if row == rank or not rows[row][column]:
                continue
            multiplier = rows[row][column]
            rows[row] = [
                value - multiplier * pivot_entry
                for value, pivot_entry in zip(rows[row], rows[rank])]
        rank += 1
        if rank == row_count:
            break
    return rank


def _integer_determinant(matrix):
    """Return the exact determinant of a square integer matrix."""
    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise ValueError("determinant requires a square matrix")
    if not size:
        return 1
    work = [list(row) for row in matrix]
    sign = 1
    previous = 1
    for column in range(size - 1):
        pivot = next((row for row in range(column, size)
                      if work[row][column]), None)
        if pivot is None:
            return 0
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            sign = -sign
        pivot_value = work[column][column]
        for row in range(column + 1, size):
            for entry in range(column + 1, size):
                numerator = (
                    work[row][entry] * pivot_value
                    - work[row][column] * work[column][entry])
                work[row][entry] = numerator // previous
        previous = pivot_value
    return sign * work[-1][-1]


def _symplectic_reduce_intersection(matrix):
    """Reduce a primitive integral skew form to canonical form.

    Rows of the returned transformation express the canonical symplectic and
    radical generators in the input basis.
    """
    matrix = tuple(tuple(int(value) for value in row) for row in matrix)
    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise ValueError("intersection matrix must be square")
    if any(matrix[row][column] != -matrix[column][row]
           for row in range(size) for column in range(size)):
        raise ValueError("intersection matrix must be skew-symmetric")

    def pairing(left, right):
        return sum(
            left[row] * matrix[row][column] * right[column]
            for row in range(size) for column in range(size))

    remaining = [tuple(1 if row == column else 0
                       for column in range(size))
                 for row in range(size)]
    a_vectors = []
    b_vectors = []
    while True:
        pair = next((
            (left, right)
            for left in range(len(remaining))
            for right in range(left + 1, len(remaining))
            if abs(pairing(remaining[left], remaining[right])) == 1
        ), None)
        if pair is None:
            break
        left_index, right_index = pair
        left = remaining[left_index]
        right = remaining[right_index]
        if pairing(left, right) == -1:
            right = tuple(-value for value in right)
        remaining = [
            vector for index, vector in enumerate(remaining)
            if index not in pair]

        orthogonal = []
        for vector in remaining:
            left_coefficient = pairing(vector, right)
            right_coefficient = pairing(vector, left)
            orthogonal.append(tuple(
                value - left_coefficient * left_value
                + right_coefficient * right_value
                for value, left_value, right_value
                in zip(vector, left, right)))
        a_vectors.append(left)
        b_vectors.append(right)
        remaining = orthogonal

    if any(pairing(left, right)
           for left in remaining for right in remaining):
        raise ValueError(
            "intersection form is not primitively symplectic")
    transformation = tuple(a_vectors + b_vectors + remaining)
    if abs(_integer_determinant(transformation)) != 1:
        raise ValueError("symplectic reduction is not unimodular")
    form = tuple(
        tuple(pairing(left, right) for right in transformation)
        for left in transformation)
    genus = len(a_vectors)
    radical_rank = len(remaining)
    return _SymplecticReduction(
        transformation=transformation,
        form=form,
        genus=genus,
        radical_rank=radical_rank,
    )


def _tree_path(adjacency, start, end):
    """Return oriented edge steps along the unique path in a tree."""
    pending = [(start, ())]
    visited = {start}
    while pending:
        vertex, path = pending.pop()
        if vertex == end:
            return path
        for neighbor, edge, orientation in adjacency[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                pending.append((
                    neighbor, path + ((edge, orientation),)))
    raise ValueError("lifted graph is disconnected")


def _ribbon_boundary_count(edges, rotation):
    """Count boundary components of the oriented thickened graph."""
    successor = {}
    for half_edges in rotation.values():
        for index, half_edge in enumerate(half_edges):
            successor[half_edge] = half_edges[(index + 1) % len(half_edges)]

    def boundary_step(half_edge):
        edge, endpoint = half_edge
        return successor[(edge, 1 - endpoint)]

    unvisited = set(successor)
    count = 0
    while unvisited:
        count += 1
        start = next(iter(unvisited))
        current = start
        while current in unvisited:
            unvisited.remove(current)
            current = boundary_step(current)
    return count


def _ribbon_cycle_intersection(edges, rotation, left, right):
    """Return the oriented intersection of two graph cycles."""
    numerator = 0
    for vertex, half_edges in rotation.items():
        left_flows = []
        right_flows = []
        for edge_index, endpoint in half_edges:
            orientation = 1 if endpoint == 0 else -1
            left_flows.append(orientation * left[edge_index])
            right_flows.append(orientation * right[edge_index])
        for first in range(len(half_edges)):
            for second in range(first + 1, len(half_edges)):
                numerator += (
                    left_flows[first] * right_flows[second]
                    - right_flows[first] * left_flows[second])
    if numerator % 2:
        raise ValueError("ribbon intersection is not integral")
    # The half-edge rotations above follow positive base-plane monodromy.
    # Their local outward-flow ordering is opposite the complex orientation
    # convention in which a canonical pair satisfies a . b = +1.
    return -numerator // 2


def _lifted_monodromy_graph(permutations):
    """Construct the lifted ribbon graph of a Hurwitz monodromy system."""
    permutations = tuple(tuple(permutation)
                         for permutation in permutations)
    if not permutations:
        raise ValueError("at least one monodromy permutation is required")
    degree = len(permutations[0])
    if not degree or any(len(permutation) != degree
                         for permutation in permutations):
        raise ValueError("monodromy permutations must have one common degree")
    for permutation in permutations:
        _permutation_cycles(permutation, include_fixed=True)
    if len(_monodromy_orbit(permutations)) != degree:
        raise ValueError("the monodromy action is not transitive")

    branch_cycles = tuple(
        _permutation_cycles(permutation, include_fixed=True)
        for permutation in permutations)
    base_vertices = tuple(("base", sheet) for sheet in range(degree))
    branch_vertices = tuple(
        ("branch", branch_index, cycle_index)
        for branch_index, cycles in enumerate(branch_cycles)
        for cycle_index in range(len(cycles)))
    vertices = base_vertices + branch_vertices

    edges = []
    edge_lookup = {}
    rotation = {vertex: [] for vertex in vertices}
    for branch_index, cycles in enumerate(branch_cycles):
        for cycle_index, cycle in enumerate(cycles):
            branch_vertex = ("branch", branch_index, cycle_index)
            # A positive base loop advances sheets in ``cycle`` order.  The
            # lifted radial half-edges point out of the ramification place,
            # so their positive surface rotation is the inverse cyclic order.
            # Transpositions conceal this distinction; higher ramification
            # makes it essential for the intersection orientation.
            for sheet in reversed(cycle):
                edge_index = len(edges)
                base_vertex = ("base", sheet)
                edges.append(_LiftedGraphEdge(
                    base_vertex, branch_vertex, branch_index, sheet))
                edge_lookup[branch_index, sheet] = edge_index
                rotation[branch_vertex].append((edge_index, 1))
    for sheet in range(degree):
        rotation[("base", sheet)] = [
            (edge_lookup[branch_index, sheet], 0)
            for branch_index in range(len(permutations))]

    parent = {vertex: vertex for vertex in vertices}

    def find(vertex):
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex

    tree_edges = []
    chord_edges = []
    for edge_index, edge in enumerate(edges):
        left_root = find(edge.tail)
        right_root = find(edge.head)
        if left_root == right_root:
            chord_edges.append(edge_index)
        else:
            parent[left_root] = right_root
            tree_edges.append(edge_index)
    if len(tree_edges) != len(vertices) - 1:
        raise ValueError("lifted monodromy graph is disconnected")

    adjacency = {vertex: [] for vertex in vertices}
    for edge_index in tree_edges:
        edge = edges[edge_index]
        adjacency[edge.tail].append((edge.head, edge_index, 1))
        adjacency[edge.head].append((edge.tail, edge_index, -1))
    cycles = []
    for chord in chord_edges:
        edge = edges[chord]
        coefficients = [0] * len(edges)
        coefficients[chord] = 1
        for tree_edge, orientation in _tree_path(
                adjacency, edge.head, edge.tail):
            coefficients[tree_edge] += orientation
        cycles.append(tuple(coefficients))

    intersection = tuple(
        tuple(_ribbon_cycle_intersection(
            edges, rotation, left, right) for right in cycles)
        for left in cycles)
    genus, unused_ramification = _riemann_hurwitz_genus(
        degree, permutations)
    boundary_components = _ribbon_boundary_count(edges, rotation)
    intersection_rank = _integer_matrix_rank(intersection)
    expected_cycle_rank = 2 * genus + degree - 1
    if len(cycles) != expected_cycle_rank:
        raise ValueError("lifted graph has an inconsistent cycle rank")
    if boundary_components != degree:
        raise ValueError("lifted graph has an inconsistent ribbon boundary")
    if intersection_rank != 2 * genus:
        raise ValueError("lifted graph has an inconsistent intersection rank")
    return _LiftedMonodromyGraph(
        degree=degree,
        genus=genus,
        permutations=permutations,
        branch_cycles=branch_cycles,
        vertices=vertices,
        edges=tuple(edges),
        rotation={vertex: tuple(half_edges)
                  for vertex, half_edges in rotation.items()},
        tree_edges=tuple(tree_edges),
        chord_edges=tuple(chord_edges),
        cycles=tuple(cycles),
        intersection=intersection,
        boundary_components=boundary_components,
        intersection_rank=intersection_rank,
    )


def _graph_cycle_word(graph, cycle):
    """Convert one simple ribbon-graph cycle to branch-loop instructions."""
    if len(cycle) != len(graph.edges):
        raise ValueError("graph cycle has the wrong edge dimension")
    outgoing = {}
    incoming = {}
    for edge_index, coefficient in enumerate(cycle):
        if coefficient not in (-1, 0, 1):
            raise ValueError("graph cycle must be a simple oriented cycle")
        if not coefficient:
            continue
        edge = graph.edges[edge_index]
        if coefficient == 1:
            start, end = edge.tail, edge.head
        else:
            start, end = edge.head, edge.tail
        if start in outgoing or end in incoming:
            raise ValueError("graph cycle is not a single simple cycle")
        outgoing[start] = end
        incoming[end] = start
    if set(outgoing) != set(incoming):
        raise ValueError("graph cycle has nonzero boundary")

    base_vertices = sorted(
        vertex for vertex in outgoing if vertex[0] == "base")
    if not base_vertices:
        raise ValueError("graph cycle contains no base-sheet vertex")
    start_vertex = base_vertices[0]
    current = start_vertex
    steps = []
    visited = set()
    while current not in visited:
        visited.add(current)
        branch_vertex = outgoing.get(current)
        if (current[0] != "base" or branch_vertex is None
                or branch_vertex[0] != "branch"):
            raise ValueError("graph cycle does not alternate vertex types")
        if branch_vertex in visited:
            raise ValueError("graph cycle repeats a branch vertex")
        visited.add(branch_vertex)
        next_base = outgoing.get(branch_vertex)
        if next_base is None or next_base[0] != "base":
            raise ValueError("graph cycle does not alternate vertex types")
        branch_index = branch_vertex[1]
        permutation = graph.permutations[branch_index]
        sheet = current[1]
        target = next_base[1]
        image = sheet
        turns = 0
        while turns < graph.degree and image != target:
            image = permutation[image]
            turns += 1
        if not turns or image != target:
            raise ValueError("branch transition disagrees with monodromy")
        steps.append(_BranchLoopStep(branch_index, turns))
        current = next_base
    if current != start_vertex or visited != set(outgoing):
        raise ValueError("graph cycle is disconnected")
    return _GraphCycleWord(start_vertex[1], tuple(steps))


def _monodromy_branch_continuations(ctx, monodromy):
    """Return finite branch loops followed by a numerical infinity loop."""
    finite = tuple(monodromy.continuations)
    if not finite:
        raise ValueError("finite monodromy continuations are required")
    finite_product = finite[0]
    for continuation in finite[1:]:
        finite_product = _concatenate_plane_curve_continuations(
            ctx, finite_product, continuation)
    infinity = _reverse_plane_curve_continuation(ctx, finite_product)
    if infinity.permutation != monodromy.infinity_permutation:
        raise ValueError("numerical infinity monodromy is inconsistent")
    return finite + (infinity,)


def _monodromy_graph_permutations(monodromy):
    """Return the permutations for branched fibres in numerical order.

    Finite entries retain their continuation indices.  Infinity is appended
    only when it is ramified: an identity permutation represents several
    ordinary points over infinity, not a branch fibre of the covering.
    """
    permutations = monodromy.permutations
    identity = tuple(range(len(monodromy.base_sheets)))
    if monodromy.infinity_permutation != identity:
        permutations += (monodromy.infinity_permutation,)
    return permutations


def _continue_graph_cycle_word(ctx, branch_continuations, word):
    """Return the numerical continuation represented by a branch-loop word."""
    result = None
    for step in word.steps:
        try:
            continuation = branch_continuations[step.branch_index]
        except IndexError:
            raise ValueError("branch-loop word has an invalid branch index")
        for unused in range(step.turns):
            if result is None:
                result = continuation
            else:
                result = _concatenate_plane_curve_continuations(
                    ctx, result, continuation)
    if result is None:
        raise ValueError("branch-loop word must not be empty")
    return result


def _numerical_graph_cycles_from_continuations(
        ctx, graph, branch_continuations):
    """Lift graph cycles using continuations in the graph's ribbon order."""
    branch_continuations = tuple(branch_continuations)
    if tuple(continuation.permutation
             for continuation in branch_continuations) != graph.permutations:
        raise ValueError("graph and numerical monodromy systems differ")
    words = tuple(_graph_cycle_word(graph, cycle)
                  for cycle in graph.cycles)
    chains = []
    for word in words:
        continuation = _continue_graph_cycle_word(
            ctx, branch_continuations, word)
        chain = _prepare_lifted_path_chain((
            (1, continuation, word.start_sheet),))
        if _lifted_path_chain_boundary(ctx, chain):
            raise ValueError("numerical graph cycle has nonzero boundary")
        chains.append(chain)
    return _NumericalGraphCycles(
        branch_continuations=branch_continuations,
        words=words,
        chains=tuple(chains),
    )


def _numerical_graph_cycles(ctx, graph, monodromy):
    """Lift cycles from a finite-then-infinity monodromy system."""
    expected = _monodromy_graph_permutations(monodromy)
    if graph.permutations != expected:
        raise ValueError("graph and numerical monodromy systems differ")
    branch_continuations = _monodromy_branch_continuations(ctx, monodromy)
    if len(branch_continuations) != len(expected):
        # The only omitted continuation is an unramified identity at infinity.
        branch_continuations = branch_continuations[:-1]
    return _numerical_graph_cycles_from_continuations(
        ctx, graph, branch_continuations)


def _ordered_monodromy_graph(monodromy):
    """Construct a graph from explicitly ribbon-ordered generators."""
    identity = tuple(range(len(monodromy.base_sheets)))
    permutations = tuple(
        generator.permutation for generator in monodromy.ribbon_generators
        if generator.permutation != identity)
    return _lifted_monodromy_graph(permutations)


def _numerical_ordered_graph_cycles(ctx, graph, monodromy):
    """Lift graph cycles using explicitly ribbon-ordered continuations."""
    identity = tuple(range(len(monodromy.base_sheets)))
    generators = tuple(
        generator for generator in monodromy.ribbon_generators
        if generator.permutation != identity)
    expected = tuple(generator.permutation for generator in generators)
    if graph.permutations != expected:
        raise ValueError("graph and ordered monodromy systems differ")
    continuations = tuple(
        generator.continuation for generator in generators)
    return _numerical_graph_cycles_from_continuations(
        ctx, graph, continuations)


def _transform_lifted_path_chains(chains, transformation):
    """Apply an integer row transformation to lifted-path chains."""
    chains = tuple(chains)
    transformed = []
    for row in transformation:
        if len(row) != len(chains):
            raise ValueError("chain transformation has the wrong width")
        terms = []
        for coefficient, chain in zip(row, chains):
            if not isinstance(coefficient, int):
                raise ValueError("chain transformation must be integral")
            terms.extend(
                (coefficient * term.coefficient,
                 term.continuation, term.sheet)
                for term in chain.terms)
        transformed.append(_prepare_lifted_path_chain(terms))
    return tuple(transformed)


def _shortest_monodromy_word(permutations, start, target):
    """Return generator indices carrying one sheet label to another."""
    pending = [(start, ())]
    visited = {start}
    while pending:
        sheet, word = pending.pop(0)
        if sheet == target:
            return word
        for index, permutation in enumerate(permutations):
            image = permutation[sheet]
            if image not in visited:
                visited.add(image)
                pending.append((image, word + (index,)))
    raise ValueError("monodromy does not connect the requested sheets")


def _concatenate_continuation_sequence(ctx, continuations):
    result = None
    for continuation in continuations:
        result = (continuation if result is None else
                  _concatenate_plane_curve_continuations(
                      ctx, result, continuation))
    return result


def _chain_as_common_base_loop(ctx, chain, monodromy):
    """Realize an integral cycle chain as one loop based on sheet zero."""
    generators = tuple(monodromy.ribbon_generators)
    permutations = tuple(
        generator.permutation for generator in generators)
    loops = []
    for term in chain.terms:
        word = _shortest_monodromy_word(
            permutations, 0, term.sheet)
        connector = _concatenate_continuation_sequence(
            ctx, tuple(generators[index].continuation for index in word))
        pieces = []
        if connector is not None:
            pieces.append(connector)
        pieces.append(term.continuation)
        if connector is not None:
            pieces.append(_reverse_plane_curve_continuation(ctx, connector))
        loop = _concatenate_continuation_sequence(ctx, pieces)
        if term.coefficient < 0:
            loop = _reverse_plane_curve_continuation(ctx, loop)
        loops.extend((loop,) * abs(term.coefficient))
    if not loops:
        raise ValueError("canonical cycle must not be empty")
    result = _concatenate_continuation_sequence(ctx, loops)
    start, end = _lifted_path_endpoints(_LiftedPathTerm(1, result, 0))
    if not _same_numerical_place(ctx, start, end):
        raise ValueError("canonical cycle did not close at the common base")
    return result


def _radial_plane_curve_monodromy(
        ctx, curve, branch_points, base_point=None, circle_steps=24,
        max_refinements=12):
    """Compute an ordered monodromy system for complex branch values.

    Finite positive loops use guarded radial spokes from an exterior base
    point.  Their counter-clockwise product order and clockwise outward
    ribbon order are retained separately.  A clockwise outer circle supplies
    the positive generator at infinity geometrically.
    """
    points = tuple(ctx.convert(point) for point in branch_points)
    base_point, center, radii, minimum_clearance = (
        _radial_branch_geometry(ctx, points, base_point=base_point))
    base_sheets = _ordered_plane_curve_sheets(ctx, curve, base_point)
    geometry = tuple(zip(points, radii))
    product_geometry = tuple(sorted(
        geometry, key=lambda item: ctx.arg(item[0] - base_point)))
    finite_generators = []
    for point, radius in product_geometry:
        path = _radial_branch_loop_path(
            ctx, base_point, point, radius, circle_steps=circle_steps)
        continuation = _continue_plane_curve_sheets_adaptive(
            ctx, curve, path, initial_sheets=base_sheets,
            max_refinements=max_refinements)
        if continuation.permutation is None:
            raise ValueError("radial monodromy paths must be closed")
        if continuation.permutation == tuple(range(curve.y_degree)):
            raise ValueError("finite branch loop has identity monodromy")
        finite_generators.append(_BranchGenerator(
            kind="finite",
            value=point,
            permutation=continuation.permutation,
            continuation=continuation,
            radius=radius,
        ))

    outer_steps = 4 * circle_steps
    outer_vector = base_point - center
    infinity_path = tuple(
        center + outer_vector * ctx.exp(
            -2 * ctx.j * ctx.pi * step / outer_steps)
        for step in range(outer_steps + 1))
    infinity_continuation = _continue_plane_curve_sheets_adaptive(
        ctx, curve, infinity_path, initial_sheets=base_sheets,
        max_refinements=max_refinements)
    if infinity_continuation.permutation is None:
        raise ValueError("infinity monodromy path must be closed")
    infinity = _BranchGenerator(
        kind="infinity",
        value=ctx.inf,
        permutation=infinity_continuation.permutation,
        continuation=infinity_continuation,
        radius=abs(outer_vector),
    )
    product_generators = tuple(finite_generators) + (infinity,)
    product = tuple(range(curve.y_degree))
    for generator in product_generators:
        product = _compose_permutations(generator.permutation, product)
    if product != tuple(range(curve.y_degree)):
        raise ValueError("radial monodromy product is not the identity")

    ribbon_generators = tuple(reversed(finite_generators)) + (infinity,)
    permutations = tuple(
        generator.permutation for generator in ribbon_generators)
    if len(_monodromy_orbit(permutations)) != curve.y_degree:
        raise ValueError("the monodromy action is not transitive")
    genus, ramification = _riemann_hurwitz_genus(
        curve.y_degree, permutations)
    return _OrderedMonodromyData(
        base_point=base_point,
        base_sheets=base_sheets,
        center=center,
        branch_points=tuple(
            generator.value for generator in finite_generators),
        product_generators=product_generators,
        ribbon_generators=ribbon_generators,
        ramification=ramification,
        genus=genus,
        minimum_clearance=minimum_clearance,
    )


def _normalise_algebraic_curve_input(ctx, curve):
    """Return ``(prepared_curve, hyperelliptic_coefficients_or_none)``."""
    if hasattr(curve, "items"):
        return _prepare_plane_curve(ctx, curve), None
    try:
        values = tuple(curve)
    except TypeError:
        raise ValueError("curve must be coefficients or sparse plane terms")
    if not values:
        raise ValueError("curve must not be empty")
    if all(isinstance(term, (tuple, list)) and len(term) == 3
           for term in values):
        terms = {}
        for x_power, y_power, coefficient in values:
            try:
                coefficient = ctx.convert(coefficient)
            except (TypeError, ValueError):
                raise ValueError("polynomial coefficients must be numbers")
            key = (x_power, y_power)
            terms[key] = terms.get(key, ctx.zero) + coefficient
        return _prepare_plane_curve(ctx, terms), None
    coefficients = tuple(ctx.convert(value) for value in values)
    terms = {(0, 2): ctx.one}
    terms.update({
        (degree, 0): -coefficient
        for degree, coefficient in enumerate(coefficients)
        if coefficient
    })
    return _prepare_plane_curve(ctx, terms), coefficients


def _period_matrix_from_columns(ctx, columns, start, count, genus):
    return ctx.matrix([
        [columns[column][start + row] for column in range(2 * genus)]
        for row in range(count)
    ])


def _normalised_differentials(ctx, differentials, a_periods):
    inverse = a_periods ** -1
    result = []
    for row in range(len(differentials)):
        coefficients = tuple(
            inverse[row, column]
            for column in range(len(differentials)))

        def differential(x, y, coefficients=coefficients):
            return ctx.fsum(
                coefficient * form(x, y)
                for coefficient, form in zip(coefficients, differentials))

        result.append(differential)
    return tuple(result)


def _riemann_constant_vector(
        ctx, curve, monodromy, canonical_cycles, differentials,
        a_periods, tau, quadrature_order):
    """Evaluate the Riemann-constant contour formula at the cycle base."""
    genus = len(differentials)
    normalised = _normalised_differentials(
        ctx, differentials, a_periods)
    cycle_integrals = []
    for cycle in canonical_cycles[:genus]:
        loop = _chain_as_common_base_loop(ctx, cycle, monodromy)
        cycle_integrals.append(_integrate_plane_curve_path_iterated(
            ctx, curve, loop, normalised, sheet=0,
            quadrature_order=quadrature_order))
    value = ctx.matrix(genus, 1)
    for row in range(genus):
        correction = ctx.fsum(
            cycle_integrals[cycle].iterated[row][cycle]
            for cycle in range(genus) if cycle != row)
        value[row] = (1 + tau[row, row]) / 2 - correction
    return value, tuple(cycle_integrals), normalised


def _jacobian_characteristic(ctx, value, tau):
    """Express a Jacobian point as the literal pair ``(a, b)``."""
    genus = tau.rows
    lattice = ctx.matrix([
        [ctx.re(1 if row == column else 0)
         for column in range(genus)]
        + [ctx.re(tau[row, column]) for column in range(genus)]
        for row in range(genus)
    ] + [
        [ctx.im(1 if row == column else 0)
         for column in range(genus)]
        + [ctx.im(tau[row, column]) for column in range(genus)]
        for row in range(genus)
    ])
    target = ctx.matrix(
        [ctx.re(entry) for entry in value]
        + [ctx.im(entry) for entry in value])
    coordinates = lattice ** -1 * target
    reduced = []
    snap_tolerance = ctx.power(ctx.eps, ctx.mpf("0.25"))
    for coordinate in coordinates:
        coordinate -= ctx.floor(coordinate)
        half = ctx.nint(2 * coordinate) / 2
        if abs(coordinate - half) <= snap_tolerance:
            coordinate = half % 1
        reduced.append(+coordinate)
    b = tuple(reduced[:genus])
    a = tuple(reduced[genus:])
    return a, b


def _normalise_curve_place(ctx, curve, place, name="place"):
    """Return a validated regular finite place on a prepared curve."""
    if isinstance(place, CurvePlace):
        place = (place.x, place.y)
    try:
        x, y = place
    except (TypeError, ValueError):
        raise ValueError(name + " must be a finite (x, y) pair")
    x = ctx.convert(x)
    y = ctx.convert(y)
    if not ctx.isfinite(x) or not ctx.isfinite(y):
        raise ValueError(name + " must be a finite (x, y) pair")
    scale = max(ctx.one, ctx.fsum(
        abs(coefficient * x ** x_power * y ** y_power)
        for x_power, y_power, coefficient in curve.terms))
    if abs(_evaluate_plane_polynomial(ctx, curve, x, y)) > (
            100 * ctx.sqrt(ctx.eps) * scale):
        raise ValueError(name + " must lie on the curve")
    if abs(_evaluate_plane_derivative(ctx, curve, x, y, "y")) <= (
            100 * ctx.sqrt(ctx.eps)):
        raise ValueError(name + " must be regular for the x projection")
    return _PlaneCurvePlace(x, y)


def _guarded_open_path(ctx, start, end, branch_points):
    """Choose a simple deterministic polyline avoiding critical values."""
    scale = max(
        [ctx.one, abs(start), abs(end)]
        + [abs(point) for point in branch_points])
    midpoint = (start + end) / 2
    candidates = [(start, end)]
    for direction in (ctx.one, -ctx.one, ctx.j, -ctx.j,
                      1 + ctx.j, 1 - ctx.j, -1 + ctx.j, -1 - ctx.j):
        candidates.append((start, midpoint + direction * scale, end))

    def clearance(path):
        return min(
            _point_segment_distance(ctx, point, left, right)
            for point in branch_points
            for left, right in zip(path, path[1:]))

    return max(candidates, key=clearance)


def _finite_base_abel_value(
        ctx, curve, monodromy, branch_points, base_place,
        normalised_differentials, quadrature_order):
    path = _guarded_open_path(
        ctx, monodromy.base_point, base_place.x, branch_points)
    continuation = _continue_plane_curve_sheets_adaptive(
        ctx, curve, path, initial_sheets=monodromy.base_sheets,
        max_refinements=20)
    target_sheet = min(
        range(curve.y_degree),
        key=lambda index: abs(
            continuation.fibres[-1][index] - base_place.y))
    match_error = abs(
        continuation.fibres[-1][target_sheet] - base_place.y)
    scale = max(ctx.one, abs(base_place.y))
    if match_error > 100 * ctx.sqrt(ctx.eps) * scale:
        raise ValueError("base_place could not be matched to a sheet")
    generators = tuple(monodromy.ribbon_generators)
    word = _shortest_monodromy_word(
        tuple(generator.permutation for generator in generators),
        0, target_sheet)
    connector = _concatenate_continuation_sequence(
        ctx, tuple(generators[index].continuation for index in word))
    if connector is not None:
        continuation = _concatenate_plane_curve_continuations(
            ctx, connector, continuation)
    integral = _integrate_plane_curve_path(
        ctx, curve, continuation, normalised_differentials, sheet=0,
        quadrature_order=quadrature_order)
    return ctx.matrix(integral.values)


def _jacobian_lattice_matrix(ctx, tau):
    genus = tau.rows
    return ctx.matrix([
        [ctx.re(1 if row == column else 0)
         for column in range(genus)]
        + [ctx.re(tau[row, column]) for column in range(genus)]
        for row in range(genus)
    ] + [
        [ctx.im(1 if row == column else 0)
         for column in range(genus)]
        + [ctx.im(tau[row, column]) for column in range(genus)]
        for row in range(genus)
    ])


def _reduce_jacobian_point(ctx, value, tau, lattice_inverse):
    genus = tau.rows
    coordinates = lattice_inverse * ctx.matrix(
        [ctx.re(entry) for entry in value]
        + [ctx.im(entry) for entry in value])
    shift = tuple(ctx.nint(entry) for entry in coordinates)
    return value - ctx.matrix([
        shift[row] + ctx.fsum(
            tau[row, column] * shift[genus + column]
            for column in range(genus))
        for row in range(genus)
    ])


def _theta_divisor_samples(
        ctx, curve, monodromy, branch_points, normalised_differentials,
        genus, quadrature_order):
    """Construct deterministic degree ``g-1`` Abel images."""
    if genus == 1:
        return (ctx.matrix([ctx.zero]),)
    count = 2 * genus + 1
    center = monodromy.center
    radius = max(abs(point - center) for point in branch_points)
    point_values = []
    for index in range(count):
        angle = 2 * ctx.pi * (index + ctx.mpf("0.173")) / count
        x = center + ctx.mpf("0.35") * radius * ctx.exp(ctx.j * angle)
        sheets = _ordered_plane_curve_sheets(ctx, curve, x)
        place = _PlaneCurvePlace(x, sheets[index % curve.y_degree])
        point_values.append(_finite_base_abel_value(
            ctx, curve, monodromy, branch_points, place,
            normalised_differentials, quadrature_order))
    samples = []
    for index in range(2 * genus):
        value = ctx.matrix(genus, 1)
        for offset in range(genus - 1):
            value += point_values[(index + offset) % count]
        samples.append(value)
    return tuple(samples)


def _theta_divisor_riemann_constant(
        ctx, tau, samples, initial_value):
    """Determine the common theta-divisor shift from marked Abel images."""
    genus = tau.rows
    if genus == 1:
        return ctx.matrix([(1 + tau[0, 0]) / 2])
    lattice = _jacobian_lattice_matrix(ctx, tau)
    lattice_inverse = lattice ** -1
    zero_index = (0,) * genus
    unit_indices = tuple(
        tuple(1 if row == column else 0 for row in range(genus))
        for column in range(genus))
    fit_samples = samples[:genus]
    tolerance = 100 * ctx.power(ctx.eps, ctx.mpf("0.35"))

    def reduced(value):
        return _reduce_jacobian_point(
            ctx, value, tau, lattice_inverse)

    def residual(value, selected):
        result = []
        for sample in selected:
            theta = ctx.rtheta(reduced(sample + value), tau)
            result.extend((ctx.re(theta), ctx.im(theta)))
        return ctx.matrix(result)

    def residual_jacobian(value):
        result = []
        jacobian = ctx.matrix(2 * genus, 2 * genus)
        for sample_index, sample in enumerate(fit_samples):
            jet = ctx.rtheta_jet(reduced(sample + value), tau, 1)
            theta = jet[zero_index]
            result.extend((ctx.re(theta), ctx.im(theta)))
            for column, unit_index in enumerate(unit_indices):
                derivative = jet[unit_index]
                jacobian[2 * sample_index, column] = ctx.re(derivative)
                jacobian[2 * sample_index, genus + column] = -ctx.im(
                    derivative)
                jacobian[2 * sample_index + 1, column] = ctx.im(
                    derivative)
                jacobian[2 * sample_index + 1, genus + column] = ctx.re(
                    derivative)
        return ctx.matrix(result), jacobian

    def solve(start):
        value = +start
        for unused in range(20):
            current, jacobian = residual_jacobian(value)
            current_norm = ctx.norm(current)
            if current_norm <= tolerance:
                return value
            try:
                correction = ctx.lu_solve(jacobian, -current)
            except (ZeroDivisionError, ValueError):
                return None
            step = ctx.one
            while step >= ctx.mpf("0.001"):
                candidate = value + ctx.matrix([
                    correction[index]
                    + ctx.j * correction[genus + index]
                    for index in range(genus)
                ]) * step
                if ctx.norm(residual(candidate, fit_samples)) < current_norm:
                    value = candidate
                    break
                step /= 2
            else:
                return None
        return None

    best_residual = ctx.inf
    best_value = None
    for mask in range(1 << (2 * genus)):
        half_shift = tuple(
            ctx.mpf("0.5") if mask & (1 << index) else ctx.zero
            for index in range(2 * genus))
        start = initial_value + ctx.matrix([
            half_shift[row] + ctx.fsum(
                tau[row, column] * half_shift[genus + column]
                for column in range(genus))
            for row in range(genus)
        ])
        candidate = solve(start)
        if candidate is None:
            continue
        check = residual(candidate, samples)
        check_residual = max(abs(entry) for entry in check)
        if check_residual < best_residual:
            best_residual = check_residual
            best_value = candidate
        if check_residual <= tolerance:
            return reduced(candidate)
    raise ctx.NoConvergence(
        "failed to determine the Riemann constant from theta-divisor samples")


# Cached computational stages
# ---------------------------

_MONODROMY_CIRCLE_STEPS = 12
_MONODROMY_MAX_REFINEMENTS = 20


def _curve_cache_state(ctx):
    """Return the numerical state that keys the curve stage caches."""
    rounding = getattr(ctx, "rounding", None)
    trap_complex = getattr(ctx, "trap_complex", None)
    return ctx.prec, rounding, trap_complex


def _curve_stage_cache(maxsize):
    """Cache a curve stage by its key and the context's numerical state.

    Stage keys are prepared curves or tuples containing them, so the cache
    never sees unhashable user input.  Cached values are immutable records;
    public functions assemble matrices from them, so a cached result can
    never be mutated through the public API.
    """
    def decorator(f):
        @lru_cache(maxsize=maxsize)
        def cached(unused_state, ctx, key):
            return f(ctx, key)

        @wraps(f)
        def wrapper(ctx, key):
            return cached(_curve_cache_state(ctx), ctx, key)

        wrapper.cache_info = cached.cache_info
        wrapper.cache_clear = cached.cache_clear
        return wrapper
    return decorator


@_curve_stage_cache(32)
def _stage_branch_locus(ctx, curve):
    """Return ``(branch_values, resultant)`` for a prepared plane curve."""
    return _plane_curve_critical_values(ctx, curve)


@_curve_stage_cache(8)
def _stage_monodromy(ctx, curve):
    """Return the guarded radial monodromy system of a prepared curve."""
    branch_values, unused_resultant = _stage_branch_locus(ctx, curve)
    return _radial_plane_curve_monodromy(
        ctx, curve, branch_values,
        circle_steps=_MONODROMY_CIRCLE_STEPS,
        max_refinements=_MONODROMY_MAX_REFINEMENTS)


@_curve_stage_cache(8)
def _stage_monodromy_graph(ctx, curve):
    """Return ``(lifted_graph, symplectic_reduction)`` for a curve."""
    monodromy = _stage_monodromy(ctx, curve)
    graph = _ordered_monodromy_graph(monodromy)
    reduction = _symplectic_reduce_intersection(graph.intersection)
    return graph, reduction


@_curve_stage_cache(8)
def _stage_canonical_cycles(ctx, curve):
    """Return lifted-path chains realizing a canonical homology basis."""
    graph, reduction = _stage_monodromy_graph(ctx, curve)
    numerical = _numerical_ordered_graph_cycles(
        ctx, graph, _stage_monodromy(ctx, curve))
    return _transform_lifted_path_chains(
        numerical.chains, reduction.transformation)


@_curve_stage_cache(16)
def _stage_cycle_integrals(ctx, key):
    """Integrate differential forms over the canonical cycles of a curve.

    ``key`` is ``(curve, forms, quadrature_order)``.  The result is
    ``(columns, max_sheet_residual)`` with one column of form values per
    canonical cycle.
    """
    curve, forms, quadrature_order = key
    forms = tuple(forms)
    genus = _stage_monodromy(ctx, curve).genus
    chains = _stage_canonical_cycles(ctx, curve)
    columns = []
    max_sheet_residual = ctx.zero
    for chain in chains[:2 * genus]:
        integral = _integrate_lifted_path_chain(
            ctx, curve, chain, forms, quadrature_order=quadrature_order)
        columns.append(integral.values)
        max_sheet_residual = max(
            max_sheet_residual, integral.max_sheet_residual)
    return tuple(columns), max_sheet_residual


def _curve_differential_sequence(differentials, name):
    """Return a validated tuple of differential callables."""
    try:
        forms = tuple(differentials)
    except TypeError:
        raise ValueError(name + " must be a sequence of callables")
    if not forms or any(not callable(value) for value in forms):
        raise ValueError(name + " must be a sequence of callables")
    return forms


def _tau_imaginary_eigenvalues(ctx, tau):
    """Return the eigenvalues of the imaginary part of a period matrix."""
    genus = tau.rows
    imaginary_tau = ctx.matrix([
        [ctx.im(tau[row, column]) for column in range(genus)]
        for row in range(genus)])
    return tuple(ctx.eigsy(imaginary_tau, eigvals_only=True))

# Public curve functions
# ---------------------


@defun
def curve_branch_locus(ctx, curve):
    r"""Return the finite branch locus of a plane algebraic curve.

    ``curve`` may be an ascending coefficient sequence defining
    ``y**2 = P(x)``, a sparse mapping from ``(x_power, y_power)`` pairs to
    a coefficient, or a sequence of ``(x_power, y_power, coefficient)``
    terms.

    The returned ``CurveBranchLocus`` record contains the degree of the
    ``x`` projection, the distinct finite branch values above which the
    projection ramifies, and the ascending coefficients of the
    y-derivative resultant whose roots they are.  Ramification above
    infinity is reported by :func:`~mpmath.curve_monodromy` instead,
    because it requires monodromy rather than the resultant alone.

    The lemniscatic curve :math:`y^2 = x^3 - x` has a two-sheeted
    projection with three finite branch values::

        >>> from mpmath import curve_branch_locus
        >>> locus = curve_branch_locus((0, -1, 0, 1))
        >>> locus.degree
        2
        >>> locus.branch_values
        (mpf('-1.0'), mpf('0.0'), mpf('1.0'))
    """
    prepared, unused_hyperelliptic = _normalise_algebraic_curve_input(
        ctx, curve)
    branch_values, resultant = _stage_branch_locus(ctx, prepared)
    return CurveBranchLocus(prepared.y_degree, branch_values, resultant)


@defun
def curve_monodromy(ctx, curve):
    r"""Return the monodromy of a plane algebraic curve over the x-line.

    The curve is continued numerically along guarded radial loops around
    the finite branch values, from an exterior base point chosen
    automatically.  A large outer loop supplies the monodromy at infinity
    geometrically.  The returned ``CurveMonodromy`` record contains the
    computational base point, its ordered fibre, the branch values, the
    counter-clockwise product-ordered local permutations, the permutation
    at infinity, the total ramification, the genus from
    Riemann--Hurwitz, the transitivity and product identities of the
    permutation system, and the minimum geometric clearance of the
    continuation paths.

    The sheet labels refer to the internally selected base fibre.  The
    routing continuations used to compute them are private.

    Each finite branch value of the lemniscatic curve
    :math:`y^2 = x^3 - x` exchanges its two sheets, as does infinity::

        >>> from mpmath import curve_monodromy
        >>> monodromy = curve_monodromy((0, -1, 0, 1))
        >>> monodromy.genus
        1
        >>> monodromy.permutations
        ((1, 0), (1, 0), (1, 0))
        >>> monodromy.infinity_permutation
        (1, 0)
    """
    prepared, unused_hyperelliptic = _normalise_algebraic_curve_input(
        ctx, curve)
    monodromy = _stage_monodromy(ctx, prepared)
    degree = prepared.y_degree
    generators = tuple(monodromy.product_generators)
    permutations = tuple(
        generator.permutation for generator in generators[:-1])
    infinity_permutation = generators[-1].permutation
    all_permutations = permutations + (infinity_permutation,)
    product = tuple(range(degree))
    for permutation in all_permutations:
        product = _compose_permutations(permutation, product)
    transitive = len(_monodromy_orbit(all_permutations)) == degree
    return CurveMonodromy(
        base_point=monodromy.base_point,
        base_sheets=monodromy.base_sheets,
        branch_values=monodromy.branch_points,
        permutations=permutations,
        infinity_permutation=infinity_permutation,
        ramification=monodromy.ramification,
        genus=monodromy.genus,
        transitive=transitive,
        product_identity=product == tuple(range(degree)),
        minimum_clearance=monodromy.minimum_clearance)


@defun
def curve_genus(ctx, curve):
    r"""Return the genus of a plane algebraic curve by monodromy.

    The genus is obtained from the Riemann--Hurwitz formula applied to
    the monodromy of the ``x`` projection, including the permutation at
    infinity.  The returned ``CurveGenus`` record also records the
    projection degree and the total ramification, so the Riemann--Hurwitz
    balance :math:`2g-2 = -2d+r` can be checked directly::

        >>> from mpmath import curve_genus
        >>> curve_genus((0, -1, 0, 1))
        CurveGenus(genus=1, degree=2, ramification=4)
    """
    prepared, unused_hyperelliptic = _normalise_algebraic_curve_input(
        ctx, curve)
    monodromy = _stage_monodromy(ctx, prepared)
    return CurveGenus(
        monodromy.genus, prepared.y_degree, monodromy.ramification)


@defun
def curve_homology(ctx, curve):
    r"""Return a canonical homology basis of a plane algebraic curve.

    The lifted monodromy graph of the ``x`` projection is reduced to a
    primitive symplectic homology basis.  The returned
    ``CurveHomology`` record contains the genus, the number of independent
    graph cycles, the number of boundary components of the lifted ribbon
    graph (the places above infinity), the rank of the graph intersection
    form, the dimension of its radical, the resulting canonical
    intersection form, and the integer transformation realizing the
    canonical basis from the graph cycles.

    The lifted graph of the lemniscatic curve :math:`y^2 = x^3 - x` has
    three independent cycles and two boundary components::

        >>> from mpmath import curve_homology
        >>> homology = curve_homology((0, -1, 0, 1))
        >>> homology.genus, homology.boundary_components
        (1, 2)
        >>> homology.intersection_form
        ((0, 1, 0), (-1, 0, 0), (0, 0, 0))
    """
    prepared, unused_hyperelliptic = _normalise_algebraic_curve_input(
        ctx, curve)
    graph, reduction = _stage_monodromy_graph(ctx, prepared)
    return CurveHomology(
        genus=reduction.genus,
        cycle_count=len(graph.cycles),
        boundary_components=graph.boundary_components,
        intersection_rank=graph.intersection_rank,
        radical_rank=reduction.radical_rank,
        intersection_form=reduction.form,
        transformation=reduction.transformation)


@defun
def curve_periods(ctx, curve, differentials=None, *,
                  second_differentials=None):
    r"""Return the period matrices of a plane algebraic curve.

    ``curve`` uses the input forms accepted by
    :func:`~mpmath.curve_branch_locus`.  A hyperelliptic coefficient
    sequence without supplied differentials is dispatched to the
    specialized engine :func:`~mpmath.hyperelliptic_periods`.

    A general plane curve requires ``differentials``, a sequence of one
    holomorphic differential callable ``f(x, y)`` per genus, supplying the
    coefficient of ``dx``.  Optional ``second_differentials`` supply the
    same number of second-kind forms; they are integrated on the same
    cycles, with the classical convention ``2*eta = -integral_a(dr)``.

    The returned ``CurvePeriods`` record contains the differential basis,
    the half-period matrices ``omega`` and ``omega_prime``, the normalized
    Riemann matrix ``tau = omega**-1 * omega_prime``, the optional
    second-kind half-period matrices ``eta`` and ``eta_prime`` and
    ``kappa = eta * omega**-1``, and numerical quality residuals.  A
    non-positive-definite normalized period matrix raises ``ValueError``,
    because it always indicates an invalid differential count or basis.

    The normalized Riemann matrix of the lemniscatic curve
    :math:`y^2 = x^3 - x` is :math:`i`::

        >>> from mpmath import curve_periods, curve_validate, mp
        >>> mp.dps = 15
        >>> data = curve_periods((0, -1, 0, 1))
        >>> mp.re(data.tau[0, 0]), mp.im(data.tau[0, 0])
        (mpf('0.0'), mpf('1.0'))

    A general plane curve needs a supplied holomorphic basis, given as
    callables returning the coefficient of ``dx``::

        >>> curve = {(0, 2): 1, (1, 0): 1, (3, 0): -1}
        >>> data = curve_periods(curve, (lambda x, y: 1 / y,))
        >>> curve_validate(data).passed
        True
    """
    prepared, hyperelliptic_coefficients = _normalise_algebraic_curve_input(
        ctx, curve)
    if hyperelliptic_coefficients is not None and differentials is None:
        if second_differentials is not None:
            raise ValueError(
                "supplied second-kind differentials require a supplied "
                "first-kind basis")
        omega, omega_prime, eta, eta_prime, tau, kappa = (
            ctx.hyperelliptic_periods(
                hyperelliptic_coefficients, second_kind=True))
        genus = omega.rows
        return CurvePeriods(
            genus, None, omega, omega_prime, tau, eta, eta_prime, kappa,
            ctx.norm(tau - tau.T), ctx.norm(kappa - kappa.T),
            _tau_imaginary_eigenvalues(ctx, tau), None)

    first_kind = _curve_differential_sequence(
        differentials, "differentials")
    if second_differentials is None:
        second_kind = ()
    else:
        second_kind = _curve_differential_sequence(
            second_differentials, "second_differentials")
    monodromy = _stage_monodromy(ctx, prepared)
    genus = monodromy.genus
    if len(first_kind) != genus:
        raise ValueError(
            "differentials must contain one form per genus")
    if second_kind and len(second_kind) != genus:
        raise ValueError(
            "second_differentials must contain one form per genus")
    forms = first_kind + second_kind
    quadrature_order = max(12, ctx.dps // 2)
    columns, max_sheet_residual = _stage_cycle_integrals(
        ctx, (prepared, forms, quadrature_order))

    periods = _period_matrix_from_columns(
        ctx, columns, 0, genus, genus)
    omega = periods[:, :genus] / 2
    omega_prime = periods[:, genus:] / 2
    raw_tau = omega**-1 * omega_prime
    symmetry_residual = ctx.norm(raw_tau - raw_tau.T)
    tau = (raw_tau + raw_tau.T) / 2
    imaginary_eigenvalues = _tau_imaginary_eigenvalues(ctx, tau)
    if min(imaginary_eigenvalues) <= 0:
        raise ValueError("normalized period matrix is not positive definite")

    eta = eta_prime = kappa = None
    kappa_symmetry_residual = None
    if second_kind:
        second_periods = _period_matrix_from_columns(
            ctx, columns, genus, genus, genus)
        eta = -second_periods[:, :genus] / 2
        eta_prime = -second_periods[:, genus:] / 2
        raw_kappa = eta * omega**-1
        kappa_symmetry_residual = ctx.norm(raw_kappa - raw_kappa.T)
        kappa = (raw_kappa + raw_kappa.T) / 2
    return CurvePeriods(
        genus, first_kind, omega, omega_prime, tau, eta, eta_prime, kappa,
        symmetry_residual, kappa_symmetry_residual, imaginary_eigenvalues,
        max_sheet_residual)


@defun
def curve_riemann_matrix(ctx, curve, differentials=None):
    r"""Return the normalized Riemann matrix of a plane algebraic curve.

    This is a convenience wrapper returning
    ``curve_periods(curve, differentials).tau``; see
    :func:`~mpmath.curve_periods` for the input conventions.

        >>> from mpmath import curve_riemann_matrix, mp
        >>> mp.dps = 15
        >>> tau = curve_riemann_matrix((0, -1, 0, 1))
        >>> mp.im(tau[0, 0])
        mpf('1.0')
    """
    return curve_periods(ctx, curve, differentials).tau


@defun
def curve_validate(ctx, result):
    r"""Validate a result record returned by the curve functions.

    ``result`` is one of ``CurveBranchLocus``, ``CurveMonodromy``,
    ``CurveHomology`` or ``CurvePeriods``.  The returned
    ``CurveValidation`` record contains one named ``CurveCheck`` per
    invariant, the largest
    numerical residual among them, and whether every check passed.  The
    checks are recomputed from the record itself; the underlying curve
    data is not recomputed.

        >>> from mpmath import curve_periods, curve_validate
        >>> report = curve_validate(curve_periods((0, -1, 0, 1)))
        >>> report.passed
        True
        >>> report.checks[0]
        CurveCheck(name='tau_symmetry_residual', value=mpf('0.0'), passed=True)
    """
    checks = []
    residuals = []
    if isinstance(result, CurveBranchLocus):
        values = tuple(result.branch_values)
        scale = max([ctx.one] + [abs(value) for value in values])
        tolerance = ctx.sqrt(ctx.eps) * scale
        separation = min(
            (abs(left - right)
             for index, left in enumerate(values)
             for right in values[index + 1:]),
            default=ctx.zero)
        checks.append(CurveCheck(
            "branch_values_distinct", separation, separation > tolerance))
        checks.append(CurveCheck(
            "resultant_nonconstant", len(result.resultant),
            len(result.resultant) > 1))
    elif isinstance(result, CurveMonodromy):
        degree = len(result.base_sheets)
        all_permutations = result.permutations + (
            result.infinity_permutation,)
        valid = all(
            sorted(permutation) == list(range(degree))
            for permutation in all_permutations)
        checks.append(CurveCheck(
            "permutations_valid", valid, valid))
        orbit = _monodromy_orbit(all_permutations)
        checks.append(CurveCheck(
            "monodromy_transitive", len(orbit), len(orbit) == degree))
        product = tuple(range(degree))
        for permutation in all_permutations:
            product = _compose_permutations(permutation, product)
        checks.append(CurveCheck(
            "monodromy_product_identity", product,
            product == tuple(range(degree))))
        balanced = (2 * result.genus - 2
                    == -2 * degree + result.ramification)
        checks.append(CurveCheck(
            "riemann_hurwitz_balance", balanced, balanced))
    elif isinstance(result, CurveHomology):
        genus = result.genus
        form = result.intersection_form
        antisymmetric = all(
            form[row][column] == -form[column][row]
            for row in range(len(form)) for column in range(len(form)))
        checks.append(CurveCheck(
            "intersection_form_antisymmetric", antisymmetric,
            antisymmetric))
        form_rank = _integer_matrix_rank(form)
        checks.append(CurveCheck(
            "intersection_form_rank", form_rank, form_rank == 2 * genus))
        integral = all(
            isinstance(entry, int)
            for row in result.transformation for entry in row)
        checks.append(CurveCheck(
            "transformation_integral", integral, integral))
        checks.append(CurveCheck(
            "cycle_count", result.cycle_count,
            result.cycle_count == result.intersection_rank
            + result.radical_rank))
    elif isinstance(result, CurvePeriods):
        tau = result.tau
        scale = max([ctx.one] + [
            abs(tau[row, column]) for row in range(tau.rows)
            for column in range(tau.cols)])
        tolerance = 100 * ctx.sqrt(ctx.eps) * scale
        symmetry_residual = ctx.norm(tau - tau.T)
        residuals.append(symmetry_residual)
        checks.append(CurveCheck(
            "tau_symmetry_residual", symmetry_residual,
            symmetry_residual <= tolerance))
        eigenvalues = _tau_imaginary_eigenvalues(ctx, tau)
        checks.append(CurveCheck(
            "tau_imaginary_positive_definite", min(eigenvalues),
            min(eigenvalues) > 0))
        if result.kappa is not None:
            kappa = result.kappa
            kappa_scale = max([ctx.one] + [
                abs(kappa[row, column]) for row in range(kappa.rows)
                for column in range(kappa.cols)])
            kappa_residual = ctx.norm(kappa - kappa.T)
            residuals.append(kappa_residual)
            checks.append(CurveCheck(
                "kappa_symmetry_residual", kappa_residual,
                kappa_residual <= 100 * ctx.sqrt(ctx.eps) * kappa_scale))
        if result.max_sheet_residual is not None:
            residuals.append(result.max_sheet_residual)
            checks.append(CurveCheck(
                "max_sheet_residual", result.max_sheet_residual,
                result.max_sheet_residual <= tolerance))
    else:
        raise ValueError("curve_validate requires a curve result record")
    maximum_residual = max(residuals) if residuals else None
    return CurveValidation(
        type(result).__name__, all(check.passed for check in checks),
        maximum_residual, tuple(checks))


@defun
def curve_fibre(ctx, curve, x):
    r"""Return the labelled fibre of a plane algebraic curve over x.

    ``curve`` uses the input forms accepted by
    :func:`~mpmath.curve_branch_locus`, and ``x`` must be a finite regular
    value of the ``x`` projection: not a branch value, and one over which
    the projection does not drop degree.  The returned tuple contains one
    ``CurvePlace`` record per sheet, ordered deterministically by the real
    and imaginary parts of ``y``.  The labelling agrees with the base fibre
    used by :func:`~mpmath.curve_monodromy`.

    >>> from mpmath import curve_fibre, mp
    >>> mp.dps = 15
    >>> [mp.nstr(place.y, 6) for place in curve_fibre((0, -1, 0, 1), 2)]
    ['-2.44949', '2.44949']
    """
    prepared, unused_hyperelliptic = _normalise_algebraic_curve_input(
        ctx, curve)
    x = ctx.convert(x)
    if not ctx.isfinite(x):
        raise ValueError("x must be finite")
    sheets = _ordered_plane_curve_sheets(ctx, prepared, x)
    scale = max([ctx.one] + [abs(value) for value in sheets])
    separation = min(
        abs(left - right)
        for index, left in enumerate(sheets)
        for right in sheets[index + 1:])
    if separation <= 100 * ctx.sqrt(ctx.eps) * scale:
        raise ValueError("x must not be a finite branch value")
    return tuple(CurvePlace(x, value) for value in sheets)


@defun
def curve_path(ctx, curve, start, end):
    r"""Return a lifted path between two regular finite places.

    ``start`` and ``end`` are regular finite places, each given as a
    ``(x, y)`` pair or a ``CurvePlace`` from :func:`~mpmath.curve_fibre`.
    A guarded polyline in the x-plane avoids the branch values, and the
    path is lifted by numerical continuation from ``start``.  The returned
    ``CurvePath`` record contains the endpoint places, the sheet index
    reached, and the continuation record carrying the numerical routing
    data used by :func:`~mpmath.curve_integral`.

    Both places must have distinct ``x`` values, and ``end`` must lie on
    the sheet reached by continuation; otherwise ``ValueError`` is raised.

    >>> from mpmath import curve_path, mp
    >>> mp.dps = 15
    >>> curve = {(0, 2): 1, (1, 0): -1}
    >>> path = curve_path(curve, (1, 1), (4, 2))
    >>> mp.nstr(path.start.y, 6), mp.nstr(path.end.y, 6)
    ('1.0', '2.0')
    """
    prepared, unused_hyperelliptic = _normalise_algebraic_curve_input(
        ctx, curve)
    start = _normalise_curve_place(ctx, prepared, start, "start")
    end = _normalise_curve_place(ctx, prepared, end, "end")
    if start.x == end.x:
        raise ValueError(
            "path endpoints must have distinct x values")
    branch_values, unused_resultant = _stage_branch_locus(ctx, prepared)
    path = _guarded_open_path(ctx, start.x, end.x, branch_values)
    lifted = _lift_plane_curve_path(ctx, prepared, path, start.y)
    scale = max(ctx.one, abs(end.x), abs(end.y), abs(lifted.end.y))
    if abs(lifted.end.y - end.y) > 100 * ctx.sqrt(ctx.eps) * scale:
        raise ValueError(
            "the lifted path from start does not reach end; the two "
            "places lie on different sheets along the guarded path")
    return CurvePath(
        start=lifted.start,
        end=lifted.end,
        sheet=lifted.sheet,
        continuation=lifted.continuation)


@defun
def curve_integral(ctx, curve, differentials, path):
    r"""Integrate one differential or a differential basis along a path.

    ``differentials`` is either a single callable ``f(x, y)`` returning the
    coefficient of ``dx``, or a sequence of such callables; ``path`` is a
    ``CurvePath`` from :func:`~mpmath.curve_path`.  A single differential
    gives a scalar ``values`` entry, a sequence gives one entry per form.
    The returned ``CurveIntegral`` record also carries the maximum
    curve-equation residual encountered on the integration nodes and the
    number of path segments.

    >>> from mpmath import curve_integral, curve_path, mp
    >>> mp.dps = 15
    >>> curve = {(0, 2): 1, (1, 0): -1}
    >>> path = curve_path(curve, (1, 1), (4, 2))
    >>> integral = curve_integral(curve, lambda x, y: 1 / y, path)
    >>> mp.nstr(integral.values, 12)
    '2.0'
    """
    prepared, unused_hyperelliptic = _normalise_algebraic_curve_input(
        ctx, curve)
    if callable(differentials):
        single = True
        forms = (differentials,)
    else:
        single = False
        forms = _curve_differential_sequence(
            differentials, "differentials")
    if not isinstance(path, CurvePath):
        raise ValueError("path must be a CurvePath from curve_path")
    integral = _integrate_plane_curve_path(
        ctx, prepared, path.continuation, forms, sheet=path.sheet)
    values = integral.values[0] if single else integral.values
    return CurveIntegral(
        values, integral.max_sheet_residual, integral.segments)


def _normalise_curve_places(ctx, curve, target):
    """Return normalised places from a single place or a divisor."""
    if isinstance(target, CurvePlace):
        return (_normalise_curve_place(ctx, curve, target),)
    try:
        left, right = target
        pair = True
    except (TypeError, ValueError):
        pair = False
    if pair and not any(
            isinstance(value, (list, tuple, CurvePlace))
            for value in (left, right)):
        return (_normalise_curve_place(ctx, curve, target),)
    places = []
    for place in target:
        places.append(_normalise_curve_place(ctx, curve, place))
    return tuple(places)


@defun
def curve_abel_map(ctx, curve, target, differentials=None,
                   base_place=None, reduce=False):
    r"""Evaluate the Abel map of a place or divisor on a plane curve.

    ``target`` is one regular finite place, given as a ``(x, y)`` pair or
    ``CurvePlace``, or a sequence of places representing an effective
    divisor; an empty sequence returns the zero vector.  A hyperelliptic
    coefficient sequence without supplied differentials is dispatched to
    :func:`~mpmath.hyperelliptic_abel_map`; a general plane curve requires
    ``differentials``, one first-kind callable per genus, and returns the
    unnormalized Abelian coordinates they integrate to.

    ``base_place`` selects a regular finite base place; the default is
    sheet zero over the internally selected computational base point.
    With ``reduce=True`` the result is reduced modulo the period lattice
    of the supplied basis, equivalent to applying
    :func:`~mpmath.curve_lattice_reduce`.

    >>> from mpmath import curve_abel_map, mp
    >>> mp.dps = 15
    >>> curve = {(0, 2): 1, (1, 0): 1, (3, 0): -1}
    >>> point = (mp.mpf(2), mp.sqrt(6))
    >>> forms = (lambda x, y: 1 / y,)
    >>> value = curve_abel_map(curve, point, forms, base_place=point)
    >>> mp.nstr(mp.norm(value), 3)
    '0.0'
    """
    prepared, hyperelliptic_coefficients = _normalise_algebraic_curve_input(
        ctx, curve)
    if hyperelliptic_coefficients is not None and differentials is None:
        if base_place is None:
            return ctx.hyperelliptic_abel_map(
                hyperelliptic_coefficients, target, reduce=reduce)
        result = (ctx.hyperelliptic_abel_map(
                      hyperelliptic_coefficients, target)
                  - ctx.hyperelliptic_abel_map(
                      hyperelliptic_coefficients, base_place))
        if reduce:
            tau = curve_periods(ctx, curve).tau
            lattice = _jacobian_lattice_matrix(ctx, tau)
            result = _reduce_jacobian_point(
                ctx, result, tau, lattice ** -1)
        return result

    forms = _curve_differential_sequence(
        differentials, "differentials")
    monodromy = _stage_monodromy(ctx, prepared)
    genus = monodromy.genus
    if len(forms) != genus:
        raise ValueError(
            "differentials must contain one form per genus")
    if base_place is None:
        base = _PlaneCurvePlace(
            monodromy.base_point, monodromy.base_sheets[0])
    else:
        base = _normalise_curve_place(
            ctx, prepared, base_place, "base_place")
    branch_values, unused_resultant = _stage_branch_locus(ctx, prepared)
    quadrature_order = max(12, ctx.dps // 2)

    def place_value(place):
        if _same_numerical_place(
                ctx, (place.x, place.y),
                (monodromy.base_point, monodromy.base_sheets[0])):
            return ctx.zeros(genus, 1)
        return _finite_base_abel_value(
            ctx, prepared, monodromy, branch_values, place, forms,
            quadrature_order)

    places = _normalise_curve_places(ctx, prepared, target)
    result = ctx.zeros(genus, 1)
    for place in places:
        result += place_value(place)
    result -= len(places) * place_value(base)
    if reduce:
        tau = curve_periods(ctx, curve, differentials).tau
        lattice = _jacobian_lattice_matrix(ctx, tau)
        result = _reduce_jacobian_point(ctx, result, tau, lattice ** -1)
    return result


@defun
def curve_lattice_reduce(ctx, value, periods):
    r"""Reduce a Jacobian vector modulo the period lattice.

    ``value`` is a genus-length column vector of Abelian coordinates, and
    ``periods`` is either a ``CurvePeriods`` record or a normalized
    Riemann matrix ``tau``.  The returned ``CurveLatticeReduction`` record
    contains the equivalent reduced vector and the integer lattice shift
    ``(m, n)`` with ``value - (m + tau*n)`` equal to the reduced vector.

    >>> from mpmath import curve_lattice_reduce, curve_riemann_matrix, mp
    >>> mp.dps = 15
    >>> tau = curve_riemann_matrix((0, -1, 0, 1))
    >>> reduced = curve_lattice_reduce(mp.matrix([2 + 1j]), tau)
    >>> reduced.shift
    (2, 1)
    >>> mp.nstr(reduced.value[0, 0], 3)
    '0.0'
    """
    if isinstance(periods, CurvePeriods):
        tau = periods.tau
    else:
        tau = periods
    if tau.rows != tau.cols:
        raise ValueError("periods must be square or a CurvePeriods record")
    genus = tau.rows
    try:
        value = ctx.matrix(value)
    except (TypeError, ValueError):
        raise ValueError("value must be a genus-length column vector")
    if value.rows != genus or value.cols != 1:
        raise ValueError("value must be a genus-length column vector")
    lattice = _jacobian_lattice_matrix(ctx, tau)
    coordinates = lattice ** -1 * ctx.matrix(
        [ctx.re(entry) for entry in value]
        + [ctx.im(entry) for entry in value])
    shift = tuple(int(ctx.nint(entry)) for entry in coordinates)
    reduced = value - ctx.matrix([
        shift[row] + ctx.fsum(
            tau[row, column] * shift[genus + column]
            for column in range(genus))
        for row in range(genus)])
    return CurveLatticeReduction(reduced, shift)


def _real_plane_curve_monodromy(
        ctx, curve, base_point, branch_points, circle_steps=24,
        corridor_height=None, max_refinements=12):
    """Compute monodromy for distinct real finite branch values.

    Infinity is inferred as the inverse of the ordered finite product.  The
    returned continuations retain the numerical evidence for every branch
    permutation.
    """
    branch_points = tuple(sorted(ctx.convert(point)
                                 for point in branch_points))
    if not branch_points:
        raise ValueError("at least one finite branch point is required")
    base_point = ctx.convert(base_point)
    base_sheets = _ordered_plane_curve_sheets(ctx, curve, base_point)
    continuations = []
    for branch_point in branch_points:
        path = _real_branch_loop_path(
            ctx, base_point, branch_point, branch_points,
            circle_steps=circle_steps, corridor_height=corridor_height)
        continuations.append(_continue_plane_curve_sheets_adaptive(
            ctx, curve, path, initial_sheets=base_sheets,
            max_refinements=max_refinements))
    permutations = tuple(result.permutation for result in continuations)
    if any(permutation is None for permutation in permutations):
        raise ValueError("monodromy paths must be closed")

    finite_product = tuple(range(curve.y_degree))
    for permutation in permutations:
        finite_product = _compose_permutations(
            permutation, finite_product)
    infinity_permutation = _inverse_permutation(finite_product)
    all_permutations = permutations + (infinity_permutation,)
    if len(_monodromy_orbit(all_permutations)) != curve.y_degree:
        raise ValueError("the monodromy action is not transitive")
    genus, ramification = _riemann_hurwitz_genus(
        curve.y_degree, all_permutations)
    return _MonodromyData(
        base_point=base_point,
        base_sheets=base_sheets,
        branch_points=branch_points,
        permutations=permutations,
        infinity_permutation=infinity_permutation,
        ramification=ramification,
        genus=genus,
        continuations=tuple(continuations),
    )
