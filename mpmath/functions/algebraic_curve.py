"""Experimental numerical building blocks for plane algebraic curves.

This module is private while the representation and numerical contracts are
validated.  Its separate conceptual sections cover projection, continuation,
lifted paths, monodromy, homology and numerical period assembly.
"""

from collections import namedtuple
from fractions import Fraction
from math import comb


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
_PathIntegrals = namedtuple(
    "_PathIntegrals", "values max_sheet_residual segments")
_PlaneCurvePlace = namedtuple(
    "_PlaneCurvePlace", "x y")
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


# Curve representation and evaluation
# -----------------------------------

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


def _numerical_graph_cycles(ctx, graph, monodromy):
    """Lift every fundamental graph cycle to a numerical path chain."""
    expected = _monodromy_graph_permutations(monodromy)
    if graph.permutations != expected:
        raise ValueError("graph and numerical monodromy systems differ")
    branch_continuations = _monodromy_branch_continuations(ctx, monodromy)
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
