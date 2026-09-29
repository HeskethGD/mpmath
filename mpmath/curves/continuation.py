"""Path construction and continuation on plane algebraic curves."""

import itertools
from bisect import bisect_right

from ._context import _curve_cache_state
from ._records import (
    _BoundaryPlace,
    _BranchContinuation,
    _GeometricCover,
    _LiftedPathChain,
    _LiftedPathTerm,
    _LiftedPlaneCurvePath,
    _PlaneCurvePlace,
    _SheetContinuation,
)
from .geometry import _stable_complex_order
from .polynomial import (
    _evaluate_plane_derivative,
    _evaluate_plane_polynomial,
    _minimum_cost_assignment,
    _minimum_root_separation,
    _newton_plane_curve_sheet,
    _ordered_plane_curve_sheets,
    _plane_curve_sheets,
)

# Base-plane paths
# ----------------


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
        clearances = tuple(clearance(candidate) for candidate in candidates)
        best_clearance = max(clearances)
        # Symmetric branch configurations can give two or more equally good
        # exterior points.  Tiny precision-dependent differences must not
        # choose a different radial marking, so retain candidate order for
        # scores indistinguishable at the working precision.
        tie_tolerance = (100 * ctx.sqrt(ctx.eps)
                         * max(ctx.one, best_clearance))
        base_point = next(
            candidate for candidate, value in zip(candidates, clearances)
            if best_clearance - value <= tie_tolerance)
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
    for left, right in itertools.pairwise(path):
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
        (candidate, _unused_residual, _unused_derivative, _unused_scale,
         converged) = _newton_plane_curve_sheet(
             ctx, curve, right, prediction, maxsteps=max_newton_steps)

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
    for left, right in itertools.pairwise(path):
        value = advance(left, right, value, 0)
    return _BranchContinuation(
        values=tuple(accepted_values),
        max_residual=diagnostics["max_residual"],
        min_derivative=diagnostics["min_derivative"],
        steps=len(accepted_path) - 1,
        path=tuple(accepted_path),
        refinements=diagnostics["refinements"],
    )


# Lifted-path composition
# -----------------------


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
            raise TypeError("chain coefficients must be integers")
        if not coefficient:
            continue
        if not isinstance(sheet, int):
            raise TypeError("chain sheets must be integer indices")
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


def _lift_plane_graph(ctx, curve, geometry):
    """Continue every base edge once, retaining its endpoint permutation."""
    fibres = []
    for x in geometry.vertices:
        roots = _plane_curve_sheets(ctx, curve, x)
        tolerance = 100 * ctx.sqrt(ctx.eps) * max(ctx.one, *map(abs, roots))
        if _minimum_root_separation(ctx, roots) <= tolerance:
            raise ctx.NoConvergence("graph vertex has unresolved fibre separation")
        fibres.append(_stable_complex_order(ctx, roots, tolerance))
    continuations, permutations = [], []
    for left, right in geometry.edges:
        lift = _continue_plane_curve_sheets_adaptive(
            ctx, curve, (geometry.vertices[left], geometry.vertices[right]),
            initial_sheets=fibres[left], max_refinements=20)
        permutation = _minimum_cost_assignment(ctx, lift.fibres[-1], fibres[right])
        scale = max(ctx.one, *map(abs, fibres[right]))
        if any(abs(y - fibres[right][j]) > 10000 * ctx.eps * scale
               for y, j in zip(lift.fibres[-1], permutation)):
            raise ctx.NoConvergence("graph edge endpoint fibre did not match")
        continuations.append(lift)
        permutations.append(tuple(permutation))
    return _GeometricCover(geometry, tuple(fibres), tuple(continuations),
                           tuple(permutations))


class _LiftedEdgeSampler:
    """Stage-local sheet queries on one straight, already continued edge.

    Continuation knots are independent of quadrature panels. An ambiguous
    correction refines its bracket, rather than accepting the nearest root
    of an independently solved fibre. Refinement is bounded; the separation
    checks are numerical consistency tests, not a certification theorem.
    """

    def __init__(self, ctx, curve, continuation):
        self.ctx = ctx
        self.curve = curve
        self.state = _curve_cache_state(ctx)
        self.left = continuation.path[0]
        self.delta = continuation.path[-1] - self.left
        if not self.delta:
            raise ValueError("a lifted edge must have distinct endpoints")
        parameters = tuple((x - self.left) / self.delta
                           for x in continuation.path)
        tolerance = 100 * ctx.eps
        if (any(abs(ctx.im(t)) > tolerance for t in parameters)
                or any(ctx.re(a) >= ctx.re(b)
                       for a, b in itertools.pairwise(parameters))):
            raise ValueError("a lifted edge must follow a straight ordered path")
        self.positions = [ctx.re(t) for t in parameters]
        self.fibres = list(continuation.fibres)
        self.refinements = 0

    def sample(self, parameter, sheet):
        ctx = self.ctx
        if _curve_cache_state(ctx) != self.state:
            raise ValueError("lifted edge sampler cannot cross precision contexts")
        if not 0 <= parameter <= 1 or not 0 <= sheet < self.curve.y_degree:
            raise ValueError("invalid edge parameter or sheet")
        x = self.left + parameter * self.delta
        for unused in range(21):
            index = min(len(self.positions) - 2,
                        max(0, bisect_right(self.positions, parameter) - 1))
            a, b = self.positions[index:index + 2]
            fraction = (parameter - a) / (b - a)
            predictions = tuple(
                u + fraction * (v - u)
                for u, v in zip(self.fibres[index], self.fibres[index + 1]))
            candidate, residual, _unused_derivative, _unused_scale, converged = (
                _newton_plane_curve_sheet(ctx, self.curve, x, predictions[sheet]))
            separation = min((abs(predictions[sheet] - y)
                              for j, y in enumerate(predictions) if j != sheet),
                             default=ctx.inf)
            if converged and abs(candidate - predictions[sheet]) < separation / 4:
                return x, candidate, abs(residual)
            midpoint = (a + b) / 2
            midpoint_x = self.left + midpoint * self.delta
            if (midpoint == a or midpoint == b
                    or midpoint_x == self.left + a * self.delta
                    or midpoint_x == self.left + b * self.delta):
                break
            lift = _continue_plane_curve_sheets_adaptive(
                ctx, self.curve,
                tuple(self.left + t * self.delta for t in (a, midpoint, b)),
                initial_sheets=self.fibres[index], max_refinements=12)
            expected = self.fibres[index + 1]
            scale = max(ctx.one, *map(abs, expected))
            if any(abs(u - v) > 10000 * ctx.eps * scale
                   for u, v in zip(lift.fibres[-1], expected)):
                raise ctx.NoConvergence("edge refinement changed the sheet labels")
            positions = [ctx.re((z - self.left) / self.delta) for z in lift.path[1:-1]]
            self.positions[index + 1:index + 1] = positions
            self.fibres[index + 1:index + 1] = lift.fibres[1:-1]
            self.refinements += 1
        raise ctx.NoConvergence("lifted edge query did not resolve its sheet")
