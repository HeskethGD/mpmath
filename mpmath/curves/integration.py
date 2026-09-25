"""Numerical integration along lifted algebraic-curve paths."""

from ._records import (
    _IteratedPathIntegrals, _PathIntegrals, _PlaneCurvePeriods,
)
from .polynomial import (
    _evaluate_plane_derivative, _evaluate_plane_polynomial,
    _minimum_cost_assignment, _newton_plane_curve_sheet_with_derivatives,
    _plane_curve_sheets, _polynomial_multiply,
)
from .quadrature import _geometric_quadrature_order

# Lifted-path integration
# -----------------------

def _full_plane_curve_segment_samples(
        ctx, curve, left_x, right_x, left_fibre, right_fibre,
        parameters, sheet):
    """Resolve one selected sheet independently at every quadrature node."""
    delta_x = right_x - left_x
    samples = []
    for parameter in parameters:
        x = left_x + parameter * delta_x
        predictions = tuple(
            left + parameter * (right - left)
            for left, right in zip(left_fibre, right_fibre))
        candidates = _plane_curve_sheets(
            ctx, curve, x, roots_init=predictions)
        assignment = _minimum_cost_assignment(ctx, predictions, candidates)
        y = candidates[assignment[sheet]]
        residual = abs(_evaluate_plane_polynomial(ctx, curve, x, y))
        samples.append((x, y, residual))
    return tuple(samples)


def _newton_plane_curve_segment_samples(
        ctx, curve, left_x, right_x, left_fibre, right_fibre,
        parameters, sheet):
    """Continue one sheet through ordered nodes, or return ``None``.

    The known right endpoint certifies the resulting branch.  A caller must
    fall back to independent full-fibre solves when this fast path fails.
    """
    current_x = left_x
    current_y = left_fibre[sheet]
    current_derivative_x = _evaluate_plane_derivative(
        ctx, curve, current_x, current_y, "x")
    current_derivative_y = _evaluate_plane_derivative(
        ctx, curve, current_x, current_y, "y")
    delta_x = right_x - left_x
    samples = []

    def advance(next_x):
        nonlocal current_x, current_y
        nonlocal current_derivative_x, current_derivative_y
        derivative_scale = max(
            ctx.one, abs(current_derivative_x), abs(current_derivative_y))
        if (abs(current_derivative_y)
                <= ctx.sqrt(ctx.eps) * derivative_scale):
            return None
        prediction = current_y - (
            current_derivative_x * (next_x - current_x)
            / current_derivative_y)
        (candidate, residual, candidate_derivative_x,
         candidate_derivative_y, unused_scale,
         converged) = _newton_plane_curve_sheet_with_derivatives(
             ctx, curve, next_x, prediction)
        correction = abs(candidate - prediction)
        motion = abs(candidate - current_y)
        if (not converged
                or correction > max(ctx.one, motion) / 4):
            return None
        current_x = next_x
        current_y = candidate
        current_derivative_x = candidate_derivative_x
        current_derivative_y = candidate_derivative_y
        return candidate, abs(residual)

    for parameter in parameters:
        x = left_x + parameter * delta_x
        result = advance(x)
        if result is None:
            return None
        y, residual = result
        samples.append((x, y, residual))

    endpoint = advance(right_x)
    if endpoint is None:
        return None
    expected = right_fibre[sheet]
    endpoint_scale = max(ctx.one, abs(expected), abs(current_y))
    tolerance = 100 * ctx.sqrt(ctx.eps) * endpoint_scale
    if abs(current_y - expected) > tolerance:
        return None
    if any(abs(current_y - other) < abs(current_y - expected)
           for index, other in enumerate(right_fibre) if index != sheet):
        return None
    return tuple(samples)


def _plane_curve_segment_samples(
        ctx, curve, left_x, right_x, left_fibre, right_fibre,
        parameters, sheet):
    """Sample one lifted segment, using certified Newton continuation."""
    if curve.y_degree > 2:
        samples = _newton_plane_curve_segment_samples(
            ctx, curve, left_x, right_x, left_fibre, right_fibre,
            parameters, sheet)
        if samples is not None:
            return samples
    return _full_plane_curve_segment_samples(
        ctx, curve, left_x, right_x, left_fibre, right_fibre,
        parameters, sheet)


def _integrate_plane_curve_path(
        ctx, curve, continuation, differentials, sheet=0,
        quadrature_order=None, branch_values=None,
        differential_evaluator=None):
    """Integrate coefficients of dx along one continued sheet.

    Each differential is a callable ``differential(x, y)`` returning the
    coefficient of ``dx``.  An internal ``differential_evaluator`` may return
    all coefficients together when their algebraic structure permits shared
    work.  Ordered quadrature nodes use predictor-corrector continuation of
    the selected sheet, certified against each segment's known endpoint
    fibre. Unsafe segments fall back to independent full-fibre solves matched
    against the interpolated endpoint fibres.
    """
    try:
        differentials = tuple(differentials)
    except TypeError:
        raise ValueError("differentials must be a sequence of callables")
    if not differentials or any(not callable(value)
                                for value in differentials):
        raise ValueError("differentials must be a sequence of callables")
    if (differential_evaluator is not None
            and not callable(differential_evaluator)):
        raise ValueError("differential_evaluator must be callable")
    if not isinstance(sheet, int) or not 0 <= sheet < curve.y_degree:
        raise ValueError("sheet must index the initial fibre")
    if len(continuation.path) != len(continuation.fibres):
        raise ValueError("continuation path and fibres are inconsistent")
    if any(len(fibre) != curve.y_degree
           for fibre in continuation.fibres):
        raise ValueError("continuation fibres have the wrong degree")
    geometric = quadrature_order == "geometry"
    if geometric and not branch_values:
        raise ValueError("geometry quadrature requires branch values")
    if quadrature_order is None:
        quadrature_order = max(16, 2 * ctx.dps)
    if not geometric and (not isinstance(quadrature_order, int)
                          or quadrature_order < 2):
        raise ValueError("quadrature_order must be an integer at least 2")
    rules = {}

    def rule(order):
        if order not in rules:
            nodes, weights = ctx.gauss_quadrature(order, "legendre")
            rules[order] = (
                tuple((nodes[index] + 1) / 2 for index in range(order)),
                tuple(weights[index] / 2 for index in range(order)))
        return rules[order]

    def integrate_segment(
            left_x, right_x, left_fibre, right_fibre, order):
        parameters, weights = rule(order)
        contributions = [[] for unused in differentials]
        samples = _plane_curve_segment_samples(
            ctx, curve, left_x, right_x, left_fibre, right_fibre,
            parameters, sheet)
        residual = ctx.zero
        for (x, y, sample_residual), weight in zip(samples, weights):
            residual = max(residual, sample_residual)
            sample_values = (tuple(differential(x, y)
                                   for differential in differentials)
                             if differential_evaluator is None else
                             tuple(differential_evaluator(x, y)))
            if len(sample_values) != len(differentials):
                raise ValueError(
                    "differential_evaluator returned the wrong number "
                    "of values")
            for index, value in enumerate(sample_values):
                contributions[index].append(
                    weight * value)
        delta_x = right_x - left_x
        return (tuple(delta_x * ctx.fsum(terms)
                      for terms in contributions), residual)

    values = [ctx.zero] * len(differentials)
    max_sheet_residual = ctx.zero
    for segment in range(len(continuation.path) - 1):
        left_x = continuation.path[segment]
        right_x = continuation.path[segment + 1]
        if left_x == right_x:
            continue
        left_fibre = continuation.fibres[segment]
        right_fibre = continuation.fibres[segment + 1]
        if geometric:
            order = _geometric_quadrature_order(
                ctx, left_x, right_x, branch_values)
        else:
            order = quadrature_order
        segment_values, residual = integrate_segment(
            left_x, right_x, left_fibre, right_fibre, order)
        max_sheet_residual = max(max_sheet_residual, residual)
        for index, value in enumerate(segment_values):
            values[index] += value

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
        lifted_samples = _plane_curve_segment_samples(
            ctx, curve, left_x, right_x, left_fibre, right_fibre,
            parameters, sheet)
        samples = []
        for x, y, residual in lifted_samples:
            max_sheet_residual = max(max_sheet_residual, residual)
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


def _concatenate_iterated_path_integrals(ctx, left, right):
    """Compose level-two path integrals using Chen concatenation.

    See K.-T. Chen, *Iterated path integrals*, Bull. Amer. Math. Soc. 83
    (1977), 831--879, doi:10.1090/S0002-9904-1977-14320-6.  Our matrix
    entry ``[outer][inner]`` integrates ``inner`` before ``outer``.
    """
    count = len(left.values)
    if (len(right.values) != count
            or len(left.iterated) != count
            or len(right.iterated) != count
            or any(len(row) != count
                   for row in left.iterated + right.iterated)):
        raise ValueError("iterated path integrals have incompatible sizes")
    values = tuple(
        left.values[index] + right.values[index]
        for index in range(count))
    iterated = tuple(tuple(
        left.iterated[outer][inner]
        + right.iterated[outer][inner]
        + right.values[outer] * left.values[inner]
        for inner in range(count)) for outer in range(count))
    return _IteratedPathIntegrals(
        values=values,
        iterated=iterated,
        max_sheet_residual=max(
            left.max_sheet_residual, right.max_sheet_residual),
        segments=left.segments + right.segments,
    )


def _reverse_iterated_path_integrals(ctx, integral):
    """Reverse level-two path integrals by Chen's reversal identity."""
    count = len(integral.values)
    if (len(integral.iterated) != count
            or any(len(row) != count for row in integral.iterated)):
        raise ValueError("iterated path integral has an incompatible size")
    return _IteratedPathIntegrals(
        values=tuple(-value for value in integral.values),
        iterated=tuple(tuple(
            integral.iterated[inner][outer]
            for inner in range(count)) for outer in range(count)),
        max_sheet_residual=integral.max_sheet_residual,
        segments=integral.segments,
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
        ctx, curve, chain, differentials, quadrature_order=None,
        integral_cache=None, branch_values=None,
        differential_evaluator=None):
    """Integrate supplied differentials termwise over a lifted-path chain.

    ``integral_cache`` may be a stage-local dictionary shared by chains that
    reuse the same continuation objects. It stores only completed path
    integrals; term coefficients and diagnostic segment counts are still
    applied for every algebraic use of a path.
    """
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
        cache_key = id(term.continuation), term.sheet
        cached = (None if integral_cache is None
                  else integral_cache.get(cache_key))
        if cached is not None and cached[0] is term.continuation:
            integral = cached[1]
        else:
            integral = _integrate_plane_curve_path(
                ctx, curve, term.continuation, differentials,
                sheet=term.sheet, quadrature_order=quadrature_order,
                branch_values=branch_values,
                differential_evaluator=differential_evaluator)
            if integral_cache is not None:
                # Retaining the continuation both guards against object-ID
                # reuse and documents that identity, rather than structural
                # hashing of its large path and fibre tuples, defines reuse.
                integral_cache[cache_key] = term.continuation, integral
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
    integral_cache = {}
    for chain in canonical_chains[:2 * genus]:
        integral = _integrate_lifted_path_chain(
            ctx, curve, chain, differentials,
            quadrature_order=quadrature_order,
            integral_cache=integral_cache)
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
