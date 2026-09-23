"""Numerical integration along lifted algebraic-curve paths."""

from ._records import (
    _IteratedPathIntegrals, _PathIntegrals, _PlaneCurvePeriods,
)
from .polynomial import (
    _evaluate_plane_derivative, _evaluate_plane_polynomial,
    _minimum_cost_assignment, _plane_curve_sheets, _polynomial_multiply,
)

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
