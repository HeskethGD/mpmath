"""Quadrature policies for numerical algebraic-curve integration."""


_REUSABLE_GAUSS_ORDERS = (8, 12, 16, 24, 32, 48, 64, 96, 128)
# Geometry-selected panels can use finer buckets without changing the
# faster-growing refinement ladder used to check opaque differentials.
_GEOMETRIC_GAUSS_ORDERS = (
    8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64,
    80, 96, 112, 128,
)


def _geometric_quadrature_order(ctx, left, right, singularities):
    """Estimate a Gauss order from known singularities off an interval.

    Each singularity is mapped to the standard interval ``[-1, 1]``.  The
    Bernstein ellipse through the nearest mapped point supplies the expected
    ``rho**(-2*n)`` convergence factor.  This is an order-selection heuristic,
    not a rigorous error bound; see Trefethen, *Is Gauss Quadrature Better than
    Clenshaw--Curtis?*, SIAM Review 50 (2008), for the analytic-integrand
    convergence principle. Known branch values need not describe singularities
    of a caller-supplied differential.
    """
    singularities = tuple(singularities)
    if left == right:
        raise ValueError("integration segment must have distinct endpoints")
    if not singularities:
        raise ValueError("at least one known singularity is required")
    midpoint = (left + right) / 2
    half_width = (right - left) / 2
    ellipse = min(
        (abs((singularity - midpoint) / half_width - 1)
         + abs((singularity - midpoint) / half_width + 1)) / 2
        for singularity in singularities)
    rho = ellipse + ctx.sqrt(ellipse * ellipse - 1)
    if rho <= 1:
        raise ValueError("integration segment meets a known singularity")
    estimate = int(ctx.ceil(
        (ctx.dps + 5) * ctx.log(10) / (2 * ctx.log(rho))))
    for order in _GEOMETRIC_GAUSS_ORDERS:
        if estimate <= order:
            return order
    return 32 * ((estimate + 31) // 32)


def _geometric_edge_panels(ctx, left, right, singularities, max_order=64,
                           max_panels=1024):
    """Plan geometric Gauss panels with bounded work, not bounded depth.

    A nearby branch value can require deep but highly localized refinement.
    Limit the total leaf panels instead; reject subdivisions whose parameter
    or mapped midpoint can no longer be distinguished at working precision.
    The order estimate and its accuracy margin remain unchanged.
    """
    if max_order < 8:
        raise ValueError("maximum geometric order must be at least eight")
    if not isinstance(max_panels, int) or max_panels < 1:
        raise ValueError("maximum geometric panel count must be a positive integer")
    singularities = tuple(singularities)
    pending = [(ctx.zero, ctx.one)]
    panels = []
    delta = right - left
    while pending:
        lower, upper = pending.pop()
        start, end = left + lower * delta, left + upper * delta
        if start == end:
            raise ctx.NoConvergence("geometric panel endpoints are unresolved")
        order = _geometric_quadrature_order(
            ctx, start, end, singularities)
        if order <= max_order:
            panels.append((lower, upper, order))
        else:
            if len(panels) + len(pending) + 2 > max_panels:
                raise ctx.NoConvergence(
                    "geometric quadrature exceeded its panel budget")
            midpoint = (lower + upper) / 2
            mapped_midpoint = left + midpoint * delta
            if midpoint in (lower, upper) or mapped_midpoint in (start, end):
                raise ctx.NoConvergence("geometric panel midpoint is unresolved")
            pending.extend(((midpoint, upper), (lower, midpoint)))
    return tuple(panels)


def _legendre_edge_rule(ctx, order):
    """Compute a Gauss rule on [0, 1] by symmetric Legendre root iteration.

    Callers cache rules within their numerical stage, never across precision
    contexts. Unlike ``GaussLegendre.calc_nodes``, this accepts the arbitrary
    orders chosen by the geometry policy, without rounding up the node count.
    ``ctx.gauss_quadrature`` supports those orders via a tridiagonal
    eigensolver; this recurrence retains the direct root iteration used for
    the automatic-basis path.
    """
    if not isinstance(order, int) or order < 2:
        raise ValueError("Gauss order must be an integer at least two")

    def pair(x):
        previous, current = ctx.one, x
        for degree in range(2, order + 1):
            previous, current = current, (
                (2 * degree - 1) * x * current - (degree - 1) * previous) / degree
        return current, previous

    nodes, weights = [None] * order, [None] * order
    for index in range(1, order // 2 + 1):
        root = ctx.cos(ctx.pi * (index - ctx.mpf(1) / 4) / (order + ctx.mpf(1) / 2))
        for unused in range(30):
            value, previous = pair(root)
            derivative = order * (root * value - previous) / (root**2 - 1)
            step = value / derivative
            root -= step
            if abs(step) <= 4 * ctx.eps:
                break
        else:
            raise ctx.NoConvergence("Legendre node did not converge")
        value, previous = pair(root)
        derivative = order * (root * value - previous) / (root**2 - 1)
        weight = 1 / ((1 - root**2) * derivative**2)
        nodes[index - 1], nodes[order - index] = (1 - root) / 2, (1 + root) / 2
        weights[index - 1] = weights[order - index] = weight
    if order % 2:
        unused_value, previous = pair(ctx.zero)
        nodes[order // 2] = ctx.mpf(1) / 2
        weights[order // 2] = 1 / (order * previous)**2
    return tuple(zip(nodes, weights))
