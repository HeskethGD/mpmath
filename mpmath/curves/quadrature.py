"""Quadrature policies for numerical algebraic-curve integration."""


_REUSABLE_GAUSS_ORDERS = (8, 12, 16, 24, 32, 48, 64, 96, 128)


def _geometric_quadrature_order(ctx, left, right, singularities):
    """Estimate a Gauss order from known singularities off an interval.

    Each singularity is mapped to the standard interval ``[-1, 1]``.  The
    Bernstein ellipse through the nearest mapped point supplies the expected
    ``rho**(-2*n)`` convergence factor.  This is an order-selection heuristic,
    not a rigorous error bound.
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
    for order in _REUSABLE_GAUSS_ORDERS:
        if estimate <= order:
            return order
    return 32 * ((estimate + 31) // 32)
