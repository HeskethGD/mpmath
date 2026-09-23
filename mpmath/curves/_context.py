"""Numerical-context helpers shared by the curve package."""


def _curve_cache_state(ctx):
    """Return the numerical state that keys curve computations."""
    rounding = getattr(ctx, "rounding", None)
    trap_complex = getattr(ctx, "trap_complex", None)
    return ctx.prec, rounding, trap_complex
