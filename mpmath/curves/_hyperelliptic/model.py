"""Hyperelliptic model preparation and Baker path geometry."""

import itertools

from ..polynomial import _polynomial_derivative, _polynomial_gcd


def _hyperelliptic_coefficients(ctx, coefficients):
    """Validate polynomial coefficients in ascending order."""
    try:
        coefficients = tuple(ctx.convert(value) for value in coefficients)
    except (TypeError, ValueError):
        raise ValueError("coefficients must be a sequence of numbers")
    if not coefficients or not coefficients[-1]:
        raise ValueError("the leading coefficient must be nonzero")
    if any(not ctx.isfinite(value) for value in coefficients):
        raise ValueError("coefficients must be finite numbers")
    degree = len(coefficients) - 1
    if degree < 3:
        raise ValueError("the polynomial degree must be at least 3")
    return coefficients


def _hyperelliptic_roots(ctx, coefficients):
    """Return distinct roots ordered by real and then imaginary part."""
    try:
        roots = ctx.polyroots(coefficients, maxsteps=200, error=False)
    except ctx.NoConvergence:
        # A repeated factor cannot be resolved by giving Durand-Kerner more
        # iterations. Check it before paying for guarded retry attempts.
        with ctx.extraprec(30):
            numeric_coefficients = tuple(ctx.convert(value)
                                         for value in coefficients)
            common = _polynomial_gcd(
                ctx, numeric_coefficients,
                _polynomial_derivative(ctx, numeric_coefficients))
        if len(common) > 1:
            raise ValueError("the polynomial must have distinct roots")
        for guard_bits in (50, 100):
            try:
                roots = ctx.polyroots(
                    coefficients, maxsteps=200,
                    extraprec=guard_bits, error=False)
                break
            except ctx.NoConvergence:
                continue
        else:
            raise ValueError("failed to resolve hyperelliptic roots")
    scale = max([ctx.one] + [abs(root) for root in roots])
    tolerance = ctx.sqrt(ctx.eps) * scale
    roots = sorted(roots, key=lambda root: (ctx.re(root), ctx.im(root)))
    if any(abs(right - left) <= tolerance
           for left, right in itertools.pairwise(roots)):
        raise ValueError("the polynomial must have distinct roots")
    return tuple(roots), tolerance


def _real_hyperelliptic_data(ctx, coefficients, roots, tolerance):
    """Return real coefficients and roots, rejecting nonreal data."""
    if any(ctx.im(value) for value in coefficients):
        raise ValueError("coefficients must be finite real numbers")
    if any(abs(ctx.im(root)) > tolerance for root in roots):
        raise ValueError("the polynomial must have only real roots")
    return (tuple(ctx.re(value) for value in coefficients),
            tuple(ctx.re(root) for root in roots))


def _prepare_hyperelliptic_curve(ctx, coefficients, method):
    """Normalize a curve and select its real or complex path construction."""
    if method not in ("auto", "real", "complex"):
        raise ValueError("method must be 'auto', 'real' or 'complex'")
    coefficients = _hyperelliptic_coefficients(ctx, coefficients)
    roots, root_tolerance = _hyperelliptic_roots(ctx, coefficients)
    real_data = (
        not any(ctx.im(value) for value in coefficients)
        and not any(abs(ctx.im(root)) > root_tolerance for root in roots)
    )
    use_real_method = method == "real" or (method == "auto" and real_data)
    if use_real_method:
        coefficients, roots = _real_hyperelliptic_data(
            ctx, coefficients, roots, root_tolerance)
    genus = (len(coefficients) - 2) // 2
    even_degree = not (len(coefficients) - 1) % 2
    return (coefficients, roots, root_tolerance, use_real_method,
            genus, even_degree)



def _infinity_direction(ctx, roots):
    """Choose a deterministic root-free ray leaving the terminal root."""
    terminal = roots[-1]
    initial = terminal - roots[-2]
    clearance = 100 * ctx.sqrt(ctx.eps)
    count = 2 * len(roots) + 1
    for step in range(count):
        direction = initial * ctx.exp(ctx.j * ctx.pi * step / count)
        blocked = False
        for root in roots[:-1]:
            ratio = (root - terminal) / direction
            if (ctx.re(ratio) > 0
                    and abs(ctx.im(ratio)) <= clearance * max(
                        ctx.one, abs(ratio))):
                blocked = True
                break
        if not blocked:
            return direction
    raise ValueError("failed to choose a root-free path from infinity")



def _normalise_abel_targets(ctx, target):
    """Normalize one affine point or a sequence of affine points."""
    try:
        items = tuple(target)
    except TypeError:
        raise ValueError(
            "target must be an affine point (x, y) or a sequence of points")
    if not items:
        return ()

    def point(value):
        try:
            entries = tuple(value)
        except TypeError:
            raise ValueError("each target point must be a pair (x, y)")
        if len(entries) != 2:
            raise ValueError("each target point must be a pair (x, y)")
        try:
            x, y = (ctx.convert(entry) for entry in entries)
        except (TypeError, ValueError):
            raise ValueError("target coordinates must be numbers")
        if not ctx.isfinite(x) or not ctx.isfinite(y):
            raise ValueError("target coordinates must be finite")
        return x, y

    if len(items) == 2:
        try:
            x, y = (ctx.convert(item) for item in items)
        except (TypeError, ValueError):
            pass
        else:
            if not ctx.isfinite(x) or not ctx.isfinite(y):
                raise ValueError("target coordinates must be finite")
            return ((x, y),)
    return tuple(point(value) for value in items)


def _evaluate_polynomial(ctx, coefficients, value):
    """Evaluate an ascending coefficient vector by Horner's rule."""
    result = ctx.zero
    for coefficient in reversed(coefficients):
        result = result * value + coefficient
    return result


def _target_branch_index(ctx, x, y, roots, target_eps):
    """Return the index if an affine target is a branch point."""
    scale = max([ctx.one, abs(x)] + [abs(root) for root in roots])
    tolerance = 100 * target_eps * scale
    for index, root in enumerate(roots):
        if abs(x - root) <= tolerance and abs(y) <= tolerance:
            return index
    return None


def _admissible_branch_vertex(ctx, target, roots, target_eps):
    """Choose the nearest branch point with an unobstructed final segment."""
    path_tolerance = 100 * ctx.sqrt(target_eps)
    candidates = []
    for index, branch in enumerate(roots):
        difference = target - branch
        obstructed = False
        for other_index, other in enumerate(roots):
            if other_index == index:
                continue
            ratio = (other - branch) / difference
            if (0 < ctx.re(ratio) < 1
                    and abs(ctx.im(ratio)) <= path_tolerance * max(
                        ctx.one, abs(ratio))):
                obstructed = True
                break
        if not obstructed:
            candidates.append((abs(difference), index))
    if not candidates:
        raise ValueError("failed to find a branch-point path to target")
    return min(candidates)[1]
