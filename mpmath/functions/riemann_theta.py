from .functions import ctx_lru_cache, defun


def _matrix_tuple(A):
    """Return a matrix as an immutable tuple of row tuples."""
    return tuple(tuple(A[i, j] for j in range(A.cols))
                 for i in range(A.rows))


def _as_vector(ctx, value, name, length=None):
    """Convert a public vector argument to a tuple of scalars."""
    try:
        vector = ctx.matrix(value)
    except (TypeError, ValueError):
        raise ValueError("%s must be a vector" % name)
    if vector.cols != 1:
        raise ValueError("%s must be a vector" % name)
    if length is not None and vector.rows != length:
        raise ValueError("%s must have length %i" % (name, length))
    result = tuple(ctx.convert(vector[i]) for i in range(vector.rows))
    if not all(ctx.isfinite(x) for x in result):
        raise ValueError("%s entries must be finite" % name)
    return result


def _normalise_tau(ctx, tau):
    """Validate and symmetrise a Riemann matrix."""
    try:
        tau = ctx.matrix(tau)
    except (TypeError, ValueError):
        raise ValueError("tau must be a square matrix")
    if not tau.rows or tau.rows != tau.cols:
        raise ValueError("tau must be a nonempty square matrix")
    tau = tau.copy()
    for i in range(tau.rows):
        for j in range(tau.cols):
            tau[i, j] = ctx.convert(tau[i, j])
            if not ctx.isfinite(tau[i, j]):
                raise ValueError("tau entries must be finite")
    for i in range(tau.rows):
        for j in range(i):
            if not ctx.almosteq(tau[i, j], tau[j, i]):
                raise ValueError("tau must be symmetric")
            entry = (tau[i, j] + tau[j, i]) / 2
            tau[i, j] = tau[j, i] = entry
    return tau


def _normalise_characteristic(ctx, characteristic, genus):
    """Validate literal characteristic vectors."""
    if characteristic is None:
        zero = ctx.zero
        return (zero,) * genus, (zero,) * genus
    if not isinstance(characteristic, (list, tuple)) or len(characteristic) != 2:
        raise ValueError("characteristic must be a pair (a, b)")
    a = _as_vector(ctx, characteristic[0], "characteristic a", genus)
    b = _as_vector(ctx, characteristic[1], "characteristic b", genus)
    if any(ctx.im(x) for x in a + b):
        raise ValueError("characteristic entries must be real")
    return tuple(ctx.re(x) for x in a), tuple(ctx.re(x) for x in b)


def _normalise_derivative(derivative, genus):
    """Convert a derivative request to a multi-index."""
    if isinstance(derivative, int):
        if derivative < 0 or genus != 1 and derivative != 0:
            raise ValueError("integer derivative orders require genus 1")
        return (derivative,) if genus == 1 else (0,) * genus
    try:
        derivative = tuple(derivative)
    except TypeError:
        raise ValueError("derivative must be a nonnegative multi-index")
    if len(derivative) != genus:
        raise ValueError("derivative must have length %i" % genus)
    if any(not isinstance(d, int) or d < 0 for d in derivative):
        raise ValueError("derivative entries must be nonnegative integers")
    return derivative


@defun
@ctx_lru_cache(maxsize=16)
def _rtheta_tau_data(ctx, tau_key):
    """Cache immutable Cholesky data depending only on tau."""
    genus = len(tau_key)
    X = ctx.matrix(genus)
    Y = ctx.matrix(genus)
    for i in range(genus):
        for j in range(genus):
            X[i, j] = ctx.re(tau_key[i][j])
            Y[i, j] = ctx.im(tau_key[i][j])
    try:
        # mpmath returns Y = L L^T; the enumeration uses T = L^T.
        T = ctx.cholesky(Y).T
    except (ValueError, ZeroDivisionError):
        raise ValueError("imaginary part of tau must be positive definite")
    invT = ctx.inverse(T)
    invT_frob = ctx.sqrt(ctx.fsum(invT[i, j] ** 2
                                  for i in range(genus)
                                  for j in range(genus)))
    # ||sqrt(pi) T n|| >= sqrt(pi)/||T^-1||_F for nonzero integer n.
    rho = ctx.sqrt(ctx.pi) / invT_frob
    value_radius = _truncation_radius(
        ctx, genus, (0,), rho, invT_frob, ctx.zero)
    return (_matrix_tuple(X), _matrix_tuple(Y), _matrix_tuple(T),
            invT_frob, rho, value_radius)


def _upper_gamma_half_integer(ctx, twice_s, x):
    """Evaluate upper gamma for a positive integer or half-integer s."""
    exponential = ctx.exp(-x)
    if twice_s & 1:
        s = ctx.mpf('0.5')
        value = ctx.sqrt(ctx.pi) * ctx.erfc(ctx.sqrt(x))
    else:
        s = ctx.one
        value = exponential
    target = ctx.mpf(twice_s) / 2
    while s < target:
        value = s * value + ctx.power(x, s) * exponential
        s += 1
    return value


def _tail_bound(ctx, genus, degree, radius, rho, invT_frob, shift_norm):
    """Bound the omitted differentiated Gaussian lattice tail."""
    x = (radius - rho / 2) ** 2
    scale = genus * ctx.power(2 / rho, genus) / 2
    transform = invT_frob / ctx.sqrt(ctx.pi)
    tail = ctx.zero
    for k in range(degree + 1):
        coefficient = (ctx.binomial(degree, k)
                       * shift_norm ** (degree - k) * transform ** k)
        tail += coefficient * _upper_gamma_half_integer(
            ctx, genus + k, x)
    return (2 * ctx.pi) ** degree * scale * tail


def _truncation_radius(ctx, genus, degrees, rho, invT_frob, shift_norm):
    """Choose a radius giving working-precision absolute tail error."""
    degree = max(degrees)
    threshold = (ctx.sqrt(genus + 2 * degree
                          + ctx.sqrt(genus ** 2 + 8 * degree)) + rho) / 2
    target = ctx.eps / 8
    high = max(ctx.one, threshold)
    while max(_tail_bound(ctx, genus, d, high, rho,
                          invT_frob, shift_norm) for d in degrees) > target:
        high = 1 + 5 * high / 4
    low = threshold
    # Only the conservative upper endpoint is returned. Twelve bisections
    # locate the discrete ellipsoid boundary adequately without evaluating
    # a high-precision transcendental tail dozens of unnecessary times.
    for unused in range(12):
        middle = (low + high) / 2
        error = max(_tail_bound(ctx, genus, d, middle, rho,
                                invT_frob, shift_norm) for d in degrees)
        if error > target:
            low = middle
        else:
            high = middle
    return high


def _ellipsoid_points(ctx, T, center, radius):
    """Yield integer points in ||T (n-center)|| <= radius."""
    genus = len(center)
    point = [0] * genus
    displacement = [ctx.zero] * genus

    def enumerate_coordinate(j, remaining):
        offset = ctx.fsum(T[j][k] * displacement[k]
                          for k in range(j + 1, genus))
        reach = ctx.sqrt(max(ctx.zero, remaining)) / abs(T[j][j])
        midpoint = center[j] - offset / T[j][j]
        lower = int(ctx.ceil(midpoint - reach))
        upper = int(ctx.floor(midpoint + reach))
        for value in range(lower, upper + 1):
            point[j] = value
            displacement[j] = value - center[j]
            row = T[j][j] * displacement[j] + offset
            new_remaining = remaining - row ** 2
            # The interval bounds imply this is nonnegative; retain the check
            # only for a possible last-bit rounding overshoot.
            if new_remaining < 0:  # pragma: no cover
                continue
            if j:
                yield from enumerate_coordinate(j - 1, new_remaining)
            else:
                yield tuple(point)

    yield from enumerate_coordinate(genus - 1, radius ** 2)


def _rtheta_sum(ctx, z, tau_key, a, b, derivatives):
    """Evaluate one or more derivatives in one lattice traversal."""
    genus = len(z)
    X_key, Y_key, T, invT_frob, rho, value_radius = (
        ctx._rtheta_tau_data(tau_key))
    Y = ctx.matrix(Y_key)
    y = ctx.matrix([ctx.im(value) for value in z])
    shift = ctx.lu_solve(Y, y)
    shift_tuple = tuple(shift[i] for i in range(genus))
    shift_norm = ctx.sqrt(ctx.fsum(value ** 2 for value in shift_tuple))
    degrees = tuple(sum(d) for d in derivatives)
    if all(degree == 0 for degree in degrees):
        radius = value_radius
    else:
        radius = _truncation_radius(
            ctx, genus, degrees, rho, invT_frob, shift_norm)
    center = tuple(-a[i] - shift_tuple[i] for i in range(genus))
    growth = ctx.exp(ctx.pi * ctx.fsum(y[i] * shift[i]
                                      for i in range(genus)))
    x_plus_b = tuple(ctx.re(z[i]) + b[i] for i in range(genus))

    sums = [[] for unused in derivatives]
    for n in _ellipsoid_points(ctx, T, center, radius / ctx.sqrt(ctx.pi)):
        u = tuple(n[i] + a[i] for i in range(genus))
        shifted = tuple(u[i] + shift_tuple[i] for i in range(genus))
        magnitude_form = ctx.fsum(
            shifted[i] * Y_key[i][j] * shifted[j]
            for i in range(genus) for j in range(genus))
        phase = (ctx.pi * ctx.fsum(u[i] * X_key[i][j] * u[j]
                                  for i in range(genus) for j in range(genus))
                 + 2 * ctx.pi * ctx.fsum(u[i] * x_plus_b[i]
                                         for i in range(genus)))
        term = ctx.exp(-ctx.pi * magnitude_form + ctx.j * phase)
        for index, derivative in enumerate(derivatives):
            factor = ctx.one
            for i, order in enumerate(derivative):
                if order:
                    factor *= (2 * ctx.pi * ctx.j * u[i]) ** order
            sums[index].append(factor * term)
    return tuple(growth * ctx.fsum(terms) for terms in sums)


@defun
def rtheta(ctx, z, tau, characteristic=None, derivative=0):
    r"""
    Evaluates the Riemann theta function of the vector :math:`z` and Riemann
    matrix :math:`\tau`. In genus :math:`g`, the convention is

    .. math ::

        \theta(z\mid\tau) = \sum_{n\in\mathbb Z^g}
        \exp(\pi i n^T\tau n + 2\pi i n^Tz).

    The argument :math:`z` must be a length-:math:`g` sequence or an mpmath
    column matrix. The period matrix :math:`\tau` must be a square nested
    sequence or mpmath matrix. It must be symmetric and its imaginary part
    must be positive definite. Entries opposite the diagonal that differ
    only by the current ``almosteq`` tolerance are averaged; materially
    asymmetric matrices raise ``ValueError``.

    **Characteristics**

    Passing ``characteristic=(a, b)`` evaluates

    .. math ::

        \theta\!\begin{bmatrix}a\\b\end{bmatrix}(z\mid\tau)
        = \sum_{n\in\mathbb Z^g}
        \exp\!\left(\pi i(n+a)^T\tau(n+a)
        +2\pi i(n+a)^T(z+b)\right).

    Here :math:`a` and :math:`b` are literal length-:math:`g` real vectors,
    rather than packed characteristic bits. Arbitrary real characteristics
    are accepted; half-integer characteristics are the most common.

    **Derivatives**

    ``derivative=(d0, ..., d(g-1))`` requests the multi-index derivative

    .. math ::

        \frac{\partial^{d_0+\cdots+d_{g-1}}\theta}
        {\partial z_0^{d_0}\cdots\partial z_{g-1}^{d_{g-1}}}.

    The entries must be nonnegative integers. In genus one, an integer
    derivative order can be supplied directly for consistency with
    ``jtheta``. Derivatives are with respect to the components of :math:`z`,
    not the angular argument used by ``jtheta``.

    **Examples**

    Evaluate a genus-two theta value::

        >>> from mpmath import mp, rtheta
        >>> mp.dps = 15
        >>> mp.pretty = True
        >>> tau = [[1.0j, 0.05j], [0.05j, 1.2j]]
        >>> rtheta([0.1, 0.2], tau)
        (1.08592896533944 + 0.0j)

    A half-characteristic and a mixed derivative use literal vectors and a
    derivative multi-index::

        >>> characteristic = ([mp.mpf('0.5'), 0], [0, mp.mpf('0.5')])
        >>> rtheta([0.1, 0.2], tau, characteristic, derivative=(1, 1))
        (-0.289794493099238 + 0.0j)

    In genus one, set :math:`q=\exp(\pi i\tau)` and write :math:`z=w/\pi`.
    The zero characteristic then agrees with the third Jacobi theta
    function::

        >>> from mpmath import almosteq, expjpi, jtheta, pi
        >>> tau1 = mp.mpc('0.2', '0.9')
        >>> w = mp.mpc('0.3', '0.1')
        >>> almosteq(rtheta([w/pi], [[tau1]]),
        ...          jtheta(3, w, expjpi(tau1)))
        True

    The other standard Jacobi functions correspond to characteristics
    ``([1/2], [1/2])``, ``([1/2], [0])`` and ``([0], [1/2])`` for
    ``jtheta(1)``, ``jtheta(2)`` and ``jtheta(4)`` respectively. With this
    convention, the ``jtheta(1)`` identity has an additional minus sign.

    **Numerical method and limitations**

    The defining series is truncated using an incomplete-gamma tail bound
    and the required lattice points are enumerated in a Cholesky ellipsoid.
    The exponentially growing factor caused by :math:`\operatorname{Im}(z)`
    is separated from the damped sum. Data depending only on :math:`\tau` is
    cached with the active context precision included in the cache key.

    This direct-summation implementation is intended for modest genus and
    precision. Siegel reduction is not yet performed, so poorly reduced
    matrices can be substantially slower than equivalent reduced matrices.
    Near a zero of theta, cancellation can prevent full relative accuracy;
    the truncation controls absolute error in the scaled oscillatory sum.

    **References**

    1. [DLMF]_ Chapter 21, Multidimensional Theta Functions.
    2. B. Deconinck, M. Heil, A. Bobenko, M. van Hoeij and M. Schmies,
       *Computing Riemann Theta Functions*, Mathematics of Computation 73
       (2004), 1417-1442.

    """
    tau = _normalise_tau(ctx, tau)
    genus = tau.rows
    z = _as_vector(ctx, z, "z", genus)
    a, b = _normalise_characteristic(ctx, characteristic, genus)
    derivative = _normalise_derivative(derivative, genus)
    tau_key = _matrix_tuple(tau)

    degree = sum(derivative)
    input_magnitude = max([ctx.mag(value) for value in z + a + b]
                          + [0])
    extra = 10 * (degree + 1) + 2 * max(0, input_magnitude)
    with ctx.extraprec(extra):
        result = _rtheta_sum(ctx, z, tau_key, a, b, (derivative,))[0]
    return +result
