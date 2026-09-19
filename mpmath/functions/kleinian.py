import math
from itertools import product

from .functions import ctx_lru_cache, defun
from .riemann_theta import (
    _as_vector, _matrix_tuple, _multiindices, _normalise_characteristic,
    _normalise_tau,
)


def _normalise_kleinian_matrix(ctx, value, name, genus=None, symmetric=False):
    """Validate a square matrix used in a Kleinian function."""
    try:
        matrix = ctx.matrix(value)
    except (TypeError, ValueError):
        raise ValueError("%s must be a square matrix" % name)
    if not matrix.rows or matrix.rows != matrix.cols:
        raise ValueError("%s must be a nonempty square matrix" % name)
    if genus is not None and matrix.rows != genus:
        raise ValueError("%s must be a %i by %i matrix" % (
            name, genus, genus))
    matrix = matrix.copy()
    for i in range(matrix.rows):
        for j in range(matrix.cols):
            matrix[i, j] = ctx.convert(matrix[i, j])
            if not ctx.isfinite(matrix[i, j]):
                raise ValueError("%s entries must be finite" % name)
    if symmetric:
        for i in range(matrix.rows):
            for j in range(i):
                if not ctx.almosteq(matrix[i, j], matrix[j, i]):
                    raise ValueError("%s must be symmetric" % name)
                entry = (matrix[i, j] + matrix[j, i]) / 2
                matrix[i, j] = matrix[j, i] = entry
    return matrix


@defun
@ctx_lru_cache(maxsize=16)
def _kleinian_period_data(ctx, omega_key, tau_key):
    """Cache the inverse complete a-period matrix and validate tau."""
    omega = ctx.matrix(omega_key)
    try:
        inverse_period = ctx.inverse(2 * omega)
    except (ValueError, ZeroDivisionError):
        raise ValueError("omega must be invertible")
    # This validates positive definiteness and shares rtheta's cached data.
    ctx._rtheta_tau_data(tau_key)
    return _matrix_tuple(inverse_period)


def _prepare_kleinian_data(ctx, u, omega, tau, kappa, characteristic):
    """Normalize Kleinian inputs and return reusable numerical data."""
    omega = _normalise_kleinian_matrix(ctx, omega, "omega")
    genus = omega.rows
    u = _as_vector(ctx, u, "u", genus)
    tau = _normalise_tau(ctx, tau)
    if tau.rows != genus:
        raise ValueError("tau must be a %i by %i matrix" % (genus, genus))
    kappa = _normalise_kleinian_matrix(
        ctx, kappa, "kappa", genus, symmetric=True)
    characteristic = _normalise_characteristic(
        ctx, characteristic, genus)
    omega_key = _matrix_tuple(omega)
    tau_key = _matrix_tuple(tau)
    inverse_period = ctx.matrix(
        ctx._kleinian_period_data(omega_key, tau_key))
    return u, inverse_period, tau_key, kappa, characteristic


# The multivariate Faà di Bruno formula expresses derivatives of log(theta)
# as a sum over set partitions. This single construction handles every order
# and replaces separate, increasingly complicated formulas for each tensor.
def _set_partitions(size):
    """Yield each set partition of range(size) once."""
    if not size:
        yield ()
        return
    new = size - 1
    for partition in _set_partitions(new):
        yield partition + ((new,),)
        for index, block in enumerate(partition):
            yield (partition[:index] + (block + (new,),)
                   + partition[index + 1:])


def _theta_u_derivative(ctx, theta_jet, inverse_period, indices, cache):
    """Return a theta derivative transformed to Abelian coordinates."""
    indices = tuple(sorted(indices))
    if indices in cache:
        return cache[indices]
    genus = inverse_period.rows
    terms = []
    for theta_indices in product(range(genus), repeat=len(indices)):
        derivative = [0] * genus
        factor = ctx.one
        for theta_index, u_index in zip(theta_indices, indices):
            derivative[theta_index] += 1
            factor *= inverse_period[theta_index, u_index]
        terms.append(factor * theta_jet[tuple(derivative)])
    value = ctx.fsum(terms)
    cache[indices] = value
    return value


def _theta_log_derivative(ctx, theta_jet, inverse_period, indices,
                          derivative_cache, partition_cache):
    """Return an arbitrary logarithmic theta derivative in u coordinates."""
    indices = tuple(sorted(indices))
    order = len(indices)
    if order not in partition_cache:
        partition_cache[order] = tuple(_set_partitions(order))
    theta = derivative_cache[()]
    terms = []
    for partition in partition_cache[order]:
        blocks = len(partition)
        coefficient = (-1) ** (blocks - 1) * math.factorial(blocks - 1)
        numerator = ctx.fprod(
            _theta_u_derivative(
                ctx, theta_jet, inverse_period,
                tuple(indices[position] for position in block),
                derivative_cache)
            for block in partition)
        terms.append(coefficient * numerator / theta ** blocks)
    return ctx.fsum(terms)


def _kleinian_theta_data(ctx, u, omega, tau, kappa, characteristic, degree,
                         logarithmic=False):
    """Return normalized data and theta derivatives in Abelian coordinates."""
    data = _prepare_kleinian_data(
        ctx, u, omega, tau, kappa, characteristic)
    u, inverse_period, tau_key, kappa, characteristic = data
    v = tuple(ctx.fsum(inverse_period[i, j] * u[j]
                       for j in range(len(u)))
              for i in range(len(u)))
    theta_jet = ctx.rtheta_jet(v, tau_key, degree, characteristic)
    theta = theta_jet[(0,) * len(v)]
    if logarithmic and not theta:
        raise ZeroDivisionError(
            "Kleinian logarithmic derivative is singular on the theta divisor")
    return u, inverse_period, tau_key, kappa, characteristic, theta_jet


@defun
@ctx_lru_cache(maxsize=16)
def _hyperelliptic_sigma_normalization(ctx, inverse_period_key, tau_key,
                                       characteristic):
    """Normalize sigma by its leading Schur--Weierstrass polynomial."""
    inverse_period = ctx.matrix(inverse_period_key)
    genus = inverse_period.rows
    degree = (genus + 1) // 2
    coordinate = degree - 1
    theta_jet = ctx.rtheta_jet(
        (ctx.zero,) * genus, tau_key, degree, characteristic)
    directional_derivative = ctx.zero
    degree_factorial = math.factorial(degree)
    for derivative, value in theta_jet.items():
        if sum(derivative) != degree:
            continue
        multinomial = degree_factorial
        factor = ctx.one
        for index, order in enumerate(derivative):
            multinomial //= math.factorial(order)
            factor *= inverse_period[index, coordinate] ** order
        directional_derivative += multinomial * factor * value
    a, b = characteristic
    half_integer = all(ctx.isint(2 * value) for value in a + b)
    characteristic_parity = int(ctx.nint(
        4 * ctx.fsum(left * right for left, right in zip(a, b)))) & 1
    incompatible_parity = (
        half_integer and characteristic_parity != (degree & 1))
    if incompatible_parity or not directional_derivative:
        raise ValueError(
            "characteristic is incompatible with hyperelliptic "
            "sigma normalization")
    # In zero-based notation, the Hankel determinant delta(u)=det(u[i+j]) has
    # u[degree-1]**degree coefficient equal to the reversing permutation sign.
    sign = -1 if (degree * (degree - 1) // 2) & 1 else 1
    return sign * degree_factorial / directional_derivative


def _sigma_normalization_constant(ctx, normalization, inverse_period,
                                  tau_key, characteristic):
    """Return the requested multiplicative normalization of sigma."""
    if normalization == "theta":
        return ctx.one
    if normalization == "hyperelliptic":
        return ctx._hyperelliptic_sigma_normalization(
            _matrix_tuple(inverse_period), tau_key, characteristic)
    raise ValueError("normalization must be 'theta' or 'hyperelliptic'")


def _sigma_sharp_index(genus):
    """Return Onishi's sigma-sharp derivative as a zero-based multi-index."""
    return tuple(index & 1 for index in range(genus))


def _kleinian_sigma_sharp(ctx, u, omega, tau, kappa, characteristic):
    """Evaluate the canonical sigma derivative on the one-point stratum."""
    index = _sigma_sharp_index(len(u))
    order = sum(index)
    jet = ctx.kleinian_sigma_jet(
        u, omega, tau, kappa, order, characteristic,
        normalization="hyperelliptic")
    return jet[index]


def _quadratic_exponential_jet(ctx, u, kappa, multiindices):
    """Return derivatives of exp(u.T*kappa*u/2) for given multi-indices."""
    genus = len(u)
    zero = (0,) * genus
    quadratic = ctx.fsum(
        u[i] * kappa[i, j] * u[j]
        for i in range(genus) for j in range(genus))
    gradient = tuple(ctx.fsum(kappa[i, j] * u[j]
                              for j in range(genus))
                     for i in range(genus))
    result = {zero: ctx.exp(quadratic / 2)}
    for index in multiindices[1:]:
        coordinate = next(i for i, value in enumerate(index) if value)
        previous = list(index)
        previous[coordinate] -= 1
        previous = tuple(previous)
        terms = [gradient[coordinate] * result[previous]]
        for j, multiplicity in enumerate(previous):
            if multiplicity:
                lower = list(previous)
                lower[j] -= 1
                terms.append(
                    multiplicity * kappa[coordinate, j]
                    * result[tuple(lower)])
        result[index] = ctx.fsum(terms)
    return result


def _multiindex_subindices(index):
    """Yield componentwise subindices and their binomial coefficients."""
    for subindex in product(*(range(value + 1) for value in index)):
        coefficient = math.prod(
            math.comb(value, subvalue)
            for value, subvalue in zip(index, subindex))
        yield subindex, coefficient


def _normalise_p_indices(indices, genus):
    """Normalize one or several Kleinian P-function index tuples."""
    if isinstance(indices, (list, tuple)) and indices and all(
            isinstance(index, int) for index in indices):
        requested = (tuple(indices),)
        scalar = True
    else:
        try:
            requested = tuple(tuple(item) for item in indices)
        except (TypeError, ValueError):
            raise ValueError("indices must contain coordinate index tuples")
        scalar = False
    if not requested:
        raise ValueError("at least one index tuple is required")
    for item in requested:
        if len(item) < 2:
            raise ValueError("each index tuple must have length at least 2")
        if any(not isinstance(index, int) or index < 0 or index >= genus
               for index in item):
            raise ValueError("coordinate indices must be between 0 and %i"
                             % (genus - 1))
    return requested, scalar


@defun
def kleinian_sigma(ctx, u, omega, tau, kappa, characteristic=None,
                    normalization="theta"):
    r"""
    Kleinian sigma function.

    The input uses unnormalized Abelian coordinates ``u`` and the convention

    .. math::

        \sigma(u) = C \exp(u^T\varkappa u/2)
        \theta[K]((2\omega)^{-1}u\mid\tau).

    ``omega`` is the first-kind a-half-period matrix, ``tau`` is the
    normalized Riemann matrix, and ``kappa`` is the symmetric matrix
    :math:`\eta\omega^{-1}`. The characteristic uses the literal
    ``(a, b)`` convention of :func:`~mpmath.rtheta`.

    The default ``normalization="theta"`` uses :math:`C=1`, which is defined
    for arbitrary coherent period data. ``normalization="hyperelliptic"``
    chooses :math:`C` so that the leading term at the origin is the
    Schur--Weierstrass polynomial

    .. math::

        \delta(u)=\det\big(u_{i+j-1}\big)_{i,j=1}^{
        \lfloor(g+1)/2\rfloor}.

    This mode assumes the differential ordering and Riemann characteristic
    returned by :func:`~mpmath.hyperelliptic_data`. In genus one it
    agrees with the conventional Weierstrass sigma function. See
    [BEL1997]_, Definition 1.

    """
    with ctx.extraprec(10):
        data = _kleinian_theta_data(
            ctx, u, omega, tau, kappa, characteristic, 0)
        u, inverse_period, tau_key, kappa, characteristic, theta_data = data
        theta = theta_data[(0,) * len(u)]
        constant = _sigma_normalization_constant(
            ctx, normalization, inverse_period, tau_key, characteristic)
        quadratic = ctx.fsum(
            u[i] * kappa[i, j] * u[j]
            for i in range(len(u)) for j in range(len(u)))
        result = constant * ctx.exp(quadratic / 2) * theta
    return +result


@defun
def kleinian_baker_akhiezer(ctx, u, abel, second_kind, omega, tau, kappa,
                             characteristic=None):
    r"""
    Evaluate the normalized odd-degree Kleinian Baker--Akhiezer function.

    Given compatible first- and second-kind integral vectors
    :math:`A(P)` and :math:`R(P)`, this evaluates

    .. math::

        \Psi(P,u)=
        \frac{\sigma(A(P)-u)}
             {\sigma_\sharp(A(P))\sigma(u)}
        \exp\!\left(-R(P)^T u\right).

    Here :math:`\sigma_\sharp` is the first nonzero sigma derivative on the
    one-point theta stratum: :math:`\sigma` in genus one and, in the present
    coordinate order, the derivative with multi-index ``(0, 1, 0, 1, ...)``
    in higher genus. This is Onishi's ``sigma_sharp`` normalization; in
    genus two it is the conventional :math:`\sigma_2(A(P))` factor of the
    normalized Baker function. It fixes the formerly arbitrary factor
    depending on :math:`P` and gives the standard leading local behavior at
    the unique point at infinity. See [Onishi2005]_, Definition 6.1 and
    Proposition 6.6, and [BEH2005]_, equation (3.3).

    ``abel`` and ``second_kind`` should normally be obtained together from
    :func:`~mpmath.hyperelliptic_abel_map` with ``second_kind=True``. The
    remaining inputs use the same conventions as
    :func:`~mpmath.kleinian_sigma`. Sigma is evaluated with its canonical
    ``normalization="hyperelliptic"``; this is required because its
    multiplicative constant does not cancel from the normalized expression.
    The denominator is singular when ``u`` lies on the sigma divisor.

    The sign in the exponential converts the positive second-kind periods of
    Christiansen, Eilbeck, Enolskii and Kostov to the classical BEL convention
    used here, :math:`2\eta=-\oint_a dr`. See [CEEK2000]_, equations (3.21)
    and (3.22).

    For an odd-degree hyperelliptic curve,
    :func:`~mpmath.hyperelliptic_abel_map` uses the unique point at infinity
    required by this standard spectral interpretation. Its even-degree base
    point is instead finite; constructing a BA function with an essential
    singularity at either of the two even-degree points at infinity requires
    an additional marked-infinity convention.

    """
    with ctx.extraprec(10):
        omega_matrix = _normalise_kleinian_matrix(ctx, omega, "omega")
        genus = omega_matrix.rows
        u = _as_vector(ctx, u, "u", genus)
        abel = _as_vector(ctx, abel, "abel", genus)
        second_kind = _as_vector(
            ctx, second_kind, "second_kind", genus)
        shifted = tuple(abel[index] - u[index]
                        for index in range(genus))
        numerator = ctx.kleinian_sigma(
            shifted, omega_matrix, tau, kappa, characteristic,
            normalization="hyperelliptic")
        denominator = ctx.kleinian_sigma(
            u, omega_matrix, tau, kappa, characteristic,
            normalization="hyperelliptic")
        spectral_normalization = _kleinian_sigma_sharp(
            ctx, abel, omega_matrix, tau, kappa, characteristic)
        exponent = -ctx.fsum(
            second_kind[index] * u[index] for index in range(genus))
        result = (numerator * ctx.exp(exponent)
                  / (spectral_normalization * denominator))
    return +result


@defun
def kleinian_sigma_jet(ctx, u, omega, tau, kappa, order,
                       characteristic=None, normalization="theta"):
    r"""
    Evaluate a derivative jet of the Kleinian sigma function.

    A jet is the function value together with all ordinary mixed partial
    derivatives through the requested total order at the same point. The
    result is a dictionary keyed by derivative-count tuples. For example, in
    genus two an order-two jet contains

    .. math::

        \begin{aligned}
        (0,0)&:\ \sigma, &
        (1,0)&:\ \partial_{u_0}\sigma, &
        (0,1)&:\ \partial_{u_1}\sigma,\\
        (2,0)&:\ \partial_{u_0}^2\sigma, &
        (1,1)&:\ \partial_{u_0}\partial_{u_1}\sigma, &
        (0,2)&:\ \partial_{u_1}^2\sigma.
        \end{aligned}

    ``order`` must be a nonnegative integer. The keys use the graded reverse
    lexicographic ordering of :func:`~mpmath.rtheta_jet`; the values are not
    divided by multi-index factorials. All theta derivatives are accumulated
    in one theta jet. Unlike logarithmic derivatives such as the Kleinian
    zeta and P-functions, a sigma jet remains defined on the theta divisor.

    Period, characteristic, and normalization conventions are the same as
    for :func:`~mpmath.kleinian_sigma`.

    **Example**

    Evaluate the sigma value, gradient, and Hessian together::

        >>> from mpmath import kleinian_sigma_jet
        >>> omega = [[1, 0], [0, 1]]
        >>> tau = [[1j, 0], [0, 1.2j]]
        >>> kappa = [[0.2, 0.1], [0.1, 0.3]]
        >>> jet = kleinian_sigma_jet([0.1, 0.2], omega, tau, kappa, 2)
        >>> list(jet)
        [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]

    """
    if not isinstance(order, int) or order < 0:
        raise ValueError("order must be a nonnegative integer")
    with ctx.extraprec(10):
        data = _kleinian_theta_data(
            ctx, u, omega, tau, kappa, characteristic, order)
        u, inverse_period, tau_key, kappa, characteristic, theta_jet = data
        genus = len(u)
        indices = tuple(_multiindices(genus, order))
        constant = _sigma_normalization_constant(
            ctx, normalization, inverse_period, tau_key, characteristic)

        theta_derivatives = {}
        derivative_cache = {(): theta_jet[(0,) * genus]}
        for index in indices:
            repeated = tuple(
                coordinate
                for coordinate, multiplicity in enumerate(index)
                for unused in range(multiplicity))
            theta_derivatives[index] = _theta_u_derivative(
                ctx, theta_jet, inverse_period, repeated, derivative_cache)

        exponential_derivatives = _quadratic_exponential_jet(
            ctx, u, kappa, indices)
        result = {}
        for index in indices:
            terms = []
            for subindex, coefficient in _multiindex_subindices(index):
                complement = tuple(
                    value - subvalue
                    for value, subvalue in zip(index, subindex))
                terms.append(
                    coefficient * exponential_derivatives[subindex]
                    * theta_derivatives[complement])
            result[index] = constant * ctx.fsum(terms)
    return {index: +value for index, value in result.items()}


@defun
def kleinian_zeta(ctx, u, omega, tau, kappa, characteristic=None):
    r"""
    Kleinian zeta vector in unnormalized Abelian coordinates.

    This evaluates :math:`\nabla_u\log\sigma(u)`. The result is an mpmath
    column matrix. Period and characteristic conventions are the same as for
    :func:`~mpmath.kleinian_sigma`.

    """
    with ctx.extraprec(10):
        data = _kleinian_theta_data(
            ctx, u, omega, tau, kappa, characteristic, 1,
            logarithmic=True)
        u, inverse_period, unused_tau, kappa, unused_char, theta_jet = data
        genus = len(u)
        derivative_cache = {(): theta_jet[(0,) * genus]}
        partition_cache = {}
        result = ctx.matrix(genus, 1)
        for i in range(genus):
            result[i] = (
                ctx.fsum(kappa[i, j] * u[j] for j in range(genus))
                + _theta_log_derivative(
                    ctx, theta_jet, inverse_period, (i,),
                    derivative_cache, partition_cache)
            )
    return +result


@defun
def kleinian_p(ctx, u, omega, tau, kappa, indices, characteristic=None):
    r"""
    Evaluate one or several Kleinian P-functions.

    ``indices`` is either one tuple containing at least two zero-based
    coordinate indices, or a sequence of such tuples. For example, ``(0, 1)``
    requests :math:`\wp_{0,1}`, while
    ``[(0, 0), (0, 0, 1), (0, 0, 1, 1)]`` evaluates functions of orders two,
    three and four using a single theta jet. A single index tuple returns a
    scalar; a sequence returns a tuple.

    The definitions are

    .. math::

        \wp_{i_1,\ldots,i_n}
        =-\partial_{u_{i_1}}\cdots\partial_{u_{i_n}}\log\sigma,
        \qquad n\geq 2.

    Arbitrary orders are supported. Their cost grows rapidly with the highest
    requested order.

    Period and characteristic conventions are the same as for
    :func:`~mpmath.kleinian_sigma`.

    """
    omega_matrix = _normalise_kleinian_matrix(ctx, omega, "omega")
    requested, scalar = _normalise_p_indices(indices, omega_matrix.rows)
    degree = max(map(len, requested))
    with ctx.extraprec(10):
        data = _kleinian_theta_data(
            ctx, u, omega_matrix, tau, kappa, characteristic, degree,
            logarithmic=True)
        (unused_u, inverse_period, unused_tau, kappa, unused_char,
         theta_jet) = data
        genus = inverse_period.rows
        derivative_cache = {(): theta_jet[(0,) * genus]}
        partition_cache = {}
        results = []
        for item in requested:
            logarithmic = _theta_log_derivative(
                ctx, theta_jet, inverse_period, item,
                derivative_cache, partition_cache)
            if len(item) == 2:
                results.append(-kappa[item[0], item[1]] - logarithmic)
            else:
                results.append(-logarithmic)
    results = tuple(+result for result in results)
    return results[0] if scalar else results
