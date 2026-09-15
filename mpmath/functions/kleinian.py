import math
from itertools import product

from .functions import ctx_lru_cache, defun
from .riemann_theta import (
    _as_vector, _matrix_tuple, _normalise_characteristic, _normalise_tau,
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
    """Cache the inverse first-kind period matrix and validate tau."""
    omega = ctx.matrix(omega_key)
    try:
        inverse_omega = ctx.inverse(omega)
    except (ValueError, ZeroDivisionError):
        raise ValueError("omega must be invertible")
    # This validates positive definiteness and shares rtheta's cached data.
    ctx._rtheta_tau_data(tau_key)
    return _matrix_tuple(inverse_omega)


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
    inverse_omega = ctx.matrix(
        ctx._kleinian_period_data(omega_key, tau_key))
    return u, inverse_omega, tau_key, kappa, characteristic


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


def _theta_u_derivative(ctx, theta_jet, inverse_omega, indices, cache):
    """Return a theta derivative transformed to Abelian coordinates."""
    indices = tuple(sorted(indices))
    if indices in cache:
        return cache[indices]
    genus = inverse_omega.rows
    terms = []
    for theta_indices in product(range(genus), repeat=len(indices)):
        derivative = [0] * genus
        factor = ctx.one
        for theta_index, u_index in zip(theta_indices, indices):
            derivative[theta_index] += 1
            factor *= inverse_omega[theta_index, u_index]
        terms.append(factor * theta_jet[tuple(derivative)])
    value = ctx.fsum(terms)
    cache[indices] = value
    return value


def _theta_log_derivative(ctx, theta_jet, inverse_omega, indices,
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
                ctx, theta_jet, inverse_omega,
                tuple(indices[position] for position in block),
                derivative_cache)
            for block in partition)
        terms.append(coefficient * numerator / theta ** blocks)
    return ctx.fsum(terms)


def _kleinian_theta_data(ctx, u, omega, tau, kappa, characteristic, degree):
    """Return normalized data and theta log derivatives in Abelian coordinates."""
    data = _prepare_kleinian_data(
        ctx, u, omega, tau, kappa, characteristic)
    u, inverse_omega, tau_key, kappa, characteristic = data
    v = tuple(ctx.fsum(inverse_omega[i, j] * u[j]
                       for j in range(len(u)))
              for i in range(len(u)))
    theta_jet = ctx.rtheta_jet(v, tau_key, degree, characteristic)
    theta = theta_jet[(0,) * len(v)]
    if degree and not theta:
        raise ZeroDivisionError(
            "Kleinian logarithmic derivative is singular on the theta divisor")
    return u, inverse_omega, tau_key, kappa, characteristic, theta_jet


@defun
@ctx_lru_cache(maxsize=16)
def _hyperelliptic_sigma_normalization(ctx, inverse_omega_key, tau_key,
                                       characteristic):
    """Normalize sigma by its leading Schur--Weierstrass polynomial."""
    inverse_omega = ctx.matrix(inverse_omega_key)
    genus = inverse_omega.rows
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
            factor *= inverse_omega[index, coordinate] ** order
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

        \sigma(u) = C \exp(-u^T\varkappa u/2)
        \theta[K](\omega^{-1}u\mid\tau).

    ``omega`` is the first-kind a-period matrix, ``tau`` is the normalized
    Riemann matrix, and ``kappa`` is the symmetric matrix
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
        u, inverse_omega, tau_key, kappa, characteristic, theta_data = data
        theta = theta_data[(0,) * len(u)]
        if normalization == "theta":
            constant = ctx.one
        elif normalization == "hyperelliptic":
            constant = ctx._hyperelliptic_sigma_normalization(
                _matrix_tuple(inverse_omega), tau_key,
                characteristic)
        else:
            raise ValueError(
                "normalization must be 'theta' or 'hyperelliptic'")
        quadratic = ctx.fsum(
            u[i] * kappa[i, j] * u[j]
            for i in range(len(u)) for j in range(len(u)))
        result = constant * ctx.exp(-quadratic / 2) * theta
    return +result


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
            ctx, u, omega, tau, kappa, characteristic, 1)
        u, inverse_omega, unused_tau, kappa, unused_char, theta_jet = data
        genus = len(u)
        derivative_cache = {(): theta_jet[(0,) * genus]}
        partition_cache = {}
        result = ctx.matrix(genus, 1)
        for i in range(genus):
            result[i] = (
                -ctx.fsum(kappa[i, j] * u[j] for j in range(genus))
                + _theta_log_derivative(
                    ctx, theta_jet, inverse_omega, (i,),
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
            ctx, u, omega_matrix, tau, kappa, characteristic, degree)
        (unused_u, inverse_omega, unused_tau, kappa, unused_char,
         theta_jet) = data
        genus = inverse_omega.rows
        derivative_cache = {(): theta_jet[(0,) * genus]}
        partition_cache = {}
        results = []
        for item in requested:
            logarithmic = _theta_log_derivative(
                ctx, theta_jet, inverse_omega, item,
                derivative_cache, partition_cache)
            if len(item) == 2:
                results.append(kappa[item[0], item[1]] - logarithmic)
            else:
                results.append(-logarithmic)
    results = tuple(+result for result in results)
    return results[0] if scalar else results
