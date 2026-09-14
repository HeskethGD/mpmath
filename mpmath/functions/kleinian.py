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


def _add_indices(genus, *indices):
    """Return the derivative multi-index for repeated coordinates."""
    derivative = [0] * genus
    for index in indices:
        derivative[index] += 1
    return tuple(derivative)


def _theta_log_jet(ctx, v, tau_key, characteristic, degree):
    """Return theta and its logarithmic derivatives through degree three."""
    genus = len(v)
    theta_derivatives = ctx.rtheta_jet(
        v, tau_key, degree, characteristic)
    zero = (0,) * genus
    theta = theta_derivatives[zero]
    if degree and not theta:
        raise ZeroDivisionError(
            "Kleinian logarithmic derivative is singular on the theta divisor")
    if not degree:
        return theta, None, None, None

    gradient = [
        theta_derivatives[_add_indices(genus, i)] / theta
        for i in range(genus)
    ]
    if degree == 1:
        return theta, gradient, None, None

    hessian = [[ctx.zero] * genus for unused in range(genus)]
    for i in range(genus):
        for j in range(genus):
            theta_ij = theta_derivatives[_add_indices(genus, i, j)]
            hessian[i][j] = theta_ij / theta - gradient[i] * gradient[j]
    if degree == 2:
        return theta, gradient, hessian, None

    third = [[[ctx.zero] * genus for unused in range(genus)]
             for unused in range(genus)]
    theta_squared = theta ** 2
    theta_cubed = theta ** 3
    first = [theta_derivatives[_add_indices(genus, i)]
             for i in range(genus)]
    for i in range(genus):
        for j in range(genus):
            theta_ij = theta_derivatives[_add_indices(genus, i, j)]
            for k in range(genus):
                theta_ik = theta_derivatives[_add_indices(genus, i, k)]
                theta_jk = theta_derivatives[_add_indices(genus, j, k)]
                theta_ijk = theta_derivatives[
                    _add_indices(genus, i, j, k)]
                third[i][j][k] = (
                    theta_ijk / theta
                    - (theta_ij * first[k] + theta_ik * first[j]
                       + theta_jk * first[i]) / theta_squared
                    + 2 * first[i] * first[j] * first[k] / theta_cubed
                )
    return theta, gradient, hessian, third


def _kleinian_theta_data(ctx, u, omega, tau, kappa, characteristic, degree):
    """Return normalized data and theta log derivatives in Abelian coordinates."""
    data = _prepare_kleinian_data(
        ctx, u, omega, tau, kappa, characteristic)
    u, inverse_omega, tau_key, kappa, characteristic = data
    v = tuple(ctx.fsum(inverse_omega[i, j] * u[j]
                       for j in range(len(u)))
              for i in range(len(u)))
    theta_data = _theta_log_jet(
        ctx, v, tau_key, characteristic, degree)
    return u, inverse_omega, kappa, theta_data


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
        if len(item) not in (2, 3):
            raise ValueError("each index tuple must have length 2 or 3")
        if any(not isinstance(index, int) or index < 0 or index >= genus
               for index in item):
            raise ValueError("coordinate indices must be between 0 and %i"
                             % (genus - 1))
    return requested, scalar


@defun
def kleinian_sigma(ctx, u, omega, tau, kappa, characteristic=None,
                    constant=1):
    r"""
    Kleinian sigma function, with a supplied normalization constant.

    The input uses unnormalized Abelian coordinates ``u`` and the convention

    .. math::

        \sigma(u) = C \exp(-u^T\varkappa u/2)
        \theta[K](\omega^{-1}u\mid\tau).

    ``omega`` is the first-kind a-period matrix, ``tau`` is the normalized
    Riemann matrix, and ``kappa`` is the symmetric matrix
    :math:`\eta\omega^{-1}`. The characteristic uses the literal
    ``(a, b)`` convention of :func:`~mpmath.rtheta`. The default ``constant``
    is one; its curve-dependent canonical value is not inferred.

    """
    with ctx.extraprec(10):
        data = _kleinian_theta_data(
            ctx, u, omega, tau, kappa, characteristic, 0)
        u, unused_inverse, kappa, theta_data = data
        theta = theta_data[0]
        constant = ctx.convert(constant)
        if not ctx.isfinite(constant):
            raise ValueError("constant must be finite")
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
        u, inverse_omega, kappa, theta_data = data
        gradient = theta_data[1]
        genus = len(u)
        result = ctx.matrix(genus, 1)
        for i in range(genus):
            result[i] = (
                -ctx.fsum(kappa[i, j] * u[j] for j in range(genus))
                + ctx.fsum(inverse_omega[j, i] * gradient[j]
                           for j in range(genus))
            )
    return +result


@defun
def kleinian_p(ctx, u, omega, tau, kappa, indices, characteristic=None):
    r"""
    Evaluate one or several Kleinian P-functions.

    ``indices`` is either one tuple of two or three zero-based coordinate
    indices, or a sequence of such tuples. For example, ``(0, 1)`` requests
    :math:`\wp_{0,1}`, while ``[(0, 0), (0, 0, 1)]`` evaluates two functions
    using a single theta jet. A single index tuple returns a scalar; a
    sequence returns a tuple.

    The definitions are

    .. math::

        \wp_{i,j} = -\partial_{u_i}\partial_{u_j}\log\sigma,
        \qquad
        \wp_{i,j,k} = -\partial_{u_i}\partial_{u_j}
        \partial_{u_k}\log\sigma.

    Period and characteristic conventions are the same as for
    :func:`~mpmath.kleinian_sigma`.

    """
    omega_matrix = _normalise_kleinian_matrix(ctx, omega, "omega")
    requested, scalar = _normalise_p_indices(indices, omega_matrix.rows)
    degree = max(map(len, requested))
    with ctx.extraprec(10):
        data = _kleinian_theta_data(
            ctx, u, omega_matrix, tau, kappa, characteristic, degree)
        unused_u, inverse_omega, kappa, theta_data = data
        hessian, third = theta_data[2], theta_data[3]
        genus = inverse_omega.rows
        results = []
        for item in requested:
            i, j = item[:2]
            if len(item) == 2:
                logarithmic = ctx.fsum(
                    inverse_omega[p, i] * hessian[p][q]
                    * inverse_omega[q, j]
                    for p in range(genus) for q in range(genus))
                results.append(kappa[i, j] - logarithmic)
            else:
                k = item[2]
                logarithmic = ctx.fsum(
                    inverse_omega[p, i] * inverse_omega[q, j]
                    * inverse_omega[r, k] * third[p][q][r]
                    for p in range(genus) for q in range(genus)
                    for r in range(genus))
                results.append(-logarithmic)
    results = tuple(+result for result in results)
    return results[0] if scalar else results
