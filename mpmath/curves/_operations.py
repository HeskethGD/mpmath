"""Internal orchestration for :class:`~mpmath.curves.AlgebraicCurve`."""

from ..functions.hyperelliptic import _normalise_abel_targets
from ._context import _curve_cache_state
from ._records import (
    CurveBranchLocus, CurveCheck, CurveGenus, CurveHomology, CurveIntegral,
    CurveLatticeReduction, CurveMonodromy, CurvePath, CurvePeriods, CurvePlace,
    CurveRiemannConstant, CurveValidation, _PlaneCurvePlace,
)
from ._stages import (
    _curve_differential_sequence, _stage_branch_locus,
    _stage_canonical_polygon, _stage_cycle_integrals, _stage_monodromy,
    _stage_monodromy_graph, _stage_riemann_constant,
    _tau_imaginary_eigenvalues,
)
from .charts import _validated_chart_coordinate_map
from .continuation import (
    _lift_plane_curve_path, _same_numerical_place,
)
from .integration import (
    _integrate_plane_curve_branch, _integrate_plane_curve_path,
    _pullback_plane_curve_differentials,
)
from .jacobian import (
    _finite_base_abel_value, _guarded_open_path, _jacobian_characteristic,
    _normalise_algebraic_curve_input, _normalise_curve_endpoint,
    _period_matrix_from_columns,
)
from .monodromy import (
    _compose_permutations, _integer_matrix_rank, _monodromy_orbit,
)
from .polynomial import _ordered_plane_curve_sheets

# Curve operations
# ----------------


def branch_locus(ctx, curve):
    r"""Return the finite branch locus of a plane algebraic curve.

    ``curve`` may be an ascending coefficient sequence defining
    ``y**2 = P(x)``, a sparse mapping from ``(x_power, y_power)`` pairs to
    a coefficient, or a sequence of ``(x_power, y_power, coefficient)``
    terms.

    The returned ``CurveBranchLocus`` record contains the degree of the
    ``x`` projection, the distinct finite branch values above which the
    projection ramifies, and the ascending coefficients of the
    y-derivative resultant whose roots they are.  Ramification above
    infinity is reported by :attr:`AlgebraicCurve.monodromy` instead,
    because it requires monodromy rather than the resultant alone.

    The lemniscatic curve :math:`y^2 = x^3 - x` has a two-sheeted
    projection with three finite branch values::

        >>> from mpmath import algebraic_curve
        >>> locus = algebraic_curve((0, -1, 0, 1)).branch_locus
        >>> locus.degree
        2
        >>> locus.branch_values
        (mpf('-1.0'), mpf('0.0'), mpf('1.0'))
    """
    prepared, unused_hyperelliptic = _normalise_algebraic_curve_input(
        ctx, curve)
    branch_values, resultant = _stage_branch_locus(ctx, prepared)
    return CurveBranchLocus(prepared.y_degree, branch_values, resultant)


def monodromy(ctx, curve):
    r"""Return the monodromy of a plane algebraic curve over the x-line.

    The curve is continued numerically along guarded radial loops around
    the finite branch values, from an exterior base point chosen
    automatically.  A large outer loop supplies the monodromy at infinity
    geometrically.  The returned ``CurveMonodromy`` record contains the
    computational base point, its ordered fibre, the branch values, the
    counter-clockwise product-ordered local permutations, the permutation
    at infinity, the total ramification, the genus from
    Riemann--Hurwitz, the transitivity and product identities of the
    permutation system, and the minimum geometric clearance of the
    continuation paths.

    The sheet labels refer to the internally selected base fibre.  The
    routing continuations used to compute them are private.

    Each finite branch value of the lemniscatic curve
    :math:`y^2 = x^3 - x` exchanges its two sheets, as does infinity::

        >>> from mpmath import algebraic_curve
        >>> monodromy = algebraic_curve((0, -1, 0, 1)).monodromy
        >>> monodromy.genus
        1
        >>> monodromy.permutations
        ((1, 0), (1, 0), (1, 0))
        >>> monodromy.infinity_permutation
        (1, 0)
    """
    prepared, unused_hyperelliptic = _normalise_algebraic_curve_input(
        ctx, curve)
    monodromy = _stage_monodromy(ctx, prepared)
    degree = prepared.y_degree
    generators = tuple(monodromy.product_generators)
    permutations = tuple(
        generator.permutation for generator in generators[:-1])
    infinity_permutation = generators[-1].permutation
    all_permutations = permutations + (infinity_permutation,)
    product = tuple(range(degree))
    for permutation in all_permutations:
        product = _compose_permutations(permutation, product)
    transitive = len(_monodromy_orbit(all_permutations)) == degree
    return CurveMonodromy(
        base_point=monodromy.base_point,
        base_sheets=monodromy.base_sheets,
        branch_values=monodromy.branch_points,
        permutations=permutations,
        infinity_permutation=infinity_permutation,
        ramification=monodromy.ramification,
        genus=monodromy.genus,
        transitive=transitive,
        product_identity=product == tuple(range(degree)),
        minimum_clearance=monodromy.minimum_clearance)


def genus_data(ctx, curve):
    r"""Return the genus of a plane algebraic curve by monodromy.

    The genus is obtained from the Riemann--Hurwitz formula applied to
    the monodromy of the ``x`` projection, including the permutation at
    infinity.  The returned ``CurveGenus`` record also records the
    projection degree and the total ramification, so the Riemann--Hurwitz
    balance :math:`2g-2 = -2d+r` can be checked directly::

        >>> from mpmath import algebraic_curve
        >>> algebraic_curve((0, -1, 0, 1)).genus_data
        CurveGenus(genus=1, degree=2, ramification=4)
    """
    prepared, unused_hyperelliptic = _normalise_algebraic_curve_input(
        ctx, curve)
    monodromy = _stage_monodromy(ctx, prepared)
    return CurveGenus(
        monodromy.genus, prepared.y_degree, monodromy.ramification)


def homology(ctx, curve):
    r"""Return a canonical homology basis of a plane algebraic curve.

    The lifted monodromy graph of the ``x`` projection is reduced to a
    primitive symplectic homology basis.  The returned
    ``CurveHomology`` record contains the genus, the number of independent
    graph cycles, the number of boundary components of the lifted ribbon
    graph (the places above infinity), the rank of the graph intersection
    form, the dimension of its radical, the resulting canonical
    intersection form, and the integer transformation realizing the
    canonical basis from the graph cycles.

    The lifted graph of the lemniscatic curve :math:`y^2 = x^3 - x` has
    three independent cycles and two boundary components::

        >>> from mpmath import algebraic_curve
        >>> homology = algebraic_curve((0, -1, 0, 1)).homology
        >>> homology.genus, homology.boundary_components
        (1, 2)
        >>> homology.intersection_form
        ((0, 1, 0), (-1, 0, 0), (0, 0, 0))
    """
    prepared, unused_hyperelliptic = _normalise_algebraic_curve_input(
        ctx, curve)
    graph, reduction = _stage_monodromy_graph(ctx, prepared)
    polygon = _stage_canonical_polygon(ctx, prepared)
    return CurveHomology(
        genus=reduction.genus,
        cycle_count=len(graph.cycles),
        boundary_components=graph.boundary_components,
        intersection_rank=graph.intersection_rank,
        radical_rank=reduction.radical_rank,
        intersection_form=polygon.intersection_form,
        transformation=polygon.transformation)


def periods(ctx, curve, differentials=None, *,
                  second_differentials=None):
    r"""Return the period matrices of a plane algebraic curve.

    ``curve`` uses the input forms accepted by
    :attr:`AlgebraicCurve.branch_locus`.  A hyperelliptic coefficient
    sequence without supplied differentials is dispatched to the
    specialized engine :func:`~mpmath.hyperelliptic_periods`.

    A general plane curve requires ``differentials``, a sequence of one
    holomorphic differential callable ``f(x, y)`` per genus, supplying the
    coefficient of ``dx``.  Optional ``second_differentials`` supply the
    same number of second-kind forms; they are integrated on the same
    cycles, with the classical convention ``2*eta = -integral_a(dr)``.

    The returned ``CurvePeriods`` record contains the differential basis,
    the half-period matrices ``omega`` and ``omega_prime``, the normalized
    Riemann matrix ``tau = omega**-1 * omega_prime``, the optional
    second-kind half-period matrices ``eta`` and ``eta_prime`` and
    ``kappa = eta * omega**-1``, and numerical quality residuals.  A
    non-positive-definite normalized period matrix raises ``ValueError``,
    because it always indicates an invalid differential count or basis.

    The normalized Riemann matrix of the lemniscatic curve
    :math:`y^2 = x^3 - x` is :math:`i`::

        >>> from mpmath import algebraic_curve, mp
        >>> mp.dps = 15
        >>> curve = algebraic_curve((0, -1, 0, 1))
        >>> data = curve.periods()
        >>> mp.re(data.tau[0, 0]), mp.im(data.tau[0, 0])
        (mpf('0.0'), mpf('1.0'))

    A general plane curve needs a supplied holomorphic basis, given as
    callables returning the coefficient of ``dx``::

        >>> curve = algebraic_curve({(0, 2): 1, (1, 0): 1, (3, 0): -1})
        >>> data = curve.periods((lambda x, y: 1 / y,))
        >>> curve.validate(data).passed
        True
    """
    prepared, hyperelliptic_coefficients = _normalise_algebraic_curve_input(
        ctx, curve)
    if hyperelliptic_coefficients is not None and differentials is None:
        if second_differentials is not None:
            raise ValueError(
                "supplied second-kind differentials require a supplied "
                "first-kind basis")
        omega, omega_prime, eta, eta_prime, tau, kappa = (
            ctx.hyperelliptic_periods(
                hyperelliptic_coefficients, second_kind=True))
        genus = omega.rows
        return CurvePeriods(
            genus, None, omega, omega_prime, tau, eta, eta_prime, kappa,
            ctx.norm(tau - tau.T), ctx.norm(kappa - kappa.T),
            _tau_imaginary_eigenvalues(ctx, tau), None)

    first_kind = _curve_differential_sequence(
        differentials, "differentials")
    if second_differentials is None:
        second_kind = ()
    else:
        second_kind = _curve_differential_sequence(
            second_differentials, "second_differentials")
    monodromy = _stage_monodromy(ctx, prepared)
    genus = monodromy.genus
    if len(first_kind) != genus:
        raise ValueError(
            "differentials must contain one form per genus")
    if second_kind and len(second_kind) != genus:
        raise ValueError(
            "second_differentials must contain one form per genus")
    forms = first_kind + second_kind
    quadrature_order = max(12, ctx.dps // 2)
    columns, max_sheet_residual = _stage_cycle_integrals(
        ctx, (prepared, forms, quadrature_order))

    periods = _period_matrix_from_columns(
        ctx, columns, 0, genus, genus)
    omega = periods[:, :genus] / 2
    omega_prime = periods[:, genus:] / 2
    raw_tau = omega**-1 * omega_prime
    symmetry_residual = ctx.norm(raw_tau - raw_tau.T)
    tau = (raw_tau + raw_tau.T) / 2
    imaginary_eigenvalues = _tau_imaginary_eigenvalues(ctx, tau)
    if min(imaginary_eigenvalues) <= 0:
        raise ValueError("normalized period matrix is not positive definite")

    eta = eta_prime = kappa = None
    kappa_symmetry_residual = None
    if second_kind:
        second_periods = _period_matrix_from_columns(
            ctx, columns, genus, genus, genus)
        eta = -second_periods[:, :genus] / 2
        eta_prime = -second_periods[:, genus:] / 2
        raw_kappa = eta * omega**-1
        kappa_symmetry_residual = ctx.norm(raw_kappa - raw_kappa.T)
        kappa = (raw_kappa + raw_kappa.T) / 2
    return CurvePeriods(
        genus, first_kind, omega, omega_prime, tau, eta, eta_prime, kappa,
        symmetry_residual, kappa_symmetry_residual, imaginary_eigenvalues,
        max_sheet_residual)


def riemann_matrix(ctx, curve, differentials=None):
    r"""Return the normalized Riemann matrix of a plane algebraic curve.

    This is a convenience wrapper returning
    ``curve.periods(differentials).tau``; see
    :meth:`AlgebraicCurve.periods` for the input conventions.

        >>> from mpmath import algebraic_curve, mp
        >>> mp.dps = 15
        >>> tau = algebraic_curve((0, -1, 0, 1)).riemann_matrix()
        >>> mp.im(tau[0, 0])
        mpf('1.0')
    """
    return periods(ctx, curve, differentials).tau


def riemann_constant(ctx, curve, differentials=None, *,
                           base_place=None):
    r"""Return the vector of Riemann constants for a plane curve.

    The returned ``CurveRiemannConstant`` contains a direct representative of
    the normalized Jacobian vector ``value`` in mpmath's additive convention
    ``theta(A(D) + value, tau) = 0``, its literal ``(a, b)`` coordinates
    ``value = tau*a + b`` modulo the period lattice, the requested
    ``base_place`` (``None`` denotes the engine's natural base), and the
    maximum sheet residual of the direct contour integrations.

    For a general plane curve, ``differentials`` supplies one holomorphic
    differential per genus, in exactly the basis accepted by
    :meth:`AlgebraicCurve.periods`.  The value is computed directly from the
    certified canonical polygon and level-two contour integrals; theta
    functions and characteristic searches are not used.  ``base_place`` may
    be a regular finite place or a chart-backed place.  Changing the base
    uses ``K_Q = K_P + (g-1) A_P(Q)`` in normalized coordinates.

    Hyperelliptic coefficient input without supplied differentials dispatches
    to the established hyperelliptic periods and characteristic convention.

    In genus one the answer is the odd half-period ``(1+tau)/2``::

        >>> from mpmath import algebraic_curve, mp
        >>> mp.dps = 15
        >>> curve = algebraic_curve({(0, 2): 1, (1, 0): 1, (3, 0): -1})
        >>> forms = (lambda x, y: 1 / y,)
        >>> constant = curve.riemann_constant(forms)
        >>> periods = curve.periods(forms)
        >>> mp.almosteq(constant.value[0], (1 + periods.tau[0, 0]) / 2)
        True

    The direct level-two integrations are substantially more expensive than
    ordinary periods, although their cost does not include an exponential
    characteristic enumeration.
    """
    prepared, hyperelliptic_coefficients = _normalise_algebraic_curve_input(
        ctx, curve)
    if hyperelliptic_coefficients is not None and differentials is None:
        omega, tau, unused_kappa, characteristic = ctx.hyperelliptic_data(
            hyperelliptic_coefficients)
        a, b = characteristic
        genus = tau.rows
        value = ctx.matrix([
            ctx.fsum(tau[row, column] * a[column]
                     for column in range(genus)) + b[row]
            for row in range(genus)])
        if base_place is not None:
            displacement = abel_map(
                ctx, curve, base_place, differentials=None)
            value += (genus - 1) * ((2 * omega) ** -1 * displacement)
        characteristic = _jacobian_characteristic(ctx, value, tau)
        return CurveRiemannConstant(
            value, characteristic, base_place, None)

    forms = _curve_differential_sequence(
        differentials, "differentials")
    periods_data = periods(ctx, curve, forms)
    genus = periods_data.genus
    quadrature_order = max(12, ctx.dps // 2)
    value, cycle_integrals, unused_normalised = _stage_riemann_constant(
        ctx, (prepared, forms, quadrature_order))
    if base_place is not None:
        displacement = abel_map(
            ctx, curve, base_place, forms)
        value += (genus - 1) * (
            (2 * periods_data.omega) ** -1 * displacement)
    characteristic = _jacobian_characteristic(
        ctx, value, periods_data.tau)
    max_sheet_residual = max(
        (integral.max_sheet_residual for integral in cycle_integrals),
        default=ctx.zero)
    return CurveRiemannConstant(
        value, characteristic, base_place, max_sheet_residual)


def validate(ctx, result):
    r"""Validate a result record returned by the curve functions.

    ``result`` is one of ``CurveBranchLocus``, ``CurveMonodromy``,
    ``CurveGenus``, ``CurveHomology``, ``CurvePeriods`` or
    ``CurveRiemannConstant``.  The returned
    ``CurveValidation`` record contains one named ``CurveCheck`` per
    invariant, the largest
    numerical residual among them, and whether every check passed.  The
    checks are recomputed from the record itself; the underlying curve
    data is not recomputed.

        >>> from mpmath import algebraic_curve
        >>> curve = algebraic_curve((0, -1, 0, 1))
        >>> report = curve.validate(curve.periods())
        >>> report.passed
        True
        >>> report.checks[0]
        CurveCheck(name='tau_symmetry_residual', value=mpf('0.0'), passed=True)
    """
    checks = []
    residuals = []
    if isinstance(result, CurveBranchLocus):
        values = tuple(result.branch_values)
        scale = max([ctx.one] + [abs(value) for value in values])
        tolerance = ctx.sqrt(ctx.eps) * scale
        separation = min(
            (abs(left - right)
             for index, left in enumerate(values)
             for right in values[index + 1:]),
            default=ctx.inf)
        checks.append(CurveCheck(
            "branch_values_distinct", separation, separation > tolerance))
        checks.append(CurveCheck(
            "resultant_nonconstant", len(result.resultant),
            len(result.resultant) > 1))
    elif isinstance(result, CurveGenus):
        balanced = (2 * result.genus - 2
                    == -2 * result.degree + result.ramification)
        checks.append(CurveCheck(
            "riemann_hurwitz_balance", balanced, balanced))
    elif isinstance(result, CurveMonodromy):
        degree = len(result.base_sheets)
        all_permutations = result.permutations + (
            result.infinity_permutation,)
        valid = all(
            sorted(permutation) == list(range(degree))
            for permutation in all_permutations)
        checks.append(CurveCheck(
            "permutations_valid", valid, valid))
        orbit = _monodromy_orbit(all_permutations)
        checks.append(CurveCheck(
            "monodromy_transitive", len(orbit), len(orbit) == degree))
        product = tuple(range(degree))
        for permutation in all_permutations:
            product = _compose_permutations(permutation, product)
        checks.append(CurveCheck(
            "monodromy_product_identity", product,
            product == tuple(range(degree))))
        balanced = (2 * result.genus - 2
                    == -2 * degree + result.ramification)
        checks.append(CurveCheck(
            "riemann_hurwitz_balance", balanced, balanced))
    elif isinstance(result, CurveHomology):
        genus = result.genus
        form = result.intersection_form
        antisymmetric = all(
            form[row][column] == -form[column][row]
            for row in range(len(form)) for column in range(len(form)))
        checks.append(CurveCheck(
            "intersection_form_antisymmetric", antisymmetric,
            antisymmetric))
        form_rank = _integer_matrix_rank(form)
        checks.append(CurveCheck(
            "intersection_form_rank", form_rank, form_rank == 2 * genus))
        integral = all(
            isinstance(entry, int)
            for row in result.transformation for entry in row)
        checks.append(CurveCheck(
            "transformation_integral", integral, integral))
        checks.append(CurveCheck(
            "cycle_count", result.cycle_count,
            result.cycle_count == result.intersection_rank
            + result.radical_rank))
    elif isinstance(result, CurvePeriods):
        tau = result.tau
        scale = max([ctx.one] + [
            abs(tau[row, column]) for row in range(tau.rows)
            for column in range(tau.cols)])
        tolerance = 100 * ctx.sqrt(ctx.eps) * scale
        symmetry_residual = result.symmetry_residual
        residuals.append(symmetry_residual)
        checks.append(CurveCheck(
            "tau_symmetry_residual", symmetry_residual,
            symmetry_residual <= tolerance))
        eigenvalues = _tau_imaginary_eigenvalues(ctx, tau)
        checks.append(CurveCheck(
            "tau_imaginary_positive_definite", min(eigenvalues),
            min(eigenvalues) > 0))
        if result.kappa is not None:
            kappa = result.kappa
            kappa_scale = max([ctx.one] + [
                abs(kappa[row, column]) for row in range(kappa.rows)
                for column in range(kappa.cols)])
            kappa_residual = result.kappa_symmetry_residual
            residuals.append(kappa_residual)
            checks.append(CurveCheck(
                "kappa_symmetry_residual", kappa_residual,
                kappa_residual <= 100 * ctx.sqrt(ctx.eps) * kappa_scale))
        if result.max_sheet_residual is not None:
            residuals.append(result.max_sheet_residual)
            checks.append(CurveCheck(
                "max_sheet_residual", result.max_sheet_residual,
                result.max_sheet_residual <= tolerance))
    elif isinstance(result, CurveRiemannConstant):
        value = result.value
        column_vector = value.cols == 1 and value.rows > 0
        checks.append(CurveCheck(
            "jacobian_column_vector", column_vector, column_vector))
        finite = column_vector and all(ctx.isfinite(entry) for entry in value)
        checks.append(CurveCheck("jacobian_finite", finite, finite))
        try:
            a, b = result.characteristic
            characteristic_shape = (
                len(a) == value.rows and len(b) == value.rows)
            characteristic_finite = characteristic_shape and all(
                ctx.isfinite(entry) for entry in a + b)
        except (TypeError, ValueError):
            characteristic_shape = characteristic_finite = False
        checks.append(CurveCheck(
            "characteristic_shape", characteristic_shape,
            characteristic_shape))
        checks.append(CurveCheck(
            "characteristic_finite", characteristic_finite,
            characteristic_finite))
        if result.max_sheet_residual is not None:
            residuals.append(result.max_sheet_residual)
            tolerance = 100 * ctx.sqrt(ctx.eps)
            checks.append(CurveCheck(
                "max_sheet_residual", result.max_sheet_residual,
                result.max_sheet_residual <= tolerance))
    else:
        raise ValueError("validate requires a curve result record")
    maximum_residual = max(residuals) if residuals else None
    return CurveValidation(
        type(result).__name__, all(check.passed for check in checks),
        maximum_residual, tuple(checks))


def fibre(ctx, curve, x):
    r"""Return the labelled fibre of a plane algebraic curve over x.

    ``curve`` uses the input forms accepted by
    :attr:`AlgebraicCurve.branch_locus`, and ``x`` must be a finite regular
    value of the ``x`` projection: not a branch value, and one over which
    the projection does not drop degree.  The returned tuple contains one
    ``CurvePlace`` record per sheet, ordered deterministically by the real
    and imaginary parts of ``y``.  The labelling agrees with the base fibre
    used by :attr:`AlgebraicCurve.monodromy`.

    >>> from mpmath import algebraic_curve, mp
    >>> mp.dps = 15
    >>> curve = algebraic_curve((0, -1, 0, 1))
    >>> [mp.nstr(place.y, 6) for place in curve.fibre(2)]
    ['-2.44949', '2.44949']
    """
    prepared, unused_hyperelliptic = _normalise_algebraic_curve_input(
        ctx, curve)
    x = ctx.convert(x)
    if not ctx.isfinite(x):
        raise ValueError("x must be finite")
    sheets = _ordered_plane_curve_sheets(ctx, prepared, x)
    scale = max([ctx.one] + [abs(value) for value in sheets])
    separation = min(
        (abs(left - right)
         for index, left in enumerate(sheets)
         for right in sheets[index + 1:]),
        default=ctx.inf)
    if separation <= 100 * ctx.sqrt(ctx.eps) * scale:
        raise ValueError("x must not be a finite branch value")
    return tuple(CurvePlace(x, value) for value in sheets)


def path(ctx, curve, start, end):
    r"""Return a lifted path between two regular finite places.

    ``start`` and ``end`` are regular finite places, each given as a
    ``(x, y)`` pair, a ``CurvePlace`` from :meth:`AlgebraicCurve.fibre`, or
    a chart-backed place from :meth:`AlgebraicCurve.chart_place`.  A guarded
    polyline in the x-plane avoids the branch values and is lifted by
    numerical continuation between the places' affine junction points;
    chart tails are joined at those junctions.  The returned ``CurvePath``
    record contains an opaque curve and numerical-context identity, the
    endpoint places, the sheet index reached, the continuation record
    carrying the numerical routing data used by
    :meth:`AlgebraicCurve.integral`, and any joined chart tails.

    Both junction points must have distinct ``x`` values, and ``end`` must
    lie on the sheet reached by continuation; otherwise ``ValueError`` is
    raised.

    >>> from mpmath import algebraic_curve, mp
    >>> mp.dps = 15
    >>> curve = algebraic_curve({(0, 2): 1, (1, 0): -1})
    >>> path = curve.path((1, 1), (4, 2))
    >>> mp.nstr(path.start.y, 6), mp.nstr(path.end.y, 6)
    ('1.0', '2.0')
    """
    prepared, unused_hyperelliptic = _normalise_algebraic_curve_input(
        ctx, curve)
    start_junction, start_tail, start_place = _normalise_curve_endpoint(
        ctx, prepared, start, "start")
    end_junction, end_tail, end_place = _normalise_curve_endpoint(
        ctx, prepared, end, "end")
    if start_junction.x == end_junction.x:
        raise ValueError(
            "path endpoints must have distinct x values")
    branch_values, unused_resultant = _stage_branch_locus(ctx, prepared)
    path = _guarded_open_path(
        ctx, start_junction.x, end_junction.x, branch_values)
    lifted = _lift_plane_curve_path(
        ctx, prepared, path, start_junction.y)
    scale = max(ctx.one, abs(end_junction.x), abs(end_junction.y),
                abs(lifted.end.y))
    if abs(lifted.end.y - end_junction.y) > 100 * ctx.sqrt(ctx.eps) * scale:
        raise ValueError(
            "the lifted path from start does not reach end; the two "
            "places lie on different sheets along the guarded path")
    return CurvePath(
        curve_key=(prepared, _curve_cache_state(ctx)),
        start=start_place,
        end=end_place,
        sheet=lifted.sheet,
        continuation=lifted.continuation,
        start_tail=start_tail,
        end_tail=end_tail)


def integral(ctx, curve, differentials, path):
    r"""Integrate one differential or a differential basis along a path.

    ``differentials`` is either a single callable ``f(x, y)`` returning the
    coefficient of ``dx``, or a sequence of such callables; ``path`` is a
    ``CurvePath`` from :meth:`AlgebraicCurve.path`.  A single differential
    gives a scalar ``values`` entry, a sequence gives one entry per form.
    Chart tails joined to the path are integrated through their coordinate
    maps with the same differentials.  The returned ``CurveIntegral`` record
    also carries the maximum curve-equation residual encountered on the
    integration nodes and the number of path segments.  A path is bound to
    the curve and working precision at which it was constructed and cannot
    be reused with a different curve or precision.

    >>> from mpmath import algebraic_curve, mp
    >>> mp.dps = 15
    >>> curve = algebraic_curve({(0, 2): 1, (1, 0): -1})
    >>> path = curve.path((1, 1), (4, 2))
    >>> integral = curve.integral(lambda x, y: 1 / y, path)
    >>> mp.nstr(integral.values, 12)
    '2.0'
    """
    prepared, unused_hyperelliptic = _normalise_algebraic_curve_input(
        ctx, curve)
    if callable(differentials):
        single = True
        forms = (differentials,)
    else:
        single = False
        forms = _curve_differential_sequence(
            differentials, "differentials")
    if not isinstance(path, CurvePath):
        raise ValueError("path must be a CurvePath from AlgebraicCurve.path")
    if path.curve_key != (prepared, _curve_cache_state(ctx)):
        raise ValueError(
            "path was constructed for a different curve or precision")
    integral = _integrate_plane_curve_path(
        ctx, prepared, path.continuation, forms, sheet=path.sheet)
    totals = list(integral.values)
    max_residual = integral.max_sheet_residual
    segments = integral.segments
    for tail, sign in ((path.start_tail, 1), (path.end_tail, -1)):
        if tail is None:
            continue
        pullbacks = _pullback_plane_curve_differentials(
            forms, _validated_chart_coordinate_map(ctx, tail.chart))
        local = _integrate_plane_curve_branch(
            ctx, tail.chart.curve, tail.branch, pullbacks)
        totals = [total + sign * value
                  for total, value in zip(totals, local.values)]
        max_residual = max(max_residual, local.max_sheet_residual)
        segments += local.segments
    values = totals[0] if single else tuple(totals)
    return CurveIntegral(values, max_residual, segments)


def _normalise_curve_places(ctx, curve, target):
    """Return ``(junction, tail, place)`` triples from a place or divisor."""
    if isinstance(target, CurvePlace):
        candidates = [target]
    else:
        try:
            left, right = target
            pair = True
        except (TypeError, ValueError):
            pair = False
        if pair and not any(
                isinstance(value, (list, tuple, CurvePlace))
                for value in (left, right)):
            candidates = [target]
        else:
            candidates = list(target)
    return tuple(
        _normalise_curve_endpoint(ctx, curve, place, "target")
        for place in candidates)


def _curve_contains_chart_place(value):
    """Return whether a target or base place input is chart-backed."""
    if isinstance(value, CurvePlace):
        return value.chart is not None
    try:
        return any(isinstance(place, CurvePlace) and place.chart is not None
                   for place in value)
    except TypeError:
        return False


def abel_map(ctx, curve, target, differentials=None,
                   base_place=None, reduce=False):
    r"""Evaluate the Abel map of a place or divisor on a plane curve.

    ``target`` is one regular finite place, given as a ``(x, y)`` pair or
    ``CurvePlace``, a chart-backed place from
    :meth:`AlgebraicCurve.chart_place`, or a sequence of places representing
    an effective divisor; an empty sequence returns the zero vector.  A
    hyperelliptic coefficient sequence without supplied differentials is
    dispatched to :func:`~mpmath.hyperelliptic_abel_map`; a general plane
    curve requires ``differentials``, one first-kind callable per genus,
    and returns the unnormalized Abelian coordinates they integrate to.
    Chart-backed places require the general pipeline.

    ``base_place`` selects a regular finite base place; the default is
    sheet zero over the internally selected computational base point.
    With ``reduce=True`` the unnormalized result is reduced modulo the full
    period lattice ``[2*omega, 2*omega_prime]`` of the supplied basis,
    equivalent to applying
    :meth:`AlgebraicCurve.lattice_reduce`.

    >>> from mpmath import algebraic_curve, mp
    >>> mp.dps = 15
    >>> curve = algebraic_curve({(0, 2): 1, (1, 0): 1, (3, 0): -1})
    >>> point = (mp.mpf(2), mp.sqrt(6))
    >>> forms = (lambda x, y: 1 / y,)
    >>> value = curve.abel_map(point, forms, base_place=point)
    >>> mp.nstr(mp.norm(value), 3)
    '0.0'
    """
    prepared, hyperelliptic_coefficients = _normalise_algebraic_curve_input(
        ctx, curve)
    if hyperelliptic_coefficients is not None and differentials is None:
        if (_curve_contains_chart_place(target)
                or (base_place is not None
                    and _curve_contains_chart_place(base_place))):
            raise ValueError(
                "chart-backed places require the general pipeline with "
                "supplied differentials")
        targets = _normalise_abel_targets(ctx, target)
        if base_place is None:
            return ctx.hyperelliptic_abel_map(
                hyperelliptic_coefficients, targets, reduce=reduce)
        result = (ctx.hyperelliptic_abel_map(
                      hyperelliptic_coefficients, targets)
                  - len(targets) * ctx.hyperelliptic_abel_map(
                      hyperelliptic_coefficients, base_place))
        if reduce:
            result = lattice_reduce(
                ctx, result, periods(ctx, curve)).value
        return result

    forms = _curve_differential_sequence(
        differentials, "differentials")
    monodromy = _stage_monodromy(ctx, prepared)
    genus = monodromy.genus
    if len(forms) != genus:
        raise ValueError(
            "differentials must contain one form per genus")
    if base_place is None:
        base_junction = _PlaneCurvePlace(
            monodromy.base_point, monodromy.base_sheets[0])
        base_tail = None
    else:
        base_junction, base_tail, unused_base = (
            _normalise_curve_endpoint(
                ctx, prepared, base_place, "base_place"))
    branch_values, unused_resultant = _stage_branch_locus(ctx, prepared)
    quadrature_order = max(12, ctx.dps // 2)

    def place_value(junction, tail):
        if _same_numerical_place(
                ctx, (junction.x, junction.y),
                (monodromy.base_point, monodromy.base_sheets[0])):
            affine_value = ctx.zeros(genus, 1)
        else:
            affine_value = _finite_base_abel_value(
                ctx, prepared, monodromy, branch_values, junction, forms,
                quadrature_order)
        if tail is None:
            return affine_value
        pullbacks = _pullback_plane_curve_differentials(
            forms, _validated_chart_coordinate_map(ctx, tail.chart))
        local = _integrate_plane_curve_branch(
            ctx, tail.chart.curve, tail.branch, pullbacks)
        return affine_value - ctx.matrix(local.values)

    places = _normalise_curve_places(ctx, prepared, target)
    result = ctx.zeros(genus, 1)
    for junction, tail, unused_place in places:
        result += place_value(junction, tail)
    result -= len(places) * place_value(base_junction, base_tail)
    if reduce:
        result = lattice_reduce(
            ctx, result, periods(ctx, curve, differentials)).value
    return result


def lattice_reduce(ctx, value, periods):
    r"""Reduce a Jacobian vector modulo the period lattice.

    ``value`` is a genus-length column vector of Abelian coordinates.  If
    ``periods`` is a ``CurvePeriods`` record, ``value`` is in the original
    differential basis and is reduced by the full lattice
    ``[2*omega, 2*omega_prime]``.  If ``periods`` is a normalized Riemann
    matrix ``tau``, ``value`` is in normalized coordinates and is reduced
    by ``[I, tau]``.  The returned ``CurveLatticeReduction`` record contains
    the equivalent vector and the integer lattice shift ``(m, n)``.

    >>> from mpmath import algebraic_curve, mp
    >>> mp.dps = 15
    >>> curve = algebraic_curve((0, -1, 0, 1))
    >>> tau = curve.riemann_matrix()
    >>> reduced = curve.lattice_reduce(mp.matrix([2 + 1j]), tau)
    >>> reduced.shift
    (2, 1)
    >>> mp.nstr(reduced.value[0, 0], 3)
    '0.0'
    """
    if isinstance(periods, CurvePeriods):
        genus = periods.genus
        period_matrix = ctx.matrix(genus, 2 * genus)
        period_matrix[:, :genus] = 2 * periods.omega
        period_matrix[:, genus:] = 2 * periods.omega_prime
    else:
        try:
            tau = ctx.matrix(periods)
        except (TypeError, ValueError):
            raise ValueError(
                "periods must be square or a CurvePeriods record")
        if tau.rows != tau.cols:
            raise ValueError(
                "periods must be square or a CurvePeriods record")
        genus = tau.rows
        period_matrix = ctx.matrix(genus, 2 * genus)
        period_matrix[:, :genus] = ctx.eye(genus)
        period_matrix[:, genus:] = tau
    try:
        value = ctx.matrix(value)
    except (TypeError, ValueError):
        raise ValueError("value must be a genus-length column vector")
    if value.rows != genus or value.cols != 1:
        raise ValueError("value must be a genus-length column vector")
    lattice = ctx.matrix([
        [ctx.re(period_matrix[row, column])
         for column in range(2 * genus)]
        for row in range(genus)
    ] + [
        [ctx.im(period_matrix[row, column])
         for column in range(2 * genus)]
        for row in range(genus)
    ])
    coordinates = lattice ** -1 * ctx.matrix(
        [ctx.re(entry) for entry in value]
        + [ctx.im(entry) for entry in value])
    shift = tuple(int(ctx.nint(entry)) for entry in coordinates)
    reduced = value - period_matrix * ctx.matrix(shift)
    return CurveLatticeReduction(reduced, shift)
