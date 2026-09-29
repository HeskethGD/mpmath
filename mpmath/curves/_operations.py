"""Internal orchestration for :class:`~mpmath.curves.Curve`."""

from ._context import _curve_cache_state
from ._hyperelliptic import _hyperelliptic_abel_map
from ._hyperelliptic.jacobian import _hyperelliptic_characteristic
from ._hyperelliptic.model import _normalize_abel_targets
from ._records import (
    CurveBranchLocus,
    CurveCheck,
    CurveFirstKindPeriods,
    CurveGenus,
    CurveHomology,
    CurveIntegral,
    CurveLatticeReduction,
    CurveMonodromy,
    CurvePath,
    CurvePlace,
    CurveRiemannConstant,
    CurveSecondKindAbelMap,
    CurveSecondKindPeriods,
    CurveValidation,
)
from ._stages import (
    _GEOMETRIC_GUARD_BITS,
    _curve_differential_sequence,
    _geometric_abel_divisor,
    _stage_branch_locus,
    _stage_geometric_custom_periods,
    _stage_geometric_custom_riemann_constant,
    _stage_geometric_periods,
    _stage_geometric_polygon,
    _stage_geometric_riemann_constant,
    _stage_hyperelliptic_homology,
    _stage_hyperelliptic_periods,
    _stage_monodromy,
    _tau_imaginary_eigenvalues,
)
from .charts import _validated_chart_coordinate_map
from .continuation import (
    _lift_plane_curve_path,
)
from .differentials import _baker_callable
from .integration import (
    _integrate_plane_curve_branch,
    _integrate_plane_curve_path,
    _pullback_plane_curve_differentials,
)
from .jacobian import (
    _guarded_open_path,
    _jacobian_characteristic,
    _normalize_algebraic_curve_input,
    _normalize_curve_endpoint,
    _period_matrix_from_columns,
    _to_hyperelliptic_points,
)
from .homology import _integer_matrix_rank
from .monodromy import (
    _compose_permutations,
    _monodromy_orbit,
)
from .polynomial import _ordered_plane_curve_sheets

# Curve operations
# ----------------


def _geometric_second_forms(second_kind, second_differentials):
    if second_kind and second_differentials is None:
        raise ValueError("second-kind operations require second_differentials")
    return (() if second_differentials is None else
            _curve_differential_sequence(second_differentials, "second_differentials"))


def _geometric_second_kind_periods(ctx, prepared, first, forms):
    if len(forms) != first.genus:
        raise ValueError("second_differentials must contain one form per genus")
    data = _stage_geometric_custom_periods(ctx, (prepared, forms))
    full = _period_matrix_from_columns(ctx, data.columns, 0, first.genus, first.genus)
    # The supplied forms are dr, while the public half-period convention is
    # 2*eta = -integral_a(dr), 2*eta_prime = -integral_b(dr). Retain the
    # unsymmetrized eta*omega**-1 residual before averaging roundoff-level
    # asymmetry in kappa.
    eta, eta_prime = -full[:, :first.genus]/2, -full[:, first.genus:]/2
    raw_kappa = eta * first.omega**-1
    return CurveSecondKindPeriods(
        first.genus, forms, eta, eta_prime, (raw_kappa+raw_kappa.T)/2,
        ctx.norm(raw_kappa-raw_kappa.T), data.max_sheet_residual,
        "general", "geometric-polygon")


def _geometric_first_kind_periods(ctx, prepared, forms=None):
    data = (_stage_geometric_periods(ctx, prepared) if forms is None else
            _stage_geometric_custom_periods(ctx, (prepared, forms)))
    genus = data.genus
    full = _period_matrix_from_columns(ctx, data.columns, 0, genus, genus)
    omega, omega_prime = full[:, :genus] / 2, full[:, genus:] / 2
    raw_tau = omega ** -1 * omega_prime
    tau = (raw_tau + raw_tau.T) / 2
    eigenvalues = _tau_imaginary_eigenvalues(ctx, tau)
    if min(eigenvalues) <= 0:
        raise ValueError("normalized period matrix is not positive definite")
    if forms is None:
        forms = tuple(_baker_callable(ctx, data.basis, i) for i in range(genus))
    return CurveFirstKindPeriods(
        genus, forms, omega, omega_prime, tau, ctx.norm(raw_tau - raw_tau.T),
        eigenvalues, data.max_sheet_residual, "general", "geometric-polygon")


def branch_locus(ctx, curve):
    """Implement :meth:`Curve.branch_locus`."""
    prepared, _unused_hyperelliptic = _normalize_algebraic_curve_input(
        ctx, curve)
    branch_values, resultant = _stage_branch_locus(ctx, prepared)
    return CurveBranchLocus(prepared.y_degree, branch_values, resultant)


def monodromy(ctx, curve):
    """Implement :meth:`Curve.monodromy`."""
    prepared, _unused_hyperelliptic = _normalize_algebraic_curve_input(
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
    """Implement :meth:`Curve.genus_data`."""
    prepared, unused_hyperelliptic = _normalize_algebraic_curve_input(
        ctx, curve)
    if unused_hyperelliptic is None:
        with ctx.extraprec(_GEOMETRIC_GUARD_BITS):
            graph, _unused_polygon = _stage_geometric_polygon(ctx, prepared)
        return CurveGenus(graph.genus, prepared.y_degree,
                          2 * graph.genus - 2 + 2 * prepared.y_degree)
    monodromy = _stage_monodromy(ctx, prepared)
    return CurveGenus(
        monodromy.genus, prepared.y_degree, monodromy.ramification)


def homology(ctx, curve):
    """Implement :meth:`Curve.homology`."""
    prepared, hyperelliptic_model = _normalize_algebraic_curve_input(
        ctx, curve)
    if hyperelliptic_model is not None:
        genus = _stage_hyperelliptic_homology(
            ctx, hyperelliptic_model.coefficients)
        cycle_count = 2 * genus
        intersection_form = tuple(tuple(
            1 if row < genus and column == genus + row
            else -1 if column < genus and row == genus + column
            else 0
            for column in range(cycle_count))
            for row in range(cycle_count))
        transformation = tuple(tuple(
            int(row == column) for column in range(cycle_count))
            for row in range(cycle_count))
        return CurveHomology(
            genus=genus,
            cycle_count=cycle_count,
            boundary_components=0,
            intersection_rank=cycle_count,
            radical_rank=0,
            intersection_form=intersection_form,
            transformation=transformation,
            engine="hyperelliptic",
            marking="baker")


    with ctx.extraprec(_GEOMETRIC_GUARD_BITS):
        graph, polygon = _stage_geometric_polygon(ctx, prepared)
    count = 2 * graph.genus
    identity = tuple(tuple(int(i == j) for j in range(count))
                     for i in range(count))
    # Report the compact canonical basis itself, not the punctured
    # graph's redundant cycles. Its transformation is the identity.
    return CurveHomology(graph.genus, count, 0, count, 0,
                         polygon.polygon.intersection, identity,
                         "general", "geometric-polygon")


def periods(ctx, curve, differentials=None, *, second_kind=False,
            second_differentials=None, _return_first=False):
    """Dispatch first- or second-kind periods and assemble their result records."""
    prepared, hyperelliptic_model = _normalize_algebraic_curve_input(
        ctx, curve)
    if hyperelliptic_model is not None and differentials is None:
        if second_differentials is not None:
            raise ValueError(
                "supplied second-kind differentials require a supplied "
                "first-kind basis")
        cached = _stage_hyperelliptic_periods(
            ctx, (hyperelliptic_model.coefficients, second_kind))
        if second_kind:
            omega, omega_prime, eta, eta_prime, tau, kappa = (
                +matrix for matrix in cached)
            kappa_symmetry_residual = ctx.norm(kappa - kappa.T)
        else:
            omega, omega_prime, tau = (+matrix for matrix in cached)
            eta = eta_prime = kappa = None
            kappa_symmetry_residual = None
        genus = omega.rows
        first_record = CurveFirstKindPeriods(
            genus, None, omega, omega_prime, tau,
            ctx.norm(tau - tau.T),
            _tau_imaginary_eigenvalues(ctx, tau), None,
            "hyperelliptic", "baker")
        if second_kind:
            second_record = CurveSecondKindPeriods(
                genus, None, eta, eta_prime, kappa,
                kappa_symmetry_residual, None,
                "hyperelliptic", "baker")
            if _return_first:
                return first_record, second_record
            return second_record
        return first_record


    second_forms = _geometric_second_forms(second_kind, second_differentials)
    forms = (None if differentials is None else
             _curve_differential_sequence(differentials, "differentials"))
    first = _geometric_first_kind_periods(ctx, prepared, forms)
    if second_forms:
        second = _geometric_second_kind_periods(ctx, prepared, first, second_forms)
        return (first, second) if _return_first else second
    return first


def riemann_matrix(ctx, curve, differentials=None):
    """Implement :meth:`Curve.riemann_matrix`."""
    return periods(ctx, curve, differentials).tau


def riemann_constant(ctx, curve, differentials=None, *,
                           base_place=None):
    """Implement :meth:`Curve.riemann_constant`."""
    prepared, hyperelliptic_model = _normalize_algebraic_curve_input(
        ctx, curve)
    if hyperelliptic_model is not None and differentials is None:
        cached = _stage_hyperelliptic_periods(
            ctx, (hyperelliptic_model.coefficients, False))
        omega, tau = +cached[0], +cached[2]
        characteristic = _hyperelliptic_characteristic(ctx, omega.rows)
        a, b = characteristic
        genus = tau.rows
        value = ctx.matrix([
            ctx.fsum(tau[row, column] * a[column]
                     for column in range(genus)) + b[row]
            for row in range(genus)])
        if base_place is not None:
            # In our additive convention, A_P(D) = A_Q(D) - (g-1) A_Q(P).
            # Thus theta(A_P(D) + K_P) keeps the same argument when
            # K_P = K_Q + (g-1) A_Q(P). The Abel map below supplies A_Q(P).
            displacement = abel_map(
                ctx, curve, base_place, differentials=None)
            value += (genus - 1) * ((2 * omega) ** -1 * displacement)
        characteristic = _jacobian_characteristic(ctx, value, tau)
        return CurveRiemannConstant(
            value, characteristic, base_place, None,
            "hyperelliptic", "baker")


    forms = (None if differentials is None else
             _curve_differential_sequence(differentials, "differentials"))
    data = _geometric_first_kind_periods(ctx, prepared, forms)
    entries, cycles = (_stage_geometric_riemann_constant(ctx, prepared)
                      if forms is None else
                      _stage_geometric_custom_riemann_constant(ctx, (prepared, forms)))
    value = ctx.matrix(entries)
    if base_place is not None:
        # A_P(D) = A_Q(D) - (g-1) A_Q(P); our additive theta shift gains
        # the same (g-1) A_Q(P) so its vanishing divisor is unchanged.
        displacement = abel_map(ctx, curve, base_place, forms)
        value += (data.genus - 1) * ((2 * data.omega) ** -1 * displacement)
    return CurveRiemannConstant(
        value, _jacobian_characteristic(ctx, value, data.tau), base_place,
        max((cycle.max_sheet_residual for cycle in cycles), default=ctx.zero),
        "general", "geometric-polygon")


def validate(ctx, result):
    """Implement :meth:`Curve.validate`."""
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
    elif isinstance(result, CurveFirstKindPeriods):
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
        if result.max_sheet_residual is not None:
            residuals.append(result.max_sheet_residual)
            checks.append(CurveCheck(
                "max_sheet_residual", result.max_sheet_residual,
                result.max_sheet_residual <= tolerance))
    elif isinstance(result, CurveSecondKindPeriods):
        kappa = result.kappa
        kappa_scale = max([ctx.one] + [
            abs(kappa[row, column]) for row in range(kappa.rows)
            for column in range(kappa.cols)])
        tolerance = 100 * ctx.sqrt(ctx.eps) * kappa_scale
        kappa_residual = result.kappa_symmetry_residual
        residuals.append(kappa_residual)
        checks.append(CurveCheck(
            "kappa_symmetry_residual", kappa_residual,
            kappa_residual <= tolerance))
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
        raise TypeError("validate requires a curve result record")
    maximum_residual = max(residuals) if residuals else None
    return CurveValidation(
        type(result).__name__, all(check.passed for check in checks),
        maximum_residual, tuple(checks))


def fibre(ctx, curve, x):
    """Implement :meth:`Curve.fibre`."""
    prepared, _unused_hyperelliptic = _normalize_algebraic_curve_input(
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
    """Implement :meth:`Curve.path`."""
    prepared, _unused_hyperelliptic = _normalize_algebraic_curve_input(
        ctx, curve)
    start_junction, start_tail, start_place = _normalize_curve_endpoint(
        ctx, prepared, start, "start")
    end_junction, end_tail, end_place = _normalize_curve_endpoint(
        ctx, prepared, end, "end")
    if start_junction.x == end_junction.x:
        raise ValueError(
            "path endpoints must have distinct x values")
    branch_values, _unused_resultant = _stage_branch_locus(ctx, prepared)
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
    """Implement :meth:`Curve.integral`."""
    prepared, _unused_hyperelliptic = _normalize_algebraic_curve_input(
        ctx, curve)
    if callable(differentials):
        single = True
        forms = (differentials,)
    else:
        single = False
        forms = _curve_differential_sequence(
            differentials, "differentials")
    if not isinstance(path, CurvePath):
        raise TypeError("path must be a CurvePath from Curve.path")
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


def _normalize_curve_places(ctx, curve, target):
    """Return ``(junction, tail, place)`` triples from a place or divisor."""
    if isinstance(target, CurvePlace):
        candidates = [target]
    else:
        left = right = None
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
        _normalize_curve_endpoint(ctx, curve, place, "target")
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


def _reduce_second_kind_abel(
        ctx, first, second, first_periods, second_periods):
    """Reduce paired Abelian integrals with one shared lattice shift."""
    reduction = lattice_reduce(ctx, first, first_periods)
    genus = first_periods.genus
    matrix = ctx.matrix(genus, 2 * genus)
    matrix[:, :genus] = 2 * second_periods.eta
    matrix[:, genus:] = 2 * second_periods.eta_prime
    second += matrix * ctx.matrix(reduction.shift)
    return second, reduction.shift


def abel_map(ctx, curve, target, differentials=None, *, second_kind=False,
             second_differentials=None, base_place=None, reduce=False):
    """Implement :meth:`Curve.abel_map_kind_1`."""
    prepared, hyperelliptic_model = _normalize_algebraic_curve_input(
        ctx, curve)
    if hyperelliptic_model is not None and differentials is None:
        if second_differentials is not None:
            raise ValueError(
                "second_differentials require a supplied first-kind basis")
        if (_curve_contains_chart_place(target)
                or (base_place is not None
                    and _curve_contains_chart_place(base_place))):
            raise ValueError(
                "chart-backed places require the general pipeline with "
                "supplied differentials")
        targets = _normalize_abel_targets(ctx, target)
        transformed_targets = _to_hyperelliptic_points(
            ctx, hyperelliptic_model, targets)
        if base_place is None:
            result = _hyperelliptic_abel_map(
                ctx, hyperelliptic_model.coefficients, transformed_targets,
                reduce=reduce, second_kind=second_kind,
                _return_shift=second_kind and reduce)
            if second_kind:
                second = result[1]
                if reduce:
                    shift = result[-1] if len(result) == 3 else None
                else:
                    shift = None
                return CurveSecondKindAbelMap(
                    second, shift, "hyperelliptic", "baker")
            return result
        base_points = _normalize_abel_targets(ctx, base_place)
        if len(base_points) != 1:
            raise ValueError("base_place must be one affine point (x, y)")
        transformed_base = _to_hyperelliptic_points(
            ctx, hyperelliptic_model, base_points)[0]
        target_result = _hyperelliptic_abel_map(
            ctx, hyperelliptic_model.coefficients, transformed_targets,
            second_kind=second_kind)
        base_result = _hyperelliptic_abel_map(
            ctx, hyperelliptic_model.coefficients, transformed_base,
            second_kind=second_kind)
        if second_kind:
            first = target_result[0] - len(targets) * base_result[0]
            second = target_result[1] - len(targets) * base_result[1]
            shift = None
            if reduce:
                first_periods = periods(ctx, curve)
                second_periods = periods(
                    ctx, curve, second_kind=True)
                second, shift = _reduce_second_kind_abel(
                    ctx, first, second, first_periods, second_periods)
            return CurveSecondKindAbelMap(
                second, shift, "hyperelliptic", "baker")
        result = target_result - len(targets) * base_result
        if reduce:
            result = lattice_reduce(
                ctx, result, periods(ctx, curve)).value
        return result


    second_forms = _geometric_second_forms(second_kind, second_differentials)
    forms = (None if differentials is None else
             _curve_differential_sequence(differentials, "differentials"))
    places = _normalize_curve_places(ctx, prepared, target)
    result = ctx.matrix(_geometric_abel_divisor(
        ctx, prepared, tuple(place for junction, tail, place in places),
        base_place=base_place, forms=forms, second_forms=second_forms))
    if second_forms:
        genus = len(second_forms)
        first, second = result[:genus, :], result[genus:, :]
        shift = None
        if reduce:
            first_periods = _geometric_first_kind_periods(ctx, prepared, forms)
            second_periods = _geometric_second_kind_periods(
                ctx, prepared, first_periods, second_forms)
            second, shift = _reduce_second_kind_abel(
                ctx, first, second, first_periods, second_periods)
        return CurveSecondKindAbelMap(second, shift, "general", "geometric-polygon")
    if reduce:
        result = lattice_reduce(ctx, result, periods(ctx, curve, forms)).value
    return result


def lattice_reduce(ctx, value, periods):
    """Implement :meth:`Curve.lattice_reduce`."""
    if isinstance(periods, CurveFirstKindPeriods):
        genus = periods.genus
        period_matrix = ctx.matrix(genus, 2 * genus)
        period_matrix[:, :genus] = 2 * periods.omega
        period_matrix[:, genus:] = 2 * periods.omega_prime
    else:
        try:
            tau = ctx.matrix(periods)
        except (TypeError, ValueError):
            raise ValueError(
                "periods must be square or a CurveFirstKindPeriods record")
        if tau.rows != tau.cols:
            raise ValueError(
                "periods must be square or a CurveFirstKindPeriods record")
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
