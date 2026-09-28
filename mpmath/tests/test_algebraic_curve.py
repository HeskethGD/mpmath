import pytest

import mpmath
import mpmath.curves.integration as curve_integration
from mpmath import (
    AlgebraicCurve, CurveBranchLocus, CurveChart, CurveCheck,
    CurveFirstKindPeriods, CurveGenus, CurveHomology, CurveIntegral,
    CurveLatticeReduction, CurveMonodromy, CurvePath, CurvePlace,
    CurveRiemannConstant, CurveSecondKindAbelMap, CurveSecondKindPeriods,
    CurveValidation, mp,
)
from mpmath.curves._hyperelliptic import (
    _hyperelliptic_abel_map as _specialized_hyperelliptic_abel_map,
    _hyperelliptic_periods as _specialized_hyperelliptic_periods,
)
from mpmath.curves.charts import (
    chart_fibre as _chart_fibre,
    chart_integral as _chart_integral,
    monomial_chart as _chart_monomial,
)
from mpmath.curves.continuation import (
    _continue_plane_curve_branch,
    _continue_plane_curve_sheets_adaptive,
    _lift_plane_curve_path,
    _radial_branch_geometry,
)
from mpmath.curves.integration import (
    _integrate_plane_curve_path,
    _integrate_plane_curve_branch,
    _pullback_plane_curve_differentials,
)
from mpmath.curves.homology import _brahana_canonical_words
from mpmath.curves.monodromy import _radial_plane_curve_monodromy
from mpmath.curves.polynomial import (
    _evaluate_plane_derivative,
    _evaluate_plane_polynomial,
    _finite_plane_curve_sheets,
    _monomial_plane_curve_chart,
    _ordered_plane_curve_sheets,
    _plane_curve_critical_values,
    _plane_polynomial_y_coefficients,
    _prepare_plane_curve,
)
from mpmath.curves import _operations


def _curve(specification):
    return AlgebraicCurve(mp, specification)


def hyperelliptic_periods(coefficients, **kwargs):
    """Private specialized oracle for facade-dispatch tests."""
    return _specialized_hyperelliptic_periods(
        mp, coefficients, **kwargs)


def hyperelliptic_abel_map(coefficients, target, **kwargs):
    """Private specialized oracle for facade-dispatch tests."""
    return _specialized_hyperelliptic_abel_map(
        mp, coefficients, target, **kwargs)


def curve_branch_locus(curve):
    return _curve(curve).branch_locus


def curve_monodromy(curve):
    return _curve(curve).monodromy


def curve_genus(curve):
    return _curve(curve).genus_data


def curve_homology(curve):
    return _curve(curve).homology


def curve_periods(curve, differentials=None, *, second_kind=False,
                  second_differentials=None):
    instance = _curve(curve)
    if second_kind or second_differentials is not None:
        return instance.second_kind_periods(
            differentials, second_differentials=second_differentials)
    return instance.first_kind_periods(differentials)


def curve_riemann_matrix(curve, differentials=None):
    return _curve(curve).riemann_matrix(differentials)


def curve_riemann_constant(curve, differentials=None, *, base_place=None):
    return _curve(curve).riemann_constant(
        differentials, base_place=base_place)


def curve_validate(result):
    return _operations.validate(mp, result)


def curve_fibre(curve, x):
    return _curve(curve).fibre(x)


def curve_path(curve, start, end):
    return _curve(curve).path(start, end)


def curve_integral(curve, differentials, path):
    return _curve(curve).integral(differentials, path)


def curve_abel_map(curve, target, differentials=None, base_place=None,
                   reduce=False, second_kind=False,
                   second_differentials=None):
    instance = _curve(curve)
    if second_kind or second_differentials is not None:
        return instance.second_kind_abel_map(
            target, differentials,
            second_differentials=second_differentials,
            base_place=base_place, reduce=reduce)
    return instance.abel_map(
        target, differentials, base_place=base_place, reduce=reduce)


def curve_lattice_reduce(value, periods):
    return _operations.lattice_reduce(mp, value, periods)


def curve_chart(curve, chart_curve, coordinate_map):
    return _curve(curve).chart(chart_curve, coordinate_map)


def curve_chart_monomial(source, x_power, y_power):
    return _chart_monomial(mp, source, x_power, y_power)


def curve_chart_fibre(chart, t):
    return _chart_fibre(mp, chart, t)


def curve_chart_place(curve, chart, seed, cutoff):
    return _curve(curve).chart_place(chart, seed, cutoff)


def curve_chart_integral(chart, differentials, t_path, seed):
    return _chart_integral(mp, chart, differentials, t_path, seed)


def test_brahana_polygon_word_reduction():
    # A maximally interlaced orientable genus-two polygon is deliberately not
    # in commutator order.  The reducer itself verifies the exact free-word
    # identity between this relator and its returned commutators.
    word = tuple((symbol, 1) for symbol in range(4)) + tuple(
        (symbol, -1) for symbol in range(4))
    pairs = _brahana_canonical_words(word)
    assert len(pairs) == 2
    assert all(a and b for a, b in pairs)


def test_plane_curve_representation_and_fibre():
    # F(x,y) = y^2-x is the simplest branched two-sheeted projection.
    mp.dps = 40
    curve = _prepare_plane_curve(mp, {(0, 2): 1, (1, 0): -1})
    assert curve.x_degree == 1
    assert curve.y_degree == 2
    assert _evaluate_plane_polynomial(mp, curve, 4, 2) == 0
    assert _evaluate_plane_derivative(mp, curve, 4, 2, "x") == -1
    assert _evaluate_plane_derivative(mp, curve, 4, 2, "y") == 4
    assert _plane_polynomial_y_coefficients(mp, curve, 4) == [-4, 0, 1]
    sheets = _ordered_plane_curve_sheets(mp, curve, 4)
    assert mp.almosteq(sheets[0], -2)
    assert mp.almosteq(sheets[1], 2)


def test_plane_curve_sheet_continuation_around_branch_point():
    # A positive loop around x=0 exchanges the two branches of sqrt(x).
    mp.dps = 40
    curve = _prepare_plane_curve(mp, {(0, 2): 1, (1, 0): -1})
    path = [mp.exp(2j * mp.pi * step / 32) for step in range(33)]
    continuation = _continue_plane_curve_sheets_adaptive(mp, curve, path)
    assert continuation.permutation == (1, 0)
    assert continuation.steps == 32
    assert continuation.max_residual < mp.mpf("1e-38")
    assert continuation.min_separation > mp.mpf("1.9")


def test_lifted_path_integrals_on_square_root_curve():
    # On the positive sheet of y^2=x, integral_1^4 dx/y = 2.  Around one
    # positive circuit of x=0 that sheet ends on the negative sheet, and the
    # lifted integrals of dx/y and y dx are -4 and -4/3 respectively.
    mp.dps = 30
    curve = _prepare_plane_curve(mp, {(0, 2): 1, (1, 0): -1})

    interval = _continue_plane_curve_sheets_adaptive(mp, curve, (1, 4))
    assert interval.permutation is None
    interval_integral = _integrate_plane_curve_path(
        mp, curve, interval, (lambda x, y: 1 / y,), sheet=1)
    assert mp.almosteq(interval_integral.values[0], 2)
    assert interval_integral.max_sheet_residual < mp.mpf("1e-28")

    lifted_interval = _lift_plane_curve_path(mp, curve, (1, 4), 1)
    assert mp.almosteq(lifted_interval.start.y, 1)
    assert mp.almosteq(lifted_interval.end.y, 2)
    selected_integral = _integrate_plane_curve_path(
        mp, curve, lifted_interval.continuation,
        (lambda x, y: 1 / y,), sheet=lifted_interval.sheet)
    assert abs(selected_integral.values[0] - 2) < mp.mpf("1e-27")
    with pytest.raises(ValueError, match="does not identify a sheet"):
        _lift_plane_curve_path(mp, curve, (1, 4), mp.mpf("1.1"))

    circle = tuple(mp.exp(2j * mp.pi * step / 16)
                   for step in range(17))
    lifted_circle = _continue_plane_curve_sheets_adaptive(mp, curve, circle)
    circle_integrals = _integrate_plane_curve_path(
        mp, curve, lifted_circle,
        (lambda x, y: 1 / y, lambda x, y: y), sheet=1)
    assert abs(circle_integrals.values[0] + 4) < mp.mpf("1e-27")
    assert abs(circle_integrals.values[1] + mp.mpf(4)/3) < mp.mpf("1e-27")
    # Two circuits close the lift; the logarithmic integral counts winding.
    twice = _continue_plane_curve_sheets_adaptive(mp, curve, circle+circle[1:])
    assert twice.permutation == (0, 1)
    value = _integrate_plane_curve_path(mp, curve, twice, (lambda x,y:1/x,),sheet=1)
    assert abs(value.values[0]-4j*mp.pi) < mp.mpf("1e-27")


def test_lifted_path_integrals_use_single_sheet_newton(monkeypatch):
    # Once the degree-three endpoint fibres have been continued, integration
    # should correct only the selected branch instead of resolving all three
    # roots again at every Gauss node.
    mp.dps = 30
    curve = _prepare_plane_curve(mp, {(0, 3): 1, (1, 0): -1})
    continuation = _continue_plane_curve_sheets_adaptive(mp, curve, (1, 8))
    evaluate_derivative = curve_integration._evaluate_plane_derivative
    derivative_calls = []

    def reject_full_fibre_solve(*unused_args, **unused_kwargs):
        raise AssertionError("unexpected full-fibre solve")

    def counted_derivative(*args, **kwargs):
        derivative_calls.append(args[-1])
        return evaluate_derivative(*args, **kwargs)

    monkeypatch.setattr(
        curve_integration, "_plane_curve_sheets", reject_full_fibre_solve)
    monkeypatch.setattr(
        curve_integration, "_evaluate_plane_derivative", counted_derivative)
    integral = _integrate_plane_curve_path(
        mp, curve, continuation, (lambda x, y: 1 / y,),
        sheet=2, quadrature_order=32)
    assert abs(integral.values[0] - mp.mpf("4.5")) < mp.mpf("1e-20")
    assert integral.max_sheet_residual < mp.mpf("1e-28")
    assert derivative_calls == ["x", "y"] * (len(continuation.path)-1)


def test_lifted_path_integrals_fall_back_to_full_fibres(monkeypatch):
    mp.dps = 30
    curve = _prepare_plane_curve(mp, {(0, 3): 1, (1, 0): -1})
    continuation = _continue_plane_curve_sheets_adaptive(mp, curve, (1, 8))
    full_solve = curve_integration._plane_curve_sheets
    calls = []

    def reject_newton(*unused_args, **unused_kwargs):
        return None

    def counted_full_solve(*args, **kwargs):
        calls.append(args[2])
        return full_solve(*args, **kwargs)

    monkeypatch.setattr(
        curve_integration, "_newton_plane_curve_segment_samples",
        reject_newton)
    monkeypatch.setattr(
        curve_integration, "_plane_curve_sheets", counted_full_solve)
    integral = _integrate_plane_curve_path(
        mp, curve, continuation, (lambda x, y: 1 / y,),
        sheet=2, quadrature_order=16)
    assert len(calls) == 16 * (len(continuation.path)-1)
    assert abs(integral.values[0] - mp.mpf("4.5")) < mp.mpf("2e-11")


def test_symmetric_radial_geometry_has_stable_base_point():
    branch_points = (-2, -1, 0, 1, 2)
    with mp.workdps(30):
        base_30 = +_radial_branch_geometry(mp, branch_points)[0]
    with mp.workdps(50):
        base_50 = +_radial_branch_geometry(mp, branch_points)[0]
    assert mp.im(base_30) > 0
    assert mp.im(base_50) > 0
    assert abs(base_30 - base_50) < mp.mpf("1e-28")


def test_trigonal_infinity_uses_positive_local_orientation():
    # The outer circle is clockwise in x but counter-clockwise in z=1/x.
    # Treating it as a negative branch loop gives the wrong number of ribbon
    # boundary components for this three-sheeted cover.
    mp.dps = 20
    curve = _prepare_plane_curve(mp, {
        (0, 3): -1,
        (4, 0): 1,
        (3, 0): 3,
        (2, 0): 7,
        (1, 0): 16,
        (0, 0): 9,
        (2, 1): 4,
        (1, 1): 5,
        (0, 1): 11,
    })
    branch_points, unused_resultant = _plane_curve_critical_values(
        mp, curve)
    monodromy = _radial_plane_curve_monodromy(
        mp, curve, branch_points, circle_steps=12, max_refinements=20)
    assert monodromy.genus == 3
    assert monodromy.ramification == 10


def test_curve_branch_locus_and_validate():
    mp.dps = 25
    locus = curve_branch_locus((0, -1, 0, 1))
    assert locus.degree == 2
    assert all(mp.almosteq(value, expected) for value, expected
               in zip(locus.branch_values, (-1, 0, 1)))
    assert len(locus.resultant) > 1
    validation = curve_validate(locus)
    assert validation.kind == "CurveBranchLocus"
    assert validation.passed
    # A single finite branch value is distinct vacuously; the other
    # ramification of y**2=x lies above infinity.
    assert curve_validate(curve_branch_locus((0, 1))).passed


def test_curve_monodromy_and_genus():
    mp.dps = 25
    monodromy = curve_monodromy((0, -1, 0, 1))
    assert monodromy.transitive
    assert monodromy.product_identity
    assert monodromy.genus == 1
    assert monodromy.ramification == 4
    assert all(permutation == (1, 0)
               for permutation in monodromy.permutations)
    assert monodromy.infinity_permutation == (1, 0)
    assert len(monodromy.base_sheets) == 2
    assert curve_genus((0, -1, 0, 1)) == (1, 2, 4)
    assert curve_validate(curve_genus((0, -1, 0, 1))).passed
    validation = curve_validate(monodromy)
    assert validation.kind == "CurveMonodromy"
    assert validation.passed


def test_curve_homology_lemniscatic():
    mp.dps = 25
    homology = curve_homology((0, -1, 0, 1))
    assert homology.genus == 1
    assert homology.cycle_count == 2
    assert homology.boundary_components == 0
    assert homology.intersection_rank == 2
    assert homology.radical_rank == 0
    assert homology.intersection_form == ((0, 1), (-1, 0))
    assert homology.transformation == ((1, 0), (0, 1))
    assert homology.engine == "hyperelliptic"
    assert homology.marking == "baker"
    assert all(isinstance(entry, int)
               for row in homology.transformation for entry in row)
    assert curve_validate(homology).passed


def test_curve_homology_general_curve_uses_polygon_marking():
    mp.dps = 20
    homology = curve_homology({
        (3, 0): 1,
        (0, 3): 1,
        (0, 0): -1,
    })
    assert homology.genus == 1
    assert homology.engine == "general"
    assert homology.marking == "geometric-polygon"
    assert homology.cycle_count == (
        homology.intersection_rank + homology.radical_rank)
    assert curve_validate(homology).passed


def test_curve_periods_hyperelliptic_dispatch():
    mp.dps = 25
    data = curve_periods((0, -1, 0, 1))
    expected = hyperelliptic_periods((0, -1, 0, 1))
    assert data.genus == 1
    assert data.differentials is None
    assert data.max_sheet_residual is None
    assert data.engine == "hyperelliptic"
    assert data.marking == "baker"
    assert isinstance(data, CurveFirstKindPeriods)
    for actual, reference in zip(
            (data.omega, data.omega_prime, data.tau), expected):
        assert mp.norm(actual - reference) < mp.mpf("1e-22")

    second = curve_periods((0, -1, 0, 1), second_kind=True)
    expected_second = hyperelliptic_periods(
        (0, -1, 0, 1), second_kind=True)
    assert isinstance(second, CurveSecondKindPeriods)
    for actual, reference in zip(
            (second.eta, second.eta_prime, second.kappa),
            (expected_second[2], expected_second[3], expected_second[5])):
        assert mp.norm(actual - reference) < mp.mpf("1e-22")
    assert curve_validate(data).passed
    assert mp.norm(curve_riemann_matrix((0, -1, 0, 1)) - data.tau) == 0
    constant = curve_riemann_constant((0, -1, 0, 1))
    assert isinstance(constant, CurveRiemannConstant)
    half = mp.mpf("0.5")
    assert constant.characteristic == ((half,), (half,))


def test_curve_hyperelliptic_dispatch_is_representation_independent():
    mp.dps = 25
    coefficients = (0, -1, 0, 1)
    sparse = {(0, 2): 1, (1, 0): 1, (3, 0): -1}
    expected = curve_periods(coefficients)
    actual = curve_periods(sparse)
    assert actual.differentials is None
    for name in ("omega", "omega_prime", "tau"):
        assert mp.norm(getattr(actual, name) - getattr(expected, name)) < (
            mp.mpf("1e-22"))


def test_curve_hyperelliptic_linear_y_normalization():
    mp.dps = 25
    # (y + x)**2 = x**3 - x.  The specialized coordinate is z = y + x.
    shifted = {
        (0, 2): 1,
        (1, 1): 2,
        (1, 0): 1,
        (2, 0): 1,
        (3, 0): -1,
    }
    coefficients = (0, -1, 0, 1)
    expected = curve_periods(coefficients)
    actual = curve_periods(shifted)
    for name in ("omega", "omega_prime", "tau"):
        assert mp.norm(getattr(actual, name) - getattr(expected, name)) < (
            mp.mpf("1e-22"))

    x = mp.mpf(2)
    z = mp.sqrt(6)
    shifted_image = curve_abel_map(shifted, (x, z - x))
    expected_image = hyperelliptic_abel_map(coefficients, (x, z))
    assert mp.norm(shifted_image - expected_image) < mp.mpf("1e-22")

    base_z = mp.sqrt(mp.mpf(3) / 8)
    shifted_base = (mp.mpf("-0.5"), base_z + mp.mpf("0.5"))
    shifted_image = curve_abel_map(
        shifted, (x, z - x), base_place=shifted_base)
    expected_image = (
        hyperelliptic_abel_map(coefficients, (x, z))
        - hyperelliptic_abel_map(
            coefficients, (shifted_base[0], base_z)))
    assert mp.norm(shifted_image - expected_image) < mp.mpf("1e-22")


def test_curve_riemann_constant_hyperelliptic_base_change():
    mp.dps = 25
    coefficients = (0, 4, 0, -5, 0, 1)
    x = mp.mpf(5)
    y = mp.sqrt(mp.fsum(
        coefficient * x**degree
        for degree, coefficient in enumerate(coefficients)))
    base_place = (x, y)
    periods = curve_periods(coefficients)
    default = curve_riemann_constant(coefficients)
    shifted = curve_riemann_constant(
        coefficients, base_place=base_place)
    displacement = hyperelliptic_abel_map(coefficients, base_place)
    expected_shift = (2 * periods.omega) ** -1 * displacement
    assert mp.norm(
        shifted.value - default.value - expected_shift) < mp.mpf("1e-22")


def test_curve_periods_general_plane_curve():
    mp.dps = 25
    curve = {
        (0, 2): 1,
        (3, 0): -1,
        (2, 0): 3,
        (1, 0): -3 - 1j,
    }
    differentials = (lambda x, y: 1 / y,)
    data = curve_periods(curve, differentials)
    assert data.genus == 1
    assert data.differentials == differentials
    assert data.symmetry_residual < mp.mpf("1e-24")
    assert min(data.imaginary_eigenvalues) > 0
    assert data.max_sheet_residual < mp.mpf("1e-23")
    tau = curve_riemann_matrix(curve, differentials)
    assert mp.norm(tau - data.tau) == 0
    constant = curve_riemann_constant(curve, differentials)
    assert isinstance(constant, CurveRiemannConstant)
    assert curve_validate(constant).passed
    assert abs(constant.value[0] - (1 + data.tau[0, 0]) / 2) < mp.mpf(
        "1e-22")
    point = curve_fibre(curve, 2)[-1]
    shifted = curve_riemann_constant(
        curve, differentials, base_place=point)
    assert abs(shifted.value[0] - constant.value[0]) < mp.mpf("1e-22")
    validation = curve_validate(data)
    assert validation.kind == "CurveFirstKindPeriods"
    assert validation.passed
    assert validation.maximum_residual < mp.mpf("1e-23")
    assert not curve_validate(data._replace(
        symmetry_residual=mp.one)).passed
    with pytest.raises(ValueError, match="one form per (positive )?genus"):
        curve_periods(curve, (lambda x, y: 1 / y,) * 2)
    automatic = curve_periods(
        {(0, 3): 1, (3, 0): 1, (0, 0): -1})
    assert automatic.genus == 1
    assert automatic.engine == "general"
    assert curve_validate(automatic).passed
    with pytest.raises(
            ValueError, match="second_differentials must contain"):
        curve_periods(
            curve, differentials,
            second_differentials=(lambda x, y: 1 / y,) * 2)


def test_curve_periods_second_kind_general_plane_curve():
    mp.dps = 20
    curve = {(0, 2): 1, (1, 0): 1, (3, 0): -1}
    data = curve_periods(
        curve, (lambda x, y: 1 / y,),
        second_differentials=(lambda x, y: x / y,))
    assert data.genus == 1
    assert data.eta is not None and data.eta_prime is not None
    assert data.kappa is not None
    assert data.kappa_symmetry_residual < mp.mpf("1e-18")
    assert curve_validate(data).passed
    assert not curve_validate(data._replace(
        kappa_symmetry_residual=mp.one)).passed


def test_curve_stage_caching_and_input_forms():
    mp.dps = 20
    from mpmath.curves import _stages as stages
    for stage in (stages._stage_branch_locus, stages._stage_monodromy,
                  stages._stage_geometric_cover,
                  stages._stage_geometric_polygon,
                  stages._stage_geometric_periods):
        stage.cache_clear()
    misses = stages._stage_monodromy.cache_info().misses
    curve_monodromy((0, -1, 0, 1))
    assert stages._stage_monodromy.cache_info().misses == misses + 1
    # Equivalent sparse and term representations share the cache key.
    curve_genus({(0, 2): 1, (1, 0): 1, (3, 0): -1})
    curve_homology(((0, 2, 1), (1, 0, 1), (3, 0, -1)))
    assert stages._stage_monodromy.cache_info().misses == misses + 1
    # Later stages reuse the cached monodromy rather than recomputing it.
    curve_periods((0, -1, 0, 1), (lambda x, y: 1 / y,))
    assert stages._stage_monodromy.cache_info().misses == misses + 1

    class UnhashableDifferential:
        __hash__ = None

        def __call__(self, x, y):
            return 1 / y

    data = curve_periods(
        {(0, 2): 1, (1, 0): 1, (3, 0): -1},
        (UnhashableDifferential(),))
    assert data.genus == 1


def test_curve_fibre():
    mp.dps = 25
    fibre = curve_fibre((0, -1, 0, 1), 2)
    assert len(fibre) == 2
    assert all(isinstance(place, CurvePlace) for place in fibre)
    assert all(place.x == 2 for place in fibre)
    assert mp.almosteq(fibre[0].y, -mp.sqrt(6))
    assert mp.almosteq(fibre[1].y, mp.sqrt(6))
    with pytest.raises(ValueError, match="branch value"):
        curve_fibre((0, -1, 0, 1), 0)
    with pytest.raises(ValueError, match="must be finite"):
        curve_fibre((0, -1, 0, 1), mp.inf)
    assert curve_fibre({(0, 1): 1, (1, 0): -1}, 2) == (
        CurvePlace(mp.mpf(2), mp.mpf(2)),)


def test_curve_path_and_curve_integral():
    mp.dps = 25
    curve = {(0, 2): 1, (1, 0): -1}
    path = curve_path(curve, (1, 1), (4, 2))
    assert mp.almosteq(path.start.y, 1)
    assert mp.almosteq(path.end.y, 2)
    integral = curve_integral(curve, lambda x, y: 1 / y, path)
    assert mp.almosteq(integral.values, 2)
    assert integral.segments >= 1
    assert integral.max_sheet_residual < mp.mpf("1e-22")
    vector = curve_integral(
        curve, (lambda x, y: 1 / y, lambda x, y: y), path)
    assert mp.almosteq(vector.values[0], 2)
    assert mp.almosteq(vector.values[1], mp.mpf(14) / 3)
    reversed_path = curve_path(curve, (4, 2), (1, 1))
    reversed_integral = curve_integral(
        curve, lambda x, y: 1 / y, reversed_path)
    assert mp.almosteq(reversed_integral.values, -2)
    with pytest.raises(ValueError, match="different sheets"):
        curve_path(curve, (1, 1), (4, -2))
    with pytest.raises(ValueError, match="distinct x values"):
        curve_path(curve, (1, 1), (1, -1))
    with pytest.raises(ValueError, match="lie on the curve"):
        curve_path(curve, (1, 1), (4, 3))
    with pytest.raises(ValueError, match="different curve or precision"):
        curve_integral(
            {(0, 2): 1, (1, 0): -4}, lambda x, y: 1 / y, path)
    with mp.workdps(30):
        with pytest.raises(ValueError, match="different curve or precision"):
            curve_integral(curve, lambda x, y: 1 / y, path)


def test_curve_abel_map_hyperelliptic_dispatch():
    mp.dps = 25
    point = (mp.mpf(2), mp.sqrt(6))
    raw = curve_abel_map((0, -1, 0, 1), point)
    assert mp.norm(raw - hyperelliptic_abel_map((0, -1, 0, 1), point)) == 0
    reduced = curve_abel_map((0, -1, 0, 1), point, reduce=True)
    periods = curve_periods((0, -1, 0, 1))
    normalized = (2 * periods.omega) ** -1 * raw
    difference = normalized - (2 * periods.omega) ** -1 * reduced
    assert mp.norm(curve_lattice_reduce(difference, periods.tau).value) < (
        mp.mpf("1e-20"))

    base = (mp.mpf("-0.5"), mp.sqrt(mp.mpf(3) / 8))
    divisor = curve_abel_map(
        (0, -1, 0, 1), [point, point], base_place=base)
    assert mp.norm(
        divisor
        - 2 * curve_abel_map(
            (0, -1, 0, 1), point, base_place=base)) < mp.mpf("1e-20")
    assert mp.norm(curve_abel_map(
        (0, -1, 0, 1), [], base_place=base)) == 0
    shifted = curve_abel_map((0, -1, 0, 1), point, base_place=base)
    shifted_reduced = curve_abel_map(
        (0, -1, 0, 1), point, base_place=base, reduce=True)
    assert mp.norm(
        shifted_reduced
        - curve_lattice_reduce(shifted, periods).value) < mp.mpf("1e-20")


def test_curve_abel_map_hyperelliptic_second_kind_dispatch():
    mp.dps = 25
    coefficients = (0, 4, 0, -5, 0, 1)
    x = mp.mpf(5)
    y = mp.sqrt(mp.fsum(
        coefficient * x**degree
        for degree, coefficient in enumerate(coefficients)))
    actual = curve_abel_map(
        coefficients, (x, y), second_kind=True, reduce=True)
    expected_first, expected_second = hyperelliptic_abel_map(
        coefficients, (x, y), second_kind=True, reduce=True)
    assert isinstance(actual, CurveSecondKindAbelMap)
    assert actual.engine == "hyperelliptic"
    assert actual.marking == "baker"
    assert mp.norm(actual.value - expected_second) < mp.mpf("1e-22")
    reduced_first = curve_abel_map(
        coefficients, (x, y), reduce=True)
    assert mp.norm(reduced_first - expected_first) < mp.mpf("1e-22")

    base_x = mp.mpf(4)
    base_y = mp.sqrt(mp.fsum(
        coefficient * base_x**degree
        for degree, coefficient in enumerate(coefficients)))
    based = curve_abel_map(
        coefficients, (x, y), base_place=(base_x, base_y),
        second_kind=True, reduce=True)
    raw_first, raw_second = hyperelliptic_abel_map(
        coefficients, (x, y), second_kind=True)
    base_first, base_second = hyperelliptic_abel_map(
        coefficients, (base_x, base_y), second_kind=True)
    first_periods = curve_periods(coefficients)
    second_data = curve_periods(coefficients, second_kind=True)
    reduction = curve_lattice_reduce(raw_first - base_first, first_periods)
    second_periods = mp.matrix(2, 4)
    second_periods[:, :2] = 2 * second_data.eta
    second_periods[:, 2:] = 2 * second_data.eta_prime
    expected_based_second = (
        raw_second - base_second
        + second_periods * mp.matrix(reduction.shift))
    assert based.reduction_shift == reduction.shift
    assert mp.norm(based.value - expected_based_second) < mp.mpf("1e-20")


def test_curve_abel_map_general_plane_curve():
    mp.dps = 20
    curve = {(0, 2): 1, (1, 0): 1, (3, 0): -1}
    forms = (lambda x, y: 1 / y,)
    place1 = (mp.mpf(2), mp.sqrt(6))
    place2 = (mp.mpf("-0.5"), mp.sqrt(mp.mpf(3) / 8))
    self_value = curve_abel_map(curve, place1, forms, base_place=place1)
    assert mp.norm(self_value) < mp.mpf("1e-18")
    first = curve_abel_map(curve, place1, forms)
    second = curve_abel_map(curve, place2, forms)
    divisor = curve_abel_map(curve, [place1, place2], forms)
    assert mp.norm(divisor - first - second) < mp.mpf("1e-18")
    shifted = curve_abel_map(curve, place1, forms, base_place=place2)
    assert mp.norm(shifted - (first - second)) < mp.mpf("1e-18")
    empty = curve_abel_map(curve, [], forms)
    assert mp.norm(empty) == 0
    paired = curve_abel_map(
        curve, place1, forms, second_kind=True,
        second_differentials=(lambda x, y: x / y,))
    assert isinstance(paired, CurveSecondKindAbelMap)
    assert paired.engine == "general"
    assert paired.marking == "geometric-polygon"
    based_pair = curve_abel_map(
        curve, place1, forms, base_place=place1, second_kind=True,
        second_differentials=(lambda x, y: x / y,))
    assert mp.norm(based_pair.value) == 0
    curve_place = curve_fibre(curve, 2)[1]
    assert mp.norm(
        curve_abel_map(curve, curve_place, forms) - first) < mp.mpf("1e-18")
    with pytest.raises(ValueError, match="one form per (positive )?genus"):
        curve_abel_map(curve, place1, forms * 2)
    with pytest.raises(ValueError, match="lie on the curve"):
        curve_abel_map(curve, (2, 1), forms)


def test_curve_abel_map_reduce_and_lattice():
    mp.dps = 20
    curve = {(0, 2): 1, (1, 0): 1, (3, 0): -1}
    forms = (lambda x, y: 1 / y,)
    point = (mp.mpf(2), mp.sqrt(6))
    value = curve_abel_map(curve, point, forms)
    reduced = curve_abel_map(curve, point, forms, reduce=True)
    periods = curve_periods(curve, forms)
    reduction = curve_lattice_reduce(value, periods)
    shift_m, shift_n = reduction.shift
    assert isinstance(shift_m, int) and isinstance(shift_n, int)
    lattice_vector = (2 * periods.omega[0, 0] * shift_m
                      + 2 * periods.omega_prime[0, 0] * shift_n)
    assert mp.norm(value - lattice_vector - reduction.value) == 0
    assert mp.norm(value - reduced - lattice_vector) < mp.mpf("1e-18")
    normalized = (2 * periods.omega) ** -1 * value
    normalized_reduction = curve_lattice_reduce(normalized, periods.tau)
    assert normalized_reduction.shift == reduction.shift
    assert mp.norm(
        2 * periods.omega * normalized_reduction.value
        - reduction.value) < mp.mpf("1e-18")


def test_curve_lattice_reduce_exact_lattice():
    mp.dps = 20
    tau = curve_riemann_matrix((0, -1, 0, 1))
    reduction = curve_lattice_reduce(mp.matrix([2 + 1j]), tau)
    assert reduction.shift == (2, 1)
    assert mp.norm(reduction.value) < mp.mpf("1e-18")
    value = mp.matrix([mp.mpf("1.7") - mp.mpf("2.3") * 1j])
    reduction = curve_lattice_reduce(value, tau)
    shift_m, shift_n = reduction.shift
    assert mp.norm(
        value - (shift_m + tau[0, 0] * shift_n) - reduction.value) == 0
    with pytest.raises(ValueError, match="column vector"):
        curve_lattice_reduce(mp.matrix([[1, 2]]), tau)


def test_curve_result_records_are_public():
    assert all(record.__module__ == "mpmath.curves._records"
               for record in (
                   CurveBranchLocus, CurveChart, CurveGenus, CurveHomology,
                   CurveCheck, CurveIntegral, CurveLatticeReduction,
                   CurveMonodromy, CurvePath, CurveFirstKindPeriods,
                   CurveSecondKindPeriods, CurveSecondKindAbelMap, CurvePlace,
                   CurveRiemannConstant, CurveValidation))


def test_critical_values_remove_repeated_resultant_factors():
    mp.dps = 20
    curve = _prepare_plane_curve(mp, {
        (2, 4): 1,
        (3, 2): -4,
        (2, 2): 6,
        (1, 2): -2,
        (2, 0): mp.mpf(27) / 5,
        (1, 0): -mp.mpf(26) / 5,
        (0, 0): 1,
    })
    points, resultant = _plane_curve_critical_values(mp, curve)
    expected = tuple([mp.zero]
                     + mp.polyroots([5, -26, 27])
                     + mp.polyroots([-2, 19, -30, 10]))
    assert len(resultant) - 1 == 18
    assert len(points) == 6
    assert max(min(abs(point - value) for value in expected)
               for point in points) < mp.mpf("1e-17")
    assert max(min(abs(point - reference) for point in points)
               for reference in expected) < mp.mpf("1e-17")


def test_curve_chart_public_surface_and_ownership():
    mp.dps = 25
    curve = {(0, 2): 1, (1, 0): 1, (3, 0): -1}
    chart = curve_chart_monomial(curve, -2, -3)
    assert isinstance(chart, CurveChart)
    assert curve_chart_fibre(chart, 0) == (mp.mpf(-1), mp.mpf(1))
    place = curve_chart_place(curve, chart, 1, mp.mpf("0.05"))
    assert isinstance(place, CurvePlace) and place.chart is not None
    integral = curve_chart_integral(
        chart, lambda x, y: 1 / y, (0, mp.mpf("0.05")), 1)
    assert mp.isfinite(integral.values)
    forms = (lambda x, y: 1 / y,)
    default_constant = curve_riemann_constant(curve, forms)
    chart_constant = curve_riemann_constant(
        curve, forms, base_place=place)
    assert abs(chart_constant.value[0] - default_constant.value[0]) < (
        mp.mpf("1e-22"))

    custom = curve_chart(
        curve, chart.curve.terms,
        lambda t, w: (t**-2, w * t**-3, -2 * t**-3))
    assert curve_chart_fibre(custom, 0) == (mp.mpf(-1), mp.mpf(1))
    invalid = curve_chart(
        curve, chart.curve.terms,
        lambda t, w: (t**-2, w * t**-3 + 1, -2 * t**-3))
    with pytest.raises(ValueError, match="does not parametrize"):
        curve_chart_place(curve, invalid, 1, mp.mpf("0.05"))

    with pytest.raises(ValueError, match="different curve"):
        curve_chart_place(
            {(0, 2): 1, (1, 0): 2, (3, 0): -1},
            chart, 1, mp.mpf("0.05"))
    with mp.workdps(30):
        with pytest.raises(ValueError, match="working precision"):
            curve_chart_fibre(chart, 0)

    assert not hasattr(mpmath, "curve_chart_reciprocal_y")
    assert not hasattr(mpmath, "curve_chart_blow_up")


def test_curve_chart_place_cutoff_stability_and_composition():
    mp.dps = 25

    # A chart at the ramification place (1, 0) of y^2 = x^3 - x.
    elliptic = {(0, 2): 1, (1, 0): 1, (3, 0): -1}
    ramification = curve_chart(
        elliptic,
        {(0, 2): 1, (0, 0): -2, (2, 0): -3, (4, 0): -1},
        lambda t, w: (1 + t**2, t * w, 2 * t))

    # y^2 - x*y - 1 = 0 has one growing and one vanishing place above
    # infinity.  The two monomial charts separate those behaviours.
    rational = {(0, 2): 1, (1, 1): -1, (0, 0): -1}
    growing = curve_chart_monomial(rational, -1, -1)
    vanishing = curve_chart_monomial(rational, -1, 1)

    def cutoff_loop(curve, chart, seed, differential):
        outer = curve_chart_place(curve, chart, seed, mp.mpf("0.08"))
        inner = curve_chart_place(curve, chart, seed, mp.mpf("0.04"))
        path = curve_path(curve, outer, inner)
        return curve_integral(curve, differential, path).values

    assert abs(cutoff_loop(
        elliptic, ramification, mp.sqrt(2),
        lambda x, y: 1 / y)) < mp.mpf("1e-20")
    assert abs(cutoff_loop(
        rational, growing, 1,
        lambda x, y: 1 / (1 + x**2))) < mp.mpf("1e-20")
    assert abs(cutoff_loop(
        rational, vanishing, -1,
        lambda x, y: 1 / (1 + x**2))) < mp.mpf("1e-20")

    first = curve_chart_monomial(elliptic, 2, 1)
    composed = curve_chart_monomial(first, 2, 1)
    direct = curve_chart_monomial(elliptic, 4, 3)
    t = mp.mpf("0.7")
    w = curve_chart_fibre(composed, t)[0]
    assert max(abs(left - right) for left, right in zip(
        composed.coordinate_map(t, w),
        direct.coordinate_map(t, w))) < mp.mpf("1e-23")


def test_kovalevskaya_four_sheet_monodromy_at_zero():
    # This is the first frozen non-hyperelliptic oracle.  Root and loop
    # orderings may differ from Abelfunctions, but x=0 must exchange both
    # pairs of sheets in the degree-four projection.
    mp.dps = 35
    curve = _prepare_plane_curve(mp, {
        (2, 4): 1,
        (3, 2): -4,
        (2, 2): 6,
        (1, 2): -2,
        (2, 0): mp.mpf(27) / 5,
        (1, 0): -mp.mpf(26) / 5,
        (0, 0): 1,
    })
    stem = [-1 + mp.mpf("0.95") * step / 16 for step in range(17)]
    loop = [
        mp.mpf("0.05") * mp.exp(
            1j * (mp.pi + 2 * mp.pi * step / 40))
        for step in range(1, 41)
    ]
    path = stem + loop + list(reversed(stem[:-1]))
    continuation = _continue_plane_curve_sheets_adaptive(mp, curve, path)
    permutation = continuation.permutation
    assert all(permutation[index] != index for index in range(4))
    assert tuple(permutation[permutation[index]]
                 for index in range(4)) == (0, 1, 2, 3)
    assert continuation.max_residual < mp.mpf("1e-32")


def test_kovalevskaya_local_coordinate_charts():
    mp.dps = 35
    curve = _prepare_plane_curve(mp, {
        (2, 4): 1,
        (3, 2): -4,
        (2, 2): 6,
        (1, 2): -2,
        (2, 0): mp.mpf(27) / 5,
        (1, 0): -mp.mpf(26) / 5,
        (0, 0): 1,
    })

    differentials = (
        lambda x, y: 1 / (
            4 * x * y**3 - 8 * x**2 * y + 12 * x * y - 4 * y),
        lambda x, y: 1 / (
            4 * x * y**2 - 8 * x**2 + 12 * x - 4),
        lambda x, y: ((x * y**2 - 1)
                      / (4 * x**2 * y**3 - 8 * x**3 * y
                         + 12 * x**2 * y - 4 * x * y)),
    )

    def check_local_integral(chart, branch, coordinate_map):
        pullbacks = _pullback_plane_curve_differentials(
            differentials, coordinate_map)
        forward = _integrate_plane_curve_branch(
            mp, chart, branch, pullbacks, quadrature_order=16)
        assert all(mp.isfinite(value) for value in forward.values)
        assert forward.max_sheet_residual < mp.mpf("1e-32")
        midpoint = (branch.path[0] + branch.path[-1]) / 2
        first = _continue_plane_curve_branch(
            mp, chart, (branch.path[0], midpoint), branch.values[0])
        second = _continue_plane_curve_branch(
            mp, chart, (midpoint, branch.path[-1]), first.values[-1])
        first_integral = _integrate_plane_curve_branch(
            mp, chart, first, pullbacks, quadrature_order=16)
        second_integral = _integrate_plane_curve_branch(
            mp, chart, second, pullbacks, quadrature_order=16)
        assert all(abs(total - left - right) < mp.mpf("1e-28")
                   for total, left, right in zip(
                       forward.values, first_integral.values,
                       second_integral.values))
        return forward

    # Over x=0 use v=1/y, then x=t^2, v=t*w.  The first chart has
    # double roots w=+/-1; blowing either one up separates the two tangent
    # directions into finite simple roots u=+/-i/sqrt(5).
    reciprocal = _prepare_plane_curve(mp, {(i,4-j):v for i,j,v in curve.terms})
    zero_chart = _monomial_plane_curve_chart(mp, reciprocal, 2, 1)
    for root in (-1, 1):
        assert _evaluate_plane_polynomial(mp, zero_chart, 0, root) == 0
        assert _evaluate_plane_derivative(
            mp, zero_chart, 0, root, "y") == 0
    for center in (-1, 1):
        # Substitute w=center+t*u into the displayed local equation and
        # cancel t^2. The remaining constant term is 4*u^2+4/5.
        from math import comb
        terms = {}
        for i,j,v in zero_chart.terms:
            for k in range(j+1):
                key = (i+k-2,k)
                terms[key] = terms.get(key,0)+v*comb(j,k)*center**(j-k)
        resolved = _prepare_plane_curve(mp, {k:v for k,v in terms.items() if v})
        roots = _finite_plane_curve_sheets(mp, resolved, 0)
        assert len(roots) == 2
        assert all(abs(root**2 + mp.mpf(1) / 5) < mp.mpf("1e-30")
                   for root in roots)
        branch = _continue_plane_curve_branch(
            mp, resolved, (0, mp.mpf("0.05")), roots[0])
        t = branch.path[-1]
        u = branch.values[-1]
        v = t * (center + t * u)
        assert abs(_evaluate_plane_polynomial(
            mp, curve, t**2, 1 / v)) < mp.mpf("1e-27")
        local_integral = check_local_integral(
            resolved, branch,
            lambda t, u, center=center:
            (t**2, 1 / (t * (center + t * u)), 2 * t))
        affine = _lift_plane_curve_path(
            mp, curve, (t**2, mp.mpf("0.01")), 1 / v)
        affine_integral = _integrate_plane_curve_path(
            mp, curve, affine.continuation, differentials,
            sheet=affine.sheet, quadrature_order=16)
        assert all(mp.isfinite(local + ordinary)
                   for local, ordinary in zip(
                       local_integral.values, affine_integral.values))

    # At infinity, x=t^-2.  The y=t^-1*w and y=t*w charts separate the
    # growing and vanishing places respectively.
    infinity_growing = _monomial_plane_curve_chart(mp, curve, -2, -1)
    growing_roots = _finite_plane_curve_sheets(mp, infinity_growing, 0)
    assert any(abs(root - 2) < mp.mpf("1e-30")
               for root in growing_roots)
    assert any(abs(root + 2) < mp.mpf("1e-30")
               for root in growing_roots)
    growing = _continue_plane_curve_branch(
        mp, infinity_growing, (0, mp.mpf("0.05")), 2)
    t = growing.path[-1]
    assert abs(_evaluate_plane_polynomial(
        mp, curve, t**-2, growing.values[-1] / t)) < mp.mpf("1e-22")
    growing_integral = check_local_integral(
        infinity_growing, growing,
        lambda t, w: (t**-2, w / t, -2 * t**-3))
    growing_affine = _lift_plane_curve_path(
        mp, curve, (t**-2, 10), growing.values[-1] / t)
    growing_affine_integral = _integrate_plane_curve_path(
        mp, curve, growing_affine.continuation, differentials,
        sheet=growing_affine.sheet, quadrature_order=16)
    assert all(mp.isfinite(local + ordinary)
               for local, ordinary in zip(
                   growing_integral.values, growing_affine_integral.values))

    infinity_vanishing = _monomial_plane_curve_chart(mp, curve, -2, 1)
    vanishing_roots = _finite_plane_curve_sheets(
        mp, infinity_vanishing, 0)
    expected = mp.sqrt(mp.mpf(27) / 20)
    assert len(vanishing_roots) == 2
    assert any(abs(root - expected) < mp.mpf("1e-30")
               for root in vanishing_roots)
    assert any(abs(root + expected) < mp.mpf("1e-30")
               for root in vanishing_roots)
    vanishing = _continue_plane_curve_branch(
        mp, infinity_vanishing, (0, mp.mpf("0.05")), expected)
    t = vanishing.path[-1]
    assert abs(_evaluate_plane_polynomial(
        mp, curve, t**-2, t * vanishing.values[-1])) < mp.mpf("1e-22")
    vanishing_integral = check_local_integral(
        infinity_vanishing, vanishing,
        lambda t, w: (t**-2, t * w, -2 * t**-3))
    vanishing_affine = _lift_plane_curve_path(
        mp, curve, (t**-2, 10), t * vanishing.values[-1])
    vanishing_affine_integral = _integrate_plane_curve_path(
        mp, curve, vanishing_affine.continuation, differentials,
        sheet=vanishing_affine.sheet, quadrature_order=16)
    assert all(mp.isfinite(local + ordinary)
               for local, ordinary in zip(
                   vanishing_integral.values,
                   vanishing_affine_integral.values))


def test_plane_curve_validation():
    with pytest.raises(ValueError, match="must depend on y"):
        _prepare_plane_curve(mp, {(2, 0): 1, (0, 0): -1})
    with pytest.raises(ValueError, match="nonnegative integers"):
        _prepare_plane_curve(mp, {(-1, 2): 1})

    # F(x,y)=x*y+1 loses projection degree at x=0.
    curve = _prepare_plane_curve(mp, {(1, 1): 1, (0, 0): 1})
    with pytest.raises(ValueError, match="projection degree drops"):
        _ordered_plane_curve_sheets(mp, curve, 0)
