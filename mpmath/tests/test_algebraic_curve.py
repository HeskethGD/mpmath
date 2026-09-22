import pytest

import mpmath
from mpmath import (
    CurveBranchLocus, CurveChart, CurveCheck, CurveGenus, CurveHomology,
    CurveIntegral, CurveLatticeReduction, CurveMonodromy, CurvePath,
    CurvePeriods, CurvePlace, CurveRiemannConstant, CurveValidation,
    curve_abel_map, curve_branch_locus, curve_fibre, curve_genus,
    curve_chart, curve_chart_fibre, curve_chart_integral,
    curve_chart_monomial, curve_chart_place, curve_homology, curve_integral,
    curve_lattice_reduce, curve_monodromy, curve_path, curve_periods,
    curve_riemann_constant, curve_riemann_matrix, curve_validate,
    hyperelliptic_abel_map, hyperelliptic_periods, mp,
)
from mpmath.functions.algebraic_curve import (
    _assemble_plane_curve_periods,
    _align_closed_continuation_base_fibre,
    _blow_up_plane_curve_y,
    _brahana_canonical_words,
    _canonical_polygon_riemann_constant,
    _canonical_ribbon_polygon,
    _close_monodromy_lift,
    _concatenate_iterated_path_integrals,
    _concatenate_plane_curve_continuations,
    _continue_plane_curve_branch,
    _continue_plane_curve_sheets,
    _evaluate_plane_derivative,
    _evaluate_plane_polynomial,
    _finite_plane_curve_sheets,
    _integrate_plane_curve_path,
    _integrate_plane_curve_path_iterated,
    _integrate_plane_curve_branch,
    _integrate_lifted_path_chain,
    _lift_plane_curve_path,
    _lifted_monodromy_graph,
    _lifted_path_chain_boundary,
    _numerical_graph_cycles,
    _numerical_canonical_polygon,
    _numerical_ordered_canonical_polygon,
    _numerical_ordered_graph_cycles,
    _ordered_plane_curve_sheets,
    _ordered_monodromy_graph,
    _monomial_plane_curve_chart,
    _monodromy_graph_permutations,
    _plane_polynomial_y_coefficients,
    _plane_curve_critical_values,
    _permutation_cycles,
    _prepare_plane_curve,
    _prepare_lifted_path_chain,
    _real_branch_loop_path,
    _real_plane_curve_monodromy,
    _radial_plane_curve_monodromy,
    _reciprocal_y_plane_curve,
    _ribbon_tree_cotree_cut_system,
    _pullback_plane_curve_differentials,
    _reverse_plane_curve_branch,
    _reverse_plane_curve_continuation,
    _reverse_iterated_path_integrals,
    _symplectic_reduce_intersection,
    _transform_lifted_path_chains,
)


def _canonical_intersection_form(genus, radical_rank):
    size = 2 * genus + radical_rank
    return tuple(tuple(
        1 if row < genus and column == genus + row
        else -1 if column < genus and row == genus + column
        else 0
        for column in range(size)) for row in range(size))


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
    continuation = _continue_plane_curve_sheets(mp, curve, path)
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

    interval = _continue_plane_curve_sheets(mp, curve, (1, 4))
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
    first_half = _continue_plane_curve_sheets(mp, curve, circle[:9])
    second_half = _continue_plane_curve_sheets(mp, curve, circle[8:])
    lifted_circle = _concatenate_plane_curve_continuations(
        mp, first_half, second_half)
    assert lifted_circle.permutation == (1, 0)
    circle_integrals = _integrate_plane_curve_path(
        mp, curve, lifted_circle,
        (lambda x, y: 1 / y, lambda x, y: y), sheet=1)
    assert abs(circle_integrals.values[0] + 4) < mp.mpf("1e-27")
    assert abs(circle_integrals.values[1] + mp.mpf(4) / 3) < mp.mpf("1e-27")
    assert circle_integrals.max_sheet_residual < mp.mpf("1e-28")

    reversed_circle = _reverse_plane_curve_continuation(mp, lifted_circle)
    reverse_integrals = _integrate_plane_curve_path(
        mp, curve, reversed_circle,
        (lambda x, y: 1 / y, lambda x, y: y), sheet=1)
    for forward, reverse in zip(
            circle_integrals.values, reverse_integrals.values):
        assert abs(forward + reverse) < mp.mpf("1e-27")

    round_trip = _concatenate_plane_curve_continuations(
        mp, lifted_circle, reversed_circle)
    assert round_trip.permutation == (0, 1)

    open_lift = _prepare_lifted_path_chain(((1, lifted_circle, 1),))
    boundary = _lifted_path_chain_boundary(mp, open_lift)
    assert sorted(place.multiplicity for place in boundary) == [-1, 1]
    cancelling_chain = _prepare_lifted_path_chain((
        (2, lifted_circle, 1),
        (2, reversed_circle, 1),
    ))
    assert not _lifted_path_chain_boundary(mp, cancelling_chain)

    closed_lift = _close_monodromy_lift(mp, lifted_circle, 1)
    closed_chain = _prepare_lifted_path_chain(((1, closed_lift, 1),))
    assert closed_lift.permutation == (0, 1)
    assert not _lifted_path_chain_boundary(mp, closed_chain)
    closed_integral = _integrate_lifted_path_chain(
        mp, curve, closed_chain, (lambda x, y: 1 / x,))
    assert abs(closed_integral.values[0] - 4j * mp.pi) < mp.mpf("1e-27")


def test_reverse_continuation_inverts_three_sheet_monodromy():
    mp.dps = 30
    curve = _prepare_plane_curve(mp, {(0, 3): 1, (1, 0): -1})
    circle = tuple(mp.exp(2j * mp.pi * step / 24)
                   for step in range(25))
    forward = _continue_plane_curve_sheets(mp, curve, circle)
    reverse = _reverse_plane_curve_continuation(mp, forward)
    assert tuple(len(cycle) for cycle in _permutation_cycles(
        forward.permutation)) == (3,)
    assert reverse.permutation == tuple(
        forward.permutation.index(index) for index in range(3))
    closed = _close_monodromy_lift(mp, forward, 0)
    assert closed.permutation == (0, 1, 2)
    chain = _prepare_lifted_path_chain(((1, closed, 0),))
    assert not _lifted_path_chain_boundary(mp, chain)
    # These inverse permutations describe positive local loops around the two
    # branch places of the sphere, not a path and its negative orientation.
    graph = _lifted_monodromy_graph(
        (forward.permutation, reverse.permutation), (1, 1))
    assert graph.genus == 0
    assert len(graph.cycles) == 2
    assert graph.boundary_components == 3
    assert graph.intersection_rank == 0
    reduction = _symplectic_reduce_intersection(graph.intersection)
    assert reduction.genus == 0
    assert reduction.radical_rank == 2
    assert reduction.form == _canonical_intersection_form(0, 2)

    forms = (lambda x, y: 1 / y, lambda x, y: x / y)
    forward_integral = _integrate_plane_curve_path_iterated(
        mp, curve, forward, forms, sheet=0, quadrature_order=12)
    reverse_integral = _integrate_plane_curve_path_iterated(
        mp, curve, reverse, forms, sheet=0,
        quadrature_order=12)
    algebraic_reverse = _reverse_iterated_path_integrals(
        mp, forward_integral)
    assert max(abs(left - right) for left, right in zip(
        reverse_integral.values,
        algebraic_reverse.values)) < mp.mpf("1e-28")
    assert max(abs(reverse_integral.iterated[row][column]
                   - algebraic_reverse.iterated[row][column])
               for row in range(2) for column in range(2)) < mp.mpf("1e-27")
    closed_integral = _concatenate_iterated_path_integrals(
        mp, forward_integral, reverse_integral)
    assert max(abs(value) for value in closed_integral.values) < mp.mpf(
        "1e-28")
    assert max(abs(closed_integral.iterated[row][column])
               for row in range(2) for column in range(2)) < mp.mpf("1e-25")

    aligned_reverse = _align_closed_continuation_base_fibre(
        mp, reverse, forward.fibres[0])
    assert all(abs(left - right) < mp.mpf("1e-28")
               for left, right in zip(
                   aligned_reverse.fibres[0], forward.fibres[0]))


def test_real_hyperelliptic_monodromy_and_genus():
    # y^2=x(x^2-1) has three finite simple branch values and one at infinity.
    # This exercises the monodromy coordinator independently of the eventual
    # non-hyperelliptic target.
    mp.dps = 35
    curve = _prepare_plane_curve(mp, {
        (0, 2): 1,
        (3, 0): -1,
        (1, 0): 1,
    })
    monodromy = _real_plane_curve_monodromy(
        mp, curve, -2, (-1, 0, 1), circle_steps=16)
    assert all(tuple(len(cycle) for cycle in _permutation_cycles(
        permutation)) == (2,) for permutation in monodromy.permutations)
    assert tuple(len(cycle) for cycle in _permutation_cycles(
        monodromy.infinity_permutation)) == (2,)
    assert monodromy.ramification == 4
    assert monodromy.genus == 1

    graph = _lifted_monodromy_graph(
        monodromy.permutations + (monodromy.infinity_permutation,),
        (1,) * len(monodromy.permutations) + (-1,))
    assert len(graph.vertices) == 6
    assert len(graph.edges) == 8
    assert len(graph.cycles) == 3
    assert graph.boundary_components == 2
    assert graph.intersection_rank == 2
    assert abs(graph.intersection[0][1]) == 1
    cut_system = _ribbon_tree_cotree_cut_system(graph)
    assert len(cut_system.generator_edges) == 2
    assert len(cut_system.boundary_word) == 4
    assert abs(cut_system.intersection[0][1]) == 1
    assert (set(cut_system.tree_edges)
            | set(cut_system.cotree_edges)
            | set(cut_system.generator_edges)) == set(range(len(graph.edges)))
    assert not (set(cut_system.tree_edges) & set(cut_system.cotree_edges))
    assert not (set(cut_system.tree_edges) & set(cut_system.generator_edges))
    assert not (set(cut_system.cotree_edges)
                & set(cut_system.generator_edges))
    polygon = _canonical_ribbon_polygon(graph, cut_system)
    assert len(polygon.a_loops) == len(polygon.b_loops) == 1
    assert polygon.intersection == ((0, 1), (-1, 0))
    reduction = _symplectic_reduce_intersection(graph.intersection)
    assert reduction.genus == 1
    assert reduction.radical_rank == 1
    assert reduction.form == _canonical_intersection_form(1, 1)
    numerical_cycles = _numerical_graph_cycles(mp, graph, monodromy)
    assert len(numerical_cycles.chains) == 3
    canonical_chains = _transform_lifted_path_chains(
        numerical_cycles.chains, reduction.transformation)
    assert all(not _lifted_path_chain_boundary(mp, chain)
               for chain in canonical_chains)
    period_data = _assemble_plane_curve_periods(
        mp, curve, canonical_chains, (lambda x, y: 1 / y,), 1)
    assert period_data.imaginary_eigenvalues[0] > 0
    assert period_data.symmetry_residual == 0
    assert abs(period_data.tau[0, 0] + 1 - 1j) < mp.mpf("1e-28")

    canonical_polygon = _numerical_canonical_polygon(
        mp, graph, monodromy)
    polygon_periods = _assemble_plane_curve_periods(
        mp, curve, canonical_polygon.chains,
        (lambda x, y: 1 / y,), 1)
    assert polygon_periods.imaginary_eigenvalues[0] > 0
    assert polygon_periods.symmetry_residual == 0
    for continuation, chain in zip(
            canonical_polygon.a_continuations
            + canonical_polygon.b_continuations,
            canonical_polygon.chains):
        based_value = _integrate_plane_curve_path(
            mp, curve, continuation, (lambda x, y: 1 / y,), sheet=0)
        homology_value = _integrate_lifted_path_chain(
            mp, curve, chain, (lambda x, y: 1 / y,))
        assert abs(based_value.values[0] - homology_value.values[0]) < (
            mp.mpf("1e-28"))

    riemann_constant, _, _ = _canonical_polygon_riemann_constant(
        mp, curve, canonical_polygon, (lambda x, y: 1 / y,),
        polygon_periods.a_periods, polygon_periods.tau, 12)
    expected = (1 + polygon_periods.tau[0, 0]) / 2
    assert abs(riemann_constant[0] - expected) < mp.mpf("1e-28")

    # The middle lollipop encloses x=0 once.  This also verifies that lifted
    # integration consumes the adaptively refined fibres without retracking
    # the path independently.
    winding = _integrate_plane_curve_path(
        mp, curve, monodromy.continuations[1],
        (lambda x, y: 1 / x,), sheet=0)
    assert abs(winding.values[0] - 2j * mp.pi) < mp.mpf("1e-28")


def test_even_degree_hyperelliptic_graph_has_primitive_periods():
    # Four finite branch values leave two ordinary places over infinity.
    # The graph cycles must nevertheless span the primitive compact homology
    # lattice, rather than an index-four sublattice of it.
    mp.dps = 30
    coefficients = (0, -mp.mpf(4) / 5, mp.mpf(38) / 5, -12, 4)
    curve = _prepare_plane_curve(mp, {
        (0, 2): 1,
        (4, 0): -4,
        (3, 0): 12,
        (2, 0): -mp.mpf(38) / 5,
        (1, 0): mp.mpf(4) / 5,
    })
    branch_points = tuple(mp.polyroots(coefficients))
    monodromy = _real_plane_curve_monodromy(
        mp, curve, -1, branch_points, circle_steps=12)
    assert monodromy.infinity_permutation == (0, 1)

    graph = _lifted_monodromy_graph(
        _monodromy_graph_permutations(monodromy))
    reduction = _symplectic_reduce_intersection(graph.intersection)
    numerical_cycles = _numerical_graph_cycles(mp, graph, monodromy)
    canonical_chains = _transform_lifted_path_chains(
        numerical_cycles.chains, reduction.transformation)
    periods = _assemble_plane_curve_periods(
        mp, curve, canonical_chains, (lambda x, y: 1 / y,), 1,
        quadrature_order=16).periods

    omega, omega_prime, unused_tau = hyperelliptic_periods(
        coefficients, method="real")
    reference = mp.matrix([[2 * omega[0, 0], 2 * omega_prime[0, 0]]])
    numerical_change = mp.matrix([
        [mp.re(periods[0, 0]), mp.re(periods[0, 1])],
        [mp.im(periods[0, 0]), mp.im(periods[0, 1])],
    ]) ** -1 * mp.matrix([
        [mp.re(reference[0, 0]), mp.re(reference[0, 1])],
        [mp.im(reference[0, 0]), mp.im(reference[0, 1])],
    ])
    integer_change = mp.matrix([
        [int(mp.nint(numerical_change[row, column]))
         for column in range(2)] for row in range(2)])
    assert max(abs(value) for value in
               numerical_change - integer_change) < mp.mpf("1e-22")
    assert abs(mp.det(integer_change)) == 1


def test_complex_radial_monodromy_and_periods():
    # A generic complex elliptic curve exercises automatic exterior-base
    # selection, distinct continuation and ribbon orders, and the geometric
    # clockwise generator at infinity.
    mp.dps = 30
    branch_points = (0, 1 + 1j, 2 - 1j)
    curve = _prepare_plane_curve(mp, {
        (0, 2): 1,
        (3, 0): -1,
        (2, 0): 3,
        (1, 0): -3 - 1j,
    })
    monodromy = _radial_plane_curve_monodromy(
        mp, curve, branch_points, circle_steps=12)
    identity = (0, 1)
    product = identity
    for generator in monodromy.product_generators:
        product = tuple(
            generator.permutation[product[index]] for index in range(2))
    assert product == identity
    assert monodromy.product_generators[-1].kind == "infinity"
    assert monodromy.product_generators[-1].permutation == (1, 0)
    assert monodromy.minimum_clearance > 0
    assert monodromy.genus == 1

    graph = _ordered_monodromy_graph(monodromy)
    assert graph.genus == 1
    assert graph.intersection_rank == 2
    reduction = _symplectic_reduce_intersection(graph.intersection)
    numerical = _numerical_ordered_graph_cycles(mp, graph, monodromy)
    canonical_chains = _transform_lifted_path_chains(
        numerical.chains, reduction.transformation)
    radical_integrals = tuple(_integrate_lifted_path_chain(
        mp, curve, chain, (lambda x, y: 1 / y,),
        quadrature_order=16).values[0]
        for chain in canonical_chains[2:])
    # This is cancellation between separately quadrature-integrated graph
    # cycles.  The wrong ribbon order gives an O(1) value; the correct order
    # is zero up to the fixed-order quadrature error.
    assert max(map(abs, radical_integrals), default=mp.zero) < mp.mpf(
        "1e-12")
    periods = _assemble_plane_curve_periods(
        mp, curve, canonical_chains, (lambda x, y: 1 / y,), 1,
        quadrature_order=16)
    assert periods.symmetry_residual == 0
    assert periods.imaginary_eigenvalues[0] > 0
    assert periods.max_sheet_residual < mp.mpf("1e-25")

    polygon = _numerical_ordered_canonical_polygon(mp, graph, monodromy)
    polygon_periods = _assemble_plane_curve_periods(
        mp, curve, polygon.chains, (lambda x, y: 1 / y,), 1,
        quadrature_order=16)
    assert polygon_periods.imaginary_eigenvalues[0] > 0


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
    graph = _ordered_monodromy_graph(monodromy)
    assert monodromy.genus == graph.genus == 3
    assert graph.boundary_components == curve.y_degree == 3
    assert graph.intersection_rank == 2 * graph.genus


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
    assert homology.cycle_count == 3
    assert homology.boundary_components == 2
    assert homology.intersection_rank == 2
    assert homology.radical_rank == 1
    assert homology.intersection_form == ((0, 1, 0), (-1, 0, 0), (0, 0, 0))
    assert all(isinstance(entry, int)
               for row in homology.transformation for entry in row)
    assert curve_validate(homology).passed


def test_curve_periods_hyperelliptic_dispatch():
    mp.dps = 25
    data = curve_periods((0, -1, 0, 1))
    expected = hyperelliptic_periods((0, -1, 0, 1), second_kind=True)
    assert data.genus == 1
    assert data.differentials is None
    assert data.max_sheet_residual is None
    for actual, reference in zip(
            (data.omega, data.omega_prime, data.eta, data.eta_prime,
             data.tau, data.kappa), expected):
        assert mp.norm(actual - reference) < mp.mpf("1e-22")
    assert curve_validate(data).passed
    assert mp.norm(curve_riemann_matrix((0, -1, 0, 1)) - data.tau) == 0
    constant = curve_riemann_constant((0, -1, 0, 1))
    assert isinstance(constant, CurveRiemannConstant)
    expected_characteristic = mpmath.hyperelliptic_data(
        (0, -1, 0, 1))[3]
    assert constant.characteristic == expected_characteristic


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
    assert data.eta is None and data.eta_prime is None
    assert data.kappa is None and data.kappa_symmetry_residual is None
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
    assert validation.kind == "CurvePeriods"
    assert validation.passed
    assert validation.maximum_residual < mp.mpf("1e-23")
    assert not curve_validate(data._replace(
        symmetry_residual=mp.one)).passed
    with pytest.raises(ValueError, match="one form per genus"):
        curve_periods(curve, (lambda x, y: 1 / y,) * 2)
    with pytest.raises(ValueError, match="sequence of callables"):
        curve_periods(curve, None)
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
    stages = mpmath.functions.algebraic_curve
    for stage in (stages._stage_branch_locus, stages._stage_monodromy,
                  stages._stage_monodromy_graph,
                  stages._stage_canonical_cycles,
                  stages._stage_cycle_integrals):
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
    curve_place = curve_fibre(curve, 2)[1]
    assert mp.norm(
        curve_abel_map(curve, curve_place, forms) - first) < mp.mpf("1e-18")
    with pytest.raises(ValueError, match="one form per genus"):
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
    assert all(record.__module__ == "mpmath.functions.algebraic_curve"
               for record in (
                   CurveBranchLocus, CurveChart, CurveGenus, CurveHomology,
                   CurveCheck, CurveIntegral, CurveLatticeReduction,
                   CurveMonodromy,
                   CurvePath, CurvePeriods, CurvePlace,
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


def test_real_trigonal_monodromy_graph_and_periods():
    # y^3=x(x-1) is an equianharmonic genus-one curve.  Its three branch
    # places all have ramification index three, so this checks non-simple
    # ramification and a genuinely non-hyperelliptic projection.
    mp.dps = 30
    curve = _prepare_plane_curve(mp, {
        (0, 3): 1,
        (2, 0): -1,
        (1, 0): 1,
    })
    monodromy = _real_plane_curve_monodromy(
        mp, curve, -1, (0, 1), circle_steps=16)
    permutations = (
        monodromy.permutations + (monodromy.infinity_permutation,))
    assert all(tuple(len(cycle) for cycle in _permutation_cycles(
        permutation)) == (3,) for permutation in permutations)
    assert monodromy.ramification == 6
    assert monodromy.genus == 1

    # The clockwise affine outer loop is positive in z=1/x at infinity.
    graph = _lifted_monodromy_graph(permutations, (1, 1, 1))
    assert len(graph.cycles) == 4
    assert graph.boundary_components == 3
    assert graph.intersection_rank == 2
    reduction = _symplectic_reduce_intersection(graph.intersection)
    assert reduction.genus == 1
    assert reduction.radical_rank == 2
    assert reduction.form == _canonical_intersection_form(1, 2)

    numerical_cycles = _numerical_graph_cycles(mp, graph, monodromy)
    canonical_chains = _transform_lifted_path_chains(
        numerical_cycles.chains, reduction.transformation)
    assert all(not _lifted_path_chain_boundary(mp, chain)
               for chain in canonical_chains)
    periods = _assemble_plane_curve_periods(
        mp, curve, canonical_chains, (lambda x, y: 1 / y**2,), 1,
        quadrature_order=16)
    assert periods.imaginary_eigenvalues[0] > 0
    assert abs(mp.im(periods.tau[0, 0]) - mp.sqrt(3) / 2) < mp.mpf(
        "1e-27")
    assert abs(abs(mp.re(periods.tau[0, 0])) - mp.mpf("0.5")) < mp.mpf(
        "1e-27")
    assert periods.max_sheet_residual < mp.mpf("1e-27")


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
    continuation = _continue_plane_curve_sheets(mp, curve, path)
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
        reverse = _integrate_plane_curve_branch(
            mp, chart, _reverse_plane_curve_branch(branch), pullbacks,
            quadrature_order=16)
        assert all(mp.isfinite(value) for value in forward.values)
        assert all(abs(left + right) < mp.mpf("1e-28")
                   for left, right in zip(forward.values, reverse.values))
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
    reciprocal = _reciprocal_y_plane_curve(mp, curve)
    zero_chart = _monomial_plane_curve_chart(mp, reciprocal, 2, 1)
    for root in (-1, 1):
        assert _evaluate_plane_polynomial(mp, zero_chart, 0, root) == 0
        assert _evaluate_plane_derivative(
            mp, zero_chart, 0, root, "y") == 0
    for center in (-1, 1):
        resolved = _blow_up_plane_curve_y(mp, zero_chart, center)
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


@pytest.mark.parametrize("working_dps,circle_steps", ((25, 8), (40, 24)))
def test_kovalevskaya_finite_monodromy_and_genus(
        working_dps, circle_steps):
    # Derive all finite branch values from the two squarefree resultant
    # factors.  Kovalevskaya is an oracle here; path construction and adaptive
    # continuation receive only a general polynomial and point set.
    mp.dps = working_dps
    curve = _prepare_plane_curve(mp, {
        (2, 4): 1,
        (3, 2): -4,
        (2, 2): 6,
        (1, 2): -2,
        (2, 0): mp.mpf(27) / 5,
        (1, 0): -mp.mpf(26) / 5,
        (0, 0): 1,
    })
    branch_points = tuple(sorted(
        [mp.zero]
        + mp.polyroots([5, -26, 27])
        + mp.polyroots([-2, 19, -30, 10])))
    expected_cycles = ((2, 2), (2, 2), (2,), (2,), (2, 2), (2, 2))

    monodromy = _real_plane_curve_monodromy(
        mp, curve, -1, branch_points, circle_steps=circle_steps)
    for continuation, expected in zip(
            monodromy.continuations, expected_cycles):
        cycle_lengths = tuple(sorted(
            len(cycle) for cycle in _permutation_cycles(
                continuation.permutation)))
        assert cycle_lengths == expected
        assert continuation.max_residual < mp.power(10, 3 - working_dps)
        assert continuation.refinements > 0
    infinity_cycles = tuple(sorted(
        len(cycle) for cycle in _permutation_cycles(
            monodromy.infinity_permutation)))
    assert infinity_cycles == (2, 2)
    assert monodromy.ramification == 12
    assert monodromy.genus == 3

    graph = _lifted_monodromy_graph(
        monodromy.permutations + (monodromy.infinity_permutation,),
        (1,) * len(monodromy.permutations) + (-1,))
    assert len(graph.cycles) == 9
    assert graph.boundary_components == 4
    assert graph.intersection_rank == 6
    cut_system = _ribbon_tree_cotree_cut_system(graph)
    assert len(cut_system.cotree_edges) == 3
    assert len(cut_system.generator_edges) == 6
    assert len(cut_system.boundary_word) == 12
    assert all(sum(edge == generator and orientation == sign
                   for edge, orientation in cut_system.boundary_word) == 1
               for generator in cut_system.generator_edges
               for sign in (-1, 1))
    assert len(cut_system.loops) == 6
    assert all(len(loop) for loop in cut_system.loops)
    assert all(cut_system.intersection[row][column]
               == -cut_system.intersection[column][row]
               for row in range(6) for column in range(6))
    polygon = _canonical_ribbon_polygon(graph, cut_system)
    assert len(polygon.a_loops) == len(polygon.b_loops) == 3
    assert polygon.intersection == _canonical_intersection_form(3, 0)
    numerical_polygon = _numerical_canonical_polygon(
        mp, graph, monodromy)
    assert len(numerical_polygon.chains) == 6
    assert all(not _lifted_path_chain_boundary(mp, chain)
               for chain in numerical_polygon.chains)
    reduction = _symplectic_reduce_intersection(graph.intersection)
    assert reduction.genus == 3
    assert reduction.radical_rank == 3
    assert reduction.form == _canonical_intersection_form(3, 3)
    numerical_cycles = _numerical_graph_cycles(mp, graph, monodromy)
    assert len(numerical_cycles.chains) == 9
    canonical_chains = _transform_lifted_path_chains(
        numerical_cycles.chains, reduction.transformation)
    assert all(not _lifted_path_chain_boundary(mp, chain)
               for chain in canonical_chains)


def test_plane_curve_validation():
    with pytest.raises(ValueError, match="must depend on y"):
        _prepare_plane_curve(mp, {(2, 0): 1, (0, 0): -1})
    with pytest.raises(ValueError, match="nonnegative integers"):
        _prepare_plane_curve(mp, {(-1, 2): 1})

    # F(x,y)=x*y+1 loses projection degree at x=0.
    curve = _prepare_plane_curve(mp, {(1, 1): 1, (0, 0): 1})
    with pytest.raises(ValueError, match="projection degree drops"):
        _ordered_plane_curve_sheets(mp, curve, 0)

    with pytest.raises(ValueError, match="requires real points"):
        _real_branch_loop_path(mp, -1, 1j, (1j,))
