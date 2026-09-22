import pytest

from mpmath import (
    algebraic_curve_data, hyperelliptic_abel_map,
    hyperelliptic_periods, mp,
)
from mpmath.functions.algebraic_curve import (
    _assemble_plane_curve_periods,
    _blow_up_plane_curve_y,
    _close_monodromy_lift,
    _concatenate_plane_curve_continuations,
    _continue_plane_curve_branch,
    _continue_plane_curve_sheets,
    _evaluate_plane_derivative,
    _evaluate_plane_polynomial,
    _finite_plane_curve_sheets,
    _integrate_plane_curve_path,
    _integrate_plane_curve_branch,
    _integrate_lifted_path_chain,
    _lift_plane_curve_path,
    _lifted_monodromy_graph,
    _lifted_path_chain_boundary,
    _numerical_graph_cycles,
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
    _pullback_plane_curve_differentials,
    _reverse_plane_curve_branch,
    _reverse_plane_curve_continuation,
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
    graph = _lifted_monodromy_graph(
        (forward.permutation, reverse.permutation))
    assert graph.genus == 0
    assert len(graph.cycles) == 2
    assert graph.boundary_components == 3
    assert graph.intersection_rank == 0
    reduction = _symplectic_reduce_intersection(graph.intersection)
    assert reduction.genus == 0
    assert reduction.radical_rank == 2
    assert reduction.form == _canonical_intersection_form(0, 2)


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
        monodromy.permutations + (monodromy.infinity_permutation,))
    assert len(graph.vertices) == 6
    assert len(graph.edges) == 8
    assert len(graph.cycles) == 3
    assert graph.boundary_components == 2
    assert graph.intersection_rank == 2
    assert abs(graph.intersection[0][1]) == 1
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
    periods = _assemble_plane_curve_periods(
        mp, curve, canonical_chains, (lambda x, y: 1 / y,), 1,
        quadrature_order=16)
    assert periods.symmetry_residual == 0
    assert periods.imaginary_eigenvalues[0] > 0
    assert periods.max_sheet_residual < mp.mpf("1e-25")


def test_algebraic_curve_data_general_plane_curve():
    mp.dps = 25
    curve = {
        (0, 2): 1,
        (3, 0): -1,
        (2, 0): 3,
        (1, 0): -3 - 1j,
    }
    data = algebraic_curve_data(
        curve=curve,
        differentials_kind_1=(lambda x, y: 1 / y,),
    )
    assert data.genus == 1
    assert data.characteristic == ((mp.mpf("0.5"),), (mp.mpf("0.5"),))
    assert data.base_place is not None
    assert len(data.diagnostics.branch_points) == 3
    assert data.diagnostics.graph.intersection_rank == 2
    assert data.diagnostics.symmetry_residual == 0
    assert data.diagnostics.imaginary_eigenvalues[0] > 0


def test_algebraic_curve_data_hyperelliptic_dispatch():
    mp.dps = 25
    data = algebraic_curve_data(curve=(0, -1, 0, 1))
    expected = hyperelliptic_periods((0, -1, 0, 1), second_kind=True)
    assert data.genus == 1
    assert data.diagnostics is None
    assert data.base_place is None
    assert data.characteristic == ((mp.mpf("0.5"),), (mp.mpf("0.5"),))
    for actual, reference in zip(
            (data.omega, data.omega_prime, data.eta, data.eta_prime,
             data.tau, data.kappa), expected):
        assert mp.norm(actual - reference) < mp.mpf("1e-22")


def test_algebraic_curve_data_finite_base_place_shift():
    mp.dps = 20
    coefficients = (1, -1, 0, 0, 0, 1)
    natural = algebraic_curve_data(curve=coefficients)
    shifted = algebraic_curve_data(
        curve=coefficients, base_place=(0, 1))
    abel = hyperelliptic_abel_map(coefficients, (0, 1))
    normalised_abel = (2 * natural.omega) ** -1 * abel

    def characteristic_point(data):
        a, b = data.characteristic
        return mp.matrix([
            b[row] + sum(data.tau[row, column] * a[column]
                         for column in range(data.genus))
            for row in range(data.genus)
        ])

    difference = (
        characteristic_point(shifted)
        - characteristic_point(natural) - normalised_abel)
    lattice = mp.matrix([
        [mp.re(1 if row == column else 0)
         for column in range(2)]
        + [mp.re(natural.tau[row, column]) for column in range(2)]
        for row in range(2)
    ] + [
        [mp.im(1 if row == column else 0)
         for column in range(2)]
        + [mp.im(natural.tau[row, column]) for column in range(2)]
        for row in range(2)
    ])
    coordinates = lattice ** -1 * mp.matrix(
        [mp.re(value) for value in difference]
        + [mp.im(value) for value in difference])
    assert max(abs(value - mp.nint(value))
               for value in coordinates) < mp.mpf("1e-17")
    assert shifted.base_place == (0, 1)


def test_critical_values_remove_repeated_resultant_factors():
    mp.dps = 30
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
    assert max(min(abs(point - reference) for point in points)
               for reference in expected) < mp.mpf("1e-18")


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

    graph = _lifted_monodromy_graph(permutations)
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
        monodromy.permutations + (monodromy.infinity_permutation,))
    assert len(graph.cycles) == 9
    assert graph.boundary_components == 4
    assert graph.intersection_rank == 6
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
