"""Second-kind conventions and finite endpoints on the geometric marking."""
import pytest
from mpmath import mp
from mpmath.curves import _operations

TERMS = {(0, 3): 1, (4, 0): -1, (1, 0): 1, (0, 0): -1}


def full(ctx, left, right):
    return ctx.matrix([list(row1)+list(row2)
                       for row1, row2 in zip(left.tolist(), right.tolist())])


def test_geometric_exact_second_kind_integrals_and_failures(monkeypatch):
    ctx = mp.clone()
    ctx.dps = 18
    curve = ctx.algebraic_curve(TERMS)

    def radial_forbidden(*args, **kwargs):
        raise AssertionError('radial fallback')

    monkeypatch.setattr(_operations, '_stage_monodromy', radial_forbidden)
    # Exact meromorphic differentials d(x), d(x^2), d(x^3), poles only at infinity.
    forms = (lambda x, y: 1, lambda x, y: 2*x, lambda x, y: 3*x*x)
    periods = curve.periods_kind_2(second_differentials=forms)
    assert periods.marking == 'geometric-polygon'
    assert ctx.norm(periods.eta)+ctx.norm(periods.eta_prime) < ctx.mpf('1e-16')
    assert curve.validate(periods).passed
    first = curve.periods_kind_1()
    assert first.marking == periods.marking
    target = curve.fibre(ctx.mpc('.3', '.7'))[0]
    base = curve.fibre(ctx.mpc('.4', '.8'))[-1]
    result = curve.abel_map_kind_2(target, second_differentials=forms, base_place=base)
    expected = ctx.matrix([target.x**i-base.x**i for i in (1, 2, 3)])
    assert ctx.norm(result.value-expected) < ctx.mpf('1e-16')
    assert ctx.norm(curve.abel_map_kind_2([], second_differentials=forms).value) == 0
    reduced = curve.abel_map_kind_2(target, second_differentials=forms, base_place=base, reduce=True)
    shift = curve.lattice_reduce(curve.abel_map_kind_1(target, base_place=base), first).shift
    assert reduced.reduction_shift == shift
    assert ctx.norm(reduced.value-expected) < ctx.mpf('1e-16')
    with pytest.raises(ValueError, match='one form per genus'):
        curve.periods_kind_2(second_differentials=forms[:1])
    with pytest.raises(ValueError, match='one form per genus'):
        curve.abel_map_kind_2(target, second_differentials=forms[:1])
    chart = curve.monomial_chart(-3, -4)
    infinity = curve.chart_place(chart, 1, ctx.mpf('.3'))
    with pytest.raises(ctx.NoConvergence, match='endpoint may be a pole'):
        curve.abel_map_kind_2(infinity, second_differentials=forms)
    with pytest.raises(ctx.NoConvergence, match='endpoint may be a pole'):
        curve.abel_map_kind_2(target, second_differentials=forms, base_place=infinity)


def test_geometric_second_kind_periods_precision_and_shared_reduction():
    ctx = mp.clone()
    ctx.dps = 18
    geometric = ctx.algebraic_curve(TERMS)
    # These forms have zero residues, with their poles confined to infinity.
    forms = (lambda x, y: x*x/(3*y*y), lambda x, y: x/(3*y),
             lambda x, y: x**3/(3*y*y))
    pnew = geometric.periods_kind_1()
    snew = geometric.periods_kind_2(second_differentials=forms)
    with ctx.workdps(25):
        reference = ctx.algebraic_curve(TERMS)
        pold = reference.periods_kind_1()
        sold = reference.periods_kind_2(second_differentials=forms)
    new = full(ctx, 2*pnew.omega, 2*pnew.omega_prime)
    old = full(ctx, 2*pold.omega, 2*pold.omega_prime)

    def realify(matrix):
        return ctx.matrix([[ctx.re(z) for z in row] for row in matrix.tolist()] +
                          [[ctx.im(z) for z in row] for row in matrix.tolist()])

    change = (realify(new)**-1 * realify(old)).apply(ctx.nint)
    J = ctx.matrix(6)
    for i in range(3):
        J[i, i+3], J[i+3, i] = 1, -1
    assert change.T*J*change == J
    assert ctx.norm(new*change-old) < ctx.mpf('1e-14')
    new_second = full(ctx, -2*snew.eta, -2*snew.eta_prime)
    old_second = full(ctx, -2*sold.eta, -2*sold.eta_prime)
    assert ctx.norm(new_second*change-old_second) < ctx.mpf('1e-14')
    raw_kappa = snew.eta*pnew.omega**-1
    assert ctx.norm(snew.kappa-(raw_kappa+raw_kappa.T)/2) < ctx.mpf('1e-16')
    target = geometric.fibre(ctx.mpc('.3', '.7'))[0]
    base = geometric.fibre(ctx.mpc('.4', '.8'))[-1]
    first = geometric.abel_map_kind_1([target]*3, base_place=base)
    value = geometric.abel_map_kind_2([target]*3, second_differentials=forms, base_place=base)
    reduced = geometric.abel_map_kind_2([target]*3, second_differentials=forms,
                                            base_place=base, reduce=True)
    shift = geometric.lattice_reduce(first, pnew).shift
    assert any(shift)
    assert reduced.reduction_shift == shift
    assert ctx.norm(reduced.value-(value.value-new_second*ctx.matrix(shift))) < ctx.mpf('1e-14')


def test_second_kind_regular_chart_endpoints_match_exact_and_higher_precision():
    ctx = mp.clone()
    ctx.dps = 18
    terms = {(0, 3): 1, (4, 0): -1, (0, 0): 1}
    curve = ctx.algebraic_curve(terms)
    branch = curve.chart(
        {(0, 3): 1, (0, 0): -4, (3, 0): -6, (6, 0): -4, (9, 0): -1},
        lambda t, w: (1+t**3, t*w, 3*t**2))
    infinity = curve.monomial_chart(-3, -4)
    base = curve.fibre(ctx.mpc('.3', '.7'))[0]
    for chart, seed, forms, powers, endpoint_x, offset in (
        (branch, ctx.root(4, 3),
         (lambda x, y: 1, lambda x, y: 2*x, lambda x, y: 3*x*x), (1, 2, 3), 1, 0),
        (infinity, ctx.one,
         (lambda x, y: -1/(x-1000j)**2, lambda x, y: -2/(x-1000j)**3,
          lambda x, y: -3/(x-1000j)**4),
         (-1, -2, -3), None, 1000j),
    ):
        values = []
        for cutoff in ('.3', '.25'):
            place = curve.chart_place(chart, seed, ctx.mpf(cutoff))
            value = curve.abel_map_kind_2(place, second_differentials=forms, base_place=base)
            expected = ctx.matrix([(0 if endpoint_x is None else (endpoint_x-offset)**k)-(base.x-offset)**k
                                   for k in powers])
            assert ctx.norm(value.value-expected) < ctx.mpf('1e-14')
            values.append(value.value)
        assert ctx.norm(values[0]-values[1]) < ctx.mpf('1e-14')
        if endpoint_x is None:
            # Independently recompute the singular infinity tail at higher
            # precision; the finite branch already has an exact primitive.
            with ctx.workdps(25):
                reference = ctx.algebraic_curve(terms)
                reference_chart = reference.chart(chart.curve.terms, chart.coordinate_map)
                reference_place = reference.chart_place(
                    reference_chart, ctx.one, ctx.mpf('.25'))
                old = reference.abel_map_kind_2(
                    reference_place, second_differentials=forms, base_place=base)
                assert ctx.norm(old.value-values[-1]) < ctx.mpf('1e-16')
            reduced = curve.abel_map_kind_2(
                place, second_differentials=forms,
                base_place=base, reduce=True)
            assert ctx.norm(reduced.value-values[-1]) < ctx.mpf('1e-14')
            assert reduced.reduction_shift == curve.lattice_reduce(
                curve.abel_map_kind_1(place, base_place=base), curve.periods_kind_1()).shift
        else:
            reverse = curve.abel_map_kind_2(
                base, second_differentials=forms, base_place=place)
            assert ctx.norm(reverse.value+values[-1]) < ctx.mpf('1e-14')
