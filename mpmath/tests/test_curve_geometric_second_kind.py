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
    curve = ctx.algebraic_curve(TERMS, _general_backend='geometric')

    def radial_forbidden(*args, **kwargs):
        raise AssertionError('radial fallback')

    monkeypatch.setattr(_operations, '_stage_monodromy', radial_forbidden)
    # Exact meromorphic differentials d(x), d(x^2), d(x^3), poles only at infinity.
    forms = (lambda x, y: 1, lambda x, y: 2*x, lambda x, y: 3*x*x)
    periods = curve.second_kind_periods(second_differentials=forms)
    assert periods.marking == 'geometric-polygon'
    assert ctx.norm(periods.eta)+ctx.norm(periods.eta_prime) < ctx.mpf('1e-16')
    assert curve.validate(periods).passed
    first = curve.first_kind_periods()
    assert first.marking == periods.marking
    target = curve.fibre(ctx.mpc('.3', '.7'))[0]
    base = curve.fibre(ctx.mpc('.4', '.8'))[-1]
    result = curve.second_kind_abel_map(target, second_differentials=forms, base_place=base)
    expected = ctx.matrix([target.x**i-base.x**i for i in (1, 2, 3)])
    assert ctx.norm(result.value-expected) < ctx.mpf('1e-16')
    doubled = curve.second_kind_abel_map([target, target], second_differentials=forms, base_place=base)
    assert ctx.norm(doubled.value-2*expected) < ctx.mpf('1e-16')
    assert ctx.norm(curve.second_kind_abel_map([], second_differentials=forms).value) == 0
    reduced = curve.second_kind_abel_map(target, second_differentials=forms, base_place=base, reduce=True)
    shift = curve.lattice_reduce(curve.abel_map(target, base_place=base), first).shift
    assert reduced.reduction_shift == shift
    assert ctx.norm(reduced.value-expected) < ctx.mpf('1e-16')
    with pytest.raises(ValueError, match='one form per genus'):
        curve.second_kind_periods(second_differentials=forms[:1])
    with pytest.raises(ValueError, match='one form per genus'):
        curve.second_kind_abel_map(target, second_differentials=forms[:1])
    chart = curve.monomial_chart(-3, -4)
    infinity = curve.chart_place(chart, 1, ctx.mpf('.3'))
    with pytest.raises(NotImplementedError, match='pole-aware'):
        curve.second_kind_abel_map(infinity, second_differentials=forms)
    with pytest.raises(NotImplementedError, match='pole-aware'):
        curve.second_kind_abel_map(target, second_differentials=forms, base_place=infinity)


def test_geometric_second_kind_periods_match_radial_and_shared_reduction():
    ctx = mp.clone()
    ctx.dps = 18
    geometric = ctx.algebraic_curve(TERMS, _general_backend='geometric')
    radial = ctx.algebraic_curve(TERMS)
    # These forms have zero residues, with their poles confined to infinity.
    forms = (lambda x, y: x*x/(3*y*y), lambda x, y: x/(3*y),
             lambda x, y: x**3/(3*y*y))
    pnew, pold = geometric.first_kind_periods(), radial.first_kind_periods()
    snew = geometric.second_kind_periods(second_differentials=forms)
    sold = radial.second_kind_periods(second_differentials=forms)
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
    first = geometric.abel_map([target]*3, base_place=base)
    value = geometric.second_kind_abel_map([target]*3, second_differentials=forms, base_place=base)
    reduced = geometric.second_kind_abel_map([target]*3, second_differentials=forms,
                                            base_place=base, reduce=True)
    shift = geometric.lattice_reduce(first, pnew).shift
    assert any(shift)
    assert reduced.reduction_shift == shift
    assert ctx.norm(reduced.value-(value.value-new_second*ctx.matrix(shift))) < ctx.mpf('1e-14')
