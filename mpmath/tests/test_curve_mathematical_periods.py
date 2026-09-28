"""Independent identities for the sole general integration pipeline."""
from itertools import combinations
from math import gcd

import pytest
from mpmath import mp


def _full(ctx, data):
    return ctx.matrix([list(a)+list(b) for a,b in
                       zip((2*data.omega).tolist(), (2*data.omega_prime).tolist())])


@pytest.mark.parametrize('m,n', [(3,4), (4,4)])
def test_fermat_period_lattice_matches_beta_integrals(m, n):
    # On y^m=1-x^n, integrating x^a*y^b dx/F_y from x=0 to 1
    # gives B((a+1)/n,(b+1)/m)/(m*n). Differences between sheets and
    # radial paths to roots of unity give closed cycles. These analytic
    # cycles must generate the FULL period lattice, not a proper sublattice.
    ctx = mp.clone()
    ctx.dps = 25
    curve = ctx.algebraic_curve({(0,m):1, (n,0):1, (0,0):-1})
    data = curve.first_kind_periods()
    periods = _full(ctx, data)
    real = ctx.matrix([[ctx.re(z) for z in row] for row in periods.tolist()]+
                      [[ctx.im(z) for z in row] for row in periods.tolist()])
    inverse = real**-1
    columns = []
    for j in range(1,n):
        for k in range(1,m):
            value = ctx.matrix([
                (1-ctx.exp(2j*ctx.pi*j*(a+1)/n)) *
                (1-ctx.exp(2j*ctx.pi*k*(b+1)/m)) *
                ctx.beta(ctx.mpf(a+1)/n, ctx.mpf(b+1)/m)/(m*n)
                for a,b in (f.numerator for f in data.differentials)])
            coordinates = (inverse*ctx.matrix(
                [ctx.re(z) for z in value]+[ctx.im(z) for z in value])).apply(ctx.nint)
            assert ctx.norm(periods*coordinates-value) < ctx.mpf('1e-22')
            columns.append(list(coordinates))
    # The gcd of maximal integer minors is the sublattice index.
    index = 0
    for selected in combinations(columns, 2*data.genus):
        determinant = ctx.det(ctx.matrix(selected))
        integer = int(ctx.nint(determinant))
        assert abs(determinant-integer) < ctx.mpf('1e-20')
        index = gcd(index, abs(integer))
        if index == 1:
            break
    assert index == 1
    assert curve.validate(data).passed


@pytest.mark.parametrize('degree,genus', [(3,1), (5,2), (6,2)])
def test_monodromy_riemann_hurwitz_for_hyperelliptic_covers(degree, genus):
    ctx = mp.clone()
    ctx.dps = 20
    curve = ctx.algebraic_curve({(0,2):1, (degree,0):-1, (0,0):1})
    data = curve.monodromy
    assert len(data.permutations) == degree
    assert all(p == (1,0) for p in data.permutations)
    assert data.infinity_permutation == ((1,0) if degree%2 else (0,1))
    assert data.ramification == degree + degree%2
    assert data.genus == genus
    assert 2*genus-2 == -4+data.ramification
    assert curve.validate(data).passed


def test_general_mu_period_precision_and_exact_second_kind_forms():
    ctx = mp.clone()
    terms = {(0,3):1, (1,2):1, (0,2):4, (2,1):2, (1,1):-5,
             (0,1):-7, (4,0):-1, (3,0):3, (2,0):-6, (1,0):-8, (0,0):9}
    values = []
    for digits in (20,30):
        ctx.dps = digits
        curve = ctx.algebraic_curve(terms)
        data = curve.first_kind_periods()
        assert curve.validate(data).passed
        values.append(_full(ctx,data))
    assert ctx.norm(values[0]-values[1]) < ctx.mpf('1e-18')
    # d(x), d(x^2), d(x^3) have zero closed periods and known primitives.
    ctx.dps = 18
    curve = ctx.algebraic_curve(terms)
    forms = (lambda x,y:1, lambda x,y:2*x, lambda x,y:3*x*x)
    second = curve.second_kind_periods(second_differentials=forms)
    assert ctx.norm(second.eta)+ctx.norm(second.eta_prime) < ctx.mpf('1e-15')
    base = curve.fibre(ctx.mpc('0.3','0.7'))[0]
    target = curve.fibre(ctx.mpc('0.4','0.8'))[-1]
    result = curve.second_kind_abel_map(target,base_place=base,second_differentials=forms)
    assert ctx.norm(result.value-ctx.matrix([target.x**i-base.x**i for i in (1,2,3)])) < ctx.mpf('1e-15')


def test_genus_two_supplied_periods_match_specialized_lattice():
    ctx = mp.clone()
    ctx.dps = 20
    curve = ctx.algebraic_curve((0,4,0,-5,0,1))
    automatic = _full(ctx,curve.first_kind_periods())
    supplied = _full(ctx,curve.first_kind_periods((lambda x,y:1/y,lambda x,y:x/y)))
    def realify(matrix):
        return ctx.matrix([[ctx.re(z) for z in row] for row in matrix.tolist()]+
                          [[ctx.im(z) for z in row] for row in matrix.tolist()])
    transform = (realify(supplied)**-1*realify(automatic)).apply(ctx.nint)
    J = ctx.matrix([[0,0,1,0],[0,0,0,1],[-1,0,0,0],[0,-1,0,0]])
    assert transform.T*J*transform == J
    assert ctx.norm(supplied*transform-automatic) < ctx.mpf('1e-17')


def test_geometric_iterated_cycles_obey_shuffle_and_reversal():
    from mpmath.curves._stages import _stage_geometric_periods
    from mpmath.curves.integration import _integrate_geometric_loops_iterated
    from mpmath.curves.differentials import _baker_callable
    from mpmath.curves.polynomial import _prepare_plane_curve
    ctx = mp.clone()
    ctx.dps = 18
    terms = {(0,3):1,(4,0):1,(0,0):-1}
    curve = _prepare_plane_curve(ctx,terms)
    data = _stage_geometric_periods(ctx,curve)
    polygon = data.polygon.polygon
    forms = tuple(_baker_callable(ctx,data.basis,i) for i in range(data.genus))
    loop = polygon.a_loops[0]
    reverse = tuple((edge,-sign) for edge,sign in reversed(loop))
    forward,backward,closed = _integrate_geometric_loops_iterated(
        ctx,curve,data.cover,data.graph,polygon,(loop,reverse,loop+reverse),forms)
    for i in range(data.genus):
        assert abs(forward.values[i]+backward.values[i]) < ctx.mpf('1e-15')
        assert abs(closed.values[i]) < ctx.mpf('1e-15')
        for j in range(data.genus):
            # Chen's shuffle and path-reversal identities.
            assert abs(forward.iterated[i][j]+forward.iterated[j][i]-
                       forward.values[i]*forward.values[j]) < ctx.mpf('1e-15')
            assert abs(backward.iterated[i][j]-forward.iterated[j][i]) < ctx.mpf('1e-15')
            assert abs(closed.iterated[i][j]) < ctx.mpf('1e-15')


def test_monodromy_cubic_cover_has_inverse_three_cycles():
    ctx = mp.clone()
    ctx.dps = 20
    curve = ctx.algebraic_curve({(0,3):1, (1,0):-1})
    data = curve.monodromy
    permutation, = data.permutations
    assert all(permutation[i] != i for i in range(3))
    assert tuple(permutation[permutation[permutation[i]]] for i in range(3)) == (0,1,2)
    assert tuple(data.infinity_permutation[permutation[i]] for i in range(3)) == (0,1,2)
    assert data.genus == 0 and data.ramification == 4
    assert curve.validate(data).passed
