"""Nonmonic projection regressions for the geometric backend."""
import pytest
from mpmath import mp


def _full(ctx, data):
    return ctx.matrix([list(a)+list(b) for a, b in
                       zip((2*data.omega).tolist(), (2*data.omega_prime).tolist())])


def _same_lattice(ctx, new, old):
    def realify(matrix):
        return ctx.matrix([[ctx.re(z) for z in row] for row in matrix.tolist()] +
                          [[ctx.im(z) for z in row] for row in matrix.tolist()])
    change = (realify(new)**-1*realify(old)).apply(ctx.nint)
    genus = new.rows
    J = ctx.matrix(2*genus)
    for i in range(genus):
        J[i, i+genus], J[i+genus, i] = 1, -1
    assert change.T*J*change == J
    assert ctx.norm(new*change-old) < ctx.mpf('1e-14')


def test_geometric_nonmonic_elliptic_matches_monic_model():
    ctx = mp.clone()
    ctx.dps = 18
    # w=x*y identifies x*y^2=x^2+1 with w^2=x^3+x. The automatic
    # form dx/(2*x*y) becomes dx/(2*w); x=0 is a projection singular value.
    terms = {(1, 2): 1, (2, 0): -1, (0, 0): -1}
    curve = ctx.algebraic_curve(terms)
    periods = curve.periods_kind_1()
    assert curve.genus_data == (1, 2, 4)
    assert curve.validate(periods).passed
    monic = ctx.algebraic_curve({(0, 2): 1, (3, 0): -1, (1, 0): -1})
    reference = monic.periods_kind_1((lambda x, w: 1/(2*w),))
    _same_lattice(ctx, _full(ctx, periods), _full(ctx, reference))


def test_geometric_kovalevskaya_supplied_basis_converges_with_precision():
    ctx = mp.clone()
    ctx.dps = 18
    terms = {(2, 4): 1, (3, 2): -4, (2, 2): 6, (1, 2): -2,
             (2, 0): ctx.mpf(27)/5, (1, 0): -ctx.mpf(26)/5, (0, 0): 1}
    forms = (lambda x, y: 1/(4*x*y**3-8*x*x*y+12*x*y-4*y),
             lambda x, y: 1/(4*x*y*y-8*x*x+12*x-4),
             lambda x, y: (x*y*y-1)/(4*x*x*y**3-8*x**3*y+12*x*x*y-4*x*y))
    records = []
    for digits in (18, 25):
        ctx.dps = digits
        curve = ctx.algebraic_curve(terms)
        periods = curve.periods_kind_1(forms)
        assert curve.genus_data == (3, 4, 12)
        assert curve.validate(periods).passed
        records.append(_full(ctx, periods))
        # The automatic Newton basis restriction predates the new backend.
        with pytest.raises(ValueError, match='degenerate edge'):
            curve.periods_kind_1()
    _same_lattice(ctx, *records)
