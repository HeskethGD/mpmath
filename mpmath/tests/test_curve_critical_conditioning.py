"""Critical values must survive a change of affine x coordinate."""
from math import comb
import pytest
from mpmath import mp
from mpmath.curves.polynomial import _critical_polynomial_roots


@pytest.mark.parametrize('terms,shift', [
    ({(3,1):1, (0,3):1, (1,0):1}, 10),
    ({(0,3):1, (1,2):1, (0,2):4, (2,1):2, (1,1):-5,
      (0,1):-7, (4,0):-1, (3,0):3, (2,0):-6, (1,0):-8, (0,0):9}, 5),
])
@pytest.mark.parametrize('digits', [30, 50])
def test_translated_critical_values(terms, shift, digits):
    from mpmath.curves.polynomial import _plane_curve_critical_values
    ctx = mp.clone()
    ctx.dps = digits
    translated = {}
    for (i,j), c in terms.items():
        for k in range(i+1):
            translated[k,j] = translated.get((k,j),0) + c*comb(i,k)*(-shift)**(i-k)
    original = ctx.algebraic_curve(terms)
    moved = ctx.algebraic_curve(translated)
    roots, _ = _plane_curve_critical_values(ctx, original._prepared)
    shifted, _ = _plane_curve_critical_values(ctx, moved._prepared)
    assert len(roots) == len(shifted) == 8
    for root in roots:
        assert min(abs(other-shift-root) for other in shifted) < 1000*ctx.eps
    assert ctx.dps == digits


def test_critical_roots_reject_bad_solver_output(monkeypatch):
    ctx = mp.clone()
    monkeypatch.setattr(ctx, 'polyroots', lambda *args, **kwargs: [ctx.mpf(2)])
    with pytest.raises(ctx.NoConvergence, match='residual'):
        _critical_polynomial_roots(ctx, [-1, 1])
