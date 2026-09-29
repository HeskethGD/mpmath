"""Permutation and Riemann-Hurwitz checks for curve monodromy."""

import pytest

import mpmath.curves.monodromy as curve_monodromy
from mpmath import mp
from mpmath.curves.monodromy import (
    _compose_permutations, _monodromy_orbit, _permutation_cycles,
    _radial_plane_curve_monodromy, _riemann_hurwitz_genus,
)
from mpmath.curves.polynomial import _prepare_plane_curve


def test_permutation_composition_cycles_and_orbit():
    swap01 = (1, 0, 2)
    swap12 = (0, 2, 1)
    assert _compose_permutations(swap01, swap12) == (1, 2, 0)
    assert _permutation_cycles((1, 2, 0)) == ((0, 1, 2),)
    assert _permutation_cycles(swap01, include_fixed=True) == ((0, 1), (2,))
    assert _monodromy_orbit((swap01, swap12)) == frozenset((0, 1, 2))
    with pytest.raises(ValueError, match="equal sizes"):
        _compose_permutations((0,), swap01)
    with pytest.raises(ValueError, match="invalid permutation"):
        _permutation_cycles((0, 0))


def test_riemann_hurwitz_accepts_sphere_cover_and_rejects_bad_ramification():
    assert _riemann_hurwitz_genus(2, ((1, 0), (1, 0))) == (0, 2)
    assert _riemann_hurwitz_genus(2, ((1, 0),) * 4) == (1, 4)
    with pytest.raises(ValueError, match="valid genus"):
        _riemann_hurwitz_genus(2, ((1, 0),))


@pytest.mark.parametrize("terms,reported_branches,message", [
    ({(0, 2): 1, (1, 0): -1}, (2,), "identity monodromy"),
    ({(0, 2): 1, (2, 0): -1, (1, 0): 1}, (0,),
     "product is not the identity"),
    ({(0, 3): 1, (0, 2): -2, (1, 1): -1, (1, 0): 2}, (0,),
     "not transitive"),
])
def test_radial_monodromy_rejects_incorrect_or_reducible_covers(
        terms, reported_branches, message):
    ctx = mp.clone()
    ctx.dps = 15
    curve = _prepare_plane_curve(ctx, terms)
    with pytest.raises(ValueError, match=message):
        _radial_plane_curve_monodromy(
            ctx, curve, reported_branches, circle_steps=8)


@pytest.mark.parametrize("fault,message", [
    ("finite_open", "radial monodromy paths must be closed"),
    ("infinity_open", "infinity monodromy path must be closed"),
    ("no_tangent", "no outgoing tangent"),
])
def test_radial_monodromy_rejects_inconsistent_continuation_data(
        monkeypatch, fault, message):
    ctx = mp.clone()
    ctx.dps = 15
    curve = _prepare_plane_curve(ctx, {(0, 2): 1, (1, 0): -1})
    continue_sheets = curve_monodromy._continue_plane_curve_sheets_adaptive
    calls = []

    def faulty(*args, **kwargs):
        result = continue_sheets(*args, **kwargs)
        calls.append(result)
        if (fault == "finite_open" and len(calls) == 1
                or fault == "infinity_open" and len(calls) == 2):
            return result._replace(permutation=None)
        if fault == "no_tangent" and len(calls) == 2:
            return result._replace(path=(result.path[0],) * len(result.path))
        return result

    monkeypatch.setattr(
        curve_monodromy, "_continue_plane_curve_sheets_adaptive", faulty)
    with pytest.raises(ValueError, match=message):
        _radial_plane_curve_monodromy(ctx, curve, (0,), circle_steps=8)
