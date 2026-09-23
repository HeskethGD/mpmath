import warnings

import mpmath
from mpmath import (
    AlgebraicCurve, CurveBranchLocus, algebraic_curve, mp,
)


def test_unshipped_functional_curve_api_is_not_exported():
    names = (
        "curve_branch_locus", "curve_monodromy", "curve_genus",
        "curve_homology", "curve_periods", "curve_riemann_matrix",
        "curve_riemann_constant", "curve_validate", "curve_fibre",
        "curve_path", "curve_integral", "curve_abel_map",
        "curve_lattice_reduce", "curve_chart", "curve_chart_monomial",
        "curve_chart_fibre", "curve_chart_place", "curve_chart_integral",
    )
    assert all(not hasattr(mpmath, name) for name in names)


def test_algebraic_curve_context_factory_and_explicit_constructor():
    with mp.workdps(20):
        curve = algebraic_curve((0, -1, 0, 1))
        explicit = AlgebraicCurve(mp, (0, -1, 0, 1))
        assert isinstance(curve, AlgebraicCurve)
        assert curve.x_degree == explicit.x_degree == 3
        assert curve.y_degree == explicit.y_degree == 2
        assert curve.genus == explicit.genus == 1
        assert isinstance(curve.branch_locus, CurveBranchLocus)
        assert curve.genus_data.genus == curve.genus


def test_algebraic_curve_path_and_integral_methods():
    with mp.workdps(25):
        curve = mp.algebraic_curve({(0, 2): 1, (1, 0): -1})
        assert tuple(abs(place.y) for place in curve.fibre(4)) == (
            mp.mpf(2), mp.mpf(2))
        path = curve.path((1, 1), (4, 2))
        integral = curve.integral(lambda x, y: 1 / y, path)
        assert mp.almosteq(integral.values, 2)


def test_algebraic_curve_recomputes_after_precision_change_and_warns_once():
    with mp.workdps(15):
        curve = mp.algebraic_curve((0, -1, 0, 1))
        first = curve.branch_locus
    with mp.workdps(30):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            second = curve.branch_locus
            third = curve.branch_locus
        assert len(caught) == 1
        assert second is not first
        assert third is not second
        assert second.branch_values == third.branch_values


def test_algebraic_curve_chart_methods_preserve_curve_ownership():
    with mp.workdps(20):
        curve = mp.algebraic_curve({
            (0, 2): 1,
            (1, 0): 1,
            (3, 0): -1,
        })
        chart = curve.monomial_chart(-2, -3)
        fibre = curve.chart_fibre(chart, 0)
        assert len(fibre) == 2
        place = curve.chart_place(chart, fibre[-1], mp.mpf("0.05"))
        assert place.chart is not None
