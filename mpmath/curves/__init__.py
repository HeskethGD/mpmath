"""Numerical algebraic curves.

This module provides tools for computing with smooth plane algebraic curves,
including period matrices, Riemann constants, Abel maps, and integration on
the curve.

The primary interface is the Curve class, which provides lazy
evaluation of the computational pipeline: branch locus, monodromy, genus,
homology, periods, and Riemann constant.

Example:
    >>> from mpmath import mp
    >>> from mpmath.curves import Curve
    >>> mp.dps = 30
    >>> # Fermat cubic x^3 + y^3 = 1
    >>> curve = Curve(mp, {(3, 0): 1, (0, 3): 1, (0, 0): -1})
    >>> curve.genus
    1
    >>> curve.branch_locus.degree
    3
"""

# Public namedtuples for result records
from .algebraic_curve import (
    Curve,
    CurveBranchLocus,
    CurveChart,
    CurveCheck,
    CurveFirstKindPeriods,
    CurveGenus,
    CurveHomology,
    CurveIntegral,
    CurveLatticeReduction,
    CurveMonodromy,
    CurvePath,
    CurvePlace,
    CurveRiemannConstant,
    CurveSecondKindAbelMap,
    CurveSecondKindPeriods,
    CurveValidation,
)

__all__ = [
    'Curve',
    'CurveBranchLocus',
    'CurveChart',
    'CurveCheck',
    'CurveFirstKindPeriods',
    'CurveGenus',
    'CurveHomology',
    'CurveIntegral',
    'CurveLatticeReduction',
    'CurveMonodromy',
    'CurvePath',
    'CurvePlace',
    'CurveRiemannConstant',
    'CurveSecondKindAbelMap',
    'CurveSecondKindPeriods',
    'CurveValidation',
]
