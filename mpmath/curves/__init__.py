"""Numerical algebraic curves.

This module provides tools for computing with smooth plane algebraic curves,
including period matrices, Riemann constants, Abel maps, and integration on
the curve.

The primary interface is the AlgebraicCurve class, which provides lazy
evaluation of the computational pipeline: branch locus, monodromy, genus,
homology, periods, and Riemann constant.

Example:
    >>> from mpmath import mp
    >>> from mpmath.curves import AlgebraicCurve
    >>> mp.dps = 30
    >>> # Fermat cubic x^3 + y^3 = 1
    >>> curve = AlgebraicCurve(mp, {(3, 0): 1, (0, 3): 1, (0, 0): -1})
    >>> curve.genus
    1
    >>> curve.branch_locus.degree
    3
"""

from .algebraic_curve import AlgebraicCurve

# Public namedtuples for result records
from .algebraic_curve import (
    CurveFirstKindPeriods,
    CurveSecondKindPeriods,
    CurveSecondKindAbelMap,
    CurveBranchLocus,
    CurveMonodromy,
    CurveGenus,
    CurveHomology,
    CurveRiemannConstant,
    CurveValidation,
    CurveCheck,
    CurvePlace,
    CurveChart,
    CurvePath,
    CurveIntegral,
    CurveLatticeReduction,
)

__all__ = [
    'AlgebraicCurve',
    'CurveFirstKindPeriods',
    'CurveSecondKindPeriods',
    'CurveSecondKindAbelMap',
    'CurveBranchLocus',
    'CurveMonodromy',
    'CurveGenus',
    'CurveHomology',
    'CurveRiemannConstant',
    'CurveValidation',
    'CurveCheck',
    'CurvePlace',
    'CurveChart',
    'CurvePath',
    'CurveIntegral',
    'CurveLatticeReduction',
]
