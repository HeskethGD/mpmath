Algebraic curves
----------------

The ``AlgebraicCurve`` class represents a smooth plane algebraic curve and
provides a lazy numerical pipeline for its topology, periods and Jacobian
data. Arbitrary-precision continuation of the sheets supplies monodromy, a
lifted ribbon graph supplies homology, and validated quadrature supplies the
periods.

Create a curve with the active mpmath context::

    >>> from mpmath import algebraic_curve, mp
    >>> mp.dps = 30
    >>> curve = algebraic_curve((0, -1, 0, 1))
    >>> curve.genus
    1
    >>> curve.branch_locus.degree
    2

``mp.algebraic_curve(specification)`` is equivalent. The explicit
``AlgebraicCurve(ctx, specification)`` constructor is available when working
with a custom context.

Curves may be supplied as an ascending coefficient sequence defining
:math:`y^2 = P(x)`, a sparse mapping from :math:`(i,j)` power pairs to the
coefficients of :math:`x^i y^j`, or a sequence of :math:`(i,j,c)` terms.
Hyperelliptic input is dispatched to the specialized engine when no
differential basis is supplied. A general plane curve requires a user-supplied
holomorphic basis containing one differential callable per genus. The current
implementation uses projection onto the ``x`` coordinate.

Expensive stages are computed lazily and cached using the numerical context,
including its precision. If the context changes after construction, the curve
warns once and recomputes numerical stages for the new state. Inexact input
coefficients retain the precision at which they were constructed.

.. autoclass:: mpmath.AlgebraicCurve


Topology
........

.. autoattribute:: mpmath.AlgebraicCurve.branch_locus

.. autoattribute:: mpmath.AlgebraicCurve.monodromy

.. autoattribute:: mpmath.AlgebraicCurve.genus

.. autoattribute:: mpmath.AlgebraicCurve.genus_data

.. autoattribute:: mpmath.AlgebraicCurve.homology


Periods and Riemann data
........................

.. automethod:: mpmath.AlgebraicCurve.periods

.. automethod:: mpmath.AlgebraicCurve.riemann_matrix

.. automethod:: mpmath.AlgebraicCurve.riemann_constant

``periods`` accepts an optional caller-supplied second-kind basis and returns
``eta``, ``eta_prime`` and ``kappa`` with the first-kind period data.


Places, paths and integration
.............................

.. automethod:: mpmath.AlgebraicCurve.fibre

.. automethod:: mpmath.AlgebraicCurve.path

.. automethod:: mpmath.AlgebraicCurve.integral

Places over a finite regular value are labelled by ``fibre``. A ``CurvePath``
is bound to its curve and numerical context, and can be passed to ``integral``
with either one differential or a sequence of differentials.


Explicit local charts
.....................

.. automethod:: mpmath.AlgebraicCurve.chart

.. automethod:: mpmath.AlgebraicCurve.monomial_chart

.. automethod:: mpmath.AlgebraicCurve.chart_fibre

.. automethod:: mpmath.AlgebraicCurve.chart_place

.. automethod:: mpmath.AlgebraicCurve.chart_integral

Explicit charts extend paths and integrals to ramification points and places
over infinity. Charts are bound to their ambient curve and working precision;
automatic chart discovery is outside the present numerical API.


Abel map and lattice reduction
..............................

.. automethod:: mpmath.AlgebraicCurve.abel_map

.. automethod:: mpmath.AlgebraicCurve.lattice_reduce


Validation and result records
.............................

.. automethod:: mpmath.AlgebraicCurve.validate

The record classes ``CurveBranchLocus``, ``CurveMonodromy``, ``CurveGenus``,
``CurveHomology``, ``CurvePeriods``, ``CurveRiemannConstant``, ``CurveChart``,
``CurvePlace``, ``CurvePath``, ``CurveIntegral``, ``CurveLatticeReduction``,
``CurveCheck`` and ``CurveValidation`` are importable from the top-level
``mpmath`` namespace. They are immutable results rather than additional
stateful objects.

The curve class complements the hyperelliptic period and Kleinian-function
machinery described in :doc:`abelian`: the hyperelliptic engine supplies
automatic differential bases for :math:`y^2=P(x)`, while ``AlgebraicCurve``
also supports general smooth plane projections.
