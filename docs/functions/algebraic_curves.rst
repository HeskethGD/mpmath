Algebraic curves
----------------

The ``AlgebraicCurve`` class represents a smooth plane algebraic curve and
provides a lazy numerical pipeline for its topology, periods and Jacobian
data. Arbitrary-precision continuation of the sheets supplies monodromy, a
lifted ribbon graph supplies homology, and precision-aware quadrature supplies
the periods.

Create a curve with the active mpmath context::

    >>> from mpmath import algebraic_curve, mp
    >>> mp.dps = 30
    >>> curve = algebraic_curve({
    ...     (0, 2): 1,
    ...     (1, 0): 1,
    ...     (3, 0): -1,
    ... })
    >>> curve.genus
    1
    >>> curve.branch_locus.degree
    2

``mp.algebraic_curve(specification)`` is equivalent. The explicit
``AlgebraicCurve(ctx, specification)`` constructor is available when working
with a custom context.

The canonical curve specification is a sparse mapping from :math:`(i,j)`
power pairs to the coefficients of :math:`x^i y^j`. Ascending coefficient
sequences defining :math:`y^2=P(x)` and sequences of :math:`(i,j,c)` terms
remain accepted for compatibility, but new code should use the sparse form.

Classification is based on the normalized polynomial rather than its input
syntax. In particular, a constant-leading quadratic equation

.. math::

    A y^2+B(x)y+C(x)=0

is sent to the specialized hyperelliptic engine, when no differential basis
is supplied, after the change of coordinate
:math:`z=y+B(x)/(2A)`. The resulting model is
:math:`z^2=B(x)^2/(4A^2)-C(x)/A`. Its automatic first-kind basis is
:math:`x^k dx/z`, expressed in the original coordinate as
:math:`x^k dx/(y+B(x)/(2A))`. Root separation and smoothness are checked
lazily when the specialized computation is requested.
The transformed polynomial must currently have distinct roots. A repeated
root describes a singular plane model; periods of its normalization and
generalized-Jacobian data are not yet part of this dispatch.

For a general plane curve, the engine selects Baker first-kind differentials
from the interior lattice points of its Newton polygon when the edge and
genus checks pass. The basis consists of
:math:`x^{a-1}y^{b-1}dx/F_y` for interior points :math:`(a,b)`, ordered by
increasing :math:`b` and then :math:`a`. If those checks fail, supply one
holomorphic differential callable per genus. The general engine uses
projection onto the ``x`` coordinate.

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

``homology`` describes the marking used by the curve's default computational
engine. For an automatically classified hyperelliptic curve this is the
compact Baker basis used by its periods and Abel maps. For a general curve it
is the canonical-polygon basis reduced from the lifted monodromy graph. The
``engine`` and ``marking`` fields make the distinction explicit.


Periods and Riemann data
........................

.. automethod:: mpmath.AlgebraicCurve.first_kind_periods

.. automethod:: mpmath.AlgebraicCurve.second_kind_periods

.. automethod:: mpmath.AlgebraicCurve.riemann_matrix

.. automethod:: mpmath.AlgebraicCurve.riemann_constant

``first_kind_periods()`` computes first-kind data only.
``second_kind_periods()`` returns only the second-kind data ``eta``,
``eta_prime`` and ``kappa``. Any compatible first-kind work needed to form
``kappa`` is used internally rather than duplicated in the result.
Automatically classified hyperelliptic curves use the BEL basis; a general
curve instead requires an explicit
``second_differentials`` basis. The arbitrary-genus algebraic second-kind
basis and half-period conventions are equation (1.3) and Lemma 1.1 of
[BEL1997]_.

Automatic differential evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The general Baker basis is retained internally as numerator monomials over
one common :math:`F_y` denominator. All automatic forms are evaluated
together at each lifted quadrature node, sharing that denominator. Supplied
callables retain the fully general one-callable-per-form path.

The corresponding records are ``CurveFirstKindPeriods`` and
``CurveSecondKindPeriods``. Their ``differentials`` field is ``None`` for an
automatic hyperelliptic basis. For the general engine it contains either the
supplied callables or callable adapters for the selected Baker forms. An
automatic adapter exposes its ``numerator`` powers for inspection.

Period quadrature policy
~~~~~~~~~~~~~~~~~~~~~~~~

For general curves, period integration chooses a Gauss--Legendre order for
each continued path segment from its distance to the finite branch values.
This is an accuracy estimate for holomorphic differentials, not a rigorous
quadrature error bound. Poles of supplied meromorphic differentials are not
part of that estimate. Iterated integrals used for Riemann constants use the
same per-segment geometry policy with a stable Legendre-basis integration
matrix.

Automatic hyperelliptic calculations use the deterministic Baker cycle
marking. Supplying a callable first-kind basis is an explicit request for the
general canonical-polygon engine. The returned ``engine`` and ``marking``
fields identify that choice; results carrying different markings must not be
combined without a symplectic basis conversion.

The Baker marking orders branch points lexicographically. In a parameterized
family, roots can exchange this order without colliding, causing the returned
matrices and characteristic to change by a symplectic basis transformation
rather than vary continuously. For real ordered roots, the square-root sheet
is continued from the interval to the right of every branch point. Moving
left across a root multiplies it by :math:`i`; it is therefore incorrect to
choose the positive principal square root independently on every real oval.

Places, paths and integration
.............................

.. automethod:: mpmath.AlgebraicCurve.fibre

.. automethod:: mpmath.AlgebraicCurve.path

.. automethod:: mpmath.AlgebraicCurve.integral

Places over a finite regular value are labelled by ``fibre``. A ``CurvePath``
is bound to its curve and numerical context, and can be passed to ``integral``
with either one differential or a sequence of differentials.

General first-kind Abel maps use the same branch-geometry quadrature policy
as first-kind periods. Supplied second-kind differentials may introduce poles
that are not visible in the curve's branch locus, so their open integrals also
compare successive quadrature orders and fail if working-precision agreement
is not reached.


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

.. automethod:: mpmath.AlgebraicCurve.second_kind_abel_map

.. automethod:: mpmath.AlgebraicCurve.lattice_reduce

For an automatically classified hyperelliptic curve,
``second_kind_abel_map(target)`` returns a ``CurveSecondKindAbelMap`` record
with the second-kind ``value``. The general engine provides the same record
with either automatic or supplied first-kind forms when an explicit
``second_differentials`` basis is supplied. Lattice reduction uses the
compatible first-kind Abel map internally and records the shared cycle shift
as ``reduction_shift``.


Validation and result records
.............................

.. automethod:: mpmath.AlgebraicCurve.validate

The record classes ``CurveBranchLocus``, ``CurveMonodromy``, ``CurveGenus``,
``CurveHomology``, ``CurveFirstKindPeriods``, ``CurveSecondKindPeriods``,
``CurveRiemannConstant``, ``CurveSecondKindAbelMap``, ``CurveChart``,
``CurvePlace``, ``CurvePath``, ``CurveIntegral``,
``CurveLatticeReduction``, ``CurveCheck`` and ``CurveValidation`` are
importable from the top-level
``mpmath`` namespace. They are immutable results rather than additional
stateful objects.

The curve class is the common interface to the hyperelliptic period and
Kleinian-function machinery described in :doc:`abelian` and to the general
plane-curve pipeline. Internally the hyperelliptic engine supplies automatic
differential bases and specialized integration, while the general engine
supports smooth plane projections with automatic Baker differentials or a
caller-supplied basis.
