Algebraic curves
----------------

Numerical pipeline for smooth plane algebraic curves, computing their
topology and periods from the defining polynomial.  Given a plane curve,
mpmath determines its branch locus, monodromy, genus, a canonical homology
basis, and the period matrices of its holomorphic differentials.  The
calculations are purely numerical: arbitrary-precision continuation of the
curve sheets supplies the monodromy, a lifted ribbon graph supplies the
homology, and validated quadrature supplies the periods.

Each stage is a separate function returning a small fixed-field record, and
expensive earlier stages are cached: a user requesting the periods of a
curve whose monodromy has already been computed pays only for the new
quadrature.  Topology and period result records can be checked independently
with :func:`~mpmath.curve_validate`.

Curves are supplied as an ascending coefficient sequence defining
:math:`y^2 = P(x)`, as accepted by :func:`~mpmath.hyperelliptic_periods`, as
a sparse mapping from :math:`(i, j)` power pairs to coefficients of
:math:`x^i y^j`, or as a sequence of :math:`(i, j, c)` terms.  Hyperelliptic
input is dispatched to the specialized engine when no differential basis
is supplied; a general plane curve requires a user-supplied holomorphic
basis of one differential callable per genus.  Singular curves and
projections with repeated critical values are not supported.
The current API always uses projection onto the ``x`` coordinate; selecting
an alternative linear projection is future work.

Places over a finite regular value are labelled by
:func:`~mpmath.curve_fibre`, connected by lifted paths and integrated
along by :func:`~mpmath.curve_path` and :func:`~mpmath.curve_integral`,
and mapped into the Jacobian by :func:`~mpmath.curve_abel_map`, with
:func:`~mpmath.curve_lattice_reduce` reducing the result modulo the
period lattice.  Explicit numerical charts extend the same operations to
ramification points and places over infinity.  Charts are bound to their
ambient curve and working precision; discovering charts automatically is
outside the present numerical API.

The record classes ``CurveBranchLocus``, ``CurveMonodromy``, ``CurveGenus``,
``CurveHomology``, ``CurvePeriods``, ``CurveRiemannConstant``, ``CurveChart``,
``CurvePlace``, ``CurvePath``, ``CurveIntegral``, ``CurveLatticeReduction``,
``CurveCheck`` and ``CurveValidation`` are importable from the top-level
``mpmath`` namespace.  They contain results rather than additional methods;
ordinary use starts with the functions below.

These functions complement the hyperelliptic period and Kleinian function
machinery described in :doc:`abelian`: the hyperelliptic engine covers
:math:`y^2 = P(x)` with its automatic differential bases, while the
functions below cover general smooth projections of plane curves.


Branch locus and genus
......................

.. autofunction:: mpmath.curve_branch_locus

.. autofunction:: mpmath.curve_genus


Monodromy
.........

.. autofunction:: mpmath.curve_monodromy


Homology
........

.. autofunction:: mpmath.curve_homology


Periods
.......

.. autofunction:: mpmath.curve_periods

.. autofunction:: mpmath.curve_riemann_matrix

.. autofunction:: mpmath.curve_riemann_constant

``curve_periods`` already accepts a caller-supplied second-kind basis and
returns ``eta``, ``eta_prime`` and ``kappa``.  No separate
``curve_second_kind_periods`` step is required.


Places, paths and integrals
...........................

.. autofunction:: mpmath.curve_fibre

.. autofunction:: mpmath.curve_path

.. autofunction:: mpmath.curve_integral


Explicit local charts
.....................

.. autofunction:: mpmath.curve_chart

.. autofunction:: mpmath.curve_chart_monomial

.. autofunction:: mpmath.curve_chart_fibre

.. autofunction:: mpmath.curve_chart_place

.. autofunction:: mpmath.curve_chart_integral


Abel map and lattice reduction
..............................

.. autofunction:: mpmath.curve_abel_map

.. autofunction:: mpmath.curve_lattice_reduce


Validation
..........

.. autofunction:: mpmath.curve_validate
