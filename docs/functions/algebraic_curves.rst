Algebraic curves
----------------

The algebraic curve module extends mpmath's genus-1 elliptic function
capabilities to supported smooth plane curves of higher genus. The normalized
period lattice of an elliptic curve is described by one complex ratio
:math:`\tau`; in genus :math:`g` it is described by a :math:`g \times g`
Riemann matrix. This module computes periods, Riemann constants, and Abel maps
used with higher-genus theta and Kleinian functions.

Historically, Riemann showed that an algebraic curve of genus :math:`g` has
:math:`g` linearly independent holomorphic differentials whose period matrix
determines a :math:`g`-dimensional complex torus called the Jacobian. Abel's
theorem states that integrals of these differentials provide coordinates on
the Jacobian. Numerical period and Abel map computations are used in
integrable systems and explicit algebraic geometry.

For a genus-1 model :math:`y^2=P(x)`, an Abel map integrates a differential
such as :math:`dx/y` to give a complex number modulo its periods. In genus
:math:`g`, it integrates :math:`g` differentials to give a vector in
:math:`\mathbb{C}^g` modulo a period lattice. The matrix :math:`\tau`
generalizes the elliptic period ratio; unnormalized periods also depend on
the chosen differential basis.

The following conventions apply throughout this module.

**Baker basis.** For supported hyperelliptic curves, the automatic first-kind
basis consists of monomials :math:`x^k dx/z`. For a general curve, the module
uses interior lattice points of its Newton polygon when its edge and genus
checks pass, following Baker's construction [BakerAbel]_. Otherwise the caller
must supply a holomorphic basis.

**BEL conventions.** The automatic hyperelliptic second-kind basis and
half-period sign conventions follow [BEL1997]_ equation (1.3) and Lemma 1.1.
That arbitrary-genus algebraic basis is constructed compatibly with the
first-kind marking. For the geometric polygon engine, callers supply
``second_differentials``.

**Period matrices.** Full first-kind period matrices are :math:`2\omega` and
:math:`2\omega'`, where :math:`\omega` and :math:`\omega'` are half-period
matrices integrated over :math:`a`- and :math:`b`-cycles respectively. The
normalized Riemann matrix is :math:`\tau=\omega^{-1}\omega'`; the computed
matrix is symmetrized to remove numerical asymmetry.

**Second-kind half-periods.** The sign convention is
:math:`2\eta = -\int_a dr` and :math:`2\eta' = -\int_b dr` for a second-kind
differential :math:`dr`. The returned :math:`\kappa` is the symmetric part of
:math:`\eta \omega^{-1}`. The generalized Legendre relation [BEL1997]_ validates
compatibility of these blocks.

**Riemann constants.** The module uses the additive convention
:math:`\theta(A(D) + K, \tau) = 0`, where :math:`A(D)` is the normalized Abel map
of an effective divisor of degree :math:`g-1`. The characteristic representation
is :math:`K = \tau a + b` modulo the normalized period lattice :math:`[I, \tau]`.
The canonical-dissection formula and change-of-base-place law are discussed
in [DP2011]_; the sign here specifies this module's additive convention.

.. list-table:: Numerical results and coordinate conventions
   :header-rows: 1
   :widths: 24 34 42

   * - Method
     - Return value
     - Coordinates
   * - ``first_kind_periods``
     - ``CurveFirstKindPeriods``: ``omega``, ``omega_prime``, ``tau``
     - Half-periods in the selected differential basis; normalized ``tau``
   * - ``second_kind_periods``
     - ``CurveSecondKindPeriods``: ``eta``, ``eta_prime``, ``kappa``
     - Second-kind half-periods in the compatible cycle marking
   * - ``riemann_matrix``
     - Matrix
     - Normalized period matrix ``tau``
   * - ``riemann_constant``
     - ``CurveRiemannConstant``
     - Normalized Jacobian coordinates, lattice ``[I, tau]``
   * - ``abel_map``
     - Column matrix
     - Integrals in the selected first-kind basis, lattice
       ``[2*omega, 2*omega_prime]``
   * - ``second_kind_abel_map``
     - ``CurveSecondKindAbelMap``
     - Integrals in the selected second-kind basis, with the shared
       first-kind reduction shift when requested
   * - ``integral`` / ``chart_integral``
     - ``CurveIntegral``
     - A scalar for one supplied form, or a tuple for a sequence of forms
   * - ``lattice_reduce``
     - ``CurveLatticeReduction``: ``value``, ``shift``
     - Original basis for a period record; normalized basis for ``tau``

The AlgebraicCurve class
.........................

The ``AlgebraicCurve`` class is the unifying object for all curve computations.
It is bound to an mpmath numerical context and provides properties and methods
for topology, homology, periods, and integration. Construct curves with
``mp.algebraic_curve(specification)`` or the explicit
``AlgebraicCurve(ctx, specification)`` for custom contexts.

.. autoclass:: mpmath.AlgebraicCurve

**Input specification.** The canonical curve specification is a sparse mapping
from :math:`(i,j)` power pairs to the coefficients of :math:`x^i y^j`. Ascending
coefficient sequences defining :math:`y^2=P(x)` and sequences of :math:`(i,j,c)`
terms are also accepted. The polynomial must depend on :math:`y`::

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

The following error occurs when the polynomial is independent of :math:`y`::

    >>> algebraic_curve({(2, 0): 1, (0, 0): -1})
    Traceback (most recent call last):
      ...
    ValueError: the plane curve must depend on y

**Computational engines.** The module has two engines, selected by curve
structure and whether the caller supplies differentials.
Expensive numerical stages are computed lazily and cached by precision. If the
context changes after construction, the curve warns once and recomputes stages
for the new state. Inexact coefficients retain the precision at which they
were constructed; increasing the context precision cannot recover digits
already lost from those inputs.

*Hyperelliptic engine:* Curves of the form :math:`A y^2 + B(x) y + C(x) = 0`
with constant nonzero :math:`A` can use the specialized engine when the
transformed polynomial has distinct roots. The
change of coordinate :math:`z = y + B(x)/(2A)` yields
:math:`z^2 = B(x)^2/(4A^2) - C(x)/A`. The automatic first-kind basis is
:math:`x^k dx/z` for :math:`k = 0, \ldots, g-1`. This engine uses the Baker
marking with deterministic paths between branch values. Root separation is
checked when a specialized computation is requested.

*Geometric polygon engine:* For non-hyperelliptic curves or when custom
differentials are supplied, the module uses ribbon graph lifting and canonical
polygon construction. When no differentials are supplied, it selects a Baker
basis from interior lattice points of the Newton polygon if the edge and genus
checks pass. The basis is :math:`x^{a-1} y^{b-1} dx / F_y`, ordered by
increasing :math:`b` and then :math:`a` for interior points :math:`(a,b)`.

To override either automatic first-kind basis, supply one holomorphic
differential callable :math:`f(x,y)` per genus, giving the coefficient of
:math:`dx`. This selects the geometric polygon engine, even for a recognized
hyperelliptic model. Numerical validation cannot certify that supplied forms
are holomorphic; the caller must check their behaviour on the curve.

**Numerical implementation.** The Baker basis is retained internally as numerator
monomials over a common :math:`F_y` denominator. The geometric polygon engine
chooses Gauss-Legendre quadrature order based on distance to branch values,
following Bernstein ellipse convergence estimates [Trefethen2008]_. This is an
accuracy heuristic, not a rigorous error bound. The Baker marking orders branch
points lexicographically; in parameterized families, root crossings cause period
matrix jumps by symplectic transformations rather than continuous variation.

Topology
........

Topological properties describe the branch locus, monodromy, genus, and homology
of the curve. Homology determines the canonical cycle basis used by period and
Abel map computations.

.. autoattribute:: mpmath.AlgebraicCurve.branch_locus

.. autoattribute:: mpmath.AlgebraicCurve.monodromy

.. autoattribute:: mpmath.AlgebraicCurve.genus

.. autoattribute:: mpmath.AlgebraicCurve.genus_data

.. autoattribute:: mpmath.AlgebraicCurve.homology

The ``homology`` property describes the curve's **default** marking. Its
``engine`` and ``marking`` fields identify either
``("hyperelliptic", "baker")`` or ``("general", "geometric-polygon")``.
Supplying custom differentials selects the geometric polygon marking for that
calculation, even on a curve whose ``homology`` property reports Baker marking.
Check the ``engine`` and ``marking`` fields of the result before combining
coordinates: different markings require an integral symplectic change of
cycles. Abel values must also use a common base place and compatible periods.

The monodromy computation uses radial loops from an automatically selected base
fibre and does not depend on the choice of integration marking. The topological
construction follows [DvH2001]_, [DP2011]_, [Eppstein2003]_, and [Lazarus2001]_;
implementation details such as Voronoi graph construction and numerical guards
are specific to this module.

Periods and Riemann data
.........................

These methods generalize elliptic period computation. A normalized genus-1
lattice has one period ratio :math:`\tau`; in genus :math:`g`, the normalized
matrix :math:`\tau` is :math:`g \times g`, symmetric, and has positive-definite
imaginary part.

First-kind periods arise from integrating holomorphic differentials around the
:math:`2g` homology cycles. These integrals determine the period lattice of the
Jacobian and supply the matrix used by theta functions. Second-kind periods
play a role like the quasi-periods of the genus-1 Weierstrass zeta function.
In genus 1, the Riemann constant is the odd half-period
:math:`(1+\tau)/2`, locating a theta zero. In higher genus, it shifts the
theta-zero set to the Abel images of effective divisors of degree :math:`g-1`.

.. automethod:: mpmath.AlgebraicCurve.first_kind_periods

.. automethod:: mpmath.AlgebraicCurve.second_kind_periods

.. automethod:: mpmath.AlgebraicCurve.riemann_matrix

.. automethod:: mpmath.AlgebraicCurve.riemann_constant

The ``first_kind_periods()`` method computes first-kind data only.
``second_kind_periods()`` returns the second-kind half-periods ``eta``,
``eta_prime``, and ``kappa``. Any first-kind work needed to form ``kappa`` is
computed internally. Hyperelliptic curves use automatic BEL basis construction;
the geometric polygon engine requires an explicit ``second_differentials``
argument.

The canonical-dissection formula and change-of-base-place law are discussed in
[DP2011]_. Level-two iterated integrals used in the geometric calculation follow
Chen's identities [Chen1977]_.

Places, paths and integration
..............................

These methods generalize elliptic integration. In genus 1, a path from one
point to another determines an elliptic integral value. In higher genus,
``integral`` can integrate a supplied sequence of differentials along one
lifted path. It returns a scalar for a single callable, or one value per form
for a sequence; the latter need not contain exactly :math:`g` forms.

.. automethod:: mpmath.AlgebraicCurve.fibre

.. automethod:: mpmath.AlgebraicCurve.path

.. automethod:: mpmath.AlgebraicCurve.integral

Places over a finite regular value are labelled by ``fibre``. A ``CurvePath``
is bound to its curve and numerical context, and can be passed to ``integral``
with one differential or a sequence of differentials.

Explicit local charts
......................

Local charts extend integration to marked places that a regular affine fibre
cannot identify. A ramification point may have finite :math:`(x,y)`
coordinates, but the projection does not separate its local branches there;
places over infinity may not have finite affine coordinates at all. A chart
supplies a local parameter and branch choice for approaching such a place.

.. automethod:: mpmath.AlgebraicCurve.chart

.. automethod:: mpmath.AlgebraicCurve.monomial_chart

.. automethod:: mpmath.AlgebraicCurve.chart_fibre

.. automethod:: mpmath.AlgebraicCurve.chart_place

.. automethod:: mpmath.AlgebraicCurve.chart_integral

A chart replaces a difficult affine endpoint by a local parameter :math:`t`
and a branch coordinate :math:`w`. Its equation must have a simple root in
:math:`w` at :math:`t=0`; that root is the ``seed`` selecting a place. The
``cutoff`` is a nonzero :math:`t` value where the local branch meets an
ordinary finite path. A ``CurvePlace`` returned by ``chart_place`` remembers
the chart tail from :math:`t=0` to that junction.

For example, take :math:`y^2=x^3-x`. Its point at infinity can be reached
with :math:`x=t^{-2}`, :math:`y=t^{-3}w`, giving the regular local equation
:math:`w^2=1-t^4`. The two roots at :math:`t=0` choose branches of this
parameterization; both approach the unique infinity place of this elliptic
curve. Here we select :math:`w=1`::

    >>> from mpmath import algebraic_curve, mp
    >>> mp.dps = 15
    >>> curve = algebraic_curve({(0, 2): 1, (3, 0): -1, (1, 0): 1})
    >>> infinity_chart = curve.monomial_chart(-2, -3)
    >>> [mp.nstr(w, 3) for w in curve.chart_fibre(infinity_chart, 0)]
    ['(-1.0 + 0.0j)', '(1.0 + 0.0j)']
    >>> infinity = curve.chart_place(infinity_chart, 1, mp.mpf('0.1'))
    >>> mp.nstr(infinity.x, 6), mp.nstr(infinity.y, 6)
    ('100.0', '999.95')
    >>> form = (lambda x, y: 1/y,)
    >>> value = curve.abel_map((2, mp.sqrt(6)), form, base_place=infinity)
    >>> mp.nstr(mp.re(value[0]), 8)
    '-1.4538919'

The large ``x`` and ``y`` are only the affine junction; ``infinity`` denotes
the place at :math:`t=0`. Passing only ``(infinity.x, infinity.y)`` to another
method would select the junction instead, losing the chart tail. The supplied
``form`` also selects the geometric engine: an automatic hyperelliptic Abel
map does not accept chart-backed endpoints.

For a curve needing a more general change of coordinates, ``chart`` accepts a
sparse local equation in :math:`(t,w)` and a map returning
:math:`(x,y,dx/dt)`. For example, a resolved chart used at two places over
:math:`x=0` on the Kovalevskaya genus-three curve has
:math:`x=t^2` and :math:`y=1/[t(1+tw)]`. The local equation must separate
the desired :math:`w` values at :math:`t=0`; the caller supplies that equation
and coordinate map.

``chart_integral`` is for a path within a chart, such as a small contour used
to obtain a residue. Its ``seed`` is the value of :math:`w` at the *first*
entry of ``t_path``, whereas ``chart_place`` takes a seed at :math:`t=0`.
The elliptic chart above can also be supplied explicitly. Since
:math:`dx/x=-2\,dt/t`, its integral from :math:`t=0.1` to :math:`t=0.2`
is :math:`-2\log 2`::

    >>> local = curve.chart(
    ...     {(0, 2): 1, (0, 0): -1, (4, 0): 1},
    ...     lambda t, w: (t**-2, t**-3*w, -2*t**-3))
    >>> start = mp.mpf('0.1')
    >>> seed = mp.sqrt(1 - start**4)
    >>> result = curve.chart_integral(
    ...     local, lambda x, y: 1/x, (start, mp.mpf('0.2')), seed)
    >>> mp.nstr(result.values, 8)
    '-1.3862944'

Abel map and lattice reduction
...............................

The Abel map generalizes the elliptic integral to a multidimensional map from
divisors to the Jacobian. For genus :math:`g`, it produces a vector in
:math:`\mathbb{C}^g` that reduces modulo the period lattice. The
``lattice_reduce`` method performs this reduction, analogous to reducing an
elliptic integral modulo its periods. By default, the Abel map starts at
infinity for odd-degree hyperelliptic models, the first ordered finite branch
point for even-degree hyperelliptic models, or sheet zero over the computational
base point for the geometric polygon engine. Set ``base_place`` to choose a
different starting place.

.. automethod:: mpmath.AlgebraicCurve.abel_map

.. automethod:: mpmath.AlgebraicCurve.second_kind_abel_map

.. automethod:: mpmath.AlgebraicCurve.lattice_reduce

Second-kind Abel maps use compatible differential bases determined by the
computational engine. Lattice reduction uses the first-kind Abel map internally
and records the shared cycle shift.

Validation
..........

.. automethod:: mpmath.AlgebraicCurve.validate

Result records
..............

The record classes ``CurveBranchLocus``, ``CurveMonodromy``, ``CurveGenus``,
``CurveHomology``, ``CurveFirstKindPeriods``, ``CurveSecondKindPeriods``,
``CurveRiemannConstant``, ``CurveSecondKindAbelMap``, ``CurveChart``,
``CurvePlace``, ``CurvePath``, ``CurveIntegral``,
``CurveLatticeReduction``, ``CurveCheck`` and ``CurveValidation`` are
importable from the top-level ``mpmath`` namespace. Record fields cannot be
reassigned, but contained matrices are mutable. Returned matrices are
independent of the private cached values.

The ``AlgebraicCurve`` class provides the common interface to higher genus
computations. It serves as the foundation for Riemann theta functions and
Kleinian functions described in :doc:`abelian`, analogous to how elliptic
integrals underlie Jacobi elliptic functions in genus 1.
