"""Object-oriented interface to numerical algebraic curves."""

import warnings

from . import _operations, _records, charts


class Curve:
    """A plane algebraic curve bound to an mpmath numerical context.

    Construct curves with ``mp.algebraic_curve(specification)`` or the
    explicit form ``Curve(ctx, specification)`` for custom contexts.

    The canonical ``specification`` is a sparse ``(x_power, y_power)``
    coefficient mapping. Sequences of ``(x_power, y_power, coefficient)``
    terms and ascending coefficient sequences for ``y**2 = P(x)`` are also
    accepted.

    The module automatically selects a computational engine: hyperelliptic
    curves use specialized Baker marking with efficient branch-based integration;
    other smooth plane curves use geometric polygon marking with automatic or
    user-supplied differential bases. See the documentation for details on
    computational engines and mathematical conventions.

    Supplied first-kind differential callables must be holomorphic on the
    curve; numerical checks cannot certify absence of poles. Supplied
    second-kind forms may have poles but must have zero residues and be
    regular along integration paths. Chart-backed endpoints extend integration
    to ramification points and places at infinity when the integrals converge;
    pole regularization is not provided.
    """

    def __init__(self, ctx, specification):
        self.ctx = ctx
        self._creation_state = _operations._curve_cache_state(ctx)
        self._warned_states = set()
        if hasattr(specification, "items"):
            specification = dict(specification.items())
        else:
            try:
                specification = tuple(specification)
            except TypeError:
                # Let the shared validator provide the public error message.
                pass
        self._specification = specification
        self._prepared, self._hyperelliptic_model = (
            _operations._normalize_algebraic_curve_input(ctx, specification))
        self._classified_states = {
            self._creation_state: _records._ClassifiedCurve(
                self._prepared, self._hyperelliptic_model)
        }
        self._automatic_first_kind_periods = {}

    @staticmethod
    def _copy_first_kind_periods(result):
        """Copy mutable matrices in an automatic first-kind result."""
        return result._replace(
            omega=+result.omega,
            omega_prime=+result.omega_prime,
            tau=+result.tau,
        )

    def _check_precision(self):
        """Warn once for each numerical state different from construction."""
        state = _operations._curve_cache_state(self.ctx)
        if state != self._creation_state and state not in self._warned_states:
            warnings.warn(
                "the Curve context changed after construction; "
                "numerical stages will be recomputed for the current state, "
                "but inexact input coefficients retain their construction "
                "precision",
                UserWarning,
                stacklevel=3,
            )
            self._warned_states.add(state)

    def _call(self, function, *args, **kwargs):
        self._check_precision()
        state = _operations._curve_cache_state(self.ctx)
        classified = self._classified_states.get(state)
        if classified is None:
            prepared, hyperelliptic = (
                _operations._normalize_algebraic_curve_input(
                    self.ctx, self._specification))
            classified = _records._ClassifiedCurve(
                prepared, hyperelliptic)
            self._classified_states[state] = classified
        return function(
            self.ctx, classified, *args, **kwargs)

    @property
    def specification(self):
        """The materialized input specification used to construct the curve."""
        if isinstance(self._specification, dict):
            return dict(self._specification)
        return self._specification

    @property
    def x_degree(self):
        """Degree of the defining polynomial in ``x``."""
        return self._prepared.x_degree

    @property
    def y_degree(self):
        """Degree of the defining polynomial in ``y``."""
        return self._prepared.y_degree

    @property
    def branch_locus(self):
        r"""Return the finite branch locus of a plane algebraic curve.

        The returned ``CurveBranchLocus`` record contains the degree of the
        ``x`` projection, the distinct finite branch values above which the
        projection ramifies, and the ascending coefficients of the
        y-derivative resultant whose roots they are.  Ramification above
        infinity is reported by :attr:`Curve.monodromy` instead,
        because it requires monodromy rather than the resultant alone.

        The lemniscatic curve :math:`y^2 = x^3 - x` has a two-sheeted
        projection with three finite branch values::

            >>> from mpmath import algebraic_curve
            >>> locus = algebraic_curve((0, -1, 0, 1)).branch_locus
            >>> locus.degree
            2
            >>> locus.branch_values
            (mpf('-1.0'), mpf('0.0'), mpf('1.0'))
        """
        return self._call(_operations.branch_locus)

    @property
    def monodromy(self):
        r"""Return the monodromy of a plane algebraic curve over the x-line.

        The curve is continued numerically along guarded radial loops around
        the finite branch values, from an exterior base point chosen
        automatically.  A large outer loop supplies the monodromy at infinity
        geometrically.  The returned ``CurveMonodromy`` record contains the
        computational base point, its ordered fibre, the branch values, the
        counter-clockwise product-ordered local permutations, the permutation
        at infinity, the total ramification, the genus from
        Riemann--Hurwitz, the transitivity and product identities of the
        permutation system, and the minimum geometric clearance of the
        continuation paths.

        The sheet labels refer to the internally selected base fibre.  The
        routing continuations used to compute them are private.

        Each finite branch value of the lemniscatic curve
        :math:`y^2 = x^3 - x` exchanges its two sheets, as does infinity::

            >>> from mpmath import algebraic_curve
            >>> monodromy = algebraic_curve((0, -1, 0, 1)).monodromy
            >>> monodromy.genus
            1
            >>> monodromy.permutations
            ((1, 0), (1, 0), (1, 0))
            >>> monodromy.infinity_permutation
            (1, 0)
        """
        return self._call(_operations.monodromy)

    @property
    def genus(self):
        r"""Genus of the compact curve.

        See :attr:`genus_data` for projection degree and ramification.
        """
        return self._call(_operations.genus_data).genus

    @property
    def genus_data(self):
        r"""Return the genus of a plane algebraic curve.

        Returns a ``CurveGenus`` record with the genus, projection degree, and
        total ramification. The Riemann-Hurwitz balance :math:`2g-2 = -2d+r`
        can be verified directly::

            >>> from mpmath import algebraic_curve
            >>> algebraic_curve((0, -1, 0, 1)).genus_data
            CurveGenus(genus=1, degree=2, ramification=4)
        """
        return self._call(_operations.genus_data)

    @property
    def homology(self):
        r"""Return the homology marking used by the curve's default engine.

        Returns a ``CurveHomology`` record containing ``2*genus`` cycles in a
        canonical symplectic basis with the standard intersection form. The
        ``marking`` field identifies the computational engine: ``'baker'`` for
        hyperelliptic curves, ``'geometric-polygon'`` for others.

        The lemniscatic curve :math:`y^2 = x^3 - x` uses Baker marking::

            >>> from mpmath import algebraic_curve
            >>> homology = algebraic_curve((0, -1, 0, 1)).homology
            >>> homology.genus, homology.marking
            (1, 'baker')
            >>> homology.intersection_form
            ((0, 1), (-1, 0))
        """
        return self._call(_operations.homology)

    def periods_kind_1(self, differentials=None):
        r"""Return first-kind half-periods and the normalized Riemann matrix.

        Generalizes elliptic period computation to genus :math:`g`. Returns a
        ``CurveFirstKindPeriods`` record with half-period matrices ``omega``,
        ``omega_prime``, and the normalized Riemann matrix ``tau`` (the symmetric
        part of ``omega**-1 * omega_prime``). Full periods are ``2*omega`` and
        ``2*omega_prime``.

        With no arguments, uses the automatic differential basis: hyperelliptic
        curves get ``x**k dx/z`` in Baker marking; other curves get a basis from
        Newton polygon interior points when the edge and genus checks pass.
        Otherwise, supply ``differentials``: one holomorphic callable
        ``f(x, y)`` per genus, giving the coefficient of ``dx``. A supplied
        basis selects the geometric-polygon engine, including for a recognized
        hyperelliptic curve.

        A non-positive-definite period matrix raises ``ValueError``.

        The lemniscatic curve has normalized period matrix ``tau = i``::

            >>> from mpmath import algebraic_curve, mp
            >>> mp.dps = 15
            >>> curve = algebraic_curve((0, -1, 0, 1))
            >>> data = curve.periods_kind_1()
            >>> mp.re(data.tau[0, 0]), mp.im(data.tau[0, 0])
            (mpf('0.0'), mpf('1.0'))

        A supplied basis uses geometric-polygon marking::

            >>> data = curve.periods_kind_1((lambda x, y: 1 / y,))
            >>> data.marking
            'geometric-polygon'
            >>> curve.validate(data).passed
            True
        """
        state = _operations._curve_cache_state(self.ctx)
        if differentials is None:
            cached = self._automatic_first_kind_periods.get(state)
            if cached is not None:
                return self._copy_first_kind_periods(cached)
        result = self._call(
            _operations.periods,
            differentials,
        )
        if differentials is None:
            self._automatic_first_kind_periods[state] = (
                self._copy_first_kind_periods(result))
        return result

    def periods_kind_2(self, differentials=None, *,
                            second_differentials=None):
        r"""Return second-kind half-periods and kappa.

        The ``CurveSecondKindPeriods`` record contains ``eta``, ``eta_prime``
        and ``kappa``, with ``2*eta = -integral_a(dr)`` and
        ``2*eta_prime = -integral_b(dr)``. The returned ``kappa`` is the symmetric
        part of ``eta * omega**-1`` for the compatible first-kind half-periods.

        Recognized hyperelliptic models use the automatic BEL basis and Baker
        marking. The general engine requires one callable per genus in
        ``second_differentials``; its first-kind basis may be supplied or
        selected automatically. To override second-kind forms on a recognized
        hyperelliptic model, also supply the first-kind ``differentials``.

        Supplied second-kind forms must have zero residues and be regular on
        the integration paths; numerical convergence checks do not establish
        those properties. ``engine`` and ``marking`` describe the cycle basis.
        With automatic first-kind forms, compatible first-kind results are
        cached for later calls to :meth:`periods_kind_1`.
        """
        result = self._call(
            _operations.periods,
            differentials,
            second_kind=True,
            second_differentials=second_differentials,
            _return_first=True,
        )
        first, second = result
        if differentials is None:
            state = _operations._curve_cache_state(self.ctx)
            self._automatic_first_kind_periods[state] = (
                self._copy_first_kind_periods(first))
        return second

    def riemann_matrix(self, differentials=None):
        r"""Return the normalized Riemann matrix of a plane algebraic curve.

        This is a convenience wrapper returning
        ``curve.periods_kind_1(differentials).tau``; see
        :meth:`Curve.periods_kind_1` for the input conventions.

            >>> from mpmath import algebraic_curve, mp
            >>> mp.dps = 15
            >>> tau = algebraic_curve((0, -1, 0, 1)).riemann_matrix()
            >>> mp.im(tau[0, 0])
            mpf('1.0')
        """
        return self._call(_operations.riemann_matrix, differentials)

    def riemann_constant(self, differentials=None, *, base_place=None):
        r"""Return the vector of Riemann constants for a plane curve.

        The returned ``CurveRiemannConstant`` contains a direct representative of
        the normalized Jacobian vector ``value`` in mpmath's additive convention
        ``theta(A(D) + value, tau) = 0``, its literal ``(a, b)`` coordinates
        ``value = tau*a + b`` modulo the period lattice, the requested
        ``base_place`` (``None`` denotes the engine's natural base), and the
        maximum sheet residual of the direct contour integrations.

        For a general plane curve, ``differentials`` may supply one holomorphic
        differential per genus, in exactly the basis accepted by
        :meth:`Curve.periods_kind_1`. The value is computed directly from the
        certified canonical polygon and level-two contour integrals; theta
        functions and characteristic searches are not used.  ``base_place`` may
        be a regular finite place or a chart-backed place.  Changing the base
        uses ``K_Q = K_P + (g-1) A_P(Q)`` in normalized coordinates.

        Structurally hyperelliptic input without supplied differentials dispatches
        to the Baker-marked specialized periods and characteristic convention.

        In genus one the answer is the odd half-period ``(1+tau)/2``::

            >>> from mpmath import algebraic_curve, mp
            >>> mp.dps = 15
            >>> curve = algebraic_curve({(0, 2): 1, (1, 0): 1, (3, 0): -1})
            >>> forms = (lambda x, y: 1 / y,)
            >>> constant = curve.riemann_constant(forms)
            >>> periods = curve.periods_kind_1(forms)
            >>> mp.almosteq(constant.value[0], (1 + periods.tau[0, 0]) / 2)
            True

        The direct level-two integrations are substantially more expensive than
        ordinary periods, although their cost does not include an exponential
        characteristic enumeration.
        """
        return self._call(
            _operations.riemann_constant,
            differentials,
            base_place=base_place,
        )

    def validate(self, result):
        r"""Validate a result record returned by the curve functions.

        ``result`` is one of ``CurveBranchLocus``, ``CurveMonodromy``,
        ``CurveGenus``, ``CurveHomology``, ``CurveFirstKindPeriods``,
        ``CurveSecondKindPeriods`` or ``CurveRiemannConstant``. The returned
        ``CurveValidation`` record contains one named ``CurveCheck`` per
        invariant, the largest numerical residual among them, and whether
        every check passed.

        Structural checks are recomputed from the record; numerical
        integration residuals use recorded values. Tolerances use the
        current context precision; curve data is not recomputed.

            >>> from mpmath import algebraic_curve
            >>> curve = algebraic_curve((0, -1, 0, 1))
            >>> report = curve.validate(curve.periods_kind_1())
            >>> report.passed
            True
            >>> report.checks[0]
            CurveCheck(name='tau_symmetry_residual', value=mpf('0.0'), passed=True)
        """
        self._check_precision()
        return _operations.validate(self.ctx, result)

    def fibre(self, x):
        r"""Return the labelled fibre of a plane algebraic curve over x.

        ``x`` must be a finite regular value of the ``x`` projection:
        it must not be a branch value or a value over which the projection
        drops degree. The returned tuple contains one ``CurvePlace`` per
        sheet, ordered deterministically by the real and imaginary parts
        of ``y``. Sheet labels refer to this particular fibre. At the
        monodromy base point they agree with the labels used by
        :attr:`Curve.monodromy`.

        >>> from mpmath import algebraic_curve, mp
        >>> mp.dps = 15
        >>> curve = algebraic_curve((0, -1, 0, 1))
        >>> [mp.nstr(place.y, 6) for place in curve.fibre(2)]
        ['-2.44949', '2.44949']
        """
        return self._call(_operations.fibre, x)

    def path(self, start, end):
        r"""Return a lifted path between two regular finite places.

        ``start`` and ``end`` are regular finite places, each given as a
        ``(x, y)`` pair, a ``CurvePlace`` from :meth:`Curve.fibre`, or
        a chart-backed place from :meth:`Curve.chart_place`.  A guarded
        polyline in the x-plane avoids the branch values and is lifted by
        numerical continuation between the places' affine junction points;
        chart tails are joined at those junctions.  The returned ``CurvePath``
        record contains an opaque curve and numerical-context identity, the
        endpoint places, the sheet index reached, the continuation record
        carrying the numerical routing data used by
        :meth:`Curve.integral`, and any joined chart tails.

        Both junction points must have distinct ``x`` values, and ``end`` must
        lie on the sheet reached by continuation; otherwise ``ValueError`` is
        raised.

        >>> from mpmath import algebraic_curve, mp
        >>> mp.dps = 15
        >>> curve = algebraic_curve({(0, 2): 1, (1, 0): -1})
        >>> path = curve.path((1, 1), (4, 2))
        >>> mp.nstr(path.start.y, 6), mp.nstr(path.end.y, 6)
        ('1.0', '2.0')
        """
        return self._call(_operations.path, start, end)

    def integral(self, differentials, path):
        r"""Integrate one differential or a differential basis along a path.

        ``differentials`` is either a single callable ``f(x, y)`` returning the
        coefficient of ``dx``, or a sequence of such callables; ``path`` is a
        ``CurvePath`` from :meth:`Curve.path`.  A single differential
        gives a scalar ``values`` entry, a sequence gives one entry per form.
        Chart tails joined to the path are integrated through their coordinate
        maps with the same differentials.  The returned ``CurveIntegral`` record
        also carries the maximum curve-equation residual encountered on the
        integration nodes and the number of path segments.  A path is bound to
        the curve and working precision at which it was constructed and cannot
        be reused with a different curve or precision.

        >>> from mpmath import algebraic_curve, mp
        >>> mp.dps = 15
        >>> curve = algebraic_curve({(0, 2): 1, (1, 0): -1})
        >>> path = curve.path((1, 1), (4, 2))
        >>> integral = curve.integral(lambda x, y: 1 / y, path)
        >>> mp.nstr(integral.values, 12)
        '2.0'
        """
        return self._call(_operations.integral, differentials, path)

    def abel_map_kind_1(self, target, differentials=None, *, base_place=None,
                 reduce=False):
        r"""Evaluate the Abel map of a place or divisor on a plane curve.

        Generalizes elliptic integrals to genus :math:`g`, integrating :math:`g`
        differentials simultaneously from a base place to a target. Returns an
        unnormalized vector in :math:`\mathbb{C}^g`, defined modulo its period
        lattice.

        ``target`` is a single place (as ``(x, y)`` pair or ``CurvePlace``),
        a chart-backed place, or a sequence of places (effective divisor). An
        empty sequence returns the zero vector. Use ``differentials`` to override
        the automatic basis. For a chart-backed endpoint on a recognized
        hyperelliptic curve, supply first-kind ``differentials`` to select the
        geometric-polygon engine.

        ``base_place`` defaults to the selected engine's natural base: the
        point at infinity for odd-degree hyperelliptic models, the first
        ordered finite branch point for even-degree hyperelliptic models, or
        sheet zero over the computational base point for the geometric-polygon
        engine.
        With ``reduce=True``, the result is reduced modulo the period lattice
        ``[2*omega, 2*omega_prime]`` of the selected first-kind basis.

        >>> from mpmath import algebraic_curve, mp
        >>> mp.dps = 15
        >>> curve = algebraic_curve({(0, 2): 1, (1, 0): 1, (3, 0): -1})
        >>> point = (mp.mpf(2), mp.sqrt(6))
        >>> forms = (lambda x, y: 1 / y,)
        >>> value = curve.abel_map_kind_1(point, forms, base_place=point)
        >>> mp.nstr(mp.norm(value), 3)
        '0.0'
        """
        return self._call(
            _operations.abel_map,
            target,
            differentials,
            base_place=base_place,
            reduce=reduce,
        )

    def abel_map_kind_2(
            self, target, differentials=None, *, second_differentials=None,
            base_place=None, reduce=False):
        r"""Evaluate second-kind Abelian integrals of a place or divisor.

        ``target`` and ``base_place`` have the same meanings as in :meth:`abel_map_kind_1`.
        Returns a ``CurveSecondKindAbelMap`` record with the second-kind ``value``.

        Hyperelliptic curves use automatic BEL basis construction. The geometric
        polygon engine requires explicit ``second_differentials``; to override
        the second-kind basis on a recognized hyperelliptic curve, also supply
        first-kind ``differentials`` to select that engine. Supplied second-kind
        forms must have zero residues and be regular along the integration
        paths. With ``reduce=True``, the compatible first-kind Abel map
        determines a cycle shift applied to both first- and second-kind values,
        recorded as ``reduction_shift``.

        Chart-backed endpoints use the geometric polygon engine. Chart integration
        paths must converge; pole regularization is not performed.
        """
        return self._call(
            _operations.abel_map,
            target,
            differentials,
            second_kind=True,
            second_differentials=second_differentials,
            base_place=base_place,
            reduce=reduce,
        )

    def lattice_reduce(self, value, periods):
        r"""Reduce a Jacobian vector modulo the period lattice.

        ``value`` is a genus-length column vector of Abelian coordinates. If
        ``periods`` is a ``CurveFirstKindPeriods`` record, ``value`` is in the
        original differential basis and is reduced by the full lattice
        ``[2*omega, 2*omega_prime]``.  If ``periods`` is a normalized Riemann
        matrix ``tau``, ``value`` is in normalized coordinates and is reduced
        by ``[I, tau]``.  The returned ``CurveLatticeReduction`` record contains
        the equivalent vector and the integer lattice shift ``(m, n)``.

        >>> from mpmath import algebraic_curve, mp
        >>> mp.dps = 15
        >>> curve = algebraic_curve((0, -1, 0, 1))
        >>> tau = curve.riemann_matrix()
        >>> reduced = curve.lattice_reduce(mp.matrix([2 + 1j]), tau)
        >>> reduced.shift
        (2, 1)
        >>> mp.nstr(reduced.value[0, 0], 3)
        '0.0'
        """
        self._check_precision()
        return _operations.lattice_reduce(self.ctx, value, periods)

    def chart(self, chart_curve, coordinate_map):
        r"""Return a user-supplied local chart of a plane algebraic curve.

        ``chart_curve`` gives the local curve as a sparse mapping from
        ``(t_power, w_power)`` pairs to coefficients, or a sequence of
        ``(t_power, w_power, coefficient)`` terms.  ``coordinate_map(t, w)``
        must return the ambient triple ``(x, y, dx/dt)``, where ``x`` depends
        on ``t`` alone.  The returned ``CurveChart`` is bound to the ambient
        curve and working precision, and is accepted by the other chart
        methods and by :meth:`Curve.chart_place`.
        """
        return self._call(charts.chart, chart_curve, coordinate_map)

    def monomial_chart(self, x_power, y_power, *, source=None):
        r"""Return the monomial chart ``x = t**x_power, y = t**y_power*w``.

        ``source`` defaults to this curve; pass another ``CurveChart`` to
        compose coordinate maps. Negative powers describe places above
        infinity. Repeated factors are cleared to give a polynomial in
        ``t`` and ``w``; this does not normalize a singular chart.
        """
        self._check_precision()
        if source is None:
            source = self._specification
        return charts.monomial_chart(
            self.ctx, source, x_power, y_power)

    def chart_fibre(self, chart, t):
        r"""Return the ordered fibre of chart ``w`` values over ``t``.

        The values are ordered by real and imaginary part, like
        :meth:`Curve.fibre`.  A fibre whose values do not separate
        indicates that the chart does not resolve the requested place and is
        rejected.
        """
        self._check_precision()
        return charts.chart_fibre(self.ctx, chart, t)

    def chart_place(self, chart, seed, cutoff):
        r"""Return the chart-backed place reached by a local branch.

        ``seed`` is the branch value of ``w`` at ``t = 0``, for example from
        :meth:`Curve.chart_fibre`; the branch is continued along the
        straight chart path from ``t = 0`` to ``t = cutoff``.  The returned
        place is represented by its finite affine cutoff point together with a
        chart tail describing the local branch, and is bound to ``curve`` and
        the working precision.  The chart must parametrize ``curve``: the
        cutoff point is checked to lie on the curve.

        >>> from mpmath import algebraic_curve, mp
        >>> mp.dps = 15
        >>> curve = algebraic_curve({(0, 2): 1, (1, 0): 1, (3, 0): -1})
        >>> chart = curve.monomial_chart(-2, -3)
        >>> [mp.nstr(value, 3) for value in curve.chart_fibre(chart, 0)]
        ['(-1.0 + 0.0j)', '(1.0 + 0.0j)']
        >>> place = curve.chart_place(chart, 1, mp.mpf("0.05"))
        >>> mp.nstr(place.x, 6)
        '400.0'
        """
        return self._call(
            charts.chart_place, chart, seed, cutoff)

    def chart_integral(self, chart, differentials, t_path, seed):
        r"""Integrate ambient differentials along a local chart branch.

        ``differentials`` are ambient ``f(x, y)`` callables returning the
        coefficient of ``dx``; they are pulled back through the chart's
        coordinate map, so a single callable gives a scalar and a sequence
        gives one entry per form.  ``t_path`` is a sequence of finite ``t``
        values along which the branch is continued from ``seed`` at
        ``t_path[0]``.  Closed chart loops therefore compute residues of
        pulled-back forms at the place.
        """
        self._check_precision()
        return charts.chart_integral(
            self.ctx, chart, differentials, t_path, seed)

    def __repr__(self):
        return (
            f"Curve(x_degree={self.x_degree}, "
            f"y_degree={self.y_degree}, "
            f"ctx.prec={self._creation_state[0]})"
        )


class CurveMethods:
    """Context methods for constructing algebraic curves."""

    def algebraic_curve(ctx, specification):
        """Construct a curve with specialized hyperelliptic or general geometry.

        General curves use a common compact marking for
        first-kind general-curve operations, with automatic or supplied
        holomorphic bases. Supplied second-kind forms support periods and
        Abel maps, including convergent chart tails. Divergent pole values
        are not regularized.
        Recognized hyperelliptic models use the specialized Baker marking.
        Monodromy is a radial diagnostic with its own base fibre.
        """
        return Curve(ctx, specification)


CurveBranchLocus = _records.CurveBranchLocus
CurveFirstKindPeriods = _records.CurveFirstKindPeriods
CurveSecondKindPeriods = _records.CurveSecondKindPeriods
CurveSecondKindAbelMap = _records.CurveSecondKindAbelMap
CurveMonodromy = _records.CurveMonodromy
CurveGenus = _records.CurveGenus
CurveHomology = _records.CurveHomology
CurveRiemannConstant = _records.CurveRiemannConstant
CurveValidation = _records.CurveValidation
CurveCheck = _records.CurveCheck
CurvePlace = _records.CurvePlace
CurveChart = _records.CurveChart
CurvePath = _records.CurvePath
CurveIntegral = _records.CurveIntegral
CurveLatticeReduction = _records.CurveLatticeReduction


__all__ = [
    "Curve",
    "CurveBranchLocus",
    "CurveChart",
    "CurveCheck",
    "CurveFirstKindPeriods",
    "CurveGenus",
    "CurveHomology",
    "CurveIntegral",
    "CurveLatticeReduction",
    "CurveMonodromy",
    "CurvePath",
    "CurvePlace",
    "CurveRiemannConstant",
    "CurveSecondKindAbelMap",
    "CurveSecondKindPeriods",
    "CurveValidation",
]
