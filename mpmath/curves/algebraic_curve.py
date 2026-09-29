"""Object-oriented interface to numerical algebraic curves."""

import warnings

from . import _operations, _records, charts


class AlgebraicCurve:
    """A plane algebraic curve bound to an mpmath numerical context.

    Construct curves normally with ``mp.algebraic_curve(specification)``.
    The explicit form ``AlgebraicCurve(ctx, specification)`` is useful for
    custom contexts. Expensive stages are lazy and are cached by the core
    engine using the current precision and numerical context state.

    The canonical ``specification`` is a sparse ``(x_power, y_power)``
    coefficient mapping.  Sequences of ``(x_power, y_power, coefficient)``
    terms and ascending coefficient sequences for ``y**2 = P(x)`` are also
    accepted. Structurally hyperelliptic equations are
    recognized after normalization, independently of their input syntax.

    General curves use a common geometric polygon. Recognized hyperelliptic
    models use their specialized Baker marking, unless supplied forms
    explicitly select the general geometric integration pipeline.
    Automatic or supplied holomorphic bases and chart-backed Abel endpoints
    are supported. Supplied callables must be holomorphic on the curve;
    numerical checks cannot certify that they have no poles. Supplied
    second-kind forms must have zero residues and no poles on integration
    paths. Chart endpoints use ordinary convergent integrals; pole values
    are not regularized.
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
                "the AlgebraicCurve context changed after construction; "
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
        infinity is reported by :attr:`AlgebraicCurve.monodromy` instead,
        because it requires monodromy rather than the resultant alone.

        The lemniscatic curve :math:`y^2 = x^3 - x` has a two-sheeted
        projection with three finite branch values::

            >>> from mpmath import algebraic_curve
            >>> locus = algebraic_curve((0, -1, 0, 1)).branch_locus
            >>> locus.degree
            2
            >>> locus.branch_values
            (mpf('-1.0'), mpf('0.0'), mpf('1.0'))

        This property can be expensive; numerical stages are cached per context state.
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

        This property can be expensive; numerical stages are cached per context state.
        """
        return self._call(_operations.monodromy)

    @property
    def genus(self):
        r"""Genus of the compact curve.

        This property can be expensive; numerical stages are cached per context
        state. See :attr:`genus_data` for projection degree and ramification.
        """
        return self._call(_operations.genus_data).genus

    @property
    def genus_data(self):
        r"""Return the genus of a plane algebraic curve.

        General curves obtain genus from the compact covering graph.
        Recognized hyperelliptic models use the Riemann--Hurwitz formula
        applied to monodromy, including ramification at infinity.
        The returned ``CurveGenus`` record also records the
        projection degree and the total ramification, so the Riemann--Hurwitz
        balance :math:`2g-2 = -2d+r` can be checked directly::

            >>> from mpmath import algebraic_curve
            >>> algebraic_curve((0, -1, 0, 1)).genus_data
            CurveGenus(genus=1, degree=2, ramification=4)

        This property can be expensive; numerical stages are cached per context state.
        """
        return self._call(_operations.genus_data)

    @property
    def homology(self):
        r"""Return the homology marking used by the curve's default engine.

        A recognized hyperelliptic curve returns the compact Baker-marked basis
        used by its automatic periods and Abel maps.  This basis has ``2*genus``
        cycles, the standard symplectic intersection form, and no auxiliary
        boundary or radical cycles.  Its transformation is therefore the
        identity.

        General curves use a compact geometric polygon with a canonical
        symplectic basis, no auxiliary boundary cycles and identity transformation.

        The lemniscatic curve :math:`y^2 = x^3 - x` uses its Baker marking::

            >>> from mpmath import algebraic_curve
            >>> homology = algebraic_curve((0, -1, 0, 1)).homology
            >>> homology.genus, homology.marking
            (1, 'baker')
            >>> homology.intersection_form
            ((0, 1), (-1, 0))

        This property can be expensive; numerical stages are cached per context state.
        """
        return self._call(_operations.homology)

    def first_kind_periods(self, differentials=None):
        r"""Return first-kind half-periods and the normalized Riemann matrix.

        The ``CurveFirstKindPeriods`` record contains ``omega``, ``omega_prime``
        and ``tau``. Full periods are ``2*omega`` and ``2*omega_prime``; ``tau``
        is the symmetric part of ``omega**-1 * omega_prime``. The ``engine`` and
        ``marking`` fields identify the cycle convention.

        Recognized hyperelliptic models use the automatic basis ``x**k dx/z``
        in the Baker marking, where ``z = y + B(x)/(2*A)`` after completing the
        square in ``A*y**2 + B(x)*y + C(x) = 0``. Their ``differentials`` field
        is ``None``. Other Newton-nondegenerate plane curves use a Baker basis
        indexed by interior lattice points of the Newton polygon.

        Supplying ``differentials`` selects the general geometric-polygon
        engine and overrides either automatic basis. Supply one holomorphic
        callable ``f(x, y)`` per genus, giving the coefficient of ``dx``.
        Numerical checks do not certify that supplied forms have no poles.
        A non-positive-definite normalized period matrix raises ``ValueError``.

        Instance caching reuses assembled results in addition to cached numerical
        stages. Returned matrices are independent copies of the cached data.

        The lemniscatic curve has normalized period matrix ``tau = i``::

            >>> from mpmath import algebraic_curve, mp
            >>> mp.dps = 15
            >>> curve = algebraic_curve((0, -1, 0, 1))
            >>> data = curve.first_kind_periods()
            >>> mp.re(data.tau[0, 0]), mp.im(data.tau[0, 0])
            (mpf('0.0'), mpf('1.0'))

        A supplied basis uses the general marking::

            >>> data = curve.first_kind_periods((lambda x, y: 1 / y,))
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

    def second_kind_periods(self, differentials=None, *,
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
        cached for later calls to :meth:`first_kind_periods`.
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
        ``curve.first_kind_periods(differentials).tau``; see
        :meth:`AlgebraicCurve.first_kind_periods` for the input conventions.

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
        :meth:`AlgebraicCurve.first_kind_periods`. The value is computed directly from the
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
            >>> periods = curve.first_kind_periods(forms)
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
            >>> report = curve.validate(curve.first_kind_periods())
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
        :attr:`AlgebraicCurve.monodromy`.

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
        ``(x, y)`` pair, a ``CurvePlace`` from :meth:`AlgebraicCurve.fibre`, or
        a chart-backed place from :meth:`AlgebraicCurve.chart_place`.  A guarded
        polyline in the x-plane avoids the branch values and is lifted by
        numerical continuation between the places' affine junction points;
        chart tails are joined at those junctions.  The returned ``CurvePath``
        record contains an opaque curve and numerical-context identity, the
        endpoint places, the sheet index reached, the continuation record
        carrying the numerical routing data used by
        :meth:`AlgebraicCurve.integral`, and any joined chart tails.

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
        ``CurvePath`` from :meth:`AlgebraicCurve.path`.  A single differential
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

    def abel_map(self, target, differentials=None, *, base_place=None,
                 reduce=False):
        r"""Evaluate the Abel map of a place or divisor on a plane curve.

        ``target`` is one regular finite place, given as a ``(x, y)`` pair or
        ``CurvePlace``, a chart-backed place from
        :meth:`AlgebraicCurve.chart_place`, or a sequence of places representing
        an effective divisor; an empty sequence returns the zero vector.  A
        structurally hyperelliptic equation without supplied differentials is
        dispatched to the specialized Abel-map engine, including the ordinate
        change required after completing the square.  A general plane curve
        selects a first-kind basis automatically when Baker's construction
        applies, or accepts one callable per genus, and returns the
        unnormalized Abelian coordinates they integrate to.
        Chart-backed places require the general pipeline.

        ``base_place`` selects a finite or chart-backed base place. The
        default is the selected engine's natural base; for the general
        engine this is sheet zero over its computational base point.
        With ``reduce=True`` the unnormalized result is reduced modulo the full
        period lattice ``[2*omega, 2*omega_prime]`` of the supplied basis,
        equivalent to applying
        :meth:`AlgebraicCurve.lattice_reduce`.

        >>> from mpmath import algebraic_curve, mp
        >>> mp.dps = 15
        >>> curve = algebraic_curve({(0, 2): 1, (1, 0): 1, (3, 0): -1})
        >>> point = (mp.mpf(2), mp.sqrt(6))
        >>> forms = (lambda x, y: 1 / y,)
        >>> value = curve.abel_map(point, forms, base_place=point)
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

    def second_kind_abel_map(
            self, target, differentials=None, *, second_differentials=None,
            base_place=None, reduce=False):
        r"""Evaluate second-kind Abelian integrals of a place or divisor.

        ``target`` and ``base_place`` have the same meanings as in
        :meth:`abel_map`. The ``CurveSecondKindAbelMap`` result contains the
        unnormalized second-kind ``value`` and its ``engine`` and ``marking``.

        Recognized hyperelliptic models use their automatic BEL basis. The
        general engine requires one callable per genus in ``second_differentials``;
        its first-kind forms may be supplied or selected automatically. To use
        supplied second-kind forms on a hyperelliptic model, supply the
        first-kind ``differentials`` as well.

        With ``reduce=True``, reduction of the compatible first-kind Abel map
        determines the common integer cycle shift. The second-kind value is
        adjusted by the same shift, reported as ``reduction_shift``. Without
        reduction that field is ``None``.

        Chart-backed endpoints require the general engine. Chart tails must
        converge; divergent endpoint values are not regularized.
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
        methods and by :meth:`AlgebraicCurve.chart_place`.
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
        :meth:`AlgebraicCurve.fibre`.  A fibre whose values do not separate
        indicates that the chart does not resolve the requested place and is
        rejected.
        """
        self._check_precision()
        return charts.chart_fibre(self.ctx, chart, t)

    def chart_place(self, chart, seed, cutoff):
        r"""Return the chart-backed place reached by a local branch.

        ``seed`` is the branch value of ``w`` at ``t = 0``, for example from
        :meth:`AlgebraicCurve.chart_fibre`; the branch is continued along the
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
            f"AlgebraicCurve(x_degree={self.x_degree}, "
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
        return AlgebraicCurve(ctx, specification)


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
    "AlgebraicCurve",
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
