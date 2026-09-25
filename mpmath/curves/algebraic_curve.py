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
    terms and ascending coefficient sequences for ``y**2 = P(x)`` remain
    accepted for compatibility.  Structurally hyperelliptic equations are
    recognized after normalization, independently of their input syntax.
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
            _operations._normalise_algebraic_curve_input(ctx, specification))
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
                _operations._normalise_algebraic_curve_input(
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
        """Finite branch locus of the projection to the ``x``-line."""
        return self._call(_operations.branch_locus)

    @property
    def monodromy(self):
        """Monodromy data of the projection to the ``x``-line."""
        return self._call(_operations.monodromy)

    @property
    def genus(self):
        """Genus of the compact curve."""
        return self._call(_operations.genus_data).genus

    @property
    def genus_data(self):
        """Genus, projection degree, and total ramification."""
        return self._call(_operations.genus_data)

    @property
    def homology(self):
        """Homology data in the marking used by the default curve engine."""
        return self._call(_operations.homology)

    def first_kind_periods(self, differentials=None):
        """Return first-kind half-periods and the Riemann matrix.

        Recognized hyperelliptic curves use their automatic basis and Baker
        marking. Other Newton-nondegenerate plane curves use an automatic
        basis indexed by interior lattice points. Supplying ``differentials``
        overrides either basis and selects the general canonical-polygon
        engine.
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
        """Return second-kind half-periods and kappa.

        Recognized hyperelliptic curves use the automatic BEL basis. The
        general engine requires ``second_differentials``; its first-kind
        basis may be supplied or selected automatically.
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
        """Return the normalized Riemann matrix."""
        return self._call(_operations.riemann_matrix, differentials)

    def riemann_constant(self, differentials=None, *, base_place=None):
        """Return the vector of Riemann constants."""
        return self._call(
            _operations.riemann_constant,
            differentials,
            base_place=base_place,
        )

    def validate(self, result):
        """Validate a result record returned by a curve computation."""
        self._check_precision()
        return _operations.validate(self.ctx, result)

    def fibre(self, x):
        """Return the labelled fibre over ``x``."""
        return self._call(_operations.fibre, x)

    def path(self, start, end):
        """Return a lifted path between two regular places."""
        return self._call(_operations.path, start, end)

    def integral(self, differentials, path):
        """Integrate one differential or a basis along ``path``."""
        return self._call(_operations.integral, differentials, path)

    def abel_map(self, target, differentials=None, *, base_place=None,
                 reduce=False):
        """Evaluate the first-kind Abel map of a place or divisor."""
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
        """Evaluate second-kind Abelian integrals.

        The result is a ``CurveSecondKindAbelMap`` record. Recognized
        hyperelliptic curves use their automatic BEL basis; the general
        engine requires ``second_differentials`` and can select first-kind
        forms automatically where Baker's construction applies.
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
        """Reduce a Jacobian vector modulo a period lattice."""
        self._check_precision()
        return _operations.lattice_reduce(self.ctx, value, periods)

    def chart(self, chart_curve, coordinate_map):
        """Create a user-supplied local chart of this curve."""
        return self._call(charts.chart, chart_curve, coordinate_map)

    def monomial_chart(self, x_power, y_power, *, source=None):
        """Create a monomial chart, optionally composing an existing chart."""
        self._check_precision()
        if source is None:
            source = self._specification
        return charts.monomial_chart(
            self.ctx, source, x_power, y_power)

    def chart_fibre(self, chart, t):
        """Return the ordered fibre of a local chart over ``t``."""
        self._check_precision()
        return charts.chart_fibre(self.ctx, chart, t)

    def chart_place(self, chart, seed, cutoff):
        """Return the place reached along a local chart branch."""
        return self._call(
            charts.chart_place, chart, seed, cutoff)

    def chart_integral(self, chart, differentials, t_path, seed):
        """Integrate ambient differentials along a local chart branch."""
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
    "CurveFirstKindPeriods",
    "CurveSecondKindPeriods",
    "CurveSecondKindAbelMap",
    "CurveBranchLocus",
    "CurveChart",
    "CurveCheck",
    "CurveGenus",
    "CurveHomology",
    "CurveIntegral",
    "CurveLatticeReduction",
    "CurveMonodromy",
    "CurvePath",
    "CurvePlace",
    "CurveRiemannConstant",
    "CurveValidation",
]
