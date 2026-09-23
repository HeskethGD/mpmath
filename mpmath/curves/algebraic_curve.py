"""Object-oriented interface to numerical algebraic curves."""

import warnings

from . import _operations, _records, charts


class AlgebraicCurve:
    """A plane algebraic curve bound to an mpmath numerical context.

    Construct curves normally with ``mp.algebraic_curve(specification)``.
    The explicit form ``AlgebraicCurve(ctx, specification)`` is useful for
    custom contexts. Expensive stages are lazy and are cached by the core
    engine using the current precision and numerical context state.

    ``specification`` may be a sparse ``(x_power, y_power)`` coefficient
    mapping, a sequence of ``(x_power, y_power, coefficient)`` terms, or an
    ascending coefficient sequence for ``y**2 = P(x)``.
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
        self._prepared, self._hyperelliptic_coefficients = (
            _operations._normalise_algebraic_curve_input(ctx, specification))

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
        return function(
            self.ctx, self._specification, *args, **kwargs)

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
        """Canonical homology data for the curve."""
        return self._call(_operations.homology)

    def periods(self, differentials=None, *, second_differentials=None):
        """Return first- and optionally second-kind period matrices."""
        return self._call(
            _operations.periods,
            differentials,
            second_differentials=second_differentials,
        )

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
        """Evaluate the Abel map of a place or divisor."""
        return self._call(
            _operations.abel_map,
            target,
            differentials,
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
CurveMonodromy = _records.CurveMonodromy
CurveGenus = _records.CurveGenus
CurveHomology = _records.CurveHomology
CurvePeriods = _records.CurvePeriods
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
    "CurveGenus",
    "CurveHomology",
    "CurveIntegral",
    "CurveLatticeReduction",
    "CurveMonodromy",
    "CurvePath",
    "CurvePeriods",
    "CurvePlace",
    "CurveRiemannConstant",
    "CurveValidation",
]
