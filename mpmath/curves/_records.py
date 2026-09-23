"""Record types shared by the algebraic-curve package."""

from collections import namedtuple

_PlaneCurve = namedtuple(
    "_PlaneCurve", "terms x_degree y_degree")
_SheetContinuation = namedtuple(
    "_SheetContinuation",
    "sheets permutation max_residual min_separation "
    "max_prediction_correction steps path fibres refinements")
_BranchContinuation = namedtuple(
    "_BranchContinuation",
    "values max_residual min_derivative steps path refinements")
_MonodromyData = namedtuple(
    "_MonodromyData",
    "base_point base_sheets branch_points permutations "
    "infinity_permutation ramification genus continuations")
_BranchGenerator = namedtuple(
    "_BranchGenerator",
    "kind value permutation continuation radius")
_OrderedMonodromyData = namedtuple(
    "_OrderedMonodromyData",
    "base_point base_sheets center branch_points product_generators "
    "ribbon_generators ramification genus minimum_clearance")
_PathIntegrals = namedtuple(
    "_PathIntegrals", "values max_sheet_residual segments")
_IteratedPathIntegrals = namedtuple(
    "_IteratedPathIntegrals",
    "values iterated max_sheet_residual segments")
_LiftedPlaneCurvePath = namedtuple(
    "_LiftedPlaneCurvePath", "continuation sheet start end")
_LiftedPathTerm = namedtuple(
    "_LiftedPathTerm", "coefficient continuation sheet")
_LiftedPathChain = namedtuple(
    "_LiftedPathChain", "terms")
_BoundaryPlace = namedtuple(
    "_BoundaryPlace", "multiplicity x y")
_LiftedGraphEdge = namedtuple(
    "_LiftedGraphEdge", "tail head branch_index sheet")
_LiftedMonodromyGraph = namedtuple(
    "_LiftedMonodromyGraph",
    "degree genus permutations branch_orientations branch_cycles "
    "vertices edges rotation "
    "tree_edges chord_edges cycles intersection boundary_components "
    "intersection_rank")
_RibbonCutSystem = namedtuple(
    "_RibbonCutSystem",
    "root tree_edges cotree_edges generator_edges loops boundary_word "
    "intersection")
_CanonicalPolygon = namedtuple(
    "_CanonicalPolygon",
    "root a_words b_words a_loops b_loops relator intersection")
_SymplecticReduction = namedtuple(
    "_SymplecticReduction",
    "transformation form genus radical_rank")
_BranchLoopStep = namedtuple(
    "_BranchLoopStep", "branch_index turns")
_GraphCycleWord = namedtuple(
    "_GraphCycleWord", "start_sheet steps")
_NumericalGraphCycles = namedtuple(
    "_NumericalGraphCycles",
    "branch_continuations words chains")
_NumericalCanonicalPolygon = namedtuple(
    "_NumericalCanonicalPolygon",
    "polygon generator_chains a_chains b_chains chains "
    "generator_continuations a_continuations b_continuations "
    "transformation intersection_form")
_PlaneCurvePeriods = namedtuple(
    "_PlaneCurvePeriods",
    "periods a_periods b_periods tau symmetry_residual "
    "imaginary_eigenvalues max_sheet_residual")
CurveBranchLocus = namedtuple(
    "CurveBranchLocus", "degree branch_values resultant")
CurveMonodromy = namedtuple(
    "CurveMonodromy",
    "base_point base_sheets branch_values permutations "
    "infinity_permutation ramification genus transitive product_identity "
    "minimum_clearance")
CurveGenus = namedtuple("CurveGenus", "genus degree ramification")
CurveHomology = namedtuple(
    "CurveHomology",
    "genus cycle_count boundary_components intersection_rank radical_rank "
    "intersection_form transformation")
CurvePeriods = namedtuple(
    "CurvePeriods",
    "genus differentials omega omega_prime tau eta eta_prime kappa "
    "symmetry_residual kappa_symmetry_residual imaginary_eigenvalues "
    "max_sheet_residual")
CurveRiemannConstant = namedtuple(
    "CurveRiemannConstant",
    "value characteristic base_place max_sheet_residual")
CurveCheck = namedtuple("CurveCheck", "name value passed")
CurveValidation = namedtuple(
    "CurveValidation", "kind passed maximum_residual checks")
CurvePlace = namedtuple(
    "CurvePlace", "x y chart", defaults=(None,))
CurveChart = namedtuple(
    "CurveChart", "curve_key curve coordinate_map")
_CurveChartTail = namedtuple(
    "_CurveChartTail", "chart branch")
CurvePath = namedtuple(
    "CurvePath",
    "curve_key start end sheet continuation start_tail end_tail",
    defaults=(None, None))
CurveIntegral = namedtuple(
    "CurveIntegral", "values max_sheet_residual segments")
CurveLatticeReduction = namedtuple(
    "CurveLatticeReduction", "value shift")
# Internal place records use the public CurvePlace representation.
_PlaneCurvePlace = CurvePlace
