"""Continuation failures and refinement on an explicitly solvable cover."""

import pytest

from mpmath import mp
from mpmath.curves import continuation
from mpmath.curves.geometry import (
    _check_voronoi_disk, _clip_voronoi_halfplane, _finish_voronoi_cell,
    _register_voronoi_vertex, _voronoi_plane_graph,
)
from mpmath.curves._records import _PlaneGraph
from mpmath.curves.polynomial import _prepare_plane_curve


@pytest.mark.parametrize("path,options,message", [
    ((), {}, "at least one point"),
    ((1, mp.inf), {}, "points must be finite"),
    ((1, 4), {"max_refinements": -1}, "nonnegative integer"),
    ((1, 4), {"correction_fraction": 0}, "fractions must be positive"),
    ((1, 4), {"motion_fraction": -1}, "fractions must be positive"),
    ((1, 4), {"initial_sheets": (1,)}, "one value per sheet"),
])
def test_sheet_continuation_rejects_invalid_paths_and_controls(path, options, message):
    curve = _prepare_plane_curve(mp, {(0, 2): 1, (1, 0): -1})
    with pytest.raises(ValueError, match=message):
        continuation._continue_plane_curve_sheets_adaptive(mp, curve, path, **options)


@pytest.mark.parametrize("path,seed,options,message", [
    ((), 1, {}, "at least one point"),
    ((1, mp.inf), 1, {}, "points must be finite"),
    ((1, 4), 1, {"max_refinements": -1}, "nonnegative integer"),
    ((1, 4), 1, {"max_newton_steps": 0}, "positive integer"),
    ((1, 4), 2, {}, "not on the plane curve"),
    ((0, 1), 0, {}, "branch must be simple"),
])
def test_branch_continuation_rejects_invalid_seeds_and_controls(path, seed, options, message):
    curve = _prepare_plane_curve(mp, {(0, 2): 1, (1, 0): -1})
    with pytest.raises(ValueError, match=message):
        continuation._continue_plane_curve_branch(mp, curve, path, seed, **options)


def test_branch_and_sheet_refinement_resolve_large_steps():
    ctx = mp.clone()
    ctx.dps = 25
    curve = _prepare_plane_curve(ctx, {(0, 2): 1, (1, 0): -1})
    branch = continuation._continue_plane_curve_branch
    sheets = continuation._continue_plane_curve_sheets_adaptive
    with pytest.raises(ctx.NoConvergence, match="did not resolve"):
        branch(ctx, curve, (1, 16), 1, max_refinements=0)
    with pytest.raises(ctx.NoConvergence, match="did not resolve"):
        sheets(ctx, curve, (1, 16), max_refinements=0)
    single = branch(ctx, curve, (1, 16), 1)
    all_sheets = sheets(ctx, curve, (1, 16),
                        initial_sheets=(-1, 1), correction_fraction='.2', motion_fraction='.4')
    assert single.refinements > 0 and all_sheets.refinements > 0
    assert abs(single.values[-1] - 4) < 100*ctx.eps
    assert all(abs(a - b) < 100*ctx.eps for a, b in zip(all_sheets.sheets, (-4, 4)))
    assert single.max_residual < 1000*ctx.eps
    assert all_sheets.max_residual < 1000*ctx.eps
    assert ctx.dps == 25


def test_sheet_refinement_recovers_from_a_failed_root_solve(monkeypatch):
    ctx = mp.clone()
    ctx.dps = 20
    curve = _prepare_plane_curve(ctx, {(0, 2): 1, (1, 0): -1})
    solve = continuation._plane_curve_sheets
    failed = []

    def fail_first_step(*args, **kwargs):
        if not failed:
            failed.append(True)
            raise ctx.NoConvergence("root solve failed at the proposed step")
        return solve(*args, **kwargs)

    monkeypatch.setattr(continuation, '_plane_curve_sheets', fail_first_step)
    result = continuation._continue_plane_curve_sheets_adaptive(ctx, curve, (1, 2))
    assert result.refinements > 0
    assert all(abs(y*y - 2) < 100*ctx.eps for y in result.sheets)


@pytest.mark.parametrize("points,base,message", [
    ((), None, "at least one finite branch point"),
    ((mp.inf,), None, "branch points must be finite"),
    ((0, 0), None, "branch points must be distinct"),
    ((0, 1), mp.inf, "base point must be finite"),
    ((0, 1), 0, "insufficient clearance"),
    ((0, 1), 2, "insufficient clearance"),
])
def test_radial_geometry_rejects_unsafe_base_paths(points, base, message):
    with pytest.raises(ValueError, match=message):
        continuation._radial_branch_geometry(mp, points, base)


def test_radial_loop_orientation_and_segment_distance():
    ctx = mp.clone()
    ctx.dps = 25
    distance = continuation._point_segment_distance
    assert distance(ctx, ctx.j, 0, 0) == 1
    assert distance(ctx, ctx.j, -1, 1) == 1
    assert distance(ctx, 2, 0, 1) == 1
    base, center, radii, clearance = continuation._radial_branch_geometry(ctx, (0, 1), 2j)
    assert base == 2j and center == ctx.mpf('.5') and clearance > 0
    path = continuation._radial_branch_loop_path(ctx, base, 0, radii[0], circle_steps=8)
    assert path[0] == path[-1] == base
    assert all(abs(abs(z) - radii[0]) < 100*ctx.eps for z in path[1:-1])
    winding = ctx.fsum(ctx.arg(b/a) for a, b in zip(path[1:-2], path[2:-1]))
    assert abs(winding - 2*ctx.pi) < 100*ctx.eps


@pytest.mark.parametrize("radius,steps,message", [
    (1, 4, "integer at least 8"),
    (0, 8, "finite and positive"),
    (mp.inf, 8, "finite and positive"),
])
def test_radial_loops_reject_invalid_radius_or_order(radius, steps, message):
    with pytest.raises(ValueError, match=message):
        continuation._radial_branch_loop_path(mp, 2, 0, radius, steps)


@pytest.mark.parametrize("points", [(), (mp.inf,), (mp.nan,)])
def test_voronoi_geometry_requires_finite_branch_values(points):
    with pytest.raises(ValueError, match="finite nonempty branch values"):
        _voronoi_plane_graph(mp, points)


def test_voronoi_clipping_rejects_an_empty_cell():
    square = (-1-1j, 1-1j, 1+1j, -1+1j)
    with pytest.raises(mp.NoConvergence, match="cell is numerically degenerate"):
        _clip_voronoi_halfplane(mp, square, 10, 0, mp.eps)


def test_voronoi_vertex_identification_checks_incidence_and_position():
    positions = {}
    with pytest.raises(mp.NoConvergence, match="inconsistent incident sites"):
        _register_voronoi_vertex(mp, 0, (1, 2, 3), 0, positions, mp.eps)
    support = (0, 1, 2)
    _register_voronoi_vertex(mp, 0, support, 0, positions, mp.eps)
    assert positions[support] == 0
    with pytest.raises(mp.NoConvergence, match="could not be identified"):
        _register_voronoi_vertex(mp, 0, support, 1, positions, mp.eps)


def test_voronoi_cell_and_disk_reject_invalid_topology():
    assert _finish_voronoi_cell(mp, [0, 1, 2, 0]) == (0, 1, 2)
    with pytest.raises(mp.NoConvergence, match="invalid vertex identifications"):
        _finish_voronoi_cell(mp, [0, 1, 0, 2])
    with pytest.raises(mp.NoConvergence, match="do not form a disk"):
        _check_voronoi_disk(mp, (0, 1, 1j), (), ((0, 1, 2),), mp.eps, 1)
    with pytest.raises(mp.NoConvergence, match="edge is numerically degenerate"):
        _check_voronoi_disk(mp, (0, mp.eps/2), ((0, 1),), (), mp.eps, 1)


@pytest.mark.parametrize("start_y,tolerance,message", [
    (mp.inf, None, "start_y must be finite"),
    (2, None, "does not identify a sheet"),
    (1, -1, "finite and nonnegative"),
    (1, mp.inf, "finite and nonnegative"),
])
def test_lifted_path_requires_a_regular_start_place(start_y, tolerance, message):
    curve = _prepare_plane_curve(mp, {(0, 2): 1, (1, 0): -1})
    with pytest.raises(ValueError, match=message):
        continuation._lift_plane_curve_path(
            mp, curve, (1, 2), start_y, match_tolerance=tolerance)


def test_lifted_chain_validates_terms_and_skips_zero_coefficients():
    curve = _prepare_plane_curve(mp, {(0, 2): 1, (1, 0): -1})
    path = continuation._continue_plane_curve_sheets_adaptive(mp, curve, (1, 2))
    prepare = continuation._prepare_lifted_path_chain
    assert prepare(((0, path, 0),)).terms == ()
    assert prepare(((2, path, 1),)).terms[0].coefficient == 2
    for terms, error, message in (
        (None, ValueError, "chain terms must be"),
        (((1, path),), ValueError, "chain terms must be"),
        (((1.5, path, 0),), TypeError, "coefficients must be integers"),
        (((1, path, 0.5),), TypeError, "integer indices"),
        (((1, path, 2),), ValueError, "does not index"),
    ):
        with pytest.raises(error, match=message):
            prepare(terms)


def test_lifted_edge_sampler_rejects_non_edges_and_invalid_queries():
    curve = _prepare_plane_curve(mp, {(0, 2): 1, (1, 0): -1})
    lift = continuation._continue_plane_curve_sheets_adaptive
    sampler = continuation._LiftedEdgeSampler
    with pytest.raises(ValueError, match="distinct endpoints"):
        sampler(mp, curve, lift(mp, curve, (1, 1)))
    with pytest.raises(ValueError, match="straight ordered path"):
        sampler(mp, curve, lift(mp, curve, (1, 2, 1 + 1j)))
    edge = sampler(mp, curve, lift(mp, curve, (1, 2)))
    with pytest.raises(ValueError, match="invalid edge parameter"):
        edge.sample(-.1, 0)
    with pytest.raises(ValueError, match="invalid edge parameter"):
        edge.sample(.5, 2)
    x, y, residual = edge.sample(mp.mpf('.5'), 1)
    assert abs(y*y - x) < 100*mp.eps and residual < 100*mp.eps


def test_sheet_predictor_does_not_divide_by_zero_at_a_ramification_point():
    curve = _prepare_plane_curve(mp, {(0, 2): 1, (1, 0): -1})
    assert continuation._predict_plane_curve_sheets(mp, curve, 0, 1, (0,)) == (0,)


def test_lifted_graph_requires_regular_vertex_fibres():
    curve = _prepare_plane_curve(mp, {(0, 2): 1, (1, 0): -1})
    geometry = _PlaneGraph((0, 1), ((0, 1),), (), (0,), ())
    with pytest.raises(mp.NoConvergence, match="unresolved fibre separation"):
        continuation._lift_plane_graph(mp, curve, geometry)


def test_sheet_refinement_reports_a_segment_too_short_to_bisect(monkeypatch):
    ctx = mp.clone()
    ctx.dps = 18
    curve = _prepare_plane_curve(ctx, {(0, 2): 1, (1, 0): -1})

    def unresolved(*args, **kwargs):
        raise ctx.NoConvergence("fibre solve failed")

    monkeypatch.setattr(continuation, "_plane_curve_sheets", unresolved)
    with pytest.raises(ctx.NoConvergence, match="cannot be subdivided further"):
        continuation._continue_plane_curve_sheets_adaptive(
            ctx, curve, (1, 1 + ctx.eps), initial_sheets=(-1, 1))


def test_lifted_graph_rejects_a_changed_edge_endpoint_fibre(monkeypatch):
    ctx = mp.clone()
    curve = _prepare_plane_curve(ctx, {(0, 2): 1, (1, 0): -1})
    geometry = _PlaneGraph((1, 2), ((0, 1),), (), (0,), ())
    continue_sheets = continuation._continue_plane_curve_sheets_adaptive

    def mismatched(*args, **kwargs):
        lift = continue_sheets(*args, **kwargs)
        fibres = lift.fibres[:-1] + (tuple(y + 1 for y in lift.fibres[-1]),)
        return lift._replace(fibres=fibres)

    monkeypatch.setattr(continuation, "_continue_plane_curve_sheets_adaptive", mismatched)
    with pytest.raises(ctx.NoConvergence, match="endpoint fibre did not match"):
        continuation._lift_plane_graph(ctx, curve, geometry)


def test_edge_sampler_reports_an_unbisectable_bracket(monkeypatch):
    ctx = mp.clone()
    ctx.dps = 18
    curve = _prepare_plane_curve(ctx, {(0, 2): 1, (1, 0): -1})
    lift = continuation._continue_plane_curve_sheets_adaptive(
        ctx, curve, (1, 2 - ctx.eps, 2))
    sampler = continuation._LiftedEdgeSampler(ctx, curve, lift)

    def unresolved(context, prepared, x, prediction):
        return prediction, context.one, context.one, context.one, False

    monkeypatch.setattr(continuation, "_newton_plane_curve_sheet", unresolved)
    with pytest.raises(ctx.NoConvergence, match="did not resolve its sheet"):
        sampler.sample(ctx.one, 1)


def test_edge_sampler_rejects_refinement_that_changes_sheet_labels(monkeypatch):
    ctx = mp.clone()
    curve = _prepare_plane_curve(ctx, {(0, 2): 1, (1, 0): -1})
    lift = continuation._continue_plane_curve_sheets_adaptive(ctx, curve, (1, 2))
    sampler = continuation._LiftedEdgeSampler(ctx, curve, lift)
    continue_sheets = continuation._continue_plane_curve_sheets_adaptive

    def unresolved(context, prepared, x, prediction):
        return prediction, context.one, context.one, context.one, False

    def changed_labels(*args, **kwargs):
        refinement = continue_sheets(*args, **kwargs)
        fibres = refinement.fibres[:-1] + (
            tuple(y + 1 for y in refinement.fibres[-1]),)
        return refinement._replace(fibres=fibres)

    monkeypatch.setattr(continuation, "_newton_plane_curve_sheet", unresolved)
    monkeypatch.setattr(
        continuation, "_continue_plane_curve_sheets_adaptive", changed_labels)
    with pytest.raises(ctx.NoConvergence, match="changed the sheet labels"):
        sampler.sample(ctx.mpf('.5'), 1)
