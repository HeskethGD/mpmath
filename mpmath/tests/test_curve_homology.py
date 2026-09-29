"""Exact topological checks independent of period integration."""

import pytest

import mpmath.curves.homology as curve_homology
from mpmath import mp
from mpmath.curves._stages import _stage_geometric_cover
from mpmath.curves.homology import (
    _brahana_canonical_words, _canonical_ribbon_polygon,
    _expand_cut_system_word, _geometric_ribbon_graph, _integer_matrix_rank,
    _ribbon_tree_cotree_cut_system,
)
from mpmath.curves.integration import _integrate_geometric_loops_iterated
from mpmath.curves._records import (
    _GeometricCover, _GeometricEdge, _GeometricRibbonGraph, _PlaneGraph,
)
from mpmath.curves.homology import _ribbon_cycle_intersection, _tree_path
from mpmath.curves.polynomial import _prepare_plane_curve


@pytest.mark.parametrize("sheet", (0, 1))
@pytest.mark.parametrize("reverse", (False, True))
def test_canonical_polygon_survives_admissible_root_and_cotree_choices(sheet, reverse):
    ctx = mp.clone()
    ctx.dps = 18
    curve = _prepare_plane_curve(ctx, {(0, 2): 1, (3, 0): -1, (1, 0): 1})
    cover = _stage_geometric_cover(ctx, curve)
    graph = _geometric_ribbon_graph(ctx, cover)
    complement = tuple(i for i in range(len(graph.edges)) if i not in graph.tree_edges)
    root = (min(graph.vertices)[0], sheet)
    cut = _ribbon_tree_cotree_cut_system(
        graph, root=root,
        cotree_edge_order=complement[::-1] if reverse else complement)
    polygon = _canonical_ribbon_polygon(graph, cut)
    assert polygon.root == root
    assert polygon.intersection == ((0, 1), (-1, 0))
    for loop in polygon.a_loops + polygon.b_loops:
        current = root
        for edge_index, orientation in loop:
            edge = graph.edges[edge_index]
            start, end = ((edge.tail, edge.head) if orientation == 1
                          else (edge.head, edge.tail))
            assert start == current
            current = end
        assert current == root
    with pytest.raises(ValueError, match="spanning tree"):
        _ribbon_tree_cotree_cut_system(graph, tree_edges=())
    with pytest.raises(ValueError, match="permute the tree complement"):
        _ribbon_tree_cotree_cut_system(graph, cotree_edge_order=())
    with pytest.raises(ValueError, match="base-sheet vertex"):
        _ribbon_tree_cotree_cut_system(graph, root=(len(graph.vertices), 0))


@pytest.mark.parametrize("word,message", [
    (((0, 1), (0, -1)), "4.g sides"),
    (((0, 1), (1, 1), (0, 1), (1, -1)), "opposite signs"),
    (((0, 1), (0, -1), (1, 1), (1, -1)), "interlaced orientable handle"),
])
def test_canonical_words_reject_invalid_handle_pairings(word, message):
    with pytest.raises(ValueError, match=message):
        _brahana_canonical_words(word)


def test_expanding_cut_words_rejects_invalid_generators_and_orientations():
    loops = (((3, 1), (4, -1)),)
    with pytest.raises(ValueError, match="invalid generator"):
        _expand_cut_system_word(((1, 1),), loops)
    with pytest.raises(ValueError, match="orientations must be"):
        _expand_cut_system_word(((0, 0),), loops)
    assert _expand_cut_system_word(((0, 1), (0, -1)), loops) == ()


def test_integer_rank_is_exact_for_nearly_dependent_large_rows():
    n = 10**40
    assert _integer_matrix_rank(()) == 0
    assert _integer_matrix_rank(((0, 0), (0, 0))) == 0
    assert _integer_matrix_rank(((n, n + 1), (n + 1, n + 2))) == 2
    assert _integer_matrix_rank(((0, 1, 2), (0, 2, 4), (0, 3, 6))) == 1


def test_iterated_geometric_loops_require_continuous_closed_edge_words():
    ctx = mp.clone()
    ctx.dps = 18
    curve = _prepare_plane_curve(ctx, {(0, 2): 1, (3, 0): -1, (1, 0): 1})
    cover = _stage_geometric_cover(ctx, curve)
    graph = _geometric_ribbon_graph(ctx, cover)
    polygon = _canonical_ribbon_polygon(graph)
    integrate = _integrate_geometric_loops_iterated
    with pytest.raises(ValueError, match="orientation must be"):
        integrate(ctx, curve, cover, graph, polygon, (((0, 0),),), ())
    off_root = next((index, orientation)
                    for index, edge in enumerate(graph.edges)
                    for orientation, start in ((1, edge.tail), (-1, edge.head))
                    if start != polygon.root)
    with pytest.raises(ValueError, match="discontinuous"):
        integrate(ctx, curve, cover, graph, polygon, ((off_root,),), ())
    with pytest.raises(ValueError, match="nonempty and closed"):
        integrate(ctx, curve, cover, graph, polygon, ((),), ())


def test_lifted_graph_rejects_disconnected_sheet_actions():
    geometry = _PlaneGraph((0, 1, 1j), ((0, 1), (0, 2), (1, 2)),
                           (), (), ())
    cover = _GeometricCover(geometry, ((-1, 1),) * 3, (), ((0, 1),) * 3)
    with pytest.raises(ValueError, match="disconnected"):
        _geometric_ribbon_graph(mp, cover)


def test_tree_cotree_rejects_a_cycle_in_a_supplied_tree():
    vertices = tuple((index, 0) for index in range(4))
    edges = tuple(_GeometricEdge(vertices[left], vertices[right], index, 0)
                  for index, (left, right) in enumerate(
                      ((0, 1), (1, 2), (2, 0), (2, 3))))
    graph = _GeometricRibbonGraph(vertices, edges, {}, (0, 1, 3), 0, ())
    with pytest.raises(ValueError, match="spanning tree"):
        _ribbon_tree_cotree_cut_system(graph, tree_edges=(0, 1, 2))
    with pytest.raises(ValueError, match="disconnected"):
        _tree_path({vertices[0]: [], vertices[1]: []}, vertices[0], vertices[1])


def test_ribbon_intersection_rejects_nonintegral_open_flows():
    rotation = {(0, 0): ((0, 0), (1, 0))}
    with pytest.raises(ValueError, match="not integral"):
        _ribbon_cycle_intersection((), rotation, (1, 0), (0, 1))


def _elliptic_ribbon_data():
    ctx = mp.clone()
    ctx.dps = 18
    curve = _prepare_plane_curve(ctx, {(0, 2): 1, (3, 0): -1, (1, 0): 1})
    cover = _stage_geometric_cover(ctx, curve)
    return ctx, cover, _geometric_ribbon_graph(ctx, cover)


def test_ribbon_certification_rejects_invalid_face_data(monkeypatch):
    ctx, cover, graph = _elliptic_ribbon_data()
    boundary_orbits = curve_homology._ribbon_boundary_orbits

    def extra_face(*args, **kwargs):
        faces = boundary_orbits(*args, **kwargs)
        return faces + (faces[0],)

    with monkeypatch.context() as patch:
        patch.setattr(curve_homology, "_ribbon_boundary_orbits", extra_face)
        with pytest.raises(ValueError, match="invalid Euler characteristic"):
            _geometric_ribbon_graph(ctx, cover)

    assert curve_homology._ribbon_boundary_orbits(
        graph.edges, graph.rotation, ()) == ()
    malformed = {0: ((0, 0), (1, 0), (0, 1), (1, 1), (0, 0))}
    with pytest.raises(ValueError, match="boundary orbit does not close"):
        curve_homology._ribbon_boundary_orbits((0, 1), malformed)


@pytest.mark.parametrize("fault,message", [
    ("disconnected_dual", "dual complement .* disconnected"),
    ("wrong_genus", "does not have 2\\*g edges"),
    ("two_cut_faces", "does not have one face"),
    ("wrong_side_count", "wrong side count"),
    ("bad_pairing", "side pairing is inconsistent"),
    ("repeated_loop_edge", "repeats an oriented edge"),
    ("degenerate_intersection", "degenerate intersection"),
])
def test_tree_cotree_certification_rejects_corrupt_intermediates(
        monkeypatch, fault, message):
    unused_ctx, unused_cover, graph = _elliptic_ribbon_data()
    cut = _ribbon_tree_cotree_cut_system(graph)
    boundary_orbits = curve_homology._ribbon_boundary_orbits
    if fault == "wrong_genus":
        graph = graph._replace(genus=graph.genus + 1)
    elif fault == "disconnected_dual":
        midpoint = len(graph.edges) // 2
        faces = tuple(tuple((index, endpoint)
                            for index in indices for endpoint in (0, 1))
                      for indices in (range(midpoint),
                                      range(midpoint, len(graph.edges))))
        monkeypatch.setattr(
            curve_homology, "_ribbon_boundary_orbits",
            lambda *args, **kwargs: faces)
    elif fault in ("two_cut_faces", "wrong_side_count", "bad_pairing"):
        generator = cut.generator_edges[0]

        def corrupt_cut_orbit(*args, **kwargs):
            orbits = boundary_orbits(*args, **kwargs)
            included = args[2] if len(args) > 2 else kwargs.get("included_edges")
            if included is not None:
                if fault == "two_cut_faces":
                    return orbits + orbits
                if fault == "wrong_side_count":
                    return (tuple(dart for dart in orbits[0]
                                  if dart != (generator, 0)),)
                return (tuple((edge, 0) if edge == generator else (edge, end)
                              for edge, end in orbits[0]),)
            return orbits

        monkeypatch.setattr(
            curve_homology, "_ribbon_boundary_orbits", corrupt_cut_orbit)
    elif fault == "repeated_loop_edge":
        tree_path = curve_homology._tree_path

        def repeated(*args, **kwargs):
            return tree_path(*args, **kwargs) + ((graph.tree_edges[0], 1),) * 4

        monkeypatch.setattr(curve_homology, "_tree_path", repeated)
    else:
        monkeypatch.setattr(
            curve_homology, "_ribbon_cycle_intersection",
            lambda *args: 0)
    with pytest.raises(ValueError, match=message):
        _ribbon_tree_cotree_cut_system(graph)


@pytest.mark.parametrize("fault,message", [
    ("discontinuous", "canonical path is discontinuous"),
    ("open", "canonical path is not closed"),
    ("boundary", "canonical chain has nonzero boundary"),
])
def test_geometric_polygon_rejects_corrupt_paths(monkeypatch, fault, message):
    ctx, cover, graph = _elliptic_ribbon_data()
    polygon = _canonical_ribbon_polygon(graph)
    root = polygon.root
    if fault == "discontinuous":
        step = next((index, orientation)
                    for index, edge in enumerate(graph.edges)
                    for orientation, start in ((1, edge.tail), (-1, edge.head))
                    if start != root)
        polygon = polygon._replace(a_loops=((step,),), b_loops=())
    elif fault == "open":
        step = next((index, orientation)
                    for index, edge in enumerate(graph.edges)
                    for orientation, start in ((1, edge.tail), (-1, edge.head))
                    if start == root)
        polygon = polygon._replace(a_loops=((step,),), b_loops=())
    else:
        monkeypatch.setattr(
            curve_homology, "_lifted_path_chain_boundary", lambda *args: (1,))
    monkeypatch.setattr(curve_homology, "_canonical_ribbon_polygon",
                        lambda *args: polygon)
    with pytest.raises(ValueError, match=message):
        curve_homology._geometric_canonical_polygon(ctx, cover, graph)


def test_canonical_certificates_reject_corrupted_inverse_and_intersection(monkeypatch):
    word = ((0, 1), (1, 1), (0, -1), (1, -1))
    with monkeypatch.context() as patch:
        patch.setattr(curve_homology, "_inverse_oriented_word", lambda word: ())
        with pytest.raises(ValueError, match="do not reproduce"):
            _brahana_canonical_words(word)

    unused_ctx, unused_cover, graph = _elliptic_ribbon_data()
    cut = _ribbon_tree_cotree_cut_system(graph)
    monkeypatch.setattr(
        curve_homology, "_ribbon_cycle_intersection", lambda *args: 0)
    with pytest.raises(ValueError, match="wrong intersection form"):
        _canonical_ribbon_polygon(graph, cut)
