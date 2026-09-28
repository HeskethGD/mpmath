"""Lifted ribbon graphs, canonical polygons, and homology intersections."""

from collections import deque
from fractions import Fraction

from ._records import (
    _CanonicalPolygon,
    _GeometricEdge,
    _GeometricPolygon,
    _GeometricRibbonGraph,
    _RibbonCutSystem,
)
from .continuation import _lifted_path_chain_boundary, _prepare_lifted_path_chain


def _geometric_ribbon_graph(ctx, cover):
    """Build the covering graph with the geometric local rotations."""
    geometry = cover.geometry
    degree = len(cover.fibres[0])
    rotation = {(v, s): [] for v in range(len(geometry.vertices))
                for s in range(degree)}
    edges = []
    for base_edge, ((a, b), permutation) in enumerate(
            zip(geometry.edges, cover.permutations)):
        for sheet, target in enumerate(permutation):
            index = len(edges)
            edge = _GeometricEdge((a, sheet), (b, target), base_edge, sheet)
            edges.append(edge)
            rotation[edge.tail].append((index, 0))
            rotation[edge.head].append((index, 1))
    for vertex, half_edges in rotation.items():
        def direction(half_edge):
            index, endpoint = half_edge
            other = edges[index].head if endpoint == 0 else edges[index].tail
            return ctx.arg(geometry.vertices[other[0]] - geometry.vertices[vertex[0]])
        half_edges.sort(key=direction, reverse=True)
        rotation[vertex] = tuple(half_edges)
    vertices = tuple(sorted(rotation))
    adjacency = {vertex: [] for vertex in vertices}
    for index, edge in enumerate(edges):
        adjacency[edge.tail].append((edge.head, index))
        adjacency[edge.head].append((edge.tail, index))
    seen, queue, tree = {vertices[0]}, deque([vertices[0]]), []
    while queue:
        vertex = queue.popleft()
        for other, edge in adjacency[vertex]:
            if other not in seen:
                seen.add(other)
                queue.append(other)
                tree.append(edge)
    if len(seen) != len(vertices):
        raise ValueError("lifted geometric graph is disconnected")
    faces = _ribbon_boundary_orbits(edges, rotation)
    twice_genus = 2 - len(vertices) + len(edges) - len(faces)
    if twice_genus < 0 or twice_genus % 2:
        raise ValueError("geometric graph has an invalid Euler characteristic")
    return _GeometricRibbonGraph(vertices, tuple(edges), rotation, tuple(tree),
                                 twice_genus // 2, faces)


def _geometric_canonical_polygon(ctx, cover, graph):
    """Retain based canonical words and their additive lifted edge chains."""
    polygon = _canonical_ribbon_polygon(graph)
    chains = []
    for loop in polygon.a_loops + polygon.b_loops:
        current = polygon.root
        coefficients = {}
        for index, orientation in loop:
            edge = graph.edges[index]
            left, right = ((edge.tail, edge.head) if orientation == 1
                           else (edge.head, edge.tail))
            if left != current:
                raise ValueError("geometric canonical path is discontinuous")
            current = right
            coefficients[index] = coefficients.get(index, 0) + orientation
        if current != polygon.root:
            raise ValueError("geometric canonical path is not closed")
        terms = []
        for index, coefficient in sorted(coefficients.items()):
            edge = graph.edges[index]
            if coefficient:
                terms.append((coefficient,
                              cover.continuations[edge.base_edge], edge.sheet))
        chain = _prepare_lifted_path_chain(terms)
        if _lifted_path_chain_boundary(ctx, chain):
            raise ValueError("geometric canonical chain has nonzero boundary")
        chains.append(chain)
    return _GeometricPolygon(polygon, tuple(chains))


def _geometric_sheet_connector(graph, root, sheet):
    """Return a tree path to another sheet over the polygon's base vertex."""
    adjacency = {vertex: [] for vertex in graph.vertices}
    for index in graph.tree_edges:
        edge = graph.edges[index]
        adjacency[edge.tail].append((edge.head, index, 1))
        adjacency[edge.head].append((edge.tail, index, -1))
    return _tree_path(adjacency, root, (root[0], sheet))


def _integer_matrix_rank(matrix):
    """Return the exact rational rank of an integer matrix."""
    rows = [list(map(Fraction, row)) for row in matrix]
    if not rows:
        return 0
    row_count = len(rows)
    column_count = len(rows[0])
    rank = 0
    for column in range(column_count):
        pivot = next((row for row in range(rank, row_count)
                      if rows[row][column]), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        pivot_value = rows[rank][column]
        rows[rank] = [value / pivot_value for value in rows[rank]]
        for row in range(row_count):
            if row == rank or not rows[row][column]:
                continue
            multiplier = rows[row][column]
            rows[row] = [
                value - multiplier * pivot_entry
                for value, pivot_entry in zip(rows[row], rows[rank])]
        rank += 1
        if rank == row_count:
            break
    return rank


def _tree_path(adjacency, start, end):
    """Return oriented edge steps along the unique path in a tree."""
    pending = [(start, ())]
    visited = {start}
    while pending:
        vertex, path = pending.pop()
        if vertex == end:
            return path
        for neighbor, edge, orientation in adjacency[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                pending.append((
                    neighbor, path + ((edge, orientation),)))
    raise ValueError("lifted graph is disconnected")


def _ribbon_boundary_orbits(edges, rotation, included_edges=None):
    """Return oriented boundary orbits of a thickened ribbon subgraph."""
    if included_edges is None:
        included_edges = set(range(len(edges)))
    else:
        included_edges = set(included_edges)
    successor = {}
    for half_edges in rotation.values():
        half_edges = tuple(
            half_edge for half_edge in half_edges
            if half_edge[0] in included_edges)
        if not half_edges:
            continue
        for index, half_edge in enumerate(half_edges):
            successor[half_edge] = half_edges[(index + 1) % len(half_edges)]

    def boundary_step(half_edge):
        edge, endpoint = half_edge
        return successor[(edge, 1 - endpoint)]

    unvisited = set(successor)
    orbits = []
    while unvisited:
        start = next(iter(unvisited))
        current = start
        orbit = []
        while current in unvisited:
            unvisited.remove(current)
            orbit.append(current)
            current = boundary_step(current)
        if current != start:
            raise ValueError("ribbon boundary orbit does not close")
        orbits.append(tuple(orbit))
    return tuple(orbits)


def _ribbon_tree_cotree_cut_system(
        graph, tree_edges=None, cotree_edge_order=None, root=None):
    """Construct a certified one-face cut system of a ribbon graph.

    This is the tree--cotree construction of Eppstein, *Dynamic generators
    of topologically embedded graphs*, SODA 2003, arXiv:cs/0207082, and
    Erickson--Whittlesey, *Greedy optimal homotopy and homology generators*,
    SODA 2005.  The returned loops retain oriented graph-edge paths; unlike a
    homology matrix alone, that data can later support a canonical polygon
    and iterated path integrals.

    ``tree_edges``, ``cotree_edge_order`` and ``root`` are private robustness
    hooks: they allow independently admissible choices to be compared without
    changing the deterministic production choice.  Every supplied choice is
    certified below before it is used.
    """
    edges = graph.edges
    if tree_edges is None:
        tree_edges = tuple(graph.tree_edges)
    else:
        tree_edges = tuple(tree_edges)
    if (len(tree_edges) != len(graph.vertices) - 1
            or len(set(tree_edges)) != len(tree_edges)
            or any(not isinstance(edge, int) or edge < 0
                   or edge >= len(edges) for edge in tree_edges)):
        raise ValueError("tree edges do not form a spanning tree")

    tree_parent = {vertex: vertex for vertex in graph.vertices}

    def tree_find(vertex):
        while tree_parent[vertex] != vertex:
            tree_parent[vertex] = tree_parent[tree_parent[vertex]]
            vertex = tree_parent[vertex]
        return vertex

    for edge_index in tree_edges:
        edge = edges[edge_index]
        left = tree_find(edge.tail)
        right = tree_find(edge.head)
        if left == right:
            raise ValueError("tree edges do not form a spanning tree")
        tree_parent[left] = right
    if len({tree_find(vertex) for vertex in graph.vertices}) != 1:
        raise ValueError("tree edges do not form a spanning tree")

    tree_set = set(tree_edges)
    faces = _ribbon_boundary_orbits(edges, graph.rotation)
    half_edge_face = {
        half_edge: face_index
        for face_index, face in enumerate(faces)
        for half_edge in face
    }

    parent = list(range(len(faces)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    complement = tuple(
        edge_index for edge_index in range(len(edges))
        if edge_index not in tree_set)
    if cotree_edge_order is None:
        cotree_edge_order = complement
    else:
        cotree_edge_order = tuple(cotree_edge_order)
        if (len(cotree_edge_order) != len(complement)
                or set(cotree_edge_order) != set(complement)):
            raise ValueError(
                "cotree edge order must permute the tree complement")

    cotree_edges = []
    for edge_index in cotree_edge_order:
        left = find(half_edge_face[(edge_index, 0)])
        right = find(half_edge_face[(edge_index, 1)])
        if left != right:
            parent[left] = right
            cotree_edges.append(edge_index)
    if len(cotree_edges) != len(faces) - 1:
        raise ValueError("dual complement of ribbon tree is disconnected")

    cotree_set = set(cotree_edges)
    generator_edges = tuple(
        edge_index for edge_index in range(len(edges))
        if edge_index not in tree_set and edge_index not in cotree_set)
    if len(generator_edges) != 2 * graph.genus:
        raise ValueError("tree-cotree complement does not have 2*g edges")

    cut_edges = tree_set | set(generator_edges)
    cut_boundaries = _ribbon_boundary_orbits(
        edges, graph.rotation, cut_edges)
    if len(cut_boundaries) != 1:
        raise ValueError("tree-cotree cut system does not have one face")
    boundary_word = tuple(
        (edge_index, 1 if endpoint == 0 else -1)
        for edge_index, endpoint in cut_boundaries[0]
        if edge_index in generator_edges)
    if len(boundary_word) != 4 * graph.genus:
        raise ValueError("cut-system polygon has the wrong side count")
    for edge_index in generator_edges:
        occurrences = tuple(
            orientation for edge, orientation in boundary_word
            if edge == edge_index)
        if sorted(occurrences) != [-1, 1]:
            raise ValueError("cut-system side pairing is inconsistent")

    adjacency = {vertex: [] for vertex in graph.vertices}
    for edge_index in tree_edges:
        edge = edges[edge_index]
        adjacency[edge.tail].append((edge.head, edge_index, 1))
        adjacency[edge.head].append((edge.tail, edge_index, -1))
    if root is None:
        root = min(graph.vertices)
    elif root not in set(graph.vertices) or root[0] != "base":
        raise ValueError("cut-system root must be a base-sheet vertex")
    loops = []
    cycles = []
    for edge_index in generator_edges:
        edge = edges[edge_index]
        loop = (_tree_path(adjacency, root, edge.tail)
                + ((edge_index, 1),)
                + _tree_path(adjacency, edge.head, root))
        coefficients = [0] * len(edges)
        for loop_edge, orientation in loop:
            coefficients[loop_edge] += orientation
        if any(coefficient not in (-1, 0, 1)
               for coefficient in coefficients):
            raise ValueError("cut-system loop repeats an oriented edge")
        loops.append(loop)
        cycles.append(tuple(coefficients))
    intersection = tuple(
        tuple(_ribbon_cycle_intersection(
            edges, graph.rotation, left, right) for right in cycles)
        for left in cycles)
    if _integer_matrix_rank(intersection) != 2 * graph.genus:
        raise ValueError("cut-system loops have degenerate intersection")
    return _RibbonCutSystem(
        root=root,
        tree_edges=tree_edges,
        cotree_edges=tuple(cotree_edges),
        generator_edges=generator_edges,
        loops=tuple(loops),
        boundary_word=boundary_word,
        intersection=intersection,
    )


def _inverse_oriented_word(word):
    """Return the inverse of a word of ``(symbol, orientation)`` pairs."""
    return tuple((symbol, -orientation)
                 for symbol, orientation in reversed(word))


def _free_reduce_oriented_word(word):
    """Cancel adjacent inverse pairs in an oriented word."""
    reduced = []
    for symbol, orientation in word:
        token = (symbol, orientation)
        if reduced and reduced[-1] == (symbol, -orientation):
            reduced.pop()
        else:
            reduced.append(token)
    return tuple(reduced)


def _brahana_canonical_words(boundary_word):
    """Rewrite an orientable polygon word as canonical commutators.

    This implements Step 2 of the Brahana transformation as stated in
    Lazarus--Pocchiola--Vegter--Verroust, *Computing a Canonical Polygonal
    Schema of an Orientable Triangulated Surface*, SoCG 2001, Section 5.
    Every returned generator remains an explicit free-group word in the
    input sides; no abelianization or homology-only substitution is used.
    """
    remaining = tuple(boundary_word)
    if len(remaining) % 4:
        raise ValueError("orientable polygon must have 4*g sides")
    symbols = {symbol for symbol, unused in remaining}
    for symbol in symbols:
        occurrences = tuple(
            orientation for candidate, orientation in remaining
            if candidate == symbol)
        if sorted(occurrences) != [-1, 1]:
            raise ValueError(
                "orientable polygon sides must occur with opposite signs")

    pairs = []
    while remaining:
        a = remaining[0]
        a_inverse = (a[0], -a[1])
        try:
            a_inverse_index = remaining.index(a_inverse, 1)
        except ValueError:
            raise ValueError("polygon side has no inverse partner")

        crossing = None
        for b_index in range(1, a_inverse_index):
            b = remaining[b_index]
            b_inverse = (b[0], -b[1])
            try:
                b_inverse_index = remaining.index(
                    b_inverse, a_inverse_index + 1)
            except ValueError:
                continue
            crossing = b_index, b_inverse_index
            break
        if crossing is None:
            raise ValueError(
                "polygon word has no interlaced orientable handle")
        b_index, b_inverse_index = crossing
        b = remaining[b_index]
        x1 = remaining[1:b_index]
        x2 = remaining[b_index + 1:a_inverse_index]
        x3 = remaining[a_inverse_index + 1:b_inverse_index]
        x4 = remaining[b_inverse_index + 1:]

        # With M = a X1 b X2 a^-1 X3 b^-1 X4, Brahana's
        # x = a X1 b X2 a^-1 and y = X3 X2 a^-1 give the exact free-word
        # identity M = [x,y] X3 X2 X1 X4.
        x = _free_reduce_oriented_word(
            (a,) + x1 + (b,) + x2 + (a_inverse,))
        y = _free_reduce_oriented_word(x3 + x2 + (a_inverse,))
        pairs.append((x, y))
        remaining = x3 + x2 + x1 + x4

    relator = ()
    for a, b in pairs:
        relator = _free_reduce_oriented_word(
            relator + a + b
            + _inverse_oriented_word(a)
            + _inverse_oriented_word(b))
    if relator != tuple(boundary_word):
        raise ValueError(
            "canonical commutators do not reproduce the polygon relator")
    return tuple(pairs)


def _expand_cut_system_word(word, generator_loops):
    """Expand a free word in cut generators to oriented ribbon edges."""
    result = ()
    for generator, orientation in word:
        try:
            loop = generator_loops[generator]
        except IndexError:
            raise ValueError("canonical word has an invalid generator")
        if orientation == -1:
            loop = _inverse_oriented_word(loop)
        elif orientation != 1:
            raise ValueError("word orientations must be +1 or -1")
        result = _free_reduce_oriented_word(result + loop)
    return result


def _canonical_ribbon_polygon(graph, cut_system=None):
    """Return canonical based loops with their exact polygon relator.

    The word conversion is Brahana's algorithm in Section 5 of
    Lazarus--Pocchiola--Vegter--Verroust (SoCG 2001).  Keeping the free words
    and expanded edge paths is essential for the canonical-dissection formula
    for Riemann constants; a symplectic homology matrix is not sufficient.
    """
    if cut_system is None:
        cut_system = _ribbon_tree_cotree_cut_system(graph)
    generator_index = {
        edge: index for index, edge in enumerate(
            cut_system.generator_edges)
    }
    boundary_word = tuple(
        (generator_index[edge], orientation)
        for edge, orientation in cut_system.boundary_word)
    pairs = _brahana_canonical_words(boundary_word)
    a_words = tuple(pair[0] for pair in pairs)
    b_words = tuple(pair[1] for pair in pairs)
    a_loops = tuple(_expand_cut_system_word(
        word, cut_system.loops) for word in a_words)
    b_loops = tuple(_expand_cut_system_word(
        word, cut_system.loops) for word in b_words)

    loops = a_loops + b_loops
    cycles = []
    for loop in loops:
        coefficients = [0] * len(graph.edges)
        for edge, orientation in loop:
            coefficients[edge] += orientation
        cycles.append(tuple(coefficients))
    intersection = tuple(
        tuple(_ribbon_cycle_intersection(
            graph.edges, graph.rotation, left, right)
            for right in cycles)
        for left in cycles)
    genus = graph.genus
    expected = tuple(tuple(
        1 if row < genus and column == genus + row
        else -1 if column < genus and row == genus + column
        else 0
        for column in range(2 * genus))
        for row in range(2 * genus))
    if intersection != expected:
        raise ValueError(
            "canonical polygon loops have the wrong intersection form")
    return _CanonicalPolygon(
        root=cut_system.root,
        a_words=a_words,
        b_words=b_words,
        a_loops=a_loops,
        b_loops=b_loops,
        relator=boundary_word,
        intersection=intersection,
    )


def _ribbon_cycle_intersection(edges, rotation, left, right):
    """Return the oriented intersection of two graph cycles."""
    numerator = 0
    for vertex, half_edges in rotation.items():
        left_flows = []
        right_flows = []
        for edge_index, endpoint in half_edges:
            orientation = 1 if endpoint == 0 else -1
            left_flows.append(orientation * left[edge_index])
            right_flows.append(orientation * right[edge_index])
        for first in range(len(half_edges)):
            for second in range(first + 1, len(half_edges)):
                numerator += (
                    left_flows[first] * right_flows[second]
                    - right_flows[first] * left_flows[second])
    if numerator % 2:
        raise ValueError("ribbon intersection is not integral")
    # The half-edge rotations above follow positive base-plane monodromy.
    # Their local outward-flow ordering is opposite the complex orientation
    # convention in which a canonical pair satisfies a . b = +1.
    return -numerator // 2
