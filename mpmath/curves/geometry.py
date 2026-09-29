"""Base-plane graphs for numerical algebraic-curve computations."""

from ._records import _PlaneGraph


def _stable_complex_order(ctx, values, tolerance):
    """Order by real then imaginary part, grouping numerical real-part ties."""
    ordered = sorted(values, key=lambda z: (ctx.re(z), ctx.im(z)))
    result = []
    first = 0
    while first < len(ordered):
        last = first + 1
        anchor = ctx.re(ordered[first])
        while last < len(ordered) and ctx.re(ordered[last]) - anchor <= tolerance:
            last += 1
        result.extend(sorted(ordered[first:last], key=ctx.im))
        first = last
    return tuple(result)


def _clip_voronoi_halfplane(ctx, polygon, site, other, tolerance):
    """Clip one cell and reject a numerically collapsed polygon."""
    normal = ctx.conj(other - site)
    offset = (abs(other)**2 - abs(site)**2) / 2

    def distance(z):
        value = ctx.re(normal * z) - offset
        return ctx.zero if abs(value) <= tolerance else value

    clipped = []
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        da, db = distance(a), distance(b)
        if da <= 0:
            clipped.append(a)
        if da < 0 < db or db < 0 < da:
            clipped.append(a + (b - a) * da / (da - db))
    if len(clipped) < 3:
        raise ctx.NoConvergence("Voronoi cell is numerically degenerate")
    return clipped


def _register_voronoi_vertex(ctx, index, support, z, positions, tolerance):
    """Identify a vertex by its incident sites and consistent position."""
    if index not in support or len(support) < 3:
        raise ctx.NoConvergence(
            "Voronoi vertex has inconsistent incident sites")
    if support in positions:
        if abs(positions[support] - z) > 100 * tolerance:
            raise ctx.NoConvergence("Voronoi vertex could not be identified")
    else:
        positions[support] = z


def _finish_voronoi_cell(ctx, cell):
    """Remove a repeated closing vertex and check the cell boundary."""
    if cell[-1] == cell[0]:
        cell.pop()
    if len(cell) < 3 or len(set(cell)) != len(cell):
        raise ctx.NoConvergence("Voronoi cell has invalid vertex identifications")
    return tuple(cell)


def _check_voronoi_disk(ctx, vertices, edges, cells, tolerance, scale):
    """Certify the bounded graph's Euler count and resolved edges."""
    if len(vertices) - len(edges) + len(cells) != 1:
        raise ctx.NoConvergence("bounded Voronoi cells do not form a disk")
    if any(abs(vertices[a] - vertices[b]) <= tolerance * scale for a, b in edges):
        raise ctx.NoConvergence("Voronoi edge is numerically degenerate")


def _voronoi_plane_graph(ctx, branch_values):
    """Construct the bounded Voronoi cells of finite projection values.

    Six exterior sites enclose the original sites. Half-plane clipping and
    vertex identification use normalized coordinates at the working precision;
    configurations too close to distinguish raise rather than guess topology.
    This is numerical geometry, not an interval-certified construction.
    """
    points = tuple(ctx.convert(z) for z in branch_values)
    if not points or any(not ctx.isfinite(z) for z in points):
        raise ValueError("finite nonempty branch values are required")
    center = ctx.fsum(points) / len(points)
    scale = max(abs(z - center) for z in points) or ctx.one
    tolerance = 100 * ctx.sqrt(ctx.eps)
    branches = _stable_complex_order(
        ctx, tuple((z - center) / scale for z in points), tolerance)
    if any(abs(a - b) <= 100 * tolerance
           for i, a in enumerate(branches) for b in branches[i + 1:]):
        raise ctx.NoConvergence("branch values are too close for Voronoi geometry")
    sites = branches + tuple(
        ctx.mpf(3) / 2 * ctx.exp(2 * ctx.j * ctx.pi * k / 6)
        for k in range(6))
    positions, cells = {}, []
    for index, site in enumerate(branches):
        polygon = [ctx.mpc(x, y) for x, y in
                   ((-8, -8), (8, -8), (8, 8), (-8, 8))]
        for other_index, other in enumerate(sites):
            if index == other_index:
                continue
            polygon = _clip_voronoi_halfplane(
                ctx, polygon, site, other, tolerance)
        cell = []
        for z in polygon:
            distances = [abs(z - s)**2 for s in sites]
            nearest = min(distances)
            support = tuple(k for k, d in enumerate(distances)
                            if abs(d - nearest) <= 20 * tolerance)
            _register_voronoi_vertex(ctx, index, support, z, positions, tolerance)
            if not cell or cell[-1] != support:
                cell.append(support)
        cells.append(_finish_voronoi_cell(ctx, cell))
    supports = tuple(sorted(positions))
    indices = {key: i for i, key in enumerate(supports)}
    vertices = tuple(center + scale * positions[key] for key in supports)
    cells = tuple(tuple(indices[key] for key in cell) for cell in cells)
    edges = tuple(sorted({tuple(sorted((a, b))) for cell in cells
                          for a, b in zip(cell, cell[1:] + cell[:1])}))
    _check_voronoi_disk(ctx, vertices, edges, cells, tolerance, scale)
    return _PlaneGraph(vertices, edges, cells,
                       tuple(center + scale * z for z in branches), supports)
