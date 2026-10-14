#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _         _____ _____
# | ____|__| | ___| |_      _____(_)___ ___|  ___| ____|
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |_  |  _|
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \  _| | |___
# |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|   |_____|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  2017 - today
#
#  Manuel Hradsky manuel.hradsky@uibk.ac.at
#
#  This file is part of EdelweissFE.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissFE.
#  ---------------------------------------------------------------------

import math

import numpy as np

"""
The geometry a segment-to-segment mortar method needs and a node-to-surface one does not: the
broadphase that pairs two surfaces, and the machinery that turns a pair of facets into the polygon
they actually share.

Used by :mod:`~edelweissfe.constraints.mortarcontact`. The counterpart for the node-to-surface
constraints, which need a closest point rather than an overlap, is
:mod:`~edelweissfe.utils.facetcontactgeometry`.

Two facets that face each other are compared in an AUXILIARY PLANE rather than in space: both are
projected into the plane of the non-mortar sub-cell, clipped against each other there, and the
resulting polygon is triangulated so it can be integrated. The curvature of either facet never
enters this step -- it enters only later, when the shape functions are evaluated at the Gauss points
mapped back onto the parent elements.

The clipping is Sutherland & Hodgman (1974). Its restriction is on the CLIP WINDOW, which has to be
convex ("clipping against irregular convex windows"); the subject may be anything, since the
algorithm is "applicable to any polygon, convex or concave, planar or not" (p. 33). Note that
"reentrant" in that paper's title describes the reentrant CODE -- one clipping stage called
repeatedly -- and not a reentrant polygon. Their Appendix B gives a way of decomposing a concave
polygon into convex pieces, which is deliberately not implemented here: the caller checks convexity
and reports a violation instead, because a contact facet that is concave in the auxiliary plane is a
mesh problem worth seeing rather than a case worth accommodating.

The second half of this module maps the integration points back: the decomposition of a curved
facet into the straight sub-cells the clipping runs on, the inverse isoparametric map from the
auxiliary plane into a facet's natural coordinates, the segment quadrature rules, and the
interpolation patches for a frictional history after a mesh refinement.
"""


class BVHNode:
    """A node of the bounding volume hierarchy over the mortar-side facets.

    A leaf carries the facets themselves; an inner node carries two children and the axis-aligned
    box enclosing both. The hierarchy is the standard one of Ericson (2004): it exists so that a
    non-mortar facet can find the few mortar facets that could possibly overlap it without testing
    all of them.

    Attributes
    ----------
    aabbMin
        Lower corner of the enclosing axis-aligned box.
    aabbMax
        Upper corner of the enclosing axis-aligned box.
    left
        Left child, or ``None`` for a leaf.
    right
        Right child, or ``None`` for a leaf.
    facets
        The facets of a leaf, or ``None`` for an inner node.
    """

    def __init__(self, aabbMin, aabbMax, left=None, right=None, facets=None):
        self.aabbMin = aabbMin
        self.aabbMax = aabbMax
        self.left = left
        self.right = right
        self.facets = facets

    def isLeaf(self) -> bool:
        """Whether this node carries facets rather than children."""
        return self.facets is not None


def buildBoundingVolumeHierarchy(facetsWithBounds) -> BVHNode:
    """Build a binary bounding volume hierarchy over facets, by recursive median split.

    Each level splits along the longest axis of the current box, at the median of the facet
    centroids, so the tree stays balanced without needing a cost model.

    Parameters
    ----------
    facetsWithBounds
        One ``(facet, centroid, aabbMin, aabbMax)`` tuple per facet. The list is sorted in place.

    Returns
    -------
    BVHNode
        The root, or ``None`` for an empty input.
    """
    if not facetsWithBounds:
        return None

    mins = np.array([f[2] for f in facetsWithBounds])
    maxs = np.array([f[3] for f in facetsWithBounds])
    aabbMin = np.min(mins, axis=0)
    aabbMax = np.max(maxs, axis=0)

    # Leaf size 2: the point of the tree is to cut the candidate set down to a handful, and below a
    # couple of facets per leaf the box tests cost more than the pair tests they replace.
    if len(facetsWithBounds) <= 2:
        return BVHNode(aabbMin, aabbMax, facets=[f[0] for f in facetsWithBounds])

    extent = aabbMax - aabbMin
    splitAxis = np.argmax(extent)

    facetsWithBounds.sort(key=lambda f: f[1][splitAxis])
    mid = len(facetsWithBounds) // 2

    leftChild = buildBoundingVolumeHierarchy(facetsWithBounds[:mid])
    rightChild = buildBoundingVolumeHierarchy(facetsWithBounds[mid:])

    return BVHNode(aabbMin, aabbMax, left=leftChild, right=rightChild)


def queryBoundingVolumeHierarchy(node: BVHNode, queryMin, queryMax, candidates: list):
    """Collect every facet whose box overlaps the query box.

    Parameters
    ----------
    node
        The root to descend from; ``None`` is accepted and yields nothing.
    queryMin
        Lower corner of the query box.
    queryMax
        Upper corner of the query box.
    candidates
        Appended to in place. This is a conservative superset: a candidate still has to be clipped
        to find out whether it overlaps at all.
    """
    if node is None:
        return

    if not (np.all(queryMin <= node.aabbMax) and np.all(node.aabbMin <= queryMax)):
        return

    if node.isLeaf():
        candidates.extend(node.facets)
    else:
        queryBoundingVolumeHierarchy(node.left, queryMin, queryMax, candidates)
        queryBoundingVolumeHierarchy(node.right, queryMin, queryMax, candidates)


def tangentBasis(normal):
    """Two orthonormal vectors spanning the plane perpendicular to a normal.

    Parameters
    ----------
    normal
        The plane's unit normal.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The two in-plane unit vectors.
    """
    # Seed with whichever global axis is least aligned with the normal, so that the first cross
    # product is well conditioned. 0.9 separates the two cases with room on both sides.
    if abs(normal[0]) < 0.9:
        v = np.array([1.0, 0.0, 0.0])
    else:
        v = np.array([0.0, 1.0, 0.0])
    t1 = np.cross(normal, v)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(normal, t1)
    t2 /= np.linalg.norm(t2)
    return t1, t2


def toPlaneCoordinates(points, origin, t1, t2):
    """Express points lying in a plane in that plane's own 2D coordinates.

    Parameters
    ----------
    points
        The spatial points.
    origin
        The plane's origin.
    t1
        First in-plane unit vector.
    t2
        Second in-plane unit vector.

    Returns
    -------
    np.ndarray
        One ``[x, y]`` pair per point.
    """
    coords = []
    for p in points:
        v = p - origin
        coords.append([np.dot(v, t1), np.dot(v, t2)])
    return np.array(coords)


def sutherlandHodgmanClip(subjectPolygon, clipPolygon):
    """Clip one 2D polygon against another by the algorithm of Sutherland & Hodgman (1974).

    The clip polygon has to be convex; the subject need not be, although this implementation's
    caller requires it of both (see the module docstring).

    Parameters
    ----------
    subjectPolygon
        The polygon being clipped, as ``[x, y]`` vertices in counter-clockwise order.
    clipPolygon
        The convex clipping window, in the same form.

    Returns
    -------
    np.ndarray
        The vertices of the overlap, possibly empty.
    """

    def inside(p, cp1, cp2):
        # Left of the directed edge cp1 -> cp2. Inclusive, so that a vertex lying exactly on the
        # edge is kept rather than dropped and the overlap does not lose a sliver of area.
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) - (cp2[1] - cp1[1]) * (p[0] - cp1[0]) >= -1e-12

    def intersection(cp1, cp2, s, e):
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        den = dc[0] * dp[1] - dc[1] * dp[0]
        if den == 0.0:
            # Parallel edges. Reached only if `inside` classified the two endpoints
            # differently, which its inclusive tolerance normally prevents for
            # exactly collinear points - but "normally" is not "never", and without
            # this guard the result would be inf/NaN and poison the whole cell.
            # Falling back to the crossing endpoint keeps the polygon closed and
            # degenerates the sliver to zero area, which the caller discards.
            return [e[0], e[1]]
        n3 = 1.0 / den
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    outputList = list(subjectPolygon)

    for i in range(len(clipPolygon)):
        cp1 = clipPolygon[i]
        cp2 = clipPolygon[(i + 1) % len(clipPolygon)]

        inputList = outputList
        outputList = []
        if not inputList:
            break

        s = inputList[-1]
        for e in inputList:
            if inside(e, cp1, cp2):
                if not inside(s, cp1, cp2):
                    outputList.append(intersection(cp1, cp2, s, e))
                outputList.append(e)
            elif inside(s, cp1, cp2):
                outputList.append(intersection(cp1, cp2, s, e))
            s = e

    return np.array(outputList)


def triangulatePolygon(polygonCoordinates):
    """Tile a simple convex 2D polygon with a triangle fan from its first vertex.

    Convexity is what makes the fan valid; on a polygon with a reflex vertex it would cover area
    outside the polygon.

    Parameters
    ----------
    polygonCoordinates
        The vertices, in order.

    Returns
    -------
    np.ndarray
        One triangle per entry, each three ``[x, y]`` vertices. Empty for fewer than three vertices.
    """
    triangles = []
    if len(polygonCoordinates) < 3:
        return triangles
    for i in range(1, len(polygonCoordinates) - 1):
        triangles.append([polygonCoordinates[0], polygonCoordinates[i], polygonCoordinates[i + 1]])
    return np.array(triangles)


def clipLineSegments(slaveCoordinates: np.ndarray, masterCoordinates: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Overlap of a 2D master segment projected onto a non-mortar segment's own line.

    The 2D counterpart of the auxiliary-plane clipping above: projecting the master segment onto the
    non-mortar segment's line is the one-dimensional degeneration of projecting a master facet into
    a non-mortar sub-cell's plane.

    Parameters
    ----------
    slaveCoordinates
        Shape ``(2, 2)``: the two endpoints of the non-mortar sub-cell.
    masterCoordinates
        Shape ``(2, 2)``: the two endpoints of the mortar sub-cell.

    Returns
    -------
    tuple[float, float, np.ndarray]
        ``(start, end, tangent)`` -- the overlap in the arc-length coordinate of the non-mortar
        sub-cell, and its unit tangent, so that a point of the overlap is
        ``slaveCoordinates[0] + s * tangent``. Where there is no overlap, ``(0.0, 0.0, tangent)`` is
        returned; the caller discards the cell on the vanishing interval, so the tangent is then
        meaningless but harmless.
    """
    slaveVector = slaveCoordinates[1] - slaveCoordinates[0]
    slaveLength = np.linalg.norm(slaveVector)
    if slaveLength < 1e-14:
        return 0.0, 0.0, np.zeros(2)

    tangent = slaveVector / slaveLength

    # Project the master endpoints onto the non-mortar line coordinate s in [0, slaveLength].
    s0 = np.dot(masterCoordinates[0] - slaveCoordinates[0], tangent)
    s1 = np.dot(masterCoordinates[1] - slaveCoordinates[0], tangent)

    start = max(0.0, min(s0, s1))
    end = min(slaveLength, max(s0, s1))

    if end - start < 1e-12:
        return 0.0, 0.0, tangent

    return start, end, tangent


def mapPlaneToNatural(el, coords_2d, point_2d, max_iter=10, tol=1e-12):
    """Map a 2D local plane coordinate point_2d to the element's natural space.

    Returns (local_coords, converged).

    Termination. This is a (Gauss-)Newton solve for the natural coordinate, so the
    quantity it actually drives to zero is the UPDATE, not the residual. For 2D
    surface elements the two coincide as long as the point lies in the element's
    plane - which it does, since both facets are projected into the same auxiliary
    plane. For 1D line elements in 2D space they do NOT: the Gauss point is
    generated on the SLAVE line and then mapped into the MASTER element, so with an
    open contact gap g the smallest attainable residual is exactly g and a residual
    test can never be satisfied. The least-squares step nevertheless returns the
    exact orthogonal projection after the first pass; without an update-based test
    the loop just burns all max_iter passes to return it. Measured on the test
    suite with a residual criterion instead: 189 of 780476 calls "failed" this way,
    every one of them a 2D call at an open gap, every returned coordinate exact.

    The update test is also the scale-invariant one - the natural coordinate is
    dimensionless, whereas the residual carries the model's length unit.

    `converged` is True when either criterion was met. It is False only when the
    iteration ran out of passes or hit a singular Jacobian - i.e. when the returned
    coordinate is the last iterate rather than a solution. Nothing downstream can
    tell those apart on its own (the value looks like any other), so the caller
    counts them and reports them; see the `gp_projection_failed` diagnostic in
    computeMortarCouplingMatrices. That the closest-point projection need not be
    solvable at all, and under which conditions it fails, is the subject of
    Konyukhov & Schweizerhof (2008); the local Newton used here is the standard one
    of the surrounding solids.

    Implementation note. The linear algebra is written out in scalars instead of
    calling np.linalg.norm / np.linalg.solve. Those are 2x2 and 2-vector operations
    called several hundred thousand times per increment, where the numpy dispatch
    overhead is many times the arithmetic (measured on hertz_hex20_fine: 5.0 s in
    np.linalg.solve and 3.3 s in np.linalg.norm out of 31 s of segmentation).

    This is NOT bit-identical to what it replaces, and the difference was measured
    rather than assumed. The 2x2 solve follows LAPACK dgesv's operation order -
    partial pivoting with IDAMAX tie-breaking towards the first row, the reciprocal
    of the pivot formed once as dgetf2 does, then forward/back substitution - and
    still differs from np.linalg.solve for 39 % of random 2x2 systems, by up to
    5.5e-15 relative (dividing instead of using the reciprocal is worse: 9.9e-12).
    math.sqrt(r0*r0 + r1*r1) differs from np.linalg.norm for 8 % of random 2-vectors,
    by 1 ulp. Propagated through the Newton iteration and the segment quadrature,
    that shows up as ~1e-12 relative in D and C and ~2e-10 in the converged
    multipliers, with the active set unchanged (hertz_hex20_medium). The whole
    verification suite passes with unchanged tolerances, but this is a deliberate
    trade of exact reproducibility against roughly 10 % wall clock - if the
    reproducibility matters more, this function is the single place to revert.
    """
    el_type = el.elType.upper()

    # 1D line elements
    if "LINE" in el_type:
        xi = 0.0
        local_coords = np.zeros(1)
        for _ in range(max_iter):
            local_coords[0] = xi
            N = el.getShapeFunctions(local_coords)
            x_mapped = N @ coords_2d
            r0 = x_mapped[0] - point_2d[0]
            r1 = x_mapped[1] - point_2d[1]
            if math.sqrt(r0 * r0 + r1 * r1) < tol:
                return local_coords, True
            dN = el.getShapeFunctionDerivatives(local_coords)  # shape (1, n_nodes)
            J = dN @ coords_2d  # shape (1, dim_of_coords_2d)
            j0, j1 = J[0, 0], J[0, 1]
            J_norm = j0 * j0 + j1 * j1
            if J_norm < 1e-14:
                return local_coords, False
            delta = (j0 * r0 + j1 * r1) / J_norm
            xi -= delta
            local_coords[0] = xi
            if abs(delta) < tol:
                return local_coords, True
        return local_coords, False

    # 2D surface elements (Quads and Triangles)
    if "TRI" in el_type:
        xi = eta = 1.0 / 3.0
    else:
        xi = eta = 0.0
    local_coords = np.array([xi, eta])

    for _ in range(max_iter):
        local_coords[0] = xi
        local_coords[1] = eta
        N = el.getShapeFunctions(local_coords)
        x_mapped = N @ coords_2d
        r0 = x_mapped[0] - point_2d[0]
        r1 = x_mapped[1] - point_2d[1]
        if math.sqrt(r0 * r0 + r1 * r1) < tol:
            return local_coords, True
        dN = el.getShapeFunctionDerivatives(local_coords)  # shape (2, n_nodes)
        Jm = dN @ coords_2d  # (2, 2); the Jacobian is its transpose
        a00, a01 = Jm[0, 0], Jm[1, 0]
        a10, a11 = Jm[0, 1], Jm[1, 1]

        # 2x2 LU with partial pivoting, in LAPACK dgesv's operation order.
        if abs(a10) > abs(a00):  # IDAMAX picks the first maximum, so ties keep row 0
            a00, a01, a10, a11 = a10, a11, a00, a01
            b0, b1 = r1, r0
        else:
            b0, b1 = r0, r1
        if a00 == 0.0:
            return local_coords, False  # singular column: np.linalg.solve would raise
        # dgetf2 forms the RECIPROCAL of the pivot once and multiplies by it; that is
        # not the same in floating point as dividing, so dividing here would already
        # cost bit-identity with np.linalg.solve.
        m = a10 * (1.0 / a00)
        u11 = a11 - m * a01
        if u11 == 0.0:
            return local_coords, False
        y1 = b1 - m * b0
        d1 = y1 / u11
        d0 = (b0 - a01 * d1) / a00

        xi -= d0
        eta -= d1
        local_coords[0] = xi
        local_coords[1] = eta
        if math.sqrt(d0 * d0 + d1 * d1) < tol:
            return local_coords, True
    return local_coords, False


# Decomposition of contact facets into linear sub-cells for the mortar
# segmentation (polygon clipping). The clip polygons must be simple convex
# polygons, so curved (quadratic) facets are subdivided into linear cells
# using their mid-side nodes before the clipping, which is what the segment-based
# integration needs. The curvature of the facet enters only
# through the shape function evaluation at the mapped Gauss points, never
# through the clipping geometry itself.
#
# The three quadratic decompositions below are those of Puso, Laursen & Solberg
# (2008), Fig. 3: the nine-node patch into four quadrilaterals (b), the six-node
# patch into four linear triangles (d), and the serendipity eight-node patch into
# "four linear triangle sub-segments and one quadrilateral segment" (f). They note
# what this costs and why it is acceptable: the geometrically exact description of
# the quadratic surface is given up for the integration, but "through consideration
# of simple affine maps between the parameterizations for these linear sub-segments
# and their parent quadratic elements, we retain the ability to easily integrate
# inner products of higher order shape functions".
subCellMap = {
    "CONLINE2": [[0, 1]],
    "CONLINE3": [[0, 2], [2, 1]],
    "CONQUAD4": [[0, 1, 2, 3]],
    "CONTRI3": [[0, 1, 2]],
    "CONQUAD8": [[0, 4, 7], [4, 1, 5], [5, 2, 6], [7, 6, 3], [4, 5, 6, 7]],
    "CONQUAD9": [[0, 4, 8, 7], [4, 1, 5, 8], [8, 5, 2, 6], [7, 8, 6, 3]],
    "CONTRI6": [[0, 3, 5], [3, 4, 5], [3, 1, 4], [5, 4, 2]],
}


def isConvexPolygon(poly_2d, tol: float = 1e-14) -> bool:
    """Whether a planar polygon given in order is convex.

    BOTH sub-cells of a segmentation pair have to be convex, for two different
    reasons:

    slave (clip window)   Sutherland-Hodgman clips against the half-plane of each
        clip edge, and the intersection of those half-planes is the polygon itself
        only if the polygon is convex. A reflex vertex therefore removes area that
        genuinely belongs to the facet, silently (measured: -2/3 of the area for a
        constructed reflex quad, see 02_polygon_clipping).

    master (subject)      Sutherland-Hodgman itself accepts any subject polygon
        ("applicable to any polygon, convex or concave", Sutherland & Hodgman 1974,
        p. 33), but the overlap it returns inherits the subject's reflex vertices,
        and `triangulatePolygon` fans from vertex 0 with a per-triangle absolute area.
        For a non-convex overlap those fan triangles reach outside the polygon and
        the area is OVER-counted (measured: 0.375 instead of 0.125, i.e. +200 %).

    Triangles cannot be non-convex, so the requirement only bites on the middle
    quad of a CONQUAD8 and the four quads of a CONQUAD9.
    """
    n = len(poly_2d)
    if n < 4:
        return True
    sign = 0.0
    for i in range(n):
        a = poly_2d[(i + 1) % n] - poly_2d[i]
        b = poly_2d[(i + 2) % n] - poly_2d[(i + 1) % n]
        cross = a[0] * b[1] - a[1] * b[0]
        if abs(cross) < tol:
            continue  # collinear vertex: neither convex nor reflex
        if sign == 0.0:
            sign = cross
        elif cross * sign < 0.0:
            return False
    return True


def curvedFacetHistoryPatches(facets: list, dim: int) -> list:
    """The old non-mortar facets of a mortar contact as patches for
    :func:`~edelweissfe.utils.contactfriction.historyInterpolation`: a point is mapped into a facet's
    natural coordinates (the isoparametric map inverted iteratively, as Wriggers 2006 notes for
    Eq. 14.85), and the nodal history is interpolated with the facet's shape functions there
    (Eq. 14.86). A point outside the facet's reference domain is not located by it."""

    patches = []
    for el in facets:
        coordinates = np.array([node.coordinates for node in el.nodes])[:, :dim]
        if dim == 3:
            normal = np.cross(coordinates[1] - coordinates[0], coordinates[-1] - coordinates[0])
            normal /= np.linalg.norm(normal)
            origin = coordinates.mean(axis=0)
            t1, t2 = tangentBasis(normal)
            planeCoordinates = toPlaneCoordinates(coordinates, origin, t1, t2)
        else:
            origin = t1 = t2 = None
            planeCoordinates = coordinates

        def locate(
            point, el=el, coordinates=coordinates, planeCoordinates=planeCoordinates, origin=origin, t1=t1, t2=t2
        ):
            planePoint = point[:dim] if origin is None else toPlaneCoordinates(point[None, :dim], origin, t1, t2)[0]
            local, converged = mapPlaneToNatural(el, planeCoordinates, planePoint)
            if not converged or not isInsideReferenceElement(el, local, 1e-6):
                return None
            shapeFunctions = el.getShapeFunctions(local)
            return float(np.linalg.norm(shapeFunctions @ coordinates - point[:dim])), list(el.nodes), shapeFunctions

        patches.append((coordinates.mean(axis=0), locate))
    return patches


def isInsideReferenceElement(el, local_coords, tol: float = 1e-10) -> bool:
    """Whether a natural coordinate lies in the element's reference domain (up to ``tol``).

    Used to catch a Gauss point that the segmentation attributed to a facet which does not
    contain it. The clipping works on the straight sub-cells of a facet (see subCellMap), so
    wherever a quadratic facet has a CURVED edge the region between the chord and the true edge
    belongs to one facet geometrically and to its neighbour in the clipping. Mapped back into the
    wrong facet, such a point lands outside the reference domain and its shape functions are
    extrapolated rather than interpolated.
    """
    el_type = el.elType.upper()
    if "TRI" in el_type:
        xi, eta = local_coords[0], local_coords[1]
        return xi >= -tol and eta >= -tol and xi + eta <= 1.0 + tol
    return bool(np.all(np.abs(local_coords) <= 1.0 + tol))


def getSubCells(el) -> list[list[int]]:
    """Return the linear sub-cell decomposition (local node indices) of a contact facet."""
    el_type = el.elType.upper()
    if el_type not in subCellMap:
        raise NotImplementedError(f"No linear sub-cell decomposition defined for element type '{el_type}'.")
    return subCellMap[el_type]


def facetNormal(coords: np.ndarray) -> np.ndarray:
    """Unnormalized area-weighted normal of a flat linear facet (3 or 4 corner nodes).

    Used for the AUXILIARY PLANE of the segmentation only - one plane per slave
    sub-cell, and the sub-cells are linear by construction, so this is the exact
    normal of their plane rather than an approximation.

    It is deliberately NOT what computeNormals builds the nodal normal n_I from:
    one constant direction per facet cannot distinguish a mid-side node from its
    corners, which costs an order of convergence on a curved slave surface. Note
    also that its length is the facet AREA, whereas computeNormals sums unit
    normals and so carries no area weighting at all. See computeNormals.
    """
    if len(coords) == 3:
        return 0.5 * np.cross(coords[1] - coords[0], coords[2] - coords[0])
    return 0.5 * np.cross(coords[2] - coords[0], coords[3] - coords[1])


# 3-point Gauss-Legendre rule (degree 5) on [-1, 1] for 1D line segments
lineGaussPoints = np.array(
    [
        [-np.sqrt(0.6)],
        [0.0],
        [np.sqrt(0.6)],
    ]
)
lineGaussWeights = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])


# 7-point symmetric Gauss rule (degree 5) on the reference triangle, the rule
# tabulated by Dunavant (1985) for p = 5, n_g = 7, written here in closed form.
# 7 points per integration cell are enough here,
# since the nonlinear projection between the auxiliary plane and the curved
# element surfaces raises the polynomial degree of the integrand. That is also the
# experience of Puso & Laursen (2004), Section 3.1, who report that "seven point
# integration was used for most example problems and appears to be more than
# sufficient when compared to higher order nine and thirteen point schemes" -- with
# the caveat, which holds here too, that "the choice of integration rule is somewhat
# problem dependent".
_triangleGaussA = (6.0 - np.sqrt(15.0)) / 21.0
_triangleGaussB = (6.0 + np.sqrt(15.0)) / 21.0
triangleGaussPoints = np.array(
    [
        [1.0 / 3.0, 1.0 / 3.0],
        [_triangleGaussA, _triangleGaussA],
        [_triangleGaussA, 1.0 - 2.0 * _triangleGaussA],
        [1.0 - 2.0 * _triangleGaussA, _triangleGaussA],
        [_triangleGaussB, _triangleGaussB],
        [_triangleGaussB, 1.0 - 2.0 * _triangleGaussB],
        [1.0 - 2.0 * _triangleGaussB, _triangleGaussB],
    ]
)
triangleGaussWeights = np.array(
    [
        9.0 / 80.0,
        (155.0 - np.sqrt(15.0)) / 2400.0,
        (155.0 - np.sqrt(15.0)) / 2400.0,
        (155.0 - np.sqrt(15.0)) / 2400.0,
        (155.0 + np.sqrt(15.0)) / 2400.0,
        (155.0 + np.sqrt(15.0)) / 2400.0,
        (155.0 + np.sqrt(15.0)) / 2400.0,
    ]
)
