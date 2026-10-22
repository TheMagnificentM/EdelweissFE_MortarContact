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
"""
The geometry of the segment-to-segment mortar method (:mod:`~edelweissfe.constraints.mortarcontact`):
the broadphase that pairs the two surfaces, and the steps that turn a pair of facets into the polygon
they share. The counterpart of the node- and segment-based constraints, which need a closest point
rather than an overlap, is :mod:`~edelweissfe.utils.facetcontactgeometry`.

Two facets are compared in an auxiliary plane: both are projected into the plane of the non-mortar
sub-cell, clipped against each other there, and the overlap is triangulated for the integration. The
curvature of the facets enters only through the shape functions at the Gauss points mapped back onto
the elements.

The clipping is that of Sutherland & Hodgman (1974). The clip window has to be convex; the subject
may be any polygon (p. 33). Concave polygons are not decomposed (their Appendix B) but reported by
the caller, since a contact facet that is concave in the auxiliary plane is a mesh problem.

The second half maps the integration points back: the decomposition of a curved facet into the
straight sub-cells the clipping runs on, the inverse isoparametric map from the auxiliary plane into
a facet's natural coordinates, and the interpolation patches for a frictional history after a mesh
refinement.
"""

import math

import numpy as np


class BVHNode:
    """A node of the bounding volume hierarchy over the mortar-side facets.

    A leaf carries the facets themselves; an inner node carries two children and the axis-aligned
    box enclosing both. The hierarchy is the standard one of Ericson (2005): it exists so that a
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
        # The crossing of the subject edge s -> e with the line of the clip edge, written along the
        # subject edge: x = s + t (e - s) with t = d_s / (d_s - d_e) from the signed distances (times
        # the clip edge length) of s and e to the clip line. This is the intersection of the two lines
        # (Popp et al. 2010, Eq. (A19), case 3), with t clamped to [0, 1] -- this implementation's
        # step. Where a subject edge lies on a clip edge (conforming meshes), round-off can still put
        # s and e on different sides, and the intersection of two nearly parallel lines can then lie
        # far outside both facets; clamped, the vertex stays on the subject edge and the sliver it
        # bounds has (almost) no area.
        d_s = (cp2[0] - cp1[0]) * (s[1] - cp1[1]) - (cp2[1] - cp1[1]) * (s[0] - cp1[0])
        d_e = (cp2[0] - cp1[0]) * (e[1] - cp1[1]) - (cp2[1] - cp1[1]) * (e[0] - cp1[0])
        if d_s == d_e:
            # Parallel edges, the endpoint that crossed is the best available point.
            return [e[0], e[1]]
        t = min(max(d_s / (d_s - d_e), 0.0), 1.0)
        return [s[0] + t * (e[0] - s[0]), s[1] + t * (e[1] - s[1])]

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
    if len(polygonCoordinates) < 3:
        return np.zeros((0, 3, 2))
    triangles = []
    for i in range(1, len(polygonCoordinates) - 1):
        triangles.append([polygonCoordinates[0], polygonCoordinates[i], polygonCoordinates[i + 1]])
    return np.array(triangles)


def mapPlaneToNatural(el, planeCoords, planePoint, maxIterations=10, tol=1e-12):
    """Map a point of the auxiliary plane into the natural coordinates of a facet, by the inverse of
    the facet's isoparametric map in the plane (Newton's method).

    Parameters
    ----------
    el
        The contact element.
    planeCoords
        The plane (2D) coordinates of its nodes.
    planePoint
        The plane point.
    maxIterations, tol
        The iteration limit, and the tolerance on the residual and on the update.

    Returns
    -------
    tuple
        ``(natural coordinates, converged)``. ``converged`` is False if the iteration ran out of
        passes or met a singular Jacobian; the caller reports such points.

    For a line element the residual of a point off the line cannot vanish, so the iteration stops on
    the (dimensionless) update, which also makes the test independent of the length unit. The 2x2
    solve is written out in scalars, with partial pivoting in the operation order of LAPACK's dgesv,
    since this function is called several hundred thousand times per segmentation and the dispatch
    of ``np.linalg.solve`` would cost more than the arithmetic.
    """
    elType = el.elType.upper()

    if "LINE" in elType:
        xi = 0.0
        localCoords = np.zeros(1)
        for _ in range(maxIterations):
            localCoords[0] = xi
            N = el.getShapeFunctions(localCoords)
            xMapped = N @ planeCoords
            r0 = xMapped[0] - planePoint[0]
            r1 = xMapped[1] - planePoint[1]
            if math.sqrt(r0 * r0 + r1 * r1) < tol:
                return localCoords, True
            J = el.getShapeFunctionDerivatives(localCoords) @ planeCoords  # (1, 2)
            j0, j1 = J[0, 0], J[0, 1]
            jSquared = j0 * j0 + j1 * j1
            if jSquared < 1e-14:
                return localCoords, False
            delta = (j0 * r0 + j1 * r1) / jSquared
            xi -= delta
            localCoords[0] = xi
            if abs(delta) < tol:
                return localCoords, True
        return localCoords, False

    if "TRI" in elType:
        xi = eta = 1.0 / 3.0
    else:
        xi = eta = 0.0
    localCoords = np.array([xi, eta])

    for _ in range(maxIterations):
        localCoords[0] = xi
        localCoords[1] = eta
        N = el.getShapeFunctions(localCoords)
        xMapped = N @ planeCoords
        r0 = xMapped[0] - planePoint[0]
        r1 = xMapped[1] - planePoint[1]
        if math.sqrt(r0 * r0 + r1 * r1) < tol:
            return localCoords, True
        Jm = el.getShapeFunctionDerivatives(localCoords) @ planeCoords  # (2, 2); the Jacobian is its transpose
        a00, a01 = Jm[0, 0], Jm[1, 0]
        a10, a11 = Jm[0, 1], Jm[1, 1]

        # 2x2 LU with partial pivoting (ties keep row 0); the reciprocal of the pivot is formed once.
        if abs(a10) > abs(a00):
            a00, a01, a10, a11 = a10, a11, a00, a01
            b0, b1 = r1, r0
        else:
            b0, b1 = r0, r1
        if a00 == 0.0:
            return localCoords, False
        m = a10 * (1.0 / a00)
        u11 = a11 - m * a01
        if u11 == 0.0:
            return localCoords, False
        y1 = b1 - m * b0
        d1 = y1 / u11
        d0 = (b0 - a01 * d1) / a00

        xi -= d0
        eta -= d1
        localCoords[0] = xi
        localCoords[1] = eta
        if math.sqrt(d0 * d0 + d1 * d1) < tol:
            return localCoords, True
    return localCoords, False


# The decomposition of contact facets into linear sub-cells for the clipping, which needs convex
# polygons. The quadratic decompositions are those of Puso, Laursen & Solberg (2008), Fig. 3: the
# nine-node patch into four quadrilaterals, the six-node patch into four triangles, and the eight-node
# patch into four triangles and one quadrilateral. The curvature of the facet enters only through the
# shape functions at the mapped Gauss points.
subCellMap = {
    "CONLINE2": [[0, 1]],
    "CONLINE3": [[0, 2], [2, 1]],
    "CONQUAD4": [[0, 1, 2, 3]],
    "CONTRI3": [[0, 1, 2]],
    "CONQUAD8": [[0, 4, 7], [4, 1, 5], [5, 2, 6], [7, 6, 3], [4, 5, 6, 7]],
    "CONQUAD9": [[0, 4, 8, 7], [4, 1, 5, 8], [8, 5, 2, 6], [7, 8, 6, 3]],
    "CONTRI6": [[0, 3, 5], [3, 4, 5], [3, 1, 4], [5, 4, 2]],
}


def isConvexPolygon(polygon, tol: float = 1e-14) -> bool:
    """Whether a planar polygon given in order is convex.

    Both sub-cells of a segmentation pair have to be: the clip window, because the clipping cuts
    with the half-plane of each of its edges, which gives the window itself only if it is convex (a
    reflex vertex silently removes area); the subject, because the overlap inherits its reflex
    vertices and the fan triangulation of :func:`triangulatePolygon` then counts area outside the
    polygon. Only the quadrilateral sub-cells of CONQUAD4, CONQUAD8 and CONQUAD9 can be non-convex.
    """
    n = len(polygon)
    if n < 4:
        return True
    sign = 0.0
    for i in range(n):
        a = polygon[(i + 1) % n] - polygon[i]
        b = polygon[(i + 2) % n] - polygon[(i + 1) % n]
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


def isInsideReferenceElement(el, localCoords, tol: float = 1e-10) -> bool:
    """Whether a natural coordinate lies in the element's reference domain (up to ``tol``).

    Catches a Gauss point attributed to a facet that does not contain it: the clipping works on the
    straight sub-cells, so where a quadratic facet has a curved edge, the region between chord and
    edge belongs to the facet geometrically and to its neighbour in the clipping. Mapped back into
    the wrong facet, the point lands outside the reference domain.
    """
    elType = el.elType.upper()
    if "TRI" in elType:
        xi, eta = localCoords[0], localCoords[1]
        return xi >= -tol and eta >= -tol and xi + eta <= 1.0 + tol
    return bool(np.all(np.abs(localCoords) <= 1.0 + tol))


def getSubCells(el) -> list[list[int]]:
    """The linear sub-cell decomposition (local node indices) of a contact facet."""
    elType = el.elType.upper()
    if elType not in subCellMap:
        raise NotImplementedError(f"No linear sub-cell decomposition defined for element type '{elType}'.")
    return subCellMap[elType]
