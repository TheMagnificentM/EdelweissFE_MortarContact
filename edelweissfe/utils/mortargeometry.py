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


def toSpatialCoordinates(planeCoordinates, origin, t1, t2):
    """Map 2D in-plane coordinates back to spatial points; the inverse of :func:`toPlaneCoordinates`.

    Parameters
    ----------
    planeCoordinates
        One ``[x, y]`` pair per point.
    origin
        The plane's origin.
    t1
        First in-plane unit vector.
    t2
        Second in-plane unit vector.

    Returns
    -------
    np.ndarray
        The spatial points.
    """
    points = []
    for c in planeCoordinates:
        points.append(origin + c[0] * t1 + c[1] * t2)
    return np.array(points)


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
