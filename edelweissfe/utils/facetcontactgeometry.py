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
#  Matthias Neuner matthias.neuner@uibk.ac.at
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
from scipy.spatial import cKDTree

"""
Exact gap function, gradient, and full Hessian (including the second-derivative term arising
from the cross-product-then-normalize/rotate-then-normalize construction of the facet normal) for
a slave point against a flat contact facet (Tria3 in 3D, Line2 in 2D), expressed directly in terms
of the current nodal coordinates.

Both facets are exactly flat (a plane through 3 points, or a line through 2 points), so the
curvature/second-fundamental-form contribution present in curved-surface contact vanishes
identically -- the only surviving second-derivative term is a pose-dependent nonlinearity of the
normal's own construction from its defining nodes' positions, not a curvature effect.

The closed forms below were derived by hand and cross-verified against exact symbolic
differentiation (SymPy) at many random, non-degenerate configurations before being transcribed
here -- see the derivation/verification scripts referenced in the class docstring of
:mod:`~edelweissfe.constraints.nodetodeformablesurfacepenalty` for the underlying methodology.
Do not hand-edit these formulas without re-verifying them the same way; this kind of
normalize/rotate second-derivative algebra is very easy to get subtly wrong.
"""


def _skew(v: np.ndarray) -> np.ndarray:
    """The skew-symmetric cross-product matrix of a 3-vector, such that ``_skew(v) @ x == v x x``."""
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def tria3GapGradientHessian(
    xs: np.ndarray, x1: np.ndarray, x2: np.ndarray, x3: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Exact gap, gradient, and Hessian of a slave point against a flat Tria3 facet (3D).

    The facet plane is spanned by ``x1, x2, x3`` (in this local order); the outward normal is
    ``cross(x2-x1, x3-x1)``, normalized. The gap is positive outside the facet's half-space,
    negative when penetrating.

    Parameters
    ----------
    xs, x1, x2, x3
        Current coordinates (each shape ``(3,)``) of the slave point and the facet's three nodes,
        in this fixed local order.

    Returns
    -------
    tuple[float, numpy.ndarray, numpy.ndarray]
        The gap ``g``, its gradient ``w`` (shape ``(12,)``, blocks ``[xs, x1, x2, x3]``), and its
        Hessian ``H`` (shape ``(12, 12)``, same block order).
    """

    r = xs - x1
    e1 = x2 - x1
    e2 = x3 - x1
    c = np.cross(e1, e2)
    m = np.linalg.norm(c)
    n = c / m
    g = n.dot(r)

    blocks = ("xs", "x1", "x2", "x3")

    dr_dBlock = {"xs": np.eye(3), "x1": -np.eye(3), "x2": np.zeros((3, 3)), "x3": np.zeros((3, 3))}
    dc_dBlock = {"xs": np.zeros((3, 3)), "x1": -_skew(x2 - x3), "x2": -_skew(e2), "x3": _skew(e1)}

    projectorOntoTangentPlane = np.eye(3) - np.outer(n, n)
    dn_dBlock = {k: (projectorOntoTangentPlane @ dc_dBlock[k]) / m for k in blocks}
    dm_dBlock = {k: n @ dc_dBlock[k] for k in blocks}  # row vector

    w_dBlock = {k: dn_dBlock[k].T @ r + dr_dBlock[k].T @ n for k in blocks}
    w = np.concatenate([w_dBlock[k] for k in blocks])

    # The tangential (in-plane) component of r: normal-projected-out, used below since the
    # curvature-like second-derivative pieces only couple through r's in-plane part.
    rTangential = r - g * n

    # d(dc_dBlock[a])/d(block b) -- dc_dBlock[a] is +/- skew(u_a) for a fixed linear combination
    # u_a of the facet's nodes; this is the constant Jacobian du_a/d(block b) for each a.
    du_dBlock = {
        "x1": {"xs": np.zeros((3, 3)), "x1": np.zeros((3, 3)), "x2": -np.eye(3), "x3": np.eye(3)},  # u = x3-x2
        "x2": {"xs": np.zeros((3, 3)), "x1": -np.eye(3), "x2": np.zeros((3, 3)), "x3": np.eye(3)},  # u = x3-x1
        "x3": {"xs": np.zeros((3, 3)), "x1": -np.eye(3), "x2": np.eye(3), "x3": np.zeros((3, 3))},  # u = x2-x1
    }
    dcSign = {"x1": +1.0, "x2": -1.0, "x3": +1.0}  # dc_dBlock[a] = dcSign[a] * skew(u_a)

    H = np.zeros((12, 12))
    blockSlice = {"xs": slice(0, 3), "x1": slice(3, 6), "x2": slice(6, 9), "x3": slice(9, 12)}

    H[blockSlice["xs"], blockSlice["xs"]] = 0.0
    for b in ("x1", "x2", "x3"):
        H[blockSlice["xs"], blockSlice[b]] = dn_dBlock[b]

    for a in ("x1", "x2", "x3"):
        for b in blocks:
            # d(dn_dBlock[a])/d(block b), contracted with r on the normal's own index -- the
            # exact second derivative of the cross-product-then-normalize construction of n.
            crossNormalizeTerm = -(1.0 / m) * (
                np.outer(dm_dBlock[a], dn_dBlock[b].T @ r) + g * (dc_dBlock[a].T @ dn_dBlock[b])
            )
            skewArgumentTerm = dcSign[a] * (1.0 / m) * (_skew(rTangential) @ du_dBlock[a][b])
            normalizeDenominatorTerm = -(1.0 / m**2) * np.outer(rTangential @ dc_dBlock[a], dm_dBlock[b])

            d2n_a_contractedWithR = crossNormalizeTerm + skewArgumentTerm + normalizeDenominatorTerm
            H[blockSlice[a], blockSlice[b]] = (
                d2n_a_contractedWithR + dn_dBlock[a].T @ dr_dBlock[b] + dr_dBlock[a].T @ dn_dBlock[b]
            )

    return g, w, H


def line2GapGradientHessian(xs: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Exact gap, gradient, and Hessian of a slave point against a flat Line2 facet (2D).

    The facet edge runs from ``x1`` to ``x2``; the outward normal is the edge direction rotated
    by -90 degrees, normalized. The gap is positive outside the facet's half-plane, negative when
    penetrating.

    Parameters
    ----------
    xs, x1, x2
        Current coordinates (each shape ``(2,)``) of the slave point and the facet's two nodes,
        in this fixed local order.

    Returns
    -------
    tuple[float, numpy.ndarray, numpy.ndarray]
        The gap ``g``, its gradient ``w`` (shape ``(6,)``, blocks ``[xs, x1, x2]``), and its
        Hessian ``H`` (shape ``(6, 6)``, same block order).
    """

    r = xs - x1
    e = x2 - x1
    length = np.linalg.norm(e)
    eHat = e / length
    rotateMinus90 = np.array([[0.0, 1.0], [-1.0, 0.0]])
    n = rotateMinus90 @ eHat
    g = n.dot(r)

    blocks = ("xs", "x1", "x2")

    dr_dBlock = {"xs": np.eye(2), "x1": -np.eye(2), "x2": np.zeros((2, 2))}
    de_dBlock = {"xs": np.zeros((2, 2)), "x1": -np.eye(2), "x2": np.eye(2)}

    projectorOntoNormal = np.eye(2) - np.outer(eHat, eHat)
    dEHat_dBlock = {k: (projectorOntoNormal @ de_dBlock[k]) / length for k in blocks}
    dn_dBlock = {k: rotateMinus90 @ dEHat_dBlock[k] for k in blocks}
    dLength_dBlock = {k: eHat @ de_dBlock[k] for k in blocks}  # row vector

    w_dBlock = {k: dn_dBlock[k].T @ r + dr_dBlock[k].T @ n for k in blocks}
    w = np.concatenate([w_dBlock[k] for k in blocks])

    # r rotated into the eHat/normal frame -- required because dn_dBlock = rotateMinus90 @
    # dEHat_dBlock, so contracting r with n's own index is equivalent to contracting
    # rotateMinus90^T @ r with eHat's index instead (rotateMinus90 is constant, applied on the
    # left, so it commutes out of the contraction this way).
    rRotated = rotateMinus90.T @ r
    rRotatedDotEHat = rRotated.dot(eHat)

    H = np.zeros((6, 6))
    blockSlice = {"xs": slice(0, 2), "x1": slice(2, 4), "x2": slice(4, 6)}

    for a in blocks:
        for b in blocks:
            normalizeTerm = -(1.0 / length) * (
                np.outer(eHat @ de_dBlock[a], dEHat_dBlock[b].T @ rRotated)
                + rRotatedDotEHat * (de_dBlock[a].T @ dEHat_dBlock[b])
                + np.outer(dEHat_dBlock[a].T @ rRotated, dLength_dBlock[b])
            )
            H[blockSlice[a], blockSlice[b]] = (
                normalizeTerm + dn_dBlock[a].T @ dr_dBlock[b] + dr_dBlock[a].T @ dn_dBlock[b]
            )

    return g, w, H


def tria3ClosestPoint(xs: np.ndarray, x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> tuple[np.ndarray, float]:
    """Closest point on the (closed) triangle (x1, x2, x3) to xs, clamped to the triangle's
    interior/edge/vertex regions (Ericson's real-time-collision-detection region test), as
    barycentric weights (w1, w2, w3) with all w >= 0 and sum(w) == 1, plus the distance."""

    e1 = x2 - x1
    e2 = x3 - x1
    r1 = xs - x1

    d1 = e1.dot(r1)
    d2 = e2.dot(r1)
    if d1 <= 0.0 and d2 <= 0.0:
        weights = np.array([1.0, 0.0, 0.0])  # vertex x1
    else:
        r2 = xs - x2
        d3 = e1.dot(r2)
        d4 = e2.dot(r2)
        if d3 >= 0.0 and d4 <= d3:
            weights = np.array([0.0, 1.0, 0.0])  # vertex x2
        else:
            r3 = xs - x3
            d5 = e1.dot(r3)
            d6 = e2.dot(r3)
            vc = d1 * d4 - d3 * d2
            va = d3 * d6 - d5 * d4
            vb = d5 * d2 - d1 * d6
            if d6 >= 0.0 and d5 <= d6:
                weights = np.array([0.0, 0.0, 1.0])  # vertex x3
            elif vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
                t = d1 / (d1 - d3)  # edge x1-x2
                weights = np.array([1.0 - t, t, 0.0])
            elif vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
                t = d2 / (d2 - d6)  # edge x1-x3
                weights = np.array([1.0 - t, 0.0, t])
            elif va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
                t = (d4 - d3) / ((d4 - d3) + (d5 - d6))  # edge x2-x3
                weights = np.array([0.0, 1.0 - t, t])
            else:
                denom = 1.0 / (va + vb + vc)  # interior
                beta = vb * denom
                gamma = vc * denom
                weights = np.array([1.0 - beta - gamma, beta, gamma])

    closestPoint = weights[0] * x1 + weights[1] * x2 + weights[2] * x3
    return weights, float(np.linalg.norm(xs - closestPoint))


def line2ClosestPoint(xs: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> tuple[np.ndarray, float]:
    """Closest point on the (closed) segment (x1, x2) to xs, as parametric weights (1-t, t) with
    t clamped to [0, 1], plus the distance."""

    e = x2 - x1
    t = np.clip((xs - x1).dot(e) / e.dot(e), 0.0, 1.0)
    weights = np.array([1.0 - t, t])
    closestPoint = weights[0] * x1 + weights[1] * x2
    return weights, float(np.linalg.norm(xs - closestPoint))


def closestFacetCandidates(queryPoints: np.ndarray, facetCoords: list, searchDistance: float | None) -> list:
    """For each query point, the facets that could possibly be its closest one.

    Replaces an exhaustive slave x facet sweep with a spatial query, WITHOUT changing which facet
    is selected. The exhaustive form called the clamped closest-point routine once per
    slave-facet pair -- 778,000 calls on the anchor pry-out model, measured at 5.98 s per
    connectivity update, which is why the solver had to be told to search only every few hundred
    increments.

    The bound that makes this exact: let ``F*`` be the truly closest facet, at point-to-closed-
    domain distance ``d*``, with centroid ``C*`` and radius ``r*`` (its largest centroid-to-vertex
    distance). A triangle's centroid lies inside its own closed domain, so the distance to the
    facet owning the nearest centroid is at most ``d0``, the distance to that centroid -- hence
    ``d* <= d0``. And ``|x - C*| <= d* + r* <= d0 + rMax``. So every facet that could win, or
    even tie, has its centroid inside a ball of radius ``d0 + rMax``, and querying that ball
    cannot miss it.

    Candidates are returned in ascending facet index, so the caller's "first strict minimum"
    selection picks exactly the facet the exhaustive sweep would have. The result is bit-identical,
    not merely equivalent.

    Parameters
    ----------
    queryPoints
        Current coordinates of the points to be projected, shape ``(nPoints, nDim)``. Slave nodes
        for the node-based formulation, quadrature points for the integrated one.
    facetCoords
        Current coordinates of each facet's nodes.
    searchDistance
        The caller's cut-off, if any. A facet beyond it is rejected anyway, so the query radius
        is capped accordingly.

    Returns
    -------
    list
        One ascending list of candidate facet indices per query point.
    """

    if not facetCoords:
        return [[] for _ in range(len(queryPoints))]

    stacked = np.asarray(facetCoords, dtype=float)  # (nFacets, nNodesPerFacet, nDim)
    centroids = stacked.mean(axis=1)
    radii = np.linalg.norm(stacked - centroids[:, None, :], axis=2).max(axis=1)
    rMax = float(radii.max())

    tree = cKDTree(centroids)
    nearestCentroidDistance, _ = tree.query(queryPoints, k=1)

    radius = nearestCentroidDistance + rMax
    if searchDistance is not None:
        # A facet farther than searchDistance is rejected by the caller regardless, and its
        # closed domain is at least |x - C| - r away, so nothing within searchDistance can have
        # its centroid beyond searchDistance + rMax.
        radius = np.minimum(radius, searchDistance + rMax)

    # The bound above is exact in real arithmetic and query_ball_point includes its boundary, so
    # a facet that exactly ties -- the ordinary case on a structured or symmetric contact mesh --
    # sits precisely ON the radius, where the rounding of the sum that produced it can land an
    # ulp low and drop it from the candidate list. That would silently pick a different facet
    # than the exhaustive sweep, against what this docstring promises. A few ulps of relative
    # margin restores the superset property; it widens the ball by a distance many orders below
    # any mesh dimension, so it admits no facet that the exhaustive sweep would not also see.
    radius = radius * (1.0 + 8.0 * np.finfo(float).eps)

    ballCandidates = tree.query_ball_point(queryPoints, radius)

    # The ball is sized by rMax, the largest facet radius anywhere on the surface, so on a surface
    # of uniform facets it is about as tight as a centroid bound can be -- and on a refined one it
    # is not tight at all. Measured on the h-adaptive anchor pry-out model, whose concrete top
    # surface carries 1.2 mm facets next to 34 mm ones: rMax = 24.54 mm against a mean nearest-
    # centroid distance of 0.78 mm, so the radius is 97 % rMax and admits 457 of 4212 facets for
    # every point. The same call on that model's other contact surface, 192 facets with
    # rMax = 2.33 mm, admits 8.6. One global rMax makes the whole surface pay for its largest
    # element.
    #
    # So reject, per facet, what its own radius already rules out. Facet i's closed domain is at
    # least |x - C_i| - r_i away, and d* <= d0 from the bound above, so a facet with
    # |x - C_i| - r_i > d0 is strictly farther than the winner and can neither win nor tie. It is
    # the same argument the ball radius rests on, applied with each facet's own r_i instead of the
    # largest one, and it removes only facets the exhaustive sweep would have evaluated and
    # discarded -- the selected facet, and the caller's ascending-index tie-breaking among equally
    # close ones, are untouched. The comparison keeps the ulp margin for exactly the reason the
    # radius above does: a tie sits precisely on the boundary.
    tolerance = 1.0 + 8.0 * np.finfo(float).eps

    candidates = []
    for p, ballCandidate in enumerate(ballCandidates):
        indices = np.fromiter(ballCandidate, dtype=np.intp, count=len(ballCandidate))
        centroidDistance = np.linalg.norm(centroids[indices] - queryPoints[p], axis=1)
        surviving = indices[centroidDistance - radii[indices] <= nearestCentroidDistance[p] * tolerance]
        surviving.sort()
        candidates.append(surviving.tolist())

    return candidates


def closestFacet(
    point: np.ndarray,
    candidates: list,
    facetCoords: list,
    closestPointFunction,
    tieTolerance: float,
    lowestIndexOnTie: bool = True,
) -> tuple[int | None, np.ndarray | None, float, bool]:
    """The closest of the candidate facets to ``point``, and whether that choice is a tie.

    Facets whose distance lies within ``tieTolerance`` of the smallest one count as equally close,
    and of those the one with the lowest index is taken, so the choice does not depend on the
    rounding of the current coordinates; with ``lowestIndexOnTie=False`` the first facet of exactly
    the smallest distance is taken instead. A tie is the ordinary case, not an accident: when the
    closest point lies on an edge or a vertex shared by several facets, every one of them returns
    that same point at the same distance. It is where the projection onto a faceted surface is not
    unique (Wriggers 2006, Remark 4.1 and Fig. 4.4, p. 60; Section 10.1 and Fig. 10.8, pp. 318-319).

    Parameters
    ----------
    point
        The point to be projected.
    candidates
        The candidate facet indices, in ascending order (see :func:`closestFacetCandidates`).
    facetCoords
        The current coordinates of each facet's nodes.
    closestPointFunction
        :func:`tria3ClosestPoint` or :func:`line2ClosestPoint`.
    tieTolerance
        Distances within this of the smallest one count as equal.
    lowestIndexOnTie
        Whether a tie goes to the lowest index, or to the first facet of exactly the smallest distance.

    Returns
    -------
    tuple
        The index of the chosen facet (``None`` without candidates), its clamped closest-point
        weights, the distance, and whether another facet was equally close.
    """

    results = [(i, *closestPointFunction(point, *facetCoords[i])) for i in candidates]
    if not results:
        return None, None, np.inf, False
    smallest = min(distance for _, _, distance in results)
    equallyClose = [(i, weights, distance) for i, weights, distance in results if distance <= smallest + tieTolerance]
    if lowestIndexOnTie:
        facet, weights, distance = equallyClose[0]
    else:
        facet, weights, distance = next(result for result in results if result[2] == smallest)
    return facet, weights, distance, len(equallyClose) > 1


def edgeOrVertexNormal(point: np.ndarray, closestPoint: np.ndarray, facetNormal: np.ndarray, tolerance: float):
    """The contact normal of a point whose closest point lies on an edge or vertex shared by several
    master facets.

    No facet normal is the normal there, since the edge lies between facets of different
    orientation; a projection onto a C0 surface has to include its edges and corner points
    (Konyukhov & Schweizerhof 2008, CMAME 197, 3045-3056). The normal is taken along the distance
    vector instead, as for the node-to-edge and
    node-to-vertex contact elements (Wriggers 2006, Eq. 9.123, p. 250, and Eq. 9.131, p. 251),
    oriented like the facet normal, so that the gap ``n . (point - closestPoint)`` is the signed
    distance ``+-|point - closestPoint|`` whichever of the equally close facets was chosen.

    Where the distance vector has no tangential part beyond ``tolerance`` -- the point lies on the
    edge, or above a flat stretch of the surface, where the facet normal already is the direction of
    the distance vector -- the facet normal is kept as it is. The switch is continuous: at the
    boundary of an edge's region the distance vector is parallel to the facet normal.
    """

    distanceVector = point - closestPoint
    tangential = distanceVector - (facetNormal @ distanceVector) * facetNormal
    if float(np.linalg.norm(tangential)) <= tolerance:
        return facetNormal
    sign = 1.0 if float(facetNormal @ distanceVector) >= 0.0 else -1.0
    return sign * distanceVector / float(np.linalg.norm(distanceVector))


def facetSurfaceBoundary(facetNodes: list) -> tuple[set, set]:
    """The boundary of a surface made of flat facets: the edges that belong to one facet only, and
    the nodes on them. In 2D, where a facet is a segment, the boundary edges are empty and the
    boundary nodes are the ends of the chain (nodes of one segment only).

    Parameters
    ----------
    facetNodes
        The nodes of each facet, in the facet's order.

    Returns
    -------
    tuple
        The boundary edges as frozensets of two nodes, and the set of boundary nodes.
    """

    if facetNodes and len(facetNodes[0]) == 2:
        count = {}
        for nodes in facetNodes:
            for node in nodes:
                count[node] = count.get(node, 0) + 1
        return set(), {node for node, n in count.items() if n == 1}

    count = {}
    for nodes in facetNodes:
        for a in range(len(nodes)):
            edge = frozenset((nodes[a], nodes[(a + 1) % len(nodes)]))
            count[edge] = count.get(edge, 0) + 1
    boundaryEdges = {edge for edge, n in count.items() if n == 1}
    return boundaryEdges, {node for edge in boundaryEdges for node in edge}


def liesOnSurfaceBoundary(nodes: list, weights: np.ndarray, boundary: tuple[set, set]) -> bool:
    """Whether the closest point given by its clamped ``weights`` on the facet with ``nodes`` lies on
    the boundary of the surface (see :func:`facetSurfaceBoundary`): on a boundary vertex, or on a
    boundary edge."""

    boundaryEdges, boundaryNodes = boundary
    support = [node for node, weight in zip(nodes, weights) if weight != 0.0]
    if len(support) == 1:
        return support[0] in boundaryNodes
    if len(support) == 2 and len(nodes) == 3:
        return frozenset(support) in boundaryEdges
    return False


def facetHistoryPatches(facets: list) -> list:
    """The old slave facets of a node-based contact as patches for
    :func:`~edelweissfe.utils.contactfriction.historyInterpolation`: a point is located on its
    closest facet, and the history is interpolated from the nodes of the facet's parent face with
    the parent face's shape functions at that location (Wriggers 2006, Eq. 14.86) -- from the facet's
    own nodes with its barycentric weights where the facet carries no parent face."""

    from edelweissfe.utils.parentfacegeometry import parentFaceShapeFunctions

    patches = []
    for facet in facets:
        coordinates = np.array([node.coordinates for node in facet.nodes])
        closestPoint = tria3ClosestPoint if len(coordinates) == 3 else line2ClosestPoint

        def locate(point, facet=facet, coordinates=coordinates, closestPoint=closestPoint):
            weights, distance = closestPoint(point, *coordinates)
            if facet.parentFaceType is None:
                return distance, list(facet.nodes), weights
            parametric = weights @ facet.vertexParametricCoords
            return distance, list(facet.parentFaceNodes), parentFaceShapeFunctions(facet.parentFaceType, parametric)

        patches.append((coordinates.mean(axis=0), locate))
    return patches


def facetQuadraturePointPatches(facets: list, nQuadraturePoints: int, pointKey) -> list:
    """The old slave facets of the integrated contact as patches for
    :func:`~edelweissfe.utils.contactfriction.historyInterpolation`, whose history sits at the
    quadrature points rather than at nodes. Over each old facet, a linear field in its barycentric
    coordinates is fitted to the values at its quadrature points by least squares (Wriggers 2006,
    Eqs. 14.32-14.33, with the facet as the patch; a constant one for a single point), and evaluated
    where the new point lies on the facet. ``pointKey(facet, q)`` gives the key of quadrature point
    ``q`` of an old facet."""

    from edelweissfe.utils.parentfacegeometry import facetQuadratureRule

    patches = []
    for facet in facets:
        coordinates = np.array([node.coordinates for node in facet.nodes])
        closestPoint = tria3ClosestPoint if len(coordinates) == 3 else line2ClosestPoint
        barycentric, _ = facetQuadratureRule(len(coordinates), nQuadraturePoints)
        nTerms = len(coordinates) if nQuadraturePoints >= len(coordinates) else 1
        fit = np.linalg.pinv(np.hstack([np.ones((len(barycentric), 1)), barycentric[:, 1:nTerms]]))
        keys = [pointKey(facet, q) for q in range(nQuadraturePoints)]

        def locate(point, coordinates=coordinates, closestPoint=closestPoint, fit=fit, keys=keys, nTerms=nTerms):
            weights, distance = closestPoint(point, *coordinates)
            return distance, keys, np.concatenate([[1.0], weights[1:nTerms]]) @ fit

        patches.append((coordinates.mean(axis=0), locate))
    return patches
