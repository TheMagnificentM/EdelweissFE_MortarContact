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

import itertools

import numpy as np
from scipy.spatial import cKDTree

from edelweissfe.utils.rotations import skewMatrix


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
    dc_dBlock = {"xs": np.zeros((3, 3)), "x1": -skewMatrix(x2 - x3), "x2": -skewMatrix(e2), "x3": skewMatrix(e1)}

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
            skewArgumentTerm = dcSign[a] * (1.0 / m) * (skewMatrix(rTangential) @ du_dBlock[a][b])
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


def rowDot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise dot product over the last axis, accumulated left to right onto +0.0 (so a sum of
    negative zeros is +0.0), with separate multiplications and additions. The order is fixed here,
    not left to a BLAS: ``np.dot`` may fuse multiply and add on one CPU and not on another, and then
    differs in the last bit."""

    dot = 0.0 + a[..., 0] * b[..., 0]
    for k in range(1, a.shape[-1]):
        dot = dot + a[..., k] * b[..., k]
    return dot


def tria3ClosestPoint(xs: np.ndarray, x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> tuple[np.ndarray, float]:
    """Closest point on the (closed) triangle (x1, x2, x3) to xs, clamped to the triangle's
    interior/edge/vertex regions (Ericson's real-time-collision-detection region test), as
    barycentric weights (w1, w2, w3) with all w >= 0 and sum(w) == 1, plus the distance.

    The single-pair form of :func:`tria3ClosestPoints`, which it calls, so that the two can never
    disagree."""

    weights, distances = tria3ClosestPoints(xs[None], x1[None], x2[None], x3[None])
    return weights[0], float(distances[0])


def tria3ClosestPoints(xs: np.ndarray, x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closest points of many (point, triangle) pairs at once, see :func:`tria3ClosestPoint`.

    Ericson's region test, evaluated for all pairs with array operations instead of one Python call
    per pair. Each pair is assigned the region whose condition, in the order the branch-by-branch
    test checks them, it satisfies first (a NaN, which satisfies no condition, falls through to the
    interior). Every operation acts on each pair alone, so a pair's result does not depend on the
    other pairs in the batch.

    Parameters
    ----------
    xs
        The points, shape ``(nPairs, 3)``.
    x1, x2, x3
        The nodes of each pair's triangle, in its local order, each of shape ``(nPairs, 3)``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The barycentric weights of the closest points, shape ``(nPairs, 3)``, and the distances,
        shape ``(nPairs,)``.
    """

    e1 = x2 - x1
    e2 = x3 - x1
    r1 = xs - x1
    r2 = xs - x2
    r3 = xs - x3

    d1 = rowDot(e1, r1)
    d2 = rowDot(e2, r1)
    d3 = rowDot(e1, r2)
    d4 = rowDot(e2, r2)
    d5 = rowDot(e1, r3)
    d6 = rowDot(e2, r3)
    vc = d1 * d4 - d3 * d2
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6

    weights = np.zeros((len(d1), 3))
    unassigned = np.ones(len(d1), dtype=bool)

    def claim(condition: np.ndarray) -> np.ndarray:
        """The still unassigned pairs that satisfy ``condition``, which are assigned hereby."""
        claimed = unassigned & condition
        unassigned[claimed] = False
        return claimed

    # The regions in the order the scalar test checks them: a pair belongs to the first it satisfies.
    weights[claim((d1 <= 0.0) & (d2 <= 0.0)), 0] = 1.0  # vertex x1
    weights[claim((d3 >= 0.0) & (d4 <= d3)), 1] = 1.0  # vertex x2
    weights[claim((d6 >= 0.0) & (d5 <= d6)), 2] = 1.0  # vertex x3

    # A degenerate triangle divides by zero; as in a pairwise evaluation, the result is inf or NaN.
    with np.errstate(divide="ignore", invalid="ignore"):
        edge = claim((vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0))  # edge x1-x2
        t = d1[edge] / (d1[edge] - d3[edge])
        weights[edge, 0] = 1.0 - t
        weights[edge, 1] = t

        edge = claim((vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0))  # edge x1-x3
        t = d2[edge] / (d2[edge] - d6[edge])
        weights[edge, 0] = 1.0 - t
        weights[edge, 2] = t

        d4MinusD3 = d4 - d3
        d5MinusD6 = d5 - d6
        edge = claim((va <= 0.0) & (d4MinusD3 >= 0.0) & (d5MinusD6 >= 0.0))  # edge x2-x3
        t = d4MinusD3[edge] / (d4MinusD3[edge] + d5MinusD6[edge])
        weights[edge, 1] = 1.0 - t
        weights[edge, 2] = t

        interior = unassigned
        denom = 1.0 / (va[interior] + vb[interior] + vc[interior])
        beta = vb[interior] * denom
        gamma = vc[interior] * denom
        weights[interior, 0] = 1.0 - beta - gamma
        weights[interior, 1] = beta
        weights[interior, 2] = gamma

    closestPoints = weights[:, 0:1] * x1 + weights[:, 1:2] * x2 + weights[:, 2:3] * x3
    toClosestPoints = xs - closestPoints
    return weights, np.sqrt(rowDot(toClosestPoints, toClosestPoints))


def line2ClosestPoint(xs: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> tuple[np.ndarray, float]:
    """Closest point on the (closed) segment (x1, x2) to xs, as parametric weights (1-t, t) with
    t clamped to [0, 1], plus the distance.

    The single-pair form of :func:`line2ClosestPoints`, which it calls, so that the two can never
    disagree."""

    weights, distances = line2ClosestPoints(xs[None], x1[None], x2[None])
    return weights[0], float(distances[0])


def line2ClosestPoints(xs: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closest points of many (point, segment) pairs at once, see :func:`line2ClosestPoint`.

    Parameters
    ----------
    xs
        The points, shape ``(nPairs, 2)``.
    x1, x2
        The nodes of each pair's segment, in its local order, each of shape ``(nPairs, 2)``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The weights of the closest points, shape ``(nPairs, 2)``, and the distances, shape
        ``(nPairs,)``.
    """

    e = x2 - x1
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.clip(rowDot(xs - x1, e) / rowDot(e, e), 0.0, 1.0)
    weights = np.stack([1.0 - t, t], axis=1)
    toClosestPoints = xs - (weights[:, 0:1] * x1 + weights[:, 1:2] * x2)
    return weights, np.sqrt(rowDot(toClosestPoints, toClosestPoints))


def tria3Projection(xs: np.ndarray, x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> tuple[float, float, bool]:
    """Barycentric-like in-plane coordinates (alpha, beta) of the projection of xs onto the
    (possibly non-orthogonal) basis spanned by (x2-x1, x3-x1), and whether that projection falls
    inside the triangle.

    Unlike :func:`tria3ClosestPoint`, the projection is not clamped to the triangle."""

    e1 = x2 - x1
    e2 = x3 - x1
    r = xs - x1
    n = np.cross(e1, e2)
    n = n / np.linalg.norm(n)
    rTangential = r - r.dot(n) * n

    A = np.array([[e1.dot(e1), e1.dot(e2)], [e1.dot(e2), e2.dot(e2)]])
    b = np.array([e1.dot(rTangential), e2.dot(rTangential)])
    alpha, beta = np.linalg.solve(A, b)

    inside = alpha >= 0.0 and beta >= 0.0 and (alpha + beta) <= 1.0
    return alpha, beta, inside


def line2Projection(xs: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> tuple[float, bool]:
    """Parametric coordinate t of the projection of xs onto the edge (x1,x2), and whether that
    projection falls inside the segment.

    Unlike :func:`line2ClosestPoint`, the projection is not clamped to the segment."""

    e = x2 - x1
    t = (xs - x1).dot(e) / e.dot(e)
    return t, 0.0 <= t <= 1.0


def facetNormalAndMeasure(coords: np.ndarray) -> tuple[np.ndarray, float]:
    """The (non-unit-normalized only in intermediate steps) outward normal and measure (area for a
    Tria3 facet, length for a Line2 facet) of a flat facet, as a function of its current node
    coordinates.

    Parameters
    ----------
    coords
        Array of shape ``(3, 3)`` (Tria3, 3D) or ``(2, 2)`` (Line2, 2D) with the facet's current
        node coordinates in its fixed local order.

    Returns
    -------
    tuple[numpy.ndarray, float]
        The outward unit normal, and the facet's measure (area or length).
    """

    nNodes, domainSize = coords.shape

    if nNodes == 3 and domainSize == 3:
        e1 = coords[1] - coords[0]
        e2 = coords[2] - coords[0]
        c = np.cross(e1, e2)
        cNorm = np.linalg.norm(c)
        return c / cNorm, 0.5 * cNorm

    elif nNodes == 2 and domainSize == 2:
        e = coords[1] - coords[0]
        eNorm = np.linalg.norm(e)
        # Outward normal is e rotated by -90 degrees, consistent with a counter-clockwise
        # (node 1 -> node 2) traversal of the solid's boundary.
        n = np.array([e[1], -e[0]]) / eNorm
        return n, eNorm

    raise ValueError(f"facetNormalAndMeasure: unsupported facet shape {coords.shape}.")


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

    # Evaluated for all (point, ball candidate) pairs at once, with the same arithmetic per pair as
    # one point at a time.
    queryPoints = np.asarray(queryPoints)
    pointIndices, facetIndices = candidatePairs(ballCandidates)
    centroidDistances = np.linalg.norm(centroids[facetIndices] - queryPoints[pointIndices], axis=1)
    surviving = centroidDistances - radii[facetIndices] <= nearestCentroidDistance[pointIndices] * tolerance
    pointIndices, facetIndices = pointIndices[surviving], facetIndices[surviving]

    # Ascending facet index per point; the pairs stay grouped by point.
    order = np.lexsort((facetIndices, pointIndices))
    survivingFacets = facetIndices[order].tolist()
    ends = np.cumsum(np.bincount(pointIndices, minlength=len(queryPoints))).tolist()
    return [survivingFacets[first:end] for first, end in zip([0] + ends[:-1], ends)]


def candidatePairs(candidatesPerPoint: list) -> tuple[np.ndarray, np.ndarray]:
    """Flatten per-point candidate lists into (point, facet) index pairs.

    Parameters
    ----------
    candidatesPerPoint
        One list of candidate facet indices per point, as :func:`closestFacetCandidates` returns.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The point and the facet index of every pair, ordered by point and, per point, in the order
        of its candidate list.
    """

    counts = np.fromiter(map(len, candidatesPerPoint), dtype=np.intp, count=len(candidatesPerPoint))
    pointIndices = np.repeat(np.arange(len(candidatesPerPoint)), counts)
    facetIndices = np.fromiter(
        itertools.chain.from_iterable(candidatesPerPoint), dtype=np.intp, count=int(counts.sum())
    )
    return pointIndices, facetIndices


def facetClosestPoints(points: np.ndarray, facetCoords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closest points of many (point, flat facet) pairs at once: :func:`tria3ClosestPoints` for
    Tria3 facets in 3D, :func:`line2ClosestPoints` for Line2 facets in 2D.

    Parameters
    ----------
    points
        The points, shape ``(nPairs, nDim)``.
    facetCoords
        The node coordinates of each pair's facet, shape ``(nPairs, nNodesPerFacet, nDim)``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The weights of the closest points, shape ``(nPairs, nNodesPerFacet)``, and the distances,
        shape ``(nPairs,)``.
    """

    if facetCoords.shape[1:] == (3, 3):
        return tria3ClosestPoints(points, facetCoords[:, 0], facetCoords[:, 1], facetCoords[:, 2])
    if facetCoords.shape[1:] == (2, 2):
        return line2ClosestPoints(points, facetCoords[:, 0], facetCoords[:, 1])
    raise ValueError(f"facetClosestPoints: unsupported facet shape {facetCoords.shape[1:]}.")


def closestFacets(
    queryPoints: np.ndarray, facetCoords: list, candidatesPerPoint: list
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each query point, the closest of its candidate facets, the clamped closest point's
    weights in it, and the distance.

    The selection is exactly that of scanning each point's candidates in their order and keeping a
    facet whenever its distance is strictly below the best so far, starting from infinity: the
    first candidate at the minimal distance wins, and a point whose candidates are all at an
    infinite or NaN distance (or that has none) gets no facet. All (point, candidate) pairs are
    evaluated at once, see :func:`facetClosestPoints`, so the weights and distances are
    bit-identical to those of the scan.

    Parameters
    ----------
    queryPoints
        The points, shape ``(nPoints, nDim)``.
    facetCoords
        Current coordinates of each facet's nodes, all of the same shape.
    candidatesPerPoint
        One list of candidate facet indices per point, see :func:`closestFacetCandidates`.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Per point the closest facet's index, -1 if none; its weights, shape
        ``(nPoints, nNodesPerFacet)``, zero if none; and the distance, infinite if none.
    """

    queryPoints = np.asarray(queryPoints, dtype=float)
    nPoints = len(queryPoints)
    closestFacet = np.full(nPoints, -1, dtype=np.intp)
    closestDistance = np.full(nPoints, np.inf)
    if not facetCoords:
        return closestFacet, np.zeros((nPoints, 0)), closestDistance

    stacked = np.asarray(facetCoords, dtype=float)  # (nFacets, nNodesPerFacet, nDim)
    closestWeights = np.zeros((nPoints, stacked.shape[1]))

    pointIndices, facetIndices = candidatePairs(candidatesPerPoint)
    weights, distances = facetClosestPoints(queryPoints[pointIndices], stacked[facetIndices])

    # The scan's strict "<" against a best that starts at infinity never accepts an infinite or a
    # NaN distance, and among equal distances keeps the first.
    acceptable = distances < np.inf
    pairs = np.flatnonzero(acceptable)
    if len(pairs) == 0:
        return closestFacet, closestWeights, closestDistance
    np.minimum.at(closestDistance, pointIndices[pairs], distances[pairs])
    pairs = pairs[distances[pairs] == closestDistance[pointIndices[pairs]]]
    # Pairs are ordered by point, and per point in candidate order: keep each point's first.
    pairs = pairs[np.r_[True, pointIndices[pairs[1:]] != pointIndices[pairs[:-1]]]]

    closestFacet[pointIndices[pairs]] = facetIndices[pairs]
    closestWeights[pointIndices[pairs]] = weights[pairs]
    return closestFacet, closestWeights, closestDistance


def closestFacetsWithTies(
    queryPoints: np.ndarray,
    facetCoords: list,
    candidatesPerPoint: list,
    tieTolerance: float,
    lowestIndexOnTie: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For each query point, the closest of its candidate facets as in :func:`closestFacets`, and
    whether that choice is a tie.

    Facets whose distance lies within ``tieTolerance`` of the smallest one count as equally close,
    and of those the one with the lowest index is taken, so the choice does not depend on the
    rounding of the current coordinates; with ``lowestIndexOnTie=False`` the first facet of exactly
    the smallest distance is taken instead, which is the choice of :func:`closestFacets`. A tie is
    the ordinary case, not an accident: when the closest point lies on an edge or a vertex shared by
    several facets, every one of them returns that same point at the same distance. It is where the
    projection onto a faceted surface is not unique (Wriggers 2006, Remark 4.1 and Fig. 4.4, p. 60;
    Section 10.1 and Fig. 10.8, pp. 318-319).

    Parameters
    ----------
    queryPoints
        The points, shape ``(nPoints, nDim)``.
    facetCoords
        Current coordinates of each facet's nodes, all of the same shape.
    candidatesPerPoint
        One ascending list of candidate facet indices per point, see :func:`closestFacetCandidates`.
    tieTolerance
        Distances within this of the smallest one count as equal.
    lowestIndexOnTie
        Whether a tie goes to the lowest index, or to the first facet of exactly the smallest distance.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Per point the chosen facet's index, -1 if none; its weights, shape
        ``(nPoints, nNodesPerFacet)``, zero if none; the distance, infinite if none; and whether
        another facet was equally close.
    """

    queryPoints = np.asarray(queryPoints, dtype=float)
    nPoints = len(queryPoints)
    closestFacet = np.full(nPoints, -1, dtype=np.intp)
    closestDistance = np.full(nPoints, np.inf)
    tied = np.zeros(nPoints, dtype=bool)
    if not facetCoords:
        return closestFacet, np.zeros((nPoints, 0)), closestDistance, tied

    stacked = np.asarray(facetCoords, dtype=float)  # (nFacets, nNodesPerFacet, nDim)
    closestWeights = np.zeros((nPoints, stacked.shape[1]))

    pointIndices, facetIndices = candidatePairs(candidatesPerPoint)
    weights, distances = facetClosestPoints(queryPoints[pointIndices], stacked[facetIndices])

    acceptable = np.flatnonzero(distances < np.inf)
    if len(acceptable) == 0:
        return closestFacet, closestWeights, closestDistance, tied
    smallest = np.full(nPoints, np.inf)
    np.minimum.at(smallest, pointIndices[acceptable], distances[acceptable])

    equallyClose = acceptable[distances[acceptable] <= smallest[pointIndices[acceptable]] + tieTolerance]
    tied = np.bincount(pointIndices[equallyClose], minlength=nPoints) > 1

    if lowestIndexOnTie:
        pairs = equallyClose
    else:
        pairs = acceptable[distances[acceptable] == smallest[pointIndices[acceptable]]]
    # Pairs are ordered by point, and per point in ascending candidate order: keep each point's first.
    pairs = pairs[np.r_[True, pointIndices[pairs[1:]] != pointIndices[pairs[:-1]]]]

    closestFacet[pointIndices[pairs]] = facetIndices[pairs]
    closestWeights[pointIndices[pairs]] = weights[pairs]
    closestDistance[pointIndices[pairs]] = distances[pairs]
    return closestFacet, closestWeights, closestDistance, tied


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


def parentFaceQuadraturePointPatches(slavePoints, pointKey) -> list:
    """The old slave parent faces of the integrated contact, integrated with
    ``quadrature='parentFace'``, as patches for
    :func:`~edelweissfe.utils.contactfriction.historyInterpolation`. The counterpart of
    :func:`facetQuadraturePointPatches` with the parent face as the patch: a linear field in the
    face's parametric coordinates is fitted to the values at its Gauss points by least squares
    (Wriggers 2006, Eqs. 14.32-14.33; a constant one for a single point), and evaluated where the new
    point lies on the face, found by the closest-point projection onto the curved reference face.

    Parameters
    ----------
    slavePoints
        The old :class:`~edelweissfe.constraints.base.contactpointsonslavesurface.ContactPointsOnSlaveSurface`.
    pointKey
        ``pointKey(c, q)`` gives the key of quadrature point ``q`` of old cell ``c``.
    """

    from edelweissfe.utils.parentfacegeometry import parentFaceClosestPoints

    patches = []
    for c in range(slavePoints.nCells):
        faceType = slavePoints.cellFaceTypes[c]
        coordinates = slavePoints.parentReferenceCoordinates[c]
        parametric = slavePoints.parametricCoordinates[c]
        localDimension = parametric.shape[1]
        nTerms = 1 + localDimension if len(parametric) > localDimension else 1
        fit = np.linalg.pinv(np.hstack([np.ones((len(parametric), 1)), parametric[:, : nTerms - 1]]))
        keys = [pointKey(c, q) for q in range(len(parametric))]

        def locate(
            point, faceType=faceType, coordinates=coordinates, fit=fit, keys=keys, nTerms=nTerms, d=localDimension
        ):
            # Started from the face centre; the reference faces of a slave surface are mildly curved
            # at most, and the projection is clamped to the face, so the distance tells a point beside
            # the face from one on it.
            xi, closest, _ = parentFaceClosestPoints(
                faceType, coordinates[None], np.asarray(point, dtype=float)[None], np.zeros((1, d)), 1e-12
            )
            distance = float(np.linalg.norm(np.asarray(point) - closest[0]))
            return distance, keys, np.concatenate([[1.0], xi[0, : nTerms - 1]]) @ fit

        patches.append((coordinates.mean(axis=0), locate))
    return patches
