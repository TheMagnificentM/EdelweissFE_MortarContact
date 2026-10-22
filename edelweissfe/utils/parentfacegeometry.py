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

"""
Parametric geometry of a *parent face*: the element face a contact facet was cut from, together
with its own (possibly quadratic) shape functions and the quadrature rules needed to integrate
over the facets tiling it.

This is what separates Gauss-point-to-segment contact from the node-based formulation. A node-based
scheme can only ever deliver a non-negative force to a node, because the force *is* the spring
force there. An integrated scheme evaluates a non-negative pressure at quadrature points and
distributes it with the parent face's own shape functions, so a node whose shape function is
negative over part of the face receives a *tensile* nodal force -- which is exactly the consistent
load of a uniform pressure on a serendipity face (:math:`-pA/12` at the corners) and exactly what a
per-node scheme cannot produce. Without the parent-face basis, quadrature alone changes nothing:
the flat facets' own linear shape functions are non-negative, so the corner nodes would keep
lifting off. See the contact theory documentation.

Canonical parent-face node orderings, and the parametric coordinates they carry:

* ``quad8``: corners ``c0..c3`` counter-cycling the face, then midsides ``m0..m3`` with ``mk``
  between ``ck`` and ``c(k+1)``; corners at :math:`(\\pm 1, \\pm 1)`, midsides at the edge centres.
* ``quad4``: corners ``c0..c3`` only.
* ``line3``: end nodes at :math:`\\xi = \\mp 1`, midside at :math:`\\xi = 0`.
* ``line2``: end nodes at :math:`\\xi = \\mp 1`.

The mapping from an element's local node indices to this canonical order is *derived* from the
surface element generator's own face tables rather than transcribed again -- see
:func:`~edelweissfe.generators.surfaceelementgenerator.canonicalParentFace`.
"""

#: Parametric coordinates of each canonical parent-face node ordering, shape ``(nNodes, nDim - 1)``.
PARENT_FACE_PARAMETRIC_COORDS = {
    "quad8": np.array(
        [
            [-1.0, -1.0],
            [+1.0, -1.0],
            [+1.0, +1.0],
            [-1.0, +1.0],
            [+0.0, -1.0],
            [+1.0, +0.0],
            [+0.0, +1.0],
            [-1.0, +0.0],
        ]
    ),
    "quad4": np.array([[-1.0, -1.0], [+1.0, -1.0], [+1.0, +1.0], [-1.0, +1.0]]),
    "line3": np.array([[-1.0], [+1.0], [0.0]]),
    "line2": np.array([[-1.0], [+1.0]]),
}


def parentFaceShapeFunctions(faceType: str, parametricCoords: np.ndarray) -> np.ndarray:
    """The parent face's shape functions at one parametric point.

    Parameters
    ----------
    faceType
        One of the keys of :data:`PARENT_FACE_PARAMETRIC_COORDS`.
    parametricCoords
        The evaluation point, shape ``(1,)`` for a line face or ``(2,)`` for a quad face.

    Returns
    -------
    numpy.ndarray
        The shape function values, ordered like the canonical node ordering and summing to one. For
        ``quad8`` and ``line3`` these take *negative* values near the corners, which is the whole
        point of this module.
    """

    nodes = PARENT_FACE_PARAMETRIC_COORDS[faceType]

    if faceType == "quad8":
        xi, eta = parametricCoords
        N = np.empty(8)
        for i in range(4):
            a, b = nodes[i]
            N[i] = 0.25 * (1.0 + xi * a) * (1.0 + eta * b) * (xi * a + eta * b - 1.0)
        for i in range(4, 8):
            a, b = nodes[i]
            if a == 0.0:
                N[i] = 0.5 * (1.0 - xi**2) * (1.0 + eta * b)
            else:
                N[i] = 0.5 * (1.0 + xi * a) * (1.0 - eta**2)
        return N

    if faceType == "quad4":
        xi, eta = parametricCoords
        return np.array([0.25 * (1.0 + xi * a) * (1.0 + eta * b) for a, b in nodes])

    if faceType == "line3":
        (xi,) = parametricCoords
        return np.array([0.5 * xi * (xi - 1.0), 0.5 * xi * (xi + 1.0), 1.0 - xi**2])

    if faceType == "line2":
        (xi,) = parametricCoords
        return np.array([0.5 * (1.0 - xi), 0.5 * (1.0 + xi)])

    raise ValueError(f"parentFaceShapeFunctions: unsupported parent face type '{faceType}'.")


#: Barycentric quadrature rules on a triangle, keyed by point count: (points, weights). The weights
#: sum to one, so an integral is ``facetMeasure * sum(w_q * f(b_q))``. 1 point is exact to degree 1,
#: 3 points to degree 2, 6 points to degree 4 (Dunavant).
_TRIANGLE_RULES = {
    1: (np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]]), np.array([1.0])),
    3: (
        np.array(
            [
                [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
                [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
                [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
            ]
        ),
        np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
    ),
    6: (
        np.array(
            [
                [0.816847572980459, 0.091576213509771, 0.091576213509771],
                [0.091576213509771, 0.816847572980459, 0.091576213509771],
                [0.091576213509771, 0.091576213509771, 0.816847572980459],
                [0.108103018168070, 0.445948490915965, 0.445948490915965],
                [0.445948490915965, 0.108103018168070, 0.445948490915965],
                [0.445948490915965, 0.445948490915965, 0.108103018168070],
            ]
        ),
        np.array(
            [
                0.109951743655322,
                0.109951743655322,
                0.109951743655322,
                0.223381589678011,
                0.223381589678011,
                0.223381589678011,
            ]
        ),
    ),
}

#: Gauss-Legendre rules on a Line2 facet, in the facet's own barycentric coordinates, keyed by point
#: count. Weights sum to one, as for the triangle rules. Exact to degree 2n - 1.
_LINE_RULES = {}
for _n in (1, 2, 3):
    _positions, _weights = np.polynomial.legendre.leggauss(_n)
    _LINE_RULES[_n] = (
        np.column_stack([0.5 * (1.0 - _positions), 0.5 * (1.0 + _positions)]),
        _weights / 2.0,
    )


def facetQuadratureRule(nFacetNodes: int, nPoints: int) -> tuple[np.ndarray, np.ndarray]:
    """The quadrature rule for one contact facet, in the facet's own barycentric coordinates.

    Parameters
    ----------
    nFacetNodes
        3 for a Tria3 facet, 2 for a Line2 facet.
    nPoints
        The number of quadrature points requested.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The barycentric coordinates, shape ``(nPoints, nFacetNodes)``, and the weights, which sum to
        one -- so an integral over the facet is ``facetMeasure * sum(weights * f(points))``.
    """

    rules = _TRIANGLE_RULES if nFacetNodes == 3 else _LINE_RULES
    if nPoints not in rules:
        raise ValueError(
            f"facetQuadratureRule: no {nPoints}-point rule for a {nFacetNodes}-node facet. "
            f"Available: {sorted(rules.keys())}."
        )
    return rules[nPoints]


#: The local dimension of each parent face type: 2 for a surface face (3D), 1 for an edge face (2D).
PARENT_FACE_LOCAL_DIMENSION = {"quad8": 2, "quad4": 2, "line3": 1, "line2": 1}

#: The parent face a whole-face contact element (``facets=wholeFace`` of the surface element generator)
#: is: its node order is the canonical one of that face, so the element *is* its parent face.
PARENT_FACE_OF_CONTACT_ELEMENT = {"CONQUAD8": "quad8", "CONQUAD4": "quad4", "CONLINE3": "line3", "CONLINE2": "line2"}


def parentFaceShapeFunctionsAt(faceType: str, parametricCoords: np.ndarray) -> np.ndarray:
    """The parent face's shape functions at many parametric points at once.

    The same functions as :func:`parentFaceShapeFunctions`, evaluated in bulk.

    Parameters
    ----------
    faceType
        One of the keys of :data:`PARENT_FACE_PARAMETRIC_COORDS`.
    parametricCoords
        The evaluation points, shape ``(nPoints, nLocalDim)``.

    Returns
    -------
    numpy.ndarray
        The shape function values, shape ``(nPoints, nNodes)``.
    """

    xi = np.asarray(parametricCoords, dtype=float)
    nodes = PARENT_FACE_PARAMETRIC_COORDS[faceType]

    if faceType == "quad8":
        x, y = xi[:, 0:1], xi[:, 1:2]
        a, b = nodes[:, 0], nodes[:, 1]
        corner = 0.25 * (1.0 + x * a[:4]) * (1.0 + y * b[:4]) * (x * a[:4] + y * b[:4] - 1.0)
        midside = np.where(
            a[4:] == 0.0,
            0.5 * (1.0 - x**2) * (1.0 + y * b[4:]),
            0.5 * (1.0 + x * a[4:]) * (1.0 - y**2),
        )
        return np.hstack([corner, midside])

    if faceType == "quad4":
        x, y = xi[:, 0:1], xi[:, 1:2]
        return 0.25 * (1.0 + x * nodes[:, 0]) * (1.0 + y * nodes[:, 1])

    if faceType == "line3":
        x = xi[:, 0]
        return np.column_stack([0.5 * x * (x - 1.0), 0.5 * x * (x + 1.0), 1.0 - x**2])

    if faceType == "line2":
        x = xi[:, 0]
        return np.column_stack([0.5 * (1.0 - x), 0.5 * (1.0 + x)])

    raise ValueError(f"parentFaceShapeFunctionsAt: unsupported parent face type '{faceType}'.")


def parentFaceShapeFunctionDerivatives(faceType: str, parametricCoords: np.ndarray) -> np.ndarray:
    """The first parametric derivatives :math:`N_{a,\\alpha}` of the parent face's shape functions.

    Parameters
    ----------
    faceType
        One of the keys of :data:`PARENT_FACE_PARAMETRIC_COORDS`.
    parametricCoords
        The evaluation points, shape ``(nPoints, nLocalDim)``.

    Returns
    -------
    numpy.ndarray
        The derivatives, shape ``(nPoints, nLocalDim, nNodes)``.
    """

    xi = np.asarray(parametricCoords, dtype=float)
    nodes = PARENT_FACE_PARAMETRIC_COORDS[faceType]

    if faceType == "quad8":
        x, y = xi[:, 0:1], xi[:, 1:2]
        a, b = nodes[:4, 0], nodes[:4, 1]
        cornerX = 0.25 * a * (1.0 + y * b) * (2.0 * x * a + y * b)
        cornerY = 0.25 * b * (1.0 + x * a) * (x * a + 2.0 * y * b)
        a, b = nodes[4:, 0], nodes[4:, 1]
        midX = np.where(a == 0.0, -x * (1.0 + y * b), 0.5 * a * (1.0 - y**2))
        midY = np.where(a == 0.0, 0.5 * b * (1.0 - x**2), -y * (1.0 + x * a))
        return np.stack([np.hstack([cornerX, midX]), np.hstack([cornerY, midY])], axis=1)

    if faceType == "quad4":
        x, y = xi[:, 0:1], xi[:, 1:2]
        a, b = nodes[:, 0], nodes[:, 1]
        return np.stack([0.25 * a * (1.0 + y * b), 0.25 * b * (1.0 + x * a)], axis=1)

    if faceType == "line3":
        x = xi[:, 0]
        return np.column_stack([x - 0.5, x + 0.5, -2.0 * x])[:, None, :]

    if faceType == "line2":
        return np.tile(np.array([[[-0.5, 0.5]]]), (len(xi), 1, 1))

    raise ValueError(f"parentFaceShapeFunctionDerivatives: unsupported parent face type '{faceType}'.")


def parentFaceShapeFunctionSecondDerivatives(faceType: str, parametricCoords: np.ndarray) -> np.ndarray:
    """The second parametric derivatives :math:`N_{a,\\alpha\\beta}` of the parent face's shape
    functions, needed by the Newton iteration of :func:`parentFaceClosestPoints`.

    Parameters
    ----------
    faceType
        One of the keys of :data:`PARENT_FACE_PARAMETRIC_COORDS`.
    parametricCoords
        The evaluation points, shape ``(nPoints, nLocalDim)``.

    Returns
    -------
    numpy.ndarray
        The derivatives, shape ``(nPoints, nLocalDim, nLocalDim, nNodes)``.
    """

    xi = np.asarray(parametricCoords, dtype=float)
    nodes = PARENT_FACE_PARAMETRIC_COORDS[faceType]
    nPoints = len(xi)

    if faceType == "quad8":
        x, y = xi[:, 0:1], xi[:, 1:2]
        a, b = nodes[:4, 0], nodes[:4, 1]
        cornerXX = 0.5 * (1.0 + y * b)
        cornerXY = 0.25 * a * b * (2.0 * x * a + 2.0 * y * b + 1.0)
        cornerYY = 0.5 * (1.0 + x * a)
        a, b = nodes[4:, 0], nodes[4:, 1]
        midXX = np.where(a == 0.0, -(1.0 + y * b), 0.0 * x)
        midXY = np.where(a == 0.0, -x * b, -y * a)
        midYY = np.where(a == 0.0, 0.0 * y, -(1.0 + x * a))
        XX, XY, YY = np.hstack([cornerXX, midXX]), np.hstack([cornerXY, midXY]), np.hstack([cornerYY, midYY])
        return np.stack([np.stack([XX, XY], axis=1), np.stack([XY, YY], axis=1)], axis=1)

    if faceType == "quad4":
        XY = np.tile(0.25 * nodes[:, 0] * nodes[:, 1], (nPoints, 1))
        zero = np.zeros_like(XY)
        return np.stack([np.stack([zero, XY], axis=1), np.stack([XY, zero], axis=1)], axis=1)

    if faceType == "line3":
        return np.tile(np.array([[[[1.0, 1.0, -2.0]]]]), (nPoints, 1, 1, 1))

    if faceType == "line2":
        return np.zeros((nPoints, 1, 1, 2))

    raise ValueError(f"parentFaceShapeFunctionSecondDerivatives: unsupported parent face type '{faceType}'.")


def parentFaceGaussRule(faceType: str, nPerDirection: int) -> tuple[np.ndarray, np.ndarray]:
    """The native Gauss-Legendre rule on a parent face: ``nPerDirection`` points on a line face, the
    tensor product of ``nPerDirection`` points per direction on a quadrilateral face. Exact to degree
    ``2 nPerDirection - 1`` per direction.

    Parameters
    ----------
    faceType
        One of the keys of :data:`PARENT_FACE_PARAMETRIC_COORDS`.
    nPerDirection
        The number of points per parametric direction.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The parametric points, shape ``(nPoints, nLocalDim)``, and the weights on the parent domain
        :math:`[-1, 1]^d` (summing to :math:`2^d`).
    """

    if faceType not in PARENT_FACE_LOCAL_DIMENSION:
        raise ValueError(
            f"parentFaceGaussRule: no native rule for parent face type '{faceType}'. Available: "
            f"{sorted(PARENT_FACE_LOCAL_DIMENSION)}."
        )
    positions, weights = np.polynomial.legendre.leggauss(nPerDirection)
    if PARENT_FACE_LOCAL_DIMENSION[faceType] == 1:
        return positions[:, None], weights
    xi, eta = np.meshgrid(positions, positions, indexing="ij")
    return np.column_stack([xi.ravel(), eta.ravel()]), np.outer(weights, weights).ravel()


def parentFaceTangents(faceType: str, parametricCoords: np.ndarray, nodeCoords: np.ndarray) -> np.ndarray:
    """The tangent vectors :math:`a_\\alpha = x_{,\\alpha} = N_{a,\\alpha} x_a` of a parent face.

    Parameters
    ----------
    faceType
        One of the keys of :data:`PARENT_FACE_PARAMETRIC_COORDS`.
    parametricCoords
        The evaluation points, shape ``(nPoints, nLocalDim)``.
    nodeCoords
        The coordinates of the face's nodes in canonical order, shape ``(nNodes, nDim)``, or one set
        per point, shape ``(nPoints, nNodes, nDim)``.

    Returns
    -------
    numpy.ndarray
        The tangents, shape ``(nPoints, nLocalDim, nDim)``.
    """

    dN = parentFaceShapeFunctionDerivatives(faceType, parametricCoords)
    if nodeCoords.ndim == 2:
        return dN @ nodeCoords
    return np.einsum("pak,pkj->paj", dN, nodeCoords)


def parentFaceAreaElement(tangents: np.ndarray) -> np.ndarray:
    """The area (3D) or length (2D) element :math:`|a_1 \\times a_2|` or :math:`|a_1|` from the tangents
    of :func:`parentFaceTangents`, shape ``(nPoints,)``."""

    if tangents.shape[1] == 2:
        return np.linalg.norm(np.cross(tangents[:, 0], tangents[:, 1]), axis=1)
    return np.linalg.norm(tangents[:, 0], axis=1)


def parentFaceUnitNormals(tangents: np.ndarray) -> np.ndarray:
    """The unit normals of a parent face from the tangents of :func:`parentFaceTangents`:
    :math:`a_1 \\times a_2 / |a_1 \\times a_2|` (Wriggers 2006, Eq. 4.4) for a surface face, the
    tangent rotated by -90 degrees for an edge face (the convention of
    :func:`~edelweissfe.utils.facetcontactgeometry.facetNormalAndMeasure`). Their orientation follows
    the canonical node order; callers align it with the outward facet normal."""

    if tangents.shape[1] == 2:
        n = np.cross(tangents[:, 0], tangents[:, 1])
    else:
        t = tangents[:, 0]
        n = np.column_stack([t[:, 1], -t[:, 0]])
    return n / np.linalg.norm(n, axis=1)[:, None]


def parentFaceClosestPoints(
    faceType: str,
    nodeCoords: np.ndarray,
    points: np.ndarray,
    initialParametricCoords: np.ndarray,
    tolerance: float,
    maxIterations: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The closest point of each query point on a curved parent face, as the solution of the minimum
    distance problem (Wriggers 2006, Eqs. 4.2-4.3) restricted to the face's parametric domain.

    The stationarity condition :math:`(x - x(\\xi)) \\cdot x_{,\\alpha} = 0` is solved by Newton's
    method, starting from the closest point on a flat facet of the face, with the linearization of
    Wriggers 2006, Eqs. 9.81-9.82 (p. 243):
    :math:`(a_\\alpha \\cdot a_\\beta - x_{,\\alpha\\beta} \\cdot (x - x(\\xi))) \\Delta\\xi^\\beta =
    (x - x(\\xi)) \\cdot a_\\alpha`. The domain :math:`[-1, 1]^d` is enforced by a projected Newton
    method: a coordinate held at a bound by a gradient pointing outward is fixed for the step, the
    others are updated, and the result is clipped to the domain. Where the full matrix is not positive
    definite -- far from a strongly curved face -- its first term alone is used (the Gauss-Newton
    step), which is always a descent direction.

    Parameters
    ----------
    faceType
        One of the keys of :data:`PARENT_FACE_PARAMETRIC_COORDS`.
    nodeCoords
        The coordinates of each point's face nodes in canonical order, shape ``(nPoints, nNodes, nDim)``.
    points
        The query points, shape ``(nPoints, nDim)``.
    initialParametricCoords
        The starting points, shape ``(nPoints, nLocalDim)``, inside the domain.
    tolerance
        The convergence tolerance on the parametric update, and (scaled by the tangent length) on the
        projected gradient; a length in parametric space.
    maxIterations
        The iteration limit.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        The parametric coordinates of the closest points, shape ``(nPoints, nLocalDim)``; their
        physical positions, shape ``(nPoints, nDim)``; and whether each one converged.
    """

    xi = np.clip(np.array(initialParametricCoords, dtype=float), -1.0, 1.0)
    nPoints, d = xi.shape
    converged = np.zeros(nPoints, dtype=bool)
    identity = np.eye(d)

    for _ in range(maxIterations + 1):
        N = parentFaceShapeFunctionsAt(faceType, xi)
        x = np.einsum("pk,pkj->pj", N, nodeCoords)
        a = parentFaceTangents(faceType, xi, nodeCoords)
        r = points - x
        gradient = -np.einsum("pj,paj->pa", r, a)

        # Coordinates held at a bound by an outward-pointing descent direction are fixed.
        free = ~(((xi >= 1.0) & (gradient < 0.0)) | ((xi <= -1.0) & (gradient > 0.0)))
        projectedGradient = np.where(free, gradient, 0.0)
        scale = np.linalg.norm(a, axis=2).max(axis=1)
        stationary = np.linalg.norm(projectedGradient, axis=1) <= tolerance * scale**2
        converged |= stationary
        if converged.all():
            break

        metric = np.einsum("paj,pbj->pab", a, a)
        curvature = np.einsum("pabk,pkj,pj->pab", parentFaceShapeFunctionSecondDerivatives(faceType, xi), nodeCoords, r)
        hessian = metric - curvature
        if d == 2:
            positive = (hessian[:, 0, 0] > 0.0) & (np.linalg.det(hessian) > 0.0)
        else:
            positive = hessian[:, 0, 0] > 0.0
        hessian = np.where(positive[:, None, None], hessian, metric)

        mask = free[:, :, None] & free[:, None, :]
        reduced = np.where(mask, hessian, identity[None])
        step = -np.linalg.solve(reduced, projectedGradient[:, :, None])[:, :, 0]
        step[converged] = 0.0
        xiNew = np.clip(xi + step, -1.0, 1.0)
        small = np.linalg.norm(xiNew - xi, axis=1) <= tolerance
        xi = xiNew
        converged |= small & (np.linalg.norm(step, axis=1) <= tolerance)

    N = parentFaceShapeFunctionsAt(faceType, xi)
    return xi, np.einsum("pk,pkj->pj", N, nodeCoords), converged
