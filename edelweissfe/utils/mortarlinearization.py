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
Directional derivatives of the mortar geometry with respect to the nodal coordinates -- the
building blocks of the consistent linearisation of :mod:`~edelweissfe.constraints.mortarcontact`.

The scheme is the one of Popp, Gitterle, Gee & Wall (2010), Appendix A (3D) and Popp, Gee & Wall
(2009), Appendix A (2D): every quantity the mortar matrices are built from -- nodal normals, the
auxiliary plane, the projected nodes, the vertices of the clip polygon, the integration-cell
Jacobian, the Gauss point coordinates on both sides, the dual shape functions -- is differentiated,
and the derivatives are chained down to D, C and the weighted gap. The chain is written out here in
forward mode: every function returns a value together with its Jacobian with respect to a local
vector of nodal coordinates, and a caller composes them by matrix products.

What is differentiated is the geometry as :mod:`~edelweissfe.utils.mortargeometry` builds it, which
departs from Popp et al. in three places, each differentiated by the chain rule rather than taken
from a paper:

* the auxiliary plane belongs to a slave SUB-CELL and its normal is the sub-cell's own facet normal,
  not the interpolated nodal normal of their Eq. (A15);
* the back-projection of a Gauss point is the inverse of the in-plane isoparametric map (3D) or a
  closest-point projection (2D), not a projection along the auxiliary normal (their Eq. (A12));
* the polygon is fanned from its first vertex, not from its centroid (their Eqs. (A20)-(A21)).

As in the papers, a derivative holds for a FIXED layout: which vertices the clip polygon consists
of, which edges intersect, how it is triangulated and which facet a Gauss point is attributed to
are decided by the value computation and held fixed in the derivative.

The in-plane basis ``t1, t2`` of the auxiliary plane is varied without rotation about the normal,
``dt_i = -(t_i . dn) n``. Every quantity that leaves this module -- areas, natural coordinates,
shape-function values -- is invariant under a common in-plane rotation of all plane coordinates, so
the rotational part of the true variation of the basis contributes nothing and is dropped.

Array conventions: a Jacobian of a quantity of shape ``s`` with respect to a local coordinate
vector of length ``nZ`` has shape ``s + (nZ,)``.
"""


def skew(a: np.ndarray) -> np.ndarray:
    """The cross-product matrix: ``skew(a) @ b == np.cross(a, b)``."""
    return np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])


def cross2(a: np.ndarray, b: np.ndarray) -> float:
    """The scalar cross product of two plane vectors."""
    return a[0] * b[1] - a[1] * b[0]


def dCross2(a, da, b, db) -> np.ndarray:
    """Derivative of :func:`cross2` from the derivatives ``da``, ``db`` of shape ``(2, nZ)``."""
    return da[0] * b[1] + a[0] * db[1] - da[1] * b[0] - a[1] * db[0]


def nodeSlots(nNodes: int, dim: int, nZ: int, offset: int = 0) -> np.ndarray:
    """Jacobians ``(nNodes, dim, nZ)`` of the coordinates of ``nNodes`` nodes that occupy the slots
    ``offset ... offset + nNodes - 1`` of a local coordinate vector of ``nZ`` entries."""
    J = np.zeros((nNodes, dim, nZ))
    for a in range(nNodes):
        J[a, :, dim * (offset + a) : dim * (offset + a + 1)] = np.eye(dim)
    return J


def unitVector(v: np.ndarray, dv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``u = v / |v|`` and its derivative ``du = (I - u u^T) dv / |v|`` -- Popp et al. (2010),
    Eq. (A1)."""
    length = np.linalg.norm(v)
    u = v / length
    return u, (np.eye(len(v)) - np.outer(u, u)) @ dv / length


# ----------------------------------------------------------------------------------------- normals


def elementNormalAtNode(dN_node: np.ndarray, coords: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """The un-normalised element normal at one node and its derivative with respect to the
    element's nodal coordinates (flattened node by node, ``dim`` components each).

    3D: ``x,xi x x,eta`` at the node's own natural coordinate, Popp et al. (2010), Eq. (A3), whose
    derivative is ``(sum_k N_k,xi dx_k) x x,eta + x,xi x (sum_k N_k,eta dx_k)``. 2D: the rotated
    tangent ``[x,xi_y, -x,xi_x]``, Popp et al. (2009), Eq. (A3).

    ``dN_node`` is the ``(nLocalDim, nNodes)`` table of shape-function derivatives at the node.
    """
    n = coords.shape[0]
    t = dN_node @ coords
    if dim == 3:
        value = np.cross(t[0], t[1])
        # d(t0 x t1) = dt0 x t1 + t0 x dt1 = -skew(t1) dt0 + skew(t0) dt1, with dt_i = sum_k N_k,i dx_k.
        A0, A1 = -skew(t[1]), skew(t[0])
        d = np.zeros((3, 3 * n))
        for k in range(n):
            d[:, 3 * k : 3 * k + 3] = dN_node[0, k] * A0 + dN_node[1, k] * A1
        return value, d
    value = np.array([t[0][1], -t[0][0]])
    d = np.zeros((2, 2 * n))
    for k in range(n):
        d[0, 2 * k + 1] = dN_node[0, k]
        d[1, 2 * k] = -dN_node[0, k]
    return value, d


# ------------------------------------------------------------------------------- auxiliary plane


def subCellPlane(scCoords: np.ndarray, dscCoords: np.ndarray):
    """The auxiliary plane of a linear slave sub-cell, as
    :func:`~edelweissfe.utils.mortargeometry.facetNormal` and the caller build it, with its
    derivatives.

    Parameters
    ----------
    scCoords
        ``(k, 3)`` corner coordinates of the sub-cell, ``k`` = 3 or 4.
    dscCoords
        ``(k, 3, nZ)`` their Jacobians.

    Returns
    -------
    tuple
        ``(normal, dnormal, p0, dp0)``: the unit normal and the centroid with their Jacobians
        ``(3, nZ)``. ``None`` for a degenerate sub-cell.
    """
    if len(scCoords) == 3:
        a, b = scCoords[1] - scCoords[0], scCoords[2] - scCoords[0]
        da, db = dscCoords[1] - dscCoords[0], dscCoords[2] - dscCoords[0]
    else:
        a, b = scCoords[2] - scCoords[0], scCoords[3] - scCoords[1]
        da, db = dscCoords[2] - dscCoords[0], dscCoords[3] - dscCoords[1]
    c = 0.5 * np.cross(a, b)
    dc = 0.5 * (-skew(b) @ da + skew(a) @ db)
    if np.linalg.norm(c) < 1e-14:
        return None
    normal, dnormal = unitVector(c, dc)
    p0 = np.mean(scCoords, axis=0)
    dp0 = np.mean(dscCoords, axis=0)
    return normal, dnormal, p0, dp0


def planeCoordinates(points, dpoints, p0, dp0, normal, dnormal, t1, t2) -> tuple[np.ndarray, np.ndarray]:
    """Plane coordinates ``q = [(x - p0) . t1, (x - p0) . t2]`` of points and their Jacobians.

    The orthogonal projection into the auxiliary plane, Popp et al. (2010), Eq. (A17), expressed in
    the plane's own basis. With the rotation-free variation of the basis (see the module docstring)

        dq_i = t_i . (dx - dp0) - (t_i . dn) ((x - p0) . n) .

    ``points`` is ``(m, 3)``, ``dpoints`` ``(m, 3, nZ)``; returns ``(m, 2)`` and ``(m, 2, nZ)``.
    """
    T = np.array([t1, t2])
    rel = points - p0
    q = rel @ T.T
    height = rel @ normal
    dq = T @ (dpoints - dp0[None]) - height[:, None, None] * (T @ dnormal)[None]
    return q, dq


# ------------------------------------------------------------------------------------- clipping


def sutherlandHodgmanClipWithDerivatives(subject, dsubject, clip, dclip) -> tuple[np.ndarray, np.ndarray]:
    """:func:`~edelweissfe.utils.mortargeometry.sutherlandHodgmanClip`, with the Jacobian of every
    vertex of the overlap.

    The values are computed with exactly the same arithmetic as the function it mirrors, so the two
    return identical polygons. A vertex of the overlap is either a vertex of the subject that
    survives, or the intersection of a subject edge with the line of a clip edge -- Popp et al.
    (2010), Eq. (A19), case 3 -- and the derivative of the intersection is that of the formula used
    for its value, chained through its four input points (which may themselves be intersections of
    an earlier stage).

    Returns the overlap ``(k, 2)`` and its Jacobian ``(k, 2, nZ)``.
    """

    def inside(p, cp1, cp2):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) - (cp2[1] - cp1[1]) * (p[0] - cp1[0]) >= -1e-12

    def intersection(cp1, dcp1, cp2, dcp2, s, ds, e, de):
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        den = dc[0] * dp[1] - dc[1] * dp[0]
        if den == 0.0:
            return [e[0], e[1]], de
        n3 = 1.0 / den
        value = [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

        ddc = dcp1 - dcp2
        ddp = ds - de
        dn1 = dCross2(cp1, dcp1, cp2, dcp2)
        dn2 = dCross2(s, ds, e, de)
        dden = dCross2(dc, ddc, dp, ddp)
        derivative = np.empty_like(ds)
        for i in range(2):
            numerator = n1 * dp[i] - n2 * dc[i]
            dnumerator = dn1 * dp[i] + n1 * ddp[i] - dn2 * dc[i] - n2 * ddc[i]
            derivative[i] = dnumerator * n3 - numerator * n3 * n3 * dden
        return value, derivative

    outputList = [(subject[k], dsubject[k]) for k in range(len(subject))]

    for i in range(len(clip)):
        cp1, dcp1 = clip[i], dclip[i]
        cp2, dcp2 = clip[(i + 1) % len(clip)], dclip[(i + 1) % len(clip)]

        inputList = outputList
        outputList = []
        if not inputList:
            break

        s, ds = inputList[-1]
        for e, de in inputList:
            if inside(e, cp1, cp2):
                if not inside(s, cp1, cp2):
                    outputList.append(intersection(cp1, dcp1, cp2, dcp2, s, ds, e, de))
                outputList.append((e, de))
            elif inside(s, cp1, cp2):
                outputList.append(intersection(cp1, dcp1, cp2, dcp2, s, ds, e, de))
            s, ds = e, de

    if not outputList:
        return np.array([]), np.zeros((0, 2, dsubject.shape[-1]))
    return np.array([p for p, _ in outputList]), np.array([d for _, d in outputList])


def fanTriangleArea(v, dv) -> tuple[float, np.ndarray]:
    """``|(v1 - v0) x (v2 - v0)|`` of a fan triangle -- twice its area, the ``area_jac`` of the
    segmentation -- and its Jacobian; the plane counterpart of Popp et al. (2010), Eq. (A7)."""
    cr = cross2(v[1] - v[0], v[2] - v[0])
    dcr = dCross2(v[1] - v[0], dv[1] - dv[0], v[2] - v[0], dv[2] - dv[0])
    return abs(cr), np.sign(cr) * dcr


# ---------------------------------------------------------------------------------- back-mapping


def inversePlaneMapDerivative(el, qNodes, dqNodes, xi, dxGauss) -> tuple[np.ndarray, np.ndarray]:
    """Jacobian of the natural coordinate ``xi`` that solves ``sum_k N_k(xi) q_k = x_gp`` in the
    auxiliary plane (the 3D back-mapping of :func:`~edelweissfe.utils.mortargeometry.mapPlaneToNatural`).

    Differentiating the defining equation gives ``J dxi + sum_k N_k dq_k = dx_gp`` with the in-plane
    Jacobian ``J = sum_k q_k (x) N_k,xi``, the counterpart of Popp et al. (2010), Eqs. (A13)-(A14),
    for an in-plane rather than a normal projection.

    Returns ``dxi`` ``(2, nZ)`` and, since every caller needs it next, the shape-function
    derivatives ``N,xi`` ``(2, nNodes)`` at ``xi``. Raises ``np.linalg.LinAlgError`` where ``J`` is
    singular.
    """
    N = el.getShapeFunctions(xi)
    dN = el.getShapeFunctionDerivatives(xi)
    (a, b), (c, d) = (dN @ qNodes).T
    det = a * d - b * c
    if det == 0.0:
        raise np.linalg.LinAlgError("singular in-plane Jacobian")
    rhs = dxGauss - np.tensordot(N, dqNodes, axes=1)
    return np.array([d * rhs[0] - b * rhs[1], a * rhs[1] - c * rhs[0]]) / det, dN


def shapeFunctionSecondDerivatives(el, xi) -> np.ndarray:
    """``d^2 N / dxi^2`` of a line element, for the closest-point linearisation in 2D."""
    elType = el.elType.upper()
    if elType == "CONLINE2":
        return np.zeros(2)
    if elType == "CONLINE3":
        return np.array([1.0, 1.0, -2.0])
    raise NotImplementedError(f"No second derivatives for element type '{elType}'.")


def closestPointDerivative(el, xNodes, dxNodes, xi, xGauss, dxGauss) -> np.ndarray:
    """Jacobian ``(1, nZ)`` of the closest-point coordinate on a line element in 2D (the 2D
    back-mapping of :func:`~edelweissfe.utils.mortargeometry.mapPlaneToNatural`).

    The coordinate satisfies ``f = (x(xi) - x_gp) . x,xi(xi) = 0``; differentiating,

        (x,xi . x,xi + r . x,xixi) dxi = -[(sum_k N_k dx_k - dx_gp) . x,xi + r . sum_k N_k,xi dx_k] ,

    with ``r = x(xi) - x_gp``. Popp et al. (2009) project along the normal field instead, their
    Eqs. (A26)-(A30); this is the same step for the projection used here.
    """
    N = el.getShapeFunctions(xi)
    dN = el.getShapeFunctionDerivatives(xi)[0]
    d2N = shapeFunctionSecondDerivatives(el, xi)
    x = N @ xNodes
    tangent = dN @ xNodes
    curvature = d2N @ xNodes
    r = x - xGauss
    denominator = tangent @ tangent + r @ curvature
    dx = np.einsum("k,kiz->iz", N, dxNodes) - dxGauss
    dtangent = np.einsum("k,kiz->iz", dN, dxNodes)
    return -(tangent @ dx + r @ dtangent)[None, :] / denominator


def chordSegment(sc, dsc, mc, dmc):
    """:func:`~edelweissfe.utils.mortargeometry.clipLineSegments` with its Jacobians.

    Returns ``(start, dstart, end, dend, tangent, dtangent)``, or ``None`` where the overlap
    vanishes. The branches of the ``min``/``max`` are those the value takes, held fixed.
    """
    slaveVector = sc[1] - sc[0]
    dslaveVector = dsc[1] - dsc[0]
    slaveLength = np.linalg.norm(slaveVector)
    if slaveLength < 1e-14:
        return None
    tangent, dtangent = unitVector(slaveVector, dslaveVector)
    dslaveLength = tangent @ dslaveVector

    s0 = np.dot(mc[0] - sc[0], tangent)
    s1 = np.dot(mc[1] - sc[0], tangent)
    ds0 = tangent @ (dmc[0] - dsc[0]) + (mc[0] - sc[0]) @ dtangent
    ds1 = tangent @ (dmc[1] - dsc[0]) + (mc[1] - sc[0]) @ dtangent

    lower, dlower = (s0, ds0) if s0 <= s1 else (s1, ds1)
    upper, dupper = (s1, ds1) if s0 <= s1 else (s0, ds0)
    start, dstart = (lower, dlower) if lower > 0.0 else (0.0, np.zeros_like(ds0))
    end, dend = (upper, dupper) if upper < slaveLength else (slaveLength, dslaveLength)
    if end - start < 1e-12:
        return None
    return start, dstart, end, dend, tangent, dtangent


# --------------------------------------------------------------------------------- facet measure


def surfaceJacobian(el, xi, coords) -> tuple[float, np.ndarray]:
    """The surface Jacobian of
    :meth:`~edelweissfe.elements.contactelement.element.Element.getJacobianAndAreaWeight` and its
    derivative with respect to the element's nodal coordinates.

    3D: ``J = |x,xi x x,eta|``, ``dJ = (c/|c|) . (dx,xi x x,eta + x,xi x dx,eta)``, Popp et al.
    (2010), Eqs. (A25)-(A26). 2D: ``J = |x,xi|``, ``dJ = (x,xi/|x,xi|) . dx,xi``, Popp et al. (2009),
    Eq. (A33).
    """
    dN = el.getShapeFunctionDerivatives(xi)
    t = dN @ coords
    n, dim = coords.shape
    if dN.shape[0] == 1:
        J = float(np.linalg.norm(t[0]))
        u = t[0] / J
        return J, (dN[0][:, None] * u[None, :]).ravel()
    c = np.cross(t[0], t[1])
    J = float(np.linalg.norm(c))
    u = c / J
    # u . (dt0 x t1 + t0 x dt1) = dt0 . (t1 x u) + dt1 . (u x t0)
    a0, a1 = np.cross(t[1], u), np.cross(u, t[0])
    return J, (dN[0][:, None] * a0[None, :] + dN[1][:, None] * a1[None, :]).ravel()


# ------------------------------------------------------------------------------------- assembly


def scatterMatrix(nodes: np.ndarray, support: np.ndarray, dim: int) -> np.ndarray:
    """The 0/1 matrix ``(dim * len(nodes), dim * len(support))`` that moves a Jacobian over the
    coordinates of ``nodes`` onto the coordinates of ``support`` (which must contain them all);
    a node listed twice in ``nodes`` has its two columns summed."""
    positions = np.searchsorted(support, nodes)
    columns = (dim * positions[:, None] + np.arange(dim)).ravel()
    S = np.zeros((len(columns), dim * len(support)))
    S[np.arange(len(columns)), columns] = 1.0
    return S


def mergeOverNodes(parts: list, dim: int, support: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Sum Jacobians given over different node lists onto one node list.

    Parameters
    ----------
    parts
        ``(nodes, jacobian)`` pairs, the last axis of each Jacobian running over the ``dim``
        coordinates of each of its nodes in turn. All Jacobians share their leading shape.
    dim
        Coordinates per node.
    support
        The sorted node list to sum onto; the union of all ``nodes`` if omitted.

    Returns
    -------
    tuple
        The support and the summed Jacobian over it.
    """
    if support is None:
        support = np.unique(np.concatenate([np.asarray(n, dtype=np.intp) for n, _ in parts]))
    out = np.zeros(parts[0][1].shape[:-1] + (dim * len(support),))
    for nodes, jacobian in parts:
        out += jacobian @ scatterMatrix(np.asarray(nodes, dtype=np.intp), support, dim)
    return support, out
