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

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from edelweissfe.utils.mortargeometry import (
    buildBoundingVolumeHierarchy,
    getSubCells,
    isConvexPolygon,
    lineGaussPoints,
    lineGaussWeights,
    mapPlaneToNatural,
    queryBoundingVolumeHierarchy,
    sutherlandHodgmanClip,
    tangentBasis,
    toPlaneCoordinates,
    triangleGaussPoints,
    triangleGaussWeights,
    triangulatePolygon,
)
from edelweissfe.utils.mortarlinearization import (
    fanTriangleArea,
    nodeSlots,
    planeCoordinates,
    scatterMatrix,
    shapeFunctionSecondDerivatives,
    subCellParentMap,
    surfaceJacobian,
    surfaceJacobianNaturalDerivative,
    sutherlandHodgmanClipWithDerivatives,
    unitVector,
)

"""
The mortar coupling of a non-mortar (slave) and a mortar (master) contact surface: the matrices D and
M (called C in the constraint), the gap matrices from which the weighted gap is formed, and the nodal
scaling factor -- with, on request, their derivatives with respect to the nodal coordinates.

Every step follows a published formulation; where two sources are composed, or a source is applied
beyond the case it treats, the step says so.

Three dimensions -- Popp, Gitterle, Gee & Wall (2010), Section 4 and Appendix A:

* an auxiliary plane per slave (sub-)segment through its centre ``x0``, with the interpolated nodal
  normal there as its normal ``n0``, Eqs. (A15) and (A18);
* the orthogonal projection of slave and master (sub-)segments into it, Eq. (A17), clipping by
  Sutherland & Hodgman, triangulation into integration cells, a 7-point rule per cell (Puso &
  Laursen 2004, Section 3.1);
* the Gauss point mapped back onto both (sub-)segments along ``n0``, Eq. (A12) -- for the planar
  sub-segment this is the inverse of its own isoparametric map in the plane;
* quadratic facets split into linear sub-segments with an affine map from sub-segment to parent
  coordinates, Popp et al. (2010), Section 4.2, Eq. (42), and Puso, Laursen & Solberg (2008),
  Eqs. (31)-(32), Fig. 3;
* D, M and the weighted gap on the cells, Eqs. (39)-(41), with the interpolated normal at the
  Gauss point, Eq. (A11).

Two dimensions -- Popp, Gee & Wall (2009), Section 4 and Appendix A:

* segment end points from projecting slave nodes onto the master element along the nodal normal,
  Eq. (44), and master nodes onto the slave element along the interpolated normal field, Eq. (45);
  of quadratic elements only the end nodes, as there;
* the Gauss points mapped linearly onto both sides, Eq. (46), with the segment Jacobian of
  Eq. (48);
* the nodal form of the weighted gap, Eq. (50).

Both dimensions:

* dual shape functions from the biorthogonality on the part of the slave element that is actually
  integrated, Popp, Seitz, Gee & Wall (2013), Section 4.1, Eqs. (21)-(28) -- in 2D the treatment of
  Cichosz & Bischoff (2011), Eq. (41);
* for quadratic slave elements the basis transformation of Popp, Wohlmuth, Gee & Wall (2012),
  Section 4.4.1, with the parameter of Farah (2018), Eq. (6.20); composing it with the
  partially-integrated biorthogonality of Popp et al. (2013), which treats first-order elements, is
  this implementation's step;
* the nodal scaling factor of Popp et al. (2013), Section 4.2, Eqs. (36)-(38), with which the rows of
  a multiplier formulation are scaled; for quadratic elements it integrates the transformed shape
  functions and normalises with their own reference integrals -- again this implementation's
  extension of a first-order definition.

Derivatives follow the same steps by the chain rule, as listed in Popp et al. (2010), Appendix A,
for a fixed layout: which cells exist, how they are triangulated, which facet pairs overlap, and
whether a slave element falls back to its reference-element dual basis is decided by the values and
not differentiated. The building blocks are in :mod:`~edelweissfe.utils.mortarlinearization`.
"""


def _contract(subscripts: str, *operands):
    """``np.einsum`` with a contraction order chosen for speed (pairwise, through BLAS) where more than
    two operands meet; the result is the same sum up to rounding."""
    return np.einsum(subscripts, *operands, optimize=len(operands) > 2)


#: Inverse condition number below which the covered-area biorthogonality system of a slave element
#: is treated as singular and the reference-element coefficients are used instead. Not from a source:
#: Popp et al. (2013) need no such guard for first-order elements, whose shape functions are strictly
#: positive (their p. 71); the transformed quadratic ones are not.
CONDITION_LIMIT = 1e12


@dataclass
class MortarCoupling:
    """The coupling of one configuration.

    Attributes
    ----------
    D, M
        ``(nSlave, nSlave)`` and ``(nSlave, nMaster)`` mortar matrices, Popp et al. (2010),
        Eqs. (39)-(40).
    Dn, Mn
        Per spatial component, the gap matrices: the weighted gap of slave node ``j`` is
        ``g_j = sum_l Mn[j,l] . x_l - sum_k Dn[j,k] . x_k``, so that ``Dn[j,k] = int Phi_j N_k n``
        with the interpolated normal at the Gauss point in 3D (Popp et al. 2010, Eq. (41)) and
        ``Dn[j,k] = D[j,k] n_j`` with the nodal normal in 2D (Popp et al. 2009, Eq. (50)).
    kappa
        ``(nSlave,)`` the nodal scaling factor of Popp et al. (2013), Eq. (36): one where every
        adjacent slave element is integrated entirely, the integrated fraction otherwise.
    rowContributions
        Per slave node, the derivatives of the pieces of its rows, if requested: tuples
        ``(kind, columns, derivative, support)`` with ``kind`` one of ``"D"``, ``"M"``, ``"Dn"``,
        ``"Mn"``, ``"kappa"``.
    projectionFailures, firstProjectionFailure
        Gauss points whose back-mapping did not converge, and where the first one was.
    nonConvex, slivers
        Element numbers of non-convex sub-segments and of slave elements that took the
        reference-element fallback.
    """

    D: csr_matrix
    M: csr_matrix
    Dn: list
    Mn: list
    kappa: np.ndarray
    rowContributions: list | None = None
    projectionFailures: int = 0
    firstProjectionFailure: tuple | None = None
    nonConvex: list = field(default_factory=list)
    slivers: list = field(default_factory=list)


def _subCellElement(nNodes: int, elementClass):
    """A linear contact element to evaluate a sub-segment with: a line, triangle or quadrilateral."""
    elType = {2: "CONLINE2", 3: "CONTRI3", 4: "CONQUAD4"}[nNodes]
    return elementClass(elType, 0)


def _subCellCentre(nNodes: int) -> np.ndarray:
    return {2: np.array([0.0]), 3: np.array([1.0 / 3.0, 1.0 / 3.0]), 4: np.array([0.0, 0.0])}[nNodes]


def _referenceIntegrals(el) -> np.ndarray:
    """``alpha_{e,j} = int_ref N~_j dxi`` of the (transformed) shape functions over the reference
    element: the normalisation of the nodal scaling factor, Popp et al. (2013), Eq. (36), where it is
    written ``int_e N_j / J_e de`` -- 1 for quad4 and 1/6 for tri3 (their p. 73)."""
    T = el.getBasisTransformation()
    points, weights = el.getQuadraturePoints()
    return sum(w * (T @ el.getShapeFunctions(xi)) for xi, w in zip(points, weights))


class _Records:
    """The Gauss point records of one slave element, collected column by column."""

    def __init__(self):
        self.N_s, self.master, self.N_m, self.dG, self.normal, self.Je = [], [], [], [], [], []
        self.Z, self.dN_s, self.dN_m, self.ddG, self.dNormal, self.dJe = [], [], [], [], [], []

    def __len__(self):
        return len(self.dG)


def computeMortarCoupling(
    nonMortarFacets: list,
    mortarFacets: list,
    coordinates: dict,
    constraintIndex: dict,
    slaveIndex: dict,
    masterIndex: dict,
    normals: np.ndarray,
    normalDerivatives: list | None,
    dim: int,
    elementClass,
    warn=None,
) -> MortarCoupling:
    """Segment the two surfaces against each other and integrate the coupling.

    Parameters
    ----------
    nonMortarFacets, mortarFacets
        The contact elements of the two surfaces.
    coordinates
        Node -> current coordinates.
    constraintIndex
        Node -> index among all constraint nodes (non-mortar first); the derivatives are expressed
        over these.
    slaveIndex, masterIndex
        Node -> row / column index of D and M.
    normals
        ``(nSlave, dim)`` the nodal normals, Popp et al. (2010), Eq. (37).
    normalDerivatives
        Per slave node, ``(nodes, dn)`` as from the constraint's ``computeNormals``; ``None`` for
        values only.
    dim
        2 or 3.
    elementClass
        The contact element class, to evaluate linear sub-segments with.
    warn
        ``warn(key, message)`` for one-off diagnostics.
    """
    withDerivatives = normalDerivatives is not None
    nSlave, nMaster = len(slaveIndex), len(masterIndex)
    result = MortarCoupling(None, None, [], [], np.zeros(nSlave))
    if withDerivatives:
        result.rowContributions = [[] for _ in range(nSlave)]

    # Broad phase: a bounding volume hierarchy over the mortar facets (Ericson 2004). The margin
    # keeps a facet a candidate a little before it touches; it is a search radius, not a tolerance
    # of the formulation.
    facetsWithBounds = []
    for m_el in mortarFacets:
        m_coords = np.array([coordinates[n] for n in m_el.nodes])
        lo, hi = m_coords.min(axis=0), m_coords.max(axis=0)
        margin = max(0.15 * np.max(hi - lo), 0.1)
        facetsWithBounds.append((m_el, m_coords.mean(axis=0), lo - margin, hi + margin))
    bvh = buildBoundingVolumeHierarchy(facetsWithBounds)

    # Adjacent slave elements per slave node, the n_j^e of Popp et al. (2013), Eq. (36).
    adjacent = np.zeros(nSlave)
    for s_el in nonMortarFacets:
        for nd in s_el.nodes:
            adjacent[slaveIndex[nd]] += 1.0
    kappaSum = np.zeros(nSlave)

    blocks = {"D": ([], [], []), "M": ([], [], []), "Dn": [([], [], []) for _ in range(dim)]}
    blocks["Mn"] = [([], [], []) for _ in range(dim)]

    for s_el in nonMortarFacets:
        s_coords = np.array([coordinates[nd] for nd in s_el.nodes])
        lo, hi = s_coords.min(axis=0), s_coords.max(axis=0)
        margin = max(0.15 * np.max(hi - lo), 0.1)
        candidates = []
        queryBoundingVolumeHierarchy(bvh, lo - margin, hi + margin, candidates)
        if not candidates:
            continue

        context = _SlaveContext(s_el, coordinates, constraintIndex, slaveIndex, normals, normalDerivatives, dim)
        records = _Records()
        if dim == 3:
            _segment3D(context, candidates, coordinates, constraintIndex, elementClass, records, result, warn)
        else:
            _segment2D(context, candidates, coordinates, constraintIndex, records, result)
        if not len(records):
            continue
        _assembleSlaveElement(context, records, masterIndex, blocks, kappaSum, result, warn)

    def toCsr(triple, shape):
        rows, cols, vals = triple
        if not vals:
            return csr_matrix(shape)
        return coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=shape).tocsr()

    # The gap matrices are assembled from the same blocks as D and M, so all share one pattern.
    result.D = toCsr(blocks["D"], (nSlave, nSlave))
    result.M = toCsr(blocks["M"], (nSlave, nMaster))
    result.Dn = [toCsr(b, (nSlave, nSlave)) for b in blocks["Dn"]]
    result.Mn = [toCsr(b, (nSlave, nMaster)) for b in blocks["Mn"]]
    covered = adjacent > 0
    result.kappa[covered] = kappaSum[covered] / adjacent[covered]
    if withDerivatives:
        # The mean over the adjacent elements carries over to the derivative, Eq. (37).
        for I in range(nSlave):  # noqa: E741 - slave node index
            result.rowContributions[I] = [
                (kind, cols, (d / adjacent[I] if kind == "kappa" else d), support)
                for kind, cols, d, support in result.rowContributions[I]
            ]
    return result


class _SlaveContext:
    """What every Gauss point of one slave element needs: its nodes, coordinates and normals, and
    the local coordinate vector its derivatives run over."""

    def __init__(self, s_el, coordinates, constraintIndex, slaveIndex, normals, normalDerivatives, dim):
        self.el = s_el
        self.dim = dim
        self.coords = np.array([coordinates[nd] for nd in s_el.nodes])
        self.slaveRows = np.array([slaveIndex[nd] for nd in s_el.nodes])
        self.normals = normals[self.slaveRows]
        self.natural = np.array(s_el.getNodalNaturalCoordinates(), dtype=float).reshape(len(s_el.nodes), -1)
        self.withDerivatives = normalDerivatives is not None
        if not self.withDerivatives:
            return
        ids = np.array([constraintIndex[nd] for nd in s_el.nodes], dtype=np.intp)
        # The slave element's own nodes and every node their normals depend on.
        self.Z = np.unique(np.concatenate([ids] + [normalDerivatives[r][0] for r in self.slaveRows]))
        nZ = dim * len(self.Z)
        self.dCoords = np.zeros((len(ids), dim, nZ))
        positions = np.searchsorted(self.Z, ids)
        for a, p in enumerate(positions):
            self.dCoords[a, :, dim * p : dim * p + dim] = np.eye(dim)
        self.dNormals = np.array(
            [normalDerivatives[r][1] @ scatterMatrix(normalDerivatives[r][0], self.Z, dim) for r in self.slaveRows]
        )

    def extended(self, masterIds):
        """Local coordinate vector with the nodes of one master element appended."""
        nZs = self.dim * len(self.Z)
        nZ = nZs + self.dim * len(masterIds)

        def pad(d):
            out = np.zeros(d.shape[:-1] + (nZ,))
            out[..., : d.shape[-1]] = d
            return out

        dMaster = nodeSlots(len(masterIds), self.dim, nZ, offset=len(self.Z))
        return np.concatenate((self.Z, masterIds)), nZ, pad, dMaster


def _segment3D(context, candidates, coordinates, constraintIndex, elementClass, records, result, warn):
    """The 3D segmentation of one slave element, Popp et al. (2010), Section 4.1 and Algorithm 1, on the
    linear sub-segments of Section 4.2."""
    s_el = context.el
    derivatives = context.withDerivatives
    for s_sub in getSubCells(s_el):
        subEl = _subCellElement(len(s_sub), elementClass)
        N0 = subEl.getShapeFunctions(_subCellCentre(len(s_sub)))
        # Auxiliary plane: centre x0 and interpolated nodal normal n0 there, Eqs. (A18) and (A15).
        x0 = N0 @ context.coords[s_sub]
        n0raw = N0 @ context.normals[s_sub]
        if np.linalg.norm(n0raw) < 1e-14:
            continue
        n0 = n0raw / np.linalg.norm(n0raw)
        t1, t2 = tangentBasis(n0)
        q_s = toPlaneCoordinates(context.coords[s_sub], x0, t1, t2)
        if not isConvexPolygon(q_s):
            result.nonConvex.append(("non-mortar", s_el.elNumber))
        toParent_s = subCellParentMap(context.natural, s_sub, subEl)

        for m_el in candidates:
            m_coords = np.array([coordinates[nd] for nd in m_el.nodes])
            m_natural = np.array(m_el.getNodalNaturalCoordinates(), dtype=float).reshape(len(m_el.nodes), -1)
            if derivatives:
                m_ids = np.array([constraintIndex[nd] for nd in m_el.nodes], dtype=np.intp)
                Z, nZ, pad, dMaster = context.extended(m_ids)
                dx0 = pad(np.tensordot(N0, context.dCoords[s_sub], axes=1))
                _, dn0 = unitVector(n0raw, pad(np.tensordot(N0, context.dNormals[s_sub], axes=1)))
                _, dq_s = planeCoordinates(context.coords[s_sub], pad(context.dCoords[s_sub]), x0, dx0, n0, dn0, t1, t2)

            for m_sub in getSubCells(m_el):
                subEl_m = _subCellElement(len(m_sub), elementClass)
                q_m = toPlaneCoordinates(m_coords[m_sub], x0, t1, t2)
                if not isConvexPolygon(q_m):
                    result.nonConvex.append(("mortar", m_el.elNumber))
                if derivatives:
                    _, dq_m = planeCoordinates(m_coords[m_sub], dMaster[list(m_sub)], x0, dx0, n0, dn0, t1, t2)
                    polygon, dPolygon = sutherlandHodgmanClipWithDerivatives(q_m, dq_m, q_s, dq_s)
                else:
                    polygon = sutherlandHodgmanClip(q_m, q_s)
                if len(polygon) < 3:
                    continue
                toParent_m = subCellParentMap(m_natural, m_sub, subEl_m)

                for i_tri, tri in enumerate(triangulatePolygon(polygon)):
                    v0, v1, v2 = tri
                    area = abs((v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1]))
                    if area < 1e-14:
                        continue
                    if derivatives:
                        dTri = dPolygon[[0, i_tri + 1, i_tri + 2]]
                        _, dArea = fanTriangleArea(tri, dTri)
                    points = []
                    for gp, w in zip(triangleGaussPoints, triangleGaussWeights):
                        L1, L2 = gp
                        x_gp = (1.0 - L1 - L2) * v0 + L1 * v1 + L2 * v2
                        # Back onto both sub-segments along n0, Eq. (A12), then to the parents, Eq. (42).
                        xiSub_s, ok_s = mapPlaneToNatural(subEl, q_s, x_gp)
                        xiSub_m, ok_m = mapPlaneToNatural(subEl_m, q_m, x_gp)
                        for ok, side, el in ((ok_s, "slave", s_el), (ok_m, "master", m_el)):
                            if not ok:
                                result.projectionFailures += 1
                                result.firstProjectionFailure = result.firstProjectionFailure or (
                                    side,
                                    el.elNumber,
                                    el.elType,
                                )
                        xi_s, jac_s = toParent_s(xiSub_s)
                        xi_m, jac_m = toParent_m(xiSub_m)
                        N_s = s_el.getShapeFunctions(xi_s)
                        normal_raw = N_s @ context.normals
                        records.N_s.append(N_s)
                        records.master.append(m_el)
                        records.N_m.append(m_el.getShapeFunctions(xi_m))
                        records.dG.append(area * w)
                        records.normal.append(normal_raw / np.linalg.norm(normal_raw))
                        points.append((xiSub_s, xiSub_m, xi_s, xi_m, N_s))
                    # The element Jacobian |x,xi x x,eta| at the seven points at once (cf.
                    # ContactElement.getJacobianAndAreaWeight).
                    tangents = np.einsum(
                        "gak,ki->gai",
                        np.array([s_el.getShapeFunctionDerivatives(p[2]) for p in points]),
                        context.coords,
                    )
                    records.Je.extend(np.linalg.norm(np.cross(tangents[:, 0], tangents[:, 1]), axis=1))
                    if derivatives:
                        _gaussPointDerivatives3D(
                            context,
                            m_el,
                            (subEl, q_s, dq_s, jac_s),
                            (subEl_m, q_m, dq_m, jac_m),
                            dTri,
                            dArea,
                            points,
                            Z,
                            pad,
                            records,
                        )


def _gaussPointDerivatives3D(context, m_el, slaveSub, masterSub, dTri, dArea, points, Z, pad, records):
    """The derivatives of the seven Gauss points of one integration cell, evaluated together: their
    position in the plane (Eq. (A16)), their natural coordinates on both sub-segments by the inverse
    plane map (Eqs. (A12)-(A14)) carried to the parents (Eq. (42)), the shape functions there, the
    interpolated normal (Eq. (A11)), the element Jacobian with its natural derivative (Popp, Seitz,
    Gee & Wall 2013, Eq. (38)) and the integration weight (Eq. (A7))."""
    s_el = context.el
    coords = context.coords
    bary = np.column_stack((1.0 - triangleGaussPoints.sum(axis=1), triangleGaussPoints))
    dx = np.einsum("gv,viz->giz", bary, dTri)

    def naturalDerivatives(sub, q, dq, jac, xiSubs):
        subEl = sub
        N = np.array([subEl.getShapeFunctions(x) for x in xiSubs])
        dN = np.array([subEl.getShapeFunctionDerivatives(x) for x in xiSubs])
        J = np.einsum("gak,ki->gia", dN, q)
        rhs = dx - np.einsum("gk,kiz->giz", N, dq)
        try:
            dxiSub = np.linalg.solve(J, rhs)
        except np.linalg.LinAlgError:
            dxiSub = np.zeros_like(rhs)
        return np.einsum("ab,gbz->gaz", jac, dxiSub)

    subEl, q_s, dq_s, jac_s = slaveSub
    subEl_m, q_m, dq_m, jac_m = masterSub
    dxi_s = naturalDerivatives(subEl, q_s, dq_s, jac_s, [p[0] for p in points])
    dxi_m = naturalDerivatives(subEl_m, q_m, dq_m, jac_m, [p[1] for p in points])
    N_s = np.array([p[4] for p in points])
    dNdxi_s = np.array([s_el.getShapeFunctionDerivatives(p[2]) for p in points])
    dNdxi_m = np.array([m_el.getShapeFunctionDerivatives(p[3]) for p in points])
    dN_s = np.einsum("gak,gaz->gkz", dNdxi_s, dxi_s)
    dN_m = np.einsum("gak,gaz->gkz", dNdxi_m, dxi_m)

    # The interpolated normal, Eq. (A11), linearised through Eq. (A1).
    dNormals = pad(context.dNormals)
    raw = N_s @ context.normals
    dRaw = np.einsum("gk,kcz->gcz", N_s, dNormals) + np.einsum("kc,gak,gaz->gcz", context.normals, dNdxi_s, dxi_s)
    length = np.linalg.norm(raw, axis=1)
    unit = raw / length[:, None]
    dNormal = (dRaw - unit[:, :, None] * np.einsum("gc,gcz->gz", unit, dRaw)[:, None, :]) / length[:, None, None]

    # The element Jacobian J_e = |x,xi x x,eta| at the Gauss points and its derivative, through the
    # nodal coordinates and through the Gauss point coordinate (Popp et al. 2010, Eqs. (A25)-(A26)).
    t = np.einsum("gak,ki->gai", dNdxi_s, coords)
    c = np.cross(t[:, 0], t[:, 1])
    u = c / np.linalg.norm(c, axis=1)[:, None]
    a0, a1 = np.cross(t[:, 1], u), np.cross(u, t[:, 0])
    dJcoords = dNdxi_s[:, 0, :, None] * a0[:, None, :] + dNdxi_s[:, 1, :, None] * a1[:, None, :]
    dCoords = context.dCoords.reshape(-1, context.dCoords.shape[-1])
    dJe = pad(dJcoords.reshape(len(points), -1) @ dCoords)
    d2N = np.array([shapeFunctionSecondDerivatives(s_el, p[2]) for p in points])
    dJdxi = np.empty((len(points), 2))
    for m in range(2):
        along0 = np.einsum("gk,ki->gi", d2N[:, 0, m], coords)
        along1 = np.einsum("gk,ki->gi", d2N[:, 1, m], coords)
        dJdxi[:, m] = np.einsum("gi,gi->g", u, np.cross(along0, t[:, 1]) + np.cross(t[:, 0], along1))
    dJe = dJe + np.einsum("ga,gaz->gz", dJdxi, dxi_s)

    for g, w in enumerate(triangleGaussWeights):
        records.Z.append(Z)
        records.dN_s.append(dN_s[g])
        records.dN_m.append(dN_m[g])
        records.ddG.append(dArea * w)
        records.dNormal.append(dNormal[g])
        records.dJe.append(dJe[g])


def _projectAlongNormal2D(el, coords, point, direction, xi0=0.0, iterations=20, tol=1e-12):
    """Natural coordinate ``xi`` of a line element where ``x(xi) - point`` is parallel to ``direction``:
    Popp et al. (2009), Eq. (44), solved by the local Newton iteration they name."""
    xi = xi0
    for _ in range(iterations):
        local = np.array([xi])
        r = el.getShapeFunctions(local) @ coords - point
        t = el.getShapeFunctionDerivatives(local)[0] @ coords
        f = r[0] * direction[1] - r[1] * direction[0]
        df = t[0] * direction[1] - t[1] * direction[0]
        if df == 0.0:
            return xi, False
        step = f / df
        xi -= step
        if abs(step) < tol:
            return xi, True
    return xi, False


def _projectOntoSlave2D(el, coords, nodalNormals, point, xi0=0.0, iterations=20, tol=1e-12):
    """Natural coordinate ``xi`` of the slave element where ``x(xi) - point`` is parallel to the
    interpolated normal ``n(xi) = sum_j N_j n_j``: Popp et al. (2009), Eq. (45)."""
    xi = xi0
    for _ in range(iterations):
        local = np.array([xi])
        N = el.getShapeFunctions(local)
        dN = el.getShapeFunctionDerivatives(local)[0]
        r = N @ coords - point
        n = N @ nodalNormals
        t, dn = dN @ coords, dN @ nodalNormals
        f = r[0] * n[1] - r[1] * n[0]
        df = (t[0] * n[1] - t[1] * n[0]) + (r[0] * dn[1] - r[1] * dn[0])
        if df == 0.0:
            return xi, False
        step = f / df
        xi -= step
        if abs(step) < tol:
            return xi, True
    return xi, False


def _cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _segment2D(context, candidates, coordinates, constraintIndex, records, result):
    """The 2D segmentation of one slave element along the continuous normal field, Popp et al. (2009),
    Section 4, Eqs. (44)-(48); the derivatives of the projections are their Eqs. (A25)-(A31)."""
    s_el = context.el
    derivatives = context.withDerivatives
    s_end = [0, 1]  # end nodes; of a quadratic element only these are projected (Popp et al. 2009)
    for m_el in candidates:
        m_coords = np.array([coordinates[nd] for nd in m_el.nodes])
        # Slave end nodes onto the master element along their nodal normals, Eq. (44).
        onMaster = [_projectAlongNormal2D(m_el, m_coords, context.coords[k], context.normals[k]) for k in s_end]
        # Master end nodes onto the slave element along the interpolated normal field, Eq. (45).
        onSlave = [_projectOntoSlave2D(s_el, context.coords, context.normals, m_coords[node]) for node in (0, 1)]
        if not all(ok for _, ok in onMaster + onSlave):
            # A projection without solution cannot bound a segment; the pair does not overlap there.
            continue
        # The segment on the slave element: the overlap of [-1, 1] with the projected master element.
        # Each end is either a slave node (xi1 = -1 or 1, its master coordinate from Eq. (44)) or a
        # projected master node (its slave coordinate from Eq. (45), the master coordinate -1 or 1).
        ends = []
        for k, xi1 in ((0, -1.0), (1, 1.0)):
            xi2 = onMaster[k][0]
            if -1.0 <= xi2 <= 1.0:
                ends.append((xi1, xi2, ("slave", k)))
        for node, xi2 in ((0, -1.0), (1, 1.0)):
            xi1 = onSlave[node][0]
            if -1.0 < xi1 < 1.0:
                ends.append((xi1, xi2, ("master", node)))
        if len(ends) < 2:
            continue
        ends.sort(key=lambda e: e[0])
        (a1, a2, aKind), (b1, b2, bKind) = ends[0], ends[-1]
        if b1 - a1 < 1e-12:
            continue

        if derivatives:
            m_ids = np.array([constraintIndex[nd] for nd in m_el.nodes], dtype=np.intp)
            Z, nZ, pad, dMaster = context.extended(m_ids)
            dEnd = {}
            for kind, k in (aKind, bKind):
                dEnd[(kind, k)] = _endDerivatives2D(context, m_el, m_coords, dMaster, pad, kind, k, onMaster, onSlave)
            da1, da2 = dEnd[aKind]
            db1, db2 = dEnd[bKind]

        for gp, w in zip(lineGaussPoints[:, 0], lineGaussWeights):
            # Eq. (46): both sides linear in the segment coordinate.
            xi1 = 0.5 * (1.0 - gp) * a1 + 0.5 * (1.0 + gp) * b1
            xi2 = 0.5 * (1.0 - gp) * a2 + 0.5 * (1.0 + gp) * b2
            Je = s_el.getJacobianAndAreaWeight(np.array([xi1]), context.coords)
            records.N_s.append(s_el.getShapeFunctions(np.array([xi1])))
            records.master.append(m_el)
            records.N_m.append(m_el.getShapeFunctions(np.array([xi2])))
            # Eq. (48): J_seg = J(xi1) (b1 - a1)/2.
            records.dG.append(w * Je * 0.5 * (b1 - a1))
            records.normal.append(None)
            records.Je.append(Je)
            if not derivatives:
                continue
            dxi1 = 0.5 * (1.0 - gp) * da1 + 0.5 * (1.0 + gp) * db1
            dxi2 = 0.5 * (1.0 - gp) * da2 + 0.5 * (1.0 + gp) * db2
            local = np.array([xi1])
            _, dJe_coords = surfaceJacobian(s_el, local, context.coords)
            dJe = pad(dJe_coords @ context.dCoords.reshape(-1, context.dCoords.shape[-1]))
            dJe = dJe + surfaceJacobianNaturalDerivative(s_el, local, context.coords)[0] * dxi1
            records.Z.append(Z)
            records.dN_s.append(np.outer(s_el.getShapeFunctionDerivatives(local)[0], dxi1))
            records.dN_m.append(np.outer(m_el.getShapeFunctionDerivatives(np.array([xi2]))[0], dxi2))
            records.ddG.append(w * 0.5 * ((b1 - a1) * dJe + Je * (db1 - da1)))
            records.dNormal.append(None)
            records.dJe.append(dJe)


def _endDerivatives2D(context, m_el, m_coords, dMaster, pad, kind, k, onMaster, onSlave):
    """Derivatives ``(dxi1, dxi2)`` of one segment end, Popp et al. (2009), Eqs. (A25)-(A30): zero for
    the coordinate that is a node itself, the derivative of the projection Eq. (44) or (45) for the
    other, by the implicit function theorem."""
    s_el = context.el
    nZ = dMaster.shape[-1]
    zero = np.zeros(nZ)
    if kind == "slave":
        # xi1 = -1/1 fixed; xi2 solves (x2(xi2) - x_k) x n_k = 0.
        xi2 = onMaster[k][0]
        local = np.array([xi2])
        N, dN = m_el.getShapeFunctions(local), m_el.getShapeFunctionDerivatives(local)[0]
        r = N @ m_coords - context.coords[k]
        n = context.normals[k]
        dr = np.tensordot(N, dMaster, axes=1) - pad(context.dCoords[k])
        dn = pad(context.dNormals[k])
        df = dr[0] * n[1] - dr[1] * n[0] + r[0] * dn[1] - r[1] * dn[0]
        dfdxi = _cross2(dN @ m_coords, n)
        return zero, -df / dfdxi
    # xi2 = -1/1 fixed; xi1 solves (x1(xi1) - x_l) x n(xi1) = 0.
    xi1 = onSlave[k][0]
    local = np.array([xi1])
    N, dN = s_el.getShapeFunctions(local), s_el.getShapeFunctionDerivatives(local)[0]
    r = N @ context.coords - m_coords[k]
    n = N @ context.normals
    dr = pad(np.tensordot(N, context.dCoords, axes=1)) - dMaster[k]
    dn = pad(np.tensordot(N, context.dNormals, axes=1))
    df = dr[0] * n[1] - dr[1] * n[0] + r[0] * dn[1] - r[1] * dn[0]
    dfdxi = _cross2(dN @ context.coords, n) + _cross2(r, dN @ context.normals)
    return -df / dfdxi, zero


def _assembleSlaveElement(context, records, masterIndex, blocks, kappaSum, result, warn):
    """D, M, the gap matrices and the scaling contributions of one slave element from its Gauss point
    records, and their derivatives."""
    s_el = context.el
    dim = context.dim
    rows = context.slaveRows
    T = s_el.getBasisTransformation()
    N_s = np.array(records.N_s)
    dG = np.array(records.dG)
    N_t = N_s @ T.T

    # Biorthogonality on the integrated part of the element, Popp et al. (2013), Eqs. (21)-(25).
    M_t = _contract("ra,rb,r->ab", N_t, N_t, dG)
    D_t = N_t.T @ dG
    regular = np.linalg.cond(M_t) < CONDITION_LIMIT
    if regular:
        B = np.linalg.inv(M_t) @ T
        A_e = D_t[:, None] * B
    else:
        result.slivers.append(s_el.elNumber)
        _, D_e, A_e = s_el.computeLocalMassMatrices(context.coords)
        B = None

    Phi = N_s @ A_e.T
    masters = {}
    for r, m_el in enumerate(records.master):
        masters.setdefault(m_el.elNumber, (m_el, []))[1].append(r)

    def addBlock(target, rowIdx, colIdx, values):
        rr, cc = np.meshgrid(rowIdx, colIdx, indexing="ij")
        target[0].append(rr.ravel())
        target[1].append(cc.ravel())
        target[2].append(values.ravel())

    D_blk = _contract("ra,rb,r->ab", Phi, N_s, dG)
    addBlock(blocks["D"], rows, rows, D_blk)
    for m_num, (m_el, idx) in masters.items():
        cols = np.array([masterIndex[nd] for nd in m_el.nodes])
        N_m = np.array([records.N_m[r] for r in idx])
        M_blk = _contract("ra,rb,r->ab", Phi[idx], N_m, dG[idx])
        addBlock(blocks["M"], rows, cols, M_blk)
        if dim == 2:
            # The nodal form of the weighted gap, Popp et al. (2009), Eq. (50): M[j,l] n_j.
            for c in range(2):
                addBlock(blocks["Mn"][c], rows, cols, M_blk * context.normals[:, c][:, None])
    if dim == 2:
        for c in range(2):
            addBlock(blocks["Dn"][c], rows, rows, D_blk * context.normals[:, c][:, None])
    if dim == 3:
        # Weighted gap at the Gauss points, Popp et al. (2010), Eq. (41), with the interpolated normal
        # of Eq. (A11): g_j = sum int Phi_j n . (x2 - x1).
        normal = np.array(records.normal)
        Dn = _contract("ra,rb,rc,r->abc", Phi, N_s, normal, dG)
        for c in range(3):
            addBlock(blocks["Dn"][c], rows, rows, Dn[:, :, c])
        for m_num, (m_el, idx) in masters.items():
            cols = np.array([masterIndex[nd] for nd in m_el.nodes])
            N_m = np.array([records.N_m[r] for r in idx])
            Mn = _contract("ra,rb,rc,r->abc", Phi[idx], N_m, normal[idx], dG[idx])
            for c in range(3):
                addBlock(blocks["Mn"][c], rows, cols, Mn[:, :, c])

    # Nodal scaling, Popp et al. (2013), Eq. (37): (1/alpha) sum w N~_j J_c/J_e over the cells.
    alpha = _referenceIntegrals(s_el)
    Je = np.array(records.Je)
    kappaElement = (N_t.T @ (dG / Je)) / alpha
    kappaSum[rows] += kappaElement

    if context.withDerivatives:
        _assembleDerivatives(
            context, records, masters, masterIndex, T, N_s, N_t, dG, M_t, D_t, A_e, B, regular, alpha, Je, result
        )


def _assembleDerivatives(
    context, records, masters, masterIndex, T, N_s, N_t, dG, M_t, D_t, A_e, B, regular, alpha, Je, result
):
    """The derivatives of :func:`_assembleSlaveElement`, by the same steps: Popp et al. (2010),
    Eqs. (A4)-(A8) for D, M and the gap, Popp et al. (2013), Eqs. (26)-(28) for the dual
    coefficients and Eq. (38) for the scaling factor."""
    s_el = context.el
    dim = context.dim
    rows = context.slaveRows
    n_s = len(rows)
    support = np.unique(np.concatenate([context.Z] + records.Z))
    P = dim * len(support)
    R = len(records)

    columnCache = {}

    def onSupport(Z, d):
        # The nodes of one Gauss point's coordinate vector are distinct (slave side and one master
        # element), so moving its Jacobian onto the support is a plain assignment of columns.
        key = Z.tobytes()
        if key not in columnCache:
            columnCache[key] = (dim * np.searchsorted(support, Z)[:, None] + np.arange(dim)).ravel()
        out = np.zeros(d.shape[:-1] + (P,))
        out[..., columnCache[key]] = d
        return out

    dN_s = np.array([onSupport(Z, d) for Z, d in zip(records.Z, records.dN_s)])
    ddG = np.array([onSupport(Z, d) for Z, d in zip(records.Z, records.ddG)])
    dJe = np.array([onSupport(Z, d) for Z, d in zip(records.Z, records.dJe)])
    dN_t = _contract("ab,rbz->raz", T, dN_s)

    # Eq. (26) of Popp et al. (2013) for A_e = diag(D_t) M_t^-1 T, with Eqs. (27)-(28).
    dM_t = _contract("raz,rb,r->abz", dN_t, N_t, dG)
    dM_t = dM_t + dM_t.transpose(1, 0, 2) + _contract("ra,rb,rz->abz", N_t, N_t, ddG)
    dD_t = _contract("raz,r->az", dN_t, dG) + N_t.T @ ddG
    if regular:
        M_inv = np.linalg.inv(M_t)
        dB = -_contract("ab,bcz,cd->adz", M_inv, dM_t, B)
        dA = dD_t[:, None, :] * B[:, :, None] + D_t[:, None, None] * dB
    else:
        # The reference-element coefficients diag(D_e) M_e^-1 T integrated over the whole element; only
        # the surface Jacobian depends on the configuration (Popp et al. 2010, Eqs. (A24)-(A26)).
        coords = context.coords
        M_e = np.zeros((n_s, n_s))
        dM_e = np.zeros((n_s, n_s, P))
        D_e = np.zeros(n_s)
        dD_e = np.zeros((n_s, P))
        dCoords = context.dCoords.reshape(-1, context.dCoords.shape[-1]) @ scatterMatrix(context.Z, support, dim)
        for xi, w in zip(*s_el.getQuadraturePoints()):
            Nt = T @ s_el.getShapeFunctions(xi)
            J, dJ = surfaceJacobian(s_el, xi, coords)
            dJ = dJ @ dCoords
            M_e += np.outer(Nt, Nt) * J * w
            dM_e += np.outer(Nt, Nt)[:, :, None] * dJ * w
            D_e += Nt * J * w
            dD_e += np.outer(Nt, dJ) * w
        M_inv = np.linalg.pinv(M_e)
        B_e = M_inv @ T
        dB_e = -_contract("ab,bcz,cd->adz", M_inv, dM_e, B_e)
        dA = dD_e[:, None, :] * B_e[:, :, None] + D_e[:, None, None] * dB_e

    Phi = N_s @ A_e.T
    dPhi = _contract("abz,rb->raz", dA, N_s) + _contract("ab,rbz->raz", A_e, dN_s)

    def product(dPhiR, PhiR, otherN, dOtherN, dGR, ddGR):
        return (
            _contract("raz,rb,r->abz", dPhiR, otherN, dGR)
            + _contract("ra,rbz,r->abz", PhiR, dOtherN, dGR)
            + _contract("ra,rb,rz->abz", PhiR, otherN, ddGR)
        )

    dD = product(dPhi, Phi, N_s, dN_s, dG, ddG)
    contributions = [("D", rows, dD)]
    masterBlocks = []
    for m_num, (m_el, idx) in masters.items():
        cols = np.array([masterIndex[nd] for nd in m_el.nodes])
        N_m = np.array([records.N_m[r] for r in idx])
        dN_m = np.array([onSupport(records.Z[r], records.dN_m[r]) for r in idx])
        masterBlocks.append((cols, idx, N_m, dN_m, product(dPhi[idx], Phi[idx], N_m, dN_m, dG[idx], ddG[idx])))
    if dim == 3:
        normal = np.array(records.normal)
        dNormal = np.array([onSupport(Z, d) for Z, d in zip(records.Z, records.dNormal)])

        def gapProduct(idx, otherN, dOtherN):
            PhiR, dPhiR, nR, dnR, dGR, ddGR = Phi[idx], dPhi[idx], normal[idx], dNormal[idx], dG[idx], ddG[idx]
            return (
                _contract("raz,rb,rc,r->abcz", dPhiR, otherN, nR, dGR)
                + _contract("ra,rbz,rc,r->abcz", PhiR, dOtherN, nR, dGR)
                + _contract("ra,rb,rcz,r->abcz", PhiR, otherN, dnR, dGR)
                + _contract("ra,rb,rc,rz->abcz", PhiR, otherN, nR, ddGR)
            )

        contributions.append(("Dn", rows, gapProduct(np.arange(R), N_s, dN_s)))
        for cols, idx, N_m, dN_m, _ in masterBlocks:
            contributions.append(("Mn", cols, gapProduct(idx, N_m, dN_m)))
    for cols, idx, N_m, dN_m, dMblock in masterBlocks:
        contributions.append(("M", cols, dMblock))
    if dim == 2:
        # d(D[j,k] n_j) = dD[j,k] n_j + D[j,k] dn_j, and likewise for M, Popp et al. (2009), Eq. (A36).
        D_blk = _contract("ra,rb,r->ab", Phi, N_s, dG)
        n, dn = context.normals, context.dNormals @ scatterMatrix(context.Z, support, dim)
        contributions.append(
            ("Dn", rows, dD[:, :, None, :] * n[:, None, :, None] + D_blk[:, :, None, None] * dn[:, None, :, :])
        )
        for cols, idx, N_m, dN_m, dMblock in masterBlocks:
            M_blk = _contract("ra,rb,r->ab", Phi[idx], N_m, dG[idx])
            contributions.append(
                ("Mn", cols, dMblock[:, :, None, :] * n[:, None, :, None] + M_blk[:, :, None, None] * dn[:, None, :, :])
            )
    # Eq. (38): the scaling factor through N~_j (via the Gauss point), J_c and J_e.
    dKappa = (
        _contract("raz,r->az", dN_t, dG / Je)
        + _contract("ra,rz->az", N_t, ddG / Je[:, None])
        - _contract("ra,r,rz->az", N_t, dG / Je**2, dJe)
    ) / alpha[:, None]

    for a, I in enumerate(rows):  # noqa: E741 - slave node index
        for kind, cols, d in contributions:
            result.rowContributions[I].append((kind, cols, d[a], support))
        result.rowContributions[I].append(("kappa", None, dKappa[a], support))
