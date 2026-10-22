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
    isInsideReferenceElement,
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

"""
The mortar coupling of a non-mortar (slave) and a mortar (master) contact surface: the matrices D and
M (called C in the constraint), the gap matrices from which the weighted gap is formed, and the nodal
scaling factor.

Every step follows a published formulation; where two sources are composed, or a source is applied
beyond the case it treats, the step says so.

Three dimensions -- Popp, Gitterle, Gee & Wall (2010), Section 4 and Appendix A:

* an auxiliary plane per slave (sub-)segment through its centre ``x0``, with the interpolated nodal
  normal there as its normal ``n0``, Eqs. (A15) and (A18);
* the orthogonal projection of slave and master (sub-)segments into it, Eq. (A17), clipping by
  Sutherland & Hodgman, triangulation into integration cells, a 7-point rule per cell (Puso &
  Laursen 2004, Section 3.1);
* quadratic facets split into linear sub-segments for the clipping, Popp et al. (2010), Section 4.2,
  and Puso, Laursen & Solberg (2008), Fig. 3;
* the Gauss point mapped back along ``n0``, Eq. (A12), by the inverse of the isoparametric map of
  the whole projected facet, and attributed to the facet that contains it. For linear facets this
  is Eq. (A12) itself. For quadratic ones it replaces the affine sub-segment map of Popp et al.
  (2010), Eq. (42), and Puso et al. (2008), Eqs. (31)-(32), which keeps a point with the facet whose
  sub-segment produced it even where an edge curved in the surface puts it in the neighbour; the
  attribution is this implementation's step (see :func:`_locate`);
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
    projectionFailures, firstProjectionFailure
        Gauss points whose back-mapping did not converge, and where the first one was.
    nonConvex, slivers
        Element numbers of non-convex sub-segments and of slave elements that took the
        reference-element fallback.
    reattributedGaussPoints, droppedGaussPoints
        3D Gauss points that lie in another facet than the one whose sub-segment produced them, and
        those that lie in no facet at all (see :func:`_locate`).
    """

    D: csr_matrix
    M: csr_matrix
    Dn: list
    Mn: list
    kappa: np.ndarray
    projectionFailures: int = 0
    firstProjectionFailure: tuple | None = None
    nonConvex: list = field(default_factory=list)
    slivers: list = field(default_factory=list)
    reattributedGaussPoints: int = 0
    droppedGaussPoints: int = 0


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

    def __len__(self):
        return len(self.dG)


def computeMortarCoupling(
    nonMortarFacets: list,
    mortarFacets: list,
    coordinates: dict,
    slaveIndex: dict,
    masterIndex: dict,
    normals: np.ndarray,
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
    slaveIndex, masterIndex
        Node -> row / column index of D and M.
    normals
        ``(nSlave, dim)`` the nodal normals, Popp et al. (2010), Eq. (37).
    dim
        2 or 3.
    elementClass
        The contact element class, to evaluate linear sub-segments with.
    warn
        ``warn(key, message)`` for one-off diagnostics.
    """
    nSlave, nMaster = len(slaveIndex), len(masterIndex)
    result = MortarCoupling(None, None, [], [], np.zeros(nSlave))

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

    owners = _Owners(nonMortarFacets, coordinates, slaveIndex, normals, dim)
    for s_el in nonMortarFacets:
        s_coords = np.array([coordinates[nd] for nd in s_el.nodes])
        lo, hi = s_coords.min(axis=0), s_coords.max(axis=0)
        margin = max(0.15 * np.max(hi - lo), 0.1)
        candidates = []
        queryBoundingVolumeHierarchy(bvh, lo - margin, hi + margin, candidates)
        if not candidates:
            continue
        if dim == 3:
            _segment3D(owners.context(s_el), candidates, owners, coordinates, elementClass, result)
        else:
            _segment2D(owners.context(s_el), candidates, coordinates, owners.records(s_el), result)

    # A Gauss point may belong to another slave element than the one that produced it, so the
    # elements are assembled once the whole surface is segmented.
    for s_el in nonMortarFacets:
        records = owners.recordsIfAny(s_el)
        if records is None or not len(records):
            continue
        _assembleSlaveElement(owners.context(s_el), records, masterIndex, blocks, kappaSum, result, warn)

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
    return result


class _Owners:
    """Per slave element its context and its Gauss point records, created on first use, and the slave
    elements sharing a node with it -- where a Gauss point is looked for that its own element does not
    contain (see :func:`_locate`)."""

    def __init__(self, nonMortarFacets, coordinates, slaveIndex, normals, dim):
        self.arguments = (coordinates, slaveIndex, normals, dim)
        self._contexts, self._records = {}, {}
        byNode = {}
        for f in nonMortarFacets:
            for nd in f.nodes:
                byNode.setdefault(nd, []).append(f)
        self.neighbours = {
            f.elNumber: list({g.elNumber: g for nd in f.nodes for g in byNode[nd] if g is not f}.values())
            for f in nonMortarFacets
        }

    def context(self, el):
        if el.elNumber not in self._contexts:
            self._contexts[el.elNumber] = _SlaveContext(el, *self.arguments)
        return self._contexts[el.elNumber]

    def records(self, el):
        return self._records.setdefault(el.elNumber, _Records())

    def recordsIfAny(self, el):
        return self._records.get(el.elNumber)


class _SlaveContext:
    """What every Gauss point of one slave element needs: its nodes, coordinates and normals."""

    def __init__(self, s_el, coordinates, slaveIndex, normals, dim):
        self.el = s_el
        self.dim = dim
        self.coords = np.array([coordinates[nd] for nd in s_el.nodes])
        self.slaveRows = np.array([slaveIndex[nd] for nd in s_el.nodes])
        self.normals = normals[self.slaveRows]


class _AuxiliaryPlane:
    """The auxiliary plane of one slave sub-segment, ``x0``, ``n0`` and the tangent basis, with the
    plane coordinates of further facets computed on demand: a Gauss point that the facet whose
    sub-segment produced it does not contain is looked up in its neighbours (see :func:`_locate`)."""

    def __init__(self, x0, n0, t1, t2, coordinates):
        self.x0, self.n0, self.t1, self.t2 = x0, n0, t1, t2
        self.coordinates = coordinates
        self.cache = {}

    def of(self, el):
        if el.elNumber not in self.cache:
            coords = np.array([self.coordinates[nd] for nd in el.nodes])
            self.cache[el.elNumber] = toPlaneCoordinates(coords, self.x0, self.t1, self.t2)
        return self.cache[el.elNumber]


def _locate(el, q, x, alternatives, plane):
    """The facet that contains the plane point ``x`` and its natural coordinate there.

    The point is first mapped back by the inverse of the facet's own isoparametric map in the plane
    -- for a linear facet the inverse of Popp et al. (2010), Eq. (A12); for a quadratic one the map of
    the whole projected facet rather than the affine sub-segment map of their Eq. (42). Where a
    quadratic facet has an edge that is curved in the surface, the straight sub-segments clip the
    region between chord and edge to the wrong facet; mapped back into that facet, the point lands
    outside its reference domain and its shape functions are extrapolated. Such a point is then
    attributed to the facet among ``alternatives`` that contains it. This attribution is this
    implementation's step, not from a source; Popp et al. (2010), Section 4.2, name the
    approximation it removes, and :class:`TestCurvedEdgeAttribution` measures it.

    Returns ``(facet, natural coordinate, converged)``; ``(None, None, True)`` if the point provably
    lies in no facet, which happens only beyond the curved boundary of a surface. A back-mapping that
    does not converge proves nothing: such a point stays with ``el`` and is reported.
    """
    xi, ok = mapPlaneToNatural(el, q, x)
    if ok and isInsideReferenceElement(el, xi):
        return el, xi, True
    for other in alternatives:
        if other is el:
            continue
        xi_o, ok_o = mapPlaneToNatural(other, plane.of(other), x)
        if ok_o and isInsideReferenceElement(other, xi_o):
            return other, xi_o, True
    if not ok:
        return el, xi, False
    return None, None, True


def _segment3D(context, candidates, owners, coordinates, elementClass, result):
    """The 3D segmentation of one slave element, Popp et al. (2010), Section 4.1 and Algorithm 1, on the
    linear sub-segments of Section 4.2, with every Gauss point attributed to the slave and master
    facets that contain it (see :func:`_locate`)."""
    s_el = context.el
    neighbours = owners.neighbours[s_el.elNumber]
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
        plane = _AuxiliaryPlane(x0, n0, t1, t2, coordinates)
        q_sFull = plane.of(s_el)
        q_s = q_sFull[s_sub]
        if not isConvexPolygon(q_s):
            result.nonConvex.append(("non-mortar", s_el.elNumber))

        for m_el in candidates:
            q_mFull = plane.of(m_el)
            for m_sub in getSubCells(m_el):
                q_m = q_mFull[m_sub]
                if not isConvexPolygon(q_m):
                    result.nonConvex.append(("mortar", m_el.elNumber))
                polygon = sutherlandHodgmanClip(q_m, q_s)
                if len(polygon) < 3:
                    continue

                for tri in triangulatePolygon(polygon):
                    v0, v1, v2 = tri
                    area = abs((v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1]))
                    if area < 1e-14:
                        continue
                    # The points of the cell, grouped by the pair of facets they are attributed to.
                    groups = {}
                    for g, (L1, L2) in enumerate(triangleGaussPoints):
                        x_gp = (1.0 - L1 - L2) * v0 + L1 * v1 + L2 * v2
                        # Back onto both surfaces along n0, Eq. (A12).
                        own_s, xi_s, ok_s = _locate(s_el, q_sFull, x_gp, neighbours, plane)
                        own_m, xi_m, ok_m = _locate(m_el, q_mFull, x_gp, candidates, plane)
                        if own_s is None or own_m is None:
                            result.droppedGaussPoints += 1
                            continue
                        if own_s is not s_el or own_m is not m_el:
                            result.reattributedGaussPoints += 1
                        for ok, side, el in ((ok_s, "slave", own_s), (ok_m, "master", own_m)):
                            if not ok:
                                result.projectionFailures += 1
                                result.firstProjectionFailure = result.firstProjectionFailure or (
                                    side,
                                    el.elNumber,
                                    el.elType,
                                )
                        key = (own_s.elNumber, own_m.elNumber)
                        groups.setdefault(key, (own_s, own_m, []))[2].append((g, xi_s, xi_m))

                    for own_s, own_m, points in groups.values():
                        owner = owners.context(own_s)
                        records = owners.records(own_s)
                        weights = triangleGaussWeights[[p[0] for p in points]]
                        N_s = np.array([own_s.getShapeFunctions(p[1]) for p in points])
                        dNdxi_s = np.array([own_s.getShapeFunctionDerivatives(p[1]) for p in points])
                        normal_raw = N_s @ owner.normals
                        records.N_s.extend(N_s)
                        records.master.extend([own_m] * len(points))
                        records.N_m.extend(own_m.getShapeFunctions(p[2]) for p in points)
                        records.dG.extend(area * weights)
                        records.normal.extend(normal_raw / np.linalg.norm(normal_raw, axis=1)[:, None])
                        # The element Jacobian |x,xi x x,eta| (cf. ContactElement.getJacobianAndAreaWeight).
                        tangents = np.einsum("gak,ki->gai", dNdxi_s, owner.coords)
                        records.Je.extend(np.linalg.norm(np.cross(tangents[:, 0], tangents[:, 1]), axis=1))


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


def _segment2D(context, candidates, coordinates, records, result):
    """The 2D segmentation of one slave element along the continuous normal field, Popp et al. (2009),
    Section 4, Eqs. (44)-(48)."""
    s_el = context.el
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
                ends.append((xi1, xi2))
        for node, xi2 in ((0, -1.0), (1, 1.0)):
            xi1 = onSlave[node][0]
            if -1.0 < xi1 < 1.0:
                ends.append((xi1, xi2))
        if len(ends) < 2:
            continue
        ends.sort(key=lambda e: e[0])
        (a1, a2), (b1, b2) = ends[0], ends[-1]
        if b1 - a1 < 1e-12:
            continue

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


def _assembleSlaveElement(context, records, masterIndex, blocks, kappaSum, result, warn):
    """D, M, the gap matrices and the scaling contributions of one slave element from its Gauss point
    records."""
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
        A_e = D_t[:, None] * (np.linalg.inv(M_t) @ T)
    else:
        result.slivers.append(s_el.elNumber)
        _, _, A_e = s_el.computeLocalMassMatrices(context.coords)

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
