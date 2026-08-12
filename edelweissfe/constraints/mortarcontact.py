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
import warnings

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.models.femodel import FEModel
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.inputlanguage import InputLanguage, Module
from edelweissfe.utils.misc import (
    caseInsensitiveKwargsChecker,
    castKwargsValuesAndAddDefaults,
)

"""
A mortar contact constraint with Lagrange multipliers and dual basis functions.

SIGN CONVENTION. The multiplier follows the contact literature (Popp, Gee & Wall
2009; Gitterle et al. 2010; Popp et al. 2012; Farah 2018): lambda_I is the
NEGATIVE slave traction, hence lambda_I >= 0 in compression, and the contact
force it applies to the slave node is -lambda_I * D_IK * n_I, i.e. directed
against the outward slave normal. For a positive nodal weight D_II - the regular
case - lambda_I therefore IS the nodal contact pressure and can be read as such;
p_n = lambda_I * sgn(D_II) generalizes that to the negative weights a partially
covered CONQUAD9 can produce. This is the convention every formula in the cited
references uses, which is what makes them transferable without sign surgery -
notably the frictional NCP, where the same expression sits inside nested max()
functions and its sign decides a branch rather than a scaling.

The Signorini conditions (normal pressure p_I >= 0, weighted gap g_I >= 0,
complementarity p_I g_I = 0) are enforced through a single non-smooth
complementarity function (NCP)

    C_n,I = p_I - max(0, p_I - c_n D_II^-1 g_I) = 0
        <=>  active  iff  s_n,I = p_I - c_n D_II^-1 g_I > 0,

solved by a primal-dual active set strategy = semi-smooth Newton method
(Hueber & Wohlmuth 2005, Eq. (3.9); Gitterle et al. 2010, Eq. (55); Popp et al.
2012, Eq. (5.1); Farah 2018, Eq. (3.58)). The active set is re-evaluated in
EVERY Newton iteration and the outer (semi-smooth) loop is converged as soon as
the set no longer changes; c_n > 0 is purely algorithmic (g_I -> 0 at
convergence, hence the converged solution is c_n-independent). The admissible
values form a band: below the lower bound c_0 ~ E the active set does not
converge (Hueber & Wohlmuth 2005, sec. 7), far above it it chatters (Gitterle et
al. 2010, Table I). c_n therefore defaults to the smallest initial Young's
modulus of the materials adjacent to the interface.

D and C are stored as sparse matrices, which is what the dual basis is for
(Popp et al. 2012, sec. 4.2 and 5).

Only the saddle-point formulation (explicit multiplier DOFs) is implemented.
A "dual condensation" variant (eliminating the multipliers via lambda_I =
g_weak,I / D_II, evaluated fresh from the current gap each iteration) was
implemented and removed again: it is not the algebraic elimination described
in Farah (2018), Sec. 3.5.3, Eqs. (3.62)-(3.65)/Popp et al. (2012). The
correct elimination is a Schur complement of the *already assembled* slave
displacement row (which includes the surrounding bulk material stiffness,
not just this constraint's own D/C/n data) and replaces that row with the
linearized constraint, moving its force contribution to the master row via
P = D^-1 M. The tried shortcut instead applies lambda_I as if it were a
penalty force with penalty stiffness 1/D_II (~O(1) here vs. e.g. ~4e4 MPa
concrete stiffness), which under-transmits contact force by ~25-30x on a
real (POT_Dejori) problem while still passing an isolated single-state unit
test (which only checks that a *prescribed* lambda produces matching forces,
not that the solver's own iteration converges to the correct one). Redoing
this correctly requires the constraint interface to replace rows rather
than only additively contribute to K/PExt - a bigger architectural change,
not attempted here.
"""

module = Module(
    "mortarcontact",
    "A mortar contact constraint with Lagrange multipliers and dual basis functions.",
)

inputLanguage = InputLanguage()

keyword = "constraint"
if keyword in inputLanguage:
    inputLanguage[keyword].addModule(module)

module.addRequiredArg("nonMortarSurface", "The non-mortar (slave) surface name.", str)
module.addRequiredArg("mortarSurface", "The mortar (master) surface name.", str)
module.addOptionalArg("field", "The field this constraint acts on (e.g. displacement).", str, "displacement")
module.addOptionalArg(
    "cn",
    "Semi-smooth-Newton complementarity parameter c_n (> 0) of the normal contact "
    "NCP. It enters the semi-smooth active-set indicator s_n = p_n - c_n*g_sep "
    "(active iff s_n > 0; Gitterle et al. 2010 Eq. 55; Hueber & Wohlmuth 2005; MOOSE "
    "ComputeWeightedGapLMMechanicalContact), where g_sep = g_weak/D_II is the weak gap "
    "normalized by the nodal mortar weight D_II, i.e. the physical nodal opening, so "
    "that c_n*g_sep is a pressure directly comparable to p_n. "
    "Purely algorithmic - no effect on the converged solution (g_sep -> 0 there), so "
    "the converged normal-contact result is identical to any admissible active-set "
    "rule. It MUST be chosen at the order of Young's modulus of the softer contacting "
    "body. The admissible values form a BAND, and both ends are documented: Hueber & "
    "Wohlmuth (2005), sec. 7 and Table 7, report a lower bound c0 below which the "
    "active set does not converge, note that 'for c > c0, the influence of c on Nl is "
    "negligible', and find that 'c0 depends linearly on E'; Gitterle et al. (2010), "
    "Table I, show the other end, where c_n = 1e5 and 1e6 produce repeated active-set "
    "changes (chattering) and non-convergence. Popp & Gee & Wall (2009) suggest "
    "'the order of Young's modulus E of the contacting bodies'. Farah (2018), text to "
    "Eq. (3.58), repeats that suggestion but uses c_n = 1 for all examples. Purely "
    "algorithmic either way - no effect on the converged solution (g_sep -> 0 there), "
    "so the converged normal-contact result is identical to any admissible active-set "
    "rule. "
    "Leave it at 0 (the default) to derive it as the smallest initial Young's modulus "
    "among the materials adjacent to the two contact surfaces; the derived value is "
    "reported once so it stays checkable.",
    float,
    0.0,
)

documentation = [module]


def map_2d_to_natural(el, coords_2d, point_2d, max_iter=10, tol=1e-12):
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
    suite before this criterion was added: 189 of 780476 calls "failed" this way,
    every one of them a 2D call at an open gap, every returned coordinate exact.

    The update test is also the scale-invariant one - the natural coordinate is
    dimensionless, whereas the residual carries the model's length unit.

    `converged` is True when either criterion was met. It is False only when the
    iteration ran out of passes or hit a singular Jacobian - i.e. when the returned
    coordinate is the last iterate rather than a solution. Nothing downstream can
    tell those apart on its own (the value looks like any other), so the caller
    counts them and reports them; see the `gp_projection_failed` diagnostic in
    compute_mortar_coupling_matrices. That the closest-point projection need not be
    solvable at all, and under which conditions it fails, is the subject of
    Konyukhov & Schweizerhof (2008); the local Newton used here is the standard one
    (Wriggers, Computational Contact Mechanics).

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
# using their mid-side nodes, following MOOSE (AutomaticMortarGeneration,
# "Step 1.1: Linearize secondary face elements") and the segment-based
# integration of Farah/Popp/Wall. The curvature of the facet enters only
# through the shape function evaluation at the mapped Gauss points, never
# through the clipping geometry itself.
SUB_CELL_MAP = {
    "CONLINE2": [[0, 1]],
    "CONLINE3": [[0, 2], [2, 1]],
    "CONQUAD4": [[0, 1, 2, 3]],
    "CONTRI3": [[0, 1, 2]],
    "CONQUAD8": [[0, 4, 7], [4, 1, 5], [5, 2, 6], [7, 6, 3], [4, 5, 6, 7]],
    "CONQUAD9": [[0, 4, 8, 7], [4, 1, 5, 8], [8, 5, 2, 6], [7, 8, 6, 3]],
    "CONTRI6": [[0, 3, 5], [3, 4, 5], [3, 1, 4], [5, 4, 2]],
}


def _coo_to_csr(vals: list, rows: list, cols: list, shape: tuple) -> csr_matrix:
    """Build a CSR matrix from lists of per-block triples, summing duplicates."""
    if not vals:
        return csr_matrix(shape)
    return coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=shape,
    ).tocsr()


def _nonzero_rows(A: csr_matrix, n_rows: int, rtol: float = 1e-12) -> tuple[list, list]:
    """Per-row (column indices, values) of a CSR matrix, filtered RELATIVE to max|A|.

    The threshold is part of the formulation's arithmetic, not of the storage: it
    decides which coupling terms enter the weak gap and the stiffness, and a stored
    CSR entry can be below it (coo -> csr sums duplicates, it does not prune).

    It is deliberately a RELATIVE threshold. The entries of D and C are integrals
    int(Phi_I N_K) and therefore carry the unit of a facet AREA, so any absolute
    bound is a statement about the model's length unit rather than about the
    arithmetic. What has to be separated here is round-off dust - which is
    ~eps*max|A| by construction, independent of the unit - from a genuine coupling
    term, and rtol*max|A| with rtol = 1e-12 sits four orders above the dust and
    twelve below the signal for ANY length scale. (The bound this replaces was an
    absolute 1e-14; on a model of order one it is the same filter to within an order
    of magnitude, which is why the measured values in the documentation are
    unchanged - see the tolerance table in "Implizite Annahmen der Nachbarsuche".)

    Caching the VALUES next to the indices - rather than re-indexing A[I, nz] on every
    Newton iteration - is the other half of the point: the pattern and the values are
    fixed for the whole increment (frozen geometry), while applyConstraint runs once
    per iteration.
    """
    idx_per_row, val_per_row = [], []
    indptr, indices, data = A.indptr, A.indices, A.data
    # An empty matrix has no scale; the loop below then keeps nothing, which is right.
    tol = rtol * np.max(np.abs(data)) if data.size else 0.0
    for I in range(n_rows):
        lo, hi = indptr[I], indptr[I + 1]
        cols, vals = indices[lo:hi], data[lo:hi]
        keep = np.abs(vals) > tol
        idx_per_row.append(cols[keep])
        val_per_row.append(vals[keep])
    return idx_per_row, val_per_row


def is_convex_polygon(poly_2d, tol: float = 1e-14) -> bool:
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
        and `triangulate_polygon` fans from vertex 0 with a per-triangle |area|.
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


def get_sub_cells(el) -> list[list[int]]:
    """Return the linear sub-cell decomposition (local node indices) of a contact facet."""
    el_type = el.elType.upper()
    if el_type not in SUB_CELL_MAP:
        raise NotImplementedError(
            f"No linear sub-cell decomposition defined for element type '{el_type}'."
        )
    return SUB_CELL_MAP[el_type]


def facet_normal(coords: np.ndarray) -> np.ndarray:
    """Unnormalized area-weighted normal of a flat linear facet (3 or 4 corner nodes).

    Used for the AUXILIARY PLANE of the segmentation only - one plane per slave
    sub-cell, and the sub-cells are linear by construction, so this is the exact
    normal of their plane rather than an approximation.

    It is deliberately NOT what compute_normals builds the nodal normal n_I from:
    one constant direction per facet cannot distinguish a mid-side node from its
    corners, which costs an order of convergence on a curved slave surface. See
    compute_normals.
    """
    if len(coords) == 3:
        return 0.5 * np.cross(coords[1] - coords[0], coords[2] - coords[0])
    return 0.5 * np.cross(coords[2] - coords[0], coords[3] - coords[1])


# 3-point Gauss-Legendre rule (degree 5) on [-1, 1] for 1D line segments
LINE_GAUSS_PTS = np.array([
    [-np.sqrt(0.6)],
    [0.0],
    [np.sqrt(0.6)],
])
LINE_GAUSS_W = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])


# 7-point symmetric Gauss rule (degree 5) on the reference triangle.
# Farah (2018), App. A.1.1: 7 points per integration cell are recommended,
# since the nonlinear projection between the auxiliary plane and the curved
# element surfaces raises the polynomial degree of the integrand.
_TRI_A = (6.0 - np.sqrt(15.0)) / 21.0
_TRI_B = (6.0 + np.sqrt(15.0)) / 21.0
TRI_GAUSS_PTS = np.array([
    [1.0 / 3.0, 1.0 / 3.0],
    [_TRI_A, _TRI_A],
    [_TRI_A, 1.0 - 2.0 * _TRI_A],
    [1.0 - 2.0 * _TRI_A, _TRI_A],
    [_TRI_B, _TRI_B],
    [_TRI_B, 1.0 - 2.0 * _TRI_B],
    [1.0 - 2.0 * _TRI_B, _TRI_B],
])
TRI_GAUSS_W = np.array([
    9.0 / 80.0,
    (155.0 - np.sqrt(15.0)) / 2400.0,
    (155.0 - np.sqrt(15.0)) / 2400.0,
    (155.0 - np.sqrt(15.0)) / 2400.0,
    (155.0 + np.sqrt(15.0)) / 2400.0,
    (155.0 + np.sqrt(15.0)) / 2400.0,
    (155.0 + np.sqrt(15.0)) / 2400.0,
])


class BVHNode:
    """A node in the Bounding Volume Hierarchy (BVH) tree for contact detection."""
    def __init__(self, aabb_min, aabb_max, left=None, right=None, facets=None):
        self.aabb_min = aabb_min
        self.aabb_max = aabb_max
        self.left = left
        self.right = right
        self.facets = facets  # Only set for leaf nodes

    def is_leaf(self) -> bool:
        return self.facets is not None


def build_bvh(facets_with_bounds) -> BVHNode:
    """Recursively build a binary BVH tree from a list of tuples: (facet, centroid, aabb_min, aabb_max)."""
    if not facets_with_bounds:
        return None

    # Compute enclosing AABB for all facets in the current subset
    mins = np.array([f[2] for f in facets_with_bounds])
    maxs = np.array([f[3] for f in facets_with_bounds])
    aabb_min = np.min(mins, axis=0)
    aabb_max = np.max(maxs, axis=0)

    # Leaf node base case: 2 or fewer facets
    if len(facets_with_bounds) <= 2:
        return BVHNode(aabb_min, aabb_max, facets=[f[0] for f in facets_with_bounds])

    # Find the longest axis to split along
    extent = aabb_max - aabb_min
    split_axis = np.argmax(extent)

    # Sort facets by their centroid along the longest axis
    facets_with_bounds.sort(key=lambda f: f[1][split_axis])
    mid = len(facets_with_bounds) // 2

    # Recursively build child nodes
    left_child = build_bvh(facets_with_bounds[:mid])
    right_child = build_bvh(facets_with_bounds[mid:])

    return BVHNode(aabb_min, aabb_max, left=left_child, right=right_child)


def query_bvh(node: BVHNode, q_min, q_max, candidates: list):
    """Query the BVH tree to find all candidate facets overlapping the query AABB."""
    if node is None:
        return

    # Check if query AABB overlaps with the node's AABB
    if not (np.all(q_min <= node.aabb_max) and np.all(node.aabb_min <= q_max)):
        return

    if node.is_leaf():
        candidates.extend(node.facets)
    else:
        query_bvh(node.left, q_min, q_max, candidates)
        query_bvh(node.right, q_min, q_max, candidates)


class Constraint(ConstraintBase):
    @caseInsensitiveKwargsChecker([kw.name for kw in module.requiredArgs], [kw.name for kw in module.optionalArgs])
    @castKwargsValuesAndAddDefaults(module)
    def __init__(self, name: str, model: FEModel, *args, **kwargs):
        super().__init__(name, model, *args, **kwargs)

        self.model = model
        kwargs = CaseInsensitiveDict(kwargs)

        self._name = name
        # Runtime diagnostics. Initialized early so that the input checks further
        # down can already use them.
        self._warned = set()
        self.field = kwargs["field"]
        self.sizeField = getFieldSize(self.field, model.domainSize)

        non_mortar_surf_name = kwargs["nonMortarSurface"]
        mortar_surf_name = kwargs["mortarSurface"]

        if non_mortar_surf_name not in model.surfaces:
            raise KeyError(f"Non-mortar surface '{non_mortar_surf_name}' not found in model.")
        if mortar_surf_name not in model.surfaces:
            raise KeyError(f"Mortar surface '{mortar_surf_name}' not found in model.")

        self.nonMortarSurface = model.surfaces[non_mortar_surf_name]
        self.mortarSurface = model.surfaces[mortar_surf_name]

        # Extract facets and validate that they are contact elements
        self.non_mortar_facets = []
        for faceID, elements in self.nonMortarSurface.items():
            for el in elements:
                if not el.elType.upper().startswith("CON"):
                    raise ValueError(
                        f"MortarContact only supports explicit contact elements starting with 'CON'. "
                        f"Got element type '{el.elType}'."
                    )
                self.non_mortar_facets.append((el, faceID))

        self.mortar_facets = []
        for faceID, elements in self.mortarSurface.items():
            for el in elements:
                if not el.elType.upper().startswith("CON"):
                    raise ValueError(
                        f"MortarContact only supports explicit contact elements starting with 'CON'. "
                        f"Got element type '{el.elType}'."
                    )
                self.mortar_facets.append((el, faceID))

        # Identify nodes
        self.non_mortar_nodes = []
        for el, faceID in self.non_mortar_facets:
            for node in el.nodes:
                if node not in self.non_mortar_nodes:
                    self.non_mortar_nodes.append(node)

        self.mortar_nodes = []
        for el, faceID in self.mortar_facets:
            for node in el.nodes:
                if node not in self.mortar_nodes:
                    self.mortar_nodes.append(node)

        # The formulation assumes a fixed pair of DISJOINT surfaces. A node listed
        # on both of them is not merely questionable input, it makes the system
        # singular: for coinciding facets C = D holds exactly, hence the weak gap
        # g_I = -sum_K D_IK (x_K.n) + sum_J C_IJ (x_J.n) vanishes identically for
        # every configuration, and the corresponding lambda row of K cancels out
        # the moment such a node becomes active. Without this check the user only
        # sees an unintelligible linear-solver failure.
        # NOTE: this rules out MISDECLARED surfaces, not self-contact. Genuine
        # self-contact would additionally require an exclusion rule for a facet's
        # own and adjacent facets, an unambiguous nodal normal at doubly
        # classified nodes and a dynamic surface pairing - see Yang & Laursen
        # (2008); it is out of scope here either way.
        shared_nodes = set(self.non_mortar_nodes) & set(self.mortar_nodes)
        if shared_nodes:
            labels = sorted(node.label for node in shared_nodes)
            raise ValueError(
                f"MortarContact '{name}': non-mortar surface '{non_mortar_surf_name}' and mortar "
                f"surface '{mortar_surf_name}' share {len(labels)} node(s): {labels}. "
                f"The two contact surfaces must be disjoint - coinciding facets yield an "
                f"identically vanishing weak gap and thus a singular system."
            )

        self._nodes = self.non_mortar_nodes + self.mortar_nodes
        self.nNonMortarNodes = len(self.non_mortar_nodes)
        self.nMortarNodes = len(self.mortar_nodes)

        self.nMultipliers = self.nNonMortarNodes
        self._nDof = self.sizeField * len(self._nodes) + self.nMultipliers

        self._fieldsOnNodes = [[self.field]] * len(self._nodes)
        self.active = True

        # Semi-smooth-Newton complementarity parameter of the normal contact NCP
        # (Gitterle et al. 2010 Eq. 55; Hueber & Wohlmuth 2005). Purely algorithmic,
        # but it carries a UNIT, so a fixed numeric default is only ever right for one
        # system of units. A non-positive value therefore means "derive it", which is
        # done lazily at the first assembly: sections are assigned to elements in
        # model.prepareYourself(), i.e. after the constraints are constructed, so the
        # materials are not reachable yet at this point.
        self.c_n = float(kwargs["cn"])
        self._derive_c_n = self.c_n <= 0.0

        # Node index lookups for fast access
        self.node_to_global_idx = {node: i for i, node in enumerate(self._nodes)}
        self.slave_node_to_idx = {node: i for i, node in enumerate(self.non_mortar_nodes)}
        self.master_node_to_idx = {node: i for i, node in enumerate(self.mortar_nodes)}

        # Undeformed coordinates of all constraint nodes (slaves first, then masters)
        self._X = np.array([node.coordinates for node in self._nodes])

        # Precompute undeformed normals
        self.undeformed_normals = self.compute_normals()

        # The orientation of the contact facets is an INPUT property: it decides
        # the direction of n_I and with it the sign of gap and pressure. Nothing
        # downstream can recover from getting it wrong - a slave normal pointing
        # into its own body turns the contact into an adhesive bond that transmits
        # tension. Two independent checks, both diagnostic rather than fatal,
        # because a legitimate mesh may be split, non-manifold at its border, or
        # deliberately one-sided.
        self._check_facet_winding(self.non_mortar_facets, non_mortar_surf_name)
        self._check_facet_winding(self.mortar_facets, mortar_surf_name)
        self._check_surfaces_face_each_other()

        # Initialize PDASS variables
        self.active_set = np.zeros(self.nNonMortarNodes, dtype=bool)
        self.use_active_set = True
        # Identifies the increment ATTEMPT (number, size, end time) - see
        # applyConstraint: a cutback re-attempt keeps the number but changes size.
        self.last_timestep_key = None
        self.current_iteration = 0
        # Termination + anti-cycling state of the semi-smooth (PDASS) iteration,
        # reset per increment in applyConstraint: the set is frozen for the rest
        # of the increment once it has settled (Hueber & Wohlmuth 2005) or once a
        # discrete state already visited this increment recurs (Bland 1977).
        self.active_set_frozen = False
        self.active_set_stable_count = 0
        self._seen_states = set()
        self._last_state = None

        # NOTE: self._warned is initialized at the very top of __init__, because the
        # input checks above already emit diagnostics through it. Every assumption
        # behind those diagnostics was measured over the Control_Tests and
        # patch-test suite before being wired up; the counts are recorded in the
        # documentation. They are emitted through `warnings` rather than the
        # journal because the constraint interface does not hand a Journal
        # instance to constraints.

    def _warn_once(self, key: str, message: str):
        """Emit a runtime diagnostic at most once per constraint instance and cause.

        These conditions repeat every increment once they occur at all, so warning
        per occurrence would bury the message in its own repetitions. One message
        per run and cause is what makes it readable.
        """
        if key in self._warned:
            return
        self._warned.add(key)
        warnings.warn(f"MortarContact '{self._name}': {message}", RuntimeWarning, stacklevel=3)

    def _check_facet_winding(self, facets, surface_name: str):
        """Whether the facets of one surface are wound consistently.

        Purely topological, no geometry involved. Two facets sharing an edge must
        traverse it in OPPOSITE directions; if they traverse it in the same
        direction, one of them is flipped and its normal points the other way.

        3D  the directed corner edges of a facet. A directed edge seen twice means
            two facets walk it the same way.
        2D  a line facet is the directed segment node0 -> node1. An interior node
            is the end of one facet and the start of the next, so a node appearing
            twice as a start (or twice as an end) marks a flipped facet.

        This catches a locally inconsistent surface. It cannot catch a surface that
        is consistently wound but globally inside-out - that is what
        _check_surfaces_face_each_other is for.
        """
        dim = self.model.domainSize
        culprits = []

        if dim == 3:
            seen = {}
            for el, _ in facets:
                nodes = el.nodes
                nCorner = 3 if len(nodes) in (3, 6) else 4
                corners = nodes[:nCorner]
                for i in range(nCorner):
                    edge = (corners[i].label, corners[(i + 1) % nCorner].label)
                    if edge in seen:
                        culprits.append((seen[edge], el.elNumber))
                    seen[edge] = el.elNumber
        else:
            starts, ends = {}, {}
            for el, _ in facets:
                a, b = el.nodes[0].label, el.nodes[1].label
                if a in starts:
                    culprits.append((starts[a], el.elNumber))
                if b in ends:
                    culprits.append((ends[b], el.elNumber))
                starts[a] = el.elNumber
                ends[b] = el.elNumber

        if culprits:
            pairs = ", ".join(f"({a}, {b})" for a, b in culprits[:5])
            self._warn_once(
                f"winding_{surface_name}",
                f"surface '{surface_name}' is not consistently wound: {len(culprits)} facet "
                f"pair(s) traverse a shared edge in the SAME direction, first {pairs}. One facet "
                f"of each pair has its normal reversed, so the weighted gap and the contact "
                f"pressure change sign there and the contact transmits tension instead of "
                f"compression. Fix the node ordering of the contact overlay elements.",
            )

    def _check_surfaces_face_each_other(self):
        """Whether the slave normals point towards the master surface at all.

        A surface can be wound perfectly consistently and still be inside-out as a
        whole; the winding check cannot see that, but the relative position of the
        two surfaces can. The contact normal must point away from the slave body,
        i.e. roughly towards the master.

        Deliberately a weak test with a wide margin: it is meant to catch a
        surface that is flipped outright, not to police curved or partially
        overlapping interfaces.
        """
        if not self.nNonMortarNodes or not self.nMortarNodes:
            return

        slave_centroid = np.mean(self._X[: self.nNonMortarNodes], axis=0)
        master_centroid = np.mean(self._X[self.nNonMortarNodes :], axis=0)
        towards_master = master_centroid - slave_centroid
        separation = np.linalg.norm(towards_master)
        if separation < 1e-14:
            return  # coincident centroids: no information, and not our problem

        mean_normal = np.mean(self.undeformed_normals, axis=0)
        if np.linalg.norm(mean_normal) < 1e-8:
            return  # normals cancel out (e.g. a closed surface) - nothing to say

        cos = float(np.dot(mean_normal, towards_master) / (np.linalg.norm(mean_normal) * separation))
        if cos < 0.0:
            self._warn_once(
                "surfaces_face_away",
                f"the averaged slave normal points AWAY from the master surface "
                f"(cos = {cos:.3f}). The contact normal must point out of the slave body, so "
                f"this indicates that the non-mortar facets are oriented inwards - or that "
                f"non-mortar and mortar surface are swapped. Gap and pressure then carry the "
                f"wrong sign and the contact bonds the surfaces instead of separating them.",
            )

    def _resolve_c_n(self):
        """Set c_n to the smallest initial Young's modulus adjacent to the interface.

        Why the initial one: Hueber & Wohlmuth (2005), sec. 7, find a lower bound c0
        for convergence of the active set that "depends linearly on E", with
        negligible influence above it, while Gitterle et al. (2010), Table I, show
        that far above it the active set starts to chatter. E therefore fixes the
        order of magnitude, not the value. Taking E at the start of the computation is
        well defined even for a damaging material (GCDP), because no damage has
        accumulated yet - and since c_n is purely algorithmic, its later evolution
        does not matter.

        Why only the ADJACENT materials: the global minimum over the model would be
        the wrong direction. A soft material somewhere far from the interface would
        push c_n BELOW the c0 of the contacting pair, which is exactly the regime
        Hueber & Wohlmuth report as non-convergent.

        Marmot materials carry their parameters as a flat array whose first entry is
        the Young's modulus (LINEARELASTIC, GCDP, ...). That is a convention, not a
        guarantee, which is why the resolved value is always reported.
        """
        contact_nodes = set(self._nodes)
        candidates = {}
        for name, section in self.model.sections.items():
            material = getattr(section, "material", None)
            if material is None:
                continue
            touches = False
            for elSet in getattr(section, "elSets", []):
                for el in elSet:
                    if contact_nodes.intersection(el.nodes):
                        touches = True
                        break
                if touches:
                    break
            if not touches:
                continue
            if isinstance(material, dict):
                # Marmot: flat parameter array, first entry is E by convention
                props = material.get("properties")
                E = float(props[0]) if props is not None and len(props) else None
                mat_name = material.get("name", "?")
            else:
                # EdelweissFE-native material classes keep it as _E (see e.g.
                # materials/linearelastic); materialProperties is the fallback.
                mat_name = type(material).__name__
                E = getattr(material, "_E", None)
                if E is None:
                    props = getattr(material, "materialProperties", None)
                    E = float(props[0]) if props is not None and len(props) else None
                E = float(E) if E is not None else None
            if E is not None and E > 0.0:
                candidates[f"{name}/{mat_name}"] = E

        if not candidates:
            self.c_n = 1.0e6
            self._warn_once(
                "cn_not_derivable",
                "no Young's modulus could be determined for the materials adjacent to the "
                f"contact surfaces, so c_n falls back to {self.c_n:.3e}. That value carries a "
                "unit and is only meaningful for a model in MPa. Set 'cn' explicitly - it must "
                "be at the order of the Young's modulus of the softer contacting body, and "
                "larger is the safe direction (Hueber & Wohlmuth 2005, Sec. 7).",
            )
            return

        chosen = min(candidates, key=candidates.get)
        self.c_n = candidates[chosen]
        listed = ", ".join(f"{k} = {v:.4g}" for k, v in sorted(candidates.items(), key=lambda kv: kv[1]))
        self._warn_once(
            "cn_derived",
            f"c_n was not given and has been derived as {self.c_n:.4g}, the smallest initial "
            f"Young's modulus adjacent to the interface ({chosen}). Considered: {listed}. This "
            f"assumes the first material constant is the Young's modulus, which is the Marmot "
            f"convention but not guaranteed - check it, or set 'cn' explicitly. Larger is the "
            f"safe direction (Hueber & Wohlmuth 2005, Sec. 7).",
        )

    def _check_converged_active_set(self, U_ref: np.ndarray):
        """Re-evaluate the NCP indicator on the CONVERGED state of the last increment.

        The active set is frozen once it has settled for two consecutive iterations
        (or on a detected cycle, or at the iteration cap). The Newton iteration then
        continues on that fixed set, so the state it finally converges to was
        decided at an intermediate iterate. Usually that is exactly right - the set
        settled because it had converged. It is not guaranteed, though, and nothing
        in the increment itself notices.

        This is the missing check, run at the start of the next increment when the
        previous one has converged: with the geometry that was in force and the
        displacements that came out, would the indicator still produce the same set?
        If not, the increment enforced the wrong branch at those nodes and its
        solution does not satisfy the Signorini conditions.
        """
        if not self.use_active_set:
            return

        dim = self.model.domainSize
        sf = self.sizeField
        nNodes = len(self._nodes)
        nSlave = self.nNonMortarNodes

        disp = U_ref[: sf * nNodes].reshape(nNodes, sf)[:, :dim]
        coords = self._X + disp
        x_slave, x_master = coords[:nSlave], coords[nSlave:]
        idx_LM_0 = sf * nNodes

        would_be = np.zeros(nSlave, dtype=bool)
        for I in range(nSlave):
            n_I = self.current_normals[I]
            nzD, nzC = self.current_D_nz[I], self.current_C_nz[I]
            g_weak = 0.0
            if len(nzD):
                g_weak -= self.current_D_row[I] @ (x_slave[nzD] @ n_I)
            if len(nzC):
                g_weak += self.current_C_row[I] @ (x_master[nzC] @ n_I)
            D_II = self.current_D_rowsum[I]
            inv_D = 1.0 / D_II if abs(D_II) > self.current_D_tol else 0.0
            p_n = U_ref[idx_LM_0 + I] * np.sign(D_II)
            would_be[I] = bool(p_n - self.c_n * g_weak * inv_D > 0.0)

        flipped = np.flatnonzero(would_be != self.active_set)
        if len(flipped):
            self._warn_once(
                "active_set_unstable_at_convergence",
                f"the active set of a converged increment does not reproduce itself: "
                f"{len(flipped)} slave node(s) (first local index {flipped[0]}) would switch "
                f"branch if the indicator were re-evaluated on the converged state. That "
                f"increment enforced g_weak = 0 on a node that wants to open, or lambda = 0 on "
                f"one that wants to close, so its solution does not satisfy the Signorini "
                f"conditions. Reduce the increment size, or verify the result (see "
                f"testfiles/mortar_tests/10_signorini_check).",
            )

    @property
    def nodes(self) -> list:
        return self._nodes

    @property
    def fieldsOnNodes(self) -> list:
        return self._fieldsOnNodes

    @property
    def nDof(self) -> int:
        return self._nDof

    def getNumberOfAdditionalNeededScalarVariables(self) -> int:
        return self.nMultipliers

    def compute_normals(self, U_np: np.ndarray = None) -> np.ndarray:
        """Averaged outward-pointing unit normal at every non-mortar (slave) node.

        The averaged nodal normal of Popp, Gitterle, Gee & Wall (2010) and Farah
        (2018), Sec. 4.1: the element normal x,xi cross x,eta is evaluated at the
        node's OWN natural coordinate in each adjacent facet, the contributions are
        summed and the result is normalized. Summing before normalizing is what
        weights a facet by its local surface Jacobian, and normalizing at the end is
        what makes the field continuous across facet boundaries.

        On a FLAT slave surface this is exactly the facet-constant normal it
        replaced - all contributions are parallel there, so any positive weighting
        normalizes to the same direction, whatever the facet sizes. The two differ
        only where the slave surface is curved, and then only for quadratic facet
        types, whose mid-side nodes a facet-constant normal cannot distinguish from
        their corners.

        If U_np is provided, the normal vectors are calculated in the deformed
        configuration. Otherwise, they are calculated in the undeformed one.
        """
        dim = self.model.domainSize
        normals = np.zeros((self.nNonMortarNodes, dim))

        node_to_idx = self.slave_node_to_idx

        # Iterate over all non-mortar facets
        for el, faceID in self.non_mortar_facets:
            facet_nodes = el.nodes

            # Retrieve coordinates
            coords = []
            for node in facet_nodes:
                X = node.coordinates
                if U_np is not None:
                    # Retrieve displacement from local solution slice
                    node_idx = self.node_to_global_idx[node]
                    u = U_np[self.sizeField * node_idx : self.sizeField * node_idx + dim]
                    coords.append(X + u)
                else:
                    coords.append(X)
            
            coords = np.array(coords)

            # Element normal evaluated AT each node's own natural coordinate, summed
            # over the adjacent facets and normalized below - the averaged nodal
            # normal of Popp, Gitterle, Gee & Wall (2010) and Farah (2018), Sec. 4.1.
            #
            # The point is that it is evaluated per NODE, not per facet. A single
            # facet-constant normal (the diagonal cross product of the corner nodes in
            # 3D, the corner chord in 2D) assigns the same direction to a corner and
            # to a mid-side node of the SAME quadratic facet, so the curvature of that
            # facet never reaches the normal and the field converges only linearly:
            # measured on the cylinder patch of 01_node_normals, CONQUAD8 and CONQUAD9
            # produced exactly the CONQUAD4 error (7.16 deg -> 3.58 deg, rate 1), i.e.
            # the quadratic facet types bought nothing at all.
            #
            # The magnitude of the un-normalized contribution is the local surface
            # Jacobian, so a large facet still outweighs a small one at a shared node;
            # for an undistorted linear facet it is the facet area up to the constant
            # factor that the normalization removes.
            dN_at_nodes = el.getShapeFunctionDerivativesAtNodes()
            for local, node in enumerate(facet_nodes):
                if node not in node_to_idx:
                    continue
                t = dN_at_nodes[local] @ coords
                if dim == 3:
                    n_node = np.cross(t[0], t[1])
                else:
                    n_node = np.array([t[0][1], -t[0][0]])
                normals[node_to_idx[node]] += n_node

        # Normalize the normal vectors
        degenerate = []
        for i in range(self.nNonMortarNodes):
            norm = np.linalg.norm(normals[i])
            if norm > 1e-14:
                normals[i] /= norm
            else:
                normals[i] = np.zeros(dim)
                degenerate.append(i)

        if degenerate:
            # A zero normal is not a harmless special case: g_weak and p_n are both
            # built by contracting with n_I, so both collapse to zero, the
            # indicator s_n is zero, and the node can NEVER become active. It
            # silently drops out of the contact. Causes are a degenerate facet
            # (zero area) or facets whose area-weighted normals cancel, which is
            # what a node on a sharp fold between two opposing facets does.
            self._warn_once(
                "degenerate_nodal_normal",
                f"{len(degenerate)} slave node(s) with a vanishing area-weighted normal "
                f"(first: local index {degenerate[0]}). Such nodes carry no contact at all - "
                f"the weighted gap and the pressure both contract with n_I and vanish with it, "
                f"so the active-set indicator can never turn them on. Usual causes: a "
                f"degenerate (zero-area) contact facet, or adjacent facets whose normals "
                f"cancel because they fold back onto each other.",
            )

        return normals

    def compute_local_dual_matrices(self, U_np: np.ndarray = None) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Compute the local standard mass matrices M_e, diagonal matrices D_e, and transformation matrices A_e for all non-mortar facets.
        
        If U_np is provided, coordinates are evaluated in the deformed configuration.
        Otherwise, they are evaluated in the undeformed configuration.
        """
        dim = self.model.domainSize
        dual_matrices = {}
        for el, faceID in self.non_mortar_facets:
            # Extract coordinates for element nodes
            coords = []
            for node in el.nodes:
                X = node.coordinates
                if U_np is not None:
                    node_idx = self.node_to_global_idx[node]
                    u = U_np[self.sizeField * node_idx : self.sizeField * node_idx + dim]
                    coords.append(X + u)
                else:
                    coords.append(X)
            coords = np.array(coords)
            
            # Compute M_e, D_e, A_e
            M_e, D_e, A_e = el.computeLocalMassMatrices(coords)
            dual_matrices[el.elNumber] = (M_e, D_e, A_e)
            
        return dual_matrices

    def compute_mortar_coupling_matrices(self, U_np: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        """Compute the global mortar coupling matrices D (slave-slave) and C (slave-master).
        Supports both 2D and 3D contact elements.
        """
        dim = self.model.domainSize
        if dim not in (2, 3):
            raise NotImplementedError("Mortar coupling matrix integration is only implemented for 2D and 3D.")

        n_slave = self.nNonMortarNodes
        n_master = self.nMortarNodes

        # D and C are assembled sparsely from the element-local blocks. The sparsity
        # is a property of the dual formulation, not an implementation detail: Popp,
        # Wohlmuth, Gee & Wall (2012), sec. 4.2, note that dual Lagrange multipliers
        # yield slave-side nodal basis functions "which have only local support",
        # algebraically visible as D becoming diagonal; sec. 5 adds that the
        # biorthogonality reduces D^-1 "either to a diagonal matrix ... or to at
        # least a sparse matrix". Dense storage would therefore cost O(n_slave^2) for
        # a structurally sparse object.
        D_rows, D_cols, D_vals = [], [], []
        C_rows, C_cols, C_vals = [], [], []

        from edelweissfe.constraints.mortar_geom_utils import (
            clip_1d_segments,
            get_tangent_basis,
            sutherland_hodgman_clip,
            to_plane_coords,
            triangulate_polygon,
        )

        # Precompute current deformed coordinates for all nodes
        current_coords = {}
        for node in self._nodes:
            X = node.coordinates
            if U_np is not None:
                node_idx = self.node_to_global_idx[node]
                u = U_np[self.sizeField * node_idx : self.sizeField * node_idx + dim]
                current_coords[node] = X + u
            else:
                current_coords[node] = X

        # Build AABB bounding boxes for all master facets to construct the BVH tree
        facets_with_bounds = []
        for m_el, m_faceID in self.mortar_facets:
            m_coords = np.array([current_coords[n] for n in m_el.nodes])
            centroid = np.mean(m_coords, axis=0)

            aabb_min = np.min(m_coords, axis=0)
            aabb_max = np.max(m_coords, axis=0)
            margin = max(0.15 * np.max(aabb_max - aabb_min), 0.1)
            aabb_min -= margin
            aabb_max += margin

            facets_with_bounds.append(((m_el, m_faceID), centroid, aabb_min, aabb_max))

        bvh_root = build_bvh(facets_with_bounds)

        # ------------------------------------------------------------------
        # PASS 1: Segmentation - collect all integration point records per
        # slave facet: (N_s, master elNumber, N_m, dGamma)
        # ------------------------------------------------------------------
        seg_records = {}  # slave elNumber -> list of records
        seg_masters = {}  # slave elNumber -> {master elNumber: m_el}
        slave_els = {}  # slave elNumber -> (s_el, s_idx)

        # Gauss point back-mapping failures (see map_2d_to_natural): a non-converged
        # natural coordinate is indistinguishable from a converged one and goes
        # straight into N_s / N_m and thus into D and C, so it has to be counted here
        # - nothing downstream can notice it.
        n_proj_failed = 0
        first_proj_failure = None

        for s_el, s_faceID in self.non_mortar_facets:
            s_nodes = s_el.nodes

            s_coords = np.array([current_coords[nd] for nd in s_nodes])
            s_idx = np.array([self.slave_node_to_idx[nd] for nd in s_nodes])

            # Query the BVH tree once per slave facet to find nearby Master candidates
            s_aabb_min = np.min(s_coords, axis=0)
            s_aabb_max = np.max(s_coords, axis=0)
            s_margin = max(0.15 * np.max(s_aabb_max - s_aabb_min), 0.1)

            candidates = []
            query_bvh(bvh_root, s_aabb_min - s_margin, s_aabb_max + s_margin, candidates)
            if not candidates:
                continue

            # Loop over the linear sub-cells of the slave facet
            for s_sub in get_sub_cells(s_el):
                sc_coords = s_coords[s_sub]

                if dim == 3:
                    # 3D surface Mortar integration (Auxiliary plane projection & Sutherland-Hodgman clipping)
                    n_vec = facet_normal(sc_coords)
                    n_norm = np.linalg.norm(n_vec)
                    if n_norm < 1e-14:
                        continue  # degenerate sub-cell
                    normal = n_vec / n_norm
                    p0 = np.mean(sc_coords, axis=0)
                    t1, t2 = get_tangent_basis(normal)

                    s_sub_2d = to_plane_coords(sc_coords, p0, t1, t2)
                    s_full_2d = to_plane_coords(s_coords, p0, t1, t2)

                    # The slave sub-cell is the CLIP polygon, so it must be convex
                    # (see is_convex_polygon). For CONQUAD8 this requires the
                    # mid-side node to stay on its own side of the element centre,
                    # for CONQUAD9 the centre node to stay clear of the corners -
                    # margins no usable volume element gets anywhere near, which is
                    # why this is a warning and not a raise.
                    if not is_convex_polygon(s_sub_2d):
                        self._warn_once(
                            "nonconvex_subcell",
                            f"non-convex sub-cell on slave facet {s_el.elNumber} (element type "
                            f"{s_el.elType}). The segmentation clips against the half-plane of "
                            f"each sub-cell edge, so contact area is lost without further notice. "
                            f"Check the mid-side/centre node positions of the quadratic contact "
                            f"facets; the element is severely distorted.",
                        )

                    for m_el, m_faceID in candidates:
                        m_nodes = m_el.nodes
                        m_coords = np.array([current_coords[nd] for nd in m_nodes])
                        m_full_2d = to_plane_coords(m_coords, p0, t1, t2)

                        for m_sub in get_sub_cells(m_el):
                            m_sub_2d = m_full_2d[m_sub]

                            # The master sub-cell is the SUBJECT of the clip, which
                            # Sutherland-Hodgman accepts non-convex - but the overlap
                            # inherits its reflex vertices and the fan triangulation
                            # below then over-counts the area (see is_convex_polygon).
                            if not is_convex_polygon(m_sub_2d):
                                self._warn_once(
                                    "nonconvex_master_subcell",
                                    f"non-convex sub-cell on master facet {m_el.elNumber} (element "
                                    f"type {m_el.elType}), projected into the plane of slave facet "
                                    f"{s_el.elNumber}. The overlap polygon inherits the reflex "
                                    f"vertex and the fan triangulation then integrates over area "
                                    f"outside the facet. Check the mid-side/centre node positions "
                                    f"of the quadratic contact facets; the element is severely "
                                    f"distorted.",
                                )

                            overlap_2d = sutherland_hodgman_clip(m_sub_2d, s_sub_2d)
                            if len(overlap_2d) < 3:
                                continue

                            for tri in triangulate_polygon(overlap_2d):
                                v0, v1, v2 = tri[0], tri[1], tri[2]
                                area_jac = abs(
                                    (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1])
                                )
                                if area_jac < 1e-14:
                                    continue

                                for gp, w in zip(TRI_GAUSS_PTS, TRI_GAUSS_W):
                                    L1, L2 = gp[0], gp[1]
                                    x_gp_2d = (1.0 - L1 - L2) * v0 + L1 * v1 + L2 * v2

                                    local_s, ok_s = map_2d_to_natural(s_el, s_full_2d, x_gp_2d)
                                    local_m, ok_m = map_2d_to_natural(m_el, m_full_2d, x_gp_2d)
                                    if not ok_s:
                                        n_proj_failed += 1
                                        if first_proj_failure is None:
                                            first_proj_failure = ("slave", s_el.elNumber, s_el.elType)
                                    if not ok_m:
                                        n_proj_failed += 1
                                        if first_proj_failure is None:
                                            first_proj_failure = ("master", m_el.elNumber, m_el.elType)

                                    N_s = s_el.getShapeFunctions(local_s)
                                    N_m = m_el.getShapeFunctions(local_m)

                                    s_num = s_el.elNumber
                                    if s_num not in seg_records:
                                        seg_records[s_num] = []
                                        seg_masters[s_num] = {}
                                        slave_els[s_num] = (s_el, s_idx)
                                    seg_records[s_num].append((N_s, m_el.elNumber, N_m, area_jac * w))
                                    seg_masters[s_num][m_el.elNumber] = m_el

                else:
                    # 2D line segment Mortar integration
                    for m_el, m_faceID in candidates:
                        m_nodes = m_el.nodes
                        m_coords = np.array([current_coords[nd] for nd in m_nodes])
                        for m_sub in get_sub_cells(m_el):
                            mc_coords = m_coords[m_sub]
                            s_start, s_end, t_vec = clip_1d_segments(sc_coords, mc_coords)
                            if s_end - s_start < 1e-12:
                                continue

                            half_len = 0.5 * (s_end - s_start)
                            mid_s = 0.5 * (s_start + s_end)
                            for gp_1d, w_1d in zip(LINE_GAUSS_PTS, LINE_GAUSS_W):
                                s_gp = mid_s + half_len * gp_1d[0]
                                x_gp = sc_coords[0] + s_gp * t_vec
                                dG = half_len * w_1d

                                local_s, ok_s = map_2d_to_natural(s_el, s_coords, x_gp)
                                local_m, ok_m = map_2d_to_natural(m_el, m_coords, x_gp)
                                if not ok_s:
                                    n_proj_failed += 1
                                    if first_proj_failure is None:
                                        first_proj_failure = ("slave", s_el.elNumber, s_el.elType)
                                if not ok_m:
                                    n_proj_failed += 1
                                    if first_proj_failure is None:
                                        first_proj_failure = ("master", m_el.elNumber, m_el.elType)

                                N_s = s_el.getShapeFunctions(local_s)
                                N_m = m_el.getShapeFunctions(local_m)

                                s_num = s_el.elNumber
                                if s_num not in seg_records:
                                    seg_records[s_num] = []
                                    seg_masters[s_num] = {}
                                    slave_els[s_num] = (s_el, s_idx)
                                seg_records[s_num].append((N_s, m_el.elNumber, N_m, dG))
                                seg_masters[s_num][m_el.elNumber] = m_el

        if n_proj_failed:
            side, el_num, el_type = first_proj_failure
            self._warn_once(
                "gp_projection_failed",
                f"the Gauss point back-mapping did not converge for {n_proj_failed} integration "
                f"point(s) (first on the {side} facet {el_num}, element type {el_type}). For those "
                f"points the last Newton iterate was used as the natural coordinate, so the shape "
                f"function values entering D and C are wrong by an unknown amount - a "
                f"non-converged coordinate is indistinguishable from a converged one. The closest "
                f"point projection is not solvable for arbitrarily distorted or strongly curved "
                f"facets (Konyukhov & Schweizerhof 2008); check the facets around the one named "
                f"above.",
            )

        # ------------------------------------------------------------------
        # PASS 2: Dual coefficients from the actual segment quadrature
        # (MOOSE-style, consistent boundary treatment) and assembly
        # ------------------------------------------------------------------
        for s_num, records in seg_records.items():
            s_el, s_idx = slave_els[s_num]
            n_s = len(s_idx)

            # Biorthogonality system on the true integration domain:
            # M_t[a,b] = int_seg(N_tilde_a * N_tilde_b), D_t[a] = int_seg(N_tilde_a)
            T_e = s_el.getBasisTransformation()
            M_t = np.zeros((n_s, n_s))
            D_t = np.zeros(n_s)
            for N_s, _, _, dG in records:
                N_tilde = T_e @ N_s
                M_t += np.outer(N_tilde, N_tilde) * dG
                D_t += N_tilde * dG

            # For sliver overlaps M_t becomes (near-)singular; fall back to the
            # reference-element coefficients in that case.
            cond_M_t = np.linalg.cond(M_t)
            if cond_M_t < 1e12:
                A_e = np.diag(D_t) @ np.linalg.inv(M_t) @ T_e
            else:
                # Measured over the Control_Tests and patch-test suite: 4 of 3989
                # slave-element evaluations take this path, all of them in the
                # curved Hertz models with CONQUAD8. It is not a dead branch, and
                # it has a consequence worth naming.
                #
                # D_II = int_overlap(N_tilde) would be non-negative for ANY
                # sub-region if N_tilde were pointwise non-negative. It is not,
                # for CONQUAD8 at alpha = 1/3: the transformed corner function
                # dips to -1/324 over exactly 12.5 % of the reference square (the
                # negative region is the triangle (1-xi) + (1-eta) < 1, of area 1/2
                # out of 4), and
                # pointwise non-negativity would need alpha >= 3/8 (CONLINE3 gets
                # it at alpha = 1/3 because its threshold is 1/4). So a partial
                # overlap can produce D_II < 0 on the CONSISTENT path already -
                # see test_partial_coverage_corner_can_turn_weight_negative in
                # testfiles/mortar_tests/05_quadratic_segmentation.
                #
                # This branch is worse by orders of magnitude: the reference-
                # element dual functions are used instead, and those change sign
                # over half the element, so the integral over a sub-region takes
                # any sign AND any magnitude. Measured: below ~0.1 % coverage
                # D_II flips sign, at -1.0 of the largest weight.
                self._warn_once(
                    "sliver_fallback",
                    f"degenerate overlap on slave facet {s_num}: cond(M_t) = {cond_M_t:.2e} >= 1e12, "
                    f"falling back to reference-element dual coefficients. Biorthogonality then "
                    f"holds on the full element only, and the nodal weights D_II can take any sign "
                    f"and magnitude there - for CONQUAD8 as well.",
                )
                # Evaluated only HERE, not for every slave facet up front: the
                # reference-element coefficients cost an element mass matrix plus its
                # inversion, and this branch is taken for 4 of 3989 slave-element
                # evaluations over the test suite. Same quantity as
                # compute_local_dual_matrices returns, for this one element and the
                # same (deformed) coordinates.
                A_e = s_el.computeLocalMassMatrices(
                    np.array([current_coords[nd] for nd in s_el.nodes])
                )[2]

            # Assemble the D block and per-master C blocks of this slave element
            D_blk = np.zeros((n_s, n_s))
            C_blks = {}
            for N_s, m_num, N_m, dG in records:
                M_bar = A_e @ N_s
                D_blk += np.outer(M_bar, N_s) * dG
                if m_num not in C_blks:
                    C_blks[m_num] = np.zeros((n_s, len(seg_masters[s_num][m_num].nodes)))
                C_blks[m_num] += np.outer(M_bar, N_m) * dG

            rr, cc = np.meshgrid(s_idx, s_idx, indexing="ij")
            D_rows.append(rr.ravel())
            D_cols.append(cc.ravel())
            D_vals.append(D_blk.ravel())
            for m_num, C_blk in C_blks.items():
                m_idx = np.array([self.master_node_to_idx[nd] for nd in seg_masters[s_num][m_num].nodes])
                rr, cc = np.meshgrid(s_idx, m_idx, indexing="ij")
                C_rows.append(rr.ravel())
                C_cols.append(cc.ravel())
                C_vals.append(C_blk.ravel())

        # coo -> csr sums duplicate entries, which is exactly what the dense "+="
        # accumulation did. The summation ORDER differs, so the result agrees with the
        # dense one to round-off rather than bit for bit.
        D = _coo_to_csr(D_vals, D_rows, D_cols, (n_slave, n_slave))
        C = _coo_to_csr(C_vals, C_rows, C_cols, (n_slave, n_master))

        return D, C

    def applyConstraint(
        self,
        U_np: np.ndarray,
        dU: np.ndarray,
        PExt: np.ndarray,
        K: np.ndarray,
        timeStep: TimeStep,
    ):
        if not self.active:
            return

        if self._derive_c_n:
            self._derive_c_n = False
            self._resolve_c_n()

        dim = self.model.domainSize
        sf = self.sizeField
        nNodes = len(self._nodes)
        nSlave = self.nNonMortarNodes

        # Detect new increment to track iterations and reset current_iteration.
        # Normals and coupling matrices are frozen within each increment
        # (staggered geometry update), so the assembled stiffness is the exact
        # Jacobian of the residual equations within the increment.
        # A RE-ATTEMPT of an increment after a solver cutback carries the SAME
        # increment number but a smaller time increment, and it restarts the Newton
        # iteration from the last converged state. It must therefore be treated
        # exactly like a new increment: the staggered geometry (normals, D, C) has
        # to be re-evaluated at the state the attempt starts from - otherwise the
        # frozen data would stem from the diverged iterate of the failed attempt -
        # and the semi-smooth active-set iteration has to restart as well.
        # Keying the reset on the increment number alone would miss this.
        #
        # REFERENCE CONFIGURATION of the frozen geometry: the LAST CONVERGED state,
        # U_n = U_np - dU, not the state handed in as U_np.
        #
        # The reference formulations do not freeze at all: Popp et al. (2009/2010),
        # Gitterle et al. (2010) and Farah (2018) re-evaluate D, M and the nodal
        # normals in every Newton iteration and carry their linearizations in K;
        # MOOSE regenerates the mortar segmentation on the displaced mesh and
        # differentiates it by AD. That is the target state (see the documentation,
        # section "Ausblick / Konsistente Linearisierung"), not what is done here.
        #
        # Within a staggered scheme, however, the reference must be the converged
        # state. EdelweissFE's solvers extrapolate the previous increment before the
        # first assembly (`extrapolation`, DEFAULT "linear": U_np = U_n + dU_extrap
        # already at iteration 0). Freezing at that predictor would make the
        # CONVERGED contact solution depend on a solver switch that must not
        # influence it - and it would be inconsistent within itself, since a cutback
        # re-attempt resets dU to zero and would then use a different kind of
        # reference than a regular increment. It would also void the error statement
        # of the staggered scheme, which is first order in the increment size only
        # with respect to an equilibrated reference configuration.
        step_key = (timeStep.number, timeStep.timeIncrement, timeStep.totalTime)
        if step_key != self.last_timestep_key or not hasattr(self, "current_normals"):
            # The previous increment has converged and U_np - dU is its result.
            # Before anything is reset, ask whether its active set reproduces
            # itself on that result - the one moment where that is checkable.
            if hasattr(self, "current_normals") and self.active_set_frozen:
                self._check_converged_active_set(U_np - dU)

            self.last_timestep_key = step_key
            self.current_iteration = 0
            # The active set (Signorini) is re-evaluated in EVERY Newton iteration
            # and the outer semi-smooth loop is converged once the set no longer
            # changes - the convergence criterion of the primal-dual active set
            # strategy (Hueber & Wohlmuth 2005) / semi-smooth Newton mortar contact
            # (Gitterle et al. 2010; Popp et al. 2012). There is deliberately NO
            # fixed iteration-count cutoff (that would be an ad-hoc heuristic and
            # could freeze a not-yet-settled set); a safeguard cap only guards
            # against pathological non-settling. Reset per increment:
            self.active_set_frozen = False
            self.active_set_stable_count = 0
            # Anti-cycling bookkeeping (Bland 1977): the discrete states already
            # visited this increment, and the immediately previous one.
            self._seen_states = set()
            self._last_state = None
            # Last converged state - see the reference-configuration note above.
            U_ref = U_np - dU
            self.current_normals = self.compute_normals(U_ref)
            D_full, C_full = self.compute_mortar_coupling_matrices(U_ref)
            # The FULL (element-locally sparse) D matrix is used for forces,
            # stiffness and weak gap. With the basis transformation T_e the
            # biorthogonality holds w.r.t. N_tilde, so D is not diagonal for
            # quadratic elements - lumping it would destroy the consistency of
            # the contact force distribution (a constant pressure could not be
            # transmitted exactly, i.e. the patch test would fail). This is
            # algebraically equivalent to the transformed formulation of
            # Popp et al. (2012) / Farah (2018), Eqs. (6.24)-(6.25).
            # Translational invariance of the weak gap is guaranteed by the
            # row-sum identity sum_K D_IK = sum_J C_IJ (same-domain integration).
            self.current_D = D_full
            self.current_C = C_full
            # Positive by construction: sum_K D_IK = int(Phi_I) = int(N_tilde_I) > 0
            self.current_D_rowsum = np.asarray(D_full.sum(axis=1)).ravel()
            # The sign-consistent contact measures below carry a negative D_II
            # correctly, but the weighted gap loses its reading as a mean opening
            # there - so it is worth saying out loud that it happened.
            # Only for weights that are meaningfully negative, though: a node far
            # outside the covered region integrates to a value that is zero up to
            # round-off, and whether that lands at +1e-19 or -1e-19 says nothing.
            # Measured range of the real cases: -0.031 (CONQUAD9 at 70 % coverage,
            # 06_active_set_pdass) down to -0.60 (sliver fallback,
            # hertz_hex20_medium), so a relative threshold of 1e-6 separates them
            # from the dust by orders of magnitude.
            weights = self.current_D_rowsum
            weight_scale = np.max(np.abs(weights)) if len(weights) else 0.0
            # Threshold below which a nodal weight counts as "no weight at all" and
            # the division 1/D_II is suppressed. RELATIVE, for the same reason as in
            # _nonzero_rows: D_II = int(Phi_I) carries the unit of an area, so an
            # absolute bound only ever fits one system of units. It also has to be
            # generous rather than tiny: at D_II = 1e-25 an absolute bound of 1e-30
            # would still divide, g_sep = g_weak/D_II would explode by 25 orders, and
            # the sign of that garbage would decide the branch of the NCP indicator -
            # a node can be switched ACTIVE by pure round-off that way, with a
            # constraint row that means nothing. Relative to the largest weight of the
            # interface, 1e-12 is far below any weight a covered node can have (the
            # smallest measured over the test suite is ~1e-3 of the largest) and far
            # above the dust of an uncovered one.
            self.current_D_tol = 1e-12 * weight_scale
            if weight_scale > 0.0:
                significant = np.flatnonzero(weights < -1e-6 * weight_scale)
                if len(significant):
                    worst = np.min(weights) / weight_scale
                    self._warn_once(
                        "negative_nodal_weight",
                        f"{len(significant)} slave node(s) with a negative nodal mortar weight "
                        f"D_II (worst: {worst:.3e} of the largest weight), first in increment "
                        f"{timeStep.number}. The active-set indicator handles the sign, but the "
                        f"weighted gap is no longer a mean opening at those nodes. Usual causes: a "
                        f"partially covered CONQUAD9, or a sliver overlap that triggered the "
                        f"reference-element fallback.",
                    )
            # Precompute the sparsity patterns AND the row values once per increment.
            # The geometry is frozen for the increment, so both are constant while
            # applyConstraint runs once per Newton iteration.
            self.current_D_nz, self.current_D_row = _nonzero_rows(D_full, nSlave)
            self.current_C_nz, self.current_C_row = _nonzero_rows(C_full, nSlave)
        else:
            self.current_iteration += 1

        normals = self.current_normals

        # Current coordinates of all constraint nodes in the deformed configuration
        disp = U_np[: sf * nNodes].reshape(nNodes, sf)[:, :dim]
        coords = self._X + disp
        x_slave = coords[:nSlave]
        x_master = coords[nSlave:]

        # ----------------------------------------------------------------------
        # SADDLE-POINT MODE: Lagrange multipliers lambda_I are explicit unknowns
        # in U_np, solved for jointly with the displacements. This is the only
        # currently supported/validated formulation; see note below on why a
        # condensed (multiplier-free) variant was removed.
        # ----------------------------------------------------------------------
        idx_LM_0 = sf * nNodes

        for I in range(nSlave):
            idx_LM_I = idx_LM_0 + I
            lambda_I = U_np[idx_LM_I]
            n_I = normals[I]
            nzD = self.current_D_nz[I]
            nzC = self.current_C_nz[I]
            dRow = self.current_D_row[I]
            cRow = self.current_C_row[I]

            g_I_weak = 0.0
            if len(nzD):
                g_I_weak -= dRow @ (x_slave[nzD] @ n_I)
            if len(nzC):
                g_I_weak += cRow @ (x_master[nzC] @ n_I)

            # Sign-consistent contact measures, valid for BOTH signs of the nodal
            # weight D_II = int(Phi_I). D_II is positive by construction for
            # CONQUAD4/8, CONTRI3/6 and CONLINE2/3, but NOT guaranteed for the
            # full-Lagrangian CONQUAD9 under partial coverage: its shape functions
            # are not pointwise non-negative, so the integral positivity required by
            # Popp et al. (2012), Eq. (4.2), can be violated there.
            #
            #   pressure  lambda_I is the multiplier in the LITERATURE convention
            #             (Popp, Gee & Wall 2009; Gitterle et al. 2010; Popp et al.
            #             2012; Farah 2018): lambda_n >= 0 in compression, i.e. the
            #             NEGATIVE slave traction. The nodal contact force along n_I
            #             is therefore -lambda_I * D_II, which points against n_I -
            #             into the slave body - for lambda_I > 0 and D_II > 0. With a
            #             negative weight the roles flip, so the physical pressure is
            #             p_n = lambda_I * sgn(D_II) >= 0 in compression for either
            #             sign of D_II.
            #   opening   Translating the master by a * n_I changes the weak gap by
            #             D_II * a (row-sum identity sum_K D_IK = sum_J C_IJ). The
            #             physical nodal opening is therefore g_weak / D_II. Dividing
            #             by the SIGNED D_II already restores exactly the property
            #             Popp et al. (2012), Sec. 4.3, demand - "a positive weighted
            #             gap if the physical gap is positive". Multiplying by
            #             sgn(D_II) on top of that flips the sign back and makes an
            #             open node look like a penetrating one; that is a bug this
            #             code carried until it was caught by the negative-weight
            #             regression test in 06_active_set_pdass. Note the asymmetry:
            #             the pressure carries sgn(D_II), the opening does not.
            D_II = self.current_D_rowsum[I]
            sgn_D = np.sign(D_II)
            inv_D = 1.0 / D_II if abs(D_II) > self.current_D_tol else 0.0
            p_n = lambda_I * sgn_D    # physical normal pressure (>= 0 in contact)
            g_sep = g_I_weak * inv_D  # physical opening (>0 open, <0 penetrating)

            if self.use_active_set and not self.active_set_frozen:
                # Semi-smooth normal complementarity (Gitterle et al. 2010 Eq. 55;
                # Hueber & Wohlmuth 2005; MOOSE ComputeWeightedGapLMMechanical
                # Contact): the Signorini KKT conditions p_n >= 0, g_sep >= 0,
                # p_n*g_sep = 0 are written as the single non-smooth function
                #   C_n = p_n - max(0, p_n - c_n*g_sep) = 0,
                # whose two branches are
                #   active   (s_n > 0):  constraint  g_weak = 0,
                #   inactive (s_n <= 0): constraint  lambda = 0,
                # with the augmented indicator s_n = p_n - c_n*g_sep. It is
                # re-evaluated EVERY Newton iteration - this is the literature-
                # standard PDASS = semi-smooth-Newton formulation with local
                # superlinear convergence. c_n is purely algorithmic: at
                # convergence g_sep -> 0, so the converged result is c_n-
                # independent and identical to any admissible active-set rule.
                #
                # Which gap measure enters the indicator differs WITHIN the
                # literature, and the form used here is the one of Hueber & Wohlmuth
                # (2005), Eq. (3.9), who write C = lambda_n - max{0, lambda_n +
                # c (u_n - g)} with the POINTWISE gap (u_n - g), i.e. a length.
                # g_sep = g_weak/D_II is exactly that: the physical nodal opening.
                # Popp et al. (2012), Eq. (5.1), Gitterle et al. (2010), Eq. (55),
                # and Farah (2018), Eq. (3.58), instead insert the mortar-weighted
                # gap g_weak, which carries length x area. Both are admissible - the
                # indicator only decides the branch, and at convergence either gap
                # vanishes - but only the length-valued form makes c_n*g_sep a
                # pressure comparable to p_n, and only for it is the recommendation
                # c_n ~ O(E) dimensionally meaningful.
                s_n = p_n - self.c_n * g_sep
                self.active_set[I] = bool(s_n > 0.0)

            # Assembly in the literature sign convention (lambda_n >= 0 in
            # compression). Throughout, K = -dPExt/dU, which is what makes the
            # multiplier block symmetric:
            #   slave force   -lambda_I * D_IK * n_I     -> K[x_s, lambda] = +D_IK*n
            #   master force  +lambda_I * C_IJ * n_I     -> K[x_m, lambda] = -C_IJ*n
            #   lambda row    +g_weak (active)           -> K[lambda, x_s] = +D_IK*n,
            #                                               K[lambda, x_m] = -C_IJ*n
            # The active lambda row therefore carries the OPPOSITE overall sign to
            # the one it had while lambda was the slave traction; that is a row
            # scaling by -1 and is exactly what keeps K[x, lambda] = K[lambda, x].
            # The inactive row (lambda = 0) has no coupling and is unaffected.
            if self.active_set[I]:
                PExt[idx_LM_I] += g_I_weak

                for K_nd, D_IK in zip(nzD, dRow):
                    s_dofs = slice(sf * K_nd, sf * K_nd + dim)
                    D_IK_n = D_IK * n_I
                    PExt[s_dofs] -= lambda_I * D_IK_n
                    K[s_dofs, idx_LM_I] += D_IK_n
                    K[idx_LM_I, s_dofs] += D_IK_n

                for J, C_IJ in zip(nzC, cRow):
                    m_global = nSlave + J
                    m_dofs = slice(sf * m_global, sf * m_global + dim)
                    C_IJ_n = C_IJ * n_I
                    PExt[m_dofs] += lambda_I * C_IJ_n
                    K[m_dofs, idx_LM_I] -= C_IJ_n
                    K[idx_LM_I, m_dofs] -= C_IJ_n
            else:
                PExt[idx_LM_I] -= lambda_I
                K[idx_LM_I, idx_LM_I] += 1.0

        # Termination of the semi-smooth active-set iteration. The discrete state
        # is the normal active set, updated every Newton iteration. Two
        # literature-grounded freeze triggers:
        #  (1) PDASS convergence (Hueber & Wohlmuth 2005): the state has settled,
        #      i.e. is unchanged for two consecutive iterations past a warm-up.
        #  (2) Anti-cycling (Bland 1977): the state CHANGED this iteration but
        #      revisits a state already seen earlier this increment -> a proven
        #      limit cycle (only finitely many states, so a revisit-after-change
        #      is a cycle). Freezing breaks it; the remaining Newton iterations
        #      are a linear solve on the fixed set.
        # A hard iteration cap remains as a final safeguard.
        if self.use_active_set and not self.active_set_frozen:
            state = self.active_set.tobytes()
            changed = state != self._last_state
            if changed:
                self.active_set_stable_count = 0
            else:
                self.active_set_stable_count += 1
            cycle = changed and (state in self._seen_states)
            self._seen_states.add(state)
            self._last_state = state
            settled = self.active_set_stable_count >= 2 and self.current_iteration >= 2
            hit_cap = self.current_iteration >= 20
            if settled or cycle or hit_cap:
                if hit_cap and not settled and not cycle:
                    # Measured over the Control_Tests and patch-test suite: this
                    # never fires - the set freezes 111 times out of 111 through the
                    # PDASS criterion, at most 11 iterations. Where it DOES fire the
                    # set was still moving, so the increment converges onto a set
                    # that was never confirmed and the converged solution may
                    # violate the Signorini conditions. Nothing checks that
                    # automatically; testfiles/mortar_tests/10_signorini_check does
                    # it for a converged model.
                    self._warn_once(
                        "active_set_iteration_cap",
                        f"the active set was frozen by the iteration cap (20), not by the PDASS "
                        f"convergence criterion, first in increment {timeStep.number}. The set had "
                        f"not settled, so the converged solution of such increments is not "
                        f"guaranteed to satisfy the Signorini conditions - verify it (see "
                        f"testfiles/mortar_tests/10_signorini_check).",
                    )
                self.active_set_frozen = True
