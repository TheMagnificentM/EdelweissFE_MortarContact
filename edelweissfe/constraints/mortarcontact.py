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

import numpy as np

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

The Signorini conditions (normal pressure p_I >= 0, weighted gap g_I >= 0,
complementarity p_I g_I = 0) are enforced through a single non-smooth
complementarity function (NCP)

    C_n,I = p_I - max(0, p_I - c_n D_II^-1 g_I) = 0
        <=>  active  iff  s_n,I = p_I - c_n D_II^-1 g_I > 0,

solved by a primal-dual active set strategy = semi-smooth Newton method
(Hueber & Wohlmuth 2005; Gitterle et al. 2010, Eq. (55); Farah 2018,
Sec. 3.5.2). The active set is re-evaluated in EVERY Newton iteration and the
outer (semi-smooth) loop is converged as soon as the set no longer changes;
c_n > 0 is purely algorithmic (g_I -> 0 at convergence, hence the converged
solution is c_n-independent) and is chosen at the order of Young's modulus of
the softer body.

Only the saddle-point formulation (explicit multiplier DOFs) is implemented.
A "dual condensation" variant (eliminating the multipliers via lambda_I =
-g_weak,I / D_II, evaluated fresh from the current gap each iteration) was
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
    "NCP. It enters the semi-smooth active-set indicator s_n = p_n - c_n*inv_D*g_sep "
    "(active iff s_n > 0; Gitterle et al. 2010 Eq. 55; Hueber & Wohlmuth 2005; MOOSE "
    "ComputeWeightedGapLMMechanicalContact). It is normalized by the nodal mortar "
    "weight D_II so that c_n*inv_D*g_sep is a pressure directly comparable to p_n. "
    "Purely algorithmic - no effect on the converged solution (g_sep -> 0 there), so "
    "the converged normal-contact result is identical to any admissible active-set "
    "rule. It MUST be chosen at the order of Young's modulus of the softer contacting "
    "body (Hueber & Wohlmuth 2005; Farah 2018 Sec. 3.5.2, c_n ~ O(E)): a too-large "
    "c_n makes the indicator gap-sign dominated, so near-boundary nodes flip "
    "active/inactive every iteration -> active-set chattering / non-convergence. Set "
    "it explicitly per problem; the default below is only a fallback.",
    float,
    1.0e6,
)

documentation = [module]


def map_2d_to_natural(el, coords_2d, point_2d, max_iter=10, tol=1e-12):
    """Map a 2D local plane coordinate point_2d to the element's natural space."""
    el_type = el.elType.upper()
    
    # 1D line elements
    if "LINE" in el_type:
        local_coords = np.zeros(1)
        for _ in range(max_iter):
            N = el.getShapeFunctions(local_coords)
            x_mapped = N @ coords_2d
            res = x_mapped - point_2d
            if np.linalg.norm(res) < tol:
                break
            dN = el.getShapeFunctionDerivatives(local_coords) # shape (1, n_nodes)
            J = dN @ coords_2d # shape (1, dim_of_coords_2d)
            J_norm = np.dot(J[0], J[0])
            if J_norm < 1e-14:
                break
            delta = np.dot(J[0], res) / J_norm
            local_coords[0] -= delta
        return local_coords

    # 2D surface elements (Quads and Triangles)
    local_coords = np.zeros(2)
    if "TRI" in el_type:
        local_coords = np.array([1.0 / 3.0, 1.0 / 3.0])
        
    for _ in range(max_iter):
        N = el.getShapeFunctions(local_coords)
        x_mapped = N @ coords_2d
        res = x_mapped - point_2d
        if np.linalg.norm(res) < tol:
            break
        dN = el.getShapeFunctionDerivatives(local_coords) # shape (2, n_nodes)
        J = (dN @ coords_2d).T # shape (2, 2)
        try:
            delta = np.linalg.solve(J, res)
        except np.linalg.LinAlgError:
            break
        local_coords -= delta
    return local_coords


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


def get_sub_cells(el) -> list[list[int]]:
    """Return the linear sub-cell decomposition (local node indices) of a contact facet."""
    el_type = el.elType.upper()
    if el_type not in SUB_CELL_MAP:
        raise NotImplementedError(
            f"No linear sub-cell decomposition defined for element type '{el_type}'."
        )
    return SUB_CELL_MAP[el_type]


def facet_normal(coords: np.ndarray) -> np.ndarray:
    """Unnormalized area-weighted normal of a flat linear facet (3 or 4 corner nodes)."""
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

        self._nodes = self.non_mortar_nodes + self.mortar_nodes
        self.nNonMortarNodes = len(self.non_mortar_nodes)
        self.nMortarNodes = len(self.mortar_nodes)

        self.nMultipliers = self.nNonMortarNodes
        self._nDof = self.sizeField * len(self._nodes) + self.nMultipliers

        self.recovered_lambdas = np.zeros(self.nNonMortarNodes)
        self._fieldsOnNodes = [[self.field]] * len(self._nodes)
        self.active = True

        # Semi-smooth-Newton complementarity parameter of the normal contact NCP
        # (Gitterle et al. 2010 Eq. 55; Hueber & Wohlmuth 2005). Purely
        # algorithmic; must be ~O(E) of the softer body (Farah 2018 Sec. 3.5.2).
        self.c_n = float(kwargs["cn"])

        # Node index lookups for fast access
        self.node_to_global_idx = {node: i for i, node in enumerate(self._nodes)}
        self.slave_node_to_idx = {node: i for i, node in enumerate(self.non_mortar_nodes)}
        self.master_node_to_idx = {node: i for i, node in enumerate(self.mortar_nodes)}

        # Undeformed coordinates of all constraint nodes (slaves first, then masters)
        self._X = np.array([node.coordinates for node in self._nodes])

        # Precompute undeformed normals
        self.undeformed_normals = self.compute_normals()

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
        """Compute area-weighted outward-pointing unit normal vectors for all non-mortar nodes.
        
        If U_np is provided, the normal vectors are calculated in the deformed configuration.
        Otherwise, they are calculated in the undeformed configuration.
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
            
            if dim == 3:
                # Differentiate between triangular and quadrilateral contact elements
                if len(facet_nodes) in (3, 6):
                    v1 = coords[1] - coords[0]
                    v2 = coords[2] - coords[0]
                    n_facet = 0.5 * np.cross(v1, v2)
                else:
                    v1 = coords[2] - coords[0]
                    v2 = coords[3] - coords[1]
                    n_facet = 0.5 * np.cross(v1, v2)
            else: # dim == 2
                t = coords[-1] - coords[0]
                n_facet = np.array([t[1], -t[0]])
            
            # Add to the normals of all nodes on this facet
            for node in facet_nodes:
                if node in node_to_idx:
                    normals[node_to_idx[node]] += n_facet

        # Normalize the normal vectors
        for i in range(self.nNonMortarNodes):
            norm = np.linalg.norm(normals[i])
            if norm > 1e-14:
                normals[i] /= norm
            else:
                normals[i] = np.zeros(dim)

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

        D = np.zeros((n_slave, n_slave))
        C = np.zeros((n_slave, n_master))

        # Reference-element dual matrices, used only as fallback for degenerate overlaps
        dual_mats = self.compute_local_dual_matrices(U_np)

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

                    for m_el, m_faceID in candidates:
                        m_nodes = m_el.nodes
                        m_coords = np.array([current_coords[nd] for nd in m_nodes])
                        m_full_2d = to_plane_coords(m_coords, p0, t1, t2)

                        for m_sub in get_sub_cells(m_el):
                            m_sub_2d = m_full_2d[m_sub]

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

                                    local_s = map_2d_to_natural(s_el, s_full_2d, x_gp_2d)
                                    local_m = map_2d_to_natural(m_el, m_full_2d, x_gp_2d)

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
                            s_start, s_end, L_slave, t_vec, n_vec = clip_1d_segments(sc_coords, mc_coords)
                            if s_end - s_start < 1e-12:
                                continue

                            half_len = 0.5 * (s_end - s_start)
                            mid_s = 0.5 * (s_start + s_end)
                            for gp_1d, w_1d in zip(LINE_GAUSS_PTS, LINE_GAUSS_W):
                                s_gp = mid_s + half_len * gp_1d[0]
                                x_gp = sc_coords[0] + s_gp * t_vec
                                dG = half_len * w_1d

                                local_s = map_2d_to_natural(s_el, s_coords, x_gp)
                                local_m = map_2d_to_natural(m_el, m_coords, x_gp)

                                N_s = s_el.getShapeFunctions(local_s)
                                N_m = m_el.getShapeFunctions(local_m)

                                s_num = s_el.elNumber
                                if s_num not in seg_records:
                                    seg_records[s_num] = []
                                    seg_masters[s_num] = {}
                                    slave_els[s_num] = (s_el, s_idx)
                                seg_records[s_num].append((N_s, m_el.elNumber, N_m, dG))
                                seg_masters[s_num][m_el.elNumber] = m_el

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
            if np.linalg.cond(M_t) < 1e12:
                A_e = np.diag(D_t) @ np.linalg.inv(M_t) @ T_e
            else:
                A_e = dual_mats[s_num][2]

            # Assemble the D block and per-master C blocks of this slave element
            D_blk = np.zeros((n_s, n_s))
            C_blks = {}
            for N_s, m_num, N_m, dG in records:
                M_bar = A_e @ N_s
                D_blk += np.outer(M_bar, N_s) * dG
                if m_num not in C_blks:
                    C_blks[m_num] = np.zeros((n_s, len(seg_masters[s_num][m_num].nodes)))
                C_blks[m_num] += np.outer(M_bar, N_m) * dG

            D[np.ix_(s_idx, s_idx)] += D_blk
            for m_num, C_blk in C_blks.items():
                m_idx = np.array([self.master_node_to_idx[nd] for nd in seg_masters[s_num][m_num].nodes])
                C[np.ix_(s_idx, m_idx)] += C_blk

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
        step_key = (timeStep.number, timeStep.timeIncrement, timeStep.totalTime)
        if step_key != self.last_timestep_key or not hasattr(self, "current_normals"):
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
            self.current_normals = self.compute_normals(U_np)
            D_full, C_full = self.compute_mortar_coupling_matrices(U_np)
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
            self.current_D_rowsum = np.sum(D_full, axis=1)
            # Precompute the sparsity patterns once per increment
            self.current_D_nz = [np.flatnonzero(np.abs(D_full[I]) > 1e-14) for I in range(nSlave)]
            self.current_C_nz = [np.flatnonzero(np.abs(C_full[I]) > 1e-14) for I in range(nSlave)]
        else:
            self.current_iteration += 1

        normals = self.current_normals
        D = self.current_D
        C = self.current_C

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

            g_I_weak = 0.0
            if len(nzD):
                g_I_weak -= D[I, nzD] @ (x_slave[nzD] @ n_I)
            if len(nzC):
                g_I_weak += C[I, nzC] @ (x_master[nzC] @ n_I)

            sgn_D = np.sign(self.current_D_rowsum[I])
            D_II = self.current_D_rowsum[I]
            inv_D = 1.0 / D_II if abs(D_II) > 1e-30 else 0.0
            p_n = -lambda_I * sgn_D   # physical normal pressure (>= 0 in contact)
            g_sep = g_I_weak * sgn_D  # separation gap (>0 open, <0 penetrating)

            if self.use_active_set and not self.active_set_frozen:
                # Semi-smooth normal complementarity (Gitterle et al. 2010 Eq. 55;
                # Hueber & Wohlmuth 2005; MOOSE ComputeWeightedGapLMMechanical
                # Contact): the Signorini KKT conditions p_n >= 0, g_sep >= 0,
                # p_n*g_sep = 0 are written as the single non-smooth function
                #   C_n = p_n - max(0, p_n - c_n*inv_D*g_sep) = 0,
                # whose two branches are
                #   active   (s_n > 0):  constraint  g_weak = 0,
                #   inactive (s_n <= 0): constraint  lambda = 0,
                # with the augmented indicator s_n = p_n - c_n*inv_D*g_sep. It is
                # re-evaluated EVERY Newton iteration - this is the literature-
                # standard PDASS = semi-smooth-Newton formulation with local
                # superlinear convergence. c_n is purely algorithmic: at
                # convergence g_sep -> 0, so the converged result is c_n-
                # independent and identical to any admissible active-set rule;
                # c_n ~ O(E) of the softer body (Farah 2018 Sec. 3.5.2). The
                # D_II-normalization makes c_n*inv_D*g_sep a pressure directly
                # comparable to p_n.
                s_n = p_n - self.c_n * inv_D * g_sep
                self.active_set[I] = bool(s_n > 0.0)

            self.recovered_lambdas[I] = lambda_I if self.active_set[I] else 0.0

            if self.active_set[I]:
                PExt[idx_LM_I] -= g_I_weak

                for K_nd in nzD:
                    s_dofs = slice(sf * K_nd, sf * K_nd + dim)
                    D_IK_n = D[I, K_nd] * n_I
                    PExt[s_dofs] += lambda_I * D_IK_n
                    K[s_dofs, idx_LM_I] -= D_IK_n
                    K[idx_LM_I, s_dofs] -= D_IK_n

                for J in nzC:
                    m_global = nSlave + J
                    m_dofs = slice(sf * m_global, sf * m_global + dim)
                    C_IJ_n = C[I, J] * n_I
                    PExt[m_dofs] -= lambda_I * C_IJ_n
                    K[m_dofs, idx_LM_I] += C_IJ_n
                    K[idx_LM_I, m_dofs] += C_IJ_n
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
            if (
                (self.active_set_stable_count >= 2 and self.current_iteration >= 2)
                or cycle
                or (self.current_iteration >= 20)
            ):
                self.active_set_frozen = True
