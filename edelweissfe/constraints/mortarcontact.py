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
    "condensation",
    "Statically condense (eliminate) the Lagrange multipliers from the assembled "
    "saddle-point system, yielding a displacement-dominated, better-conditioned "
    "system (Gitterle et al. 2010, Eqs. 85/86; Farah 2018, Sec. 3.5.3). The nodal "
    "multiplier z_I in R^dim is eliminated projection-free (valid across edges/kinks) "
    "and recovered afterwards. Frictionless only for now. Default: saddle-point.",
    bool,
    False,
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

        # Vectorial nodal Lagrange multiplier z_I in R^dim per slave node (global
        # components of the discrete contact traction), following Gitterle, Popp,
        # Gee & Wall (2010) and Popp, Wohlmuth, Gee & Wall (2012). The frictionless
        # case constrains the normal component (weighted gap, Signorini) and sets
        # the tangential traction to zero; friction later only replaces the
        # tangential constraint rows (Gitterle Eq. (86), rows St/Sl) by the
        # Coulomb stick/slip law. Using the full vector (instead of a scalar
        # normal multiplier) makes the later static condensation a projection-free
        # vectorial elimination (Gitterle Eq. (85)) that stays valid across
        # edges/kinks, and is the exact structure the frictional extension needs.
        self.dim = self.model.domainSize
        self.nMultipliers = self.dim * self.nNonMortarNodes
        self._nDof = self.sizeField * len(self._nodes) + self.nMultipliers

        # Recovered nodal normal pressure z_I . n_I (for output/verification)
        self.recovered_lambdas = np.zeros(self.nNonMortarNodes)
        # Recovered full nodal traction vectors z_I (global components)
        self.recovered_tractions = np.zeros((self.nNonMortarNodes, self.dim))

        # Opt-in static condensation of the multipliers (solver-level).
        self.use_condensation = bool(kwargs.get("condensation", False))
        # Coulomb friction is not implemented yet; kept for the friction-ready
        # interface (the tangential constraint rows become the stick/slip rows).
        self.friction_coefficient = float(
            kwargs.get("friction_coefficient", kwargs.get("frictioncoefficient", 0.0))
        )
        if self.use_condensation and self.friction_coefficient > 0.0:
            # Condensation of the tangential (friction) multipliers (Gitterle
            # Eq. 86, rows St/Sl) is not implemented yet. Fail loudly rather than
            # silently producing a frictionless result.
            raise NotImplementedError(
                "Dual condensation with Coulomb friction (mu > 0) is not implemented yet. "
                "Use condensation only for frictionless contact, or disable condensation "
                "to keep the saddle-point formulation."
            )
        self._fieldsOnNodes = [[self.field]] * len(self._nodes)
        self.active = True

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
        self.last_timestep_number = -1
        self.current_iteration = 0

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

    def _local_frame(self, n_I: np.ndarray) -> list[np.ndarray]:
        """Return the (dim-1) orthonormal tangent vectors spanning the plane
        orthogonal to the unit normal ``n_I``.

        In 2D there is a single in-plane tangent ``[n_y, -n_x]``; in 3D the two
        tangents are taken from :func:`get_tangent_basis`. These span the local
        frame in which the tangential contact constraints are expressed
        (frictionless: zero tangential traction; friction: Coulomb stick/slip).
        """
        from edelweissfe.constraints.mortar_geom_utils import get_tangent_basis

        if len(n_I) == 2:
            return [np.array([n_I[1], -n_I[0]])]
        t1, t2 = get_tangent_basis(n_I)
        return [t1, t2]

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
        if timeStep.number != self.last_timestep_number or not hasattr(self, "current_normals"):
            self.last_timestep_number = timeStep.number
            self.current_iteration = 0
            # Active set (Signorini) is updated EVERY Newton iteration and the
            # outer active-set loop is converged once the set no longer changes -
            # the convergence criterion of the primal-dual active set strategy
            # (Hüeber & Wohlmuth 2005) / semismooth Newton mortar contact
            # (Popp et al. 2012, Gitterle et al. 2010). There is deliberately NO
            # fixed iteration-count cutoff (that would be an ad-hoc heuristic and
            # could freeze a not-yet-settled set). A safeguard cap only guards
            # against pathological non-settling (e.g. chattering).
            self.active_set_frozen = False
            self.active_set_stable_count = 0
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
        # SADDLE-POINT MODE (vectorial, friction-ready): the nodal multiplier
        # z_I in R^dim (global components of the discrete contact traction) is an
        # explicit unknown, solved for jointly with the displacements. The slave
        # and master equilibrium rows carry the weighted contact traction
        # (+D z on the slave, -C z = -(M z) on the master). The dim constraint
        # rows per active node enforce, in the local frame (n_I, t_1[, t_2]):
        #   - normal:      weighted normal gap g_n,I = 0   (Signorini, active set)
        #   - tangential:  t_a . z_I = 0   (a = 1..dim-1)  (frictionless -> zero
        #                  tangential traction; the mu->0 limit of the Coulomb
        #                  stick/slip rows, which friction later replaces here).
        # This is algebraically equivalent to the scalar normal formulation
        # (once t_a.z_I = 0 holds, z_I = lambda_I n_I and D z / C z reduce to the
        # scalar lambda_I D n_I / lambda_I C n_I), but keeps the full traction
        # vector so the later condensation (Gitterle Eq. (85)) is a
        # projection-free vectorial elimination valid across edges/kinks.
        #
        # DOF layout of node I's dim scalar variables (local indices
        # z0 = sf*nNodes + dim*I, .. z0+dim-1): row z0 carries the normal-gap
        # equation, rows z0+1..z0+dim-1 the tangential-traction equations.
        # ----------------------------------------------------------------------
        idx_LM_0 = sf * nNodes
        Id = np.eye(dim)

        # Snapshot the active set to detect whether this iteration changed it
        # (stability-based freezing, see the increment-reset block above).
        active_set_before = self.active_set.copy()

        for I in range(nSlave):
            z0 = idx_LM_0 + dim * I
            z_slice = slice(z0, z0 + dim)
            z_I = np.array(U_np[z_slice], dtype=float)
            n_I = normals[I]
            nzD = self.current_D_nz[I]
            nzC = self.current_C_nz[I]

            # Weighted normal gap g_n,I (translation-invariant via the row-sum
            # identity sum_K D_IK = sum_J C_IJ).
            g_I_weak = 0.0
            if len(nzD):
                g_I_weak -= D[I, nzD] @ (x_slave[nzD] @ n_I)
            if len(nzC):
                g_I_weak += C[I, nzC] @ (x_master[nzC] @ n_I)

            lambda_I = float(z_I @ n_I)  # normal pressure component z_I . n_I

            if self.use_active_set and not self.active_set_frozen:
                sgn_D = np.sign(self.current_D_rowsum[I])
                if self.active_set[I]:
                    if lambda_I * sgn_D > 1e-10:
                        self.active_set[I] = False
                else:
                    if g_I_weak * sgn_D < -1e-10:
                        self.active_set[I] = True

            self.recovered_lambdas[I] = lambda_I if self.active_set[I] else 0.0
            self.recovered_tractions[I] = z_I if self.active_set[I] else 0.0

            if self.active_set[I]:
                # --- Equilibrium coupling: weighted contact traction forces ---
                for K_nd in nzD:
                    s_dofs = slice(sf * K_nd, sf * K_nd + dim)
                    PExt[s_dofs] += D[I, K_nd] * z_I
                    K[s_dofs, z_slice] -= D[I, K_nd] * Id
                for J in nzC:
                    m_global = nSlave + J
                    m_dofs = slice(sf * m_global, sf * m_global + dim)
                    PExt[m_dofs] -= C[I, J] * z_I
                    K[m_dofs, z_slice] += C[I, J] * Id

                # --- Constraint row z0: weighted normal gap g_n,I = 0 ---
                PExt[z0] -= g_I_weak
                for K_nd in nzD:
                    s_dofs = slice(sf * K_nd, sf * K_nd + dim)
                    K[z0, s_dofs] -= D[I, K_nd] * n_I
                for J in nzC:
                    m_global = nSlave + J
                    m_dofs = slice(sf * m_global, sf * m_global + dim)
                    K[z0, m_dofs] += C[I, J] * n_I

                # --- Constraint rows z0+1..z0+dim-1: tangential traction = 0 ---
                for a, t_a in enumerate(self._local_frame(n_I)):
                    r_t = z0 + 1 + a
                    PExt[r_t] -= float(t_a @ z_I)
                    K[r_t, z_slice] += t_a
            else:
                # Inactive: z_I = 0 (all components -> normal and tangential)
                for c in range(dim):
                    PExt[z0 + c] -= z_I[c]
                    K[z0 + c, z0 + c] += 1.0

        # Stability-based freezing of the active set: once the set is unchanged
        # for two consecutive iterations (and past a small warm-up), freeze it so
        # the Newton iteration converges on a fixed set. A safeguard cap prevents
        # an unbounded outer loop should the set fail to settle (e.g. oscillate).
        if self.use_active_set and not self.active_set_frozen:
            if np.array_equal(self.active_set, active_set_before):
                self.active_set_stable_count += 1
            else:
                self.active_set_stable_count = 0
            if (self.active_set_stable_count >= 2 and self.current_iteration >= 2) or (
                self.current_iteration >= 20
            ):
                self.active_set_frozen = True

    def _build_condensation_transform(self) -> list[list[tuple[int, float]]]:
        r"""Global basis-transformation rows ``T[I, :]`` per slave node, as a
        sparse ``[(K, T_IK), ..]`` list referencing slave node local indices.

        For linear facets ``T`` is the identity (every row is ``[(I, 1.0)]``), so
        the elimination reduces to a per-node scalar division by ``D_II``. For
        quadratic facets (CONQUAD8/9, CONTRI6, CONLINE3) the alpha = 1/3 basis
        transformation of Popp, Wohlmuth, Gee & Wall (2012) (see
        :func:`ContactElement.getBasisTransformation`) renders the *transformed*
        mortar matrix ``D_tilde = D_phys T^T`` diagonal; the elimination of
        ``z_I`` then uses the ``T``-weighted combination of the slave equilibrium
        rows of node ``I`` and its adjacent mid-side/corner nodes.

        The global ``T`` is assembled **topologically** (set, not accumulated):
        every element carries the identical local ``T_e``, so setting
        ``T[a, b] = T_e[a, b]`` avoids the double-counting on shared edges that
        additive accumulation would produce (which would give ``2*alpha`` and
        destroy the diagonality of ``D_tilde``; cf. Cichosz & Bischoff 2011).
        """
        nSlave = self.nNonMortarNodes
        rows = [dict() for _ in range(nSlave)]
        for I in range(nSlave):
            rows[I][I] = 1.0  # identity default (corners / linear nodes)
        for el, faceID in self.non_mortar_facets:
            T_e = el.getBasisTransformation()
            loc2glob = [self.slave_node_to_idx[nd] for nd in el.nodes]
            n = el.nNodes
            for a in range(n):
                for b in range(n):
                    v = T_e[a, b]
                    if abs(v) > 1e-14:
                        rows[loc2glob[a]][loc2glob[b]] = float(v)  # set, not +=
        return [sorted(r.items()) for r in rows]

    def getCondensationOperators(self) -> dict | None:
        """Operators for the solver-level static condensation of this
        constraint's vectorial Lagrange multipliers (Gitterle et al. 2010,
        Eqs. 85/86).

        Returns ``None`` unless ``condensation=True`` was requested. When on, the
        constraint keeps assembling the full (vectorial) saddle-point system so
        the solver can slice the bulk stiffness, coupling and residual out of the
        assembled matrix; this method exposes, per **active** slave node, the
        metadata the solver cannot recover from the matrix alone: the node's dim
        scalar multiplier variables ``z_I`` (global components of the contact
        traction), the slave ``Node``, the frozen unit normal ``n_I``, the
        diagonal *transformed* mortar entry ``D_tilde_II`` (== ``current_D_rowsum[I]``,
        since ``sum_K D_phys[I,K] = int(Phi_I) = D_tilde_II`` by the partition of
        unity), the transformation row ``T[I,:]`` (identity for linear facets),
        and the active flag.

        Because the multiplier is a full vector (not a scalar normal component),
        the elimination the solver performs is projection-free (no dependence on
        a single well-defined normal), hence valid across edges/kinks and the
        exact structure the frictional extension needs.

        Must be called *after* :func:`applyConstraint` in the same iteration.

        Returns
        -------
        dict | None
            ``{"field": <name>, "dim": <int>, "multipliers": [ {..}, .. ]}`` with
            one entry per slave node holding ``scalarVariables`` (list of dim
            ``ScalarVariable``), ``slaveNode``, ``localIndex`` (I), ``normal``,
            ``D_diag``, ``transform`` (``[(K, T_IK), ..]``) and ``active``.
            ``None`` if condensation is off or before the first assembly.
        """
        if not self.use_condensation:
            return None
        if self.friction_coefficient > 0.0:
            raise NotImplementedError("Dual condensation with friction is not implemented yet.")
        if not hasattr(self, "current_normals"):
            # applyConstraint has not run yet this analysis; nothing to condense.
            return None

        dim = self.dim
        transform = self._build_condensation_transform()
        multipliers = []
        for I in range(self.nNonMortarNodes):
            multipliers.append(
                {
                    "scalarVariables": [self.scalarVariables[dim * I + c] for c in range(dim)],
                    "slaveNode": self.non_mortar_nodes[I],
                    "localIndex": I,
                    "normal": self.current_normals[I],
                    # local tangent frame, in the SAME order as the tangential
                    # constraint rows assembled in applyConstraint (z-dof 1..dim-1)
                    "tangents": self._local_frame(self.current_normals[I]),
                    "D_diag": float(self.current_D_rowsum[I]),
                    "transform": transform[I],
                    "active": bool(self.active_set[I]),
                }
            )
        return {"field": self.field, "dim": dim, "multipliers": multipliers}
