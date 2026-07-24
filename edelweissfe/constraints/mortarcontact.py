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
    "friction_coefficient",
    "Coulomb friction coefficient mu (>= 0.0). mu = 0.0 (default) reproduces the "
    "frictionless behaviour exactly (no tangential multiplier DOFs are created).",
    float,
    0.0,
)
module.addOptionalArg(
    "friction_ct",
    "Semi-smooth-Newton complementarity parameter c_t (> 0) for the tangential "
    "(friction) active set. It does NOT change the converged solution, only the "
    "convergence behaviour (Gitterle et al. 2010, p. 555/565). The weighted slip is "
    "normalised by the nodal mortar weight D_II, so c_t balances the scales of the "
    "physical slip and the traction and should be chosen at the order of Young's "
    "modulus of the softer contacting body (Gitterle et al. 2010 p. 565; Farah 2018 "
    "Sec. 3.5.2).",
    float,
    1.0,
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
        self.friction_coefficient = float(
            kwargs.get("friction_coefficient", kwargs.get("frictioncoefficient", 0.0))
        )
        self.c_t = float(kwargs.get("friction_ct", kwargs.get("frictionct", 1.0)))

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

        # Normal Lagrange multipliers: one scalar per slave node (the normal
        # contact pressure), unchanged regardless of friction. Tangential
        # multipliers (Coulomb friction) add (dim - 1) scalar components per
        # slave node - the in-plane traction z_t in the frozen tangent basis
        # (see compute_tangent_basis) - laid out right after the normal block
        # in the additional-scalar-variable / DOF vector. With mu = 0 no
        # tangential DOFs are created and the layout is identical to the
        # frictionless case.
        self.nMultipliers = self.nNonMortarNodes
        self.nTangentialComponents = (model.domainSize - 1) if self.friction_coefficient > 0.0 else 0
        self.nTangentialMultipliers = self.nNonMortarNodes * self.nTangentialComponents
        self._nDof = (
            self.sizeField * len(self._nodes) + self.nMultipliers + self.nTangentialMultipliers
        )

        self.recovered_lambdas = np.zeros(self.nNonMortarNodes)
        self.recovered_tractions_t = np.zeros((self.nNonMortarNodes, self.nTangentialComponents))
        self.stick_set = np.zeros(self.nNonMortarNodes, dtype=bool)

        # Single-owner treatment of shared slave nodes (only relevant with
        # friction). When several decomposed contact surfaces meet at an edge
        # they share slave nodes. Each mortar constraint is an independent
        # object, so a shared node would receive contact constraints from EVERY
        # surface it belongs to. Without friction this is fine (a shared node
        # then carries only the normal weak-gap constraint of each surface, i.e.
        # < dim constraints on its dim displacement DOFs). WITH friction the
        # owner surface additionally enforces the (dim-1) tangential stick/slip
        # constraints, which together with its normal constraint already tie the
        # node completely (dim constraints = full basis); any further contact
        # constraint from another surface is then redundant and makes the
        # saddle-point system singular (the extra normal multiplier is
        # indeterminate; the tangential multipliers blow up). Therefore, once
        # friction is active, each shared slave node is owned ENTIRELY by exactly
        # one surface (the first in input order): the owner enforces its full
        # normal + tangential contact, every other surface skips that node
        # completely (drives its normal lambda and tangential z_t to zero) - its
        # contact is fully represented by the owner. Ownership is recorded in a
        # model-level registry shared across all mortarcontact constraints.
        self._owns_node = np.ones(self.nNonMortarNodes, dtype=bool)
        if self.friction_coefficient > 0.0:
            owners = getattr(model, "_mortarFrictionOwners", None)
            if owners is None:
                owners = {}
                model._mortarFrictionOwners = owners
            for I, node in enumerate(self.non_mortar_nodes):
                if owners.setdefault(node, self._name) != self._name:
                    self._owns_node[I] = False
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
        return self.nMultipliers + self.nTangentialMultipliers

    def compute_tangent_basis(self, normals: np.ndarray) -> np.ndarray:
        """Return an orthonormal in-plane tangent basis per slave node, shape
        (nNonMortarNodes, nTangentialComponents, dim).

        In 3D these are the two vectors t1, t2 spanning the plane perpendicular
        to the nodal normal (from get_tangent_basis); in 2D it is the single
        vector t = [-n_y, n_x] obtained by rotating n by 90 degrees. The basis
        is arbitrary but consistent per node, which is sufficient because the
        Coulomb cone |z_t| <= mu*p_n and the slip increment are both expressed
        in the SAME frame, so all physical quantities (|z_t|, |u_t|) are frame
        invariant. Only meaningful / used when friction_coefficient > 0.
        """
        dim = self.model.domainSize
        basis = np.zeros((self.nNonMortarNodes, self.nTangentialComponents, dim))
        if self.nTangentialComponents == 0:
            return basis

        # A slave node whose area-weighted facet normals cancel gets a zero
        # normal from compute_normals(). Such a node has no well-defined contact
        # frame; leaving its tangent basis at zero here (and skipping friction
        # for it in applyConstraint) avoids the 0/0 = NaN that get_tangent_basis
        # would otherwise produce (harmless without friction, fatal with it - it
        # propagates NaN tractions into the bulk and breaks the return mapping).
        if dim == 3:
            from edelweissfe.constraints.mortar_geom_utils import get_tangent_basis

            for I in range(self.nNonMortarNodes):
                if np.dot(normals[I], normals[I]) < 1e-24:
                    continue  # leave basis[I] = 0
                t1, t2 = get_tangent_basis(normals[I])
                basis[I, 0] = t1
                basis[I, 1] = t2
        else:  # dim == 2
            for I in range(self.nNonMortarNodes):
                n_I = normals[I]
                if np.dot(n_I, n_I) < 1e-24:
                    continue  # leave basis[I] = 0
                basis[I, 0] = np.array([-n_I[1], n_I[0]])
        return basis

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
        if timeStep.number != self.last_timestep_number or not hasattr(self, "current_normals"):
            self.last_timestep_number = timeStep.number
            self.current_iteration = 0
            self.current_normals = self.compute_normals(U_np)
            # Frozen per-node tangent basis for Coulomb friction (empty if mu = 0).
            # Frozen together with the normals and the mortar matrices, so the
            # assembled tangent stiffness is the exact Jacobian of the algebraic
            # (frozen-geometry) residual.
            self.current_tangents = self.compute_tangent_basis(self.current_normals)
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

        # ----------------------------------------------------------------------
        # Coulomb friction preamble (only when mu > 0). The tangential Lagrange
        # multipliers z_t are laid out in a contiguous block right after the
        # nSlave normal multipliers.
        # ----------------------------------------------------------------------
        mu = self.friction_coefficient
        ntc = self.nTangentialComponents
        idx_TAU_0 = idx_LM_0 + nSlave

        if mu > 0.0:
            # Weighted relative tangential slip INCREMENT: it must be measured
            # over the change of displacement since the start of the current
            # increment (dU), not the total displacement U_np - a node that
            # started sticking must not be treated as sliding relative to the
            # master since t = 0. D, C, n and the tangent basis are frozen
            # within the increment (staggered update), consistent with the
            # normal part and with a backward-Euler discretisation of the
            # relative tangential velocity (Gitterle et al. 2010, Eqs. (47)/(52)).
            du_disp = dU[: sf * nNodes].reshape(nNodes, sf)[:, :dim]
            du_slave = du_disp[:nSlave]
            du_master = du_disp[nSlave:]
            tangents = self.current_tangents

        for I in range(nSlave):
            idx_LM_I = idx_LM_0 + I
            lambda_I = U_np[idx_LM_I]
            n_I = normals[I]
            nzD = self.current_D_nz[I]
            nzC = self.current_C_nz[I]

            # Shared slave node owned entirely by another surface (single-owner,
            # friction only): skip its contact here - drive its normal lambda and
            # its tangential z_t to zero. Its contact is enforced by the owner.
            if not self._owns_node[I]:
                self.active_set[I] = False
                self.recovered_lambdas[I] = 0.0
                PExt[idx_LM_I] -= lambda_I
                K[idx_LM_I, idx_LM_I] += 1.0
                if mu > 0.0:
                    idx_TAU_I0 = idx_TAU_0 + I * ntc
                    self.recovered_tractions_t[I] = 0.0
                    self.stick_set[I] = False
                    for c in range(ntc):
                        idx_TAU_Ic = idx_TAU_I0 + c
                        PExt[idx_TAU_Ic] -= U_np[idx_TAU_Ic]
                        K[idx_TAU_Ic, idx_TAU_Ic] += 1.0
                continue

            g_I_weak = 0.0
            if len(nzD):
                g_I_weak -= D[I, nzD] @ (x_slave[nzD] @ n_I)
            if len(nzC):
                g_I_weak += C[I, nzC] @ (x_master[nzC] @ n_I)

            if self.use_active_set:
                if self.current_iteration < 5:
                    sgn_D = np.sign(self.current_D_rowsum[I])
                    if self.active_set[I]:
                        if lambda_I * sgn_D > 1e-10:
                            self.active_set[I] = False
                    else:
                        if g_I_weak * sgn_D < -1e-10:
                            self.active_set[I] = True

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

            # ----------------------------------------------------------------
            # COULOMB FRICTION (tangential Lagrange multipliers z_t), semi-smooth
            # / primal-dual active set formulation of Gitterle, Popp, Gee & Wall
            # (2010), "Finite deformation frictional mortar contact using a
            # semi-smooth Newton method with consistent linearization",
            # Int. J. Numer. Methods Eng. 84:543-571.
            #
            # Trial traction  z_tr = z_t + c_t * u_t  (Gitterle Eq. 58), with the
            # physical (D_II-normalized) slip increment u_t (Eq. 47) and friction
            # bound  b = mu * p_n. Node classified by ||z_tr|| vs b (Eqs. 72/73):
            #   stick (||z_tr|| <= b): exact constraint  C_t = u_t = 0  (Eq. 54/74),
            #        z_t is its Lagrange multiplier (tangential analog of g_weak=0).
            #   slip  (||z_tr|| >  b): C_t = z_t - b * z_tr/||z_tr||  (Eq. 61 divided
            #        by ||z_tr||)  ->  ||z_t|| = b along the trial direction.
            # Both branches are exact Coulomb friction (no penalty regularization);
            # c_t enters only the classification and the trial direction, so it is a
            # purely algorithmic parameter with no effect on the converged solution
            # (Gitterle p. 555/565). The stick row (zero z_t-diagonal, saddle-point)
            # mirrors the well-conditioned normal row; the slip row has an O(1)
            # z_t-diagonal. The nested c_n*g_tilde term of the bound (Eq. 60) is
            # dropped: it vanishes at convergence (normal row drives g_weak -> 0) and
            # active/inactive is decided by the normal active set above.
            # p_n is the PHYSICAL (>= 0) normal pressure; in this code's sign
            # convention p_n = -lambda_I*sgn_D for an active (compressed) node.
            # Frozen geometry (n, t, D, C) => K is the exact Jacobian of this
            # algebraic residual; verified by the FD tangent check in test9_friction.
            # Sign convention: K[row, col] = -d(PExt[row])/d(col).
            # ----------------------------------------------------------------
            if mu > 0.0:
                idx_TAU_I0 = idx_TAU_0 + I * ntc
                t_I = tangents[I]  # (ntc, dim) orthonormal in-plane basis
                z_t = np.array([U_np[idx_TAU_I0 + c] for c in range(ntc)])

                sgn_D = np.sign(self.current_D_rowsum[I])
                p_n = -lambda_I * sgn_D  # physical normal pressure
                b = mu * p_n  # Coulomb friction bound

                # A node with a degenerate (zero) nodal normal has no valid
                # tangent frame (t_I == 0); skip friction for it (enforce z_t = 0)
                # to avoid a zero/NaN tangential row.
                has_tangent = np.dot(t_I[0], t_I[0]) > 0.5

                if self.active_set[I] and b > 0.0 and has_tangent:
                    db_dlam = -mu * sgn_D  # d(b)/d(lambda_I)

                    # Weighted tangential slip increment (Gitterle Eq. (47)),
                    # D, C and the tangent basis frozen within the increment.
                    # t_I already lies in the tangent plane, so t_I @ (.) is the
                    # tangential projection (no separate P = I - n(x)n needed).
                    w_vec = np.zeros(dim)
                    if len(nzD):
                        w_vec += D[I, nzD] @ du_slave[nzD]
                    if len(nzC):
                        w_vec -= C[I, nzC] @ du_master[nzC]
                    # Normalise by the nodal mortar weight D_II (row-sum, = int Phi_I
                    # dGamma) so that u_t is the PHYSICAL relative tangential slip
                    # (displacement units, u_t = u_tilde / D_II), independent of the
                    # element size. This makes the complementarity parameter c_t
                    # dimensionally a stress/length and directly of the order of the
                    # softer body's Young's modulus, as recommended by Gitterle et al.
                    # (2010, p. 565: "choose c_t such that the scales of u_tilde and
                    # z_t are balanced ... reflect the material parameters") and Farah
                    # (2018, Sec. 3.5.2). c_t is a purely algorithmic parameter: it does
                    # not change the converged solution (at slip z_t stays parallel to
                    # u_t for any c_t), only the Newton convergence. D_II is frozen, so
                    # this is just a constant scaling of the slip term and its tangent.
                    D_II = self.current_D_rowsum[I]
                    inv_D = 1.0 / D_II if abs(D_II) > 1e-30 else 0.0
                    u_t = inv_D * (t_I @ w_vec)  # (ntc,) physical slip increment

                    z_tr = z_t + self.c_t * u_t  # trial tangential traction
                    z_tr_norm = np.linalg.norm(z_tr)

                    # stick / slip branch = generalized derivative of the max in the
                    # unified NCP (Gitterle Eqs. 72/73). Evaluated fresh every
                    # iteration (semi-smooth Newton), like MOOSE; NOT frozen.
                    slip = z_tr_norm > b
                    self.stick_set[I] = not slip

                    # Friction nodal force: same structure as the normal force
                    # lambda*D*n, with the normal n replaced by each tangent
                    # basis vector t_c. Applied for both stick and slip.
                    for c in range(ntc):
                        idx_TAU_Ic = idx_TAU_I0 + c
                        t_c = t_I[c]
                        for K_nd in nzD:
                            s_dofs = slice(sf * K_nd, sf * K_nd + dim)
                            D_IK_t = D[I, K_nd] * t_c
                            PExt[s_dofs] += z_t[c] * D_IK_t
                            K[s_dofs, idx_TAU_Ic] -= D_IK_t
                        for J in nzC:
                            m_global = nSlave + J
                            m_dofs = slice(sf * m_global, sf * m_global + dim)
                            C_IJ_t = C[I, J] * t_c
                            PExt[m_dofs] -= z_t[c] * C_IJ_t
                            K[m_dofs, idx_TAU_Ic] += C_IJ_t

                    if slip:
                        # SLIP branch of the unified NCP (max = ||z_tr||), Gitterle Eq. 61:
                        #     C_t = ||z_tr|| z_t - b z_tr   ->  ||z_t|| = b along z_tr.
                        # The un-normalized form is deliberate: its tangent has no bare
                        # 1/||z_tr|| term - every occurrence of dir = z_tr/||z_tr|| is
                        # multiplied by z_t, which vanishes at the onset of slip
                        # (z_t = 0, tiny slip). The normalized variant C_t = z_t - b*dir
                        # instead carries d(dir)/du ~ 1/||z_tr||, which blows the tangent
                        # up for just-slipping nodes and corrupts the displacement
                        # solution (breaks the bulk return mapping). See Gitterle p. 554.
                        dir_c = z_tr / z_tr_norm      # (ntc,) unit trial direction
                        dir_spatial = dir_c @ t_I     # (dim,)
                        C_t = z_tr_norm * z_t - b * z_tr  # (ntc,)
                        for c in range(ntc):
                            idx_TAU_Ic = idx_TAU_I0 + c
                            PExt[idx_TAU_Ic] -= C_t[c]
                            # d(C_t[c])/d(z_t[cp]) = dir_c[cp] z_t[c] + (||z_tr|| - b) delta
                            for cp in range(ntc):
                                idx_TAU_Icp = idx_TAU_I0 + cp
                                dCt_dzt = dir_c[cp] * z_t[c] + ((z_tr_norm - b) if cp == c else 0.0)
                                K[idx_TAU_Ic, idx_TAU_Icp] += dCt_dzt
                            # d(C_t[c])/d(lambda) = -(db/dlam) z_tr[c]
                            K[idx_TAU_Ic, idx_LM_I] += -db_dlam * z_tr[c]
                            # d(C_t[c])/d(u) via z_tr = z_t + c_t u_t (u_t normalised by D_II):
                            #   coeff = z_t[c] dir_spatial - b t_c
                            #   +c_t inv_D D[I,K] coeff (slave),  -c_t inv_D C[I,J] coeff (master)
                            coeff_vec = z_t[c] * dir_spatial - b * t_I[c]  # (dim,)
                            for K_nd in nzD:
                                s_dofs = slice(sf * K_nd, sf * K_nd + dim)
                                K[idx_TAU_Ic, s_dofs] += self.c_t * inv_D * D[I, K_nd] * coeff_vec
                            for J in nzC:
                                m_global = nSlave + J
                                m_dofs = slice(sf * m_global, sf * m_global + dim)
                                K[idx_TAU_Ic, m_dofs] += -self.c_t * inv_D * C[I, J] * coeff_vec
                    else:
                        # STICK branch of the unified NCP (max = b), Gitterle Eq. 74:
                        #     C_t = b z_t - b z_tr = -b c_t u_t   ->  u_t = 0.
                        # This is the same single expression as slip (only max differs),
                        # exactly as in MOOSE ComputeFrictionalForceLMMechanicalContact.
                        # z_t has no self-diagonal here (d/dz_t = b - b = 0): a saddle row,
                        # like the normal LM. b depends on lambda (via p_n), giving the
                        # d/dlambda coupling below.
                        C_t = b * z_t - b * z_tr  # (ntc,) == -b c_t u_t
                        for c in range(ntc):
                            idx_TAU_Ic = idx_TAU_I0 + c
                            PExt[idx_TAU_Ic] -= C_t[c]
                            # d(C_t[c])/d(lambda) = db_dlam (z_t - z_tr)[c] = -db_dlam c_t u_t[c]
                            K[idx_TAU_Ic, idx_LM_I] += -db_dlam * self.c_t * u_t[c]
                            # d(C_t[c])/d(u) = -b c_t d(u_t[c])/d(u), u_t normalised by D_II:
                            #   -b c_t inv_D D[I,K] t_c (slave),  +b c_t inv_D C[I,J] t_c (master)
                            for K_nd in nzD:
                                s_dofs = slice(sf * K_nd, sf * K_nd + dim)
                                K[idx_TAU_Ic, s_dofs] += -b * self.c_t * inv_D * D[I, K_nd] * t_I[c]
                            for J in nzC:
                                m_global = nSlave + J
                                m_dofs = slice(sf * m_global, sf * m_global + dim)
                                K[idx_TAU_Ic, m_dofs] += b * self.c_t * inv_D * C[I, J] * t_I[c]

                    self.recovered_tractions_t[I] = z_t
                else:
                    # Inactive node, or active node with no normal pressure yet
                    # (b = mu*p_n = 0): no friction. Enforce z_t = 0, which also
                    # avoids the degenerate stick row (it scales with b).
                    self.recovered_tractions_t[I] = 0.0
                    self.stick_set[I] = False
                    for c in range(ntc):
                        idx_TAU_Ic = idx_TAU_I0 + c
                        PExt[idx_TAU_Ic] -= z_t[c]
                        K[idx_TAU_Ic, idx_TAU_Ic] += 1.0

        import os as _os
        if mu > 0.0 and _os.environ.get("FRICTION_DEBUG"):
            import sys as _sys
            fin = bool(np.isfinite(PExt).all())
            nstick = int(self.stick_set.sum())
            nact = int(self.active_set.sum())
            maxzt = float(np.abs(self.recovered_tractions_t).max()) if self.recovered_tractions_t.size else 0.0
            maxpe = float(np.abs(PExt[np.isfinite(PExt)]).max()) if fin else float("nan")
            print(
                f"[FRIC ts={timeStep.number} it={self.current_iteration}] "
                f"active={nact} stick={nstick} slip={nact - nstick} "
                f"max|z_t|={maxzt:.3e} max|PExt|={maxpe:.3e} PExt_finite={fin}",
                file=_sys.stderr, flush=True,
            )
