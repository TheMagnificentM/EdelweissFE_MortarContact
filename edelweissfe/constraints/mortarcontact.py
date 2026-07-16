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
                        f"MortarContact3D only supports explicit contact elements starting with 'CON'. "
                        f"Got element type '{el.elType}'."
                    )
                self.non_mortar_facets.append((el, faceID))

        self.mortar_facets = []
        for faceID, elements in self.mortarSurface.items():
            for el in elements:
                if not el.elType.upper().startswith("CON"):
                    raise ValueError(
                        f"MortarContact3D only supports explicit contact elements starting with 'CON'. "
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

        self._fieldsOnNodes = [[self.field]] * len(self._nodes)
        self.active = True

        # Node index lookups for fast access
        self.node_to_global_idx = {node: i for i, node in enumerate(self._nodes)}
        self.slave_node_to_idx = {node: i for i, node in enumerate(self.non_mortar_nodes)}
        self.master_node_to_idx = {node: i for i, node in enumerate(self.mortar_nodes)}

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
        
        # Build node lookup map for fast index retrieval
        node_to_idx = {node: i for i, node in enumerate(self.non_mortar_nodes)}

        # Iterate over all non-mortar facets
        for el, faceID in self.non_mortar_facets:
            facet_nodes = el.nodes
            
            # Retrieve coordinates
            coords = []
            for node in facet_nodes:
                X = node.coordinates
                if U_np is not None:
                    # Retrieve displacement from local solution slice
                    # Nodes are stored in constraint order: non_mortar_nodes + mortar_nodes
                    node_idx = self._nodes.index(node)
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
                    node_idx = self._nodes.index(node)
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
        
        This method performs the core Mortar integration:
        1. For each Slave (non-mortar) boundary facet:
           - Finds overlapping Master (mortar) boundary facets.
           - Projects Master facets onto the local Slave plane.
           - Clips the polygons using Sutherland-Hodgman in 2D.
           - Triangulates the overlap into integration sub-cells.
           - Performs numerical integration using a 3-point Gauss rule on each sub-triangle.
           - Evaluates the dual Slave shape functions \bar{M} and standard Master shape functions N.
           - Assembles the local integration contributions into the global D and C matrices.
        """
        dim = self.model.domainSize
        n_slave = self.nNonMortarNodes
        n_master = self.nMortarNodes
        
        # Initialize global matrices D (n_slave x n_slave) and C (n_slave x n_master)
        D = np.zeros((n_slave, n_slave))
        C = np.zeros((n_slave, n_master))
        
        # Precompute the local dual transformation matrices A_e for all slave elements
        dual_mats = self.compute_local_dual_matrices(U_np)
        
        # Import geometry helper functions directly
        from edelweissfe.constraints.mortar_geom_utils import (
            project_point_to_plane,
            get_tangent_basis,
            to_plane_coords,
            sutherland_hodgman_clip,
            triangulate_polygon,
            to_3d_coords
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

        # Build AABB bounding boxes for all master elements to construct the BVH tree
        facets_with_bounds = []
        for m_el, m_faceID in self.mortar_facets:
            m_nodes = m_el.nodes
            m_coords = np.array([current_coords[n] for n in m_nodes])
            centroid = np.mean(m_coords, axis=0)
            
            # Extract min and max coordinates with a safety margin (e.g. 15% of size)
            aabb_min = np.min(m_coords, axis=0)
            aabb_max = np.max(m_coords, axis=0)
            margin = max(0.15 * np.max(aabb_max - aabb_min), 0.1)
            aabb_min -= margin
            aabb_max += margin
            
            facets_with_bounds.append(((m_el, m_faceID), centroid, aabb_min, aabb_max))
            
        bvh_root = build_bvh(facets_with_bounds)
        
        # Loop over each Slave facet
        for s_el, s_faceID in self.non_mortar_facets:
            s_nodes = s_el.nodes
            s_num = s_el.elNumber
            _, _, A_e = dual_mats[s_num]
            
            # Get physical coordinates of the Slave nodes
            s_coords = np.array([current_coords[nd] for nd in s_nodes])
            
            # Setup the auxiliary local projection plane for this Slave facet
            p0 = np.mean(s_coords, axis=0)
            if dim == 3:
                if len(s_nodes) in (3, 6):
                    v1 = s_coords[1] - s_coords[0]
                    v2 = s_coords[2] - s_coords[0]
                else:
                    v1 = s_coords[2] - s_coords[0]
                    v2 = s_coords[3] - s_coords[1]
                normal = np.cross(v1, v2)
                normal /= np.linalg.norm(normal)
            else: # dim == 2
                t = s_coords[-1] - s_coords[0]
                normal = np.array([t[1], -t[0]])
                normal /= np.linalg.norm(normal)
                
            t1, t2 = get_tangent_basis(normal)
            s_2d = to_plane_coords(s_coords, p0, t1, t2)

            # Query the BVH tree using the slave facet AABB to find nearby Master candidates
            s_aabb_min = np.min(s_coords, axis=0)
            s_aabb_max = np.max(s_coords, axis=0)
            s_margin = max(0.15 * np.max(s_aabb_max - s_aabb_min), 0.1)
            s_aabb_min -= s_margin
            s_aabb_max += s_margin

            candidates = []
            query_bvh(bvh_root, s_aabb_min, s_aabb_max, candidates)
            
            # Loop over candidate Master facets (accelerated search)
            for m_el, m_faceID in candidates:
                m_nodes = m_el.nodes
                
                # Get physical coordinates of the Master nodes
                m_coords = np.array([current_coords[nd] for nd in m_nodes])
                
                # Project Master facet nodes onto the Slave projection plane
                proj_m_coords = np.array([project_point_to_plane(p, p0, normal) for p in m_coords])
                m_2d = to_plane_coords(proj_m_coords, p0, t1, t2)
                
                # Clip the Master facet with the Slave facet to find overlap
                overlap_2d = sutherland_hodgman_clip(m_2d, s_2d)
                if len(overlap_2d) < 3:
                    continue # No overlap or degenerate intersection polygon
                    
                # Triangulate the overlap polygon into sub-triangles
                sub_triangles = triangulate_polygon(overlap_2d)
                
                # Integrate over each sub-triangle using a 3-point symmetric Gauss rule
                # Local coordinates and weights on standard reference triangle: [0, 1]x[0, 1]
                gauss_pts = [
                    np.array([1.0 / 6.0, 1.0 / 6.0]),
                    np.array([2.0 / 3.0, 1.0 / 6.0]),
                    np.array([1.0 / 6.0, 2.0 / 3.0])
                ]
                gauss_w = [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]
                
                for tri in sub_triangles:
                    # Vertices of the sub-triangle in local 2D plane coordinates
                    v0, v1, v2 = tri[0], tri[1], tri[2]
                    
                    # Compute the area/Jacobian of the sub-triangle mapping
                    area_jac = abs((v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1]))
                    
                    for gp, w in zip(gauss_pts, gauss_w):
                        # Interpolate the Gauss point inside the sub-triangle
                        L1, L2 = gp[0], gp[1]
                        L0 = 1.0 - L1 - L2
                        x_gp_2d = L0 * v0 + L1 * v1 + L2 * v2
                        
                        # Map the local 2D Gauss point back to 3D physical coordinates
                        x_gp_3d = to_3d_coords([x_gp_2d], p0, t1, t2)[0]
                        
                        # --- Map to Slave natural space (xi_s, eta_s) ---
                        local_s = map_2d_to_natural(s_el, s_2d, x_gp_2d)
                        
                        # --- Map to Master natural space (xi_m, eta_m) ---
                        local_m = map_2d_to_natural(m_el, m_2d, x_gp_2d)
                        
                        # Evaluate standard shape functions N at mapped locations
                        N_s = s_el.getShapeFunctions(local_s)
                        N_m = m_el.getShapeFunctions(local_m)
                        
                        # Evaluate dual shape functions: M_bar = A_e * N_s
                        M_bar = A_e @ N_s
                        
                        # Differential area element on the sub-triangle
                        dGamma = area_jac * w
                        
                        # Assemble into global coupling matrices D and C
                        for i, s_nd in enumerate(s_nodes):
                            global_s_idx = self.slave_node_to_idx[s_nd]
                            
                            # Add to D matrix (Slave shape functions)
                            for j, s_nd_inner in enumerate(s_nodes):
                                global_s_inner_idx = self.slave_node_to_idx[s_nd_inner]
                                D[global_s_idx, global_s_inner_idx] += M_bar[i] * N_s[j] * dGamma
                                
                            # Add to C matrix (Master shape functions)
                            for j, m_nd in enumerate(m_nodes):
                                global_m_idx = self.master_node_to_idx[m_nd]
                                C[global_s_idx, global_m_idx] += M_bar[i] * N_m[j] * dGamma
                                
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

        # Detect new increment to track iterations and reset current_iteration
        if timeStep.number != self.last_timestep_number:
            self.last_timestep_number = timeStep.number
            self.current_iteration = 0
        else:
            self.current_iteration += 1

        # Compute normals and mortar matrices D and C in the deformed configuration
        normals = self.compute_normals(U_np)
        D, C = self.compute_mortar_coupling_matrices(U_np)

        # Compute current coordinates of all nodes in deformed configuration
        current_coords = {}
        for node in self._nodes:
            X = node.coordinates
            node_idx = self.node_to_global_idx[node]
            u = U_np[self.sizeField * node_idx : self.sizeField * node_idx + dim]
            current_coords[node] = X + u

        # Loop over non-mortar nodes to update active set, residues, and stiffness
        for I, node_s in enumerate(self.non_mortar_nodes):
            idx_LM_I = self.sizeField * len(self._nodes) + I
            lambda_I = U_np[idx_LM_I]
            node_s_idx = self.node_to_global_idx[node_s]

            # Compute weak gap for node I:
            # g_I_weak = -D_I * (x_I . n_I) + sum_J C_IJ * (x_J . n_I)
            x_I = current_coords[node_s]
            n_I = normals[I]
            g_I_weak = -D[I, I] * np.dot(x_I, n_I)
            for J, node_m in enumerate(self.mortar_nodes):
                if C[I, J] != 0:
                    x_J = current_coords[node_m]
                    g_I_weak += C[I, J] * np.dot(x_J, n_I)

            if self.use_active_set:
                # To prevent active set oscillations/cycling, freeze the active set after 5 iterations
                if self.current_iteration < 5:
                    sgn_D = np.sign(D[I, I])
                    if self.active_set[I]:
                        # If active, check for tension (lambda_I * sgn_D > 1e-10)
                        if lambda_I * sgn_D > 1e-10:
                            self.active_set[I] = False
                    else:
                        # If inactive, check for penetration (g_I_weak * sgn_D < -1e-10)
                        if g_I_weak * sgn_D < -1e-10:
                            self.active_set[I] = True

            # Assemble based on updated active status
            if self.active_set[I]:
                # 1. Multiplier residual equation (weak gap)
                PExt[idx_LM_I] -= g_I_weak

                # 2. Slave force contribution
                PExt[self.sizeField * node_s_idx : self.sizeField * node_s_idx + dim] += lambda_I * D[I, I] * n_I

                # 3. Master forces contributions
                for J, node_m in enumerate(self.mortar_nodes):
                    if C[I, J] != 0:
                        node_m_idx = self.node_to_global_idx[node_m]
                        PExt[self.sizeField * node_m_idx : self.sizeField * node_m_idx + dim] -= lambda_I * C[I, J] * n_I

                # 4. Stiffness matrix entries (symmetric coupling terms)
                # K[u_I, lambda_I] and K[lambda_I, u_I]
                K[self.sizeField * node_s_idx : self.sizeField * node_s_idx + dim, idx_LM_I] -= D[I, I] * n_I
                K[idx_LM_I, self.sizeField * node_s_idx : self.sizeField * node_s_idx + dim] -= D[I, I] * n_I

                # K[u_J, lambda_I] and K[lambda_I, u_J]
                for J, node_m in enumerate(self.mortar_nodes):
                    if C[I, J] != 0:
                        node_m_idx = self.node_to_global_idx[node_m]
                        K[self.sizeField * node_m_idx : self.sizeField * node_m_idx + dim, idx_LM_I] += C[I, J] * n_I
                        K[idx_LM_I, self.sizeField * node_m_idx : self.sizeField * node_m_idx + dim] += C[I, J] * n_I
            else:
                # Inactive node: lambda_I = 0
                # 1. Multiplier residual equation
                PExt[idx_LM_I] -= lambda_I

                # 2. Stiffness matrix diagonal entry
                K[idx_LM_I, idx_LM_I] += 1.0
