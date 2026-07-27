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
module.addOptionalArg(
    "friction_coefficient",
    "Coulomb friction coefficient mu (>= 0.0). mu = 0.0 (default) reproduces the "
    "frictionless behaviour exactly (the tangential rows stay t_a . z_I = 0).",
    float,
    0.0,
)
module.addOptionalArg(
    "friction_ramp",
    "Ramp the friction coefficient linearly from 0 to its target value over this "
    "fraction of the step progress, then hold it. 0.0 (default) disables ramping. "
    "Purely a robustness/continuation device; the converged final state is unaffected.",
    float,
    0.0,
)
module.addOptionalArg(
    "friction_cn",
    "Semi-smooth-Newton complementarity parameter c_n (> 0) entering the augmented "
    "Coulomb bound b = mu*max(0, p_n + c_n*inv_D*g_sep) (Alart-Curnier; Gitterle et "
    "al. 2010). Normalized by the nodal mortar weight D_II. Purely algorithmic (no "
    "effect on the converged solution). Choose at the order of Young's modulus of the "
    "softer body (Hueber & Wohlmuth 2005; Farah 2018 Sec. 3.5.2, c_n ~ c_t ~ O(E)).",
    float,
    1.0e6,
)
module.addOptionalArg(
    "friction_ct",
    "Semi-smooth-Newton complementarity parameter c_t (> 0) for the tangential set. "
    "Does NOT change the converged solution, only convergence behaviour (Gitterle et "
    "al. 2010 p. 555/565). The weighted slip is normalised by D_II, so c_t ~ O(E).",
    float,
    1.0,
)
module.addOptionalArg(
    "friction_epsilon",
    "PDASS friction-activation threshold on the raw normal pressure (MOOSE "
    "'contact_pressure < epsilon', default 1e-7): friction engages only once p_n "
    "exceeds it. Prevents a contact-initiation shock. Purely algorithmic.",
    float,
    1.0e-7,
)
module.addOptionalArg(
    "frictionSymmetryBCs",
    "Symmetry / Dirichlet-aware friction restriction (opt-in, default empty). "
    "Comma-separated '<nodeSet>:<component>' entries (1-based global displacement "
    "component fixed by a Dirichlet BC, e.g. 'z_symm:3' for u_z = 0). On the listed "
    "slave nodes friction is restricted out of the fixed direction(s) (no tangential "
    "traction across the symmetry plane). Must mirror the actual Dirichlet BCs.",
    str,
    "",
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

        # ------------------------------------------------------------------
        # Coulomb friction (Gitterle, Popp, Gee & Wall 2010; Hueber & Wohlmuth
        # 2005; MOOSE ComputeFrictionalForceLMMechanicalContact). The tangential
        # traction is NOT a separate DOF here: in the vectorial formulation it is
        # the in-plane projection z_t[c] = t_c . z_I of the existing nodal
        # multiplier vector z_I (the frictionless tangential rows t_a . z_I = 0
        # are replaced by the Coulomb stick/slip law). Hence, unlike the
        # scalar-normal formulation, friction adds NO new DOFs and nMultipliers /
        # _nDof are unchanged.
        self.friction_coefficient = float(
            kwargs.get("friction_coefficient", kwargs.get("frictioncoefficient", 0.0))
        )
        # Algorithmic complementarity parameters (no effect on the converged
        # solution; Gitterle 2010 p. 555/565, Alart-Curnier 1991). c_n ~ c_t ~
        # O(E) is well-scaled (Hueber & Wohlmuth 2005; Farah 2018 Sec. 3.5.2).
        self.c_t = float(kwargs.get("friction_ct", kwargs.get("frictionct", 1.0)))
        self.c_n = float(kwargs.get("friction_cn", kwargs.get("frictioncn", 1.0e6)))
        # PDASS friction-activation threshold on the RAW normal pressure (MOOSE
        # "contact_pressure < epsilon", default 1e-7): friction engages only once
        # p_n has built up -> no contact-initiation shock.
        self.friction_epsilon = float(
            kwargs.get("friction_epsilon", kwargs.get("frictionepsilon", 1.0e-7))
        )
        # Optional mu ramp-up (continuation) over the first `friction_ramp`
        # fraction of the step progress; frozen within the increment.
        self.friction_ramp = float(kwargs.get("friction_ramp", kwargs.get("frictionramp", 0.0)))

        # (dim - 1) in-plane tangential components per active slave node (only
        # meaningful with friction). These index the tangential rows z0+1..z0+dim-1
        # of the existing vector multiplier; no DOFs are allocated here.
        self.nTangentialComponents = (self.dim - 1) if self.friction_coefficient > 0.0 else 0
        # Recovered in-plane tangential tractions z_t (frozen tangent basis) for
        # output/verification, and the stick/slip classification per slave node.
        self.recovered_tractions_t = np.zeros((self.nNonMortarNodes, self.nTangentialComponents))
        self.stick_set = np.zeros(self.nNonMortarNodes, dtype=bool)

        # Condensation of the tangential (friction) multipliers (Gitterle Eq. 86,
        # rows St/Sl) is handled generically in condenseMortarMultipliers: the
        # assembled friction constraint-row Jacobian (dC_t/dz, dC_t/dd) is folded
        # by z = W d - wf (Eq. 85) and the folded rows are rotated into the slave
        # displacement DOFs by the local frame Q = [n_I, t_1, .., t_{dim-1}]. No
        # extra guard needed.

        # Single-owner treatment of shared slave nodes (friction only): when
        # decomposed contact surfaces meet at an edge they share slave nodes;
        # with friction the owner's (normal + dim-1 tangential) constraints
        # already tie the node completely, so any further contact constraint from
        # another surface is redundant and makes the saddle system singular.
        # Once friction is active each shared slave node is owned entirely by the
        # first surface (input order); others skip it (z_I driven to 0). See
        # Gitterle 2010; recorded in a model-level registry.
        self._owns_node = np.ones(self.nNonMortarNodes, dtype=bool)
        if self.friction_coefficient > 0.0:
            owners = getattr(model, "_mortarFrictionOwners", None)
            if owners is None:
                owners = {}
                model._mortarFrictionOwners = owners
            for I, node in enumerate(self.non_mortar_nodes):
                if owners.setdefault(node, self._name) != self._name:
                    self._owns_node[I] = False

        # Symmetry / Dirichlet-aware friction restriction. On a slave node whose
        # displacement is prescribed along a global direction e_k (symmetry
        # plane u_k = 0), Coulomb friction must not generate a tangential
        # traction along e_k (by symmetry it is zero; enforcing it fights the
        # Dirichlet BC and injects a spurious force into the brittle bulk). We
        # record here which global displacement directions are fixed per slave
        # node; compute_tangent_basis geometrically removes them from the tangent
        # frame. Opt-in, self-contained: frictionSymmetryBCs="<nodeSet>:<comp>[,..]"
        # (1-based global component, must mirror the actual Dirichlet BCs). Empty
        # -> no restriction (default) so the frictionless path is unchanged.
        self._node_fixed_dirs = [set() for _ in range(self.nNonMortarNodes)]
        sym_spec = kwargs.get("frictionSymmetryBCs", kwargs.get("frictionsymmetrybcs", ""))
        if self.friction_coefficient > 0.0 and sym_spec:
            node_sets = getattr(model, "nodeSets", {})
            for pair in str(sym_spec).split(","):
                pair = pair.strip()
                if not pair:
                    continue
                try:
                    set_name, comp_str = pair.split(":")
                    set_name = set_name.strip()
                    comp = int(comp_str) - 1  # 1-based input -> 0-based direction
                except ValueError:
                    raise ValueError(
                        f"frictionSymmetryBCs entry '{pair}' must be '<nodeSet>:<component>' "
                        f"(e.g. 'z_symm:3')."
                    )
                if not (0 <= comp < model.domainSize):
                    raise ValueError(
                        f"frictionSymmetryBCs component in '{pair}' out of range 1..{model.domainSize}."
                    )
                if set_name not in node_sets:
                    raise KeyError(f"frictionSymmetryBCs node set '{set_name}' not found in model.")
                fixed_nodes = set(node_sets[set_name])
                for I, node in enumerate(self.non_mortar_nodes):
                    if node in fixed_nodes:
                        self._node_fixed_dirs[I].add(comp)
        # Number of tangential friction components actually enforced per slave
        # node (<= nTangentialComponents; reduced on symmetry nodes). Recomputed
        # each increment in compute_tangent_basis; default = all components.
        self._n_free_tangents = np.full(self.nNonMortarNodes, self.nTangentialComponents, dtype=int)

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
        # PDASS termination + anti-cycling state used by the friction path
        # (Hueber & Wohlmuth 2005; Bland 1977): the semi-smooth iteration is
        # frozen for the rest of the increment as soon as its discrete state
        # (normal active set + friction stick set) REPEATS a state seen this
        # increment. Reset per increment together with current_iteration.
        self._seen_states = set()
        self._set_frozen = False

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

    def compute_tangent_basis(self, normals: np.ndarray) -> np.ndarray:
        """Return an orthonormal in-plane tangent basis per slave node, shape
        (nNonMortarNodes, nTangentialComponents, dim), FROZEN per increment and
        used by the Coulomb friction rows (z_t[c] = t_c . z_I). In 3D the two
        vectors t1, t2 span the plane perpendicular to the nodal normal
        (get_tangent_basis); in 2D the single vector t = [-n_y, n_x]. The basis is
        arbitrary but consistent per node, sufficient because the Coulomb cone
        |z_t| <= mu*p_n and the slip increment are expressed in the SAME frame, so
        |z_t|, |u_t| are frame-invariant. Only used when friction_coefficient > 0.
        """
        dim = self.model.domainSize
        basis = np.zeros((self.nNonMortarNodes, self.nTangentialComponents, dim))
        if self.nTangentialComponents == 0:
            return basis

        # A slave node whose area-weighted facet normals cancel gets a zero normal
        # from compute_normals(); it has no well-defined contact frame. Leave its
        # tangent basis at zero (friction is skipped for it in applyConstraint) to
        # avoid the 0/0 = NaN get_tangent_basis would produce (fatal with friction:
        # it propagates NaN tractions into the bulk).
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

        # Restrict the friction tangent space on symmetry / Dirichlet-fixed nodes
        # (see __init__): reorder so friction-free directions come first and record
        # how many tangential components carry friction on each node.
        self._n_free_tangents = np.full(self.nNonMortarNodes, self.nTangentialComponents, dtype=int)
        if any(self._node_fixed_dirs):
            self._restrict_tangents_to_free_space(basis, normals)
        return basis

    def _restrict_tangents_to_free_space(self, basis: np.ndarray, normals: np.ndarray):
        """Reorder each slave node's tangent basis so the friction-FREE directions
        (within the contact tangent plane T AND orthogonal to every Dirichlet-fixed
        global displacement direction at that node) come first, followed by the
        FIXED directions, and set ``self._n_free_tangents`` to the number of free
        directions. Fixed components carry no friction (suppressed in
        applyConstraint). General: the fixed global axis is projected into the
        tangent plane, so this works for arbitrarily oriented surfaces and reduces
        to a plain component drop when a tangent is already axis-aligned. A fixed
        axis parallel to the contact normal has no in-plane part and leaves the
        node unrestricted (that direction is carried by the normal contact)."""
        dim = self.model.domainSize
        ntc = self.nTangentialComponents
        tol = 1e-8
        for I in range(self.nNonMortarNodes):
            fixed = self._node_fixed_dirs[I]
            if not fixed:
                continue
            n_I = normals[I]
            if np.dot(n_I, n_I) < 1e-24:
                continue  # degenerate normal: friction already skipped for this node
            # Orthonormal basis of the in-plane (tangent) part of the fixed axes.
            fix_basis = []
            for k in sorted(fixed):
                e = np.zeros(dim)
                e[k] = 1.0
                p = e - (e @ n_I) * n_I  # project fixed axis onto tangent plane
                for q in fix_basis:
                    p = p - (p @ q) * q
                nrm = np.linalg.norm(p)
                if nrm > tol:
                    fix_basis.append(p / nrm)
            if not fix_basis:
                continue  # fixed axis parallel to normal -> nothing tangential to drop
            # Free directions = tangent plane with the fixed in-plane subspace removed.
            free_basis = []
            for t in basis[I]:
                v = t.copy()
                for q in fix_basis + free_basis:
                    v = v - (v @ q) * q
                nrm = np.linalg.norm(v)
                if nrm > tol:
                    free_basis.append(v / nrm)
            new_basis = free_basis + fix_basis  # exactly ntc orthonormal vectors, free first
            for c in range(min(ntc, len(new_basis))):
                basis[I, c] = new_basis[c]
            self._n_free_tangents[I] = len(free_basis)

    def _assemble_friction_tangential_rows(
        self, I, z0, z_slice, z_I, n_I, t_basis, nzD, nzC, D, C, sf, dim, nSlave,
        inv_D, sgn_D, p_n, g_sep, mu, du_slave, du_master, PExt, K,
    ):
        """Coulomb friction stick/slip constraint rows for the active slave node I,
        expressed in the VECTORIAL multiplier: the in-plane tangential traction is
        the projection ``z_t[c] = t_c . z_I`` (NOT a separate DOF), so these rows
        replace the frictionless rows ``t_a . z_I = 0`` on z0+1..z0+dim-1.

        Semi-smooth PDASS of Gitterle, Popp, Gee & Wall (2010), Eqs. (47) slip
        increment, (58) trial traction, (60) augmented bound, (61)/(72)-(74)
        unified stick/slip NCP; identical formulation to MOOSE
        ComputeFrictionalForceLMMechanicalContact. c_t, c_n are purely algorithmic
        (no effect on the converged solution; Gitterle p. 555/565, Alart-Curnier
        1991). Frozen geometry (n, t, D, C) => K is the exact Jacobian of this
        algebraic residual; verified by the FD tangent check in test9_friction.
        Sign convention: K[row, col] = -d(PExt[row])/d(col), so K += d(C_t)/d(.).

        The tangential traction FORCE on the bodies is already assembled from the
        full vector z_I in the equilibrium coupling (+D z_I / -C z_I) of the
        caller, so it is NOT re-added here - only the constraint rows are set.
        """
        ntc = self.nTangentialComponents
        nf = int(self._n_free_tangents[I])
        c_t = self.c_t

        # Tangential traction components in the frozen frame: z_t[c] = t_c . z_I.
        z_t = t_basis @ z_I  # (ntc,)

        # Augmented Coulomb bound b = mu*max(0, p_n + c_n*inv_D*g_sep) (Gitterle
        # Eq. 60; MOOSE). b_arg SHRINKS under a penetrating transient (removes the
        # contact-initiation shock); at convergence g_sep -> 0 so b -> mu*p_n.
        b_arg = p_n + self.c_n * inv_D * g_sep
        b = mu * max(0.0, b_arg)
        has_tangent = np.dot(t_basis[0], t_basis[0]) > 0.5

        # PDASS friction gate (MOOSE "contact_pressure < epsilon"): friction acts
        # only once a genuine normal pressure has built up AND the bound is
        # positive. At initiation p_n ~ 0 -> enforce zero tangential traction
        # (z_t = 0), i.e. the frictionless rows, avoiding an initiation shock.
        if not (p_n > self.friction_epsilon and b > 0.0 and has_tangent):
            self.recovered_tractions_t[I] = 0.0
            self.stick_set[I] = False
            for c in range(ntc):
                r_t = z0 + 1 + c
                PExt[r_t] -= float(t_basis[c] @ z_I)
                K[r_t, z_slice] += t_basis[c]
            return

        # Derivatives of b. p_n = -lambda_I*sgn_D with lambda_I = n_I . z_I, so
        #   d(b)/d(z_I) = db_dlam * n_I ,  db_dlam = -mu*sgn_D ;
        # and via g_sep = sgn_D*g_weak (g_weak depends on displacements through
        # -D[I,K] n / +C[I,J] n):  d(b)/d(d_sK) = db_disp_fac*D[I,K]*n_I,
        #   d(b)/d(d_mJ) = -db_disp_fac*C[I,J]*n_I.
        db_dlam = -mu * sgn_D
        db_disp_fac = -mu * self.c_n * inv_D * sgn_D

        # Weighted, D_II-normalized tangential slip increment (Gitterle Eq. 47).
        w_vec = np.zeros(dim)
        if len(nzD):
            w_vec += D[I, nzD] @ du_slave[nzD]
        if len(nzC):
            w_vec -= C[I, nzC] @ du_master[nzC]
        u_t = inv_D * (t_basis @ w_vec)  # (ntc,)

        z_tr = z_t + c_t * u_t                       # trial traction (Eq. 58)
        z_tr_norm = float(np.linalg.norm(z_tr[:nf]))  # Coulomb cone over FREE comps

        # stick / slip = generalized derivative of the max in the unified NCP
        # (Gitterle Eqs. 72/73), re-evaluated every iteration until the set is
        # frozen (same freeze signal as the normal active set).
        if self.active_set_frozen:
            slip = not self.stick_set[I]
        else:
            slip = z_tr_norm > b
            self.stick_set[I] = not slip

        z_t_rec = z_t.copy()
        z_t_rec[nf:] = 0.0
        self.recovered_tractions_t[I] = z_t_rec

        if slip:
            # SLIP (max = ||z_tr||), Gitterle Eq. 61 (un-normalized form: its
            # tangent has no bare 1/||z_tr|| term, so it stays bounded at slip
            # onset; see Gitterle p. 554):  C_t = ||z_tr|| z_t - b z_tr.
            dir_c = z_tr[:nf] / z_tr_norm      # (nf,) unit trial direction
            dir_spatial = dir_c @ t_basis[:nf]  # (dim,)
            C_t = z_tr_norm * z_t - b * z_tr    # (ntc,); only [:nf] used
            for c in range(nf):
                r_t = z0 + 1 + c
                PExt[r_t] -= C_t[c]
                # d(C_t[c])/d(z_I): via z_t[cp] = t_cp . z_I  and  b(lambda_I).
                dC_dzI = np.zeros(dim)
                for cp in range(nf):
                    dCt_dzt = dir_c[cp] * z_t[c] + ((z_tr_norm - b) if cp == c else 0.0)
                    dC_dzI += dCt_dzt * t_basis[cp]
                dC_dzI += (-db_dlam * z_tr[c]) * n_I
                K[r_t, z_slice] += dC_dzI
                # d(C_t[c])/d(d): u_t via z_tr (coeff_vec) + b via g_sep.
                coeff_vec = z_t[c] * dir_spatial - b * t_basis[c]  # (dim,)
                for K_nd in nzD:
                    s_dofs = slice(sf * K_nd, sf * K_nd + dim)
                    K[r_t, s_dofs] += c_t * inv_D * D[I, K_nd] * coeff_vec
                    K[r_t, s_dofs] += -z_tr[c] * db_disp_fac * D[I, K_nd] * n_I
                for J in nzC:
                    m_global = nSlave + J
                    m_dofs = slice(sf * m_global, sf * m_global + dim)
                    K[r_t, m_dofs] += -c_t * inv_D * C[I, J] * coeff_vec
                    K[r_t, m_dofs] += z_tr[c] * db_disp_fac * C[I, J] * n_I
        else:
            # STICK (max = b), Gitterle Eq. 74:  C_t = b z_t - b z_tr = -b c_t u_t
            # -> u_t = 0. z_t cancels (no z_I self-coupling); only b(lambda_I) and
            # u_t(displacement) remain.
            C_t = b * z_t - b * z_tr  # (ntc,) == -b c_t u_t; only [:nf] used
            for c in range(nf):
                r_t = z0 + 1 + c
                PExt[r_t] -= C_t[c]
                K[r_t, z_slice] += (-db_dlam * c_t * u_t[c]) * n_I
                for K_nd in nzD:
                    s_dofs = slice(sf * K_nd, sf * K_nd + dim)
                    K[r_t, s_dofs] += -b * c_t * inv_D * D[I, K_nd] * t_basis[c]
                    K[r_t, s_dofs] += -c_t * u_t[c] * db_disp_fac * D[I, K_nd] * n_I
                for J in nzC:
                    m_global = nSlave + J
                    m_dofs = slice(sf * m_global, sf * m_global + dim)
                    K[r_t, m_dofs] += b * c_t * inv_D * C[I, J] * t_basis[c]
                    K[r_t, m_dofs] += c_t * u_t[c] * db_disp_fac * C[I, J] * n_I

        # Suppress friction on FIXED (symmetry) tangential directions [nf, ntc):
        # enforce zero tangential traction t_c . z_I = 0 there (well-conditioned).
        for c in range(nf, ntc):
            r_t = z0 + 1 + c
            PExt[r_t] -= float(t_basis[c] @ z_I)
            K[r_t, z_slice] += t_basis[c]

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
            # Frozen per-node tangent basis for Coulomb friction (all-zero / unused
            # when mu = 0). Frozen together with the normals and mortar matrices so
            # the assembled tangent stiffness is the exact Jacobian of the frozen-
            # geometry residual (Gitterle et al. 2010). Also (re)computes
            # self._n_free_tangents for symmetry-restricted nodes.
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

        # Coulomb friction coefficient with optional ramp-up (continuation),
        # frozen within the increment (stepProgress is constant here).
        mu = self.friction_coefficient
        if self.friction_ramp > 0.0 and mu > 0.0:
            prog = max(0.0, float(timeStep.stepProgress))
            mu = mu * min(1.0, prog / self.friction_ramp)
        if mu > 0.0:
            # Displacement INCREMENT (dU since the start of the increment) for the
            # weighted tangential slip u_t; slaves first, then masters.
            du_disp = dU[: sf * nNodes].reshape(nNodes, sf)[:, :dim]
            du_slave = du_disp[:nSlave]
            du_master = du_disp[nSlave:]

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
            sgn_D = np.sign(self.current_D_rowsum[I])
            D_II = self.current_D_rowsum[I]
            inv_D = 1.0 / D_II if abs(D_II) > 1e-30 else 0.0
            p_n = -lambda_I * sgn_D   # physical normal pressure (>= 0 in contact)
            g_sep = g_I_weak * sgn_D  # separation gap (>0 open, <0 penetrating)

            if self.use_active_set and not self.active_set_frozen:
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

                # --- Constraint rows z0+1..z0+dim-1: tangential ---
                if mu > 0.0:
                    # Coulomb friction (stick/slip) in the vectorial frame:
                    # z_t[c] = t_c . z_I. Replaces the frictionless rows.
                    self._assemble_friction_tangential_rows(
                        I, z0, z_slice, z_I, n_I, self.current_tangents[I],
                        nzD, nzC, D, C, sf, dim, nSlave,
                        inv_D, sgn_D, p_n, g_sep, mu, du_slave, du_master, PExt, K,
                    )
                else:
                    # Frictionless: zero tangential traction t_a . z_I = 0.
                    for a, t_a in enumerate(self._local_frame(n_I)):
                        r_t = z0 + 1 + a
                        PExt[r_t] -= float(t_a @ z_I)
                        K[r_t, z_slice] += t_a
            else:
                # Inactive: z_I = 0 (all components -> normal and tangential)
                for c in range(dim):
                    PExt[z0 + c] -= z_I[c]
                    K[z0 + c, z0 + c] += 1.0
                if mu > 0.0:
                    self.recovered_tractions_t[I] = 0.0
                    self.stick_set[I] = False

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
        if not hasattr(self, "current_normals"):
            # applyConstraint has not run yet this analysis; nothing to condense.
            return None

        dim = self.dim
        friction_on = self.friction_coefficient > 0.0
        transform = self._build_condensation_transform()
        multipliers = []
        for I in range(self.nNonMortarNodes):
            # Tangent frame MUST be the SAME one the tangential constraint rows
            # (z-dof 1..dim-1) were assembled with in applyConstraint, so the
            # condensation's local-frame rotation Q = [n_I, t_1, .., t_{dim-1}]
            # matches the assembled rows: the frozen, symmetry-restricted
            # current_tangents when friction is active, else _local_frame.
            if friction_on:
                tangents = [self.current_tangents[I][c] for c in range(self.nTangentialComponents)]
            else:
                tangents = self._local_frame(self.current_normals[I])
            multipliers.append(
                {
                    "scalarVariables": [self.scalarVariables[dim * I + c] for c in range(dim)],
                    "slaveNode": self.non_mortar_nodes[I],
                    "localIndex": I,
                    "normal": self.current_normals[I],
                    "tangents": tangents,
                    "D_diag": float(self.current_D_rowsum[I]),
                    "transform": transform[I],
                    "active": bool(self.active_set[I]),
                }
            )
        return {"field": self.field, "dim": dim, "multipliers": multipliers}
