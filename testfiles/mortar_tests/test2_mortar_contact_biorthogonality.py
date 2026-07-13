#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test 2: Mortar Contact Biorthogonality Verification Suite
=========================================================
This test verifies the correctness of the dual basis shape functions computed
on the slave boundary contact elements (CON elements) of a MortarContact3D constraint.

What this test does:
1. Generates a base 2D/3D mesh using volume elements (C3D8, C3D20, CPS4, CPS8).
2. Extracts their boundary faces and overlays explicit contact elements (CONQUAD4, CONTRI3, CONLINE2, etc.).
3. Optionally distorts/skews the nodes in space to test general distorted configurations.
4. Initializes degrees of freedom and the MortarContact3D constraint.
5. Computes local mass matrices M_e, diagonal matrices D_e, and dual transformations A_e.
6. Performs three validation checks (A, B, C) on each element to verify biorthogonality.

What to do if this test fails:
- Check A fails ("D_e is not diagonal"):
  The integration loop for D_e in computeLocalMassMatrices() has a bug, or the shape function N_i
  values are incorrect, placing non-zero weights off the diagonal.
- Check B fails ("B_e = A_e @ M_e is not equal to D_e"):
  The matrix inversion of M_e failed (singular matrix or numerical instability) or the algebraic 
  definition of A_e is wrong.
- Check C fails ("Explicit quadrature biorthogonality failed"):
  This is the most critical check. It means that evaluating standard and dual shape functions 
  point-by-point and integrating them numerically does not yield D_e. This indicates a bug in
  how shape functions (N_i), dual functions (M_bar_i), or Jacobian mapping (getJacobianAndAreaWeight) 
  are evaluated at local coordinates.
"""

import sys
import os
import numpy as np

# Add EdelweissFE to python search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from edelweissfe.models.femodel import FEModel
from edelweissfe.generators.boxgen import generateModelData as generateBoxMesh
from edelweissfe.generators.planerectquad import generateModelData as generatePlaneMesh
from edelweissfe.journal.journal import Journal
from edelweissfe.config.elementlibrary import getElementClass
from edelweissfe.constraints.mortarcontact3d import Constraint as MortarContact3D
from edelweissfe.points.node import Node

def get_local_nodes(el, faceID):
    """
    Helper to extract boundary nodes of volume elements based on face ID.
    
    Why are volume elements (like C3D8, CPS4) here?
    - The mesh generators (generateBoxMesh, generatePlaneMesh) generate standard physical volume
      elements representing the solid bodies.
    - We use these volume elements to define the top/bottom boundary surfaces.
    - We map the face nodes of these volume elements using this helper to overlay explicit boundary
      contact elements (CON elements) on the contact interface.
    - The constraint itself only works on the CON elements, but we need this mapping to set up the mesh.
    """
    elType = el.elType.upper()
    if "C3D8" in elType:
        if faceID == 1: idx = [3, 2, 1, 0]
        elif faceID == 2: idx = [4, 5, 6, 7]
        elif faceID == 3: idx = [0, 1, 5, 4]
        elif faceID == 4: idx = [6, 5, 1, 2]
        elif faceID == 5: idx = [7, 6, 2, 3]
        elif faceID == 6: idx = [4, 7, 3, 0]
    elif "C3D20" in elType:
        if faceID == 1: idx = [3, 2, 1, 0, 10, 9, 8, 11]
        elif faceID == 2: idx = [4, 5, 6, 7, 12, 13, 14, 15]
        elif faceID == 3: idx = [0, 1, 5, 4, 8, 17, 12, 16]
        elif faceID == 4: idx = [6, 5, 1, 2, 13, 17, 9, 18]
        elif faceID == 5: idx = [7, 6, 2, 3, 14, 18, 10, 19]
        elif faceID == 6: idx = [4, 7, 3, 0, 15, 19, 11, 16]
    elif "CPS4" in elType or "CPE4" in elType:
        if faceID == 1: idx = [0, 1]
        elif faceID == 2: idx = [1, 2]
        elif faceID == 3: idx = [2, 3]
        elif faceID == 4: idx = [3, 0]
    elif "CPS8" in elType or "CPE8" in elType:
        if faceID == 1: idx = [0, 1, 4]
        elif faceID == 2: idx = [1, 2, 5]
        elif faceID == 3: idx = [2, 3, 6]
        elif faceID == 4: idx = [3, 0, 7]
    else:
        raise NotImplementedError(f"Local face mapping not defined for {elType}")
    return [el.nodes[i] for i in idx]

def apply_contact_elements(model, surface_name, contact_el_type):
    """Generates contact elements on a boundary surface."""
    ConClass = getElementClass(contact_el_type, "edelweiss")
    max_el_id = max(model.elements.keys()) if model.elements else 0
    max_node_id = max(model.nodes.keys()) if model.nodes else 0
    
    contact_elements = []
    for faceID, elements in model.surfaces[surface_name].items():
        for el in elements:
            facet_nodes = get_local_nodes(el, faceID)
            
            # Special case for CONQUAD9: we need a 9th center node
            if contact_el_type == "CONQUAD9" and len(facet_nodes) == 8:
                coords = np.array([node.coordinates for node in facet_nodes[:4]])
                center_coords = np.mean(coords, axis=0)
                max_node_id += 1
                center_node = Node(max_node_id, center_coords)
                model.nodes[max_node_id] = center_node
                facet_nodes.append(center_node)
                
            # Special case for CONTRI3: split 4-node quad face into two triangles
            if contact_el_type == "CONTRI3" and len(facet_nodes) == 4:
                max_el_id += 1
                con_el1 = ConClass(contact_el_type, max_el_id)
                con_el1.setNodes([facet_nodes[0], facet_nodes[1], facet_nodes[2]])
                model.elements[max_el_id] = con_el1
                contact_elements.append(con_el1)
                
                max_el_id += 1
                con_el2 = ConClass(contact_el_type, max_el_id)
                con_el2.setNodes([facet_nodes[0], facet_nodes[2], facet_nodes[3]])
                model.elements[max_el_id] = con_el2
                contact_elements.append(con_el2)
                continue
                
            # Special case for CONTRI6: split 8-node quadratic quad face into two quadratic triangles
            if contact_el_type == "CONTRI6" and len(facet_nodes) == 8:
                mid_coords = 0.5 * (facet_nodes[0].coordinates + facet_nodes[2].coordinates)
                max_node_id += 1
                mid_node = Node(max_node_id, mid_coords)
                model.nodes[max_node_id] = mid_node
                
                max_el_id += 1
                con_el1 = ConClass(contact_el_type, max_el_id)
                con_el1.setNodes([facet_nodes[0], facet_nodes[1], facet_nodes[2], facet_nodes[4], facet_nodes[5], mid_node])
                model.elements[max_el_id] = con_el1
                contact_elements.append(con_el1)
                
                max_el_id += 1
                con_el2 = ConClass(contact_el_type, max_el_id)
                con_el2.setNodes([facet_nodes[0], facet_nodes[2], facet_nodes[3], mid_node, facet_nodes[6], facet_nodes[7]])
                model.elements[max_el_id] = con_el2
                contact_elements.append(con_el2)
                continue

            max_el_id += 1
            con_el = ConClass(contact_el_type, max_el_id)
            con_el.setNodes(facet_nodes)
            model.elements[max_el_id] = con_el
            contact_elements.append(con_el)
            
    new_surface_name = "con_" + surface_name
    model.surfaces[new_surface_name] = {1: contact_elements}
    return new_surface_name

def run_biorthogonality_test(el_type, skewed=False, dim=3):
    """Verifies the biorthogonality condition for a given contact element type."""
    config_name = "Skewed" if skewed else "Normal"
    print(f"\n* Testing {el_type} ({config_name} Geometry)...")
    
    model = FEModel(dimension=dim)
    journal = Journal()
    
    # 1. Generate base volume elements (C3D or CPS)
    if dim == 3:
        vol_type = "C3D20" if ("8" in el_type or "9" in el_type or "6" in el_type) else "C3D8"
        generateBoxMesh({"name": "gen"}, model, journal, **{
            "nX": 2, "nY": 2, "nZ": 2,
            "lX": 2.0, "lY": 2.0, "lZ": 2.0,
            "elType": vol_type, "elProvider": "edelweiss"
        })
    else: # dim == 2
        vol_type = "CPS8" if "3" in el_type else "CPS4"
        generatePlaneMesh({"name": "gen"}, model, journal, **{
            "nX": 2, "nY": 2,
            "l": 2.0, "h": 2.0,
            "elType": vol_type, "elProvider": "edelweiss"
        })
        
    # 2. Skew base geometry if requested (distorts nodes in 3D to test general mapping)
    if skewed:
        np.random.seed(42)
        for node in model.nodes.values():
            offset = np.random.uniform(-0.15, 0.15, size=dim)
            node.coordinates += offset
            
    # 3. Overlay contact boundary elements (CON elements) on the mesh surfaces
    slave_surf = apply_contact_elements(model, "gen_bottom", el_type)
    master_surf = apply_contact_elements(model, "gen_top", el_type)
    
    # 4. Initialize displacement degrees of freedom (required for Constraint init)
    from edelweissfe.variables.fieldvariable import FieldVariable
    for node in model.nodes.values():
        node.fields["displacement"] = FieldVariable(node, "displacement")
        
    # 5. Instantiate the MortarContact3D constraint
    kwargs = {"nonMortarSurface": slave_surf, "mortarSurface": master_surf, "field": "displacement"}
    contact = MortarContact3D(f"contact_{el_type}_{config_name.lower()}", model, **kwargs)
    
    # 6. Compute dual basis matrices (M_e, D_e, A_e) for all slave elements
    dual_matrices = contact.compute_local_dual_matrices()
    
    # 7. Verify biorthogonality for each slave element
    for el, faceID in contact.non_mortar_facets:
        M_e, D_e, A_e = dual_matrices[el.elNumber]
        
        # Verify sizes are consistent
        n_nodes = el.nNodes
        assert M_e.shape == (n_nodes, n_nodes), f"M_e has wrong shape: {M_e.shape}"
        assert D_e.shape == (n_nodes, n_nodes), f"D_e has wrong shape: {D_e.shape}"
        assert A_e.shape == (n_nodes, n_nodes), f"A_e has wrong shape: {A_e.shape}"
        
        # Check A: Verify that D_e is strictly diagonal (non-zero only on the main diagonal)
        # If this fails, the local integration or shape function values N_i are wrong.
        D_diag = np.diag(np.diag(D_e))
        if not np.allclose(D_e, D_diag, atol=1e-14):
            print(f"  [FAIL] D_e is not diagonal for element {el.elNumber}!")
            sys.exit(1)
            
        # Check B: Verify that A_e @ M_e equals D_e on a matrix algebraic level
        # If this fails, the matrix inversion of M_e failed or A_e is defined incorrectly.
        B_e = A_e @ M_e
        if not np.allclose(B_e, D_e, atol=1e-12):
            print(f"  [FAIL] B_e = A_e @ M_e is not equal to D_e for element {el.elNumber}!")
            print(f"  Max diff: {np.max(np.abs(B_e - D_e))}")
            sys.exit(1)
            
        # Check C: Verify the actual physical biorthogonality by integrating point-by-point
        # We loop over the Gauss points, evaluate standard shape functions N(gp) and dual functions
        # M_bar(gp) = A_e @ N(gp) at each coordinate, and perform numerical quadrature.
        # If this fails, the shape functions N_i, dual shape functions, or Jacobian weights have a bug.
        points, weights = el.getQuadraturePoints()
        coords = np.array([node.coordinates for node in el.nodes])
        B_explicit = np.zeros((n_nodes, n_nodes))
        
        for local_coords, w in zip(points, weights):
            N = el.getShapeFunctions(local_coords)
            jac = el.getJacobianAndAreaWeight(local_coords, coords)
            dGamma = jac * w
            
            # Dual function value at this point: M_bar = A_e * N
            M_bar = A_e @ N
            B_explicit += np.outer(M_bar, N) * dGamma
            
        if not np.allclose(B_explicit, D_e, atol=1e-12):
            print(f"  [FAIL] Explicit quadrature biorthogonality failed for element {el.elNumber}!")
            print(f"  Max diff: {np.max(np.abs(B_explicit - D_e))}")
            sys.exit(1)
            
    print(f"  [PASS] {el_type} {config_name} biorthogonality verified successfully!")

if __name__ == "__main__":
    print("====================================================")
    print("MORTAR CONTACT BIORTHOGONALITY STEP 2 TEST SUITE")
    print("====================================================")
    
    # 3D Quadrilateral Contact Elements
    run_biorthogonality_test("CONQUAD4", skewed=False, dim=3)
    run_biorthogonality_test("CONQUAD4", skewed=True, dim=3)
    
    run_biorthogonality_test("CONQUAD8", skewed=False, dim=3)
    run_biorthogonality_test("CONQUAD8", skewed=True, dim=3)
    
    run_biorthogonality_test("CONQUAD9", skewed=False, dim=3)
    run_biorthogonality_test("CONQUAD9", skewed=True, dim=3)
    
    # 3D Triangular Contact Elements
    run_biorthogonality_test("CONTRI3", skewed=False, dim=3)
    run_biorthogonality_test("CONTRI3", skewed=True, dim=3)
    
    run_biorthogonality_test("CONTRI6", skewed=False, dim=3)
    run_biorthogonality_test("CONTRI6", skewed=True, dim=3)
    
    # 2D Linear & Quadratic Line Contact Elements
    run_biorthogonality_test("CONLINE2", skewed=False, dim=2)
    run_biorthogonality_test("CONLINE2", skewed=True, dim=2)
    
    run_biorthogonality_test("CONLINE3", skewed=False, dim=2)
    run_biorthogonality_test("CONLINE3", skewed=True, dim=2)
    
    print("\n====================================================")
    print("ALL 14 BIORTHOGONALITY TESTS PASSED SUCCESSFULLY!")
    print("====================================================")
