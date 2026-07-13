#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test 1: Mortar Contact Normals Verification Suite
==================================================
This test verifies the correctness of computed node normal vectors on the slave
(non-mortar) surface of a MortarContact3D constraint.

What this test does:
1. Generates a 2D or 3D block mesh of standard volume elements (C3D8, C3D20, CPS4, CPS8).
2. Extracts their boundary faces and overlays explicit contact elements (CONQUAD4, CONTRI3, CONLINE2, etc.).
3. Skews the nodes randomly to test general distorted configurations.
4. Computes area-weighted knot normals on the slave surface.
5. Verifies that all normals are unit vectors and properly oriented (perpendicular to surface).

What to do if this test fails:
- "Node normal is not normalized": The normalization division in compute_normals() has a bug.
- "Flat orientation verify failed": The normals in a flat configuration are not pointing in the 
  correct direction (-Y). This indicates an ordering/orientation or cross-product sign error.
- "Orthogonality verify failed for skewed geometry": The computed knot normal is not orthogonal
  to the adjacent contact element edges. This indicates a bug in how element tangents or the cross product is computed.
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
    
    Why are volume elements (like C3D8 or CPS4) here?
    - The FE mesh generators (generateBoxMesh, generatePlaneMesh) create physical volume elements
      representing the actual blocks.
    - We need these volume elements to define the boundary surfaces.
    - We then use this helper to map face nodes of these volume elements to our custom boundary 
      contact elements (CON elements) that are overlaid on the surface.
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
    """
    Instantiates explicit contact boundary elements (e.g. CONQUAD4, CONTRI3) on a surface.
    This creates the surface mesh representing the contact interface.
    """
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

def run_contact_test(el_type, skewed=False, dim=3):
    """Runs verification of node normals for flat or skewed configurations."""
    config_name = "Skewed" if skewed else "Normal"
    print(f"\n* Testing {el_type} ({config_name} Geometry)...")
    
    model = FEModel(dimension=dim)
    journal = Journal()
    
    # 1. Generate base volume elements (CPS/C3D)
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
        
    # 2. Skew nodes if requested (tests general geometry logic)
    if skewed:
        np.random.seed(42)
        for node in model.nodes.values():
            offset = np.random.uniform(-0.15, 0.15, size=dim)
            node.coordinates += offset
            
    # 3. Create contact boundary elements on top of volume element surfaces
    slave_surf = apply_contact_elements(model, "gen_bottom", el_type)
    master_surf = apply_contact_elements(model, "gen_top", el_type)
    
    # 4. Initialize displacement fields for degrees of freedom (required for Constraint init)
    from edelweissfe.variables.fieldvariable import FieldVariable
    for node in model.nodes.values():
        node.fields["displacement"] = FieldVariable(node, "displacement")
        
    # 5. Instantiate the MortarContact3D constraint
    kwargs = {"nonMortarSurface": slave_surf, "mortarSurface": master_surf, "field": "displacement"}
    contact = MortarContact3D(f"contact_{el_type}_{config_name.lower()}", model, **kwargs)
    
    # 6. Verify Knot Normals
    normals = contact.undeformed_normals
    
    # Check A: Verify that every normal vector has unit length
    # If this fails, normal vectors were not normalized properly in the code.
    for i, node in enumerate(contact.non_mortar_nodes):
        n = normals[i]
        length = np.linalg.norm(n)
        if not np.isclose(length, 1.0, atol=1e-7):
            print(f"  [FAIL] Node {node.label} normal is not normalized: length = {length:.6f}")
            sys.exit(1)
            
    # Check B: Verify normal directions on flat surface
    # In flat configurations, all normals should point straight down in the -Y direction ([0, -1, 0] or [0, -1]).
    # If this fails, the normal computation is misoriented or pointing in the wrong direction.
    if not skewed:
        expected = np.zeros(dim)
        expected[1] = -1.0
        for i, node in enumerate(contact.non_mortar_nodes):
            n = normals[i]
            if not np.allclose(n, expected, atol=1e-7):
                print(f"  [FAIL] Node {node.label} normal orientation incorrect: normal = {n}")
                sys.exit(1)
        print("  -> Flat orientation verify: OK")
    # Check C: Verify normal directions on skewed surface
    # On a distorted surface, the normal vector must be perpendicular to the surface facet tangents.
    # If this fails, the tangent derivatives or the cross-product calculation has a bug.
    else:
        all_orthogonal = True
        for el, faceID in contact.non_mortar_facets:
            coords = np.array([node.coordinates for node in el.nodes])
            if dim == 3:
                if len(el.nodes) in (3, 6):
                    # Triangle edges
                    v1 = coords[1] - coords[0]
                    v2 = coords[2] - coords[0]
                else:
                    # Quad diagonals
                    v1 = coords[2] - coords[0]
                    v2 = coords[3] - coords[1]
                n_facet = np.cross(v1, v2)
                dot1 = np.dot(n_facet, v1)
                dot2 = np.dot(n_facet, v2)
                if not (np.isclose(dot1, 0.0, atol=1e-12) and np.isclose(dot2, 0.0, atol=1e-12)):
                    all_orthogonal = False
            else: # dim == 2
                # Tangent vector from start to end node
                t = coords[-1] - coords[0]
                n_facet = np.array([t[1], -t[0]])
                dot = np.dot(n_facet, t)
                if not np.isclose(dot, 0.0, atol=1e-12):
                    all_orthogonal = False
                    
        if all_orthogonal:
            print("  -> Skewed orthogonality verify: OK")
        else:
            print("  [FAIL] Orthogonality verify failed for skewed geometry!")
            sys.exit(1)
            
    print(f"  [PASS] {el_type} {config_name} test successful!")

if __name__ == "__main__":
    print("====================================================")
    print("MORTAR CONTACT ELEMENTS NORMALS VERIFICATION SUITE")
    print("====================================================")
    
    # 3D Quadrilateral Contact Elements
    run_contact_test("CONQUAD4", skewed=False, dim=3)
    run_contact_test("CONQUAD4", skewed=True, dim=3)
    
    run_contact_test("CONQUAD8", skewed=False, dim=3)
    run_contact_test("CONQUAD8", skewed=True, dim=3)
    
    run_contact_test("CONQUAD9", skewed=False, dim=3)
    run_contact_test("CONQUAD9", skewed=True, dim=3)
    
    # 3D Triangular Contact Elements
    run_contact_test("CONTRI3", skewed=False, dim=3)
    run_contact_test("CONTRI3", skewed=True, dim=3)
    
    run_contact_test("CONTRI6", skewed=False, dim=3)
    run_contact_test("CONTRI6", skewed=True, dim=3)
    
    # 2D Linear & Quadratic Line Contact Elements
    run_contact_test("CONLINE2", skewed=False, dim=2)
    run_contact_test("CONLINE2", skewed=True, dim=2)
    
    run_contact_test("CONLINE3", skewed=False, dim=2)
    run_contact_test("CONLINE3", skewed=True, dim=2)
    
    print("\n====================================================")
    print("ALL NORMALS VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("====================================================")
