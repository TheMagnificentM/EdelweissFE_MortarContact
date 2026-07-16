#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test 3B: Assembly and Verification of Mortar Coupling Matrices D and C
=======================================================================

This script sets up a multi-element quad mesh setup, instantiates the Mortar
constraint, calculates the global D and C matrices via polygon projection,
clipping, sub-triangulation, and Gauss quadrature, and prints the result properties:
1. Matrix dimensions check.
2. D matrix diagonal dominance / properties.
3. Row sums conservation (sum of D_row matches sum of C_row for flat patches).
"""

import sys
import os
import numpy as np

# Add local path of EdelweissFE to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from edelweissfe.models.femodel import FEModel
from edelweissfe.generators.boxgen import generateModelData as generateBoxMesh
from edelweissfe.journal.journal import Journal
from edelweissfe.config.elementlibrary import getElementClass
from edelweissfe.constraints.mortarcontact import Constraint as MortarContact

def get_local_nodes(el, faceID):
    elType = el.elType.upper()
    if "C3D8" in elType:
        if faceID == 1: return [el.nodes[i] for i in [0, 1, 5, 4]]
        elif faceID == 2: return [el.nodes[i] for i in [1, 2, 6, 5]]
        elif faceID == 3: return [el.nodes[i] for i in [2, 3, 7, 6]]
        elif faceID == 4: return [el.nodes[i] for i in [3, 0, 4, 7]]
        elif faceID == 5: return [el.nodes[i] for i in [0, 3, 2, 1]]
        elif faceID == 6: return [el.nodes[i] for i in [4, 5, 6, 7]]
    return el.nodes

def run_coupling_assembly_test():
    print("================ RUNNING MORTAR COUPLING ASSEMBLY TEST ================")
    
    # 1. Create Model with dimension=3
    model = FEModel(dimension=3)
    
    # Define Nodes directly for two solid blocks
    # Block 1 (Slave): from [0, 1]x[0, 1]x[0, 1]
    # Block 2 (Master): from [0.5, 1.5]x[0.2, 1.2]x[1.0, 2.0]
    slave_coords = [
        [0.0, 0.0, 0.0], # 1
        [1.0, 0.0, 0.0], # 2
        [1.0, 1.0, 0.0], # 3
        [0.0, 1.0, 0.0], # 4
        [0.0, 0.0, 1.0], # 5
        [1.0, 0.0, 1.0], # 6
        [1.0, 1.0, 1.0], # 7
        [0.0, 1.0, 1.0], # 8
    ]
    master_coords = [
        [0.5, 0.2, 1.0], # 9
        [1.5, 0.2, 1.0], # 10
        [1.5, 1.2, 1.0], # 11
        [0.5, 1.2, 1.0], # 12
        [0.5, 0.2, 2.0], # 13
        [1.5, 0.2, 2.0], # 14
        [1.5, 1.2, 2.0], # 15
        [0.5, 1.2, 2.0], # 16
    ]
    
    from edelweissfe.points.node import Node
    
    # Add nodes to model
    for i, pt in enumerate(slave_coords):
        n_id = i + 1
        model.nodes[n_id] = Node(n_id, np.array(pt))
        
    for i, pt in enumerate(master_coords):
        n_id = i + 9
        model.nodes[n_id] = Node(n_id, np.array(pt))
        
    # Instantiate Solid Hex elements (C3D8)
    HexClass = getElementClass("C3D8", "edelweiss")
    
    s_el = HexClass("C3D8", 1)
    s_el.setNodes([model.nodes[i+1] for i in range(8)])
    model.elements[1] = s_el
    
    m_el = HexClass("C3D8", 2)
    m_el.setNodes([model.nodes[i+9] for i in range(8)])
    model.elements[2] = m_el
    
    # Create surface boundary contact elements (CONQUAD4)
    # Slave: Top face of Block 1 (nodes 5, 6, 7, 8)
    # Master: Bottom face of Block 2 (nodes 9, 10, 11, 12)
    ConQuadClass = getElementClass("CONQUAD4", "edelweiss")
    
    s_con = ConQuadClass("CONQUAD4", 3)
    s_con.setNodes([model.nodes[5], model.nodes[6], model.nodes[7], model.nodes[8]])
    model.elements[3] = s_con
    
    m_con = ConQuadClass("CONQUAD4", 4)
    m_con.setNodes([model.nodes[9], model.nodes[10], model.nodes[11], model.nodes[12]])
    model.elements[4] = m_con
    
    # Setup surfaces
    model.surfaces = {
        "slave_surf": {1: [s_con]},
        "master_surf": {1: [m_con]}
    }
    
    # Initialize displacement fields on all nodes
    from edelweissfe.variables.fieldvariable import FieldVariable
    for node in model.nodes.values():
        node.fields["displacement"] = FieldVariable(node, "displacement")
        
    # 2. Instantiate Mortar constraint
    constraint = MortarContact(
        "contact_constraint",
        model,
        nonMortarSurface="slave_surf",
        mortarSurface="master_surf",
        field="displacement"
    )
    model.constraints["contact_constraint"] = constraint
    
    # 3. Assemble coupling matrices D and C
    D, C = constraint.compute_mortar_coupling_matrices()
    
    # Write output to test results log
    output_dir = "testfiles/mortar_tests/test3_mortar_intersection"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "coupling_assembly_results.txt")
    
    with open(out_path, 'w') as f:
        f.write("================ MORTAR COUPLING ASSEMBLY RESULTS ================\n")
        f.write(f"Slave nodes in contact boundary: {[nd.label for nd in constraint.non_mortar_nodes]}\n")
        f.write(f"Master nodes in contact boundary: {[nd.label for nd in constraint.mortar_nodes]}\n\n")
        
        f.write(f"Matrix D (Slave-Slave) size: {D.shape}\n")
        f.write("D Matrix Content:\n")
        f.write(np.array2string(D, precision=6, suppress_small=True) + "\n\n")
        
        f.write(f"Matrix C (Slave-Master) size: {C.shape}\n")
        f.write("C Matrix Content:\n")
        f.write(np.array2string(C, precision=6, suppress_small=True) + "\n\n")
        
        # Verify row sum balance (exact overlap patch area conservation)
        f.write("Row-wise sum conservation check:\n")
        for i in range(len(D)):
            sum_d = np.sum(D[i])
            sum_c = np.sum(C[i])
            f.write(f"  Row {i+1}: Sum(D) = {sum_d:.6f}, Sum(C) = {sum_c:.6f}, Diff = {abs(sum_d - sum_c):.6e}\n")
            
    print(f"Saved coupling assembly verification logs to: {out_path}")

if __name__ == '__main__':
    run_coupling_assembly_test()
