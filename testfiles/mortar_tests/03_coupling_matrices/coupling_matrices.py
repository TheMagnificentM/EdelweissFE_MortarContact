#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test 3: Assembly and Verification of Mortar Coupling Matrices D and C
=====================================================================

This script sets up a multi-element quad mesh setup, instantiates the Mortar
constraint, calculates the global D and C matrices via polygon projection,
clipping, sub-triangulation, and Gauss quadrature, and verifies:
1. Matrix dimensions (D is n_slave x n_slave, C is n_slave x n_master).
2. Row sum conservation sum_K D_IK = sum_J C_IJ for every slave node.
   This is the momentum conservation / translational invariance identity
   (Farah 2018, Eq. (4.99)); it holds because D and C are integrated over the
   SAME clipped cells, so it must be satisfied even though the master facet
   only partially covers the slave facet in this setup.
3. sum(D) = sum(C) = the analytically known overlap area. The dual shape
   functions form a partition of unity, hence sum(D) integrates 1 over the
   overlap. Slave facet [0,1]x[0,1] against master facet [0.5,1.5]x[0.2,1.2]
   overlaps on [0.5,1]x[0.2,1] -> area 0.5 * 0.8 = 0.4.
4. Strictly positive nodal weights sum_K D_IK > 0 (integral positivity,
   Popp et al. 2012, Eq. (4.2)) - here for a linear CONQUAD4 slave facet.

The numeric matrices are additionally written to coupling_assembly_results.txt
for inspection; the checks above decide whether the test passes.
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

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------
    n_slave = constraint.nNonMortarNodes
    n_master = constraint.nMortarNodes
    # Slave facet [0,1]x[0,1] vs. master facet [0.5,1.5]x[0.2,1.2]
    # -> overlap [0.5,1]x[0.2,1] -> 0.5 * 0.8
    OVERLAP_AREA = 0.4

    ok = True

    if D.shape != (n_slave, n_slave):
        print(f"  [FAIL] D has shape {D.shape}, expected {(n_slave, n_slave)}")
        ok = False
    if C.shape != (n_slave, n_master):
        print(f"  [FAIL] C has shape {C.shape}, expected {(n_slave, n_master)}")
        ok = False

    rowsum_D = np.sum(D, axis=1)
    rowsum_C = np.sum(C, axis=1)

    # Row sum identity: momentum conservation / translational invariance.
    rowsum_err = float(np.max(np.abs(rowsum_D - rowsum_C)))
    print(f"  max|sum_K D_IK - sum_J C_IJ| = {rowsum_err:.3e}  (tolerance 1e-14)")
    if rowsum_err > 1e-14:
        print("  [FAIL] Row sum identity violated - D and C are not integrated over the same cells!")
        ok = False

    # Partition of unity of the dual basis: the total weight is the overlap area.
    for label, total in (("sum(D)", float(np.sum(D))), ("sum(C)", float(np.sum(C)))):
        err = abs(total - OVERLAP_AREA)
        print(f"  {label} = {total:.12f}, expected {OVERLAP_AREA:.12f} (diff {err:.3e})")
        if err > 1e-12:
            print(f"  [FAIL] {label} does not match the analytical overlap area!")
            ok = False

    # Integral positivity of the nodal weights (linear facet: no transformation needed).
    print(f"  min_I sum_K D_IK = {rowsum_D.min():.12f}")
    if np.any(rowsum_D <= 0.0):
        print(f"  [FAIL] Nodal dual weights not strictly positive: {rowsum_D}")
        ok = False

    # Write output to test results log
    # Next to this script, independent of the current working directory (a
    # relative path created a stray nested testfiles/ tree when run from here).
    output_dir = os.path.dirname(os.path.abspath(__file__))
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

    if not ok:
        print("\n[FAIL] MORTAR COUPLING ASSEMBLY TEST FAILED!")
        return False

    print("\n[PASS] MORTAR COUPLING ASSEMBLY TEST SUCCESSFUL!")
    return True

if __name__ == '__main__':
    if not run_coupling_assembly_test():
        sys.exit(1)
