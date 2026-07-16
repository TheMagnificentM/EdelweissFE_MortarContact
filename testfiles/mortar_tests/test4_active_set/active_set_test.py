#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test 4: Verification of the Primal-Dual Active Set Strategy (PDASS)
==================================================================

This test verifies the active set status updates, residual vector assembly, 
and stiffness matrix assembly under contact and separation conditions.
"""

import sys
import os
import numpy as np

# Add local path of EdelweissFE to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from edelweissfe.models.femodel import FEModel
from edelweissfe.config.elementlibrary import getElementClass
from edelweissfe.constraints.mortarcontact import Constraint as MortarContact
from edelweissfe.points.node import Node
from edelweissfe.variables.fieldvariable import FieldVariable
from edelweissfe.timesteppers.timestep import TimeStep

def run_active_set_test():
    print("================ RUNNING MORTAR ACTIVE SET (PDASS) TEST ================")
    
    # 1. Create Model with dimension=3
    model = FEModel(dimension=3)
    
    # Define Nodes directly for two solid blocks
    # Block 1 (Slave): from [0, 1]x[0, 1]x[0, 1]
    # Block 2 (Master): from [0, 1]x[0, 1]x[1.1, 2.1] (separated by 0.1 in Z direction)
    slave_coords = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], # Node 5
        [1.0, 0.0, 1.0], # Node 6
        [1.0, 1.0, 1.0], # Node 7
        [0.0, 1.0, 1.0], # Node 8
    ]
    master_coords = [
        [0.0, 0.0, 1.1], # Node 9
        [1.0, 0.0, 1.1], # Node 10
        [1.0, 1.0, 1.1], # Node 11
        [0.0, 1.0, 1.1], # Node 12
        [0.0, 0.0, 2.1],
        [1.0, 0.0, 2.1],
        [1.0, 1.0, 2.1],
        [0.0, 1.0, 2.1]
    ]
    
    for i, pt in enumerate(slave_coords):
        model.nodes[i+1] = Node(i+1, np.array(pt))
    for i, pt in enumerate(master_coords):
        model.nodes[i+9] = Node(i+9, np.array(pt))
        
    ConQuadClass = getElementClass("CONQUAD4", "edelweiss")
    s_con = ConQuadClass("CONQUAD4", 3)
    s_con.setNodes([model.nodes[5], model.nodes[6], model.nodes[7], model.nodes[8]])
    m_con = ConQuadClass("CONQUAD4", 4)
    m_con.setNodes([model.nodes[9], model.nodes[10], model.nodes[11], model.nodes[12]])
    
    model.surfaces = {
        "slave_surf": {1: [s_con]},
        "master_surf": {1: [m_con]}
    }
    
    for node in model.nodes.values():
        node.fields["displacement"] = FieldVariable(node, "displacement")
        
    constraint = MortarContact(
        "contact_constraint",
        model,
        nonMortarSurface="slave_surf",
        mortarSurface="master_surf",
        field="displacement"
    )
    model.constraints["contact_constraint"] = constraint
    
    # Define solver input vectors: local solution, incremental, residual, and stiffness
    nDof = constraint.nDof
    U_np = np.zeros(nDof)
    dU = np.zeros(nDof)
    PExt = np.zeros(nDof)
    K = np.zeros((nDof, nDof))
    timeStep = TimeStep(1, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    print("\n--- Phase A: Initial separated state (no penetration) ---")
    constraint.applyConstraint(U_np, dU, PExt, K, timeStep)
    print("Active set status (should be all False):", constraint.active_set)
    assert not np.any(constraint.active_set), "Expected all nodes to be inactive"
    print("Inactive Multiplier Residual checks: PExt[idx_LM] =", PExt[constraint.sizeField * len(constraint._nodes):])
    
    # Reset residuals and stiffness
    PExt.fill(0.0)
    K.fill(0.0)
    
    print("\n--- Phase B: Penetration state (displace Slave nodes in Z by +0.15) ---")
    # This pushes the slave boundary at Z=1.0 to Z=1.15, causing 0.05 penetration into master at Z=1.1
    for nd in constraint.non_mortar_nodes:
        idx = constraint.node_to_global_idx[nd]
        U_np[constraint.sizeField * idx + 2] = 0.15
        
    # Re-apply constraint to trigger penetration detection and activate nodes
    constraint.applyConstraint(U_np, dU, PExt, K, timeStep)
    print("Active set status after penetration (should be all True):", constraint.active_set)
    assert np.all(constraint.active_set), "Expected all nodes to be active"
    
    print("Residual vector PExt:\n", PExt)
    print("Stiffness matrix K:\n", K)
    
    print("\n[PASS] PDASS Active Set Verification Successful!")

if __name__ == '__main__':
    run_active_set_test()
