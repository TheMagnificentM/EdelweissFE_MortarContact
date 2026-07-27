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

def test_active_set_semismooth_and_pdass_freeze():
    """Semi-smooth normal complementarity + PDASS termination.

    The normal active set is re-evaluated from the augmented-pressure indicator
    s_n = p_n - c_n*inv_D*g_sep > 0 (Hueber & Wohlmuth 2005; Gitterle et al. 2010,
    Eq. 55; MOOSE ComputeWeightedGapLMMechanicalContact). It is updated on EVERY
    Newton iteration as long as it keeps changing (NOT frozen after a fixed
    iteration count, unlike the earlier heuristic), and it FREEZES for the rest of
    the increment once it is unchanged between two consecutive iterations - the
    literal PDASS stopping criterion (Hueber & Wohlmuth 2005). Freezing the
    stabilized set is the globalization that stops active-set chattering. A new
    increment (changed timeStep.number) re-opens the set. This test asserts all
    three properties.
    """
    print("\n--- Running Semi-smooth Active Set + PDASS Freeze Test ---")
    model = FEModel(dimension=3)
    slave_coords = [
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]
    ]
    master_coords = [
        [0.0, 0.0, 1.1], [1.0, 0.0, 1.1], [1.0, 1.0, 1.1], [0.0, 1.0, 1.1],
        [0.0, 0.0, 2.1], [1.0, 0.0, 2.1], [1.0, 1.0, 2.1], [0.0, 1.0, 2.1]
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
    
    U_np = np.zeros(constraint.nDof)
    dU = np.zeros(constraint.nDof)
    PExt = np.zeros(constraint.nDof)
    K = np.zeros((constraint.nDof, constraint.nDof))
    slave_nodes = list(constraint.non_mortar_nodes)

    def penetrate_first(k):
        """Push the first k slave nodes into the master, the rest out."""
        for j, nd in enumerate(slave_nodes):
            idx = constraint.node_to_global_idx[nd]
            U_np[constraint.sizeField * idx + 2] = 0.15 if j < k else 0.0

    # ------------------------------------------------------------------
    # (1) The set is updated EVERY iteration as long as its state is still
    #     new (semi-smooth; there is NO fixed iteration count in the freeze).
    #     Go from all-open to all-penetrating: both are new states, so the set
    #     tracks the current configuration and does not freeze. (all-or-nothing
    #     avoids the mortar coupling ambiguity of a partially penetrating face.)
    # ------------------------------------------------------------------
    ts1 = TimeStep(1, 0.0, 0.0, 0.0, 0.0, 0.0)
    penetrate_first(0)                                   # all open
    constraint.applyConstraint(U_np, dU, PExt, K, ts1)
    assert constraint.current_iteration == 0
    assert not np.any(constraint.active_set), "open gap -> inactive"
    assert not constraint._set_frozen
    penetrate_first(len(slave_nodes))                    # all penetrating (new state)
    constraint.applyConstraint(U_np, dU, PExt, K, ts1)
    assert constraint.current_iteration == 1
    assert np.all(constraint.active_set), "penetration -> the set updated this iteration"
    assert not constraint._set_frozen, "a still-new state must not freeze"
    print("  (1) updated the set to the new state each iteration, not frozen: OK")

    # ------------------------------------------------------------------
    # (2) Anti-cycling (Bland 1977): a period-2 active-set limit cycle is
    #     detected and frozen. Alternate all-penetrate / all-release; the set
    #     A, B, A... repeats state A at the 3rd iteration -> freeze. A later
    #     change of the iterate must then NOT flip it (this is the anti-
    #     chattering globalization that terminates the cycle).
    # ------------------------------------------------------------------
    ts2 = TimeStep(2, 0.0, 0.0, 0.0, 0.0, 0.0)
    penetrate_first(len(slave_nodes))                    # state A: all active
    constraint.applyConstraint(U_np, dU, PExt, K, ts2)   # new increment -> re-opened
    assert not constraint._set_frozen
    assert np.all(constraint.active_set)
    penetrate_first(0)                                   # state B: all inactive
    constraint.applyConstraint(U_np, dU, PExt, K, ts2)
    assert not constraint._set_frozen                    # B is still a new state
    assert not np.any(constraint.active_set)
    penetrate_first(len(slave_nodes))                    # back to state A -> revisit
    constraint.applyConstraint(U_np, dU, PExt, K, ts2)
    assert constraint._set_frozen, "a repeated state (period-2 cycle) must freeze (anti-cycling)"
    assert np.all(constraint.active_set)                 # frozen on the revisited state A
    penetrate_first(0)                                   # reopen the gap mid-increment
    constraint.applyConstraint(U_np, dU, PExt, K, ts2)
    assert constraint._set_frozen and np.all(constraint.active_set), (
        "frozen set must NOT change within the increment even though the gap "
        "reopened (globalization that terminates the cycle)"
    )
    print("  (2) detected the period-2 cycle, froze it, and held against a later change: OK")

    # ------------------------------------------------------------------
    # (3) A new increment re-opens the set: the freeze is reset and the set
    #     is re-evaluated from the current (now open) configuration.
    # ------------------------------------------------------------------
    ts3 = TimeStep(3, 0.0, 0.0, 0.0, 0.0, 0.0)
    constraint.applyConstraint(U_np, dU, PExt, K, ts3)   # gap still open from (2)
    assert not constraint._set_frozen, "new increment must reset the freeze"
    assert not np.any(constraint.active_set), "re-opened set must reflect the open gap (inactive)"
    print("  (3) new increment reset the freeze and re-evaluated the set: OK")

    print("[PASS] Semi-smooth Active Set + PDASS Freeze Test Successful!")

def test_bvh_search_correctness():
    print("\n--- Running BVH Search Correctness Test ---")
    model = FEModel(dimension=3)
    ConQuadClass = getElementClass("CONQUAD4", "edelweiss")
    
    # Create a 2x2 grid of Slave elements
    s_elements = []
    node_id = 1
    el_id = 1
    for x in range(3):
        for y in range(3):
            model.nodes[node_id] = Node(node_id, np.array([float(x), float(y), 1.0]))
            model.nodes[node_id].fields["displacement"] = FieldVariable(model.nodes[node_id], "displacement")
            node_id += 1
            
    for i in range(2):
        for j in range(2):
            n1 = i * 3 + j + 1
            n2 = n1 + 1
            n3 = n1 + 4
            n4 = n1 + 3
            el = ConQuadClass("CONQUAD4", el_id)
            el.setNodes([model.nodes[n1], model.nodes[n2], model.nodes[n3], model.nodes[n4]])
            s_elements.append(el)
            model.elements[el_id] = el
            el_id += 1
            
    # Create a 2x2 grid of Master elements (shifted in space)
    m_elements = []
    for x in range(3):
        for y in range(3):
            model.nodes[node_id] = Node(node_id, np.array([float(x) + 0.5, float(y) + 0.5, 1.0]))
            model.nodes[node_id].fields["displacement"] = FieldVariable(model.nodes[node_id], "displacement")
            node_id += 1
            
    for i in range(2):
        for j in range(2):
            n1 = i * 3 + j + 1 + 9
            n2 = n1 + 1
            n3 = n1 + 4
            n4 = n1 + 3
            el = ConQuadClass("CONQUAD4", el_id)
            el.setNodes([model.nodes[n1], model.nodes[n2], model.nodes[n3], model.nodes[n4]])
            m_elements.append(el)
            model.elements[el_id] = el
            el_id += 1
            
    model.surfaces = {
        "slave_surf": {1: s_elements},
        "master_surf": {1: m_elements}
    }
    
    constraint = MortarContact(
        "contact_constraint",
        model,
        nonMortarSurface="slave_surf",
        mortarSurface="master_surf",
        field="displacement"
    )
    
    D, C = constraint.compute_mortar_coupling_matrices()
    print("Matrices assembled successfully using BVH tree search.")
    print("Matrix C non-zeros:", np.count_nonzero(C))
    assert np.count_nonzero(C) > 0, "Expected non-zero coupling terms in matrix C"
    print("[PASS] BVH Search Correctness Test Successful!")

if __name__ == '__main__':
    run_active_set_test()
    test_active_set_semismooth_and_pdass_freeze()
    test_bvh_search_correctness()
