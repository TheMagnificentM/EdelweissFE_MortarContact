#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test 6: Verification of the Primal-Dual Active Set Strategy (PDASS)
==================================================================

This test verifies the active set status updates, residual vector assembly,
and stiffness matrix assembly under contact and separation conditions.

The active set follows the semi-smooth normal complementarity function (NCP)

    C_n,I = p_n - max(0, p_n - c_n * inv_D * g_sep) = 0
        <=>  active  iff  s_n = p_n - c_n * inv_D * g_sep > 0

(Gitterle et al. 2010, Eq. (55); Hueber & Wohlmuth 2005), re-evaluated in EVERY
Newton iteration. The semi-smooth (outer) iteration is terminated once the
discrete state has settled - NOT after a fixed number of iterations.
"""

import sys
import os
import warnings

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

def build_two_block_contact(cn=1000.0):
    """Two unit blocks, slave face at z = 1.0, master face at z = 1.1 (gap 0.1)."""
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
        field="displacement",
        cn=cn,
    )
    nDof = constraint.nDof
    return (constraint, np.zeros(nDof), np.zeros(nDof), np.zeros(nDof), np.zeros((nDof, nDof)))


def set_slave_z(constraint, U_np, values):
    """Prescribe the z-displacement of every slave node (order of non_mortar_nodes)."""
    for local, nd in enumerate(constraint.non_mortar_nodes):
        idx = constraint.node_to_global_idx[nd]
        U_np[constraint.sizeField * idx + 2] = values[local]


def test_ncp_indicator():
    """The NCP indicator s_n = p_n - c_n*inv_D*g_sep decides - not penetration alone.

    At a CLOSED gap (g_sep = 0) a positive normal pressure keeps the node active
    and a tensile (negative) pressure releases it. This is the branch a pure
    penetration heuristic cannot represent (Gitterle et al. 2010, Eq. (55)).
    """
    print("\n--- Running Normal NCP Indicator Test ---")
    constraint, U_np, dU, PExt, K = build_two_block_contact()

    # Close the gap exactly (slave face 1.0 -> 1.1 = master face): g_sep = 0
    set_slave_z(constraint, U_np, [0.1] * 4)

    # Increment 1: g_sep = 0 and lambda = 0 -> s_n = 0, NOT > 0 -> inactive
    constraint.applyConstraint(U_np, dU, PExt, K, TimeStep(1, 0.0, 0.0, 0.0, 0.0, 0.0))
    print("  g_sep = 0, p_n = 0 -> active set:", constraint.active_set)
    assert not np.any(constraint.active_set), "s_n = 0 must not activate"

    idx_LM_0 = constraint.sizeField * len(constraint._nodes)
    sgn_D = np.sign(constraint.current_D_rowsum)

    # Increment 2: g_sep = 0 but COMPRESSIVE pressure p_n = -lambda*sgn_D = +1 > 0
    U_np[idx_LM_0:] = -sgn_D
    PExt.fill(0.0); K.fill(0.0)
    constraint.applyConstraint(U_np, dU, PExt, K, TimeStep(2, 0.0, 0.0, 0.0, 0.0, 0.0))
    print("  g_sep = 0, p_n = +1 -> active set:", constraint.active_set)
    assert np.all(constraint.active_set), "positive pressure at closed gap must be active"

    # Increment 3: g_sep = 0 but TENSILE pressure p_n = -1 < 0 -> released
    U_np[idx_LM_0:] = sgn_D
    PExt.fill(0.0); K.fill(0.0)
    constraint.applyConstraint(U_np, dU, PExt, K, TimeStep(3, 0.0, 0.0, 0.0, 0.0, 0.0))
    print("  g_sep = 0, p_n = -1 -> active set:", constraint.active_set)
    assert not np.any(constraint.active_set), "tensile pressure must release the node"
    print("[PASS] Normal NCP Indicator Test Successful!")


def test_active_set_reevaluated_every_iteration():
    """No fixed iteration cutoff: the indicator is re-evaluated in EVERY Newton
    iteration as long as the discrete state has not settled - also past iteration
    5, where the previous heuristic stopped updating (Hueber & Wohlmuth 2005).
    """
    print("\n--- Running Active Set Re-evaluation Test ---")
    constraint, U_np, dU, PExt, K = build_two_block_contact()
    timeStep = TimeStep(1, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Eight pairwise DIFFERENT active-set patterns of the four slave nodes, so the
    # state neither settles (freeze trigger 1) nor recurs (anti-cycling trigger 2).
    # For linear facets D is diagonal, hence each nodal gap is controlled by that
    # node's own displacement alone: z = 0.15 penetrates, z = 0.0 stays separated.
    patterns = [
        (0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (1, 1, 0, 0),
        (0, 0, 1, 0), (1, 0, 1, 0), (0, 1, 1, 0), (1, 1, 1, 0),
    ]
    for it, pat in enumerate(patterns):
        set_slave_z(constraint, U_np, [0.15 if p else 0.0 for p in pat])
        PExt.fill(0.0); K.fill(0.0)
        constraint.applyConstraint(U_np, dU, PExt, K, timeStep)
        got = tuple(int(b) for b in constraint.active_set)
        print(f"  Iteration {constraint.current_iteration}: active set {got}, expected {pat}")
        assert constraint.current_iteration == it
        assert not constraint.active_set_frozen, "set has not settled - must not be frozen"
        assert got == pat, f"iteration {it}: expected {pat}, got {got}"
    print("[PASS] Active Set Re-evaluation Test Successful (updated up to iteration 7)!")


def test_active_set_pdass_termination():
    """PDASS convergence criterion (Hueber & Wohlmuth 2005): the semi-smooth
    iteration is frozen once the discrete state has SETTLED - unchanged for two
    consecutive iterations past a warm-up - and not after a fixed iteration count.
    The next increment restarts it.
    """
    print("\n--- Running PDASS Termination Test ---")
    constraint, U_np, dU, PExt, K = build_two_block_contact()
    timeStep = TimeStep(1, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Iterations 0, 1, 2 without penetration: the state (all inactive) settles.
    for it in range(3):
        PExt.fill(0.0); K.fill(0.0)
        constraint.applyConstraint(U_np, dU, PExt, K, timeStep)
        print(f"  Iteration {constraint.current_iteration}: frozen = {constraint.active_set_frozen}")
        assert constraint.current_iteration == it
    assert constraint.active_set_frozen, "settled state must terminate the PDASS loop"

    # A penetration applied AFTER the freeze must not change the set any more.
    set_slave_z(constraint, U_np, [0.15] * 4)
    PExt.fill(0.0); K.fill(0.0)
    constraint.applyConstraint(U_np, dU, PExt, K, timeStep)
    print("  Active set after penetration in the frozen increment:", constraint.active_set)
    assert not np.any(constraint.active_set), "frozen set must not change within the increment"

    # The next increment restarts the semi-smooth iteration -> penetration detected.
    #
    # This step doubles as the check of the freeze safeguard. The state handed in
    # here is precisely the case the safeguard exists for: the increment was frozen
    # on "all inactive" and then acquired a penetration, so its set does NOT
    # reproduce itself on the state it ended with. The constraint re-evaluates the
    # indicator at the start of the following increment - the one moment where a
    # converged state is available - and must say so.
    PExt.fill(0.0); K.fill(0.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        constraint.applyConstraint(U_np, dU, PExt, K, TimeStep(2, 0.0, 0.0, 0.0, 0.0, 0.0))
    messages = [str(w.message) for w in caught]
    print("  Active set in the new increment:", constraint.active_set)
    assert any("does not reproduce itself" in m for m in messages), (
        "the constraint must report that the frozen active set of the previous increment "
        f"does not reproduce itself on its converged state; got {messages}"
    )
    print("  [OK]   Der eingefrorene Satz wurde als nicht selbstkonsistent gemeldet")
    assert not constraint.active_set_frozen, "new increment must reset the freeze"
    assert np.all(constraint.active_set), "penetration must be detected again"
    print("[PASS] PDASS Termination Test Successful!")

def build_quad9_partial_overlap(shift_x=0.3, cn=1000.0):
    """CONQUAD9 slave against a CONQUAD9 master shifted in x -> partial coverage.

    This is the one configuration in which the nodal weight D_II = int(Phi_I) turns
    NEGATIVE: the full-Lagrangian CONQUAD9 shape functions are not pointwise
    non-negative, so integral positivity (Popp et al. 2012, Eq. (4.2)) is not
    guaranteed once only part of the slave facet is covered. Both facets lie in the
    z = 0 plane, so the slave nodal normal is +e_z and the master can be opened by
    simply translating it in +z.
    """
    def quad9(sx, z=0.0):
        return [
            [0.0 + sx, 0.0, z], [1.0 + sx, 0.0, z], [1.0 + sx, 1.0, z], [0.0 + sx, 1.0, z],
            [0.5 + sx, 0.0, z], [1.0 + sx, 0.5, z], [0.5 + sx, 1.0, z], [0.0 + sx, 0.5, z],
            [0.5 + sx, 0.5, z],
        ]

    model = FEModel(dimension=3)
    ConClass = getElementClass("CONQUAD9", "edelweiss")
    slave_nodes = [Node(1 + i, np.array(p, dtype=float)) for i, p in enumerate(quad9(0.0))]
    master_nodes = [Node(100 + i, np.array(p, dtype=float)) for i, p in enumerate(quad9(shift_x))]
    for nd in slave_nodes + master_nodes:
        model.nodes[nd.label] = nd

    s_con = ConClass("CONQUAD9", 1)
    s_con.setNodes(slave_nodes)
    model.elements[1] = s_con
    m_con = ConClass("CONQUAD9", 2)
    m_con.setNodes(master_nodes)
    model.elements[2] = m_con

    model.surfaces = {"slave_surf": {1: [s_con]}, "master_surf": {1: [m_con]}}
    for nd in model.nodes.values():
        nd.fields["displacement"] = FieldVariable(nd, "displacement")

    constraint = MortarContact(
        "contact_constraint", model,
        nonMortarSurface="slave_surf", mortarSurface="master_surf",
        field="displacement", cn=cn,
    )
    nDof = constraint.nDof
    return constraint, np.zeros(nDof), np.zeros(nDof), np.zeros(nDof), np.zeros((nDof, nDof))


def test_negative_nodal_weight_opens_correctly():
    """A node with NEGATIVE D_II must still deactivate when the gap is open.

    Regression test for a sign bug in the active-set indicator. With D_II < 0 both
    sign conventions flip: compression means lambda*D_II < 0, and translating the
    master away by a changes the weak gap by D_II*a, i.e. an OPEN gap gives
    g_weak < 0. The indicator therefore has to use the opening normalized by the
    SIGNED weight, g_sep = g_weak/D_II, which is positive-when-open for either sign.
    The earlier implementation multiplied by sgn(D_II) on top of dividing by D_II,
    which flipped the gap sign back: a wide open node was reported as penetrating,
    stayed active, and glued the surfaces (transmitting tension).

    Popp et al. (2012), Sec. 4.3, state the requirement this enforces: the discrete
    condition "should yield a positive weighted gap if the value of the unweighted
    physical gap function evaluated at slave node j is positive and vice versa.
    Otherwise, the numerical algorithm will generate nonphysical gaps and
    penetrations".
    """
    print("\n--- Running Negative Nodal Weight Test (CONQUAD9, partial overlap) ---")
    constraint, U_np, dU, PExt, K = build_quad9_partial_overlap()

    def evaluate(offset, increment):
        """Translate the whole master facet by offset*n_I and re-evaluate the set.

        Every configuration gets its OWN increment number. That matters twice: the
        semi-smooth loop freezes the set once it has settled (so re-using one
        increment would silently stop updating after two stable iterations), and a
        new increment re-evaluates the frozen geometry at the current state - which
        is exactly what the solver does. Translating along the slave normal leaves
        the auxiliary-plane overlap unchanged, so D and C (and hence the negative
        weights) are the same in every configuration.
        """
        U_np[:] = 0.0
        for nd in constraint.mortar_nodes:
            idx = constraint.node_to_global_idx[nd]
            U_np[constraint.sizeField * idx : constraint.sizeField * idx + 3] = offset * n_I
        PExt.fill(0.0)
        K.fill(0.0)
        constraint.applyConstraint(U_np, dU, PExt, K, TimeStep(increment, 0.0, 0.0, 0.0, 0.0, 0.0))
        return constraint.active_set.copy()

    constraint.applyConstraint(U_np, dU, PExt, K, TimeStep(1, 0.0, 0.0, 0.0, 0.0, 0.0))
    rowsum = constraint.current_D_rowsum
    negative = np.flatnonzero(rowsum < 0.0)
    print(f"  Nodal weights D_II: {np.round(rowsum, 5)}")
    print(f"  Nodes with NEGATIVE weight: {list(negative)} -> {np.round(rowsum[negative], 6)}")
    assert len(negative) > 0, (
        "This configuration is supposed to produce negative nodal weights - "
        "without them the test cannot exercise the sign handling."
    )

    n_I = constraint.current_normals[0]
    assert abs(n_I[2]) > 0.99, f"expected a +/-z slave normal, got {n_I}"

    # Open the gap by translating the master away from the slave. lambda stays zero,
    # so the decision rests entirely on the gap term of the indicator.
    for k, opening in enumerate((0.01, 0.05, 0.2)):
        active = evaluate(+opening, 10 + k)
        print(f"  opening     = {opening:.2f}, lambda = 0 -> active set {active.astype(int)}")
        if np.any(active[negative]):
            print(
                f"  [FAIL] Node(s) {list(negative[active[negative]])} with D_II < 0 are ACTIVE "
                f"although the gap is open by {opening} - the indicator reads the gap sign wrong!"
            )
            sys.exit(1)
        assert not np.any(active), (
            f"no node may be active at an open gap with lambda = 0, got {active.astype(int)}"
        )

    # Counter-check: pushing the master INTO the slave must activate the very same
    # nodes. Without this a test that simply never activates anything would pass.
    for k, penetration in enumerate((0.01, 0.05)):
        active = evaluate(-penetration, 20 + k)
        print(f"  penetration = {penetration:.2f}, lambda = 0 -> active set {active.astype(int)}")
        assert np.all(active[negative]), (
            f"nodes {list(negative)} with D_II < 0 must activate on penetration, got {active.astype(int)}"
        )
        assert np.all(active), f"all nodes must activate on a uniform penetration, got {active.astype(int)}"

    print("[PASS] Negative Nodal Weight Test Successful!")


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
    test_ncp_indicator()
    test_negative_nodal_weight_opens_correctly()
    test_active_set_reevaluated_every_iteration()
    test_active_set_pdass_termination()
    test_bvh_search_correctness()
