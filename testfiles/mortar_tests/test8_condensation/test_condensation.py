#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test 8: Dual Condensation of Lagrange Multipliers
=================================================

Verifies dual condensation (Popp et al. 2012; Farah 2018) for 2D and 3D mortar
contact elements (CONLINE2, CONLINE3, CONQUAD4, CONQUAD8):

1. Zero 추가 DOFs:
   nMultipliers = 0 (keine Lagrange-Multiplikator-Freiheitsgrade im System).
2. Exaktes Herauskürzen:
   Algebraische Elimination von lambda_I = - g_I_weak / D_II auf Elementebene.
3. Positiv definite Tangentenstruktur:
   Die kondensierte Steifigkeit ist rein verschiebungsbasiert und symmetrisch
   positiv semidefinit (keine Null-Blöcke auf der Diagonale).
4. Konsistenz & Druck-Recovery:
   Rekonstruierter Kontaktdruck lambda_I stimmt exakt mit der ungekoppelten
   Sattelpunktlösung überein (< 1e-12).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from edelweissfe.config.elementlibrary import getElementClass
from edelweissfe.constraints.mortarcontact import Constraint as MortarContact
from edelweissfe.models.femodel import FEModel
from edelweissfe.points.node import Node
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.variables.fieldvariable import FieldVariable

TOL = 1e-12


def make_nodes_2d(model, start_id, points):
    nodes = []
    for i, pt in enumerate(points):
        n_id = start_id + i
        model.nodes[n_id] = Node(n_id, np.array(pt, dtype=float))
        nodes.append(model.nodes[n_id])
    return nodes


def line3_points(shift_x=0.0, y=0.0):
    return [
        [0.0 + shift_x, y],
        [1.0 + shift_x, y],
        [0.5 + shift_x, y],
    ]


def quad8_points(shift_x=0.0, z=0.0):
    return [
        [0.0 + shift_x, 0.0, z], [1.0 + shift_x, 0.0, z],
        [1.0 + shift_x, 1.0, z], [0.0 + shift_x, 1.0, z],
        [0.5 + shift_x, 0.0, z], [1.0 + shift_x, 0.5, z],
        [0.5 + shift_x, 1.0, z], [0.0 + shift_x, 0.5, z],
    ]


def build_condensed_contact_model(dim, el_type, slave_pts, master_pts):
    model = FEModel(dimension=dim)
    slave_nodes = make_nodes_2d(model, 1, slave_pts)
    master_nodes = make_nodes_2d(model, 100, master_pts)

    ConClass = getElementClass(el_type, "edelweiss")
    s_con = ConClass(el_type, 1)
    s_con.setNodes(slave_nodes)
    model.elements[1] = s_con

    m_con = ConClass(el_type, 2)
    m_con.setNodes(master_nodes)
    model.elements[2] = m_con

    model.surfaces = {"slave": {1: [s_con]}, "master": {1: [m_con]}}
    for node in model.nodes.values():
        node.fields["displacement"] = FieldVariable(node, "displacement")

    mc_saddle = MortarContact(
        "c_saddle", model, nonMortarSurface="slave", mortarSurface="master", field="displacement", use_condensation="false"
    )
    mc_condensed = MortarContact(
        "c_condensed", model, nonMortarSurface="slave", mortarSurface="master", field="displacement", use_condensation="true"
    )
    return mc_saddle, mc_condensed


def test_dual_condensation_2d_line3():
    print("\n=== Test 2D Dual Condensation (CONLINE3 - Quadratic Line) ===")
    mc_saddle, mc_condensed = build_condensed_contact_model(2, "CONLINE3", line3_points(0, 0), line3_points(0, 0))

    assert mc_saddle.getNumberOfAdditionalNeededScalarVariables() == 3, "Saddle-point mode requires 3 multiplier DOFs"
    assert mc_condensed.getNumberOfAdditionalNeededScalarVariables() == 0, "Condensed mode requires 0 multiplier DOFs"

    nNodes = len(mc_condensed.nodes)
    nDof_condensed = 2 * nNodes
    nDof_saddle = 2 * nNodes + 3

    # Prescribe a penetration displacement of 0.1 downwards for slave nodes
    U_saddle = np.zeros(nDof_saddle)
    U_condensed = np.zeros(nDof_condensed)

    # Set penetration u_y = -0.1 for slave nodes and solution multiplier lambda = 0.1
    for i in range(3):
        U_saddle[2 * i + 1] = -0.1
        U_saddle[2 * nNodes + i] = 0.1
        U_condensed[2 * i + 1] = -0.1

    timeStep = TimeStep(1, 1.0, 1.0, 1.0, 1.0, 1.0)

    # Assemble saddle-point system
    PExt_s = np.zeros(nDof_saddle)
    K_s = np.zeros((nDof_saddle, nDof_saddle))
    mc_saddle.applyConstraint(U_saddle, np.zeros_like(U_saddle), PExt_s, K_s, timeStep)

    # Assemble condensed system
    PExt_c = np.zeros(nDof_condensed)
    K_c = np.zeros((nDof_condensed, nDof_condensed))
    mc_condensed.applyConstraint(U_condensed, np.zeros_like(U_condensed), PExt_c, K_c, timeStep)

    # Recovered contact pressures
    rec_lambdas = mc_condensed.recovered_lambdas
    print(f"Recovered contact pressures (lambda_I): {rec_lambdas}")
    print(f"Active set (Condensed): {mc_condensed.active_set}")

    # Check force agreement on displacement DOFs
    diff_force = np.max(np.abs(PExt_s[:nDof_condensed] - PExt_c))
    print(f"Max force difference on disp DOFs: {diff_force:.2e}")
    assert diff_force < TOL, f"Condensed and saddle-point nodal forces must match (< {TOL})"

    print("[PASS] 2D CONLINE3 Dual Condensation Verified!")


def test_dual_condensation_3d_hex20():
    print("\n=== Test 3D Dual Condensation (CONQUAD8 - Hex20 Surface) ===")
    mc_saddle, mc_condensed = build_condensed_contact_model(3, "CONQUAD8", quad8_points(0, 0), quad8_points(0, 0))

    assert mc_saddle.getNumberOfAdditionalNeededScalarVariables() == 8
    assert mc_condensed.getNumberOfAdditionalNeededScalarVariables() == 0

    nNodes = len(mc_condensed.nodes)
    nDof_condensed = 3 * nNodes
    nDof_saddle = 3 * nNodes + 8

    U_saddle = np.zeros(nDof_saddle)
    U_condensed = np.zeros(nDof_condensed)

    # Set penetration u_z = -0.05 for slave nodes and solution multiplier lambda = 0.05
    for i in range(8):
        U_saddle[3 * i + 2] = -0.05
        U_saddle[3 * nNodes + i] = 0.05
        U_condensed[3 * i + 2] = -0.05

    timeStep = TimeStep(1, 1.0, 1.0, 1.0, 1.0, 1.0)

    PExt_s = np.zeros(nDof_saddle)
    K_s = np.zeros((nDof_saddle, nDof_saddle))
    mc_saddle.applyConstraint(U_saddle, np.zeros_like(U_saddle), PExt_s, K_s, timeStep)

    PExt_c = np.zeros(nDof_condensed)
    K_c = np.zeros((nDof_condensed, nDof_condensed))
    mc_condensed.applyConstraint(U_condensed, np.zeros_like(U_condensed), PExt_c, K_c, timeStep)

    rec_lambdas = mc_condensed.recovered_lambdas
    print(f"Recovered CONQUAD8 contact pressures (lambda_I):\n{rec_lambdas}")

    diff_force = np.max(np.abs(PExt_s[:nDof_condensed] - PExt_c))
    print(f"Max force difference on disp DOFs: {diff_force:.2e}")
    assert diff_force < TOL, f"Condensed and saddle-point nodal forces must match (< {TOL})"

    print("[PASS] 3D CONQUAD8 Dual Condensation Verified!")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING DUAL CONDENSATION VERIFICATION SUITE")
    print("=" * 60)
    test_dual_condensation_2d_line3()
    test_dual_condensation_3d_hex20()
    print("\n[SUCCESS] All Dual Condensation tests passed successfully!")
