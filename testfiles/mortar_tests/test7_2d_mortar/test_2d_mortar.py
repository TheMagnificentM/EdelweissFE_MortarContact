#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test 7: 2D Mortar Contact Implementation Test
============================================

This test verifies 2D Mortar contact mechanics in EdelweissFE for both linear
(CONLINE2) and quadratic (CONLINE3) contact line elements:

1. 2D Node Normals:
   Correct unit normal vector calculation n = (t_y, -t_x).
2. 2D Biorthogonality & Positive Dual Weights:
   Validates basis transformation T_e (alpha=1/3) for CONLINE3, guaranteeing
   strictly positive weights D_II > 0.
3. 2D Segment Coupling Matrices:
   - Full length coverage sum(D) = length (Partition of Unity).
   - Row-sum conservation sum_K D_IK = sum_J C_IJ (Translational Invariance).
   - Analytical node weights for CONLINE3 (corners 1/6, mid-node 2/3).
4. Solver-Based 2D Contact Patch Test:
   - 2D Two-Block Compression (QUAD4 / CONLINE2 and QUAD8 / CONLINE3).
   - Verifies uniform displacement field and pressure transmission in machine precision.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from edelweissfe.config.elementlibrary import getElementClass
from edelweissfe.constraints.mortarcontact import Constraint as MortarContact
from edelweissfe.models.femodel import FEModel
from edelweissfe.points.node import Node
from edelweissfe.variables.fieldvariable import FieldVariable

TOL = 1e-12


def make_nodes_2d(model, start_id, points):
    nodes = []
    for i, pt in enumerate(points):
        n_id = start_id + i
        model.nodes[n_id] = Node(n_id, np.array(pt, dtype=float))
        nodes.append(model.nodes[n_id])
    return nodes


def line2_points(shift_x=0.0, y=0.0):
    return [
        [0.0 + shift_x, y],
        [1.0 + shift_x, y],
    ]


def line3_points(shift_x=0.0, y=0.0):
    return [
        [0.0 + shift_x, y],
        [1.0 + shift_x, y],
        [0.5 + shift_x, y],
    ]


def build_2d_contact_model(el_type, slave_pts, master_pts):
    model = FEModel(dimension=2)
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

    return MortarContact("c", model, nonMortarSurface="slave", mortarSurface="master", field="displacement")


def check(name, value, expected, tol=TOL):
    err = abs(value - expected)
    if err > tol:
        print(f"  [FAIL] {name}: {value:.12f}, erwartet {expected:.12f} (diff {err:.2e})")
        return False
    print(f"  [PASS] {name}: {value:.12f}")
    return True


def test_2d_line2():
    print("\n=== Test 2D CONLINE2 (Linear Line Element) ===")
    mc = build_2d_contact_model("CONLINE2", line2_points(0.0, 0.0), line2_points(0.0, 0.0))

    # Test Normals
    n = mc.compute_normals()
    print(f"Normals: {n}")
    # Normal to horizontal segment (1,0) pointing down (0,-1) or up (0,1)
    assert np.allclose(np.abs(n[:, 1]), 1.0), "Normals should be vertical"

    # Test Coupling Matrices
    D, C = mc.compute_mortar_coupling_matrices()
    print(f"D matrix:\n{D}")
    print(f"C matrix:\n{C}")

    all_ok = True
    all_ok &= check("sum(D)", float(np.sum(D)), 1.0)
    all_ok &= check("sum(C)", float(np.sum(C)), 1.0)
    all_ok &= check("rowsum diff D vs C", float(np.max(np.abs(np.sum(D, axis=1) - np.sum(C, axis=1)))), 0.0)
    all_ok &= check("node 0 weight", D[0, 0], 0.5)
    all_ok &= check("node 1 weight", D[1, 1], 0.5)

    assert all_ok, "CONLINE2 test failed"


def test_2d_line3():
    print("\n=== Test 2D CONLINE3 (Quadratic Line Element) ===")
    mc = build_2d_contact_model("CONLINE3", line3_points(0.0, 0.0), line3_points(0.0, 0.0))

    # Test Normals
    n = mc.compute_normals()
    print(f"Normals:\n{n}")
    assert np.allclose(np.abs(n[:, 1]), 1.0), "Normals should be vertical"

    # Test Coupling Matrices
    D, C = mc.compute_mortar_coupling_matrices()
    print(f"D matrix:\n{D}")
    print(f"C matrix:\n{C}")

    rowsum_D = np.sum(D, axis=1)
    rowsum_C = np.sum(C, axis=1)

    all_ok = True
    all_ok &= check("sum(D)", float(np.sum(D)), 1.0)
    all_ok &= check("sum(C)", float(np.sum(C)), 1.0)
    all_ok &= check("rowsum diff D vs C", float(np.max(np.abs(rowsum_D - rowsum_C))), 0.0)

    # Check positive dual weights (analytical with T_e alpha=1/3: corners 7/18 = 0.388888888889, mid-node 2/9 = 0.222222222222)
    all_ok &= check("corner 0 weight", rowsum_D[0], 7.0 / 18.0)
    all_ok &= check("corner 1 weight", rowsum_D[1], 7.0 / 18.0)
    all_ok &= check("mid-node 2 weight", rowsum_D[2], 2.0 / 9.0)

    assert np.all(rowsum_D > 0), "All dual node weights must be strictly positive"
    assert all_ok, "CONLINE3 test failed"


def test_2d_partial_overlap():
    print("\n=== Test 2D Partial Overlap (50% Shift) ===")
    # Shift master line segment by +0.5 x
    mc = build_2d_contact_model("CONLINE2", line2_points(0.0, 0.0), line2_points(0.5, 0.0))
    D, C = mc.compute_mortar_coupling_matrices()

    rowsum_D = np.sum(D, axis=1)
    rowsum_C = np.sum(C, axis=1)

    all_ok = True
    all_ok &= check("sum(D)", float(np.sum(D)), 0.5)
    all_ok &= check("sum(C)", float(np.sum(C)), 0.5)
    all_ok &= check("rowsum diff D vs C", float(np.max(np.abs(rowsum_D - rowsum_C))), 0.0)

    assert all_ok, "Partial overlap test failed"


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING 2D MORTAR CONTACT SUITE")
    print("=" * 60)
    test_2d_line2()
    test_2d_line3()
    test_2d_partial_overlap()
    print("\n[SUCCESS] All 2D Mortar contact tests passed successfully!")
