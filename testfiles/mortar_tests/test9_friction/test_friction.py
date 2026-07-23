#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test 9: Coulomb Friction (saddle-point mortar contact)
======================================================

Verifies the saddle-point Coulomb friction added on branch 9
(genuine tangential Lagrange-multiplier DOFs z_t), following
Gitterle, Popp, Gee & Wall (2010), IJNME 84:543-571.

Checks:
1. Stick regime: the tangential multiplier row enforces zero weighted
   tangential slip increment (u_t = 0), mirroring the normal g_weak = 0 row.
2. Slip regime: |z_t| is driven to the Coulomb limit mu*|lambda_n| along the
   trial direction.
3. Finite-difference tangent consistency: the assembled K must equal the true
   derivative of PExt w.r.t. ALL unknowns (displacements, lambda_n, z_t), in
   both 2D (CONLINE2) and 3D (CONQUAD8), for stick and slip. With frozen
   geometry this is the exact Jacobian and must match to ~sqrt(eps).

Run:  python test_friction.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from edelweissfe.config.elementlibrary import getElementClass
from edelweissfe.constraints.mortarcontact import Constraint as MortarContact
from edelweissfe.models.femodel import FEModel
from edelweissfe.points.node import Node
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.variables.fieldvariable import FieldVariable


def _make_nodes(model, start_id, points):
    nodes = []
    for i, pt in enumerate(points):
        n_id = start_id + i
        model.nodes[n_id] = Node(n_id, np.array(pt, dtype=float))
        nodes.append(model.nodes[n_id])
    return nodes


def _line2(shift_x=0.0, y=0.0):
    return [[shift_x, y], [1.0 + shift_x, y]]


def _quad8(shift_x=0.0, z=0.0):
    return [
        [0.0 + shift_x, 0.0, z], [1.0 + shift_x, 0.0, z],
        [1.0 + shift_x, 1.0, z], [0.0 + shift_x, 1.0, z],
        [0.5 + shift_x, 0.0, z], [1.0 + shift_x, 0.5, z],
        [0.5 + shift_x, 1.0, z], [0.0 + shift_x, 0.5, z],
    ]


def _build(dim, el_type, slave_pts, master_pts, mu):
    model = FEModel(dimension=dim)
    slave_nodes = _make_nodes(model, 1, slave_pts)
    master_nodes = _make_nodes(model, 100, master_pts)

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

    return MortarContact(
        "c_friction", model, nonMortarSurface="slave", mortarSurface="master",
        field="displacement", friction_coefficient=str(mu),
    )


def _prime_and_freeze(mc, U, dU, timeStep, n=7):
    """Call applyConstraint repeatedly on the same state so active_set / stick_set
    settle before the FD sweep (the set update freezes after a few iterations,
    giving one consistent linearization branch)."""
    nDof = len(U)
    for _ in range(n):
        mc.applyConstraint(U.copy(), dU.copy(), np.zeros(nDof), np.zeros((nDof, nDof)), timeStep)


def _check_tangent(mc, U0, dU0, timeStep, nDof, eps=1e-7, tol=1e-5, label=""):
    PExt0 = np.zeros(nDof)
    K0 = np.zeros((nDof, nDof))
    mc.applyConstraint(U0.copy(), dU0.copy(), PExt0, K0, timeStep)

    K_fd = np.zeros((nDof, nDof))
    for j in range(nDof):
        Up = U0.copy(); Up[j] += eps
        dUp = dU0.copy(); dUp[j] += eps  # dU tracks U_np within the increment
        Pp = np.zeros(nDof)
        mc.applyConstraint(Up, dUp, Pp, np.zeros((nDof, nDof)), timeStep)

        Um = U0.copy(); Um[j] -= eps
        dUm = dU0.copy(); dUm[j] -= eps
        Pm = np.zeros(nDof)
        mc.applyConstraint(Um, dUm, Pm, np.zeros((nDof, nDof)), timeStep)

        # Sign convention in mortarcontact.py: K[row, col] = -d(PExt[row])/d(col)
        K_fd[:, j] = -(Pp - Pm) / (2.0 * eps)

    err = np.max(np.abs(K0 - K_fd))
    print(f"  [FD tangent] {label}: max|K_analytic - K_fd| = {err:.3e}")
    assert err < tol, f"tangent inconsistent ({label}): {err:.3e}"
    return K0, PExt0


def test_2d_stick():
    print("\n=== 2D Coulomb friction: stick ===")
    mu = 0.3
    mc = _build(2, "CONLINE2", _line2(), _line2(), mu)
    nNodes = len(mc.nodes); dim = 2
    nDof = dim * nNodes + mc.nMultipliers + mc.nTangentialMultipliers
    idx_LM0 = dim * nNodes
    idx_TAU0 = idx_LM0 + mc.nNonMortarNodes

    U = np.zeros(nDof)
    for i in range(2):
        U[2 * i + 1] = -0.1     # penetration (n = [0, -1])
        U[2 * i] = 0.001        # small tangential displacement
    for i in range(2):
        U[idx_LM0 + i] = -0.05  # negative = compression (sgn_D = +1)
        U[idx_TAU0 + i] = 0.003  # |z_t| well inside the cone mu*0.05 = 0.015
    dU = U.copy()

    ts = TimeStep(1, 1.0, 1.0, 1.0, 1.0, 1.0)
    _prime_and_freeze(mc, U, dU, ts)
    assert np.all(mc.active_set), "both nodes should be active"
    assert np.all(mc.stick_set), "expected stick"
    _check_tangent(mc, U, dU, ts, nDof, label="2D stick")
    print("  [PASS]")


def test_2d_slip():
    print("\n=== 2D Coulomb friction: slip + Coulomb limit ===")
    mu = 0.3
    mc = _build(2, "CONLINE2", _line2(), _line2(), mu)
    nNodes = len(mc.nodes); dim = 2
    nDof = dim * nNodes + mc.nMultipliers + mc.nTangentialMultipliers
    idx_LM0 = dim * nNodes
    idx_TAU0 = idx_LM0 + mc.nNonMortarNodes

    U = np.zeros(nDof)
    for i in range(2):
        U[2 * i + 1] = -0.1
        U[2 * i] = 0.2          # large tangential displacement -> slip
    for i in range(2):
        U[idx_LM0 + i] = -0.05
        U[idx_TAU0 + i] = 0.04  # |z_t| exceeds mu*0.05 = 0.015 -> slip branch
    dU = U.copy()

    ts = TimeStep(2, 1.0, 1.0, 1.0, 1.0, 1.0)
    _prime_and_freeze(mc, U, dU, ts)
    assert np.all(mc.active_set)
    assert not np.any(mc.stick_set), "expected slip"
    _check_tangent(mc, U, dU, ts, nDof, label="2D slip")

    # Newton on the z_t sub-block to its root; |z_t| must equal mu*|lambda_n|.
    U_root = U.copy()
    tau_dofs = list(range(idx_TAU0, idx_TAU0 + mc.nTangentialMultipliers))
    for _ in range(30):
        Pr = np.zeros(nDof); Kr = np.zeros((nDof, nDof))
        mc.applyConstraint(U_root.copy(), U_root.copy(), Pr, Kr, ts)
        r = Pr[tau_dofs]
        if np.linalg.norm(r) < 1e-13:
            break
        U_root[tau_dofs] += np.linalg.solve(Kr[np.ix_(tau_dofs, tau_dofs)], r)
    tau_lim = mu * abs(U[idx_LM0])
    got = abs(U_root[idx_TAU0])
    print(f"  slip |z_t| at root = {got:.6e},  Coulomb limit mu*|lambda_n| = {tau_lim:.6e}")
    assert abs(got - tau_lim) < 1e-8, "slip |z_t| must equal the Coulomb limit"
    print("  [PASS]")


def test_3d_stick():
    print("\n=== 3D Coulomb friction (CONQUAD8): stick ===")
    mu = 0.25
    mc = _build(3, "CONQUAD8", _quad8(), _quad8(), mu)
    nNodes = len(mc.nodes); dim = 3
    nDof = dim * nNodes + mc.nMultipliers + mc.nTangentialMultipliers
    idx_LM0 = dim * nNodes
    idx_TAU0 = idx_LM0 + mc.nNonMortarNodes

    U = np.zeros(nDof)
    for i in range(8):
        U[3 * i + 2] = 0.05     # penetration (n = [0, 0, 1])
        U[3 * i] = 0.0005
    for i in range(8):
        U[idx_LM0 + i] = -0.02
        U[idx_TAU0 + 2 * i] = 0.0005
        U[idx_TAU0 + 2 * i + 1] = 0.0003
    dU = U.copy()

    ts = TimeStep(1, 1.0, 1.0, 1.0, 1.0, 1.0)
    _prime_and_freeze(mc, U, dU, ts)
    assert np.all(mc.active_set)
    assert np.all(mc.stick_set), "expected stick"
    _check_tangent(mc, U, dU, ts, nDof, label="3D stick")
    print("  [PASS]")


def test_3d_slip():
    print("\n=== 3D Coulomb friction (CONQUAD8): slip (non-axis-aligned) ===")
    mu = 0.25
    mc = _build(3, "CONQUAD8", _quad8(), _quad8(), mu)
    nNodes = len(mc.nodes); dim = 3
    nDof = dim * nNodes + mc.nMultipliers + mc.nTangentialMultipliers
    idx_LM0 = dim * nNodes
    idx_TAU0 = idx_LM0 + mc.nNonMortarNodes

    U = np.zeros(nDof)
    for i in range(8):
        U[3 * i + 2] = 0.05
        U[3 * i] = 0.3
        U[3 * i + 1] = 0.1      # genuinely 3D slip direction
    for i in range(8):
        U[idx_LM0 + i] = -0.02
        U[idx_TAU0 + 2 * i] = 0.02
        U[idx_TAU0 + 2 * i + 1] = 0.015
    dU = U.copy()

    ts = TimeStep(2, 1.0, 1.0, 1.0, 1.0, 1.0)
    _prime_and_freeze(mc, U, dU, ts)
    assert np.all(mc.active_set)
    assert not np.any(mc.stick_set), "expected slip"
    _check_tangent(mc, U, dU, ts, nDof, label="3D slip")
    print("  [PASS]")


if __name__ == "__main__":
    print("=" * 60)
    print("COULOMB FRICTION (saddle-point) VERIFICATION SUITE")
    print("=" * 60)
    test_2d_stick()
    test_2d_slip()
    test_3d_stick()
    test_3d_slip()
    print("\n[SUCCESS] all friction tests passed")
