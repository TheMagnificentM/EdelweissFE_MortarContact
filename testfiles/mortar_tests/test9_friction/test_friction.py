#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test 9: Coulomb Friction in the VECTORIAL mortar contact (branch 12)
====================================================================

Verifies the Coulomb friction ported into the vectorial dual-mortar
formulation (branch 6 / 12), following Gitterle, Popp, Gee & Wall (2010),
IJNME 84:543-571, and MOOSE ComputeFrictionalForceLMMechanicalContact.

Unlike the scalar-normal branch 9, the tangential traction is NOT a separate
DOF: it is the in-plane projection z_t[c] = t_c . z_I of the nodal multiplier
vector z_I in R^dim. The Coulomb stick/slip rows therefore REPLACE the
frictionless tangential rows t_a . z_I = 0 on the DOFs z0+1..z0+dim-1, and the
consistent linearization is re-expressed on z_I via the chain rule
(d z_t[c]/d z_I = t_c, d b/d z_I = db_dlam * n_I). No new DOFs are created, so
nDof = dim*nNodes + dim*nSlave.

Checks (2D CONLINE2 and 3D CONQUAD8, stick + slip + symmetry):
1. Correct stick/slip classification for the prescribed state.
2. Finite-difference tangent consistency: the assembled K must equal the true
   derivative of PExt w.r.t. ALL unknowns (displacements + z_I). With frozen
   geometry this is the exact Jacobian and must match to ~sqrt(eps).

Run:  python test_friction.py   (or via pytest)
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


def _build(dim, el_type, slave_pts, master_pts, mu, cn=1.0e6, sym=None):
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

    kwargs = dict(
        nonMortarSurface="slave", mortarSurface="master", field="displacement",
        friction_coefficient=str(mu), friction_cn=str(cn),
    )
    if sym is not None:
        set_name, comp = sym
        if not hasattr(model, "nodeSets") or model.nodeSets is None:
            model.nodeSets = {}
        model.nodeSets[set_name] = slave_nodes
        kwargs["frictionSymmetryBCs"] = f"{set_name}:{comp}"
    return MortarContact("c_friction", model, **kwargs)


def _prime_and_freeze(mc, U, dU, timeStep, n=7):
    """Call applyConstraint repeatedly on the same state so active_set / stick_set
    settle and freeze before the FD sweep (one consistent linearization branch)."""
    nDof = len(U)
    for _ in range(n):
        mc.applyConstraint(U.copy(), dU.copy(), np.zeros(nDof), np.zeros((nDof, nDof)), timeStep)


def _set_vector_multipliers(U, mc, dim, nNodes, lam, zt_scale, normals, tangents):
    """Set the vectorial multipliers z_I = lam*n_I + sum_c zt_scale[c]*t_c
    (layout z0 = dim*nNodes + dim*I)."""
    idx_LM0 = dim * nNodes
    for i in range(mc.nNonMortarNodes):
        z0 = idx_LM0 + dim * i
        zI = lam * normals[i].astype(float)
        for c in range(mc.nTangentialComponents):
            zI = zI + zt_scale[c] * tangents[i][c]
        U[z0:z0 + dim] = zI


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


def _run(dim, eltype, mkpts, disp_setter, zt_scale, mu, cn, ts_num, expect_stick,
         sym=None, label=""):
    mc = _build(dim, eltype, mkpts(), mkpts(), mu, cn=cn, sym=sym)
    nNodes = len(mc.nodes)
    nDof = dim * nNodes + mc.nMultipliers
    U = np.zeros(nDof)
    disp_setter(U, dim)
    ts = TimeStep(ts_num, 1.0, 1.0, 1.0, 1.0, 1.0)
    # prime once to freeze geometry (normals, tangents, D, sgn_D)
    mc.applyConstraint(U.copy(), U.copy(), np.zeros(nDof), np.zeros((nDof, nDof)), ts)
    normals = mc.current_normals
    tangents = mc.current_tangents
    sgn_D = np.sign(mc.current_D_rowsum[0])
    lam = -0.5 * sgn_D  # p_n = -lam*sgn_D = +0.5 (compression)
    _set_vector_multipliers(U, mc, dim, nNodes, lam, zt_scale, normals, tangents)
    dU = U.copy()
    _prime_and_freeze(mc, U, dU, ts)
    print(f"\n=== {label} ===")
    assert np.all(mc.active_set), f"{label}: all nodes should be active"
    if expect_stick:
        assert np.all(mc.stick_set), f"{label}: expected stick"
    else:
        assert not np.any(mc.stick_set), f"{label}: expected slip"
    _check_tangent(mc, U, dU, ts, nDof, label=label)
    print(f"  [PASS] {label}")


def _disp_2d(U, dim):
    for i in range(2):
        U[2 * i + 1] = -0.002   # small penetration
        U[2 * i] = 0.001        # tangential offset


def _disp_3d(U, dim):
    for i in range(8):
        U[3 * i + 2] = 0.002
        U[3 * i] = 0.0005


def test_2d_stick():
    _run(2, "CONLINE2", _line2, _disp_2d, [0.01], 0.3, 1.0, 1, True, label="2D stick")


def test_2d_slip():
    _run(2, "CONLINE2", _line2, _disp_2d, [0.5], 0.3, 0.0, 2, False, label="2D slip")


def test_3d_stick():
    _run(3, "CONQUAD8", _quad8, _disp_3d, [0.001, 0.0], 0.25, 1.0, 3, True, label="3D stick")


def test_3d_slip():
    _run(3, "CONQUAD8", _quad8, _disp_3d, [0.5, 0.3], 0.25, 0.0, 4, False, label="3D slip")


def test_3d_symmetry():
    # Fix global component 3 (z) on all slave nodes: friction restricted out of z;
    # one tangential direction becomes FIXED (z_t suppressed there).
    _run(3, "CONQUAD8", _quad8, _disp_3d, [0.4, 0.2], 0.25, 1.0, 5, False,
         sym=("z_symm", 3), label="3D symmetry")


if __name__ == "__main__":
    test_2d_stick()
    test_2d_slip()
    test_3d_stick()
    test_3d_slip()
    test_3d_symmetry()
    print("\nALLE REIBUNGS-FD-TESTS (vektoriell) ERFOLGREICH PASSIERT!")
