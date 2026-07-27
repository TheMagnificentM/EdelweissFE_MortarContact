#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Condensation WITH Coulomb friction: condensed == saddle (linear-algebra check)
==============================================================================

Verifies the static condensation of the FRICTIONAL mortar rows (Gitterle, Popp,
Gee & Wall 2010, Eqs. 85/86, rows St/Sl). The friction constraint-row Jacobian
(dC_t/dz, dC_t/dd, FD-verified in test9_friction) is folded by z = W d - wf
(Eq. 85) and the folded rows are rotated into the slave displacement DOFs by the
local frame Q = [n_I, t_1, .., t_{dim-1}] (Eq. 86). The condensation is an exact
Gaussian elimination, so the reduced-system displacement solution AND the
recovered multipliers must equal the full saddle solution to machine precision.

Linear 2D CONLINE2 patch (transform = identity); the quadratic transform is
already verified frictionless in condensation_test.py (hex20 ~1e-14). Stick+slip.
"""
import os
import sys

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test9_friction")))

from test_friction import _build, _prime_and_freeze, _line2  # noqa: E402
from edelweissfe.timesteppers.timestep import TimeStep  # noqa: E402
from edelweissfe.solvers.base.mortarcondensation import condenseMortarMultipliers  # noqa: E402


def _build_case(zt_scale, mu, cn, ts_num):
    dim = 2
    mc = _build(dim, "CONLINE2", _line2(), _line2(), mu, cn=cn)
    nNodes = len(mc.nodes)
    nSlave = mc.nNonMortarNodes
    nDof = dim * nNodes + mc.nMultipliers
    idx_LM0 = dim * nNodes

    U = np.zeros(nDof)
    for i in range(nSlave):
        U[2 * i + 1] = -0.002
        U[2 * i] = 0.001
    ts = TimeStep(ts_num, 1.0, 1.0, 1.0, 1.0, 1.0)
    mc.applyConstraint(U.copy(), U.copy(), np.zeros(nDof), np.zeros((nDof, nDof)), ts)
    normals, tangents = mc.current_normals, mc.current_tangents
    sgn_D = np.sign(mc.current_D_rowsum[0])
    lam = -0.5 * sgn_D  # p_n = +0.5 (compression)
    for i in range(nSlave):
        z0 = idx_LM0 + dim * i
        zI = lam * normals[i].astype(float)
        for c in range(mc.nTangentialComponents):
            zI = zI + zt_scale[c] * tangents[i][c]
        U[z0:z0 + dim] = zI
    dU = U.copy()
    _prime_and_freeze(mc, U, dU, ts)

    Pc = np.zeros(nDof)
    Kc = np.zeros((nDof, nDof))
    mc.applyConstraint(U.copy(), dU.copy(), Pc, Kc, ts)

    nDisp = dim * nNodes
    K = Kc.copy()
    K[:nDisp, :nDisp] += np.eye(nDisp)  # SPD bulk so the disp block is regular
    R = Pc.copy()
    R[:nDisp] += 0.01 * np.sin(0.3 * np.arange(nDisp) + 1.0)  # arbitrary disp load

    ops, active_z = [], []
    for I in range(nSlave):
        z_idx = np.array([idx_LM0 + dim * I + c for c in range(dim)], dtype=int)
        active_z.extend(z_idx.tolist())
        if not mc.active_set[I]:
            continue
        slave_dofs = np.array([dim * I + c for c in range(dim)], dtype=int)
        ops.append({
            "z_idx": z_idx, "slave_dofs": slave_dofs, "normal": normals[I],
            "tangents": [tangents[I][c] for c in range(mc.nTangentialComponents)],
            "D_diag": float(mc.current_D_rowsum[I]),
            "transform": [(slave_dofs, 1.0)], "active": True,
        })
    return mc, K, R, ops, np.array(active_z, dtype=int), nDisp


def _run(label, zt_scale, mu, cn, ts_num, expect_stick):
    mc, K, R, ops, stripZ, nDisp = _build_case(zt_scale, mu, cn, ts_num)
    x_saddle = np.linalg.solve(K, R)
    K3, R3 = condenseMortarMultipliers(csr_matrix(K), R.copy(), ops, stripZ=stripZ)
    x_cond = np.linalg.solve(K3.toarray(), R3)

    z_all = np.concatenate([op["z_idx"] for op in ops])
    d_err = np.max(np.abs(x_cond[:nDisp] - x_saddle[:nDisp]))
    z_err = np.max(np.abs(x_cond[z_all] - x_saddle[z_all]))
    dref = max(np.max(np.abs(x_saddle[:nDisp])), 1e-30)
    zref = max(np.max(np.abs(x_saddle[z_all])), 1e-30)
    print(f"=== {label} ===  stick={mc.stick_set.tolist()}")
    print(f"  rel|d_cond-d_saddle|={d_err/dref:.3e}   |z_cond-z_saddle|={z_err:.3e}")
    if expect_stick:
        assert np.all(mc.stick_set), f"{label}: expected stick"
    else:
        assert not np.any(mc.stick_set), f"{label}: expected slip"
    assert d_err / dref < 1e-9, f"{label}: condensed displacement != saddle"
    assert z_err / zref < 1e-9, f"{label}: recovered multiplier != saddle"
    print(f"  [PASS] {label}")


def test_condensation_friction_stick():
    _run("2D friction STICK", [0.01], 0.3, 1.0, 1, True)


def test_condensation_friction_slip():
    _run("2D friction SLIP", [0.5], 0.3, 0.0, 2, False)


if __name__ == "__main__":
    test_condensation_friction_stick()
    test_condensation_friction_slip()
    print("\nKONDENSATION MIT REIBUNG == SATTELPUNKT: ERFOLGREICH!")
