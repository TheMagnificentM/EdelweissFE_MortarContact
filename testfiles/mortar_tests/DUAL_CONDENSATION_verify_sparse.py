"""
Verification of the SPARSE solver-level condensation routine
``edelweissfe.solvers.base.mortarcondensation.condenseMortarMultipliers``
against a direct dense solve of the full saddle-point system.

This is the sparse counterpart of DUAL_CONDENSATION_verify_algebra.py: it
builds a saddle-point system with the SAME block layout that
mortarcontact.py assembles (including an INACTIVE multiplier, whose row is
the assembled identity z=0 and which must be left untouched), feeds the
assembled (K_csr, R) through the production condensation function, solves
the resulting displacement-form system, and checks that both the
displacements d and the recovered multipliers z match the reference
saddle-point solution to ~1e-10.

Convention (Gitterle et al. 2010, matching mortarcontact.py):
  B[z_I, slaveK]  = -D_IK n_I ,  B[z_I, masterJ] = +C_IJ n_I ,  D diagonal.
  Newton system  K ddU = R :   [ A  B^T ; B  C_ll ] [d; z] = [f; g].
  Active node   -> C_ll row = 0        (the constraint  B d = g).
  Inactive node -> C_ll row = identity (z = 0), no B / B^T coupling.
"""
import os
import sys

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from edelweissfe.solvers.base.mortarcondensation import condenseMortarMultipliers


def build_spd(n, shift=0.0):
    A = np.array([[np.cos(0.3 * (i + 1) * (j + 1) + shift) for j in range(n)] for i in range(n)])
    return A @ A.T + n * np.eye(n)


def main():
    dim = 3
    n_slave, n_master, n_int = 3, 2, 2
    nodes = n_slave + n_master + n_int
    ndisp = dim * nodes
    nz = n_slave
    ntot = ndisp + nz

    def sl(i):
        return np.arange(dim * i, dim * i + dim)

    def ma(j):
        return np.arange(dim * (n_slave + j), dim * (n_slave + j) + dim)

    # global multiplier DOF indices packed after all displacement DOFs
    def zi(i):
        return ndisp + i

    A = build_spd(ndisp) * 1.0e4

    normals = np.array([[0.0, 0.0, 1.0], [0.2, 0.0, 1.0], [0.0, 0.1, 1.0]])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    D = np.array([1.0 + 0.3 * i for i in range(n_slave)])   # diagonal entries
    C = np.array([[0.6, 0.4], [0.5, 0.5], [0.7, 0.3]])

    # Slave node 2 is INACTIVE (z=0 identity row); nodes 0,1 are active.
    active = [True, True, False]

    Ktot = np.zeros((ntot, ntot))
    Ktot[:ndisp, :ndisp] = A
    f = np.array([np.sin(0.7 * (i + 1)) for i in range(ndisp)]) * 1.0e2
    g = np.array([0.05 * (I + 1) for I in range(nz)])
    rhs = np.zeros(ntot)
    rhs[:ndisp] = f

    for I in range(n_slave):
        if active[I]:
            # B (z-row) and B^T (z-col)
            Ktot[zi(I), sl(I)] += -D[I] * normals[I]
            Ktot[sl(I), zi(I)] += -D[I] * normals[I]
            for J in range(n_master):
                Ktot[zi(I), ma(J)] += C[I, J] * normals[I]
                Ktot[ma(J), zi(I)] += C[I, J] * normals[I]
            rhs[zi(I)] = g[I]
        else:
            Ktot[zi(I), zi(I)] += 1.0     # identity row -> z_I = g_I (=0 physically)
            rhs[zi(I)] = 0.0

    # reference solution of the full saddle-point system
    sol_ref = np.linalg.solve(Ktot, rhs)
    d_ref = sol_ref[:ndisp]
    z_ref = sol_ref[ndisp:]

    # ---- run the production sparse condensation ----
    operators = []
    for I in range(n_slave):
        operators.append(
            {
                "z_idx": zi(I),
                "slave_dofs": sl(I),
                "normal": normals[I],
                "D_diag": D[I],
                "active": active[I],
            }
        )

    Kcsr = csr_matrix(Ktot)
    Kc, Rc = condenseMortarMultipliers(Kcsr, rhs.copy(), operators)
    sol_c = np.linalg.solve(Kc.toarray(), Rc)
    d_c = sol_c[:ndisp]
    z_c = sol_c[ndisp:]

    err_d = np.max(np.abs(d_c - d_ref)) / max(np.max(np.abs(d_ref)), 1e-30)
    err_z = np.max(np.abs(z_c - z_ref)) / max(np.max(np.abs(z_ref)), 1e-30)
    print(f"rel err d = {err_d:.3e}  (|d_ref|~{np.max(np.abs(d_ref)):.2e})")
    print(f"rel err z = {err_z:.3e}  (|z_ref|~{np.max(np.abs(z_ref)):.2e})")
    print(f"inactive z (must be ~0): z_ref[2]={z_ref[2]:.2e}, z_c[2]={z_c[2]:.2e}")
    ok = err_d < 1e-9 and err_z < 1e-9
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if main() else 1)
