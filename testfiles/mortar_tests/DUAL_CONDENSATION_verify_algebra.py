"""
Standalone verification of the dual-mortar CONDENSATION algebra (frictionless,
normal contact), Gitterle et al. (2010) Eqs. (85)/(86), before wiring it into
the EdelweissFE solver.

Saddle-point system with the SAME block structure mortarcontact.py assembles:
    [ K_dd   B^T ] [ d ]   [ f ]
    [ B      0   ] [ z ] = [ g ]
per active slave node I (scalar normal multiplier z_I, unit normal n_I):
  B[z_I, slaveK] = -D_IK n_I ,  B[z_I, masterJ] = +C_IJ n_I   (D diagonal)
K_dd is SPD "bulk".

Dual condensation:
  slave-disp equilibrium row projected on n_I  (D diagonal -> per-node scalar):
      n_I.K_dd[sI,:] d - D_II z_I = n_I.f_sI
   => z_I = (1/D_II)( n_I.K_dd[sI,:] d - n_I.f_sI ).                       (Eq. 85)
  z appears ONLY in the master rows (B^T[M,z_I]=C n) and in the slave normal
  rows (replaced by the constraint). Interior and slave-tangential rows have no
  z coupling (n.t = 0). Reduced system (Eq. 86):
   - interior N rows:            K_dd[N,:] d = f_N
   - master M rows:              K_dd[M,:] d + sum_I C-coupling * z_I(d) = f_M
   - slave I tangential rows:    (I - n n^T) K_dd[sI,:] d = (I - n n^T) f_sI
   - slave I normal row:         B[z_I,:] d = g_I
  Recover z afterwards from (85).
"""
import numpy as np


def build_spd(n, shift=0.0):
    A = np.array([[np.cos(0.3 * (i + 1) * (j + 1) + shift) for j in range(n)] for i in range(n)])
    return A @ A.T + n * np.eye(n)


def main():
    dim = 3
    n_slave, n_master, n_int = 3, 2, 2
    nodes = n_slave + n_master + n_int
    ndisp = dim * nodes
    nz = n_slave

    def sl(i): return np.arange(dim * i, dim * i + dim)
    def ma(j): return np.arange(dim * (n_slave + j), dim * (n_slave + j) + dim)

    K_dd = build_spd(ndisp) * 1.0e4

    normals = np.array([[0.0, 0.0, 1.0], [0.2, 0.0, 1.0], [0.0, 0.1, 1.0]])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    D = np.diag([1.0 + 0.3 * i for i in range(n_slave)])          # dual -> diagonal
    C = np.array([[0.6, 0.4], [0.5, 0.5], [0.7, 0.3]])

    B = np.zeros((nz, ndisp))
    for I in range(n_slave):
        for K in range(n_slave):
            B[I, sl(K)] = -D[I, K] * normals[I]
        for J in range(n_master):
            B[I, ma(J)] = +C[I, J] * normals[I]

    Ktot = np.zeros((ndisp + nz, ndisp + nz))
    Ktot[:ndisp, :ndisp] = K_dd
    Ktot[:ndisp, ndisp:] = B.T
    Ktot[ndisp:, :ndisp] = B

    f = np.array([np.sin(0.7 * (i + 1)) for i in range(ndisp)]) * 1.0e2
    g = np.array([0.05 * (I + 1) for I in range(nz)])
    rhs = np.concatenate([f, g])

    sol = np.linalg.solve(Ktot, rhs)
    d_ref, z_ref = sol[:ndisp], sol[ndisp:]

    # ------- condensation -------
    # z(d)_I = (1/D_II)( n_I . K_dd[sI,:] d - n_I . f_sI )
    WK = np.zeros((nz, ndisp))     # z = WK d - wf
    wf = np.zeros(nz)
    for I in range(n_slave):
        rows = sl(I)
        WK[I, :] = (normals[I] @ K_dd[rows, :]) / D[I, I]
        wf[I] = (normals[I] @ f[rows]) / D[I, I]

    A_mod = K_dd.copy()
    b_mod = f.copy()

    # master rows: add B^T[M,z]*z(d).  B^T[maJ, z_I] = C[I,J] n_I
    for J in range(n_master):
        rows = ma(J)
        for I in range(n_slave):
            coupl = C[I, J] * normals[I]          # (dim,)  = B^T[maJ, z_I]
            A_mod[np.ix_(rows, np.arange(ndisp))] += np.outer(coupl, WK[I, :])
            b_mod[rows] += coupl * wf[I]

    # slave rows: rotate to (n, t) frame per node
    for I in range(n_slave):
        rows = sl(I)
        n_I = normals[I]
        P = np.eye(dim) - np.outer(n_I, n_I)      # tangential projector
        # tangential equilibrium (dim-1 eqs, embedded) + normal constraint (1 eq)
        A_mod[rows, :] = P @ K_dd[rows, :] + np.outer(n_I, B[I, :])
        b_mod[rows] = P @ f[rows] + n_I * g[I]

    d_c = np.linalg.solve(A_mod, b_mod)
    z_c = WK @ d_c - wf

    err_d = np.max(np.abs(d_c - d_ref)) / max(np.max(np.abs(d_ref)), 1e-30)
    err_z = np.max(np.abs(z_c - z_ref)) / max(np.max(np.abs(z_ref)), 1e-30)
    print(f"rel err d = {err_d:.3e}  (|d_ref|~{np.max(np.abs(d_ref)):.2e})")
    print(f"rel err z = {err_z:.3e}  (|z_ref|~{np.max(np.abs(z_ref)):.2e})")
    print("RESULT:", "PASS" if (err_d < 1e-9 and err_z < 1e-9) else "FAIL")


if __name__ == "__main__":
    main()
