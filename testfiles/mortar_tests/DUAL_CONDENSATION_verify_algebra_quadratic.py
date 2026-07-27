"""
Standalone verification of the dual-mortar CONDENSATION algebra for QUADRATIC
slave facets (e.g. CONQUAD8 from hex20), before wiring it into the solver.

Difference to the linear check (DUAL_CONDENSATION_verify_algebra.py): for
quadratic elements the physical mortar matrix D_phys = int(Phi_a * N_b) is NOT
diagonal, so the per-node scalar elimination z_I = (1/D_II)(...) of Gitterle
Eq. (85) does not apply directly. The dual-quadratic method of Popp, Wohlmuth,
Gee & Wall (2012) restores a diagonal structure via the basis transformation
N_tilde = T_e * N (the alpha = 1/3 shift already implemented in
ContactElement.getBasisTransformation): the TRANSFORMED mortar matrix

    D_tilde = int(Phi_a * N_tilde_b) = D_e   (diagonal, strictly positive)

is diagonal, and relates to the physical one by the exact identity

    D_tilde = D_phys * T_e^T          (derived from Phi = A_e N, A_e = D_e M_e^-1 T_e).

Consequences for the elimination (this is what we verify here):

  * The multiplier z couples into the slave-displacement equilibrium via
    B^T[slaveK, z_I] = -D_phys[I,K] n_I. Stacking the slave equilibrium rows and
    left-multiplying by  L[m,(K)] = T[m,K] n_K  gives, for a FLAT interface
    (all n_K equal),  L * (z-coupling) = -D_tilde  (diagonal). Hence

        z = D_tilde^-1 * T * ( n . (K_dd[S,:] d - f[S]) )                    (*)

    which is the cheap, per-node transformed elimination: one sparse mat-vec
    with T, then a diagonal scaling. EXACT for constant normals (the mortar
    contact patch test), reproducing the saddle-point solution to ~1e-12.

  * For a CURVED interface the cross terms (n_K . n_I) with K != I no longer
    vanish, so (*) is exact only up to O(normal variation). The algebraically
    exact elimination then solves the small coupled system
        M_proj[K,I] = -D_phys[I,K] (n_K . n_I)
    per contact patch (still much smaller than the bulk). We verify BOTH:
    (*) is flat-exact but curved-approximate, and the M_proj solve is exact in
    both cases. This documents the Cichosz & Bischoff (2011) boundary/curvature
    caveat honestly instead of hiding it.

The saddle-point block structure is identical to the linear check and to what
mortarcontact.py assembles.
"""
import numpy as np


def build_spd(n, shift=0.0):
    A = np.array([[np.cos(0.3 * (i + 1) * (j + 1) + shift) for j in range(n)] for i in range(n)])
    return A @ A.T + n * np.eye(n)


def build_quad8_transformation():
    """The alpha = 1/3 basis transformation T of a single CONQUAD8 facet
    (ContactElement.getBasisTransformation): corners 0..3, mid-side 4..7,
    mid_to_corners = {4:(0,1), 5:(1,2), 6:(2,3), 7:(3,0)}."""
    alpha = 1.0 / 3.0
    T = np.eye(8)
    mid_to_corners = {4: (0, 1), 5: (1, 2), 6: (2, 3), 7: (3, 0)}
    for mid, (c1, c2) in mid_to_corners.items():
        T[mid, mid] = 1.0 - 2.0 * alpha
        T[c1, mid] = alpha
        T[c2, mid] = alpha
    return T


def run(curved: bool):
    dim = 3
    n_slave, n_master, n_int = 8, 4, 3           # one quad8 slave facet
    nodes = n_slave + n_master + n_int
    ndisp = dim * nodes
    nz = n_slave

    def sl(i): return np.arange(dim * i, dim * i + dim)
    def ma(j): return np.arange(dim * (n_slave + j), dim * (n_slave + j) + dim)

    K_dd = build_spd(ndisp) * 1.0e4

    # Per-node normals. Flat: all +z (patch-test case). Curved: perturbed.
    normals = np.tile(np.array([0.0, 0.0, 1.0]), (n_slave, 1)).astype(float)
    if curved:
        for I in range(n_slave):
            normals[I] += np.array([0.15 * np.sin(I + 1), 0.12 * np.cos(2 * I + 1), 0.0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    # Diagonal TRANSFORMED mortar matrix D_tilde = D_e > 0, and the exact
    # physical D_phys = D_tilde * T^-T (so that D_tilde = D_phys * T^T holds).
    T = build_quad8_transformation()
    D_tilde = np.diag([1.0 + 0.25 * i for i in range(n_slave)])
    D_phys = D_tilde @ np.linalg.inv(T).T

    # slave-master coupling C (deterministic, non-trivial)
    C = np.array([[0.3 + 0.1 * ((i * 3 + j) % 5) for j in range(n_master)] for i in range(n_slave)])

    # Assemble B (constraint rows), same convention as mortarcontact.py:
    #   B[z_I, slaveK] = -D_phys[I,K] n_I ,  B[z_I, masterJ] = +C[I,J] n_I
    B = np.zeros((nz, ndisp))
    for I in range(n_slave):
        for K in range(n_slave):
            B[I, sl(K)] = -D_phys[I, K] * normals[I]
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

    # normal-projected slave equilibrium operator: R_slave (nz x ndisp), rp (nz,)
    R_slave = np.zeros((nz, ndisp))
    rp = np.zeros(nz)
    for K in range(n_slave):
        R_slave[K, :] = normals[K] @ K_dd[sl(K), :]
        rp[K] = normals[K] @ f[sl(K)]

    def condense(elimination):
        """elimination -> (W, wf) with z = W d - wf."""
        if elimination == "cheap":
            # z = D_tilde^-1 T (R_slave d - rp)      (Eq. * ; flat-exact)
            Dinv = np.diag(1.0 / np.diag(D_tilde))
            W = Dinv @ T @ R_slave
            wf = Dinv @ T @ rp
        else:
            # exact per-patch: M_proj z = (rp - R_slave d),  M_proj[K,I] = -D_phys[I,K](n_K.n_I)
            M_proj = np.zeros((nz, nz))
            for K in range(n_slave):
                for I in range(n_slave):
                    M_proj[K, I] = -D_phys[I, K] * (normals[K] @ normals[I])
            Minv = np.linalg.inv(M_proj)
            # M_proj z = rp - R_slave d  ->  z = Minv(rp) - Minv R_slave d
            W = -Minv @ R_slave
            wf = -Minv @ rp
        return W, wf

    def solve_condensed(W, wf):
        # Fold z(d) = W d - wf into ALL displacement rows (interior/master/slave)
        # via the rank update B^T[:,z] @ W. This is the crux: the tangential
        # slave rows also carry z-coupling (-D_phys[J,I] n_J) for non-diagonal
        # D_phys, which must be substituted out too - not only the master rows.
        Bt = B.T                                   # ndisp x nz  (= B^T[:, z])
        A_full = K_dd + Bt @ W
        b_full = f + Bt @ wf
        A_mod = A_full.copy()
        b_mod = b_full.copy()
        # slave rows: rotate the FOLDED row to (n_I, t) frame; the normal
        # component (the equation used up to define z) -> constraint B[z_I,:].
        for I in range(n_slave):
            rows = sl(I)
            n_I = normals[I]
            P = np.eye(dim) - np.outer(n_I, n_I)
            A_mod[rows, :] = P @ A_full[rows, :] + np.outer(n_I, B[I, :])
            b_mod[rows] = P @ b_full[rows] + n_I * g[I]
        d_c = np.linalg.solve(A_mod, b_mod)
        z_c = W @ d_c - wf
        return d_c, z_c

    results = {}
    for mode in ("cheap", "exact"):
        W, wf = condense(mode)
        d_c, z_c = solve_condensed(W, wf)
        err_d = np.max(np.abs(d_c - d_ref)) / max(np.max(np.abs(d_ref)), 1e-30)
        err_z = np.max(np.abs(z_c - z_ref)) / max(np.max(np.abs(z_ref)), 1e-30)
        results[mode] = (err_d, err_z)
    return results


def main():
    print("=" * 68)
    print("Quadratic (CONQUAD8) dual-condensation algebra check")
    print("  D_tilde = D_phys * T^T  (transformed mortar matrix is diagonal)")
    print("=" * 68)

    ok = True
    for curved in (False, True):
        label = "CURVED interface (varying normals)" if curved else "FLAT interface (constant normal = patch test)"
        res = run(curved)
        print(f"\n{label}")
        for mode in ("cheap", "exact"):
            err_d, err_z = res[mode]
            tag = "diagonal per-node z=D_tilde^-1 T(...)" if mode == "cheap" else "exact per-patch M_proj solve   "
            print(f"  [{tag}]  rel err d = {err_d:.3e},  rel err z = {err_z:.3e}")

        # Acceptance: the diagonal per-node elimination must be machine-exact on
        # the flat (patch-test) interface; the exact per-patch solve must be
        # machine-exact in both cases.
        if not curved:
            ok = ok and res["cheap"][0] < 1e-9 and res["cheap"][1] < 1e-9
        ok = ok and res["exact"][0] < 1e-9 and res["exact"][1] < 1e-9

    print("\n" + "-" * 68)
    print("Interpretation:")
    print("  * FLAT: diagonal per-node elimination is EXACT -> guarantees the")
    print("    quad8 patch test (constant pressure) will pass after wiring.")
    print("  * CURVED: diagonal per-node elimination carries an O(normal")
    print("    variation) error; the exact per-patch solve stays machine-exact.")
    print("    -> Cichosz & Bischoff (2011) boundary/curvature treatment matters")
    print("       for curved contact; flat/mildly-curved (e.g. POT) is fine.")
    print("-" * 68)
    print("RESULT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
