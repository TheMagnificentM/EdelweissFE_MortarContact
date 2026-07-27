"""
De-risking check for the QUADRATIC dual-condensation *global* transformation.

The element-level identity  D_tilde = D_phys * T^T  (diagonal) was verified in
DUAL_CONDENSATION_verify_algebra_quadratic.py. The open question before any
solver work: does it still hold on an ASSEMBLED patch where two CONQUAD8 facets
share an edge (shared corner + shared mid-side node)? This is the consistent
boundary treatment of Cichosz & Bischoff (2011).

Key point tested here: the global basis transformation T must be built
TOPOLOGICALLY (each mid-side node contributes alpha to each of its two
edge-corner neighbours exactly once), NOT by naive element-additive assembly
(which would give a shared mid node 2*alpha and break the identity).

We reuse the real ContactElement code paths (getShapeFunctions, getQuadrature-
Points, getJacobianAndAreaWeight, computeLocalMassMatrices) so the D_phys blocks
are exactly what mortarcontact.py assembles (D_block = A_e @ M0, M0 = int N N^T).
"""
import os
import sys

import numpy as np

TESTDIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(TESTDIR, "..", "..")))

from edelweissfe.elements.contactelement.element import ContactElement
from edelweissfe.points.node import Node

ALPHA = 1.0 / 3.0


def make_quad8(coords_by_local, elnum):
    """coords_by_local: list of 8 (x,y,z). Standard CONQUAD8 local order:
    corners 0..3, mids 4..7 (4:0-1, 5:1-2, 6:2-3, 7:3-0)."""
    el = ContactElement("CONQUAD8", elnum)
    nodes = [Node(i, np.array(c, dtype=float)) for i, c in enumerate(coords_by_local)]
    el.setNodes(nodes)
    return el


def facet_mass_and_dualcoeff(el):
    """M0 = int N N^T over the facet, and A_e (dual coefficients), via real code."""
    coords = np.array([nd.coordinates for nd in el.nodes])
    pts, wts = el.getQuadraturePoints()
    n = el.nNodes
    M0 = np.zeros((n, n))
    for lc, w in zip(pts, wts):
        N = el.getShapeFunctions(lc)
        jac = el.getJacobianAndAreaWeight(lc, coords)
        M0 += np.outer(N, N) * (jac * w)
    _, _, A_e = el.computeLocalMassMatrices(coords)
    return M0, A_e


def main():
    # Two flat CONQUAD8 facets sharing the edge x = 1 (z = 0 plane).
    # Facet 1: [0,1] x [0,1];  Facet 2: [1,2] x [0,1].
    f1 = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
          (.5, 0, 0), (1, .5, 0), (.5, 1, 0), (0, .5, 0)]
    f2 = [(1, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0),
          (1.5, 0, 0), (2, .5, 0), (1.5, 1, 0), (1, .5, 0)]

    el1 = make_quad8(f1, 1)
    el2 = make_quad8(f2, 2)

    # Global node bookkeeping: dedupe by coordinate.
    gcoords = []

    def gid(c):
        c = np.array(c, float)
        for i, gc in enumerate(gcoords):
            if np.linalg.norm(gc - c) < 1e-9:
                return i
        gcoords.append(c)
        return len(gcoords) - 1

    loc2glob = {}
    for el, fc in ((el1, f1), (el2, f2)):
        loc2glob[el.elNumber] = [gid(c) for c in fc]

    ng = len(gcoords)
    gcoords = np.array(gcoords)
    print(f"assembled patch: {ng} global nodes (expected 13)")

    # ---- assemble global physical D = int(Phi_a N_b) ----
    D = np.zeros((ng, ng))
    for el in (el1, el2):
        M0, A_e = facet_mass_and_dualcoeff(el)
        D_block = A_e @ M0                      # exactly mortarcontact.py's D_blk
        g = loc2glob[el.elNumber]
        D[np.ix_(g, g)] += D_block

    # ---- build global T TOPOLOGICALLY ----
    # classify global nodes as corner/mid via local roles; find each mid's 2
    # edge-corner neighbours from the element mid->corner maps.
    mid_local = {4: (0, 1), 5: (1, 2), 6: (2, 3), 7: (3, 0)}
    is_mid = np.zeros(ng, dtype=bool)
    mid_corners = {}   # global mid id -> set of global corner ids
    for el in (el1, el2):
        g = loc2glob[el.elNumber]
        for m, (c1, c2) in mid_local.items():
            gm = g[m]
            is_mid[gm] = True
            mid_corners.setdefault(gm, set()).update((g[c1], g[c2]))

    T = np.eye(ng)
    for gm, corners in mid_corners.items():
        T[gm, gm] = 1.0 - 2.0 * ALPHA
        for gc in corners:
            T[gc, gm] = ALPHA                    # set (not +=): topological, once

    # ---- the identity under test ----
    D_tilde = D @ T.T
    offdiag = D_tilde - np.diag(np.diag(D_tilde))
    max_off = np.max(np.abs(offdiag))
    scale = np.max(np.abs(np.diag(D_tilde)))
    print(f"max |offdiag(D_phys T^T)| / scale = {max_off / scale:.3e}")
    print(f"min diag(D_tilde) = {np.min(np.diag(D_tilde)):.4e}  (must be > 0)")

    rowsum = np.sum(D, axis=1)
    err_rowsum = np.max(np.abs(rowsum - np.diag(D_tilde))) / scale
    print(f"max |rowsum(D_phys) - diag(D_tilde)| / scale = {err_rowsum:.3e}  "
          f"(current_D_rowsum must equal D_tilde_II)")

    # naive element-additive T (the WRONG way) for contrast
    T_naive = np.eye(ng)
    for el in (el1, el2):
        g = loc2glob[el.elNumber]
        for m, (c1, c2) in mid_local.items():
            T_naive[g[m], g[m]] = 1.0 - 2.0 * ALPHA
            T_naive[g[c1], g[m]] += ALPHA
            T_naive[g[c2], g[m]] += ALPHA
    off_naive = np.max(np.abs((D @ T_naive.T) - np.diag(np.diag(D @ T_naive.T)))) / scale
    print(f"[contrast] naive additive T -> max offdiag/scale = {off_naive:.3e}")

    ok = (max_off / scale < 1e-10) and (np.min(np.diag(D_tilde)) > 0) and (err_rowsum < 1e-10)
    print("\nRESULT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
