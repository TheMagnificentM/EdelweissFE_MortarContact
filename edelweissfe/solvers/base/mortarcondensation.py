#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _         _____ _____
# | ____|__| | ___| |_      _____(_)___ ___|  ___| ____|
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |_  |  _|
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \  _| | |___
# |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|   |_____|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  2017 - today
#
#  This file is part of EdelweissFE.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissFE.
#  ---------------------------------------------------------------------
r"""
Static (dual) condensation of mortar-contact Lagrange multipliers -- vectorial,
edge/kink-robust, friction-ready.

This module eliminates the discrete nodal Lagrange multipliers of a dual-mortar
contact constraint from the assembled saddle-point Newton system, yielding a
displacement-dominated system solved by the existing direct solver; the
multipliers are recovered afterwards.

Formulation
-----------
The multiplier is the **full nodal traction vector** ``z_I in R^dim`` (global
components), not a scalar normal pressure. Following

    M. Gitterle, A. Popp, M. W. Gee, W. A. Wall (2010),
    "Finite deformation frictional mortar contact using a semi-smooth Newton
    method with consistent linearization", Int. J. Numer. Methods Eng.
    84:543-571,

the slave force-equilibrium rows carry ``D_tilde_II z_I`` (dual shape functions
render the transformed mortar matrix ``D_tilde = D_phys T^T`` diagonal, Popp,
Wohlmuth, Gee & Wall 2012). Eq. (85) solves those rows for ``z_I``:

    z_I = D_tilde_II^-1 * sum_K T_IK ( A[s_K,:] d - f[s_K] )   =  W_I d - wf_I,

where ``A[s_K,:]`` are the pure-bulk (multiplier-column-stripped) slave
equilibrium rows of node ``K`` and ``f[s_K]`` the corresponding residual. This
elimination is **vectorial and projection-free**: unlike a scalar
normal-pressure elimination (``z_I = (1/D_II)(n_I . A d - n_I . f)``) it never
projects onto a single nodal normal, so it stays valid where the nodal normal is
not well defined -- across **edges/kinks** of the contact surface -- and it is
the exact structure the frictional extension needs (the tangential components of
``z_I`` become the Coulomb stick/slip tractions).

Substituting ``z = W d - wf`` (Eq. 86) folds ``B^T[:, z] W`` into the master rows
(mortar projector ``M_hat = D^-1 M``, Eq. 87). The active slave *displacement*
rows are replaced by the contact constraints in the local frame ``(n_I, t_a)``:
the normal direction carries the weighted normal gap, the tangential directions
the *exact* condensed tangential equation -- the tangential multiplier-row
residual ``D_tilde_II * R[z_a]`` is retained (a plain tangential-equilibrium
replacement would drop it, which fails across edges/kinks where the tangential
traction is non-zero during the iteration). The multiplier rows are replaced by
the recovery equation ``z_I - W_I d = -wf_I`` (the ``+1`` diagonal keeps the
system well-posed for the direct solver). The multiplier DOFs are *kept* in the
system (block-triangular), so the DofManager / CSR structure is untouched, the
system has the same size, and the solver returns ``z`` directly in the solution
vector; the displacement sub-block is the positive-definite Schur complement of
Eq. (86).

The frictionless-only equivalent of the elimination is Popp et al. (2012) /
Farah (2018), Sec. 3.5.3, Eqs. (3.62)-(3.65). For linear slave facets ``T`` is
the identity and ``D_tilde_II = D_II`` is the ordinary diagonal mortar entry.

Limitation
----------
The contact constraints are placed in the slave *displacement* rows. A Dirichlet
boundary condition applied directly to a slave contact-surface DOF therefore
overwrites the constraint component in that direction. This is consistent (and
correct) when the constrained direction is tangential (zero tangential traction
== no tangential motion), but over-constrains the contact if a Dirichlet BC is
imposed along a direction with a non-zero contact-normal component on the contact
surface. In practice contact surfaces are left free; do not Dirichlet-fix contact
nodes along the (possibly kinked) contact normal.

Safety checks
-------------
A per-node consistency check verifies that the transform-combined assembled
multiplier coupling equals ``-D_tilde_II I`` (a sign/topology/transform check
that -- being purely algebraic -- holds regardless of surface curvature). A
floor guards barely-covered nodes with a degenerate ``D_tilde_II``. There is
deliberately **no** curvature guard: the vectorial elimination does not assume a
flat interface.
"""

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags

# Relative tolerance of the algebraic consistency check
#   sum_K T_IK * K[slave_dofs_K, z_idx_I]  ?=  -D_tilde_II * I_dim.
# A mismatch signals a sign, topology or transformation-row error. It is
# curvature-independent (no normal enters), so it is a pure assembly check.
_D_CONSISTENCY_TOL = 1e-8

# Floor below which |D_tilde_II| is considered degenerate (barely-covered slave
# node). Such a node cannot be condensed by division; guarded here as in the
# saddle-point branch of mortarcontact.py.
_D_FLOOR = 1e-12


class MortarCondensationError(RuntimeError):
    """Raised when the assembled system is inconsistent with the diagonal-D
    elimination assumed by :func:`condenseMortarMultipliers`."""


def condenseMortarMultipliers(
    Kcsr: csr_matrix,
    R: np.ndarray,
    operators: list[dict],
    dirichletIndices: np.ndarray = None,
    stripZ: np.ndarray = None,
) -> tuple[csr_matrix, np.ndarray]:
    r"""Condense the active vectorial Lagrange multipliers of one or more mortar
    contact constraints out of the assembled saddle-point system.

    Parameters
    ----------
    Kcsr
        The assembled system matrix (scipy CSR), full (vectorial) saddle-point
        form.
    R
        The assembled residual / right-hand side (length ``Kcsr.shape[0]``).
    operators
        One dict per **active** slave node of every condensable constraint, with
        resolved *global* DOF indices:

        - ``"z_idx"``      : np.ndarray[int], the dim global multiplier DOFs z_I
          (component 0 == the normal-gap constraint row, 1.. == tangential rows).
        - ``"slave_dofs"`` : np.ndarray[int], the dim global displacement DOFs of
          the slave node.
        - ``"normal"``     : np.ndarray[float], the unit normal n_I (len dim).
        - ``"D_diag"``     : float, the transformed diagonal mortar entry
          D_tilde_II.
        - ``"transform"``  : list of ``(slave_dofs_K, coeff)`` with
          ``slave_dofs_K`` the dim global slave DOFs of node K and ``coeff`` the
          transformation weight T_IK (``[(slave_dofs_I, 1.0)]`` for linear
          facets).
    dirichletIndices
        Optional global indices whose right-hand side must be preserved exactly
        (Dirichlet BCs take precedence over the condensation).

    Returns
    -------
    tuple[csr_matrix, np.ndarray]
        The condensed system matrix and right-hand side. If no active multiplier
        is present, the inputs are returned unchanged.
    """
    if not operators:
        return Kcsr, R

    n = Kcsr.shape[0]
    dim = len(operators[0]["normal"])
    R = np.asarray(R, dtype=np.float64).copy()

    activeZ = np.concatenate([np.asarray(op["z_idx"], dtype=int) for op in operators])

    # Column mask that zeroes the multiplier columns, used to slice the pure-bulk
    # block A out of the assembled rows. It must zero ALL contact multiplier
    # columns (active-condensed, inactive, and Dirichlet-skipped-saddle ones):
    # for quadratic facets a condensed node's transform references neighbour slave
    # rows that may couple to non-condensed multipliers, and those couplings must
    # not leak into the pure-bulk A. `stripZ` carries the full set; it falls back
    # to `activeZ` (exact for linear facets, where a node couples only to its own
    # multiplier).
    stripCols = np.asarray(stripZ, dtype=int) if stripZ is not None else activeZ
    dispColMask = np.ones(n, dtype=np.float64)
    dispColMask[stripCols] = 0.0
    Ddisp = diags(dispColMask, format="csr")

    nA = len(operators)

    # ------------------------------------------------------------------
    # 1) Vectorial elimination  z = W d - wf  (Gitterle Eq. 85), dim rows per
    #    active node, from the transform-combined pure-bulk slave equilibrium
    #    rows. No normal projection -> valid across edges/kinks.
    # ------------------------------------------------------------------
    W_rows, W_cols, W_vals = [], [], []
    wf = np.zeros(nA * dim)
    Ident = np.eye(dim)

    for k, op in enumerate(operators):
        zIdx = np.asarray(op["z_idx"], dtype=int)
        D_II = float(op["D_diag"])
        if abs(D_II) < _D_FLOOR:
            raise MortarCondensationError(
                f"Degenerate transformed mortar entry D_tilde_II={D_II:g} at active slave "
                f"DOFs {np.asarray(op['slave_dofs']).tolist()}; cannot condense "
                f"(barely-covered node?)."
            )

        A_comb = csr_matrix((dim, n), dtype=np.float64)  # transform-combined bulk rows
        f_comb = np.zeros(dim)
        B_II = np.zeros((dim, dim))  # transform-combined multiplier coupling
        for sdofsK, coeff in op["transform"]:
            sdofsK = np.asarray(sdofsK, dtype=int)
            subRows = Kcsr[sdofsK, :] @ Ddisp  # dim x n, pure bulk (z cols stripped)
            A_comb = A_comb + coeff * subRows
            f_comb = f_comb + coeff * R[sdofsK]
            B_II += coeff * Kcsr[sdofsK, :][:, zIdx].toarray()

        # Consistency: transform-combined coupling must be -D_tilde_II I. This is
        # a pure assembly/topology check (no normal enters -> curvature-agnostic).
        err = np.max(np.abs(B_II + D_II * Ident))
        if err > _D_CONSISTENCY_TOL * max(abs(D_II), 1.0):
            raise MortarCondensationError(
                f"Inconsistent transformed mortar diagonal: constraint D_tilde_II={D_II:g} "
                f"but assembled -sum_K T_IK K[s_K, z_I]={-B_II.diagonal()} "
                f"(off-diagonal max {np.max(np.abs(B_II - np.diag(B_II.diagonal()))):g}). "
                f"Signals a sign, topology or transformation-row error."
            )

        Wblock = (A_comb / D_II).tocoo()
        W_rows.extend((k * dim + Wblock.row).tolist())
        W_cols.extend(Wblock.col.tolist())
        W_vals.extend(Wblock.data.tolist())
        wf[k * dim : (k + 1) * dim] = f_comb / D_II

    W = coo_matrix((W_vals, (W_rows, W_cols)), shape=(nA * dim, n)).tocsr()

    # ------------------------------------------------------------------
    # 2) Fold z = W d - wf into the master rows (mortar projector
    #    M_hat = D^-1 M, Eq. 87). Slave and multiplier rows are overwritten below.
    # ------------------------------------------------------------------
    Bt = Kcsr[:, activeZ].tocsr()  # n x (nA*dim)
    K1 = (Kcsr + Bt @ W).tocsr()
    R1 = R + np.asarray(Bt @ wf).ravel()

    # ------------------------------------------------------------------
    # 3) Zero the rows that will be replaced (active slave disp rows + active z
    #    rows) and the active z columns (their coupling has been folded in).
    # ------------------------------------------------------------------
    slaveDofsAll = np.concatenate([np.asarray(op["slave_dofs"], dtype=int) for op in operators])
    replacedRows = np.concatenate([slaveDofsAll, activeZ])
    rowMask = np.ones(n, dtype=np.float64)
    rowMask[replacedRows] = 0.0
    colMask = np.ones(n, dtype=np.float64)
    colMask[activeZ] = 0.0
    K2 = (diags(rowMask, format="csr") @ K1 @ diags(colMask, format="csr")).tocsr()

    # ------------------------------------------------------------------
    # 4) Replacement rows (local frame n_I, t_a; z-dof 0 == normal gap, z-dof
    #    a>0 == tangential). After z is eliminated from the slave EQUILIBRIUM
    #    rows (-> recovery), the slave displacements are governed ONLY by the
    #    contact CONSTRAINTS (the multiplier rows), not by equilibrium:
    #      slave disp rows s_I : n_I (x) Brow_n  +  P_I W_I      (P_I = I - n n^T)
    #          normal dir  -> weighted normal gap  Brow_n d = R[z0],
    #          tangential  -> condensed tangential constraint t_a . z = 0, i.e.
    #                         (t_a . W) d = R[z_a] + t_a . wf.
    #      multiplier rows z_I : z_I - W_I d = -wf_I   (recovery, Eq. 85; the +1
    #          diagonal keeps the system well-posed for the direct solver).
    #    Mixing in the tangential equilibrium here would be wrong: equilibrium is
    #    already consumed by the recovery, and the two can cancel in the
    #    tangential subspace (masking an unconverged constraint at edges/kinks).
    # ------------------------------------------------------------------
    rep_rows, rep_cols, rep_vals = [], [], []
    R2 = R1
    for k, op in enumerate(operators):
        sdofs = np.asarray(op["slave_dofs"], dtype=int)
        zIdx = np.asarray(op["z_idx"], dtype=int)
        n_I = np.asarray(op["normal"], dtype=np.float64)
        tangents = [np.asarray(t, dtype=np.float64) for t in op["tangents"]]
        P = Ident - np.outer(n_I, n_I)  # tangential projector

        Wnode = W[k * dim : (k + 1) * dim, :]  # dim x n, the node's recovery rows
        wf_node = wf[k * dim : (k + 1) * dim]

        # normal-gap constraint row (multiplier row z0), z-columns stripped
        Brow_n = (Kcsr[[zIdx[0]], :] @ Ddisp).tocsr()  # 1 x n

        slaveBlock = (csr_matrix(n_I.reshape(dim, 1)) @ Brow_n) + (csr_matrix(P) @ Wnode)
        slaveBlock = slaveBlock.tocoo()
        rep_rows.extend(sdofs[slaveBlock.row].tolist())
        rep_cols.extend(slaveBlock.col.tolist())
        rep_vals.extend(slaveBlock.data.tolist())

        # RHS: normal gap residual (normal dir) + condensed tangential constraint
        #   residual (tangential dir): n R[z0] + sum_a t_a R[z_a] + P wf.
        tang_res = np.zeros(dim)
        for a, t_a in enumerate(tangents):
            tang_res += t_a * R[zIdx[1 + a]]
        R2[sdofs] = n_I * R[zIdx[0]] + tang_res + P @ wf_node

        # recovery rows z_I - W_I d = -wf_I
        for c in range(dim):
            wrow = W.getrow(k * dim + c).tocoo()
            rep_rows.extend([zIdx[c]] * wrow.nnz)
            rep_cols.extend(wrow.col.tolist())
            rep_vals.extend((-wrow.data).tolist())
            rep_rows.append(zIdx[c])
            rep_cols.append(zIdx[c])
            rep_vals.append(1.0)
        R2[zIdx] = -wf[k * dim : (k + 1) * dim]

    Rrep = coo_matrix((rep_vals, (rep_rows, rep_cols)), shape=(n, n)).tocsr()
    K3 = (K2 + Rrep).tocsr()

    # Dirichlet BCs take precedence: restore their untouched right-hand side.
    if dirichletIndices is not None and len(dirichletIndices):
        R2[dirichletIndices] = R[dirichletIndices]

    # Match the format expected by the direct solvers.
    K3.sum_duplicates()
    K3.sort_indices()
    K3.eliminate_zeros()
    K3.indices = K3.indices.astype(np.int32, copy=False)
    K3.indptr = K3.indptr.astype(np.int32, copy=False)
    K3.data = np.ascontiguousarray(K3.data, dtype=np.float64)

    return K3, R2
