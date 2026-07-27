# Dual Condensation of Mortar Contact Lagrange Multipliers — Design Note

Literature-grounded design for eliminating the (normal + tangential) Lagrange
multipliers from the mortar-contact saddle-point system, to obtain a robust,
well-conditioned displacement-only system.

## Why (recap of the diagnosis)

The un-condensed saddle-point system converges for frictionless normal contact
but the **frictional** tangential-multiplier rows are `c_t`-scaled and
asymmetric, which wrecks the conditioning and blows up the multipliers once
slip is triggered (confirmed for c_t = 1, 100, 1000, E, both with GCDP and with
linear-elastic concrete). Gitterle, Popp, Gee & Wall (2010) obtain robustness
by **condensing** the multipliers: their reduced system (Eq. 86) is positive
definite. MOOSE instead keeps explicit multipliers but relies on PETSc
variational-inequality solvers + automatic variable scaling, which our direct
(Pardiso) solver + additive constraint interface do not provide. Hence
condensation is the appropriate path for EdelweissFE.

## Authoritative reference

Gitterle, Popp, Gee & Wall (2010), *Finite deformation frictional mortar
contact using a semi-smooth Newton method with consistent linearization*,
IJNME 84:543–571. Full frictional saddle-point system Eq. (84), local
elimination Eq. (85), condensed system Eq. (86), projection operator Eq. (87).
(Frictionless-only equivalent: Farah 2018 Eqs. 3.62–3.65; Popp et al. 2012.)

## Node-set partition (per semi-smooth-Newton iteration)

All DOFs are split into (Gitterle Eq. 81):
- `N`  : interior displacement nodes (no contact)
- `M`  : master (mortar) displacement nodes
- `I`  : inactive slave nodes            (z_j = 0)
- `St` : active + sticking slave nodes
- `Sl` : active + slipping slave nodes
- `A = St ∪ Sl` : active slave nodes
- `S = I ∪ St ∪ Sl` : all slave nodes

`z_j` is the **nodal multiplier vector** (normal + tangential components) at
slave node j. The nodal contact force on the slave is `D_jj z_j`; on the master
it is `-(M^T z)_l`. Because dual shape functions make `D` **diagonal**, `D^-1`
is trivial.

## Full saddle-point system (Gitterle Eq. 84)

Unknowns `[Δd_N, Δd_M, Δd_I, Δd_St, Δd_Sl, z_I, z_St, z_Sl]`, RHS `-[...]`:

```
row N (interior eq):   K_NN Δd_N + K_NM Δd_M + K_NI Δd_I + K_NSt Δd_St + K_NSl Δd_Sl                         = -r_N
row M (master eq):     K_MN + K̃_MM + K̃_MI + K̃_MSt + K̃_MSl  - M_I^T z_I - M_St^T z_St - M_Sl^T z_Sl        = -r_M
row I (slave eq):      K_IN + K̃_IM + K̃_II + K̃_ISt + K̃_ISl  + D_I  z_I                                      = -r_I
row St(slave eq):      K_StN+ K̃_StM+ K̃_StI+ K̃_StSt+ K̃_StSl + D_St z_St                                     = -r_St
row Sl(slave eq):      K_SlN+ K̃_SlM+ K̃_SlI+ K̃_SlSt+ K̃_SlSl + D_Sl z_Sl                                     = -r_Sl
row I (constraint):    I_I z_I                                                                                = 0        (inactive: z_I = 0)
row A (normal constr): M̃_A Δd_M + S̃_AI Δd_I + S̃_ASt Δd_St + S̃_ASl Δd_Sl                                    = -g̃_A     (weighted gap = 0)
row St(stick constr):  H_St Δd_M + F_StI Δd_I + F_StSt Δd_St + F_StSl Δd_Sl + P_St z_St                       = -C_t,St
row Sl(slip constr):   J_Sl Δd_M + G_SlI Δd_I + G_SlSt Δd_St + G_SlSl Δd_Sl + L_Sl z_Sl                       = -C_t,Sl
```

- Rows N,M,I,St,Sl are the linearized balance equation (34); `K̃` = bulk
  stiffness incl. contact-geometry linearization terms.
- `D_I, D_St, D_Sl` are the diagonal mortar blocks coupling slave eq ↔ its z.
- `M_*^T` couple master eq ↔ z (the mortar `M` matrix).
- Rows 7/8/9 are the discrete contact constraints (normal / stick / slip) and
  their consistent linearizations (S̃, M̃, H, F, G, P, L blocks).

## Local elimination of z (Gitterle Eq. 85) — this is the crux

The **slave equilibrium rows (I, St, Sl)** contain `D z`, and `D` is diagonal,
so solve them for z:

```
z = D^-1 ( r_S − K_SN Δd_N − K̃_SM Δd_M − K̃_SS Δd_S )      (Eq. 85)
```

i.e. the multiplier is recovered from the **already-assembled slave force
balance** (bulk stiffness `K_SN, K̃_SM, K̃_SS` + residual `r_S`), NOT from the
constraint rows. `D^-1` is a cheap diagonal inverse.

## Condensed displacement-only system (Gitterle Eq. 86)

Substituting (85) into rows M, 7, 8, 9 and dropping the inactive z_I row/col:

```
row N :  K_NN, K_NM, K_NI, K_NSt, K_NSl                                                  = -r_N
row M :  K_MN, K̃_MM + M̂^T K̃_AM, ... + M̂^T K̃_A*, ...                                    = -(r_M + M̂^T r_A)
row I :  K_IN, K̃_IM, K̃_II, K̃_ISt, K̃_ISl                                                = -r_I
row A :  0,    M̃_A,  S̃_AI, S̃_ASt, S̃_ASl                                                 = -g̃_A
row St:  P_St D_St^-1 K_StN,  (P_St D_St^-1 K̃_StM − H_St), ...                            = -(P_St D_St^-1 r_St − C_t,St)
row Sl:  L_Sl D_Sl^-1 K_SlN,  (L_Sl D_Sl^-1 K̃_SlM − G_SlM), ...                           = -(L_Sl D_Sl^-1 r_Sl − C_t,Sl)
```

with the **mortar projection operator** (Eq. 87)
```
M̂ = D^-1 M .
```
The resulting system is **positive definite** (Eq. 86 text) and solvable by our
existing direct solver. `z` is recovered afterwards from (85).

## Mapping onto EdelweissFE (implementation plan)

The condensation operates on the **assembled** global system (it needs the bulk
stiffness sub-blocks `K_SN, K_SM, K_SS, K_MM`), so it must live at the solver
level, not inside the additive constraint. Proposed pieces:

1. **Constraint side (`mortarcontact.py`)** — expose, per increment:
   - the LM→slave-DOF and LM→master-DOF maps,
   - the diagonal `D` (already `current_D_rowsum`-like) and `M` (= `current_C`),
     hence `M̂ = D^-1 M`,
   - which LM are normal vs tangential, and the node-set tags (I / St / Sl).
   A new interface method, e.g. `getCondensationOperators()`.

2. **Constraint interface (`constraintbase.py`)** — optional method flagging a
   constraint as "condensable" and returning its operators; default: none.

3. **Solver (`nonlinearimplicitstatic.py`, `solveIncrement`)** — after
   assembling `K, R` and before the linear solve:
   - gather condensation operators from all condensable constraints,
   - build the static-condensation transform (eliminate LM rows/cols via Eq. 85,
     add `M̂^T·(slave rows)` into master rows, modify constraint rows per Eq. 86),
   - solve the reduced displacement system,
   - recover `Δz` via (85) and scatter back.
   A sparse implementation using the CSR structure + index sets.

## Incremental milestones (each independently verifiable)

- **M1**: Frictionless condensation (normal LM only). Reproduce the existing
  frictionless saddle-point POT_Dejori result as a condensed SPD system
  (same forces/displacements to ~1e-10, fewer DOFs). Verifies the machinery.
- **M2**: Add friction condensation (stick/slip rows, Eq. 86 rows St/Sl).
  Validate on test9-style synthetic + the POT_Dejori pull-out.
- Regression: the full mortar test suite (test1–test9) must stay green.

## Open questions to resolve while coding

- Do we form the consistent-linearization blocks (S̃, M̃, H, F, G, P, L)
  explicitly, or reuse the tangent our constraint already assembles? Our current
  constraint already produces a consistent tangent for the residual it assembles;
  the condensation is a re-arrangement of that same tangent, so we should be able
  to derive the condensed blocks from the assembled K without re-deriving the
  linearization from scratch.
- Handling of partially-covered nodes with tiny/negative `D_jj` (guard as in the
  saddle-point branch).
