#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test 11: Dual condensation for QUADRATIC (hex20 / CONQUAD8) slave facets
========================================================================

Extends milestone M1 (test10, linear hex8/CONQUAD4) to quadratic contact
facets. For quadratic dual mortar the physical mortar matrix D_phys is NOT
diagonal, so the scalar per-node elimination is done in the TRANSFORMED basis
of Popp, Wohlmuth, Gee & Wall (2012): the transformed matrix D_tilde =
D_phys T^T (T = the alpha = 1/3 basis transformation) is diagonal, and the
multiplier is eliminated as

    z_I = (1 / D_tilde_II) sum_K T_IK ( n_K . A[sK,:] d - n_K . f[sK] )

(Gitterle et al. 2010 Eq. 85 in the transformed basis). The standalone algebra
of this is verified in DUAL_CONDENSATION_verify_algebra_quadratic.py and the
global (shared-edge) transformation in DUAL_CONDENSATION_verify_globalT.py;
this test drives it END-TO-END through the real NIST(Parallel) solver.

Setup: the classical mortar contact patch test (two elastic blocks pressed
across a non-matching interface; cf. Puso & Laursen 2004, Farah 2018 App. A.2.1)
with hex20 blocks and CONQUAD8 contact facets. The interface is FLAT (y = 1
plane), i.e. constant facet normal, which is exactly the regime in which the
diagonal transformed elimination is algebraically exact (the curvature guard
passes) and where the quad8 patch test must hold to machine precision.

Checks, for BOTH a matching (2x2 vs 2x2) and a non-matching (2x2 vs 3x3)
interface (the latter exercises the genuinely non-diagonal D_phys):
  1. condensation=False (saddle-point baseline) passes the patch test.
  2. condensation=True passes the same patch test.
  3. The two agree: displacement field and recovered multipliers to ~1e-9.
     This is the M1-for-quadratic statement -- condensation changes only the
     linear algebra, not the solution.
"""

import os
import sys

import numpy as np

TESTDIR = os.path.abspath(os.path.dirname(__file__))
# reuse the hex20 contact-facet generator from test6
TEST6DIR = os.path.abspath(os.path.join(TESTDIR, "..", "test6_hex20_patch_test"))
# repo root must precede any installed edelweissfe
sys.path.insert(0, os.path.abspath(os.path.join(TESTDIR, "..", "..", "..")))

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.utils.inputfileparser import parseInputFile

E_MOD = 1000.0
PRESSURE = 10.0

INP_TEMPLATE = """*modelGenerator, generator=boxGen, name=genA
nX=2
nY=1
nZ=2
lX=1.0
lY=1.0
lZ=1.0
elType=C3D20

*modelGenerator, generator=boxGen, name=genB
y0=1.0
nX={nxB}
nY=1
nZ={nzB}
lX=1.0
lY=1.0
lZ=1.0
elType=C3D20

*modelGenerator, generator=executePythonCode, name=contactgen
import generated_setup_{name} as vs
vs.setup(model)

*material, name=LinearElastic, id=mat
{E}, 0.0

*section, name=secA, material=mat, type=solid
genA_all
*section, name=secB, material=mat, type=solid
genB_all

*constraint, type=mortarcontact, name=contact
nonMortarSurface=con_slave
mortarSurface=con_master
condensation={cond}

*job, name=condqjob, domain=3d
*solver, name=theSolver, solver=NISTParallel

*step, solver=theSolver
maxInc=0.5, minInc=1e-3, maxNumInc=100, maxIter=25, stepLength=1
>>dirichlet, name=bot, nSet=genA_bottom, field=displacement, 2=0.0
>>dirichlet, name=top, nSet=genB_top, field=displacement, 2={utop}
>>dirichlet, name=lat, nSet=allnodes, field=displacement, 1=0.0, 3=0.0
"""


def run(name, cond, nxB, nzB):
    setup_lines = [
        "import sys",
        f"sys.path.insert(0, r'{TEST6DIR}')",
        "import make_contact_elements as mce",
        "from edelweissfe.sets.nodeset import NodeSet",
        "",
        "def setup(model):",
        "    mce.apply(model, 'genA_top', 'con_slave', 'CONQUAD8', (0.0, 1.0, 0.0))",
        "    mce.apply(model, 'genB_bottom', 'con_master', 'CONQUAD8', (0.0, -1.0, 0.0))",
        "    model.nodeSets['allnodes'] = NodeSet('allnodes', list(model.nodes.values()))",
    ]
    setup_path = os.path.join(TESTDIR, f"generated_setup_{name}.py")
    with open(setup_path, "w") as f:
        f.write("\n".join(setup_lines) + "\n")
    if TESTDIR not in sys.path:
        sys.path.insert(0, TESTDIR)

    inp_text = INP_TEMPLATE.format(
        name=name, nxB=nxB, nzB=nzB, E=E_MOD, utop=-2.0 * PRESSURE / E_MOD, cond=cond
    )
    inp_path = os.path.join(TESTDIR, f"generated_{name}.inp")
    with open(inp_path, "w") as f:
        f.write(inp_text)

    inputFile = parseInputFile(inp_path)
    model, _ = finiteElementSimulation(inputFile, verbose=False, suppressPlots=True)

    nf = model.nodeFields["displacement"]
    U = np.array(nf["U"])
    lambdas = np.array([v.value for v in model.scalarVariables.values()]).flatten()

    os.remove(inp_path)
    os.remove(setup_path)
    return nf, U, lambdas


def check_patch(tag, nf, U, lambdas, tol_u=1e-7, tol_lam=1e-7):
    u_ref = PRESSURE * 2.0 / E_MOD
    max_err_lateral = 0.0
    max_err_uy = 0.0
    for node, u in zip(nf.nodes, U):
        y = node.coordinates[1]
        uy_exact = -PRESSURE * y / E_MOD
        max_err_lateral = max(max_err_lateral, abs(u[0]), abs(u[2]))
        max_err_uy = max(max_err_uy, abs(u[1] - uy_exact))
    print(f"  [{tag}] max|u_lateral| = {max_err_lateral:.3e}, max|u_y-u_y_exact| = {max_err_uy:.3e}")
    if len(lambdas) == 0:
        print(f"  [FAIL] {tag}: no contact multipliers found!")
        sys.exit(1)
    lam_err = np.max(np.abs(np.abs(lambdas) - PRESSURE)) / PRESSURE
    same_sign = np.all(lambdas > 0) or np.all(lambdas < 0)
    print(f"  [{tag}] multipliers n={len(lambdas)}, max rel dev from p = {lam_err:.3e}, same_sign={same_sign}")
    if max_err_lateral > tol_u * u_ref or max_err_uy > tol_u * u_ref:
        print(f"  [FAIL] {tag}: displacement field deviates from exact patch solution!")
        sys.exit(1)
    if not same_sign or lam_err > tol_lam:
        print(f"  [FAIL] {tag}: contact pressure not constant = p!")
        sys.exit(1)
    print(f"  [PASS] {tag}: patch test satisfied.")


def run_case(caselabel, nxB, nzB):
    print(f"\n{'=' * 60}\n{caselabel}\n{'=' * 60}")
    print("* Saddle-point (condensation=False)")
    nf0, U0, lam0 = run(f"q_saddle_{nxB}{nzB}", "False", nxB, nzB)
    check_patch("saddle", nf0, U0, lam0)

    print("* Condensed (condensation=True)")
    nf1, U1, lam1 = run(f"q_condensed_{nxB}{nzB}", "True", nxB, nzB)
    check_patch("condensed", nf1, U1, lam1)

    print("* Agreement between the two formulations")
    du = np.max(np.abs(U1 - U0)) / max(np.max(np.abs(U0)), 1e-30)
    dl = np.max(np.abs(lam1 - lam0)) / max(np.max(np.abs(lam0)), 1e-30)
    print(f"  rel diff displacements = {du:.3e}")
    print(f"  rel diff multipliers   = {dl:.3e}")
    if du > 1e-9 or dl > 1e-9:
        print("  [FAIL] condensed solution differs from saddle-point beyond tolerance!")
        sys.exit(1)
    print("  [PASS] condensation reproduces the saddle-point solution.")


def main():
    print("=" * 60)
    print("TEST 11: DUAL CONDENSATION (hex20 / CONQUAD8, transformed basis)")
    print("=" * 60)
    run_case("Matching interface (2x2 vs 2x2)", 2, 2)
    run_case("Non-matching interface (2x2 vs 3x3) -- non-diagonal D_phys", 3, 3)
    print("\n" + "=" * 60)
    print("TEST 11 (QUADRATIC CONDENSATION) PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
