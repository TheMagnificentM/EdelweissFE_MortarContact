#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test 12: Kondensation bei PARTIELLEM Kontakt (Active-Set)
========================================================

Kleiner, schneller Reproduktionsfall für das POT_Dejori-Verhalten: der Master ist
schmaler als der Slave, sodass nur ein Teil der Slave-Kontaktknoten aktiv ist
(Signorini / Active-Set). Vergleicht kondensiert vs. Sattelpunkt.
"""

import os
import sys

import numpy as np

TESTDIR = os.path.abspath(os.path.dirname(__file__))
MORTARDIR = os.path.abspath(os.path.join(TESTDIR, ".."))
sys.path.insert(0, os.path.join(MORTARDIR, "test8_2d_patch_test"))

import patch_test_2d as p2  # noqa: E402
from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation  # noqa: E402
from edelweissfe.utils.inputfileparser import parseInputFile  # noqa: E402

E_MOD = 1000.0
UTOP = -0.01

INP = """*modelGenerator, generator=planeRectQuad, name=genA
x0=0.0
y0=0.0
l=2.0
h=1.0
nX=4
nY=1
elType=CPE4

*modelGenerator, generator=planeRectQuad, name=genB
x0=0.5
y0=1.0
l=1.0
h=1.0
nX=2
nY=1
elType=CPE4

*modelGenerator, generator=executePythonCode, name=contactgen
import gen_partial as vs
vs.setup(model)

*material, name=LinearElastic, id=mat
{E}, 0.0

*section, name=secA, material=mat, type=plane, thickness=1.0
genA_all
*section, name=secB, material=mat, type=plane, thickness=1.0
genB_all

*constraint, type=mortarcontact, name=contact
nonMortarSurface=con_slave
mortarSurface=con_master{cond}

*job, name=partialjob, domain=2d
*solver, name=theSolver, solver=NIST

*step, solver=theSolver
maxInc=1.0, minInc=1e-3, maxNumInc=50, maxIter=30, stepLength=1
>>dirichlet, name=bot, nSet=genA_bottom, field=displacement, 1=0.0, 2=0.0
>>dirichlet, name=top, nSet=genB_top, field=displacement, 1=0.0, 2={utop}
"""

SETUP = '''
import sys
sys.path.insert(0, r'{d}')
import patch_test_2d as p2
from edelweissfe.sets.nodeset import NodeSet
def setup(model):
    p2.attach_2d_contact(model, 'genA_top', 'con_slave', 'CONLINE2', (0.0, 1.0))
    p2.attach_2d_contact(model, 'genB_bottom', 'con_master', 'CONLINE2', (0.0, -1.0))
'''.format(d=os.path.join(MORTARDIR, "test8_2d_patch_test"))


def solve(condensation):
    with open(os.path.join(TESTDIR, "gen_partial.py"), "w") as f:
        f.write(SETUP)
    if TESTDIR not in sys.path:
        sys.path.insert(0, TESTDIR)
    cond = ", condensation=True" if condensation else ""
    with open(os.path.join(TESTDIR, "gen_partial.inp"), "w") as f:
        f.write(INP.format(E=E_MOD, utop=UTOP, cond=cond))
    model, _ = finiteElementSimulation(
        parseInputFile(os.path.join(TESTDIR, "gen_partial.inp")), verbose=False, suppressPlots=True)
    U = np.array(model.nodeFields["displacement"]["U"])
    c = model.constraints["contact"]
    return U, np.asarray(c.recovered_lambdas).flatten(), np.asarray(c.active_set)


if __name__ == "__main__":
    print("=" * 60)
    print("MORTAR KONDENSATION - PARTIELLER KONTAKT (Active-Set)")
    print("=" * 60)
    U_s, lam_s, act_s = solve(False)
    print(f"[saddle]   aktive Knoten: {int(act_s.sum())}/{len(act_s)}")
    U_c, lam_c, act_c = solve(True)
    print(f"[condense] aktive Knoten: {int(act_c.sum())}/{len(act_c)}")
    dU = np.max(np.abs(U_c - U_s)) / abs(UTOP)
    print(f"\n  ||U_cond - U_saddle||/|utop| = {dU:.3e}")
    print(f"  active-set gleich: {np.array_equal(act_s, act_c)}")
    for fn in ("gen_partial.py", "gen_partial.inp"):
        p = os.path.join(TESTDIR, fn)
        if os.path.exists(p):
            os.remove(p)
    print("\n[PASS]" if dU < 1e-8 else "\n[FAIL]")
