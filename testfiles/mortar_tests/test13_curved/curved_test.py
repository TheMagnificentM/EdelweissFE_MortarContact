#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test 13: gekrümmte Kontaktfläche + partieller Kontakt (POT_Dejori-artig)
=======================================================================

Slave-Unterseite ist eine Parabel (variierende Normale über mehrere Facetten),
gedrückt auf einen flachen Master. Nur der mittlere Bereich kommt in Kontakt
(Active-Set). Trennt die Fehlerursache: laeuft der VEKTORIELLE SATTELPUNKT
(condensation=False), aber die Kondensation nicht -> Kondensationsfehler;
laeuft schon der Sattelpunkt nicht -> Reformulierungsfehler.

Ausgabe je Modus: Konvergenz (increment/iter) + max Multiplikator-Residuum.
"""

import os
import sys

import numpy as np

TESTDIR = os.path.abspath(os.path.dirname(__file__))

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.utils.inputfileparser import parseInputFile

E_MOD = 1000.0

SETUP = r'''
import numpy as np
from edelweissfe.points.node import Node
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.config.elementlibrary import getElementClass

def N(model, lbl, x, y):
    n = Node(lbl, np.array([x, y])); model.nodes[lbl] = n; return n

def setup(model):
    CPE4 = getElementClass("CPE4", None)
    CON = getElementClass("CONLINE2", "edelweiss")
    xs = [0.0, 0.5, 1.0, 1.5, 2.0]
    # slave: curved bottom (parabola, dips to 0 at x=1), flat top y=1
    s_bot = [N(model, 1+i, x, 0.1*(x-1.0)**2) for i, x in enumerate(xs)]
    s_top = [N(model, 10+i, x, 1.0) for i, x in enumerate(xs)]
    sels = []
    for i in range(4):
        e = CPE4("CPE4", 1+i); e.setNodes([s_bot[i], s_bot[i+1], s_top[i+1], s_top[i]])
        model.elements[1+i] = e; sels.append(e)
    # master: flat block y in [-0.3, 0]
    m_top = [N(model, 20+i, x, 0.0) for i, x in enumerate(xs)]
    m_bot = [N(model, 30+i, x, -0.3) for i, x in enumerate(xs)]
    mels = []
    for i in range(4):
        e = CPE4("CPE4", 20+i); e.setNodes([m_bot[i], m_bot[i+1], m_top[i+1], m_top[i]])
        model.elements[20+i] = e; mels.append(e)
    model._populateNodeFieldVariablesFromElements()

    def facet(lbl, a, b, want_ny_pos):
        t = b.coordinates[:2] - a.coordinates[:2]
        nn = np.array([t[1], -t[0]])
        nodes = [a, b] if (nn[1] > 0) == want_ny_pos else [b, a]
        c = CON("CONLINE2", lbl); c.setNodes(nodes); model.elements[lbl] = c; return c

    scon = [facet(100+i, s_bot[i], s_bot[i+1], False) for i in range(4)]   # slave normal down
    mcon = [facet(200+i, m_top[i], m_top[i+1], True) for i in range(4)]    # master normal up
    model.surfaces["con_slave"] = {1: scon}
    model.surfaces["con_master"] = {1: mcon}
    model.elementSets["slave_all"] = ElementSet("slave_all", sels)
    model.elementSets["master_all"] = ElementSet("master_all", mels)
    model.nodeSets["slave_top"] = NodeSet("slave_top", s_top)
    model.nodeSets["master_bot"] = NodeSet("master_bot", m_bot)
'''

INP = """*modelGenerator, generator=executePythonCode, name=meshgen
import gen_curved as vs
vs.setup(model)

*material, name=LinearElastic, id=mat
{E}, 0.0

*section, name=secS, material=mat, type=plane, thickness=1.0
slave_all
*section, name=secM, material=mat, type=plane, thickness=1.0
master_all

*constraint, type=mortarcontact, name=contact
nonMortarSurface=con_slave
mortarSurface=con_master{cond}

*job, name=curvedjob, domain=2d
*solver, name=theSolver, solver=NIST

*step, solver=theSolver
maxInc=0.25, minInc=1e-4, maxNumInc=100, maxIter=30, stepLength=1
>>dirichlet, name=mbot, nSet=master_bot, field=displacement, 1=0.0, 2=0.0
>>dirichlet, name=stop, nSet=slave_top, field=displacement, 1=0.0, 2=-0.15
"""


def solve(condensation):
    with open(os.path.join(TESTDIR, "gen_curved.py"), "w") as f:
        f.write(SETUP)
    cond = ", condensation=True" if condensation else ""
    with open(os.path.join(TESTDIR, "gen_curved.inp"), "w") as f:
        f.write(INP.format(E=E_MOD, cond=cond))
    model, _ = finiteElementSimulation(
        parseInputFile(os.path.join(TESTDIR, "gen_curved.inp")), verbose=False, suppressPlots=True)
    U = np.array(model.nodeFields["displacement"]["U"])
    c = model.constraints["contact"]
    return U, np.asarray(c.recovered_lambdas).flatten(), np.asarray(c.active_set)


if __name__ == "__main__":
    print("=" * 60)
    print("GEKRÜMMTE FLÄCHE + PARTIELLER KONTAKT")
    print("=" * 60)
    print("\n--- VEKTORIELLER SATTELPUNKT (condensation=False) ---")
    try:
        U_s, lam_s, act_s = solve(False)
        print(f"[saddle] konvergiert. aktiv {int(act_s.sum())}/{len(act_s)}, "
              f"lam(active)={np.round(lam_s[act_s],3)}")
        saddle_ok = True
    except SystemExit:
        raise
    except Exception as e:
        print(f"[saddle] FEHLER: {e}"); saddle_ok = False; U_s = None

    print("\n--- KONDENSIERT (condensation=True) ---")
    try:
        U_c, lam_c, act_c = solve(True)
        print(f"[condense] gelöst. aktiv {int(act_c.sum())}/{len(act_c)}")
        cond_ok = True
    except Exception as e:
        print(f"[condense] FEHLER: {e}"); cond_ok = False; U_c = None

    for fn in ("gen_curved.py", "gen_curved.inp"):
        p = os.path.join(TESTDIR, fn)
        if os.path.exists(p):
            os.remove(p)

    if saddle_ok and cond_ok and U_s is not None and U_c is not None:
        dU = np.max(np.abs(U_c - U_s)) / 0.15
        print(f"\n||U_cond - U_saddle||/uref = {dU:.3e}, active-set gleich: {np.array_equal(act_s, act_c)}")
        print("[PASS]" if dU < 1e-7 else "[FAIL: condensed != saddle]")
    else:
        print("\n[INFO] siehe oben, welcher Modus fehlschlägt")
