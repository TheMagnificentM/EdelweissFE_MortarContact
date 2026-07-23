#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test 14: viele MARGINALE Kontaktknoten (Active-Set-Churn) - POT_Dejori-Reproduktion
==================================================================================

Feiner gekrümmter Slave (Parabel) auf flachem Master, moderate Last -> nur der
mittlere Bereich kommt in Kontakt, an der Kontaktgrenze liegen mehrere marginale
Knoten. Soll den Active-Set-Churn reproduzieren, der bei POT_Dejori den
kondensierten Lauf stehenbleiben lässt (Sattelpunkt konvergiert dort einwandfrei).

Erwartung VOR dem Fix: Sattelpunkt konvergiert, kondensiert bleibt stehen.
"""

import os
import sys

import numpy as np

TESTDIR = os.path.abspath(os.path.dirname(__file__))

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.utils.inputfileparser import parseInputFile

E_MOD = 1000.0
NX = 12
UTOP = -0.05  # partial contact

SETUP = r'''
import numpy as np
from edelweissfe.points.node import Node
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.config.elementlibrary import getElementClass

NX = {NX}

def N(model, lbl, x, y):
    n = Node(lbl, np.array([x, y])); model.nodes[lbl] = n; return n

def setup(model):
    CPE4 = getElementClass("CPE4", None)
    CON = getElementClass("CONLINE2", "edelweiss")
    xs = [2.0*i/NX for i in range(NX+1)]
    yb = lambda x: 0.15*((x-1.0))**2           # parabola, min 0 at x=1, 0.15 at edges
    s_bot = [N(model, 1+i, x, yb(x)) for i, x in enumerate(xs)]
    s_top = [N(model, 1000+i, x, 1.0) for i, x in enumerate(xs)]
    sels = []
    for i in range(NX):
        e = CPE4("CPE4", 1+i); e.setNodes([s_bot[i], s_bot[i+1], s_top[i+1], s_top[i]])
        model.elements[1+i] = e; sels.append(e)
    m_top = [N(model, 2000+i, x, 0.0) for i, x in enumerate(xs)]
    m_bot = [N(model, 3000+i, x, -0.3) for i, x in enumerate(xs)]
    mels = []
    for i in range(NX):
        e = CPE4("CPE4", 2000+i); e.setNodes([m_bot[i], m_bot[i+1], m_top[i+1], m_top[i]])
        model.elements[2000+i] = e; mels.append(e)
    model._populateNodeFieldVariablesFromElements()

    def facet(lbl, a, b, want_ny_pos):
        t = b.coordinates[:2] - a.coordinates[:2]
        nn = np.array([t[1], -t[0]])
        nodes = [a, b] if (nn[1] > 0) == want_ny_pos else [b, a]
        c = CON("CONLINE2", lbl); c.setNodes(nodes); model.elements[lbl] = c; return c

    scon = [facet(100+i, s_bot[i], s_bot[i+1], False) for i in range(NX)]
    mcon = [facet(500+i, m_top[i], m_top[i+1], True) for i in range(NX)]
    model.surfaces["con_slave"] = {1: scon}
    model.surfaces["con_master"] = {1: mcon}
    model.elementSets["slave_all"] = ElementSet("slave_all", sels)
    model.elementSets["master_all"] = ElementSet("master_all", mels)
    model.nodeSets["slave_top"] = NodeSet("slave_top", s_top)
    model.nodeSets["master_bot"] = NodeSet("master_bot", m_bot)
'''.replace("{NX}", str(NX))

INP = """*modelGenerator, generator=executePythonCode, name=meshgen
import gen_marg as vs
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

*job, name=margjob, domain=2d
*solver, name=theSolver, solver=NIST

*step, solver=theSolver
maxInc=0.5, minInc=1e-4, maxNumInc=100, maxIter=40, stepLength=1
>>dirichlet, name=mbot, nSet=master_bot, field=displacement, 1=0.0, 2=0.0
>>dirichlet, name=stop, nSet=slave_top, field=displacement, 1=0.0, 2={utop}
"""


def solve(condensation):
    with open(os.path.join(TESTDIR, "gen_marg.py"), "w") as f:
        f.write(SETUP)
    cond = ", condensation=True" if condensation else ""
    with open(os.path.join(TESTDIR, "gen_marg.inp"), "w") as f:
        f.write(INP.format(E=E_MOD, cond=cond, utop=UTOP))
    model, _ = finiteElementSimulation(
        parseInputFile(os.path.join(TESTDIR, "gen_marg.inp")), verbose=False, suppressPlots=True)
    U = np.array(model.nodeFields["displacement"]["U"])
    c = model.constraints["contact"]
    return U, np.asarray(c.recovered_lambdas).flatten(), np.asarray(c.active_set)


if __name__ == "__main__":
    print("=" * 60)
    print(f"VIELE MARGINALE KNOTEN (NX={NX}, utop={UTOP})")
    print("=" * 60)
    try:
        U_s, lam_s, act_s = solve(False)
        print(f"[saddle]   OK, aktiv {int(act_s.sum())}/{len(act_s)}")
        s_ok = True
    except Exception as e:
        print(f"[saddle]   FEHLER {e}"); s_ok = False
    try:
        U_c, lam_c, act_c = solve(True)
        print(f"[condense] OK, aktiv {int(act_c.sum())}/{len(act_c)}")
        c_ok = True
    except Exception as e:
        print(f"[condense] FEHLER {e}"); c_ok = False
    for fn in ("gen_marg.py", "gen_marg.inp"):
        p = os.path.join(TESTDIR, fn)
        if os.path.exists(p):
            os.remove(p)
    if s_ok and c_ok:
        dU = np.max(np.abs(U_c - U_s)) / abs(UTOP)
        print(f"\n||U_cond-U_saddle||/|utop| = {dU:.3e}, active-set gleich: {np.array_equal(act_s,act_c)}")
        print("[PASS]" if dU < 1e-7 else "[FAIL: mismatch]")
    else:
        print("\n[INFO] Modus-Fehlschlag siehe oben")
