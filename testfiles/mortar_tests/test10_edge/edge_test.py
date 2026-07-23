#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test 10: Kondensation über eine KANTE/KNICK in der Kontaktfläche
================================================================

Der eigentliche Härtetest der vektoriellen Kondensation. Die Slave-(Non-Mortar-)
Oberfläche hat einen ausgeprägten Knick: zwei Facetten treffen am Scheitelknoten
unter ~53 Grad aufeinander. Die gemittelte Knotennormale weicht dort stark von
den Facettennormalen ab (|1 - n_Facette . n_Knoten| ~ 0.11, weit über der
Krümmungs-Guard-Schwelle 1e-2, an der der frühere skalare Kondensations-Ansatz
(Branch 4) bewusst ABBRACH).

Die vektorielle Elimination (Gitterle 2010, Eq. 85) projiziert NICHT auf eine
einzelne Normale und ist daher an Kanten gültig. Verifiziert:
  - die Kondensation bricht NICHT ab (kein MortarCondensationError),
  - kondensierte Lösung == Sattelpunkt-Lösung (U + Multiplikatoren) bis ~1e-9.

(Der Spannungszustand ist an der Kante nicht uniform, daher kein analytischer
Patch-Vergleich - der maßgebliche wissenschaftliche Nachweis ist die exakte
Reproduktion der Sattelpunkt-Referenz.)
"""

import os
import sys

import numpy as np

TESTDIR = os.path.abspath(os.path.dirname(__file__))

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.utils.inputfileparser import parseInputFile

E_MOD = 1000.0
UTOP = -0.02

# Kinked interface y-coordinate at x = 0, 1, 2 (Scheitel bei x=1).
Y_IF = [0.5, 1.0, 0.5]

SETUP = r'''
import numpy as np
from edelweissfe.points.node import Node
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.config.elementlibrary import getElementClass

Y_IF = [0.5, 1.0, 0.5]

def _mk_node(model, lbl, x, y):
    n = Node(lbl, np.array([x, y]))
    model.nodes[lbl] = n
    return n

def setup(model):
    CPE4 = getElementClass("CPE4", None)  # default (Marmot) provider for bulk
    CON = getElementClass("CONLINE2", "edelweiss")  # contact facets

    # --- slave body (bottom), kinked top ---
    s_bot = [_mk_node(model, 1 + i, x, 0.0) for i, x in enumerate((0.0, 1.0, 2.0))]
    s_top = [_mk_node(model, 10 + i, x, Y_IF[i]) for i, x in enumerate((0.0, 1.0, 2.0))]
    sA = CPE4("CPE4", 1); sA.setNodes([s_bot[0], s_bot[1], s_top[1], s_top[0]])
    sB = CPE4("CPE4", 2); sB.setNodes([s_bot[1], s_bot[2], s_top[2], s_top[1]])
    model.elements[1] = sA; model.elements[2] = sB

    # --- master body (top), conforming kinked bottom, flat top ---
    m_bot = [_mk_node(model, 20 + i, x, Y_IF[i]) for i, x in enumerate((0.0, 1.0, 2.0))]
    m_top = [_mk_node(model, 30 + i, x, y) for i, (x, y) in enumerate(
        [(0.0, 1.5), (1.0, 2.0), (2.0, 1.5)])]
    mA = CPE4("CPE4", 3); mA.setNodes([m_bot[0], m_bot[1], m_top[1], m_top[0]])
    mB = CPE4("CPE4", 4); mB.setNodes([m_bot[1], m_bot[2], m_top[2], m_top[1]])
    model.elements[3] = mA; model.elements[4] = mB

    model._populateNodeFieldVariablesFromElements()

    # --- contact facets ---
    def facet(lbl, a, b, expected_ny_positive):
        t = b.coordinates[:2] - a.coordinates[:2]
        n = np.array([t[1], -t[0]])
        nodes = [a, b] if (n[1] > 0) == expected_ny_positive else [b, a]
        c = CON("CONLINE2", lbl); c.setNodes(nodes)
        model.elements[lbl] = c
        return c

    # slave top facets: normal must point UP (+y) toward master
    sc1 = facet(101, s_top[0], s_top[1], True)
    sc2 = facet(102, s_top[1], s_top[2], True)
    # master bottom facets: normal must point DOWN (-y) toward slave
    mc1 = facet(201, m_bot[0], m_bot[1], False)
    mc2 = facet(202, m_bot[1], m_bot[2], False)

    model.surfaces["con_slave"] = {1: [sc1, sc2]}
    model.surfaces["con_master"] = {1: [mc1, mc2]}

    model.elementSets["slave_all"] = ElementSet("slave_all", [sA, sB])
    model.elementSets["master_all"] = ElementSet("master_all", [mA, mB])
    model.nodeSets["slave_bottom"] = NodeSet("slave_bottom", s_bot)
    model.nodeSets["master_top"] = NodeSet("master_top", m_top)
    model.nodeSets["allnodes"] = NodeSet(
        "allnodes", [n for n in model.nodes.values() if "displacement" in n.fields])
'''

INP = """*modelGenerator, generator=executePythonCode, name=meshgen
import gen_edge as vs
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

*job, name=edgejob, domain=2d
*solver, name=theSolver, solver=NIST

*step, solver=theSolver
maxInc=0.5, minInc=1e-3, maxNumInc=100, maxIter=25, stepLength=1
>>dirichlet, name=bot, nSet=slave_bottom, field=displacement, 1=0.0, 2=0.0
>>dirichlet, name=top, nSet=master_top, field=displacement, 1=0.0, 2={utop}
"""


def _check_kink():
    # report the curvature deviation of the apex node vs its two facets
    p0 = np.array([0.0, Y_IF[0]]); p1 = np.array([1.0, Y_IF[1]]); p2 = np.array([2.0, Y_IF[2]])
    def fn(a, b):
        t = b - a; n = np.array([t[1], -t[0]]); return n / np.linalg.norm(n) * np.sign(n[1])
    n1 = fn(p0, p1); n2 = fn(p1, p2)
    n_apex = n1 + n2; n_apex /= np.linalg.norm(n_apex)
    dev = max(abs(1.0 - n1 @ n_apex), abs(1.0 - n2 @ n_apex))
    return dev


def solve(condensation):
    with open(os.path.join(TESTDIR, "gen_edge.py"), "w") as f:
        f.write(SETUP)
    if TESTDIR not in sys.path:
        sys.path.insert(0, TESTDIR)
    cond = ", condensation=True" if condensation else ""
    inp = INP.format(E=E_MOD, utop=UTOP, cond=cond)
    path = os.path.join(TESTDIR, "gen_edge.inp")
    with open(path, "w") as f:
        f.write(inp)
    model, _ = finiteElementSimulation(parseInputFile(path), verbose=False, suppressPlots=True)
    os.remove(path)
    U = np.array(model.nodeFields["displacement"]["U"])
    c = model.constraints["contact"]
    return U, np.asarray(c.recovered_lambdas).flatten()


if __name__ == "__main__":
    print("=" * 60)
    print("MORTAR KONDENSATION MIT KANTE/KNICK IN DER KONTAKTFLAECHE")
    print("=" * 60)
    dev = _check_kink()
    print(f"\nKrümmungsabweichung am Scheitelknoten |1 - n_Facette.n_Knoten| = {dev:.3e}")
    print(f"  (Branch-4-Guard-Schwelle war 1e-2 -> dort ABBRUCH; hier zulässig)")
    assert dev > 1e-2, "Testgeometrie hat keinen echten Knick"

    U_s, lam_s = solve(condensation=False)
    print("\n[OK] Sattelpunkt-Referenz gelöst")
    try:
        U_c, lam_c = solve(condensation=True)
    except Exception as e:
        print(f"\n[FAIL] Kondensation brach an der Kante ab: {type(e).__name__}: {e}")
        sys.exit(1)
    print("[OK] Kondensation gelöst (kein Abbruch an der Kante)")

    dU = np.max(np.abs(U_c - U_s)) / abs(UTOP)
    dLam = np.max(np.abs(lam_c - lam_s)) / max(np.max(np.abs(lam_s)), 1e-30)
    print(f"\n  ||U_cond - U_saddle||_inf / |utop|  = {dU:.3e}")
    print(f"  ||lam_cond - lam_saddle||_inf / |lam| = {dLam:.3e}")

    os.remove(os.path.join(TESTDIR, "gen_edge.py"))
    if dU < 1e-9 and dLam < 1e-8:
        print("\n[PASS] Kondensation reproduziert den Sattelpunkt über die Kante!")
        print("=" * 60)
    else:
        print("\n[FAIL] Kondensierte Lösung weicht vom Sattelpunkt ab!")
        sys.exit(1)
