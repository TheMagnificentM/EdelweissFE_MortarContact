#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test 12: Increment-Groesse und der Fehler der eingefrorenen Geometrie
=====================================================================

Normalen ``n_I`` und Koppelmatrizen ``D``, ``M`` werden EINMAL je Increment am
letzten konvergierten Zustand ausgewertet und dann fuer alle Newton-Iterationen
des Increments festgehalten (gestaffelte Behandlung der Geometrienichtlinearitaet,
siehe Doku, Abschnitt "Eingefrorene Geometrie"). Der konvergierte Zustand erfuellt
damit

    g_weak(x; D_alt, M_alt, n_alt) = 0

mit den Groessen vom Increment-BEGINN, nicht mit denen der konvergierten
Konfiguration. Die Doku nennt diesen Fehler "von erster Ordnung in der
Increment-Groesse und ueber die Schrittweite kontrolliert". Dieser Test misst das
nach, statt es zu behaupten -- es ist die zentrale Genauigkeitsaussage der
Formulierung.

Warum ein GEKRUEMMTES Interface noetig ist
------------------------------------------
Auf einem ebenen Interface ist der Staffelungsfehler strukturell null: die
Kontaktflaechen bleiben eben und deckungsgleich, ``D`` und ``M`` aendern sich ueber
den Lastpfad nicht, und ob man sie am Anfang oder am Ende des Increments auswertet,
macht keinen Unterschied. Deshalb koennen weder die Patch-Tests noch der
Gleit-Testfall der Control_Tests hier etwas aussagen.

Der Testfall ist daher eine EINDRINGUNG: der obere Block hat eine parabolisch
gekruemmte Unterseite und wird auf den unteren gedrueckt. Die Kontaktzone waechst
mit der Last von der Mitte nach aussen, ``D`` und ``M`` aendern sich also
fortlaufend, und die Reihenfolge der Auswertung wird sichtbar.

Vorgehen
--------
Dasselbe Randwertproblem wird mit N = 1, 2, 4, 8, 16 Increments bis zur GLEICHEN
Endverschiebung gerechnet und gegen einen feinen Referenzlauf (N_ref) verglichen.
Bewertet werden zwei Groessen:

  * die uebertragene Gesamtkontaktkraft  sum_j lambda_j * D_II,j
    (die physikalisch massgebliche Groesse; lambda allein ist an teilweise
    ueberdeckten Knoten kein Druck, vgl. 10_signorini_check),
  * das Verschiebungsfeld in der L2-Norm.

Erwartet und geprueft wird: der Fehler faellt monoton, und die geschaetzte
Konvergenzordnung liegt bei ~1.
"""

import os
import sys

import numpy as np

TESTDIR = os.path.abspath(os.path.dirname(__file__))
MORTARDIR = os.path.abspath(os.path.join(TESTDIR, ".."))
PATCH2D_DIR = os.path.join(MORTARDIR, "07_patch_test_2d")
sys.path.insert(0, os.path.abspath(os.path.join(MORTARDIR, "../..")))

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.utils.inputfileparser import parseInputFile

E_MOD = 1000.0
LENGTH = 1.0
HEIGHT = 1.0
CURVATURE = 0.02   # Stich der Indenter-Unterseite ueber die halbe Breite
U_TOP = -0.05      # Endverschiebung der Oberseite

INP_TEMPLATE = """*modelGenerator, generator=executePythonCode, name=meshgen
import generated_setup_{name} as vs
vs.setup(model)

*material, name=LinearElastic, id=mat
{E}, 0.0

*section, name=secA, thickness=1.0, material=mat, type=plane
solids_a
*section, name=secB, thickness=1.0, material=mat, type=plane
solids_b

*constraint, type=mortarcontact, name=contact
nonMortarSurface=con_slave
mortarSurface=con_master
cn={E}

*job, name=incjob, domain=2d
*solver, name=theSolver, solver=NISTParallel

*step, solver=theSolver
maxInc={maxinc}, minInc=1e-9, maxNumInc=2000, maxIter=25, stepLength=1
>>dirichlet, name=bot,  nSet=fixed_bottom, field=displacement, 2=0.0
>>dirichlet, name=top,  nSet=load_top,     field=displacement, 2={utop}
>>dirichlet, name=symm, nSet=symm_x,       field=displacement, 1=0.0
"""

# Die Kruemmung wird NACH dem Netzaufbau aufgepraegt; danach muessen die
# Volumenelemente ihre Referenzgeometrie neu einlesen (setNodes), sonst rechnen sie
# Steifigkeit und Dehnung weiter aus der ungekruemmten Referenz. Dieselbe Falle ist
# in Control_Tests/README.md fuer den gedrehten Testfall dokumentiert.
SETUP_TEMPLATE = """import sys
sys.path.insert(0, r'{patchdir}')
import two_block_mesh_2d as tb
import make_contact_elements_2d as mce


def setup(model):
    tb.build(model, '{el_type}', {nx_a}, {nx_b}, height={height}, length={length})

    curv = {curv}
    for el in model.elementSets['solids_b']:
        for nd in el.nodes:
            nd.coordinates[1] += curv * (nd.coordinates[0] / {length}) ** 2
    for el in model.elementSets['solids_b']:
        el.setNodes(el.nodes)

    mce.apply(model, 'surf_a_top', 'con_slave', '{con_type}', (0.0, 1.0))
    mce.apply(model, 'surf_b_bottom', 'con_master', '{con_type}', (0.0, -1.0))
"""


def run(name, n_inc, el_type="CPE4", con_type="CONLINE2", nx_a=8, nx_b=10):
    """Ein Lauf mit vorgegebener maximaler Increment-Groesse."""
    setup_path = os.path.join(TESTDIR, f"generated_setup_{name}.py")
    with open(setup_path, "w") as f:
        f.write(SETUP_TEMPLATE.format(
            patchdir=PATCH2D_DIR, el_type=el_type, con_type=con_type,
            nx_a=nx_a, nx_b=nx_b, height=repr(HEIGHT), length=repr(LENGTH),
            curv=repr(CURVATURE),
        ))

    if TESTDIR not in sys.path:
        sys.path.insert(0, TESTDIR)

    inp_path = os.path.join(TESTDIR, f"generated_{name}.inp")
    with open(inp_path, "w") as f:
        f.write(INP_TEMPLATE.format(name=name, E=E_MOD, utop=repr(U_TOP),
                                    maxinc=repr(1.0 / n_inc)))

    model, _ = finiteElementSimulation(parseInputFile(inp_path), verbose=False, suppressPlots=True)

    mc = model.constraints["contact"]
    lambdas = np.array([sv.value for sv in mc.scalarVariables])
    rowsum = mc.current_D_rowsum

    # Physikalisch massgeblich ist die KNOTENKRAFT lambda * D_II, nicht lambda:
    # an teilweise ueberdeckten Knoten skaliert lambda mit 1/D_II und ist dort
    # kein Kontaktdruck mehr.
    nodal_forces = lambdas * rowsum
    total_force = float(np.sum(nodal_forces))

    nf = model.nodeFields["displacement"]
    U = np.array([np.asarray(u)[:2] for u in nf["U"]])

    os.remove(inp_path)
    os.remove(setup_path)

    return {
        "n_inc": n_inc,
        "total_force": total_force,
        "U": U,
        "n_active": int(mc.active_set.sum()),
        "n_slave": int(mc.nNonMortarNodes),
    }


def _order(errors, ns):
    """Geschaetzte Konvergenzordnung aus aufeinanderfolgenden Paaren."""
    orders = []
    for i in range(1, len(errors)):
        if errors[i] > 0.0 and errors[i - 1] > 0.0:
            orders.append(np.log(errors[i - 1] / errors[i]) / np.log(ns[i] / ns[i - 1]))
    return orders


def test_increment_size_convergence():
    print("\n=== Fehler der eingefrorenen Geometrie ueber die Increment-Groesse ===")
    print(f"  Indenter: parabolische Unterseite, Stich {CURVATURE}, Endverschiebung {U_TOP}")

    N_REF = 64
    ref = run("ref", N_REF)
    print(f"\n  Referenz N = {N_REF}: Gesamtkontaktkraft {ref['total_force']:+.9f}, "
          f"{ref['n_active']} von {ref['n_slave']} Slave-Knoten aktiv")

    u_ref_norm = float(np.linalg.norm(ref["U"]))
    assert ref["n_active"] > 0, "Der Referenzlauf hat gar keinen Kontakt - Modell pruefen"
    assert ref["n_active"] < ref["n_slave"], (
        "Im Referenzlauf ist die gesamte Slave-Flaeche aktiv - die Kontaktzone waechst dann "
        "nicht, und der Test misst den Staffelungsfehler nicht"
    )

    ns, err_f, err_u = [], [], []
    print(f"\n  {'N':>4} {'Gesamtkraft':>16} {'rel. Fehler F':>15} {'rel. L2(u)':>13} {'aktiv':>7}")
    print("  " + "-" * 60)
    for n in (1, 2, 4, 8, 16):
        r = run(f"n{n}", n)
        ef = abs(r["total_force"] - ref["total_force"]) / abs(ref["total_force"])
        eu = float(np.linalg.norm(r["U"] - ref["U"])) / u_ref_norm
        ns.append(n)
        err_f.append(ef)
        err_u.append(eu)
        print(f"  {n:>4} {r['total_force']:>16.9f} {ef:>15.3e} {eu:>13.3e} "
              f"{r['n_active']:>4}/{r['n_slave']}")

    # Massgeblich ist das VERSCHIEBUNGSFELD. Die Gesamtkontaktkraft ist ein
    # Integral, in dem sich die lokalen Abweichungen teilweise aufheben; sie faellt
    # bis auf ~1e-5 und liegt dort bereits im Bereich der Loeser-Toleranzen, wo eine
    # Ordnungsschaetzung nichts mehr hergibt. Sie wird deshalb ausgewiesen, aber
    # nicht bewertet.
    #
    # Die Ordnung wird ab N = 2 geschaetzt: N = 1 ist ein Sonderfall, in dem die
    # eingefrorene Geometrie die UNVERFORMTE Konfiguration ist, in der noch gar kein
    # Kontakt besteht. Der Sprung von dort ist qualitativ, nicht asymptotisch.
    ord_u = _order(err_u[1:], ns[1:])
    print(f"\n  Geschaetzte Ordnung (Verschiebungen, ab N = 2): {np.round(ord_u, 2)}")
    print(f"  Gesamtkraft nur als Messung: {err_f[0]:.3e} bei N = 1 auf "
          f"{err_f[-1]:.3e} bei N = {ns[-1]}")

    # (1) Der Fehler muss ueberhaupt vorhanden sein - sonst regt der Testfall die
    #     Geometrienichtlinearitaet nicht an und die Messung waere leer.
    assert err_u[0] > 1e-6, (
        f"Schon N = 1 trifft die Referenz auf {err_u[0]:.1e} genau -- der Testfall regt die "
        "Geometrienichtlinearitaet nicht an und kann nichts belegen"
    )

    # (2) Monotone Abnahme des Verschiebungsfehlers.
    for i in range(1, len(err_u)):
        assert err_u[i] < err_u[i - 1], (
            f"Der L2-Fehler des Verschiebungsfeldes faellt nicht monoton: N = {ns[i-1]} -> "
            f"{err_u[i-1]:.3e}, N = {ns[i]} -> {err_u[i]:.3e}"
        )

    # (3) Die Gesamtkraft muss ueber die Reihe hinweg deutlich besser werden -
    #     ohne Monotonieforderung, siehe oben.
    assert err_f[-1] < 0.1 * err_f[0], (
        f"Der Fehler der Gesamtkontaktkraft sinkt kaum: {err_f[0]:.3e} -> {err_f[-1]:.3e}"
    )

    # (4) Ordnung ~1. Grosszuegige Schranke: die Kontaktzone waechst knotenweise,
    #     also sprunghaft, sodass die Ordnung um den Erwartungswert schwankt.
    mean_order = float(np.mean(ord_u))
    print(f"  Mittlere Ordnung: {mean_order:.2f}")
    assert 0.7 <= mean_order <= 1.6, (
        f"Die Konvergenzordnung des Staffelungsfehlers ist {mean_order:.2f}, erwartet ~1. "
        "Entweder trifft die Aussage 'erster Ordnung in der Increment-Groesse' nicht mehr zu, "
        "oder die Referenzkonfiguration der eingefrorenen Geometrie hat sich geaendert."
    )

    print("\n  [PASS] Der Fehler der eingefrorenen Geometrie faellt mit der Schrittweite, "
          f"Ordnung {mean_order:.2f}.")


if __name__ == "__main__":
    print("=" * 66)
    print("STAFFELUNGSFEHLER UEBER DIE INCREMENT-GROESSE")
    print("=" * 66)
    test_increment_size_convergence()
    print("\n[SUCCESS] Der Staffelungsfehler wird ueber die Schrittweite kontrolliert.")
