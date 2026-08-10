#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test 11: Skaleninvarianz (Längeneinheiten)
==========================================

Der Kontakt-Constraint enthält an mehreren Stellen ABSOLUTE, also
einheitenabhängige Toleranzen. Die praktisch relevanten sind:

  * die Suchmarge der Nachbarsuche
        ``margin = max(0.15 * Ausdehnung, 0.1)``
    (``mortarcontact.py`` in ``compute_mortar_coupling_matrices``, beide
    AABB-Aufweitungen). Der zweite Term ist eine absolute Länge;
  * ``inside()`` in Sutherland--Hodgman, das ein Kreuzprodukt -- also eine
    FLÄCHE -- gegen ``-1e-12`` vergleicht (``mortar_geom_utils.py``);
  * die absolute Abbruchschranke ``1e-12`` der Rückabbildung
    ``map_2d_to_natural``, die auf einer LÄNGE arbeitet.

Dieselbe Rechnung in verschiedenen Längeneinheiten ist der direkte Test für alle
drei auf einmal: ein rein geometrisch skaliertes Problem (Längen x k,
vorgeschriebene Verschiebung x k, E unverändert) hat dieselben Dehnungen,
Spannungen und Kontaktdrücke; nur die Verschiebungen skalieren mit k.

Geprüft werden k = 1e-3, 1 und 1e3 gegen die exakte Patch-Lösung. Damit ist
sowohl der Fall abgedeckt, in dem der relative Term der Marge dominiert (grosse
Modelle), als auch der, in dem der absolute Term ``0.1`` alles überdeckt (kleine
Modelle) -- dort entartet die BVH-Suche zur vollständigen Paarung. Das ist ein
Laufzeit-, kein Korrektheitsproblem, und der Test macht es sichtbar.

Ausdrücklich mitgeprüft wird auch der semi-smooth Aktivierungsindikator
``s_n = p_n - c_n * g_sep``: ``p_n`` ist eine Spannung, ``g_sep`` eine LÄNGE, das
Produkt ``c_n * g_sep`` also nur dann eine Spannung, wenn ``c_n`` die Einheit
Spannung/Länge trägt. Bei festem ``c_n`` verschiebt sich die Schaltschwelle mit
der Längenskala. Auf die KONVERGIERTE Lösung wirkt sich das nicht aus (dort ist
``g_sep`` an aktiven Knoten null, die Lösung ist c_n-unabhängig) -- wohl aber auf
den Weg dorthin. Der Test rechnet deshalb alle Skalen mit demselben ``c_n = E``
und prüft, dass das Ergebnis trotzdem übereinstimmt.

Netz und Kontaktelemente aus 07_patch_test_2d.
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
PRESSURE = 10.0          # skaleninvariant
BASE_LENGTH = 1.0
BASE_HEIGHT = 1.0

# Absoluter Term der Suchmarge, aus mortarcontact.py. Wird hier nur zur
# Diagnose ausgewertet, nicht nachgebaut.
MARGIN_ABS = 0.1
MARGIN_REL = 0.15

RTOL = 1e-8

INP_TEMPLATE = """*modelGenerator, generator=executePythonCode, name=meshgen
import generated_setup_{name} as vs
vs.setup(model)

*material, name=LinearElastic, id=mat
{E}, 0.0

*section, name=secA, thickness=1.0, material=mat, type=plane
solids_a
*section, name=secB, thickness=1.0, material=mat, type=plane
solids_b

** cn ~ O(E) des weicheren Koerpers (Farah 2018, Abschn. 3.5.2). Bewusst auf
** ALLEN Skalen derselbe Wert -- siehe Modul-Docstring.
*constraint, type=mortarcontact, name=contact
nonMortarSurface=con_slave
mortarSurface=con_master
cn={E}

*job, name=scalejob, domain=2d
*solver, name=theSolver, solver=NISTParallel

*step, solver=theSolver
maxInc=0.5, minInc=1e-3, maxNumInc=100, maxIter=25, stepLength=1
>>dirichlet, name=bot,  nSet=fixed_bottom, field=displacement, 2=0.0
>>dirichlet, name=top,  nSet=load_top,     field=displacement, 2={utop}
>>dirichlet, name=symm, nSet=symm_x,       field=displacement, 1=0.0
"""

SETUP_TEMPLATE = """import sys
sys.path.insert(0, r'{patchdir}')
import two_block_mesh_2d as tb
import make_contact_elements_2d as mce


def setup(model):
    tb.build(model, '{el_type}', {nx_a}, {nx_b}, length={length}, height={height})
    mce.apply(model, 'surf_a_top', 'con_slave', '{con_type}', (0.0, 1.0))
    mce.apply(model, 'surf_b_bottom', 'con_master', '{con_type}', (0.0, -1.0))
"""


def run_scaled(name, el_type, con_type, nx_a, nx_b, k):
    """Zwei-Block-Patch-Test, geometrisch um den Faktor k skaliert."""
    length = BASE_LENGTH * k
    height = BASE_HEIGHT * k
    utop = -2.0 * height * PRESSURE / E_MOD

    setup_path = os.path.join(TESTDIR, f"generated_setup_{name}.py")
    with open(setup_path, "w") as f:
        f.write(
            SETUP_TEMPLATE.format(
                patchdir=PATCH2D_DIR, el_type=el_type, con_type=con_type,
                nx_a=nx_a, nx_b=nx_b, length=repr(length), height=repr(height),
            )
        )

    if TESTDIR not in sys.path:
        sys.path.insert(0, TESTDIR)

    inp_path = os.path.join(TESTDIR, f"generated_{name}.inp")
    with open(inp_path, "w") as f:
        f.write(INP_TEMPLATE.format(name=name, E=E_MOD, utop=repr(utop)))

    model, _ = finiteElementSimulation(parseInputFile(inp_path), verbose=False, suppressPlots=True)

    nf = model.nodeFields["displacement"]
    max_err_ux = 0.0
    max_err_uy = 0.0
    for node, u in zip(nf.nodes, nf["U"]):
        y = node.coordinates[1]
        max_err_ux = max(max_err_ux, abs(u[0]))
        max_err_uy = max(max_err_uy, abs(u[1] - (-PRESSURE * y / E_MOD)))

    mc = model.constraints["contact"]
    lambdas = np.array([sv.value for sv in mc.scalarVariables])
    rowsum = mc.current_D_rowsum
    lam_mean = np.sum(lambdas * rowsum) / np.sum(rowsum)

    os.remove(inp_path)
    os.remove(setup_path)

    return {
        "u_ref": abs(utop),
        "err_ux": max_err_ux,
        "err_uy": max_err_uy,
        "lambdas": lambdas,
        "lam_mean": lam_mean,
        "facet_size": length / nx_a,
        "extent": max(length, 2.0 * height),
    }


def report(k, res):
    """Bewertet einen Lauf und meldet nebenbei die Entartung der Suchmarge."""
    tol = RTOL * res["u_ref"]
    lam_err = np.max(np.abs(np.abs(res["lambdas"]) - PRESSURE)) / PRESSURE
    mean_err = abs(abs(res["lam_mean"]) - PRESSURE) / PRESSURE

    print(f"\n* Skalierungsfaktor k = {k:g}  (Modellausdehnung {res['extent']:g})")
    print(f"  max|u_x|             = {res['err_ux']:.3e}   (Toleranz {tol:.1e})")
    print(f"  max|u_y - u_y_exakt| = {res['err_uy']:.3e}   (Toleranz {tol:.1e})")
    print(f"  Kontaktdruck: max. rel. Abweichung von p = {lam_err:.3e}, "
          f"gewichtetes Mittel {res['lam_mean']:+.9f} (rel. Fehler {mean_err:.3e})")

    # Diagnose der Suchmarge (Formel aus mortarcontact.py, hier nur ausgewertet)
    margin = max(MARGIN_REL * res["extent"], MARGIN_ABS)
    ratio = margin / res["facet_size"]
    dominating = "absolut (0.1)" if MARGIN_ABS > MARGIN_REL * res["extent"] else "relativ (0.15*L)"
    print(f"  Suchmarge = {margin:g} = {ratio:.1f} x Facettengroesse -- dominierender Term: {dominating}")
    if ratio > 10.0:
        print("  [INFO] Die Marge uebersteigt die Facettengroesse um mehr als das Zehnfache:")
        print("         die BVH-Anfrage liefert praktisch ALLE Masterfacetten zurueck, die")
        print("         Nachbarsuche entartet zur vollstaendigen Paarung (O(n^2)). Das ist ein")
        print("         LAUFZEIT-, kein Korrektheitsproblem -- die Ergebnisse oben belegen das.")

    ok = True
    if res["err_ux"] > tol or res["err_uy"] > tol:
        print("  [FAIL] Verschiebungsfeld skaliert nicht wie die exakte Loesung!")
        ok = False
    if lam_err > RTOL or mean_err > RTOL:
        print("  [FAIL] Kontaktdruck ist nicht skaleninvariant!")
        ok = False
    if ok:
        print(f"  [PASS] k = {k:g}: Spannungen und Kontaktdruck unveraendert, "
              f"Verschiebungen um k skaliert.")
    return ok


VARIANTS = [
    ("cpe4", "CPE4", "CONLINE2", 2, 3),
    ("cpe8", "CPE8", "CONLINE3", 2, 3),
]

SCALES = [1e-3, 1.0, 1e3]


def test_scale_invariance():
    print("\n=== Skaleninvarianz des Zwei-Block-Patch-Tests ===")
    results = []
    pressures = {}
    for tag, el_type, con_type, nx_a, nx_b in VARIANTS:
        print(f"\n--- Variante {el_type}/{con_type} (nicht passend, {nx_a} gegen {nx_b}) ---")
        for k in SCALES:
            name = f"{tag}_k{SCALES.index(k)}"
            res = run_scaled(name, el_type, con_type, nx_a, nx_b, k)
            results.append(report(k, res))
            pressures[(tag, k)] = np.sort(np.abs(res["lambdas"]))

        # Direkter Vergleich der Skalen untereinander, nicht nur gegen die
        # analytische Loesung: die Multiplikatoren muessen knotenweise gleich sein.
        ref = pressures[(tag, 1.0)]
        for k in SCALES:
            diff = np.max(np.abs(pressures[(tag, k)] - ref)) / PRESSURE
            print(f"  Multiplikatoren k = {k:g} gegen k = 1: max. rel. Differenz {diff:.3e}")
            if diff > RTOL:
                print("  [FAIL] Multiplikatoren unterscheiden sich zwischen den Skalen!")
                results.append(False)

    n_fail = results.count(False)
    assert n_fail == 0, f"{n_fail} Skalierungspruefungen fehlgeschlagen"


if __name__ == "__main__":
    print("=" * 60)
    print("SKALENINVARIANZ DES MORTAR-KONTAKTS")
    print("=" * 60)
    test_scale_invariance()
    print("\n[SUCCESS] Die Loesung ist skaleninvariant.")
