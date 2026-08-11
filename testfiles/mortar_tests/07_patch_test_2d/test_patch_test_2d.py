#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test 7: 2D-Mortar-Kontakt
=========================

Prüft den 2D-Mortar-Kontakt für lineare (CONLINE2) und quadratische (CONLINE3)
Kontakt-Linienelemente auf zwei Ebenen: erst die Bausteine einzeln, dann den
vollständigen Patch-Test durch den Löser.

Teil A -- Bausteine (ohne Löser)
--------------------------------
1. Knotennormalen: n = (t_y, -t_x) aus der Kantentangente.
2. Koppelmatrizen einer Einzelfacette:
   - Volle Überdeckung: sum(D) = sum(C) = Kantenlänge (Partition der Eins).
   - Zeilensummen-Identität sum_K D_IK = sum_J C_IJ (Translationsinvarianz).
   - Analytische Knotengewichte für CONLINE3 MIT Basistransformation
     (alpha = 1/3, Popp et al. 2012): Ecken 7/18, Mittelknoten 2/9. Ohne die
     Transformation wären es 1/6 bzw. 2/3 -- die Transformation verschiebt
     alpha * int(N_mid) von jedem Mittelknoten auf seine beiden Eckknoten,
     also 1/6 + (1/3)(2/3) = 7/18 bzw. (1/3)(2/3) = 2/9.
3. Teilüberdeckung (Master um 50 % verschoben): sum(D) = sum(C) = 0.5.
4. Eingabeschutz: eine Slave- und eine Masterfläche, die sich Knoten teilen,
   werden vom Konstruktor abgewiesen (sie führten sonst auf ein singuläres
   System, nicht auf einen Indizierungsfehler -- Begründung im Test).

Teil B -- Kontakt-Patch-Test durch den Löser
--------------------------------------------
Zwei linear-elastische Blöcke (E = 1000, nu = 0) werden über ein nicht
notwendigerweise passendes Interface gegeneinander gepresst. Block A liegt in
y in [0, 1], Block B in y in [1, 2]; die Oberseite von B wird um u_y = -0.02
verschoben, die Unterseite von A in y gehalten. Exakte Lösung:
sigma_yy = -p = -10 überall, u_y(y) = -p*y/E, u_x = 0, Kontaktdruck = p.

Geprüfte Varianten:
  1. CPE4 / CONLINE2, passend      (2 gegen 2 Elemente)
  2. CPE4 / CONLINE2, nicht-passend (2 gegen 3)
  3. CPE8 / CONLINE3, passend      (2 gegen 2)
  4. CPE8 / CONLINE3, nicht-passend (2 gegen 3)

Alle Facetten haben gerade Kanten, die segmentbasierte Integration ist damit
exakt -- der Patch-Test muss in Maschinengenauigkeit bestehen.

WICHTIG -- u_x bleibt frei: Anders als beim 3D-Patch-Test (08_patch_test_hex20),
der u_x = u_z = 0 auf ALLEN Knoten vorschreibt, wird hier u_x nur auf der linken
Kante gehalten (Symmetriebedingung, mit der exakten Lösung vereinbar, da nu = 0).
Dort ist die Kontrolle max|u_x| ~ 0 deshalb zwangsläufig erfüllt und damit leer;
hier sind die Interface-Knoten in x tatsächlich frei, sodass ein fälschlich
tangential klemmender Kontakt als u_x != 0 sichtbar würde.

Das Zwei-Block-Netz wird explizit aufgebaut (two_block_mesh_2d.py) und nicht mit
dem planeRectQuad-Generator -- Begründung im Docstring jenes Moduls.
"""

import os
import sys

import numpy as np

TESTDIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(TESTDIR, "../..")))

from edelweissfe.config.elementlibrary import getElementClass
from edelweissfe.constraints.mortarcontact import Constraint as MortarContact
from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.models.femodel import FEModel
from edelweissfe.points.node import Node
from edelweissfe.utils.inputfileparser import parseInputFile
from edelweissfe.variables.fieldvariable import FieldVariable

TOL = 1e-12

E_MOD = 1000.0
PRESSURE = 10.0
BLOCK_HEIGHT = 1.0


# ===========================================================================
# Teil A: Bausteine
# ===========================================================================


def make_nodes_2d(model, start_id, points):
    nodes = []
    for i, pt in enumerate(points):
        n_id = start_id + i
        model.nodes[n_id] = Node(n_id, np.array(pt, dtype=float))
        nodes.append(model.nodes[n_id])
    return nodes


def line2_points(shift_x=0.0, y=0.0):
    return [
        [0.0 + shift_x, y],
        [1.0 + shift_x, y],
    ]


def line3_points(shift_x=0.0, y=0.0):
    return [
        [0.0 + shift_x, y],
        [1.0 + shift_x, y],
        [0.5 + shift_x, y],
    ]


def build_2d_contact_model(el_type, slave_pts, master_pts, share_nodes=False):
    """Minimalmodell aus je einer Slave- und einer Masterfacette.

    Mit ``share_nodes=True`` wird für beide Facetten DASSELBE Knotenobjekt
    verwendet -- die Fehlkonfiguration, gegen die der Konstruktor schützt.
    """
    model = FEModel(dimension=2)
    slave_nodes = make_nodes_2d(model, 1, slave_pts)
    master_nodes = slave_nodes if share_nodes else make_nodes_2d(model, 100, master_pts)

    ConClass = getElementClass(el_type, "edelweiss")
    s_con = ConClass(el_type, 1)
    s_con.setNodes(slave_nodes)
    model.elements[1] = s_con

    m_con = ConClass(el_type, 2)
    m_con.setNodes(master_nodes)
    model.elements[2] = m_con

    model.surfaces = {"slave": {1: [s_con]}, "master": {1: [m_con]}}
    for node in model.nodes.values():
        node.fields["displacement"] = FieldVariable(node, "displacement")

    return MortarContact("c", model, nonMortarSurface="slave", mortarSurface="master", field="displacement")


def check(name, value, expected, tol=TOL):
    err = abs(value - expected)
    if err > tol:
        print(f"  [FAIL] {name}: {value:.12f}, erwartet {expected:.12f} (diff {err:.2e})")
        return False
    print(f"  [PASS] {name}: {value:.12f}")
    return True


def test_2d_line2():
    print("\n=== Test 2D CONLINE2 (lineares Linienelement) ===")
    mc = build_2d_contact_model("CONLINE2", line2_points(0.0, 0.0), line2_points(0.0, 0.0))

    n = mc.compute_normals()
    print(f"Normalen: {n}")
    assert np.allclose(np.abs(n[:, 1]), 1.0), "Normalen muessen vertikal sein"

    D, C = mc.compute_mortar_coupling_matrices()
    D, C = D.toarray(), C.toarray()  # sparse -> dicht, nur fuer diesen Test
    print(f"D:\n{D}")
    print(f"C:\n{C}")

    all_ok = True
    all_ok &= check("sum(D)", float(np.sum(D)), 1.0)
    all_ok &= check("sum(C)", float(np.sum(C)), 1.0)
    all_ok &= check("rowsum diff D vs C", float(np.max(np.abs(np.sum(D, axis=1) - np.sum(C, axis=1)))), 0.0)
    all_ok &= check("Gewicht Knoten 0", D[0, 0], 0.5)
    all_ok &= check("Gewicht Knoten 1", D[1, 1], 0.5)

    assert all_ok, "CONLINE2-Test fehlgeschlagen"


def test_2d_line3():
    print("\n=== Test 2D CONLINE3 (quadratisches Linienelement) ===")
    mc = build_2d_contact_model("CONLINE3", line3_points(0.0, 0.0), line3_points(0.0, 0.0))

    n = mc.compute_normals()
    print(f"Normalen:\n{n}")
    assert np.allclose(np.abs(n[:, 1]), 1.0), "Normalen muessen vertikal sein"

    D, C = mc.compute_mortar_coupling_matrices()
    D, C = D.toarray(), C.toarray()  # sparse -> dicht, nur fuer diesen Test
    print(f"D:\n{D}")
    print(f"C:\n{C}")

    rowsum_D = np.sum(D, axis=1)
    rowsum_C = np.sum(C, axis=1)

    all_ok = True
    all_ok &= check("sum(D)", float(np.sum(D)), 1.0)
    all_ok &= check("sum(C)", float(np.sum(C)), 1.0)
    all_ok &= check("rowsum diff D vs C", float(np.max(np.abs(rowsum_D - rowsum_C))), 0.0)

    # Mit Basistransformation T_e (alpha = 1/3): Ecken 7/18, Mittelknoten 2/9.
    all_ok &= check("Gewicht Ecke 0 (7/18)", rowsum_D[0], 7.0 / 18.0)
    all_ok &= check("Gewicht Ecke 1 (7/18)", rowsum_D[1], 7.0 / 18.0)
    all_ok &= check("Gewicht Mittelknoten 2 (2/9)", rowsum_D[2], 2.0 / 9.0)

    assert np.all(rowsum_D > 0), "Alle dualen Knotengewichte muessen strikt positiv sein"
    assert all_ok, "CONLINE3-Test fehlgeschlagen"


def test_2d_partial_overlap():
    print("\n=== Test 2D Teilüberdeckung (50 % Verschiebung) ===")
    mc = build_2d_contact_model("CONLINE2", line2_points(0.0, 0.0), line2_points(0.5, 0.0))
    D, C = mc.compute_mortar_coupling_matrices()
    D, C = D.toarray(), C.toarray()  # sparse -> dicht, nur fuer diesen Test

    rowsum_D = np.sum(D, axis=1)
    rowsum_C = np.sum(C, axis=1)

    all_ok = True
    all_ok &= check("sum(D)", float(np.sum(D)), 0.5)
    all_ok &= check("sum(C)", float(np.sum(C)), 0.5)
    all_ok &= check("rowsum diff D vs C", float(np.max(np.abs(rowsum_D - rowsum_C))), 0.0)

    assert all_ok, "Teilüberdeckungstest fehlgeschlagen"


def test_overlapping_surfaces_rejected():
    """Slave- und Masterfläche dürfen sich keinen Knoten teilen.

    Bei deckungsgleichen Facetten ist C = D exakt, damit verschwindet der
    gewichtete Spalt g = -sum_K D_IK (x_K.n) + sum_J C_IJ (x_J.n) für JEDE
    Konfiguration identisch und die zugehörige lambda-Zeile von K hebt sich auf,
    sobald der Knoten aktiv wird -- das System wird singulär. Der Konstruktor
    muss das mit einer verständlichen Meldung abweisen statt es dem Löser zu
    überlassen.
    """
    print("\n=== Test Schutz gegen überlappende Kontaktflächen ===")
    try:
        build_2d_contact_model("CONLINE2", line2_points(0.0, 0.0), None, share_nodes=True)
    except ValueError as e:
        msg = str(e)
        print(f"  Meldung: {msg}")
        assert "disjoint" in msg, "Die Meldung muss die Ursache benennen"
        assert "[1, 2]" in msg, "Die Meldung muss die betroffenen Knotenlabels nennen"
        print("  [PASS] Überlappende Flächen werden mit klarer Meldung abgewiesen.")
        return
    raise AssertionError("Überlappende Kontaktflächen wurden NICHT abgewiesen!")


# ===========================================================================
# Teil B: Kontakt-Patch-Test durch den Löser
# ===========================================================================

INP_TEMPLATE = """*modelGenerator, generator=executePythonCode, name=meshgen
import generated_setup_{name} as vs
vs.setup(model)

*material, name=LinearElastic, id=mat
{E}, 0.0

*section, name=secA, thickness=1.0, material=mat, type=plane
solids_a
*section, name=secB, thickness=1.0, material=mat, type=plane
solids_b

** cn: Komplementaritaetsparameter c_n der Normalkontakt-NCP, rein algorithmisch
** und ~ O(E) des weicheren Koerpers zu waehlen (Farah 2018, Abschn. 3.5.2).
*constraint, type=mortarcontact, name=contact
nonMortarSurface=con_slave
mortarSurface=con_master
cn={E}

*job, name=patch2djob, domain=2d
*solver, name=theSolver, solver=NISTParallel

*step, solver=theSolver
maxInc=0.5, minInc=1e-3, maxNumInc=100, maxIter=25, stepLength=1
>>dirichlet, name=bot,  nSet=fixed_bottom, field=displacement, 2=0.0
>>dirichlet, name=top,  nSet=load_top,     field=displacement, 2={utop}
>>dirichlet, name=symm, nSet=symm_x,       field=displacement, 1=0.0
"""

SETUP_TEMPLATE = """import sys
sys.path.insert(0, r'{testdir}')
import two_block_mesh_2d as tb
import make_contact_elements_2d as mce


def setup(model):
    tb.build(model, '{el_type}', {nx_a}, {nx_b}, height={height})
    mce.apply(model, 'surf_a_top', 'con_slave', '{con_type}', (0.0, 1.0))
    mce.apply(model, 'surf_b_bottom', 'con_master', '{con_type}', (0.0, -1.0))
"""


def run_patch_variant(name, el_type, con_type, nx_a, nx_b, tol=1e-8):
    print(f"\n* Variante: {name}")

    setup_path = os.path.join(TESTDIR, f"generated_setup_{name}.py")
    with open(setup_path, "w") as f:
        f.write(
            SETUP_TEMPLATE.format(
                testdir=TESTDIR, el_type=el_type, con_type=con_type,
                nx_a=nx_a, nx_b=nx_b, height=BLOCK_HEIGHT,
            )
        )

    if TESTDIR not in sys.path:
        sys.path.insert(0, TESTDIR)

    inp_path = os.path.join(TESTDIR, f"generated_{name}.inp")
    with open(inp_path, "w") as f:
        f.write(
            INP_TEMPLATE.format(
                name=name, E=E_MOD, utop=-2.0 * BLOCK_HEIGHT * PRESSURE / E_MOD,
            )
        )

    model, _ = finiteElementSimulation(parseInputFile(inp_path), verbose=False, suppressPlots=True)

    # --- Kontrolle 1: Verschiebungsfeld gleich der exakten Loesung ---
    nf = model.nodeFields["displacement"]
    U = nf["U"]
    u_ref = 2.0 * BLOCK_HEIGHT * PRESSURE / E_MOD  # |u_y| an der Oberseite

    max_err_ux = 0.0
    max_err_uy = 0.0
    for node, u in zip(nf.nodes, U):
        y = node.coordinates[1]
        max_err_ux = max(max_err_ux, abs(u[0]))
        max_err_uy = max(max_err_uy, abs(u[1] - (-PRESSURE * y / E_MOD)))

    print(f"  max|u_x|             = {max_err_ux:.3e}  (Toleranz {tol * u_ref:.1e}, x ist FREI)")
    print(f"  max|u_y - u_y_exakt| = {max_err_uy:.3e}  (Toleranz {tol * u_ref:.1e})")
    if max_err_ux > tol * u_ref or max_err_uy > tol * u_ref:
        print("  [FAIL] Verschiebungsfeld weicht von der exakten Patch-Loesung ab!")
        return False

    # --- Kontrolle 2: konstanter Kontaktdruck ---
    lambdas = np.array([v.value for v in model.scalarVariables.values()]).flatten()
    if len(lambdas) == 0:
        print("  [FAIL] Keine Kontakt-Multiplikatoren im Modell gefunden!")
        return False

    # lambda folgt der Literaturkonvention (Druck POSITIV), ist an voll ueber-
    # deckten Knoten also unmittelbar der Kontaktdruck. Geprueft wird daher der
    # vorzeichenrichtige Wert und nicht sein Betrag.
    lam_err = np.max(np.abs(lambdas - PRESSURE)) / PRESSURE
    same_sign = np.all(lambdas > 0)
    print(f"  Multiplikatoren: n = {len(lambdas)}, max. rel. Abweichung von p = {lam_err:.3e}")

    # Uebertragene Gesamtkraft: mit den dualen Gewichten (Zeilensummen von D)
    # gewichtetes Mittel der Multiplikatoren.
    rowsum = model.constraints["contact"].current_D_rowsum
    lam_mean = np.sum(lambdas * rowsum) / np.sum(rowsum)
    lam_mean_err = abs(lam_mean - PRESSURE) / PRESSURE
    print(f"  Gewichtetes Mittel (Gesamtkraft/Laenge): {lam_mean:.9f}, rel. Fehler = {lam_mean_err:.3e}")

    if not same_sign:
        print("  [FAIL] Kontaktdruck nicht durchgehend positiv (Zug oder Oszillation)!")
        return False
    if lam_err > tol:
        print("  [FAIL] Kontaktdruck nicht konstant = p!")
        return False
    if lam_mean_err > tol:
        print("  [FAIL] Uebertragene Gesamtkraft weicht von p*L ab!")
        return False

    print(f"  [PASS] Variante '{name}' erfolgreich!")
    os.remove(inp_path)
    os.remove(setup_path)
    return True


def test_2d_solver_patch_tests():
    print("\n=== Kontakt-Patch-Test durch den Loeser (2D) ===")
    variants = [
        ("cpe4_matching", "CPE4", "CONLINE2", 2, 2),
        ("cpe4_nonmatching", "CPE4", "CONLINE2", 2, 3),
        ("cpe8_matching", "CPE8", "CONLINE3", 2, 2),
        ("cpe8_nonmatching", "CPE8", "CONLINE3", 2, 3),
    ]
    results = [run_patch_variant(*v) for v in variants]
    assert all(results), f"{results.count(False)} von {len(results)} 2D-Patch-Test-Varianten fehlgeschlagen"


if __name__ == "__main__":
    print("=" * 60)
    print("2D-MORTAR-KONTAKT-TESTREIHE")
    print("=" * 60)
    test_2d_line2()
    test_2d_line3()
    test_2d_partial_overlap()
    test_overlapping_surfaces_rejected()
    test_2d_solver_patch_tests()
    print("\n[SUCCESS] Alle 2D-Mortar-Kontakt-Tests erfolgreich bestanden!")
