#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test 1: Verifizierung der Knotennormalen
========================================

Die Knotennormale traegt den gewichteten Spalt und damit die Richtung der
Kontaktkraft. Sie muss deshalb gegen eine *unabhaengige* Referenz geprueft
werden -- nicht gegen die Formel, mit der der Code sie berechnet.

Die Referenz muss dabei von AUSSEN kommen. Eine Pruefung der Form

    n_facet = cross(v1, v2)   und dann   dot(n_facet, v1) == 0

baut die Formel des Codes im Test nach und verifiziert anschliessend eine
Identitaet, die fuer JEDE Eingabe gilt: sie kann nicht fehlschlagen und sagt
ueber die Knotennormale nichts aus. Wer diesen Test erweitert, prueft bitte
gegen eine geschlossene Loesung, nicht gegen die Konstruktion.

Geprueft wird
-------------
A  Einheitslaenge          |n| = 1 an jedem Slave-Knoten.
B  Ebene Referenz          Auf einer ebenen Flaeche exakt -e_y.
C  Analytische Normale     Auf einem ZYLINDERausschnitt gegen die exakte
                           Normale n = (x - x_achse)/|x - x_achse|, mit
                           Konvergenz unter Netzverfeinerung. Das ist die
                           eigentliche Pruefung: eine unabhaengige Referenz.
D  Orientierung            Alle Normalen zeigen aus dem Koerper heraus (positive
                           Projektion auf die Radialrichtung). Eine nach innen
                           gedrehte Normale kehrt Spalt- und Druckvorzeichen um
                           und macht aus dem Kontakt eine Verklebung; nichts im
                           Constraint faengt das ab.
E  2D-Sehnenformel         Auf einer gekruemmten CONLINE3-Kante ist die exakte
                           flaechengewichtete Facettennormale die gedrehte Sehne
                           zwischen den beiden ECKknoten, denn
                              int n dGamma = int [t_y, -t_x] dxi = R (x_1 - x_0)
                           unabhaengig von der Kruemmung. Genau das wird
                           nachgerechnet -- der Test, der den obigen Fehler
                           gefunden haette.

Elementtypen: CONQUAD4/8/9, CONTRI3/6 (3D) und CONLINE2/3 (2D).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from edelweissfe.config.elementlibrary import getElementClass
from edelweissfe.constraints.mortarcontact import Constraint as MortarContact
from edelweissfe.generators.boxgen import generateModelData as generateBoxMesh
from edelweissfe.generators.planerectquad import generateModelData as generatePlaneMesh
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.points.node import Node
from edelweissfe.variables.fieldvariable import FieldVariable

TOL_UNIT = 1e-12
TOL_FLAT = 1e-12
TOL_EXACT = 1e-12


# ===========================================================================
# Modellaufbau
# ===========================================================================


def get_local_nodes(el, faceID):
    """Knoten einer Aussenflaeche eines Volumenelements, Abaqus-Konvention.

    Die Reihenfolge ist so gewaehlt, dass die Facettennormale nach AUSSEN zeigt.
    """
    elType = el.elType.upper()
    if "C3D8" in elType:
        idx = {1: [3, 2, 1, 0], 2: [4, 5, 6, 7], 3: [0, 1, 5, 4],
               4: [6, 5, 1, 2], 5: [7, 6, 2, 3], 6: [4, 7, 3, 0]}[faceID]
    elif "C3D20" in elType:
        idx = {1: [3, 2, 1, 0, 10, 9, 8, 11], 2: [4, 5, 6, 7, 12, 13, 14, 15],
               3: [0, 1, 5, 4, 8, 17, 12, 16], 4: [6, 5, 1, 2, 13, 17, 9, 18],
               5: [7, 6, 2, 3, 14, 18, 10, 19], 6: [4, 7, 3, 0, 15, 19, 11, 16]}[faceID]
    elif "CPS4" in elType or "CPE4" in elType:
        idx = {1: [0, 1], 2: [1, 2], 3: [2, 3], 4: [3, 0]}[faceID]
    elif "CPS8" in elType or "CPE8" in elType:
        idx = {1: [0, 1, 4], 2: [1, 2, 5], 3: [2, 3, 6], 4: [3, 0, 7]}[faceID]
    else:
        raise NotImplementedError(f"Local face mapping not defined for {elType}")
    return [el.nodes[i] for i in idx]


def apply_contact_elements(model, surface_name, contact_el_type):
    """Legt Kontaktelemente ueber eine Oberflaeche und liefert den neuen Flaechennamen."""
    ConClass = getElementClass(contact_el_type, "edelweiss")
    max_el_id = max(model.elements.keys()) if model.elements else 0
    max_node_id = max(model.nodes.keys()) if model.nodes else 0

    contact_elements = []
    for faceID, elements in model.surfaces[surface_name].items():
        for el in elements:
            facet_nodes = get_local_nodes(el, faceID)

            if contact_el_type == "CONQUAD9" and len(facet_nodes) == 8:
                coords = np.array([node.coordinates for node in facet_nodes[:4]])
                max_node_id += 1
                center_node = Node(max_node_id, np.mean(coords, axis=0))
                model.nodes[max_node_id] = center_node
                facet_nodes.append(center_node)

            if contact_el_type == "CONTRI3" and len(facet_nodes) == 4:
                for tri in ([0, 1, 2], [0, 2, 3]):
                    max_el_id += 1
                    con_el = ConClass(contact_el_type, max_el_id)
                    con_el.setNodes([facet_nodes[i] for i in tri])
                    model.elements[max_el_id] = con_el
                    contact_elements.append(con_el)
                continue

            if contact_el_type == "CONTRI6" and len(facet_nodes) == 8:
                max_node_id += 1
                mid_node = Node(max_node_id, 0.5 * (facet_nodes[0].coordinates + facet_nodes[2].coordinates))
                model.nodes[max_node_id] = mid_node

                max_el_id += 1
                con_el1 = ConClass(contact_el_type, max_el_id)
                con_el1.setNodes([facet_nodes[0], facet_nodes[1], facet_nodes[2],
                                  facet_nodes[4], facet_nodes[5], mid_node])
                model.elements[max_el_id] = con_el1
                contact_elements.append(con_el1)

                max_el_id += 1
                con_el2 = ConClass(contact_el_type, max_el_id)
                con_el2.setNodes([facet_nodes[0], facet_nodes[2], facet_nodes[3],
                                  mid_node, facet_nodes[6], facet_nodes[7]])
                model.elements[max_el_id] = con_el2
                contact_elements.append(con_el2)
                continue

            max_el_id += 1
            con_el = ConClass(contact_el_type, max_el_id)
            con_el.setNodes(facet_nodes)
            model.elements[max_el_id] = con_el
            contact_elements.append(con_el)

    new_surface_name = "con_" + surface_name
    model.surfaces[new_surface_name] = {1: contact_elements}
    return new_surface_name


def build_model(el_type, dim, n=2, warp=None):
    """Zwei-Block-Modell mit Kontaktelementen auf Ober- und Unterseite.

    warp: optionale Funktion coords -> coords, die die Knoten VOR dem Anlegen der
    Kontaktelemente verschiebt (fuer die gekruemmte Geometrie).
    """
    model = FEModel(dimension=dim)
    journal = Journal()

    if dim == 3:
        vol_type = "C3D20" if ("8" in el_type or "9" in el_type or "6" in el_type) else "C3D8"
        generateBoxMesh({"name": "gen"}, model, journal,
                        nX=n, nY=n, nZ=n, lX=2.0, lY=2.0, lZ=2.0,
                        elType=vol_type, elProvider="edelweiss")
    else:
        vol_type = "CPS8" if "3" in el_type else "CPS4"
        generatePlaneMesh({"name": "gen"}, model, journal,
                          nX=n, nY=n, l=2.0, h=2.0,
                          elType=vol_type, elProvider="edelweiss")

    if warp is not None:
        for node in model.nodes.values():
            node.coordinates = warp(node.coordinates)

    slave_surf = apply_contact_elements(model, "gen_bottom", el_type)
    master_surf = apply_contact_elements(model, "gen_top", el_type)

    for node in model.nodes.values():
        node.fields["displacement"] = FieldVariable(node, "displacement")

    contact = MortarContact(f"contact_{el_type}", model,
                            nonMortarSurface=slave_surf, mortarSurface=master_surf,
                            field="displacement")
    return model, contact


# ===========================================================================
# A/B: Einheitslaenge und ebene Referenz
# ===========================================================================


def run_unit_and_flat(el_type, dim):
    print(f"\n* {el_type} ({dim}D), ebene Flaeche")
    _, contact = build_model(el_type, dim)
    normals = contact.undeformed_normals

    lengths = np.linalg.norm(normals, axis=1)
    err_unit = float(np.max(np.abs(lengths - 1.0)))
    print(f"  max| |n| - 1 |            = {err_unit:.3e}   (Toleranz {TOL_UNIT:.0e})")
    if err_unit > TOL_UNIT:
        print("  [FAIL] Knotennormalen sind nicht normiert!")
        return False

    expected = np.zeros(dim)
    expected[1] = -1.0
    err_dir = float(np.max(np.abs(normals - expected)))
    print(f"  max|n - (-e_y)|           = {err_dir:.3e}   (Toleranz {TOL_FLAT:.0e})")
    if err_dir > TOL_FLAT:
        print("  [FAIL] Normale der ebenen Flaeche zeigt nicht nach -y!")
        return False

    print(f"  [PASS] {el_type} ({dim}D), eben")
    return True


# ===========================================================================
# C/D: analytische Normale auf einem Zylinderausschnitt + Orientierung
# ===========================================================================

CYL_R = 4.0  # Zylinderradius; die Achse liegt bei y = -CYL_R, parallel zu z


def cylinder_warp(X):
    """Bildet die ebene Unterseite y = 0 auf einen Zylindermantel vom Radius R ab.

    x bleibt die Bogenlaenge, y wird radial verschoben. Die Achse liegt bei
    (x_m, -R) parallel zu z, sodass die exakte Aussennormale der Unterseite
    radial NACH INNEN zur Achse zeigt -- also vom Koerper weg.
    """
    X = np.asarray(X, dtype=float)
    s = X[0] - 1.0                      # Bogenlaenge um die Mitte
    phi = s / CYL_R
    r = CYL_R + X[1]                    # y = 0 -> r = R (Mantel), y > 0 nach aussen
    out = X.copy()
    out[0] = 1.0 + r * np.sin(phi)
    out[1] = -CYL_R + r * np.cos(phi)
    return out


def exact_cylinder_normal(X, dim):
    """Exakte Aussennormale der gekruemmten Unterseite an der Stelle X.

    Die Unterseite ist die Flaeche r = R; ihre nach aussen (vom Material weg,
    also zur Achse hin) gerichtete Normale ist -e_r.
    """
    axis = np.zeros(dim)
    axis[0] = 1.0
    axis[1] = -CYL_R
    v = np.asarray(X, dtype=float) - axis
    if dim == 3:
        v[2] = 0.0                      # Achse parallel zu z
    return -v / np.linalg.norm(v)


def run_curved(el_type, dim, refinements=(2, 4)):
    """Knotennormale gegen die analytische Zylindernormale, mit Konvergenz."""
    print(f"\n* {el_type} ({dim}D), Zylinderausschnitt R = {CYL_R}")
    errors = []
    for n in refinements:
        _, contact = build_model(el_type, dim, n=n, warp=cylinder_warp)
        normals = contact.undeformed_normals

        worst = 0.0
        for i, node in enumerate(contact.non_mortar_nodes):
            n_exact = exact_cylinder_normal(node.coordinates, dim)
            # Winkel zwischen berechneter und exakter Normale
            c = float(np.clip(np.dot(normals[i], n_exact), -1.0, 1.0))
            worst = max(worst, np.degrees(np.arccos(c)))

            # D: Orientierung. Eine nach innen gedrehte Normale ist nicht ein
            # bisschen falsch, sondern macht aus Druck Zug.
            if c <= 0.0:
                print(f"  [FAIL] Normale an Knoten {node.label} zeigt in den Koerper "
                      f"(n . n_exakt = {c:+.3f})!")
                return False

        errors.append(worst)
        print(f"  n = {n:2d}:  max. Winkelfehler = {worst:8.4f} Grad")

    # C: Konvergenz. Die Facettennormale ist stueckweise linear rekonstruiert,
    # der Fehler muss mit dem Netz kleiner werden. Verlangt wird eine echte
    # Reduktion, nicht eine bestimmte Ordnung - die haengt am Elementtyp.
    if not errors[-1] < errors[0]:
        print(f"  [FAIL] Der Winkelfehler faellt nicht mit der Netzverfeinerung "
              f"({errors[0]:.4f} -> {errors[-1]:.4f} Grad)!")
        return False
    ratio = errors[0] / errors[-1] if errors[-1] > 0 else np.inf
    print(f"  Fehlerreduktion bei Halbierung der Elementgroesse: Faktor {ratio:.2f}")

    print(f"  [PASS] {el_type} ({dim}D), gekruemmt")
    return True


# ===========================================================================
# E: exakte Sehnenformel der 2D-Facettennormale
# ===========================================================================


def run_line_chord_exactness():
    """Auf einer gekruemmten CONLINE3-Kante ist die Sehne der Eckknoten exakt.

    Es gilt   int n dGamma = int [t_y, -t_x] dxi = R (x_1 - x_0),
    unabhaengig davon, wo der Mittelknoten liegt. Die Knotennormale einer
    einzelnen gekruemmten Facette muss daher fuer JEDE Mittelknotenlage die
    gedrehte Eckknoten-Sehne sein.

    Genau hier lag der Fehler, den die alte Fassung dieses Tests nicht sehen
    konnte: mit coords[-1] (dem Mittelknoten) statt coords[1] ergaben sich fuer
    einen um 0.15 ausgelenkten Mittelknoten 16.7 Grad Abweichung.
    """
    print("\n* CONLINE3: exakte Sehnenformel auf gekruemmter Kante")
    from edelweissfe.constraints.mortarcontact import Constraint as MC

    ok = True
    for bulge in (0.0, 0.05, 0.15, 0.4, -0.25):
        model = FEModel(dimension=2)
        ConClass = getElementClass("CONLINE3", "edelweiss")

        # Knotenreihenfolge Ende - Ende - Mitte
        slave_pts = [[0.0, 0.0], [1.0, 0.0], [0.5, bulge]]
        master_pts = [[0.0, 1.0], [1.0, 1.0], [0.5, 1.0 + bulge]]
        for i, p in enumerate(slave_pts + master_pts):
            lbl = i + 1
            model.nodes[lbl] = Node(lbl, np.array(p, dtype=float))

        s_con = ConClass("CONLINE3", 1)
        s_con.setNodes([model.nodes[i] for i in (1, 2, 3)])
        m_con = ConClass("CONLINE3", 2)
        m_con.setNodes([model.nodes[i] for i in (4, 5, 6)])
        model.elements[1], model.elements[2] = s_con, m_con
        model.surfaces = {"slave": {1: [s_con]}, "master": {1: [m_con]}}
        for nd in model.nodes.values():
            nd.fields["displacement"] = FieldVariable(nd, "displacement")

        c = MC("c", model, nonMortarSurface="slave", mortarSurface="master", field="displacement")

        # Referenz: gedrehte Sehne der beiden ECKknoten, normiert
        t = np.array(slave_pts[1]) - np.array(slave_pts[0])
        n_ref = np.array([t[1], -t[0]]) / np.linalg.norm(t)

        err = float(np.max(np.abs(c.undeformed_normals - n_ref)))
        angle = np.degrees(np.arccos(np.clip(float(c.undeformed_normals[0] @ n_ref), -1.0, 1.0)))
        print(f"  Mittelknoten-Auslenkung {bulge:+.2f}:  max|n - n_Sehne| = {err:.3e}, "
              f"Winkel {angle:.4f} Grad")
        if err > TOL_EXACT:
            print("  [FAIL] Die 2D-Facettennormale ist nicht die Eckknoten-Sehne! "
                  "(Vermutlich wird wieder coords[-1] statt coords[1] verwendet.)")
            ok = False

    if ok:
        print("  [PASS] CONLINE3-Sehnenformel exakt fuer jede Mittelknotenlage")
    return ok


# ===========================================================================


def test_node_normals():
    results = []

    for el_type in ("CONQUAD4", "CONQUAD8", "CONQUAD9", "CONTRI3", "CONTRI6"):
        results.append(run_unit_and_flat(el_type, dim=3))
    for el_type in ("CONLINE2", "CONLINE3"):
        results.append(run_unit_and_flat(el_type, dim=2))

    for el_type in ("CONQUAD4", "CONQUAD8", "CONQUAD9", "CONTRI3", "CONTRI6"):
        results.append(run_curved(el_type, dim=3))
    for el_type in ("CONLINE2", "CONLINE3"):
        results.append(run_curved(el_type, dim=2))

    results.append(run_line_chord_exactness())

    n_fail = results.count(False)
    assert n_fail == 0, f"{n_fail} von {len(results)} Normalenpruefungen fehlgeschlagen"


if __name__ == "__main__":
    print("====================================================")
    print("MORTAR KONTAKT: VERIFIKATION DER KNOTENNORMALEN")
    print("====================================================")
    test_node_normals()
    print("\n====================================================")
    print("ALLE NORMALENTESTS ERFOLGREICH PASSIERT!")
    print("====================================================")
