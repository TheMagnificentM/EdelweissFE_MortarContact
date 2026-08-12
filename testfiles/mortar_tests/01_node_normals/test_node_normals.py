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


# Referenzfacette je Elementtyp, in der Ebene y = y0 liegend. Dient als
# Master-Gegenstueck; ihre Form ist fuer diesen Test bedeutungslos, ihre LAGE
# nicht: sie muss UNTERHALB der Slave-Flaeche liegen, damit die Slave-Normale
# (-e_y auf der Unterseite) zum Master zeigt. Der Constraint verlangt genau das --
# der gewichtete Spalt
#     g_weak = -sum_K D_IK (x_K . n_I) + sum_J C_IJ (x_J . n_I)
# wird nur dann positiv bei geoeffnetem Kontakt, wenn n_I vom Slave zum Master
# weist. Eine Paarung von Ober- UND Unterseite DESSELBEN Blocks erfuellt das nicht
# und wird vom Eingabeschutz des Constraints gemeldet.
def _reference_facet(el_type, y0):
    a, b = 0.0, 2.0
    m = 0.5 * (a + b)
    if el_type == "CONLINE2":
        return [[a, y0], [b, y0]]
    if el_type == "CONLINE3":
        return [[a, y0], [b, y0], [m, y0]]
    if el_type == "CONTRI3":
        return [[a, y0, a], [b, y0, a], [a, y0, b]]
    if el_type == "CONTRI6":
        return [[a, y0, a], [b, y0, a], [a, y0, b],
                [m, y0, a], [m, y0, m], [a, y0, m]]
    corners = [[a, y0, a], [b, y0, a], [b, y0, b], [a, y0, b]]
    if el_type == "CONQUAD4":
        return corners
    mids = [[m, y0, a], [b, y0, m], [m, y0, b], [a, y0, m]]
    if el_type == "CONQUAD8":
        return corners + mids
    return corners + mids + [[m, y0, m]]  # CONQUAD9


def add_master_facet(model, el_type, y0):
    """Legt eine einzelne Master-Kontaktfacette unterhalb der Slave-Flaeche an."""
    ConClass = getElementClass(el_type, "edelweiss")
    next_node = (max(model.nodes.keys()) if model.nodes else 0) + 1
    next_el = (max(model.elements.keys()) if model.elements else 0) + 1

    nodes = []
    for i, pt in enumerate(_reference_facet(el_type, y0)):
        lbl = next_node + i
        model.nodes[lbl] = Node(lbl, np.array(pt, dtype=float))
        nodes.append(model.nodes[lbl])

    el = ConClass(el_type, next_el)
    el.setNodes(nodes)
    model.elements[next_el] = el
    model.surfaces["con_master"] = {1: [el]}
    return "con_master"


def build_model(el_type, dim, n=2, warp=None, snap=None):
    """Ein Block mit Kontaktelementen auf der Unterseite, Master-Facette darunter.

    Slave ist die UNTERSEITE des Blocks (Aussennormale -e_y), Master eine einzelne
    Facette darunter (siehe ``_reference_facet``). Geprueft werden ausschliesslich
    die Slave-Normalen; der Master ist das Gegenstueck, das der Constraint
    braucht.

    Der Master wird NICHT mit einem zweiten Generatoraufruf erzeugt:
    ``planeRectQuad`` beginnt seine Knotenlabels immer bei 1 und ueberschreibt bei
    einem zweiten Aufruf still das erste Netz (``boxGen`` versetzt die Labels
    dagegen, siehe boxgen.py).

    warp: optionale Funktion coords -> coords, die die Knoten VOR dem Anlegen der
    Kontaktelemente verschiebt (fuer die gekruemmte Geometrie).

    snap: optionale Funktion coords -> coords, die NACH dem Anlegen der
    Kontaktelemente auf alle Slave-Knoten wirkt. Sie erwischt damit auch die Knoten,
    die erst dabei entstehen (CONQUAD9-Zentrum, CONTRI6-Diagonalmitte); siehe
    ``cylinder_snap``.
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

    # Erst JETZT, denn apply_contact_elements legt fuer CONQUAD9 und CONTRI6 eigene
    # Knoten an, die die Geometrie sonst nicht trifft (siehe cylinder_snap).
    if snap is not None:
        for _faceID, elements in model.surfaces[slave_surf].items():
            for el in elements:
                for nd in el.nodes:
                    nd.coordinates[:] = snap(nd.coordinates)

    # Deutlich unterhalb der (ggf. gekruemmten) Slave-Flaeche
    master_surf = add_master_facet(model, el_type, -3.0 if warp is None else -3.0)

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


def cylinder_snap(X):
    """Zieht einen Punkt radial auf den Mantel r = CYL_R.

    Noetig fuer die Knoten, die ``apply_contact_elements`` SELBST anlegt: der
    Zentrumsknoten des CONQUAD9 und der Diagonalmittelknoten des CONTRI6 entstehen
    dort als arithmetisches Mittel zweier ECKknoten, liegen also auf der Sehne und
    nicht auf dem Zylinder. Ohne diese Projektion misst der Test fuer diese beiden
    Typen die Lage jenes Zusatzknotens statt der Normalenformel -- und zwar so
    deutlich, dass er den Unterschied zwischen einer facettenkonstanten und einer
    knotenweisen Normale gar nicht mehr sieht: beide lieferten dann denselben
    Winkelfehler von 7.16 Grad wie das lineare CONQUAD4.

    Fuer alle Knoten, die aus ``cylinder_warp`` stammen, ist das die Identitaet --
    die Abbildung y = 0 -> r = R trifft den Mantel exakt.
    """
    X = np.asarray(X, dtype=float)
    axis = np.array([1.0, -CYL_R])
    d = X[:2] - axis
    r = float(np.linalg.norm(d))
    if r > 1e-14:
        X = X.copy()
        X[:2] = axis + d * (CYL_R / r)
    return X


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


def run_curved(el_type, dim, refinements=(2, 4), min_ratio=1.8):
    """Knotennormale gegen die analytische Zylindernormale, mit Konvergenz.

    min_ratio ist die geforderte Fehlerreduktion bei Halbierung der Elementgroesse.
    Fuer die quadratischen Typen wird sie deutlich ueber 2 gesetzt (Aufrufer), und
    das ist der eigentliche Regressionsschutz dieses Tests: eine facettenkonstante
    Normale kann die Kruemmung einer quadratischen Facette nicht sehen und faellt
    damit auf dieselbe Rate ~2 zurueck wie die linearen Typen, bei rund dem
    250-fachen Fehler. Ohne eine Schranke an die ORDNUNG faellt das nicht auf --
    eine blosse Fehlerreduktion erfuellen beide Varianten.
    """
    print(f"\n* {el_type} ({dim}D), Zylinderausschnitt R = {CYL_R}")
    errors = []
    for n in refinements:
        _, contact = build_model(el_type, dim, n=n, warp=cylinder_warp, snap=cylinder_snap)
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

    # C: Konvergenz - und zwar mit einer Mindest-ORDNUNG, siehe Docstring.
    ratio = errors[0] / errors[-1] if errors[-1] > 0 else np.inf
    print(f"  Fehlerreduktion bei Halbierung der Elementgroesse: Faktor {ratio:.2f} "
          f"(gefordert {min_ratio:.1f})")
    if not ratio >= min_ratio:
        print(f"  [FAIL] Der Winkelfehler faellt zu langsam ({errors[0]:.4f} -> "
              f"{errors[-1]:.4f} Grad, Faktor {ratio:.2f} < {min_ratio:.1f})!")
        return False

    print(f"  [PASS] {el_type} ({dim}D), gekruemmt")
    return True


# ===========================================================================
# E: exakte Sehnenformel der 2D-Facettennormale
# ===========================================================================


def run_nodal_natural_coordinates():
    """Die Knotenkoordinaten passen zur Knotenreihenfolge der Formfunktionen.

    ``compute_normals`` wertet die Elementnormale an der Naturkoordinate JEDES
    Knotens aus (``getNodalNaturalCoordinates``, ``getShapeFunctionDerivativesAtNodes``).
    Diese Tabelle und ``getShapeFunctions`` muessen dieselbe Knotenreihenfolge
    benutzen; ein Vertauscher darin gibt einem Knoten stillschweigend die Normale
    eines anderen -- am ehesten die eines Mittelknotens, dessen Tangente auf einer
    gekruemmten Facette in eine ganz andere Richtung zeigt.

    Geprueft wird die definierende Eigenschaft, unabhaengig von jeder Geometrie:

        N_a(xi_b) = delta_ab .

    Das ist der Nachfolger der frueheren Sehnenpruefung. Die fing dieselbe Falle
    (CONLINE3 hat die Reihenfolge Ende-Ende-MITTE, sodass der letzte Listeneintrag
    NICHT der zweite Eckknoten ist), tat es aber ueber eine inzwischen abgeloeste
    Definition der Knotennormale -- die flaechengewichtete Facettensehne, die allen
    Knoten einer Facette dieselbe Richtung gab. Die Pruefung hier ist von der
    Definition der Normale unabhaengig und deckt alle sieben Elementtypen ab statt
    nur CONLINE3.
    """
    print("\n* Knoten-Naturkoordinaten gegen die Formfunktionen (alle Typen)")
    ok = True
    for el_type in ("CONLINE2", "CONLINE3", "CONQUAD4", "CONQUAD8",
                    "CONQUAD9", "CONTRI3", "CONTRI6"):
        el = getElementClass(el_type, "edelweiss")(el_type, 1)
        xi = el.getNodalNaturalCoordinates()
        N = np.array([el.getShapeFunctions(x) for x in xi])
        err = float(np.max(np.abs(N - np.eye(len(xi)))))

        # Und die zwischengespeicherte Ableitungstabelle muss dieselbe Tabelle sein.
        dN = el.getShapeFunctionDerivativesAtNodes()
        err_dn = max(
            float(np.max(np.abs(dN[a] - el.getShapeFunctionDerivatives(xi[a]))))
            for a in range(len(xi))
        )

        print(f"  {el_type:9s} max|N_a(xi_b) - delta_ab| = {err:.3e}, "
              f"max|dN_cache - dN| = {err_dn:.3e}")
        if err > TOL_EXACT or err_dn > TOL_EXACT:
            print(f"  [FAIL] Knotenreihenfolge von {el_type} passt nicht zu den "
                  f"Formfunktionen!")
            ok = False

    if ok:
        print("  [PASS] Knoten-Naturkoordinaten konsistent fuer alle sieben Typen")
    return ok


# ===========================================================================


def test_node_normals():
    results = []

    for el_type in ("CONQUAD4", "CONQUAD8", "CONQUAD9", "CONTRI3", "CONTRI6"):
        results.append(run_unit_and_flat(el_type, dim=3))
    for el_type in ("CONLINE2", "CONLINE3"):
        results.append(run_unit_and_flat(el_type, dim=2))

    # Mindestordnung: die linearen Typen rekonstruieren die Normale stueckweise
    # konstant und koennen nur erster Ordnung sein; die quadratischen sehen die
    # Kruemmung ihrer Facette und muessen deutlich schneller fallen (gemessen:
    # Faktor 7.95 gegenueber 2.00). Siehe run_curved.
    for el_type in ("CONQUAD4", "CONTRI3"):
        results.append(run_curved(el_type, dim=3, min_ratio=1.8))
    for el_type in ("CONQUAD8", "CONQUAD9", "CONTRI6"):
        results.append(run_curved(el_type, dim=3, min_ratio=3.5))
    results.append(run_curved("CONLINE2", dim=2, min_ratio=1.8))
    results.append(run_curved("CONLINE3", dim=2, min_ratio=3.5))

    results.append(run_nodal_natural_coordinates())

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
