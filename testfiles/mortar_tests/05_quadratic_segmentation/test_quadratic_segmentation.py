#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test 5: Sub-Zellen-Segmentierung für quadratische Kontaktelemente
=================================================================

Dieser Test verifiziert die segmentbasierte Mortar-Integration für quadratische
Kontaktelemente (CONQUAD8, CONQUAD9, CONTRI6). Quadratische Facetten werden für
das Polygon-Clipping in lineare Sub-Zellen zerlegt (Puso, Laursen & Solberg 2008;
Farah 2018, App. A.1.4; analog zu MOOSE AutomaticMortarGeneration). Ein direktes
Clipping mit allen Knoten in Speicherreihenfolge würde ein selbstschneidendes
Polygon erzeugen und Kontaktfläche verlieren (historischer Fehler: 17% Verlust
beim CONQUAD8).

Kontrollen für identische, flache Patches (Einheitsquadrat):
- Die Gesamtsumme von D und C muss exakt der Überlappungsfläche entsprechen
  (sum(D) = Fläche, da die dualen Formfunktionen eine Partition der Eins bilden).
- Die gelumpten Gewichte D_II = integral(N_tilde_I) müssen den analytischen
  Werten entsprechen (CONQUAD8 mit Basistransformation alpha=1/3:
  Ecken 5/36, Mittelknoten 1/9) und strikt positiv sein.
- Zeilensummen von D und C müssen übereinstimmen (Partition der Eins /
  Translationsinvarianz, Farah 2018, Gl. (4.99)).
Für teilweise überlappende Patches wird die exakte Überlappungsfläche geprüft.

Zusätzlich wird die Voraussetzung des Verfahrens vermessen: Sutherland--Hodgman
und die Fächer-Triangulierung verlangen KONVEXE Sub-Zellen (02_polygon_clipping
hält den Verlust fest, der sonst entsteht). ``run_subcell_convexity_test``
bestimmt, wie weit ein Mittel- bzw. Zentralknoten wandern muss, bis das
Mittel-Quad des CONQUAD8 bzw. die Quads des CONQUAD9 nicht-konvex werden, und
prüft, dass die integrierte Fläche bis dahin exakt bleibt.
"""

import os
import sys

import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from edelweissfe.config.elementlibrary import getElementClass
from edelweissfe.constraints.mortarcontact import Constraint as MortarContact, get_sub_cells
from edelweissfe.models.femodel import FEModel
from edelweissfe.points.node import Node
from edelweissfe.variables.fieldvariable import FieldVariable

TOL = 1e-12


def make_nodes(model, start_id, points):
    nodes = []
    for i, pt in enumerate(points):
        n_id = start_id + i
        model.nodes[n_id] = Node(n_id, np.array(pt, dtype=float))
        nodes.append(model.nodes[n_id])
    return nodes


def quad8_points(shift_x=0.0, z=0.0):
    return [
        [0.0 + shift_x, 0.0, z], [1.0 + shift_x, 0.0, z],
        [1.0 + shift_x, 1.0, z], [0.0 + shift_x, 1.0, z],
        [0.5 + shift_x, 0.0, z], [1.0 + shift_x, 0.5, z],
        [0.5 + shift_x, 1.0, z], [0.0 + shift_x, 0.5, z],
    ]


def quad9_points(shift_x=0.0, z=0.0):
    return quad8_points(shift_x, z) + [[0.5 + shift_x, 0.5, z]]


def tri6_pair_points(shift_x=0.0, z=0.0):
    """Two TRI6 covering the unit square: corners + all mid-side nodes."""
    s = shift_x
    return {
        # triangle 1: (0,0)-(1,0)-(1,1)
        "t1": [[0 + s, 0, z], [1 + s, 0, z], [1 + s, 1, z],
               [0.5 + s, 0, z], [1 + s, 0.5, z], [0.5 + s, 0.5, z]],
        # triangle 2: (0,0)-(1,1)-(0,1)
        "t2": [[0 + s, 0, z], [1 + s, 1, z], [0 + s, 1, z],
               [0.5 + s, 0.5, z], [0.5 + s, 1, z], [0 + s, 0.5, z]],
    }


def build_single_facet_model(el_type, slave_pts, master_pts, master_el_type=None):
    model = FEModel(dimension=3)
    slave_nodes = make_nodes(model, 1, slave_pts)
    master_nodes = make_nodes(model, 100, master_pts)

    ConClass = getElementClass(el_type, "edelweiss")
    s_con = ConClass(el_type, 1)
    s_con.setNodes(slave_nodes)
    model.elements[1] = s_con

    master_el_type = master_el_type or el_type
    MConClass = getElementClass(master_el_type, "edelweiss")
    m_con = MConClass(master_el_type, 2)
    m_con.setNodes(master_nodes)
    model.elements[2] = m_con

    model.surfaces = {"slave": {1: [s_con]}, "master": {1: [m_con]}}
    for node in model.nodes.values():
        node.fields["displacement"] = FieldVariable(node, "displacement")

    return MortarContact("c", model, nonMortarSurface="slave", mortarSurface="master", field="displacement")


def check(name, value, expected, tol=TOL):
    if abs(value - expected) > tol:
        print(f"  [FAIL] {name}: {value:.12f}, erwartet {expected:.12f}")
        raise AssertionError("Segmentierungspruefung fehlgeschlagen -- siehe [FAIL] oben")
    print(f"  [OK]   {name}: {value:.12f}")


def run_quad_test(el_type, points_func):
    print(f"\n* Teste {el_type}: identische Patches (Einheitsquadrat)...")
    constraint = build_single_facet_model(el_type, points_func(), points_func())
    D, C = constraint.compute_mortar_coupling_matrices()
    D, C = D.toarray(), C.toarray()  # sparse -> dicht, nur fuer diesen Test

    check("sum(D) = Überlappungsfläche", np.sum(D), 1.0)
    check("sum(C) = Überlappungsfläche", np.sum(C), 1.0)
    check("max|rowsum(D)-rowsum(C)|", np.max(np.abs(np.sum(D, axis=1) - np.sum(C, axis=1))), 0.0)

    lumped = np.sum(D, axis=1)
    if np.any(lumped <= 0.0):
        print(f"  [FAIL] Gelumpte Gewichte nicht strikt positiv: {lumped}")
        raise AssertionError("Segmentierungspruefung fehlgeschlagen -- siehe [FAIL] oben")
    print(f"  [OK]   Alle gelumpten Gewichte positiv (min = {np.min(lumped):.6f})")

    if el_type == "CONQUAD8":
        # Analytisch mit Basistransformation alpha = 1/3: Ecken 5/36, Mittelknoten 1/9
        expected = np.array([5.0 / 36.0] * 4 + [1.0 / 9.0] * 4)
        if not np.allclose(lumped, expected, atol=1e-10):
            print(f"  [FAIL] Gelumpte Gewichte {lumped} != analytisch {expected}")
            raise AssertionError("Segmentierungspruefung fehlgeschlagen -- siehe [FAIL] oben")
        print("  [OK]   Gelumpte Gewichte entsprechen analytischen Werten (5/36, 1/9)")

    print(f"* Teste {el_type}: Master um 0.3 in x verschoben...")
    constraint = build_single_facet_model(el_type, points_func(), points_func(shift_x=0.3))
    D, C = constraint.compute_mortar_coupling_matrices()
    D, C = D.toarray(), C.toarray()  # sparse -> dicht, nur fuer diesen Test
    check("sum(D) = Überlappungsfläche", np.sum(D), 0.7, tol=1e-10)
    check("max|rowsum(D)-rowsum(C)|", np.max(np.abs(np.sum(D, axis=1) - np.sum(C, axis=1))), 0.0, tol=1e-12)

    # Gelumpte Gewichte bei PARTIELLER Überdeckung. Die segmentquadratur-
    # basierten dualen Koeffizienten (konsistente Randbehandlung, Cichosz &
    # Bischoff 2011 / MOOSE reinitDual) halten sie hier positiv -- aber NICHT
    # aufgrund eines Positivitätssatzes: die transformierte CONQUAD8-Eckfunktion
    # ist bei alpha = 1/3 nicht punktweise nicht-negativ (Minimum -1/324 auf
    # 12,4 % der Elementfläche; dafür wäre alpha >= 3/8 nötig). Bei dieser
    # STREIFEN-Überdeckung liegt nur ein kleiner Teil des Negativgebiets in der
    # Überlappung und wird vom positiven Anteil überwogen -- deshalb hält die
    # scharfe Schranke. Wird die Überlappung dagegen in das Negativgebiet
    # gelegt, kippt das Gewicht; das hält
    # test_partial_coverage_corner_can_turn_weight_negative fest.
    # Echt abgesichert ist nur CONQUAD4 (bilineare N sind punktweise >= 0).
    # Die CONQUAD9-Formfunktionen wechseln punktweise das Vorzeichen (Minimum
    # -1/8 auf 49,8 % der Fläche), daher sind dort winzige negative Gewichte an
    # überdeckungsfernen Knoten möglich (vom sgn_D-Guard im Active-Set
    # abgefangen). Toleriert werden 5% des größten Gewichts.
    lumped = np.sum(D, axis=1)
    bound = -0.05 * np.max(lumped)
    if el_type in ("CONQUAD4", "CONQUAD8"):
        bound = -1e-12
    if np.any(lumped < bound):
        print(f"  [FAIL] Gelumpte Gewichte bei Teilüberdeckung zu negativ: {lumped}")
        raise AssertionError("Segmentierungspruefung fehlgeschlagen -- siehe [FAIL] oben")
    print(f"  [OK]   Gelumpte Gewichte bei Teilüberdeckung im zulässigen Bereich (min = {np.min(lumped):.6f})")

    print(f"  [PASS] {el_type} erfolgreich verifiziert!")


COVERING_MASTER = [[-2.0, -2.0, 0.0], [3.0, -2.0, 0.0], [3.0, 3.0, 0.0], [-2.0, 3.0, 0.0]]


def _shoelace(points):
    p = np.asarray(points)[:, :2]
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _is_convex(points):
    """Konvex, wenn alle aufeinanderfolgenden Kantenkreuzprodukte dasselbe
    Vorzeichen haben (numpy 2 kennt kein 2D-``cross`` mehr, daher explizit)."""
    p = np.asarray(points)[:, :2]
    n = len(p)
    cr = []
    for i in range(n):
        a = p[(i + 1) % n] - p[i]
        b = p[(i + 2) % n] - p[(i + 1) % n]
        cr.append(a[0] * b[1] - a[1] * b[0])
    return all(c > 0 for c in cr) or all(c < 0 for c in cr)


def _integrated_vs_geometric_area(el_type, slave_pts):
    """Integrierte Fläche (sum(D)) und geometrische Sub-Zellen-Fläche einer
    vollständig überdeckten Facette.

    Fängt zugleich die Laufzeitmeldung des Constraints ab, damit geprüft werden
    kann, dass sie genau dann kommt, wenn eine Sub-Zelle nicht-konvex ist.
    """
    constraint = build_single_facet_model(el_type, slave_pts, COVERING_MASTER, master_el_type="CONQUAD4")
    s_el = constraint.non_mortar_facets[0][0]
    coords = np.array(slave_pts, dtype=float)
    sub_cells = get_sub_cells(s_el)

    geometric = sum(_shoelace(coords[sub]) for sub in sub_cells)
    all_convex = all(_is_convex(coords[sub]) for sub in sub_cells)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        D, _ = constraint.compute_mortar_coupling_matrices()
        D = D.toarray()  # sparse -> dicht, nur fuer diesen Test
    warned = any("non-convex sub-cell" in str(w.message) for w in caught)
    return float(np.sum(D)), geometric, all_convex, warned


def run_subcell_convexity_test():
    """Ab wann werden die Sub-Zellen nicht-konvex -- und was kostet es?

    Sutherland--Hodgman und die Fächer-Triangulierung setzen konvexe Polygone
    voraus (siehe 02_polygon_clipping). Dreiecke sind immer konvex; gefährdet
    sind das Mittel-Quad [4,5,6,7] des CONQUAD8 und die vier Quads des CONQUAD9.
    Der Test misst die Schwelle, statt sie zu behaupten:

    CONQUAD8, unterer Mittelknoten von y = 0 nach y = t verschoben
        Das Mittel-Quad ist konvex, solange t < 0.5 -- der Mittelknoten müsste
        also über den ELEMENTMITTELPUNKT hinaus wandern. Bis dahin stimmt die
        integrierte Fläche exakt; darüber geht sie still verloren
        (t = 0.51: 1.3 %, t = 0.8: 12.5 %).

    CONQUAD9, Zentralknoten von (0.5, 0.5) nach (s, s) verschoben
        Die vier Quads sind konvex, solange s > 0.25 -- der Zentralknoten müsste
        also bis auf ein Viertel an die Ecke heranrücken. Darüber hinaus fehlen
        bis zu 4.3 %.

    Beide Schwellen liegen weit jenseits jeder Verzerrung, die ein brauchbares
    Volumenelement überlebt (die Jacobi-Determinante wäre längst negativ). Der
    Befund ist deshalb: die Annahme trägt mit großem Abstand -- aber sie wird
    zur Laufzeit NICHT geprüft, und ihre Verletzung ist lautlos.
    """
    print("\n* Teste Konvexität der Sub-Zellen (Flächenerhalt bei Verzerrung)...")

    def quad8_with_mid_bottom(t):
        return [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.5, t, 0.0], [1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.5, 0.0],
        ]

    def quad9_with_center(s):
        return quad8_with_mid_bottom(0.0) + [[s, s, 0.0]]

    print("  CONQUAD8, unterer Mittelknoten auf y = t:")
    for t, expect_convex in [(0.0, True), (0.2, True), (0.4, True), (0.49, True), (0.6, False)]:
        integrated, geometric, convex, warned = _integrated_vs_geometric_area(
            "CONQUAD8", quad8_with_mid_bottom(t)
        )
        loss = 1.0 - integrated / geometric
        print(
            f"    t = {t:.2f}: konvex = {str(convex):5s}  Fläche {integrated:.9f} von {geometric:.9f}"
            f"  Verlust {100 * loss:6.2f} %  Meldung: {'ja' if warned else 'nein'}"
        )
        if warned == convex:
            print(f"  [FAIL] Laufzeitmeldung bei t = {t} passt nicht zur Konvexität")
            raise AssertionError("Segmentierungspruefung fehlgeschlagen -- siehe [FAIL] oben")
        if convex != expect_convex:
            print(f"  [FAIL] Konvexität bei t = {t} anders als erwartet ({convex})")
            raise AssertionError("Segmentierungspruefung fehlgeschlagen -- siehe [FAIL] oben")
        if convex and abs(loss) > 1e-10:
            print(f"  [FAIL] Flächenverlust {loss:.3e} bei KONVEXEN Sub-Zellen (t = {t})!")
            raise AssertionError("Segmentierungspruefung fehlgeschlagen -- siehe [FAIL] oben")
        if not convex and loss < 1e-3:
            print(f"  [FAIL] Nicht-konvexe Sub-Zelle ohne messbaren Verlust bei t = {t} -- "
                  "die dokumentierte Schwelle stimmt nicht mehr")
            raise AssertionError("Segmentierungspruefung fehlgeschlagen -- siehe [FAIL] oben")

    print("  CONQUAD9, Zentralknoten auf (s, s):")
    for s, expect_convex in [(0.5, True), (0.4, True), (0.3, True), (0.26, True), (0.15, False)]:
        integrated, geometric, convex, warned = _integrated_vs_geometric_area(
            "CONQUAD9", quad9_with_center(s)
        )
        loss = 1.0 - integrated / geometric
        print(
            f"    s = {s:.2f}: konvex = {str(convex):5s}  Fläche {integrated:.9f} von {geometric:.9f}"
            f"  Verlust {100 * loss:6.2f} %  Meldung: {'ja' if warned else 'nein'}"
        )
        if warned == convex:
            print(f"  [FAIL] Laufzeitmeldung bei s = {s} passt nicht zur Konvexität")
            raise AssertionError("Segmentierungspruefung fehlgeschlagen -- siehe [FAIL] oben")
        if convex != expect_convex:
            print(f"  [FAIL] Konvexität bei s = {s} anders als erwartet ({convex})")
            raise AssertionError("Segmentierungspruefung fehlgeschlagen -- siehe [FAIL] oben")
        if convex and abs(loss) > 1e-10:
            print(f"  [FAIL] Flächenverlust {loss:.3e} bei KONVEXEN Sub-Zellen (s = {s})!")
            raise AssertionError("Segmentierungspruefung fehlgeschlagen -- siehe [FAIL] oben")
        if not convex and loss < 1e-3:
            print(f"  [FAIL] Nicht-konvexe Sub-Zelle ohne messbaren Verlust bei s = {s} -- "
                  "die dokumentierte Schwelle stimmt nicht mehr")
            raise AssertionError("Segmentierungspruefung fehlgeschlagen -- siehe [FAIL] oben")

    print("  [PASS] Flächenerhalt exakt, solange die Sub-Zellen konvex sind; "
          "Schwellen (CONQUAD8 t = 0.5, CONQUAD9 s = 0.25) bestätigt.")


def run_tri6_test():
    print("\n* Teste CONTRI6: identische Patches (2 Dreiecke, Einheitsquadrat)...")
    model = FEModel(dimension=3)
    s_pts = tri6_pair_points()
    m_pts = tri6_pair_points()

    ConClass = getElementClass("CONTRI6", "edelweiss")
    s1_nodes = make_nodes(model, 1, s_pts["t1"])
    s2_nodes = make_nodes(model, 20, s_pts["t2"])
    m1_nodes = make_nodes(model, 100, m_pts["t1"])
    m2_nodes = make_nodes(model, 120, m_pts["t2"])

    els = {}
    for el_id, nodes in [(1, s1_nodes), (2, s2_nodes), (3, m1_nodes), (4, m2_nodes)]:
        el = ConClass("CONTRI6", el_id)
        el.setNodes(nodes)
        model.elements[el_id] = el
        els[el_id] = el

    model.surfaces = {"slave": {1: [els[1], els[2]]}, "master": {1: [els[3], els[4]]}}
    for node in model.nodes.values():
        node.fields["displacement"] = FieldVariable(node, "displacement")

    constraint = MortarContact("c", model, nonMortarSurface="slave", mortarSurface="master", field="displacement")
    D, C = constraint.compute_mortar_coupling_matrices()
    D, C = D.toarray(), C.toarray()  # sparse -> dicht, nur fuer diesen Test

    check("sum(D) = Überlappungsfläche", np.sum(D), 1.0, tol=1e-10)
    check("sum(C) = Überlappungsfläche", np.sum(C), 1.0, tol=1e-10)
    check("max|rowsum(D)-rowsum(C)|", np.max(np.abs(np.sum(D, axis=1) - np.sum(C, axis=1))), 0.0, tol=1e-12)
    print("  [PASS] CONTRI6 erfolgreich verifiziert!")


def run_sliver_fallback_test():
    """Ab wann greift der Rueckfallpfad -- und was kostet er?

    Die dualen Koeffizienten werden auf dem TATSAECHLICHEN Ueberlappungsgebiet
    gebildet (konsistente Randbehandlung nach Cichosz & Bischoff 2011). Bei sehr
    schmalen Ueberlappungen wird die zugehoerige Massenmatrix M_t nahezu singulaer;
    der Code prueft ihre Konditionszahl und faellt bei cond(M_t) >= 1e12 auf die
    REFERENZELEMENT-Koeffizienten zurueck.

    Das ist kein theoretischer Zweig: ueber die Control_Tests und Patch-Tests wird
    er vier von 3989 Slave-Elementauswertungen betreten, ausschliesslich in den
    gekruemmten Hertz-Modellen. Und er hat eine Konsequenz, die festgehalten
    gehoert. Auf dem konsistenten Weg ist D_II = int_ueberlappung(N_tilde); das
    kann fuer CONQUAD8 schon dort negativ werden, weil die transformierte Basis
    bei alpha = 1/3 nicht punktweise nicht-negativ ist (siehe
    test_partial_coverage_corner_can_turn_weight_negative), aber nur knapp und nur
    bei einer Ueberdeckung, die im kleinen Negativgebiet der Eckfunktion liegt.
    Auf dem Rueckfallpfad steht dort dagegen das Integral der dualen Funktionen
    des VOLLEN Elements. Die wechseln ueber das halbe Element das Vorzeichen, und
    das Integral ueber ein Teilgebiet nimmt jedes Vorzeichen UND jede
    Groessenordnung an -- hier bis -1.0 des groessten Gewichts.

    Der Test misst den Uebergang, statt ihn zu behaupten.
    """
    print("\n* Rueckfallpfad bei entarteter Ueberlappung (CONQUAD8)")
    print(f"  {'Ueberdeckung':>14} {'Rueckfall':>10} {'min D_II':>14} {'min/max':>10}")
    print("  " + "-" * 52)

    rows = []
    for coverage in (1e-1, 1e-2, 1e-3, 1e-5):
        slave_pts = quad8_points()
        master_pts = quad8_points(shift_x=1.0 - coverage)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            constraint = build_single_facet_model("CONQUAD8", slave_pts, master_pts)
            D, _ = constraint.compute_mortar_coupling_matrices()
            D = D.toarray()  # sparse -> dicht, nur fuer diesen Test
        fell_back = any("degenerate overlap" in str(w.message) for w in caught)

        weights = np.sum(D, axis=1)
        w_min = float(np.min(weights))
        ratio = w_min / float(np.max(np.abs(weights)))
        rows.append((coverage, fell_back, w_min, ratio))
        print(f"  {coverage:>14.0e} {('ja' if fell_back else 'nein'):>10} "
              f"{w_min:>14.3e} {ratio:>10.3f}")

    # (1) Der Uebergang existiert und liegt zwischen 1e-2 und 1e-3 Ueberdeckung.
    assert not rows[0][1] and not rows[1][1], (
        "Bei 10 % und 1 % Ueberdeckung darf der Rueckfallpfad NICHT greifen; "
        "die Konditionsschranke 1e12 hat sich verschoben"
    )
    assert rows[2][1] and rows[3][1], (
        "Bei 0.1 % und 0.001 % Ueberdeckung MUSS der Rueckfallpfad greifen; "
        "die Konditionsschranke 1e12 hat sich verschoben"
    )

    # (2) Solange konsistent gerechnet wird, bleiben die Gewichte positiv --
    #     das ist die Aussage, die der Rueckfall aufgibt.
    for coverage, fell_back, w_min, _ in rows[:2]:
        assert w_min > 0.0, (
            f"Ohne Rueckfall muessen alle Knotengewichte positiv sein, bei "
            f"{coverage:.0e} Ueberdeckung ist min D_II = {w_min:.3e}"
        )

    # (3) Und mit Rueckfall eben nicht mehr. Das ist kein Fehler, sondern der
    #     dokumentierte Preis: eine unbrauchbare Inverse gegen den Verlust der
    #     Integral-Positivitaet. Der Vorzeichen-Guard im Constraint traegt das.
    for coverage, fell_back, w_min, ratio in rows[2:]:
        assert w_min < 0.0, (
            f"Auf dem Rueckfallpfad wird bei {coverage:.0e} Ueberdeckung ein negatives "
            f"Knotengewicht erwartet (dokumentierter Verlust der Positivitaet), "
            f"gemessen min D_II = {w_min:.3e}"
        )

    print("  [PASS] Der Uebergang liegt zwischen 1 % und 0.1 % Ueberdeckung und faellt "
          "mit dem Vorzeichenwechsel von D_II zusammen.")


def _quad8_shape_functions(xi, eta):
    """Serendipity-Formfunktionen des quad8, hier unabhaengig von
    ``element.py`` notiert, damit der Vergleich keine Tautologie ist.
    Knotenreihenfolge wie ueblich: erst die vier Ecken, dann die vier
    Kantenmitten (Ecke 0 bei (-1,-1), Mitte 4 zwischen Ecke 0 und 1)."""
    return np.array([
        0.25 * (1 - xi) * (1 - eta) * (-xi - eta - 1),
        0.25 * (1 + xi) * (1 - eta) * (xi - eta - 1),
        0.25 * (1 + xi) * (1 + eta) * (xi + eta - 1),
        0.25 * (1 - xi) * (1 + eta) * (-xi + eta - 1),
        0.5 * (1 - xi * xi) * (1 - eta),
        0.5 * (1 + xi) * (1 - eta * eta),
        0.5 * (1 - xi * xi) * (1 + eta),
        0.5 * (1 - xi) * (1 - eta * eta),
    ])


def _transformed_corner_min(alpha, n=401):
    """Kleinster Wert der transformierten Eckfunktion ueber das Referenzquadrat.

    N_tilde_0 = N_0 + alpha*(N_4 + N_7), Popp et al. (2012), Gl. (4.5).
    """
    g = np.linspace(-1.0, 1.0, n)
    X, Y = np.meshgrid(g, g)
    N = _quad8_shape_functions(X, Y)
    return float((N[0] + alpha * (N[4] + N[7])).min())


def _corner_weight_closed_form(s):
    """Geschlossene Form fuer das Knotengewicht des Eckknotens 0 bei einer
    Ueberlappung, die nur die gegenueberliegende Ecke [s,1]^2 bedeckt.

    Slave = CONQUAD8 auf [0,1]^2 mit alpha = 1/3, lokal xi = 2x-1. Damit ist

        D_II(Ecke 0) = int_{[s,1]^2} N_tilde_0 dA = -(s-1)^4 (8s-5) / 36,

    hergeleitet durch symbolische Integration von N_tilde_0 und hier als
    Referenz festgehalten (der Test braucht dafuer kein sympy). Nullstellen bei
    s = 5/8 und s = 1; fuer s > 5/8, also unterhalb von (3/8)^2 = 14.0625 %
    Ueberdeckung, ist das Gewicht NEGATIV.
    """
    return -((s - 1.0) ** 4) * (8.0 * s - 5.0) / 36.0


def test_partial_coverage_corner_can_turn_weight_negative():
    """CONQUAD8 hat bei Teilueberdeckung keine Positivitaetsgarantie.

    Die Basistransformation nach Popp et al. (2012), Gl. (4.5), sichert die
    INTEGRAL-Positivitaet ueber das VOLLE Element -- unverzerrt ab alpha > 1/8;
    Popp waehlt 1/5, Farah (2018, Gl. (6.20)) fuer tet10 den hier verwendeten
    Wert 1/3. Keine der beiden Quellen behauptet punktweise Nichtnegativitaet
    fuer die transformierte quadratische Basis; Popp nennt diese staerkere
    Eigenschaft (S. B429) ausdruecklich nur fuer die ordnungsreduzierte
    (bi)lineare Multiplikatorwahl.

    Das ist relevant, weil segmentbasiert ueber die TATSAECHLICHE Ueberlappung
    integriert wird: fuer beliebige Teilgebiete braucht es die punktweise
    Bedingung, und die gilt fuer die quad8-Eckfunktion erst ab alpha >= 3/8.
    Bei alpha = 1/3 bleibt ein Negativgebiet von 12.4 % der Elementflaeche mit
    Minimum -1/324. Liegt die Ueberlappung darin, wird D_II negativ -- ohne
    Sliver-Rueckfall, auf dem konsistenten Weg.

    Der Test haelt beides fest: die punktweise Eigenschaft der Basis und ihre
    Folge auf dem Produktionspfad, gegen die geschlossene Form gerechnet.
    """
    print("\n* CONQUAD8: Teilueberdeckung im Negativgebiet der Eckfunktion")

    # (1) Die transformierte Basis ist bei alpha = 1/3 NICHT punktweise
    #     nicht-negativ; erst ab alpha* = 3/8 ist sie es.
    ConClass = getElementClass("CONQUAD8", "edelweiss")
    T_e = ConClass("CONQUAD8", 1).getBasisTransformation()
    alpha = float(T_e[0, 4])
    assert abs(alpha - 1.0 / 3.0) < 1e-14, (
        f"Dieser Test ist auf alpha = 1/3 geschrieben, implementiert ist {alpha}. Wurde alpha "
        f"bewusst geaendert (z.B. auf 3/8, ab dem die Basis punktweise nicht-negativ waere), "
        f"sind die geschlossene Form in _corner_weight_closed_form, die Schwelle s = 5/8 und die "
        f"Doku-Abschnitte 8.2/8.3/8.4/13.4 nachzuziehen"
    )
    print(f"  [OK]   implementiertes alpha: {alpha:.12f}")
    check("min N_tilde_Ecke (alpha = 1/3)", _transformed_corner_min(alpha), -1.0 / 324.0, tol=1e-6)
    assert _transformed_corner_min(3.0 / 8.0) > -1e-12, (
        "Ab alpha = 3/8 muss die transformierte Eckfunktion punktweise nicht-negativ sein"
    )
    assert _transformed_corner_min(1.0 / 5.0) < -0.07, (
        "Bei Popps alpha = 1/5 muss das Negativgebiet deutlich tiefer sein als bei 1/3"
    )

    # (2) Folge auf dem Produktionspfad: Master so verschoben, dass nur die
    #     gegenueberliegende Ecke [s,1]^2 ueberdeckt ist.
    print(f"  {'s':>6} {'Ueberdeckung':>13} {'D_II Ecke 0':>14} {'geschlossen':>14} {'Rueckfall':>10}")
    print("  " + "-" * 62)
    slave_pts = quad8_points()
    for s in (0.3, 0.5, 0.6, 0.65, 0.7, 0.8):
        master_pts = [[x + s, y + s, z] for x, y, z in quad8_points()]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            constraint = build_single_facet_model("CONQUAD8", slave_pts, master_pts)
            D, C = constraint.compute_mortar_coupling_matrices()
            D, C = D.toarray(), C.toarray()
        fell_back = any("degenerate overlap" in str(w.message) for w in caught)

        weights = np.sum(D, axis=1)
        expected = _corner_weight_closed_form(s)
        print(f"  {s:>6.3f} {100 * (1 - s) ** 2:>12.2f}% {weights[0]:>14.3e} {expected:>14.3e} "
              f"{('ja' if fell_back else 'nein'):>10}")

        # Der Rueckfallpfad darf hier nirgends greifen -- sonst misst der Test
        # die Sliver-Entartung statt der punktweisen Eigenschaft der Basis.
        assert not fell_back, (
            f"Bei s = {s} darf der Sliver-Rueckfall NICHT greifen; sonst prueft dieser Test "
            f"nicht mehr den konsistenten Weg (dafuer ist run_sliver_fallback_test da)"
        )
        # Produktionspfad gegen die geschlossene Form. Die Segmentquadratur
        # trifft sie auf ~1e-14 absolut; bei Werten der Groessenordnung 1e-4
        # sind das acht signifikante Stellen. Die Schranke 1e-12 laesst dafuer
        # Luft, ohne die Aussage zu verwaessern.
        check(f"D_II(Ecke 0) bei s = {s}", float(weights[0]), expected, tol=1e-12)
        # Ueberdeckte Flaeche und Zeilensummen-Identitaet bleiben unberuehrt --
        # ein negatives D_II verletzt die Impulserhaltung NICHT.
        check(f"Ueberlappungsflaeche bei s = {s}", float(np.sum(D)), (1.0 - s) ** 2, tol=1e-12)
        check(
            f"max|rowsum(D)-rowsum(C)| bei s = {s}",
            float(np.max(np.abs(weights - np.sum(C, axis=1)))),
            0.0,
            tol=1e-12,
        )

        # (3) Das Vorzeichen kippt bei s = 5/8, nicht irgendwo.
        if s < 5.0 / 8.0:
            assert weights[0] > 0.0, (
                f"Oberhalb von 14.06 % Ueberdeckung (s = {s} < 5/8) muss das Eckgewicht positiv sein"
            )
        else:
            assert weights[0] < 0.0, (
                f"Unterhalb von 14.06 % Ueberdeckung (s = {s} > 5/8) MUSS das Eckgewicht negativ "
                f"werden -- alpha = 1/3 sichert nur das Voll-Element-Integral. Schlaegt das fehl, "
                f"wurde alpha geaendert; dann sind Doku-Abschnitte 8.3/8.4/13.4 nachzuziehen"
            )

    print("  [PASS] Vorzeichenwechsel bei s = 5/8 (14.06 % Ueberdeckung), ohne Sliver-Rueckfall, "
          "in Uebereinstimmung mit der geschlossenen Form.")


def test_quadratic_segmentation():
    run_quad_test("CONQUAD4", lambda shift_x=0.0: quad8_points(shift_x)[:4])
    run_quad_test("CONQUAD8", quad8_points)
    run_quad_test("CONQUAD9", quad9_points)
    run_tri6_test()
    run_subcell_convexity_test()
    run_sliver_fallback_test()


if __name__ == "__main__":
    print("====================================================")
    print("MORTAR SUB-ZELLEN-SEGMENTIERUNG VERIFIKATIONSTEST")
    print("====================================================")

    test_quadratic_segmentation()
    test_partial_coverage_corner_can_turn_weight_negative()

    print("\n====================================================")
    print("ALLE SEGMENTIERUNGSTESTS ERFOLGREICH PASSIERT!")
    print("====================================================")
