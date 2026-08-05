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
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from edelweissfe.config.elementlibrary import getElementClass
from edelweissfe.constraints.mortarcontact import Constraint as MortarContact
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


def build_single_facet_model(el_type, slave_pts, master_pts):
    model = FEModel(dimension=3)
    slave_nodes = make_nodes(model, 1, slave_pts)
    master_nodes = make_nodes(model, 100, master_pts)

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
    if abs(value - expected) > tol:
        print(f"  [FAIL] {name}: {value:.12f}, erwartet {expected:.12f}")
        sys.exit(1)
    print(f"  [OK]   {name}: {value:.12f}")


def run_quad_test(el_type, points_func):
    print(f"\n* Teste {el_type}: identische Patches (Einheitsquadrat)...")
    constraint = build_single_facet_model(el_type, points_func(), points_func())
    D, C = constraint.compute_mortar_coupling_matrices()

    check("sum(D) = Überlappungsfläche", np.sum(D), 1.0)
    check("sum(C) = Überlappungsfläche", np.sum(C), 1.0)
    check("max|rowsum(D)-rowsum(C)|", np.max(np.abs(np.sum(D, axis=1) - np.sum(C, axis=1))), 0.0)

    lumped = np.sum(D, axis=1)
    if np.any(lumped <= 0.0):
        print(f"  [FAIL] Gelumpte Gewichte nicht strikt positiv: {lumped}")
        sys.exit(1)
    print(f"  [OK]   Alle gelumpten Gewichte positiv (min = {np.min(lumped):.6f})")

    if el_type == "CONQUAD8":
        # Analytisch mit Basistransformation alpha = 1/3: Ecken 5/36, Mittelknoten 1/9
        expected = np.array([5.0 / 36.0] * 4 + [1.0 / 9.0] * 4)
        if not np.allclose(lumped, expected, atol=1e-10):
            print(f"  [FAIL] Gelumpte Gewichte {lumped} != analytisch {expected}")
            sys.exit(1)
        print("  [OK]   Gelumpte Gewichte entsprechen analytischen Werten (5/36, 1/9)")

    print(f"* Teste {el_type}: Master um 0.3 in x verschoben...")
    constraint = build_single_facet_model(el_type, points_func(), points_func(shift_x=0.3))
    D, C = constraint.compute_mortar_coupling_matrices()
    check("sum(D) = Überlappungsfläche", np.sum(D), 0.7, tol=1e-10)
    check("max|rowsum(D)-rowsum(C)|", np.max(np.abs(np.sum(D, axis=1) - np.sum(C, axis=1))), 0.0, tol=1e-12)

    # Gelumpte Gewichte bei PARTIELLER Überdeckung: durch die segment-
    # quadraturbasierten dualen Koeffizienten (konsistente Randbehandlung,
    # Cichosz & Bischoff 2011 / MOOSE reinitDual) sind sie robust positiv,
    # solange die transformierte Basis punktweise nicht-negativ ist (CONQUAD4,
    # CONQUAD8). Die Lagrange-Mittelknotenfunktionen des CONQUAD9 wechseln
    # punktweise das Vorzeichen, daher sind dort winzige negative Gewichte an
    # überdeckungsfernen Knoten möglich (vom sgn_D-Guard im Active-Set
    # abgefangen). Toleriert werden 5% des größten Gewichts.
    lumped = np.sum(D, axis=1)
    bound = -0.05 * np.max(lumped)
    if el_type in ("CONQUAD4", "CONQUAD8"):
        bound = -1e-12
    if np.any(lumped < bound):
        print(f"  [FAIL] Gelumpte Gewichte bei Teilüberdeckung zu negativ: {lumped}")
        sys.exit(1)
    print(f"  [OK]   Gelumpte Gewichte bei Teilüberdeckung im zulässigen Bereich (min = {np.min(lumped):.6f})")

    print(f"  [PASS] {el_type} erfolgreich verifiziert!")


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

    check("sum(D) = Überlappungsfläche", np.sum(D), 1.0, tol=1e-10)
    check("sum(C) = Überlappungsfläche", np.sum(C), 1.0, tol=1e-10)
    check("max|rowsum(D)-rowsum(C)|", np.max(np.abs(np.sum(D, axis=1) - np.sum(C, axis=1))), 0.0, tol=1e-12)
    print("  [PASS] CONTRI6 erfolgreich verifiziert!")


if __name__ == "__main__":
    print("====================================================")
    print("MORTAR SUB-ZELLEN-SEGMENTIERUNG VERIFIKATIONSTEST")
    print("====================================================")

    run_quad_test("CONQUAD4", lambda shift_x=0.0: quad8_points(shift_x)[:4])
    run_quad_test("CONQUAD8", quad8_points)
    run_quad_test("CONQUAD9", quad9_points)
    run_tri6_test()

    print("\n====================================================")
    print("ALLE SEGMENTIERUNGSTESTS ERFOLGREICH PASSIERT!")
    print("====================================================")
