#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hilfsmodul für den 2D-Kontakt-Patch-Test: erzeugt CONLINE2/CONLINE3-Overlay-
Elemente auf planeRectQuad-Kanten und richtet ihre Normalen aus (der Mortar-Löser
prüft die Orientierung nicht selbst).

2D-Gegenstück zu 08_patch_test_hex20/make_contact_elements.py. Wird vom
executePythonCode-Modelgenerator in den generierten .inp-Dateien importiert.
"""

import numpy as np

from edelweissfe.config.elementlibrary import getElementClass

# Lokale Kantenknoten je 2D-Element-Face (Abaqus-Konvention, 0-basiert).
# planeRectQuad numeriert die Ecken gegen den Uhrzeigersinn ab links unten und
# die Mittelknoten 4..7 auf den Kanten unten/rechts/oben/links; die Face-IDs des
# Generators sind 1=unten, 2=rechts, 3=oben, 4=links.
FACE_MAPS = {
    4: {
        1: [0, 1],
        2: [1, 2],
        3: [2, 3],
        4: [3, 0],
    },
    8: {
        1: [0, 1, 4],
        2: [1, 2, 5],
        3: [2, 3, 6],
        4: [3, 0, 7],
    },
}

# Permutation, die die Orientierung (Normale) einer Kante umkehrt. Der
# Mittelknoten bleibt an Position 2 - das erwartet CONLINE3 (SUB_CELL_MAP
# [[0, 2], [2, 1]] setzt Index 2 als Mittelknoten voraus).
REVERSE_PERM = {
    2: [1, 0],
    3: [1, 0, 2],
}


def _edge_normal(coords):
    """Normale einer Kante wie in mortarcontact.compute_normals: n = [t_y, -t_x]
    mit t = coords[-1] - coords[0]. Für CONLINE3 ist coords[-1] der Mittelknoten,
    die Richtung ist auf geraden Kanten dieselbe."""
    t = coords[-1] - coords[0]
    return np.array([t[1], -t[0]])


def apply(model, surface_name, new_surface_name, con_type, expected_normal):
    """Erzeugt Kontaktelemente vom Typ con_type auf der Kante surface_name und
    registriert sie als neue Oberfläche new_surface_name. Die Knotenreihenfolge
    wird ggf. umgedreht, sodass die Kantennormale in Richtung expected_normal
    zeigt."""
    expected_normal = np.asarray(expected_normal, dtype=float)
    ConClass = getElementClass(con_type, "edelweiss")
    max_el_id = max(model.elements.keys()) if model.elements else 0

    n_con = 2 if con_type.upper() == "CONLINE2" else 3

    contact_elements = []
    for faceID, elements in model.surfaces[surface_name].items():
        for el in elements:
            idx = FACE_MAPS[el.nNodes][faceID]
            facet_nodes = [el.nodes[i] for i in idx]

            # CONLINE2 auf quadratischen Volumenelementen: nur die Eckknoten.
            if n_con == 2 and len(facet_nodes) == 3:
                facet_nodes = facet_nodes[:2]

            coords = np.array([nd.coordinates[:2] for nd in facet_nodes])
            if np.dot(_edge_normal(coords), expected_normal) < 0.0:
                facet_nodes = [facet_nodes[i] for i in REVERSE_PERM[len(facet_nodes)]]

            max_el_id += 1
            con_el = ConClass(con_type, max_el_id)
            con_el.setNodes(facet_nodes)
            model.elements[max_el_id] = con_el
            contact_elements.append(con_el)

    model.surfaces[new_surface_name] = {1: contact_elements}
