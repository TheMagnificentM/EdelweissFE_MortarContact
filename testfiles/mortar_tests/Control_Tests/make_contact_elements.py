#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hilfsmodul für den Kontakt-Patch-Test: erzeugt CONQUAD4/CONQUAD8-Overlay-Elemente
auf boxGen-Oberflächen und stellt sicher, dass die Facettennormalen in die erwartete
Richtung zeigen (der Mortar-Löser prüft die Orientierung nicht selbst).

Wird vom executePythonCode-Modelgenerator in den generierten .inp-Dateien importiert.
"""

import numpy as np

from edelweissfe.config.elementlibrary import getElementClass

# Lokale Facettenknoten je Volumenelement-Face (Abaqus-Konvention, 0-basiert)
FACE_MAPS = {
    "C3D8": {
        1: [3, 2, 1, 0],
        2: [4, 5, 6, 7],
        3: [0, 1, 5, 4],
        4: [6, 5, 1, 2],
        5: [7, 6, 2, 3],
        6: [4, 7, 3, 0],
    },
    "C3D20": {
        1: [3, 2, 1, 0, 10, 9, 8, 11],
        2: [4, 5, 6, 7, 12, 13, 14, 15],
        3: [0, 1, 5, 4, 8, 17, 12, 16],
        4: [6, 5, 1, 2, 13, 17, 9, 18],
        5: [7, 6, 2, 3, 14, 18, 10, 19],
        6: [4, 7, 3, 0, 15, 19, 11, 16],
    },
}

# Permutation, die die Orientierung (Normale) einer Facette umkehrt
REVERSE_PERM = {
    4: [0, 3, 2, 1],
    8: [0, 3, 2, 1, 7, 6, 5, 4],
}


def _corner_normal(coords):
    if len(coords) in (3, 6):
        return np.cross(coords[1] - coords[0], coords[2] - coords[0])
    return np.cross(coords[2] - coords[0], coords[3] - coords[1])


def apply(model, surface_name, new_surface_name, con_type, expected_normal):
    """Erzeugt Kontaktelemente vom Typ con_type auf der Volumen-Oberfläche
    surface_name und registriert sie als neue Oberfläche new_surface_name.
    Die Knotenreihenfolge wird ggf. umgedreht, sodass die Facettennormale in
    Richtung expected_normal zeigt."""
    expected_normal = np.asarray(expected_normal, dtype=float)
    ConClass = getElementClass(con_type, "edelweiss")
    max_el_id = max(model.elements.keys()) if model.elements else 0

    contact_elements = []
    for faceID, elements in model.surfaces[surface_name].items():
        for el in elements:
            base_type = "C3D20" if "C3D20" in el.elType.upper() else "C3D8"
            idx = FACE_MAPS[base_type][faceID]
            facet_nodes = [el.nodes[i] for i in idx]

            # Bei CONQUAD4 auf quadratischen Volumina: nur die Eckknoten verwenden
            n_con = int(con_type.upper().replace("CONQUAD", ""))
            if n_con == 4 and len(facet_nodes) == 8:
                facet_nodes = facet_nodes[:4]

            coords = np.array([nd.coordinates for nd in facet_nodes])
            if np.dot(_corner_normal(coords), expected_normal) < 0.0:
                perm = REVERSE_PERM[len(facet_nodes)]
                facet_nodes = [facet_nodes[i] for i in perm]

            max_el_id += 1
            con_el = ConClass(con_type, max_el_id)
            con_el.setNodes(facet_nodes)
            model.elements[max_el_id] = con_el
            contact_elements.append(con_el)

    model.surfaces[new_surface_name] = {1: contact_elements}


def distort_inplane(model, amplitude=0.05):
    """Verzerrt die x/z-Koordinaten ALLER Knoten mit einer glatten Funktion, die auf
    dem Rand des Einheitsquadrats verschwindet. Die Belastungsrichtung y bleibt
    unverändert (Säulen bleiben vertikal), sodass die exakte Patch-Test-Lösung
    gültig bleibt; die Interface-Netze (inkl. Mittelknoten) werden dadurch in der
    Ebene gekrümmt - Elementkanten der quadratischen Facetten sind dann gekrümmt."""
    for node in model.nodes.values():
        x, z = node.coordinates[0], node.coordinates[2]
        f = amplitude * np.sin(np.pi * x) * np.sin(np.pi * z)
        node.coordinates[0] = x + f
        node.coordinates[2] = z - f
