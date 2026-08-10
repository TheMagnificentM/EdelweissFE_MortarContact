#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Zwei-Block-Netzgenerator für den 2D-Kontakt-Patch-Test.

Warum nicht der eingebaute planeRectQuad-Generator?
---------------------------------------------------
``planeRectQuad`` versetzt die Knoten- und Elementlabels NICHT gegen bereits im
Modell vorhandene Entitäten: es startet immer bei ``currentNodeLabel = 1``
(``generators/planerectquad.py:130,141``). ``boxGen`` tut das dagegen
(``generators/boxgen.py:145-147,180-182``: ``if model.nodes:
currentNodeLabel += max(model.nodes.keys())``), weshalb der 3D-Patch-Test
(08_patch_test_hex20) zwei Blöcke problemlos aus zwei boxGen-Aufrufen bauen kann.

Zwei planeRectQuad-Aufrufe im selben Modell überschreiben sich deshalb
gegenseitig in ``model.nodes``/``model.elements``, während die Elemente des ersten
Blocks weiter auf die verdrängten Node-Objekte zeigen. Der DofManager bricht dann
mit einem KeyError auf deren FieldVariables ab. Damit der Patch-Test nicht von
diesem Verhalten abhängt, wird das Netz hier explizit aufgebaut.

Das Netz folgt der Knotenkonvention von planeRectQuad (= Abaqus): Ecken gegen den
Uhrzeigersinn ab links unten, Mittelknoten 4..7 auf den Kanten unten/rechts/oben/
links. Die Face-IDs sind 1=unten, 2=rechts, 3=oben, 4=links.
"""

import numpy as np

from edelweissfe.config.elementlibrary import getElementClass
from edelweissfe.points.node import Node
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.sets.nodeset import NodeSet


def _build_block(model, el_type, x0, y0, lx, ly, nX, nY):
    """Ein rechteckiger Block aus CPE4- oder CPE8-Elementen.

    Rückgabe: (elements, lattice) mit lattice[(iX, iY)] -> Node. Der Lattice-Index
    zählt für CPE8 in Halbschritten (0 .. 2*nX bzw. 0 .. 2*nY); die für
    Serendipity-Elemente unbenutzten Elementmittelknoten (beide Indizes ungerade)
    werden gar nicht erzeugt.

    Es wird der Standard-Elementprovider (marmot) verwendet - nur er ist mit der
    Materialangabe ``*material, name=LinearElastic`` der .inp-Datei kompatibel
    (``sections/plane.py:127-134``: der edelweiss-Provider erwartet ein
    Material-Objekt statt Name + Parameterliste).
    """
    ElClass = getElementClass(el_type)
    quadratic = ElClass(el_type, 0).nNodes == 8
    step = 2 if quadratic else 1

    nLatX, nLatY = step * nX + 1, step * nY + 1
    dx, dy = lx / (nLatX - 1), ly / (nLatY - 1)

    nextNode = (max(model.nodes.keys()) + 1) if model.nodes else 1
    lattice = {}
    for iX in range(nLatX):
        for iY in range(nLatY):
            if quadratic and iX % 2 == 1 and iY % 2 == 1:
                continue  # Elementmittelknoten: bei Serendipity nicht vorhanden
            node = Node(nextNode, np.array([x0 + iX * dx, y0 + iY * dy]))
            model.nodes[nextNode] = node
            lattice[(iX, iY)] = node
            nextNode += 1

    nextEl = (max(model.elements.keys()) + 1) if model.elements else 1
    elements = {}
    for ix in range(nX):
        for iy in range(nY):
            a, b = step * ix, step * iy
            if quadratic:
                idx = [
                    (a, b), (a + 2, b), (a + 2, b + 2), (a, b + 2),
                    (a + 1, b), (a + 2, b + 1), (a + 1, b + 2), (a, b + 1),
                ]
            else:
                idx = [(a, b), (a + 1, b), (a + 1, b + 1), (a, b + 1)]
            el = ElClass(el_type, nextEl)
            el.setNodes([lattice[k] for k in idx])
            model.elements[nextEl] = el
            elements[(ix, iy)] = el
            nextEl += 1

    return elements, lattice, step


def build(model, el_type, nX_a, nX_b, nY=1, length=1.0, height=1.0, length_b=None):
    """Zwei aufeinanderliegende Blöcke: A in y in [0, height], B in y in [height, 2*height].

    Block A reicht in x von 0 bis ``length``, Block B von 0 bis ``length_b``; ohne
    Angabe ist B deckungsgleich mit A. Ein schmalerer Block B lässt einen Teil der
    Slave-Fläche ohne Gegenüber -- der Lastfall, in dem das Active Set tatsächlich
    arbeitet (10_signorini_check). Beide Blöcke beginnen bei x = 0, damit ``symm_x``
    für beide auf derselben Symmetrieachse liegt.

    Legt an:
      Elementsets  ``solids_a``, ``solids_b``
      Nodesets     ``fixed_bottom`` (Unterseite A), ``load_top`` (Oberseite B),
                   ``symm_x`` (linke Kante beider Blöcke)
      Surfaces     ``surf_a_top`` (faceID 3), ``surf_b_bottom`` (faceID 1)
    """
    if length_b is None:
        length_b = length

    els_a, lat_a, step = _build_block(model, el_type, 0.0, 0.0, length, height, nX_a, nY)
    els_b, lat_b, _ = _build_block(model, el_type, 0.0, height, length_b, height, nX_b, nY)

    model.elementSets["solids_a"] = ElementSet("solids_a", list(els_a.values()))
    model.elementSets["solids_b"] = ElementSet("solids_b", list(els_b.values()))

    topY = step * nY

    model.nodeSets["fixed_bottom"] = NodeSet(
        "fixed_bottom", [nd for (iX, iY), nd in sorted(lat_a.items()) if iY == 0]
    )
    model.nodeSets["load_top"] = NodeSet(
        "load_top", [nd for (iX, iY), nd in sorted(lat_b.items()) if iY == topY]
    )
    # u_x = 0 nur auf der linken Kante (Symmetriebedingung). Sie ist mit der
    # exakten Loesung vereinbar (nu = 0 -> u_x = 0 ueberall) und entfernt die
    # x-Starrkoerpermoden. Entscheidend: die uebrigen Interface-Knoten bleiben in
    # x FREI, sodass ein faelschlich tangential klemmender Kontakt als u_x != 0
    # sichtbar wuerde.
    model.nodeSets["symm_x"] = NodeSet(
        "symm_x",
        [nd for (iX, iY), nd in sorted(lat_a.items()) if iX == 0]
        + [nd for (iX, iY), nd in sorted(lat_b.items()) if iX == 0],
    )

    model.surfaces["surf_a_top"] = {3: [els_a[(ix, nY - 1)] for ix in range(nX_a)]}
    model.surfaces["surf_b_bottom"] = {1: [els_b[(ix, 0)] for ix in range(nX_b)]}

    model._populateNodeFieldVariablesFromElements()
