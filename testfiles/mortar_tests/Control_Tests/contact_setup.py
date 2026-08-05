#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemeinsamer executePythonCode-Helfer fuer die Control_Tests.

Baut aus den beiden boxGen-Bloecken (genA = unten/Slave, genB = oben/Master) die
Mortar-Kontaktflaechen als CONQUAD-Overlay-Elemente auf:

    genA_top    ->  con_slave   (nonMortarSurface, Normale +y)
    genB_bottom ->  con_master  (mortarSurface,    Normale -y)

Der CONQUAD-Typ wird automatisch aus dem Elementtyp des jeweiligen Blocks
abgeleitet (C3D8/C3D8R -> CONQUAD4, C3D20/C3D20R -> CONQUAD8), sodass ein und
dieselbe .inp-Zeile fuer hex8-, hex20- und hex20R-Varianten funktioniert.

Zusaetzlich wird der Knotensatz 'allnodes' angelegt (fuer die laterale
Randbedingung 1=0, 3=0).

Wird aus den .inp-Dateien so aufgerufen (quoten-, komma- und '='-frei, weil der
Inputparser Datalines an Kommas splittet, Quotes entfernt und '='-Zeilen als
kwargs interpretiert):

    *modelGenerator, generator=executePythonCode, name=contactgen
    import os
    import sys
    sys.path.append(os.getcwd())
    sys.path.append(os.path.dirname(os.getcwd()))
    import contact_setup
    contact_setup.setup(model)
"""

import make_contact_elements as mce
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.sets.nodeset import NodeSet


def add_allnodes(model):
    """Knotensatz 'allnodes' anlegen (fuer die Monolith-Referenz ohne Kontakt)."""
    model.nodeSets["allnodes"] = NodeSet("allnodes", list(model.nodes.values()))


HERTZ_R = 10.0  # Kruemmungsradius des parabolischen Indenters (Hertz-Test)


def setup_hertz(model, slaveBlock="genA", masterBlock="genB"):
    """Hertz-Kontakt: die Unterseite des Indenters (masterBlock) wird parabolisch
    verschoben (y += x^2/(2R)), sodass sie eine gekruemmte Kontaktflaeche bekommt
    und ueber der flachen Foundation (slaveBlock) einen Anfangsspalt x^2/(2R) bildet.

    WICHTIG: NUR die Knoten des Indenters verschieben (ueber die Elementmenge
    masterBlock_all bestimmt) - nicht ueber die y-Koordinate, sonst wuerde auch die
    Foundation-Oberseite (gleiche y-Hoehe) mitgekruemmt und der Spalt verschwaende.
    Danach die Element-Referenzgeometrie neu erfassen (wie bei setup_inclined)."""
    import numpy as np

    master_ids = set()
    for el in model.elementSets[masterBlock + "_all"]:
        for nd in el.nodes:
            master_ids.add(id(nd))
    for n in model.nodes.values():
        if id(n) in master_ids:
            x = n.coordinates[0]
            n.coordinates[1] = float(n.coordinates[1]) + x * x / (2.0 * HERTZ_R)
    for el in model.elements.values():
        el.setNodes(el.nodes)

    setup(model, slaveBlock, masterBlock)  # Slave=flache Foundation-Oberseite, Master=gekruemmt


def setup_inclined(model, slaveBlock="genA", masterBlock="genB"):
    """Wie setup(), aber die gesamte Geometrie wird vorher um 30 Grad um die
    z-Achse gedreht. Damit liegt die Kontaktflaeche schief im Raum und der Test
    prueft, ob die Knoten-Normalen auf einer nicht achsparallelen Flaeche korrekt
    berechnet werden. Die Drehung erfolgt VOR dem Aufbau der Kontaktelemente,
    damit deren Normalen auf der gedrehten Geometrie bestimmt werden."""
    import numpy as np

    deg = 30.0
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    for n in model.nodes.values():
        n.coordinates[:] = R @ np.asarray(n.coordinates, dtype=float)

    # WICHTIG: Die Volumenelemente wurden von boxGen VOR der Drehung erzeugt und
    # haben dabei ihre Referenz-Knotenkoordinaten (undeformte Geometrie) erfasst.
    # Nach dem Drehen der Knoten muessen sie diese neu einlesen, sonst rechnen sie
    # Steifigkeit und Dehnung aus der UNGEDREHTEN Referenz (B_ungedreht * u_gedreht)
    # -> falsche Spannung/Loesung. setNodes() erfasst die (gedrehten) Koordinaten neu.
    for el in model.elements.values():
        el.setNodes(el.nodes)

    setup(model, slaveBlock, masterBlock)  # Kontaktelemente werden erst hier (gedreht) erzeugt


def _con_type_from_surface(model, surfaceName):
    """CONQUAD-Typ aus dem ersten Volumenelement der Flaeche ableiten."""
    for _faceID, elements in model.surfaces[surfaceName].items():
        for el in elements:
            return "CONQUAD8" if "C3D20" in el.elType.upper() else "CONQUAD4"
    raise Exception(f"Surface '{surfaceName}' ist leer - kann CONQUAD-Typ nicht ableiten.")


def setup(model, slaveBlock="genA", masterBlock="genB"):
    """Erzeuge con_slave / con_master und den Knotensatz allnodes."""
    slaveSurf = slaveBlock + "_top"
    masterSurf = masterBlock + "_bottom"

    conSlave = _con_type_from_surface(model, slaveSurf)
    conMaster = _con_type_from_surface(model, masterSurf)

    # Slave (Block A oben): Facettennormale soll nach +y (zum Master) zeigen
    mce.apply(model, slaveSurf, "con_slave", conSlave, (0.0, 1.0, 0.0))
    # Master (Block B unten): Facettennormale soll nach -y (zum Slave) zeigen
    mce.apply(model, masterSurf, "con_master", conMaster, (0.0, -1.0, 0.0))

    # elSet 'solids' = nur die Volumenelemente (ohne CONQUAD-Kontaktelemente),
    # damit perElement-Spannungs-/Dehnungs-Output nicht ueber die Kontaktelemente
    # laeuft (die haben keine Spannungsergebnisse).
    solids = list(model.elementSets[slaveBlock + "_all"]) + list(model.elementSets[masterBlock + "_all"])
    model.elementSets["solids"] = ElementSet("solids", solids)

    model.nodeSets["allnodes"] = NodeSet("allnodes", list(model.nodes.values()))
