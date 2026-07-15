#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test 1: Verifizierung der Knotennormalen
========================================

Dieser Test prüft die geometrische Korrektheit der Knotennormalen auf der Slave-Fläche.

Was dieser Test schrittweise tut:
1. Er erzeugt ein einfaches Blockgitter (2D oder 3D) mit EdelweissFE-Standardelementen.
2. Er identifiziert die Ränder und legt die Geometrie-Kontaktelemente (CON-Elemente) darüber.
3. Er verzerrt (skews) optional die Knotenpositionen mit Zufallswerten, um gekrümmte Oberflächen zu testen.
4. Er initialisiert das MortarContact3D-Constraint.
5. Er prüft, ob die berechneten Knotennormalen Einheitsvektoren sind (Länge = 1) und
   ob sie senkrecht auf den Elementfacetten stehen.
"""

import sys
import os
import numpy as np

# Den lokalen Pfad von EdelweissFE hinzufügen, damit die Python-Imports funktionieren
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from edelweissfe.models.femodel import FEModel
from edelweissfe.generators.boxgen import generateModelData as generateBoxMesh
from edelweissfe.generators.planerectquad import generateModelData as generatePlaneMesh
from edelweissfe.journal.journal import Journal
from edelweissfe.config.elementlibrary import getElementClass
from edelweissfe.constraints.mortarcontact3d import Constraint as MortarContact3D
from edelweissfe.points.node import Node

def get_local_nodes(el, faceID):
    """
    Hilfsfunktion, um die Knoten einer bestimmten Außenfläche (faceID) eines 3D-Volumenelements
    zu sammeln. Die Reihenfolge folgt der Abaqus-Konvention, was sicherstellt, 
    dass die Normale am Ende nach außen zeigt.
    """
    elType = el.elType.upper()
    if "C3D8" in elType:
        if faceID == 1: idx = [3, 2, 1, 0]
        elif faceID == 2: idx = [4, 5, 6, 7]
        elif faceID == 3: idx = [0, 1, 5, 4]
        elif faceID == 4: idx = [6, 5, 1, 2]
        elif faceID == 5: idx = [7, 6, 2, 3]
        elif faceID == 6: idx = [4, 7, 3, 0]
    elif "C3D20" in elType:
        if faceID == 1: idx = [3, 2, 1, 0, 10, 9, 8, 11]
        elif faceID == 2: idx = [4, 5, 6, 7, 12, 13, 14, 15]
        elif faceID == 3: idx = [0, 1, 5, 4, 8, 17, 12, 16]
        elif faceID == 4: idx = [6, 5, 1, 2, 13, 17, 9, 18]
        elif faceID == 5: idx = [7, 6, 2, 3, 14, 18, 10, 19]
        elif faceID == 6: idx = [4, 7, 3, 0, 15, 19, 11, 16]
    elif "CPS4" in elType or "CPE4" in elType:
        if faceID == 1: idx = [0, 1]
        elif faceID == 2: idx = [1, 2]
        elif faceID == 3: idx = [2, 3]
        elif faceID == 4: idx = [3, 0]
    elif "CPS8" in elType or "CPE8" in elType:
        if faceID == 1: idx = [0, 1, 4]
        elif faceID == 2: idx = [1, 2, 5]
        elif faceID == 3: idx = [2, 3, 6]
        elif faceID == 4: idx = [3, 0, 7]
    else:
        raise NotImplementedError(f"Local face mapping not defined for {elType}")
    return [el.nodes[i] for i in idx]

def apply_contact_elements(model, surface_name, contact_el_type):
    """
    Erzeugt Kontaktelemente auf einer Oberfläche.
    Teilt z. B. Vierecksflächen in Dreiecke auf, falls Dreieckskontakt (CONTRI3/CONTRI6) getestet wird,
    und erstellt das Kontakt-Oberflächen-Set ("con_...").
    """
    ConClass = getElementClass(contact_el_type, "edelweiss")
    max_el_id = max(model.elements.keys()) if model.elements else 0
    max_node_id = max(model.nodes.keys()) if model.nodes else 0
    
    contact_elements = []
    for faceID, elements in model.surfaces[surface_name].items():
        for el in elements:
            facet_nodes = get_local_nodes(el, faceID)
            
            # Spezialfall CONQUAD9: benötigt einen zusätzlichen Mittelknoten
            if contact_el_type == "CONQUAD9" and len(facet_nodes) == 8:
                coords = np.array([node.coordinates for node in facet_nodes[:4]])
                center_coords = np.mean(coords, axis=0)
                max_node_id += 1
                center_node = Node(max_node_id, center_coords)
                model.nodes[max_node_id] = center_node
                facet_nodes.append(center_node)
                
            # Spezialfall CONTRI3: teilt Viereck in zwei Dreiecke
            if contact_el_type == "CONTRI3" and len(facet_nodes) == 4:
                max_el_id += 1
                con_el1 = ConClass(contact_el_type, max_el_id)
                con_el1.setNodes([facet_nodes[0], facet_nodes[1], facet_nodes[2]])
                model.elements[max_el_id] = con_el1
                contact_elements.append(con_el1)
                
                max_el_id += 1
                con_el2 = ConClass(contact_el_type, max_el_id)
                con_el2.setNodes([facet_nodes[0], facet_nodes[2], facet_nodes[3]])
                model.elements[max_el_id] = con_el2
                contact_elements.append(con_el2)
                continue
                
            # Spezialfall CONTRI6: teilt quadratisches Viereck in zwei quadratische Dreiecke
            if contact_el_type == "CONTRI6" and len(facet_nodes) == 8:
                mid_coords = 0.5 * (facet_nodes[0].coordinates + facet_nodes[2].coordinates)
                max_node_id += 1
                mid_node = Node(max_node_id, mid_coords)
                model.nodes[max_node_id] = mid_node
                
                max_el_id += 1
                con_el1 = ConClass(contact_el_type, max_el_id)
                con_el1.setNodes([facet_nodes[0], facet_nodes[1], facet_nodes[2], facet_nodes[4], facet_nodes[5], mid_node])
                model.elements[max_el_id] = con_el1
                contact_elements.append(con_el1)
                
                max_el_id += 1
                con_el2 = ConClass(contact_el_type, max_el_id)
                con_el2.setNodes([facet_nodes[0], facet_nodes[2], facet_nodes[3], mid_node, facet_nodes[6], facet_nodes[7]])
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

def run_contact_test(el_type, skewed=False, dim=3):
    """
    Führt den Normalen-Verifikationstest aus.
    """
    config_name = "Skewed" if skewed else "Normal"
    print(f"\n* Teste Elementtyp: {el_type} ({config_name} Geometrie)...")
    
    model = FEModel(dimension=dim)
    journal = Journal()
    
    # 1. Block-Mesh generieren
    if dim == 3:
        vol_type = "C3D20" if ("8" in el_type or "9" in el_type or "6" in el_type) else "C3D8"
        generateBoxMesh({"name": "gen"}, model, journal, **{
            "nX": 2, "nY": 2, "nZ": 2,
            "lX": 2.0, "lY": 2.0, "lZ": 2.0,
            "elType": vol_type, "elProvider": "edelweiss"
        })
    else: # dim == 2
        vol_type = "CPS8" if "3" in el_type else "CPS4"
        generatePlaneMesh({"name": "gen"}, model, journal, **{
            "nX": 2, "nY": 2,
            "l": 2.0, "h": 2.0,
            "elType": vol_type, "elProvider": "edelweiss"
        })
        
    # 2. Knoten verzerren (falls 'skewed' gewählt wurde)
    if skewed:
        np.random.seed(42)
        for node in model.nodes.values():
            offset = np.random.uniform(-0.15, 0.15, size=dim)
            node.coordinates += offset
            
    # 3. Kontaktelemente erzeugen
    slave_surf = apply_contact_elements(model, "gen_bottom", el_type)
    master_surf = apply_contact_elements(model, "gen_top", el_type)
    
    # 4. Verschiebungsfelder initialisieren
    from edelweissfe.variables.fieldvariable import FieldVariable
    for node in model.nodes.values():
        node.fields["displacement"] = FieldVariable(node, "displacement")
        
    # 5. Mortar-Constraint instanziieren
    kwargs = {"nonMortarSurface": slave_surf, "mortarSurface": master_surf, "field": "displacement"}
    contact = MortarContact3D(f"contact_{el_type}_{config_name.lower()}", model, **kwargs)
    
    # 6. Normalen verifizieren
    normals = contact.undeformed_normals
    
    # Test A: Prüfen, ob die Knotennormalen normiert sind (Länge = 1)
    for i, node in enumerate(contact.non_mortar_nodes):
        n = normals[i]
        length = np.linalg.norm(n)
        if not np.isclose(length, 1.0, atol=1e-7):
            print(f"  [FAIL] Knotennormale an Knoten {node.label} nicht normiert: Länge = {length:.6f}")
            sys.exit(1)
            
    # Test B: Richtungsprüfung auf ebener Fläche (muss straight nach unten zeigen: -Y)
    if not skewed:
        expected = np.zeros(dim)
        expected[1] = -1.0  # In Y-Richtung nach unten
        for i, node in enumerate(contact.non_mortar_nodes):
            n = normals[i]
            if not np.allclose(n, expected, atol=1e-7):
                print(f"  [FAIL] Knoten {node.label} Normale falsch ausgerichtet: normal = {n}")
                sys.exit(1)
        print("  -> Ebener Ausrichtungstest: OK")
        
    # Test C: Orthogonalitätsprüfung auf verzerrter Geometrie
    # Das Skalarprodukt zwischen der Normale und den Tangenten der Facette muss Null sein.
    else:
        all_orthogonal = True
        for el, faceID in contact.non_mortar_facets:
            coords = np.array([node.coordinates for node in el.nodes])
            if dim == 3:
                if len(el.nodes) in (3, 6):
                    v1 = coords[1] - coords[0]
                    v2 = coords[2] - coords[0]
                else:
                    v1 = coords[2] - coords[0]
                    v2 = coords[3] - coords[1]
                n_facet = np.cross(v1, v2)
                dot1 = np.dot(n_facet, v1)
                dot2 = np.dot(n_facet, v2)
                if not (np.isclose(dot1, 0.0, atol=1e-12) and np.isclose(dot2, 0.0, atol=1e-12)):
                    all_orthogonal = False
            else: # dim == 2
                t = coords[-1] - coords[0]
                n_facet = np.array([t[1], -t[0]])
                dot = np.dot(n_facet, t)
                if not np.isclose(dot, 0.0, atol=1e-12):
                    all_orthogonal = False
                    
        if all_orthogonal:
            print("  -> Verzerrter Orthogonalitätstest: OK")
        else:
            print("  [FAIL] Orthogonalitätstest fehlgeschlagen für verzerrte Geometrie!")
            sys.exit(1)
            
    print(f"  [PASS] Test für {el_type} ({config_name}) erfolgreich!")

if __name__ == "__main__":
    print("====================================================")
    print("MORTAR KONTAKT NORMALE VERIFIKATIONSTEST")
    print("====================================================")
    
    # 3D Vierecke
    run_contact_test("CONQUAD4", skewed=False, dim=3)
    run_contact_test("CONQUAD4", skewed=True, dim=3)
    
    run_contact_test("CONQUAD8", skewed=False, dim=3)
    run_contact_test("CONQUAD8", skewed=True, dim=3)
    
    run_contact_test("CONQUAD9", skewed=False, dim=3)
    run_contact_test("CONQUAD9", skewed=True, dim=3)
    
    # 3D Dreiecke
    run_contact_test("CONTRI3", skewed=False, dim=3)
    run_contact_test("CONTRI3", skewed=True, dim=3)
    
    run_contact_test("CONTRI6", skewed=False, dim=3)
    run_contact_test("CONTRI6", skewed=True, dim=3)
    
    # 2D Linien
    run_contact_test("CONLINE2", skewed=False, dim=2)
    run_contact_test("CONLINE2", skewed=True, dim=2)
    
    run_contact_test("CONLINE3", skewed=False, dim=2)
    run_contact_test("CONLINE3", skewed=True, dim=2)
    
    print("\n====================================================")
    print("ALLE GEOMETRISCHEN NORMALENTESTS ERFOLGREICH PASSIERT!")
    print("====================================================")
