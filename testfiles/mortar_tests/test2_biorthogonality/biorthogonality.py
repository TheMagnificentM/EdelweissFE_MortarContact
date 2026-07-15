#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test 2: Verifizierung der Biorthogonalität (Schritt 2)
======================================================

Dieser Test prüft die Korrektheit der dualen Basisfunktionen auf den Slave-Grenzflächen.

Die Biorthogonalität besagt, dass:
   integral( M_bar_i * N_j * dGamma ) = 0   für alle i != j.

Der Test führt für jedes Kontaktelement drei Kontrollen durch:
- Kontrolle A: Ist D_e eine echte Diagonalmatrix (Nebendiagonaleinträge = 0)?
- Kontrolle B: Stimmt das Matrix-Produkt A_e * M_e exakt mit D_e überein?
- Kontrolle C: Ergibt die punktweise Gauß-Integration des Produkts aus dualer Formfunktion
               M_bar_i und Standard-Formfunktion N_j exakt die Diagonale D_e?
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
    zu sammeln. Die Reihenfolge folgt der Abaqus-Konvention.
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
    """
    ConClass = getElementClass(contact_el_type, "edelweiss")
    max_el_id = max(model.elements.keys()) if model.elements else 0
    max_node_id = max(model.nodes.keys()) if model.nodes else 0
    
    contact_elements = []
    for faceID, elements in model.surfaces[surface_name].items():
        for el in elements:
            facet_nodes = get_local_nodes(el, faceID)
            
            # Spezialfall CONQUAD9
            if contact_el_type == "CONQUAD9" and len(facet_nodes) == 8:
                coords = np.array([node.coordinates for node in facet_nodes[:4]])
                center_coords = np.mean(coords, axis=0)
                max_node_id += 1
                center_node = Node(max_node_id, center_coords)
                model.nodes[max_node_id] = center_node
                facet_nodes.append(center_node)
                
            # Spezialfall CONTRI3
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
                
            # Spezialfall CONTRI6
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

def run_biorthogonality_test(el_type, skewed=False, dim=3):
    """
    Verifiziert die Biorthogonalität für ein Kontaktelement.
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
    
    # 6. Berechnen der lokalen Biorthogonal-Matrizen
    dual_matrices = contact.compute_local_dual_matrices()
    
    # 7. Kontrolle der Biorthogonalitäts-Bedingungen an jedem Element
    for el, faceID in contact.non_mortar_facets:
        M_e, D_e, A_e = dual_matrices[el.elNumber]
        n_nodes = el.nNodes
        
        # Kontrolle A: Ist D_e eine echte Diagonalmatrix?
        # Nebendiagonalelemente müssen numerisch Null sein.
        D_diag = np.diag(np.diag(D_e))
        if not np.allclose(D_e, D_diag, atol=1e-14):
            print(f"  [FAIL] D_e ist nicht diagonal für Element {el.elNumber}!")
            sys.exit(1)
            
        # Kontrolle B: Prüfen der algebraischen Relation A_e * M_e = D_e
        B_e = A_e @ M_e
        if not np.allclose(B_e, D_e, atol=1e-12):
            print(f"  [FAIL] A_e @ M_e ist ungleich D_e für Element {el.elNumber}!")
            sys.exit(1)
            
        # Kontrolle C: Punktweise Gauß-Integration (physikalischer Biorthogonalitätstest)
        # Wir integrieren integral( M_bar_i * N_j * dGamma ) explizit über die Gauß-Punkte.
        # Dies ist der ultimative Test für die mathematische Richtigkeit der dualen Basisfunktion.
        points, weights = el.getQuadraturePoints()
        coords = np.array([node.coordinates for node in el.nodes])
        B_explicit = np.zeros((n_nodes, n_nodes))
        
        for local_coords, w in zip(points, weights):
            N = el.getShapeFunctions(local_coords)
            jac = el.getJacobianAndAreaWeight(local_coords, coords)
            dGamma = jac * w
            
            # Wert der dualen Formfunktion an diesem Punkt: M_bar = A_e * N
            M_bar = A_e @ N
            B_explicit += np.outer(M_bar, N) * dGamma
            
        if not np.allclose(B_explicit, D_e, atol=1e-12):
            print(f"  [FAIL] Explizite Biorthogonalitäts-Integration fehlgeschlagen für Element {el.elNumber}!")
            sys.exit(1)
            
    print(f"  [PASS] Element {el_type} ({config_name}) erfolgreich verifiziert!")

if __name__ == "__main__":
    print("====================================================")
    print("MORTAR KONTAKT BIORTHOGONALITÄT VERIFIKATIONSTEST")
    print("====================================================")
    
    # 3D Vierecke
    run_biorthogonality_test("CONQUAD4", skewed=False, dim=3)
    run_biorthogonality_test("CONQUAD4", skewed=True, dim=3)
    
    run_biorthogonality_test("CONQUAD8", skewed=False, dim=3)
    run_biorthogonality_test("CONQUAD8", skewed=True, dim=3)
    
    run_biorthogonality_test("CONQUAD9", skewed=False, dim=3)
    run_biorthogonality_test("CONQUAD9", skewed=True, dim=3)
    
    # 3D Dreiecke
    run_biorthogonality_test("CONTRI3", skewed=False, dim=3)
    run_biorthogonality_test("CONTRI3", skewed=True, dim=3)
    
    run_biorthogonality_test("CONTRI6", skewed=False, dim=3)
    run_biorthogonality_test("CONTRI6", skewed=True, dim=3)
    
    # 2D Linien
    run_biorthogonality_test("CONLINE2", skewed=False, dim=2)
    run_biorthogonality_test("CONLINE2", skewed=True, dim=2)
    
    run_biorthogonality_test("CONLINE3", skewed=False, dim=2)
    run_biorthogonality_test("CONLINE3", skewed=True, dim=2)
    
    print("\n====================================================")
    print("ALLE BIORTHOGONALITÄTSTESTS ERFOLGREICH PASSIERT!")
    print("====================================================")
