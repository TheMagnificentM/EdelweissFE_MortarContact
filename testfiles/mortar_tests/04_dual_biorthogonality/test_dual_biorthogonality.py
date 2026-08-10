#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test 4: Verifizierung der Biorthogonalität der dualen Basis
===========================================================

Dieser Test prüft die Korrektheit der dualen Basisfunktionen auf den Slave-Grenzflächen.

Die Biorthogonalität wird bezüglich der TRANSFORMIERTEN Basis N_tilde = T_e * N
erzwungen (Basistransformation nach Popp et al. 2012 / Farah 2018, Abschn. 6.2.3.2;
T_e = Identität für lineare Elemente):

   integral( M_bar_i * N_tilde_j * dGamma ) = delta_ij * integral( N_tilde_i * dGamma )

Der Test führt für jedes Kontaktelement vier Kontrollen durch:
- Kontrolle A: Ist D_e eine echte Diagonalmatrix (Nebendiagonaleinträge = 0)?
- Kontrolle B: Stimmt das Matrix-Produkt A_e * inv(T_e) * M_e exakt mit D_e überein?
               (M_e ist die Massenmatrix der transformierten Basis, A_e bildet
               Standard-Formfunktionswerte N auf duale Werte M_bar ab.)
- Kontrolle C: Ergibt die punktweise Gauß-Integration des Produkts aus dualer Formfunktion
               M_bar_i und transformierter Formfunktion N_tilde_j exakt die Diagonale D_e?
- Kontrolle D: Sind alle dualen Gewichte D_e[i,i] strikt positiv? (Zweck der
               Basistransformation; ohne sie wären z.B. die Eckknoten-Gewichte
               eines CONQUAD8 negativ.)
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
from edelweissfe.constraints.mortarcontact import Constraint as MortarContact3D
from edelweissfe.points.node import Node
from edelweissfe.variables.fieldvariable import FieldVariable

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
        T_e = el.getBasisTransformation()

        # Kontrolle A: Ist D_e eine echte Diagonalmatrix?
        # Nebendiagonalelemente müssen numerisch Null sein.
        D_diag = np.diag(np.diag(D_e))
        if not np.allclose(D_e, D_diag, atol=1e-14):
            print(f"  [FAIL] D_e ist nicht diagonal für Element {el.elNumber}!")
            raise AssertionError("Biorthogonalitaetspruefung fehlgeschlagen -- siehe [FAIL] oben")

        # Kontrolle B: Prüfen der algebraischen Relation A_e * inv(T_e) * M_e = D_e
        # (A_e = D_e * inv(M_e) * T_e, mit M_e als Massenmatrix der transformierten Basis)
        B_e = A_e @ np.linalg.solve(T_e, M_e)
        if not np.allclose(B_e, D_e, atol=1e-12):
            print(f"  [FAIL] A_e @ inv(T_e) @ M_e ist ungleich D_e für Element {el.elNumber}!")
            raise AssertionError("Biorthogonalitaetspruefung fehlgeschlagen -- siehe [FAIL] oben")

        # Kontrolle C: Punktweise Gauß-Integration (physikalischer Biorthogonalitätstest)
        # Wir integrieren integral( M_bar_i * N_tilde_j * dGamma ) explizit über die
        # Gauß-Punkte. Dies ist der ultimative Test für die mathematische Richtigkeit
        # der dualen Basisfunktion bezüglich der transformierten Basis.
        points, weights = el.getQuadraturePoints()
        coords = np.array([node.coordinates for node in el.nodes])
        B_explicit = np.zeros((n_nodes, n_nodes))

        for local_coords, w in zip(points, weights):
            N = el.getShapeFunctions(local_coords)
            N_tilde = T_e @ N
            jac = el.getJacobianAndAreaWeight(local_coords, coords)
            dGamma = jac * w

            # Wert der dualen Formfunktion an diesem Punkt: M_bar = A_e * N
            M_bar = A_e @ N
            B_explicit += np.outer(M_bar, N_tilde) * dGamma

        if not np.allclose(B_explicit, D_e, atol=1e-12):
            print(f"  [FAIL] Explizite Biorthogonalitäts-Integration fehlgeschlagen für Element {el.elNumber}!")
            raise AssertionError("Biorthogonalitaetspruefung fehlgeschlagen -- siehe [FAIL] oben")

        # Kontrolle D: Positivität der dualen Gewichte (Popp et al. 2012)
        if np.any(np.diag(D_e) <= 0.0):
            print(f"  [FAIL] Duale Gewichte D_e[i,i] nicht strikt positiv für Element {el.elNumber}!")
            print(f"         diag(D_e) = {np.diag(D_e)}")
            raise AssertionError("Biorthogonalitaetspruefung fehlgeschlagen -- siehe [FAIL] oben")
            
    print(f"  [PASS] Element {el_type} ({config_name}) erfolgreich verifiziert!")

# ===========================================================================
# Unabhaengige Referenzen
#
# Die Kontrollen A-D oben pruefen die Konstruktion GEGEN SICH SELBST: D_e ist per
# Konstruktion diagonal, A_e @ inv(T_e) @ M_e = D_e folgt in einer Zeile aus
# A_e = D_e inv(M_e) T_e, und die "explizite" Integration benutzt DIESELBE
# Quadraturregel, mit der A_e gebaut wurde -- ausmultipliziert steht dort wieder
# D_e inv(M_e) M_e = D_e. Drei der vier Kontrollen koennen also nicht fehlschlagen,
# solange numpy rechnet. Nur Kontrolle D (Positivitaet) prueft etwas.
#
# Die folgenden drei Kontrollen bringen unabhaengige Referenzen ins Spiel.
# ===========================================================================


def dual_reference_quad4(xi, eta):
    """Geschlossene duale Basis des bilinearen Vierecks auf dem Referenzquadrat.

    Fuer den linearen 1D-Ansatz auf [-1,1] ist M = [[2/3,1/3],[1/3,2/3]], D = I,
    also A = inv(M) = [[2,-1],[-1,2]] und damit

        Phi_1 = 2 N_1 - N_2 = (1 - 3 xi)/2,   Phi_2 = (1 + 3 xi)/2.

    Das Viereck ist das Tensorprodukt zweier solcher Ansaetze, die dualen
    Funktionen sind daher die Produkte der eindimensionalen. Diese Formel steckt
    nirgends im Code -- sie ist die unabhaengige Referenz.
    """
    p = lambda t: (1.0 - 3.0 * t) / 2.0  # noqa: E731
    m = lambda t: (1.0 + 3.0 * t) / 2.0  # noqa: E731
    return np.array([
        p(xi) * p(eta),
        m(xi) * p(eta),
        m(xi) * m(eta),
        p(xi) * m(eta),
    ])


def high_order_rule(el_type):
    """Eine ANDERE, hoehergradige Quadraturregel als die der Konstruktion.

    Die Biorthogonalitaet ist diskret bezueglich der Regel definiert, mit der A_e
    gebaut wurde. Auf einem UNVERZERRTEN Element ist diese Regel exakt, die
    Biorthogonalitaet gilt dann auch fuer jede andere Regel -- das ist die
    Aussage, die hier geprueft wird. Auf einem verzerrten Element ist der
    Integrand rational, keine Regel ist exakt, und die Differenz ist der
    Quadraturfehler: dort wird gemessen, nicht bestanden/durchgefallen.
    """
    el_type = el_type.upper()
    if "LINE" in el_type:
        # 5-Punkt-Gauss-Legendre auf [-1,1], exakt bis Grad 9
        x = np.array([0.0, np.sqrt(5 - 2 * np.sqrt(10 / 7.0)) / 3,
                      -np.sqrt(5 - 2 * np.sqrt(10 / 7.0)) / 3,
                      np.sqrt(5 + 2 * np.sqrt(10 / 7.0)) / 3,
                      -np.sqrt(5 + 2 * np.sqrt(10 / 7.0)) / 3])
        w = np.array([128 / 225.0,
                      (322 + 13 * np.sqrt(70)) / 900.0, (322 + 13 * np.sqrt(70)) / 900.0,
                      (322 - 13 * np.sqrt(70)) / 900.0, (322 - 13 * np.sqrt(70)) / 900.0])
        return [np.array([xi]) for xi in x], list(w)

    if "TRI" in el_type:
        # Dunavant Grad 6, 12 Punkte -- deutlich hoeher als die Grad-5-Regel des Codes
        a1, w1 = 0.063089014491502, 0.050844906370207
        a2, w2 = 0.249286745170910, 0.116786275726379
        a3, b3, w3 = 0.310352451033785, 0.053145049844816, 0.082851075618374
        pts, wts = [], []
        for a in (a1, a2):
            pts += [np.array([a, a]), np.array([1 - 2 * a, a]), np.array([a, 1 - 2 * a])]
        wts += [w1] * 3 + [w2] * 3
        for u, v in ((a3, b3), (b3, a3), (a3, 1 - a3 - b3), (1 - a3 - b3, a3),
                     (b3, 1 - a3 - b3), (1 - a3 - b3, b3)):
            pts.append(np.array([u, v]))
        wts += [w3] * 6
        # Dunavant-Gewichte summieren zu 1, das Referenzdreieck hat Flaeche 1/2
        return pts, [0.5 * w for w in wts]

    # Vierecke: 5x5-Gauss-Produktregel, exakt bis Grad 9 je Richtung
    x = np.array([0.0, np.sqrt(5 - 2 * np.sqrt(10 / 7.0)) / 3,
                  -np.sqrt(5 - 2 * np.sqrt(10 / 7.0)) / 3,
                  np.sqrt(5 + 2 * np.sqrt(10 / 7.0)) / 3,
                  -np.sqrt(5 + 2 * np.sqrt(10 / 7.0)) / 3])
    w = np.array([128 / 225.0,
                  (322 + 13 * np.sqrt(70)) / 900.0, (322 + 13 * np.sqrt(70)) / 900.0,
                  (322 - 13 * np.sqrt(70)) / 900.0, (322 - 13 * np.sqrt(70)) / 900.0])
    pts = [np.array([x[i], x[j]]) for i in range(5) for j in range(5)]
    wts = [w[i] * w[j] for i in range(5) for j in range(5)]
    return pts, wts


def test_dual_basis_against_closed_form():
    """Kontrolle E: CONQUAD4 gegen die geschlossene duale Basis.

    Auf dem Referenzquadrat muss A_e @ N genau ``dual_reference_quad4`` liefern.
    Das ist die einzige Kontrolle der Reihe, die den Code gegen eine Formel prueft,
    die nicht aus ihm stammt.
    """
    print("\n* Kontrolle E: CONQUAD4 gegen die geschlossene duale Basis")
    ConClass = getElementClass("CONQUAD4", "edelweiss")
    el = ConClass("CONQUAD4", 1)

    model = FEModel(dimension=3)
    coords = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]])
    nodes = []
    for i, c in enumerate(coords):
        model.nodes[i + 1] = Node(i + 1, c)
        nodes.append(model.nodes[i + 1])
    el.setNodes(nodes)

    _, _, A_e = el.computeLocalMassMatrices(coords)

    worst = 0.0
    for xi in (-0.9, -0.3, 0.0, 0.42, 0.87):
        for eta in (-0.75, -0.1, 0.0, 0.55, 0.95):
            lc = np.array([xi, eta])
            phi_code = A_e @ el.getShapeFunctions(lc)
            phi_ref = dual_reference_quad4(xi, eta)
            worst = max(worst, float(np.max(np.abs(phi_code - phi_ref))))

    print(f"  max|Phi_Code - Phi_geschlossen| = {worst:.3e}   (Toleranz 1e-13)")
    assert worst < 1e-13, "Die duale Basis des CONQUAD4 stimmt nicht mit der geschlossenen Form ueberein"
    print("  [PASS] Geschlossene Form reproduziert")


def test_biorthogonality_with_independent_quadrature():
    """Kontrolle F: Biorthogonalitaet, mit einer ANDEREN Quadraturregel nachgerechnet.

    Auf unverzerrten Elementen muss das exakt aufgehen -- dort ist die
    Konstruktionsregel exakt, die Biorthogonalitaet gilt also nicht nur diskret.
    Auf verzerrten Elementen wird der Unterschied nur gemessen: er ist der
    Quadraturfehler der Konstruktionsregel, kein Fehler der Implementierung.
    """
    print("\n* Kontrolle F: Biorthogonalitaet mit hoehergradiger Regel")

    UNDISTORTED = {
        "CONQUAD4": np.array([[-1.0, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=float),
        "CONQUAD8": np.array([[-1.0, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0],
                              [0, -1, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0]], dtype=float),
        "CONQUAD9": np.array([[-1.0, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0],
                              [0, -1, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, 0, 0]], dtype=float),
        "CONTRI3": np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        "CONTRI6": np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0],
                             [0.5, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0]]),
        "CONLINE2": np.array([[-1.0, 0.0], [1.0, 0.0]]),
        "CONLINE3": np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 0.0]]),
    }

    for el_type, coords in UNDISTORTED.items():
        dim = coords.shape[1]
        ConClass = getElementClass(el_type, "edelweiss")
        el = ConClass(el_type, 1)
        model = FEModel(dimension=dim)
        nodes = []
        for i, c in enumerate(coords):
            model.nodes[i + 1] = Node(i + 1, np.asarray(c, dtype=float))
            nodes.append(model.nodes[i + 1])
        el.setNodes(nodes)

        _, D_e, A_e = el.computeLocalMassMatrices(coords)
        T_e = el.getBasisTransformation()

        pts, wts = high_order_rule(el_type)
        B = np.zeros_like(D_e)
        for lc, w in zip(pts, wts):
            N = el.getShapeFunctions(lc)
            dG = el.getJacobianAndAreaWeight(lc, coords) * w
            B += np.outer(A_e @ N, T_e @ N) * dG

        err = float(np.max(np.abs(B - D_e)))
        scale = float(np.max(np.abs(np.diag(D_e))))
        print(f"  {el_type:9s} max|B_hochgradig - D_e| / max(D_e) = {err / scale:.3e}")
        assert err / scale < 1e-12, (
            f"{el_type}: die Biorthogonalitaet haelt einer unabhaengigen Quadraturregel nicht "
            "stand -- die Konstruktionsregel ist auf dem unverzerrten Element nicht exakt"
        )
    print("  [PASS] Biorthogonalitaet ist regelunabhaengig auf unverzerrten Elementen")


def test_segment_consistent_dual_basis():
    """Kontrolle G: die TATSAECHLICH verwendete duale Basis (Segmentquadratur).

    Die Kontrollen A-F pruefen ``compute_local_dual_matrices`` -- das ist der
    REFERENZELEMENT-Rueckfallpfad. Produktiv laeuft die konsistente Randbehandlung
    nach Cichosz & Bischoff (2011): A_e wird in PASS 2 von
    ``compute_mortar_coupling_matrices`` aus der echten Segmentquadratur gebaut,
    damit die Biorthogonalitaet auf dem tatsaechlichen Ueberlappungsgebiet gilt.
    Genau dieser Pfad war bisher numerisch unbelegt.

    Geprueft wird die beobachtbare Folge davon: aus
    int_ovl(Phi_j * N_tilde_b) = delta_jb * int_ovl(N_tilde_j) und
    N_tilde = T_e N folgt fuer die assemblierte Slave-Slave-Matrix

        (D @ T_e^T)_jb = int_ovl(Phi_j * N_tilde_b)   ist DIAGONAL,

    und ihre Diagonale sind die Knotengewichte. Das muss auch bei TEILueberdeckung
    gelten -- das ist der ganze Zweck der konsistenten Randbehandlung.
    """
    print("\n* Kontrolle G: Segmentkonsistente duale Basis (Produktionspfad)")

    def unit_quad(el_type, shift_x=0.0):
        z = 0.0
        c = [[0.0 + shift_x, 0, z], [1 + shift_x, 0, z], [1 + shift_x, 1, z], [0 + shift_x, 1, z]]
        if el_type in ("CONQUAD8", "CONQUAD9"):
            c += [[0.5 + shift_x, 0, z], [1 + shift_x, 0.5, z],
                  [0.5 + shift_x, 1, z], [0 + shift_x, 0.5, z]]
        if el_type == "CONQUAD9":
            c += [[0.5 + shift_x, 0.5, z]]
        return np.array(c, dtype=float)

    for el_type in ("CONQUAD4", "CONQUAD8", "CONQUAD9"):
        for shift, label in ((0.0, "volle Ueberdeckung"), (0.3, "70 % Ueberdeckung")):
            model = FEModel(dimension=3)
            ConClass = getElementClass(el_type, "edelweiss")

            s_nodes, m_nodes = [], []
            for i, c in enumerate(unit_quad(el_type)):
                model.nodes[i + 1] = Node(i + 1, c)
                s_nodes.append(model.nodes[i + 1])
            for i, c in enumerate(unit_quad(el_type, shift)):
                model.nodes[100 + i] = Node(100 + i, c)
                m_nodes.append(model.nodes[100 + i])

            s_el = ConClass(el_type, 1)
            s_el.setNodes(s_nodes)
            m_el = ConClass(el_type, 2)
            m_el.setNodes(m_nodes)
            model.elements[1], model.elements[2] = s_el, m_el
            model.surfaces = {"slave": {1: [s_el]}, "master": {1: [m_el]}}
            for nd in model.nodes.values():
                nd.fields["displacement"] = FieldVariable(nd, "displacement")

            c = MortarContact3D("c", model, nonMortarSurface="slave",
                                mortarSurface="master", field="displacement")
            D, _ = c.compute_mortar_coupling_matrices()

            # Reihenfolge der Slave-Knoten im Constraint = Reihenfolge in s_el.nodes
            T_e = s_el.getBasisTransformation()
            B = D @ T_e.T
            offdiag = float(np.max(np.abs(B - np.diag(np.diag(B)))))
            scale = float(np.max(np.abs(np.diag(B))))

            print(f"  {el_type:9s} {label:20s} max|Nebendiagonale| / max(Diagonale) = "
                  f"{offdiag / scale:.3e}")
            assert offdiag / scale < 1e-10, (
                f"{el_type} bei {label}: D @ T_e^T ist nicht diagonal -- die Biorthogonalitaet "
                "auf dem tatsaechlichen Ueberlappungsgebiet ist verletzt (PASS 2 von "
                "compute_mortar_coupling_matrices)"
            )

            # Und die Diagonale sind genau die Knotengewichte
            err_w = float(np.max(np.abs(np.diag(B) - np.sum(D, axis=1))))
            assert err_w / scale < 1e-10, (
                f"{el_type} bei {label}: die Diagonale von D @ T_e^T stimmt nicht mit den "
                "Zeilensummen von D ueberein"
            )

    print("  [PASS] Biorthogonalitaet gilt auf dem tatsaechlichen Ueberlappungsgebiet, "
          "auch bei Teilueberdeckung")


def test_dual_biorthogonality():
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


if __name__ == "__main__":
    print("====================================================")
    print("MORTAR KONTAKT BIORTHOGONALITÄT VERIFIKATIONSTEST")
    print("====================================================")
    test_dual_biorthogonality()
    print("\n====================================================")
    print("ALLE BIORTHOGONALITÄTSTESTS ERFOLGREICH PASSIERT!")
    print("====================================================")
