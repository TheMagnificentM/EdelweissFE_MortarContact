#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test 8: Kontakt-Patch-Test für 2D-Elemente (CPE4 / CPE8)
========================================================

2D-Analogon zum hex20-Patch-Test (test6). Zwei elastische Blöcke werden über ein
NICHT-passendes Interface gegeneinander gepresst; ein konsistenter Mortar-Kontakt
muss den konstanten Druckzustand EXAKT übertragen (Puso & Laursen 2004; Farah 2018,
App. A.2.1).

Aufbau: Block A (unten, y in [0,1]), Block B (oben, y in [1,2]), linear elastisch
(E=1000, nu=0). Verschiebungsgesteuert: u_y=-2p/E an der Oberseite von B, Unterseite
von A in y gehalten, seitlich u_x=0. Exakte Lösung: sigma_yy=-p=-10, u_y=-p*y/E,
u_x=0, Kontaktdruck = p.

Dieser Test verifiziert die VEKTORIELLE, reibungsbereite Sattelpunkt-Formulierung
(z_I in R^dim) in 2D: die Normalkomponente z_I.n_I muss dem Druck p entsprechen,
die Tangentialkomponente muss verschwinden (reibungsfrei).

Varianten:
  1. CPE4 / CONLINE2, nicht-passend (2 gegen 3 Elemente) - lineare Facetten
  2. CPE8 / CONLINE3, nicht-passend (2 gegen 3 Elemente) - quadratische Facetten
"""

import os
import sys

import numpy as np

TESTDIR = os.path.abspath(os.path.dirname(__file__))

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.utils.inputfileparser import parseInputFile

E_MOD = 1000.0
PRESSURE = 10.0

# Lokale Kantenknoten je 2D-Volumenelement-Face (Abaqus-Konvention, 0-basiert).
# faceID 1 = Unterkante (y_min), faceID 3 = Oberkante (y_max).
FACE_MAPS_2D = {
    "CPE4": {1: [0, 1], 3: [2, 3]},
    "CPS4": {1: [0, 1], 3: [2, 3]},
    "CPE8": {1: [0, 1, 4], 3: [2, 3, 6]},
    "CPS8": {1: [0, 1, 4], 3: [2, 3, 6]},
}

INP_TEMPLATE = """*modelGenerator, generator=planeRectQuad, name=genA
x0=0.0
y0=0.0
l=1.0
h=1.0
nX={nxA}
nY=1
elType={elA}

*modelGenerator, generator=planeRectQuad, name=genB
x0=0.0
y0=1.0
l=1.0
h=1.0
nX={nxB}
nY=1
elType={elB}

*modelGenerator, generator=executePythonCode, name=contactgen
import generated_setup_{name} as vs
vs.setup(model)

*material, name=LinearElastic, id=mat
{E}, 0.0

*section, name=secA, material=mat, type=plane, thickness=1.0
genA_all
*section, name=secB, material=mat, type=plane, thickness=1.0
genB_all

*constraint, type=mortarcontact, name=contact
nonMortarSurface=con_slave
mortarSurface=con_master

*job, name=patchjob, domain=2d
*solver, name=theSolver, solver=NIST

*step, solver=theSolver
maxInc=0.5, minInc=1e-3, maxNumInc=100, maxIter=25, stepLength=1
>>dirichlet, name=bot, nSet=genA_bottom, field=displacement, 2=0.0
>>dirichlet, name=top, nSet=genB_top, field=displacement, 2={utop}
>>dirichlet, name=lat, nSet=allnodes, field=displacement, 1=0.0
"""


def attach_2d_contact(model, surface_name, new_surface_name, con_type, expected_normal):
    """Erzeugt CONLINE2/CONLINE3-Kontaktelemente auf einer 2D-Volumen-Oberfläche
    und dreht die Knotenreihenfolge ggf. um, sodass die Facettennormale in Richtung
    expected_normal zeigt (der Mortar-Löser prüft die Orientierung nicht selbst)."""
    from edelweissfe.config.elementlibrary import getElementClass

    expected_normal = np.asarray(expected_normal, dtype=float)
    ConClass = getElementClass(con_type, "edelweiss")
    max_el_id = max(model.elements.keys()) if model.elements else 0

    contact_elements = []
    for faceID, elements in model.surfaces[surface_name].items():
        for el in elements:
            base_type = el.elType.upper()
            idx = FACE_MAPS_2D[base_type][faceID]
            facet_nodes = [el.nodes[i] for i in idx]

            coords = np.array([n.coordinates[:2] for n in facet_nodes])
            t = coords[1] - coords[0]
            n = np.array([t[1], -t[0]])
            if np.dot(n, expected_normal) < 0.0:
                # Kanten umdrehen: Endknoten tauschen (Mittelknoten bleibt hinten)
                if len(facet_nodes) == 2:
                    facet_nodes = [facet_nodes[1], facet_nodes[0]]
                else:
                    facet_nodes = [facet_nodes[1], facet_nodes[0], facet_nodes[2]]

            max_el_id += 1
            con = ConClass(con_type, max_el_id)
            con.setNodes(facet_nodes)
            model.elements[max_el_id] = con
            contact_elements.append(con)

    model.surfaces[new_surface_name] = {1: contact_elements}


def setup_factory(name, conA, conB, path):
    setup_lines = [
        "import sys",
        f"sys.path.insert(0, r'{TESTDIR}')",
        "import patch_test_2d as p2",
        "from edelweissfe.sets.nodeset import NodeSet",
        "",
        "def setup(model):",
        f"    p2.attach_2d_contact(model, 'genA_top', 'con_slave', '{conA}', (0.0, 1.0))",
        f"    p2.attach_2d_contact(model, 'genB_bottom', 'con_master', '{conB}', (0.0, -1.0))",
        "    # Nur Knoten mit displacement-Feld (CPE8: ungenutzte Zentrumsknoten des",
        "    # planeRectQuad-Gitters ausschließen).",
        "    alln = [n for n in model.nodes.values() if 'displacement' in n.fields]",
        "    model.nodeSets['allnodes'] = NodeSet('allnodes', alln)",
    ]
    with open(path, "w") as f:
        f.write("\n".join(setup_lines) + "\n")


def run_variant(name, elA, nxA, conA, elB, nxB, conB, tol_u=1e-8, tol_lam=1e-8):
    print(f"\n* Variante: {name}")

    setup_path = os.path.join(TESTDIR, f"generated_setup_{name}.py")
    setup_factory(name, conA, conB, setup_path)
    if TESTDIR not in sys.path:
        sys.path.insert(0, TESTDIR)

    inp_text = INP_TEMPLATE.format(
        name=name, nxA=nxA, elA=elA, nxB=nxB, elB=elB,
        E=E_MOD, utop=-2.0 * PRESSURE / E_MOD,
    )
    inp_path = os.path.join(TESTDIR, f"generated_{name}.inp")
    with open(inp_path, "w") as f:
        f.write(inp_text)

    inputFile = parseInputFile(inp_path)
    model, _ = finiteElementSimulation(inputFile, verbose=False, suppressPlots=True)

    # --- Kontrolle 1: Verschiebungsfeld gleich der exakten Lösung ---
    nf = model.nodeFields["displacement"]
    U = nf["U"]
    u_ref = PRESSURE * 2.0 / E_MOD

    max_err_lateral = 0.0
    max_err_uy = 0.0
    for node, u in zip(nf.nodes, U):
        y = node.coordinates[1]
        uy_exact = -PRESSURE * y / E_MOD
        max_err_lateral = max(max_err_lateral, abs(u[0]))
        max_err_uy = max(max_err_uy, abs(u[1] - uy_exact))

    print(f"  max|u_x|             = {max_err_lateral:.3e}  (Toleranz {tol_u * u_ref:.1e})")
    print(f"  max|u_y - u_y_exakt| = {max_err_uy:.3e}  (Toleranz {tol_u * u_ref:.1e})")
    ok = not (max_err_lateral > tol_u * u_ref or max_err_uy > tol_u * u_ref)

    # --- Kontrolle 2: konstanter Kontaktdruck (Normalkomponente z_I.n_I) ---
    contact = model.constraints["contact"]
    lambdas = np.asarray(contact.recovered_lambdas).flatten()
    tractions = np.asarray(contact.recovered_tractions)
    active = np.asarray(contact.active_set)

    lam_active = lambdas[active]
    lam_err = np.max(np.abs(np.abs(lam_active) - PRESSURE)) / PRESSURE
    same_sign = np.all(lam_active > 0) or np.all(lam_active < 0)

    # Tangentialkomponente z_I - (z_I.n_I) n_I muss verschwinden (reibungsfrei)
    normals = contact.current_normals
    tang_norm = 0.0
    for I in np.flatnonzero(active):
        z = tractions[I]
        n = normals[I]
        z_t = z - (z @ n) * n
        tang_norm = max(tang_norm, np.linalg.norm(z_t))

    print(f"  aktive Knoten: {int(np.sum(active))}/{len(active)}, "
          f"max. rel. Abw. Normaldruck von p = {lam_err:.3e}")
    print(f"  max |Tangentialtraktion| (reibungsfrei -> 0) = {tang_norm:.3e}")

    ok &= same_sign and lam_err <= tol_lam and tang_norm <= tol_lam * PRESSURE

    if ok:
        print(f"  [PASS] Variante '{name}' erfolgreich!")
        os.remove(inp_path)
        os.remove(setup_path)
    else:
        print(f"  [FAIL] Variante '{name}'!")
        sys.exit(1)


if __name__ == "__main__":
    print("====================================================")
    print("MORTAR KONTAKT-PATCH-TEST (2D, CPE4 / CPE8)")
    print("====================================================")
    run_variant("cpe4_nonmatching", "CPE4", 2, "CONLINE2", "CPE4", 3, "CONLINE2")
    run_variant("cpe8_nonmatching", "CPE8", 2, "CONLINE3", "CPE8", 3, "CONLINE3")
    print("\n====================================================")
    print("ALLE 2D-PATCH-TESTS ERFOLGREICH PASSIERT!")
    print("====================================================")
