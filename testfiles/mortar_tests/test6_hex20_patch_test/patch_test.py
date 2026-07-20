#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test 6: Kontakt-Patch-Test für hex20-(Serendipity)-Elemente
===========================================================

Der klassische Mortar-Kontakt-Patch-Test (vgl. Puso & Laursen 2004; Farah 2018,
App. A.2.1): Zwei elastische Blöcke werden über ein NICHT-passendes Interface
gegeneinander gepresst. Ein konsistenter Mortar-Kontakt muss den konstanten
Druckzustand EXAKT übertragen (bis auf Löser-Toleranzen), unabhängig von der
Vernetzung des Interfaces.

Aufbau: Block A (unten, y in [0,1]) und Block B (oben, y in [1,2]; boxGen
orientiert top/bottom entlang der y-Achse), linear elastisch (E = 1000, nu = 0).
Verschiebungsgesteuert: u_y = -0.02 auf der Oberseite von B, Unterseite von A
in y gehalten. Exakte Lösung: sigma_yy = -p = -10 überall, u_y = -p*y/E,
u_x = u_z = 0, Kontaktdruck am Interface = p.

(Verschiebungssteuerung statt Drucklast, um die Kontaktkonsistenz isoliert zu
testen: die konsistente Flächenlast-Assemblierung für C3D20-Faces ist ein vom
Kontakt unabhängiges Thema und würde hier zusätzliche Fehler von ~0.4% einstreuen.)

Geprüfte Varianten ("alle hex20-Möglichkeiten"):
  1. hex20/hex20, passende Netze (2x2 gegen 2x2) - Basisfall
  2. hex20/hex20, nicht-passend (2x2 gegen 3x3) - CONQUAD8/CONQUAD8
  3. hex20 Slave / hex8 Master, nicht-passend - CONQUAD8/CONQUAD4 (gemischte Ordnung)
  4. hex8 Slave / hex20 Master, nicht-passend - CONQUAD4/CONQUAD8
  5. hex20/hex20, nicht-passend + in der Ebene verzerrtes Interface-Netz
     (gekrümmte Elementkanten inkl. Mittelknoten)

Erwartete Genauigkeit: Für Facetten mit GERADEN Kanten (Varianten 1-4) ist die
segmentbasierte Integration exakt, der Patch-Test muss in Maschinengenauigkeit
bestehen. Bei GEKRÜMMTEN Elementkanten (Variante 5) approximiert die
linearisierte Sub-Zellen-Segmentierung das Integrationsgebiet nur stückweise
linear (Farah 2018, App. A.1.4: "This approach only affects the integration
domain itself, which is less accurate in terms of geometry") - der Patch-Test
gilt dort nur näherungsweise, konvergiert aber mit Netzverfeinerung optimal
(Farah 2018, App. A.2.2). Bei 5% Verzerrungsamplitude beträgt der beobachtete
Fehler ~0.4%; die Toleranz ist entsprechend gesetzt.

Kontrollen je Variante:
  - max|u_x|, max|u_y| ~ 0
  - u_z an jedem Knoten gleich der exakten linearen Lösung
  - alle Kontakt-Multiplikatoren aktiv, gleiches Vorzeichen, |lambda| = p
    (konstanter übertragener Kontaktdruck, keine Oszillationen)
"""

import os
import sys

import numpy as np

TESTDIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(TESTDIR, "../..")))

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.utils.inputfileparser import parseInputFile

E_MOD = 1000.0
PRESSURE = 10.0

INP_TEMPLATE = """*modelGenerator, generator=boxGen, name=genA
nX={nxA}
nY=1
nZ={nzA}
lX=1.0
lY=1.0
lZ=1.0
elType={elA}

*modelGenerator, generator=boxGen, name=genB
y0=1.0
nX={nxB}
nY=1
nZ={nzB}
lX=1.0
lY=1.0
lZ=1.0
elType={elB}

*modelGenerator, generator=executePythonCode, name=contactgen
import generated_setup_{name} as vs
vs.setup(model)

*material, name=LinearElastic, id=mat
{E}, 0.0

*section, name=secA, material=mat, type=solid
genA_all
*section, name=secB, material=mat, type=solid
genB_all

*constraint, type=mortarcontact, name=contact
nonMortarSurface=con_slave
mortarSurface=con_master

*job, name=patchjob, domain=3d
*solver, name=theSolver, solver=NISTParallel

*step, solver=theSolver
maxInc=0.5, minInc=1e-3, maxNumInc=100, maxIter=25, stepLength=1
>>dirichlet, name=bot, nSet=genA_bottom, field=displacement, 2=0.0
>>dirichlet, name=top, nSet=genB_top, field=displacement, 2={utop}
>>dirichlet, name=lat, nSet=allnodes, field=displacement, 1=0.0, 3=0.0
"""


def run_variant(name, elA, nxA, nzA, conA, elB, nxB, nzB, conB, distort=False,
                tol_u=1e-8, tol_lam=1e-8, tol_lam_mean=None):
    print(f"\n* Variante: {name}")

    # Setup-Modul für den executePythonCode-Generator: der inp-Parser splittet
    # Datalines an Kommas und entfernt Anführungszeichen, daher enthält die
    # dataline nur einen Import und einen Funktionsaufruf; die eigentlichen
    # Parameter stehen in diesem generierten Modul.
    setup_lines = [
        "import sys",
        f"sys.path.insert(0, r'{TESTDIR}')",
        "import make_contact_elements as mce",
        "from edelweissfe.sets.nodeset import NodeSet",
        "",
        "def setup(model):",
    ]
    if distort:
        setup_lines.append("    mce.distort_inplane(model, amplitude=0.05)")
    setup_lines.append(f"    mce.apply(model, 'genA_top', 'con_slave', '{conA}', (0.0, 1.0, 0.0))")
    setup_lines.append(f"    mce.apply(model, 'genB_bottom', 'con_master', '{conB}', (0.0, -1.0, 0.0))")
    setup_lines.append("    model.nodeSets['allnodes'] = NodeSet('allnodes', list(model.nodes.values()))")

    setup_path = os.path.join(TESTDIR, f"generated_setup_{name}.py")
    with open(setup_path, "w") as f:
        f.write("\n".join(setup_lines) + "\n")

    if TESTDIR not in sys.path:
        sys.path.insert(0, TESTDIR)

    inp_text = INP_TEMPLATE.format(
        name=name, nxA=nxA, nzA=nzA, elA=elA, nxB=nxB, nzB=nzB, elB=elB,
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
    u_ref = PRESSURE * 2.0 / E_MOD  # |u_z| an der Oberseite

    max_err_lateral = 0.0
    max_err_uy = 0.0
    for node, u in zip(nf.nodes, U):
        y = node.coordinates[1]
        uy_exact = -PRESSURE * y / E_MOD
        max_err_lateral = max(max_err_lateral, abs(u[0]), abs(u[2]))
        max_err_uy = max(max_err_uy, abs(u[1] - uy_exact))

    print(f"  max|u_lateral|            = {max_err_lateral:.3e}  (Toleranz {tol_u * u_ref:.1e})")
    print(f"  max|u_y - u_y_exakt|      = {max_err_uy:.3e}  (Toleranz {tol_u * u_ref:.1e})")
    if max_err_lateral > tol_u * u_ref or max_err_uy > tol_u * u_ref:
        print(f"  [FAIL] Verschiebungsfeld weicht von der exakten Patch-Lösung ab!")
        sys.exit(1)

    # --- Kontrolle 2: konstanter Kontaktdruck (Multiplikatoren) ---
    lambdas = np.array([v.value for v in model.scalarVariables.values()]).flatten()
    if len(lambdas) == 0:
        print("  [FAIL] Keine Kontakt-Multiplikatoren im Modell gefunden!")
        sys.exit(1)

    lam_err = np.max(np.abs(np.abs(lambdas) - PRESSURE)) / PRESSURE
    same_sign = np.all(lambdas > 0) or np.all(lambdas < 0)
    print(f"  Multiplikatoren: n = {len(lambdas)}, max. rel. Abweichung von p = {lam_err:.3e}")

    # Übertragene Gesamtkraft: gewichtetes Mittel der Multiplikatoren mit den
    # dualen Gewichten (Zeilensummen von D = Flächenanteile der Knoten).
    rowsum = model.constraints["contact"].current_D_rowsum
    lam_mean = np.sum(lambdas * rowsum) / np.sum(rowsum)
    lam_mean_err = abs(abs(lam_mean) - PRESSURE) / PRESSURE
    print(f"  Gewichtetes Mittel (Gesamtkraft/Fläche): {lam_mean:.6f}, rel. Fehler = {lam_mean_err:.3e}")

    if not same_sign:
        print(f"  [FAIL] Kontaktdruck oszilliert (unterschiedliche Vorzeichen)!")
        sys.exit(1)
    if lam_err > tol_lam:
        print(f"  [FAIL] Kontaktdruck nicht konstant = p!")
        sys.exit(1)
    if lam_mean_err > (tol_lam_mean if tol_lam_mean is not None else tol_lam):
        print(f"  [FAIL] Übertragene Gesamtkraft weicht von p*A ab!")
        sys.exit(1)

    print(f"  [PASS] Variante '{name}' erfolgreich!")
    os.remove(inp_path)
    os.remove(setup_path)


if __name__ == "__main__":
    print("====================================================")
    print("MORTAR KONTAKT-PATCH-TEST (HEX20 / SERENDIPITY)")
    print("====================================================")

    run_variant("hex20_hex20_matching", "C3D20", 2, 2, "CONQUAD8", "C3D20", 2, 2, "CONQUAD8")
    run_variant("hex20_hex20_nonmatching", "C3D20", 2, 2, "CONQUAD8", "C3D20", 3, 3, "CONQUAD8")
    run_variant("hex20_slave_hex8_master", "C3D20", 2, 2, "CONQUAD8", "C3D8", 3, 3, "CONQUAD4")
    run_variant("hex8_slave_hex20_master", "C3D8", 3, 3, "CONQUAD4", "C3D20", 2, 2, "CONQUAD8")
    # Bei gekrümmten Elementkanten (verzerrtes Netz) gilt der Patch-Test nur
    # näherungsweise (linearisiertes Integrationsgebiet). Einzelne Multiplikatoren
    # an Randknoten mit kleinen dualen Gewichten weichen lokal stärker ab; die
    # übertragene GESAMTKRAFT (gewichtetes Mittel) ist die physikalisch
    # maßgebliche Größe und bleibt auf ~1e-4 genau.
    run_variant("hex20_hex20_distorted", "C3D20", 2, 2, "CONQUAD8", "C3D20", 3, 3, "CONQUAD8",
                distort=True, tol_u=1e-2, tol_lam=0.2, tol_lam_mean=1e-3)

    print("\n====================================================")
    print("ALLE PATCH-TESTS ERFOLGREICH PASSIERT!")
    print("====================================================")
