#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test 11: Skaleninvarianz (Längeneinheiten)
==========================================

Der Kontakt-Constraint enthält an mehreren Stellen ABSOLUTE, also
einheitenabhängige Toleranzen. Dieselbe Rechnung in verschiedenen Längeneinheiten
ist der direkte Test dafür: ein rein geometrisch skaliertes Problem (Längen x k,
vorgeschriebene Verschiebung x k, E unverändert) hat dieselben Dehnungen,
Spannungen und Kontaktdrücke; nur die Verschiebungen skalieren mit k.

WICHTIG -- 2D und 3D schneiden mit VERSCHIEDENEN Routinen. Der Test läuft deshalb
in beiden Dimensionen; eine reine 2D-Studie könnte die flächenwertigen Toleranzen
des Polygon-Clippings grundsätzlich nicht erreichen, gleichgültig wie weit sie die
Skala spreizt:

  Toleranz                                    Einheit  Pfad    erreicht von
  ------------------------------------------  -------  ------  -----------------
  margin = max(0.15*Ausdehnung, 0.1)          Länge    2D+3D   beiden Varianten
  tol = 1e-12 in map_2d_to_natural            Länge    2D+3D   beiden Varianten
  -1e-12 in inside() (Kreuzprodukt!)          FLÄCHE   nur 3D  der 3D-Variante
  area_jac < 1e-14                            FLÄCHE   nur 3D  der 3D-Variante
  |D_jk| > 1e-14 (Besetzungsmuster)           Fläche   2D+3D   beiden Varianten

Im 2D-Pfad wird ``sutherland_hodgman_clip`` gar nicht aufgerufen -- dort schneidet
``clip_1d_segments`` Intervalle. Eine reine 2D-Skalenstudie erreicht die beiden
flächenwertigen Toleranzen deshalb grundsätzlich nicht, gleichgültig wie weit sie
die Skala spreizt.

Geprüft werden k = 1e-3, 1 und 1e3. Damit ist sowohl der Fall abgedeckt, in dem
der relative Term der Suchmarge dominiert (grosse Modelle), als auch der, in dem
der absolute Term ``0.1`` alles überdeckt (kleine Modelle) -- dort entartet die
BVH-Suche zur vollständigen Paarung. Das ist ein Laufzeit-, kein
Korrektheitsproblem, und der Test macht es sichtbar.

Ausdrücklich mitgeprüft wird auch der semi-smooth Aktivierungsindikator
``s_n = p_n - c_n * g_sep``: ``p_n`` ist eine Spannung, ``g_sep`` eine LÄNGE, das
Produkt ``c_n * g_sep`` also nur dann eine Spannung, wenn ``c_n`` die Einheit
Spannung/Länge trägt. Bei festem ``c_n`` verschiebt sich die Schaltschwelle mit
der Längenskala. Auf die KONVERGIERTE Lösung wirkt sich das nicht aus (dort ist
``g_sep`` an aktiven Knoten null, die Lösung ist c_n-unabhängig) -- wohl aber auf
den Weg dorthin. Der Test rechnet deshalb alle Skalen mit demselben ``c_n = E``
und prüft, dass das Ergebnis trotzdem übereinstimmt.

Netz und Kontaktelemente: 2D aus 07_patch_test_2d, 3D aus 08_patch_test_hex20.
"""

import os
import sys

import numpy as np

TESTDIR = os.path.abspath(os.path.dirname(__file__))
MORTARDIR = os.path.abspath(os.path.join(TESTDIR, ".."))
PATCH2D_DIR = os.path.join(MORTARDIR, "07_patch_test_2d")
PATCH3D_DIR = os.path.join(MORTARDIR, "08_patch_test_hex20")
sys.path.insert(0, os.path.abspath(os.path.join(MORTARDIR, "../..")))

from contextlib import contextmanager

from edelweissfe.config import phenomena
from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.utils.inputfileparser import parseInputFile

E_MOD = 1000.0
PRESSURE = 10.0          # skaleninvariant
BASE_LENGTH = 1.0
BASE_HEIGHT = 1.0

# Absoluter Term der Suchmarge, aus mortarcontact.py. Wird hier nur zur
# Diagnose ausgewertet, nicht nachgebaut.
MARGIN_ABS = 0.1
MARGIN_REL = 0.15

RTOL = 1e-8

# Standardwert aus edelweissfe/config/phenomena.py. ABSOLUT, nicht relativ zum
# mittleren Fluss wie bei den Knotenfeldern.
DEFAULT_SCALAR_FLUX_TOL = 1e-8


def scalar_flux_tolerance(k, dim):
    """Absolute Schranke fuer das Residuum der lambda-Zeile, mit der Skala gefuehrt.

    Das Residuum der Multiplikatorzeile ist der GEWICHTETE Spalt
    g_weak = D_II * g_sep, in 3D also von der Dimension Laenge x Flaeche, in 2D
    Laenge x Laenge. Sein erreichbarer Boden ist durch die Maschinengenauigkeit der
    Koordinaten gegeben,

        Boden ~ D_II * |x| * eps ~ (k*L)^dim * eps,

    waechst also mit der dritten (2D: zweiten) Potenz der Modellausdehnung. Der
    Loeser prueft Skalarvariablen aber gegen eine ABSOLUTE Schranke
    (phenomena.py: fluxResidualTolerance["scalar variables"] = 1e-8), anders als
    die Knotenfelder, die relativ zum raeumlich gemittelten Fluss geprueft werden.
    Ab einer gewissen Modellausdehnung ist das Kriterium daher unerreichbar --
    NICHT weil die Loesung falsch waere, sondern weil die Schranke es ist. Genau
    das misst ``check_lambda_residual_floor`` unten.

    Diese Funktion fuehrt die Schranke mit der Skala und laesst den Standardwert
    dort stehen, wo er erreichbar ist.
    """
    L = 2.0 * BASE_LENGTH * k
    return float(max(DEFAULT_SCALAR_FLUX_TOL, 100.0 * L**dim * np.finfo(float).eps))


UPDATE_CONFIG = """*updateConfiguration, configuration=fluxResidualTolerance
scalar variables={tol}

"""


@contextmanager
def default_tolerances_restored():
    """Stellt die Loeser-Toleranzen nach dem Test wieder her.

    ``*updateConfiguration`` wirkt PROZESSWEIT und nicht nur auf den eigenen Job:
    ``loadConfiguration`` reicht die Modul-Dicts aus ``config/phenomena.py`` per
    Referenz durch, und ``updateConfiguration`` schreibt in genau dieses Objekt.
    Ohne Wiederherstellung rechnet jeder spaeter im selben Python-Prozess gestartete
    Job mit der hier gelockerten Schranke weiter -- was in einer Testreihe still
    andere Ergebnisse erzeugt (nachgewiesen an 12_increment_size, dessen
    Referenzlauf dadurch um 5e-5 relativ verschoben wurde).
    """
    saved = dict(phenomena.fluxResidualTolerance)
    try:
        yield
    finally:
        phenomena.fluxResidualTolerance.clear()
        phenomena.fluxResidualTolerance.update(saved)

# ===========================================================================
# 2D-Variante (Linienelemente, clip_1d_segments)
# ===========================================================================

INP_TEMPLATE_2D = """*modelGenerator, generator=executePythonCode, name=meshgen
import generated_setup_{name} as vs
vs.setup(model)

*material, name=LinearElastic, id=mat
{E}, 0.0

*section, name=secA, thickness=1.0, material=mat, type=plane
solids_a
*section, name=secB, thickness=1.0, material=mat, type=plane
solids_b

** cn ~ O(E) des weicheren Koerpers (Farah 2018, Abschn. 3.5.2). Bewusst auf
** ALLEN Skalen derselbe Wert -- siehe Modul-Docstring.
*constraint, type=mortarcontact, name=contact
nonMortarSurface=con_slave
mortarSurface=con_master
cn={E}

*job, name=scalejob, domain=2d
*solver, name=theSolver, solver=NISTParallel

*step, solver=theSolver
maxInc=0.5, minInc=1e-3, maxNumInc=100, maxIter=25, stepLength=1
>>dirichlet, name=bot,  nSet=fixed_bottom, field=displacement, 2=0.0
>>dirichlet, name=top,  nSet=load_top,     field=displacement, 2={utop}
>>dirichlet, name=symm, nSet=symm_x,       field=displacement, 1=0.0
"""

SETUP_TEMPLATE_2D = """import sys
sys.path.insert(0, r'{patchdir}')
import two_block_mesh_2d as tb
import make_contact_elements_2d as mce


def setup(model):
    tb.build(model, '{el_type}', {nx_a}, {nx_b}, length={length}, height={height})
    mce.apply(model, 'surf_a_top', 'con_slave', '{con_type}', (0.0, 1.0))
    mce.apply(model, 'surf_b_bottom', 'con_master', '{con_type}', (0.0, -1.0))
"""

# ===========================================================================
# 3D-Variante (Flaechenelemente, Sutherland-Hodgman + Dreiecksquadratur)
# ===========================================================================

INP_TEMPLATE_3D = """*modelGenerator, generator=boxGen, name=genA
nX={nx_a}
nY=1
nZ={nx_a}
lX={length}
lY={height}
lZ={length}
elType={el_type}

*modelGenerator, generator=boxGen, name=genB
y0={height}
nX={nx_b}
nY=1
nZ={nx_b}
lX={length}
lY={height}
lZ={length}
elType={el_type}

*modelGenerator, generator=executePythonCode, name=contactgen
import generated_setup_{name} as vs
vs.setup(model)

*material, name=LinearElastic, id=mat
{E}, 0.0

*section, name=secA, material=mat, type=solid
genA_all
*section, name=secB, material=mat, type=solid
genB_all

** cn ~ O(E) des weicheren Koerpers (Farah 2018, Abschn. 3.5.2). Bewusst auf
** ALLEN Skalen derselbe Wert -- siehe Modul-Docstring.
*constraint, type=mortarcontact, name=contact
nonMortarSurface=con_slave
mortarSurface=con_master
cn={E}

*job, name=scalejob3d, domain=3d
*solver, name=theSolver, solver=NISTParallel

*step, solver=theSolver
maxInc=0.5, minInc=1e-3, maxNumInc=100, maxIter=25, stepLength=1
>>dirichlet, name=bot, nSet=genA_bottom, field=displacement, 2=0.0
>>dirichlet, name=top, nSet=genB_top,    field=displacement, 2={utop}
>>dirichlet, name=lat, nSet=allnodes,    field=displacement, 1=0.0, 3=0.0
"""

SETUP_TEMPLATE_3D = """import sys
sys.path.insert(0, r'{patchdir}')
import make_contact_elements as mce
from edelweissfe.sets.nodeset import NodeSet


def setup(model):
    mce.apply(model, 'genA_top', 'con_slave', '{con_type}', (0.0, 1.0, 0.0))
    mce.apply(model, 'genB_bottom', 'con_master', '{con_type}', (0.0, -1.0, 0.0))
    model.nodeSets['allnodes'] = NodeSet('allnodes', list(model.nodes.values()))
"""


# ===========================================================================


def _weighted_gaps(model, dim):
    """Rekonstruiert g_weak und g_sep je Slave-Knoten aus der konvergierten Loesung.

    Dieselben Formeln wie in ``applyConstraint`` und in 10_signorini_check: mit der
    im letzten Increment eingefrorenen Geometrie und den konvergierten Koordinaten.
    """
    mc = model.constraints["contact"]
    nSlave = mc.nNonMortarNodes
    nf = model.nodeFields[mc.field]
    u_of_node = {node: u for node, u in zip(nf.nodes, nf["U"])}
    coords = mc._X + np.array([u_of_node[node][:dim] for node in mc.nodes])
    x_s, x_m = coords[:nSlave], coords[nSlave:]

    D_dense = mc.current_D.toarray()  # sparse -> dicht, nur fuer diesen Test
    C_dense = mc.current_C.toarray()
    g_weak = np.array([
        -D_dense[I] @ (x_s @ mc.current_normals[I])
        + C_dense[I] @ (x_m @ mc.current_normals[I])
        for I in range(nSlave)
    ])
    rowsum = mc.current_D_rowsum
    g_sep = np.where(np.abs(rowsum) > 1e-30, g_weak / np.where(rowsum == 0.0, 1.0, rowsum), 0.0)
    return g_weak, g_sep, mc.active_set.copy()


def _run(name, inp_text, setup_text, k, nx_a, dim):
    """Schreibt Setup + Input, rechnet und wertet gegen die exakte Loesung aus."""
    setup_path = os.path.join(TESTDIR, f"generated_setup_{name}.py")
    with open(setup_path, "w") as f:
        f.write(setup_text)

    if TESTDIR not in sys.path:
        sys.path.insert(0, TESTDIR)

    inp_path = os.path.join(TESTDIR, f"generated_{name}.inp")
    with open(inp_path, "w") as f:
        f.write(inp_text)

    model, _ = finiteElementSimulation(parseInputFile(inp_path), verbose=False, suppressPlots=True)

    length = BASE_LENGTH * k
    height = BASE_HEIGHT * k
    utop = -2.0 * height * PRESSURE / E_MOD

    nf = model.nodeFields["displacement"]
    max_err_lat = 0.0
    max_err_uy = 0.0
    for node, u in zip(nf.nodes, nf["U"]):
        y = node.coordinates[1]
        max_err_lat = max(max_err_lat, abs(u[0]))
        if dim == 3:
            max_err_lat = max(max_err_lat, abs(u[2]))
        max_err_uy = max(max_err_uy, abs(u[1] - (-PRESSURE * y / E_MOD)))

    mc = model.constraints["contact"]
    lambdas = np.array([sv.value for sv in mc.scalarVariables])
    rowsum = mc.current_D_rowsum
    lam_mean = np.sum(lambdas * rowsum) / np.sum(rowsum)
    g_weak, g_sep, active = _weighted_gaps(model, dim)

    os.remove(inp_path)
    os.remove(setup_path)

    return {
        "u_ref": abs(utop),
        "err_ux": max_err_lat,
        "err_uy": max_err_uy,
        "lambdas": lambdas,
        "lam_mean": lam_mean,
        "facet_size": length / nx_a,
        "extent": max(length, 2.0 * height),
        "g_weak_active": float(np.max(np.abs(g_weak[active]))) if active.any() else 0.0,
        "g_sep_active": float(np.max(np.abs(g_sep[active]))) if active.any() else 0.0,
    }


def _with_scalar_tol(inp_text, tol):
    """Stellt dem *job-Block eine angepasste Skalar-Toleranz voran (falls noetig)."""
    if tol <= DEFAULT_SCALAR_FLUX_TOL:
        return inp_text
    return inp_text.replace("*job,", UPDATE_CONFIG.format(tol=repr(tol)) + "*job,", 1)


def run_scaled_2d(name, el_type, con_type, nx_a, nx_b, k, scalar_tol=None):
    length = BASE_LENGTH * k
    height = BASE_HEIGHT * k
    utop = -2.0 * height * PRESSURE / E_MOD
    setup_text = SETUP_TEMPLATE_2D.format(
        patchdir=PATCH2D_DIR, el_type=el_type, con_type=con_type,
        nx_a=nx_a, nx_b=nx_b, length=repr(length), height=repr(height),
    )
    inp_text = INP_TEMPLATE_2D.format(name=name, E=E_MOD, utop=repr(utop))
    tol = scalar_flux_tolerance(k, 2) if scalar_tol is None else scalar_tol
    return _run(name, _with_scalar_tol(inp_text, tol), setup_text, k, nx_a, dim=2)


def run_scaled_3d(name, el_type, con_type, nx_a, nx_b, k, scalar_tol=None):
    length = BASE_LENGTH * k
    height = BASE_HEIGHT * k
    utop = -2.0 * height * PRESSURE / E_MOD
    setup_text = SETUP_TEMPLATE_3D.format(patchdir=PATCH3D_DIR, con_type=con_type)
    inp_text = INP_TEMPLATE_3D.format(
        name=name, el_type=el_type, nx_a=nx_a, nx_b=nx_b,
        length=repr(length), height=repr(height), E=E_MOD, utop=repr(utop),
    )
    tol = scalar_flux_tolerance(k, 3) if scalar_tol is None else scalar_tol
    return _run(name, _with_scalar_tol(inp_text, tol), setup_text, k, nx_a, dim=3)


def report(k, res, dim):
    """Bewertet einen Lauf und meldet nebenbei die Entartung der Suchmarge."""
    tol = RTOL * res["u_ref"]
    lam_err = np.max(np.abs(res["lambdas"] - PRESSURE)) / PRESSURE
    mean_err = abs(res["lam_mean"] - PRESSURE) / PRESSURE

    lat = "max|u_x|            " if dim == 2 else "max|u_x|, max|u_z|  "
    print(f"\n* {dim}D, Skalierungsfaktor k = {k:g}  (Modellausdehnung {res['extent']:g})")
    print(f"  {lat} = {res['err_ux']:.3e}   (Toleranz {tol:.1e})")
    print(f"  max|u_y - u_y_exakt| = {res['err_uy']:.3e}   (Toleranz {tol:.1e})")
    print(f"  Kontaktdruck: max. rel. Abweichung von p = {lam_err:.3e}, "
          f"gewichtetes Mittel {res['lam_mean']:+.9f} (rel. Fehler {mean_err:.3e})")

    # Diagnose der Suchmarge (Formel aus mortarcontact.py, hier nur ausgewertet)
    margin = max(MARGIN_REL * res["extent"], MARGIN_ABS)
    ratio = margin / res["facet_size"]
    dominating = "absolut (0.1)" if MARGIN_ABS > MARGIN_REL * res["extent"] else "relativ (0.15*L)"
    print(f"  Suchmarge = {margin:g} = {ratio:.1f} x Facettengroesse -- dominierender Term: {dominating}")
    if ratio > 10.0:
        print("  [INFO] Die Marge uebersteigt die Facettengroesse um mehr als das Zehnfache:")
        print("         die BVH-Anfrage liefert praktisch ALLE Masterfacetten zurueck, die")
        print("         Nachbarsuche entartet zur vollstaendigen Paarung (O(n^2)). Das ist ein")
        print("         LAUFZEIT-, kein Korrektheitsproblem -- die Ergebnisse oben belegen das.")

    ok = True
    if res["err_ux"] > tol or res["err_uy"] > tol:
        print("  [FAIL] Verschiebungsfeld skaliert nicht wie die exakte Loesung!")
        ok = False
    if lam_err > RTOL or mean_err > RTOL:
        print("  [FAIL] Kontaktdruck ist nicht skaleninvariant!")
        ok = False
    if ok:
        print(f"  [PASS] k = {k:g}: Spannungen und Kontaktdruck unveraendert, "
              f"Verschiebungen um k skaliert.")
    return ok


VARIANTS_2D = [
    ("cpe4", "CPE4", "CONLINE2", 2, 3),
    ("cpe8", "CPE8", "CONLINE3", 2, 3),
]

# 3D: nicht-passendes Interface, damit tatsaechlich geclippt wird (bei passenden
# Netzen faellt die Ueberlappung mit der Facette zusammen und die Toleranzen der
# Schnittbildung werden nie erreicht).
VARIANTS_3D = [
    ("hex8", "C3D8", "CONQUAD4", 2, 3),
    ("hex20", "C3D20", "CONQUAD8", 2, 3),
]

SCALES = [1e-3, 1.0, 1e3]


def _sweep(tag, runner, el_type, con_type, nx_a, nx_b, dim, results, pressures):
    print(f"\n--- {dim}D-Variante {el_type}/{con_type} (nicht passend, {nx_a} gegen {nx_b}) ---")
    for k in SCALES:
        name = f"{tag}_k{SCALES.index(k)}"
        res = runner(name, el_type, con_type, nx_a, nx_b, k)
        results.append(report(k, res, dim))
        pressures[(tag, k)] = np.sort(res["lambdas"])

    # Direkter Vergleich der Skalen untereinander, nicht nur gegen die
    # analytische Loesung: die Multiplikatoren muessen knotenweise gleich sein.
    ref = pressures[(tag, 1.0)]
    for k in SCALES:
        diff = np.max(np.abs(pressures[(tag, k)] - ref)) / PRESSURE
        print(f"  Multiplikatoren k = {k:g} gegen k = 1: max. rel. Differenz {diff:.3e}")
        if diff > RTOL:
            print("  [FAIL] Multiplikatoren unterscheiden sich zwischen den Skalen!")
            results.append(False)


def test_lambda_residual_floor_is_scale_dependent():
    with default_tolerances_restored():
        """Haelt fest, WO die absolute Skalar-Schranke des Loesers unerreichbar wird.

        Die Aussage ist nicht ``der Kontakt rechnet falsch``, sondern: das Residuum der
        lambda-Zeile ist der GEWICHTETE Spalt und traegt deshalb die Einheit
        Laenge x Flaeche, waehrend die Schranke absolut ist. Gemessen wird beides an
        derselben, nachweislich exakten Loesung:

          * g_sep (die physikalische Knotenoeffnung) ist an aktiven Knoten null bis auf
            die Koordinaten-Rundung -- die Kontaktbedingung ist also erfuellt;
          * g_weak = D_II * g_sep ist es dem Betrage nach NICHT, weil D_II mit der
            Flaeche waechst.

        Ab einer Modellausdehnung, bei der max|g_weak| die Standardschranke 1e-8
        ueberschreitet, kann der Loeser das Increment nicht mehr als konvergiert
        annehmen, obwohl die Loesung stimmt. Der Test belegt, dass genau das bei
        k = 1e3 in 3D eintritt und bei k = 1 noch nicht.
        """
        print("\n=== Boden des lambda-Residuums ueber der Laengenskala (3D) ===")
        rows = []
        for k in (1.0, 1e3):
            res = run_scaled_3d(f"floor_k{k:g}", "C3D8", "CONQUAD4", 2, 3, k)
            rows.append((k, res))
            print(f"\n* k = {k:g}")
            print(f"  max|u_y - u_y_exakt|                = {res['err_uy']:.3e}")
            print(f"  max|lambda| - p                     = "
                  f"{np.max(np.abs(res['lambdas'] - PRESSURE)):.3e}")
            print(f"  max|g_sep|  an aktiven Knoten       = {res['g_sep_active']:.3e}   (Laenge)")
            print(f"  max|g_weak| an aktiven Knoten       = {res['g_weak_active']:.3e}   "
                  f"(Laenge x Flaeche)  <-- das prueft der Loeser")
            print(f"  Standardschranke (phenomena.py)     = {DEFAULT_SCALAR_FLUX_TOL:.1e}")
            print(f"  mitgefuehrte Schranke dieses Tests  = {scalar_flux_tolerance(k, 3):.3e}")

        (k1, r1), (k3, r3) = rows

        # Beide Rechnungen sind exakt - die Loesung haengt nicht an der Schranke.
        for k, r in rows:
            assert r["err_uy"] < 1e-8 * r["u_ref"], f"k = {k:g}: Verschiebungsfeld nicht exakt"
            assert np.max(np.abs(r["lambdas"] - PRESSURE)) / PRESSURE < RTOL, \
                f"k = {k:g}: Kontaktdruck nicht exakt"
            assert r["g_sep_active"] < 1e-9 * r["u_ref"], \
                f"k = {k:g}: die physikalische Knotenoeffnung ist an aktiven Knoten nicht null"

        # Und trotzdem liegt g_weak bei k = 1e3 ueber der Standardschranke, bei k = 1
        # weit darunter. Das ist die Grenze, um die es geht.
        assert r1["g_weak_active"] < DEFAULT_SCALAR_FLUX_TOL, (
            "Bei k = 1 sollte die Standardschranke erreichbar sein "
            f"(max|g_weak| = {r1['g_weak_active']:.3e})"
        )
        assert r3["g_weak_active"] > DEFAULT_SCALAR_FLUX_TOL, (
            "Bei k = 1e3 sollte die Standardschranke unerreichbar sein; ist sie es nicht mehr, "
            "wurde die Skalierung des lambda-Residuums geaendert und dieser Test ist "
            f"anzupassen (max|g_weak| = {r3['g_weak_active']:.3e})"
        )
        print(f"\n  [PASS] Die Grenze liegt zwischen k = 1 (max|g_weak| = {r1['g_weak_active']:.2e}) "
              f"und k = 1e3 ({r3['g_weak_active']:.2e}).")
        print("         Beide Loesungen sind exakt; unerreichbar ist die SCHRANKE, nicht die Loesung.")
        print("         Abhilfe im Anwendungsfall: *updateConfiguration, "
              "configuration=fluxResidualTolerance / scalar variables=<passend>.")


def test_scale_invariance():
    with default_tolerances_restored():
        print("\n=== Skaleninvarianz des Zwei-Block-Patch-Tests ===")
        results = []
        pressures = {}

        for tag, el_type, con_type, nx_a, nx_b in VARIANTS_2D:
            _sweep(tag, run_scaled_2d, el_type, con_type, nx_a, nx_b, 2, results, pressures)

        for tag, el_type, con_type, nx_a, nx_b in VARIANTS_3D:
            _sweep(tag, run_scaled_3d, el_type, con_type, nx_a, nx_b, 3, results, pressures)

        n_fail = results.count(False)
        assert n_fail == 0, f"{n_fail} Skalierungspruefungen fehlgeschlagen"


if __name__ == "__main__":
    print("=" * 60)
    print("SKALENINVARIANZ DES MORTAR-KONTAKTS (2D UND 3D)")
    print("=" * 60)
    test_scale_invariance()
    test_lambda_residual_floor_is_scale_dependent()
    print("\n[SUCCESS] Die Loesung ist skaleninvariant.")
