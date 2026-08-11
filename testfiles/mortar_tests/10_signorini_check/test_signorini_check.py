#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test 10: Signorini-Nachprüfung der konvergierten Lösung
=======================================================

Alle übrigen Tests dieser Reihe prüfen Bausteine (Normalen, Clipping, Koppel-
matrizen, Biorthogonalität, Tangente) oder das Verschiebungsfeld und die
Multiplikatoren einer Patch-Lösung. **Keiner** prüft, ob die konvergierte Lösung
die Hertz--Signorini--Moreau-Bedingungen tatsächlich erfüllt:

    p_n >= 0,        g_sep >= 0,        p_n * g_sep = 0.

Genau das tut dieser Test. Er ist die einzige Kontrolle, die ein zu früh
eingefrorenes Active Set aufdecken kann: die Newton-Iteration konvergiert auch
dann sauber, wenn der eingefrorene Satz die falschen Knoten enthält -- sie löst
dann nur eben ein anderes Problem. Sichtbar wird das ausschließlich an einem
aktiven Knoten unter Zug (p_n < 0) oder einem inaktiven Knoten mit Durchdringung
(g_sep < 0).

Rekonstruktion
--------------
``check_signorini`` baut p_n und g_sep je Slave-Knoten aus **denselben** Größen
und mit **denselben** Formeln auf wie ``Constraint.applyConstraint``
(``mortarcontact.py``, Abschnitt "Sign-consistent contact measures"):

    g_weak_I = -sum_K D_IK (x_K . n_I) + sum_J C_IJ (x_J . n_I)
    p_n      =  lambda_I * sgn(D_II)
    g_sep    =  g_weak_I / D_II

mit den im letzten Increment eingefrorenen ``current_normals``, ``current_D``,
``current_C``, ``current_D_rowsum`` (staggered geometry update) und den
konvergierten Knotenkoordinaten. Damit ist die Prüfung exakt die Auswertung der
Bedingungen, die der Löser durchgesetzt hat -- keine unabhängige Nachrechnung mit
anderer Geometrie, die eine Abweichung erzeugen würde, die es gar nicht gibt.

Die Vorzeichenkonvention ist die des Constraints und trägt auch negative
Knotengewichte D_II (CONQUAD9 bei Teilüberdeckung, vgl. 06_active_set_pdass).

Knotenklassen
-------------
uncovered  D_II ~ 0: kein Master gegenüber. p_n und g_sep sind nicht definiert
           (der Constraint setzt beide auf 0), die Bedingungen sind trivial
           erfüllt. Diese Knoten MÜSSEN lambda = 0 haben.
active     Der Constraint erzwingt g_weak = 0. Zu prüfen ist p_n >= 0.
inactive   Der Constraint erzwingt lambda = 0. Zu prüfen ist g_sep >= 0.

Lastfälle
---------
1. ``teilkontakt``  Block B halb so breit wie A: die Slave-Knoten rechts davon
                    haben kein Gegenüber, die übrigen sind geschlossen.
2. ``abheben``      Zug auf die Oberseite: alle Knoten müssen loslassen.
3. ``schief``       Verkippte Oberseitenverschiebung (analyticalField): links
                    geschlossen, rechts offen -- im selben Increment, bei voller
                    Überdeckung. Der eigentliche Active-Set-Lastfall.
4. ``voll``         Der Patch-Testfall aus 07: alle Knoten aktiv. Gegenprobe.

Jeder Lastfall läuft mit CPE4/CONLINE2 und CPE8/CONLINE3.

Abnahme: alle Fälle erfüllen die drei Bedingungen, und über die Testreihe
hinweg müssen sowohl echt offene (lambda = 0 bei vorhandenem Gegenüber) als
auch geschlossene Knoten (g_sep = 0) vorkommen -- sonst prüft der Test das
Active Set nicht wirklich, sondern nur einen Sonderfall.

Netz und Kontaktelemente stammen aus 07_patch_test_2d (two_block_mesh_2d.py,
make_contact_elements_2d.py); dort ist auch begründet, warum das Zwei-Block-Netz
nicht mit planeRectQuad gebaut wird.
"""

import os
import sys

import numpy as np

TESTDIR = os.path.abspath(os.path.dirname(__file__))
MORTARDIR = os.path.abspath(os.path.join(TESTDIR, ".."))
PATCH2D_DIR = os.path.join(MORTARDIR, "07_patch_test_2d")
sys.path.insert(0, os.path.abspath(os.path.join(MORTARDIR, "../..")))

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.utils.inputfileparser import parseInputFile

E_MOD = 1000.0
BLOCK_HEIGHT = 1.0
LENGTH = 1.0
U_TOP = 0.02  # Betrag der aufgebrachten Oberseitenverschiebung

# Relative Toleranzen. Die Bedingungen p_n >= 0 und g_sep >= 0 sind reine
# Vorzeichenaussagen; toleriert wird nur das Löserrauschen, bezogen auf die
# jeweilige Referenzgröße (Druck bzw. Länge) -- absolute Schranken wären
# einheitenabhängig.
RTOL_P = 1e-7
RTOL_G = 1e-7

# Schranke fuer die Pruefung an NEU gerechneter Geometrie. Sie ist notwendig
# groesser als RTOL_G: dort wird nicht mehr geprueft, ob der Loeser seine eigenen
# Gleichungen erfuellt (das tut er bis auf Loeserrauschen), sondern was davon im
# TATSAECHLICHEN Zustand uebrig bleibt. Die Differenz ist der Staffelungsfehler
# der eingefrorenen Geometrie.
#
# Der Wert ist deshalb KEINE Eigenschaft der Formulierung, sondern eine Schranke
# fuer die hier verwendete Schrittweite (maxInc = 0.1, also rund zehn Increments).
# Gemessen wird ueber die Lastfaelle hinweg maximal etwa 6.0e-5 der aufgebrachten
# Verschiebung (2D-Zweig, teilkontakt_cpe8), im 3D-Zweig rund drei
# Groessenordnungen weniger; die Schranke laesst damit etwa den Faktor acht Luft.
# Dass diese Groesse mit der Schrittweite faellt (Ordnung ~1), misst
# 12_increment_size.
RTOL_G_FRESH = 5e-4


# ===========================================================================
# Die Nachprüfung
# ===========================================================================


def check_signorini(model, constraint_name="contact"):
    """Rekonstruiert p_n und g_sep je Slave-Knoten aus dem konvergierten Modell.

    Rückgabe: dict mit den Arrays ``p_n``, ``g_sep``, ``lam``, ``D_II``,
    ``active_set`` sowie den Klassifikationsmasken ``uncovered``, ``active``,
    ``inactive`` und dem Zustand der Active-Set-Iteration des letzten
    Increments (``frozen``, ``iterations``).
    """
    mc = model.constraints[constraint_name]
    dim = model.domainSize
    nSlave = mc.nNonMortarNodes

    # Konvergierte Knotenkoordinaten in derselben Reihenfolge wie mc._X
    # (Slaves zuerst, dann Master).
    nf = model.nodeFields[mc.field]
    u_of_node = {node: u for node, u in zip(nf.nodes, nf["U"])}
    disp = np.array([u_of_node[node][:dim] for node in mc.nodes])
    coords = mc._X + disp
    x_slave = coords[:nSlave]
    x_master = coords[nSlave:]

    # Im letzten Increment eingefrorene Geometrie -- exakt die Größen, mit denen
    # der Löser die Bedingungen aufgestellt hat.
    D = mc.current_D.toarray()  # sparse -> dicht, nur fuer diesen Test
    C = mc.current_C.toarray()
    normals = mc.current_normals
    rowsum = mc.current_D_rowsum

    lam = np.array([sv.value for sv in mc.scalarVariables])

    p_n = np.zeros(nSlave)
    g_sep = np.zeros(nSlave)
    for I in range(nSlave):
        n_I = normals[I]
        g_weak = -D[I] @ (x_slave @ n_I) + C[I] @ (x_master @ n_I)
        D_II = rowsum[I]
        p_n[I] = lam[I] * np.sign(D_II)
        g_sep[I] = g_weak / D_II if abs(D_II) > 1e-30 else 0.0

    # "Kein Gegenüber": die Zeilensumme von D ist das Integral der dualen
    # Formfunktion über den ÜBERDECKTEN Teil der Facette; ohne Überdeckung ist
    # sie exakt null. Als Schwelle dient ein Bruchteil des größten Gewichts,
    # damit die Aussage längenskaleninvariant bleibt.
    uncovered = np.abs(rowsum) < 1e-10 * np.max(np.abs(rowsum))
    active = mc.active_set & ~uncovered
    inactive = ~mc.active_set & ~uncovered

    return {
        "p_n": p_n,
        "g_sep": g_sep,
        "lam": lam,
        "D_II": rowsum,
        "active_set": mc.active_set.copy(),
        "uncovered": uncovered,
        "active": active,
        "inactive": inactive,
        "frozen": mc.active_set_frozen,
        "iterations": mc.current_iteration,
    }


def check_signorini_fresh_geometry(model, constraint_name="contact"):
    """Prueft den PHYSIKALISCHEN Spalt an neu gerechneter Geometrie.

    ``check_signorini`` oben rekonstruiert die Bedingungen mit der im letzten
    Increment EINGEFRORENEN Geometrie -- also mit genau den Groessen, mit denen der
    Loeser sie aufgestellt hat. Das ist die richtige Pruefung dafuer, ob der Loeser
    seine eigenen Gleichungen erfuellt, und genau deshalb kann sie den
    Staffelungsfehler prinzipiell nicht sehen: eine Durchdringung, die erst dadurch
    entsteht, dass D, M und n_I vom Increment-Beginn stammen, ist in diesen
    Groessen unsichtbar.

    Diese Funktion schliesst die Luecke: sie wertet Normalen und Koppelmatrizen an
    der KONVERGIERTEN Konfiguration neu aus und misst den Spalt damit. Was
    herauskommt, ist der tatsaechlich verbliebene Kontaktzustand -- die Groesse,
    die den Staffelungsfehler traegt (Test 12 misst seine Ordnung).

    Rueckgabe: max. Durchdringung (positiv = es dringt ein) an aktiven Knoten und
    an inaktiven Knoten mit Gegenueber.
    """
    mc = model.constraints[constraint_name]
    dim = model.domainSize
    sf = mc.sizeField
    nSlave = mc.nNonMortarNodes

    nf = model.nodeFields[mc.field]
    u_of_node = {node: u for node, u in zip(nf.nodes, nf["U"])}

    # Zustandsvektor in der Constraint-eigenen Anordnung
    U = np.zeros(mc.nDof)
    for node, idx in mc.node_to_global_idx.items():
        U[sf * idx : sf * idx + dim] = np.asarray(u_of_node[node])[:dim]

    normals = mc.compute_normals(U)
    D, C = mc.compute_mortar_coupling_matrices(U)
    D, C = D.toarray(), C.toarray()  # sparse -> dicht, nur fuer diesen Test
    rowsum = np.sum(D, axis=1)

    coords = mc._X + U[: sf * len(mc.nodes)].reshape(len(mc.nodes), sf)[:, :dim]
    x_slave, x_master = coords[:nSlave], coords[nSlave:]

    g_sep = np.zeros(nSlave)
    for I in range(nSlave):
        n_I = normals[I]
        g_weak = -D[I] @ (x_slave @ n_I) + C[I] @ (x_master @ n_I)
        g_sep[I] = g_weak / rowsum[I] if abs(rowsum[I]) > 1e-30 else 0.0

    covered = np.abs(rowsum) > 1e-10 * np.max(np.abs(rowsum))
    active = mc.active_set & covered
    inactive = ~mc.active_set & covered

    pen_active = float(-g_sep[active].min()) if active.any() else 0.0
    pen_inactive = float(-g_sep[inactive].min()) if inactive.any() else 0.0
    return pen_active, pen_inactive


def report_and_assert(name, res, p_ref, g_ref):
    """Gibt die Auswertung aus und prüft die drei Bedingungen."""
    eps_p = RTOL_P * p_ref
    eps_g = RTOL_G * g_ref

    p_n, g_sep, lam = res["p_n"], res["g_sep"], res["lam"]
    act, inact, unc = res["active"], res["inactive"], res["uncovered"]

    print(
        f"  Knoten: {len(p_n)} gesamt, {act.sum()} aktiv, {inact.sum()} offen, "
        f"{unc.sum()} ohne Gegenueber"
    )
    # 'iterations' zaehlt die WIEDERHOLTEN Assemblierungen des letzten Increments;
    # 0 heisst, dass die extrapolierte Startloesung bereits konvergiert war (bei
    # linearer Elastizitaet mit gleich bleibendem Active Set der Normalfall).
    print(
        f"  Active Set im letzten Increment: {'EINGEFROREN' if res['frozen'] else 'frei'}"
        f" nach {res['iterations']} Wiederholungen"
        + ("  <-- Iterationsgrenze erreicht!" if res["iterations"] >= 20 else "")
    )

    ok = True

    # (1) p_n >= 0 -- ein aktiver Knoten darf nicht ziehen.
    if act.any():
        p_min = p_n[act].min()
        print(f"  min p_n  (aktiv)   = {p_min:+.6e}   (Schranke {-eps_p:.1e}, p_ref = {p_ref:g})")
        if p_min < -eps_p:
            print("  [FAIL] Aktiver Knoten unter ZUG -- Signorini p_n >= 0 verletzt!")
            ok = False

    # (2) g_sep >= 0 -- ein inaktiver Knoten darf nicht durchdringen.
    if inact.any():
        g_min = g_sep[inact].min()
        print(f"  min g_sep (offen)  = {g_min:+.6e}   (Schranke {-eps_g:.1e}, g_ref = {g_ref:g})")
        if g_min < -eps_g:
            print("  [FAIL] Inaktiver Knoten DURCHDRINGT -- Signorini g_sep >= 0 verletzt!")
            ok = False

    # (3) Komplementaritaet. Der Constraint erzwingt je Zweig genau eine der
    # beiden Gleichungen; geprueft wird, dass die jeweils andere Groesse
    # tatsaechlich verschwindet -- und damit das Produkt.
    if act.any():
        g_act = np.abs(g_sep[act]).max()
        print(f"  max |g_sep| (aktiv) = {g_act:.6e}   (Schranke {eps_g:.1e})")
        if g_act > eps_g:
            print("  [FAIL] Aktiver Knoten mit Restspalt -- g_weak = 0 nicht erfuellt!")
            ok = False
    if inact.any():
        p_inact = np.abs(p_n[inact]).max()
        print(f"  max |p_n| (offen)   = {p_inact:.6e}   (Schranke {eps_p:.1e})")
        if p_inact > eps_p:
            print("  [FAIL] Inaktiver Knoten mit Druck -- lambda = 0 nicht erfuellt!")
            ok = False
    if unc.any():
        lam_unc = np.abs(lam[unc]).max()
        print(f"  max |lambda| (ohne Gegenueber) = {lam_unc:.6e}   (Schranke {eps_p:.1e})")
        if lam_unc > eps_p:
            print("  [FAIL] Knoten ohne Gegenueber traegt einen Multiplikator!")
            ok = False

    # Das Produkt ist eine ABGELEITETE Groesse: seine erreichbare Genauigkeit ist
    # die durch eps_p und eps_g induzierte Stoerung, also
    # eps_p * |g| + eps_g * |p| -- NICHT eps_p * eps_g. Als Skalen dienen die
    # tatsaechlich auftretenden Maxima (mindestens die nominellen Referenzen),
    # damit eine Druckspitze an einer Splitterueberdeckung die Schranke
    # mitzieht, statt sie zu sprengen.
    p_scale = max(np.abs(p_n).max(), p_ref)
    g_scale = max(np.abs(g_sep).max(), g_ref)
    eps_prod = eps_p * g_scale + eps_g * p_scale
    complementarity = np.abs(p_n * g_sep).max()
    print(f"  max |p_n * g_sep|   = {complementarity:.6e}   (Schranke {eps_prod:.1e})")
    if complementarity > eps_prod:
        print("  [FAIL] Komplementaritaet p_n * g_sep = 0 verletzt!")
        ok = False

    # Diagnose (kein Kriterium): ein aktiver Knoten, dessen Gewicht D_II nur ein
    # winziger Bruchteil des groessten ist, wird ueber eine fast verschwindende
    # Flaeche gezwungen. Sein Multiplikator skaliert dann mit 1/D_II, waehrend die
    # uebertragene Knotenkraft lambda*D_II endlich bleibt -- nachgemessen an der
    # Kante der Teilueberdeckung: D_II 2.1e-2 / 5.2e-3 / 8.3e-6 ergibt
    # lambda 58 / 173 / 8.2e4, die Knotenkraft dagegen durchweg O(1).
    # Das ist kein Fehler, sondern die Eigenschaft eines dualen Multiplikators
    # ueber schrumpfendem Traeger; es heisst aber, dass lambda an teilweise
    # ueberdeckten Knoten NICHT als Kontaktdruck gelesen werden darf.
    # Sichtbar machen, nicht bewerten.
    D_II = res["D_II"]
    D_max = np.abs(D_II).max()
    slivers = act & (np.abs(D_II) < 1e-3 * D_max)
    for I in np.flatnonzero(slivers):
        print(
            f"  [INFO] Splitterueberdeckung an Slave-Knoten {I}: "
            f"D_II/max = {abs(D_II[I]) / D_max:.2e}, p_n = {p_n[I]:.4e}, "
            f"Knotenkraft -lambda*D_II = {-lam[I] * D_II[I]:+.4e}"
        )

    if ok:
        print(f"  [PASS] '{name}': Signorini-Bedingungen erfuellt.")
    return ok


# ===========================================================================
# Modellaufbau
# ===========================================================================

INP_TEMPLATE = """*modelGenerator, generator=executePythonCode, name=meshgen
import generated_setup_{name} as vs
vs.setup(model)

*material, name=LinearElastic, id=mat
{E}, 0.0

*section, name=secA, thickness=1.0, material=mat, type=plane
solids_a
*section, name=secB, thickness=1.0, material=mat, type=plane
solids_b

** cn: Komplementaritaetsparameter c_n der Normalkontakt-NCP, rein algorithmisch
** und ~ O(E) des weicheren Koerpers zu waehlen (Farah 2018, Abschn. 3.5.2).
*constraint, type=mortarcontact, name=contact
nonMortarSurface=con_slave
mortarSurface=con_master
cn={E}
{fields}
*job, name=signorinijob, domain=2d
*solver, name=theSolver, solver=NISTParallel

*step, solver=theSolver
maxInc=0.1, minInc=1e-3, maxNumInc=100, maxIter=25, stepLength=1
>>dirichlet, name=bot,  nSet=fixed_bottom, field=displacement, 2=0.0
>>dirichlet, name=top,  nSet=load_top,     field=displacement, 2={utop}{topfield}
>>dirichlet, name=symm, nSet=symm_x,       field=displacement, 1=0.0
"""

# Verkippung der Oberseitenverschiebung: Faktor 1 - 2x/L, also volle Stauchung
# bei x = 0 und derselbe Betrag als Zug bei x = L. Der Nulldurchgang liegt in der
# Mitte, sodass die Slave-Flaeche im selben Increment halb geschlossen und halb
# offen ist.
TILT_FIELD = """
*analyticalField, name=tilt, type=scalarExpression
"f(x,y,z)"="1.0 - 2.0*x/{length}"
"""

SETUP_TEMPLATE = """import sys
sys.path.insert(0, r'{patchdir}')
import two_block_mesh_2d as tb
import make_contact_elements_2d as mce


def setup(model):
    tb.build(model, '{el_type}', {nx_a}, {nx_b}, height={height}, length={length},
             length_b={length_b})
    mce.apply(model, 'surf_a_top', 'con_slave', '{con_type}', (0.0, 1.0))
    mce.apply(model, 'surf_b_bottom', 'con_master', '{con_type}', (0.0, -1.0))
"""


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
lX={length_b}
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

*constraint, type=mortarcontact, name=contact
nonMortarSurface=con_slave
mortarSurface=con_master
cn={E}
{fields}
*job, name=signorinijob3d, domain=3d
*solver, name=theSolver, solver=NISTParallel

*step, solver=theSolver
maxInc=0.1, minInc=1e-3, maxNumInc=100, maxIter=25, stepLength=1
>>dirichlet, name=bot, nSet=genA_bottom, field=displacement, 2=0.0
>>dirichlet, name=top, nSet=genB_top,    field=displacement, 2={utop}{topfield}
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

PATCH3D_DIR = os.path.join(MORTARDIR, "08_patch_test_hex20")


def run_case_3d(case, el_type, con_type, nx_a, nx_b, length_b, utop, tilt):
    """Derselbe Lastfallkatalog in 3D, mit Flaechen- statt Linienelementen.

    Damit laufen die Signorini-Bedingungen erstmals ueber den 3D-Pfad --
    Hilfsebenen-Projektion, Sutherland--Hodgman und Dreiecksquadratur -- statt
    ueber den 1D-Intervallschnitt des 2D-Pfads.
    """
    name = f"{case}_{el_type.lower()}_3d"
    print(f"\n* Lastfall '{case}' mit {el_type}/{con_type} (3D)")

    setup_path = os.path.join(TESTDIR, f"generated_setup_{name}.py")
    with open(setup_path, "w") as f:
        f.write(SETUP_TEMPLATE_3D.format(patchdir=PATCH3D_DIR, con_type=con_type))

    if TESTDIR not in sys.path:
        sys.path.insert(0, TESTDIR)

    inp_path = os.path.join(TESTDIR, f"generated_{name}.inp")
    with open(inp_path, "w") as f:
        f.write(INP_TEMPLATE_3D.format(
            name=name, el_type=el_type, nx_a=nx_a, nx_b=nx_b,
            length=repr(LENGTH), length_b=repr(length_b), height=repr(BLOCK_HEIGHT),
            E=E_MOD, utop=repr(utop),
            fields=TILT_FIELD.format(length=LENGTH) if tilt else "",
            topfield=", analyticalField=tilt" if tilt else "",
        ))

    model, _ = finiteElementSimulation(parseInputFile(inp_path), verbose=False, suppressPlots=True)

    res = check_signorini(model)
    p_ref = E_MOD * abs(utop) / (2.0 * BLOCK_HEIGHT)
    ok = report_and_assert(name, res, p_ref, abs(utop))
    ok = report_fresh_geometry(name, model, abs(utop)) and ok

    if ok:
        os.remove(inp_path)
        os.remove(setup_path)
    return ok, res


def report_fresh_geometry(name, model, u_ref):
    """Misst und bewertet die Durchdringung an NEU gerechneter Geometrie."""
    pen_active, pen_inactive = check_signorini_fresh_geometry(model)
    print(f"  Neu gerechnete Geometrie: max. Durchdringung aktiv {pen_active:+.3e}, "
          f"offen {pen_inactive:+.3e}   (Schranke {RTOL_G_FRESH * u_ref:.1e})")

    ok = True
    if pen_active > RTOL_G_FRESH * u_ref:
        print("  [FAIL] Aktiver Knoten dringt im tatsaechlichen Zustand ein -- der "
              "Staffelungsfehler ist groesser als zugelassen!")
        ok = False
    if pen_inactive > RTOL_G_FRESH * u_ref:
        print("  [FAIL] Inaktiver Knoten dringt im tatsaechlichen Zustand ein!")
        ok = False
    return ok


def run_case(case, el_type, con_type, nx_a, nx_b, length_b, utop, tilt):
    name = f"{case}_{el_type.lower()}"
    print(f"\n* Lastfall '{case}' mit {el_type}/{con_type}")

    setup_path = os.path.join(TESTDIR, f"generated_setup_{name}.py")
    with open(setup_path, "w") as f:
        f.write(
            SETUP_TEMPLATE.format(
                patchdir=PATCH2D_DIR, el_type=el_type, con_type=con_type,
                nx_a=nx_a, nx_b=nx_b, height=BLOCK_HEIGHT, length=LENGTH,
                length_b=length_b,
            )
        )

    if TESTDIR not in sys.path:
        sys.path.insert(0, TESTDIR)

    inp_path = os.path.join(TESTDIR, f"generated_{name}.inp")
    with open(inp_path, "w") as f:
        f.write(
            INP_TEMPLATE.format(
                name=name, E=E_MOD, utop=utop,
                fields=TILT_FIELD.format(length=LENGTH) if tilt else "",
                topfield=", analyticalField=tilt" if tilt else "",
            )
        )

    model, _ = finiteElementSimulation(parseInputFile(inp_path), verbose=False, suppressPlots=True)

    res = check_signorini(model)

    # Referenzgroessen: der Druck, der bei voller Ueberdeckung entstuende, und
    # die aufgebrachte Verschiebung.
    p_ref = E_MOD * abs(utop) / (2.0 * BLOCK_HEIGHT)
    ok = report_and_assert(name, res, p_ref, abs(utop))
    ok = report_fresh_geometry(name, model, abs(utop)) and ok

    if ok:
        os.remove(inp_path)
        os.remove(setup_path)
    return ok, res


# ===========================================================================
# Lastfaelle
# ===========================================================================

# (Name, nx_a, nx_b, length_b, u_top, tilt)
CASES = [
    ("teilkontakt", 4, 2, 0.5 * LENGTH, -U_TOP, False),
    ("abheben", 2, 2, LENGTH, +U_TOP, False),
    ("schief", 4, 4, LENGTH, -U_TOP, True),
    ("voll", 2, 3, LENGTH, -U_TOP, False),
]

ELEMENTS = [("CPE4", "CONLINE2"), ("CPE8", "CONLINE3")]

# 3D: dieselben Lastfaelle mit Flaechenelementen. Der 'schief'-Fall entfaellt, weil
# die seitliche Dirichlet-Fixierung des 3D-Modells (u_x = u_z = 0 auf allen Knoten)
# mit einer verkippten Oberseitenverschiebung nicht vertraeglich ist; Teilkontakt,
# Abheben und volle Ueberdeckung decken beide Zweige des Active Sets ab.
CASES_3D = [
    ("teilkontakt", 3, 2, 0.5 * LENGTH, -U_TOP, False),
    ("abheben", 2, 2, LENGTH, +U_TOP, False),
    ("voll", 2, 3, LENGTH, -U_TOP, False),
]

ELEMENTS_3D = [("C3D8", "CONQUAD4"), ("C3D20", "CONQUAD8")]


def test_signorini_conditions():
    print("\n=== Signorini-Nachpruefung der konvergierten Loesung ===")
    results = []
    has_open = False   # Knoten MIT Gegenueber, aber lambda = 0
    has_closed = False  # Knoten mit g_sep = 0
    for case, nx_a, nx_b, length_b, utop, tilt in CASES:
        for el_type, con_type in ELEMENTS:
            ok, res = run_case(case, el_type, con_type, nx_a, nx_b, length_b, utop, tilt)
            results.append(ok)
            has_open |= bool(res["inactive"].any())
            has_closed |= bool(res["active"].any())

    print("\n--- 3D ---")
    for case, nx_a, nx_b, length_b, utop, tilt in CASES_3D:
        for el_type, con_type in ELEMENTS_3D:
            ok, res = run_case_3d(case, el_type, con_type, nx_a, nx_b, length_b, utop, tilt)
            results.append(ok)
            has_open |= bool(res["inactive"].any())
            has_closed |= bool(res["active"].any())

    n_fail = results.count(False)
    assert n_fail == 0, f"{n_fail} von {len(results)} Signorini-Pruefungen fehlgeschlagen"

    # Ohne beide Knotenklassen prueft die Testreihe das Active Set nicht, sondern
    # nur einen Sonderfall, in dem alle Knoten denselben Zweig nehmen.
    assert has_open, "Kein Lastfall erzeugt offene Knoten MIT Gegenueber -- Active Set ungeprueft"
    assert has_closed, "Kein Lastfall erzeugt geschlossene Knoten -- Active Set ungeprueft"
    print("\n  Beide Knotenklassen kamen vor: offen (lambda = 0) und geschlossen (g_sep = 0).")


if __name__ == "__main__":
    print("=" * 60)
    print("SIGNORINI-NACHPRUEFUNG DER KONVERGIERTEN MORTAR-LOESUNG")
    print("=" * 60)
    test_signorini_conditions()
    print("\n[SUCCESS] Alle Signorini-Pruefungen bestanden!")
