#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auswertung eines Control_Tests-Laufs gegen die Referenzloesung.
===============================================================

Fuehrt einen Zwei-Block-Kontakt-Job (und die zugehoerige Monolith-Referenz) in
EdelweissFE aus und wertet aus:

  1. L2-Fehler im Verschiebungsfeld
       - gegen die exakte analytische Loesung (Primaerreferenz, nu=0, uniaxial)
       - gegen den verschmolzenen Monolith-Lauf (isoliert den Kontakt)
  2. Spannungen an ALLEN Gausspunkten
       - Vergleich jeder GP-Spannung mit der analytischen Loesung
       - schreibt gp_report_<variante>.csv (fuer jeden GP alles: sigma, strain, Fehler)
  3. Kontaktbedingungen ("springen sie?")
       - Lagrange-Multiplikatoren lambda (= Kontaktdruck) je Slave-Knoten
       - gleiches Vorzeichen? |lambda| = p? gewichtetes Mittel = p? Oszillationsamplitude?

Nutzung (aus dem jeweiligen Testordner heraus, z.B. 02_pressure/):

    python ../evaluate.py pressure_hex8_matching.inp
    python ../evaluate.py pressure_hex8_matching.inp reference_monolith_hex8.inp

Der Testtyp (selfweight/pressure/dispcontrol) wird am Ordnernamen erkannt, die
Monolith-Referenz am Dateinamen (elementtyp-abhaengig) automatisch gefunden.

Materialparameter/Geometrie sind fuer alle Control_Tests fix:
    E = 1000, nu = 0, Bloecke y in [0, 2], p = 10 (Druck bzw. Kontaktdruck),
    Eigengewichts-Volumenkraft b = 10 (nach unten).
"""

import os
import re
import sys

import numpy as np

# ---- Pfade: Control_Tests und EdelweissFE-Wurzel auf sys.path ----------------
_THISDIR = os.path.abspath(os.path.dirname(__file__))          # .../Control_Tests
sys.path.insert(0, _THISDIR)
sys.path.insert(0, os.path.abspath(os.path.join(_THISDIR, "../..")))  # .../EdelweissFE

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.utils.inputfileparser import parseInputFile

# ---- feste Modellparameter ---------------------------------------------------
E = 1000.0
NU = 0.0
H = 2.0        # Gesamthoehe (zwei Einheitswuerfel)
P = 10.0       # Druck / Kontaktdruck
B = 10.0       # Eigengewichts-Volumenkraft (nach unten)

# inclined-Test: gedrehte Achse a = R(30 Grad)*e_y
_TH = np.radians(30.0)
AXIS = np.array([-np.sin(_TH), np.cos(_TH), 0.0])

# Steifigkeitskontrast-Test (07): Block A weich (Slave), Block B steif
E_A_STIFF = 1000.0
E_B_STIFF = 100000.0
U_TOP_STIFF = -0.02
SIGMA_STIFF = U_TOP_STIFF / (1.0 / E_A_STIFF + 1.0 / E_B_STIFF)  # Reihenschaltung, konstant

# Testklassen
COMPRESSION_TESTS = ("selfweight", "pressure", "dispcontrol", "stiffness")
BEHAVIOUR_TESTS = ("separation", "sliding", "inclined")


# =============================================================================
#  analytische Loesungen (uniaxial, nu = 0)
# =============================================================================
def analytical_uy(y, test):
    """Exakte vertikale Verschiebung u_y(y)."""
    if test == "selfweight":
        return (0.5 * B * y * y - B * H * y) / E
    # pressure / dispcontrol: konstanter Druck P
    return -P * y / E


def analytical_sigma_yy(y, test):
    """Exakte (globale) Vertikalspannung sigma_yy(y)."""
    if test == "selfweight":
        return -B * (H - y)
    if test == "separation":
        return 0.0                 # Kontakt trennt sich -> spannungsfrei
    if test == "inclined":
        return -P * AXIS[1] ** 2   # globales sigma_yy = -10 * a_y^2 = -7.5
    if test == "stiffness":
        return SIGMA_STIFF         # Reihenschaltung -> konstant ueber beide Bloecke
    return -P                      # pressure, dispcontrol, sliding


def analytical_disp(coords, test):
    """Exakter Verschiebungsvektor an einem Knoten mit gegebenen Koordinaten.

    Fuer separation/sliding ist das Feld am Interface unstetig (Bloecke A/B); diese
    Tests werden nicht ueber den Verschiebungs-L2 beurteilt, sondern ueber
    spezifische Kennzahlen (siehe evaluate_separation/-sliding)."""
    x, y, z = coords
    if test == "selfweight":
        return np.array([0.0, (0.5 * B * y * y - B * H * y) / E, 0.0])
    if test == "inclined":
        return -0.01 * (AXIS @ coords) * AXIS
    if test == "separation":
        # Kontakt trennt sich: unterer Block bleibt (0), oberer Block starr +0.02
        return np.array([0.0, 0.0 if y < 1.0 else 0.02, 0.0])
    if test == "stiffness":
        # stueckweise linear, stetig am Interface (u(1)=sigma/E_A):
        if y <= 1.0:
            uy = SIGMA_STIFF / E_A_STIFF * y
        else:
            uy = SIGMA_STIFF / E_A_STIFF + SIGMA_STIFF / E_B_STIFF * (y - 1.0)
        return np.array([0.0, uy, 0.0])
    # pressure / dispcontrol (und Naeherung fuer sliding-uy):
    return np.array([0.0, -P * y / E, 0.0])


def analytical_pressure(test=None):
    """Erwarteter Kontaktdruck am Interface (y = 1)."""
    if test == "stiffness":
        return abs(SIGMA_STIFF)   # Reihenschaltung -> 19.80198
    return P  # selfweight: B*hoehe_B = 10 ; pressure/dispcontrol/sliding: P = 10


# =============================================================================
#  Ausfuehrung eines .inp im Speicher
# =============================================================================
# Alle Control_Tests nutzen einen Step mit stepLength=1 -> Endzeit 1.0 bei Erfolg.
_END_TIME = 1.0


def run_job(inp_path):
    """Fuehrt inp_path aus (cwd = dessen Ordner) und liefert (model, foc, converged).

    finiteElementSimulation faengt StepFailed intern ab und liefert das Modell im
    letzten konvergierten Zustand zurueck. Konvergenz wird an model.time erkannt
    (erreicht 1.0 nur bei vollstaendig aufgebrachter Last)."""
    inp_path = os.path.abspath(inp_path)
    workdir = os.path.dirname(inp_path)
    prev = os.getcwd()
    os.chdir(workdir)
    try:
        inputFile = parseInputFile(os.path.basename(inp_path))
        model, foc = finiteElementSimulation(inputFile, verbose=False, suppressPlots=True)
    finally:
        os.chdir(prev)
    converged = model.time >= _END_TIME - 1e-6
    return model, foc, converged


# =============================================================================
#  L2-Fehler im Verschiebungsfeld
# =============================================================================
def l2_disp_vs_analytical(model, test):
    nf = model.nodeFields["displacement"]
    U = np.asarray(nf["U"])
    coords = np.array([n.coordinates for n in nf.nodes])
    u_ex = np.array([analytical_disp(c, test) for c in coords])
    num = np.sqrt(np.sum((U - u_ex) ** 2))
    den = np.sqrt(np.sum(u_ex ** 2))
    rel = num / den if den > 0 else num
    return dict(rel_l2=rel, abs_l2=num, max_abs=np.max(np.abs(U - u_ex)),
                max_lateral=np.max(np.abs(U[:, [0, 2]])))


def l2_disp_vs_monolith(model, mono_model):
    """Vergleicht Knotenverschiebungen ueber Koordinaten-Lookup gegen den Monolith."""
    nf = model.nodeFields["displacement"]
    U = np.asarray(nf["U"])
    coords = np.array([n.coordinates for n in nf.nodes])

    mnf = mono_model.nodeFields["displacement"]
    mU = np.asarray(mnf["U"])
    mco = np.array([n.coordinates for n in mnf.nodes])
    lookup = {tuple(np.round(c, 7)): mU[i] for i, c in enumerate(mco)}

    diffs = []
    missing = 0
    for c, u in zip(coords, U):
        key = tuple(np.round(c, 7))
        if key in lookup:
            diffs.append(u - lookup[key])
        else:
            missing += 1
    diffs = np.array(diffs)
    num = np.sqrt(np.sum(diffs ** 2))
    den = np.sqrt(np.sum(mU ** 2))
    return dict(rel_l2=num / den if den > 0 else num,
                max_abs=np.max(np.abs(diffs)) if len(diffs) else float("nan"),
                missing=missing)


# =============================================================================
#  Gausspunkt-Auswertung
# =============================================================================
# Marmot-GP-Ordnung: EMPIRISCH bestimmt (siehe README/final_report Verifikation).
# Bei den boxGen-Bloecken faellt die lokale zeta-Achse mit der globalen y-Achse
# zusammen (die 8 Eckknoten sind 4+4 entlang y aufgeteilt). Marmot durchlaeuft die
# GP zeta-major (zeta = langsamster Index): erst alle GP der untersten y-Ebene,
# dann die naechste usw. So stimmen die ausgegebenen GP-Koordinaten mit den
# tatsaechlichen Marmot-Spannungswerten ueberein (verifiziert ueber das
# Eigengewicht: y_GP = 2 + sigma_yy/10). Die x/z-Zuordnung innerhalb einer y-Ebene
# ist fuer die uniaxialen Tests ohne Belang.
_HEX8_NAT = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
], dtype=float)


def _N_hex8(xi, eta, zeta, nodeNat=_HEX8_NAT):
    return np.prod(0.5 * (1 + nodeNat * np.array([xi, eta, zeta])), axis=1)


def gp_natural_coords(nGP):
    """Natuerliche GP-Koordinaten (nGP x 3) in Marmot-Ordnung (eta = y = langsamster Index)."""
    if nGP == 8:
        axis = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
    elif nGP == 27:
        g = np.sqrt(0.6)
        axis = [-g, 0.0, g]
    else:
        raise ValueError(f"nGP={nGP} nicht unterstuetzt")
    pts = []
    for zeta in axis:         # zeta = globale y-Achse = langsamster Index (Marmot)
        for eta in axis:
            for xi in axis:
                pts.append((xi, eta, zeta))
    return np.array(pts)


def gp_world_coords(element, nGP):
    """GP-Weltkoordinaten (nGP x 3) via trilineare Abbildung der 8 Eckknoten
    (exakt, da boxGen-Elemente affin sind)."""
    Xc = np.array([n.coordinates for n in element.nodes[:8]])
    nat = gp_natural_coords(nGP)
    return np.array([_N_hex8(xi, eta, zeta) @ Xc for xi, eta, zeta in nat])


def element_centroid(element):
    return np.mean([n.coordinates for n in element.nodes], axis=0)


def evaluate_stress(foc, test, variant, outdir):
    """Per-GP-Spannungsvergleich + CSV-Dump. Liefert Kennzahlen-Dict."""
    fo = foc.fieldOutputs["stressGP"]
    fe = foc.fieldOutputs["strainGP"]
    sig = np.asarray(fo.getLastResult())   # [nEl, nGP, 6]
    eps = np.asarray(fe.getLastResult())
    elements = list(fo.associatedSet)
    nEl, nGP, _ = sig.shape

    # Voigt-Reihenfolge Marmot: [11, 22, 33, 12, 13, 23]
    iyy, ixx, izz = 1, 0, 2
    shear = [3, 4, 5]

    rows = []
    err_syy = []      # |sigma_yy - analytisch(gp_y)| ueber alle GP
    max_sxx = 0.0
    max_shear = 0.0

    for e, el in enumerate(elements):
        c = element_centroid(el)
        gpw = gp_world_coords(el, nGP)  # (nGP x 3), Marmot-Ordnung
        for g in range(nGP):
            syy_ex = analytical_sigma_yy(gpw[g, 1], test)
            err_syy.append(abs(sig[e, g, iyy] - syy_ex))
            rows.append([
                el.elNumber, g, gpw[g, 0], gpw[g, 1], gpw[g, 2],
                *sig[e, g, :], *eps[e, g, :],
                syy_ex, sig[e, g, iyy] - syy_ex,
            ])
        max_sxx = max(max_sxx, np.max(np.abs(sig[e, :, ixx])), np.max(np.abs(sig[e, :, izz])))
        max_shear = max(max_shear, np.max(np.abs(sig[e, :, shear])))

    err_syy = np.array(err_syy)

    # CSV schreiben
    header = ("elNumber,gp,gp_x,gp_y,gp_z,"
              "s11,s22,s33,s12,s13,s23,"
              "e11,e22,e33,e12,e13,e23,"
              "sigma_yy_exact,err_sigma_yy")
    path = os.path.join(outdir, f"gp_report_{variant}.csv")
    np.savetxt(path, np.array(rows), delimiter=",", header=header, comments="",
               fmt=["%d", "%d"] + ["%.10e"] * (len(header.split(",")) - 2))

    return dict(nEl=nEl, nGP=nGP, gp_csv=path,
                max_err_syy=float(err_syy.max()),
                rms_err_syy=float(np.sqrt(np.mean(err_syy ** 2))),
                max_sxx=float(max_sxx), max_shear=float(max_shear),
                syy_min=float(sig[:, :, iyy].min()), syy_max=float(sig[:, :, iyy].max()))


def element_mean_syy(foc):
    """Dict centroid(gerundet) -> mittlere sigma_yy je Element (ordnungsfrei)."""
    fo = foc.fieldOutputs["stressGP"]
    sig = np.asarray(fo.getLastResult())   # [nEl, nGP, 6]
    els = list(fo.associatedSet)
    return {tuple(np.round(element_centroid(el), 6)): float(sig[e, :, 1].mean())
            for e, el in enumerate(els)}


def stress_vs_monolith(foc_test, foc_mono):
    """Max|sigma_yy(Kontakt) - sigma_yy(Monolith)| ueber deckungsgleiche Elemente.

    Isoliert den vom KONTAKT verursachten Spannungsanteil von der reinen
    Element-Diskretisierung (die im Monolith identisch steckt). Nur sinnvoll fuer
    deckungsgleiche Netze (matching); bei non-matching gibt es keine gemeinsamen
    Element-Zentren -> None."""
    a = element_mean_syy(foc_test)
    b = element_mean_syy(foc_mono)
    common = set(a) & set(b)
    if len(common) < len(a) // 2:
        return None
    return max(abs(a[k] - b[k]) for k in common)


# =============================================================================
#  Kontaktbedingungen
# =============================================================================
def evaluate_contact(model, test=None):
    lam = np.array([v.value for v in model.scalarVariables.values()]).flatten()
    if lam.size == 0:
        return None
    p = analytical_pressure(test)
    c = model.constraints["contact"]
    rowsum = np.asarray(c.current_D_rowsum)
    same_sign = bool(np.all(lam > 0) or np.all(lam < 0))
    absdev = np.max(np.abs(np.abs(lam) - p)) / p
    weighted_mean = np.sum(np.abs(lam) * rowsum) / np.sum(rowsum)
    wm_err = abs(weighted_mean - p) / p
    osc = (lam.max() - lam.min()) / abs(np.mean(lam))
    return dict(n=int(lam.size), same_sign=same_sign, abs_mean=float(np.mean(np.abs(lam))),
                max_rel_dev=float(absdev), weighted_mean=float(weighted_mean),
                weighted_mean_rel_err=float(wm_err), oscillation=float(osc),
                lam_min=float(lam.min()), lam_max=float(lam.max()))


# =============================================================================
#  Reaktionskraft & effektive Steifigkeit (Punkt 1)
# =============================================================================
def reaction_stiffness(model):
    """Gesamte Vertikal-Reaktionskraft an der Unterseite und effektive Steifigkeit
    k = |F| / |u_oben| (E_eff = k*L/A, L=2, A=1). Sinnvoll fuer die Druck-Familie:
    erwartet F = 10 (pressure/dispcontrol) bzw. 20 (selfweight), k = 500, E_eff = 1000."""
    nf = model.nodeFields["displacement"]
    P = np.asarray(nf["P"])
    U = np.asarray(nf["U"])
    co = np.array([n.coordinates for n in nf.nodes])
    bot = np.isclose(co[:, 1], co[:, 1].min())
    top = np.isclose(co[:, 1], co[:, 1].max())
    Fy = float(P[bot, 1].sum())
    u_top = float(U[top, 1].mean())
    k = abs(Fy) / abs(u_top) if u_top != 0 else float("nan")
    return dict(F_react=Fy, u_top=u_top, k_eff=k, E_eff=k * 2.0)


# =============================================================================
#  Verhaltens-Tests (Punkte 2, 3, 5)
# =============================================================================
def _lambdas(model):
    return np.array([v.value for v in model.scalarVariables.values()]).flatten()


def write_lambda_vtk(model, path):
    """Schreibt die Kontaktdruecke lambda als Punktwolke (Legacy-VTK PolyData) an den
    Slave-Knoten des Interface. Direkt in ParaView ladbar: die Punkte lassen sich nach
    'lambda' (= Kontaktdruck) einfaerben. So sieht man den Kontaktdruck raeumlich,
    den EnSight (nur U/P/stress/strain) nicht zeigt."""
    if "contact" not in model.constraints:
        return None
    c = model.constraints["contact"]
    slaves = c.non_mortar_nodes
    lam = _lambdas(model)
    if len(slaves) != len(lam):
        return None
    X = np.array([n.coordinates for n in slaves])
    n = len(slaves)
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\ncontact pressure lambda\nASCII\nDATASET POLYDATA\n")
        f.write(f"POINTS {n} float\n")
        for p in X:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")
        f.write(f"VERTICES {n} {2 * n}\n")
        for i in range(n):
            f.write(f"1 {i}\n")
        f.write(f"POINT_DATA {n}\nSCALARS lambda float 1\nLOOKUP_TABLE default\n")
        for lv in lam:
            f.write(f"{float(lv)}\n")
    return path


def evaluate_separation(model, foc):
    """Zug/Abheben: erwartet Trennung -> Kontaktdruck 0, Spannung 0."""
    lam = _lambdas(model)
    sig = np.asarray(foc.fieldOutputs["stressGP"].getLastResult())
    nf = model.nodeFields["displacement"]
    U = np.asarray(nf["U"])
    co = np.array([n.coordinates for n in nf.nodes])
    lower = co[:, 1] < 0.99
    max_sig = float(np.abs(sig).max())
    max_lam = float(np.abs(lam).max()) if lam.size else 0.0
    u_lower = float(np.abs(U[lower]).max())
    return dict(max_sigma=max_sig, max_lam=max_lam, u_lowerblock=u_lower,
                separated=bool(max_lam < 1e-2 and max_sig < 1e-2))


def evaluate_sliding(model, foc):
    """Reibungsfreies Gleiten: Normaldruck bleibt, kein Schub, unterer Block ruht."""
    lam = _lambdas(model)
    sig = np.asarray(foc.fieldOutputs["stressGP"].getLastResult())
    nf = model.nodeFields["displacement"]
    U = np.asarray(nf["U"])
    co = np.array([n.coordinates for n in nf.nodes])
    lower = co[:, 1] < 0.99
    return dict(syy_mean=float(sig[:, :, 1].mean()),
                max_syy_err=float(np.abs(sig[:, :, 1] + P).max()),
                max_shear=float(np.abs(sig[:, :, 3]).max()),
                ux_lowerblock=float(np.abs(U[lower, 0]).max()),
                lam_mean=float(np.mean(lam)) if lam.size else float("nan"))


def evaluate_inclined(model, foc):
    """Schiefes Interface (Normalen-Test): der gesamte Aussenrand traegt die exakte
    lineare Loesung; der Innenraum inkl. schiefer Kontaktflaeche muss sie
    reproduzieren. Kennzahl: L2-Fehler der Verschiebung + Vorzeichen/Streuung des
    Kontaktdrucks (konsistente Normalen -> einheitliches Vorzeichen)."""
    l2 = l2_disp_vs_analytical(model, "inclined")
    lam = _lambdas(model)
    active = lam[np.abs(lam) > 1e-6]
    same_sign = bool(np.all(active > 0) or np.all(active < 0)) if active.size else True
    # Axialspannung sigma_aa = a^T sigma a soll -10 sein (Kontaktnormale = gedrehte Achse)
    sig = np.asarray(foc.fieldOutputs["stressGP"].getLastResult()).reshape(-1, 6)
    a = AXIS
    saa = np.array([a[0] * a[0] * v[0] + a[1] * a[1] * v[1] + a[2] * a[2] * v[2]
                    + 2 * a[0] * a[1] * v[3] + 2 * a[0] * a[2] * v[4] + 2 * a[1] * a[2] * v[5]
                    for v in sig])
    return dict(rel_l2=l2["rel_l2"], max_abs=l2["max_abs"],
                max_saa_err=float(np.abs(saa + P).max()),
                n_active=int(active.size), n_total=int(lam.size),
                lam_min=float(lam.min()) if lam.size else float("nan"),
                lam_max=float(lam.max()) if lam.size else float("nan"),
                same_sign=same_sign)


# =============================================================================
#  Hertz'scher Kontakt (Test 08)
# =============================================================================
HERTZ_THICK = 0.1  # z-Dicke der Scheibe (muss zum .inp passen)


def _full_facet_weights(c, model):
    """Tributaere Flaeche je Slave-Knoten, ueber die VOLLEN Facetten integriert.

    A_j = sum_{Facetten f, die j enthalten} int_f N_tilde_j dGamma.

    Nicht zu verwechseln mit dem dualen Knotengewicht D_II = int_UEBERLAPPUNG:
    an voll ueberdeckten Knoten sind beide gleich, an teilweise ueberdeckten ist
    D_II kleiner -- genau das ist der Punkt der folgenden Auswertung.

    Ausgewertet wird in der DEFORMIERTEN Konfiguration, weil D_II es ebenfalls ist.
    Mit undeformierten Koordinaten traegt das Verhaeltnis D_II/A_j sonst die
    Flaechenaenderung durch die Verformung mit und verfaelscht den Vergleich
    systematisch (auf den Hertz-Modellen um rund ein Prozent).
    """
    dim = model.domainSize
    nf = model.nodeFields[c.field]
    u_of_node = {node: np.asarray(u)[:dim] for node, u in zip(nf.nodes, nf["U"])}

    A = np.zeros(c.nNonMortarNodes)
    for el, _ in c.non_mortar_facets:
        coords = np.array([nd.coordinates + u_of_node[nd] for nd in el.nodes])
        _, D_e, _ = el.computeLocalMassMatrices(coords)
        for a, nd in enumerate(el.nodes):
            A[c.slave_node_to_idx[nd]] += D_e[a, a]
    return A


def _hertz_slave_pressure(model):
    """Slave-Knoten-x und Kontaktdruck je EINDEUTIGEM x (ueber z gemittelt).

    Rueckgabe: (x, p_lambda, p_kraft) -- ZWEI Lesarten desselben Ergebnisses:

    p_lambda = -lambda_j
        Der Multiplikator selbst. An voll ueberdeckten Knoten ist das der
        Kontaktdruck. An TEILWEISE ueberdeckten Knoten nicht: dort ist lambda die
        Amplitude einer dualen Formfunktion ueber schrumpfendem Traeger und
        skaliert mit 1/D_II, waehrend die Knotenkraft lambda*D_II endlich bleibt
        (nachgemessen in 10_signorini_check: D_II/max = 8.3e-6 bei lambda = 8.2e4).
        Genau solche Knoten liegen am KONTAKTRAND -- dort, wo die Zickzack-
        Oszillation quadratischer Elemente ausgewiesen wird.

    p_kraft = -lambda_j * D_II,j / A_j
        Die Knotenkraft, verteilt ueber die tributaere Flaeche der ganzen Facette.
        An voll ueberdeckten Knoten ist D_II = A_j, beide Lesarten fallen zusammen.
        An teilweise ueberdeckten Knoten ist p_kraft kleiner -- der Knoten traegt
        seine Kraft eben nur ueber einen Teil seines Traegers. Die uebertragene
        Gesamtkraft sum_j p_kraft_j * A_j = sum_j lambda_j * D_II,j ist in beiden
        Lesarten dieselbe.

    Der Vergleich beider Profile trennt den echten Ecke/Mittelknoten-Effekt vom
    Auswertungsartefakt am Kontaktrand.
    """
    c = model.constraints["contact"]
    slaves = c.non_mortar_nodes
    lam = _lambdas(model)
    xs = np.array([n.coordinates[0] for n in slaves])

    D_II = np.asarray(c.current_D_rowsum)
    A = _full_facet_weights(c, model)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(np.abs(A) > 1e-30, D_II / A, 0.0)

    p_lam = -lam                  # Druck positiv
    p_force = -lam * share

    # ueber gleiche x zusammenfassen (2 Knoten je x wegen z=0/0.1)
    xu = np.unique(np.round(xs, 6))
    pu = np.array([p_lam[np.isclose(xs, x)].mean() for x in xu])
    pfu = np.array([p_force[np.isclose(xs, x)].mean() for x in xu])
    return xu, pu, pfu


def evaluate_hertz(model):
    import contact_setup
    R = contact_setup.HERTZ_R
    Estar = 1.0 / ((1 - NU ** 2) / E + (1 - NU ** 2) / E)  # ebener Verzerrungszustand
    x, p, p_force = _hertz_slave_pressure(model)

    # uebertragene Last P' (Kraft je Laenge): Reaktion unten, Halbmodell -> *2
    nf = model.nodeFields["displacement"]
    P_react = np.asarray(nf["P"])
    co = np.array([n.coordinates for n in nf.nodes])
    Fy = P_react[np.isclose(co[:, 1], co[:, 1].min()), 1].sum()
    Pprime = abs(Fy) / HERTZ_THICK * 2.0

    a = np.sqrt(4 * Pprime * R / (np.pi * Estar))
    p0 = 2 * Pprime / (np.pi * a)

    active = p > 1e-6 * p0

    # Maximaldruck: der FE-Wert am Zentrum (groesster Knotendruck) - der uebliche,
    # unmittelbare Vergleichswert.
    p0_num = float(p.max())
    # Kontakthalbbreite robuster als "letzter aktiver Knoten": Hertz -> p^2 linear in
    # x^2. Least-squares-Fit p^2 = A + B*x^2 ueber die aktiven Knoten, Nullstelle bei
    # x^2 = -A/B -> a = sqrt(-A/B). (Die Steigung/Nullstelle ist robust; nur der
    # Achsenabschnitt A waere durch Randeffekte verzerrt, daher p0 aus dem Peak.)
    xa, pa = x[active], p[active]
    if xa.size >= 3:
        B, A = np.linalg.lstsq(np.vstack([xa ** 2, np.ones_like(xa)]).T, pa ** 2, rcond=None)[0]
        a_num = float(np.sqrt(max(-A / B, 0.0))) if B < 0 else float(xa.max())
    else:
        a_num = float(xa.max()) if xa.size else 0.0

    inside = x < a
    p_hertz = p0 * np.sqrt(np.clip(1 - (x / a) ** 2, 0.0, None))
    err = np.abs(p[inside] - p_hertz[inside])
    # Knoten-zu-Knoten-Oszillation ("Zickzack"): mittlere |2. Differenz| des Drucks
    # ueber die aktiven Knoten, relativ zum Maximaldruck. Misst die Ecke/Mittelknoten-
    # Schwankung, die bei quadratischen Elementen (hex20) auftritt.
    pa_sorted = p[active][np.argsort(x[active])]
    zigzag = float(np.mean(np.abs(np.diff(pa_sorted, 2))) / p0_num) if pa_sorted.size >= 3 and p0_num > 0 else float("nan")

    # Gegenrechnung mit der kraftbasierten Lesart (siehe _hertz_slave_pressure).
    # Weicht das Zickzack der beiden Profile deutlich voneinander ab, so stammt
    # ein Teil der Oszillation aus der Auswertung am Kontaktrand und nicht aus dem
    # Ecke/Mittelknoten-Effekt.
    pf_sorted = p_force[active][np.argsort(x[active])]
    zigzag_force = (float(np.mean(np.abs(np.diff(pf_sorted, 2))) / p0_num)
                    if pf_sorted.size >= 3 and p0_num > 0 else float("nan"))

    return dict(Pprime=float(Pprime), a_hertz=float(a), p0_hertz=float(p0),
                a_num=a_num, p0_num=p0_num, p_peak=float(p.max()), zigzag=zigzag,
                p0_force=float(p_force.max()), zigzag_force=zigzag_force,
                peak_err_rel=float(abs(p0_num - p0) / p0),
                a_err_rel=float(abs(a_num - a) / a) if a > 0 else float("nan"),
                max_p_err=float(err.max()) if err.size else float("nan"),
                rms_p_err=float(np.sqrt(np.mean(err ** 2))) if err.size else float("nan"),
                n_active=int(active.sum()))


def write_hertz_profile(model, path):
    """CSV: x, Kontaktdruck_num, Hertz_p(x) - zum Ansehen/Plotten der Druckverteilung."""
    import contact_setup
    R = contact_setup.HERTZ_R
    Estar = 1.0 / ((1 - NU ** 2) / E + (1 - NU ** 2) / E)
    x, p, p_force = _hertz_slave_pressure(model)
    nf = model.nodeFields["displacement"]
    P_react = np.asarray(nf["P"])
    co = np.array([n.coordinates for n in nf.nodes])
    Fy = P_react[np.isclose(co[:, 1], co[:, 1].min()), 1].sum()
    Pprime = abs(Fy) / HERTZ_THICK * 2.0
    a = np.sqrt(4 * Pprime * R / (np.pi * Estar))
    p0 = 2 * Pprime / (np.pi * a)
    p_h = p0 * np.sqrt(np.clip(1 - (x / a) ** 2, 0.0, None))
    rows = np.column_stack([x, p, p_force, p_h])
    np.savetxt(path, rows, delimiter=",", header="x,pressure_lambda,pressure_force,pressure_hertz",
               comments="", fmt="%.8e")
    return path


# =============================================================================
#  Haupt
# =============================================================================
ALL_TESTS = COMPRESSION_TESTS + BEHAVIOUR_TESTS


def detect_test_type(inp_path):
    parent = os.path.basename(os.path.dirname(os.path.abspath(inp_path)))
    b = os.path.basename(inp_path)
    if "hertz" in parent or "hertz" in b:
        return "hertz"
    for t in ALL_TESTS:
        if t in parent or t in b:
            return t
    raise ValueError("Testtyp nicht erkennbar aus " + inp_path)


def find_monolith(inp_path):
    folder = os.path.dirname(os.path.abspath(inp_path))
    b = os.path.basename(inp_path)
    m = re.search(r"(hex20R|hex20|hex8)", b)
    if not m:
        return None
    cand = os.path.join(folder, f"reference_monolith_{m.group(1)}.inp")
    return cand if os.path.exists(cand) else None


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    inp = argv[1]
    test = detect_test_type(inp)
    variant = os.path.splitext(os.path.basename(inp))[0]
    outdir = os.path.dirname(os.path.abspath(inp))
    mono = argv[2] if len(argv) > 2 else find_monolith(inp)

    print(f"\n{'=' * 70}\n Control-Test-Auswertung: {variant}   (Typ: {test})\n{'=' * 70}")

    model, foc, converged = run_job(inp)
    if not converged:
        print(f"\n  !!! LOESER NICHT KONVERGIERT (model.time = {model.time:.4f} < 1.0) !!!")
        print("  Die Last konnte nicht voll aufgebracht werden - Ergebnis unbrauchbar.")
        print("  (Das IST das Ergebnis fuer diese Variante: Kontakt+Element konvergiert nicht.)")
        print(f"\n{'=' * 70}\n")
        return 2

    # Kontaktdruck als ParaView-Punktwolke (falls Kontakt vorhanden)
    vtk = write_lambda_vtk(model, os.path.join(outdir, f"lambda_{variant}.vtk"))
    if vtk:
        print(f"\n[Kontaktdruck fuer ParaView]  -> {os.path.basename(vtk)}  (Punkte nach 'lambda' einfaerben)")

    # --- Hertz'scher Kontakt -------------------------------------------------
    if test == "hertz":
        h = evaluate_hertz(model)
        prof = write_hertz_profile(model, os.path.join(outdir, f"hertz_profile_{variant}.csv"))
        print("\n[Hertz'scher Kontakt]  Vergleich der Druckverteilung gegen die Hertz-Loesung")
        print(f"    uebertragene Last P'      = {h['Pprime']:.4f}")
        print(f"    Kontakthalbbreite a       = {h['a_num']:.4f} (num, Fit) vs {h['a_hertz']:.4f} (Hertz)  "
              f"-> {h['a_err_rel'] * 100:.1f}% Abw.")
        print(f"    Maximaldruck p0           = {h['p0_num']:.4f} (num, Fit) vs {h['p0_hertz']:.4f} (Hertz)  "
              f"-> {h['peak_err_rel'] * 100:.1f}% Abw.")
        print(f"    max|p_num - p_Hertz|      = {h['max_p_err']:.4f}  (rms {h['rms_p_err']:.4f})")
        print(f"    aktive Kontaktknoten      = {h['n_active']}")
        print(f"    -> Druckprofil (x, p_num, p_Hertz): {os.path.basename(prof)}")
        print(f"\n{'=' * 70}\n")
        return 0

    # --- Verhaltens-Tests (separation / sliding / inclined) ------------------
    if test in BEHAVIOUR_TESTS:
        evaluate_stress(foc, test, variant, outdir)  # schreibt gp_report_*.csv
        if test == "separation":
            s = evaluate_separation(model, foc)
            print("\n[Zug / Abheben]  Erwartung: Kontakt trennt sich (Druck 0, Spannung 0)")
            print(f"    max|Kontaktdruck lambda|  = {s['max_lam']:.3e}   (-> 0)")
            print(f"    max|Spannung|             = {s['max_sigma']:.3e}   (-> 0)")
            print(f"    max|u| unterer Block      = {s['u_lowerblock']:.3e}   (-> 0, bleibt in Ruhe)")
            print(f"    URTEIL: {'getrennt (unilateral) - OK' if s['separated'] else 'NICHT getrennt (wie Verklebung)'}")
        elif test == "sliding":
            s = evaluate_sliding(model, foc)
            print("\n[Tangentiales Gleiten]  Erwartung reibungsfrei: Normaldruck bleibt, kein Schub")
            print(f"    sigma_yy Mittel           = {s['syy_mean']:.5f}   (-> -10, Normaldruck bleibt)")
            print(f"    max|sigma_yy + 10|        = {s['max_syy_err']:.3e}")
            print(f"    max|Schub sigma_xy|       = {s['max_shear']:.3e}   (-> 0, reibungsfrei)")
            print(f"    max|u_x| unterer Block    = {s['ux_lowerblock']:.3e}   (-> 0, nicht mitgeschleppt)")
            print(f"    Kontaktdruck lambda Mittel= {s['lam_mean']:.5f}   (-> -10)")
        else:  # inclined
            s = evaluate_inclined(model, foc)
            print("\n[Schiefes Interface 30 Grad]  Normalen-Test (Patch-Loesung auf dem Aussenrand)")
            print(f"    rel. L2 (vs exakt gedreht)= {s['rel_l2']:.3e}")
            print(f"    max|u - u_exakt|          = {s['max_abs']:.3e}")
            print(f"    max|Axialspannung + 10|   = {s['max_saa_err']:.3e}   (Normale korrekt -> sigma_aa = -10)")
            print(f"    Kontaktdruck lambda       = [{s['lam_min']:.4f}, {s['lam_max']:.4f}], "
                  f"aktiv {s['n_active']}/{s['n_total']}, gleiches Vorzeichen = {s['same_sign']}")
        print(f"\n{'=' * 70}\n")
        return 0

    # --- Druck-Familie (selfweight / pressure / dispcontrol) -----------------
    # 1) L2 Verschiebung
    la = l2_disp_vs_analytical(model, test)
    print("\n[1] L2-Fehler Verschiebungsfeld")
    print(f"    rel. L2 (vs analytisch)   = {la['rel_l2']:.3e}")
    print(f"    max|u - u_exakt|          = {la['max_abs']:.3e}")
    print(f"    max|u_lateral| (x,z)      = {la['max_lateral']:.3e}")
    mono_foc = None
    if mono:
        mono_model, mono_foc, mono_conv = run_job(mono)
        if not mono_conv:
            print("    (Monolith-Referenz nicht konvergiert - uebersprungen)")
            mono = None
    if mono:
        lm = l2_disp_vs_monolith(model, mono_model)
        print(f"    rel. L2 (vs Monolith)     = {lm['rel_l2']:.3e}   "
              f"max|diff| = {lm['max_abs']:.3e}")
    else:
        print("    (kein Monolith gefunden)")

    # 2) Spannungen an allen GP
    st = evaluate_stress(foc, test, variant, outdir)
    print(f"\n[2] Spannungen an allen Gausspunkten  (nEl={st['nEl']}, nGP={st['nGP']})")
    print(f"    max|sigma_yy - exakt|     = {st['max_err_syy']:.3e}")
    print(f"    rms|sigma_yy - exakt|     = {st['rms_err_syy']:.3e}")
    print(f"    sigma_yy Bereich          = [{st['syy_min']:.5f}, {st['syy_max']:.5f}]")
    print(f"    max|sigma_xx|,|sigma_zz|  = {st['max_sxx']:.3e}")
    print(f"    max|Schub|                = {st['max_shear']:.3e}")
    if mono_foc is not None:
        svm = stress_vs_monolith(foc, mono_foc)
        if svm is not None:
            print(f"    max|sigma_yy - MONOLITH|  = {svm:.3e}   "
                  f"(= reiner KONTAKT-Anteil; Rest oben ist Element-Diskretisierung)")
    print(f"    -> Per-GP-Dump: {os.path.basename(st['gp_csv'])}")

    # 3) Kontaktbedingungen
    ct = evaluate_contact(model, test)
    print("\n[3] Kontaktbedingungen (Lagrange-Multiplikatoren = Kontaktdruck)")
    if ct is None:
        print("    (keine Kontakt-Multiplikatoren - Monolith?)")
    else:
        print(f"    n Multiplikatoren         = {ct['n']}")
        print(f"    gleiches Vorzeichen       = {ct['same_sign']}")
        print(f"    |lambda| Mittel           = {ct['abs_mean']:.5f}   (erwartet {analytical_pressure(test):.4f})")
        print(f"    max rel. Abw. von p       = {ct['max_rel_dev']:.3e}")
        print(f"    gewichtetes Mittel        = {ct['weighted_mean']:.5f}   "
              f"(rel. Fehler {ct['weighted_mean_rel_err']:.3e})")
        print(f"    Oszillationsamplitude     = {ct['oscillation']:.3e}   "
              f"(lambda in [{ct['lam_min']:.4f}, {ct['lam_max']:.4f}])")

    # 4) Reaktionskraft & effektive Steifigkeit
    rs = reaction_stiffness(model)
    F_exp = 20.0 if test == "selfweight" else (abs(SIGMA_STIFF) if test == "stiffness" else 10.0)
    print("\n[4] Reaktionskraft & effektive Steifigkeit")
    print(f"    Reaktionskraft unten F_y  = {rs['F_react']:.5f}   (erwartet {F_exp:.1f})")
    print(f"    u_oben                    = {rs['u_top']:.5f}")
    print(f"    Steifigkeit k = |F/u|     = {rs['k_eff']:.4f}")
    if test == "stiffness":
        E_eff_exp = 2.0 / (1.0 / E_A_STIFF + 1.0 / E_B_STIFF)  # Reihen-Ersatzmodul
        print(f"    E_eff = k*L/A             = {rs['E_eff']:.4f}   (Reihe erwartet {E_eff_exp:.2f})")
    else:
        print(f"    E_eff = k*L/A             = {rs['E_eff']:.4f}   (erwartet {E:.0f})")

    print(f"\n{'=' * 70}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
