"""Gemeinsame Testinfrastruktur der Mortar-Kontakt-Suite.

Der Kontakt kennt drei Formulierungen derselben diskreten Zwangsbedingung
g_A = 0.  Sie teilen die gesamte geometrische Verarbeitung (Suche, Projektion,
Clipping, Segmentierung, Knotennormalen, D und C) und unterscheiden sich allein
im Zwangsgesetz:

  lagrange       Sattelpunkt, die Multiplikatoren sind Unbekannte des
                 Gesamtsystems (Popp et al. 2009/2012; Gitterle et al. 2010).
  penalty        reines Penalty, t_A = kappa*g_A
                 (Puso & Laursen 2004, Gl. (24); Puso, Laursen & Solberg 2008,
                 Gl. (7)).
  penalty_uzawa  dasselbe im augmentierten Lagrange-Verfahren,
                 p^(k+1) = p^k + kappa*g_A (Puso et al. 2008, Gl. (18)).

Tests, die eine dieser Formulierungen pruefen, nehmen die Fixture
``formulation`` und laufen damit dreimal.  Tests der reinen Geometrie und
Algebra (01 bis 05) nehmen sie NICHT: sie sind formulierungsunabhaengig, und
drei Kopien desselben Ergebnisses waeren nur Laufzeit.
"""

import datetime
import os
import subprocess

import pytest

from edelweissfe.constraints import mortarcontact as _mc

FORMULATIONS = ("lagrange", "penalty", "penalty_uzawa")

#: Was die einzelnen Formulierungen an Zwangserfuellung leisten koennen.  Der
#: Sattelpunkt erfuellt g_A = 0 exakt, bis auf das Rauschen des linearen
#: Loesers.  Das augmentierte Verfahren treibt sie auf die Augmentierungs-
#: toleranz herunter (Puso et al. 2008 lassen sie ausdruecklich offen), reines
#: Penalty laesst eine Durchdringung der Ordnung 1/kappa stehen - das ist die
#: Eigenschaft des Verfahrens und kein Mangel der Implementierung.  Tests, die
#: eine Schranke an die Zwangserfuellung legen, skalieren sie hiermit, statt
#: eine Zahl zu fuehren, die nur fuer den Sattelpunkt gilt.
CONSTRAINT_TOLERANCE_FACTOR = {
    # Sattelpunkt: g_A = 0 ist eine Gleichung des Systems, erfuellt bis auf das
    # Rauschen des linearen Loesers. Gemessen 1e-17 bis 1e-14.
    "lagrange": 1.0,
    # Augmentiertes Verfahren: die Zwangsverletzung faellt geometrisch bis zur
    # Augmentierungstoleranz (Default 1e-6 relative Multiplikatoraenderung).
    # Gemessen ueber die Suite: Restdurchdringung 1e-11 bis 1e-9, Druckabweichung
    # gegenueber dem Sattelpunkt 1e-9 bis 1e-8. Der Faktor 1e3 laesst dafuer reichlich Luft.
    "penalty_uzawa": 1.0e3,
    # Reines Penalty: die Durchdringung ist von der Ordnung 1/kappa und bleibt
    # stehen - das ist die Eigenschaft des Verfahrens, kein Mangel. Mit dem
    # abgeleiteten kappa gemessen: 6,5e-3 (hex20 passend), 1,05e-2 (gemischte
    # Ordnungen), 8e-3 (Zwei-Block). Die Schranke liegt mit 5e-2 etwa Faktor 5
    # darueber, also weit genug, um nicht auf Rauschen zu reagieren, und eng
    # genug, um eine echte Verschlechterung zu bemerken.
    "penalty": 5.0e6,
}


def _optionsFor(formulation: str) -> dict:
    """Die Optionen, die eine Formulierung ausmachen."""
    return {
        "formulation": "lagrange" if formulation == "lagrange" else "penalty",
        "augmentedLagrange": formulation == "penalty_uzawa",
    }


@pytest.fixture(params=FORMULATIONS)
def formulation(request):
    """Laesst einen Test in allen drei Formulierungen laufen.

    Injiziert wird in den Konstruktor des Constraints, NICHT in die Vorgabewerte des
    Optionsschemas: dessen Felder sind eine frozen dataclass, deren Vorgaben beim
    Anlegen der Klasse in die ``__init__``-Signatur eingebacken werden -- ein
    nachtraeglich gesetztes ``__dataclass_fields__[...].default`` bleibt folgenlos
    und liesse alle drei Parametrisierungen still als ``lagrange`` laufen.

    Eine Option, die der Aufrufer selbst setzt (Testaufbau oder Eingabedatei),
    behaelt Vorrang; geprueft wird ohne Ruecksicht auf Gross-/Kleinschreibung, weil
    die Eingabesprache das ebenfalls nicht unterscheidet.
    """
    mode = request.param
    injected = _optionsFor(mode)
    originalInit = _mc.Constraint.__init__

    def patchedInit(self, name, model, *args, **kwargs):
        merged = dict(kwargs)
        present = {str(k).casefold() for k in merged}
        for key, value in injected.items():
            if key.casefold() not in present:
                merged[key] = value
        originalInit(self, name, model, *args, **merged)

    _mc.Constraint.__init__ = patchedInit
    try:
        yield mode
    finally:
        _mc.Constraint.__init__ = originalInit


def requires_lagrange(formulation: str, reason: str):
    """Ueberspringt einen Test, der nur fuer den Sattelpunkt eine Aussage hat.

    Nicht jeder Test ist in jeder Formulierung sinnvoll.  Was die semi-smoothe
    NCP oder die Multiplikatorzeile des Gleichungssystems prueft, hat im
    Penalty-Zweig kein Gegenstueck - dort gibt es weder das eine noch das
    andere.  Solche Tests melden das ausdruecklich, statt stillschweigend zu
    bestehen oder unverstanden zu scheitern.
    """
    if formulation != "lagrange":
        pytest.skip(f"not required for formulation={formulation}: {reason}")


def requires_pressure_estimate(formulation: str, reason: str):
    """Ueberspringt einen Test, der einen von der Durchdringung UNABHAENGIGEN
    Druckschaetzer braucht.

    Den gibt es im Sattelpunkt (der Multiplikator ist eine eigene Unbekannte) und
    im augmentierten Verfahren (der akkumulierte Schaetzer p^k), nicht aber im
    reinen Penalty: dort ist der Druck ausschliesslich kappa*g_A, das Verfahren
    IST die Durchdringungsheuristik.
    """
    if formulation == "penalty":
        pytest.skip(f"not required for formulation={formulation}: {reason}")


def set_nodal_pressure(constraint, U_np, p_n):
    """Setzt den knotenweisen Druckschaetzer p_n, formulierungsunabhaengig.

    Im Sattelpunkt ist er die Multiplikator-Unbekannte, lambda = p_n*sgn(D_II)
    (die Vorzeichenkonvention des Codes, siehe applyConstraint).  Im
    augmentierten Verfahren ist er der akkumulierte Schaetzer z_aug, der bereits
    der physikalische Druck ist.  Beide Male bedeutet ein positiver Wert Druck.
    """
    import numpy as np

    if constraint.formulation == "lagrange":
        idx_LM_0 = constraint.sizeField * len(constraint.nodes)
        U_np[idx_LM_0:] = np.asarray(p_n) * np.sign(constraint.current_D_rowsum)
    else:
        # Auch den KONVERGIERTEN Schaetzer setzen: applyConstraint startet jedes
        # Increment warm aus z_aug_converged (und ein Cutback-Neuversuch ebenso),
        # ein nur in z_aug abgelegter Wert waere beim naechsten Aufruf wieder weg.
        constraint.z_aug[:] = p_n
        constraint.z_aug_converged[:] = p_n


def constraint_tolerance(formulation: str, base: float) -> float:
    """Schranke an die ZWANGSERFUELLUNG, der Formulierung angemessen."""
    return base * CONSTRAINT_TOLERANCE_FACTOR[formulation]


# ---------------------------------------------------------------------------
# Netzgeneratoren
# ---------------------------------------------------------------------------
# Upstream hat die Generatoren von einer Funktion `generateModelData(model,
# options, journal)` auf Klassen mit Optionsschema umgestellt (Generator(name,
# model, journal, configuration=...)); die Erzeugung passiert jetzt IM
# Konstruktor. Die folgenden Adapter halten die Aufrufform der Tests stabil,
# damit die Umstellung nicht in jeden einzelnen Test hineinreicht.


def _runGenerator(generatorModule, generatorDefinition, model, journal, **kwargs):
    from edelweissfe.utils.schema import buildSchemaFromOptions

    name = (generatorDefinition or {}).get("name", "gen")
    configuration = buildSchemaFromOptions(generatorModule.Generator.schema, kwargs)
    # Knoten- und Elementnummern duerfen nur innerhalb einer Topologieaenderung
    # vergeben werden; die .inp-Strecke oeffnet den Kontext selbst, ein direkter
    # Aufruf aus einem Test muss es ebenso tun.
    with model.topologyChanges():
        generatorModule.Generator(name, model, journal, configuration=configuration)
    return model


def generateBoxMesh(generatorDefinition, model, journal, *args, **kwargs):
    """boxGen in der Aufrufform der Tests."""
    from edelweissfe.generators import boxgen

    return _runGenerator(boxgen, generatorDefinition, model, journal, **kwargs)


def generatePlaneMesh(generatorDefinition, model, journal, *args, **kwargs):
    """planeRectQuad in der Aufrufform der Tests."""
    from edelweissfe.generators import planerectquad

    return _runGenerator(planerectquad, generatorDefinition, model, journal, **kwargs)


# ---------------------------------------------------------------------------
# Auswertungstabelle
# ---------------------------------------------------------------------------

#: Bekannte offene Punkte, nach Test und Formulierung. Sie stehen hier und nicht
#: im jeweiligen Test, weil sie Aussagen ueber die FORMULIERUNG sind und nicht
#: ueber den Testaufbau - und weil der Report sie ohne Quelltextlektuere lesbar
#: machen soll. Jeder Eintrag ist gemessen, nicht vermutet.
KNOWN_OPEN = {
    ("10_signorini_check::test_signorini_conditions", "penalty_uzawa"): (
        "Penalty bounds the WEIGHTED gap g_A, not the physical opening "
        "g_sep = g_A/D_II. At nodes with sliver coverage (measured D_II/max = "
        "2.5e-05 ... 1.4e-04) that division amplifies the remaining violation: "
        "g_sep reaches 1.5e-03 where the saddle point holds 1e-16. The nodal FORCE "
        "there stays negligible (-1.4e-05), so no load statement is affected - but "
        "the Signorini conditions are checked pointwise and do fail. Open question: "
        "whether such nodes should be excluded from the pointwise check, as the "
        "sliver diagnostic already identifies them."
    ),
    # 2026-09-14 BEHOBEN (Ursache lag im Solver, nicht im Kontakt): checkConvergence
    # durfte bei Iteration 0 - ohne berechnete Korrektur - Konvergenz erklaeren,
    # weil die fehlende Korrektur als kleine Korrektur zaehlte und das
    # Flusskriterium einen ABSOLUTEN Boden (1e-1 fuer displacement) traegt, den die
    # Knotenkraefte bei kleiner Laengenskala ohnehin unterschreiten. Gemessen: ein
    # Residuum von 2,5e-03 gegen ein Flussmass von 5,4e-04 galt als konvergiert.
    # Ohne Multiplikatorzeilen (also nur im Penalty-Zweig) ist dieses gebodete
    # Kriterium das einzige, weshalb der Sattelpunkt davon nie betroffen war.
    ("11_scale_invariance::test_scale_invariance", "penalty_uzawa_BEHOBEN"): (
        "NOT a tolerance issue, and NOT the penalty law: at the small end of the "
        "length scale (k = 1e-3, model extent 0.002) the augmented Lagrangian "
        "returns a contact pressure of exactly 5.0 instead of 10.0 - a factor of "
        "two - while PURE penalty at the same scale gives 9.975 (accurate to "
        "2.5e-03) and both are exact to 1e-09 at k = 1 and k = 1e3. The defect is "
        "therefore in the OUTER loop, not in t_A = kappa*g_A. The likely mechanism "
        "is the one already found once: an augmentation whose force change falls "
        "below the solver's residual bound leaves the displacements unmoved while "
        "the pressure estimate keeps advancing. At this scale kappa is derived as "
        "1.8e+15, so that bound is reached almost immediately. UNRESOLVED - and the "
        "first thing to check is the interaction of the augmentation with the "
        "solver tolerance at extreme kappa."
    ),
    ("12_increment_size::test_increment_size_convergence", "penalty_BEHOBEN"): (
        "The test measures the ORDER of the staggering error over the step size. "
        "Pure penalty adds its own O(1/kappa) error, which does not fall with the "
        "step size and therefore masks the quantity being measured. With the "
        "augmented Lagrangian the constraint error is driven out and the order is "
        "recovered (that column passes). Expected, not a defect - but it means the "
        "staggering order cannot be measured in the pure penalty branch."
    ),
}

_RESULTS = {}
_FULL_RUN = True


def _wrap(text, width=84, indent=" " * 6):
    """Bricht einen Erlaeuterungstext auf Reportbreite um."""
    import textwrap

    return ("\n" + indent).join(textwrap.wrap(" ".join(text.split()), width))


def pytest_runtest_logreport(report):
    if report.when != "call" and not (report.when == "setup" and report.skipped):
        return
    nodeid = report.nodeid
    mode = None
    if "[" in nodeid and nodeid.endswith("]"):
        head, _, param = nodeid.rpartition("[")
        param = param[:-1]
        if param in FORMULATIONS:
            nodeid, mode = head, param
    if report.skipped:
        outcome, note = "SKIP", (report.longrepr[2] if isinstance(report.longrepr, tuple) else "")
        note = note.replace("Skipped: ", "")
    elif report.failed:
        outcome, note = "FAIL", ""
    else:
        outcome, note = "PASS", ""
    entry = _RESULTS.setdefault(nodeid, {"modes": {}, "note": ""})
    if mode:
        entry["modes"][mode] = outcome
        if note and not entry["note"]:
            entry["note"] = note
    else:
        entry["modes"]["all"] = outcome


def pytest_collection_modifyitems(config, items):
    """Merkt sich, ob dieser Lauf die ganze Suite umfasst."""
    _RESULTS.clear()
    global _FULL_RUN
    root = os.path.dirname(os.path.abspath(__file__))
    # Eingeschraenkt ist ein Lauf, sobald ein Ausdruck filtert ODER ein Pfad
    # UNTERHALB dieses Verzeichnisses angegeben wurde. Ein leeres args oder der
    # Suite-Ordner selbst gilt als vollstaendig.
    narrowed = any(
        os.path.abspath(a.split("::")[0].rstrip("/")) != root
        for a in config.args
        if not a.startswith("-")
    )
    _FULL_RUN = not (config.option.keyword or config.option.markexpr or narrowed)


def pytest_sessionfinish(session, exitstatus):
    if not _RESULTS:
        return
    # Ein gefilterter oder auf einzelne Ordner eingeschraenkter Lauf ergaebe eine
    # Tabelle, die vollstaendig AUSSIEHT, aber nur einen Ausschnitt zeigt - und die
    # den vorherigen, vollstaendigen Stand ueberschreibt. Solche Laeufe schreiben
    # deshalb nichts und sagen das auch.
    if not _FULL_RUN:
        print("\n[FORMULATION_REPORT.txt not written: this was a filtered run. "
              "Run 'pytest testfiles/mortar_tests' without -k or a path to regenerate it.]")
        return
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "FORMULATION_REPORT.txt")

    try:
        commit = subprocess.run(
            ["git", "-C", here, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or "unknown"
    except Exception:
        commit = "unknown"

    shared, per_mode = {}, {}
    for nodeid, entry in _RESULTS.items():
        (shared if "all" in entry["modes"] else per_mode)[nodeid] = entry

    def short(nodeid):
        f, _, t = nodeid.partition("::")
        return f"{os.path.basename(os.path.dirname(f))}::{t}"

    lines = []
    W = 92
    lines.append("=" * W)
    lines.append("MORTAR CONTACT - VERIFICATION MATRIX")
    lines.append("=" * W)
    lines.append(f"generated : {datetime.datetime.now():%Y-%m-%d %H:%M}")
    lines.append(f"commit    : {commit}")
    lines.append("")
    lines.append("Three formulations of the SAME discrete constraint g_A = 0. They share the whole")
    lines.append("geometric pipeline and differ only in how the constraint is enforced:")
    lines.append("")
    lines.append("  lagrange       saddle point, multipliers are unknowns of the global system")
    lines.append("                 (Popp et al. 2009/2012; Gitterle et al. 2010)")
    lines.append("  penalty        t_A = kappa * g_A")
    lines.append("                 (Puso & Laursen 2004 Eq. (24); Puso, Laursen & Solberg 2008 Eq. (7))")
    lines.append("  penalty_uzawa  the same inside the augmented Lagrangian loop,")
    lines.append("                 p^(k+1) = p^k + kappa * g_A (Puso et al. 2008 Eq. (18))")
    lines.append("")

    if shared:
        lines.append("-" * W)
        lines.append("A. FORMULATION-INDEPENDENT (geometry and algebra; run once)")
        lines.append("-" * W)
        lines.append(f"{'test':<74}{'result':>8}")
        for nodeid in sorted(shared):
            lines.append(f"{short(nodeid):<74}{shared[nodeid]['modes']['all']:>8}")
        lines.append("")

    if per_mode:
        lines.append("-" * W)
        lines.append("B. FORMULATION-DEPENDENT (run in every formulation)")
        lines.append("-" * W)
        NAMEW = 60
        lines.append(f"{'test':<{NAMEW}}{'lagrange':>10}{'penalty':>10}{'pen.+uzawa':>12}")
        skips, fails = [], []
        for nodeid in sorted(per_mode):
            m = per_mode[nodeid]["modes"]
            name = short(nodeid)
            if len(name) > NAMEW - 1:
                name = name[: NAMEW - 4] + "..."
            row = f"{name:<{NAMEW}}"
            for f in FORMULATIONS:
                width = 10 if f != "penalty_uzawa" else 12
                row += f"{m.get(f, '-'):>{width}}"
            lines.append(row)
            if per_mode[nodeid]["note"]:
                skips.append(f"  {short(nodeid)}\n      {_wrap(per_mode[nodeid]['note'])}")
            for f in FORMULATIONS:
                if m.get(f) == "FAIL":
                    why = KNOWN_OPEN.get((short(nodeid), f), "not yet analysed")
                    fails.append(f"  {short(nodeid)}  [{f}]\n      {_wrap(why)}")
        lines.append("")
        if fails:
            lines.append("OPEN POINTS (what the failures mean):")
            lines.extend(fails)
            lines.append("")
        if skips:
            lines.append("Why tests are skipped:")
            lines.extend(skips)
            lines.append("")

    total = sum(len(e["modes"]) for e in _RESULTS.values())
    failed = sum(1 for e in _RESULTS.values() for o in e["modes"].values() if o == "FAIL")
    skipped = sum(1 for e in _RESULTS.values() for o in e["modes"].values() if o == "SKIP")
    lines.append("-" * W)
    lines.append(f"SUMMARY: {total - failed - skipped} passed, {failed} failed, {skipped} skipped "
                 f"({total} checks over {len(_RESULTS)} tests)")
    lines.append("-" * W)
    lines.append("")
    lines.append("A SKIP is a deliberate statement that the test has no meaning in that")
    lines.append("formulation - never a test that was not run for technical reasons.")
    lines.append("Regenerate with:  pytest testfiles/mortar_tests")

    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
