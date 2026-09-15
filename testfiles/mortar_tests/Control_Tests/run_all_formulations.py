#!/usr/bin/env python3
"""Fahrt die gesamte Control-Test-Reihe in ALLEN drei Kontaktformulierungen und
stellt die Ergebnisse gegeneinander.

Aufruf:   python run_all_formulations.py

Erzeugt im Ordner Control_Tests:
  * final_report_<formulierung>.txt   - die vollstaendige Ausgabe je Lauf
  * final_summary<...>.json           - die Kennzahlen maschinenlesbar
  * final_gp_table<...>.csv/.txt      - Gausspunkt-Tabelle je Lauf
  * final_contact_table<...>.csv      - Kontaktdruecke je Slave-Knoten
  * CONTROL_TESTS_MATRIX.txt          - die Vergleichstabelle (englisch)

Jede Formulierung laeuft in einem EIGENEN Prozess. Das ist nicht nur sauberer,
sondern noetig: `*updateConfiguration` wirkt prozessweit (die Konfigurations-
Dicts werden per Referenz durchgereicht), und eine in einem Lauf gelockerte
Toleranz wuerde sonst in den naechsten durchschlagen.
"""

import json
import os
import subprocess
import sys
from datetime import datetime

THISDIR = os.path.abspath(os.path.dirname(__file__))
FORMULATIONS = ("lagrange", "penalty", "penalty_uzawa")

#: Reihenfolge der Laeufe - bewusst NICHT die der Tabelle. Die Auswertedateien IN
#: den Testordnern (stress_gp.csv, gp_report_*.csv, ...) stammen aus den
#: `export=`-Anweisungen der jeweiligen .inp und tragen deshalb keine Endung je
#: Formulierung: der letzte Lauf gewinnt. Damit die versionierten Dateien am Ende
#: den Sattelpunkt enthalten - den Stand, auf den sich die Dokumentation bezieht -,
#: laeuft `lagrange` zuletzt. Die Dateien im Control_Tests-Ordner selbst
#: (final_gp_table, final_contact_table, final_summary) sind dagegen je
#: Formulierung benannt und bleiben alle erhalten.
RUN_ORDER = ("penalty", "penalty_uzawa", "lagrange")

#: Wie genau eine Formulierung den Sattelpunkt reproduzieren KANN. Der Sattelpunkt
#: erfuellt g_A = 0 als Gleichung, das augmentierte Verfahren bis zu seiner
#: Augmentierungstoleranz, reines Penalty laesst O(1/kappa) stehen - das ist die
#: Eigenschaft des Verfahrens und kein Mangel. Vergleichsmassstab ist deshalb nicht
#: eine feste Zahl, sondern die der Formulierung angemessene Schranke.
TOLERANCE = {"lagrange": 0.0, "penalty_uzawa": 1.0e-5, "penalty": 5.0e-2}


def suffix(f):
    return "" if f == "lagrange" else f"_{f}"


def run(formulation):
    out = os.path.join(THISDIR, f"final_report_{formulation}.txt")
    print(f"  running {formulation} ...", flush=True)
    env = dict(os.environ, PYTHON_GIL="0")
    with open(out, "w") as fh:
        rc = subprocess.call([sys.executable, os.path.join(THISDIR, "final_report.py"),
                              "--formulation", formulation],
                             stdout=fh, stderr=subprocess.STDOUT, cwd=THISDIR, env=env)
    print(f"    -> {os.path.basename(out)} (exit {rc})", flush=True)
    path = os.path.join(THISDIR, f"final_summary{suffix(formulation)}.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def variant_key(r):
    return f"{r.get('test', '')}|{r.get('elem', r.get('name', ''))}|{r.get('mesh', '')}"


#: Je Familie die Kennzahl, an der die Formulierungen verglichen werden, und ihr
#: Sollwert. Genommen wird jeweils die Groesse, die der Testfall selbst als Urteil
#: fuehrt - nicht eine neu erfundene.
METRICS = {
    "compression": ("l2_anal", "rel. L2 displacement error", 0.0),
    "separation": ("max_lam", "max|lambda| - the contact must RELEASE under tension", 0.0),
    "sliding": ("max_shear", "max|shear stress| - frictionless, must vanish", 0.0),
    "inclined": ("rel_l2", "rel. L2 displacement error", 0.0),
    "hertz": ("p0_num", "peak contact pressure", None),
}


def main():
    print("=" * 100)
    print("CONTROL TESTS IN ALL CONTACT FORMULATIONS")
    print("=" * 100)
    reuse = "--reuse" in sys.argv
    if reuse:
        print("  reusing the summaries of the previous run (no jobs recomputed)")
        data = {}
        for f in FORMULATIONS:
            path = os.path.join(THISDIR, f"final_summary{suffix(f)}.json")
            data[f] = json.load(open(path)) if os.path.exists(path) else None
    else:
        results = {f: run(f) for f in RUN_ORDER}
        data = {f: results[f] for f in FORMULATIONS}

    missing = [f for f, d in data.items() if d is None]
    lines = []
    W = 100
    lines.append("=" * W)
    lines.append("CONTROL TESTS - CONTACT FORMULATION MATRIX")
    lines.append("=" * W)
    lines.append(f"generated : {datetime.now():%Y-%m-%d %H:%M}")
    lines.append("")
    lines.append("The two-block reference series and the Hertz benchmark, run in every formulation.")
    lines.append("The saddle point is the REFERENCE: it enforces g_A = 0 as an equation of the system.")
    lines.append("A penalty column is judged by how closely it reproduces that reference, within the")
    lines.append("accuracy its own formulation can deliver:")
    for f in FORMULATIONS:
        t = TOLERANCE[f]
        lines.append(f"   {f:<14} tolerance {'exact (reference)' if t == 0 else f'{t:.0e} relative'}")
    lines.append("")
    if missing:
        lines.append(f"!! NO DATA for: {', '.join(missing)} - see final_report_<name>.txt")
        lines.append("")

    ref = data.get("lagrange")
    for family, (key, label, _) in METRICS.items():
        if not ref or family not in ref:
            continue
        lines.append("-" * W)
        lines.append(f"{family.upper()}  -  {label}")
        lines.append("-" * W)
        head = f"{'case':<34}" + "".join(f"{f:>21}" for f in FORMULATIONS) + "  verdict"
        lines.append(head)
        by_key = {f: {variant_key(r): r for r in (data[f][family] if data[f] else [])}
                  for f in FORMULATIONS}
        for r in ref[family]:
            k = variant_key(r)
            row = f"{k.replace('|', ' ')[:33]:<34}"
            vals, verdict = {}, "ok"
            for f in FORMULATIONS:
                rr = by_key[f].get(k)
                if rr is None:
                    row += f"{'-':>21}"
                    verdict = "MISSING"
                    continue
                if not rr.get("converged", False):
                    row += f"{'NOT CONVERGED':>21}"
                    verdict = "NOT CONVERGED"
                    continue
                v = rr.get(key)
                vals[f] = v
                row += f"{(f'{v:.3e}' if isinstance(v, (int, float)) else str(v)):>21}"
            base = vals.get("lagrange")
            if verdict == "ok" and isinstance(base, (int, float)):
                scale = max(abs(base), 1.0)
                off = [f for f in ("penalty", "penalty_uzawa")
                       if isinstance(vals.get(f), (int, float))
                       and abs(vals[f] - base) / scale > TOLERANCE[f]]
                if off:
                    verdict = "DEVIATES: " + ", ".join(off)
            lines.append(row + f"  {verdict}")
        lines.append("")

    lines.append("-" * W)
    lines.append("A 'DEVIATES' row is not automatically a defect - it is a place where the formulations")
    lines.append("genuinely disagree and which therefore needs an explanation. See the per-run reports")
    lines.append("final_report_<formulation>.txt for the full numbers behind each column.")
    lines.append("-" * W)

    path = os.path.join(THISDIR, "CONTROL_TESTS_MATRIX.txt")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
