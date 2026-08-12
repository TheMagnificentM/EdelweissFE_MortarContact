#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_comparison.py -- Kraft-Weg-Kurven des Ausziehversuchs nach Dejori:
beide Diskretisierungen gegen die drei Laborversuche.

Im Unterschied zu den ordnerweisen `plot_results.py` (je eine Simulationskurve)
zeigt dieses Bild BEIDE Simulationen in einem Diagramm:

  * hex8  : GC3D8   + CONQUAD4, aus `hex8/`  -- penalty = 920
  * hex20 : GC3D20R + CONQUAD8, aus `hex20/` -- penalty = 400

Beide Laeufe wurden zeitgleich gestartet und gehoeren damit zum selben
Codestand.

Die unterschiedlichen Federwerte sind kein Tippfehler, sondern notwendig, damit
beide Modelle DIESELBE Gesamt-Auflagerbettung haben: `directionalspringpenalty`
setzt eine unabhaengige Feder der Steifigkeit `penalty` auf jeden KNOTEN des
nSet (`constraints/directionalspringpenalty.py:117-123`) -- ohne Flaechenbezug.
Die Auflagerflaechen haben in hex8 10 + 10, in hex20 23 + 23 Knoten (die
Kantenmittelknoten kommen hinzu), also

    k_ges = penalty * n_Knoten  ->  920 * 10 = 400 * 23 = 9200 N/mm je Auflager.

Aufruf (Umgebung next_v26.11, benoetigt pandas + odfpy):
  python3 plot_comparison.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

HERE = os.path.dirname(os.path.abspath(__file__))

# Simulationen: (Ordner, Beschriftung, Farbe, Federwert)
SIMULATIONS = [
    ("hex8", "FEM hex8 (GC3D8 / CONQUAD4)", "#2563eb", 920),
    ("hex20", "FEM hex20 (GC3D20R / CONQUAD8)", "#16a34a", 400),
]

# Vorgegebene Endverschiebung des Laststeps (.inp: dirichlet disp, 2=1.1).
# Wer sie erreicht, ist durchgelaufen; wer darunter endet, ist abgebrochen.
U_PRESCRIBED = 1.1

EXPERIMENT_ODS = os.path.join(HERE, "hex20", "0_VA1_EBT6cm.ods")


def load_simulation(folder):
    """disp.csv / force.csv eines Laufs. Spalte 0 = Lastfaktor, Spalte 1 = Wert.

    Der Skalierungsfaktor 20 (4-mm-Scheibe -> 80 mm Referenztiefe) steckt bereits
    im fieldOutput der .inp-Datei; hier wird nur N -> kN umgerechnet.
    """
    u = np.loadtxt(os.path.join(HERE, folder, "disp.csv"))[:, 1]
    f = -np.loadtxt(os.path.join(HERE, folder, "force.csv"))[:, 1] / 1000.0
    return u, f


def load_experiments():
    """Drei Versuchskurven aus der ODS (Zeile 1 = X/Y-Header)."""
    df = pd.read_excel(EXPERIMENT_ODS, engine="odf", header=1)
    df.columns = ["x1", "y1", "x2", "y2", "x3", "y3"]
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    return [(df[f"x{i}"].values, df[f"y{i}"].values, f"Versuch {i}") for i in (1, 2, 3)]


def secant_stiffness(u, f, u_max=0.03):
    """Sekantensteifigkeit im nahezu elastischen Anfangsbereich, Ausgleichsgerade
    durch den Ursprung (kN/mm)."""
    m = (u > 0.0) & (u <= u_max)
    if m.sum() < 2:
        return float("nan")
    return float(np.sum(f[m] * u[m]) / np.sum(u[m] ** 2))


def main():
    experiments = load_experiments()

    # Flaches Format: so passt die Abbildung in der Doku zusammen mit Tabelle und
    # Text auf eine Seite, statt als Float allein auf einer Seite zu landen.
    fig, ax = plt.subplots(figsize=(9, 4.8))

    # --- Experiment: Streuband der drei Versuche ---
    all_x = np.unique(np.concatenate([xd for xd, _, _ in experiments]))
    stacked = np.vstack(
        [interp1d(xd, yd, bounds_error=False, fill_value=np.nan)(all_x) for xd, yd, _ in experiments]
    )
    valid = ~np.any(np.isnan(stacked), axis=0)
    ax.fill_between(
        all_x[valid],
        np.nanmin(stacked, axis=0)[valid],
        np.nanmax(stacked, axis=0)[valid],
        alpha=0.22,
        color="#f97316",
        label="Experiment (Streuband)",
        zorder=1,
    )
    for (xd, yd, label), col in zip(experiments, ["#f97316", "#ea580c", "#c2410c"]):
        ax.plot(xd, yd, color=col, linewidth=1.1, alpha=0.75, label=label, zorder=2)

    # --- Simulationen ---
    summary = []
    aborted = False
    for folder, label, color, penalty in SIMULATIONS:
        u, f = load_simulation(folder)
        ax.plot(u, f, color=color, linewidth=2.3, label=label, zorder=5)
        # Endpunkt nur dann markieren, wenn die Rechnung VOR der vorgegebenen
        # Verschiebung endet -- das Kreuz steht fuer den Abbruch, nicht fuer das
        # regulaere Lastende (Abbruchgrund im Text der Doku).
        if u[-1] < 0.999 * U_PRESCRIBED:
            aborted = True
            ax.plot(u[-1], f[-1], marker="x", markersize=9, markeredgewidth=2.0, color=color, zorder=6)
        i = int(np.argmax(f))
        summary.append((label, penalty, secant_stiffness(u, f), f[i], u[i], u[-1], len(u)))

    ax.set_xlabel("Verschiebung $u$ [mm]", fontsize=13)
    ax.set_ylabel("Ausziehkraft $F$ [kN]", fontsize=13)
    ax.set_title(
        "Ausziehversuch nach Dejori (VA1, EBT 6 cm)\n"
        "Mortar-Kontakt: lineare gegen quadratische Diskretisierung",
        fontsize=12,
        pad=12,
    )
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_xlim(0.0, U_PRESCRIBED + 0.05)
    ax.set_ylim(bottom=0.0)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9, edgecolor="#cbd5e1")
    if aborted:
        ax.text(
            0.985,
            0.02,
            r"$\times$ = Abbruch der Rechnung",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="#475569",
        )

    plt.tight_layout()
    out_pdf = os.path.join(HERE, "force_disp_hex8_vs_hex20.pdf")
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"Gespeichert: {out_pdf}")

    # Auch neben die Doku legen, damit \includegraphics dort ohne Pfadakrobatik greift.
    doku_pdf = os.path.abspath(os.path.join(HERE, "..", "Doku", "force_disp_hex8_vs_hex20.pdf"))
    plt.savefig(doku_pdf, bbox_inches="tight")
    print(f"Gespeichert: {doku_pdf}")

    print("\n--- Kennwerte ---")
    print(f"{'Kurve':34s} {'penalty':>7s} {'k_sec':>10s} {'F_max':>8s} {'@u':>7s} {'u_Ende':>7s} {'Inkr.':>6s}")
    for label, penalty, k, fmax, uat, uend, n in summary:
        print(f"{label:34s} {penalty:7d} {k:7.1f} kN/mm {fmax:6.2f} kN {uat:6.3f} {uend:7.3f} {n:6d}")
    for xd, yd, label in experiments:
        print(f"{label:34s} {'-':>7s} {'-':>10s} {yd.max():6.2f} kN {xd[yd.argmax()]:6.3f}")


if __name__ == "__main__":
    main()
