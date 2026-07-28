#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_results.py — Kraft-Weg-Kurve fuer den POT Dejori Ausziehversuch

Plottet:
  1. EdelweissFE-Simulation (MortarContact3D, linear-elastisch)    aus disp.csv / force.csv
  2. Experimentelle Daten (3 Versuche)                              aus 0_VA1_EBT6cm.ods

Aufruf:
  python3 plot_results.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

current_dir = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# 1. SIMULATIONSDATEN
# =============================================================================
disp_data  = np.loadtxt(os.path.join(current_dir, "disp_friction.csv"))
force_data = np.loadtxt(os.path.join(current_dir, "force_friction.csv"))

# Spalte 1 = Lastfaktor, Spalte 2 = Wert
sim_disp_mm  =  disp_data[:, 1]                   # Y-Verschiebung [mm]
# *20 Skalierungsfaktor: 4mm-Scheibe -> 80mm Referenztiefe (wie Konstantin) (ist schon im inp file)
sim_force_kN = -force_data[:, 1] * 1 / 1000.0    # [kN]

# Berechnet das absolute Maximum der Kraft (wahrer F_max) und nicht den Endwert
sim_f_max     = sim_force_kN.max()
sim_u_at_fmax = sim_disp_mm[sim_force_kN.argmax()]

# =============================================================================
# 2. EXPERIMENTELLE DATEN (ODS)
# =============================================================================
# Zeile 0 = Headings (Dataset 1 / Dataset 2 ...), Zeile 1 = X/Y
# header=1 → zweite Zeile als Spaltenheader (X, Y, X, Y, X, Y)
df_raw = pd.read_excel(
    os.path.join(current_dir, "0_VA1_EBT6cm.ods"),
    engine="odf",
    header=1,          # "X", "Y" als Spaltenheader
)
df_raw.columns = ["x1", "y1", "x2", "y2", "x3", "y3"]
df = df_raw.apply(pd.to_numeric, errors="coerce").dropna()

# X: Meter → mm  |  Y: kN (bleibt)
exp_datasets = [
    (df["x1"].values * 1.0, df["y1"].values, "Versuch 1"),
    (df["x2"].values * 1.0, df["y2"].values, "Versuch 2"),
    (df["x3"].values * 1.0, df["y3"].values, "Versuch 3"),
]

# =============================================================================
# 3. PLOT
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 6))

# --- Experiment: Einhüllende (gefüllter Bereich) ---
all_x_exp = np.union1d(
    np.union1d(exp_datasets[0][0], exp_datasets[1][0]),
    exp_datasets[2][0],
)
# Interpoliere alle Kurven auf gemeinsames x-Raster
from scipy.interpolate import interp1d

interp_curves = []
for xd, yd, _ in exp_datasets:
    # Nur Bereich bis zum jeweiligen Maximum interpolieren
    f = interp1d(xd, yd, bounds_error=False, fill_value=np.nan)
    interp_curves.append(f(all_x_exp))

stacked = np.vstack(interp_curves)
y_min = np.nanmin(stacked, axis=0)
y_max = np.nanmax(stacked, axis=0)

# Maske: nur dort wo alle drei Kurven definiert sind
valid = ~np.any(np.isnan(stacked), axis=0)
ax.fill_between(
    all_x_exp[valid], y_min[valid], y_max[valid],
    alpha=0.25, color="#f97316", label="Experiment (Spannweite)",
)

# --- Experiment: Einzelkurven ---
exp_colors = ["#f97316", "#ea580c", "#c2410c"]
for (xd, yd, label), col in zip(exp_datasets, exp_colors):
    ax.plot(xd, yd, color=col, linewidth=1.2, alpha=0.7, label=label)

# --- Simulation ---
ax.plot(
    sim_disp_mm, sim_force_kN,
    color="#2563EB", linewidth=2.2,
    label="FEM – MortarContact",
    zorder=5,
)

# =============================================================================
# 4. FORMATIERUNG
# =============================================================================
ax.set_xlabel("Displacement [mm]", fontsize=13)
ax.set_ylabel("Pull-out Force [kN]",  fontsize=13)
ax.set_title(
    "VA1 – Pull-Out Test: EBT 6cm\n"
    "MortarContact3D vs. Experiment",
    fontsize=12, pad=12,
)
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

# Legende: Experiment-Einträge zusammenfassen
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize=10, loc="upper right",
          framealpha=0.9, edgecolor="#cbd5e1")


plt.tight_layout()

# Nur PDF speichern
pdf_path = os.path.join(current_dir, "force_disp_mortar_contact.pdf")
plt.savefig(pdf_path, bbox_inches="tight")

print(f"Gespeichert: {pdf_path}")
print(f"\nSimulation:  F_max = {sim_f_max:.2f} kN  @ u = {sim_u_at_fmax:.3f} mm")
for xd, yd, label in exp_datasets:
    idx_max = yd.argmax()
    print(f"{label}:  F_max = {yd.max():.2f} kN  @ u = {xd[idx_max]:.2f} mm")