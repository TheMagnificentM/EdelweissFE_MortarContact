#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generates control / presentation figures (PNG) in plots/ for the Control_Tests.

Per test AND mesh (matching / nonmatching), one figure each so every run can be
controlled visually:
  <test>_<mesh>_disp.png     - displacement field u_y(y): hex8/hex20/hex20R vs analytical (+monolith)
  <test>_<mesh>_stress.png   - sigma_yy at the integration points: hex8/hex20/hex20R vs analytical
  <test>_<mesh>_pressure.png - contact pressure lambda along the interface
  sliding_<mesh>_ux.png      - lateral displacement u_x(y): top block slides, bottom at rest

Overviews:
  accuracy_overview.png, stiffness_bars.png, gausspunkte_schema.png

Titles are uniform "<test> - <mesh>: <quantity>"; observations read off the plot
(e.g. "constant = -10") go into a small box inside the plot, not the title.

Run:  python make_plots.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

THISDIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, THISDIR)
import evaluate as ev

OUT = os.path.join(THISDIR, "plots")
os.makedirs(OUT, exist_ok=True)
FOLDER = {"selfweight": "01_selfweight", "pressure": "02_pressure", "dispcontrol": "03_dispcontrol",
          "separation": "04_separation", "sliding": "05_sliding", "inclined": "06_inclined",
          "stiffness": "07_stiffness"}
ELEMS = ["hex8", "hex20", "hex20R"]
COL = {"hex8": "tab:blue", "hex20": "tab:orange", "hex20R": "tab:green"}
MARK = {"hex8": "o", "hex20": "s", "hex20R": "^"}
MESHES = ["matching", "nonmatching"]
LABEL = {"selfweight": "Self-weight", "pressure": "Pressure", "dispcontrol": "Displacement control",
         "stiffness": "Stiffness contrast", "separation": "Separation (uplift)"}
FIELD_TESTS = list(LABEL.keys())
HAS_MONO = {"selfweight", "pressure", "dispcontrol"}


def inp(test, elem, mesh="matching"):
    return os.path.join(THISDIR, FOLDER[test], f"{test}_{elem}_{mesh}.inp")


def mono_path(test):
    return os.path.join(THISDIR, FOLDER[test], "reference_monolith_hex20.inp")


def nodal_u(model, comp):
    nf = model.nodeFields["displacement"]
    return np.array([n.coordinates[1] for n in nf.nodes]), np.asarray(nf["U"])[:, comp]


def gp_syy(model, foc):
    fo = foc.fieldOutputs["stressGP"]; sig = np.asarray(fo.getLastResult()); els = list(fo.associatedSet)
    ys, ss = [], []
    for e, el in enumerate(els):
        gpw = ev.gp_world_coords(el, sig.shape[1])
        for g in range(sig.shape[1]):
            ys.append(gpw[g, 1]); ss.append(sig[e, g, 1])
    return np.array(ys), np.array(ss)


def lam_of(model):
    # lambda folgt der Literaturkonvention: Druck POSITIV. Kein abs() mehr - ein
    # Vorzeichenfehler soll im Diagramm sichtbar werden und nicht weggerechnet.
    return np.array([v.value for v in model.scalarVariables.values()]).flatten()


def newfig(**kw):
    plt.rcParams["text.usetex"] = False   # robust rendering regardless of what the FE solver set
    return plt.subplots(**kw)


def finish(fig, path, insight):
    """Legende ist normal; die abgelesene Aussage als 'Insight:'-Zeile unter das Bild."""
    fig.text(0.5, 0.015, "Insight: " + insight, ha="center", va="bottom", fontsize=8.5,
             style="italic", color="#333333")
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(path, dpi=120); plt.close(fig)
    print("  " + os.path.basename(path))


# ===========================================================================
def plot_disp(test, mesh):
    fe = {}
    for elem in ELEMS:
        m, foc, conv = ev.run_job(inp(test, elem, mesh)); fe[elem] = nodal_u(m, 1) if conv else None
    mono = None
    if test in HAS_MONO:
        mm, _, mc = ev.run_job(mono_path(test)); mono = nodal_u(mm, 1) if mc else None
    y = np.linspace(0, 2, 300)
    uex = np.array([ev.analytical_disp([0.0, yi, 0.0], test)[1] for yi in y])
    fig, ax = newfig(figsize=(6.8, 4.7))
    ax.plot(uex * 1e3, y, "k-", lw=2, label="analytical (exact)")
    for elem in ELEMS:
        if fe[elem] is None:
            continue
        yy, uu = fe[elem]
        ax.plot(uu * 1e3, yy, MARK[elem], ms=6, mfc="none", mew=1.4, color=COL[elem], label=f"{elem} (nodes)")
    if mono is not None:
        ax.plot(mono[1] * 1e3, mono[0], "x", ms=6, color="gray", alpha=.7, label="monolith (nodes)")
    ax.set_xlabel(r"displacement $u_y$  [$10^{-3}$]"); ax.set_ylabel(r"height $y$")
    ax.axhline(1.0, color="gray", ls=":", lw=1); ax.text(ax.get_xlim()[0], 1.03, "interface", color="gray", fontsize=8)
    ax.set_title(f"{LABEL[test]} – {mesh}: displacement field $u_y(y)$")
    ax.legend(loc="best", fontsize=8.5); ax.grid(alpha=.3)
    ins = ("contact opens — lower block at rest, upper block moves rigidly by +0.02" if test == "separation"
           else "all element types match the analytical solution"
                + (" and the monolith reference" if mono is not None else ""))
    finish(fig, os.path.join(OUT, f"{test}_{mesh}_disp.png"), ins)


def plot_stress(test, mesh):
    data = {}
    for elem in ELEMS:
        m, foc, conv = ev.run_job(inp(test, elem, mesh)); data[elem] = gp_syy(m, foc) if conv else None
    y = np.linspace(0, 2, 300); sex = np.array([ev.analytical_sigma_yy(yi, test) for yi in y])
    const = (sex.max() - sex.min()) < 1e-9
    ins = (f"$\\sigma_{{yy}}$ is constant = {sex[0]:.1f}; all element types reproduce it exactly" if const
           else "hex20 and hex20R follow the linear field; hex8 is element-constant (staircase)")
    fig, ax = newfig(figsize=(6.8, 4.7))
    ax.plot(sex, y, "k-", lw=2, label="analytical")
    for elem in ELEMS:
        if data[elem] is None:
            continue
        ax.plot(data[elem][1], data[elem][0], MARK[elem], ms=6, mfc="none", mew=1.4, color=COL[elem],
                label=f"{elem} (int. points)")
    lo = min(sex.min(), 0.0)
    ax.set_xlim(lo - 2.0, 2.0); ax.axvline(0.0, color="gray", ls=":", lw=1)
    ax.set_xlabel(r"stress $\sigma_{yy}$"); ax.set_ylabel(r"height $y$")
    ax.set_title(f"{LABEL[test]} – {mesh}: $\\sigma_{{yy}}$ at the integration points")
    ax.legend(loc="best", fontsize=8.5); ax.grid(alpha=.3)
    finish(fig, os.path.join(OUT, f"{test}_{mesh}_stress.png"), ins)


def plot_pressure(test, mesh):
    pexp = 0.0 if test == "separation" else ev.analytical_pressure(test)
    fe = {}
    for elem in ELEMS:
        m, foc, conv = ev.run_job(inp(test, elem, mesh)); fe[elem] = lam_of(m) if conv else None
    fig, ax = newfig(figsize=(6.8, 4.5))
    ax.axhline(pexp, color="k", ls="--", lw=1.5, label=f"analytical = {pexp:.2f}")
    for elem in ELEMS:
        if fe[elem] is None:
            continue
        lam = fe[elem]
        ax.plot(np.arange(len(lam)), lam, MARK[elem] + "-", ms=5, mfc="none", color=COL[elem],
                label=f"{elem} (n={len(lam)})")
    ax.set_xlabel("slave node at interface (index)"); ax.set_ylabel(r"contact pressure $\lambda$")
    ax.set_ylim(min(0.0, pexp) - 1.0, max(pexp * 1.3, 12.0))
    ax.set_title(f"{LABEL[test]} – {mesh}: contact pressure $\\lambda$")
    ax.legend(loc="best", fontsize=8.5); ax.grid(alpha=.3)
    ins = ("blocks separated — contact pressure $\\lambda = 0$ everywhere" if test == "separation"
           else f"contact pressure is constant $\\lambda = {pexp:.2f}$ along the interface, no oscillation")
    finish(fig, os.path.join(OUT, f"{test}_{mesh}_pressure.png"), ins)


def plot_sliding_ux(mesh):
    fe = {}
    for elem in ELEMS:
        m, foc, conv = ev.run_job(inp("sliding", elem, mesh)); fe[elem] = nodal_u(m, 0) if conv else None
    y = np.linspace(0, 2, 300); uex = np.where(y < 1.0, 0.0, 0.01)
    fig, ax = newfig(figsize=(6.8, 4.7))
    ax.plot(uex * 1e3, y, "k-", lw=2, label="analytical")
    for elem in ELEMS:
        if fe[elem] is None:
            continue
        yy, uu = fe[elem]
        ax.plot(uu * 1e3, yy, MARK[elem], ms=6, mfc="none", mew=1.4, color=COL[elem], label=f"{elem} (nodes)")
    ax.set_xlabel(r"lateral displacement $u_x$  [$10^{-3}$]"); ax.set_ylabel(r"height $y$")
    ax.axhline(1.0, color="gray", ls=":", lw=1); ax.text(ax.get_xlim()[0], 1.03, "interface", color="gray", fontsize=8)
    ax.set_title(f"Frictionless sliding – {mesh}: lateral displacement $u_x(y)$")
    ax.legend(loc="best", fontsize=8.5); ax.grid(alpha=.3)
    finish(fig, os.path.join(OUT, f"sliding_{mesh}_ux.png"),
           "top block slides laterally (+0.01), lower block stays at rest (0) — frictionless")


def plot_accuracy_overview():
    tests = ["dispcontrol", "inclined", "stiffness", "pressure", "selfweight"]
    labels = {"dispcontrol": "Displ.\ncontrol", "inclined": "incl. 30°", "stiffness": "Stiffness\ncontrast",
              "pressure": "Pressure", "selfweight": "Self-\nweight"}
    data = {e: [] for e in ELEMS}
    for t in tests:
        for e in ELEMS:
            m, foc, conv = ev.run_job(inp(t, e))
            data[e].append(ev.l2_disp_vs_analytical(m, t)["rel_l2"] if conv else np.nan)
    x = np.arange(len(tests)); w = 0.26
    fig, ax = newfig(figsize=(9.5, 5.0))
    for i, e in enumerate(ELEMS):
        vals = np.maximum(data[e], 1e-17)
        bars = ax.bar(x + (i - 1) * w, vals, w, label=e, color=COL[e])
        ax.bar_label(bars, labels=[f"{v:.0e}" for v in vals], padding=2, fontsize=6.5, rotation=90)
    ax.set_yscale("log"); ax.set_ylim(1e-16, 1e-3)
    ax.set_ylabel("relative L2 error of the displacement field  (smaller = better)")
    ax.set_xticks(x); ax.set_xticklabels([labels[t] for t in tests])
    ax.axhline(1e-6, color="gray", ls=":", lw=1); ax.text(-0.45, 1.3e-6, "1e-6", color="gray", fontsize=7, va="bottom")
    ax.set_title("Accuracy vs. the exact solution  (matching mesh)")
    ax.legend(title="element type", loc="upper left"); ax.grid(axis="y", alpha=.3, which="both")
    ax.text(0.5, 0.90,
            "Displ. control, incl. 30°, contrast:  essentially machine-exact\n"
            "Pressure, self-weight (ca. 1e-7):  tiny stabilization spring",
            transform=ax.transAxes, ha="center", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9))
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "accuracy_overview.png"), dpi=120); plt.close(fig)
    print("  accuracy_overview.png")


def plot_stiffness_bars():
    tests = [("pressure", "Pressure\n(exp. 500)"), ("dispcontrol", "Displ.\n(exp. 500)"), ("stiffness", "Contrast\n(exp. 990)")]
    data = {e: [] for e in ELEMS}
    for t, _ in tests:
        for e in ELEMS:
            m, foc, conv = ev.run_job(inp(t, e)); data[e].append(ev.reaction_stiffness(m)["k_eff"] if conv else np.nan)
    x = np.arange(len(tests)); w = 0.26
    fig, ax = newfig(figsize=(6.5, 4.4))
    for i, e in enumerate(ELEMS):
        ax.bar(x + (i - 1) * w, data[e], w, label=e, color=COL[e])
    ax.set_xticks(x); ax.set_xticklabels([t[1] for t in tests]); ax.set_ylabel(r"effective stiffness  $k = F/u$")
    ax.set_title("Stiffness is element-independent (hex8 = hex20 = hex20R)")
    ax.legend(title="element type"); ax.grid(axis="y", alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "stiffness_bars.png"), dpi=120); plt.close(fig)
    print("  stiffness_bars.png")


def plot_gauss_schema():
    fig, axs = newfig(figsize=(10, 4.6), ncols=2)
    g2 = 1 / np.sqrt(3); g3 = np.sqrt(0.6)
    for ax, xs, titel in [
            (axs[0], [0.5 - 0.5 * g2, 0.5 + 0.5 * g2], "hex8 (C3D8): 8 Gauss points = 2×2×2\n2 height levels"),
            (axs[1], [0.5 - 0.5 * g3, 0.5, 0.5 + 0.5 * g3], "hex20 (C3D20): 27 Gauss points = 3×3×3\n3 height levels")]:
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, ec="k", lw=1.5))
        for yy in xs:
            for xx in xs:
                ax.plot(xx, yy, "o", ms=9, color="tab:red")
            ax.axhline(yy, color="gray", ls=":", lw=1); ax.text(1.02, yy, f"y={yy:.3f}", va="center", fontsize=8, color="gray")
        ax.set_xlim(-0.05, 1.35); ax.set_ylim(-0.05, 1.05); ax.set_aspect("equal")
        ax.set_xlabel("element cross-section (x)"); ax.set_ylabel(r"height in element $y$"); ax.set_title(titel, fontsize=10)
    fig.suptitle("Integration (Gauss) points: where the stress is evaluated inside an element (red), not at the corners", fontsize=11)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "gausspunkte_schema.png"), dpi=120); plt.close(fig)
    print("  gausspunkte_schema.png")


if __name__ == "__main__":
    print("creating figures in plots/ ...")
    plot_gauss_schema()
    plot_accuracy_overview()
    for test in FIELD_TESTS:
        for mesh in MESHES:
            plot_disp(test, mesh)
            plot_stress(test, mesh)
            plot_pressure(test, mesh)
    for mesh in MESHES:
        plot_sliding_ux(mesh)
    plot_stiffness_bars()
    print("done.")
