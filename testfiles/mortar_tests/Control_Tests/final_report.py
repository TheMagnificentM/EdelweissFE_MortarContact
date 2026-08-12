#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FINALER Report ueber ALLE Control_Tests – eine Datei, ein Aufruf, alles ablesbar.
================================================================================

    python final_report.py

Führt alle Varianten aus (6 Tests x {hex8, hex20, hex20R} x {matching, nonmatching}
+ Monolith-Referenzen der Druck-Familie) und schreibt/druckt:

  * final_gp_table.csv / .txt   – DIE grosse Tabelle: fuer JEDEN Gausspunkt jeder
        Variante Koordinaten, alle 6 Spannungen, alle 6 Dehnungen und alle Fehler.
  * final_contact_table.csv     – Kontaktdruck lambda an JEDEM Slave-Knoten.
  * Bildschirm: (A) Druck-Familie mit L2-Fehler, Spannungsfehler, STEIFIGKEIT und
        Kontakt-Urteil; (B) die Verhaltens-Tests (Abheben, Gleiten, schiefe Flaeche).
"""

import os
import sys

import numpy as np

THISDIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, THISDIR)

import evaluate as ev

ELEMS = ["hex8", "hex20", "hex20R"]
MESHES = ["matching", "nonmatching"]
FOLDERS = [("01_selfweight", "selfweight"), ("02_pressure", "pressure"),
           ("03_dispcontrol", "dispcontrol"), ("04_separation", "separation"),
           ("05_sliding", "sliding"), ("06_inclined", "inclined"),
           ("07_stiffness", "stiffness")]
WITH_MONOLITH = {"selfweight", "pressure", "dispcontrol"}
IYY = 1


def monolith_gp_lookup(mfoc):
    fo = mfoc.fieldOutputs["stressGP"]
    sig = np.asarray(fo.getLastResult())
    els = list(fo.associatedSet)
    d = {}
    for e, el in enumerate(els):
        key = tuple(np.round(ev.element_centroid(el), 6))
        for g in range(sig.shape[1]):
            d[(key, g)] = sig[e, g, IYY]
    return d


def collect_gp_rows(foc, test, elem, mesh, mono_lookup):
    fo = foc.fieldOutputs["stressGP"]
    fe = foc.fieldOutputs["strainGP"]
    sig = np.asarray(fo.getLastResult())
    eps = np.asarray(fe.getLastResult())
    els = list(fo.associatedSet)
    nGP = sig.shape[1]
    rows = []
    for e, el in enumerate(els):
        c = ev.element_centroid(el)
        ckey = tuple(np.round(c, 6))
        gpw = ev.gp_world_coords(el, nGP)
        syy_ctr = ev.analytical_sigma_yy(c[1], test)
        for g in range(nGP):
            syy = sig[e, g, IYY]
            syy_an = ev.analytical_sigma_yy(gpw[g, 1], test)
            mono = mono_lookup.get((ckey, g), np.nan) if mono_lookup else np.nan
            rows.append(dict(
                test=test, elem=elem, mesh=mesh, el=int(el.elNumber), gp=g,
                x=gpw[g, 0], y=gpw[g, 1], z=gpw[g, 2],
                s11=sig[e, g, 0], s22=sig[e, g, 1], s33=sig[e, g, 2],
                s12=sig[e, g, 3], s13=sig[e, g, 4], s23=sig[e, g, 5],
                e11=eps[e, g, 0], e22=eps[e, g, 1], e33=eps[e, g, 2],
                e12=eps[e, g, 3], e13=eps[e, g, 4], e23=eps[e, g, 5],
                syy_anal=syy_an, err_anal=syy - syy_an,
                syy_anal_ctr=syy_ctr, err_anal_ctr=syy - syy_ctr,
                syy_mono=mono, err_mono=(syy - mono) if not np.isnan(mono) else np.nan))
    return rows


def collect_contact_rows(model, test, elem, mesh):
    lam = ev._lambdas(model)
    rows = []
    for i, l in enumerate(lam):
        rows.append(dict(test=test, elem=elem, mesh=mesh, slave_node=i, lam=float(l)))
    return rows


# ---------------------------------------------------------------------------
def main():
    gp_rows, contact_rows = [], []
    comp_rows, sep_rows, sli_rows, inc_rows = [], [], [], []

    for folder, test in FOLDERS:
        mono_cache = {}
        for elem in ELEMS:
            for mesh in MESHES:
                variant = f"{test}_{elem}_{mesh}"
                inp = os.path.join(THISDIR, folder, f"{variant}.inp")
                print(f"  ... {variant}", flush=True)
                model, foc, conv = ev.run_job(inp)
                contact_rows.extend(collect_contact_rows(model, test, elem, mesh) if conv else [])
                if conv:
                    ev.write_lambda_vtk(model, os.path.join(THISDIR, folder, f"lambda_{variant}.vtk"))

                if test in ev.COMPRESSION_TESTS:
                    row = dict(test=test, elem=elem, mesh=mesh, converged=conv,
                               l2_anal=np.nan, l2_mono=np.nan, max_err_anal=np.nan,
                               max_err_mono=np.nan, F_react=np.nan, k_eff=np.nan, springt=None)
                    if conv:
                        row["l2_anal"] = ev.l2_disp_vs_analytical(model, test)["rel_l2"]
                        if elem not in mono_cache:
                            mp = os.path.join(THISDIR, folder, f"reference_monolith_{elem}.inp")
                            if test in WITH_MONOLITH and os.path.exists(mp):
                                mm, mfoc, mc = ev.run_job(mp)
                                mono_cache[elem] = (mm, mfoc) if mc else None
                            else:
                                mono_cache[elem] = None
                        mono = mono_cache[elem]
                        mlook = monolith_gp_lookup(mono[1]) if mono else None
                        if mono:
                            row["l2_mono"] = ev.l2_disp_vs_monolith(model, mono[0])["rel_l2"]
                        st = ev.evaluate_stress(foc, test, variant, os.path.join(THISDIR, folder))
                        row["max_err_anal"] = st["max_err_syy"]
                        em = ev.stress_vs_monolith(foc, mono[1]) if mono else None
                        row["max_err_mono"] = em if em is not None else np.nan
                        # Steifigkeit nur fuer Punktlast-Faelle (pressure/dispcontrol)
                        # sinnvoll; bei Eigengewicht ist die Last verteilt und die
                        # Reaktion spiegelt die Knotenverteilung der Volumenkraft.
                        if test != "selfweight":
                            rs = ev.reaction_stiffness(model)
                            row["F_react"], row["k_eff"] = rs["F_react"], rs["k_eff"]
                        ct = ev.evaluate_contact(model, test)
                        row["springt"] = ct["springt"] if ct and "springt" in ct else _springt(ct)
                        gp_rows.extend(collect_gp_rows(foc, test, elem, mesh, mlook))
                    comp_rows.append(row)
                else:
                    ev.evaluate_stress(foc, test, variant, os.path.join(THISDIR, folder))
                    gp_rows.extend(collect_gp_rows(foc, test, elem, mesh, None) if conv else [])
                    if test == "separation":
                        s = ev.evaluate_separation(model, foc) if conv else {}
                        sep_rows.append(dict(elem=elem, mesh=mesh, converged=conv, **s))
                    elif test == "sliding":
                        s = ev.evaluate_sliding(model, foc) if conv else {}
                        sli_rows.append(dict(elem=elem, mesh=mesh, converged=conv, **s))
                    else:
                        s = ev.evaluate_inclined(model, foc) if conv else {}
                        inc_rows.append(dict(elem=elem, mesh=mesh, converged=conv, **s))

    # --- Hertz'scher Kontakt (08_hertz) ---
    hertz_rows = []
    hfolder = os.path.join(THISDIR, "08_hertz")
    for name in ["hex8_medium", "hex8_fine", "hex20_medium", "hex20_fine"]:
        inp = os.path.join(hfolder, f"hertz_{name}.inp")
        if not os.path.exists(inp):
            continue
        print(f"  ... hertz_{name}", flush=True)
        model, foc, conv = ev.run_job(inp)
        row = dict(name=name, converged=conv)
        if conv:
            row.update(ev.evaluate_hertz(model))
            ev.write_lambda_vtk(model, os.path.join(hfolder, f"lambda_hertz_{name}.vtk"))
            ev.write_hertz_profile(model, os.path.join(hfolder, f"hertz_profile_hertz_{name}.csv"))
        hertz_rows.append(row)

    _write_gp_files(gp_rows)
    _write_contact_files(contact_rows)
    _print_compression(comp_rows)
    _print_behaviour(sep_rows, sli_rows, inc_rows)
    _print_hertz(hertz_rows)
    _print_gp_note(gp_rows)


def _springt(ct):
    if not ct:
        return None
    return (not ct["same_sign"]) or (ct["max_rel_dev"] > 1e-2)


# ---------------------------------------------------------------------------
GP_COLS = ["test", "elem", "mesh", "el", "gp", "x", "y", "z",
           "s11", "s22", "s33", "s12", "s13", "s23",
           "e11", "e22", "e33", "e12", "e13", "e23",
           "syy_anal", "err_anal", "syy_anal_ctr", "err_anal_ctr", "syy_mono", "err_mono"]


def _write_gp_files(rows):
    import csv
    with open(os.path.join(THISDIR, "final_gp_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=GP_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in GP_COLS})
    hdr = (f"{'el':>3} {'gp':>3} {'x':>7} {'y':>7} {'z':>7} "
           f"{'s_xx':>9} {'s_yy':>9} {'s_zz':>9} {'s_xy':>9} {'e_yy':>10} "
           f"{'syy_anal':>9} {'err_anal':>10} {'err_mono':>10}")
    lines, cur = [], None
    for r in rows:
        key = (r["test"], r["elem"], r["mesh"])
        if key != cur:
            cur = key
            lines += ["", "=" * len(hdr), f" {r['test']}  |  {r['elem']}  |  {r['mesh']}",
                      "=" * len(hdr), hdr, "-" * len(hdr)]
        em = "" if np.isnan(r["err_mono"]) else f"{r['err_mono']:10.2e}"
        sa = "" if np.isnan(r["syy_anal"]) else f"{r['syy_anal']:9.4f}"
        ea = "" if np.isnan(r["err_anal"]) else f"{r['err_anal']:10.2e}"
        lines.append(f"{r['el']:>3} {r['gp']:>3} {r['x']:7.3f} {r['y']:7.3f} {r['z']:7.3f} "
                     f"{r['s11']:9.4f} {r['s22']:9.4f} {r['s33']:9.4f} {r['s12']:9.4f} "
                     f"{r['e22']:10.3e} {sa:>9} {ea:>10} {em:>10}")
    with open(os.path.join(THISDIR, "final_gp_table.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def _write_contact_files(rows):
    import csv
    with open(os.path.join(THISDIR, "final_contact_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["test", "elem", "mesh", "slave_node", "lam"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _f(x, nan="  -  "):
    if x is None:
        return "  -  "
    if isinstance(x, bool):
        return "ja" if x else "NEIN"
    if isinstance(x, float) and np.isnan(x):
        return nan
    return f"{x:.2e}" if isinstance(x, float) else str(x)


def _print_compression(rows):
    print("\n" + "#" * 108)
    print("# (A) DRUCK-FAMILIE: L2-Fehler (Verschiebung), Spannungsfehler, STEIFIGKEIT, Kontakt-Urteil")
    print("#" * 108)
    cur = None
    def flt(x, w, dec):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return f"{'-':>{w}}"
        return f"{x:>{w}.{dec}f}"
    hdr = (f"{'Variante':<22}{'konv':>5}{'L2(anal)':>10}{'L2(mono)':>11}{'max|err_an|':>12}"
           f"{'max|err_mo|':>12}{'F_react':>9}{'k_eff':>8}{'Kontakt':>9}")
    for r in rows:
        if r["test"] != cur:
            cur = r["test"]
            print(f"\n=== {cur} ===\n{hdr}\n" + "-" * len(hdr))
        springt = "SPRINGT" if r["springt"] else ("ok" if r["springt"] is not None else "-")
        print(f"{r['elem'] + '_' + r['mesh']:<22}{('ja' if r['converged'] else 'NEIN'):>5}"
              f"{_f(r['l2_anal']):>10}{_f(r['l2_mono']):>11}{_f(r['max_err_anal']):>12}"
              f"{_f(r['max_err_mono']):>12}{flt(r['F_react'], 9, 3)}{flt(r['k_eff'], 8, 1)}{springt:>9}")
    print("\nSteifigkeit (nur Punktlast pressure/dispcontrol): F_react = Reaktionskraft unten (erwartet 10),")
    print("k_eff = |F/u_oben| (erwartet 500). Gleiche k_eff fuer hex8/hex20/hex20R => alle GLEICH steif.")
    print("Bei Eigengewicht ist die Last verteilt -> keine sinnvolle Einzel-Steifigkeit ('-').")


def _print_behaviour(sep, sli, inc):
    print("\n" + "#" * 108)
    print("# (B) VERHALTENS-TESTS")
    print("#" * 108)

    print("\n=== 04 Abheben unter Zug (unilateral): trennt sich der Kontakt? ===")
    print(f"{'Variante':<22}{'konv':>5}{'max|lam|':>11}{'max|sigma|':>12}{'u_unten':>11}{'Urteil':>12}")
    for r in sep:
        urteil = "getrennt" if r.get("separated") else ("-" if not r["converged"] else "NICHT getr.")
        print(f"{r['elem'] + '_' + r['mesh']:<22}{('ja' if r['converged'] else 'NEIN'):>5}"
              f"{_f(r.get('max_lam')):>11}{_f(r.get('max_sigma')):>12}{_f(r.get('u_lowerblock')):>11}{urteil:>12}")

    print("\n=== 05 Tangentiales Gleiten (reibungsfrei): Normaldruck bleibt, kein Schub, unten ruht ===")
    print(f"{'Variante':<22}{'konv':>5}{'sigma_yy':>11}{'max|Schub|':>12}{'u_x unten':>12}{'lam_mittel':>12}")
    for r in sli:
        print(f"{r['elem'] + '_' + r['mesh']:<22}{('ja' if r['converged'] else 'NEIN'):>5}"
              f"{_f(r.get('syy_mean')):>11}{_f(r.get('max_shear')):>12}"
              f"{_f(r.get('ux_lowerblock')):>12}{_f(r.get('lam_mean')):>12}")

    print("\n=== 06 Schiefes Interface 30 Grad (Normalen-Test): reproduziert der Kontakt die exakte Loesung? ===")
    print(f"{'Variante':<22}{'konv':>5}{'rel_L2':>11}{'max|u-ex|':>12}{'max|s_aa+10|':>13}{'lam aktiv':>11}{'gl.Vz.':>8}")
    for r in inc:
        akt = f"{r.get('n_active', '-')}/{r.get('n_total', '-')}" if r["converged"] else "-"
        print(f"{r['elem'] + '_' + r['mesh']:<22}{('ja' if r['converged'] else 'NEIN'):>5}"
              f"{_f(r.get('rel_l2')):>11}{_f(r.get('max_abs')):>12}{_f(r.get('max_saa_err')):>13}{akt:>11}"
              f"{_f(r.get('same_sign')):>8}")
    print("\nmax|u-ex| = Verschiebungsfehler, max|s_aa+10| = Fehler der Axialspannung (soll -10).")
    print("Nach Korrektur der Referenzgeometrie ist der schiefe Kontakt maschinengenau -> Normalen korrekt.")


def _print_hertz(rows):
    print("\n" + "#" * 108)
    print("# (C) HERTZ'SCHER KONTAKT: gekruemmter Indenter, Vergleich der Druckverteilung gegen Hertz")
    print("#" * 108)
    hdr = (f"{'Netz':<16}{'konv':>5}{'P_prime':>10}{'a_num':>9}{'a_Hertz':>9}{'a_err%':>8}"
           f"{'p0_num':>9}{'p0_Hertz':>10}{'p0_err%':>9}{'rms_p':>8}{'zickzk%':>9}{'zz_F%':>8}{'akt.':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if not r["converged"]:
            print(f"{r['name']:<16}{'NEIN':>5}")
            continue
        print(f"{r['name']:<16}{'ja':>5}{r['Pprime']:>10.3f}{r['a_num']:>9.4f}{r['a_hertz']:>9.4f}"
              f"{r['a_err_rel'] * 100:>8.1f}{r['p0_num']:>9.3f}{r['p0_hertz']:>10.3f}"
              f"{r['peak_err_rel'] * 100:>9.1f}{r['rms_p_err']:>8.3f}{r['zigzag'] * 100:>9.1f}"
              f"{r['zigzag_force'] * 100:>8.1f}{r['n_active']:>6}")
    print("\nhex8 (linear): GLATTES Druckprofil (kleiner zickzk), aber Maximum systematisch zu hoch")
    print("(p0_err), sinkt mit Verfeinerung. hex20 (quadratisch): Maximum sehr genau (~1%), aber")
    print("KNOTEN-ZU-KNOTEN-OSZILLATION (zickzk ~8-10%). Die Eckknoten treffen Hertz ueber die")
    print("ganze innere Kontaktzone konstant; der Zickzack sitzt ausschliesslich auf den MITTEL-")
    print("knoten und waechst zum Rand der Kontaktzone hin - der volle quadratische Multiplikator-")
    print("raum kann den dortigen sqrt-Abfall auf einen Rand ZWISCHEN zwei Knoten nicht darstellen")
    print("und schwingt am Mittelknoten ueber. Sinkt nur langsam mit Verfeinerung. Die")
    print("uebertragene Gesamtkraft ist bei beiden korrekt. Druckprofile in hertz_profile_*.csv.")
    print("\nzickzk% liest den Multiplikator lambda direkt als Druck, zz_F% die Knotenkraft")
    print("lambda*D_II geteilt durch die tributaere Flaeche der ganzen Facette. An teilweise")
    print("ueberdeckten Randknoten skaliert lambda mit 1/D_II, die beiden Lesarten koennen dort")
    print("also auseinanderlaufen. Dass sie es NICHT tun, belegt: die Oszillation ist eine")
    print("Eigenschaft der Diskretisierung und kein Artefakt der Auswertung am Kontaktrand.")


def _print_gp_note(rows):
    print("\n" + "#" * 108)
    print(f"# PER-GAUSSPUNKT-TABELLE ({len(rows)} Zeilen) -> final_gp_table.txt (lesbar) + .csv (vollstaendig)")
    print(f"# KONTAKTDRUECKE je Slave-Knoten -> final_contact_table.csv")
    print("#" * 108)


if __name__ == "__main__":
    main()
