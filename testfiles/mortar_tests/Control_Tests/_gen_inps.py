#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generator fuer alle Control_Tests-.inp-Dateien.

Erzeugt in 01_selfweight/, 02_pressure/, 03_dispcontrol/, 04_separation/,
05_sliding/, 06_inclined/ je:
  <test>_<elem>_<mesh>.inp       fuer elem in {hex8, hex20, hex20R}, mesh in {matching, nonmatching}
  reference_monolith_<elem>.inp  nur fuer die Druck-Familie (01/02/03)

Die erzeugten Dateien sind eigenstaendige, lesbare EdelweissFE-Beispiele und
direkt via `edelweissfe <datei>.inp` ausfuehrbar.

Aufruf:  python _gen_inps.py
"""

import os

THISDIR = os.path.abspath(os.path.dirname(__file__))

ELEM = {  # token -> (elType, nGP)
    "hex8": ("C3D8", 8),
    "hex20": ("C3D20", 27),
    "hex20R": ("C3D20R", 8),
}
MESH = {  # token -> (nA, nB)  (nX=nZ je Block)
    "matching": (2, 2),
    "nonmatching": (2, 3),
}

TESTNUM = {"selfweight": "01", "pressure": "02", "dispcontrol": "03",
           "separation": "04", "sliding": "05", "inclined": "06", "stiffness": "07"}
FOLDER = {"selfweight": "01_selfweight", "pressure": "02_pressure",
          "dispcontrol": "03_dispcontrol", "separation": "04_separation",
          "sliding": "05_sliding", "inclined": "06_inclined", "stiffness": "07_stiffness"}
TESTLABEL = {"selfweight": "Eigengewicht", "pressure": "Druck von oben",
             "dispcontrol": "verschiebungsgesteuert", "separation": "Zug / Abheben",
             "sliding": "tangentiales Gleiten (reibungsfrei)",
             "inclined": "schiefes Interface (30 Grad, Normalen-Test)",
             "stiffness": "Steifigkeitskontrast (weich auf steif)"}
WITH_MONOLITH = {"selfweight", "pressure", "dispcontrol"}

# Materialien fuer den Steifigkeitskontrast-Test
E_A_STIFF = 1000.0      # Block A (unten, Slave) = weich
E_B_STIFF = 100000.0    # Block B (oben, Master) = steif (Verhaeltnis 100)

# Komplementaritaetsparameter c_n der semismooth Normalkontakt-NCP. Rein
# algorithmisch (bei Konvergenz g -> 0, also loesungsunabhaengig) und in der
# Groessenordnung des E-Moduls des WEICHEREN Koerpers zu waehlen (Hueber &
# Wohlmuth 2005; Farah 2018, Abschn. 3.5.2). Der weichere Block ist in allen
# Tests Block A mit E = 1000 (auch im Steifigkeitskontrast-Test 07).
CN = 1000.0

# --- exakte / erwartete Loesung, als Kommentarblock je Test --------------------
EXACT = {
    "selfweight": (
        "** EXAKTE LOESUNG (E=1000, nu=0, Volumenkraft b=10 nach unten):\n"
        "**   sigma_yy(y) = -b*(2-y)  ->  0 (oben), -10 (Interface y=1), -20 (unten)\n"
        "**   u_y(y) = (0.5*b*y^2 - b*2*y)/E = (5*y^2 - 20*y)/1000 ; u_x=u_z=0\n"
        "**   Kontaktdruck am Interface = b*hoehe_B = 10 (konstant)\n"
    ),
    "pressure": (
        "** EXAKTE LOESUNG (E=1000, nu=0, Druck p=10 auf B-Oberseite):\n"
        "**   sigma_yy = -p = -10 (konstant) ; u_y(y) = -p*y/E = -0.01*y ; u_x=u_z=0\n"
        "**   Kontaktdruck am Interface = p = 10 (konstant)\n"
    ),
    "dispcontrol": (
        "** EXAKTE LOESUNG (E=1000, nu=0, u_y=-0.02 auf B-Oberseite vorgeschrieben):\n"
        "**   eps_yy = -0.01 ; sigma_yy = -E*0.01 = -10 (konstant) ; u_y(y) = -0.01*y\n"
        "**   Kontaktdruck am Interface = 10 (konstant)\n"
    ),
    "separation": (
        "** ERWARTETE LOESUNG (Oberseite wird um +0.02 nach OBEN gezogen):\n"
        "**   Ein einseitiger (unilateraler) Kontakt TRENNT sich unter Zug:\n"
        "**   Kontaktdruck -> 0, Spannung -> 0 in beiden Bloecken, ein Spalt oeffnet sich.\n"
        "**   Unterer Block bleibt in Ruhe (u=0), oberer Block verschiebt sich starr um +0.02.\n"
    ),
    "sliding": (
        "** ERWARTETE LOESUNG (Oberseite: u_x=+0.01 seitlich UND u_y=-0.02 druecken):\n"
        "**   Reibungsfreier Kontakt -> freies tangentiales Gleiten:\n"
        "**   Normaldruck bleibt konstant 10, sigma_yy=-10, KEIN Schub (sigma_xy=0),\n"
        "**   der untere Block wird NICHT mitgeschleppt (u_x=0 unten).\n"
    ),
    "inclined": (
        "** ERWARTETE LOESUNG (gesamte Geometrie um 30 Grad um z gedreht, dann axial gedrueckt):\n"
        "**   Die Kontaktflaeche liegt schief -> Test der Knoten-Normalen.\n"
        "**   Axialspannung entlang der (gedrehten) Achse = -10, Kontaktdruck = 10 konstant.\n"
        "**   Achse a = (-sin30, cos30, 0) = (-0.5, 0.8660254, 0).\n"
    ),
    "stiffness": (
        "** EXAKTE LOESUNG (Block A E_A=1000 weich=Slave, Block B E_B=100000 steif; u_y=-0.02 oben):\n"
        "**   Reihenschaltung: sigma_yy = -0.02/(1/E_A + 1/E_B) = -19.80198 (konstant, Gleichgewicht)\n"
        "**   u_y stueckweise linear (steiler im weichen Block A): u(1)=sigma/E_A, u(2)=-0.02\n"
        "**   Kontaktdruck = |sigma_yy| = 19.80198. Alle Elementtypen ergeben dieselbe Reihensteifigkeit.\n"
    ),
}

# --- lastspezifischer Step-Block (Zwei-Block-Kontakttest) ---------------------
STEP_CONTACT = {
    "selfweight": (
        ">>dirichlet, name=bot, nSet=genA_bottom, field=displacement, 2=0.0\n"
        ">>dirichlet, name=lat, nSet=allnodes,    field=displacement, 1=0.0, 3=0.0\n"
        ">>bodyforce, name=gravity, elSet=solids, forceVector='0.0, -10.0, 0.0', f(t)=t\n"
    ),
    "pressure": (
        ">>dirichlet, name=bot, nSet=genA_bottom, field=displacement, 2=0.0\n"
        ">>dirichlet, name=lat, nSet=allnodes,    field=displacement, 1=0.0, 3=0.0\n"
        ">>distributedload, name=press, surface=genB_top, type=pressure, magnitude=10.0, f(t)=t\n"
    ),
    "dispcontrol": (
        ">>dirichlet, name=bot, nSet=genA_bottom, field=displacement, 2=0.0\n"
        ">>dirichlet, name=lat, nSet=allnodes,    field=displacement, 1=0.0, 3=0.0\n"
        ">>dirichlet, name=top, nSet=genB_top,    field=displacement, 2=-0.02, f(t)=t\n"
    ),
    "separation": (
        ">>dirichlet, name=bot, nSet=genA_bottom, field=displacement, 2=0.0\n"
        ">>dirichlet, name=lat, nSet=allnodes,    field=displacement, 1=0.0, 3=0.0\n"
        ">>dirichlet, name=top, nSet=genB_top,    field=displacement, 2=0.02, f(t)=t\n"
    ),
    "sliding": (
        ">>dirichlet, name=zfix, nSet=allnodes,    field=displacement, 3=0.0\n"
        ">>dirichlet, name=bot,  nSet=genA_bottom, field=displacement, 1=0.0, 2=0.0\n"
        ">>dirichlet, name=top,  nSet=genB_top,    field=displacement, 1=0.01, 2=-0.02, f(t)=t\n"
    ),
    # inclined: gedrehte einachsige Kompression. Ober- und Unterseite werden auf die
    # (gedrehten) exakten Werte gesetzt, die Seiten sind frei -> der KONTAKT traegt
    # die Last ueber die SCHIEFE Flaeche. u_oben = R(30)*(0,-0.02,0) = (0.01, -0.01732, 0).
    # setup_inclined dreht die Geometrie und erfasst die Element-Referenz neu.
    "inclined": (
        ">>dirichlet, name=zfix, nSet=allnodes,    field=displacement, 3=0.0\n"
        ">>dirichlet, name=bot,  nSet=genA_bottom, field=displacement, 1=0.0, 2=0.0\n"
        ">>dirichlet, name=top,  nSet=genB_top,    field=displacement, 1=0.01, 2=-0.0173205081, f(t)=t\n"
    ),
    "stiffness": (
        ">>dirichlet, name=bot, nSet=genA_bottom, field=displacement, 2=0.0\n"
        ">>dirichlet, name=lat, nSet=allnodes,    field=displacement, 1=0.0, 3=0.0\n"
        ">>dirichlet, name=top, nSet=genB_top,    field=displacement, 2=-0.02, f(t)=t\n"
    ),
}

# Material-/Section-Block je Test (Standard: ein Material E=1000 auf beiden Bloecken;
# Steifigkeitskontrast: zwei Materialien).
_MATSEC_DEFAULT = (
    "*material, name=linearelastic, id=mat\n"
    "1000.0, 0.0\n\n"
    "*section, name=secA, material=mat, type=solid\n"
    "genA_all\n"
    "*section, name=secB, material=mat, type=solid\n"
    "genB_all"
)
_MATSEC_STIFFNESS = (
    f"*material, name=linearelastic, id=matA\n{E_A_STIFF}, 0.0\n"
    f"*material, name=linearelastic, id=matB\n{E_B_STIFF}, 0.0\n\n"
    "*section, name=secA, material=matA, type=solid\n"
    "genA_all\n"
    "*section, name=secB, material=matB, type=solid\n"
    "genB_all"
)
MATSEC = {t: _MATSEC_DEFAULT for t in TESTNUM}
MATSEC["stiffness"] = _MATSEC_STIFFNESS

ANALYTICAL_FIELDS = ""
ANALYTICAL_FOR = {t: False for t in TESTNUM}
MAXINC = {t: (0.25 if t == "inclined" else 0.5) for t in TESTNUM}

# --- lastspezifischer Step-Block (Monolith-Referenz, nur Druck-Familie) -------
STEP_MONO = {
    "selfweight": (
        ">>dirichlet, name=bot, nSet=gen_bottom, field=displacement, 2=0.0\n"
        ">>dirichlet, name=lat, nSet=allnodes,   field=displacement, 1=0.0, 3=0.0\n"
        ">>bodyforce, name=gravity, elSet=gen_all, forceVector='0.0, -10.0, 0.0', f(t)=t\n"
    ),
    "pressure": (
        ">>dirichlet, name=bot, nSet=gen_bottom, field=displacement, 2=0.0\n"
        ">>dirichlet, name=lat, nSet=allnodes,   field=displacement, 1=0.0, 3=0.0\n"
        ">>distributedload, name=press, surface=gen_top, type=pressure, magnitude=10.0, f(t)=t\n"
    ),
    "dispcontrol": (
        ">>dirichlet, name=bot, nSet=gen_bottom, field=displacement, 2=0.0\n"
        ">>dirichlet, name=lat, nSet=allnodes,   field=displacement, 1=0.0, 3=0.0\n"
        ">>dirichlet, name=top, nSet=gen_top,    field=displacement, 2=-0.02, f(t)=t\n"
    ),
}

# Minimale Stabilisierung nur fuer die LASTGESTEUERTEN Tests (selfweight, pressure):
# dort haelt der Kontakt den oberen Block vertikal allein -> Starrkoerpermodus ->
# singulaere Tangente. Eine winzige y-Feder (penalty=1e-5) entfernt genau diesen
# Modus; ihr Fussabdruck ist ~1e-6 (siehe README). Alle anderen Tests schreiben
# die bewegte Flaeche per Dirichlet vor und brauchen KEINE Feder.
STAB_PENALTY = 1e-5
STAB_BLOCK = (
    "** --- minimale Stabilisierung: schwache y-Feder gegen den Starrkoerpermodus\n"
    "** --- von Block B unter Laststeuerung (siehe README). Fussabdruck ~1e-6.\n"
    "*constraint, type=directionalspringpenalty, name=stab_B_y\n"
    "nSet=genB_top\n"
    "field=displacement\n"
    "component=1\n"
    f"penalty={STAB_PENALTY}\n\n"
)
STAB_FOR = {"selfweight": True, "pressure": True, "dispcontrol": False,
            "separation": False, "sliding": False, "inclined": False,
            "stiffness": False}

# Aufruf im executePythonCode-Block: inclined dreht die Geometrie vorher.
PYCALL = {t: "contact_setup.setup(model)" for t in TESTNUM}
PYCALL["inclined"] = "contact_setup.setup_inclined(model)"

PYHEAD = (
    "import os\n"
    "import sys\n"
    "sys.path.append(os.getcwd())\n"
    "sys.path.append(os.path.dirname(os.getcwd()))\n"
    "import contact_setup\n"
)


def contact_inp(test, elemTok, meshTok):
    elType, nGP = ELEM[elemTok]
    nA, nB = MESH[meshTok]
    return f"""** ==============================================================================
** Control Test {TESTNUM[test]} - {TESTLABEL[test]}   ({elemTok}, {meshTok} interface)
** ==============================================================================
**
** Zwei linear-elastische Einheitswuerfel, Block A (unten, y in [0,1], Slave) und
** Block B (oben, y in [1,2], Master), ueber MortarContact3D verbunden.
** Elementtyp {elType} ; Interface {meshTok} (A {nA}x{nA} gegen B {nB}x{nB}).
**
{EXACT[test]}**
** Ausfuehren:  edelweissfe {test}_{elemTok}_{meshTok}.inp
** Auswerten:   python ../evaluate.py {test}_{elemTok}_{meshTok}.inp
** ------------------------------------------------------------------------------

*modelGenerator, generator=boxGen, name=genA
nX={nA}
nY=1
nZ={nA}
lX=1.0
lY=1.0
lZ=1.0
elType={elType}

*modelGenerator, generator=boxGen, name=genB
y0=1.0
nX={nB}
nY=1
nZ={nB}
lX=1.0
lY=1.0
lZ=1.0
elType={elType}

*modelGenerator, generator=executePythonCode, name=contactgen
{PYHEAD}{PYCALL[test]}

{MATSEC[test]}

** cn: Komplementaritaetsparameter c_n der Normalkontakt-NCP, rein algorithmisch
** (kein Einfluss auf die konvergierte Loesung) und in der Groessenordnung des
** E-Moduls des weicheren Koerpers zu waehlen -> E_A = 1000 (Farah 2018, 3.5.2).
*constraint, type=mortarcontact, name=contact
nonMortarSurface=con_slave
mortarSurface=con_master
field=displacement
cn={CN}

*job, name={test}job, domain=3d
*solver, name=theSolver, solver=NISTParallel

*fieldOutput
>>perNode,    name=displacement, elSet=all, field=displacement, result=U
>>perNode,    name=P,            elSet=all, field=displacement, result=P
>>perNode,    name=nodal_U,      elSet=all, field=displacement, result=U, saveHistory=True, export=nodal_U
>>perNode,    name=RF,           nSet=genA_bottom, field=displacement, result=P, saveHistory=True, export=reaction
>>perElement, name=stress,       elSet=solids, result=stress, quadraturePoint=0:{nGP}, f(x)='np.mean(x,axis=1)'
>>perElement, name=strain,       elSet=solids, result=strain, quadraturePoint=0:{nGP}, f(x)='np.mean(x,axis=1)'
>>perElement, name=stressGP,     elSet=solids, result=stress, quadraturePoint=0:{nGP}, saveHistory=True, export=stress_gp
>>perElement, name=strainGP,     elSet=solids, result=strain, quadraturePoint=0:{nGP}, saveHistory=True, export=strain_gp
>>perNode,    name=utop,         nSet=genB_top, field=displacement, result=U, f(x)='np.mean(x[:,1])', saveHistory=True, export=utop

*output, type=ensight, name=ensight_{test}_{elemTok}_{meshTok}
>>perNode,    fieldOutput=displacement
>>perNode,    fieldOutput=P
>>perElement, fieldOutput=stress
>>perElement, fieldOutput=strain
>>configuration, overwrite=yes

*output, type=monitor, name=myMonitor
fieldOutput=utop

{ANALYTICAL_FIELDS if ANALYTICAL_FOR[test] else ""}{STAB_BLOCK if STAB_FOR[test] else ""}*step, solver=theSolver
maxInc={MAXINC[test]}, minInc=1e-4, maxNumInc=400, maxIter=30, stepLength=1
{STEP_CONTACT[test]}"""


def mono_inp(test, elemTok):
    elType, nGP = ELEM[elemTok]
    return f"""** ==============================================================================
** Control Test {TESTNUM[test]} - REFERENZ (Monolith, {elemTok})
** ==============================================================================
**
** Ein verschmolzener Block 1 x 2 x 1 (y in [0,2]), KEIN Kontakt, Elementtyp {elType}.
** Gleiches Netz wie die zwei gestapelten Bloecke (nY=2), gleiche Last.
** Liefert die FE-Referenz -> isoliert den Kontakt von der Element-Diskretisierung.
**
** Ausfuehren:  edelweissfe reference_monolith_{elemTok}.inp
** ------------------------------------------------------------------------------

*modelGenerator, generator=boxGen, name=gen
nX=2
nY=2
nZ=2
lX=1.0
lY=2.0
lZ=1.0
elType={elType}

*modelGenerator, generator=executePythonCode, name=allnodesgen
{PYHEAD}contact_setup.add_allnodes(model)

*material, name=linearelastic, id=mat
1000.0, 0.0

*section, name=sec, material=mat, type=solid
gen_all

*job, name={test}monojob, domain=3d
*solver, name=theSolver, solver=NISTParallel

*fieldOutput
>>perNode,    name=displacement, elSet=all, field=displacement, result=U
>>perNode,    name=P,            elSet=all, field=displacement, result=P
>>perNode,    name=nodal_U,      elSet=all, field=displacement, result=U, saveHistory=True, export=mono_nodal_U
>>perElement, name=stress,       elSet=all, result=stress, quadraturePoint=0:{nGP}, f(x)='np.mean(x,axis=1)'
>>perElement, name=strain,       elSet=all, result=strain, quadraturePoint=0:{nGP}, f(x)='np.mean(x,axis=1)'
>>perElement, name=stressGP,     elSet=all, result=stress, quadraturePoint=0:{nGP}, saveHistory=True, export=mono_stress_gp

*output, type=ensight, name=ensight_{test}_monolith_{elemTok}
>>perNode,    fieldOutput=displacement
>>perNode,    fieldOutput=P
>>perElement, fieldOutput=stress
>>configuration, overwrite=yes

*step, solver=theSolver
maxInc=0.5, minInc=1e-3, maxNumInc=200, maxIter=25, stepLength=1
{STEP_MONO[test]}"""


# ---------------------------------------------------------------------------
# Hertz-Kontakt (08_hertz): parabolischer Indenter auf flacher Foundation, 2D
# (duenne Scheibe, ebener Verzerrungszustand ueber z-Fixierung). Halbmodell x>=0
# mit Symmetrie bei x=0. Vergleich der Druckverteilung gegen die Hertz-Loesung.
HERTZ_R = 10.0      # Kruemmungsradius (muss zu contact_setup.HERTZ_R passen)
HERTZ_LX = 4.0      # Halbbreite der Bloecke (>> Kontakthalbbreite a ~ 0.5)
HERTZ_LY = 5.0      # Hoehe je Block: Tiefe >> a (Halbraum-Naeherung, a/LY ~ 0.1)
HERTZ_DELTA = 0.03  # vorgeschriebene Eindrueckung
# ausreichende Aufloesung in BEIDEN Richtungen nahe der Kontaktzone noetig
HERTZ_MESHES = [
    ("hex8_medium", "C3D8", 100, 12, 8),    # ~11 Elemente ueber a
    ("hex8_fine", "C3D8", 200, 16, 8),      # ~22 Elemente ueber a
    ("hex20_medium", "C3D20", 100, 12, 27),  # ~11 Elemente ueber a
    ("hex20_fine", "C3D20", 160, 14, 27),    # ~17 Elemente ueber a
]


def hertz_inp(name, elType, nX, nY, nGP):
    return f"""** ==============================================================================
** Control Test 08 - Hertz'scher Kontakt   ({name})
** ==============================================================================
**
** Parabolischer Indenter (Block B, Unterseite y += x^2/2R, R={HERTZ_R})
** auf flacher Foundation (Block A). Beide linear-elastisch (E=1000, nu=0), duenne
** Scheibe mit z-Fixierung (ebener Verzerrungszustand), Halbmodell x>=0 mit
** Symmetrie bei x=0. Verschiebungsgesteuert (Indenter oben um {HERTZ_DELTA} gedrueckt).
**
** VERGLEICH: Kontaktdruck-Verteilung lambda(x) gegen Hertz p(x)=p0*sqrt(1-(x/a)^2),
**   Kontakthalbbreite a und Maximaldruck p0 aus der berechneten Last P' (2D Hertz):
**   a = sqrt(4 P' R / (pi E*)),  p0 = 2 P'/(pi a),  1/E* = (1-nu^2)/E1 + (1-nu^2)/E2.
** Hertz ist eine Naeherung (Halbraum) -> nicht maschinengenau, konvergiert aber mit
** Netzverfeinerung (coarse -> medium -> fine).
**
** Ausfuehren:  edelweissfe hertz_{name}.inp
** Auswerten:   python ../evaluate.py hertz_{name}.inp
** ------------------------------------------------------------------------------

*modelGenerator, generator=boxGen, name=genA
x0=0.0
nX={nX}
nY={nY}
nZ=1
lX={HERTZ_LX}
lY={HERTZ_LY}
lZ=0.1
elType={elType}

*modelGenerator, generator=boxGen, name=genB
x0=0.0
y0={HERTZ_LY}
nX={nX}
nY={nY}
nZ=1
lX={HERTZ_LX}
lY={HERTZ_LY}
lZ=0.1
elType={elType}

*modelGenerator, generator=executePythonCode, name=hertzgen
{PYHEAD}contact_setup.setup_hertz(model)

*material, name=linearelastic, id=mat
1000.0, 0.0

*section, name=secA, material=mat, type=solid
genA_all
*section, name=secB, material=mat, type=solid
genB_all

** cn: Komplementaritaetsparameter c_n der Normalkontakt-NCP ~ O(E) = 1000.
*constraint, type=mortarcontact, name=contact
nonMortarSurface=con_slave
mortarSurface=con_master
field=displacement
cn={CN}

*job, name=hertzjob, domain=3d
*solver, name=theSolver, solver=NISTParallel

*fieldOutput
>>perNode,    name=displacement, elSet=all, field=displacement, result=U
>>perNode,    name=P,            elSet=all, field=displacement, result=P
>>perElement, name=stress,       elSet=solids, result=stress, quadraturePoint=0:{nGP}, f(x)='np.mean(x,axis=1)'

*output, type=ensight, name=ensight_hertz_{name}
>>perNode,    fieldOutput=displacement
>>perNode,    fieldOutput=P
>>perElement, fieldOutput=stress
>>configuration, overwrite=yes

*step, solver=theSolver
maxInc=0.1, minInc=1e-4, maxNumInc=400, maxIter=30, stepLength=1
>>dirichlet, name=zfix, nSet=allnodes,    field=displacement, 3=0.0
>>dirichlet, name=symA, nSet=genA_left,   field=displacement, 1=0.0
>>dirichlet, name=symB, nSet=genB_left,   field=displacement, 1=0.0
>>dirichlet, name=bot,  nSet=genA_bottom, field=displacement, 1=0.0, 2=0.0
>>dirichlet, name=top,  nSet=genB_top,    field=displacement, 2=-{HERTZ_DELTA}, f(t)=t
"""


def generate_hertz():
    folder = os.path.join(THISDIR, "08_hertz")
    os.makedirs(folder, exist_ok=True)
    for name, elType, nX, nY, nGP in HERTZ_MESHES:
        with open(os.path.join(folder, f"hertz_{name}.inp"), "w") as f:
            f.write(hertz_inp(name, elType, nX, nY, nGP))
    print("generated 08_hertz")


def main():
    for test in TESTNUM:
        folder = os.path.join(THISDIR, FOLDER[test])
        os.makedirs(folder, exist_ok=True)
        for elemTok in ELEM:
            for meshTok in MESH:
                fn = os.path.join(folder, f"{test}_{elemTok}_{meshTok}.inp")
                with open(fn, "w") as f:
                    f.write(contact_inp(test, elemTok, meshTok))
            if test in WITH_MONOLITH:
                fn = os.path.join(folder, f"reference_monolith_{elemTok}.inp")
                with open(fn, "w") as f:
                    f.write(mono_inp(test, elemTok))
        print("generated", FOLDER[test])
    generate_hertz()


if __name__ == "__main__":
    main()
