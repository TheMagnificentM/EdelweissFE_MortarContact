#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test 9: Verifikation der statischen (dualen) Kondensation der Mortar-Multiplikatoren
===================================================================================

Löst identische Kontakt-Patch-Modelle ZWEIMAL:
  (a) Sattelpunkt (explizite Multiplikatoren, condensation=False) - Referenz,
  (b) kondensiert  (condensation=True) - vektorielle Elimination (Gitterle 2010,
      Eqs. 85/86).

Verifiziert:
  - kondensierte Lösung == Sattelpunkt-Lösung (Verschiebungen + Normaldruck) bis ~1e-9,
  - beide == exakte Patch-Lösung (sigma = -p),
  - Tangentialtraktion ~ 0 (reibungsfrei).

Deckt 2D (CPE4/CONLINE2, CPE8/CONLINE3) und 3D (hex8/CONQUAD4, hex20/CONQUAD8),
jeweils nicht-passende Netze, ab.
"""

import os
import sys

import numpy as np

TESTDIR = os.path.abspath(os.path.dirname(__file__))
MORTARDIR = os.path.abspath(os.path.join(TESTDIR, ".."))
sys.path.insert(0, os.path.join(MORTARDIR, "test6_hex20_patch_test"))
sys.path.insert(0, os.path.join(MORTARDIR, "test8_2d_patch_test"))

import make_contact_elements as mce  # noqa: E402  (3D contact overlay)
import patch_test_2d as p2  # noqa: E402  (2D contact overlay)

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation  # noqa: E402
from edelweissfe.sets.nodeset import NodeSet  # noqa: E402
from edelweissfe.utils.inputfileparser import parseInputFile  # noqa: E402

E_MOD = 1000.0
PRESSURE = 10.0

INP_3D = """*modelGenerator, generator=boxGen, name=genA
nX={nxA}
nY=1
nZ={nzA}
lX=1.0
lY=1.0
lZ=1.0
elType={elA}

*modelGenerator, generator=boxGen, name=genB
y0=1.0
nX={nxB}
nY=1
nZ={nzB}
lX=1.0
lY=1.0
lZ=1.0
elType={elB}

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
mortarSurface=con_master{cond}

*job, name=patchjob, domain=3d
*solver, name=theSolver, solver=NIST

*step, solver=theSolver
maxInc=0.5, minInc=1e-3, maxNumInc=100, maxIter=25, stepLength=1
>>dirichlet, name=bot, nSet=genA_bottom, field=displacement, 2=0.0
>>dirichlet, name=top, nSet=genB_top, field=displacement, 2={utop}
>>dirichlet, name=lat, nSet=allnodes, field=displacement, 1=0.0, 3=0.0
"""

INP_2D = """*modelGenerator, generator=planeRectQuad, name=genA
x0=0.0
y0=0.0
l=1.0
h=1.0
nX={nxA}
nY=1
elType={elA}

*modelGenerator, generator=planeRectQuad, name=genB
x0=0.0
y0=1.0
l=1.0
h=1.0
nX={nxB}
nY=1
elType={elB}

*modelGenerator, generator=executePythonCode, name=contactgen
import generated_setup_{name} as vs
vs.setup(model)

*material, name=LinearElastic, id=mat
{E}, 0.0

*section, name=secA, material=mat, type=plane, thickness=1.0
genA_all
*section, name=secB, material=mat, type=plane, thickness=1.0
genB_all

*constraint, type=mortarcontact, name=contact
nonMortarSurface=con_slave
mortarSurface=con_master{cond}

*job, name=patchjob, domain=2d
*solver, name=theSolver, solver=NIST

*step, solver=theSolver
maxInc=0.5, minInc=1e-3, maxNumInc=100, maxIter=25, stepLength=1
>>dirichlet, name=bot, nSet=genA_bottom, field=displacement, 2=0.0
>>dirichlet, name=top, nSet=genB_top, field=displacement, 2={utop}
>>dirichlet, name=lat, nSet=allnodes, field=displacement, 1=0.0
"""


def _write_setup_3d(name, conA, conB):
    lines = [
        "import sys",
        f"sys.path.insert(0, r'{os.path.join(MORTARDIR, 'test6_hex20_patch_test')}')",
        "import make_contact_elements as mce",
        "from edelweissfe.sets.nodeset import NodeSet",
        "",
        "def setup(model):",
        f"    mce.apply(model, 'genA_top', 'con_slave', '{conA}', (0.0, 1.0, 0.0))",
        f"    mce.apply(model, 'genB_bottom', 'con_master', '{conB}', (0.0, -1.0, 0.0))",
        "    model.nodeSets['allnodes'] = NodeSet('allnodes', list(model.nodes.values()))",
    ]
    path = os.path.join(TESTDIR, f"generated_setup_{name}.py")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def _write_setup_2d(name, conA, conB):
    lines = [
        "import sys",
        f"sys.path.insert(0, r'{os.path.join(MORTARDIR, 'test8_2d_patch_test')}')",
        "import patch_test_2d as p2",
        "from edelweissfe.sets.nodeset import NodeSet",
        "",
        "def setup(model):",
        f"    p2.attach_2d_contact(model, 'genA_top', 'con_slave', '{conA}', (0.0, 1.0))",
        f"    p2.attach_2d_contact(model, 'genB_bottom', 'con_master', '{conB}', (0.0, -1.0))",
        "    alln = [n for n in model.nodes.values() if 'displacement' in n.fields]",
        "    model.nodeSets['allnodes'] = NodeSet('allnodes', alln)",
    ]
    path = os.path.join(TESTDIR, f"generated_setup_{name}.py")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def _solve(template, name, fmt, condensation):
    cond = ", condensation=True" if condensation else ""
    inp_text = template.format(cond=cond, E=E_MOD, utop=-2.0 * PRESSURE / E_MOD, name=name, **fmt)
    inp_path = os.path.join(TESTDIR, f"generated_{name}.inp")
    with open(inp_path, "w") as f:
        f.write(inp_text)
    inputFile = parseInputFile(inp_path)
    model, _ = finiteElementSimulation(inputFile, verbose=False, suppressPlots=True)
    os.remove(inp_path)
    nf = model.nodeFields["displacement"]
    U = np.array(nf["U"])
    contact = model.constraints["contact"]
    lam = np.asarray(contact.recovered_lambdas).flatten()
    tract = np.asarray(contact.recovered_tractions)
    normals = contact.current_normals
    active = np.asarray(contact.active_set)
    return model, U, lam, tract, normals, active


def run_case(kind, name, template, fmt, conA, conB):
    print(f"\n* {kind}-Fall: {name}")
    if kind == "3D":
        setup_path = _write_setup_3d(name + "_ref", conA, conB)
        os.replace(setup_path, os.path.join(TESTDIR, f"generated_setup_{name}.py"))
    else:
        setup_path = _write_setup_2d(name + "_ref", conA, conB)
        os.replace(setup_path, os.path.join(TESTDIR, f"generated_setup_{name}.py"))
    if TESTDIR not in sys.path:
        sys.path.insert(0, TESTDIR)

    # (a) saddle-point reference
    _, U_s, lam_s, _, _, _ = _solve(template, name, fmt, condensation=False)
    # (b) condensed
    _, U_c, lam_c, tract_c, normals_c, active_c = _solve(template, name, fmt, condensation=True)

    u_ref = PRESSURE * 2.0 / E_MOD

    # condensed vs saddle
    dU = np.max(np.abs(U_c - U_s)) / u_ref
    dLam = np.max(np.abs(lam_c - lam_s)) / PRESSURE
    print(f"  ||U_cond - U_saddle||_inf / u_ref     = {dU:.3e}")
    print(f"  ||lam_cond - lam_saddle||_inf / p     = {dLam:.3e}")

    # condensed vs exact patch solution
    err_uy = 0.0
    err_lat = 0.0
    nf_nodes = None
    # recompute from U_c against coordinates via a fresh model? reuse last model's nodeField
    # (we only kept arrays; recompute exact check from saddle model geometry is identical)
    # -> use the condensed model geometry:
    model_c, U_c2, lam_c2, tract_c2, normals_c2, active_c2 = _solve(template, name, fmt, condensation=True)
    nf = model_c.nodeFields["displacement"]
    for node, u in zip(nf.nodes, np.array(nf["U"])):
        y = node.coordinates[1]
        err_uy = max(err_uy, abs(u[1] + PRESSURE * y / E_MOD))
        err_lat = max(err_lat, abs(u[0]))
        if len(u) == 3:
            err_lat = max(err_lat, abs(u[2]))
    lam_act = lam_c2[active_c2]
    lam_err = np.max(np.abs(np.abs(lam_act) - PRESSURE)) / PRESSURE if len(lam_act) else 1.0

    tang = 0.0
    for I in np.flatnonzero(active_c2):
        z = tract_c2[I]
        nn = normals_c2[I]
        tang = max(tang, np.linalg.norm(z - (z @ nn) * nn))

    print(f"  kondensiert vs exakt: max|u_y-u_exakt|={err_uy:.2e}, max|u_lat|={err_lat:.2e}, "
          f"lam_err={lam_err:.2e}, tang={tang:.2e}")

    ok = (dU < 1e-9 and dLam < 1e-8 and err_uy < 1e-8 * u_ref and err_lat < 1e-8 * u_ref
          and lam_err < 1e-8 and tang < 1e-8 * PRESSURE)

    os.remove(os.path.join(TESTDIR, f"generated_setup_{name}.py"))
    if ok:
        print(f"  [PASS] {name}")
    else:
        print(f"  [FAIL] {name}")
        sys.exit(1)


if __name__ == "__main__":
    print("=" * 60)
    print("MORTAR DUAL-KONDENSATION: kondensiert == Sattelpunkt == exakt")
    print("=" * 60)

    # 2D
    run_case("2D", "cpe4", INP_2D, dict(nxA=2, elA="CPE4", nxB=3, elB="CPE4"), "CONLINE2", "CONLINE2")
    run_case("2D", "cpe8", INP_2D, dict(nxA=2, elA="CPE8", nxB=3, elB="CPE8"), "CONLINE3", "CONLINE3")

    # 3D
    run_case("3D", "hex8", INP_3D,
             dict(nxA=2, nzA=2, elA="C3D8", nxB=3, nzB=3, elB="C3D8"), "CONQUAD4", "CONQUAD4")
    run_case("3D", "hex20", INP_3D,
             dict(nxA=2, nzA=2, elA="C3D20", nxB=3, nzB=3, elB="C3D20"), "CONQUAD8", "CONQUAD8")

    print("\n" + "=" * 60)
    print("ALLE KONDENSATIONS-TESTS ERFOLGREICH PASSIERT!")
    print("=" * 60)
