#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _         _____ _____
# | ____|__| | ___| |_      _____(_)___ ___|  ___| ____|
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |_  |  _|
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \  _| | |___
# |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|   |_____|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  2017 - today
#
#  Matthias Neuner matthias.neuner@uibk.ac.at
#
#  This file is part of EdelweissFE.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissFE.
#  ---------------------------------------------------------------------
"""
AMR and restart for every contact formulation, end to end.

The setup of ``testfiles/marmot/AMR_ContactRefineShear``: two C3D20R blocks with a non-conforming
interface, pressed together (step 1) and dragged across each other with friction (step 2), while
hAdaptivity refines both bodies -- contact surfaces included -- on a strain marker. It is run with the
node-to-surface, the integrated and the mortar constraint (penalty with friction, and Lagrange
multipliers without), and checked twice:

* **AMR**: the run completes, the contact surfaces were actually refined (the constraint holds more
  slave points than at setup), and the contact state is admissible (|t| <= mu p).
* **Restart with AMR**: a run truncated after the refinement and resumed from its last checkpoint --
  which replays the refinement, then restores the contact state onto the refined surfaces -- ends
  with the displacements of the uninterrupted run.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.utils.inputfileparser import parseInputFile

pytest.importorskip("edelweissfe.elements.marmotelement.element", reason="the AMR decks use Marmot elements")

MU = 0.3

_MESH = """
*material, name=linearelastic, id=mat
18000, 0.0

*section, name=sec, material=mat, type=solid
fixed_all
moving_all

*modelGenerator, generator=boxGen, name=fixed
nX=2, nY=2, nZ=1
x0=0, y0=0, z0=0
lX=1, lY=1, lZ=1
elType=C3D20R

*modelGenerator, generator=boxGen, name=moving
nX=3, nY=3, nZ=2
x0=0.07, y0=0.11, z0=1.05
lX=0.83, lY=0.79, lZ=1
elType=C3D20R

*modelGenerator, generator=surfaceElementGenerator, name=genMaster
surface = fixed_front
name    = masterSurf
{facetOption}

*modelGenerator, generator=surfaceElementGenerator, name=genSlave
surface = moving_back
name    = slaveSurf
{facetOption}

*constraint, name=contact, type={constraintType}
{constraintOptions}

*modelModifier, type=hadaptivity, name=amr0
>>marker, type=fieldOutput, fieldOutput=strainFixedForAMR, operator='>', threshold=0.02
>>marker, type=fieldOutput, fieldOutput=strainMovingForAMR, operator='>', threshold=0.02
maxLevel=1

*job, name=amrcontact, domain=3d
*solver, solver=NIST, name=theSolver

*fieldOutput
>>perElement, elSet=fixed_all, result=strain, quadraturePoint=0:8, name=strainFixedForAMR, f(x)='np.abs(x)'
>>perElement, elSet=moving_all, result=strain, quadraturePoint=0:8, name=strainMovingForAMR, f(x)='np.abs(x)'
"""

_STEP_1 = """
*step, solver=theSolver
maxInc=0.2, minInc=1e-4, maxNumInc=50, maxIter=25, stepLength=1
>>options, name=theSolver, extrapolation=off
>>dirichlet, name=fixBack, nSet=fixed_back, field=displacement, 1=0.0, 2=0.0, 3=0.0
>>dirichlet, name=moveFront, nSet=moving_front, field=displacement, 1=0.0, 2=0.0, 3=-0.18
"""


def _step2(maxNumInc: int) -> str:
    return f"""
*step, solver=theSolver
maxInc=0.2, minInc=1e-4, maxNumInc={maxNumInc}, maxIter=25, stepLength=1
>>options, name=theSolver, extrapolation=off
>>dirichlet, name=fixBack, nSet=fixed_back, field=displacement, 1=0.0, 2=0.0, 3=0.0
>>dirichlet, name=moveFront, nSet=moving_front, field=displacement, 1=0.05, 2=0.0, 3=0.0
"""


_FACETS = "facets = wholeFace"
_TRIANGLES = "triangulation = midside"

_FORMULATIONS = {
    "nodeToSurface": dict(
        facetOption=_TRIANGLES,
        constraintType="nodeToDeformableSurfacePenalty",
        constraintOptions=(
            "slaveSurface=slaveSurf_facets, masterSurface=masterSurf_facets, penalty=1e6, type=quadratic, "
            f"searchDistance=2.0, sliding=small, mu={MU}, tangentPenalty=1e6"
        ),
    ),
    "integrated": dict(
        facetOption=_TRIANGLES,
        constraintType="surfaceToDeformableSurfacePenalty",
        constraintOptions=(
            "slaveSurface=slaveSurf_facets, masterSurface=masterSurf_facets, penalty=1e6, type=quadratic, "
            f"searchDistance=2.0, mu={MU}, tangentPenalty=1e6"
        ),
    ),
    "mortarPenalty": dict(
        facetOption=_FACETS,
        constraintType="mortarcontact",
        constraintOptions=(
            "nonMortarSurface=slaveSurf_facets, mortarSurface=masterSurf_facets, field=displacement, "
            f"formulation=penalty, penaltyStiffness=1e6, mu={MU}"
        ),
    ),
    "mortarLagrange": dict(
        facetOption=_FACETS,
        constraintType="mortarcontact",
        constraintOptions=(
            "nonMortarSurface=slaveSurf_facets, mortarSurface=masterSurf_facets, field=displacement, "
            "formulation=lagrange"
        ),
    ),
}


def _deck(formulation: str, steps: str, extra: str = "") -> str:
    return _MESH.format(**_FORMULATIONS[formulation]) + extra + steps


def _run(path: Path):
    model, _ = finiteElementSimulation(parseInputFile(str(path)), verbose=False, suppressPlots=True)
    return model


def _nSlavePoints(constraint) -> int:
    for attribute in ("nNonMortarNodes", "nSlaves", "nPoints"):
        if hasattr(constraint, attribute):
            return getattr(constraint, attribute)
    raise AttributeError("unknown contact constraint")


def _setupSlavePoints(formulation: str, tmp_path: Path) -> int:
    """The number of slave points before any refinement: the constraint of a model that is built
    but not run."""

    path = tmp_path / "setup.inp"
    path.write_text(_deck(formulation, ""))
    model, _ = finiteElementSimulation(parseInputFile(str(path)), verbose=False, suppressPlots=True)
    return _nSlavePoints(model.constraints["contact"])


def _mostRecentCheckpoint(tmp_path: Path) -> Path:
    checkpoints = list(tmp_path.glob("ckpt_*.h5"))
    assert checkpoints, "the truncated run did not write any restart checkpoint"

    def checkpointTime(path):
        with h5py.File(path, "r") as f:
            return f.attrs["time"]

    return max(checkpoints, key=checkpointTime)


@pytest.mark.parametrize("formulation", list(_FORMULATIONS))
def test_contact_survives_refinement_of_both_surfaces(formulation, tmp_path):
    nSetup = _setupSlavePoints(formulation, tmp_path)

    path = tmp_path / "full.inp"
    path.write_text(_deck(formulation, _STEP_1 + _step2(50)))
    model = _run(path)
    constraint = model.constraints["contact"]

    assert model.time == pytest.approx(2.0), "the run did not complete"
    assert _nSlavePoints(constraint) > nSetup, "the slave surface was never refined -- test is vacuous"

    pressures = constraint.getNormalPressures()
    assert pressures.max() > 0.0, "no contact pressure after the refinement"
    if "Lagrange" not in formulation:
        tractions = constraint.getTangentialTractions()
        if tractions.ndim == 2:
            tractions = np.linalg.norm(tractions, axis=1)
        np.testing.assert_array_less(tractions, MU * np.maximum(pressures, 0.0) * (1 + 1e-8) + 1e-12)
        assert tractions.max() > 0.0, "friction transmitted nothing -- test is vacuous"


@pytest.mark.parametrize("formulation", list(_FORMULATIONS))
def test_restart_after_refinement_matches_the_uninterrupted_run(formulation, tmp_path):
    fullPath = tmp_path / "full.inp"
    fullPath.write_text(_deck(formulation, _STEP_1 + _step2(50)))
    reference = _run(fullPath)
    referenceU = reference.nodeFields["displacement"]["U"].copy()

    checkpointBaseName = tmp_path / "ckpt"
    restartWriter = (
        "\n*output, type=restart, name=restartwriter\n"
        f"writeInterval=1, baseName={checkpointBaseName}, numberOfFilesToKeep=3\n"
    )
    truncatedPath = tmp_path / "truncated.inp"
    truncatedPath.write_text(_deck(formulation, _STEP_1 + _step2(2), restartWriter))
    truncated = _run(truncatedPath)
    assert truncated.time < 2.0, "the truncated run already finished -- not a real truncation"
    assert len(truncated.elements) > 3 * 3 * 2 + 2 * 2 * 1 + 1, "truncated before any refinement"

    resumePath = tmp_path / "resume.inp"
    resumePath.write_text(
        _deck(formulation, _STEP_1 + _step2(50), f"\n*restart, readFrom={_mostRecentCheckpoint(tmp_path)}\n")
    )
    resumed = _run(resumePath)

    assert set(resumed.elements.keys()) == set(reference.elements.keys())
    np.testing.assert_allclose(resumed.nodeFields["displacement"]["U"], referenceU, atol=1e-9)
    for name, variable in reference.scalarVariables.items():
        assert resumed.scalarVariables[name].value == pytest.approx(variable.value, abs=1e-6), name
