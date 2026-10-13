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
One friction law for all contact formulations, checked end to end.

The same drag -- a small block pressed onto a larger one and dragged across it, the setup of
``MortarContactFrictionSlide`` -- is computed with the node-to-surface, the integrated
(Gauss-point-to-surface) and the mortar constraint, all with the same penalty, the same linear normal
law and the same Coulomb law (mu = 0.3, eps_T = eps_N). The formulations differ in how they
discretise the contact condition, so the displacements may differ slightly; what may not differ is
the friction law: at the end of the drag every loaded contact point slips with ``|t| = mu p``, and
the friction force transmitted to the lower block is mu times the normal force, for all three alike.
"""

import contextlib
import io
import os

import numpy as np
import pytest

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.utils.inputfileparser import parseInputFile

MU = 0.3

_DECK = """
*material, name=LinearElastic, id=linearelastic, provider=edelweiss
1.8e4, 0.3

*job, name=frictionjob, domain=3d
*solver, solver=NIST, name=theSolver

*modelGenerator, generator=boxGen, name=lower
nX=3, nY=3, nZ=1
x0=0, y0=0, z0=0
lX=2, lY=2, lZ=1
elProvider=edelweiss
elType=C3D8

*modelGenerator, generator=boxGen, name=upper
nX=2, nY=2, nZ=2
x0=0.3, y0=0.7, z0=1.05
lX=0.6, lY=0.6, lZ=1
elProvider=edelweiss
elType=C3D8

*modelGenerator, generator=surfaceElementGenerator, name=gen1
surface = upper_back
name    = slaveSurf
{facetOption}

*modelGenerator, generator=surfaceElementGenerator, name=gen2
surface = lower_front
name    = masterSurf
{facetOption}

*section, name=section1, material=linearelastic, type=solid
lower_all
upper_all

*constraint, name=contact, type={constraintType}
{constraintOptions}

*fieldOutput
>>perNode, name=RFDrag,   nSet=lower_back, field=displacement, result=P, f(x)='sum(x[:,0])'
>>perNode, name=RFNormal, nSet=lower_back, field=displacement, result=P, f(x)='sum(x[:,2])'

*step, solver=theSolver
maxInc=0.34, minInc=1e-6, maxNumInc=200, maxIter=25, stepLength=1
>>options, name=theSolver, extrapolation=off
>>dirichlet, name=fixLower, nSet=lower_back, field=displacement, 1=0.0, 2=0.0, 3=0.0
>>dirichlet, name=holdUpper, nSet=upper_front, field=displacement, 1=0.0, 2=0.0, 3=-0.1

*step, solver=theSolver
maxInc=0.1, minInc=1e-6, maxNumInc=200, maxIter=25, stepLength=1
>>options, name=theSolver, extrapolation=off
>>dirichlet, name=fixLower, nSet=lower_back, field=displacement, 1=0.0, 2=0.0, 3=0.0
>>dirichlet, name=holdUpper, nSet=upper_front, field=displacement, 1=0.4, 2=0.0, 3=0.0
"""

_FORMULATIONS = {
    "nodeToSurface": dict(
        facetOption="triangulation = midside",
        constraintType="nodeToDeformableSurfacePenalty",
        constraintOptions=(
            f"slaveSurface=slaveSurf_facets, masterSurface=masterSurf_facets, penalty=1e6, type=linear, "
            f"searchDistance=2.0, sliding=small, mu={MU}"
        ),
    ),
    "integrated": dict(
        facetOption="triangulation = midside",
        constraintType="surfaceToDeformableSurfacePenalty",
        constraintOptions=(
            f"slaveSurface=slaveSurf_facets, masterSurface=masterSurf_facets, penalty=1e6, type=linear, "
            f"searchDistance=2.0, mu={MU}"
        ),
    ),
    "mortar": dict(
        facetOption="facets = wholeFace",
        constraintType="mortarcontact",
        constraintOptions=(
            "nonMortarSurface=slaveSurf_facets, mortarSurface=masterSurf_facets, field=displacement, "
            f"formulation=penalty, penaltyStiffness=1e6, mu={MU}"
        ),
    ),
}


def _runDrag(formulation: str, directory):
    previous = os.getcwd()
    try:
        os.chdir(directory)
        with open("test.inp", "w") as deck:
            deck.write(_DECK.format(**_FORMULATIONS[formulation]))
        inputFile = parseInputFile("test.inp")
        with contextlib.redirect_stdout(io.StringIO()):
            model, fieldOutputs = finiteElementSimulation(inputFile, verbose=False, suppressPlots=True)
    finally:
        os.chdir(previous)
    drag = float(np.asarray(fieldOutputs.fieldOutputs["RFDrag"].getLastResult()).ravel()[0])
    normal = float(np.asarray(fieldOutputs.fieldOutputs["RFNormal"].getLastResult()).ravel()[0])
    return model.constraints["contact"], drag, normal


@pytest.mark.parametrize("formulation", list(_FORMULATIONS))
def test_every_formulation_slides_on_the_same_friction_cone(formulation, tmp_path):
    constraint, drag, normal = _runDrag(formulation, tmp_path)

    pressures = constraint.getNormalPressures()
    tractions = constraint.getTangentialTractions()
    if tractions.ndim == 2:  # the mortar constraint reports the traction vector
        tractions = np.linalg.norm(tractions, axis=1)
    states = constraint.getSlipStates()
    loaded = pressures > 1e-8 * pressures.max()

    assert loaded.sum() > 3, "hardly any contact point is loaded -- the test is vacuous"
    np.testing.assert_array_equal(states[loaded], 2.0, "every loaded point slips at the end of the drag")
    np.testing.assert_allclose(tractions[loaded], MU * pressures[loaded], rtol=1e-8)

    # The friction resultant on the lower block is mu times its normal resultant -- up to the small
    # inclination of the contact normals the drag tilts the upper block into.
    assert normal > 0.0
    assert drag / normal == pytest.approx(-MU, rel=0.02)
