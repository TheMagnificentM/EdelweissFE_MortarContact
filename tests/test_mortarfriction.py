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
The Coulomb conditions of the mortar penalty friction, checked on the converged state of the two
friction decks rather than as a diff of displacements.

``MortarContactFrictionSlide`` (implicit) ends in a drag that has carried every contact node into
slip, so there the cone has to be met with equality: ``|t_I| = mu p_I`` at every active node, and the
traction has to point along the drag. ``NEDMortarContactFriction`` (explicit) ends mid-motion, where
nodes may stick or slip, so the condition left is the cone itself, ``|t_I| <= mu p_I`` -- together
with the fact that friction did transmit something, without which the check would be vacuous.
"""

import contextlib
import io
import os
from pathlib import Path

import numpy as np

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.utils.inputfileparser import parseInputFile

_DECK_ROOT = Path(__file__).resolve().parents[1] / "testfiles" / "edelweiss-only"


def _runDeck(deckName: str, withOutputs: bool = False):
    """Run one regression deck and return its mortar constraint, holding the last assembled state."""

    previous = os.getcwd()
    try:
        os.chdir(_DECK_ROOT / deckName)
        inputFile = parseInputFile("test.inp")
        with contextlib.redirect_stdout(io.StringIO()):
            model, fieldOutputs = finiteElementSimulation(inputFile, verbose=False, suppressPlots=True)
    finally:
        os.chdir(previous)

    constraints = [c for c in model.constraints.values() if hasattr(c, "nonMortarNodes")]
    assert len(constraints) == 1, f"{deckName} does not hold exactly one mortar constraint"
    if withOutputs:
        return constraints[0], fieldOutputs
    return constraints[0]


def test_a_dragged_block_slides_on_the_friction_cone():
    constraint = _runDeck("MortarContactFrictionSlide")

    active = constraint.activeSet
    assert np.sum(active) > 0, "no node is in contact at the end of the drag"
    np.testing.assert_array_equal(constraint._friction.slipState[active], 2)

    pressure = np.abs(constraint.nodalMultipliers[active])
    traction = constraint._friction.tractionCurrent[active]
    magnitude = np.linalg.norm(traction, axis=1)
    np.testing.assert_allclose(magnitude, constraint._friction.mu * pressure, rtol=1e-10)
    # The upper (non-mortar) block is dragged in +x over the fixed lower one, so the friction it
    # receives opposes that: -x.
    assert np.all(traction[:, 0] < -0.9 * magnitude)


def test_explicit_friction_stays_inside_the_cone_and_transmits_force():
    constraint = _runDeck("NEDMortarContactFriction")

    pressure = np.abs(constraint.nodalMultipliers)
    magnitude = np.linalg.norm(constraint._friction.tractionCurrent, axis=1)
    assert np.max(magnitude) > 0.0, "no frictional traction at all -- test is vacuous"
    assert np.all(magnitude <= constraint._friction.mu * pressure * (1.0 + 1e-12) + 1e-14 * np.max(pressure))


def test_the_nodal_contact_results_reach_the_field_outputs():
    """The fromExpression outputs over the published node set carry the constraint's own state."""

    constraint, fieldOutputs = _runDeck("MortarContactFrictionSlide", withOutputs=True)
    outputs = fieldOutputs.fieldOutputs
    n = constraint.nNonMortarNodes

    traction = np.asarray(outputs["contactTraction"].getLastResult())
    assert traction.shape == (n, 3)
    np.testing.assert_allclose(traction, constraint._friction.tractionCurrent)
    np.testing.assert_allclose(
        np.asarray(outputs["contactPressure"].getLastResult()).ravel(), constraint.getNormalPressures()
    )
    np.testing.assert_array_equal(
        np.asarray(outputs["contactSlipState"].getLastResult()).ravel(), constraint._friction.slipState
    )
    gap = np.asarray(outputs["contactGap"].getLastResult()).ravel()
    assert np.all(gap[constraint.activeSet] <= 0.0), "an active node reports an opening"
