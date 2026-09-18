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
#  Manuel Hradsky manuel.hradsky@uibk.ac.at
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
"""Does the converged mortar contact solution actually satisfy the Signorini conditions?

.. code-block:: console

    p_n >= 0,        g_sep >= 0,        p_n * g_sep = 0

The regression decks under ``testfiles/edelweiss-only/MortarContact*`` compare a converged
displacement field against a stored one, and the unit tests beside
:mod:`~edelweissfe.constraints.mortarcontact` check the pieces the constraint is assembled from.
Neither can see the failure this file is for: an active set that froze on the wrong nodes. The
Newton iteration then converges just as cleanly, because it is solving a well-posed problem -- only
not the contact problem. The result is visible in exactly two places, at an active node carrying
tension or at an inactive node that has penetrated, and nowhere in the displacement norm.

That is a property of the converged state rather than a value to diff, which ``run_tests_edelweissfe``'s
single ``test.inp``/``U.ref`` model cannot express -- so, like ``tests/test_restart_integration.py``,
it lives here as a pytest test driving ``finiteElementSimulation`` over the decks that already exist.

The three quantities are rebuilt from the same frozen quantities and with the same sign convention
the constraint assembles from (``mortarcontact.py``, "Sign-consistent contact measures"), not
recomputed from a fresh geometry: the question is whether the conditions the solver enforced hold,
and an independently re-segmented geometry would answer a different one.

Three classes of non-mortar node, which the conditions mean different things for:

``uncovered``
    The nodal weight vanishes -- nothing lies opposite the node. Neither a pressure nor an opening is
    defined there and the constraint sets both to zero; what has to hold is that such a node carries
    no multiplier.
``active``
    The constraint enforces a vanishing weighted gap, so the condition left to check is
    ``p_n >= 0``.
``inactive``
    The constraint enforces a vanishing multiplier, so the condition left to check is
    ``g_sep >= 0``.

The decks are chosen so that every class is reached. ``MortarContactPartialOverlap`` is the only one
with nodes that have nothing opposite them, and ``MortarContactTilted`` the only one with a node that
has a counterpart and is separated from it anyway -- which is the case the active set exists for, and
which no deck produced until that one was written for this test. ``MortarContactPatchHexa20`` is the
quadratic one, and ``MortarContactPenaltyHexa20`` is the penalty branch, whose conditions hold only
to the accuracy of its stiffness rather than exactly.
"""

import contextlib
import io
import os
from pathlib import Path

import numpy as np
import pytest

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.utils.inputfileparser import parseInputFile

_DECK_ROOT = Path(__file__).resolve().parents[1] / "testfiles" / "edelweiss-only"

#: The penalty branch satisfies the conditions only approximately -- its pressure is manufactured
#: from the gap, so a node in contact keeps a gap of the order of pressure over stiffness. The
#: saddle-point branch satisfies them to round-off.
_TOLERANCE = {"lagrange": 1e-9, "penalty": 1e-6}


def _runDeck(deckName: str):
    """Run one regression deck to convergence and return its mortar constraint.

    Parameters
    ----------
    deckName
        The directory name under ``testfiles/edelweiss-only``.

    Returns
    -------
    Constraint
        The mortar contact constraint, holding the frozen quantities of the last assembly.
    """

    previous = os.getcwd()
    try:
        os.chdir(_DECK_ROOT / deckName)
        inputFile = parseInputFile("test.inp")
        # The decks write Ensight output and print a solution table; neither is of interest here.
        with contextlib.redirect_stdout(io.StringIO()):
            model, _ = finiteElementSimulation(inputFile, verbose=False, suppressPlots=True)
    finally:
        os.chdir(previous)

    constraints = [c for c in model.constraints.values() if hasattr(c, "currentNodalWeights")]
    assert len(constraints) == 1, f"{deckName} does not hold exactly one mortar constraint"
    return constraints[0]


def _signoriniMeasures(constraint):
    """The nodal pressure and physical opening of the converged state, per non-mortar node.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        The pressure, the opening, and the mask of nodes that have nothing opposite them.
    """

    weights = constraint.currentNodalWeights
    uncovered = np.abs(weights) <= constraint.currentWeightTolerance

    safeWeights = np.where(uncovered, 1.0, weights)
    pressure = np.where(uncovered, 0.0, constraint.nodalMultipliers * np.sign(safeWeights))
    opening = np.where(uncovered, 0.0, constraint.currentWeakGap / safeWeights)
    return pressure, opening, uncovered


@pytest.mark.parametrize(
    "deckName",
    [
        "MortarContactPartialOverlap",
        "MortarContactTilted",
        "MortarContactSeparation",
        "MortarContactPunchEdge",
        "MortarContactPatchHexa20",
        "MortarContactPenaltyHexa20",
    ],
)
def test_the_converged_solution_satisfies_the_signorini_conditions(deckName):
    """No tension is transmitted, nothing has penetrated, and no node does both at once."""

    constraint = _runDeck(deckName)
    pressure, opening, uncovered = _signoriniMeasures(constraint)
    active = constraint.activeSet
    tolerance = _TOLERANCE[constraint.formulation]

    scale = max(float(np.max(np.abs(pressure))), 1.0)

    assert not np.any(uncovered & (np.abs(constraint.nodalMultipliers) > tolerance * scale)), (
        f"{deckName}: a node with nothing opposite it carries a multiplier " f"{constraint.nodalMultipliers[uncovered]}"
    )

    checkedActive = active & ~uncovered
    assert np.all(pressure[checkedActive] >= -tolerance * scale), (
        f"{deckName}: {int(np.sum(pressure[checkedActive] < -tolerance * scale))} active node(s) "
        f"transmit tension, the most tensile being {float(np.min(pressure[checkedActive])):.3e}"
    )

    checkedInactive = ~active & ~uncovered
    if np.any(checkedInactive):
        openingScale = max(float(np.max(np.abs(opening))), 1.0)
        assert np.all(opening[checkedInactive] >= -tolerance * openingScale), (
            f"{deckName}: {int(np.sum(opening[checkedInactive] < -tolerance * openingScale))} "
            f"inactive node(s) have penetrated, the deepest by "
            f"{float(np.min(opening[checkedInactive])):.3e}"
        )

    complementarity = np.abs(pressure * opening)[~uncovered]
    assert np.all(complementarity <= tolerance * scale * max(float(np.max(np.abs(opening))), 1.0)), (
        f"{deckName}: pressure and opening are both non-zero at "
        f"{int(np.sum(complementarity > tolerance * scale))} node(s), the largest product being "
        f"{float(np.max(complementarity)):.3e}"
    )


def test_the_decks_still_exercise_every_node_class():
    """A guard on the decks rather than on the constraint.

    The check above is only worth running while its decks still produce all three classes of node.
    Each class comes from a different deck, and each could lose it to an innocuous-looking edit: a
    wider mortar surface would cover ``MortarContactPartialOverlap``'s free nodes, a larger tilt
    would separate all of ``MortarContactTilted``'s. The parametrised test would keep passing while
    checking a third of what it claims, so the classes are asserted separately here.
    """

    withoutCounterpart = _runDeck("MortarContactPartialOverlap")
    _, _, uncovered = _signoriniMeasures(withoutCounterpart)
    assert np.any(uncovered), "no node without a counterpart -- the uncovered branch is untested"

    partlyOpen = _runDeck("MortarContactTilted")
    _, _, uncoveredTilted = _signoriniMeasures(partlyOpen)
    assert np.any(
        ~partlyOpen.activeSet & ~uncoveredTilted
    ), "no covered but inactive node -- the non-penetration branch is untested"
    assert np.any(partlyOpen.activeSet), "no active node -- the compression branch is untested"
