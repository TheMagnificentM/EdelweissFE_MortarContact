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
"""What the two geometry updates of the mortar contact cost in the converged result.

``geometryUpdate=frozen`` builds the geometry once per increment at the last converged state, so the
converged state of an increment satisfies the contact conditions in the configuration the increment
STARTED from: the result carries an error of first order in the increment size. Under
``geometryUpdate=incomplete`` the geometry is rebuilt at every iterate, so the converged state
satisfies them in its own configuration and does not depend on how the load was split.

That is a property of the converged state of a whole run, which ``run_tests_edelweissfe``'s single
reference solution cannot express, so it is checked here by running ``MortarContactTilted`` -- a
partly closed, partly lifted interface on non-matching meshes -- at several increment sizes.
"""

import contextlib
import io
import os
from pathlib import Path

import numpy as np
import pytest

from edelweissfe.drivers.inputfiledrivensimulation import finiteElementSimulation
from edelweissfe.utils.inputfileparser import parseInputFile

_DECK = Path(__file__).resolve().parents[1] / "testfiles" / "edelweiss-only" / "MortarContactTilted" / "test.inp"


def _finalDisplacement(tmp_path: Path, geometryUpdate: str, maxInc: float) -> np.ndarray:
    """Run the deck with the given geometry update and increment size, return the final displacements."""

    text = _DECK.read_text()
    assert "geometryUpdate=incomplete" in text and "maxInc=0.5," in text
    text = text.replace("geometryUpdate=incomplete", f"geometryUpdate={geometryUpdate}")
    text = text.replace("maxInc=0.5,", f"maxInc={maxInc}, startInc={maxInc},")
    directory = tmp_path / f"{geometryUpdate}_{maxInc}"
    directory.mkdir()
    (directory / "test.inp").write_text(text)
    previous = os.getcwd()
    try:
        os.chdir(directory)
        with contextlib.redirect_stdout(io.StringIO()):
            model, _ = finiteElementSimulation(parseInputFile("test.inp"), verbose=False, suppressPlots=True)
    finally:
        os.chdir(previous)
    return model.nodeFields["displacement"]["U"].copy()


@pytest.mark.parametrize("geometryUpdate", ["frozen", "incomplete"])
def test_dependence_on_the_increment_size(tmp_path, geometryUpdate):
    coarse = _finalDisplacement(tmp_path, geometryUpdate, 0.5)
    fine = _finalDisplacement(tmp_path, geometryUpdate, 0.125)
    deviation = float(np.max(np.abs(coarse - fine)) / np.max(np.abs(fine)))
    if geometryUpdate == "incomplete":
        # No staggering: the converged states agree to the tolerance of the Newton iteration, which
        # converges linearly here and stops further from the solution than a quadratic one would.
        assert deviation < 1e-6, deviation
    else:
        # Staggered: an error of first order in the increment size, measurable on this deck.
        assert deviation > 1e-3, deviation
