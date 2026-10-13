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
"""Unit tests of the node-to-surface contact constraint's Coulomb friction.

The friction law is the shared one of :mod:`edelweissfe.utils.contactfriction`; what is checked here
is the node-to-surface mapping onto it -- the consistent tangent of the assembled residual in stick
and slip, and the friction cone in terms of the reported pressure and traction.
"""

import unittest

import numpy as np

from edelweissfe.constraints.nodetodeformablesurfacepenalty import (
    Constraint as NodeToSurfaceContact,
)
from edelweissfe.constraints.nodetodeformablesurfacepenalty import (
    NodeToDeformableSurfacePenaltySchema,
)
from edelweissfe.constraints.test_surfacetodeformablesurfacepenalty import twoBlockModel
from edelweissfe.journal.journal import Journal
from edelweissfe.utils.contactfriction import SLIP, STICK


class TestNodeToSurfaceFriction(unittest.TestCase):
    def setUp(self):
        self.journal = Journal()

    def _constraint(self, **options) -> NodeToSurfaceContact:
        model, slaveSurface, masterSurface = twoBlockModel(self.journal, penetration=0.05)
        configuration = NodeToDeformableSurfacePenaltySchema(
            penalty=1.0e4, searchDistance=1.0, sliding="small", contactType="quadratic", **options
        )
        constraint = NodeToSurfaceContact(
            "theContact", model, slaveSurface, masterSurface, self.journal, configuration=configuration
        )
        constraint.updateConnectivity(model)
        return constraint

    @staticmethod
    def _slaveBlocks(constraint) -> list:
        """The local DOF offset of each slave node, and of its facet block (None if unassigned)."""

        blocks, offset = [], 0
        for s in range(constraint.nSlaves):
            pStart = offset
            offset += constraint.nDim
            fStart = None
            if constraint._assignedFacetIdx[s] is not None:
                fStart = offset
                offset += len(constraint.facetElements[constraint._assignedFacetIdx[s]].nodes) * constraint.nDim
            blocks.append((pStart, fStart, offset))
        return blocks

    def _slide(self, constraint, amount) -> np.ndarray:
        """Moves every slave node by ``amount`` in x (tangential), plus a little noise everywhere."""

        U = 1e-4 * np.random.default_rng(3).standard_normal(constraint.nDof)
        for pStart, _, _ in self._slaveBlocks(constraint):
            U[pStart] += amount
        return U

    def _forces(self, constraint, U, dU) -> np.ndarray:
        PExt = np.zeros(constraint.nDof)
        constraint.applyConstraint(U, dU, PExt, None, None)
        return PExt

    def _assembledTangent(self, constraint, U, dU) -> np.ndarray:
        flat = np.zeros(constraint.getVIJContributionSize())
        view = constraint.shapeVIJContribution(flat)
        constraint.applyConstraint(U, dU, np.zeros(constraint.nDof), view, None)

        K = np.zeros((constraint.nDof, constraint.nDof))
        activeIdx = 0
        for pStart, fStart, end in self._slaveBlocks(constraint):
            if fStart is None:
                continue
            p, f = np.arange(pStart, fStart), np.arange(fStart, end)
            K[np.ix_(p, p)] += view.K_pp[activeIdx]
            K[np.ix_(f, f)] += view.K_ff[activeIdx]
            K[np.ix_(p, f)] += view.K_pf[activeIdx]
            K[np.ix_(f, p)] += view.K_fp[activeIdx]
            activeIdx += 1
        return K

    def _checkTangent(self, slide, expectedState, **options):
        constraint = self._constraint(**options)
        U = self._slide(constraint, slide)
        K = self._assembledTangent(constraint, U, U)
        loaded = constraint.getNormalPressures() > 0.0
        self.assertTrue(loaded.any(), "no slave node is in contact -- test is vacuous")
        np.testing.assert_array_equal(constraint._friction.slipState[loaded], expectedState)

        h = 1e-8
        numeric = np.zeros_like(K)
        for j in range(constraint.nDof):
            step = np.zeros(constraint.nDof)
            step[j] = h
            plus = self._forces(constraint, U + step, U + step)
            minus = self._forces(constraint, U - step, U - step)
            numeric[:, j] = -(plus - minus) / (2.0 * h)
        np.testing.assert_array_equal(constraint._friction.slipState[loaded], expectedState, "state flipped")
        scale = np.abs(K).max()
        np.testing.assert_allclose(K / scale, numeric / scale, atol=1e-6)
        return constraint

    def test_the_stick_tangent_differentiates_the_residual(self):
        self._checkTangent(1e-5, STICK, mu=10.0)

    def test_the_slip_tangent_differentiates_the_residual(self):
        constraint = self._checkTangent(5e-2, SLIP, mu=0.3)
        loaded = constraint.getNormalPressures() > 0.0
        np.testing.assert_allclose(
            constraint.getTangentialTractions()[loaded], 0.3 * constraint.getNormalPressures()[loaded], rtol=1e-12
        )

    def test_the_slip_tolerance_tangent_differentiates_the_residual(self):
        self._checkTangent(1e-3, STICK, mu=0.3, slipTolerance=0.05)
        self._checkTangent(5e-2, SLIP, mu=0.3, slipTolerance=0.05)

    def test_slip_tolerance_is_refused_explicitly(self):
        constraint = self._constraint(mu=0.3, slipTolerance=0.005)
        U = np.zeros(constraint.nDof)
        with self.assertRaises(NotImplementedError):
            constraint.applyConstraintExplicit(U, U, np.zeros(constraint.nDof), None)

    def test_a_force_checkpoint_is_restored_as_traction(self):
        constraint = self._constraint(mu=0.3)
        force = np.tile([1.0, 0.0, 0.0], (constraint.nSlaves, 1))
        constraint.setRestartData({"tangentialForceConverged": force, "lambdaN": np.zeros(constraint.nSlaves)})
        areas = constraint.tributaryAreas
        np.testing.assert_allclose(constraint._friction.tractionConverged[areas > 0, 0], 1.0 / areas[areas > 0])


if __name__ == "__main__":
    unittest.main()
