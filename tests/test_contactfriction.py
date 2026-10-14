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
"""Unit tests of the shared Coulomb friction law, independent of any contact formulation.

A generic contact point is coupled to one slave and three master nodes through the coefficients
``c = [1, -N1, -N2, -N3]``; its pressure is a linear penalty of the normal gap along ``n``, so that
``p`` and ``q = dp/dU`` are exact and the full tangent can be checked by finite differences.
"""

import unittest

import numpy as np

from edelweissfe.utils.contactfriction import (
    OPEN,
    SLIP,
    STICK,
    CoulombFriction,
    characteristicFacetSize,
)

nDim = 3
c = np.array([1.0, -0.2, -0.5, -0.3])
nLoc = len(c) * nDim
A = 0.7
n = np.array([0.0, 0.0, 1.0])
epsN = 1.0e3
gap0 = -1.0e-2  # initial penetration: p = epsN * 1e-2 = 10


def _pressureAndGradient(U):
    g = gap0 + n @ (c @ U.reshape(len(c), nDim))
    p = max(-epsN * g, 0.0)
    q = -epsN * np.kron(c, n) if p > 0 else np.zeros(nLoc)
    return p, q


def _forces(friction, U, dU, **options):
    p, q = _pressureAndGradient(U)
    return friction.evaluatePoint(0, c, A, n, p, q, dU, epsN, **options)


def _slide(amount, direction=(1.0, 0.0, 0.0)):
    """Moves the slave node tangentially by ``amount`` along ``direction``."""

    U = np.zeros(nLoc)
    U[:nDim] = amount * np.asarray(direction)
    return U


class TestCoulombFriction(unittest.TestCase):
    def _checkTangent(self, friction, U, expectedState, h=1e-8):
        P, K = _forces(friction, U, U)
        self.assertEqual(friction.slipState[0], expectedState)
        numeric = np.zeros((nLoc, nLoc))
        for j in range(nLoc):
            step = np.zeros(nLoc)
            step[j] = h
            plus, _ = _forces(friction, U + step, U + step, computeTangent=False)
            minus, _ = _forces(friction, U - step, U - step, computeTangent=False)
            numeric[:, j] = -(plus - minus) / (2 * h)
        self.assertEqual(friction.slipState[0], expectedState, "state flipped under the perturbation")
        scale = np.max(np.abs(K))
        np.testing.assert_allclose(K / scale, numeric / scale, atol=1e-6)
        return P, K

    def test_the_stick_tangent_differentiates_the_residual(self):
        self._checkTangent(CoulombFriction(1, nDim, mu=10.0), _slide(1e-4, (1.0, 0.5, 0.2)), STICK)

    def test_the_slip_tangent_differentiates_the_residual(self):
        _, K = self._checkTangent(CoulombFriction(1, nDim, mu=0.3), _slide(1e-2, (1.0, 0.5, 0.2)), SLIP)
        self.assertGreater(np.max(np.abs(K - K.T)), 0.0, "the slip tangent must couple to the pressure")

    def test_the_slip_tolerance_tangents_differentiate_the_residual(self):
        stick = CoulombFriction(1, nDim, mu=0.3, slipTolerance=0.01)
        stick.setCharacteristicLength(1.0)
        self._checkTangent(stick, _slide(1e-3, (1.0, 0.5, 0.0)), STICK)
        slip = CoulombFriction(1, nDim, mu=0.3, slipTolerance=0.01)
        slip.setCharacteristicLength(1.0)
        self._checkTangent(slip, _slide(5e-2, (1.0, 0.5, 0.0)), SLIP)

    def test_the_frozen_tangent_differentiates_the_residual(self):
        friction = CoulombFriction(1, nDim, mu=0.3)
        _forces(friction, _slide(1e-2), _slide(1e-2))
        friction.slipStateFrozen = True
        friction.freezeDirections(np.array([True]))
        self._checkTangent(friction, _slide(1e-2, (1.0, 0.3, 0.0)), SLIP)

    def test_stick_is_elastic_and_slip_is_on_the_cone(self):
        friction = CoulombFriction(1, nDim, mu=0.3, tangentPenalty=2.0e3)
        _forces(friction, _slide(1e-4), _slide(1e-4))
        np.testing.assert_allclose(friction.tractionCurrent[0], [-2.0e3 * 1e-4, 0.0, 0.0])
        P, _ = _forces(friction, _slide(1.0), _slide(1.0))
        np.testing.assert_allclose(friction.tractionCurrent[0], [-0.3 * 10.0, 0.0, 0.0])
        # PExt = A c (x) t: the slave gets A t, the master nodes -A N_a t
        np.testing.assert_allclose(P.reshape(len(c), nDim), A * np.outer(c, friction.tractionCurrent[0]))

    def test_a_point_without_pressure_carries_nothing(self):
        friction = CoulombFriction(1, nDim, mu=0.3)
        friction.tractionConverged[0] = [1.0, 0.0, 0.0]
        P, K = friction.evaluatePoint(0, c, A, n, 0.0, np.zeros(nLoc), _slide(1e-3), epsN)
        np.testing.assert_array_equal(P, 0.0)
        np.testing.assert_array_equal(K, 0.0)
        self.assertEqual(friction.slipState[0], SLIP)

    def test_the_history_is_promoted_on_accept_only(self):
        friction = CoulombFriction(1, nDim, mu=10.0)
        _forces(friction, _slide(1e-4), _slide(1e-4))
        np.testing.assert_array_equal(friction.tractionConverged, 0.0)
        friction.accept()
        np.testing.assert_allclose(friction.tractionConverged, friction.tractionCurrent)
        self.assertTrue(friction.activeConverged[0])
        # Next increment, no further slip: the traction stays.
        before = friction.tractionCurrent.copy()
        _forces(friction, _slide(1e-4), np.zeros(nLoc))
        np.testing.assert_allclose(friction.tractionCurrent, before)

    def test_carry_rotates_on_smooth_surfaces_and_resets_at_creases(self):
        friction = CoulombFriction(3, nDim, mu=0.5)
        friction.tractionConverged[:] = [1.0, 0.0, 0.0]
        friction.pressureConverged[:] = 10.0
        friction.activeConverged[:] = True
        old = np.tile(n, (3, 1))
        smooth = np.array([np.sin(0.2), 0.0, np.cos(0.2)])
        crease = np.array([np.sin(1.0), 0.0, np.cos(1.0)])
        discarded = friction.carry(np.array([smooth, crease, smooth]), old, assigned=np.array([True, True, False]))
        self.assertEqual(discarded, 1)
        self.assertAlmostEqual(friction.tractionConverged[0] @ smooth, 0.0)
        self.assertAlmostEqual(np.linalg.norm(friction.tractionConverged[0]), np.cos(0.2))
        np.testing.assert_array_equal(friction.tractionConverged[1:], 0.0)

    def test_carry_with_a_gap_correction_stays_on_the_cone(self):
        friction = CoulombFriction(1, nDim, mu=0.5)
        friction.tractionConverged[0] = [4.0, 0.0, 0.0]
        friction.pressureConverged[0] = 10.0
        friction.activeConverged[0] = True
        friction.carry(np.array([n]), deltaGap=np.array([[1.0, 0.0, 0.0]]), normalPenalty=epsN)
        np.testing.assert_allclose(friction.tractionConverged[0], [5.0, 0.0, 0.0])

    def test_a_recurring_state_freezes_after_the_free_iterations(self):
        friction = CoulombFriction(2, nDim, mu=0.3)
        friction.tractionCurrent[:] = [0.0, 1.0, 0.0]
        for iteration, state in enumerate((STICK, SLIP, STICK, SLIP, STICK)):
            friction.slipState[:] = state
            friction.freezeIfChattering(iteration)
            self.assertEqual(friction.slipStateFrozen, iteration >= 3, f"iteration {iteration}")
        np.testing.assert_array_equal(friction.frozenSlip, True)

    def test_begin_evaluation_counts_iterations_per_attempt(self):
        class Step:
            def __init__(self, number, dt):
                self.number, self.timeIncrement, self.totalTime = number, dt, number * dt

        friction = CoulombFriction(1, nDim, mu=0.3)
        self.assertEqual([friction.beginEvaluation(Step(1, 0.1)) for _ in range(3)], [0, 1, 2])
        friction.slipStateFrozen = True
        self.assertEqual(friction.beginEvaluation(Step(1, 0.05)), 0, "a cutback is a new attempt")
        self.assertFalse(friction.slipStateFrozen)
        self.assertEqual(friction.beginEvaluation(None), 0)

    def test_restart_and_remap_keep_the_converged_history(self):
        friction = CoulombFriction(2, nDim, mu=0.3)
        friction.tractionConverged[:] = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
        friction.pressureConverged[:] = [5.0, 6.0]
        friction.activeConverged[:] = [True, False]
        restored = CoulombFriction(2, nDim, mu=0.3)
        restored.setRestartData({k: v.copy() for k, v in friction.restartData().items()})
        np.testing.assert_array_equal(restored.tractionConverged, friction.tractionConverged)
        np.testing.assert_array_equal(restored.activeConverged, friction.activeConverged)
        friction.remap(["a", "b"], ["b", "new", "a"])
        np.testing.assert_array_equal(friction.tractionConverged, [[0.0, 2.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        np.testing.assert_array_equal(friction.slipState, OPEN)

    def test_accept_reports_only_a_freeze_that_held_a_point_in_slip_inside_the_cone(self):
        free = CoulombFriction(1, nDim, mu=0.3)
        _forces(free, _slide(1e-2), _slide(1e-2))
        self.assertIsNone(free.acceptAndReport("point"), "an unfrozen state has nothing to report")

        frozen = CoulombFriction(1, nDim, mu=0.3)
        _forces(frozen, _slide(1e-2), _slide(1e-2))
        frozen.slipStateFrozen = True
        frozen.freezeDirections(np.array([True]))
        # A tiny slip: the stick predictor lies well inside the cone, but the point is held in slip.
        _forces(frozen, _slide(1e-6), _slide(1e-6))
        report = frozen.acceptAndReport("point")
        self.assertIsNotNone(report)
        self.assertIn("1 point(s)", report)

    def test_the_characteristic_length_is_fixed_once_and_reported(self):
        friction = CoulombFriction(1, nDim, mu=0.3, slipTolerance=0.01)
        self.assertIn("gamma_crit = 0.02", friction.setCharacteristicLength(2.0))
        self.assertIsNone(friction.setCharacteristicLength(5.0), "fixed once, later calls keep it")
        self.assertEqual(friction.elasticSlip, 0.02)
        for inactive in (CoulombFriction(1, nDim, mu=0.0, slipTolerance=0.01), CoulombFriction(1, nDim, mu=0.3)):
            self.assertIsNone(inactive.setCharacteristicLength(2.0))
            self.assertIsNone(inactive.elasticSlip)

    def test_options_are_validated(self):
        for options in ({"mu": -1.0}, {"mu": 0.3, "slipTolerance": -1.0}, {"mu": 0.3, "tangentPenalty": -1.0}):
            with self.assertRaises(ValueError):
                CoulombFriction(1, nDim, **options)
        with self.assertRaises(ValueError):
            CoulombFriction(1, nDim, mu=0.3, tangentPenalty=1.0, slipTolerance=0.005)
        with self.assertRaises(NotImplementedError):
            CoulombFriction(1, nDim, mu=0.3, slipTolerance=0.005).checkExplicitIntegrability()

    def test_characteristic_facet_size(self):
        self.assertAlmostEqual(characteristicFacetSize(16.0, 4, 3), 2.0)
        self.assertAlmostEqual(characteristicFacetSize(6.0, 3, 2), 2.0)


if __name__ == "__main__":
    unittest.main()
