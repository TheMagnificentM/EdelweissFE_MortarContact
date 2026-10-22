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
The plumbing a penalty contact constraint between facet surfaces needs around the Coulomb law it
shares with all contact formulations (:class:`~edelweissfe.utils.contactfriction.CoulombFriction`):
refusing an explicit integration the law cannot honour, counting the Newton iterations of the
anti-chattering rule, evaluating the law at a contact point, carrying the history into the new frozen
tangent planes at a connectivity update, promoting it on acceptance, and the per-point results.
The law itself and its state live in ``self._friction``; this mixin only connects it to the
constraint and to its journal.
"""

import numpy as np

from edelweissfe.timesteppers.timestep import TimeStep


class FrictionalContact:
    """Mixin for a contact constraint that holds its friction law as ``self._friction``, and has
    ``self.name`` and ``self.journal``. Its journal messages are sent under the class attribute
    ``identification`` where the constraint has one, else under its name.

    List it before :class:`~edelweissfe.constraints.base.forcesonlyexplicitevaluation.ForcesOnlyExplicitEvaluation`
    (or the constraint base), whose :meth:`applyConstraintExplicit` it extends.
    """

    def _frictionMessage(self, text: str, level: int):
        # A constraint with a class-level identification sends under it and names itself in the text;
        # one without sends under its own name, which then already identifies it.
        identification = getattr(self, "identification", None)
        if identification is None:
            self.journal.message(text, self.name, level=level)
        else:
            self.journal.message(f"{self.name}: {text}", identification, level=level)

    def applyConstraintExplicit(
        self,
        U_np: np.ndarray,
        dU: np.ndarray,
        PExt: np.ndarray,
        timeStep: TimeStep,
    ):
        """Reject a friction law the explicit path cannot integrate, then evaluate as usual."""

        self._friction.checkExplicitIntegrability()
        super().applyConstraintExplicit(U_np, dU, PExt, timeStep)

    def _beginFriction(self, timeStep: TimeStep):
        """Count the Newton iteration of the current increment attempt, at the start of an assembly
        (see :meth:`~edelweissfe.utils.contactfriction.CoulombFriction.beginEvaluation`)."""

        if self._friction.enabled:
            self._friction.beginEvaluation(timeStep)

    def _endFriction(self):
        """Apply the anti-chattering rule of the shared law, once all points of an assembly were
        evaluated (see :meth:`~edelweissfe.utils.contactfriction.CoulombFriction.freezeIfChattering`)."""

        if self._friction.enabled:
            self._friction.freezeIfChattering(self._friction.iteration)

    def _pointFriction(
        self,
        k: int,
        c: np.ndarray,
        area: float,
        normal: np.ndarray,
        normalForce: float,
        stiffness: float,
        w: np.ndarray,
        dU: np.ndarray,
        computeTangent: bool,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Friction at the contact point ``k`` of a penalty constraint: the point with the coefficients
        ``c``, the measure ``area`` and the frozen ``normal``, carrying the normal force ``normalForce``
        (negative in contact) with ``d normalForce / dg = stiffness`` along the gap gradient ``w``. Its
        pressure is ``p = -normalForce / area`` with ``dp/dU = -stiffness * w / area``."""

        return self._friction.evaluatePoint(
            k,
            c,
            area,
            normal,
            -normalForce / area,
            (-stiffness / area) * w,
            dU,
            self.penalty,
            computeTangent=computeTangent,
        )

    def _setFrictionCharacteristicLength(self, h: float):
        """Fix the elastic slip of ``slipTolerance`` from the characteristic facet size ``h``."""

        report = self._friction.setCharacteristicLength(h)
        if report:
            self._frictionMessage(f"mu = {self._friction.mu:.4g}, {report}", level=1)

    def _carryFrictionHistory(self, newNormals: np.ndarray, previousNormals: np.ndarray, assigned: np.ndarray):
        """Rotate the frictional history into the new frozen tangent planes at a connectivity update,
        and discard it where a point lost its facet or crossed a crease of the master surface (see
        :meth:`~edelweissfe.utils.contactfriction.CoulombFriction.carry`)."""

        if not self._friction.enabled:
            return
        nDiscarded = self._friction.carry(newNormals, previousNormals, assigned, normalPenalty=self.penalty)
        if nDiscarded:
            self._frictionMessage(f"{nDiscarded} point(s) crossed a crease, frictional history reset", level=2)

    def _acceptFrictionHistory(self):
        """Promote the tangential tractions of the converged iterate to the frictional history, and
        report what a frozen stick/slip state may have cost."""

        if not self._friction.enabled:
            return
        report = self._friction.acceptAndReport("contact point")
        if report:
            self._frictionMessage(report, level=1)

    def getTangentialTractions(self) -> np.ndarray:
        """The current tangential (frictional) traction magnitude at each contact point, ordered like
        :meth:`getNormalPressures`; zero without friction."""

        return self._friction.tractionMagnitudes()

    def getSlipStates(self) -> np.ndarray:
        """The current stick/slip state of each contact point as a float for output -- 0 open,
        1 stick, 2 slip -- ordered like :meth:`getNormalPressures`. Always 0 without friction."""

        return self._friction.slipStates()
