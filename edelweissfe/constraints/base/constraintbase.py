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

from abc import ABC, abstractmethod

import numpy as np

from edelweissfe.models.femodel import FEModel
from edelweissfe.numerics.vijentitybase import VIJEntityBase
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.variables.scalarvariable import ScalarVariable


class ConstraintBase(ABC, VIJEntityBase):
    @abstractmethod
    def __init__(self, name: str, model: FEModel, *args, **kwargs):
        """The constraint base class.

        Constraints can act on nodal variables, and scalar variables.
        If scalar variables are required, the can be created on demand by
        defining
        :func:`~ConstraintBase.getNumberOfAdditionalNeededScalarVariables` and
        :func:`~ConstraintBase.assignAdditionalScalarVariables`, which are called at the beginning of an analysis.

        If scalar variables are used, EdelweissFE expects the layout of the external load vector PExt
        (and the stiffness) to be of the form

        .. code-block:: console

            [ node 1 - dofs field 1,
              node 1 - dofs field 2,
              node 1 - ... ,
              node 1 - dofs field n,
              node 2 - dofs field 1,
              ... ,
              node N - dofs field n,
              scalar variable 1,
              scalar variable 2,
              ... ,
              scalar variable J ].

        Parameters
        ----------
        name
            The name of the constraint.
        model
            A dictionary containing the model tree.
        kwargs
            Key value pairs.
        """

        self.scalarVariables = []

    @property
    @abstractmethod
    def nodes(self) -> list:
        """The nodes this constraint is acting on."""

    @property
    @abstractmethod
    def fieldsOnNodes(self) -> list:
        """The fields on the nodes this constraint is acting on."""

    @property
    @abstractmethod
    def nDof(self) -> int:
        """The total number of degrees of freedom this constraint is associated with."""

    def getNumberOfAdditionalNeededScalarVariables(
        self,
    ) -> int:
        """This method is called to determine the additional number of scalar variables
        this Constraint needs.

        Returns
        -------
        int
            Number of ScalarVariables required.

        """

        return 0

    def assignAdditionalScalarVariables(self, scalarVariables: list[ScalarVariable]):
        """Assign a list of scalar variables associated with this constraint.

        Parameters
        ----------
        list
            The list of ScalarVariables associated with this constraint.

        """

        self.scalarVariables = scalarVariables

    def requiresCorrectionBeforeConvergence(self) -> bool:
        """May the increment be declared converged on the state as it stands?

        A constraint that changed its own internal state since the last Newton
        correction has invalidated the equilibrium the solver is about to accept -
        the assembled forces are not the ones that produced the current
        displacements. The typical case is a nested scheme whose multiplier
        estimate is warm-started at the beginning of an increment.

        This matters because the state under test may never have been corrected at
        all: at iteration 0 there is no correction yet, and a convergence criterion
        cannot distinguish "the correction is small" from "there is no correction".

        Constraints that are fully enforced by the equations of the global system -
        every Lagrange-multiplier constraint - inherit the no-op below.

        Returns
        -------
        bool
            True if at least one Newton correction has to be taken before the
            convergence of the increment may be tested.

        """

        return False

    def augmentConstraint(self) -> bool:
        """Perform one outer (augmentation) iteration once the Newton loop of the
        current increment has converged.

        This is the hook for nested solution strategies in which the constraint is
        enforced by an OUTER loop around the equilibrium iteration - most notably
        the augmented Lagrangian (Uzawa) scheme, where the multiplier estimate is
        updated from the converged constraint violation and equilibrium is then
        re-established (Puso, Laursen & Solberg 2008, Eq. (18): the augmented
        Lagrangian counter is advanced only "once convergence of the Newton-Raphson
        loop is achieved").

        Constraints that are fully enforced within the Newton loop - every
        constraint using Lagrange multipliers as unknowns, and pure penalty - do
        not need this and inherit the no-op below.

        Returns
        -------
        bool
            True if the constraint updated its internal state and requires the
            equilibrium iteration to be resumed, False if it is satisfied.

        """

        return False

    @abstractmethod
    def applyConstraint(
        self,
        U_np: np.ndarray,
        dU: np.ndarray,
        PExt: np.ndarray,
        V: np.ndarray,
        timeStep: TimeStep,
    ):
        """Apply the constraint.  Add the contributions to the external load vector and the system matrix.

        Parameters
        ----------
        U_np
            The current total solution vector.
        dU
            The current increment since the last time the constraint was applied.
        PExt
            The external load vector.
        K
            The system (stiffness) matrix.
        dT
            The time increment.
        time
            The current step and total time.
        """
