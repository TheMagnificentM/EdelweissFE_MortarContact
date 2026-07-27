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

    def snapshotState(self) -> dict:
        """Capture all mutable state of this constraint so a side-effect-free
        (residual-only) re-evaluation of :func:`applyConstraint` can be rolled back.

        Used by the solver's opt-in line-search globalisation: the merit function is
        evaluated by re-assembling the residual at trial displacements *within* a
        single Newton iteration, and those trial evaluations must not advance any
        per-iteration bookkeeping (e.g. a contact active-set / anti-cycling state
        machine).

        numpy arrays, sets and dicts are copied because :func:`applyConstraint` may
        mutate them in place; scalars/immutables and large per-increment-frozen
        objects (e.g. the model) are kept by reference (they are not mutated within an
        increment). A constraint without mutable state simply snapshots references.

        Returns
        -------
        dict
            An opaque snapshot to be passed to :func:`restoreState`.
        """

        snapshot = {}
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                snapshot[key] = value.copy()
            elif isinstance(value, (set, dict)):
                snapshot[key] = value.copy()
            else:
                snapshot[key] = value

        return snapshot

    def restoreState(self, snapshot: dict):
        """Restore a snapshot produced by :func:`snapshotState`.

        Parameters
        ----------
        snapshot
            The snapshot returned by :func:`snapshotState`.
        """

        self.__dict__.update(snapshot)

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
