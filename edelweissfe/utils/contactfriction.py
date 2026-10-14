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

import numpy as np
from scipy.spatial import cKDTree

from edelweissfe.utils.schema import schemaField

"""
The penalty-regularised Coulomb friction law shared by all contact formulations of EdelweissFE --
node-to-surface (:mod:`~edelweissfe.constraints.nodetodeformablesurfacepenalty`), integrated
Gauss-point-to-surface (:mod:`~edelweissfe.constraints.surfacetodeformablesurfacepenalty`) and
mortar (:mod:`~edelweissfe.constraints.mortarcontact`).

WHAT IS SHARED AND WHAT IS NOT. A friction law is a statement about a POINT of the interface: given
the tangential relative displacement of that point within the increment, its normal pressure and
its frictional history, it returns a tangential traction. Nothing in it depends on how the point is
chosen or how its pressure is enforced. The three formulations differ only in what a "point" is and
how it is coupled to the nodes, and every one of them reduces to the same four ingredients:

    c      coefficients of the point's relative position over its local nodes,
           u_rel = sum_a c_a u_a   (slave nodes with +, master nodes with -),
    A      the interface measure the point stands for,
    n      the (frozen) normal of the point,
    p      its normal contact pressure (>= 0), with the gradient q = dp/dU_loc.

=================  ======================  ======================  ===============================
formulation        point                   c                       A
=================  ======================  ======================  ===============================
node-to-surface    slave node              [1, -N_1 .. -N_k]       tributary area of the node
Gauss-point        slave quadrature point  [N_s, -N_m]             quadrature weight x measure
mortar             non-mortar node I       [D_IK, -C_IJ] / |D_II|  |D_II|
=================  ======================  ======================  ===============================

With these, the slip increment of the point, its force on the nodes and the consistent tangent are

    du_T   = P_T (c (x) I) dU_loc                                  pointwise slip increment
    PExt  += A  c (x) t                                            nodal forces
    K     += -A (c c^T) (x) dt/du  -  A (c (x) dt/dp) q^T          K = -dPExt/dU

for any formulation, ``P_T = I - n n^T``. The law itself -- the stick predictor, the return onto the
cone, the derivatives dt/du and dt/dp -- and the bookkeeping around it (history, its transport into
a new tangent plane, the anti-chattering of the stick/slip state, restart) live in
:class:`CoulombFriction` and nowhere else.

THE LAW. Penalty-regularised Coulomb friction with an elastic predictor and a radial return
(Simo & Laursen 1992, Eqs. 3.9 and 3.12-3.14, pp. 103-104; Wriggers 2006, Section 10.5.2,
Eqs. 10.121-10.126 and Box 14, pp. 361-363; in the mortar setting Yang, Laursen & Meng 2005,
Eqs. 39 and 47-49, in 2D, and Puso, Laursen & Solberg 2008, Eqs. 16-21, in 3D):

    t_tr = P_T t_n - eps_T du_T                       stick predictor
    t    = t_tr                  if |t_tr| <= mu p    stick
           mu p t_tr / |t_tr|    otherwise            slip

The history ``t_n`` is the converged TRACTION (force per area), in global components, as in the
trial state of Simo & Laursen 1992, Eq. 3.12, and Puso et al. 2008, Eq. 19 (Wriggers 2006 keeps the
plastic slip instead, Eq. 10.121); the law is then independent of the size of the interface measure
each formulation assigns to a point. The tangential modulus ``eps_T`` is a traction per unit slip --
either constant (``tangentPenalty``; by default the formulation's normal penalty modulus, the one
parameter for normal contact and stick of Wriggers 2006, p. 236, and Puso et al. 2008, Eqs. 18-19)
or, with ``slipTolerance``, proportional to the pressure, ``eps_T = mu p / (slipTolerance h)`` with
``h`` the characteristic facet size of the slave surface: a sticking point then always slips
elastically by up to ``slipTolerance * h``, however lightly it is loaded. This is the SLIP
TOLERANCE of Abaqus/Standard, "the ratio of allowable maximum elastic slip to characteristic contact
surface face dimension", default 0.005 (Abaqus Keywords Reference, *FRICTION,
https://abaqus-docs.mit.edu/2017/English/SIMACAEKEYRefMap/simakey-r-friction.htm). A pressure-proportional tangential stiffness is the form of Wriggers 2006, Eq. 5.54, p. 92,
with exponent one; the definition of ``h`` is this implementation's.

A point without pressure carries no friction (Simo & Laursen 1992, p. 104: the algorithm requires
``t_T = 0`` if ``t_N = 0``) and contributes nothing to the tangent; it is reported as slipping (it
offers no resistance). A stick spring there would tie the surfaces together over a cone of radius
zero.
"""

#: Cosine of the largest change of a point's normal across one connectivity update over which its
#: frictional history is still rotated into the new tangent plane rather than discarded. The 30
#: degrees are chosen here, not taken from the contact literature: a smooth surface of curvature
#: radius R facetted with facet size h turns its normal by about h/R between neighbouring facets --
#: below 10 degrees for any surface resolved finely enough for a meaningful pressure distribution --
#: while the creases of engineering geometries (chamfers, steps, notch flanks) turn it by 45 degrees
#: and more.
cosOfSmoothNormalChangeLimit = np.cos(np.deg2rad(30.0))

#: Anti-chattering of the stick/slip state in the implicit iteration (see
#: :meth:`CoulombFriction.freezeIfChattering`; both numbers are chosen here): Newton iterations of an
#: increment that are always left alone, so that the predictor and the first corrections may move
#: the state freely ...
slipStateFreeIterations = 3
#: ... and the iteration from which ANY further change of the state freezes it, recurring or not.
slipStateFreezeIteration = 6

#: Stick/slip states of a contact point, as reported by :meth:`CoulombFriction.slipStates`.
OPEN, STICK, SLIP = 0, 1, 2


def frictionCoefficientField(remark: str = ""):
    """The schema field ``mu`` every frictional contact constraint declares, documented once."""

    return schemaField(
        description=(
            "The Coulomb friction coefficient. 0, the default, is frictionless. A positive value adds the "
            "penalty-regularised Coulomb law shared by all contact formulations (see "
            "edelweissfe.utils.contactfriction): an elastic (stick) predictor from the converged tangential "
            "traction and the tangential slip increment, and a radial return onto the cone |t_T| <= mu * p. "
            "The implicit tangent is non-symmetric on slip, which is inherent to Coulomb friction." + remark
        ),
        dtype=float,
        default=0.0,
    )


def tangentPenaltyField(normalPenaltyOption: str, remark: str = ""):
    """The schema field ``tangentPenalty``, documented once. ``normalPenaltyOption`` names the normal
    penalty option of the constraint it defaults to."""

    return schemaField(
        description=(
            "The tangential penalty modulus eps_T of the stick predictor, used only for mu > 0: a traction per "
            f"unit tangential slip, in the same unit as {normalPenaltyOption}. It sets the elastic slip a "
            f"sticking point accumulates, mu*p/eps_T at most. 0, the default, takes {normalPenaltyOption}." + remark
        ),
        dtype=float,
        default=0.0,
    )


def slipToleranceField():
    """The schema field ``slipTolerance``, documented once."""

    return schemaField(
        description=(
            "The elastic slip of the stick predictor as a fraction of the characteristic facet size h of the "
            "slave surface, used only for mu > 0 under the implicit solvers. A positive value replaces the "
            "constant tangentPenalty by the pressure-proportional eps_T = mu * p / (slipTolerance * h), so that "
            "a sticking point always slips elastically by up to slipTolerance * h, whatever its pressure -- "
            "Abaqus' penalty friction, whose default is 0.005. Recommended for real interfaces: with a constant "
            "eps_T the elastic range mu*p/eps_T of a lightly loaded point is far below one Newton correction, "
            "and the stick/slip state then chatters. 0, the default, keeps the constant modulus."
        ),
        dtype=float,
        default=0.0,
    )


def characteristicFacetSize(surfaceMeasure: float, nFacets: int, nDim: int) -> float:
    """The characteristic facet size ``h`` of a slave surface: the edge length of its mean facet,
    ``(measure / nFacets) ** (1 / (nDim - 1))``."""

    return (surfaceMeasure / nFacets) ** (1.0 / (nDim - 1))


def matchRestartKeys(storedKeys, currentKeys) -> tuple[np.ndarray, np.ndarray]:
    """Match the contact points of a checkpoint to the current ones by key -- a node label, or a
    tuple of node labels and a quadrature point index -- for a restart whose points need not be in
    the same order or of the same number (e.g. after a topology replay of an AMR run).

    Returns
    -------
    tuple
        ``(current, stored)`` index arrays: point ``current[i]`` receives entry ``stored[i]`` of the
        checkpoint. Points without a counterpart are left out.
    """

    storedIndex = {key: i for i, key in enumerate(storedKeys)}
    pairs = [(i, storedIndex[key]) for i, key in enumerate(currentKeys) if key in storedIndex]
    if not pairs:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    current, stored = zip(*pairs)
    return np.array(current, dtype=int), np.array(stored, dtype=int)


def historyInterpolation(
    newPoints: np.ndarray, newKeys: list, oldKeys: list, oldPatches: list, maxDistance: float
) -> dict:
    """How the frictional history of the points a mesh refinement created is interpolated from the
    points of the old surface.

    A refinement transfers the history of every point it keeps unchanged and has to interpolate it
    for every point it creates, with the shape functions of the old element that contains the new
    point (Wriggers 2006, Section 14.7.2, Eqs. 14.83-14.86, pp. 452-454; Peric et al. 1996,
    Section 4.1). Each old patch -- an old facet, or a parent face -- knows where a point lies in it
    and returns the interpolation weights over its own points.

    Parameters
    ----------
    newPoints
        ``(nNew, nDim)`` the reference coordinates of the new points.
    newKeys
        The keys of the new points; one that is also an old key is kept, not interpolated.
    oldKeys
        The keys of the old points, in the order of the old state arrays.
    oldPatches
        ``(centroid, locate)`` per old patch: ``locate(point)`` returns ``(distance, keys, weights)``
        -- the distance of the point from the patch, the keys of the old points the patch
        interpolates from, and their weights -- or ``None`` where the patch cannot locate the point.
    maxDistance
        A point farther than this from every patch is not interpolated (it starts at rest).

    Returns
    -------
    dict
        New point index -> ``(old point indices, weights)``, for the new points that were located.
        Weights of patch points without history are dropped and the rest renormalised.
    """

    oldIndex = {key: i for i, key in enumerate(oldKeys)}
    if not oldPatches:
        return {}
    centroids = np.array([centroid for centroid, _ in oldPatches])
    tree = cKDTree(centroids)
    nNeighbours = min(8, len(oldPatches))

    interpolation = {}
    for k, key in enumerate(newKeys):
        if key in oldIndex:
            continue
        _, nearest = tree.query(newPoints[k], k=nNeighbours)
        best = None
        for patch in np.atleast_1d(nearest):
            located = oldPatches[patch][1](newPoints[k])
            if located is not None and (best is None or located[0] < best[0]):
                best = located
        if best is None or best[0] > maxDistance:
            continue
        _, keys, weights = best
        pairs = [(oldIndex[key], weight) for key, weight in zip(keys, weights) if key in oldIndex]
        total = sum(weight for _, weight in pairs)
        if not pairs or abs(total) < 1e-12:
            continue
        interpolation[k] = (
            np.array([i for i, _ in pairs], dtype=int),
            np.array([weight for _, weight in pairs]) / total,
        )
    return interpolation


def frozenNormalsForRestart(frozenNormals: list, nDim: int) -> np.ndarray:
    """The frozen normals of the contact points as an ``(nPoints, nDim)`` array for a checkpoint,
    NaN where a point has none."""

    rows = np.full((len(frozenNormals), nDim), np.nan)
    for k, normal in enumerate(frozenNormals):
        if normal is not None:
            rows[k] = normal
    return rows


def restoreFrozenNormals(frozenNormals: list, stored: np.ndarray, current, storedIndex) -> None:
    """Write the checkpointed normals ``stored`` (see :func:`frozenNormalsForRestart`) back into the
    list ``frozenNormals``: point ``current[i]`` receives row ``storedIndex[i]``, ``None`` for NaN."""

    for k, j in zip(current, storedIndex):
        row = np.asarray(stored[j], dtype=float)
        frozenNormals[k] = None if np.isnan(row[0]) else row.copy()


class CoulombFriction:
    """The penalty-regularised Coulomb law and the frictional state of the contact points of one
    constraint.

    The owning constraint decides what its points are and supplies, per point, the coefficients
    ``c``, the measure ``A``, the normal ``n``, the pressure ``p`` and its gradient ``q`` (see the
    module documentation); this class returns the nodal forces and the tangent, and keeps the
    history.

    State per point (``nPoints`` rows):

    * the tangential traction of the current iterate and the converged one (the history of the
      stick predictor, promoted in :meth:`accept` only, so a cutback re-attempt -- which recomputes
      the slip from a fresh dU -- needs no reset),
    * the pressure and the contact status that went with the converged traction, which bound the
      history when it is carried into a new tangent plane (:meth:`carry`) before the new increment
      has assembled a pressure of its own,
    * the stick/slip state (0 open, 1 stick, 2 slip) and the anti-chattering bookkeeping.

    Parameters
    ----------
    nPoints
        The number of contact points.
    nDim
        The spatial dimension.
    mu
        The friction coefficient, >= 0; 0 disables friction.
    tangentPenalty
        The constant tangential modulus eps_T; 0 follows the normal modulus passed to
        :meth:`evaluate`.
    slipTolerance
        If positive, eps_T = mu p / (slipTolerance h) instead; h is set by
        :meth:`setCharacteristicLength`.
    owner
        The name of the owning constraint, for error messages.
    """

    def __init__(
        self,
        nPoints: int,
        nDim: int,
        mu: float,
        tangentPenalty: float | None = 0.0,
        slipTolerance: float = 0.0,
        owner: str = "",
    ):
        self.mu = float(mu)
        self.tangentPenalty = 0.0 if tangentPenalty is None else float(tangentPenalty)
        self.slipTolerance = float(slipTolerance)
        self.nDim = nDim
        self._owner = owner

        if self.mu < 0.0:
            raise ValueError(f"Contact '{owner}': the friction coefficient mu must not be negative.")
        if self.tangentPenalty < 0.0:
            raise ValueError(
                f"Contact '{owner}': tangentPenalty must not be negative; 0 takes the normal penalty modulus."
            )
        if self.slipTolerance < 0.0:
            raise ValueError(f"Contact '{owner}': slipTolerance must not be negative.")
        if self.slipTolerance > 0.0 and self.tangentPenalty > 0.0:
            raise ValueError(
                f"Contact '{owner}': tangentPenalty and slipTolerance both set the tangential modulus of the "
                "friction law. Give one of them."
            )

        #: The elastic slip gamma_crit = slipTolerance * h, once h is known (setCharacteristicLength).
        self.elasticSlip = None
        #: How many increments needed the stick/slip state frozen; a diagnostic only.
        self.slipStateFreezeCount = 0
        self._stepKey = None
        #: The Newton iteration within the current increment attempt, counted by beginEvaluation.
        self.iteration = 0
        self.resize(nPoints)

    @property
    def enabled(self) -> bool:
        """Whether friction is active at all (mu > 0)."""

        return self.mu > 0.0

    # ------------------------------------------------------------------ state

    def resize(self, nPoints: int):
        """(Re)allocate the state for ``nPoints`` points, all at rest without history."""

        nDim = self.nDim
        self.nPoints = nPoints
        self.tractionConverged = np.zeros((nPoints, nDim))
        self.tractionCurrent = np.zeros((nPoints, nDim))
        self.pressureConverged = np.zeros(nPoints)
        self.pressureCurrent = np.zeros(nPoints)
        self.activeConverged = np.zeros(nPoints, dtype=bool)
        self.slipState = np.zeros(nPoints, dtype=int)
        # Predictor magnitude, cone radius and alignment with a frozen direction of the last
        # assembly, for the admissibility check of a frozen stick/slip state in accept().
        self.trialMagnitude = np.zeros(nPoints)
        self.slipLimit = np.zeros(nPoints)
        self.trialAlignment = np.ones(nPoints)
        self.frozenDirection = np.zeros((nPoints, nDim))
        self.frozenSlip = np.zeros(nPoints, dtype=bool)
        self.startIncrementAttempt()

    def remap(self, oldKeys: list, newKeys: list, interpolation: dict | None = None):
        """Carry the converged history over to a new set of points, matched by key (e.g. the
        ``Node`` objects of a slave surface after a mesh refinement). A point whose key is new
        takes the history interpolated from the old points as given by ``interpolation`` (see
        :func:`historyInterpolation`) -- traction and pressure as the weighted sums, the contact
        status where the weighted status is more than one half -- or, without an entry there,
        starts at rest. The interpolated traction is projected onto the new tangent plane and
        returned onto the cone of the interpolated pressure by the next :meth:`carry`, since a
        transferred state need not satisfy the slip condition (Peric et al. 1996, Remark 4.3)."""

        oldIndex = {key: i for i, key in enumerate(oldKeys)}
        traction, pressure, active = self.tractionConverged, self.pressureConverged, self.activeConverged
        self.resize(len(newKeys))
        for i, key in enumerate(newKeys):
            j = oldIndex.get(key)
            if j is not None:
                self.tractionConverged[i] = traction[j]
                self.pressureConverged[i] = pressure[j]
                self.activeConverged[i] = active[j]
            elif interpolation is not None and i in interpolation:
                indices, weights = interpolation[i]
                self.tractionConverged[i] = weights @ traction[indices]
                self.pressureConverged[i] = max(0.0, float(weights @ pressure[indices]))
                self.activeConverged[i] = float(weights @ active[indices]) > 0.5
        self.tractionCurrent[:] = self.tractionConverged

    def markOpen(self, k):
        """Point(s) ``k`` are not in contact in this iterate: no traction, state open. The history
        is not touched here; it is replaced by the zero traction once the iterate is accepted."""

        self.tractionCurrent[k] = 0.0
        self.slipState[k] = OPEN
        self.pressureCurrent[k] = 0.0
        self.trialMagnitude[k] = 0.0
        self.slipLimit[k] = 0.0

    def setCharacteristicLength(self, h: float) -> float | None:
        """Fix the elastic slip ``slipTolerance * h`` (once; later calls keep the first value).
        Returns it, or ``None`` without slipTolerance."""

        if self.slipTolerance > 0.0 and self.elasticSlip is None:
            self.elasticSlip = self.slipTolerance * h
        return self.elasticSlip

    def tangentModulus(self, p, normalPenalty: float):
        """The pointwise tangential modulus eps_T at pressure ``p`` and its derivative d eps_T/dp.

        Constant (``tangentPenalty``, or ``normalPenalty`` when that is 0) by default; with
        ``slipTolerance``, ``mu * p / gamma_crit``.
        """

        if self.slipTolerance > 0.0:
            slope = self.mu / self.elasticSlip
            return slope * p, slope
        return (self.tangentPenalty if self.tangentPenalty > 0.0 else normalPenalty), 0.0

    def checkExplicitIntegrability(self):
        """Refuse what an explicit integration cannot honour: a pressure-dependent eps_T is not
        bounded by the stability limit the time step is checked against."""

        if self.enabled and self.slipTolerance > 0.0:
            raise NotImplementedError(
                f"Constraint '{self._owner}': slipTolerance cannot be integrated explicitly. Its tangential "
                "modulus grows with the contact pressure and is therefore not bounded by the stability limit "
                "the explicit time step is checked against. Use a constant tangentPenalty (or its default) "
                "instead."
            )

    # ------------------------------------------------------------------ the law

    def evaluate(
        self,
        k: np.ndarray,
        c: np.ndarray,
        A: np.ndarray,
        normals: np.ndarray,
        p: np.ndarray,
        dp_dU: np.ndarray,
        dU: np.ndarray,
        normalPenalty: float,
        pressureTolerance=0.0,
        computeTangent: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Friction at the points ``k``, all coupled to the same number of local nodes.

        Parameters
        ----------
        k
            ``(m,)`` indices of the points.
        c
            ``(m, nNodes)`` coefficients of the relative position over the local nodes.
        A
            ``(m,)`` interface measure of each point.
        normals
            ``(m, nDim)`` frozen unit normals.
        p
            ``(m,)`` normal pressures, >= 0.
        dp_dU
            ``(m, nNodes*nDim)`` gradients of the pressures with respect to the local DOFs.
        dU
            ``(m, nNodes*nDim)`` increments of the local DOFs since the last converged state, node
            by node, ``nDim`` components each.
        normalPenalty
            The normal penalty modulus, which eps_T follows when ``tangentPenalty`` is 0.
        pressureTolerance
            Scalar or ``(m,)``: a pressure at or below it counts as no pressure.
        computeTangent
            Whether to return the tangent.

        Returns
        -------
        tuple
            ``(m, nNodes*nDim)`` nodal forces to add to PExt, and the ``(m, nNodes*nDim,
            nNodes*nDim)`` tangent contributions ``-dPExt/dU`` to add to K (``None`` if not asked
            for). Points without pressure contribute zero rows.
        """

        k = np.asarray(k)
        m, nNodes = c.shape
        nDim = self.nDim
        nLoc = nNodes * nDim
        eye = np.eye(nDim)

        projector = eye - normals[:, :, None] * normals[:, None, :]
        slipLimit = self.mu * p
        self.pressureCurrent[k] = p
        self.slipLimit[k] = slipLimit

        forces = np.zeros((m, nLoc))
        tangent = np.zeros((m, nLoc, nLoc)) if computeTangent else None

        # No pressure: no friction, no tangent, reported as slipping (no resistance).
        loaded = p > pressureTolerance
        unloaded = k[~loaded]
        self.slipLimit[unloaded] = 0.0
        self.tractionCurrent[unloaded] = 0.0
        self.slipState[unloaded] = SLIP
        self.trialMagnitude[unloaded] = 0.0
        self.trialAlignment[unloaded] = 1.0
        if not np.any(loaded):
            return forces, tangent

        epsT, slope = self.tangentModulus(p, normalPenalty)
        epsT = np.broadcast_to(epsT, p.shape)

        relative = np.einsum("ia,iaj->ij", c, dU.reshape(m, nNodes, nDim))
        slip = np.einsum("ijk,ik->ij", projector, relative)
        trial = np.einsum("ijk,ik->ij", projector, self.tractionConverged[k]) - epsT[:, None] * slip
        trialMagnitude = np.linalg.norm(trial, axis=1)

        frozen = loaded & self.frozenSlip[k] if self.slipStateFrozen else np.zeros(m, dtype=bool)
        stick = loaded & ~frozen & (trialMagnitude <= slipLimit)
        slipping = loaded & ~frozen & ~stick

        traction = np.zeros((m, nDim))
        dt_du = np.zeros((m, nDim, nDim))
        dt_dp = np.zeros((m, nDim))
        alignment = np.ones(m)
        state = np.full(m, SLIP)

        if np.any(frozen):
            # Held in slip along the direction it had when frozen: t = mu p d.
            direction = np.einsum("ijk,ik->ij", projector[frozen], self.frozenDirection[k[frozen]])
            direction /= np.linalg.norm(direction, axis=1)[:, None]
            magnitude = trialMagnitude[frozen]
            alignment[frozen] = np.divide(
                np.einsum("ij,ij->i", trial[frozen], direction),
                magnitude,
                out=np.ones_like(magnitude),
                where=magnitude > 0.0,
            )
            traction[frozen] = slipLimit[frozen, None] * direction
            dt_dp[frozen] = self.mu * direction

        if np.any(stick):
            traction[stick] = trial[stick]
            dt_du[stick] = -epsT[stick, None, None] * projector[stick]
            dt_dp[stick] = -slope * slip[stick]
            state[stick] = STICK

        if np.any(slipping):
            # The slip tangent (Simo & Laursen 1992, p. 106; Wriggers 2006, Eqs. 8.39-8.45,
            # pp. 193-194): the derivative of the direction t_tr/|t_tr| carries the factor
            # mu p/|t_tr|, which the printed Eqs. 8.42 and 8.45 omit; composing their Eqs. 8.39 and
            # 8.40 restores it. The dt/dp term is what makes the tangent non-symmetric (Wriggers 2006,
            # p. 359, with Eq. 9.47).
            direction = trial[slipping] / trialMagnitude[slipping, None]
            radial = eye - direction[:, :, None] * direction[:, None, :]
            scale = slipLimit[slipping] / trialMagnitude[slipping]
            traction[slipping] = slipLimit[slipping, None] * direction
            dt_du[slipping] = -(scale * epsT[slipping])[:, None, None] * (radial @ projector[slipping])
            dt_dp[slipping] = self.mu * direction - (scale * slope)[:, None] * np.einsum(
                "ijk,ik->ij", radial, slip[slipping]
            )

        kLoaded = k[loaded]
        self.tractionCurrent[kLoaded] = traction[loaded]
        self.slipState[kLoaded] = state[loaded]
        self.trialMagnitude[kLoaded] = trialMagnitude[loaded]
        self.trialAlignment[kLoaded] = alignment[loaded]

        # PExt += A c (x) t
        forces[loaded] = (A[loaded, None, None] * c[loaded, :, None] * traction[loaded, None, :]).reshape(-1, nLoc)

        if computeTangent:
            # K = -A (c c^T) (x) dt/du - A (c (x) dt/dp) q^T
            cl, Al = c[loaded], A[loaded]
            KUU = np.einsum("ia,ib,ijk->iajbk", cl, cl, dt_du[loaded]).reshape(-1, nLoc, nLoc)
            cdp = (cl[:, :, None] * dt_dp[loaded, None, :]).reshape(-1, nLoc)
            tangent[loaded] = -Al[:, None, None] * (KUU + cdp[:, :, None] * dp_dU[loaded, None, :])

        return forces, tangent

    def evaluatePoint(
        self,
        k: int,
        c: np.ndarray,
        A: float,
        n: np.ndarray,
        p: float,
        dp_dU: np.ndarray,
        dU: np.ndarray,
        normalPenalty: float,
        pressureTolerance: float = 0.0,
        computeTangent: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """:meth:`evaluate` for the single point ``k``; returns its force vector and tangent block."""

        forces, tangent = self.evaluate(
            np.array([k]),
            np.asarray(c, dtype=float)[None, :],
            np.array([A], dtype=float),
            np.asarray(n, dtype=float)[None, :],
            np.array([p], dtype=float),
            np.asarray(dp_dU, dtype=float)[None, :],
            np.asarray(dU, dtype=float)[None, :],
            normalPenalty,
            pressureTolerance,
            computeTangent,
        )
        return forces[0], (tangent[0] if tangent is not None else None)

    # ------------------------------------------------------------------ anti-chattering

    def startIncrementAttempt(self):
        """The stick/slip state is free again in every increment attempt."""

        self.slipStateFrozen = False
        self.frozenSlip[:] = False
        self._seenSlipStates = set()
        self._lastSlipState = None

    def beginEvaluation(self, timeStep) -> int:
        """Count the Newton iteration of the increment attempt ``timeStep`` identifies, and free the
        stick/slip state when a new attempt begins (a cutback keeps the number but changes the
        increment size). Returns the iteration. Without a time step every call is a fresh attempt.

        For formulations that keep no iteration count of their own; the mortar constraint counts
        its iterations itself and calls :meth:`startIncrementAttempt` directly.
        """

        key = None if timeStep is None else (timeStep.number, timeStep.timeIncrement, timeStep.totalTime)
        if key is None or key != self._stepKey:
            self._stepKey = key
            self.iteration = 0
            self.startIncrementAttempt()
        else:
            self.iteration += 1
        return self.iteration

    def freezeIfChattering(self, iteration: int):
        """Freeze the stick/slip state of this increment once it chatters. Called once per assembly,
        after all points were evaluated.

        Coulomb friction is a non-smooth law, and a Newton iteration on it can flip points near the
        boundary of the cone back and forth between stick and slip without ever settling -- the
        jamming or zig-zagging that Wriggers 2006, Remark 10.2(1), p. 335, describes for the normal
        active set, where he keeps a constraint active until the new set is a subset of the old one.
        The same idea of a state that may only move one way is applied here to stick and slip; the
        rule, with its two iteration counts chosen here, is applied per increment attempt and reset
        with it:

        * the first ``slipStateFreeIterations`` (3) iterations are always free: the predictor and
          the first corrections legitimately move many points at once;
        * after that, a state that RECURS after a change -- a proven cycle, since there are only
          finitely many states -- freezes it at once;
        * from iteration ``slipStateFreezeIteration`` (6) on, any change at all freezes it.

        Frozen means: a slipping point keeps slipping, in the direction of its traction at the
        moment of freezing; a sticking point may still go to slip, and its direction is frozen with
        it. Stick to slip is the safe direction of change -- a frozen stick could carry a traction
        outside the cone -- and because the state can then only move one way, the iteration
        terminates. Holding the direction and not merely the state keeps the frozen law smooth: a
        point held in slip with a free direction is the unregularised ``mu p sign(slip)``, on which
        Newton does not converge. What the freeze may cost is measured in :meth:`accept`.
        """

        if self.slipStateFrozen:
            # Monotone from here: a point that has gone to slip under the freeze stays there too,
            # in the direction it went with.
            self.freezeDirections((self.slipState == SLIP) & ~self.frozenSlip)
            return

        state = self.slipState.tobytes()
        changed = self._lastSlipState is not None and state != self._lastSlipState
        cycle = changed and state in self._seenSlipStates
        self._seenSlipStates.add(state)
        self._lastSlipState = state

        if (cycle and iteration >= slipStateFreeIterations) or (changed and iteration >= slipStateFreezeIteration):
            self.slipStateFrozen = True
            self.frozenSlip[:] = False
            self.freezeDirections(self.slipState == SLIP)
            self.slipStateFreezeCount += 1

    def freezeDirections(self, points: np.ndarray):
        """Hold the given slipping points in slip along their current traction direction. A point
        whose traction is zero -- no pressure -- has no direction to hold and stays free."""

        magnitude = np.linalg.norm(self.tractionCurrent, axis=1)
        points = points & (magnitude > 0.0)
        self.frozenDirection[points] = self.tractionCurrent[points] / magnitude[points, None]
        self.frozenSlip |= points

    # ------------------------------------------------------------------ history

    def accept(self) -> tuple[int, int]:
        """Promote the tractions of the converged iterate to the history.

        If the stick/slip state was frozen in this increment, the accepted state is checked for what
        the freeze may have cost. Returns the number of points held in slip whose predictor lies
        inside the cone (they should have stuck) and of those whose predictor points more than 30
        degrees away from the frozen direction; the owner reports them.
        """

        self.tractionConverged[:] = self.tractionCurrent
        self.pressureConverged[:] = self.pressureCurrent
        self.activeConverged[:] = self.slipState > OPEN

        if not self.slipStateFrozen:
            return 0, 0
        held = self.frozenSlip & (self.slipState == SLIP) & (self.slipLimit > 0.0)
        insideCone = held & (self.trialMagnitude < (1.0 - 1e-3) * self.slipLimit)
        turnedAway = held & (self.trialAlignment < cosOfSmoothNormalChangeLimit)
        return int(np.sum(insideCone)), int(np.sum(turnedAway))

    def carry(
        self,
        normals: np.ndarray,
        previousNormals: np.ndarray | None = None,
        assigned: np.ndarray | None = None,
        deltaGap: np.ndarray | None = None,
        normalPenalty: float = 0.0,
    ) -> int:
        """Carry the accepted history into a new tangent frame, at a connectivity update.

        * A point without a contact partner (``assigned`` False) loses its history.
        * A point whose normal turned by more than 30 degrees -- a crease of the master surface --
          loses it too: projecting it would drop the part that now points along the new normal and
          leave a predictor unrelated to the frictional state. It enters like a fresh contact,
          which is harmless where it matters (a sticking point does not cross a crease, a slipping
          one is returned onto the cone within the increment).
        * Every other point that was in contact at the accepted state keeps its traction,
          optionally corrected by ``eps_T * deltaGap`` (the objective slip correction of a
          formulation whose coupling operator changes at the update, see the mortar constraint),
          projected onto the new tangent plane and returned onto the cone of the accepted pressure.

        Parameters
        ----------
        normals
            ``(nPoints, nDim)`` the new normals (rows of unassigned points are ignored).
        previousNormals
            ``(nPoints, nDim)`` the normals before the update; ``None``, or NaN rows, where unknown --
            such points are rotated without the crease test.
        assigned
            ``(nPoints,)`` whether each point has a contact partner; ``None`` means all.
        deltaGap
            ``(nPoints, nDim)`` the change of the pointwise gap vector across the update; NaN rows
            leave that point as it is.
        normalPenalty
            The normal modulus eps_T follows when ``tangentPenalty`` is 0.

        Returns
        -------
        int
            The number of points whose history was discarded at a crease.
        """

        nPoints = self.nPoints
        assigned = np.ones(nPoints, dtype=bool) if assigned is None else np.asarray(assigned, dtype=bool)
        self.tractionConverged[~assigned] = 0.0

        turned = np.zeros(nPoints, dtype=bool)
        if previousNormals is not None:
            known = assigned & np.isfinite(previousNormals[:, 0])
            turned[known] = np.einsum("ij,ij->i", previousNormals[known], normals[known]) < cosOfSmoothNormalChangeLimit
        nDiscarded = int(np.sum(turned & self.activeConverged & (np.linalg.norm(self.tractionConverged, axis=1) > 0)))
        self.tractionConverged[turned] = 0.0

        carried = assigned & self.activeConverged & ~turned
        if deltaGap is not None:
            carried &= np.isfinite(deltaGap[:, 0])

        for k in np.flatnonzero(carried):
            n = normals[k]
            traction = self.tractionConverged[k].copy()
            if deltaGap is not None:
                epsT, _ = self.tangentModulus(self.pressureConverged[k], normalPenalty)
                traction += epsT * deltaGap[k]
            traction -= (traction @ n) * n
            limit = self.mu * self.pressureConverged[k]
            magnitude = float(np.linalg.norm(traction))
            if magnitude > limit:
                traction *= limit / magnitude
            self.tractionConverged[k] = traction

        return nDiscarded

    def restartData(self) -> dict[str, np.ndarray]:
        """The converged frictional history, for a checkpoint."""

        return {
            "frictionTractionConverged": self.tractionConverged,
            "frictionPressureConverged": self.pressureConverged,
            "frictionActiveConverged": self.activeConverged.astype(np.int8),
        }

    def setRestartData(self, data: dict[str, np.ndarray], current=None, stored=None):
        """Restore the converged frictional history from :meth:`restartData`. A checkpoint that holds
        only the traction leaves pressure and contact status at rest.

        ``current`` and ``stored`` optionally select which point of this state receives which entry
        of the checkpoint (see :func:`matchRestartKeys`); by default the two are taken to be in the
        same order and of the same size.
        """

        if current is None:
            current = stored = slice(None)
        traction = np.asarray(data["frictionTractionConverged"])
        self.tractionConverged[current] = traction[stored]
        self.tractionCurrent[current] = traction[stored]
        if "frictionPressureConverged" in data:
            pressure = np.asarray(data["frictionPressureConverged"])
            self.pressureConverged[current] = pressure[stored]
            self.pressureCurrent[current] = pressure[stored]
            self.activeConverged[current] = np.asarray(data["frictionActiveConverged"])[stored] > 0.5

    # ------------------------------------------------------------------ output

    def tractionMagnitudes(self) -> np.ndarray:
        """``|t|`` per point, as of the last assembly."""

        return np.linalg.norm(self.tractionCurrent, axis=1)

    def slipStates(self) -> np.ndarray:
        """The stick/slip state per point as of the last assembly, as a float for output: 0 open,
        1 stick, 2 slip."""

        return self.slipState.astype(float)
