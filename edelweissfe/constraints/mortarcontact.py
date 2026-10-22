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
"""
A unilateral contact constraint between two deformable surfaces, enforced segment by segment with a
dual mortar method. Both surfaces are element sets of geometry-only contact elements
(:mod:`~edelweissfe.elements.contactelement.element`, typically created by
:mod:`~edelweissfe.generators.surfaceelementgenerator` with ``facets=wholeFace``). The non-mortar
(slave) side carries the constraint; the mortar (master) side is projected onto it.

Instead of a gap per node, the non-penetration condition is integrated over the overlap of the two
surfaces against the dual basis function of each non-mortar node, which gives one WEIGHTED gap per
node. This is what lets the method transfer a constant pressure across non-matching meshes. The
nodal mortar weight ``D_II`` (the integral of the node's basis function) converts between the two
worlds: a weighted gap divided by ``D_II`` is the physical opening, a multiplier times ``D_II`` a
nodal force. The dual basis keeps the slave coupling matrix diagonal for linear elements, so a
multiplier belongs to exactly one node.

Sign convention: ``lambda_I`` is the negative normal traction of the non-mortar surface, i.e. a
pressure, positive in compression. Where ``D_II`` is negative (possible for quadratic faces under
partial coverage) the pressure is ``lambda_I * sgn(D_II)``.

The Signorini conditions are enforced through the complementarity function::

    C_n,I = p_I - max(0, p_I - c_n * g_I / D_II) = 0

which a semi-smooth Newton method (primal-dual active set strategy) solves by re-deciding the active
set in every iteration. Three ways to enforce the same discrete condition share the whole geometric
pipeline (search, projection, clipping, segmentation, nodal normals, coupling matrices):

``formulation=lagrange``
    The multipliers are unknowns of the global system (saddle point); the condition holds exactly.
``formulation=penalty``
    The pressure is ``penaltyStiffness`` times the pointwise penetration.
``formulation=penalty`` with ``augmentedLagrange``
    As penalty, with the pressure corrected in an outer (Uzawa) loop after each converged equilibrium.

Coulomb friction (``mu > 0``) is available for ``formulation=penalty``, implicit and explicit, with
the law shared by all contact formulations (:mod:`~edelweissfe.utils.contactfriction`).

The geometry -- segmentation, coupling matrices, nodal normals -- is built once per increment at
the last converged configuration (``geometryUpdate=frozen``, the default, and always under explicit
dynamics) or at every Newton iterate with a tangent that omits its derivatives
(``geometryUpdate=incomplete``).

Results per non-mortar node (:meth:`Constraint.getNormalPressures`,
:meth:`Constraint.getTangentialTractions`, :meth:`Constraint.getSlipStates`,
:meth:`Constraint.getGaps`) are ordered like the node set ``<name>_nonMortarNodes`` the constraint
registers, so they can be written as a ``fromExpression`` field output over that set::

    *fieldOutput
    >>fromExpression, name=pN, nSet=contact_nonMortarNodes,
    expression='model.constraints["contact"].getNormalPressures()'

The theory, the sources and the deviations from them are documented on
:doc:`the mortar theory page </documentation/mortartheory>`.
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.elements.contactelement.element import ContactElement
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.models.meshdependent import MeshDependent
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.contactfriction import (
    HISTORY_DISTANCE_FACTOR,
    CoulombFriction,
    characteristicFacetSize,
    frictionCoefficientField,
    historyInterpolation,
    matchRestartKeys,
    slipToleranceField,
    tangentPenaltyField,
)
from edelweissfe.utils.mortarcoupling import SEARCH_MARGIN_FACTOR, computeMortarCoupling
from edelweissfe.utils.mortargeometry import curvedFacetHistoryPatches, extent
from edelweissfe.utils.schema import buildSchemaFromOptions, schemaField

#: Relative noise floor of the mortar quantities (entries of D and C, nodal weights, weighted gaps):
#: values below ``RELATIVE_NOISE_FLOOR`` times the largest of their kind count as zero. Relative because
#: these quantities carry the unit of an area (times a length for the gap), so an absolute bound would
#: depend on the length unit of the model. Chosen here, not taken from a source: four orders above
#: the round-off ``eps * max|A|`` and far below any weight of a covered node.
RELATIVE_NOISE_FLOOR = 1e-12

#: Factor C of the default penalty stiffness ``eps_N = C E / h``. The literature gives the scale, not a
#: value: Wriggers (2006), Remark 6.4, pp. 119-120, finds a penalty of at least 100 times the axial
#: stiffness necessary and recommends choosing it as a function of the mesh size; Ghesquiere-Dierickx
#: et al. (2026), Eq. (12), state the guideline ``k = alpha E / h`` with alpha from 10^1 to 10^4.
PENALTY_STIFFNESS_FACTOR = 100.0

#: Fallback of c_n and eps_N where no Young's modulus or no contact area is available. It carries a
#: unit (suited to a model in MPa and mm) and is reported whenever it is used.
FALLBACK_MODULUS = 1.0e6

#: The active set of the semi-smooth Newton method is frozen for the rest of an increment once it has
#: been unchanged for this many consecutive iterations (and at least this many iterations were made),
#: or once a state recurs after a change, or at the iteration cap below. Chosen here.
ACTIVE_SET_SETTLE_ITERATIONS = 2
ACTIVE_SET_ITERATION_CAP = 20


@dataclass(frozen=True)
class MortarContactSchema:
    """The options this constraint accepts, owned by this module and never mutated from outside
    it.

    Name resolution, case folding, type coercion and defaults are all applied by
    :func:`~edelweissfe.utils.schema.buildSchemaFromOptions`, so an option is spelled once, here,
    and nowhere else.
    """

    nonMortarSurface: str | None = schemaField(
        description=(
            "The element set of contact elements (CONLINE2/CONLINE3 in 2D, CONQUAD4/CONQUAD8/CONQUAD9/CONTRI3/"
            "CONTRI6 in 3D) forming the non-mortar (slave) surface, e.g. the '<name>_facets' set of a "
            "surfaceElementGenerator run with facets=wholeFace. The constraint is enforced on this side: its "
            "nodes carry the weighted gap and the contact pressure, and the mortar side is projected onto it."
        ),
        dtype=str,
        default=None,
        required=True,
    )

    mortarSurface: str | None = schemaField(
        description=(
            "The element set of contact elements forming the mortar (master) surface. It carries no "
            "unknowns of its own; the contact forces reach it through the projection onto the non-mortar "
            "side. The two sets must be disjoint."
        ),
        dtype=str,
        default=None,
        required=True,
    )

    field: str = schemaField(
        description=("The field this constraint acts on (e.g. displacement)."),
        dtype=str,
        default="displacement",
    )

    cn: float = schemaField(
        description=(
            "Complementarity parameter of the semi-smooth activation test s_n = p_n - cn * g_n, used only for "
            "formulation=lagrange; g_n is the weighted gap divided by the nodal mortar weight, i.e. the "
            "physical opening at the node, and a node is in contact where s_n is positive. It is purely "
            "algorithmic: the converged solution does not depend on it, only the path to it. Its unit is a "
            "stress per length, so a value that suits a model in metres does not suit the same model in "
            "millimetres. Too small and the active set does not settle, far too large and it oscillates. Leave "
            "it at 0, the default, to take the smallest Young's modulus among the materials adjacent to the two "
            "surfaces; the value used is reported."
        ),
        dtype=float,
        default=0.0,
    )

    formulation: str = schemaField(
        description=(
            "How the non-penetration condition is enforced. 'lagrange', the default, keeps the multipliers "
            "as unknowns of the global system (a saddle-point problem) and satisfies the condition exactly at "
            "convergence. 'penalty' replaces them by a pressure proportional to the penetration, which "
            "satisfies the condition only approximately -- unless augmentedLagrange is switched on as well. "
            "Both share the entire geometric pipeline and differ in the constraint law alone."
        ),
        dtype=str,
        default="lagrange",
    )

    geometryUpdate: str = schemaField(
        description=(
            "How the mortar geometry -- segmentation, coupling matrices D and C, nodal normals -- follows the "
            "deformation in an implicit analysis. 'frozen', the default, builds it once per increment at the "
            "last converged configuration and holds it fixed; the tangent is exact for that residual, and the "
            "result carries an error of first order in the increment size. 'incomplete' rebuilds it at every "
            "Newton iterate and holds it fixed within each Newton step: the converged state satisfies the "
            "contact conditions in its own configuration, but the tangent omits the derivatives of the "
            "normals and of D and C (Hartmann 2007; the incomplete linearization of Popp, Gitterle, Gee & "
            "Wall 2010), which costs the quadratic convergence of the iteration. Explicit dynamics always uses "
            "the frozen geometry, rebuilt every contact-update-frequency increments."
        ),
        dtype=str,
        default="frozen",
    )

    penaltyStiffness: float = schemaField(
        description=(
            "Penalty stiffness eps_N of formulation=penalty: the pressure is eps_N times the pointwise "
            "penetration. It is the classical pointwise penalty parameter, a pressure per length, in the same "
            "unit as the penalty of the node- and segment-based penalty constraints. Leave it at 0, the "
            "default, to derive it as 100 times the smallest adjacent Young's modulus divided by the "
            "characteristic face size (under explicit dynamics, as the largest stiffness the time increment "
            "can carry); the value used is reported. With augmentedLagrange the converged result does not "
            "depend on it; it then only sets how fast the outer loop converges."
        ),
        dtype=float,
        default=0.0,
    )

    augmentedLagrange: bool = schemaField(
        description=(
            "Add an outer correction loop to formulation=penalty: after each converged equilibrium the nodal "
            "pressure estimate is advanced by the penalty stiffness times the remaining penetration, and "
            "equilibrium is re-established. The converged solution then satisfies the condition to the "
            "augmentation tolerance and no longer depends on the penalty stiffness. Implicit solvers only. "
            "Ignored for formulation=lagrange."
        ),
        dtype=bool,
        default=False,
    )

    augmentationTolerance: float = schemaField(
        description=(
            "Termination tolerance of the augmented outer loop, as the relative change of the nodal pressure "
            "estimate from one augmentation to the next: max|p_new - p_old| <= tol * max|p_new|. Relative, "
            "and therefore independent of the length unit of the model. Where the penalty term alone already "
            "satisfies the condition to rounding, the ratio stops decreasing and the loop runs to "
            "maxAugmentations."
        ),
        dtype=float,
        default=1e-06,
    )

    maxAugmentations: int = schemaField(
        description=(
            "Upper limit of the augmented-Lagrangian outer iterations per increment. Reaching it is reported: "
            "the increment then converged on a pressure estimate that had not settled."
        ),
        dtype=int,
        default=10,
    )

    mu: float = frictionCoefficientField(
        " Used only for formulation=penalty, at every non-mortar node, with the mortar-weighted "
        "tangential slip (Yang, Laursen & Meng 2005; Puso, Laursen & Solberg 2008); implicit and explicit."
    )

    tangentPenalty: float = tangentPenaltyField(
        "penaltyStiffness",
        " The default follows eps_N as resolved, including its derivation from the stability limit under "
        "explicit dynamics.",
    )

    slipTolerance: float = slipToleranceField()


def _storedRows(A: csr_matrix, nRows: int, rtol: float = RELATIVE_NOISE_FLOOR) -> tuple[list, list]:
    """Per row, the column indices and values of every entry a CSR matrix stores, with the values
    below ``rtol * max|A|`` set to zero. A stored entry below the floor is kept with the value zero,
    which leaves every sum it enters unchanged; the floor decides which coupling terms enter the
    weighted gap and the stiffness."""

    tol = rtol * np.max(np.abs(A.data)) if A.data.size else 0.0
    data = np.where(np.abs(A.data) > tol, A.data, 0.0)
    return (
        [A.indices[A.indptr[I] : A.indptr[I + 1]] for I in range(nRows)],  # noqa: E741
        [data[A.indptr[I] : A.indptr[I + 1]] for I in range(nRows)],  # noqa: E741
    )


def _storedVectorRows(components: list, nRows: int, rtol: float = RELATIVE_NOISE_FLOOR) -> list:
    """Per row, the ``(len(row), dim)`` values of a vector-valued matrix given as one CSR matrix per
    component, all of the same pattern; small values set to zero as in :func:`_storedRows`."""

    scale = max((float(np.max(np.abs(A.data))) for A in components if A.data.size), default=0.0)
    columns = []
    for A in components:
        data = np.where(np.abs(A.data) > rtol * scale, A.data, 0.0)
        columns.append([data[A.indptr[I] : A.indptr[I + 1]] for I in range(nRows)])  # noqa: E741
    return [np.stack([c[I] for c in columns], axis=1) for I in range(nRows)]  # noqa: E741


class Constraint(ConstraintBase, MeshDependent):
    """A segment-to-segment mortar contact constraint between two deformable surfaces.

    The module documentation states the formulation, its sign convention and the three ways the
    discrete condition can be enforced. Under ``formulation=lagrange`` the constraint contributes one
    scalar multiplier per non-mortar node to the global system.

    The constraint is a :class:`~edelweissfe.models.meshdependent.MeshDependent`: after an h-adaptive
    refinement of either surface it rebuilds its contact system from the regenerated contact elements
    and carries every nodal state over by node identity (see :meth:`refresh`); a restart of a refined
    run restores that state by node label (see :meth:`setRestartData`).

    Parameters
    ----------
    name
        The name of this constraint.
    model
        The model tree.
    journal
        The journal to report to. ``None`` is accepted for a constraint built directly; diagnostics
        then go to :mod:`warnings`.
    kwargs
        The options of :class:`MortarContactSchema`, by their input-file names.
    """

    #: Option schema for this constraint, per OptionSchemaProvider.
    schema = MortarContactSchema

    #: Sender of this constraint's journal messages; the messages carry the instance name in their text.
    identification = "MortarContact"

    def __init__(self, name: str, model: FEModel, journal: Journal = None, **kwargs):
        super().__init__(name, model, **kwargs)

        self.name = name
        self.journal = journal
        self.model = model
        self._lastSeenTopologyVersion = model.topology.version
        model.topology.registerMeshDependent(self)
        kwargs = CaseInsensitiveDict(vars(buildSchemaFromOptions(self.schema, kwargs)))

        # Diagnostics, already used by the input checks below.
        self._warned = set()
        self.field = kwargs["field"]
        self.sizeField = getFieldSize(self.field, model.domainSize)

        self._nonMortarSurfaceSetName = kwargs["nonMortarSurface"]
        self._mortarSurfaceSetName = kwargs["mortarSurface"]
        self._identifySurfaces(model)

        self.formulation = str(kwargs["formulation"]).strip().lower()
        if self.formulation not in ("lagrange", "penalty"):
            raise ValueError(
                f"MortarContact '{name}': unknown formulation '{kwargs['formulation']}'. "
                f"Valid choices are 'lagrange' (default) and 'penalty'."
            )
        self.geometryUpdate = str(kwargs["geometryUpdate"]).strip().lower()
        if self.geometryUpdate not in ("frozen", "incomplete"):
            raise ValueError(
                f"MortarContact '{name}': unknown geometryUpdate '{kwargs['geometryUpdate']}'. "
                f"Valid choices are 'frozen' (default) and 'incomplete'."
            )

        self._updateFootprint()
        self.active = True

        # c_n and eps_N carry a unit, so a non-positive value means "derive it". c_n is derived at the
        # first assembly (the sections are assigned to the elements only after the constraints are
        # built), eps_N at the first geometry rebuild (it needs the nodal weights).
        self.cn = float(kwargs["cn"])
        self._deriveCn = self.cn <= 0.0
        self.epsN = float(kwargs["penaltyStiffness"])
        self._deriveEpsN = self.epsN <= 0.0
        # Where c_n and eps_N came from, for the setup line of _reportSetupOnce.
        self._cnSource = "given"
        self._epsNSource = "given"
        self._frictionSetup = None
        self._setupReported = False
        self._reportEpsNWhenDerived = False

        self.useAugmentedLagrange = bool(kwargs["augmentedLagrange"])
        self.augmentationTolerance = float(kwargs["augmentationTolerance"])
        self.maxAugmentations = int(kwargs["maxAugmentations"])

        # The friction law and the per-node frictional state are those shared by all contact
        # formulations. eps_T <= 0 follows eps_N, which may itself be derived only later.
        self._friction = CoulombFriction(
            self.nNonMortarNodes,
            model.domainSize,
            float(kwargs["mu"]),
            float(kwargs["tangentPenalty"]),
            float(kwargs["slipTolerance"]),
            owner=name,
        )
        if self._friction.enabled and self.formulation != "penalty":
            raise ValueError(
                f"MortarContact '{name}': friction (mu > 0) is implemented for formulation=penalty only. "
                f"Set formulation=penalty."
            )

        # An option of the other formulation is inert; say so once.
        if self.formulation == "penalty" and float(kwargs["cn"]) > 0.0:
            self._warnOnce("cn_ignored_in_penalty", "WARNING: 'cn' is ignored for formulation=penalty")
        if self.formulation == "lagrange" and float(kwargs["penaltyStiffness"]) > 0.0:
            self._warnOnce(
                "penalty_ignored_in_lagrange", "WARNING: 'penaltyStiffness' is ignored for formulation=lagrange"
            )

        self._deriveNodalGeometry()

        # Semi-smooth Newton (primal-dual active set) state, reset per increment attempt in
        # _refreshGeometry. useActiveSet=False pins the current set from outside (tests, bilateral use).
        self.activeSet = np.zeros(self.nNonMortarNodes, dtype=bool)
        self.useActiveSet = True
        self.activeSetFrozen = False
        self.activeSetStableCount = 0
        self._seenStates = set()
        self._lastState = None
        self.currentIteration = 0
        # Identifies the increment attempt (number, size, end time): a cutback keeps the number but
        # changes the size, and has to restart from the last converged state like a new increment.
        self.lastTimeStepKey = None
        # Set by updateConnectivity, which the solver calls once per increment (implicit) or every
        # 'contact-update-frequency' increments (explicit); True so that the first assembly builds it.
        self._geometryStale = True
        # The broadphase boxes are widened by the motion of the contact nodes between the last two
        # geometry builds at converged states, from the node coordinates of the last such build (see
        # _updateSearchMotion).
        self.searchMotion = 0.0
        self._searchReferenceCoordinates = None
        self._searchReserve = np.inf

        # Explicit dynamics: known from the first call of applyConstraintExplicit. The lumped mass
        # arrives through computeCriticalTimeStepForExplicitDynamics, before the increment loop.
        self._explicit = False
        self._explicitLumpedMass = None
        self._explicitTimeIncrement = 0.0

        # Penalty and augmented Lagrange: the nodal pressure estimate (>= 0 in compression; zero for
        # pure penalty), its value at the last converged increment (warm start, cutback reset), and the
        # number of augmentations in the current increment.
        self.augmentedMultipliers = np.zeros(self.nNonMortarNodes)
        self.augmentedMultipliersConverged = np.zeros(self.nNonMortarNodes)
        self._augmentationCounter = 0
        # The weighted gap and the nodal multiplier of the last assembly, for both formulations.
        self.currentWeakGap = np.zeros(self.nNonMortarNodes)
        self.nodalMultipliers = np.zeros(self.nNonMortarNodes)

        # Friction: the pointwise weighted gap vector of every non-mortar node at the last geometry
        # rebuild (NaN where undefined), the reference of the objective slip correction; and whether a
        # mesh refinement invalidated that reference.
        self.frictionGapReference = np.full((self.nNonMortarNodes, model.domainSize), np.nan)
        self._discretisationChanged = False

        self._publishNonMortarNodeSet(model)

    # ------------------------------------------------------------------ surfaces and setup

    def _identifySurfaces(self, model: FEModel):
        """Collect the contact elements and the nodes of both sides, at construction and after an
        AMR refinement (see :meth:`refresh`)."""

        self.nonMortarFacets = self._collectFacets(model, self._nonMortarSurfaceSetName, "Non-mortar")
        self.mortarFacets = self._collectFacets(model, self._mortarSurfaceSetName, "Mortar")

        # First-seen order, so the numbering of the contact system depends on the element sets alone.
        self.nonMortarNodes = list(dict.fromkeys(node for el in self.nonMortarFacets for node in el.nodes))
        self.mortarNodes = list(dict.fromkeys(node for el in self.mortarFacets for node in el.nodes))

        # A node on both surfaces makes the system singular: coinciding facets give C = D, so their
        # weighted gap vanishes identically. Self-contact is not supported.
        sharedNodes = set(self.nonMortarNodes) & set(self.mortarNodes)
        if sharedNodes:
            labels = sorted(node.label for node in sharedNodes)
            raise ValueError(
                f"MortarContact '{self.name}': non-mortar surface '{self._nonMortarSurfaceSetName}' and mortar "
                f"surface '{self._mortarSurfaceSetName}' share {len(labels)} node(s): {labels}. "
                f"The two contact surfaces must be disjoint - coinciding facets yield an "
                f"identically vanishing weak gap and thus a singular system."
            )

        self._nodes = self.nonMortarNodes + self.mortarNodes
        self.nNonMortarNodes = len(self.nonMortarNodes)
        self.nMortarNodes = len(self.mortarNodes)

    def _updateFootprint(self):
        """The degrees of freedom of the contact system: the field on every contact node, plus one
        multiplier per non-mortar node under formulation=lagrange."""

        self.nMultipliers = self.nNonMortarNodes if self.formulation == "lagrange" else 0
        self._nDof = self.sizeField * len(self._nodes) + self.nMultipliers
        self._fieldsOnNodes = [[self.field]] * len(self._nodes)

    def _deriveNodalGeometry(self):
        """The index lookups, the reference coordinates, the interface diameter, and the orientation
        checks of the two surfaces, all in the undeformed configuration."""

        self.nodeToGlobalIndex = {node: i for i, node in enumerate(self._nodes)}
        self.nonMortarNodeToIndex = {node: i for i, node in enumerate(self.nonMortarNodes)}
        self.mortarNodeToIndex = {node: i for i, node in enumerate(self.mortarNodes)}

        # Undeformed coordinates of all constraint nodes (non-mortar first, then mortar).
        self._X = np.array([node.coordinates for node in self._nodes])

        # The largest extent of the non-mortar surface, the length the noise floor of the weighted gap
        # and the orientation check are measured against. Undeformed, so that it does not move with the
        # solution.
        slaveCoordinates = self._X[: self.nNonMortarNodes]
        self._interfaceDiameter = float(np.max(np.ptp(slaveCoordinates, axis=0))) if len(slaveCoordinates) else 0.0

        # The orientation of the facets decides the sign of gap and pressure: a non-mortar normal
        # pointing into its own body turns the contact into a bond. Both checks only warn, since a
        # legitimate mesh may be split or one-sided.
        self._checkFacetWinding(self.nonMortarFacets, self._nonMortarSurfaceSetName)
        self._checkFacetWinding(self.mortarFacets, self._mortarSurfaceSetName)
        self._checkSurfacesFaceEachOther(self.computeNormals())

    def _currentCoordinates(self, U_np: np.ndarray = None) -> np.ndarray:
        """The coordinates of all contact nodes (non-mortar first, then mortar), deformed by the
        constraint-local solution ``U_np``, or undeformed without one."""

        if U_np is None:
            return self._X
        nNodes = len(self._nodes)
        return self._X + U_np[: self.sizeField * nNodes].reshape(nNodes, self.sizeField)[:, : self.model.domainSize]

    def _weightedGap(self, I: int, xSlave: np.ndarray, xMaster: np.ndarray) -> float:  # noqa: E741
        """The weighted gap of non-mortar node ``I``, ``g_I = sum_J Cn_IJ . x_J - sum_K Dn_IK . x_K``,
        with the gap matrices of the current geometry: Popp, Gitterle, Gee & Wall (2010), Eq. (41), in
        3D, Popp, Gee & Wall (2009), Eq. (50), in 2D.

        The coordinates enter relative to the node's own position. Since ``sum_K Dn_IK = sum_J Cn_IJ``
        this changes nothing but the round-off: with absolute coordinates both sums are of the size of
        the coordinates times the nodal weight and cancel to the gap, and their round-off would be a
        spurious gap that the penalty turns into a force."""

        nzD, nzC = self.currentDNonzero[I], self.currentCNonzero[I]
        origin = xSlave[I]
        gWeak = 0.0
        if len(nzD):
            gWeak -= float(np.sum(self.currentDnRow[I] * (xSlave[nzD] - origin)))
        if len(nzC):
            gWeak += float(np.sum(self.currentCnRow[I] * (xMaster[nzC] - origin)))
        return gWeak

    def _multiplierRowScale(self, I: int) -> float:  # noqa: E741
        """``1/kappa_I``, the factor of node ``I``'s multiplier row (Popp, Seitz, Gee & Wall 2013,
        Eqs. (30)-(35)); one for a node without any integrated facet, whose row is empty anyway."""

        kappa = self.currentNodalScale[I]
        return 1.0 / kappa if kappa > 1e-12 else 1.0

    def _warnOnce(self, key: str, message: str):
        """Emit a diagnostic at most once per constraint and cause ``key``; these conditions tend to
        repeat in every increment."""

        if key in self._warned:
            return
        self._warned.add(key)
        text = f"{self.name}: {message}"
        if self.journal is not None:
            self.journal.message(text, self.identification, level=1)
        else:
            warnings.warn(f"MortarContact {text}", RuntimeWarning, stacklevel=3)

    def _reportSetupOnce(self):
        """Report the parameters the constraint runs with, once, after the first geometry rebuild,
        when every derived value (c_n, eps_N, the elastic slip) is known."""

        if self._setupReported:
            return
        self._setupReported = True
        if self.formulation == "lagrange":
            line = f"lagrange, c_n = {self.cn:.4g} ({self._cnSource})"
        else:
            # "AL": augmentedLagrange=True, abbreviated so that the line fits the journal column.
            augmented = " (AL)" if self.useAugmentedLagrange else ""
            line = f"penalty{augmented}, eps_N = {self.epsN:.4g} ({self._epsNSource})"
        lines = [line]
        if self._friction.enabled:
            if self._frictionSetup:
                lines.append(f"mu = {self._friction.mu:.4g}, {self._frictionSetup}")
            else:
                lines.append(
                    f"mu = {self._friction.mu:.4g}, eps_T = {self._friction.constantTangentModulus(self.epsN):.4g}"
                )
        for line in lines:
            if self.journal is not None:
                self.journal.message(f"{self.name}: {line}", self.identification, level=1)

    def _publishNonMortarNodeSet(self, model: FEModel):
        """Register the non-mortar nodes, in the order of the contact system, as the node set
        ``<name>_nonMortarNodes``, which the per-node accessors line up with. An existing set of that
        name that is not this constraint's own is left alone and reported."""

        setName = f"{self.name}_nonMortarNodes"
        self.nonMortarNodeSetName = setName
        nodeSets = getattr(model, "nodeSets", None)
        if nodeSets is None:
            return
        if getattr(self, "_ownsNonMortarNodeSet", False):
            # Ours: brought back into the order of the contact system after a refinement.
            current = nodeSets.get(setName)
            if current is None:
                nodeSets[setName] = NodeSet(setName, list(self.nonMortarNodes))
            else:
                current.replaceMembers(list(self.nonMortarNodes))
            return
        if setName in nodeSets:
            self._warnOnce(
                "node_set_exists",
                f"WARNING: node set '{setName}' already exists, non-mortar nodes not published",
            )
            return
        nodeSets[setName] = NodeSet(setName, list(self.nonMortarNodes))
        self._ownsNonMortarNodeSet = True

    # ------------------------------------------------------------------ AMR

    def refresh(self, model: FEModel, change) -> bool:
        """Rebuild the contact system from the regenerated contact elements after an AMR refinement
        touched either surface, carrying every nodal state over by ``Node`` identity.

        A retained non-mortar node keeps its frictional history, augmented pressure estimate,
        active-set entry, friction gap reference and (formulation=lagrange) multiplier. A new node
        takes the frictional history and the augmented estimate interpolated from the old facet it lies
        on (see :meth:`_remapNodalState`). The geometry is rebuilt at the next assembly, and the
        frictional history is carried into it without the objective slip correction, whose reference
        belongs to the old discretisation. Under formulation=lagrange the scalar variables are
        re-created (:meth:`~edelweissfe.models.femodel.FEModel.reassignScalarVariablesOfConstraint`).

        Returns True if either surface was touched, so that the solver rebuilds its equation system.
        """

        recipes = getattr(model, "contactFacetRecipes", {})
        touched = False
        for setName in (self._nonMortarSurfaceSetName, self._mortarSurfaceSetName):
            recipe = recipes.get(setName)
            if recipe is not None and change.touchesSurface(recipe[0]):
                touched = True
            if setName in getattr(change, "changedElementSets", ()):
                touched = True
        if not touched:
            return False

        oldNonMortarNodes = list(self.nonMortarNodes)
        oldNonMortarFacets = list(self.nonMortarFacets)
        oldMultipliers = (
            np.array([variable.value for variable in self.scalarVariables])
            if self.formulation == "lagrange" and len(self.scalarVariables) == len(oldNonMortarNodes)
            else None
        )

        self._identifySurfaces(model)
        self._updateFootprint()
        self._invalidateGeometry()
        # The node set changed; the motion measured so far is kept for the next build.
        self._searchReferenceCoordinates = None
        self._deriveNodalGeometry()
        self._remapNodalState(oldNonMortarNodes, oldNonMortarFacets)
        self._publishNonMortarNodeSet(model)

        if self.formulation == "lagrange":
            variables = model.reassignScalarVariablesOfConstraint(self)
            if oldMultipliers is not None:
                old = dict(zip(oldNonMortarNodes, oldMultipliers))
                for node, variable in zip(self.nonMortarNodes, variables):
                    variable.value = old.get(node, 0.0)

        if self.journal is not None:
            self.journal.message(
                f"{self.name}: rebuilt after refinement, {self.nNonMortarNodes} non-mortar nodes "
                f"(before {len(oldNonMortarNodes)}), {self.nMortarNodes} mortar nodes",
                self.identification,
                level=1,
            )
        return True

    def _invalidateGeometry(self):
        """Drop every quantity derived from the contact surfaces, so that the next assembly rebuilds
        the geometry from scratch."""

        for attribute in (
            "currentNormals",
            "currentD",
            "currentC",
            "currentNodalWeights",
            "currentNodalScale",
            "currentDNonzero",
            "currentDRow",
            "currentCNonzero",
            "currentCRow",
            "currentDnRow",
            "currentCnRow",
            "currentWeightTolerance",
            "currentGapTolerance",
            "lastCoupling",
        ):
            self.__dict__.pop(attribute, None)
        self._geometryStale = True
        # A new attempt of the current increment as far as the per-increment bookkeeping goes.
        self.lastTimeStepKey = None

    def _remapNodalState(self, oldNonMortarNodes: list, oldNonMortarFacets: list):
        """Carry the per-node state over to the current non-mortar nodes. A retained node keeps its
        state; a new node takes its frictional history and augmented pressure estimate interpolated
        with the shape functions of the old facet it lies on (Wriggers 2006, Section 14.7.2,
        Eq. 14.86); its active-set entry, weighted gap and multiplier start at rest."""

        index = {node: i for i, node in enumerate(oldNonMortarNodes)}
        newToOld = np.array([index.get(node, -1) for node in self.nonMortarNodes], dtype=int)
        kept = newToOld >= 0

        def remapped(array, fill):
            result = np.full((self.nNonMortarNodes,) + array.shape[1:], fill, dtype=array.dtype)
            result[kept] = array[newToOld[kept]]
            return result

        diameters = [float(np.max(np.ptp([n.coordinates for n in el.nodes], axis=0))) for el in oldNonMortarFacets]
        interpolation = historyInterpolation(
            np.array([node.coordinates for node in self.nonMortarNodes]),
            self.nonMortarNodes,
            oldNonMortarNodes,
            curvedFacetHistoryPatches(oldNonMortarFacets, self.model.domainSize),
            HISTORY_DISTANCE_FACTOR * float(np.mean(diameters)) if diameters else 0.0,
        )

        def interpolated(array, fill):
            result = remapped(array, fill)
            for k, (indices, weights) in interpolation.items():
                result[k] = weights @ array[indices]
            return result

        self.activeSet = remapped(self.activeSet, False)
        self.augmentedMultipliers = interpolated(self.augmentedMultipliers, 0.0)
        self.augmentedMultipliersConverged = interpolated(self.augmentedMultipliersConverged, 0.0)
        self.currentWeakGap = remapped(self.currentWeakGap, 0.0)
        self.nodalMultipliers = remapped(self.nodalMultipliers, 0.0)
        self.frictionGapReference = remapped(self.frictionGapReference, np.nan)
        self._discretisationChanged = True
        self._friction.remap(oldNonMortarNodes, self.nonMortarNodes, interpolation)

    @staticmethod
    def _collectFacets(model: FEModel, elementSetName: str, side: str) -> list:
        """The contact elements of one side, in the element set's own order.

        The set must hold contact elements only: the nodes of a solid element among them would enter
        the coupling matrices as if they lay on the interface.

        Parameters
        ----------
        model
            The model tree.
        elementSetName
            The name of the element set holding this side's contact elements.
        side
            'Non-mortar' or 'Mortar', for the error messages.

        Returns
        -------
        list
            The contact elements of that side.
        """

        if elementSetName not in model.elementSets:
            raise KeyError(
                f"{side} surface '{elementSetName}' is not an element set of this model. It has to name "
                "a set of contact elements, e.g. the '<name>_facets' set of a surfaceElementGenerator "
                "run with facets=wholeFace."
            )

        facets = list(model.elementSets[elementSetName])
        if not facets:
            raise ValueError(f"{side} surface '{elementSetName}' contains no elements.")

        offending = sorted({el.elType for el in facets if not el.elType.upper().startswith("CON")})
        if offending:
            raise ValueError(
                f"{side} surface '{elementSetName}' must contain contact elements only (element types "
                f"starting with 'CON'), but also holds: {', '.join(offending)}."
            )

        return facets

    def _checkFacetWinding(self, facets, surfaceName: str):
        """Warn if the facets of one surface are not wound consistently: two facets sharing an edge
        must traverse it in opposite directions (3D), and a node must not be the start, or the end, of
        two line facets (2D). A surface that is consistently wound but inside out as a whole is caught
        by :meth:`_checkSurfacesFaceEachOther`."""

        culprits = []
        if self.model.domainSize == 3:
            seen = {}
            for el in facets:
                nodes = el.nodes
                nCorner = 3 if len(nodes) in (3, 6) else 4
                corners = nodes[:nCorner]
                for i in range(nCorner):
                    edge = (corners[i].label, corners[(i + 1) % nCorner].label)
                    if edge in seen:
                        culprits.append((seen[edge], el.elNumber))
                    seen[edge] = el.elNumber
        else:
            starts, ends = {}, {}
            for el in facets:
                a, b = el.nodes[0].label, el.nodes[1].label
                if a in starts:
                    culprits.append((starts[a], el.elNumber))
                if b in ends:
                    culprits.append((ends[b], el.elNumber))
                starts[a] = el.elNumber
                ends[b] = el.elNumber

        if culprits:
            pairs = ", ".join(f"({a}, {b})" for a, b in culprits[:5])
            # One facet of each pair has its normal reversed: gap and pressure change sign there.
            self._warnOnce(
                f"winding_{surfaceName}",
                f"WARNING: surface '{surfaceName}' inconsistently wound, {len(culprits)} facet pair(s), first {pairs}",
            )

    def _checkSurfacesFaceEachOther(self, normals: np.ndarray):
        """Warn if the non-mortar normals point away from the mortar surface as a whole.

        A deliberately weak test: the mortar centroid has to lie more than a tenth of the interface
        diameter behind the non-mortar one along the mean normal. A flipped surface puts it a body
        thickness behind, an initial interpenetration only a fraction of an element, so the margin
        separates the two.
        """

        if not self.nNonMortarNodes or not self.nMortarNodes:
            return

        slaveCentroid = np.mean(self._X[: self.nNonMortarNodes], axis=0)
        masterCentroid = np.mean(self._X[self.nNonMortarNodes :], axis=0)
        towardsMaster = masterCentroid - slaveCentroid
        if np.linalg.norm(towardsMaster) < 1e-14 * extent(self._X):
            return  # coincident centroids: no information

        meanNormal = np.mean(normals, axis=0)
        if np.linalg.norm(meanNormal) < 1e-8:
            return  # normals cancel (e.g. a closed surface): no information

        meanNormal = meanNormal / np.linalg.norm(meanNormal)
        normalSeparation = float(np.dot(meanNormal, towardsMaster))

        if self._interfaceDiameter <= 0.0:
            return

        if normalSeparation < -0.1 * self._interfaceDiameter:
            self._warnOnce(
                "surfaces_face_away",
                f"WARNING: non-mortar normals point away from the mortar surface (offset {-normalSeparation:.4g}, "
                f"interface size {self._interfaceDiameter:.4g}); facets inverted or surfaces swapped?",
            )

    def _adjacentYoungsModuli(self) -> dict:
        """The Young's moduli of the materials whose elements touch the contact surfaces, as
        ``{section/material: E}``.

        Marmot materials carry their parameters as a flat array whose first entry is the Young's
        modulus by convention (LINEARELASTIC, GCDP, ...), not by guarantee, which is why every value
        derived from it is reported.
        """

        contactNodes = set(self._nodes)
        candidates = {}
        for name, section in self.model.sections.items():
            material = getattr(section, "material", None)
            if material is None:
                continue
            touches = False
            for elSet in getattr(section, "elSets", []):
                for el in elSet:
                    if contactNodes.intersection(el.nodes):
                        touches = True
                        break
                if touches:
                    break
            if not touches:
                continue
            if isinstance(material, dict):
                # Marmot: flat parameter array, first entry is E by convention.
                props = material.get("properties")
                E = float(props[0]) if props is not None and len(props) else None
                materialName = material.get("name", "?")
            else:
                # EdelweissFE materials keep E as _E (see materials/linearelastic); materialProperties
                # is the fallback.
                materialName = type(material).__name__
                E = getattr(material, "_E", None)
                if E is None:
                    props = getattr(material, "materialProperties", None)
                    E = float(props[0]) if props is not None and len(props) else None
                E = float(E) if E is not None else None
            if E is not None and E > 0.0:
                candidates[f"{name}/{materialName}"] = E

        return candidates

    def _smallestAdjacentYoungsModulus(self):
        """The smallest of :meth:`_adjacentYoungsModuli`, or None if there is none."""

        candidates = self._adjacentYoungsModuli()
        return min(candidates.values()) if candidates else None

    def _resolveCn(self):
        """Set c_n to the smallest initial Young's modulus adjacent to the interface.

        That c_n is purely algorithmic and does not influence the solution, only the convergence, is
        Gitterle, Popp, Gee & Wall (2010); that its lower bound depends on the material, roughly
        linearly on E, is the observation of Hueber & Wohlmuth (2005), Section 7, for their inexact
        active-set strategy. The value E fixes the order of magnitude; the materials far from the
        interface are left out, since a soft material there would push c_n below the lower bound of
        the contacting pair.
        """

        candidates = self._adjacentYoungsModuli()

        if not candidates:
            self.cn = FALLBACK_MODULUS
            self._warnOnce(
                "cn_not_derivable",
                f"WARNING: no Young's modulus found next to the contact, c_n = {self.cn:.3e} (fallback); set 'cn'",
            )
            self._cnSource = "fallback"
            return

        chosen = min(candidates, key=candidates.get)
        self.cn = candidates[chosen]
        self._cnSource = f"E of {chosen}"

    def _resolveEpsN(self):
        """Derive the penalty stiffness eps_N of ``p_I = eps_N * penetration_I``.

        Implicitly, ``eps_N = C E / h`` with C = :data:`PENALTY_STIFFNESS_FACTOR`, E the smallest
        initial Young's modulus adjacent to the interface and h the characteristic facet size of the
        non-mortar surface (:meth:`_characteristicFacetSize`). Under explicit dynamics, the largest
        stiffness the time increment can carry (:meth:`_resolveEpsNExplicit`). The nodal weight is not
        part of eps_N; it enters node by node in :meth:`_nodalPenaltyStiffness`.

        While the two surfaces do not overlap, no node is coupled and none needs a stiffness; eps_N is
        then derived at the first geometry build at which they do.
        """

        if not np.any(self.currentNodalWeights):
            self.epsN = 0.0
            self._epsNSource = "not yet derived, the surfaces do not overlap"
            return
        if self._setupReported and self._epsNSource.startswith("not yet derived"):
            # The setup line reported the pending state; the derived value is reported once below.
            self._reportEpsNWhenDerived = True

        if self._explicit and self._explicitLumpedMass is not None:
            if self._resolveEpsNExplicit():
                self._reportDerivedEpsN()
                return
            # No covered interface, or no lumped mass on the contact nodes.
            self._warnOnce(
                "epsN_not_derivable_explicit",
                "WARNING: stability limit not available, eps_N from the implicit rule; check for instability",
            )

        self._deriveEpsN = False

        h = self._characteristicFacetSize()
        if h <= 0.0:
            self.epsN = FALLBACK_MODULUS
            self._warnOnce(
                "epsN_not_derivable",
                f"WARNING: no contact area to derive eps_N, eps_N = {self.epsN:.3e} (fallback); set 'penaltyStiffness'",
            )
            self._epsNSource = "fallback"
            return

        E = self._smallestAdjacentYoungsModulus()
        if E is None:
            self.epsN = FALLBACK_MODULUS
            self._warnOnce(
                "epsN_not_derivable_E",
                f"WARNING: no Young's modulus found next to the contact, eps_N = {self.epsN:.3e} (fallback); "
                f"set 'penaltyStiffness'",
            )
            self._epsNSource = "fallback"
            return

        self.epsN = PENALTY_STIFFNESS_FACTOR * E / h
        self._epsNSource = f"E={E:.4g}, h={h:.4g}"
        self._reportDerivedEpsN()

    def _reportDerivedEpsN(self):
        """Report eps_N once where it was derived only after the setup line (first overlap)."""

        if self._reportEpsNWhenDerived:
            self._reportEpsNWhenDerived = False
            self._warnOnce(
                "epsN_derived_at_overlap", f"eps_N = {self.epsN:.4g} at the first overlap ({self._epsNSource})"
            )

    def _characteristicFacetSize(self) -> float:
        """The characteristic facet size h of the non-mortar surface, from its measure in the
        undeformed configuration (:func:`~edelweissfe.utils.contactfriction.characteristicFacetSize`).
        Taken from the whole surface rather than from the part the other surface covers, so that a
        first contact over a small part of it does not shrink h."""

        if not self.nonMortarFacets:
            return 0.0
        measure = sum(
            float(np.trace(el.computeLocalMassMatrices(np.array([node.coordinates for node in el.nodes]))[1]))
            for el in self.nonMortarFacets
        )
        return characteristicFacetSize(measure, len(self.nonMortarFacets), self.model.domainSize)

    def _resolveEpsNExplicit(self) -> bool:
        """Under explicit dynamics, set eps_N to half the largest stiffness the current time increment
        can carry, and return whether that was possible.

        The central-difference stability limit ``dt_cr = 2 / sqrt(lambda_max)`` (Kwon, Kim, Cho &
        Gonzalez 2025, Eqs. (20)-(21); Ghesquiere-Dierickx et al. 2026, Eq. (17)), with lambda_max
        bounded by the Gershgorin row sums of :meth:`_stabilityRowSums`, gives
        ``eps_N <= 4 / (dt^2 R)``. Half of it is taken, because the geometry and with it R change
        between two connectivity updates. With friction the tangential spring shares the budget row by
        row: ``max_a (eps_N r_N,a + eps_T r_T,a) <= 2 / dt^2``. Re-derived at every geometry rebuild.
        """

        rows = self._stabilityRowSums(self._explicitLumpedMass)
        factor = float(np.max(rows[0])) if rows is not None else 0.0
        dt = self._explicitTimeIncrement
        if not (factor > 0.0 and dt > 0.0):
            return False

        budget = 2.0 / (dt * dt)
        previous = getattr(self, "epsN", None)
        if not self._friction.enabled:
            self.epsN = budget / factor
        elif self._friction.tangentPenalty <= 0.0:
            factor = float(np.max(rows[0] + rows[1]))
            self.epsN = budget / factor
        else:
            loaded = rows[0] > 0.0
            self.epsN = float(np.min((budget - self._friction.tangentPenalty * rows[1][loaded]) / rows[0][loaded]))
            if self.epsN <= 0.0:
                raise ValueError(
                    f"MortarContact '{self.name}': tangentPenalty = {self._friction.tangentPenalty:.4g} alone exceeds "
                    f"the stability limit of the time increment {dt:.4g}, so no normal penalty "
                    f"stiffness is left. Lower tangentPenalty or leave it at 0 to follow eps_N."
                )
        if previous is None or not np.isclose(previous, self.epsN, rtol=0.25):
            # The setup line reports the first value; a later change by more than 25 % is reported once.
            self._epsNSource = f"stability limit, dt={dt:.4g}"
            if self._setupReported:
                self._warnOnce(
                    "epsN_rederived_explicit",
                    f"eps_N re-derived = {self.epsN:.4g} (stability limit, dt = {dt:.4g})",
                )
        return True

    def _isActiveLagrange(self, pN: float, gWeak: float, invD: float) -> bool:
        """The semi-smooth activation test of formulation=lagrange, ``s_n = p_n - c_n g_n`` with the
        physical opening ``g_n = g_weak / D_II``.

        A closed node without pressure -- ``s_n`` within the noise floor of the gap -- counts as
        active: a semi-smooth Newton method may use any element of the generalized derivative at the
        kink of max(0, .) (Hintermueller, Ito & Kunisch 2002, Lemma 3.1 and Remark 3.1), and only the
        active side keeps the tangent of a body carried by the contact alone regular when it starts
        exactly touching.
        """

        sN = pN - self.cn * (gWeak * invD)
        return bool(sN > -self.cn * self.currentGapTolerance * abs(invD))

    def _checkConvergedActiveSet(self, U_ref: np.ndarray):
        """Re-evaluate the activation test on the converged state of the last increment.

        The active set is frozen once it has settled (or on a cycle, or at the iteration cap), so the
        converged state was decided at an intermediate iterate. Warns if the test applied to the
        converged displacements and multipliers would give another set; the increment then does not
        satisfy the Signorini conditions at those nodes. Only formulation=lagrange freezes its set.
        """

        if not self.useActiveSet or self.formulation == "penalty":
            return

        nSlave = self.nNonMortarNodes
        coords = self._currentCoordinates(U_ref)
        xSlave, xMaster = coords[:nSlave], coords[nSlave:]
        firstMultiplier = self.sizeField * len(self._nodes)

        wouldBe = np.zeros(nSlave, dtype=bool)
        for I in range(nSlave):  # noqa: E741 - I is the non-mortar node index of the formulation
            gWeak = self._weightedGap(I, xSlave, xMaster)
            D_II = self.currentNodalWeights[I]
            invD = 1.0 / D_II if abs(D_II) > self.currentWeightTolerance else 0.0
            pN = U_ref[firstMultiplier + I] * np.sign(D_II)
            wouldBe[I] = self._isActiveLagrange(pN, gWeak, invD)

        flipped = np.flatnonzero(wouldBe != self.activeSet)
        if len(flipped):
            self._warnOnce(
                "active_set_unstable_at_convergence",
                f"WARNING: converged active set not self-consistent at {len(flipped)} node(s); reduce the increment size",
            )

    @classmethod
    def fromConstraintDefinition(cls, name: str, definition: dict, model: FEModel, journal: Journal) -> "Constraint":
        """Build this constraint from a parsed ``*constraint`` definition, passing the journal on (see
        :class:`~edelweissfe.constraints.base.constraintbase.ConstraintBase`)."""

        return cls(name, model, journal, **definition)

    @property
    def nodes(self) -> list:
        return self._nodes

    @property
    def fieldsOnNodes(self) -> list:
        return self._fieldsOnNodes

    @property
    def nDof(self) -> int:
        return self._nDof

    @property
    def liveGeometry(self) -> bool:
        """Whether the geometry is rebuilt at every Newton iterate (``geometryUpdate=incomplete``,
        implicit solvers only)."""

        return self.geometryUpdate == "incomplete" and not self._explicit

    @property
    def currentCoverage(self) -> np.ndarray:
        """The integrated fraction of each non-mortar node's facets, the nodal scaling factor of
        Popp, Seitz, Gee & Wall (2013), Eq. (36): one for a fully covered node."""

        return self.currentNodalScale

    def getNumberOfAdditionalNeededScalarVariables(self) -> int:
        return self.nMultipliers

    # ------------------------------------------------------------------ geometry

    def computeNormals(self, U_np: np.ndarray = None) -> np.ndarray:
        """The averaged unit normal at every non-mortar node, in the configuration ``U_np`` (undeformed
        without one).

        The element normal ``x,xi x x,eta`` (2D: the tangent rotated by -90 degrees) is evaluated at
        the node's own natural coordinate in every adjacent facet, normalised, summed, and the sum is
        normalised again: the averaged nodal normal of Popp, Gitterle, Gee & Wall (2010), Eq. (37); Farah
        (2018), Eq. (4.41), writes the same average out with unit element normals (for the master side
        of his node-to-surface projection). Evaluated per node, the normal follows the curvature of
        quadratic facets; summed unweighted, it does not depend on the size of the reference element. A node whose contributions vanish or cancel gets a zero normal and
        cannot become active; that is reported.
        """

        dim = self.model.domainSize
        normals = np.zeros((self.nNonMortarNodes, dim))

        nodeToIndex = self.nonMortarNodeToIndex
        current = self._currentCoordinates(U_np)

        for el in self.nonMortarFacets:
            facetNodes = el.nodes
            coords = current[[self.nodeToGlobalIndex[node] for node in facetNodes]]
            dNAtNodes = el.getShapeFunctionDerivativesAtNodes()
            for local, node in enumerate(facetNodes):
                if node not in nodeToIndex:
                    continue
                t = dNAtNodes[local] @ coords
                if dim == 3:
                    nNode = np.cross(t[0], t[1])
                else:
                    nNode = np.array([t[0][1], -t[0][0]])
                length = np.linalg.norm(nNode)
                # The tangents carry the facet size: the cross product an area, the 2D normal a length.
                size = extent(coords)
                if length <= 1e-14 * (size * size if dim == 3 else size):
                    # A degenerate facet at this node contributes nothing.
                    continue
                normals[nodeToIndex[node]] += nNode / length

        degenerate = []
        for i in range(self.nNonMortarNodes):
            norm = np.linalg.norm(normals[i])
            if norm > 1e-14:
                normals[i] /= norm
            else:
                normals[i] = np.zeros(dim)
                degenerate.append(i)

        if degenerate:
            # Usual causes: a zero-area facet, or adjacent facets folding back onto each other.
            self._warnOnce(
                "degenerate_nodal_normal",
                f"WARNING: {len(degenerate)} non-mortar node(s) with zero normal carry no contact",
            )

        return normals

    def computeMortarCouplingMatrices(self, U_np: np.ndarray = None) -> tuple[csr_matrix, csr_matrix]:
        """The mortar matrices D (non-mortar x non-mortar) and C (non-mortar x mortar) at the
        configuration ``U_np``; see :mod:`~edelweissfe.utils.mortarcoupling`. The whole coupling of
        that configuration is kept as :attr:`lastCoupling`."""

        coupling = self._couple(U_np, self.computeNormals(U_np))
        return coupling.D, coupling.M

    def _couple(self, U_np: np.ndarray, normals: np.ndarray):
        """Run :func:`~edelweissfe.utils.mortarcoupling.computeMortarCoupling` on the configuration
        ``U_np`` and report its diagnostics."""

        coupling = computeMortarCoupling(
            self.nonMortarFacets,
            self.mortarFacets,
            dict(zip(self._nodes, self._currentCoordinates(U_np))),
            self.nonMortarNodeToIndex,
            self.mortarNodeToIndex,
            normals,
            self.model.domainSize,
            ContactElement,
            searchMargin=self.searchMotion,
        )
        if coupling.projectionFailures:
            side, number, elType = coupling.firstProjectionFailure
            # Those points keep the last Newton iterate, so their shape functions are inexact.
            self._warnOnce(
                "gp_projection_failed",
                f"WARNING: Gauss point projection not converged for {coupling.projectionFailures} point(s), "
                f"first on {side} facet {number} ({elType})",
            )
        for side, number in coupling.nonConvex[:1]:
            # A clip polygon has to be convex; a reflex vertex means a severely distorted quadratic facet.
            self._warnOnce(
                f"nonconvex_{side}",
                f"WARNING: non-convex sub-segment on {side} facet {number}, contact area inexact",
            )
        if coupling.slivers:
            # The biorthogonality system of the integrated part is singular: the reference dual basis is used.
            self._warnOnce(
                "sliver_fallback",
                f"WARNING: degenerate overlap on non-mortar facet {coupling.slivers[0]}, reference dual basis used",
            )
        self.lastCoupling = coupling
        return coupling

    # ------------------------------------------------------------------ solver hooks

    def requiresCorrectionBeforeConvergence(self) -> bool:
        """True on the first assembly of an increment under formulation=penalty.

        There the pressure is a function of the state, not an unknown, so the forces assembled on the
        extrapolated state at iteration 0 (and, with the augmented Lagrangian, on the warm-started
        estimate) have never been equilibrated; an absent correction must not pass the convergence
        test. Under formulation=lagrange the multiplier rows are checked as scalar variables.
        """

        return self.active and self.formulation == "penalty" and self.currentIteration == 0

    def augmentConstraint(self) -> bool:
        """One augmented-Lagrangian (Uzawa) iteration on the converged state of the increment.

        The update ``p^(k+1)_I = max(0, p^k_I + eps_N * penetration_I)`` is that of Puso, Laursen &
        Solberg (2008), Eq. (18), applied only once the Newton loop has converged, as they require;
        the projection onto ``p >= 0`` is applied directly (Wriggers 2006, Eq. (10.79), p. 339, prints a
        first-order update with a min). The node-wise stiffness ``eps_N / |D_II|`` turns the weighted gap into the pointwise
        penetration. Gaps within the noise floor (:attr:`currentGapTolerance`) count as zero, so that
        a load-free interface does not feed its own round-off into the estimate. The loop terminates on
        the relative change of the estimate (see ``augmentationTolerance``).

        Returns
        -------
        bool
            True if the estimate moved and the increment has to be re-equilibrated.
        """

        if not self.active or self.formulation != "penalty" or not self.useAugmentedLagrange:
            return False

        # Nothing assembled yet.
        if not hasattr(self, "currentNodalWeights"):
            return False

        sgnD = np.sign(self.currentNodalWeights)
        gPen = -self.currentWeakGap * sgnD
        gPen = np.where(np.abs(self.currentWeakGap) > self.currentGapTolerance, gPen, 0.0)
        zNew = np.maximum(0.0, self.augmentedMultipliers + self._nodalPenaltyStiffness() * gPen)

        change = float(np.max(np.abs(zNew - self.augmentedMultipliers))) if len(zNew) else 0.0
        scale = float(np.max(np.abs(zNew))) if len(zNew) else 0.0
        self.augmentedMultipliers = zNew
        self._augmentationCounter += 1

        # scale == 0: no node carries pressure, nothing to augment.
        converged = change <= self.augmentationTolerance * scale or scale == 0.0

        if converged:
            self.augmentedMultipliersConverged[:] = self.augmentedMultipliers
            return False

        if self._augmentationCounter >= self.maxAugmentations:
            self.augmentedMultipliersConverged[:] = self.augmentedMultipliers
            self._warnOnce(
                "augmentation_cap",
                f"WARNING: augmented Lagrange stopped at {self.maxAugmentations} augmentations (change "
                f"{change / scale:.2e} > {self.augmentationTolerance:.1e}); raise maxAugmentations or penaltyStiffness",
            )
            return False

        return True

    # ------------------------------------------------------------------ assembly

    def applyConstraint(
        self,
        U_np: np.ndarray,
        dU: np.ndarray,
        PExt: np.ndarray,
        K: np.ndarray,
        timeStep: TimeStep,
    ):
        """Assemble the contact forces and the tangent with the geometry of :meth:`_refreshGeometry`
        held fixed, i.e. without the derivatives of the normals and of D and C."""

        if not self.active:
            return

        # c_n is used by formulation=lagrange only; it is not derived (and not reported) otherwise.
        if self._deriveCn and self.formulation == "lagrange":
            self._deriveCn = False
            self._resolveCn()

        stepKey = (timeStep.number, timeStep.timeIncrement, timeStep.totalTime)
        self._refreshGeometry(stepKey, U_np, dU, timeStep)

        nSlave = self.nNonMortarNodes
        coords = self._currentCoordinates(U_np)
        xSlave = coords[:nSlave]
        xMaster = coords[nSlave:]

        penalty = self.formulation == "penalty"
        # One stiffness per non-mortar node, see _nodalPenaltyStiffness.
        nodalPenalty = self._nodalPenaltyStiffness() if penalty else None
        friction = penalty and self._friction.enabled

        for I in range(nSlave):  # noqa: E741 - I is the non-mortar node index of the formulation
            gWeak = self._weightedGap(I, xSlave, xMaster)

            # The nodal weight D_II = int(Phi_I) is positive for a fully covered node of the linear
            # elements, but not in general (quadratic faces under partial coverage, the sliver
            # fallback). The pressure carries its sign, the opening g_weak / D_II does not: dividing by
            # the signed weight already gives a positive opening for a positive physical gap.
            D_II = self.currentNodalWeights[I]
            sgnD = np.sign(D_II)
            invD = 1.0 / D_II if abs(D_II) > self.currentWeightTolerance else 0.0

            if penalty:
                self._assemblePenaltyNode(I, gWeak, sgnD, D_II, nodalPenalty[I], friction, dU, PExt, K)
            else:
                self._assembleLagrangeNode(I, gWeak, sgnD, invD, U_np, PExt, K)

        self._freezeActiveSetIfSettled(penalty, timeStep)
        if friction:
            # The anti-chattering rule of the shared law, counted in this constraint's Newton iterations.
            self._friction.freezeIfChattering(self.currentIteration)
        self._reportThinlyCoveredNodes(timeStep)

    def _couplingDofs(self, I: int) -> tuple[np.ndarray, np.ndarray]:  # noqa: E741
        """The weights ``w`` of node ``I``'s weighted gap (``+D_IK`` on the non-mortar, ``-C_IJ`` on the
        mortar nodes) and the local degrees of freedom of the coupled nodes."""

        # astype: both index arrays can be empty (a node that lost all coverage while still carrying an
        # augmented pressure), and concatenating two empty arrays does not keep an integer dtype.
        nodeIds = np.concatenate((self.currentDNonzero[I], self.nNonMortarNodes + self.currentCNonzero[I])).astype(
            np.intp
        )
        w = np.concatenate((self.currentDRow[I], -self.currentCRow[I]))
        idcs = (self.sizeField * nodeIds[:, None] + np.arange(self.model.domainSize)).ravel()
        return w, idcs

    def _assemblePenaltyNode(
        self,
        I: int,  # noqa: E741 - the non-mortar node index of the formulation
        gWeak: float,
        sgnD: float,
        D_II: float,
        kappa: float,
        friction: bool,
        dU: np.ndarray,
        PExt: np.ndarray,
        K: np.ndarray,
    ):
        """The penalty (and augmented-Lagrange) contribution of one non-mortar node.

        The pressure is ``p = max(0, z_I + kappa_I g_pen)``, with the augmented estimate ``z_I`` (zero
        for pure penalty), the penetration measure ``g_pen = -g_weak sgn(D_II)`` (weighted, positive
        under penetration) and the node-wise stiffness ``kappa_I = eps_N / |D_II|`` of
        :meth:`_nodalPenaltyStiffness`, so that ``kappa_I g_pen = eps_N`` times the pointwise
        penetration. The assembled multiplier is ``lambda_I = p sgn(D_II)``, as in the Lagrange branch.

        With ``w`` the weights of the weighted gap, ``v = w (x) n_I`` and the gap matrix row
        ``v_gap`` (equal to ``v`` in 2D, Popp, Gee & Wall 2009, Eq. (50); with the normal at the Gauss
        points in 3D, Popp et al. 2010, Eq. (41), which makes the tangent non-symmetric), the
        contribution is rank one: ``PExt -= lambda_I v`` and ``K += kappa_I v v_gap^T``.

        A closed node without pressure -- ``p_trial`` within ``kappa_I`` times the noise floor of the
        gap -- is active, for the same reason as in :meth:`_isActiveLagrange`; it carries the
        stiffness but no tension. A node without coverage has no spring (``kappa_I = 0``).
        """

        self.currentWeakGap[I] = gWeak
        gPen = -gWeak * sgnD
        pTrial = self.augmentedMultipliers[I] + kappa * gPen
        if self.useActiveSet:
            self.activeSet[I] = bool(kappa > 0.0 and pTrial >= -kappa * self.currentGapTolerance)

        if not self.activeSet[I]:
            self.nodalMultipliers[I] = 0.0
            if friction:
                # An open node carries no traction.
                self._friction.markOpen(I)
            return

        lambda_I = max(pTrial, 0.0) * sgnD
        self.nodalMultipliers[I] = lambda_I

        w, idcs = self._couplingDofs(I)
        v = (w[:, None] * self.currentNormals[I][None, :]).ravel()
        vGap = np.concatenate((self.currentDnRow[I], -self.currentCnRow[I])).ravel()
        PExt[idcs] -= lambda_I * v
        K[np.ix_(idcs, idcs)] += kappa * np.outer(v, vGap)
        if friction:
            self._assembleFriction(
                I, w, vGap, idcs, self.currentNormals[I], max(pTrial, 0.0), kappa * sgnD, abs(D_II), dU, PExt, K
            )

    def _assembleLagrangeNode(
        self,
        I: int,  # noqa: E741 - the non-mortar node index of the formulation
        gWeak: float,
        sgnD: float,
        invD: float,
        U_np: np.ndarray,
        PExt: np.ndarray,
        K: np.ndarray,
    ):
        """The contribution of one non-mortar node under formulation=lagrange.

        The node is active or inactive by :meth:`_isActiveLagrange`, re-decided in every Newton
        iteration until the active set is frozen (see :meth:`_freezeActiveSetIfSettled`): the semi-smooth
        Newton method of Hintermueller, Ito & Kunisch (2002), whose complementarity function, Eq. (3.1),
        is the max form above, in the contact form of Hueber & Wohlmuth (2005), Eq. (3.9), and Gitterle
        et al. (2010), Eq. (55). The indicator uses the physical opening ``g_weak / D_II``, a
        length, where Gitterle et al. use the weighted gap, so that ``c_n g_n`` is a pressure and
        ``c_n ~ E`` is dimensionally meaningful; at convergence either gap vanishes.

        Active, the multiplier row enforces ``g_weak = 0``; with the slave force ``-lambda_I D_IK n_I``
        and the master force ``+lambda_I C_IJ n_I`` (K = -dPExt/dU), and the gap matrix rows in the
        multiplier row. Inactive, it enforces ``lambda_I = 0``. The whole row is scaled by the nodal
        factor ``1/kappa_I`` of Popp, Seitz, Gee & Wall (2013), Eqs. (30)-(35), so that lambda_I is
        their scaled multiplier, Eq. (29), and the entries keep one order of magnitude however little
        of a facet is integrated.
        """

        dim = self.model.domainSize
        sf = self.sizeField
        nSlave = self.nNonMortarNodes
        multiplierIdx = sf * len(self._nodes) + I
        lambda_I = U_np[multiplierIdx]
        n_I = self.currentNormals[I]

        self.nodalMultipliers[I] = lambda_I
        self.currentWeakGap[I] = gWeak

        if self.useActiveSet and not self.activeSetFrozen:
            self.activeSet[I] = self._isActiveLagrange(lambda_I * sgnD, gWeak, invD)

        if self.activeSet[I]:
            scale = self._multiplierRowScale(I)
            PExt[multiplierIdx] += scale * gWeak

            for K_nd, D_IK, Dn_IK in zip(self.currentDNonzero[I], self.currentDRow[I], self.currentDnRow[I]):
                sDofs = slice(sf * K_nd, sf * K_nd + dim)
                D_IK_n = scale * D_IK * n_I
                PExt[sDofs] -= lambda_I * D_IK_n
                K[sDofs, multiplierIdx] += D_IK_n
                K[multiplierIdx, sDofs] += scale * Dn_IK

            for J, C_IJ, Cn_IJ in zip(self.currentCNonzero[I], self.currentCRow[I], self.currentCnRow[I]):
                mGlobal = nSlave + J
                mDofs = slice(sf * mGlobal, sf * mGlobal + dim)
                C_IJ_n = scale * C_IJ * n_I
                PExt[mDofs] += lambda_I * C_IJ_n
                K[mDofs, multiplierIdx] -= C_IJ_n
                K[multiplierIdx, mDofs] -= scale * Cn_IJ
        else:
            PExt[multiplierIdx] -= lambda_I
            K[multiplierIdx, multiplierIdx] += 1.0

    def _freezeActiveSetIfSettled(self, penalty: bool, timeStep: TimeStep):
        """Freeze the active set for the rest of the increment once it has settled.

        Three triggers: the set is unchanged for :data:`ACTIVE_SET_SETTLE_ITERATIONS` consecutive
        iterations (the termination criterion of the primal-dual active set strategy), a discrete state
        already visited in this increment recurs after a change (a cycle, since there are only finitely
        many states), or the iteration cap :data:`ACTIVE_SET_ITERATION_CAP` is reached. Only the cap is
        reported, since only it leaves the set unconfirmed. The penalty branch has no active set to
        freeze: there the set follows the pressure.
        """

        if self.useActiveSet and not self.activeSetFrozen and not penalty:
            state = self.activeSet.tobytes()
            changed = state != self._lastState
            if changed:
                self.activeSetStableCount = 0
            else:
                self.activeSetStableCount += 1
            cycle = changed and (state in self._seenStates)
            self._seenStates.add(state)
            self._lastState = state
            settled = (
                self.activeSetStableCount >= ACTIVE_SET_SETTLE_ITERATIONS
                and self.currentIteration >= ACTIVE_SET_SETTLE_ITERATIONS
            )
            hitCap = self.currentIteration >= ACTIVE_SET_ITERATION_CAP
            if settled or cycle or hitCap:
                if hitCap and not settled and not cycle:
                    # The converged solution of such an increment may violate the Signorini conditions.
                    self._warnOnce(
                        "active_set_iteration_cap",
                        f"WARNING: active set frozen by the iteration cap ({ACTIVE_SET_ITERATION_CAP}) in increment "
                        f"{timeStep.number}, not settled",
                    )
                self.activeSetFrozen = True

    def _reportThinlyCoveredNodes(self, timeStep: TimeStep):
        """Warn about active non-mortar nodes whose own facet area is almost entirely unopposed (less
        than 1 % coverage). The constraint is exact there, but the force such a node transmits is set
        by the surrounding mechanics; putting the smaller surface on the non-mortar side avoids it. The
        bound only decides when to warn."""

        if hasattr(self, "currentNodalScale"):
            coverage = self.currentNodalScale
            thinlyCovered = np.flatnonzero((coverage < 1e-2) & (coverage > 0.0) & self.activeSet)
            if len(thinlyCovered):
                self._warnOnce(
                    "thinly_covered_nodes",
                    f"WARNING: {len(thinlyCovered)} contacting node(s) with < 1 % coverage "
                    f"(min {float(np.min(coverage[thinlyCovered])):.2e}) in increment {timeStep.number}; "
                    f"put the smaller surface on the non-mortar side",
                )

    def _nodalPenaltyStiffness(self, modulus: float | None = None) -> np.ndarray:
        """The penalty stiffness of each non-mortar node, ``kappa_I = eps_N / |D_II|`` (or the given
        ``modulus`` in place of eps_N, for the tangential law).

        The penalty law is pointwise, ``p_I = eps_N * (-g_I / D_II)``; the assembly works with the
        weighted gap ``g_I`` and folds the division into a stiffness per node. A single stiffness for
        all nodes would make the pointwise stiffness proportional to the nodal weight, a factor of four
        between corner and interior nodes of a plain interface, and the contact patch test would fail.
        The scaling is that of Yang, Laursen & Meng (2005), Eqs. (36)-(37), with the nodal weight of
        the current geometry. Nodes whose weight lies below the noise floor carry no spring.
        """

        weights = np.asarray(self.currentNodalWeights, dtype=float)
        magnitude = np.abs(weights)
        covered = magnitude > self.currentWeightTolerance
        stiffness = np.zeros_like(magnitude)
        stiffness[covered] = (self.epsN if modulus is None else modulus) / magnitude[covered]
        return stiffness

    # ------------------------------------------------------------------ friction

    def _assembleFriction(
        self,
        I: int,  # noqa: E741 - the non-mortar node index of the formulation
        w: np.ndarray,
        v: np.ndarray,
        idcs: np.ndarray,
        n_I: np.ndarray,
        p: float,
        dp_dv: float,
        nodalWeight: float,
        dU: np.ndarray,
        PExt: np.ndarray,
        K: np.ndarray,
    ):
        """Coulomb friction at one active non-mortar node, on the geometry of the assembly.

        The law is that of :class:`~edelweissfe.utils.contactfriction.CoulombFriction`; in the mortar
        setting Yang, Laursen & Meng (2005), Eqs. (39), (47)-(49), in the 3D form of Puso, Laursen &
        Solberg (2008), Eqs. (16)-(21). The node is a contact point with ``c = w / |D_II|``,
        ``A = |D_II|``, the normal ``n_I`` and the pressure ``p`` with ``dp/dU = dp_dv * v``, so that
        its slip increment is the mortar-weighted slip ``D dU_s - C dU_m`` over ``|D_II|`` and its
        nodal forces ``w (x) t``.

        That slip is measured on the geometry of the assembly (the frozen one, or that of the iterate
        under ``incomplete``) and is not objective on its own: a rigid rotation of a penetrating pair
        reads as a slip of the gap vector. :meth:`_carryFrictionHistory` removes the difference to the
        objective slip of Yang et al. (2005), Eq. (47), at every rebuild. Under ``incomplete`` a
        remainder of second order in the increment is left. Friction under ``incomplete`` is this
        implementation's extension; Hartmann (2007) treats no Coulomb sliding friction.

        Parameters
        ----------
        I
            The non-mortar node.
        w, v, idcs
            The weight row, ``w (x) n_I`` (the gap matrix row), and the local degrees of freedom.
        n_I
            The nodal normal of the geometry of the assembly.
        p
            The contact pressure, >= 0.
        dp_dv
            The derivative of p along v, i.e. ``dp/dU = dp_dv * v``.
        nodalWeight
            ``|D_II|``.
        dU, PExt, K
            As in :meth:`applyConstraint`.
        """

        forces, tangent = self._friction.evaluatePoint(
            I,
            w / nodalWeight,
            nodalWeight,
            n_I,
            p,
            dp_dv * v,
            dU[idcs],
            self.epsN,
            pressureTolerance=abs(dp_dv) * self.currentGapTolerance,
        )
        PExt[idcs] += forces
        K[np.ix_(idcs, idcs)] += tangent

    def _carryFrictionHistory(self, U_ref: np.ndarray, previousNormals: np.ndarray | None):
        """Carry the accepted tangential tractions into a rebuilt geometry, with the objective slip.

        Runs at every rebuild at a converged configuration. Between two rebuilds the measured slip
        sums to ``D_old (x_new - x_old)_s - C_old (x_new - x_old)_m``, while the objective slip of Yang,
        Laursen & Meng (2005), Eq. (47), is ``-(D_new - D_old) x_new,s + (C_new - C_old) x_new,m``;
        the difference is exactly the change of the weighted gap vector ``g = D x_s - C x_m``. The
        accepted traction is therefore corrected by ``eps_T (g_new / |D_II,new| - g_old / |D_II,old|)``
        before it is projected onto the new tangent plane and returned onto the cone
        (:meth:`~edelweissfe.utils.contactfriction.CoulombFriction.carry`). A pure slide of a flat pair
        leaves g unchanged, a rigid rotation is removed. A node whose normal turned by more than
        30 degrees loses its history; a node whose gap vector is undefined on either side is left as it
        is. After a mesh refinement the history is carried without the correction.

        Parameters
        ----------
        U_ref
            The configuration the geometry was just rebuilt at, the last accepted one.
        previousNormals
            The nodal normals of the previous rebuild, ``None`` at the first.
        """

        nSlave = self.nNonMortarNodes
        dim = self.model.domainSize
        coords = self._currentCoordinates(U_ref)
        xSlave, xMaster = coords[:nSlave], coords[nSlave:]

        weights = np.abs(np.asarray(self.currentNodalWeights, dtype=float))
        gap = np.full((nSlave, dim), np.nan)
        for I in range(nSlave):  # noqa: E741 - I is the non-mortar node index of the formulation
            if weights[I] <= self.currentWeightTolerance:
                continue
            nzD, nzC = self.currentDNonzero[I], self.currentCNonzero[I]
            g = np.zeros(dim)
            if len(nzD):
                g += self.currentDRow[I] @ xSlave[nzD]
            if len(nzC):
                g -= self.currentCRow[I] @ xMaster[nzC]
            gap[I] = g / weights[I]

        previousGap = self.frictionGapReference.copy()
        self.frictionGapReference[:] = gap

        deltaGap = gap - previousGap
        if self._discretisationChanged:
            deltaGap = np.where(np.isfinite(gap), 0.0, np.nan)
            self._discretisationChanged = False
        self._friction.carry(self.currentNormals, previousNormals, deltaGap=deltaGap, normalPenalty=self.epsN)

    def acceptLastState(self):
        """Promote the tangential tractions of the converged iterate to the frictional history, and
        report what a frozen stick/slip state may have cost. Called on acceptance of an increment,
        after the augmented outer loop, and under explicit dynamics after every time step."""

        if not self._friction.enabled:
            return
        report = self._friction.acceptAndReport("non-mortar node")
        if report:
            self._warnOnce("slip_state_frozen_inadmissible", report)

    # ------------------------------------------------------------------ results

    def getNormalPressures(self) -> np.ndarray:
        """The contact pressure of each non-mortar node, >= 0 in compression, as of the last assembly,
        ordered like the node set ``<name>_nonMortarNodes``.

        Under formulation=penalty eps_N times the pointwise penetration (plus the augmented estimate);
        under formulation=lagrange the traction recovered from the scaled multiplier, ``lambda_I /
        kappa_I`` (Popp, Seitz, Gee & Wall 2013, Eq. (39)). At a thinly integrated node this is the
        coefficient of a dual shape function over a small support and exceeds the pressure around it
        (Cichosz & Bischoff 2011); the force the node transmits is ``lambda_I D_II / kappa_I`` in
        either case.
        """

        weights = getattr(self, "currentNodalWeights", None)
        sign = np.sign(weights) if weights is not None else np.ones(self.nNonMortarNodes)
        pressures = self.nodalMultipliers * np.where(sign == 0.0, 1.0, sign)
        if self.formulation == "lagrange" and hasattr(self, "currentNodalScale"):
            kappa = self.currentNodalScale
            pressures = np.divide(pressures, kappa, out=np.zeros_like(pressures), where=kappa > 1e-12)
        return pressures

    def getTangentialTractions(self) -> np.ndarray:
        """The frictional traction vector of each non-mortar node, in global components, as of the last
        assembly; zero without friction. It is the traction the mortar side exerts on the non-mortar
        side. Unlike the node- and segment-based constraints, which return the magnitude, this returns
        the vector, of shape ``(nNonMortarNodes, nDim)``."""

        return self._friction.tractionCurrent.copy()

    def getSlipStates(self) -> np.ndarray:
        """The stick/slip state of each non-mortar node as of the last assembly, as a float for output:
        0 open, 1 stick, 2 slip. Always 0 without friction."""

        return self._friction.slipStates()

    def getGaps(self) -> np.ndarray:
        """The pointwise normal opening of each non-mortar node, ``g_I / D_II``: positive open,
        negative penetrating, 0 where the node has no weight."""

        weights = getattr(self, "currentNodalWeights", None)
        if weights is None:
            return np.zeros(self.nNonMortarNodes)
        covered = np.abs(weights) > self.currentWeightTolerance
        return np.divide(self.currentWeakGap, weights, out=np.zeros(self.nNonMortarNodes), where=covered)

    # ------------------------------------------------------------------ restart

    def getRestartData(self) -> dict[str, np.ndarray] | None:
        """The converged frictional history, the friction gap reference and normals, the augmented
        pressure estimate and, under formulation=lagrange, the multipliers, each per non-mortar node,
        with the node labels they belong to; and the search motion with the coordinates of the last
        converged geometry build, per contact node. The geometry, the active set and the derived stiffnesses
        are rebuilt from the restored solution. The multipliers are also scalar variables of the model,
        but after a refinement only the labels identify the node they belong to."""

        data = {
            **self._friction.restartData(),
            "frictionGapReference": self.frictionGapReference,
            "frictionNormals": (
                np.asarray(self.currentNormals, dtype=float)
                if getattr(self, "currentNormals", None) is not None
                else np.full((self.nNonMortarNodes, self.model.domainSize), np.nan)
            ),
            "augmentedMultipliersConverged": self.augmentedMultipliersConverged,
            "nonMortarNodeLabels": np.array([node.label for node in self.nonMortarNodes], dtype=int),
            "searchMotion": np.array([self.searchMotion]),
            "contactNodeLabels": np.array([node.label for node in self._nodes], dtype=int),
            "searchReferenceCoordinates": (
                self._searchReferenceCoordinates
                if self._searchReferenceCoordinates is not None
                else np.full((len(self._nodes), self.model.domainSize), np.nan)
            ),
        }
        if self.formulation == "lagrange":
            data["lagrangeMultipliers"] = np.array([variable.value for variable in self.scalarVariables])
        return data

    def setRestartData(self, data: dict[str, np.ndarray]):
        """Restore the state of :meth:`getRestartData`, matched to the non-mortar nodes by label.
        Called after the topology replay of an AMR run has re-created the refined mesh (see
        :meth:`~edelweissfe.models.femodel.FEModel.readRestart`)."""

        current, stored = matchRestartKeys(
            [int(label) for label in data["nonMortarNodeLabels"]], [node.label for node in self.nonMortarNodes]
        )

        self._friction.setRestartData(data, current, stored)
        self.frictionGapReference[current] = np.asarray(data["frictionGapReference"])[stored]
        # The reference belongs to the mesh it was written on, which the replay re-created.
        self._discretisationChanged = False
        normals = np.full((self.nNonMortarNodes, self.model.domainSize), np.nan)
        normals[current] = np.asarray(data["frictionNormals"])[stored]
        self._restoredFrictionNormals = normals
        augmented = np.asarray(data["augmentedMultipliersConverged"])[stored]
        self.augmentedMultipliersConverged[current] = augmented
        self.augmentedMultipliers[current] = augmented
        if self.formulation == "lagrange":
            multipliers = np.asarray(data["lagrangeMultipliers"])
            for i, j in zip(current, stored):
                self.scalarVariables[i].value = multipliers[j]
        if "searchMotion" in data:
            self.searchMotion = float(np.asarray(data["searchMotion"]).ravel()[0])
            row = {int(label): k for k, label in enumerate(data["contactNodeLabels"])}
            stored = np.asarray(data["searchReferenceCoordinates"], dtype=float)
            if all(node.label in row for node in self._nodes):
                reference = stored[[row[node.label] for node in self._nodes]]
                self._searchReferenceCoordinates = None if np.isnan(reference).any() else reference

    # ------------------------------------------------------------------ explicit dynamics

    def _stabilityRowSums(self, lumpedMass: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """The mass-scaled Gershgorin row sums of the normal and of the tangential contact stiffness,
        ``(r_N, r_T)`` per degree of freedom and free of the moduli, so that the largest contact
        eigenvalue of ``M^-1 K`` is bounded by ``max_a (eps_N r_N,a + eps_T r_T,a)``.

        The normal contribution of a node is rank one, ``(eps_N / |D_II|) v_I v_I^T`` with
        ``v_I = w_I (x) n_I``, so ``r_N,a = (1/m_a) sum_I (1/|D_II|) |v_I,a| ||v_I||_1``; the tangential
        one is the stick stiffness ``(eps_T / |D_II|) (w w^T) (x) P_T``, bounded the same way with
        ``|P_T|``. Gershgorin bounds of the row sums of ``M^-1 K``: Kulak (1989), Eqs. (11)-(12);
        Ghesquiere-Dierickx et al. (2026), Eq. (18). The bound is conservative and covers the contact
        stiffness alone; the solver takes the smaller of it and the elements' own limit (as Kulak 1989,
        Eq. (14)). Every non-mortar node is counted, active or not, since the active set changes.
        ``None`` while no geometry exists or a touched row has no mass.
        """

        if not hasattr(self, "currentNormals"):
            return None

        dim = self.model.domainSize
        rowSums = np.zeros(len(lumpedMass))
        rowSumsT = np.zeros(len(lumpedMass))

        # The stiffness-free part of kappa_I = eps_N / |D_II|; this method also runs before eps_N is known.
        weights = np.abs(np.asarray(self.currentNodalWeights, dtype=float))
        covered = weights > self.currentWeightTolerance

        for I in range(self.nNonMortarNodes):  # noqa: E741 - the non-mortar node index of the formulation
            nzD, nzC = self.currentDNonzero[I], self.currentCNonzero[I]
            if (len(nzD) == 0 and len(nzC) == 0) or not covered[I]:
                continue
            w, idcs = self._couplingDofs(I)
            v = np.abs((w[:, None] * self.currentNormals[I][None, :]).ravel())
            rowSums[idcs] += v * float(np.sum(v)) / weights[I]
            if self._friction.enabled:
                n_I = self.currentNormals[I]
                absW = np.abs(w)
                projectorRowSums = np.abs(np.eye(dim) - np.outer(n_I, n_I)).sum(axis=1)
                rowSumsT[idcs] += np.kron(absW * float(np.sum(absW)), projectorRowSums) / weights[I]

        mass = np.asarray(lumpedMass, dtype=float)
        touched = (rowSums > 0.0) | (rowSumsT > 0.0)
        if not np.any(touched) or np.any(mass[touched] <= 0.0):
            return None
        scaledN = np.zeros_like(rowSums)
        scaledT = np.zeros_like(rowSumsT)
        scaledN[touched] = rowSums[touched] / mass[touched]
        scaledT[touched] = rowSumsT[touched] / mass[touched]
        return scaledN, scaledT

    def _omegaSquaredBound(self, lumpedMass: np.ndarray) -> float:
        """Gershgorin bound of the largest contact eigenvalue of ``M^-1 K`` at the current moduli."""

        rows = self._stabilityRowSums(lumpedMass)
        if rows is None:
            return 0.0
        epsT = self._friction.constantTangentModulus(self.epsN) if self._friction.enabled else 0.0
        return float(np.max(self.epsN * rows[0] + epsT * rows[1]))

    def computeCriticalTimeStepForExplicitDynamics(self, lumpedMass: np.ndarray) -> float:
        """The stable time increment the penalty stiffness permits, ``2 / sqrt(omega^2)``, and the hook
        through which the lumped mass reaches this constraint (for :meth:`_resolveEpsNExplicit`).
        ``inf`` until the geometry exists, i.e. before the first assembly."""

        self._explicitLumpedMass = np.asarray(lumpedMass, dtype=float).copy()

        if self.formulation != "penalty" or not hasattr(self, "currentNormals"):
            return np.inf

        omegaSquared = self._omegaSquaredBound(self._explicitLumpedMass)
        if omegaSquared <= 0.0:
            return np.inf
        return 2.0 / np.sqrt(omegaSquared)

    def applyConstraintExplicit(
        self,
        U_np: np.ndarray,
        dU: np.ndarray,
        PExt: np.ndarray,
        timeStep: TimeStep,
    ):
        """Refuse what an explicit solver cannot honour, then evaluate as usual.

        ``formulation=lagrange`` has multipliers without inertia, which an explicit increment never
        updates. ``augmentedLagrange`` would never augment, since only the implicit solvers call
        :meth:`augmentConstraint`, and would silently reduce to pure penalty. A user-given
        ``penaltyStiffness`` is checked against the stability limit at every geometry rebuild, a
        derived one is taken from it (see :meth:`_resolveEpsNExplicit`).
        """

        if self.formulation == "lagrange":
            raise NotImplementedError(
                f"Constraint '{self.name}': formulation=lagrange cannot be integrated explicitly. "
                "Its multipliers are unknowns of the global system, and an explicit increment "
                "solves no system: the multiplier degrees of freedom carry no inertia and would "
                "never be updated. Use formulation=penalty."
            )
        if self.useAugmentedLagrange:
            raise NotImplementedError(
                f"Constraint '{self.name}': augmentedLagrange cannot be integrated explicitly. "
                "The augmented update runs on a converged equilibrium, which an explicit "
                "increment does not produce, so it would never run and the constraint would "
                "quietly reduce to the pure penalty form. Set augmentedLagrange=False and accept "
                "the penalty error, or run implicitly."
            )

        self._friction.checkExplicitIntegrability()

        # Only explicit solvers call this method.
        self._explicit = True
        self._explicitTimeIncrement = timeStep.timeIncrement

        super().applyConstraintExplicit(U_np, dU, PExt, timeStep)

    # ------------------------------------------------------------------ geometry update

    def updateConnectivity(self, model: FEModel) -> bool:
        """Mark the geometry for a rebuild at its next use. The solver calls this once per increment
        (implicit) or every ``contact-update-frequency`` increments (explicit). The rebuild needs the
        constraint-local solution, so it happens in :meth:`applyConstraint`. Returns False: the DOF
        footprint does not change at a tick (a mesh refinement is reported by :meth:`refresh`)."""

        self._geometryStale = True
        return False

    def _refreshGeometry(self, stepKey: tuple, U_np: np.ndarray, dU: np.ndarray, timeStep: TimeStep):
        """Start a new increment attempt or count an iteration, and rebuild the geometry where due.

        A new increment attempt -- a new increment, or a cutback re-attempt, which keeps the number
        but changes the size -- resets the active-set bookkeeping and the augmented estimate to the last
        converged ones. The geometry is built at the last converged state ``U_np - dU``, not at the
        extrapolated predictor the solver starts the increment from, so that the converged result does
        not depend on the extrapolation. Under ``geometryUpdate=frozen`` (and explicitly) that happens
        whenever the solver marked the geometry stale; under ``incomplete`` once per increment attempt,
        and then again at every iterate (Hartmann 2007, Section 6.2.4).

        Parameters
        ----------
        stepKey
            Identifies the increment and the attempt.
        U_np
            The current solution vector.
        dU
            The increment of it, so that ``U_np - dU`` is the last converged state.
        timeStep
            The current time step, for the increment number in the diagnostics.
        """

        newIncrement = stepKey != self.lastTimeStepKey
        rebuildGeometry = self._geometryStale or not hasattr(self, "currentNormals")

        if newIncrement:
            # U_np - dU is the converged result of the previous increment; check its active set first.
            if hasattr(self, "currentNormals") and self.activeSetFrozen:
                self._checkConvergedActiveSet(U_np - dU)

            self.lastTimeStepKey = stepKey
            self.currentIteration = 0
            self.activeSetFrozen = False
            self.activeSetStableCount = 0
            self._seenStates = set()
            self._lastState = None
            # The augmented Lagrangian restarts from the last converged estimate (warm start, and the
            # reset of a cutback re-attempt).
            self._augmentationCounter = 0
            self.augmentedMultipliers[:] = self.augmentedMultipliersConverged
            self._friction.startIncrementAttempt()
        else:
            self.currentIteration += 1

        if self.liveGeometry:
            self._geometryStale = False
            atConverged = not np.any(dU)
            if newIncrement or rebuildGeometry:
                # Once per increment attempt at the last converged state, which the frictional history
                # is carried into and the once-derived quantities (eps_N, the elastic slip) are taken from.
                self._buildGeometry(U_np - dU, timeStep)
                if atConverged:
                    return
            self._buildGeometry(U_np, timeStep, carryHistory=False)
            return

        if not rebuildGeometry and self._outrunsSearch(U_np):
            if self._explicit:
                # Before the next connectivity tick: a pair beyond the margin could otherwise come into
                # contact unseen.
                rebuildGeometry = True
            else:
                # The frozen geometry cannot be rebuilt within the increment.
                self._warnOnce(
                    "search_outrun",
                    f"WARNING: contact nodes moved by more than the search reserve ({self._searchReserve:.3g}) "
                    f"within increment {timeStep.number}; a contact may be missed, reduce the increment size",
                )

        if not rebuildGeometry:
            return

        self._geometryStale = False
        self._buildGeometry(U_np - dU, timeStep)

    def _updateSearchMotion(self, U_ref: np.ndarray):
        """Measure the largest displacement of a contact node since the last geometry build at a
        converged state, by which :func:`~edelweissfe.utils.mortarcoupling.computeMortarCoupling` widens
        the broadphase boxes. The geometry is held until the next build (an increment, or
        ``contact-update-frequency`` explicit increments), and a pair that approaches by as much again
        until then is still segmented now. The same state built again (a cutback re-attempt) keeps the
        motion measured before it; the first build has none to go by. The margin is this
        implementation's choice."""

        coordinates = self._currentCoordinates(U_ref)
        reference = self._searchReferenceCoordinates
        if reference is not None and reference.shape == coordinates.shape:
            if np.array_equal(coordinates, reference):
                return
            self.searchMotion = float(np.max(np.linalg.norm(coordinates - reference, axis=1)))
        self._searchReferenceCoordinates = coordinates.copy()

    def _outrunsSearch(self, U_np: np.ndarray) -> bool:
        """Whether a contact node has moved, since the last geometry build at a converged state, by more
        than the broadphase boxes were widened for (see :meth:`_updateSearchMotion`): two surfaces
        approaching each other at most twice as fast cannot have closed a gap the search left out
        before that."""

        reference = self._searchReferenceCoordinates
        if reference is None:
            return False
        motion = np.linalg.norm(self._currentCoordinates(U_np) - reference, axis=1)
        if not np.all(np.isfinite(motion)):
            return False  # a diverged state, which the solver stops on; no geometry to build there
        return bool(np.max(motion) > self._searchReserve)

    def _buildGeometry(self, U_ref: np.ndarray, timeStep: TimeStep, carryHistory: bool = True):
        """Build the geometry -- nodal normals, segmentation, coupling matrices -- at the configuration
        ``U_ref``, and everything derived from it.

        The full (for quadratic elements not diagonal) D enters forces, stiffness and weighted gap,
        which is algebraically equivalent to the transformed formulation of Popp, Wohlmuth, Gee & Wall
        (2012); lumping it would break the patch test. The weighted gap is translation invariant
        through the row-sum identity ``sum_K D_IK = sum_J C_IJ``.

        Parameters
        ----------
        U_ref
            The constraint-local solution vector the geometry is built at.
        timeStep
            The current time step, for the increment number in the diagnostics.
        carryHistory
            Whether ``U_ref`` is a converged configuration the frictional history is carried into.
        """

        nSlave = self.nNonMortarNodes
        previousNormals = getattr(self, "currentNormals", None)
        if previousNormals is None:
            # The normals of the last rebuild before a restart, if the checkpoint held them.
            previousNormals = self.__dict__.pop("_restoredFrictionNormals", None)
        if carryHistory:
            self._updateSearchMotion(U_ref)
            coordinates = self._currentCoordinates(U_ref)
            smallestFacet = min(
                extent(coordinates[[self.nodeToGlobalIndex[node] for node in el.nodes]])
                for el in (*self.nonMortarFacets, *self.mortarFacets)
            )
            self._searchReserve = SEARCH_MARGIN_FACTOR * smallestFacet + self.searchMotion
        self.currentNormals = self.computeNormals(U_ref)
        coupling = self._couple(U_ref, self.currentNormals)
        self.currentD = coupling.D
        self.currentC = coupling.M
        # The nodal scaling factor of Popp, Seitz, Gee & Wall (2013), Eq. (36).
        self.currentNodalScale = coupling.kappa
        # D_II = sum_K D_IK = int(Phi_I).
        self.currentNodalWeights = np.asarray(coupling.D.sum(axis=1)).ravel()
        self._deriveTolerances(timeStep)

        # The rows of D, C and the gap matrices, with round-off values set to zero (see _storedRows).
        self.currentDNonzero, self.currentDRow = _storedRows(coupling.D, nSlave)
        self.currentCNonzero, self.currentCRow = _storedRows(coupling.M, nSlave)
        self.currentDnRow = _storedVectorRows(coupling.Dn, nSlave)
        self.currentCnRow = _storedVectorRows(coupling.Mn, nSlave)

        if self.formulation == "penalty" and self._deriveEpsN:
            self._resolveEpsN()
        elif self.formulation == "penalty" and self._explicit and self._explicitLumpedMass is not None:
            self._checkExplicitStability()

        if self._friction.enabled and self._friction.elasticSlip is None:
            self._frictionSetup = self._friction.setCharacteristicLength(self._characteristicFacetSize())

        if self._friction.enabled and carryHistory:
            self._carryFrictionHistory(U_ref, previousNormals)

        self._reportSetupOnce()

    def _deriveTolerances(self, timeStep: TimeStep):
        """The noise floors of the current geometry, relative to its largest nodal weight (see
        :data:`RELATIVE_NOISE_FLOOR`): :attr:`currentWeightTolerance`, below which a nodal weight
        counts as none and ``1/D_II`` is not formed, and :attr:`currentGapTolerance`, below which a
        weighted gap counts as none (its scale is the interface diameter times the largest weight).
        The penalty branch needs the gap floor because it manufactures a pressure from the gap: without
        it, the round-off of a load-free interface would activate nodes. Also warns about significantly
        negative nodal weights (below -1e-6 of the largest), where the weighted gap is no longer a mean
        opening."""

        weights = self.currentNodalWeights
        weightScale = np.max(np.abs(weights)) if len(weights) else 0.0
        self.currentWeightTolerance = RELATIVE_NOISE_FLOOR * weightScale
        self.currentGapTolerance = RELATIVE_NOISE_FLOOR * weightScale * self._interfaceDiameter
        if weightScale > 0.0:
            significant = np.flatnonzero(weights < -1e-6 * weightScale)
            if len(significant):
                worst = np.min(weights) / weightScale
                # Usual causes: a partially covered CONQUAD9, or the reference-element fallback at a sliver.
                self._warnOnce(
                    "negative_nodal_weight",
                    f"WARNING: {len(significant)} non-mortar node(s) with negative mortar weight (worst "
                    f"{worst:.3e} of max) in increment {timeStep.number}",
                )

    def _checkExplicitStability(self):
        """Warn if a user-given penalty stiffness exceeds the (conservative) stability limit of the
        explicit time increment; tangentPenalty shares the budget."""

        omegaSquared = self._omegaSquaredBound(self._explicitLumpedMass)
        dt = self._explicitTimeIncrement
        if omegaSquared > 0.0 and dt > 0.0:
            dtStable = 2.0 / np.sqrt(omegaSquared)
            if dt > dtStable:
                self._warnOnce(
                    "epsN_above_stability_limit",
                    f"WARNING: penalty stiffness needs dt <= {dtStable:.4g}, analysis runs at dt = {dt:.4g}; "
                    f"lower penaltyStiffness or courant-number by {(dt / dtStable) ** 2:.4g}",
                )
