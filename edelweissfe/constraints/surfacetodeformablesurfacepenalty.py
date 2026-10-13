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

from dataclasses import dataclass

import numpy as np

from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.elements.contactsurfaceelement import facetNormalAndMeasure
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.models.meshdependent import MeshDependent
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.contactfriction import (
    CoulombFriction,
    characteristicFacetSize,
    frictionCoefficientField,
    matchRestartKeys,
    slipToleranceField,
    tangentPenaltyField,
)
from edelweissfe.utils.facetcontactgeometry import (
    closestFacet,
    closestFacetCandidates,
    edgeOrVertexNormal,
    facetSurfaceBoundary,
    liesOnSurfaceBoundary,
    line2ClosestPoint,
    tria3ClosestPoint,
)
from edelweissfe.utils.parentfacegeometry import (
    facetQuadratureRule,
    parentFaceShapeFunctions,
)
from edelweissfe.utils.schema import buildSchemaFromOptions, schemaField

"""
A penalty based unilateral contact constraint between two deformable surfaces, integrated over the
slave surface at quadrature points and distributed with each side's *parent element face* shape
functions -- a Gauss-point-to-segment (GPTS) formulation.

Why this exists alongside :mod:`~edelweissfe.constraints.nodetodeformablesurfacepenalty`: a
node-based penalty scheme cannot transmit a correct pressure across a *serendipity* (quad8/hexa20)
face, and no choice of nodal weights can repair it. The consistent nodal loads of a uniform
pressure :math:`p` on a quad8 face of area :math:`A` are :math:`-pA/12` at the corners and
:math:`+pA/3` at the midsides -- the corner load is *tensile*, while a unilateral spring can only
push. The discrete solution resolves that contradiction by opening the corner gaps: on a matched
hexa20 patch test all corner nodes lift off, carrying no load, and the transmitted force falls ~1.7%
short with a visibly undulating interface. See issue #112.

Integrating instead removes the contradiction at its root. The pressure is evaluated *pointwise*
at quadrature points, where it is non-negative as physics requires, and is then distributed with
the parent face's own shape functions, so the nodal force at node :math:`a`,

.. math::
    f_a = \\sum_q w_q \\, J_q \\, p_q \\, N_a(\\xi_q) ,

is negative wherever :math:`N_a` is -- a tensile *nodal* force arising from a non-negative
*pressure field*, with no unilateral condition ever imposed at a corner node. Note that quadrature
alone would achieve nothing: the flat facets' own linear shape functions are non-negative, so
distributing with those would leave the corners lifting off exactly as before. The parent-face
basis (:mod:`~edelweissfe.utils.parentfacegeometry`) is the load-bearing half of the formulation.

Scope of this implementation: normal penalty contact under ``sliding=small``, for both the implicit
and the explicit solver path, with optional Coulomb friction -- the law shared by all contact
formulations (:mod:`~edelweissfe.utils.contactfriction`), evaluated at every quadrature point.
Augmented Lagrange is *not* implemented (the node-based constraint has it). ``sliding=finite`` is
rejected rather than approximated, see :class:`Constraint`.

What this formulation does *not* fix: pointwise enforcement at quadrature points still
over-constrains a non-matching interface, and the integrand is discontinuous inside a slave facet
wherever the assigned master facet changes, so that quadrature error does not vanish under
refinement of the rule. Exactness on non-matching meshes requires segment clipping and weak
enforcement through a multiplier field -- mortar. This is the strictly weaker, strictly simpler
member of that family, and it is what the corner-liftoff defect actually calls for.
"""


class IntegratedSurfaceContactStiffnessView:
    """Provides structured 2-D sub-views for the sparse stiffness matrix slice of
    :class:`Constraint`.

    Each active contact point couples the nodes of its slave parent face to the nodes of its
    assigned master facet's parent face, and to nothing else -- so the contribution is one dense
    square block per active point, over that point's own ``[slave parent, master parent]`` node
    list. This is a flatter layout than the node-based constraint's four-block
    ``K_pp``/``K_ff``/``K_pf``/``K_fp`` split, because here the slave side is a face like the
    master side rather than a single node.

    Attributes
    ----------
    blocks : list[numpy.ndarray]
        One view per active contact point, of shape ``(m, m)`` with
        ``m = (nSlaveParentNodes + nMasterParentNodes) * nDim``.
    """

    def __init__(self, flat_array: np.ndarray, blockSizes: list[int]):
        """Carve the flat value slice into one dense square block per active contact point.

        Parameters
        ----------
        flat_array
            This constraint's slice of the global VIJ value array.
        blockSizes
            The DOF count ``m`` of each active contact point's block, in assembly order.
        """

        self.blocks = []
        offset = 0
        for m in blockSizes:
            self.blocks.append(flat_array[offset : offset + m * m].reshape((m, m)))
            offset += m * m


@dataclass(frozen=True)
class SurfaceToDeformableSurfacePenaltySchema:
    """The options this constraint accepts, owned by this module and never mutated from outside it.

    Mirrors :class:`~edelweissfe.constraints.nodetodeformablesurfacepenalty.
    NodeToDeformableSurfacePenaltySchema` where the meaning is the same, including the ``type`` /
    ``contactType`` spelling (a dataclass field literally called ``type`` would shadow the builtin,
    which this project's conventions avoid). ``penalty`` here is an interface stiffness modulus per
    unit area exactly as there, but it multiplies a quadrature weight rather than a nodal tributary
    area -- the same physical dimension, applied pointwise.
    """

    slaveSurface: str | None = schemaField(
        description="The element set of contact facet elements (Tria3ContactFacet/Line2ContactFacet) "
        "forming the slave surface; contact is integrated over these facets at quadrature points.",
        dtype=str,
        default=None,
        required=True,
    )
    masterSurface: str | None = schemaField(
        description="The element set of contact facet elements (Tria3ContactFacet/Line2ContactFacet) "
        "forming the master surface.",
        dtype=str,
        default=None,
        required=True,
    )
    penalty: float | None = schemaField(
        description="The numerical penalty value, an interface stiffness modulus per unit slave " "surface area.",
        dtype=float,
        default=None,
        required=True,
    )
    contactType: str = schemaField(
        description="The formulation type: 'linear' (linear force, constant stiffness with jump) "
        "or 'quadratic' (quadratic force, linear stiffness).",
        dtype=str,
        default="linear",
        optionName="type",
    )
    nQuadraturePoints: int = schemaField(
        description="The number of quadrature points per slave facet: 1, 3 or 6 for a Tria3 facet "
        "(3D), 1, 2 or 3 for a Line2 facet (2D). The default of 3 integrates the parent face's "
        "shape functions exactly over a facet, which is what makes the consistent nodal loads -- "
        "including the negative corner loads of a serendipity face -- come out exactly.",
        dtype=int,
        default=3,
    )
    searchDistance: float | None = schemaField(
        description="An optional broadphase distance for the per-increment candidate-facet search. "
        "If not given, every contact point is always assigned its single closest facet, without a "
        "distance gate.",
        dtype=float,
        default=None,
    )
    sliding: str = schemaField(
        description="The kinematic treatment of the contact geometry. Only 'small' (Abaqus-style "
        "small sliding: the closest-point projection -- master facet, parametric location and "
        "normal -- is frozen once per increment from the last converged configuration, making the "
        "gap linear in the displacement DOFs) is implemented; 'finite' is rejected.",
        dtype=str,
        default="small",
    )
    mu: float = frictionCoefficientField(" Evaluated at every quadrature point.")
    tangentPenalty: float = tangentPenaltyField("penalty")
    slipTolerance: float = slipToleranceField()


class Constraint(ConstraintBase, MeshDependent):
    """
    Penalty based unilateral contact between two deformable surfaces, integrated at quadrature
    points over the slave surface's facets and distributed with both sides' parent element face
    shape functions.

    Theoretical background
    -----------------------
    The contact points are *quadrature points*, not nodes. Each slave facet carries an
    ``nQuadraturePoints``-point rule; a point at facet-barycentric coordinates :math:`b` sits at
    parent-face parametric coordinates :math:`\\xi = b \\cdot \\Xi` (with :math:`\\Xi` the facet
    vertices' own parametric coordinates in the parent face, stamped by the surface element
    generator), and carries the constant weight :math:`J_q = A_{\\text{facet}}^{\\text{ref}} w_q`.
    Its position follows the *curved* parent surface, :math:`x_q = N^s(\\xi_q) \\cdot x^s`, so the
    slave-side faceting error of the node-based formulation disappears as well.

    Under ``sliding=small`` the projection is frozen once per increment: the assigned master facet,
    the parametric location of the closest point *in that facet's parent face*, and the facet's unit
    normal :math:`\\bar{n}`. The gap is then

    .. math::
        g = \\bar{n} \\cdot \\left( N^s(\\xi_q) \\cdot x^s - N^m(\\xi_m) \\cdot x^m \\right) ,

    which is *linear* in the displacement DOFs, so the gradient
    :math:`w = \\bar{n} \\otimes [N^s, -N^m]` is constant, the geometric Hessian vanishes
    identically, and the tangent :math:`\\mathrm{d}f_n/\\mathrm{d}g \\, w \\otimes w` is
    symmetric.
    Structurally this is the same algebra as the node-based constraint's frozen form, with that
    formulation's slave coefficient ``1`` replaced by the slave face's shape-function vector -- which
    is precisely the generalization that lets a corner node carry a tensile force.

    Note that the gap is measured to the *parent surface* point :math:`N^m(\\xi_m) \\cdot x^m`, not
    to the flat facet point :math:`b \\cdot x^{\\text{facet}}`. This matters: distributing the force
    with :math:`N^m` while measuring the gap to the flat point would make the force distribution
    differ from the transpose of the gap gradient, i.e. a non-symmetric, variationally inconsistent
    operator. The flat facets serve only as the search and parametrization scaffold; the contact
    geometry itself is the curved parent surface. For straight-edged faces the two points coincide
    identically, so nothing is lost in the common case.

    ``sliding=finite`` is rejected rather than approximated. It would require the exact gap gradient
    and Hessian of a point against a *curved* quadratic surface, whose closest-point projection is
    itself an iterative sub-problem -- a substantially different formulation from the flat-facet
    algebra of :mod:`~edelweissfe.utils.facetcontactgeometry`, not a variation on it.

    Coulomb friction (``mu > 0``) is the law shared by all contact formulations
    (:class:`~edelweissfe.utils.contactfriction.CoulombFriction`): every quadrature point is a
    contact point with the coefficients ``c = [N^s, -N^m]`` of its frozen projection, the measure
    ``A = J_q`` and the pressure ``p = -f_n / J_q``, so the tangential traction is integrated and
    distributed with the parent-face shape functions exactly as the pressure is. The history is
    rotated into the new frozen tangent plane at every connectivity update and discarded where a
    point loses its facet or crosses a crease of the master surface.

    Like the node-based constraint, this is a
    :class:`~edelweissfe.models.meshdependent.MeshDependent`: if either surface's source solid
    elements are refined mid-run, it rebuilds that side from the regenerated facets at its own next
    :meth:`updateConnectivity` tick. Quadrature points, unlike nodes, have no identity of their own,
    so the frictional history is keyed by the nodes of its slave facet and the quadrature point
    index: a slave facet that survives the retiling unchanged keeps the history of its points, the
    points of a newly created facet start at rest.

    Parameters
    ----------
    name
        The name of this constraint.
    model
        The model tree.
    slaveSurface
        The element set of contact facet elements forming the slave surface.
    masterSurface
        The element set of contact facet elements forming the master surface.
    journal
        The journal instance.
    configuration
        The options this constraint accepts, see :class:`SurfaceToDeformableSurfacePenaltySchema`.
    """

    #: Option schema for this constraint, per OptionSchemaProvider.
    schema = SurfaceToDeformableSurfacePenaltySchema

    #: Journal sender tag: a short, class-level identification, not this instance's own (potentially
    #: long, user-chosen) name -- so the log's fixed-width sender column never has to truncate it. The
    #: instance name is still reported in full, in the message text itself (see __init__/updateConnectivity).
    identification = "SurfaceContact"

    def __init__(
        self,
        name: str,
        model: FEModel,
        slaveSurface: ElementSet,
        masterSurface: ElementSet,
        journal: Journal,
        *,
        configuration: SurfaceToDeformableSurfacePenaltySchema = SurfaceToDeformableSurfacePenaltySchema(),
    ):
        super().__init__(name, model)

        self.name = name
        self.journal = journal
        self._model = model
        self._lastSeenTopologyVersion = model.topologyVersion
        model.registerMeshDependent(self)

        self.nDim = model.domainSize

        self._slaveSurfaceSetName = slaveSurface.name
        self._masterSurfaceSetName = masterSurface.name

        self.penalty = configuration.penalty
        if self.penalty <= 0.0:
            raise ValueError("The penalty must be positive: a non-positive penalty silently disables contact.")

        self.type = configuration.contactType.lower()
        if self.type not in ["linear", "quadratic"]:
            raise ValueError(f"Constraint type '{self.type}' is not supported. Use 'linear' or 'quadratic'.")

        self.sliding = configuration.sliding.lower()
        if self.sliding != "small":
            raise ValueError(
                f"Constraint sliding '{self.sliding}' is not supported by "
                "surfaceToDeformableSurfacePenalty: only 'small' is implemented. Finite sliding "
                "would require the exact gap gradient and Hessian against a curved quadratic "
                "surface, whose closest-point projection is itself an iterative sub-problem."
            )

        self.nQuadraturePoints = configuration.nQuadraturePoints
        self.searchDistance = configuration.searchDistance

        # The friction law and its per-point state; sized (and, after a retiling, remapped) by
        # _buildFromSurfaces.
        self._friction = CoulombFriction(
            0, self.nDim, configuration.mu, configuration.tangentPenalty, configuration.slipTolerance, owner=name
        )
        self._pointKeys = None

        self._buildFromSurfaces(slaveSurface, masterSurface)

        self.totalNormalForce = 0.0

        self.journal.message(
            f"contact '{self.name}': {self.nPoints} points ({len(self._slaveFacets)} slave facets x "
            f"{self.nQuadraturePoints} quadrature points), {len(self.facetElements)} master facets, "
            f"sliding={self.sliding}, type={self.type}, friction={'on' if self._friction.enabled else 'off'}",
            self.identification,
        )

    def _buildFromSurfaces(self, slaveSurface: ElementSet, masterSurface: ElementSet):
        """Build (or rebuild) every cached quantity from the two facet element sets.

        Called once from ``__init__`` and again after an AMR retiling. Everything here derives from
        the reference configuration and the facets' stamped parent faces, so it is safe to redo
        wholesale -- except the frictional history, which is carried over for every point whose
        slave facet (identified by its nodes) and quadrature point survive the retiling.

        Parameters
        ----------
        slaveSurface
            The element set of slave contact facets.
        masterSurface
            The element set of master contact facets.
        """

        self._slaveFacets = list(slaveSurface)
        self.facetElements = list(masterSurface)

        for side, facets in (("slave", self._slaveFacets), ("master", self.facetElements)):
            unstamped = [facet.elNumber for facet in facets if facet.parentFaceType is None]
            if unstamped:
                raise ValueError(
                    f"Constraint '{self.name}': {side} facet {unstamped[0]} carries no parent face. "
                    "The integrated contact formulation distributes the contact pressure with the "
                    "parent element face's shape functions, so the facets must come from the "
                    "surface element generator, which stamps them."
                )

        # A per-node weighting has no meaning in this formulation and, unlike the node-based
        # constraint, there is no code path that could apply one: the pressure is distributed with
        # the parent face's own shape functions, evaluated at the quadrature points. Accepting a
        # surface stamped with a weight transform would silently give a different answer from the
        # sibling constraint on the same input, which is the one outcome worth refusing outright --
        # the whole point of the parent-face basis is that it fixes the corner mismatch the
        # weighting only minimises.
        for side, facets, setName in (
            ("slave", self._slaveFacets, self._slaveSurfaceSetName),
            ("master", self.facetElements, self._masterSurfaceSetName),
        ):
            weighted = [facet.elNumber for facet in facets if facet.weightTransform is not None]
            if weighted:
                raise ValueError(
                    f"Constraint '{self.name}': {side} surface '{setName}' was generated with "
                    f"nodalWeights='serendipityOptimal' (facet {weighted[0]} and "
                    f"{len(weighted) - 1} more carry a weight transform), which this constraint "
                    "cannot honour -- it distributes the contact pressure with the parent face's "
                    "shape functions, not with per-node weights, and needs no corner reweighting "
                    "to begin with. Generate these facets with the default "
                    "nodalWeights='facetConsistent'."
                )

        masterNodes = {node for facet in self.facetElements for node in facet.parentFaceNodes}
        slaveNodes = {node for facet in self._slaveFacets for node in facet.parentFaceNodes}
        if not masterNodes.isdisjoint(slaveNodes):
            raise ValueError(
                f"Constraint '{self.name}': slave surface '{self._slaveSurfaceSetName}' and master "
                f"surface '{self._masterSurfaceSetName}' share nodes -- self-contact is not "
                "supported."
            )

        # Per slave facet: its parent face's nodes and reference coordinates, the parent-face shape
        # functions at each of its quadrature points (constant -- the parametric locations are
        # fixed), and each point's integration weight in the reference configuration (consistent
        # with the small-deformation setting, as the node-based constraint's tributary areas are).
        self._slaveParentNodes = []
        self._slaveParentRefCoords = []
        self._slaveShapeFunctions = []
        self._slaveIntegrationWeights = []

        for facet in self._slaveFacets:
            barycentric, weights = facetQuadratureRule(facet.nNodes, self.nQuadraturePoints)
            _, measure = facetNormalAndMeasure(np.array([n.coordinates for n in facet.nodes]))
            parametric = barycentric @ facet.vertexParametricCoords
            self._slaveParentNodes.append(list(facet.parentFaceNodes))
            self._slaveParentRefCoords.append(np.array([n.coordinates for n in facet.parentFaceNodes]))
            self._slaveShapeFunctions.append(
                np.array([parentFaceShapeFunctions(facet.parentFaceType, xi) for xi in parametric])
            )
            self._slaveIntegrationWeights.append(measure * weights)

        # The contact points, flattened: (slave facet index, quadrature point index) pairs.
        self._pointFacet = np.repeat(np.arange(len(self._slaveFacets)), self.nQuadraturePoints)
        self._pointQuadraturePoint = np.tile(np.arange(self.nQuadraturePoints), len(self._slaveFacets))
        self.nPoints = len(self._pointFacet)

        # The frictional state per point, keyed by (slave facet nodes, quadrature point) across a
        # retiling; the characteristic facet size of slipTolerance is the mean slave facet size.
        pointKeys = [
            (tuple(self._slaveFacets[f].nodes), q) for f, q in zip(self._pointFacet, self._pointQuadraturePoint)
        ]
        if self._pointKeys is None:
            self._friction.resize(self.nPoints)
        else:
            self._friction.remap(self._pointKeys, pointKeys)
        self._pointKeys = pointKeys
        if self._friction.enabled and self._friction.slipTolerance > 0.0 and self._slaveFacets:
            surfaceMeasure = float(sum(np.sum(w) for w in self._slaveIntegrationWeights))
            self._friction.setCharacteristicLength(
                characteristicFacetSize(surfaceMeasure, len(self._slaveFacets), self.nDim)
            )

        self._referenceCoordsFacets = [np.array([n.coordinates for n in el.nodes]) for el in self.facetElements]
        self._masterBoundary = facetSurfaceBoundary([el.nodes for el in self.facetElements])
        # Two facets within this distance of each other count as equally close (see closestFacet).
        self._tieTolerance = 1e-10 * characteristicFacetSize(
            float(sum(np.sum(w) for w in self._slaveIntegrationWeights)), max(len(self._slaveFacets), 1), self.nDim
        )
        self._masterParentRefCoords = [
            np.array([n.coordinates for n in el.parentFaceNodes]) for el in self.facetElements
        ]

        # Small-sliding frozen projection data, refreshed once per increment in updateConnectivity:
        # the assigned master facet, the parent-face shape functions of the closest point within
        # that facet's parent face, and the facet's unit normal.
        self._assignedFacetIdx = [None] * self.nPoints
        self._frozenMasterShapeFunctions = [None] * self.nPoints
        self._frozenNormals = [None] * self.nPoints

        # Per-point results of the current Newton iterate, for output. _gapCurrent is also the
        # scratch value an augmented-Lagrange update would consume once implemented.
        self._normalForceCurrent = np.zeros(self.nPoints)
        self._gapCurrent = np.zeros(self.nPoints)

        # The unique nodes of the slave facets, in the same first-encounter order the surface
        # element generator uses for its '<prefix>_nodes' node set -- so a fromExpression field
        # output over that set lines up with getSlaveNodalNormalForces().
        self.slaveSurfaceNodes = list(dict.fromkeys(node for facet in self._slaveFacets for node in facet.nodes))

        self._nodes = []
        self._fieldsOnNodes = []
        self._nDof = 0

        # Batched active-point arrays, rebuilt by every connectivity update; None means the
        # per-point loop is used (no active points, or non-uniform parent-node counts).
        self._batched = None
        self._activePoints = np.zeros(0, dtype=int)
        self._blockSizes = None
        self._lastReportedActivity = None

    @classmethod
    def fromConstraintDefinition(cls, name: str, definition: dict, model: FEModel, journal: Journal) -> "Constraint":
        """Build this constraint from a parsed ``*constraint`` definition. See
        :class:`~edelweissfe.constraints.base.constraintbase.ConstraintBase` for why this is
        separate from ``__init__``."""
        configuration = buildSchemaFromOptions(cls.schema, definition)
        return cls(
            name,
            model,
            model.elementSets[configuration.slaveSurface],
            model.elementSets[configuration.masterSurface],
            journal,
            configuration=configuration,
        )

    @property
    def nodes(self) -> list:
        return self._nodes

    @property
    def fieldsOnNodes(self) -> list:
        return self._fieldsOnNodes

    @property
    def nDof(self) -> int:
        return self._nDof

    def _currentCoordinates(self, nodes: list, model: FEModel, referenceCoords: np.ndarray) -> np.ndarray:
        dispField = model.nodeFields.get("displacement")
        if dispField is None or "U" not in dispField:
            return referenceCoords.copy()
        idcs = dispField._indicesOfNodesInArray
        u = np.array([dispField["U"][idcs[n]] if n in idcs else np.zeros(self.nDim) for n in nodes])
        return referenceCoords + u

    def updateConnectivity(self, model: FEModel) -> bool:
        """Freeze each contact point's closest-point projection onto the master surface, and
        redeclare the constraint's DOF footprint accordingly.

        Called once per increment by the solver, before the equation system is (re)built. The
        candidate search runs on the *flat* master facets, which keeps it robust and lets it reuse
        the node-based constraint's exact broadphase; what is frozen from it is the parametric
        location of the closest point within the facet's *parent* face, from which the parent-face
        shape functions -- the ones that may go negative at a corner -- are evaluated.
        """

        slavePointCoords = self._currentSlavePointCoordinates(model)
        facetCoords = [
            self._currentCoordinates(el.nodes, model, self._referenceCoordsFacets[i])
            for i, el in enumerate(self.facetElements)
        ]

        closestPointFunction = tria3ClosestPoint if self.nDim == 3 else line2ClosestPoint
        candidatesPerPoint = closestFacetCandidates(slavePointCoords, facetCoords, self.searchDistance)

        # The normal is a property of the facet, not of the point projecting onto it, and there are
        # nQuadraturePoints times more points than facets. Memoised per facet for this update only;
        # the arrays are shared between points, which is safe because nothing ever writes through
        # _frozenNormals -- both evaluation paths only read it into a row of nBar.
        facetNormals = {}

        # The frozen normals before and after the update, NaN where there is none, for the transport
        # of the frictional history into the new tangent planes.
        previousNormals = np.full((self.nPoints, self.nDim), np.nan)
        newNormals = np.full((self.nPoints, self.nDim), np.nan)

        newAssignment = [None] * self.nPoints
        for p in range(self.nPoints):
            if self._frozenNormals[p] is not None:
                previousNormals[p] = self._frozenNormals[p]
            # The closest facet; a tie -- the closest point on an edge or vertex shared by several
            # facets -- goes to the lowest index, so the distribution basis does not depend on the
            # last bits of the coordinates.
            bestFacet, bestWeights, bestDistance, tied = closestFacet(
                slavePointCoords[p], candidatesPerPoint[p], facetCoords, closestPointFunction, self._tieTolerance
            )

            if bestFacet is not None and (self.searchDistance is None or bestDistance <= self.searchDistance):
                newAssignment[p] = bestFacet
                facet = self.facetElements[bestFacet]
                # The clamped barycentric weights locate the closest point in the flat facet; map
                # that location into the parent face's parametric space and evaluate the parent
                # face's shape functions there.
                parametric = bestWeights @ facet.vertexParametricCoords
                self._frozenMasterShapeFunctions[p] = parentFaceShapeFunctions(facet.parentFaceType, parametric)
                normal = facetNormals.get(bestFacet)
                if normal is None:
                    normal, _ = facetNormalAndMeasure(facetCoords[bestFacet])
                    facetNormals[bestFacet] = normal
                if tied and not liesOnSurfaceBoundary(facet.nodes, bestWeights, self._masterBoundary):
                    # On an edge or vertex between facets each facet normal would give its own gap
                    # (Wriggers 2006, Fig. 10.8); the normal along the distance vector gives the one
                    # signed distance, continuous across the edge. Not on the boundary of the master
                    # surface, where the point is beside the surface rather than over it.
                    normal = edgeOrVertexNormal(
                        slavePointCoords[p], bestWeights @ facetCoords[bestFacet], normal, self._tieTolerance
                    )
                self._frozenNormals[p] = normal
                newNormals[p] = normal
            else:
                self._frozenMasterShapeFunctions[p] = None
                self._frozenNormals[p] = None

        if self._friction.enabled:
            # Rotate the frictional history into the new frozen tangent planes; discard it where a
            # point lost its facet or crossed a crease of the master surface (CoulombFriction.carry).
            assigned = np.array([a is not None for a in newAssignment], dtype=bool)
            nDiscarded = self._friction.carry(newNormals, previousNormals, assigned, normalPenalty=self.penalty)
            if nDiscarded:
                self.journal.message(
                    f"contact '{self.name}': {nDiscarded} points crossed a crease of the master surface; their "
                    "frictional history was discarded",
                    self.identification,
                    level=2,
                )

        hasChanged = newAssignment != self._assignedFacetIdx
        self._assignedFacetIdx = newAssignment

        # Only *assigned* points contribute DOFs. Gap activation is a per-iteration decision within
        # the increment and deliberately does not enter here: an assigned point that happens to be
        # open still owns its block, which is then left at zero.
        newNodes = []
        newFieldsOnNodes = []
        for p in range(self.nPoints):
            if newAssignment[p] is None:
                continue
            pointNodes = (
                self._slaveParentNodes[self._pointFacet[p]] + self.facetElements[newAssignment[p]].parentFaceNodes
            )
            newNodes.extend(pointNodes)
            newFieldsOnNodes.extend([["displacement"]] * len(pointNodes))

        if newNodes != self._nodes:
            hasChanged = True

        self._nodes = newNodes
        self._fieldsOnNodes = newFieldsOnNodes
        self._nDof = self.nDim * len(newNodes)

        self._prepareActiveArrays()

        # A point that lost its facet has no gap any more, so its last value must not survive: it
        # backs the public getGaps(), whose consumers cannot mask it because they do not know the
        # assignment, and it would otherwise keep counting as 'closed' below forever. Zero is the
        # neutral value this array is initialized with. Resetting here is what makes the closed
        # count correct by construction, so it deliberately needs no mask of its own.
        for p in range(self.nPoints):
            if newAssignment[p] is None:
                self._gapCurrent[p] = 0.0

        # Reported because the assembly cost is paid per *assigned* point while the transmitted load
        # is carried only by the closed ones: without a searchDistance every point is assigned, so
        # these two numbers are what tell an explicit run whether its contact cost is doing work.
        # The closed count is from the last force evaluation, i.e. one increment behind the
        # assignment on the same line. Emitted only when it changes -- an explicit run updates
        # connectivity thousands of times and an unchanging pair of numbers is pure log noise.
        nAssigned = sum(1 for facetIdx in newAssignment if facetIdx is not None)
        activity = (nAssigned, int((self._gapCurrent < 0.0).sum()))
        if activity != self._lastReportedActivity:
            self._lastReportedActivity = activity
            self.journal.message(
                f"contact '{self.name}': {activity[0]}/{self.nPoints} active, {activity[1]} closed",
                self.identification,
                level=2,
            )

        return hasChanged

    def _currentSlavePointCoordinates(self, model: FEModel) -> np.ndarray:
        """The current positions of all contact points, on the *curved* slave parent surface."""

        coords = np.empty((self.nPoints, self.nDim))
        for f, nodes in enumerate(self._slaveParentNodes):
            currentCoords = self._currentCoordinates(nodes, model, self._slaveParentRefCoords[f])
            first = f * self.nQuadraturePoints
            coords[first : first + self.nQuadraturePoints] = self._slaveShapeFunctions[f] @ currentCoords
        return coords

    def refresh(self, model: FEModel, change) -> bool:
        """Rebuild both sides from the regenerated facet sets if ``change`` touched either surface's
        source elements.

        The facets themselves were already regenerated in the topology-update phase by the implicit
        ``surfaceFacets`` modifier; this constraint is a pure reader of the resulting element sets.
        Everything cached here is derived from the reference configuration, so the rebuild is
        wholesale, except the frictional history of the points whose slave facet survives the
        retiling (see the class docstring).
        """

        slaveRecipe = model.contactFacetRecipes.get(self._slaveSurfaceSetName)
        masterRecipe = model.contactFacetRecipes.get(self._masterSurfaceSetName)
        touchedSlave = slaveRecipe is not None and change.touchesSurface(slaveRecipe[0])
        touchedMaster = masterRecipe is not None and change.touchesSurface(masterRecipe[0])
        if not (touchedSlave or touchedMaster):
            return False

        self._buildFromSurfaces(
            model.elementSets[self._slaveSurfaceSetName],
            model.elementSets[self._masterSurfaceSetName],
        )
        return True

    def _prepareActiveArrays(self):
        """Batch the frozen per-point data into contiguous arrays, once per connectivity update.

        The force evaluation runs every explicit increment -- 1.5M times on the anchor pry-out --
        over thousands of contact points whose individual work is a couple of 8x3 matmuls and a
        48-entry outer product. Looping in Python costs about 29 microseconds per point, essentially
        all of it interpreter and numpy dispatch overhead rather than arithmetic, which put contact
        at roughly half the wall clock of an explicit run. Batching lets the whole surface be
        evaluated in a fixed handful of array operations instead.

        Only the *uniform* case is batched: every active point must share the same slave and master
        parent-node counts, which holds whenever each surface is discretized with a single element
        type (the ordinary case, and the pry-out's). Anything else falls back to the per-point loop,
        which remains the reference implementation.
        """

        active = [p for p in range(self.nPoints) if self._assignedFacetIdx[p] is not None]
        self._activePoints = np.array(active, dtype=int)
        self._blockSizes = None

        if not active:
            self._batched = None
            return

        slaveCounts = {len(self._slaveShapeFunctions[self._pointFacet[p]][0]) for p in active}
        masterCounts = {len(self._frozenMasterShapeFunctions[p]) for p in active}
        if len(slaveCounts) != 1 or len(masterCounts) != 1:
            self._batched = None
            return

        nSlaveNodes, nMasterNodes = slaveCounts.pop(), masterCounts.pop()
        nActive, nDim = len(active), self.nDim

        Ns = np.empty((nActive, nSlaveNodes))
        Nm = np.empty((nActive, nMasterNodes))
        nBar = np.empty((nActive, nDim))
        weights = np.empty(nActive)
        slaveRef = np.empty((nActive, nSlaveNodes, nDim))
        masterRef = np.empty((nActive, nMasterNodes, nDim))

        for k, p in enumerate(active):
            f = self._pointFacet[p]
            q = self._pointQuadraturePoint[p]
            Ns[k] = self._slaveShapeFunctions[f][q]
            Nm[k] = self._frozenMasterShapeFunctions[p]
            nBar[k] = self._frozenNormals[p]
            weights[k] = self._slaveIntegrationWeights[f][q]
            slaveRef[k] = self._slaveParentRefCoords[f]
            masterRef[k] = self._masterParentRefCoords[self._assignedFacetIdx[p]]

        # Local DOF layout, mirroring the node declaration above exactly: each active point owns a
        # contiguous run of (nSlaveNodes + nMasterNodes) * nDim entries, slave nodes first, and no
        # two points share a slot -- the sharing between points happens only when the *solver*
        # scatters this local vector into the global one. So the whole local vector is just a
        # (nActive, blockDof) matrix, and both the displacement gather and the force scatter are
        # plain reshapes: no index arrays, no bincount, no np.add.at.
        blockDof = (nSlaveNodes + nMasterNodes) * nDim
        assert self._nDof == nActive * blockDof, "the local DOF layout is not one block per point"

        self._batched = {
            "Ns": Ns,
            "Nm": Nm,
            "nBar": nBar,
            "weights": weights,
            "slaveRef": slaveRef,
            "masterRef": masterRef,
            "nSlaveNodes": nSlaveNodes,
            "nMasterNodes": nMasterNodes,
            "nSlaveDof": nSlaveNodes * nDim,
            "blockDof": blockDof,
        }

    def _applyConstraintBatched(self, U_np: np.ndarray, dU: np.ndarray, PExt: np.ndarray, K) -> None:
        """The batched force evaluation. Physics identical to the per-point loop, expressed over all
        active points at once; see :meth:`_prepareActiveArrays`."""

        b = self._batched
        nDim = self.nDim
        active = self._activePoints

        blocks = U_np.reshape((-1, b["blockDof"]))
        xS = b["slaveRef"] + blocks[:, : b["nSlaveDof"]].reshape((-1, b["nSlaveNodes"], nDim))
        xM = b["masterRef"] + blocks[:, b["nSlaveDof"] :].reshape((-1, b["nMasterNodes"], nDim))

        # Both points ride the curved parent surfaces; see the class docstring on why the master
        # point must not be taken on the flat facet instead.
        relative = np.einsum("ai,aij->aj", b["Ns"], xS) - np.einsum("ai,aij->aj", b["Nm"], xM)
        gaps = np.einsum("aj,aj->a", b["nBar"], relative)

        self._gapCurrent[active] = gaps

        closed = gaps < 0.0
        if not closed.any():
            return

        g = gaps[closed]
        penaltyTimesArea = self.penalty * b["weights"][closed]

        if self.type == "linear":
            f_n = penaltyTimesArea * g
            stiffness = penaltyTimesArea
        else:
            f_n = -0.5 * penaltyTimesArea * g**2
            stiffness = -penaltyTimesArea * g

        # w = kron([Ns, -Nm], nBar), per point
        c = np.concatenate((b["Ns"][closed], -b["Nm"][closed]), axis=1)
        w = (c[:, :, None] * b["nBar"][closed][:, None, :]).reshape((len(g), -1))

        rows = np.flatnonzero(closed)
        PExt.reshape((-1, b["blockDof"]))[rows] += -f_n[:, None] * w

        self._normalForceCurrent[active[closed]] = f_n
        self.totalNormalForce = float(f_n.sum())

        frictionTangent = None
        if self._friction.enabled:
            # Every closed point as a contact point of the shared law: c, A = J_q, the frozen normal,
            # p = -f_n / J_q and dp/dU = -stiffness * w / J_q -- all points at once.
            area = b["weights"][closed]
            frictionForces, frictionTangent = self._friction.evaluate(
                active[closed],
                c,
                area,
                b["nBar"][closed],
                -f_n / area,
                (-stiffness / area)[:, None] * w,
                dU.reshape((-1, b["blockDof"]))[rows],
                self.penalty,
                computeTangent=K is not None,
            )
            PExt.reshape((-1, b["blockDof"]))[rows] += frictionForces

        if K is not None:
            # The tangent is only ever requested by the implicit path, so it stays a loop: one
            # (nSlave+nMaster)*nDim square block per point, written into the view's own per-active
            # -point slots, which the batched form has no way to address in bulk.
            for i, activeIdx in enumerate(rows):
                K.blocks[activeIdx] += stiffness[i] * np.outer(w[i], w[i])
                if frictionTangent is not None:
                    K.blocks[activeIdx] += frictionTangent[i]

    def _activeBlockSizes(self) -> list[int]:
        """The DOF count of each assigned contact point's dense block, in assembly order.

        Cached, because the implicit path asks for it twice per evaluation (once to size the VIJ
        contribution, once to shape it) and recomputing it walks every point and dereferences its
        facet's parent-face node list -- which profiled as the single largest cost of an evaluation,
        larger than all of the actual contact arithmetic put together.
        """

        if self._blockSizes is None:
            self._blockSizes = self._computeActiveBlockSizes()
        return self._blockSizes

    def _computeActiveBlockSizes(self) -> list[int]:
        """Uncached :meth:`_activeBlockSizes`."""

        return [
            self.nDim
            * (
                len(self._slaveParentNodes[self._pointFacet[p]])
                + len(self.facetElements[self._assignedFacetIdx[p]].parentFaceNodes)
            )
            for p in range(self.nPoints)
            if self._assignedFacetIdx[p] is not None
        ]

    def getVIJContributionSize(self) -> int:
        return sum(m * m for m in self._activeBlockSizes())

    def initializeVIJContribution(self, idcs: np.ndarray, I_: np.ndarray, J_: np.ndarray, offset: int) -> None:
        k = offset
        localOffset = 0
        for m in self._activeBlockSizes():
            blockIdcs = idcs[localOffset : localOffset + m]
            localOffset += m
            I_[k : k + m * m] = np.repeat(blockIdcs, m)
            J_[k : k + m * m] = np.tile(blockIdcs, m)
            k += m * m

    def shapeVIJContribution(self, flat_view: np.ndarray) -> IntegratedSurfaceContactStiffnessView:
        return IntegratedSurfaceContactStiffnessView(flat_view, self._activeBlockSizes())

    def applyConstraintExplicit(
        self,
        U_np: np.ndarray,
        dU: np.ndarray,
        PExt: np.ndarray,
        timeStep: TimeStep,
    ):
        """Forces without a tangent, by running the one loop with ``K=None``.

        Overriding this is a necessity here rather than an optimization, and the override must match
        the base class's spelling of the hook exactly. The base implementation asks for a tangent
        container through the ordinary VIJ protocol and then discards it; for this constraint that
        container is ``nPoints * ((nSlaveParentNodes + nMasterParentNodes) * nDim)**2`` doubles --
        of order 50 MB per increment on the anchor pry-out -- plus one dense outer product per
        contact point to fill it. An override under the wrong name is therefore not a cosmetic slip:
        it is silently dead code, every result is unchanged, and the only symptom is an
        inexplicably slow run. One loop rather than two, so the physics cannot drift between the
        implicit and explicit paths."""

        self._friction.checkExplicitIntegrability()
        self.applyConstraint(U_np, dU, PExt, None, timeStep)

    def applyConstraint(
        self,
        U_np: np.ndarray,
        dU: np.ndarray,
        PExt: np.ndarray,
        K: IntegratedSurfaceContactStiffnessView | None,
        timeStep: TimeStep,
    ):
        """Evaluate the contact forces, and the tangent unless ``K`` is None.

        The local DOF walk must mirror :meth:`updateConnectivity`'s node declaration exactly: an
        unassigned point contributes no nodes there, so it must not advance ``localOffset`` here.
        """

        self.totalNormalForce = 0.0
        self._normalForceCurrent[:] = 0.0

        friction = self._friction.enabled
        if friction:
            self._friction.beginEvaluation(timeStep)
            self._friction.markOpen(slice(None))

        if self._batched is not None:
            self._applyConstraintBatched(U_np, dU, PExt, K)
        else:
            self._applyConstraintPointwise(U_np, dU, PExt, K)

        if friction:
            self._friction.freezeIfChattering(self._friction.iteration)

    def _applyConstraintPointwise(
        self,
        U_np: np.ndarray,
        dU: np.ndarray,
        PExt: np.ndarray,
        K: IntegratedSurfaceContactStiffnessView | None,
    ):
        """The per-point force evaluation, the reference implementation of the batched one."""

        localOffset = 0
        activeIdx = 0
        for p in range(self.nPoints):

            facetIdx = self._assignedFacetIdx[p]
            if facetIdx is None:
                continue

            f = self._pointFacet[p]
            slaveShapeFunctions = self._slaveShapeFunctions[f][self._pointQuadraturePoint[p]]
            masterShapeFunctions = self._frozenMasterShapeFunctions[p]
            nBar = self._frozenNormals[p]

            nSlaveDof = self.nDim * len(slaveShapeFunctions)
            nMasterDof = self.nDim * len(masterShapeFunctions)

            slaveIdcs = list(range(localOffset, localOffset + nSlaveDof))
            localOffset += nSlaveDof
            masterIdcs = list(range(localOffset, localOffset + nMasterDof))
            localOffset += nMasterDof

            slaveCoords = self._slaveParentRefCoords[f] + U_np[slaveIdcs].reshape((-1, self.nDim))
            masterCoords = self._masterParentRefCoords[facetIdx] + U_np[masterIdcs].reshape((-1, self.nDim))

            # Both points ride the *curved* parent surfaces; see the class docstring on why the
            # master point must not be taken on the flat facet instead.
            g = nBar.dot(slaveShapeFunctions @ slaveCoords - masterShapeFunctions @ masterCoords)
            self._gapCurrent[p] = g

            if g >= 0.0:
                activeIdx += 1
                continue

            # Frozen projection: the gap is linear in the DOFs, so the gradient is constant and the
            # geometric (Hessian) term vanishes identically. Built only for a point that is actually
            # in contact -- without a searchDistance every point is assigned a facet and most of
            # them are open, so this kron of 2 * nParentNodes * nDim entries would otherwise be the
            # dominant per-increment cost of an explicit run and be discarded every time.
            c = np.concatenate((slaveShapeFunctions, -masterShapeFunctions))
            w = np.kron(c, nBar)

            # The integration weight plays the role the node-based formulation's tributary area
            # plays, so the force laws below are identical to that constraint's, sign conventions
            # included: f_n carries the sign of g (negative in contact) so that PExt -= f_n * w
            # pushes the surfaces apart, and stiffness = df_n/dg is positive for g < 0.
            penaltyTimesArea = self.penalty * self._slaveIntegrationWeights[f][self._pointQuadraturePoint[p]]

            if self.type == "linear":
                f_n = penaltyTimesArea * g
                stiffness = penaltyTimesArea
            else:
                f_n = -0.5 * penaltyTimesArea * g**2
                stiffness = -penaltyTimesArea * g

            globalIdcs = slaveIdcs + masterIdcs
            PExt[globalIdcs] += -f_n * w

            # K is None when the caller discards the tangent -- see
            # ConstraintBase.applyConstraintExplicit.
            if K is not None:
                K.blocks[activeIdx] += stiffness * np.outer(w, w)

            if self._friction.enabled:
                # The point as a contact point of the shared law: c, A = J_q, the frozen normal,
                # p = -f_n / J_q and dp/dU = -stiffness * w / J_q.
                area = self._slaveIntegrationWeights[f][self._pointQuadraturePoint[p]]
                PFriction, KFriction = self._friction.evaluatePoint(
                    p,
                    c,
                    area,
                    nBar,
                    -f_n / area,
                    (-stiffness / area) * w,
                    dU[globalIdcs],
                    self.penalty,
                    computeTangent=K is not None,
                )
                PExt[globalIdcs] += PFriction
                if K is not None:
                    K.blocks[activeIdx] += KFriction

            self._normalForceCurrent[p] = f_n
            self.totalNormalForce += f_n
            activeIdx += 1

    def getNormalPressures(self) -> np.ndarray:
        """The current normal contact pressures (positive in compression), one per contact point.

        Ordered by slave facet and, within a facet, by quadrature point -- *not* by node, unlike the
        node-based constraint's accessor of the same name. A ``fromExpression`` field output reading
        this therefore cannot be tied to a node set; use a reduction (``np.max``, ``np.mean``, ...)
        or :meth:`getSlaveNodalNormalForces` instead.
        """

        weights = np.concatenate(self._slaveIntegrationWeights) if self.nPoints else np.zeros(0)
        return np.divide(
            -self._normalForceCurrent,
            weights,
            out=np.zeros_like(self._normalForceCurrent),
            where=weights > 0,
        )

    def getGaps(self) -> np.ndarray:
        """The current gap at each contact point (negative when penetrating), ordered like
        :meth:`getNormalPressures`.

        A point with no assigned master facet reports 0, not a stale value from when it last had
        one -- see :meth:`updateConnectivity`. Returned as a copy, so a caller cannot corrupt the
        state a later augmented-Lagrange update would read.
        """

        return self._gapCurrent.copy()

    def getTangentialTractions(self) -> np.ndarray:
        """The current tangential (frictional) traction magnitude at each contact point, ordered like
        :meth:`getNormalPressures`; zero without friction."""

        return self._friction.tractionMagnitudes()

    def getSlipStates(self) -> np.ndarray:
        """The current stick/slip state of each contact point as a float for output -- 0 open,
        1 stick, 2 slip -- ordered like :meth:`getNormalPressures`. Always 0 without friction."""

        return self._friction.slipStates()

    def acceptLastState(self):
        """Promote the tangential tractions of the converged iterate to the frictional history.
        Called by :meth:`~edelweissfe.models.femodel.FEModel.advanceToTime` upon increment
        acceptance."""

        if not self._friction.enabled:
            return
        insideCone, turnedAway = self._friction.accept()
        if insideCone or turnedAway:
            self.journal.message(
                f"contact '{self.name}': the stick/slip state chattered and was frozen; {insideCone} point(s) "
                f"were accepted in slip although their stick predictor lies inside the friction cone, "
                f"{turnedAway} with a predictor more than 30 degrees away from the frozen slip direction. "
                "They carry the full friction mu*p in the frozen direction; smaller increments, or a "
                "slipTolerance, remove it.",
                self.identification,
                level=1,
            )

    def getRestartData(self) -> dict[str, np.ndarray] | None:
        """The converged frictional history per contact point, with the point keys (the labels of
        its slave facet's nodes and its quadrature point index); ``None`` without friction, where
        there is none. The frozen projection is recomputed by the first connectivity update after a
        restart."""

        if not self._friction.enabled:
            return None
        return {**self._friction.restartData(), "pointKeys": self._pointKeyLabels()}

    def setRestartData(self, data: dict[str, np.ndarray]):
        """Restore the converged frictional history. Called after the topology replay of an AMR run
        re-created the refined mesh but before any solver refreshed the mesh dependents, so the
        constraint first catches up with the refined surfaces (:meth:`refresh`) and then matches the
        checkpoint to its points by key."""

        self.refreshIfMeshChanged(self._model)

        if "pointKeys" in data:
            current, stored = matchRestartKeys(
                [tuple(int(v) for v in row) for row in np.asarray(data["pointKeys"])],
                [tuple(int(v) for v in row) for row in self._pointKeyLabels()],
            )
            self._friction.setRestartData(data, current, stored)
        else:
            self._friction.setRestartData(data)

    def _pointKeyLabels(self) -> np.ndarray:
        """``(nPoints, nFacetNodes + 1)``: the node labels of each point's slave facet and its
        quadrature point index -- the key the frictional history is carried by."""

        if not self.nPoints:
            return np.zeros((0, 1), dtype=int)
        return np.array([[node.label for node in nodes] + [q] for nodes, q in self._pointKeys], dtype=int)

    def getSlaveNodalNormalForces(self) -> np.ndarray:
        """The normal component of the contact force delivered to each slave surface node.

        Positive means pushed along the master's outward normal, i.e. repelled; **negative means
        pulled**, which is exactly the tensile corner load a serendipity face requires and a
        node-based scheme cannot produce. This is the accessor that demonstrates the formulation
        works, and the one a patch test should assert on.

        Ordered like the surface element generator's ``<prefix>_nodes`` node set of the slave
        surface (see :attr:`slaveSurfaceNodes`), so a ``fromExpression`` output over that set lines
        up. A parent-face node that is not itself a facet node -- possible only with
        ``triangulation=corner``, where the facets keep no midside nodes -- receives contact force
        but has no slot in that set, and is omitted here; use ``triangulation=midside`` for a
        complete picture.
        """

        forceOfNode = dict.fromkeys(self.slaveSurfaceNodes, 0.0)
        for p in range(self.nPoints):
            if self._assignedFacetIdx[p] is None or self._normalForceCurrent[p] == 0.0:
                continue
            shapeFunctions = self._slaveShapeFunctions[self._pointFacet[p]][self._pointQuadraturePoint[p]]
            for node, shapeFunction in zip(self._slaveParentNodes[self._pointFacet[p]], shapeFunctions):
                if node in forceOfNode:
                    forceOfNode[node] += -self._normalForceCurrent[p] * shapeFunction
        return np.array([forceOfNode[node] for node in self.slaveSurfaceNodes])
