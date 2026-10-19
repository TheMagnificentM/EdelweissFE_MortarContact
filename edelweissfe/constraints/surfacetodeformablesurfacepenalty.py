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
from edelweissfe.constraints.base.contactpointsonslavesurface import (
    ContactPointsOnSlaveSurface,
    checkFacetsCarryParentFaces,
    groupFacetsByParentFace,
)
from edelweissfe.constraints.base.forcesonlyexplicitevaluation import (
    ForcesOnlyExplicitEvaluation,
)
from edelweissfe.constraints.base.frictionalcontact import FrictionalContact
from edelweissfe.constraints.base.frozencontactsearch import (
    FrozenContactSearch,
    packAssignment,
    packPerPointArrays,
    unpackAssignment,
    unpackPerPointArrays,
)
from edelweissfe.constraints.base.penaltylaw import (
    normalPenaltyForce,
    validatedContactType,
)
from edelweissfe.constraints.base.surfacecontactpenaltyschema import (
    SurfaceContactPenaltySchema,
)
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.models.meshdependent import MeshDependent
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.contactfriction import (
    CoulombFriction,
    characteristicFacetSize,
    frictionCoefficientField,
    historyInterpolation,
    slipToleranceField,
    tangentPenaltyField,
)
from edelweissfe.utils.facetcontactgeometry import (
    closestFacetCandidates,
    closestFacetsWithTies,
    edgeOrVertexNormal,
    facetClosestPoints,
    facetNormalAndMeasure,
    facetQuadraturePointPatches,
    facetSurfaceBoundary,
    liesOnSurfaceBoundary,
    parentFaceQuadraturePointPatches,
)
from edelweissfe.utils.meshtools import currentNodeCoordinates
from edelweissfe.utils.parentfacegeometry import (
    PARENT_FACE_LOCAL_DIMENSION,
    parentFaceClosestPoints,
    parentFaceShapeFunctions,
    parentFaceShapeFunctionsAt,
    parentFaceTangents,
    parentFaceUnitNormals,
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
class SurfaceToDeformableSurfacePenaltySchema(SurfaceContactPenaltySchema):
    """The options this constraint accepts: the shared slave-side and penalty options of
    :class:`~edelweissfe.constraints.base.surfacecontactpenaltyschema.SurfaceContactPenaltySchema`,
    plus the master surface. ``penalty`` here is an interface stiffness modulus per unit area exactly
    as in the node-based constraint, but it multiplies a quadrature weight rather than a nodal
    tributary area -- the same physical dimension, applied pointwise.
    """

    masterSurface: str | None = schemaField(
        description="The element set of contact facet elements (Tria3ContactFacet/Line2ContactFacet) "
        "forming the master surface.",
        dtype=str,
        default=None,
        required=True,
    )
    masterProjection: str = schemaField(
        description="Where each contact point is projected onto the master surface: 'facet' (the "
        "closest point on the flat master facet, mapped into its parent face, with the facet's "
        "normal) or 'parentFace' (that point refined by Newton's method to the closest point on the "
        "curved parent face, clamped to the face and passed to a neighbouring face where it lies "
        "closer, with the surface normal there). The facets remain the search either way.",
        dtype=str,
        default="facet",
    )
    mu: float = frictionCoefficientField(" Evaluated at every quadrature point.")
    tangentPenalty: float = tangentPenaltyField("penalty")
    slipTolerance: float = slipToleranceField()


class Constraint(FrictionalContact, FrozenContactSearch, ForcesOnlyExplicitEvaluation, ConstraintBase, MeshDependent):
    """
    Penalty based unilateral contact between two deformable surfaces, integrated at quadrature
    points over the slave surface and distributed with both sides' parent element face shape
    functions.

    Theoretical background
    -----------------------
    The contact points are *quadrature points*, not nodes, see
    :class:`~edelweissfe.constraints.base.contactpointsonslavesurface.ContactPointsOnSlaveSurface`.
    With ``quadrature=facet`` each slave facet carries an ``nQuadraturePoints``-point rule; a point
    at facet-barycentric coordinates :math:`b` sits at parent-face parametric coordinates
    :math:`\\xi = b \\cdot \\Xi` (with :math:`\\Xi` the facet vertices' own parametric coordinates in
    the parent face, stamped by the surface element generator), and carries the constant weight
    :math:`J_q = A_{\\text{facet}}^{\\text{ref}} w_q`. With ``quadrature=parentFace`` each slave parent
    face carries its native Gauss rule, weighted with the face's own surface Jacobian. Either way a
    point's position follows the *curved* parent surface, :math:`x_q = N^s(\\xi_q) \\cdot x^s`, so the
    slave-side faceting error of the node-based formulation disappears as well.

    Under ``sliding=small`` the projection is frozen once per increment: the assigned master parent
    face, the parametric location of the closest point *in that parent face*, and the unit normal
    :math:`\\bar{n}`. With ``masterProjection=facet`` the closest point is that on the flat master
    facet and :math:`\\bar{n}` the facet's normal; with ``masterProjection=parentFace`` it is the
    closest point on the curved parent face (Wriggers 2006, Eqs. 4.2-4.3, by Newton's method from the
    flat-facet point, Eqs. 9.81-9.82) and :math:`\\bar{n}` the surface normal there (Eq. 4.4). The gap
    is then

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
    operator. The flat facets serve as the search and, with ``masterProjection=facet``, as the
    parametrization scaffold; the contact geometry itself is the curved parent surface. For
    straight-edged faces the two points coincide identically.

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
    so the frictional history is keyed by the nodes of its slave cell (facet or parent face) and the
    quadrature point index: a cell that survives the retiling unchanged keeps the history of its
    points, and a point of a newly created cell takes the history interpolated from the quadrature
    points of the old cell it lies on (Wriggers 2006, Section 14.7.2; see
    :func:`~edelweissfe.utils.facetcontactgeometry.facetQuadraturePointPatches` and
    :func:`~edelweissfe.utils.facetcontactgeometry.parentFaceQuadraturePointPatches`).

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
        self._lastSeenTopologyVersion = model.topology.version
        model.topology.registerMeshDependent(self)

        self.nDim = model.domainSize

        self._slaveSurfaceSetName = slaveSurface.name
        self._masterSurfaceSetName = masterSurface.name

        self.penalty = configuration.penalty
        if self.penalty <= 0.0:
            raise ValueError("The penalty must be positive: a non-positive penalty silently disables contact.")

        self.type = validatedContactType(configuration.contactType)

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
        self.quadrature = configuration.quadrature
        self.gaussPointsPerDirection = configuration.gaussPointsPerDirection
        if not 1 <= self.gaussPointsPerDirection <= 4:
            raise ValueError(f"gaussPointsPerDirection must be 1 to 4, got {self.gaussPointsPerDirection}.")
        self.masterProjection = configuration.masterProjection
        if self.masterProjection not in ("facet", "parentFace"):
            raise ValueError(
                f"Constraint '{name}': unknown masterProjection '{self.masterProjection}'; "
                "choose 'facet' or 'parentFace'."
            )
        self._projectionFallbacks = 0

        # The friction law and its per-point state; sized (and, after a retiling, remapped) by
        # _buildFromSurfaces.
        self._friction = CoulombFriction(
            0, self.nDim, configuration.mu, configuration.tangentPenalty, configuration.slipTolerance, owner=name
        )
        self._pointKeys = None

        self._buildFromSurfaces(slaveSurface, masterSurface)

        self.totalNormalForce = 0.0

        self.journal.message(
            f"{self.name}: {self.nPoints} points ({self.slave.nCells} slave {self.quadrature} cells x "
            f"{self.slave.pointsPerCell} quadrature points), {len(self.facetElements)} master facets, "
            f"masterProjection={self.masterProjection}, sliding={self.sliding}, type={self.type}, "
            f"mu={self._friction.mu:g}",
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

        oldSlave = getattr(self, "slave", None)
        self.slave = ContactPointsOnSlaveSurface(
            slaveSurface, self.nQuadraturePoints, self.name, self.quadrature, self.gaussPointsPerDirection
        )
        self.facetElements = list(masterSurface)

        checkFacetsCarryParentFaces(self.facetElements, "master", self._masterSurfaceSetName, self.name)

        masterNodes = {node for facet in self.facetElements for node in facet.parentFaceNodes}
        if not masterNodes.isdisjoint(self.slave.allParentNodes()):
            raise ValueError(
                f"Constraint '{self.name}': slave surface '{self._slaveSurfaceSetName}' and master "
                f"surface '{self._masterSurfaceSetName}' share nodes -- self-contact is not "
                "supported."
            )

        # The frictional state per point, keyed by (slave cell nodes, quadrature point) across a
        # retiling; the characteristic facet size of slipTolerance is the mean slave facet size,
        # whatever the integration cells.
        facetSize = characteristicFacetSize(self.slave.facetMeasure, max(len(self.slave.facets), 1), self.nDim)
        pointKeys = [(self.slave.cellKeys[c], q) for c, q in zip(self.slave.pointCell, self.slave.pointQuadraturePoint)]
        if self._pointKeys is None:
            self._friction.resize(self.nPoints)
        else:
            # A point of a retained slave facet keeps its history; a point of a new facet takes it
            # interpolated from the quadrature points of the old facet it lies on (Wriggers 2006,
            # Section 14.7.2).
            interpolation = None
            if self._friction.enabled:
                if oldSlave.quadrature == "facet":
                    oldPatches = facetQuadraturePointPatches(
                        oldSlave.facets, self.nQuadraturePoints, lambda facet, q: (tuple(facet.nodes), q)
                    )
                else:
                    oldPatches = parentFaceQuadraturePointPatches(oldSlave, lambda c, q: (oldSlave.cellKeys[c], q))
                interpolation = historyInterpolation(
                    self.slave.referencePointCoordinates(), pointKeys, self._pointKeys, oldPatches, 0.1 * facetSize
                )
            self._friction.remap(self._pointKeys, pointKeys, interpolation)
        self._pointKeys = pointKeys
        if self.slave.facets:
            self._setFrictionCharacteristicLength(facetSize)

        self._buildMasterParentFaces()

        self._referenceCoordsFacets = [np.array([n.coordinates for n in el.nodes]) for el in self.facetElements]
        self._masterBoundary = facetSurfaceBoundary([el.nodes for el in self.facetElements])
        # Two facets within this distance of each other count as equally close (see
        # closestFacetsWithTies).
        self._tieTolerance = 1e-10 * facetSize
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

    @property
    def nPoints(self) -> int:
        """The number of contact points, see :class:`ContactPointsOnSlaveSurface`."""
        return self.slave.nPoints

    @property
    def slaveSurfaceNodes(self) -> list:
        """The unique slave facet nodes, see :attr:`ContactPointsOnSlaveSurface.surfaceNodes`."""
        return self.slave.surfaceNodes

    def updateConnectivity(self, model: FEModel) -> bool:
        """Freeze each contact point's closest-point projection onto the master surface, and
        redeclare the constraint's DOF footprint accordingly.

        Called once per increment by the solver, before the equation system is (re)built. The
        candidate search runs on the *flat* master facets, which keeps it robust and lets it reuse
        the node-based constraint's exact broadphase; what is frozen from it is the parametric
        location of the closest point within the facet's *parent* face, from which the parent-face
        shape functions -- the ones that may go negative at a corner -- are evaluated.
        """

        slavePointCoords = self.slave.currentPointCoordinates(model)
        facetCoords = [
            currentNodeCoordinates(el.nodes, model, self._referenceCoordsFacets[i])
            for i, el in enumerate(self.facetElements)
        ]

        candidatesPerPoint = closestFacetCandidates(slavePointCoords, facetCoords, self.searchDistance)
        # The closest facet; a tie -- the closest point on an edge or vertex shared by several
        # facets -- goes to the lowest index, so the distribution basis does not depend on the last
        # bits of the coordinates.
        closestFacet, closestWeights, closestDistance, closestTied = closestFacetsWithTies(
            slavePointCoords, facetCoords, candidatesPerPoint, self._tieTolerance
        )

        # The normal is a property of the facet, not of the point projecting onto it, and there are
        # nQuadraturePoints times more points than facets. Memoised per facet for this update only;
        # the arrays are shared between points, which is safe because nothing ever writes through
        # _frozenNormals -- both evaluation paths only read it into a row of nBar.
        facetNormals = {}

        # The frozen normals before and after the update, NaN where there is none, for the transport
        # of the frictional history into the new tangent planes.
        previousNormals = np.full((self.nPoints, self.nDim), np.nan)
        newNormals = np.full((self.nPoints, self.nDim), np.nan)

        assigned = (closestFacet >= 0) & (
            closestDistance <= self.searchDistance if self.searchDistance is not None else True
        )
        if self.masterProjection == "parentFace":
            surfaceProjection = self._projectOntoParentFaces(
                model, slavePointCoords, facetCoords, candidatesPerPoint, np.flatnonzero(assigned), closestFacet
            )

        newAssignment = [None] * self.nPoints
        for p in range(self.nPoints):
            if self._frozenNormals[p] is not None:
                previousNormals[p] = self._frozenNormals[p]
            bestFacet, bestWeights = int(closestFacet[p]), closestWeights[p]

            if assigned[p] and self.masterProjection == "parentFace":
                face, xi, normal, useEdgeNormal, closestPoint = surfaceProjection[p]
                newAssignment[p] = self._masterFaceFacets[face][0]
                self._frozenMasterShapeFunctions[p] = parentFaceShapeFunctions(self._masterFaceTypes[face], xi)
                if useEdgeNormal:
                    # As for the facets below (Wriggers 2006, Fig. 10.8): on an edge or vertex shared by
                    # equally close parent faces, the normal along the distance vector.
                    normal = edgeOrVertexNormal(slavePointCoords[p], closestPoint, normal, self._tieTolerance)
                self._frozenNormals[p] = normal
                newNormals[p] = normal
            elif assigned[p]:
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
                if closestTied[p] and not liesOnSurfaceBoundary(facet.nodes, bestWeights, self._masterBoundary):
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

        self._carryFrictionHistory(
            newNormals, previousNormals, np.array([a is not None for a in newAssignment], dtype=bool)
        )

        return self._assignMasterFacets(newAssignment)

    def _buildMasterParentFaces(self):
        """Group the master facets by parent face, for ``masterProjection='parentFace'``: per face its
        facets, type, reference node coordinates, the orientation of its canonical node order relative
        to the outward facet normal, which of its parametric bounds lie on the boundary of the master
        surface, and the faces it shares a corner with."""

        if self.masterProjection != "parentFace":
            return

        facetIndex = {id(facet): i for i, facet in enumerate(self.facetElements)}
        groups = list(groupFacetsByParentFace(self.facetElements).values())
        unsupported = [facets[0] for facets in groups if facets[0].parentFaceType not in PARENT_FACE_LOCAL_DIMENSION]
        if unsupported:
            raise ValueError(
                f"Constraint '{self.name}': masterProjection='parentFace' does not support parent face type "
                f"'{unsupported[0].parentFaceType}' (master facet {unsupported[0].elNumber})."
            )

        self._masterFaceFacets = [[facetIndex[id(facet)] for facet in facets] for facets in groups]
        self._masterFaceTypes = [facets[0].parentFaceType for facets in groups]
        self._masterFaceNodes = [list(facets[0].parentFaceNodes) for facets in groups]
        self._masterFaceRefCoords = [np.array([n.coordinates for n in nodes]) for nodes in self._masterFaceNodes]
        self._facetMasterFace = np.empty(len(self.facetElements), dtype=np.intp)
        for face, facets in enumerate(self._masterFaceFacets):
            self._facetMasterFace[facets] = face

        # The canonical node order fixes the sign of a_1 x a_2 (or of the rotated tangent); align it
        # with the outward normal of a facet of the same face, at that facet's parametric centroid.
        self._masterFaceSign = np.empty(len(groups))
        for face, facets in enumerate(groups):
            facet = facets[0]
            facetNormal, _ = facetNormalAndMeasure(np.array([n.coordinates for n in facet.nodes]))
            centroid = np.mean(np.asarray(facet.vertexParametricCoords, dtype=float), axis=0)[None]
            tangents = parentFaceTangents(facet.parentFaceType, centroid, self._masterFaceRefCoords[face])
            self._masterFaceSign[face] = 1.0 if parentFaceUnitNormals(tangents)[0] @ facetNormal > 0.0 else -1.0

        # Each parametric bound of a face, as the corner nodes it runs through: for a quadrilateral the
        # edge of the canonical corner cycle c0 (-1,-1), c1 (1,-1), c2 (1,1), c3 (-1,1); for a line the
        # end node. A bound shared with no other master face lies on the surface boundary.
        def boundCorners(face):
            c = self._masterFaceNodes[face]
            if PARENT_FACE_LOCAL_DIMENSION[self._masterFaceTypes[face]] == 2:
                return [
                    [frozenset((c[3], c[0])), frozenset((c[1], c[2]))],
                    [frozenset((c[0], c[1])), frozenset((c[2], c[3]))],
                ]
            return [[frozenset((c[0],)), frozenset((c[1],))]]

        bounds = [boundCorners(face) for face in range(len(groups))]
        count = {}
        for faceBounds in bounds:
            for alphaBounds in faceBounds:
                for bound in alphaBounds:
                    count[bound] = count.get(bound, 0) + 1
        self._masterFaceOnBoundary = [
            np.array([[count[bound] == 1 for bound in alphaBounds] for alphaBounds in faceBounds])
            for faceBounds in bounds
        ]

        corners = [
            set(nodes[: 2 if PARENT_FACE_LOCAL_DIMENSION[t] == 1 else 4])
            for nodes, t in zip(self._masterFaceNodes, self._masterFaceTypes)
        ]
        facesOfNode = {}
        for face, nodes in enumerate(corners):
            for node in nodes:
                facesOfNode.setdefault(node, []).append(face)
        self._masterFaceNeighbours = [
            sorted({other for node in nodes for other in facesOfNode[node]}) for nodes in corners
        ]

    def _projectOntoParentFaces(
        self,
        model: FEModel,
        slavePointCoords: np.ndarray,
        facetCoords: list,
        candidatesPerPoint: list,
        points: np.ndarray,
        closestFacet: np.ndarray,
    ) -> dict:
        """The closest point of each assigned contact point on the curved master parent faces.

        The flat facets remain the search: the candidates are the parent faces of the point's candidate
        facets and every face sharing a corner with the face of its closest facet, so that a point whose
        curved projection leaves that face is passed to the neighbour it lies over. On each candidate
        face, the closest point on its flat facets is the starting point of Newton's method on the
        curved face (Wriggers 2006, Eqs. 9.81-9.82, p. 243: "using as a starting value the solution for
        a flat surface"), see :func:`~edelweissfe.utils.parentfacegeometry.parentFaceClosestPoints`. The
        closest face wins, a tie within the tie tolerance going to the lowest face index; where the
        iteration does not converge, the flat facet's point is kept for that face.

        Returns
        -------
        dict
            Per point: the face, the parametric coordinates of the closest point, the outward unit
            surface normal there (Wriggers 2006, Eq. 4.4), whether the point projects onto an edge or
            vertex shared by equally close faces inside the master surface, and the closest point.
        """

        if not len(points):
            return {}

        nDim = self.nDim
        pairPoint, pairFace = [], []
        for p in points:
            faces = {int(self._facetMasterFace[f]) for f in candidatesPerPoint[p]}
            faces.update(self._masterFaceNeighbours[int(self._facetMasterFace[closestFacet[p]])])
            faces = sorted(faces)
            pairPoint.extend([p] * len(faces))
            pairFace.extend(faces)
        pairPoint = np.array(pairPoint, dtype=np.intp)
        pairFace = np.array(pairFace, dtype=np.intp)
        nPairs = len(pairPoint)
        faceCoords = {
            face: currentNodeCoordinates(self._masterFaceNodes[face], model, self._masterFaceRefCoords[face])
            for face in np.unique(pairFace).tolist()
        }

        # The starting point on each candidate face: its closest flat facet point, mapped into the face.
        startFacets = [(k, f) for k, face in enumerate(pairFace) for f in self._masterFaceFacets[face]]
        startPair = np.array([k for k, _ in startFacets], dtype=np.intp)
        startFacet = np.array([f for _, f in startFacets], dtype=np.intp)
        stacked = np.asarray(facetCoords, dtype=float)
        weights, distances = facetClosestPoints(slavePointCoords[pairPoint[startPair]], stacked[startFacet])
        best = np.full(nPairs, np.inf)
        np.minimum.at(best, startPair, distances)
        first = np.flatnonzero(distances == best[startPair])
        first = first[np.r_[True, startPair[first[1:]] != startPair[first[:-1]]]]

        localDim = {PARENT_FACE_LOCAL_DIMENSION[t] for t in self._masterFaceTypes}.pop()
        xi = np.zeros((nPairs, localDim))
        xi[startPair[first]] = np.einsum(
            "pk,pkd->pd",
            weights[first],
            np.array([self.facetElements[f].vertexParametricCoords for f in startFacet[first]], dtype=float),
        )

        closestPoint = np.zeros((nPairs, nDim))
        normal = np.zeros((nPairs, nDim))
        tolerance = 1e-12
        for faceType in sorted(set(self._masterFaceTypes)):
            k = np.flatnonzero(np.array([self._masterFaceTypes[face] == faceType for face in pairFace], dtype=bool))
            if not len(k):
                continue
            nodeCoords = np.array([faceCoords[face] for face in pairFace[k]])
            xiStart = xi[k]
            xiNew, xNew, converged = parentFaceClosestPoints(
                faceType, nodeCoords, slavePointCoords[pairPoint[k]], xiStart, tolerance
            )
            self._projectionFallbacks += int((~converged).sum())
            xiNew[~converged] = xiStart[~converged]
            xNew[~converged] = np.einsum(
                "pk,pkj->pj", parentFaceShapeFunctionsAt(faceType, xiStart[~converged]), nodeCoords[~converged]
            )
            xi[k], closestPoint[k] = xiNew, xNew
            normal[k] = parentFaceUnitNormals(parentFaceTangents(faceType, xiNew, nodeCoords)) * (
                self._masterFaceSign[pairFace[k]][:, None]
            )

        distance = np.linalg.norm(slavePointCoords[pairPoint] - closestPoint, axis=1)
        smallest = np.full(self.nPoints, np.inf)
        np.minimum.at(smallest, pairPoint, distance)
        equallyClose = np.flatnonzero(distance <= smallest[pairPoint] + self._tieTolerance)
        tied = np.bincount(pairPoint[equallyClose], minlength=self.nPoints) > 1
        # Pairs are ordered by point and, per point, by ascending face: keep each point's first.
        chosen = equallyClose[np.r_[True, pairPoint[equallyClose[1:]] != pairPoint[equallyClose[:-1]]]]

        result = {}
        for k in chosen:
            p, face = int(pairPoint[k]), int(pairFace[k])
            atBound = np.isclose(np.abs(xi[k]), 1.0, rtol=0.0, atol=tolerance)
            onBoundary = any(
                self._masterFaceOnBoundary[face][alpha, 1 if xi[k, alpha] > 0.0 else 0]
                for alpha in np.flatnonzero(atBound)
            )
            result[p] = (face, xi[k].copy(), normal[k].copy(), bool(tied[p] and not onBoundary), closestPoint[k])
        return result

    def _assignMasterFacets(self, newAssignment: list) -> bool:
        """Assign each contact point its master facet (searched or restored) and update the coupled DOFs."""

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
                self.slave.parentNodes[self.slave.pointCell[p]] + self.facetElements[newAssignment[p]].parentFaceNodes
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
                f"{self.name}: {activity[0]}/{self.nPoints} active, {activity[1]} closed",
                self.identification,
                level=2,
            )

        return hasChanged

    def _searchLayout(self) -> dict[str, np.ndarray]:
        layout = {
            "slaveFacets": np.array([facet.elNumber for facet in self.slave.facets], dtype=np.int64),
            "nQuadraturePoints": np.array([self.nQuadraturePoints], dtype=np.int64),
            "masterFacets": np.array([facet.elNumber for facet in self.facetElements], dtype=np.int64),
        }
        # The point layout and the meaning of the frozen projection depend on these as well; entered
        # only where they differ from the defaults, so that a checkpoint written without them still fits.
        if self.quadrature != "facet":
            layout["gaussPointsPerDirection"] = np.array([self.gaussPointsPerDirection], dtype=np.int64)
        if self.masterProjection != "facet":
            layout["masterProjectionParentFace"] = np.array([1], dtype=np.int64)
        return layout

    def _frozenProjection(self) -> dict[str, np.ndarray]:
        shapeFunctionCounts, shapeFunctions = packPerPointArrays(self._frozenMasterShapeFunctions)
        normalCounts, normals = packPerPointArrays(self._frozenNormals)
        return {
            "assignedFacetIdx": packAssignment(self._assignedFacetIdx),
            "shapeFunctionCounts": shapeFunctionCounts,
            "shapeFunctions": shapeFunctions,
            "normalCounts": normalCounts,
            "normals": normals,
        }

    def _adoptFrozenProjection(self, projection: dict[str, np.ndarray]) -> bool:
        # The frictional history was already rotated into this tangent plane by the search that froze it.
        self._frozenMasterShapeFunctions = unpackPerPointArrays(
            projection["shapeFunctionCounts"], projection["shapeFunctions"]
        )
        self._frozenNormals = unpackPerPointArrays(projection["normalCounts"], projection["normals"])
        return self._assignMasterFacets(unpackAssignment(projection["assignedFacetIdx"]))

    def getRestartData(self) -> dict[str, np.ndarray]:
        """Return the frozen projection and, with friction, the converged frictional history per
        contact point. Overridden rather than declared: the frozen projection is packed by
        :class:`FrozenContactSearch`."""

        data = super().getRestartData()
        if self._friction.enabled:
            data |= self._friction.restartData()
        return data

    def setRestartData(self, data: dict[str, np.ndarray]):
        """Restore the frozen projection and, with friction, the frictional history. Called after the
        topology replay and after the mesh dependents caught up with it (see
        :meth:`~edelweissfe.models.femodel.FEModel.readRestart`), so the contact points are those of
        the checkpoint, which :class:`FrozenContactSearch` verifies."""

        super().setRestartData(data)
        if self._friction.enabled:
            self._friction.setRestartData(data)

    def refresh(self, model: FEModel, change) -> bool:
        """Rebuild both sides from the regenerated facet sets if ``change`` touched either surface's
        source elements.

        The facets themselves were already regenerated in the topology-update phase by the implicit
        ``surfaceFacets`` modifier; this constraint is a pure reader of the resulting element sets.
        Everything cached here is derived from the reference configuration, so the rebuild is
        wholesale, except the frictional history of the points whose slave facet survives the
        retiling (see the class docstring).
        """

        masterRecipe = model.contactFacetRecipes.get(self._masterSurfaceSetName)
        touchedSlave = self.slave.isTouchedBy(model, change)
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

        slaveCounts = {len(self.slave.shapeFunctions[self.slave.pointCell[p]][0]) for p in active}
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
            f = self.slave.pointCell[p]
            q = self.slave.pointQuadraturePoint[p]
            Ns[k] = self.slave.shapeFunctions[f][q]
            Nm[k] = self._frozenMasterShapeFunctions[p]
            nBar[k] = self._frozenNormals[p]
            weights[k] = self.slave.integrationWeights[f][q]
            slaveRef[k] = self.slave.parentReferenceCoordinates[f]
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

        f_n, stiffness = normalPenaltyForce(self.type, penaltyTimesArea, g)

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
                len(self.slave.parentNodes[self.slave.pointCell[p]])
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

            f = self.slave.pointCell[p]
            slaveShapeFunctions = self.slave.shapeFunctions[f][self.slave.pointQuadraturePoint[p]]
            masterShapeFunctions = self._frozenMasterShapeFunctions[p]
            nBar = self._frozenNormals[p]

            nSlaveDof = self.nDim * len(slaveShapeFunctions)
            nMasterDof = self.nDim * len(masterShapeFunctions)

            slaveIdcs = list(range(localOffset, localOffset + nSlaveDof))
            localOffset += nSlaveDof
            masterIdcs = list(range(localOffset, localOffset + nMasterDof))
            localOffset += nMasterDof

            slaveCoords = self.slave.parentReferenceCoordinates[f] + U_np[slaveIdcs].reshape((-1, self.nDim))
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
            penaltyTimesArea = self.penalty * self.slave.integrationWeights[f][self.slave.pointQuadraturePoint[p]]

            f_n, stiffness = normalPenaltyForce(self.type, penaltyTimesArea, g)

            globalIdcs = slaveIdcs + masterIdcs
            PExt[globalIdcs] += -f_n * w

            # K is None in explicit runs -- see ForcesOnlyExplicitEvaluation.
            if K is not None:
                K.blocks[activeIdx] += stiffness * np.outer(w, w)

            if self._friction.enabled:
                # The point as a contact point of the shared law: c, A = J_q, the frozen normal,
                # p = -f_n / J_q and dp/dU = -stiffness * w / J_q.
                area = self.slave.integrationWeights[f][self.slave.pointQuadraturePoint[p]]
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

        Ordered by slave integration cell (facet, or parent face with ``quadrature=parentFace``)
        and, within a cell, by quadrature point -- *not* by node, unlike the node-based constraint's
        accessor of the same name. A ``fromExpression`` field output reading this therefore cannot be
        tied to a node set; use a reduction (``np.max``, ``np.mean``, ...) or
        :meth:`getSlaveNodalNormalForces` instead, over a set whose size divides the point count
        (the facet set, or with ``quadrature=parentFace`` the elements owning the slave faces).
        """

        return self.slave.normalPressures(self._normalForceCurrent)

    def getGaps(self) -> np.ndarray:
        """The current gap at each contact point (negative when penetrating), ordered like
        :meth:`getNormalPressures`.

        A point with no assigned master facet reports 0, not a stale value from when it last had
        one -- see :meth:`updateConnectivity`. Returned as a copy, so a caller cannot corrupt the
        state a later augmented-Lagrange update would read.
        """

        return self._gapCurrent.copy()

    def acceptLastState(self):
        """Promote the tangential tractions of the converged iterate to the frictional history.
        Called by :meth:`~edelweissfe.models.femodel.FEModel.advanceToTime` upon increment
        acceptance."""

        self._acceptFrictionHistory()

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

        return self.slave.nodalNormalForces(self._normalForceCurrent)
