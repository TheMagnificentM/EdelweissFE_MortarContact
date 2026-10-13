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
"""Unit tests for the integrated (Gauss-point-to-segment) surface contact constraint.

The consistency of the tangent is checked here by finite differences rather than left to the
regression tests: the hexa20 contact patch test is very nearly a linear problem, so it converges in
two iterations even with a wrong tangent and would not expose one.
"""

import unittest

import numpy as np

import edelweissfe.utils.inputfileparser  # noqa: F401 bootstrap input language
from edelweissfe.constraints.surfacetodeformablesurfacepenalty import (
    Constraint as IntegratedContact,
)
from edelweissfe.constraints.surfacetodeformablesurfacepenalty import (
    SurfaceToDeformableSurfacePenaltySchema,
)
from edelweissfe.elements.contactsurfaceelement import Tria3ContactFacet
from edelweissfe.elements.displacementelement.element import DisplacementElement
from edelweissfe.fields.nodefield import NodeField
from edelweissfe.generators.surfaceelementgenerator import buildContactFacets
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.points.node import Node
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.utils.contactfriction import SLIP, STICK

_HEXA20_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)

_SIDE = 2.0

#: Face numbers of the hexa20 node ordering, per the generator's face tables.
_YMIN, _YMAX = 1, 2


def _hexa20Coordinates(yOffset: float) -> list:
    """The 20 node coordinates of a side-``_SIDE`` cube whose y span starts at ``yOffset``.

    The corner ring order is boxGen's own -- (0,0), (0,S), (S,S), (S,0) in the (x, z) plane -- which
    is what the generator's face tables were verified against. Transposing two corners still yields
    a geometrically valid cube whose face areas and shape-function integrals are unchanged, but it
    reverses the face normals, so a contact fixture built that way reports penetration where there
    is separation. Hence the orientation assertion in the tests below.
    """

    corners = [
        np.array([0.0, yOffset, 0.0]),
        np.array([0.0, yOffset, _SIDE]),
        np.array([_SIDE, yOffset, _SIDE]),
        np.array([_SIDE, yOffset, 0.0]),
        np.array([0.0, yOffset + _SIDE, 0.0]),
        np.array([0.0, yOffset + _SIDE, _SIDE]),
        np.array([_SIDE, yOffset + _SIDE, _SIDE]),
        np.array([_SIDE, yOffset + _SIDE, 0.0]),
    ]
    return corners + [0.5 * (corners[a] + corners[b]) for a, b in _HEXA20_EDGES]


def twoBlockModel(journal: Journal, penetration: float, nodalWeights: str = "facetConsistent") -> tuple:
    """Two hexa20 cubes meeting at y = 0, the lower one overlapping the upper by ``penetration``,
    with the lower block's Ymax face as slave and the upper block's Ymin face as master, both as
    contact facet sets. Shared by the tests of the facet-based contact formulations.

    Returns
    -------
    tuple
        ``(model, slaveSurface, masterSurface)``.
    """

    model = FEModel(3)
    label = 1
    elements = {}
    with model.topologyChanges():
        for key, yOffset in (("upper", 0.0), ("lower", -_SIDE + penetration)):
            nodes = []
            for x in _hexa20Coordinates(yOffset):
                node = Node(label, x)
                model.nodes[label] = node
                nodes.append(node)
                label += 1
            (elNumber,) = model.reserveElementNumbers(1)
            element = DisplacementElement("C3D20", elNumber)
            element.setNodes(nodes)
            model.createElement(element)
            elements[key] = element

        model.surfaces["masterFace"] = {_YMIN: ElementSet("m", [elements["upper"]])}
        model.surfaces["slaveFace"] = {_YMAX: ElementSet("s", [elements["lower"]])}

        masterSetName, _ = buildContactFacets(model, "masterFace", "mst", "midside", nodalWeights, journal)
        slaveSetName, _ = buildContactFacets(model, "slaveFace", "slv", "midside", nodalWeights, journal)

    return model, model.elementSets[slaveSetName], model.elementSets[masterSetName]


class TestIntegratedSurfaceContact(unittest.TestCase):
    def setUp(self):
        self.journal = Journal()

    def _twoBlockModel(self, penetration: float, nodalWeights: str = "facetConsistent") -> tuple:
        """See :func:`twoBlockModel`."""

        return twoBlockModel(self.journal, penetration, nodalWeights)

    def _constraint(self, model, slaveSurface, masterSurface, **options) -> IntegratedContact:
        configuration = SurfaceToDeformableSurfacePenaltySchema(
            penalty=options.pop("penalty", 1.0e4),
            searchDistance=options.pop("searchDistance", 1.0),
            **options,
        )
        constraint = IntegratedContact(
            "theContact", model, slaveSurface, masterSurface, self.journal, configuration=configuration
        )
        constraint.updateConnectivity(model)
        return constraint

    def _forces(self, constraint, U) -> np.ndarray:
        PExt = np.zeros(constraint.nDof)
        constraint.applyConstraint(U, np.zeros_like(U), PExt, None, None)
        return PExt

    def _assembledTangent(self, constraint, U) -> np.ndarray:
        """The tangent as one dense ``nDof x nDof`` matrix, scattered from the per-point blocks."""

        flat = np.zeros(constraint.getVIJContributionSize())
        view = constraint.shapeVIJContribution(flat)
        PExt = np.zeros(constraint.nDof)
        constraint.applyConstraint(U, np.zeros_like(U), PExt, view, None)

        K = np.zeros((constraint.nDof, constraint.nDof))
        localOffset = 0
        for block in view.blocks:
            m = block.shape[0]
            idcs = np.arange(localOffset, localOffset + m)
            K[np.ix_(idcs, idcs)] += block
            localOffset += m
        return K

    def test_tangent_is_the_negative_force_jacobian(self):
        """``K == -dPExt/dU``, by central differences, for both penalty laws.

        Every contact point stays well inside contact over the perturbation, so the gap-activation
        switch never fires and the comparison is against a smooth branch.
        """

        for contactType in ("linear", "quadratic"):
            model, slaveSurface, masterSurface = self._twoBlockModel(penetration=0.05)
            constraint = self._constraint(model, slaveSurface, masterSurface, contactType=contactType)

            rng = np.random.default_rng(0)
            U = 1e-3 * rng.standard_normal(constraint.nDof)

            K = self._assembledTangent(constraint, U)

            h = 1e-8
            KNumeric = np.zeros_like(K)
            for i in range(constraint.nDof):
                UPlus, UMinus = U.copy(), U.copy()
                UPlus[i] += h
                UMinus[i] -= h
                KNumeric[:, i] = -(self._forces(constraint, UPlus) - self._forces(constraint, UMinus)) / (2.0 * h)

            scale = max(np.abs(K).max(), 1.0)
            np.testing.assert_allclose(K, KNumeric, atol=1e-5 * scale, err_msg=f"type={contactType}")

    def test_batched_and_per_point_paths_agree(self):
        """The batched evaluation must reproduce the per-point loop, which is the reference
        implementation. Without this the loop is unreachable in tests -- every ordinary model takes
        the batched path -- and the two could drift apart unnoticed.

        Checked for both penalty laws, on forces and on the assembled tangent, at a displacement
        state where some points are closed and some are open.
        """

        for contactType in ("linear", "quadratic"):
            model, slaveSurface, masterSurface = self._twoBlockModel(penetration=0.02)
            constraint = self._constraint(model, slaveSurface, masterSurface, contactType=contactType)

            rng = np.random.default_rng(7)
            U = 2e-2 * rng.standard_normal(constraint.nDof)

            self.assertIsNotNone(constraint._batched, "this model should take the batched path")
            gaps = None

            results = {}
            for label in ("batched", "loop"):
                if label == "loop":
                    constraint._batched = None
                P = self._forces(constraint, U)
                K = self._assembledTangent(constraint, U)
                results[label] = (P.copy(), K.copy(), constraint.getGaps().copy(), constraint.totalNormalForce)
                if gaps is None:
                    gaps = constraint.getGaps().copy()

            # a mixed active set, or the comparison proves little
            self.assertTrue((gaps < 0.0).any() and (gaps >= 0.0).any(), "expected a mixed active set")

            pB, kB, gB, fB = results["batched"]
            pL, kL, gL, fL = results["loop"]
            np.testing.assert_allclose(gB, gL, rtol=1e-13, atol=1e-15, err_msg=f"gaps, {contactType}")
            np.testing.assert_allclose(pB, pL, rtol=1e-11, atol=1e-13, err_msg=f"forces, {contactType}")
            np.testing.assert_allclose(kB, kL, rtol=1e-11, atol=1e-13, err_msg=f"tangent, {contactType}")
            self.assertAlmostEqual(fB, fL, delta=1e-10 * max(abs(fL), 1.0))

    def test_tangent_is_symmetric(self):
        """The frozen-projection formulation must produce a symmetric operator; a weight
        substitution that broke the gradient/distribution transpose relation would show up here."""

        model, slaveSurface, masterSurface = self._twoBlockModel(penetration=0.05)
        constraint = self._constraint(model, slaveSurface, masterSurface)
        K = self._assembledTangent(constraint, np.zeros(constraint.nDof))
        np.testing.assert_allclose(K, K.T, atol=1e-12 * max(np.abs(K).max(), 1.0))

    def test_explicit_path_matches_the_implicit_forces(self):
        """``applyConstraintExplicit`` must deliver exactly the forces the implicit path does."""

        model, slaveSurface, masterSurface = self._twoBlockModel(penetration=0.05)
        constraint = self._constraint(model, slaveSurface, masterSurface)
        U = np.zeros(constraint.nDof)

        PImplicit = np.zeros(constraint.nDof)
        flat = np.zeros(constraint.getVIJContributionSize())
        constraint.applyConstraint(U, np.zeros_like(U), PImplicit, constraint.shapeVIJContribution(flat), None)

        PExplicit = np.zeros(constraint.nDof)
        constraint.applyConstraintExplicit(U, np.zeros_like(U), PExplicit, None)

        np.testing.assert_array_equal(PImplicit, PExplicit)

    def test_corner_nodes_receive_tensile_force(self):
        """The whole point of the formulation: with the two faces pressed flat together, the corner
        nodes of the slave face carry a NEGATIVE (tensile) normal force, of exactly the consistent
        magnitude -p*A/12, while the midside nodes carry +p*A/3."""

        model, slaveSurface, masterSurface = self._twoBlockModel(penetration=0.05)
        constraint = self._constraint(model, slaveSurface, masterSurface, penalty=1.0e4)
        self._forces(constraint, np.zeros(constraint.nDof))

        pressures = constraint.getNormalPressures()
        np.testing.assert_allclose(pressures, pressures[0], rtol=1e-12)
        p = pressures[0]
        area = _SIDE**2

        forces = constraint.getSlaveNodalNormalForces()
        nodalForce = dict(zip(constraint.slaveSurfaceNodes, forces))
        sourceNodes = model.surfaces["slaveFace"][_YMAX][0].nodes
        # Ymax face of the hexa20 ordering: corners 4..7, midsides 12..15
        corners = np.array([nodalForce[sourceNodes[i]] for i in (4, 5, 6, 7)])
        midsides = np.array([nodalForce[sourceNodes[i]] for i in (12, 13, 14, 15)])

        np.testing.assert_allclose(corners, -p * area / 12.0, rtol=1e-10)
        np.testing.assert_allclose(midsides, p * area / 3.0, rtol=1e-10)
        self.assertTrue(np.all(corners < 0.0), "the corner nodal forces must be tensile")
        self.assertAlmostEqual(corners.sum() + midsides.sum(), p * area, delta=1e-8 * p * area)

    def test_unassigned_points_report_no_stale_gap(self):
        """A point that loses its master facet must report gap 0, not the value it had while still
        assigned.

        ``_gapCurrent`` backs the public :meth:`getGaps`, whose callers cannot mask unassigned
        entries because they do not know the assignment, and the connectivity diagnostic counts
        negative entries as 'closed'. Without the reset, a point that penetrated and then dropped
        out of the search radius would keep being reported as closed forever.
        """

        model, slaveSurface, masterSurface = self._twoBlockModel(penetration=0.05)
        constraint = self._constraint(model, slaveSurface, masterSurface, searchDistance=1.0)
        self._forces(constraint, np.zeros(constraint.nDof))

        self.assertTrue((constraint.getGaps() < 0.0).any(), "expected penetrating points to start")

        # Shrink the search radius so nothing can be assigned, and re-run the connectivity update.
        constraint.searchDistance = 1e-12
        constraint.updateConnectivity(model)

        self.assertTrue(all(f is None for f in constraint._assignedFacetIdx), "expected no assignments")
        np.testing.assert_array_equal(
            constraint.getGaps(), 0.0, err_msg="unassigned points must not retain a stale gap"
        )

    def _twoElementMasterModel(self, penetration: float) -> tuple:
        """Two hexa20 cubes side by side in x forming the master surface, and one hexa20 cube
        straddling their seam as the slave, overlapping by ``penetration``.

        The point of the second master element is that its Ymin face is a *different* parent face:
        a contact point sliding in +x crosses from one parent face to the other, which is the
        reassignment that :meth:`test_reprojection_across_parent_faces_leaves_the_gap_unchanged`
        pins. The two cubes do not share nodes -- their coincident faces are topologically
        separate -- which is exactly the situation the closest-point search sees on a real mesh
        once each element has contributed its own facets.

        Returns
        -------
        tuple
            ``(model, slaveSurface, masterSurface)``.
        """

        model = FEModel(3)
        label = 1
        elements = {}
        with model.topologyChanges():
            for key, shift in (
                ("masterLeft", np.array([0.0, 0.0, 0.0])),
                ("masterRight", np.array([_SIDE, 0.0, 0.0])),
                ("slave", np.array([0.5 * _SIDE, -_SIDE + penetration, 0.0])),
            ):
                nodes = []
                for x in _hexa20Coordinates(0.0):
                    node = Node(label, x + shift)
                    model.nodes[label] = node
                    nodes.append(node)
                    label += 1
                (elNumber,) = model.reserveElementNumbers(1)
                element = DisplacementElement("C3D20", elNumber)
                element.setNodes(nodes)
                model.createElement(element)
                elements[key] = element

            model.surfaces["masterFace"] = {_YMIN: ElementSet("m", [elements["masterLeft"], elements["masterRight"]])}
            model.surfaces["slaveFace"] = {_YMAX: ElementSet("s", [elements["slave"]])}

            masterSetName, _ = buildContactFacets(
                model, "masterFace", "mst", "midside", "facetConsistent", self.journal
            )
            slaveSetName, _ = buildContactFacets(model, "slaveFace", "slv", "midside", "facetConsistent", self.journal)

        return model, model.elementSets[slaveSetName], model.elementSets[masterSetName]

    def _addDisplacementField(self, model) -> NodeField:
        """Give ``model`` a zeroed displacement field, so the constraint's connectivity search can
        be re-run at a configuration other than the reference one."""

        for node in model.nodes.values():
            node.fields["displacement"] = model.domainSize
        field = NodeField("displacement", model.domainSize, NodeSet("all", list(model.nodes.values())))
        field.createFieldValueEntry("U")
        model.nodeFields["displacement"] = field
        return field

    def _gapsFromFrozenProjection(self, constraint, model) -> np.ndarray:
        """The gap of every assigned point at the model's *current* configuration, evaluated from
        whatever projection is frozen right now.

        Deliberately independent of :meth:`applyConstraint`: it takes the frozen data directly, so
        calling it either side of an ``updateConnectivity`` isolates the effect of the re-projection
        from everything else.
        """

        slavePoints = constraint._currentSlavePointCoordinates(model)
        gaps = np.full(constraint.nPoints, np.nan)
        for p in range(constraint.nPoints):
            facetIdx = constraint._assignedFacetIdx[p]
            if facetIdx is None:
                continue
            masterCoords = constraint._currentCoordinates(
                constraint.facetElements[facetIdx].parentFaceNodes,
                model,
                constraint._masterParentRefCoords[facetIdx],
            )
            masterPoint = constraint._frozenMasterShapeFunctions[p] @ masterCoords
            gaps[p] = constraint._frozenNormals[p] @ (slavePoints[p] - masterPoint)
        return gaps

    def _slideAndReproject(self, constraint, model, field, slaveNodes, offset: np.ndarray) -> tuple:
        """Translate ``slaveNodes`` tangentially by ``offset``, re-run the connectivity search, and
        return ``(assignmentBefore, assignmentAfter, gapsBefore, gapsAfter)``, both gap fields taken
        at the *same* (slid) configuration.

        The master surface here is flat and undeformed and the slide is in its plane, so the normal
        gap is a geometric invariant of the slide: ``n`` is constant and ``n . x`` is unchanged for
        a tangential motion of either surface. Whatever the search then picks -- a different facet,
        a different parent face, a different parametric location within one -- must reproduce the
        same gap exactly. This is the sharp form of the statement that re-projection is a change of
        *representation*, not of physics.
        """

        for node in slaveNodes:
            field["U"][field._indicesOfNodesInArray[node]] = offset

        assignmentBefore = list(constraint._assignedFacetIdx)
        gapsBefore = self._gapsFromFrozenProjection(constraint, model)
        constraint.updateConnectivity(model)
        return (
            assignmentBefore,
            list(constraint._assignedFacetIdx),
            gapsBefore,
            self._gapsFromFrozenProjection(constraint, model),
        )

    def test_reprojection_within_one_parent_face_leaves_the_gap_unchanged(self):
        """Sliding the slave tangentially reassigns points among the facets tiling one parent face,
        and the gap field is unchanged to the last bit.

        This is the benign half of the reassignment behaviour, and it is worth pinning because the
        mapping it exercises -- clamped facet barycentrics through ``vertexParametricCoords`` into
        the parent face's own parametric space -- is the one place a facet swap could silently move
        the evaluation point. A single hexa20 face is tiled by six Tria3 facets, so this happens
        without any second element being involved.
        """

        model, slaveSurface, masterSurface = self._twoBlockModel(penetration=0.01)
        field = self._addDisplacementField(model)
        constraint = self._constraint(model, slaveSurface, masterSurface, searchDistance=1.0)
        slaveNodes = [node for node in slaveSurface.extractNodeSet()]

        before, after, gapsBefore, gapsAfter = self._slideAndReproject(
            constraint, model, field, slaveNodes, np.array([0.31, 0.0, 0.17])
        )

        reassigned = [p for p in range(constraint.nPoints) if before[p] != after[p]]
        self.assertTrue(reassigned, "the slide must actually reassign facets or the test is vacuous")
        for p in reassigned:
            self.assertEqual(
                constraint.facetElements[before[p]].parentFaceNodes,
                constraint.facetElements[after[p]].parentFaceNodes,
                "a single-element master offers only one parent face to reassign within",
            )

        np.testing.assert_array_equal(
            gapsAfter, gapsBefore, err_msg="re-projecting within one parent face must not move the gap"
        )

    def test_reprojection_across_parent_faces_leaves_the_gap_unchanged(self):
        """A point sliding from one master element's parent face onto the next one's reproduces the
        same gap exactly.

        This is the reassignment that actually occurs in service -- 166 times over the
        NEDSurfaceContact run -- and the one that makes an explicit trajectory a step function of
        rounding, because *which* side of the seam a borderline point lands on is decided at the
        1e-16 level. That sensitivity is legitimate only if the two branches agree where they meet,
        which is what this asserts. On a flat master they agree exactly; what remains in a deformed
        model is the genuine kink between two non-coplanar element faces, i.e. the frozen facet
        normal changing -- a property of facet-based contact, not of this mapping. See the header of
        testfiles/edelweiss-only/NEDSurfaceContact/test.inp.
        """

        model, slaveSurface, masterSurface = self._twoElementMasterModel(penetration=0.01)
        field = self._addDisplacementField(model)
        constraint = self._constraint(model, slaveSurface, masterSurface, searchDistance=1.0)
        slaveNodes = [node for node in slaveSurface.extractNodeSet()]

        before, after, gapsBefore, gapsAfter = self._slideAndReproject(
            constraint, model, field, slaveNodes, np.array([0.37, 0.0, 0.0])
        )

        crossings = [
            p
            for p in range(constraint.nPoints)
            if before[p] != after[p]
            and constraint.facetElements[before[p]].parentFaceNodes
            != constraint.facetElements[after[p]].parentFaceNodes
        ]
        self.assertTrue(crossings, "the slide must carry points across the seam or the test is vacuous")

        np.testing.assert_array_equal(
            gapsAfter, gapsBefore, err_msg="crossing a parent-face boundary must not move the gap"
        )

    def test_getGaps_returns_a_copy(self):
        """Mutating the returned array must not corrupt the state a later Uzawa update would read."""

        model, slaveSurface, masterSurface = self._twoBlockModel(penetration=0.05)
        constraint = self._constraint(model, slaveSurface, masterSurface)
        self._forces(constraint, np.zeros(constraint.nDof))

        gaps = constraint.getGaps()
        before = gaps.copy()
        gaps[:] = 1234.0
        np.testing.assert_array_equal(constraint.getGaps(), before)

    def test_open_gap_carries_no_force(self):
        """Separated surfaces produce no contact force at all."""

        model, slaveSurface, masterSurface = self._twoBlockModel(penetration=-0.01)
        constraint = self._constraint(model, slaveSurface, masterSurface)
        PExt = self._forces(constraint, np.zeros(constraint.nDof))
        np.testing.assert_array_equal(PExt, 0.0)
        self.assertEqual(constraint.totalNormalForce, 0.0)

    def test_quadrature_point_count_is_honoured(self):
        for nQuadraturePoints in (1, 3, 6):
            model, slaveSurface, masterSurface = self._twoBlockModel(penetration=0.05)
            constraint = self._constraint(model, slaveSurface, masterSurface, nQuadraturePoints=nQuadraturePoints)
            self.assertEqual(constraint.nPoints, len(slaveSurface) * nQuadraturePoints)
            self.assertEqual(len(constraint.getNormalPressures()), constraint.nPoints)

    def test_resultant_is_independent_of_the_quadrature_rule(self):
        """A uniform pressure is integrated exactly by every rule, so the transmitted resultant must
        not depend on the point count -- whereas the nodal *distribution* needs at least the 3-point
        rule to be exact."""

        resultants = []
        for nQuadraturePoints in (1, 3, 6):
            model, slaveSurface, masterSurface = self._twoBlockModel(penetration=0.05)
            constraint = self._constraint(model, slaveSurface, masterSurface, nQuadraturePoints=nQuadraturePoints)
            self._forces(constraint, np.zeros(constraint.nDof))
            resultants.append(constraint.totalNormalForce)
        np.testing.assert_allclose(resultants, resultants[0], rtol=1e-12)

    def test_finite_sliding_is_rejected(self):
        model, slaveSurface, masterSurface = self._twoBlockModel(penetration=0.05)
        with self.assertRaises(ValueError) as ctx:
            self._constraint(model, slaveSurface, masterSurface, sliding="finite")
        self.assertIn("only 'small' is implemented", str(ctx.exception))

    def test_unstamped_facets_are_rejected(self):
        """A facet built by hand, without a parent face, cannot be integrated over."""

        model, _, masterSurface = self._twoBlockModel(penetration=0.05)
        handMade = Tria3ContactFacet("Tria3ContactFacet", 9999)
        handMade.setNodes([Node(9000 + i, x) for i, x in enumerate(np.eye(3))])
        handMade.initializeElement()

        with self.assertRaises(ValueError) as ctx:
            self._constraint(model, ElementSet("hand_made", [handMade]), masterSurface)
        self.assertIn("carries no parent face", str(ctx.exception))

    def test_self_contact_is_rejected(self):
        model, slaveSurface, _ = self._twoBlockModel(penetration=0.05)
        with self.assertRaises(ValueError) as ctx:
            self._constraint(model, slaveSurface, slaveSurface)
        self.assertIn("self-contact is not supported", str(ctx.exception))

    def test_serendipity_weighted_surfaces_are_rejected(self):
        """A surface generated with nodalWeights='serendipityOptimal' must be refused, not silently
        ignored.

        There is no code path here that could honour a per-node weighting: the pressure is
        distributed with the parent face's shape functions at the quadrature points. Accepting the
        surface would give a different answer from the node-based constraint on identical input,
        with nothing in the log to say why -- and the corner mismatch the weighting exists to
        minimise is one this formulation removes outright.
        """

        model, slaveSurface, masterSurface = self._twoBlockModel(penetration=0.05, nodalWeights="serendipityOptimal")
        with self.assertRaises(ValueError) as ctx:
            self._constraint(model, slaveSurface, masterSurface)
        self.assertIn("cannot honour", str(ctx.exception))
        self.assertIn("nodalWeights='facetConsistent'", str(ctx.exception))

    def test_invalid_penalty_and_type_are_rejected(self):
        model, slaveSurface, masterSurface = self._twoBlockModel(penetration=0.05)
        with self.assertRaises(ValueError) as ctx:
            self._constraint(model, slaveSurface, masterSurface, penalty=0.0)
        self.assertIn("penalty must be positive", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            self._constraint(model, slaveSurface, masterSurface, contactType="cubic")
        self.assertIn("is not supported", str(ctx.exception))


class TestIntegratedSurfaceFriction(unittest.TestCase):
    """Coulomb friction of the integrated constraint: the shared law of
    :mod:`edelweissfe.utils.contactfriction`, evaluated at every quadrature point."""

    def setUp(self):
        self.journal = Journal()

    def _constraint(self, penetration=0.05, **options) -> IntegratedContact:
        model, slaveSurface, masterSurface = twoBlockModel(self.journal, penetration)
        configuration = SurfaceToDeformableSurfacePenaltySchema(
            penalty=1.0e4, searchDistance=1.0, contactType="quadratic", **options
        )
        constraint = IntegratedContact(
            "theContact", model, slaveSurface, masterSurface, self.journal, configuration=configuration
        )
        constraint.updateConnectivity(model)
        return constraint

    @staticmethod
    def _slide(constraint, amount) -> np.ndarray:
        """Moves the slave nodes of every point block by ``amount`` in x, plus a little noise."""

        U = 1e-4 * np.random.default_rng(5).standard_normal(constraint.nDof)
        offset = 0
        for m in constraint._activeBlockSizes():
            nSlaveDof = constraint.nDim * len(constraint._slaveShapeFunctions[0][0])
            U[offset : offset + nSlaveDof : constraint.nDim] += amount
            offset += m
        return U

    @staticmethod
    def _forces(constraint, U, dU) -> np.ndarray:
        PExt = np.zeros(constraint.nDof)
        constraint.applyConstraint(U, dU, PExt, None, None)
        return PExt

    @staticmethod
    def _assembledTangent(constraint, U, dU) -> np.ndarray:
        flat = np.zeros(constraint.getVIJContributionSize())
        view = constraint.shapeVIJContribution(flat)
        constraint.applyConstraint(U, dU, np.zeros(constraint.nDof), view, None)
        K = np.zeros((constraint.nDof, constraint.nDof))
        offset = 0
        for block in view.blocks:
            idcs = np.arange(offset, offset + block.shape[0])
            K[np.ix_(idcs, idcs)] += block
            offset += block.shape[0]
        return K

    def _checkTangent(self, constraint, U, expectedState):
        K = self._assembledTangent(constraint, U, U)
        loaded = constraint.getNormalPressures() > 0.0
        self.assertTrue(loaded.any(), "no point is in contact -- test is vacuous")
        np.testing.assert_array_equal(constraint._friction.slipState[loaded], expectedState)
        h = 1e-8
        numeric = np.zeros_like(K)
        for j in range(constraint.nDof):
            step = np.zeros(constraint.nDof)
            step[j] = h
            numeric[:, j] = -(
                self._forces(constraint, U + step, U + step) - self._forces(constraint, U - step, U - step)
            ) / (2.0 * h)
        np.testing.assert_array_equal(constraint._friction.slipState[loaded], expectedState, "state flipped")
        scale = np.abs(K).max()
        np.testing.assert_allclose(K / scale, numeric / scale, atol=1e-6)

    def test_the_stick_and_slip_tangents_differentiate_the_residual(self):
        for batched in (True, False):
            for slide, state, options in (
                (1e-5, STICK, {"mu": 10.0}),
                (5e-2, SLIP, {"mu": 0.3}),
                (5e-2, SLIP, {"mu": 0.3, "slipTolerance": 0.05}),
            ):
                constraint = self._constraint(**options)
                if not batched:
                    constraint._batched = None
                with self.subTest(batched=batched, slide=slide, **options):
                    self._checkTangent(constraint, self._slide(constraint, slide), state)

    def test_batched_and_per_point_friction_agree(self):
        constraint = self._constraint(penetration=0.02, mu=0.3)
        rng = np.random.default_rng(11)
        U = 2e-2 * rng.standard_normal(constraint.nDof)
        results = {}
        for label in ("batched", "loop"):
            if label == "loop":
                constraint._batched = None
            results[label] = (
                self._forces(constraint, U, U),
                self._assembledTangent(constraint, U, U),
                constraint.getTangentialTractions().copy(),
                constraint.getSlipStates().copy(),
            )
        states = results["batched"][3]
        self.assertTrue((states == STICK).any() or (states == SLIP).any(), "no frictional point -- vacuous")
        for b, l in zip(results["batched"], results["loop"]):
            np.testing.assert_allclose(b, l, rtol=1e-11, atol=1e-12)

    def test_sliding_points_carry_mu_times_the_pressure(self):
        constraint = self._constraint(mu=0.3)
        U = self._slide(constraint, 5e-2)
        P = self._forces(constraint, U, U)
        loaded = constraint.getNormalPressures() > 0.0
        np.testing.assert_allclose(
            constraint.getTangentialTractions()[loaded], 0.3 * constraint.getNormalPressures()[loaded], rtol=1e-12
        )
        frictionless = self._constraint(mu=0.0)
        P0 = self._forces(frictionless, U, U)
        # the frictional force resultant opposes the slide of the slave (x) and equals mu * N
        nDim = constraint.nDim
        resultant = (P - P0).reshape(-1, nDim)
        normal = -frictionless.totalNormalForce
        self.assertAlmostEqual(abs(resultant[:, 0].sum()), 0.0, delta=1e-9 * normal)  # slave and master cancel
        slaveForce = 0.0
        offset = 0
        nSlaveDof = nDim * len(constraint._slaveShapeFunctions[0][0])
        for m in constraint._activeBlockSizes():
            slaveForce += (P - P0)[offset : offset + nSlaveDof : nDim].sum()
            offset += m
        self.assertAlmostEqual(slaveForce, -0.3 * normal, delta=1e-6 * normal)

    def test_history_is_promoted_on_accept_and_restored(self):
        constraint = self._constraint(mu=10.0)
        U = self._slide(constraint, 1e-5)
        self._forces(constraint, U, U)
        np.testing.assert_array_equal(constraint._friction.tractionConverged, 0.0)
        constraint.acceptLastState()
        np.testing.assert_allclose(constraint._friction.tractionConverged, constraint._friction.tractionCurrent)
        restored = self._constraint(mu=10.0)
        restored.setRestartData({k: v.copy() for k, v in constraint.getRestartData().items()})
        np.testing.assert_array_equal(restored._friction.tractionConverged, constraint._friction.tractionConverged)
        self.assertIsNone(self._constraint().getRestartData(), "no friction, no restart data")

    def test_slip_tolerance_is_refused_explicitly(self):
        constraint = self._constraint(mu=0.3, slipTolerance=0.005)
        U = np.zeros(constraint.nDof)
        with self.assertRaises(NotImplementedError):
            constraint.applyConstraintExplicit(U, U, np.zeros(constraint.nDof), None)


if __name__ == "__main__":
    unittest.main()
