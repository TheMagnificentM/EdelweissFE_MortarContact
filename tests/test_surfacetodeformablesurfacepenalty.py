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
from _hexa20cube import (  # noqa: E402  (tests/ is on sys.path)
    _SIDE,
    _YMAX,
    _YMIN,
    _hexa20Coordinates,
    twoBlockModel,
)

import edelweissfe.utils.inputfileparser  # noqa: F401 bootstrap input language
from edelweissfe.constraints.base.contactpointsonslavesurface import surfaceFaceNodes
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
from edelweissfe.models.modelchange import ModelChange
from edelweissfe.models.modelchangeobserver import ModelChangeType
from edelweissfe.points.node import Node
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.utils.contactfriction import SLIP, STICK
from edelweissfe.utils.meshtools import currentNodeCoordinates


class TestIntegratedSurfaceContact(unittest.TestCase):
    """The integrated contact on surfaces discretized with flat facets (``facets=triangulated``, the
    default); :class:`TestIntegratedSurfaceContactOnWholeFaces` runs every test again on whole element
    faces (``facets=wholeFace``)."""

    #: The surface element generator's discretization of both surfaces.
    FACETS = "triangulated"

    def setUp(self):
        self.journal = Journal()

    def _twoBlockModel(self, penetration: float, nodalWeights: str = "facetConsistent") -> tuple:
        """See :func:`twoBlockModel`."""

        return twoBlockModel(self.journal, penetration, nodalWeights, facets=self.FACETS)

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
        with model.topology.changes():
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
                (elNumber,) = model.topology.reserveElementNumbers(1)
                element = DisplacementElement("C3D20", elNumber)
                element.setNodes(nodes)
                model.createElement(element)
                elements[key] = element

            model.surfaces["masterFace"] = {_YMIN: ElementSet("m", [elements["masterLeft"], elements["masterRight"]])}
            model.surfaces["slaveFace"] = {_YMAX: ElementSet("s", [elements["slave"]])}

            triangulation = "midside" if self.FACETS == "triangulated" else "corner"
            masterSetName, _ = buildContactFacets(
                model, "masterFace", "mst", triangulation, "facetConsistent", self.journal, facets=self.FACETS
            )
            slaveSetName, _ = buildContactFacets(
                model, "slaveFace", "slv", triangulation, "facetConsistent", self.journal, facets=self.FACETS
            )

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

        slavePoints = constraint.slave.currentPointCoordinates(model)
        gaps = np.full(constraint.nPoints, np.nan)
        for p in range(constraint.nPoints):
            facetIdx = constraint._assignedFacetIdx[p]
            if facetIdx is None:
                continue
            masterCoords = currentNodeCoordinates(
                surfaceFaceNodes(constraint.facetElements[facetIdx]),
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
        """Sliding the slave tangentially moves every point's closest point within the one master
        parent face; the face, not a facet, is what is assigned, and the gap is unchanged."""

        model, slaveSurface, masterSurface = self._twoBlockModel(penetration=0.01)
        field = self._addDisplacementField(model)
        constraint = self._constraint(model, slaveSurface, masterSurface, searchDistance=1.0)
        slaveNodes = [node for node in slaveSurface.extractNodeSet()]
        shapeFunctionsBefore = [N.copy() for N in constraint._frozenMasterShapeFunctions]

        before, after, gapsBefore, gapsAfter = self._slideAndReproject(
            constraint, model, field, slaveNodes, np.array([0.31, 0.0, 0.17])
        )

        self.assertEqual(before, after)
        moved = [
            p
            for p in range(constraint.nPoints)
            if not np.allclose(shapeFunctionsBefore[p], constraint._frozenMasterShapeFunctions[p])
        ]
        self.assertTrue(moved, "the slide must actually move the closest points or the test is vacuous")
        np.testing.assert_allclose(gapsAfter, gapsBefore, rtol=0.0, atol=1e-15)

    def test_curved_master_gap_is_the_signed_distance(self):
        """On a curved master face, the frozen gap of each point is its signed distance from the
        curved surface, here the parabolic cylinder y = c (x - 1)^2 which the quad8 map represents
        exactly."""

        c, penetration = 0.3, 0.02

        def curved(x):
            return np.array([x[0], x[1] + c * (x[0] - 1.0) ** 2, x[2]])

        def signedDistance(point):
            x = point[0]
            for _ in range(50):
                s = x - 1.0
                f = (x - point[0]) + (c * s * s - point[1]) * 2.0 * c * s
                df = 1.0 + 2.0 * c * (c * s * s - point[1]) + (2.0 * c * s) ** 2
                x -= f / df
            closest = np.array([x, c * (x - 1.0) ** 2, point[2]])
            # Penetration (negative) where the point lies above the master's lower face.
            return -np.linalg.norm(point - closest) if point[1] > closest[1] else np.linalg.norm(point - closest)

        model, slaveSurface, masterSurface = twoBlockModel(
            self.journal, penetration, mapping=curved, facets=self.FACETS
        )
        constraint = self._constraint(model, slaveSurface, masterSurface)
        self._forces(constraint, np.zeros(constraint.nDof))
        exact = np.array([signedDistance(x) for x in constraint.slave.currentPointCoordinates(model)])
        np.testing.assert_allclose(constraint.getGaps(), exact, rtol=0.0, atol=1e-12)

    def test_reprojection_across_parent_faces_leaves_the_gap_unchanged(self):
        """A point sliding from one master element's parent face onto the next one's reproduces the
        same gap exactly.

        This is the reassignment that occurs in service under a periodic re-search, and *which* side
        of the seam a borderline point lands on is decided at the 1e-16 level. That is harmless only
        if the two branches agree where they meet, which is what this asserts. On a flat master they
        agree exactly; on an edge between two non-coplanar faces of a deformed master they agree
        through the normal along the distance vector (tests/test_facet_edge_normal.py). See the
        header of testfiles/edelweiss-only/NEDSurfaceContact/test.inp.
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
            and surfaceFaceNodes(constraint.facetElements[before[p]])
            != surfaceFaceNodes(constraint.facetElements[after[p]])
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

    def test_refresh_ignores_changes_that_do_not_touch_the_surfaces(self):
        model, slaveSurface, masterSurface = self._twoBlockModel(penetration=0.05)
        constraint = self._constraint(model, slaveSurface, masterSurface)
        self.assertFalse(constraint.refresh(model, ModelChange(kind=ModelChangeType.REFINEMENT)))


class TestIntegratedSurfaceContactOnWholeFaces(TestIntegratedSurfaceContact):
    """Every test of the base class again on whole element faces (``facets=wholeFace``): native Gauss
    rule on each slave face, the master faces searched and projected onto as they are."""

    FACETS = "wholeFace"

    def test_quadrature_point_count_is_honoured(self):
        for n in (1, 2, 3, 4):
            model, slaveSurface, masterSurface = self._twoBlockModel(penetration=0.05)
            constraint = self._constraint(model, slaveSurface, masterSurface, gaussPointsPerDirection=n)
            # One hexa20 face.
            self.assertEqual(constraint.nPoints, n * n)
            self.assertEqual(len(constraint.getNormalPressures()), constraint.nPoints)

    def test_resultant_is_independent_of_the_quadrature_rule(self):
        resultants = []
        for n in (1, 2, 3, 4):
            model, slaveSurface, masterSurface = self._twoBlockModel(penetration=0.05)
            constraint = self._constraint(model, slaveSurface, masterSurface, gaussPointsPerDirection=n)
            self._forces(constraint, np.zeros(constraint.nDof))
            resultants.append(constraint.totalNormalForce)
        np.testing.assert_allclose(resultants, resultants[0], rtol=1e-12)

    def test_unstamped_facets_are_rejected(self):
        self.skipTest("a whole-face surface has no facets")

    def test_serendipity_weighted_surfaces_are_rejected(self):
        self.skipTest("a whole-face surface carries no nodal weights")

    def test_mixed_surfaces_are_rejected(self):
        model, slaveSurface, masterSurface = self._twoBlockModel(penetration=0.05)
        _, facetSlave, _ = twoBlockModel(self.journal, 0.05)
        mixed = ElementSet("mixed", list(slaveSurface) + list(facetSlave))
        with self.assertRaisesRegex(ValueError, "mixes whole element faces and facets"):
            self._constraint(model, mixed, masterSurface)


class TestSlaveSurfaceQuadrature(unittest.TestCase):
    """The consistent nodal loads of a uniform pressure on a curved or distorted slave face, through
    the surface element generator's parent-face stamping, against an independent high-order
    integration (issue #216, item 1)."""

    def _loadErrors(self, mapping) -> dict:
        from test_parentfacegeometry import _referenceIntegrals

        from edelweissfe.constraints.base.contactpointsonslavesurface import (
            ContactPointsOnSlaveSurface,
        )

        errors = {}
        for name, facets, nPoints in (
            ("triangulated 3", "triangulated", 3),
            ("triangulated 6", "triangulated", 6),
            ("wholeFace", "wholeFace", 3),
        ):
            _, slaveSurface, _ = twoBlockModel(Journal(), 0.0, mapping=mapping, facets=facets)
            slave = ContactPointsOnSlaveSurface(slaveSurface, nPoints, "c", 3)
            p = 1.0
            loads = dict(
                zip(slave.surfaceNodes, slave.nodalNormalForces(-p * np.concatenate(slave.integrationWeights)))
            )
            nodes = slave.parentNodes[0]
            reference = _referenceIntegrals("quad8", np.array([n.coordinates for n in nodes]))
            errors[name] = np.abs(np.array([loads[n] for n in nodes]) - p * reference).max() / np.abs(reference).max()
        return errors

    def test_curved_face(self):
        """Measured: 4.0e-3, 3.7e-5 and 4.7e-4."""

        errors = self._loadErrors(lambda x: np.array([x[0], x[1] + 0.3 * (x[0] - 1.0) ** 2, x[2]]))
        self.assertLess(errors["triangulated 3"], 5e-3)
        self.assertLess(errors["triangulated 6"], 1e-4)
        self.assertLess(errors["wholeFace"], 1e-3)

    def test_distorted_flat_face(self):
        """A trapezoid with off-centre midside nodes: exact for the 6-point facet rule and the 3 x 3
        Gauss rule, 1.3e-3 for the 3-point facet rule (exact to degree 2 only); weighting the facets
        with their flat areas instead would give 7.2e-2."""

        def distorted(x):
            return np.array([x[0] + 0.2 * x[0] * (2.0 - x[0]), x[1], x[2] * (1.0 + 0.1 * x[0])])

        errors = self._loadErrors(distorted)
        self.assertLess(errors["triangulated 3"], 2e-3)
        self.assertLess(errors["triangulated 6"], 1e-9)
        self.assertLess(errors["wholeFace"], 1e-9)

    def test_subcell_weights_equal_the_flat_facet_areas_on_an_affine_face(self):
        from edelweissfe.constraints.base.contactpointsonslavesurface import (
            ContactPointsOnSlaveSurface,
        )
        from edelweissfe.utils.facetcontactgeometry import facetNormalAndMeasure

        _, slaveSurface, _ = twoBlockModel(Journal(), 0.0)
        slave = ContactPointsOnSlaveSurface(slaveSurface, 3, "c")
        for facet, weights in zip(slave.facets, slave.integrationWeights):
            _, area = facetNormalAndMeasure(np.array([n.coordinates for n in facet.nodes]))
            np.testing.assert_allclose(weights, area / 3.0, rtol=1e-13)


class TestIntegratedSurfaceFriction(unittest.TestCase):
    """Coulomb friction of the integrated constraint: the shared law of
    :mod:`edelweissfe.utils.contactfriction`, evaluated at every quadrature point."""

    FACETS = "triangulated"

    def setUp(self):
        self.journal = Journal()

    def _configuration(self, **options) -> SurfaceToDeformableSurfacePenaltySchema:
        return SurfaceToDeformableSurfacePenaltySchema(
            penalty=1.0e4, searchDistance=1.0, contactType="quadratic", **options
        )

    def _constraint(self, penetration=0.05, **options) -> IntegratedContact:
        model, slaveSurface, masterSurface = twoBlockModel(self.journal, penetration, facets=self.FACETS)
        configuration = self._configuration(**options)
        constraint = IntegratedContact(
            "theContact", model, slaveSurface, masterSurface, self.journal, configuration=configuration
        )
        constraint.updateConnectivity(model)
        self.model = model
        return constraint

    @staticmethod
    def _slide(constraint, amount) -> np.ndarray:
        """Moves the slave nodes of every point block by ``amount`` in x, plus a little noise."""

        U = 1e-4 * np.random.default_rng(5).standard_normal(constraint.nDof)
        offset = 0
        for m in constraint._activeBlockSizes():
            nSlaveDof = constraint.nDim * len(constraint.slave.shapeFunctions[0][0])
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

    #: A slide beyond the regularized stick range of ``slipTolerance=0.05``, which is relative to the
    #: characteristic size of the slave surface elements.
    REGULARIZED_SLIDE = 5e-2

    def test_the_stick_and_slip_tangents_differentiate_the_residual(self):
        for batched in (True, False):
            for slide, state, options in (
                (1e-5, STICK, {"mu": 10.0}),
                (5e-2, SLIP, {"mu": 0.3}),
                (self.REGULARIZED_SLIDE, SLIP, {"mu": 0.3, "slipTolerance": 0.05}),
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
        nSlaveDof = nDim * len(constraint.slave.shapeFunctions[0][0])
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
        self.assertNotIn(
            "frictionTractionConverged", self._constraint().getRestartData(), "no friction, no frictional history"
        )

    def test_a_restart_keeps_the_crease_decision_of_the_uninterrupted_run(self):
        """The frozen normals are history: the first connectivity update after a restart compares
        against them whether a point slid over a crease and loses its frictional history. Here the
        last normals are turned by 90 degrees, so the uninterrupted run discards the history; the
        restarted one must do the same, which it does only if the checkpoint carries the normals."""

        constraint = self._constraint(mu=10.0)
        U = self._slide(constraint, 1e-5)
        self._forces(constraint, U, U)
        constraint.acceptLastState()
        held = np.linalg.norm(constraint._friction.tractionConverged, axis=1) > 0.0
        self.assertTrue(held.any(), "no frictional history -- test is vacuous")
        for k, normal in enumerate(constraint._frozenNormals):
            if normal is not None:
                constraint._frozenNormals[k] = np.cross(
                    normal, [0.0, 0.0, 1.0] if abs(normal[2]) < 0.9 else [1.0, 0.0, 0.0]
                )
                constraint._frozenNormals[k] /= np.linalg.norm(constraint._frozenNormals[k])
        checkpoint = {k: v.copy() for k, v in constraint.getRestartData().items()}
        uninterruptedModel = self.model

        model, slaveSurface, masterSurface = twoBlockModel(self.journal, penetration=0.05, facets=self.FACETS)
        restored = IntegratedContact(
            "theContact",
            model,
            slaveSurface,
            masterSurface,
            self.journal,
            configuration=self._configuration(mu=10.0),
        )
        restored.setRestartData(checkpoint)

        constraint.updateConnectivity(uninterruptedModel)
        restored.updateConnectivity(model)
        np.testing.assert_array_equal(constraint._friction.tractionConverged[held], 0.0)
        np.testing.assert_array_equal(restored._friction.tractionConverged, constraint._friction.tractionConverged)

    def test_slip_tolerance_is_refused_explicitly(self):
        constraint = self._constraint(mu=0.3, slipTolerance=0.005)
        U = np.zeros(constraint.nDof)
        with self.assertRaises(NotImplementedError):
            constraint.applyConstraintExplicit(U, U, np.zeros(constraint.nDof), None)


class TestIntegratedSurfaceFrictionOnWholeFaces(TestIntegratedSurfaceFriction):
    """Every friction test again on whole element faces (``facets=wholeFace``)."""

    FACETS = "wholeFace"
    # A whole face is larger than a facet, so its stick range is too.
    REGULARIZED_SLIDE = 2e-1


if __name__ == "__main__":
    unittest.main()
