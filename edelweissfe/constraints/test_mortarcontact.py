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
"""Unit tests for the segment-to-segment mortar contact constraint.

The regression decks under ``testfiles/edelweiss-only/MortarContact*`` check the constraint through
the answers it produces on a converged model. That is the right check for the formulation as a
whole, and the wrong one for the pieces it is assembled from: a patch test on a flat interface is
very nearly a linear problem and converges in two iterations even with a defective tangent, and an
interface whose two sides overlap completely never reaches the code that decides what happens when
they do not. Both are tested here instead, on models small enough to state the expected answer in
closed form.

The pieces, in the order the constraint uses them:

* the geometry of one overlap -- clipping two faces against each other and triangulating the result
* the nodal normals the overlap is projected along
* the coupling matrices those overlaps assemble into, the row-sum identity that makes them a
  partition of the non-mortar surface, and the weighting that keeps a boundary row's multiplier a
  pressure rather than a ratio of a force to a vanishing area
* the noise floor below which the penalty branch must not read a gap as contact
* the tangent, against finite differences of the residual it claims to differentiate

NOT covered here, and worth naming so the list above is not read as a complete one: the semi-smooth
active-set indicator of the multiplier branch -- its re-decision in every Newton iteration, the
freeze once it has settled, and the anti-cycling that stops a state from recurring. Those are
exercised end to end by the regression decks, which walk a model through closing, separating and
closing again, but nothing tests them as a building block.
"""

import unittest
import warnings

import numpy as np

import edelweissfe.utils.inputfileparser  # noqa: F401 bootstrap input language
from edelweissfe.constraints.mortarcontact import Constraint as MortarContact
from edelweissfe.constraints.mortarcontact import facetNormal, isConvexPolygon
from edelweissfe.generators.surfaceelementgenerator import buildContactFacets
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.points.node import Node
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.utils.mortargeometry import (
    clipLineSegments,
    sutherlandHodgmanClip,
    tangentBasis,
    toPlaneCoordinates,
    toSpatialCoordinates,
    triangulatePolygon,
)


def _polygonArea(polygon: np.ndarray) -> float:
    """The shoelace area of a planar polygon given as a list of 2D vertices."""

    x, y = np.asarray(polygon)[:, 0], np.asarray(polygon)[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


class TestOverlapGeometry(unittest.TestCase):
    """Clipping one face against another, which is where the area that everything else is weighted
    by comes from."""

    def test_two_offset_squares_clip_to_their_analytic_overlap(self):
        """Unit square against a unit square shifted by (0.5, 0.25): the overlap is a rectangle of
        0.5 by 0.75."""

        subject = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        clip = subject + np.array([0.5, 0.25])

        overlap = np.asarray(sutherlandHodgmanClip(subject, clip))

        self.assertEqual(len(overlap), 4)
        self.assertAlmostEqual(_polygonArea(overlap), 0.5 * 0.75, places=13)

    def test_a_contained_face_clips_to_itself(self):
        """A face entirely inside the other returns its own area, not the larger one -- the case
        every non-matching interface is full of."""

        subject = np.array([[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]])
        clip = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

        overlap = np.asarray(sutherlandHodgmanClip(subject, clip))
        self.assertAlmostEqual(_polygonArea(overlap), 0.25, places=13)

    def test_disjoint_faces_clip_to_nothing(self):
        """No overlap has to mean an empty polygon rather than a degenerate one: a sliver of
        near-zero area would enter the coupling matrices as a real, tiny nodal weight, and the
        constraint divides by those weights."""

        subject = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        clip = subject + np.array([3.0, 0.0])

        overlap = sutherlandHodgmanClip(subject, clip)
        self.assertEqual(len(overlap), 0)

    def test_the_triangulation_preserves_the_area(self):
        """The overlap is integrated triangle by triangle, so the fan has to tile it exactly."""

        polygon = np.array([[0.0, 0.0], [2.0, 0.0], [2.5, 1.0], [1.0, 1.8], [-0.3, 0.9]])
        triangles = triangulatePolygon(polygon)

        total = sum(_polygonArea(np.asarray(triangle)) for triangle in triangles)
        self.assertEqual(len(triangles), len(polygon) - 2)
        self.assertAlmostEqual(total, _polygonArea(polygon), places=13)

    def test_a_non_convex_clip_window_loses_area(self):
        """Sutherland-Hodgman clips against each edge's infinite half-plane, so a non-convex clip
        window cuts away parts of the subject that lie inside it. That is a property of the
        algorithm, not a defect, and it is the reason the constraint splits quadratic faces into
        convex sub-cells before clipping rather than clipping them whole.

        Measured here so the sub-cell machinery cannot be removed as redundant without this test
        going red.
        """

        subject = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        # an L-shaped, i.e. non-convex, window covering three quarters of the subject
        window = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.5], [0.5, 0.5], [0.5, 1.0], [0.0, 1.0]])

        overlap = sutherlandHodgmanClip(subject, window)
        clippedArea = _polygonArea(np.asarray(overlap)) if len(overlap) else 0.0

        self.assertFalse(isConvexPolygon(window))
        self.assertLess(clippedArea, 0.75 - 1e-9, "the non-convex window did not lose any area -- test is vacuous")

    def test_one_dimensional_segments_clip_to_their_overlap(self):
        """The two-dimensional path intersects intervals instead of polygons, and reaches none of
        the code above."""

        slaveSegment = np.array([[0.0, 0.0], [2.0, 0.0]])
        masterSegment = np.array([[1.5, 0.0], [4.0, 0.0]])

        start, end, direction = clipLineSegments(slaveSegment, masterSegment)
        # The interval is returned in the ARC LENGTH of the slave segment, not as a fraction of it:
        # the slave runs from 0 to 2, the master covers it from 1.5 onwards, so the overlap is 0.5
        # long and begins at 1.5.
        self.assertAlmostEqual(start, 1.5, places=13)
        self.assertAlmostEqual(end - start, 0.5, places=13)
        np.testing.assert_allclose(direction, [1.0, 0.0], atol=1e-13)

    def test_the_tangent_basis_and_the_plane_map_are_inverse(self):
        """Every overlap is computed in the tangent plane of the non-mortar face and mapped back.
        A basis that is not orthonormal would distort the overlap area without changing its shape,
        which is exactly the kind of error a patch test on a uniform pressure cannot see."""

        normal = np.array([1.0, 2.0, -0.5])
        normal /= np.linalg.norm(normal)
        t1, t2 = tangentBasis(normal)

        np.testing.assert_allclose([t1 @ t1, t2 @ t2], 1.0, atol=1e-13)
        np.testing.assert_allclose([t1 @ t2, t1 @ normal, t2 @ normal], 0.0, atol=1e-13)

        origin = np.array([0.3, -1.2, 4.0])
        points = origin + np.outer([0.0, 1.0, 2.0, -1.5], t1) + np.outer([0.0, -0.7, 1.1, 2.0], t2)

        planar = toPlaneCoordinates(points, origin, t1, t2)
        np.testing.assert_allclose(toSpatialCoordinates(planar, origin, t1, t2), points, atol=1e-13)


class TestFacetNormals(unittest.TestCase):
    def test_the_facet_normal_is_a_unit_vector_along_the_winding(self):
        """Direction follows the node ordering, so a face and its reverse give opposite normals.
        The constraint never corrects the sign, it relies on the generator winding faces outward."""

        face = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

        normal = facetNormal(face)
        self.assertAlmostEqual(float(np.linalg.norm(normal)), 1.0, places=13)

        reversed_normal = facetNormal(face[::-1])
        np.testing.assert_allclose(reversed_normal, -normal, atol=1e-13)

    def test_the_facet_normal_of_a_tilted_face_is_exact(self):
        """A face tilted by 45 degrees about the x axis: the normal is known in closed form, and an
        error in it tilts every projection the overlap is computed in."""

        s = np.sqrt(0.5)
        face = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, s, s], [0.0, s, s]])

        normal = facetNormal(face)
        expected = np.array([0.0, -s, s])
        np.testing.assert_allclose(np.abs(normal), np.abs(expected), atol=1e-13)


class _TwoBlockModel:
    """A pair of flat, non-matching contact surfaces facing each other across a gap.

    Built from two single hexahedra rather than from a mesh generator: the point of these tests is
    the constraint, and a model small enough to write the expected coupling matrices down by hand is
    worth more here than a realistic one.
    """

    @staticmethod
    def build(
        gap: float = 0.0, elementType: str = "C3D8", masterShift: float = 0.0, flipNonMortar: bool = False
    ) -> FEModel:
        from edelweissfe.elements.displacementelement.element import DisplacementElement

        model = FEModel(3)
        journal = Journal(verbose=False)

        def cube(originX, originY, originZ, size, firstLabel):
            corners = [
                np.array([originX, originY, originZ]),
                np.array([originX, originY, originZ + size]),
                np.array([originX + size, originY, originZ + size]),
                np.array([originX + size, originY, originZ]),
                np.array([originX, originY + size, originZ]),
                np.array([originX, originY + size, originZ + size]),
                np.array([originX + size, originY + size, originZ + size]),
                np.array([originX + size, originY + size, originZ]),
            ]
            nodes = [Node(firstLabel + i, x) for i, x in enumerate(corners)]
            for node in nodes:
                model.nodes[node.label] = node
            return nodes

        lowerNodes = cube(0.0, 0.0, 0.0, 1.0, 1)
        upperNodes = cube(masterShift, 1.0 + gap, 0.0, 1.0, 101)

        with model.topologyChanges():
            for nodes, name in ((lowerNodes, "lower"), (upperNodes, "upper")):
                (elNumber,) = model.reserveElementNumbers(1)
                element = DisplacementElement(elementType, elNumber)
                element.setNodes(nodes)
                model.createElement(element)
                if name == "lower":
                    # Face 2 is the lower block's top, which faces the upper block. Face 1 is
                    # its bottom, on the far side of the body -- the stand-in for a surface
                    # that was declared inside out.
                    faceNumber = 1 if flipNonMortar else 2
                else:
                    faceNumber = 1
                model.surfaces[name] = {faceNumber: ElementSet(name, [element])}

            buildContactFacets(model, "lower", "nonMortar", "corner", "facetConsistent", journal, facets="wholeFace")
            buildContactFacets(model, "upper", "mortar", "corner", "facetConsistent", journal, facets="wholeFace")

        model.nodeSets["all"] = __import__("edelweissfe.sets.nodeset", fromlist=["NodeSet"]).NodeSet(
            "all", list(model.nodes.values())
        )
        return model

    @staticmethod
    def constraint(model: FEModel, journal: Journal = None, **options) -> MortarContact:
        journal = journal if journal is not None else Journal(verbose=False)
        # cn belongs to the multiplier branch; handing it to the penalty branch is refused as a
        # contradiction, so the helper only supplies it where it means something.
        settings = dict(nonMortarSurface="nonMortar_facets", mortarSurface="mortar_facets")
        if options.get("formulation", "lagrange") == "lagrange":
            settings["cn"] = 1000.0
        settings.update(options)
        return MortarContact("contact", model, journal, **settings)


class _RecordingJournal(Journal):
    """A journal that keeps what was written to it, so a diagnostic can be asserted on.

    The constraint reports through the journal like every other one in the package, so a test that
    wants to see a diagnostic has to read it there rather than from :mod:`warnings`.
    """

    def __init__(self):
        super().__init__(verbose=False)
        self.records = []

    def message(self, message: str, senderIdentification: str, level: int = 1):
        self.records.append((senderIdentification, message))


class TestConstraintSetup(unittest.TestCase):
    def test_the_surfaces_are_read_from_element_sets(self):
        """The two sides are named by element set, like every other contact constraint in this
        code base, and the elements in them have to be contact elements."""

        model = _TwoBlockModel.build()
        constraint = _TwoBlockModel.constraint(model)

        self.assertEqual(len(constraint.nonMortarFacets), 1)
        self.assertEqual(len(constraint.mortarFacets), 1)
        self.assertEqual(constraint.nonMortarFacets[0].elType, "CONQUAD4")
        self.assertEqual(len(constraint.nonMortarNodes), 4)

    def test_a_missing_element_set_is_named_in_the_error(self):
        model = _TwoBlockModel.build()
        with self.assertRaises(KeyError) as ctx:
            _TwoBlockModel.constraint(model, nonMortarSurface="doesNotExist")
        self.assertIn("doesNotExist", str(ctx.exception))

    def test_a_set_of_solid_elements_is_refused(self):
        """Not ignored: the nodes of a solid element would enter the coupling matrices as if they
        lay on the interface, and the resulting system would be quietly wrong rather than broken."""

        model = _TwoBlockModel.build()
        model.elementSets["solids"] = ElementSet("solids", list(model.elements.values())[:1])

        with self.assertRaises(ValueError) as ctx:
            _TwoBlockModel.constraint(model, nonMortarSurface="solids")
        self.assertIn("CON", str(ctx.exception))

    def test_a_surface_facing_away_from_its_partner_is_reported(self):
        """A non-mortar surface on the far side of its own body still passes the winding check --
        every facet of it is wound consistently -- and would silently bond the two bodies together
        instead of keeping them apart. Only its position relative to the other surface gives it
        away."""

        model = _TwoBlockModel.build(flipNonMortar=True)
        journal = _RecordingJournal()
        _TwoBlockModel.constraint(model, journal=journal)
        reported = [text for sender, text in journal.records if "points AWAY" in text]
        self.assertTrue(reported, f"the inverted surface was not reported; journal held {journal.records}")
        self.assertEqual([sender for sender, _ in journal.records][0], "MortarContact")

    def test_an_initial_overlap_is_not_reported_as_facing_away(self):
        """The counterpart, and the reason the check carries a margin rather than a bare sign test:
        starting the two surfaces slightly inside each other is an ordinary way to set a contact
        problem up, and puts the mortar centroid marginally behind the non-mortar one without
        anything being wrong."""

        model = _TwoBlockModel.build(gap=-0.01, masterShift=0.2)
        journal = _RecordingJournal()
        _TwoBlockModel.constraint(model, journal=journal)
        facingAway = [text for _, text in journal.records if "points AWAY" in text]
        self.assertEqual(facingAway, [], "an initial overlap was mistaken for an inverted surface")

    def test_sharing_a_node_between_the_two_surfaces_is_refused(self):
        """Two surfaces that share a node make the system singular rather than merely odd: for
        coinciding faces the two coupling matrices are equal, the weighted gap vanishes identically
        for every configuration, and the multiplier row cancels out the moment the node goes
        active. Without the check the user sees an unintelligible linear solver failure."""

        model = _TwoBlockModel.build()
        with self.assertRaises(Exception) as ctx:
            _TwoBlockModel.constraint(model, mortarSurface="nonMortar_facets")
        self.assertTrue(len(str(ctx.exception)) > 0)


class TestCouplingMatrices(unittest.TestCase):
    """The two coupling matrices, assembled from the overlaps.

    D couples the multiplier of a non-mortar node to the non-mortar displacements, C to the mortar
    ones. What makes them a consistent pair is a single identity: each row of D sums to the same
    value as the corresponding row of C. That is what lets a rigid translation of both bodies
    produce no gap, and it holds regardless of how the two meshes are related.
    """

    @staticmethod
    def _assembled(model: FEModel) -> MortarContact:
        """The constraint with its geometry evaluated once, at the undeformed configuration.

        The coupling matrices are built inside applyConstraint rather than at construction, because
        they depend on the current configuration; assembling once here is what gives the test
        something to look at.
        """

        constraint = _TwoBlockModel.constraint(model)
        nDof = constraint.nDof
        constraint.applyConstraint(
            np.zeros(nDof), np.zeros(nDof), np.zeros(nDof), np.zeros((nDof, nDof)), _frozenTimeStep()
        )
        return constraint

    def test_the_row_sums_of_the_two_coupling_matrices_agree(self):
        """The partition-of-unity identity, on a non-matching and laterally shifted interface where
        no two faces coincide."""

        model = _TwoBlockModel.build(masterShift=0.3)
        constraint = self._assembled(model)

        D = np.asarray(constraint.currentD.todense())
        C = np.asarray(constraint.currentC.todense())

        active = np.abs(D).sum(axis=1) > 1e-14
        self.assertTrue(active.any(), "no node was coupled at all -- test is vacuous")
        np.testing.assert_allclose(D[active].sum(axis=1), C[active].sum(axis=1), atol=1e-12)

    def test_the_nodal_weights_are_the_area_of_the_facets_not_of_the_overlap(self):
        """The weights are deliberately NOT the shared area.

        Rows belonging to a partly covered facet are rescaled so that their weight is the integral
        over the whole facet again, which is what keeps the multiplier a pressure of ordinary
        magnitude there instead of the ratio of a finite force to a vanishing area. Shifting the
        mortar face by 0.3 therefore leaves the trace at the full unit face rather than reducing it
        to 0.7 -- while the coverage, which the next test looks at, does drop to 0.7.

        This is the property to watch if the rescaling is ever removed: the constraint itself is
        unchanged by it -- both coupling matrices carry the same factor, so the weighted gap and the
        solution are the same -- but every reading of a nodal weight as an area depends on it.
        """

        for shift in (0.0, 0.3):
            model = _TwoBlockModel.build(masterShift=shift)
            constraint = self._assembled(model)

            D = np.asarray(constraint.currentD.todense())
            self.assertAlmostEqual(float(np.trace(D)), 1.0, places=11, msg=f"shift {shift}")

    def test_the_coverage_reports_the_shared_area(self):
        """What the weights no longer say, the coverage does: the fraction of a node's own facet
        area that the other surface opposes. One where the two faces coincide, 0.7 where the mortar
        face is shifted by 0.3 out of a unit face, and zero for a node with nothing opposite it."""

        model = _TwoBlockModel.build()
        constraint = self._assembled(model)
        np.testing.assert_allclose(constraint.currentCoverage, 1.0, atol=1e-11)

        model = _TwoBlockModel.build(masterShift=0.3)
        constraint = self._assembled(model)
        coverage = constraint.currentCoverage
        weights = np.asarray(constraint.currentNodalWeights)

        # Area-weighted mean over the nodes, which is the covered fraction of the whole face.
        self.assertAlmostEqual(float(coverage @ weights / weights.sum()), 0.7, places=11)
        self.assertTrue(np.all(coverage <= 1.0 + 1e-12))
        self.assertLess(float(coverage.min()), 1.0, "nothing is partially covered -- test is vacuous")

    def test_the_rescaling_is_the_identity_where_the_facet_is_fully_covered(self):
        """The reason no case distinction is needed: on a fully covered facet the covered and the
        whole integral coincide, so the factor is exactly one and an inner facet passes through the
        same expression as a boundary facet, untouched."""

        model = _TwoBlockModel.build()
        constraint = self._assembled(model)

        facet = constraint.nonMortarFacets[0]
        coordinates = np.array([node.coordinates for node in facet.nodes])
        fullFacetWeights = np.diag(facet.computeLocalMassMatrices(coordinates)[1])
        weights = np.asarray(constraint.currentNodalWeights)

        indices = [constraint.nonMortarNodeToIndex[node] for node in facet.nodes]
        np.testing.assert_allclose(weights[indices], fullFacetWeights, rtol=1e-11)


class TestPenaltyActivation(unittest.TestCase):
    """The penalty branch manufactures its contact pressure out of the weighted gap, which makes it
    sensitive to a kind of noise the multiplier branch is immune to.

    There the pressure is an unknown of the system, and a node that touches nothing solves to zero.
    Here it is kappa times the gap, so on a load-free interface -- where the two surfaces coincide
    and the weighted gap is the rounding of the coordinate arithmetic that produced it -- a stiffness
    of 1e6 turns 5e-16 of nothing into a pressure of 5e-10. Read against a bare `> 0` that is
    contact, and everything downstream follows: nodes flicker in and out of the active set between
    augmentations, and the outer loop measures one rounding error against another until it exhausts
    its iteration cap and reports a failure to converge on an interface that carries no load at all.
    """

    @staticmethod
    def _penaltyConstraint(model: FEModel) -> MortarContact:
        return _TwoBlockModel.constraint(model, formulation="penalty", penaltyStiffness=1e6, augmentedLagrange=True)

    @staticmethod
    def _assembleOnce(constraint: MortarContact) -> None:
        nDof = constraint.nDof
        constraint.applyConstraint(
            np.zeros(nDof), np.zeros(nDof), np.zeros(nDof), np.zeros((nDof, nDof)), _frozenTimeStep()
        )

    def test_a_load_free_interface_activates_no_node(self):
        """Two surfaces exactly in contact but carrying no load. Their weighted gap is round-off,
        and round-off is not a gap."""

        model = _TwoBlockModel.build(gap=0.0)
        constraint = self._penaltyConstraint(model)
        self._assembleOnce(constraint)

        self.assertGreater(constraint.currentGapTolerance, 0.0)
        self.assertLess(
            float(np.max(np.abs(constraint.currentWeakGap))),
            constraint.currentGapTolerance,
            "the fixture does not actually produce a round-off gap -- test is vacuous",
        )
        self.assertEqual(int(np.sum(constraint.activeSet)), 0)

    def test_the_outer_loop_stops_at_once_when_there_is_nothing_to_correct(self):
        """And therefore does not spend its iteration cap, nor report a failure to converge, on an
        interface that carries no pressure."""

        model = _TwoBlockModel.build(gap=0.0)
        constraint = self._penaltyConstraint(model)
        self._assembleOnce(constraint)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(constraint.maxAugmentations + 1):
                if not constraint.augmentConstraint():
                    break

        self.assertEqual(constraint._augmentation_counter, 1, "the loop iterated on an idle interface")
        self.assertEqual([w for w in caught if "augmented Lagrangian hit its cap" in str(w.message)], [])

    def test_a_loaded_interface_still_activates_and_still_augments(self):
        """The counterpart, so the floor above cannot be raised until it silences real contact:
        with the two bodies pressed into each other, every node is active and the loop asks for a
        second round."""

        model = _TwoBlockModel.build(gap=-0.01)
        constraint = self._penaltyConstraint(model)
        self._assembleOnce(constraint)

        self.assertEqual(int(np.sum(constraint.activeSet)), len(constraint.nonMortarNodes))
        self.assertGreater(
            float(np.max(np.abs(constraint.currentWeakGap))),
            1e3 * constraint.currentGapTolerance,
            "a genuine gap has to sit orders of magnitude above the noise floor",
        )
        self.assertTrue(constraint.augmentConstraint())
        self.assertGreater(float(np.max(constraint.augmentedMultipliers)), 0.0)


class TestConsistentTangent(unittest.TestCase):
    def test_the_tangent_differentiates_the_residual_it_assembles(self):
        """Central differences of the assembled contact forces against the assembled tangent, with
        the active set and the geometry held fixed.

        Both freezes are necessary and neither weakens the test. Differentiating across a change of
        the active set differentiates the kink of a max function, where no tangent exists; and the
        geometry is deliberately evaluated once per increment rather than per iteration, so the
        residual this tangent belongs to is the one with frozen geometry. What is checked is that
        the tangent is consistent with the residual the solver actually sees -- which is what
        governs how the Newton iteration converges.
        """

        model = _TwoBlockModel.build(gap=-0.01, masterShift=0.2)
        constraint = _TwoBlockModel.constraint(model)

        nDof = constraint.nDof
        U = np.zeros(nDof)
        dU = np.zeros(nDof)
        timeStep = _frozenTimeStep()

        # One assembly WITH the active set live, so that the constraint decides which nodes are in
        # contact on the penetrating configuration, and only then freeze it. Freezing straight away
        # would leave the set empty: it starts empty, and the only thing assembled for an inactive
        # node is the unit entry that keeps its multiplier row non-singular. Differentiating that
        # reproduces it exactly, so the comparison would pass while testing nothing -- which is what
        # this test did before, and why the check below counts active nodes rather than merely
        # asking whether the tangent is non-zero.
        constraint.applyConstraint(U, dU, np.zeros(nDof), np.zeros((nDof, nDof)), timeStep)
        self.assertGreater(int(np.sum(constraint.activeSet)), 0, "no node is in contact -- test is vacuous")
        constraint.useActiveSet = False

        P0 = np.zeros(nDof)
        K0 = np.zeros((nDof, nDof))
        constraint.applyConstraint(U, dU, P0, K0, timeStep)

        h = 1e-7
        numeric = np.zeros((nDof, nDof))
        for j in range(nDof):
            plus, minus = np.zeros(nDof), np.zeros(nDof)
            step = np.zeros(nDof)
            step[j] = h

            constraint.applyConstraint(U + step, dU, plus, np.zeros((nDof, nDof)), timeStep)
            constraint.applyConstraint(U - step, dU, minus, np.zeros((nDof, nDof)), timeStep)
            numeric[:, j] = -(plus - minus) / (2.0 * h)

        scale = float(np.max(np.abs(K0)))
        self.assertGreater(scale, 0.0)
        np.testing.assert_allclose(K0 / scale, numeric / scale, atol=5e-6)


def _frozenTimeStep():
    """One time step object, reused for every evaluation of the tangent test.

    The constraint recomputes its geometry when it sees a new increment, and identifies an increment
    by number, size and end time. Handing out a fresh step per evaluation would therefore recompute
    the geometry under the perturbation and compare the tangent against a residual it never
    assembled.
    """

    from edelweissfe.timesteppers.timestep import TimeStep

    return TimeStep(
        number=1,
        stepProgressIncrement=1.0,
        stepProgress=1.0,
        timeIncrement=1.0,
        stepTime=1.0,
        totalTime=1.0,
    )


if __name__ == "__main__":
    unittest.main()
