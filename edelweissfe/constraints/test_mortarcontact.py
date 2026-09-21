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
* the semi-smooth active set of the multiplier branch: what decides it, that it is re-decided in
  every Newton iteration, when it stops being re-decided, and that it reads the gap correctly where
  a nodal weight has turned negative
* the tangent, against finite differences of the residual it claims to differentiate

NOT covered here, and worth naming so the list above is not read as a complete one: whether the
CONVERGED solution satisfies the Signorini conditions. That is a property of a solved model rather
than of a piece of one, and it is checked in ``tests/test_mortarsignorini.py``, which runs the
regression decks and reads the conditions off the state they converged to.
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


def _frozenTimeStep(number: int = 1):
    """One time step object, identical for every call with the same increment number.

    The constraint recomputes its geometry when it sees a new increment, and identifies an increment
    by number, size and end time. For the tangent test that means handing out the SAME step for
    every evaluation, since a fresh one would recompute the geometry under the perturbation and
    compare the tangent against a residual it never assembled. For the active-set tests it means the
    opposite: a new ``number`` is how a new increment is staged, which is what thaws a frozen set
    and refreezes the geometry, exactly as the solver does.

    Parameters
    ----------
    number
        The increment number.

    Returns
    -------
    TimeStep
        The time step.
    """

    from edelweissfe.timesteppers.timestep import TimeStep

    return TimeStep(
        number=number,
        stepProgressIncrement=1.0,
        stepProgress=1.0,
        timeIncrement=1.0,
        stepTime=1.0,
        totalTime=float(number),
    )


class _Quad9PartialOverlapModel:
    """Two nine-node contact facets in the same plane, the mortar one shifted sideways.

    The one configuration in which a nodal weight turns NEGATIVE. ``CONQUAD9`` receives no basis
    transformation -- its corner integrals are already positive over the whole facet, so it needs
    none -- and its shape functions are therefore not pointwise non-negative. Once only part of the
    facet is covered, the integral that defines the weight can run over the region where the
    function is negative and come out below zero.

    Built from contact elements directly rather than from the generator, because the generator
    cannot emit ``CONQUAD9``: it emits faces of solid elements, and EdelweissFE has no 27-node
    hexahedron for such a face to come from.
    """

    @staticmethod
    def build(shiftX: float = 0.3) -> FEModel:
        from edelweissfe.config.elementlibrary import getElementClass
        from edelweissfe.variables.fieldvariable import FieldVariable

        def facet(originX):
            return [
                [originX, 0.0, 0.0],
                [originX + 1.0, 0.0, 0.0],
                [originX + 1.0, 1.0, 0.0],
                [originX, 1.0, 0.0],
                [originX + 0.5, 0.0, 0.0],
                [originX + 1.0, 0.5, 0.0],
                [originX + 0.5, 1.0, 0.0],
                [originX, 0.5, 0.0],
                [originX + 0.5, 0.5, 0.0],
            ]

        model = FEModel(3)
        contactElementClass = getElementClass("CONQUAD9", "edelweiss")

        elements = {}
        for label, (name, originX) in enumerate((("nonMortar", 0.0), ("mortar", shiftX)), start=1):
            nodes = [Node(100 * label + i, np.array(x, dtype=float)) for i, x in enumerate(facet(originX))]
            for node in nodes:
                model.nodes[node.label] = node
                node.fields["displacement"] = FieldVariable(node, "displacement")
            element = contactElementClass("CONQUAD9", label)
            element.setNodes(nodes)
            model.elements[label] = element
            elements[name] = element

        for name, element in elements.items():
            model.elementSets[f"{name}_facets"] = ElementSet(f"{name}_facets", [element])
        return model

    @staticmethod
    def constraint(model: FEModel) -> MortarContact:
        return MortarContact(
            "contact",
            model,
            Journal(verbose=False),
            nonMortarSurface="nonMortar_facets",
            mortarSurface="mortar_facets",
            cn=1000.0,
        )


class TestActiveSet(unittest.TestCase):
    """The semi-smooth active set of the multiplier branch: what decides it, how often, and when it
    stops being decided.

    This is the machinery that chooses which nodes are in contact, and a defect in it is silent. The
    Newton iteration converges just as cleanly onto a wrong set as onto the right one -- it then
    solves a different problem. The regression decks compare converged displacements and cannot see
    that, since they never exercise the intermediate states the decision is made in.
    """

    @staticmethod
    def _setNonMortarGap(constraint: MortarContact, U: np.ndarray, openings) -> None:
        """Displace each non-mortar node along the interface normal, in its own order.

        The two blocks of the fixture face each other across y, and for linear facets D is diagonal,
        so each nodal gap follows that node's own displacement alone.
        """
        for local, node in enumerate(constraint.nonMortarNodes):
            index = constraint.nodeToGlobalIndex[node]
            U[constraint.sizeField * index + 1] = openings[local]

    @staticmethod
    def _setNodalPressure(constraint: MortarContact, U: np.ndarray, pressure: float) -> None:
        """Prescribe the nodal contact pressure through the multiplier unknowns.

        ``lambda_I = p_n * sgn(D_II)`` is the constraint's own convention, so that a positive value
        means compression for either sign of the nodal weight.
        """
        firstMultiplier = constraint.sizeField * len(constraint.nodes)
        U[firstMultiplier:] = pressure * np.sign(constraint.currentNodalWeights)

    def _assemble(self, constraint: MortarContact, U: np.ndarray, increment: int) -> None:
        nDof = constraint.nDof
        constraint.applyConstraint(
            U, np.zeros(nDof), np.zeros(nDof), np.zeros((nDof, nDof)), _frozenTimeStep(increment)
        )

    def test_the_indicator_decides_by_pressure_where_the_gap_is_closed(self):
        """At a closed gap the branch is decided by the pressure, which a penetration test cannot
        represent: the same geometry must be active under compression and released under tension.

        This is the content of the complementarity function -- active where
        ``p_n - c_n * g_sep > 0`` -- as opposed to the heuristic ``active where the node has
        penetrated``, which the two cases below are indistinguishable for.
        """

        model = _TwoBlockModel.build(gap=0.1)
        constraint = _TwoBlockModel.constraint(model)
        U = np.zeros(constraint.nDof)

        # Close the gap exactly: g_sep = 0, so the gap term of the indicator vanishes.
        self._setNonMortarGap(constraint, U, [0.1] * len(constraint.nonMortarNodes))

        self._assemble(constraint, U, 1)
        self.assertLess(
            float(np.max(np.abs(constraint.currentWeakGap))),
            1e-12,
            "the fixture did not actually close the gap -- test is vacuous",
        )
        self.assertEqual(int(np.sum(constraint.activeSet)), 0, "a vanishing indicator must not activate")

        self._setNodalPressure(constraint, U, +1.0)
        self._assemble(constraint, U, 2)
        self.assertTrue(np.all(constraint.activeSet), "compression at a closed gap must be active")

        self._setNodalPressure(constraint, U, -1.0)
        self._assemble(constraint, U, 3)
        self.assertEqual(int(np.sum(constraint.activeSet)), 0, "tension at a closed gap must release the node")

    def test_the_set_is_re_decided_in_every_iteration(self):
        """Not on a fixed schedule and not once per increment: the indicator is evaluated again in
        every iteration for as long as the set keeps changing.

        Driven through eight pairwise different patterns so that the set neither settles nor repeats
        a state, which are the two things that would legitimately stop the re-evaluation.
        """

        model = _TwoBlockModel.build(gap=0.1)
        constraint = _TwoBlockModel.constraint(model)
        U = np.zeros(constraint.nDof)

        patterns = [
            (0, 0, 0, 0),
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (1, 1, 0, 0),
            (0, 0, 1, 0),
            (1, 0, 1, 0),
            (0, 1, 1, 0),
            (1, 1, 1, 0),
        ]
        for iteration, pattern in enumerate(patterns):
            self._setNonMortarGap(constraint, U, [0.15 if p else 0.0 for p in pattern])
            self._assemble(constraint, U, 1)

            self.assertEqual(constraint.currentIteration, iteration)
            self.assertFalse(constraint.activeSetFrozen, "a set that is still moving must not be frozen")
            self.assertEqual(
                tuple(int(a) for a in constraint.activeSet),
                pattern,
                f"iteration {iteration}: the set was not re-decided",
            )

    def test_the_set_freezes_once_it_has_settled_and_thaws_next_increment(self):
        """The semi-smooth loop terminates on the set reproducing itself, not on an iteration count,
        and the freeze lasts exactly one increment.

        The second half is the more valuable one: a frozen set that is then contradicted within its
        own increment must stay frozen -- otherwise the freeze is no termination criterion at all --
        and the contradiction must be reported at the start of the next increment, which is the one
        moment a converged state is available to check it against.
        """

        model = _TwoBlockModel.build(gap=0.1)
        constraint = _TwoBlockModel.constraint(model)
        journal = _RecordingJournal()
        constraint.journal = journal
        U = np.zeros(constraint.nDof)

        for iteration in range(3):
            self._assemble(constraint, U, 1)
            self.assertEqual(constraint.currentIteration, iteration)
        self.assertTrue(constraint.activeSetFrozen, "a settled set must terminate the iteration")

        self._setNonMortarGap(constraint, U, [0.15] * len(constraint.nonMortarNodes))
        self._assemble(constraint, U, 1)
        self.assertEqual(int(np.sum(constraint.activeSet)), 0, "a frozen set must not move within its increment")

        self._assemble(constraint, U, 2)
        self.assertFalse(constraint.activeSetFrozen, "a new increment must thaw the set")
        self.assertTrue(np.all(constraint.activeSet), "the penetration must be found again")
        self.assertTrue(
            [text for _, text in journal.records if "does not reproduce itself" in text],
            f"the contradicted frozen set was not reported; journal held {journal.records}",
        )

    def test_a_node_of_negative_weight_still_opens_and_still_closes(self):
        """A regression test for a sign defect that made an open node look like a penetrating one.

        Where the nodal weight is negative both conventions flip. The nodal force along the normal
        is ``-lambda * D_II``, so compression means ``lambda < 0`` once ``D_II < 0``, which is why
        the pressure carries ``sgn(D_II)``. Translating the mortar surface away by ``a`` changes the
        weighted gap by ``D_II * a``, so an OPEN gap gives a NEGATIVE weighted gap there. The
        indicator therefore has to use the opening normalised by the SIGNED weight, which is
        positive-when-open either way. Multiplying by ``sgn(D_II)`` on top of that -- the defect --
        turns a wide open node into a penetrating one, keeps it active and transmits tension.

        Both directions are asserted: a test that merely never activates anything would pass the
        first half on its own.
        """

        model = _Quad9PartialOverlapModel.build()
        constraint = _Quad9PartialOverlapModel.constraint(model)
        U = np.zeros(constraint.nDof)

        self._assemble(constraint, U, 1)
        negative = np.flatnonzero(constraint.currentNodalWeights < 0.0)
        self.assertTrue(
            len(negative),
            "the fixture no longer produces a negative nodal weight -- test is vacuous",
        )

        normal = constraint.currentNormals[0]
        self.assertGreater(abs(normal[2]), 0.99, f"expected an out-of-plane normal, got {normal}")

        def evaluateAt(offset: float, increment: int) -> np.ndarray:
            """Translate the whole mortar facet along the normal and re-decide the set.

            A fresh increment per configuration, because a settled set would otherwise stop being
            re-decided, and because the geometry is refrozen per increment -- which is what the
            solver does too. Translating along the normal leaves the overlap in the auxiliary plane
            unchanged, so the weights, and with them the negative ones, are the same throughout.
            """
            U[:] = 0.0
            for node in constraint.mortarNodes:
                index = constraint.nodeToGlobalIndex[node]
                U[constraint.sizeField * index : constraint.sizeField * index + 3] = offset * normal
            self._assemble(constraint, U, increment)
            return constraint.activeSet.copy()

        for k, opening in enumerate((0.01, 0.05, 0.2)):
            active = evaluateAt(+opening, 10 + k)
            self.assertEqual(
                int(np.sum(active)),
                0,
                f"an interface open by {opening} activated nodes {list(np.flatnonzero(active))}; "
                f"of those, {list(np.intersect1d(np.flatnonzero(active), negative))} carry a negative weight",
            )

        for k, penetration in enumerate((0.01, 0.05)):
            active = evaluateAt(-penetration, 20 + k)
            self.assertTrue(
                np.all(active),
                f"a uniform penetration of {penetration} left nodes " f"{list(np.flatnonzero(~active))} inactive",
            )


class TestExplicitDynamics(unittest.TestCase):
    """What changes, and what must not, when this constraint is integrated explicitly.

    Two separate concerns. One is REFUSAL: of the three ways this constraint can enforce its
    condition, only the pure penalty survives an explicit increment, and the other two have to say
    so rather than run and mean something else. The other is THROTTLING: the segmentation is 93 to
    98 % of this constraint's cost, so who decides how often it runs is the difference between a
    tractable explicit analysis and an impossible one.
    """

    def _penetratingModel(self):
        """Two blocks overlapping slightly, so that the contact is active from the first call."""

        return _TwoBlockModel.build(gap=-1e-3)

    def test_lagrange_is_refused_explicitly(self):
        """The multipliers are unknowns of a system an explicit increment never solves.

        They would also carry no inertia, so the explicit update would never move them: the
        constraint would look active in every output and enforce nothing at all.
        """

        model = self._penetratingModel()
        constraint = _TwoBlockModel.constraint(model, formulation="lagrange")
        nDof = constraint.nDof
        with self.assertRaises(NotImplementedError) as raised:
            constraint.applyConstraintExplicit(
                np.zeros(nDof), np.zeros(nDof), np.zeros(nDof), _frozenTimeStep()
            )
        self.assertIn("lagrange", str(raised.exception))

    def test_augmented_lagrange_is_refused_explicitly(self):
        """The dangerous one: it would not fail, it would quietly become a pure penalty.

        ``augmentConstraint`` is called by the implicit solvers and by nobody else, so under an
        explicit solver the outer update simply never runs. Without this refusal a deck asking for
        an augmented-Lagrangian result would receive a penalty one and report nothing.
        """

        model = self._penetratingModel()
        constraint = _TwoBlockModel.constraint(
            model, formulation="penalty", penaltyStiffness=1.0e6, augmentedLagrange=True
        )
        nDof = constraint.nDof
        with self.assertRaises(NotImplementedError) as raised:
            constraint.applyConstraintExplicit(
                np.zeros(nDof), np.zeros(nDof), np.zeros(nDof), _frozenTimeStep()
            )
        self.assertIn("augmentedLagrange", str(raised.exception))

    def test_pure_penalty_is_accepted_and_transmits_force(self):
        """The variant that does work, checked on the force rather than on the absence of a raise."""

        model = self._penetratingModel()
        constraint = _TwoBlockModel.constraint(model, formulation="penalty", penaltyStiffness=1.0e6)
        nDof = constraint.nDof
        PExt = np.zeros(nDof)
        constraint.applyConstraintExplicit(np.zeros(nDof), np.zeros(nDof), PExt, _frozenTimeStep())
        self.assertGreater(
            np.max(np.abs(PExt)), 0.0, "penetrating blocks produced no contact force at all"
        )

    def test_geometry_is_rebuilt_only_when_the_solver_ticks(self):
        """The throttle itself: a new increment alone must NOT re-segment.

        This is the whole point of routing the rebuild through ``updateConnectivity``. An implicit
        solver ticks once per increment, so for it the two coincide and nothing changes. An explicit
        solver ticks every ``contact-update-frequency`` increments, and every increment in between
        has to reuse the geometry it already has -- otherwise the segmentation runs per time step,
        which is what makes an explicit mortar analysis unaffordable.
        """

        model = self._penetratingModel()
        constraint = _TwoBlockModel.constraint(model, formulation="penalty", penaltyStiffness=1.0e6)
        nDof = constraint.nDof

        rebuilds = []
        original = constraint.computeMortarCouplingMatrices

        def counting(U_np=None):
            rebuilds.append(True)
            return original(U_np)

        constraint.computeMortarCouplingMatrices = counting

        def assemble(incrementNumber):
            constraint.applyConstraint(
                np.zeros(nDof), np.zeros(nDof), np.zeros(nDof), np.zeros((nDof, nDof)),
                _frozenTimeStep(incrementNumber),
            )

        # First assembly: no solver ticked, so the constraint builds its own geometry.
        assemble(1)
        self.assertEqual(len(rebuilds), 1, "the first assembly must build the geometry")

        # Three further increments without a tick: the geometry is reused.
        for increment in (2, 3, 4):
            assemble(increment)
        self.assertEqual(
            len(rebuilds), 1, f"a new increment alone re-segmented: {len(rebuilds)} rebuilds after 4 increments"
        )

        # The solver ticks; the next assembly rebuilds, and only that one.
        constraint.updateConnectivity(model)
        assemble(5)
        assemble(6)
        self.assertEqual(len(rebuilds), 2, "a tick must cause exactly one rebuild")

    def test_the_connectivity_tick_reports_no_dof_change(self):
        """The tick marks the geometry stale; it must not ask for an equation-system rebuild.

        This constraint's DOF footprint is fixed at construction -- every node of both surfaces is in
        :attr:`nodes` from the outset, whether or not the segmentation currently couples it. Reporting
        a change would make the solver rebuild the equation system on every tick for nothing.
        """

        model = self._penetratingModel()
        constraint = _TwoBlockModel.constraint(model, formulation="penalty", penaltyStiffness=1.0e6)
        self.assertFalse(constraint.updateConnectivity(model))


if __name__ == "__main__":
    unittest.main()
