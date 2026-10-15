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
from edelweissfe.generators.surfaceelementgenerator import buildContactFacets
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.points.node import Node
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.utils.mortargeometry import (
    facetNormal,
    isConvexPolygon,
    sutherlandHodgmanClip,
    tangentBasis,
    toPlaneCoordinates,
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

    def test_the_tangent_basis_and_the_plane_map_are_inverse(self):
        """Every overlap is computed in the tangent plane of the non-mortar face.
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
        # The plane coordinates have to be the coordinates along t1 and t2, i.e. invert exactly.
        np.testing.assert_allclose(origin + np.outer(planar[:, 0], t1) + np.outer(planar[:, 1], t2), points, atol=1e-13)


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

        with model.topology.changes():
            for nodes, name in ((lowerNodes, "lower"), (upperNodes, "upper")):
                (elNumber,) = model.topology.reserveElementNumbers(1)
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
        reported = [text for sender, text in journal.records if "point away" in text]
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
        facingAway = [text for _, text in journal.records if "point away" in text]
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

    def test_the_nodal_weights_are_the_integrated_area_and_the_scaled_ones_the_facet_area(self):
        """D is integrated over the overlap only, so shifting the mortar face by 0.3 out of a unit face
        reduces its trace from 1 to 0.7. The rows the multiplier formulation works with are scaled by
        the nodal factor of Popp, Seitz, Gee & Wall (2013), Eq. (36), and the scaled weights
        D_II/kappa_I are the integrals over the whole facets again -- exactly so for a constant facet
        Jacobian -- which is what keeps the entries of the system "of constant order of magnitude"
        however little of a facet is integrated (their Section 4.2)."""

        for shift, trace in ((0.0, 1.0), (0.3, 0.7)):
            model = _TwoBlockModel.build(masterShift=shift)
            constraint = self._assembled(model)

            D = np.asarray(constraint.currentD.todense())
            self.assertAlmostEqual(float(np.trace(D)), trace, places=11, msg=f"shift {shift}")
            scaled = np.asarray(constraint.currentNodalWeights) / constraint.currentNodalScale
            np.testing.assert_allclose(scaled, 0.25, rtol=1e-11, err_msg=f"shift {shift}")

    def test_the_coverage_is_the_nodal_scaling_factor(self):
        """The coverage reported is the scaling factor of Popp et al. (2013), Eq. (36): per node, the
        parametric integral of its shape function over the integrated part of its facets, normalised
        to one for a facet integrated entirely. One where the two faces coincide; for the mortar face
        shifted by 0.3, x in [0.3, 1] is xi in [-0.4, 1], where the bilinear shape functions integrate
        to 0.49 at the nodes on the shifted side and 0.91 at the others."""

        model = _TwoBlockModel.build()
        constraint = self._assembled(model)
        np.testing.assert_allclose(constraint.currentCoverage, 1.0, atol=1e-11)

        model = _TwoBlockModel.build(masterShift=0.3)
        constraint = self._assembled(model)
        coverage = np.sort(constraint.currentCoverage)
        np.testing.assert_allclose(coverage, [0.49, 0.49, 0.91, 0.91], atol=1e-11)

    def test_the_scaling_is_the_identity_where_the_facet_is_fully_covered(self):
        """On a fully covered facet the factor is exactly one and the weights are the whole-facet
        integrals, so an inner facet passes through the same expression as a boundary facet,
        untouched."""

        model = _TwoBlockModel.build()
        constraint = self._assembled(model)

        facet = constraint.nonMortarFacets[0]
        coordinates = np.array([node.coordinates for node in facet.nodes])
        fullFacetWeights = np.diag(facet.computeLocalMassMatrices(coordinates)[1])
        weights = np.asarray(constraint.currentNodalWeights)

        indices = [constraint.nonMortarNodeToIndex[node] for node in facet.nodes]
        np.testing.assert_allclose(weights[indices], fullFacetWeights, rtol=1e-11)
        np.testing.assert_allclose(constraint.currentNodalScale[indices], 1.0, rtol=1e-11)


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

    def test_a_load_free_interface_is_closed_but_carries_no_pressure(self):
        """Two surfaces exactly in contact but carrying no load. Their weighted gap is round-off,
        and round-off is neither a gap nor a penetration: the nodes are CLOSED. A closed node keeps
        its spring in the tangent -- otherwise a body carried by the contact alone has a singular
        tangent the moment it touches -- but its pressure is exactly zero, not kappa times the
        rounding of the geometry."""

        model = _TwoBlockModel.build(gap=0.0)
        constraint = self._penaltyConstraint(model)
        self._assembleOnce(constraint)

        self.assertGreater(constraint.currentGapTolerance, 0.0)
        self.assertLess(
            float(np.max(np.abs(constraint.currentWeakGap))),
            constraint.currentGapTolerance,
            "the fixture does not actually produce a round-off gap -- test is vacuous",
        )
        self.assertEqual(int(np.sum(constraint.activeSet)), len(constraint.nonMortarNodes))
        self.assertEqual(float(np.max(np.abs(constraint.nodalMultipliers))), 0.0)

    def test_a_closed_interface_keeps_its_stiffness(self):
        """The reason the kink is put on the active side: the assembled tangent of a closed,
        load-free interface is not zero, so the contact still carries a body that rests on it."""

        model = _TwoBlockModel.build(gap=0.0)
        constraint = self._penaltyConstraint(model)
        nDof = constraint.nDof
        K = np.zeros((nDof, nDof))
        constraint.applyConstraint(np.zeros(nDof), np.zeros(nDof), np.zeros(nDof), K, _frozenTimeStep())

        self.assertGreater(float(np.max(np.abs(K))), 0.0)

    def test_an_open_interface_activates_no_node(self):
        """The counterpart to the closed case: a genuine opening, orders of magnitude above the
        noise floor, releases every node."""

        model = _TwoBlockModel.build(gap=0.01)
        constraint = self._penaltyConstraint(model)
        self._assembleOnce(constraint)

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
        # A vanishing indicator is the kink of the complementarity function, where either branch is
        # admissible; the constraint puts it on the ACTIVE side, so that a body resting on a closed,
        # unloaded interface does not get a singular tangent (see the activation in applyConstraint).
        self.assertTrue(np.all(constraint.activeSet), "a closed gap without pressure counts as closed")

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
            [text for _, text in journal.records if "not self-consistent" in text],
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
            constraint.applyConstraintExplicit(np.zeros(nDof), np.zeros(nDof), np.zeros(nDof), _frozenTimeStep())
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
            constraint.applyConstraintExplicit(np.zeros(nDof), np.zeros(nDof), np.zeros(nDof), _frozenTimeStep())
        self.assertIn("augmentedLagrange", str(raised.exception))

    def test_pure_penalty_is_accepted_and_transmits_force(self):
        """The variant that does work, checked on the force rather than on the absence of a raise."""

        model = self._penetratingModel()
        constraint = _TwoBlockModel.constraint(model, formulation="penalty", penaltyStiffness=1.0e6)
        nDof = constraint.nDof
        PExt = np.zeros(nDof)
        constraint.applyConstraintExplicit(np.zeros(nDof), np.zeros(nDof), PExt, _frozenTimeStep())
        self.assertGreater(np.max(np.abs(PExt)), 0.0, "penetrating blocks produced no contact force at all")

    def test_geometry_is_rebuilt_only_when_the_solver_ticks(self):
        """The throttle itself: a new increment alone must NOT re-segment.

        This is the whole point of routing the rebuild through ``updateConnectivity``. An implicit
        solver ticks once per increment, so for it the two coincide and nothing changes. An explicit
        solver ticks every ``contact-update-frequency`` increments, and every increment in between
        has to reuse the geometry it already has -- otherwise the segmentation runs per time step,
        which is what makes an explicit mortar analysis unaffordable. The throttle belongs to the
        frozen geometry; a consistent one is rebuilt at every iterate by definition.
        """

        model = self._penetratingModel()
        constraint = _TwoBlockModel.constraint(
            model, formulation="penalty", penaltyStiffness=1.0e6, geometryUpdate="frozen"
        )
        nDof = constraint.nDof

        rebuilds = []
        original = constraint._couple

        def counting(U_np, normals, withDerivatives):
            rebuilds.append(True)
            return original(U_np, normals, withDerivatives)

        constraint._couple = counting

        def assemble(incrementNumber):
            constraint.applyConstraint(
                np.zeros(nDof),
                np.zeros(nDof),
                np.zeros(nDof),
                np.zeros((nDof, nDof)),
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

    def _explicitConstraint(self, model, lumpedMassPerDof=1.0, **options):
        """A penalty constraint set up as an explicit solver would leave it.

        The solver hands the lumped mass over through
        ``computeCriticalTimeStepForExplicitDynamics`` before the increment loop and marks the
        run explicit on the first assembly; both are done here by hand so the stability logic can
        be tested without a solver.
        """

        constraint = _TwoBlockModel.constraint(model, formulation="penalty", **options)
        mass = np.full(constraint.nDof, lumpedMassPerDof)
        constraint.computeCriticalTimeStepForExplicitDynamics(mass)
        return constraint, mass

    def _assembleOnce(self, constraint, timeIncrement, increment=1):
        nDof = constraint.nDof
        constraint._explicit = True
        constraint._explicitTimeIncrement = timeIncrement
        constraint.applyConstraint(
            np.zeros(nDof),
            np.zeros(nDof),
            np.zeros(nDof),
            np.zeros((nDof, nDof)),
            _frozenTimeStep(increment),
        )

    def test_the_stability_bound_is_infinite_before_the_geometry_exists(self):
        """Asked before the first assembly it cannot answer, and must not pretend to.

        The segmentation needs a solution vector that this hook is not given, so on the call the
        solver makes before its increment loop there is nothing to bound yet. Returning anything
        finite there would cap the time step on no evidence.
        """

        model = self._penetratingModel()
        constraint, mass = self._explicitConstraint(model, penaltyStiffness=1.0e6)
        self.assertEqual(constraint.computeCriticalTimeStepForExplicitDynamics(mass), np.inf)

    def test_the_stability_bound_scales_as_one_over_the_square_root_of_the_stiffness(self):
        """dt <= 2/sqrt(kappa*R): quadrupling the stiffness must halve the stable increment.

        R carries the geometry and the inertia and not the stiffness, so this scaling is the
        defining property of the bound rather than an incidental one -- and it is checkable
        without knowing R at all.
        """

        bounds = {}
        for kappa in (1.0e6, 4.0e6):
            model = self._penetratingModel()
            constraint, mass = self._explicitConstraint(model, penaltyStiffness=kappa)
            self._assembleOnce(constraint, timeIncrement=1.0e-6)
            bounds[kappa] = constraint.computeCriticalTimeStepForExplicitDynamics(mass)
            self.assertTrue(np.isfinite(bounds[kappa]), "no bound was produced once the geometry exists")
        self.assertAlmostEqual(
            bounds[1.0e6] / bounds[4.0e6],
            2.0,
            places=10,
            msg=f"quadrupling kappa changed the bound by {bounds[1.0e6] / bounds[4.0e6]}, not 2",
        )

    def test_a_derived_stiffness_satisfies_its_own_stability_bound(self):
        """The property the whole mechanism exists for, checked end to end.

        With no ``penaltyStiffness`` given, the stiffness is derived from the stability limit of
        the time increment in use. The test of that derivation is not what number it produces but
        whether the number it produces is one the increment can carry: the bound it then reports
        must not be smaller than the increment it was derived for.
        """

        timeIncrement = 1.0e-6
        model = self._penetratingModel()
        constraint, mass = self._explicitConstraint(model)
        self.assertTrue(constraint._derive_epsN, "the fixture was expected to leave eps_N underived")

        self._assembleOnce(constraint, timeIncrement)

        bound = constraint.computeCriticalTimeStepForExplicitDynamics(mass)
        self.assertGreaterEqual(
            bound,
            timeIncrement,
            f"the derived eps_N = {constraint.epsN:.4g} permits only {bound:.4g}, "
            f"below the increment {timeIncrement:.4g} it was derived for",
        )
        # Derived at half the limit, so the bound should sit near sqrt(2) above the increment
        # rather than far above it -- a stiffness far below the limit would be needlessly soft.
        self.assertLess(
            bound,
            4.0 * timeIncrement,
            f"the derived eps_N = {constraint.epsN:.4g} is far softer than the increment requires",
        )

    def test_a_softer_stiffness_is_derived_for_a_larger_time_increment(self):
        """The derivation must follow the increment, since that is what it is derived from."""

        stiffnesses = {}
        for timeIncrement in (1.0e-6, 4.0e-6):
            model = self._penetratingModel()
            constraint, _ = self._explicitConstraint(model)
            self._assembleOnce(constraint, timeIncrement)
            stiffnesses[timeIncrement] = constraint.epsN
        self.assertAlmostEqual(
            stiffnesses[1.0e-6] / stiffnesses[4.0e-6],
            16.0,
            places=6,
            msg="eps_N ~ 1/dt^2 was expected, since omega^2 = eps_N*R <= 4/dt^2",
        )

    def test_other_constraints_keep_the_meshs_time_step(self):
        """The hook is additive: anything that does not implement it must not constrain anything.

        This is what keeps every constraint in the package -- the other contact formulations
        above all -- behaving exactly as it did before the hook existed.
        """

        from edelweissfe.constraints.base.constraintbase import ConstraintBase

        model = self._penetratingModel()
        constraint = _TwoBlockModel.constraint(model, formulation="penalty", penaltyStiffness=1.0e6)
        self.assertEqual(
            ConstraintBase.computeCriticalTimeStepForExplicitDynamics(constraint, np.ones(constraint.nDof)),
            np.inf,
        )

    def test_the_connectivity_tick_reports_no_dof_change(self):
        """The tick marks the geometry stale; it must not ask for an equation-system rebuild.

        This constraint's DOF footprint is fixed at construction -- every node of both surfaces is in
        :attr:`nodes` from the outset, whether or not the segmentation currently couples it. Reporting
        a change would make the solver rebuild the equation system on every tick for nothing.
        """

        model = self._penetratingModel()
        constraint = _TwoBlockModel.constraint(model, formulation="penalty", penaltyStiffness=1.0e6)
        self.assertFalse(constraint.updateConnectivity(model))


class TestCurvedEdgeAttribution(unittest.TestCase):
    """Quadratic facets are segmented on their linear sub-segments (Popp, Gitterle, Gee & Wall 2010,
    Section 4.2). Where an edge shared by two mortar facets is CURVED in the surface, the sub-segments
    follow its chord, and the strip between chord and edge is clipped to the wrong facet. Carried into
    the parent by the affine sub-segment map of their Eq. (42), a point there stays with that facet,
    and a uniform multiplier reproduces the consistent mortar-side loads of a uniform pressure only up
    to an error proportional to the curvature of the edge -- measured 5.8e-3, 1.2e-2 and 4.4e-2 for
    bulges of 0.02, 0.04 and 0.15.

    Here every Gauss point is mapped back by the inverse of the whole projected facet and attributed
    to the facet that contains it (see :func:`edelweissfe.utils.mortarcoupling._locate`); what remains
    is the quadrature of the cells that straddle the curved edge.

    Fixture: one flat CONQUAD8 non-mortar facet on the unit square, opposed by two CONQUAD8 mortar
    facets that tile the same square along a shared edge whose mid-side node is moved off the chord.
    Both surfaces cover exactly the same area, so every discrepancy is the treatment of the edge.
    """

    @staticmethod
    def build(bulge: float = 0.15) -> FEModel:
        from edelweissfe.config.elementlibrary import getElementClass
        from edelweissfe.variables.fieldvariable import FieldVariable

        model = FEModel(3)
        elementClass = getElementClass("CONQUAD8", "edelweiss")
        pool = {}

        def node(x):
            key = tuple(np.round(x, 12))
            if key not in pool:
                n = Node(1000 + len(model.nodes), np.array(x, dtype=float))
                n.fields["displacement"] = FieldVariable(n, "displacement")
                model.nodes[n.label] = n
                pool[key] = n
            return pool[key]

        def facet(label, points):
            element = elementClass("CONQUAD8", label)
            element.setNodes([node(p) for p in points])
            model.elements[label] = element
            return element

        nonMortar = facet(
            1, [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0, 0], [1, 0.5, 0], [0.5, 1, 0], [0, 0.5, 0]]
        )
        pool.clear()  # the two surfaces do not share nodes
        c = 0.5 + bulge
        left = facet(
            2, [[0, 0, 0], [0.5, 0, 0], [0.5, 1, 0], [0, 1, 0], [0.25, 0, 0], [c, 0.5, 0], [0.25, 1, 0], [0, 0.5, 0]]
        )
        right = facet(
            3, [[0.5, 0, 0], [1, 0, 0], [1, 1, 0], [0.5, 1, 0], [0.75, 0, 0], [1, 0.5, 0], [0.75, 1, 0], [c, 0.5, 0]]
        )
        model.elementSets["nonMortar_facets"] = ElementSet("nonMortar_facets", [nonMortar])
        model.elementSets["mortar_facets"] = ElementSet("mortar_facets", [left, right])
        return model

    @staticmethod
    def mortarConsistencyError(constraint: MortarContact) -> float:
        """max |C^T 1 - int N_m| / max |int N_m| -- zero when a uniform multiplier reproduces the
        consistent mortar-side nodal loads of a uniform pressure."""
        _, C = constraint.computeMortarCouplingMatrices()
        loads = np.zeros(constraint.nMortarNodes)
        xg, wg = np.polynomial.legendre.leggauss(12)
        for el in constraint.mortarFacets:
            X = np.array([n.coordinates for n in el.nodes])
            for a, wa in zip(xg, wg):
                for b, wb in zip(xg, wg):
                    xi = np.array([a, b])
                    dX = el.getShapeFunctionDerivatives(xi) @ X
                    J = np.linalg.norm(np.cross(dX[0], dX[1]))
                    for k, n in enumerate(el.nodes):
                        loads[constraint.mortarNodeToIndex[n]] += el.getShapeFunctions(xi)[k] * J * wa * wb
        return float(np.max(np.abs(np.asarray(C.sum(axis=0)).ravel() - loads)) / np.max(np.abs(loads)))

    def test_points_are_attributed_to_the_facet_that_contains_them(self):
        errors, moved = [], []
        for bulge in (0.0, 0.02, 0.04, 0.15):
            model = self.build(bulge)
            constraint = MortarContact(
                "contact",
                model,
                Journal(verbose=False),
                nonMortarSurface="nonMortar_facets",
                mortarSurface="mortar_facets",
                cn=1000.0,
            )
            errors.append(self.mortarConsistencyError(constraint))
            moved.append(constraint.lastCoupling.reattributedGaussPoints)

        # Exact for a straight edge; measured 9.1e-5, 3.2e-4 and 2.1e-4 for the curved ones, against
        # 5.8e-3, 1.2e-2 and 4.4e-2 with the affine sub-segment map.
        self.assertLess(errors[0], 1e-14)
        self.assertEqual(moved[0], 0)
        self.assertLess(max(errors[1:]), 5e-4)
        self.assertGreater(moved[3], 0)


class TestFriction(unittest.TestCase):
    """Penalty-regularised Coulomb friction of the penalty branch (``mu > 0``).

    The two blocks of :class:`_TwoBlockModel` touch across the plane y = 1, so the nodal normals are
    the y axis and the tangent plane is x-z. A tangential increment is staged by moving the mortar
    (upper) block in x through ``dU``, which is what the slip measure reads.

    Tests that state an exact property of the slip measure of the frozen geometry -- ``D dU_s - C
    dU_m`` within an increment and its correction at a rebuild -- and the finite-difference checks of
    its tangent run with ``geometryUpdate=frozen``. The consistent geometry measures the objective slip
    of Gitterle et al. (2010), Eq. (53), which equals the relative motion only up to second order where
    the overlap changes, as it does here; and its tangent cannot be checked by finite differences on
    these blocks, whose facet edges coincide in z, which puts the overlap on a kink. It is checked in
    ``tests/test_mortarlinearization.py``.
    """

    penaltyStiffness = 1.0e3

    def _constraint(self, mu, gap=-0.01, **options):
        model = _TwoBlockModel.build(gap=gap, masterShift=0.2)
        return _TwoBlockModel.constraint(
            model, formulation="penalty", penaltyStiffness=self.penaltyStiffness, mu=mu, **options
        )

    @staticmethod
    def _masterSlide(constraint, amount):
        """The increment that moves every mortar node by ``amount`` in x."""

        dU = np.zeros(constraint.nDof)
        sf = constraint.sizeField
        for i in range(constraint.nNonMortarNodes, len(constraint.nodes)):
            dU[sf * i] = amount
        return dU

    @staticmethod
    def _assemble(constraint, U, dU, timeStep):
        nDof = constraint.nDof
        P, K = np.zeros(nDof), np.zeros((nDof, nDof))
        constraint.applyConstraint(U, dU, P, K, timeStep)
        return P, K

    def _finiteDifferenceTangent(self, constraint, U, dU, timeStep, h=1e-8):
        """-dP/dU with U and dU perturbed together, as a Newton correction moves both."""

        nDof = constraint.nDof
        numeric = np.zeros((nDof, nDof))
        for j in range(nDof):
            step = np.zeros(nDof)
            step[j] = h
            plus, _ = self._assemble(constraint, U + step, dU + step, timeStep)
            minus, _ = self._assemble(constraint, U - step, dU - step, timeStep)
            numeric[:, j] = -(plus - minus) / (2.0 * h)
        return numeric

    def test_friction_is_refused_for_the_multiplier_branch(self):
        model = _TwoBlockModel.build(gap=-0.01)
        with self.assertRaises(ValueError):
            _TwoBlockModel.constraint(model, formulation="lagrange", mu=0.3)

    def test_without_tangential_motion_friction_changes_nothing(self):
        """No slip and no history: the frictional forces vanish and the normal ones are untouched."""

        timeStep = _frozenTimeStep()
        frictionless = self._constraint(mu=0.0)
        frictional = self._constraint(mu=0.5)
        U = np.zeros(frictionless.nDof)
        P0, K0 = self._assemble(frictionless, U, U, timeStep)
        P1, K1 = self._assemble(frictional, U, U, timeStep)
        self.assertGreater(int(np.sum(frictional.activeSet)), 0, "no node is in contact -- test is vacuous")
        np.testing.assert_allclose(P1, P0, rtol=0.0, atol=1e-14 * np.max(np.abs(P0)))
        np.testing.assert_allclose(np.abs(frictional._friction.tractionCurrent), 0.0, atol=1e-14)

    def _checkTangent(self, mu, slide, expectedState, **options):
        timeStep = _frozenTimeStep()
        constraint = self._constraint(mu=mu, geometryUpdate="frozen", **options)
        U = self._masterSlide(constraint, slide)
        dU = U.copy()
        self._assemble(constraint, U, dU, timeStep)
        active = constraint.activeSet.copy()
        self.assertGreater(int(np.sum(active)), 0, "no node is in contact -- test is vacuous")
        np.testing.assert_array_equal(constraint._friction.slipState[active], expectedState)
        constraint.useActiveSet = False

        _, K = self._assemble(constraint, U, dU, timeStep)
        numeric = self._finiteDifferenceTangent(constraint, U, dU, timeStep)
        np.testing.assert_array_equal(constraint._friction.slipState[active], expectedState, "state flipped under h")
        scale = float(np.max(np.abs(K)))
        np.testing.assert_allclose(K / scale, numeric / scale, atol=1e-6)
        return constraint, K

    def test_the_stick_tangent_differentiates_the_residual_it_assembles(self):
        self._checkTangent(mu=10.0, slide=1e-4, expectedState=1)

    def test_the_slip_tangent_differentiates_the_residual_it_assembles(self):
        """The slip branch, including the non-symmetric normal-tangential coupling."""

        _, K = self._checkTangent(mu=0.2, slide=0.05, expectedState=2)
        self.assertGreater(float(np.max(np.abs(K - K.T))), 1e-8 * float(np.max(np.abs(K))), "slip tangent is symmetric")

    def test_the_pressure_dependent_stick_tangent_differentiates_its_residual(self):
        """slipTolerance: eps_T = mu p / gamma_crit, so the stick tangent carries a pressure term."""

        constraint, _ = self._checkTangent(mu=0.3, slide=1e-4, expectedState=1, slipTolerance=0.05)
        self.assertGreater(constraint._friction.elasticSlip, 0.0)

    def test_the_pressure_dependent_slip_tangent_differentiates_its_residual(self):
        self._checkTangent(mu=0.3, slide=0.05, expectedState=2, slipTolerance=0.005)

    def test_the_elastic_slip_is_the_same_at_every_pressure(self):
        """With slipTolerance a sticking node carries mu p (slide / gamma_crit), proportional to p."""

        slide = 1e-4
        constraint = self._constraint(mu=0.3, slipTolerance=0.05, geometryUpdate="frozen")
        U = self._masterSlide(constraint, slide)
        self._assemble(constraint, U, U, _frozenTimeStep())
        active = constraint.activeSet
        np.testing.assert_array_equal(constraint._friction.slipState[active], 1)
        pressure = np.abs(constraint.nodalMultipliers[active])
        expected = 0.3 * pressure * slide / constraint._friction.elasticSlip
        np.testing.assert_allclose(constraint._friction.tractionCurrent[active, 0], expected, rtol=1e-10)

    def test_slip_tolerance_and_tangent_penalty_are_exclusive(self):
        model = _TwoBlockModel.build(gap=-0.01)
        with self.assertRaises(ValueError):
            _TwoBlockModel.constraint(model, formulation="penalty", mu=0.3, slipTolerance=0.005, tangentPenalty=1e3)

    def test_a_node_without_pressure_has_no_friction_and_no_tangential_stiffness(self):
        """A closed node at zero pressure: no stick spring may tie the surfaces over a cone of radius 0."""

        timeStep = _frozenTimeStep()
        frictionless = _TwoBlockModel.constraint(
            _TwoBlockModel.build(gap=0.0, masterShift=0.2), formulation="penalty", penaltyStiffness=1e3
        )
        frictional = self._constraint(mu=0.5, gap=0.0)
        U = self._masterSlide(frictional, 1e-4)
        P0, K0 = self._assemble(frictionless, U, U, timeStep)
        P1, K1 = self._assemble(frictional, U, U, timeStep)
        self.assertGreater(int(np.sum(frictional.activeSet)), 0, "no node is closed -- test is vacuous")
        np.testing.assert_allclose(frictional.getNormalPressures(), 0.0, atol=1e-10)  # round-off of a zero gap
        np.testing.assert_allclose(K1, K0, atol=0.0)
        np.testing.assert_allclose(P1, P0, atol=0.0)
        np.testing.assert_array_equal(frictional._friction.slipState[frictional.activeSet], 2)

    def test_a_slipping_node_carries_mu_times_its_pressure_against_the_slide(self):
        mu = 0.2
        constraint = self._constraint(mu=mu)
        U = self._masterSlide(constraint, 0.05)
        self._assemble(constraint, U, U, _frozenTimeStep())

        active = constraint.activeSet
        pressure = np.abs(constraint.nodalMultipliers[active])
        traction = constraint._friction.tractionCurrent[active]
        np.testing.assert_allclose(np.linalg.norm(traction, axis=1), mu * pressure, rtol=1e-12)
        # Purely tangential, and on the slave side it drags along with the mortar block: +x.
        np.testing.assert_allclose(traction[:, 1:], 0.0, atol=1e-12 * float(np.max(pressure)))
        self.assertTrue(np.all(traction[:, 0] > 0.0))

    def test_a_sticking_node_carries_eps_T_times_the_relative_slide(self):
        """In stick, the traction is the pointwise penalty law: eps_T times the relative motion."""

        slide = 1e-4
        constraint = self._constraint(mu=10.0, tangentPenalty=2.0e3, geometryUpdate="frozen")
        U = self._masterSlide(constraint, slide)
        self._assemble(constraint, U, U, _frozenTimeStep())
        active = constraint.activeSet
        np.testing.assert_allclose(constraint._friction.tractionCurrent[active, 0], 2.0e3 * slide, rtol=1e-10)

    def test_the_history_is_promoted_only_on_acceptance(self):
        """A cutback re-attempt must start from the last ACCEPTED traction, not from the failed iterate."""

        slide = 1e-4
        constraint = self._constraint(mu=10.0)
        U = self._masterSlide(constraint, slide)
        self._assemble(constraint, U, U, _frozenTimeStep(1))
        afterFirst = constraint._friction.tractionCurrent.copy()
        np.testing.assert_allclose(constraint._friction.tractionConverged, 0.0)

        constraint.acceptLastState()
        np.testing.assert_allclose(constraint._friction.tractionConverged, afterFirst)

        # Next increment, no further motion: the accepted traction is held (stick).
        self._assemble(constraint, U, np.zeros_like(U), _frozenTimeStep(2))
        active = constraint.activeSet
        np.testing.assert_allclose(constraint._friction.tractionCurrent[active], afterFirst[active], rtol=1e-10)

        # A failed attempt that slid further back is never accepted; the retry sees the old history.
        self._assemble(constraint, U, -self._masterSlide(constraint, 5 * slide), _frozenTimeStep(3))
        self._assemble(constraint, U, np.zeros_like(U), _frozenTimeStep(4))
        np.testing.assert_allclose(constraint._friction.tractionCurrent[active], afterFirst[active], rtol=1e-10)

    def test_the_tangential_spring_enters_the_stability_bound(self):
        """With friction the bound covers the stick spring too, and never gets less conservative."""

        bounds = {}
        for mu in (0.0, 0.3):
            model = _TwoBlockModel.build(gap=-1e-3)
            constraint = _TwoBlockModel.constraint(model, formulation="penalty", penaltyStiffness=1e6, mu=mu)
            mass = np.ones(constraint.nDof)
            constraint.computeCriticalTimeStepForExplicitDynamics(mass)
            constraint._explicit = True
            constraint._explicitTimeIncrement = 1e-6
            self._assemble(constraint, np.zeros(constraint.nDof), np.zeros(constraint.nDof), _frozenTimeStep())
            bounds[mu] = constraint.computeCriticalTimeStepForExplicitDynamics(mass)
            rows = constraint._stabilityRowSums(mass)
            if mu > 0.0:
                self.assertGreater(float(np.max(rows[1])), 0.0, "the tangential rows are missing")
        self.assertLessEqual(bounds[0.3], bounds[0.0])

    def test_restart_data_round_trips(self):
        constraint = self._constraint(mu=10.0)
        U = self._masterSlide(constraint, 1e-4)
        self._assemble(constraint, U, U, _frozenTimeStep())
        constraint.acceptLastState()
        data = {k: v.copy() for k, v in constraint.getRestartData().items()}

        restored = self._constraint(mu=10.0)
        restored.setRestartData(data)
        np.testing.assert_allclose(restored._friction.tractionConverged, constraint._friction.tractionConverged)
        np.testing.assert_allclose(restored._friction.pressureConverged, constraint._friction.pressureConverged)
        np.testing.assert_array_equal(restored._friction.activeConverged, constraint._friction.activeConverged)
        np.testing.assert_array_equal(restored.frictionGapReference, constraint.frictionGapReference)
        # the normals of the last rebuild, which the first rebuild after the restart compares against
        # for the crease test
        np.testing.assert_array_equal(restored._restoredFrictionNormals, constraint.currentNormals)
        self._assemble(restored, U, np.zeros_like(U), _frozenTimeStep(2))
        self.assertFalse(hasattr(restored, "_restoredFrictionNormals"), "the restored normals were not used")

    # ------------------------------------------------------------------ objective slip

    def _rigidRotation(self, constraint, theta):
        """The displacement of a rigid rotation of BOTH blocks by ``theta`` about the z axis."""

        rotation = np.array([[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0, 0, 1]])
        centre = np.array([0.5, 1.0, 0.5])
        X = constraint._X
        motion = (X - centre) @ rotation.T + centre - X
        U = np.zeros(constraint.nDof)
        sf = constraint.sizeField
        for i in range(len(X)):
            U[sf * i : sf * i + 3] = motion[i]
        return U

    def _stage(self, constraint, increments):
        """Run accepted increments through ``(U, rebuild)`` pairs, as the implicit solver would."""

        previous = np.zeros(constraint.nDof)
        for number, U in enumerate(increments, start=1):
            constraint.updateConnectivity(constraint.model)
            self._assemble(constraint, U, U - previous, _frozenTimeStep(number))
            constraint.acceptLastState()
            previous = U

    def test_a_rigid_rotation_does_not_accumulate_a_frictional_traction(self):
        """A penetrating, sticking pair rotated rigidly over many increments keeps no history.

        On the frozen geometry alone every increment reads the turn of the penalty gap as a slip of
        about theta * penetration, and in stick that accumulates. The objective correction at each
        rebuild removes it again, so at the next rebuild nothing is left.
        """

        theta, steps = 0.01, 8
        constraint = self._constraint(mu=10.0)
        increments = [self._rigidRotation(constraint, k * theta) for k in range(steps)]
        self._stage(constraint, increments + [increments[-1]])

        active = constraint._friction.activeConverged
        self.assertGreater(int(np.sum(active)), 0, "no node is in contact -- test is vacuous")
        pressure = constraint._friction.pressureConverged[active]
        # What the frozen measure alone leaves after the same motion, for scale: eps_T times the
        # accumulated turn of the 0.01 penetration.
        accumulated = self.penaltyStiffness * (steps - 1) * theta * 0.01
        self.assertGreater(accumulated, 1e-3 * float(np.max(pressure)))
        traction = np.linalg.norm(constraint._friction.tractionConverged[active], axis=1)
        self.assertLess(float(np.max(traction)), 1e-3 * accumulated)

    def test_a_pure_slide_is_left_as_measured(self):
        """A flat slide does not change the gap vector, so the rebuild leaves the stick traction alone."""

        slide = 1e-4
        constraint = self._constraint(mu=10.0, geometryUpdate="frozen")
        U = self._masterSlide(constraint, slide)
        self._stage(constraint, [np.zeros_like(U), U])
        before = constraint._friction.tractionConverged.copy()
        constraint.updateConnectivity(constraint.model)
        self._assemble(constraint, U, np.zeros_like(U), _frozenTimeStep(3))

        active = constraint._friction.activeConverged
        np.testing.assert_allclose(before[active, 0], self.penaltyStiffness * slide, rtol=1e-10)
        np.testing.assert_allclose(
            constraint._friction.tractionConverged[active], before[active], rtol=0.0, atol=1e-10 * np.max(before)
        )

    def test_a_repeated_rebuild_at_the_same_state_changes_nothing(self):
        """A cutback re-attempt may rebuild the geometry again at the accepted state."""

        constraint = self._constraint(mu=10.0)
        increments = [self._rigidRotation(constraint, k * 0.01) for k in range(3)]
        self._stage(constraint, increments)
        constraint.updateConnectivity(constraint.model)
        self._assemble(constraint, increments[-1], np.zeros(constraint.nDof), _frozenTimeStep(10))
        once = constraint._friction.tractionConverged.copy()
        constraint.updateConnectivity(constraint.model)
        self._assemble(constraint, increments[-1], np.zeros(constraint.nDof), _frozenTimeStep(11))
        np.testing.assert_allclose(constraint._friction.tractionConverged, once, rtol=0.0, atol=1e-14)

    # ------------------------------------------------------------------ anti-chattering

    def test_a_recurring_stick_slip_state_freezes_after_the_free_iterations(self):
        constraint = self._constraint(mu=0.3)
        stick, slip = np.ones(constraint.nNonMortarNodes, dtype=int), np.full(constraint.nNonMortarNodes, 2)
        constraint._friction.tractionCurrent[:] = [0.0, 0.0, 1.0]  # a direction to freeze
        for iteration, state in enumerate((stick, slip, stick, slip, stick)):
            constraint.currentIteration = iteration
            constraint._friction.slipState[:] = state
            constraint._friction.freezeIfChattering(constraint.currentIteration)
            # A-B-A within the free iterations (0..2) is left alone; the recurrence at 3 freezes.
            self.assertEqual(constraint._friction.slipStateFrozen, iteration >= 3, f"iteration {iteration}")
        np.testing.assert_array_equal(constraint._friction.frozenSlip, True)
        np.testing.assert_allclose(constraint._friction.frozenDirection, [[0.0, 0.0, 1.0]] * constraint.nNonMortarNodes)

    def test_a_late_change_freezes_even_without_a_cycle(self):
        constraint = self._constraint(mu=0.3)
        n = constraint.nNonMortarNodes
        for iteration in range(8):
            constraint.currentIteration = iteration
            # A different state in every iteration -- the bits of the iteration number -- so that
            # nothing ever recurs and only the late change can be what freezes it.
            state = 1 + (iteration >> np.arange(n)) % 2
            constraint._friction.slipState[:] = state
            constraint._friction.freezeIfChattering(constraint.currentIteration)
            if constraint._friction.slipStateFrozen:
                break
        self.assertTrue(constraint._friction.slipStateFrozen)
        self.assertGreaterEqual(constraint.currentIteration, 6)

    def test_a_frozen_slip_node_stays_on_the_cone_and_is_reported(self):
        """Frozen: slip is kept even where the predictor falls back inside the cone, never the reverse."""

        mu = 0.2
        journal = _RecordingJournal()
        model = _TwoBlockModel.build(gap=-0.01, masterShift=0.2)
        constraint = _TwoBlockModel.constraint(
            model, journal, formulation="penalty", penaltyStiffness=self.penaltyStiffness, mu=mu
        )
        timeStep = _frozenTimeStep()
        U = self._masterSlide(constraint, 0.05)
        self._assemble(constraint, U, U, timeStep)
        active = constraint.activeSet.copy()
        np.testing.assert_array_equal(constraint._friction.slipState[active], 2)

        constraint._friction.slipStateFrozen = True
        constraint._friction.freezeDirections(constraint._friction.slipState == 2)
        frozenDirection = constraint._friction.tractionCurrent[active] / np.linalg.norm(
            constraint._friction.tractionCurrent[active], axis=1, keepdims=True
        )
        # Back the other way, by less than the elastic range: free, this would stick (and reverse);
        # frozen, it keeps slipping the way it went.
        back = -self._masterSlide(constraint, 1e-6)
        self._assemble(constraint, back, back, timeStep)
        np.testing.assert_array_equal(constraint._friction.slipState[active], 2)
        np.testing.assert_allclose(
            constraint._friction.tractionCurrent[active]
            / np.linalg.norm(constraint._friction.tractionCurrent[active], axis=1, keepdims=True),
            frozenDirection,
            atol=1e-12,
        )
        pressure = np.abs(constraint.nodalMultipliers[active])
        np.testing.assert_allclose(
            np.linalg.norm(constraint._friction.tractionCurrent[active], axis=1), mu * pressure, rtol=1e-12
        )

        constraint.acceptLastState()
        self.assertTrue(any("frozen" in message for _, message in journal.records))

    # ------------------------------------------------------------------ output

    def test_the_nodal_results_line_up_with_the_published_node_set(self):
        constraint = self._constraint(mu=0.2)
        U = self._masterSlide(constraint, 0.05)
        self._assemble(constraint, U, U, _frozenTimeStep())

        nodeSet = constraint.model.nodeSets["contact_nonMortarNodes"]
        self.assertEqual(list(nodeSet), constraint.nonMortarNodes)
        self.assertEqual(constraint.getNormalPressures().shape, (len(nodeSet),))
        self.assertEqual(constraint.getTangentialTractions().shape, (len(nodeSet), 3))

        active = constraint.activeSet
        np.testing.assert_allclose(constraint.getNormalPressures()[active], np.abs(constraint.nodalMultipliers[active]))
        self.assertTrue(np.all(constraint.getNormalPressures() >= 0.0))
        np.testing.assert_allclose(constraint.getGaps()[active], -0.01, rtol=1e-10)
        np.testing.assert_array_equal(constraint.getSlipStates()[active], 2.0)


if __name__ == "__main__":
    unittest.main()
