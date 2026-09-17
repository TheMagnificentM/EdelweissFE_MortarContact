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
"""Unit tests for the contact facet generator's per-node weighting options.

The analytic reference values are those of a *flat, straight-edged* quadratic face, for which the
midside triangulation's four corner triangles each have a quarter of the face area and the central
midside quad has half of it.
"""

import unittest

import numpy as np

import edelweissfe.utils.inputfileparser  # noqa: F401 bootstrap input language
from edelweissfe.constraints.nodetodeformablesurfacepenalty import (
    Constraint as ContactConstraint,
)
from edelweissfe.constraints.nodetodeformablesurfacepenalty import (
    NodeToDeformableSurfacePenaltySchema,
)
from edelweissfe.elements.contactsurfaceelement import (
    Tria3ContactFacet,
    facetNormalAndMeasure,
)
from edelweissfe.elements.displacementelement.element import DisplacementElement
from edelweissfe.generators.surfaceelementgenerator import (
    buildContactFacets,
    canonicalParentFace,
)
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.points.node import Node
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.utils.parentfacegeometry import (
    facetQuadratureRule,
    parentFaceShapeFunctions,
)

#: Edge-to-midside-node map of the hexa20 node ordering, local indices 8..19 in order.
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

#: Side length of the test cube; a face therefore has area 4, matching the reference values below.
_SIDE = 2.0

#: The Ymin face (face number 1) of the hexa20 node ordering: corner then midside local indices.
_YMIN_CORNERS = (0, 1, 2, 3)
_YMIN_MIDSIDES = (8, 9, 10, 11)


class TestContactFacetNodalWeights(unittest.TestCase):
    def setUp(self):
        self.journal = Journal()

    def _modelWithOneHexa20(self) -> FEModel:
        """A single straight-edged hexa20 cube of side ``_SIDE``, with its Ymin face registered as
        the surface ``theSurface``."""

        # boxGen's own corner ring order, (0,0), (0,S), (S,S), (S,0) in the (x, z) plane: the
        # generator's face tables were verified against it, and transposing two corners would
        # reverse every face normal while leaving areas and shape-function integrals untouched.
        corners = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, _SIDE]),
            np.array([_SIDE, 0.0, _SIDE]),
            np.array([_SIDE, 0.0, 0.0]),
            np.array([0.0, _SIDE, 0.0]),
            np.array([0.0, _SIDE, _SIDE]),
            np.array([_SIDE, _SIDE, _SIDE]),
            np.array([_SIDE, _SIDE, 0.0]),
        ]
        coordinates = corners + [0.5 * (corners[a] + corners[b]) for a, b in _HEXA20_EDGES]

        model = FEModel(3)
        nodes = [Node(i + 1, x) for i, x in enumerate(coordinates)]
        for node in nodes:
            model.nodes[node.label] = node

        with model.topologyChanges():
            (elNumber,) = model.reserveElementNumbers(1)
            element = DisplacementElement("C3D20", elNumber)
            element.setNodes(nodes)
            model.createElement(element)
        model.surfaces["theSurface"] = {1: ElementSet("theFace", [element])}

        return model

    def _sharesPerNode(self, model: FEModel, facetsSetName: str) -> dict:
        """The tributary area of every facet node, assembled exactly as the contact constraint
        does."""

        shares = {}
        for facet in model.elementSets[facetsSetName]:
            for node, share in zip(facet.nodes, facet.nodalAreaShares):
                shares[node] = shares.get(node, 0.0) + share
        return shares

    def _facetSharesByRole(self, model: FEModel, nodalWeights: str) -> tuple:
        """Generate the Ymin face's facets under ``nodalWeights`` and return its (corner shares,
        midside shares) as arrays, plus the facet element set name."""

        with model.topologyChanges():
            facetsSetName, _ = buildContactFacets(model, "theSurface", "pfx", "midside", nodalWeights, self.journal)
        shares = self._sharesPerNode(model, facetsSetName)
        (sourceElement,) = model.surfaces["theSurface"][1]
        sourceNodes = sourceElement.nodes
        cornerShares = np.array([shares[sourceNodes[i]] for i in _YMIN_CORNERS])
        midsideShares = np.array([shares[sourceNodes[i]] for i in _YMIN_MIDSIDES])
        return cornerShares, midsideShares, facetsSetName

    def test_facetConsistent_shares_are_the_facet_tiling_lumping(self):
        """The default weighting: A/24 per corner, 5A/24 per midside node, for A = 4."""

        model = self._modelWithOneHexa20()
        cornerShares, midsideShares, _ = self._facetSharesByRole(model, "facetConsistent")

        np.testing.assert_allclose(cornerShares, 4.0 / 24.0)
        np.testing.assert_allclose(midsideShares, 5.0 * 4.0 / 24.0)
        self.assertAlmostEqual(cornerShares.sum() + midsideShares.sum(), 4.0)

    def test_serendipityOptimal_zeroes_the_corners_and_stays_resultant_exact(self):
        """The modified weighting: zero per corner, A/4 per midside node, same total area."""

        model = self._modelWithOneHexa20()
        cornerShares, midsideShares, facetsSetName = self._facetSharesByRole(model, "serendipityOptimal")

        np.testing.assert_allclose(cornerShares, 0.0)
        np.testing.assert_allclose(midsideShares, 4.0 / 4.0)
        self.assertAlmostEqual(cornerShares.sum() + midsideShares.sum(), 4.0)

        for facet in model.elementSets[facetsSetName]:
            self.assertTrue(np.all(facet.nodalAreaShares >= 0.0))

    def test_serendipityOptimal_halves_the_mismatch_against_the_consistent_loads(self):
        """The mismatch against the quad8 face's own consistent nodal loads (-A/12 per corner,
        +A/3 per midside) drops from A/8 to A/12 per node -- by a third, and that is the optimum
        over all resultant-exact, non-negative weightings."""

        area = 4.0
        consistentCorner, consistentMidside = -area / 12.0, area / 3.0

        default = self._facetSharesByRole(self._modelWithOneHexa20(), "facetConsistent")
        modified = self._facetSharesByRole(self._modelWithOneHexa20(), "serendipityOptimal")

        for cornerShares, midsideShares, expectedMismatch in (
            (*default[:2], area / 8.0),
            (*modified[:2], area / 12.0),
        ):
            np.testing.assert_allclose(cornerShares - consistentCorner, expectedMismatch)
            np.testing.assert_allclose(midsideShares - consistentMidside, -expectedMismatch)

    def test_serendipityOptimal_installs_a_force_preserving_transform_on_corner_triangles(self):
        """Only the four corner triangles carry a transform, and it is column-stochastic (so the
        transferred force is redistributed, never scaled)."""

        model = self._modelWithOneHexa20()
        with model.topologyChanges():
            facetsSetName, _ = buildContactFacets(
                model, "theSurface", "pfx", "midside", "serendipityOptimal", self.journal
            )
        facets = list(model.elementSets[facetsSetName])
        self.assertEqual(len(facets), 6)

        transformed = [facet for facet in facets if facet.weightTransform is not None]
        self.assertEqual(len(transformed), 4)
        for facet in transformed:
            np.testing.assert_allclose(facet.weightTransform.sum(axis=0), 1.0)
            # the corner (local index 1 of a (m_prev, c, m_next) triangle) receives nothing
            np.testing.assert_allclose(facet.weightTransform[1, :], 0.0)

    def test_facetConsistent_installs_no_transform(self):
        model = self._modelWithOneHexa20()
        with model.topologyChanges():
            facetsSetName, _ = buildContactFacets(
                model, "theSurface", "pfx", "midside", "facetConsistent", self.journal
            )
        for facet in model.elementSets[facetsSetName]:
            self.assertIsNone(facet.weightTransform)

    def test_serendipityOptimal_rejects_the_corner_triangulation(self):
        """With the corner reduction there are no midside nodes to reassign the corner shares to,
        which would silently zero every weight of a quadratic face."""

        model = self._modelWithOneHexa20()
        with self.assertRaises(ValueError) as ctx, model.topologyChanges():
            buildContactFacets(model, "theSurface", "pfx", "corner", "serendipityOptimal", self.journal)
        self.assertIn("requires triangulation='midside'", str(ctx.exception))

    def _modelWithOneQuad8(self) -> FEModel:
        """A single straight-edged quad8 of side ``_SIDE`` in 2D, with one of its edges registered
        as the surface ``theSurface``."""

        corners = [
            np.array([0.0, 0.0]),
            np.array([_SIDE, 0.0]),
            np.array([_SIDE, _SIDE]),
            np.array([0.0, _SIDE]),
        ]
        coordinates = corners + [0.5 * (corners[a] + corners[(a + 1) % 4]) for a in range(4)]

        model = FEModel(2)
        nodes = [Node(i + 1, x) for i, x in enumerate(coordinates)]
        for node in nodes:
            model.nodes[node.label] = node

        with model.topologyChanges():
            (elNumber,) = model.reserveElementNumbers(1)
            element = DisplacementElement("CPE8", elNumber)
            element.setNodes(nodes)
            model.createElement(element)
        model.surfaces["theSurface"] = {1: ElementSet("theEdge", [element])}

        return model

    def test_serendipityOptimal_rejects_2d_quadratic_edges(self):
        """A quadratic *edge* tiles into two Line2 facets, which reach neither branch that applies
        the corner reassignment -- so without a refusal the option would validate and then do
        nothing at all.

        The refusal is not a placeholder for an unimplemented case: a quadratic edge's consistent
        weights are Simpson's (L/6, 2L/3, L/6), all non-negative, so there is no tensile corner
        load to work around in 2D and the corner-share reassignment is simply the wrong instrument.
        """

        model = self._modelWithOneQuad8()
        with self.assertRaises(ValueError) as ctx, model.topologyChanges():
            buildContactFacets(model, "theSurface", "pfx", "midside", "serendipityOptimal", self.journal)
        self.assertIn("not available for 2D higher-order element edges", str(ctx.exception))

    def test_unknown_nodalWeights_is_rejected(self):
        model = self._modelWithOneHexa20()
        with self.assertRaises(ValueError) as ctx, model.topologyChanges():
            buildContactFacets(model, "theSurface", "pfx", "midside", "nonsense", self.journal)
        self.assertIn("nodalWeights 'nonsense' is not supported", str(ctx.exception))

    def test_transform_reassigns_the_corner_weight_of_the_contact_point(self):
        """End-to-end on the constraint: a slave whose closest point falls inside a corner triangle
        gets the corner's interpolation weight split equally onto the two adjacent midside nodes,
        so no force reaches the corner node."""

        model = self._modelWithOneHexa20()
        with model.topologyChanges():
            facetsSetName, _ = buildContactFacets(
                model, "theSurface", "pfx", "midside", "serendipityOptimal", self.journal
            )

        # The Ymin corner triangle (9, 1, 8) of the side-2 cube spans (2, 0, 1), (2, 0, 0) and
        # (1, 0, 0); its centroid is (5/3, 0, 1/3). Three slave nodes just outside the face (the
        # outward normal is -y), the first exactly above that centroid.
        slaveCoordinates = (
            np.array([5.0 / 3.0, -0.01, 1.0 / 3.0]),
            np.array([1.70, -0.01, 0.35]),
            np.array([1.75, -0.01, 0.30]),
        )
        slaveNodes = [Node(1000 + i, x) for i, x in enumerate(slaveCoordinates)]
        slaveFacet = Tria3ContactFacet("Tria3ContactFacet", 1000)
        slaveFacet.setNodes(slaveNodes)
        slaveFacet.initializeElement()

        constraint = ContactConstraint(
            "theContact",
            model,
            ElementSet("slave_facets", [slaveFacet]),
            model.elementSets[facetsSetName],
            self.journal,
            configuration=NodeToDeformableSurfacePenaltySchema(penalty=1.0, sliding="small", searchDistance=1.0),
        )
        constraint.updateConnectivity(model)

        for s, weights in enumerate(constraint._frozenWeights):
            self.assertIsNotNone(weights, f"slave {s} was not assigned a facet")
            self.assertAlmostEqual(weights.sum(), 1.0, msg="the transform must preserve the force")
            self.assertAlmostEqual(weights[1], 0.0, msg="the corner node must receive no weight")

        # the centroid slave: barycentric (1/3, 1/3, 1/3) -> (1/2, 0, 1/2)
        np.testing.assert_allclose(constraint._frozenWeights[0], [0.5, 0.0, 0.5], atol=1e-12)

    def test_transformed_master_facets_require_small_sliding(self):
        """A weight transform is variationally admissible only in the frozen-projection
        formulation; finite sliding must refuse it rather than half-support it."""

        model = self._modelWithOneHexa20()
        with model.topologyChanges():
            facetsSetName, _ = buildContactFacets(
                model, "theSurface", "pfx", "midside", "serendipityOptimal", self.journal
            )
        masterSurface = model.elementSets[facetsSetName]
        slaveSurface = ElementSet("slave_facets", [])

        for sliding, expectation in (("small", None), ("finite", "requires sliding=small")):
            configuration = NodeToDeformableSurfacePenaltySchema(penalty=1.0, sliding=sliding)
            if expectation is None:
                ContactConstraint(
                    "theContact", model, slaveSurface, masterSurface, self.journal, configuration=configuration
                )
            else:
                with self.assertRaises(ValueError) as ctx:
                    ContactConstraint(
                        "theContact", model, slaveSurface, masterSurface, self.journal, configuration=configuration
                    )
                self.assertIn(expectation, str(ctx.exception))


class TestParentFaceIntegration(unittest.TestCase):
    """The premise of integrated (Gauss-point-to-segment) contact: quadrature over the facets
    tiling a face, distributed with the *parent face's* shape functions, reproduces that face's own
    consistent nodal loads -- including the NEGATIVE corner loads of a serendipity face that no
    per-node scheme can deliver."""

    def setUp(self):
        self.journal = Journal()

    def _consistentLoadsOfUnitPressure(self, model: FEModel, facetsSetName: str, nPoints: int) -> dict:
        """Integrate a unit pressure over the facet tiling, distributing it with the parent face's
        shape functions, and accumulate the resulting nodal loads."""

        loads = {}
        for facet in model.elementSets[facetsSetName]:
            self.assertIsNotNone(facet.parentFaceType, "the generator must stamp the parent face")
            coordinates = np.array([n.coordinates for n in facet.nodes])
            _, measure = facetNormalAndMeasure(coordinates)
            barycentric, weights = facetQuadratureRule(len(facet.nodes), nPoints)
            for b, w in zip(barycentric, weights):
                N = parentFaceShapeFunctions(facet.parentFaceType, b @ facet.vertexParametricCoords)
                for node, shapeFunction in zip(facet.parentFaceNodes, N):
                    loads[node] = loads.get(node, 0.0) + measure * w * shapeFunction
        return loads

    def test_hexa20_face_recovers_the_negative_corner_loads(self):
        """The go/no-go for the whole formulation: -A/12 per corner and +A/3 per midside, for the
        side-2 cube's face of area A = 4. The integrand is quadratic in the facet's barycentric
        coordinates, so the 3-point rule is exact and this must hold to machine precision."""

        for nPoints in (3, 6):
            model = TestContactFacetNodalWeights._modelWithOneHexa20(self)
            with model.topologyChanges():
                facetsSetName, _ = buildContactFacets(
                    model, "theSurface", "pfx", "midside", "facetConsistent", self.journal
                )
            loads = self._consistentLoadsOfUnitPressure(model, facetsSetName, nPoints)
            sourceNodes = model.surfaces["theSurface"][1][0].nodes

            corners = np.array([loads[sourceNodes[i]] for i in _YMIN_CORNERS])
            midsides = np.array([loads[sourceNodes[i]] for i in _YMIN_MIDSIDES])
            np.testing.assert_allclose(corners, -4.0 / 12.0, atol=1e-13, err_msg=f"nPoints={nPoints}")
            np.testing.assert_allclose(midsides, 4.0 / 3.0, atol=1e-13, err_msg=f"nPoints={nPoints}")
            self.assertAlmostEqual(corners.sum() + midsides.sum(), 4.0, places=12)
            self.assertTrue(np.all(corners < 0.0), "the corner loads must be tensile")

    def test_corner_triangulation_recovers_them_too(self):
        """The parent face is a property of the element, not of the tiling: even the corner
        reduction, whose facets contain no midside nodes at all, distributes onto the full quad8
        basis and reproduces the same loads."""

        model = TestContactFacetNodalWeights._modelWithOneHexa20(self)
        with model.topologyChanges():
            facetsSetName, _ = buildContactFacets(model, "theSurface", "pfx", "corner", "facetConsistent", self.journal)
        loads = self._consistentLoadsOfUnitPressure(model, facetsSetName, 3)
        sourceNodes = model.surfaces["theSurface"][1][0].nodes
        np.testing.assert_allclose([loads[sourceNodes[i]] for i in _YMIN_CORNERS], -4.0 / 12.0, atol=1e-13)
        np.testing.assert_allclose([loads[sourceNodes[i]] for i in _YMIN_MIDSIDES], 4.0 / 3.0, atol=1e-13)

    def test_generated_facet_normals_point_outward(self):
        """Every facet of the cube's Ymin face must have an outward (-y) normal. Nothing else in
        this file would notice a reversed winding -- areas and shape-function integrals are
        orientation-independent -- but a contact constraint would silently see penetration where
        there is separation."""

        model = TestContactFacetNodalWeights._modelWithOneHexa20(self)
        with model.topologyChanges():
            facetsSetName, _ = buildContactFacets(
                model, "theSurface", "pfx", "midside", "facetConsistent", self.journal
            )
        for facet in model.elementSets[facetsSetName]:
            normal, _ = facetNormalAndMeasure(np.array([n.coordinates for n in facet.nodes]))
            np.testing.assert_allclose(normal, [0.0, -1.0, 0.0], atol=1e-14)

    def test_canonical_parent_face_derivation(self):
        """The derivation from the face tables, on one face of each supported source type."""

        faceType, indices = canonicalParentFace("hexa20", 1)
        self.assertEqual(faceType, "quad8")
        # Ymin corner triangles are (11,3,10), (10,2,9), (9,1,8), (8,0,11): cycle 3,2,1,0 with the
        # midside between consecutive corners taken from each triangle's last entry
        self.assertEqual(indices, (3, 2, 1, 0, 10, 9, 8, 11))

        self.assertEqual(canonicalParentFace("hexa8", 1), ("quad4", (3, 2, 1, 0)))
        self.assertEqual(canonicalParentFace("quad8", 1), ("line3", (0, 1, 4)))
        self.assertEqual(canonicalParentFace("quad4", 1), ("line2", (0, 1)))

    def test_line3_parent_edge_recovers_simpson(self):
        """In 2D the consistent weights of a quadratic edge are Simpson's -- all positive, so the
        2D case has no obstruction to begin with; this pins the line3 branch."""

        coords = {0: np.array([0.0, 0.0]), 1: np.array([2.0, 0.0]), 4: np.array([1.0, 0.0])}
        faceType, indices = canonicalParentFace("quad8", 1)
        parametric = {0: np.array([-1.0]), 1: np.array([1.0]), 4: np.array([0.0])}
        loads = {i: 0.0 for i in indices}
        for facetNodes in ((0, 4), (4, 1)):
            measure = np.linalg.norm(coords[facetNodes[1]] - coords[facetNodes[0]])
            vertexParametric = np.array([parametric[i] for i in facetNodes])
            barycentric, weights = facetQuadratureRule(2, 3)
            for b, w in zip(barycentric, weights):
                N = parentFaceShapeFunctions(faceType, b @ vertexParametric)
                for k, i in enumerate(indices):
                    loads[i] += measure * w * N[k]
        np.testing.assert_allclose([loads[0], loads[1]], 2.0 / 6.0, atol=1e-13)
        np.testing.assert_allclose(loads[4], 2.0 * 2.0 / 3.0, atol=1e-13)


class TestWholeFaceFacets(unittest.TestCase):
    """The ``facets='wholeFace'`` mode, which emits one curved contact element per source face
    instead of tiling the face into flat ones.

    What has to hold is narrow but easy to get wrong in a way no simulation would notice quickly:
    the emitted element must be the type that matches the face, its nodes must arrive in the
    ordering that element's own shape functions assume, and the resulting normal must point out of
    the source solid. A permuted ordering still yields a plausible-looking element of the right
    type, and an inward normal still integrates to the right area -- both would simply transfer the
    contact pressure in the wrong direction.
    """

    def setUp(self):
        self.journal = Journal()

    def _build(self, model, prefix="mortar"):
        with model.topologyChanges():
            facetsSetName, nodesSetName = buildContactFacets(
                model, "theSurface", prefix, "corner", "facetConsistent", self.journal, facets="wholeFace"
            )
        return facetsSetName, nodesSetName

    def _modelWithOneHexa8(self) -> FEModel:
        """A single straight-edged hexa8 cube of side ``_SIDE``, Ymin face registered as
        ``theSurface`` -- the same corner ring order as the hexa20 fixture."""

        coordinates = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, _SIDE]),
            np.array([_SIDE, 0.0, _SIDE]),
            np.array([_SIDE, 0.0, 0.0]),
            np.array([0.0, _SIDE, 0.0]),
            np.array([0.0, _SIDE, _SIDE]),
            np.array([_SIDE, _SIDE, _SIDE]),
            np.array([_SIDE, _SIDE, 0.0]),
        ]

        model = FEModel(3)
        nodes = [Node(i + 1, x) for i, x in enumerate(coordinates)]
        for node in nodes:
            model.nodes[node.label] = node

        with model.topologyChanges():
            (elNumber,) = model.reserveElementNumbers(1)
            element = DisplacementElement("C3D8", elNumber)
            element.setNodes(nodes)
            model.createElement(element)
        model.surfaces["theSurface"] = {1: ElementSet("theFace", [element])}
        return model

    def test_the_emitted_type_follows_the_parent_face(self):
        """A linear hexa face is a quad4 and becomes a CONQUAD4, a quadratic one a quad8 and becomes
        a CONQUAD8, and the two-dimensional edges become CONLINE2/CONLINE3 the same way. The type is
        taken from the FACE rather than from the solid, because the face is what the contact element
        has to reproduce."""

        for model, expected in (
            (self._modelWithOneHexa8(), "CONQUAD4"),
            (TestContactFacetNodalWeights._modelWithOneHexa20(self), "CONQUAD8"),
            (TestContactFacetNodalWeights._modelWithOneQuad8(self), "CONLINE3"),
        ):
            facetsSetName, _ = self._build(model)
            (contactElement,) = model.elementSets[facetsSetName]
            self.assertEqual(contactElement.elType, expected)

    def test_one_element_per_face_rather_than_a_tiling(self):
        """The whole point of the mode: a quadratic face that the triangulated path would split into
        six Tria3 facets stays a single element with all eight of its nodes."""

        model = TestContactFacetNodalWeights._modelWithOneHexa20(self)
        facetsSetName, nodesSetName = self._build(model)

        self.assertEqual(len(model.elementSets[facetsSetName]), 1)
        (contactElement,) = model.elementSets[facetsSetName]
        self.assertEqual(len(contactElement.nodes), 8)
        self.assertEqual(len(model.nodeSets[nodesSetName]), 8)

    def test_the_node_ordering_is_the_canonical_parent_face_ordering(self):
        """Not merely the right SET of nodes: the contact element's shape functions are written for
        the canonical ordering -- corner cycle first, then the midside of each corner-to-corner edge
        in the same cycle order -- so any permutation of it silently evaluates the wrong function at
        every quadrature point."""

        model = TestContactFacetNodalWeights._modelWithOneHexa20(self)
        (sourceElement,) = model.surfaces["theSurface"][1]
        _faceType, canonicalIndices = canonicalParentFace("hexa20", 1)

        facetsSetName, _ = self._build(model)
        (contactElement,) = model.elementSets[facetsSetName]

        expected = [sourceElement.nodes[i] for i in canonicalIndices]
        self.assertEqual(list(contactElement.nodes), expected)

    def test_generated_normals_point_out_of_the_source_solid(self):
        """The ordering the face tables produce already winds the face outward, so no orientation
        correction is applied anywhere in this path. That is only safe as long as it is true: an
        inward-wound face would reverse the contact pressure while leaving its area intact, which no
        area-based or node-count-based check could catch."""

        for model, outward in (
            (self._modelWithOneHexa8(), np.array([0.0, -1.0, 0.0])),
            (TestContactFacetNodalWeights._modelWithOneHexa20(self), np.array([0.0, -1.0, 0.0])),
        ):
            facetsSetName, _ = self._build(model)
            (contactElement,) = model.elementSets[facetsSetName]

            corners = np.array([node.coordinates for node in contactElement.nodes[:4]])
            normal = np.cross(corners[2] - corners[0], corners[3] - corners[1])
            self.assertGreater(float(np.dot(normal, outward)), 0.0)

    def test_rebuilding_replaces_rather_than_accumulates(self):
        """Re-running under the same prefix has to be idempotent: the surface facet modifier calls
        this again after an adaptive refinement, and a second set of contact elements on the same
        face would double every coupling term rather than fail."""

        model = TestContactFacetNodalWeights._modelWithOneHexa20(self)
        facetsSetName, _ = self._build(model)
        firstNumbers = {element.elNumber for element in model.elementSets[facetsSetName]}

        self._build(model)
        secondNumbers = {element.elNumber for element in model.elementSets[facetsSetName]}

        self.assertEqual(len(secondNumbers), 1)
        self.assertTrue(firstNumbers.isdisjoint(secondNumbers), "stale element numbers were handed back out")
        self.assertEqual(len(model.elements), 2, "the source solid and exactly one contact element")

    def test_the_recipe_records_the_mode(self):
        """The recipe is what a rebuild after a mesh change replays. Losing the mode from it would
        retile the surface into flat facets the constraint cannot use."""

        model = TestContactFacetNodalWeights._modelWithOneHexa20(self)
        facetsSetName, _ = self._build(model)
        self.assertEqual(model.contactFacetRecipes[facetsSetName][-1], "wholeFace")

    def test_tiling_options_are_rejected_rather_than_ignored(self):
        """Both options describe how a face is cut up and how the pieces share its area. Neither
        question arises for a face kept whole, and accepting them would leave the user believing a
        setting took effect that cannot."""

        for option in ("triangulation", "nodalWeights"):
            model = TestContactFacetNodalWeights._modelWithOneHexa20(self)
            kwargs = {"triangulation": "corner", "nodalWeights": "facetConsistent"}
            kwargs[option] = "midside" if option == "triangulation" else "serendipityOptimal"
            with self.assertRaises(ValueError) as ctx, model.topologyChanges():
                buildContactFacets(
                    model,
                    "theSurface",
                    "pfx",
                    kwargs["triangulation"],
                    kwargs["nodalWeights"],
                    self.journal,
                    facets="wholeFace",
                )
            self.assertIn(option, str(ctx.exception))
            self.assertIn("wholeFace", str(ctx.exception))

    def test_unknown_facets_mode_is_rejected(self):
        model = TestContactFacetNodalWeights._modelWithOneHexa20(self)
        with self.assertRaises(ValueError) as ctx, model.topologyChanges():
            buildContactFacets(model, "theSurface", "pfx", "corner", "facetConsistent", self.journal, facets="nonsense")
        self.assertIn("nonsense", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
