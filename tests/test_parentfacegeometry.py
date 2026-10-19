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
"""Unit tests for the parent-face geometry of the integrated contact: shape function derivatives,
the native Gauss rule, the closest-point projection onto a curved face, and the accuracy of the
facet rule that issue #216 questioned."""

import unittest

import numpy as np

from edelweissfe.elements.contactelement.element import ContactElement
from edelweissfe.utils.mortarlinearization import shapeFunctionSecondDerivatives
from edelweissfe.utils.parentfacegeometry import (
    PARENT_FACE_LOCAL_DIMENSION,
    PARENT_FACE_PARAMETRIC_COORDS,
    facetQuadratureRule,
    parentFaceAreaElement,
    parentFaceClosestPoints,
    parentFaceGaussRule,
    parentFaceShapeFunctionDerivatives,
    parentFaceShapeFunctions,
    parentFaceShapeFunctionsAt,
    parentFaceShapeFunctionSecondDerivatives,
    parentFaceTangents,
    parentFaceUnitNormals,
)

#: The contact element type whose node order is the canonical parent-face order.
_CONTACT_ELEMENT = {"quad8": "CONQUAD8", "quad4": "CONQUAD4", "line3": "CONLINE3", "line2": "CONLINE2"}

#: The midside triangulation of a quad8 face, in canonical node indices: 4 corner triangles
#: (m_prev, c, m_next) and the central quad of the midsides split in two, as the generator emits it.
_MIDSIDE_TRIANGLES = ((7, 0, 4), (4, 1, 5), (5, 2, 6), (6, 3, 7), (4, 5, 6), (4, 6, 7))


def _referenceIntegrals(faceType: str, nodeCoords: np.ndarray, nPerDirection: int = 40) -> np.ndarray:
    """Per node, the integral of its shape function over the face, by a high-order tensor rule that
    shares nothing with the code under test but the shape functions."""

    positions, weights = np.polynomial.legendre.leggauss(nPerDirection)
    integrals = np.zeros(len(nodeCoords))
    h = 1e-6
    for x, wx in zip(positions, weights):
        for y, wy in zip(positions, weights):
            xi = np.array([x, y])
            tangents = [
                (
                    parentFaceShapeFunctions(faceType, xi + h * e) @ nodeCoords
                    - parentFaceShapeFunctions(faceType, xi - h * e) @ nodeCoords
                )
                / (2 * h)
                for e in np.eye(2)
            ]
            integrals += wx * wy * np.linalg.norm(np.cross(*tangents)) * parentFaceShapeFunctions(faceType, xi)
    return integrals


def _flatQuad8(distorted: bool) -> np.ndarray:
    """A flat quad8 face in the plane y = 0: the square [0, 2]^2, or a trapezoid with off-centre
    midside nodes."""

    xi = PARENT_FACE_PARAMETRIC_COORDS["quad8"]
    coords = np.column_stack([1.0 + xi[:, 0], np.zeros(8), 1.0 + xi[:, 1]])
    if distorted:
        coords[2, 0] += 0.4  # a trapezoid
        coords[4, 0] += 0.25  # and midside nodes off their edge centres, along the straight edges
        coords[7, 2] -= 0.2
        coords[5] = 0.6 * coords[1] + 0.4 * coords[2]
    return coords


class TestParentFaceShapeFunctions(unittest.TestCase):
    def test_bulk_values_match_the_pointwise_ones(self):
        rng = np.random.default_rng(1)
        for faceType, d in PARENT_FACE_LOCAL_DIMENSION.items():
            xi = rng.uniform(-1.0, 1.0, (7, d))
            np.testing.assert_array_equal(
                parentFaceShapeFunctionsAt(faceType, xi),
                np.array([parentFaceShapeFunctions(faceType, x) for x in xi]),
            )

    def test_derivatives_match_the_contact_elements_and_finite_differences(self):
        rng = np.random.default_rng(2)
        for faceType, d in PARENT_FACE_LOCAL_DIMENSION.items():
            element = ContactElement(_CONTACT_ELEMENT[faceType], 1)
            xi = rng.uniform(-1.0, 1.0, (5, d))
            first = parentFaceShapeFunctionDerivatives(faceType, xi)
            second = parentFaceShapeFunctionSecondDerivatives(faceType, xi)
            for k, x in enumerate(xi):
                np.testing.assert_allclose(first[k], element.getShapeFunctionDerivatives(x), atol=1e-14)
                np.testing.assert_allclose(second[k], shapeFunctionSecondDerivatives(element, x), atol=1e-14)
                np.testing.assert_allclose(
                    element.getShapeFunctions(x), parentFaceShapeFunctions(faceType, x), atol=1e-15
                )
            h = 1e-6
            for alpha in range(d):
                e = h * np.eye(d)[alpha]
                np.testing.assert_allclose(
                    (parentFaceShapeFunctionsAt(faceType, xi + e) - parentFaceShapeFunctionsAt(faceType, xi - e))
                    / (2 * h),
                    first[:, alpha],
                    atol=1e-8,
                )


class TestFacetRuleAccuracy(unittest.TestCase):
    """Issue #216, item 2: the 3-point triangle rule is exact to degree 2, the serendipity functions
    contain cubic terms."""

    def _facetIntegrals(self, nPoints: int) -> np.ndarray:
        """Per midside facet of the square [-1, 1]^2, the integrals of the 8 quad8 functions."""

        xi = PARENT_FACE_PARAMETRIC_COORDS["quad8"]
        barycentric, weights = facetQuadratureRule(3, nPoints)
        result = []
        for triangle in _MIDSIDE_TRIANGLES:
            Xi = xi[list(triangle)]
            e1, e2 = Xi[1] - Xi[0], Xi[2] - Xi[0]
            area = 0.5 * abs(e1[0] * e2[1] - e1[1] * e2[0])
            result.append(area * weights @ parentFaceShapeFunctionsAt("quad8", barycentric @ Xi))
        return np.array(result)

    def test_three_point_rule_is_not_exact_per_facet_but_over_the_face(self):
        threePoint, sixPoint = self._facetIntegrals(3), self._facetIntegrals(6)
        # The 6-point rule is exact to degree 4, so it is the exact facet integral.
        self.assertGreater(np.abs(threePoint - sixPoint).max(), 1e-4)
        np.testing.assert_allclose(threePoint.sum(axis=0), sixPoint.sum(axis=0), atol=1e-14)
        np.testing.assert_allclose(sixPoint.sum(axis=0), [-1 / 3] * 4 + [4 / 3] * 4, atol=1e-14)


class TestParentFaceGaussRule(unittest.TestCase):
    def test_weights_and_exactness(self):
        for faceType, d in PARENT_FACE_LOCAL_DIMENSION.items():
            for n in (1, 2, 3, 4):
                points, weights = parentFaceGaussRule(faceType, n)
                self.assertEqual(len(points), n**d)
                self.assertAlmostEqual(weights.sum(), 2.0**d, places=14)
                # Exact to degree 2n - 1 per direction: the integral of x^(2n-2) over [-1, 1] is 2/(2n-1).
                exact = (2.0 / (2 * n - 1)) ** d
                self.assertAlmostEqual(weights @ np.prod(points ** (2 * n - 2), axis=1), exact, places=13)

    def _consistentLoads(self, coords: np.ndarray, nPerDirection: int) -> np.ndarray:
        points, weights = parentFaceGaussRule("quad8", nPerDirection)
        J = parentFaceAreaElement(parentFaceTangents("quad8", points, coords))
        return (weights * J) @ parentFaceShapeFunctionsAt("quad8", points)

    def test_two_by_two_is_exact_on_a_parallelogram(self):
        coords = _flatQuad8(distorted=False)
        np.testing.assert_allclose(self._consistentLoads(coords, 2), [-1 / 3] * 4 + [4 / 3] * 4, atol=1e-14)

    def test_three_by_three_is_exact_on_a_distorted_flat_face(self):
        """A trapezoid with off-centre midside nodes: N_a |J| is a polynomial of degree 5 per
        direction, which the 3 x 3 rule integrates exactly and the 2 x 2 rule does not."""

        coords = _flatQuad8(distorted=True)
        reference = _referenceIntegrals("quad8", coords)
        np.testing.assert_allclose(self._consistentLoads(coords, 3), reference, rtol=0, atol=1e-9)
        self.assertGreater(np.abs(self._consistentLoads(coords, 2) - reference).max(), 1e-4)

    def test_convergence_on_a_curved_face(self):
        """A quad8 face on the parabolic cylinder y = c (x - 1)^2, represented exactly by the quad8 map."""

        xi = PARENT_FACE_PARAMETRIC_COORDS["quad8"]
        coords = np.column_stack([1.0 + xi[:, 0], 0.4 * xi[:, 0] ** 2, 1.0 + xi[:, 1]])
        reference = _referenceIntegrals("quad8", coords)
        errors = [np.abs(self._consistentLoads(coords, n) - reference).max() for n in (2, 3, 4)]
        self.assertLess(errors[1], errors[0])
        self.assertLess(errors[2], errors[1])
        self.assertLess(errors[1], 1e-2 * np.abs(reference).max())


class TestParentFaceClosestPoints(unittest.TestCase):
    def _parabolicFace(self, c: float = 0.4) -> np.ndarray:
        xi = PARENT_FACE_PARAMETRIC_COORDS["quad8"]
        return np.column_stack([1.0 + xi[:, 0], c * xi[:, 0] ** 2, 1.0 + xi[:, 1]])

    @staticmethod
    def _exactClosestPoint(point: np.ndarray, c: float = 0.4) -> np.ndarray:
        """The closest point on y = c (x - 1)^2, z free, by Newton's method in x, independently."""

        x = point[0]
        for _ in range(50):
            s = x - 1.0
            f = (x - point[0]) + (c * s * s - point[1]) * 2.0 * c * s
            df = 1.0 + 2.0 * c * (c * s * s - point[1]) + (2.0 * c * s) ** 2
            x -= f / df
        return np.array([x, c * (x - 1.0) ** 2, point[2]])

    def test_projection_onto_a_curved_face_is_the_exact_closest_point(self):
        coords = self._parabolicFace()
        rng = np.random.default_rng(3)
        points = np.column_stack([rng.uniform(0.2, 1.8, 20), rng.uniform(-0.3, 0.6, 20), rng.uniform(0.1, 1.9, 20)])
        start = np.column_stack([points[:, 0] - 1.0, points[:, 2] - 1.0]) * 0.9  # a rough guess
        xi, closest, converged = parentFaceClosestPoints("quad8", np.repeat(coords[None], 20, 0), points, start, 1e-13)
        self.assertTrue(converged.all())
        exact = np.array([self._exactClosestPoint(p) for p in points])
        np.testing.assert_allclose(closest, exact, atol=1e-10)
        # The normal there is along the distance vector (Wriggers 2006, Eqs. 4.4-4.5).
        normals = parentFaceUnitNormals(parentFaceTangents("quad8", xi, np.repeat(coords[None], 20, 0)))
        distance = points - closest
        cross = np.cross(normals, distance / np.linalg.norm(distance, axis=1)[:, None])
        np.testing.assert_allclose(cross, 0.0, atol=1e-9)

    def test_projection_is_clamped_to_the_face(self):
        coords = self._parabolicFace()
        point = np.array([[2.6, 0.5, 2.4]])
        xi, closest, converged = parentFaceClosestPoints("quad8", coords[None], point, np.zeros((1, 2)), 1e-13)
        self.assertTrue(converged.all())
        np.testing.assert_allclose(xi, [[1.0, 1.0]], atol=1e-14)

    def test_line_faces(self):
        coords = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 0.3]])  # a line3 arc
        point = np.array([[0.7, 1.0]])
        xi, closest, converged = parentFaceClosestPoints("line3", coords[None], point, np.zeros((1, 1)), 1e-13)
        self.assertTrue(converged.all())
        tangent = parentFaceTangents("line3", xi, coords)[0, 0]
        self.assertAlmostEqual(float((point[0] - closest[0]) @ tangent), 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
