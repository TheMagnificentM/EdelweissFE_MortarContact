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
"""Unit tests for the geometry-only contact elements of the mortar formulation.

These elements carry no stiffness and no degrees of freedom of their own, so nothing about them is
checked by a simulation that merely converges. What a mortar constraint asks of them is a small set
of interpolation properties, and each one fails in a way that a converging run would not reveal:

* the shape functions must be the cardinal ones of the node ordering the generator supplies, so
  that a nodal quantity is interpolated by the function belonging to that node and not its neighbour
* the quadrature must integrate the element's own geometry exactly, since the coupling matrices are
  nothing but such integrals
* the dual basis must be biorthogonal to the (transformed) standard basis, which is the single
  property that makes the multiplier field condensable and the nodal weights meaningful

The reference values are those of straight-edged elements of known size, where every integral above
is available in closed form.
"""

import unittest

import numpy as np

import edelweissfe.utils.inputfileparser  # noqa: F401 bootstrap input language
from edelweissfe.config.elementlibrary import getElementClass
from edelweissfe.points.node import Node

#: Every contact element type, with the dimension its nodes live in and the number of nodes.
_TYPES = (
    ("CONLINE2", 2, 2),
    ("CONLINE3", 2, 3),
    ("CONQUAD4", 3, 4),
    ("CONQUAD8", 3, 8),
    ("CONQUAD9", 3, 9),
    ("CONTRI3", 3, 3),
    ("CONTRI6", 3, 6),
)

#: Side length of the flat reference geometries below. The quadrilateral types then span an area of
#: 4, the triangular ones half of that, and a line an arc length of 2.
_SIDE = 2.0


def _referenceCoordinates(elementType: str) -> np.ndarray:
    """A flat, straight-edged element of the given type, laid out in the plane y = 0 for the surface
    types and along the x axis for the line types.

    Built from the element's own natural coordinates rather than from a hand-written table: the
    natural coordinates are what the shape functions are written in, so mapping them through an
    affine map produces exactly the geometry the element believes it has. A hand-written table would
    be a second, independent claim about the node ordering, and if the two disagreed the test would
    report the element as broken when only the table was.
    """

    element = getElementClass(elementType, "edelweiss")(elementType, 1)
    natural = element.getNodalNaturalCoordinates()

    if elementType.startswith("CONLINE"):
        # natural coordinates on [-1, 1] -> x on [0, _SIDE], y = 0
        return np.column_stack([0.5 * (natural[:, 0] + 1.0) * _SIDE, np.zeros(len(natural))])

    if elementType.startswith("CONTRI"):
        # barycentric-style natural coordinates on the unit triangle -> the (x, z) plane at y = 0
        return np.column_stack([natural[:, 0] * _SIDE, np.zeros(len(natural)), natural[:, 1] * _SIDE])

    # quadrilaterals: natural coordinates on [-1, 1]^2 -> the (x, z) plane at y = 0
    return np.column_stack(
        [0.5 * (natural[:, 0] + 1.0) * _SIDE, np.zeros(len(natural)), 0.5 * (natural[:, 1] + 1.0) * _SIDE]
    )


def _element(elementType: str):
    """One contact element of the given type on the reference geometry, with real nodes attached."""

    coordinates = _referenceCoordinates(elementType)
    element = getElementClass(elementType, "edelweiss")(elementType, 1)
    element.setNodes([Node(i + 1, x) for i, x in enumerate(coordinates)])
    return element, coordinates


class TestShapeFunctions(unittest.TestCase):
    def test_the_shape_functions_are_cardinal_at_the_nodes(self):
        """N_a evaluated at node b is delta_ab, for every type.

        This is the property that ties the shape functions to the node ORDER, and the only one that
        does. Every other check here -- partition of unity, exact quadrature, even biorthogonality --
        is invariant under a permutation of the nodes, so a table whose entries are individually
        correct but listed in the wrong order passes all of them and still interpolates a nodal gap
        onto the wrong node.
        """

        for elementType, _dim, nNodes in _TYPES:
            element = getElementClass(elementType, "edelweiss")(elementType, 1)
            natural = element.getNodalNaturalCoordinates()
            self.assertEqual(len(natural), nNodes, elementType)

            evaluated = np.array([element.getShapeFunctions(xi) for xi in natural])
            np.testing.assert_allclose(evaluated, np.eye(nNodes), atol=1e-13, err_msg=elementType)

    def test_the_shape_functions_sum_to_one(self):
        """Partition of unity, away from the nodes as well: without it a rigid translation of the
        interface would produce a gap."""

        rng = np.random.default_rng(20260916)
        for elementType, _dim, _nNodes in _TYPES:
            element = getElementClass(elementType, "edelweiss")(elementType, 1)
            natural = element.getNodalNaturalCoordinates()
            nLocal = natural.shape[1]

            for _ in range(20):
                if elementType.startswith("CONTRI"):
                    # stay inside the unit triangle
                    a, b = rng.random(2)
                    if a + b > 1.0:
                        a, b = 1.0 - a, 1.0 - b
                    xi = np.array([a, b])
                else:
                    xi = rng.uniform(-1.0, 1.0, nLocal)
                self.assertAlmostEqual(float(np.sum(element.getShapeFunctions(xi))), 1.0, places=12, msg=elementType)

    def test_the_derivatives_agree_with_finite_differences(self):
        """The derivatives enter the surface Jacobian, hence every integration weight. An error
        there scales the coupling matrices without changing their structure, which no patch test on
        a uniform interface would expose -- a uniformly wrong area still transfers a uniform pressure
        correctly."""

        h = 1e-6
        rng = np.random.default_rng(20260916)
        for elementType, _dim, _nNodes in _TYPES:
            element = getElementClass(elementType, "edelweiss")(elementType, 1)
            nLocal = element.getNodalNaturalCoordinates().shape[1]

            xi = rng.uniform(-0.4, 0.4, nLocal)
            if elementType.startswith("CONTRI"):
                xi = np.array([0.3, 0.25])[:nLocal]

            # Index order is [local direction, node], and the line types flatten the single
            # direction away. Pinning it here is half the value of this test: the array is square
            # for CONQUAD8 in the (2, 8) sense for no type at all, but a transposed read of a
            # CONQUAD9 derivative would be shape-compatible and silently wrong.
            nNodesOfType = len(element.getShapeFunctions(xi))
            analytic = np.asarray(element.getShapeFunctionDerivatives(xi)).reshape(nLocal, nNodesOfType)
            numeric = np.zeros_like(analytic)
            for k in range(nLocal):
                step = np.zeros(nLocal)
                step[k] = h
                numeric[k, :] = (element.getShapeFunctions(xi + step) - element.getShapeFunctions(xi - step)) / (2 * h)

            np.testing.assert_allclose(analytic, numeric, atol=1e-7, err_msg=elementType)


class TestQuadrature(unittest.TestCase):
    def test_the_rule_integrates_the_element_measure_exactly(self):
        """Sum of Jacobian times weight is the area of a flat element (half that for the triangular
        types, the arc length for the line types). The coupling matrices are integrals over exactly
        this measure, so a rule that misses it misses them by the same factor."""

        expected = {
            "CONLINE2": _SIDE,
            "CONLINE3": _SIDE,
            "CONQUAD4": _SIDE**2,
            "CONQUAD8": _SIDE**2,
            "CONQUAD9": _SIDE**2,
            "CONTRI3": 0.5 * _SIDE**2,
            "CONTRI6": 0.5 * _SIDE**2,
        }

        for elementType, _dim, _nNodes in _TYPES:
            element, coordinates = _element(elementType)
            points, weights = element.getQuadraturePoints()
            measure = sum(element.getJacobianAndAreaWeight(xi, coordinates) * w for xi, w in zip(points, weights))
            self.assertAlmostEqual(measure, expected[elementType], places=12, msg=elementType)

    def test_the_rule_integrates_the_shape_functions_to_the_element_measure(self):
        """Their integrals have to sum to the measure as well, which is partition of unity carried
        through the quadrature. It is the consistency between the two that matters: a rule of too
        low an order can still return the right total area while distributing it wrongly over the
        nodes, and the nodal distribution is what becomes the nodal mortar weight."""

        for elementType, _dim, _nNodes in _TYPES:
            element, coordinates = _element(elementType)
            points, weights = element.getQuadraturePoints()

            total = 0.0
            perNode = None
            for xi, w in zip(points, weights):
                dGamma = element.getJacobianAndAreaWeight(xi, coordinates) * w
                N = element.getShapeFunctions(xi)
                perNode = N * dGamma if perNode is None else perNode + N * dGamma
                total += dGamma

            self.assertAlmostEqual(float(np.sum(perNode)), total, places=12, msg=elementType)


class TestDualBasis(unittest.TestCase):
    """The dual basis, which is what makes the multiplier field of this formulation local.

    Its defining property is biorthogonality against the transformed standard basis. Everything the
    constraint does with nodal weights -- deciding whether a node is in contact, turning a weighted
    gap into an opening, turning a multiplier into a pressure -- assumes it holds.
    """

    def test_the_dual_basis_is_biorthogonal_to_the_transformed_basis(self):
        """int(Phi_a * Ntilde_b) = delta_ab * int(Ntilde_a), recomputed here from the element's own
        quadrature rather than read back from the matrices the element returns."""

        for elementType, _dim, nNodes in _TYPES:
            element, coordinates = _element(elementType)
            _M_e, D_e, A_e = element.computeLocalMassMatrices(coordinates)
            T_e = element.getBasisTransformation()

            points, weights = element.getQuadraturePoints()
            coupling = np.zeros((nNodes, nNodes))
            for xi, w in zip(points, weights):
                N = element.getShapeFunctions(xi)
                dGamma = element.getJacobianAndAreaWeight(xi, coordinates) * w
                coupling += np.outer(A_e @ N, T_e @ N) * dGamma

            np.testing.assert_allclose(coupling, D_e, atol=1e-12, err_msg=elementType)

    def test_the_dual_basis_reproduces_a_constant(self):
        """Summed over the nodes the dual functions integrate to the element measure, so a constant
        multiplier field is represented exactly. Without it the formulation could not transfer a
        constant pressure, which is precisely what the patch test decks measure at the system
        level."""

        for elementType, _dim, _nNodes in _TYPES:
            element, coordinates = _element(elementType)
            _M_e, _D_e, A_e = element.computeLocalMassMatrices(coordinates)

            points, weights = element.getQuadraturePoints()
            integral = 0.0
            measure = 0.0
            for xi, w in zip(points, weights):
                dGamma = element.getJacobianAndAreaWeight(xi, coordinates) * w
                integral += float(np.sum(A_e @ element.getShapeFunctions(xi))) * dGamma
                measure += dGamma

            self.assertAlmostEqual(integral, measure, places=11, msg=elementType)

    def test_the_nodal_weights_of_a_fully_covered_element_are_positive(self):
        """D_e is the tributary measure of each node. On a fully covered, undistorted element every
        entry has to be positive -- a negative one is not a small error but a sign flip in the
        active-set decision, which is why the constraint carries that sign explicitly instead of
        assuming it.

        The transformation is what buys this for the serendipity and quadratic triangle types, whose
        untransformed corner functions integrate to zero or less over the full element.
        """

        for elementType, _dim, nNodes in _TYPES:
            element, coordinates = _element(elementType)
            _M_e, D_e, _A_e = element.computeLocalMassMatrices(coordinates)
            diagonal = np.diag(D_e)

            self.assertEqual(len(diagonal), nNodes, elementType)
            self.assertTrue(np.all(diagonal > 0.0), f"{elementType}: non-positive nodal weights {diagonal}")
            self.assertAlmostEqual(float(np.sum(diagonal)), float(np.trace(D_e)), places=12, msg=elementType)

    def test_the_untransformed_serendipity_corner_weight_really_is_non_positive(self):
        """The reason the transformation exists, measured rather than asserted.

        Without it the corner functions of CONQUAD8 integrate to a NEGATIVE measure over the full
        element and those of CONTRI6 to exactly zero, so the nodal weights the constraint divides by
        would be unusable at every corner of every quadratic facet. This test would go quiet if the
        transformation were ever reduced to the identity -- the previous test would then still pass
        on linear elements and fail on quadratic ones without saying why.
        """

        expected = {"CONQUAD8": -1.0 / 12.0 * _SIDE**2, "CONTRI6": 0.0}

        for elementType, reference in expected.items():
            element, coordinates = _element(elementType)
            points, weights = element.getQuadraturePoints()

            cornerIntegral = 0.0
            for xi, w in zip(points, weights):
                dGamma = element.getJacobianAndAreaWeight(xi, coordinates) * w
                cornerIntegral += element.getShapeFunctions(xi)[0] * dGamma

            self.assertAlmostEqual(cornerIntegral, reference, places=12, msg=elementType)
            self.assertLessEqual(cornerIntegral, 1e-12, elementType)


if __name__ == "__main__":
    unittest.main()
