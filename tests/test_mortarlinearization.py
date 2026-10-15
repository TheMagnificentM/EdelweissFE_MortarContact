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
"""Each building block of :mod:`edelweissfe.utils.mortarlinearization` against central finite
differences of the value computation it differentiates."""

import unittest

import numpy as np

from edelweissfe.elements.contactelement.element import ContactElement as Element
from edelweissfe.utils.mortargeometry import (
    sutherlandHodgmanClip,
    tangentBasis,
    toPlaneCoordinates,
)
from edelweissfe.utils.mortarlinearization import (
    elementNormalAtNode,
    fanTriangleArea,
    nodeSlots,
    planeCoordinates,
    shapeFunctionSecondDerivatives,
    skew,
    surfaceJacobian,
    surfaceJacobianNaturalDerivative,
    sutherlandHodgmanClipWithDerivatives,
    unitVector,
)

STEP = 1e-6


def finiteDifference(f, z, h=STEP):
    """Central-difference Jacobian of ``f`` (array-valued) at ``z``."""
    f0 = np.asarray(f(z), dtype=float)
    J = np.zeros(f0.shape + (len(z),))
    for i in range(len(z)):
        zp, zm = z.copy(), z.copy()
        zp[i] += h
        zm[i] -= h
        J[..., i] = (np.asarray(f(zp)) - np.asarray(f(zm))) / (2 * h)
    return J


def assertJacobian(testCase, analytic, numeric, rtol=1e-6):
    scale = max(np.max(np.abs(numeric)), 1.0)
    testCase.assertLess(np.max(np.abs(analytic - numeric)) / scale, rtol)


def planeFrame(z, nSub=4):
    """Plane of the first ``nSub`` points in ``z`` (flattened 3D coordinates) as the segmentation
    builds it: facet normal, centroid, tangent basis."""
    X = z.reshape(-1, 3)
    sc = X[:nSub]
    if nSub == 3:
        c = 0.5 * np.cross(sc[1] - sc[0], sc[2] - sc[0])
    else:
        c = 0.5 * np.cross(sc[2] - sc[0], sc[3] - sc[1])
    n = c / np.linalg.norm(c)
    p0 = sc.mean(axis=0)
    return n, p0


class TestNormals(unittest.TestCase):
    def test_element_normal_3d(self):
        rng = np.random.default_rng(1)
        for elType in ("CONQUAD4", "CONTRI3", "CONQUAD8", "CONQUAD9", "CONTRI6"):
            el = Element(elType, 1)
            dNn = el.getShapeFunctionDerivativesAtNodes()
            nNodes = dNn.shape[0]
            X = np.array([[*xi, 0.0] for xi in el.getNodalNaturalCoordinates()]) + 0.1 * rng.standard_normal(
                (nNodes, 3)
            )
            for node in range(nNodes):
                _, d = elementNormalAtNode(dNn[node], X, 3)
                num = finiteDifference(lambda z: elementNormalAtNode(dNn[node], z.reshape(-1, 3), 3)[0], X.ravel())
                assertJacobian(self, d, num)

    def test_element_normal_2d(self):
        for elType in ("CONLINE2", "CONLINE3"):
            el = Element(elType, 1)
            dNn = el.getShapeFunctionDerivativesAtNodes()
            X = np.array([[0.0, 0.0], [1.0, 0.2], [0.45, 0.3]])[: dNn.shape[0]]
            for node in range(dNn.shape[0]):
                _, d = elementNormalAtNode(dNn[node], X, 2)
                num = finiteDifference(lambda z: elementNormalAtNode(dNn[node], z.reshape(-1, 2), 2)[0], X.ravel())
                assertJacobian(self, d, num)


class TestAuxiliaryPlane(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        # A slave quad sub-cell and four further points, all slightly out of plane.
        base = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.2, 0.1, 0.05], [1.3, 0.2, -0.1]], float)
        self.z = (base + 0.05 * rng.standard_normal(base.shape)).ravel()

    def test_in_plane_quantities_are_invariant_to_the_dropped_rotation(self):
        """The plane coordinates are differentiated without the rotation of t1, t2 about the normal,
        so they agree with finite differences only up to that rotation; quantities that do not
        depend on it -- distances between projected points, here -- agree exactly."""
        X = self.z.reshape(-1, 3)
        dX = nodeSlots(len(X), 3, len(self.z))
        a, b = X[2] - X[0], X[3] - X[1]
        normal, dnormal = unitVector(np.cross(a, b), -skew(b) @ (dX[2] - dX[0]) + skew(a) @ (dX[3] - dX[1]))
        p0, dp0 = X[:4].mean(axis=0), dX[:4].mean(axis=0)
        t1, t2 = tangentBasis(normal)
        q, dq = planeCoordinates(X, dX, p0, dp0, normal, dnormal, t1, t2)

        def distances(z):
            Y = z.reshape(-1, 3)
            n, c = planeFrame(z)
            s1, s2 = tangentBasis(n)
            qq = toPlaneCoordinates(Y, c, s1, s2)
            return np.array([np.sum((qq[a] - qq[b]) ** 2) for a in range(len(Y)) for b in range(a)])

        analytic = np.array([2 * (q[a] - q[b]) @ (dq[a] - dq[b]) for a in range(len(X)) for b in range(a)])
        assertJacobian(self, analytic, finiteDifference(distances, self.z))


class TestClipping(unittest.TestCase):
    def test_values_identical_and_area_derivative(self):
        rng = np.random.default_rng(3)
        clip = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        subject = np.array([[0.3, -0.2], [1.4, 0.25], [1.1, 1.3], [0.2, 0.8]])
        for trial in range(20):
            s = subject + 0.1 * rng.standard_normal(subject.shape)
            c = clip + 0.05 * rng.standard_normal(clip.shape)
            z = np.concatenate([s.ravel(), c.ravel()])
            dZ = nodeSlots(8, 2, 16)
            poly, dpoly = sutherlandHodgmanClipWithDerivatives(s, dZ[:4], c, dZ[4:])
            np.testing.assert_array_equal(poly, sutherlandHodgmanClip(s, c))

            # The overlap AREA, through the fan triangulation, is what the integration uses.
            def area(zz):
                p = sutherlandHodgmanClip(zz[:8].reshape(4, 2), zz[8:].reshape(4, 2))
                return sum(fanTriangleArea(p[[0, i, i + 1]], np.zeros((3, 2, 1)))[0] for i in range(1, len(p) - 1))

            dA = sum(fanTriangleArea(poly[[0, i, i + 1]], dpoly[[0, i, i + 1]])[1] for i in range(1, len(poly) - 1))
            assertJacobian(self, dA, finiteDifference(area, z))

    def test_shared_edge_under_round_off(self):
        # Two conforming facets: the subject's lower edge lies on the clip's upper edge. Round-off
        # can classify its two ends on different sides of that edge; the crossing must then stay on
        # the subject edge instead of running off along the nearly parallel line.
        clip = np.array([[-2.0, -1.25], [2.0, -1.25], [2.0, 1.25], [-2.0, 1.25]])
        subject = np.array([[2.0, 1.25], [-2.0, 1.25], [-2.0, 3.75], [2.0, 3.75]])
        rng = np.random.default_rng(5)
        for trial in range(200):
            s = subject + 1e-13 * rng.standard_normal(subject.shape)
            c = clip + 1e-13 * rng.standard_normal(clip.shape)
            poly = sutherlandHodgmanClip(s, c)
            if len(poly):
                self.assertLess(np.max(np.abs(poly[:, 0])), 2.0 + 1e-9)
                self.assertLess(np.max(np.abs(poly[:, 1] - 1.25)), 1e-9)
            dZ = nodeSlots(8, 2, 16)
            np.testing.assert_array_equal(poly, sutherlandHodgmanClipWithDerivatives(s, dZ[:4], c, dZ[4:])[0])


    def test_nearly_parallel_edges_keep_their_crossing_parameter(self):
        # A subject edge crossing a clip edge at an angle with sine 1e-6: the crossing moves by
        # dx / sin(theta), which only holds for perturbations far below a Newton step. Below
        # PARALLEL_EDGE_SINE the parameter is held fixed, and the vertex moves with the end points.
        from edelweissfe.utils.mortarlinearization import PARALLEL_EDGE_SINE

        clip = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        tilt = 1e-6
        subject = np.array([[0.2, 1.0 - tilt * 0.3], [0.8, 1.0 + tilt * 0.3], [0.8, 2.0], [0.2, 2.0]])
        self.assertLess(tilt, PARALLEL_EDGE_SINE)
        dZ = nodeSlots(8, 2, 16)
        poly, dpoly = sutherlandHodgmanClipWithDerivatives(subject, dZ[:4], clip, dZ[4:])
        np.testing.assert_array_equal(poly, sutherlandHodgmanClip(subject, clip))
        self.assertLess(np.max(np.abs(dpoly)), 2.0)


class TestSurfaceJacobian(unittest.TestCase):
    def test_surface_jacobian(self):
        rng = np.random.default_rng(9)
        for elType in ("CONQUAD4", "CONTRI3", "CONQUAD8", "CONQUAD9", "CONTRI6", "CONLINE2", "CONLINE3"):
            el = Element(elType, 1)
            nat = np.array(el.getNodalNaturalCoordinates(), float).reshape(el.nNodes, -1)
            dim = 3 if nat.shape[1] == 2 else 2
            X = np.hstack([nat, np.zeros((el.nNodes, dim - nat.shape[1]))]) + 0.05 * rng.standard_normal(
                (el.nNodes, dim)
            )
            for xi in el.getQuadraturePoints()[0]:
                _, dJ = surfaceJacobian(el, xi, X)
                num = finiteDifference(lambda z: el.getJacobianAndAreaWeight(xi, z.reshape(-1, dim)), X.ravel())
                assertJacobian(self, dJ, num)


class TestElementDerivatives(unittest.TestCase):
    def test_second_derivatives_of_the_shape_functions(self):
        for elType in ("CONLINE2", "CONLINE3", "CONTRI3", "CONQUAD4", "CONTRI6", "CONQUAD8", "CONQUAD9"):
            el = Element(elType, 1)
            xi = np.array([0.23, -0.31])[: (1 if "LINE" in elType else 2)]
            if "TRI" in elType:
                xi = np.array([0.21, 0.33])
            numeric = np.moveaxis(finiteDifference(lambda z: el.getShapeFunctionDerivatives(z), xi), -1, 1)
            assertJacobian(self, shapeFunctionSecondDerivatives(el, xi), numeric)

    def test_natural_derivative_of_the_surface_jacobian(self):
        rng = np.random.default_rng(4)
        for elType in ("CONQUAD4", "CONTRI3", "CONQUAD8", "CONQUAD9", "CONTRI6", "CONLINE2", "CONLINE3"):
            el = Element(elType, 1)
            nat = np.array(el.getNodalNaturalCoordinates(), float).reshape(el.nNodes, -1)
            dim = 3 if nat.shape[1] == 2 else 2
            X = np.hstack([nat, np.zeros((el.nNodes, dim - nat.shape[1]))]) + 0.05 * rng.standard_normal(
                (el.nNodes, dim)
            )
            xi = el.getQuadraturePoints()[0][1]
            numeric = finiteDifference(lambda z: el.getJacobianAndAreaWeight(z, X), np.array(xi, float))
            assertJacobian(self, surfaceJacobianNaturalDerivative(el, xi, X), numeric)


class _FacetPair:
    """Two contact surfaces built from contact elements directly, facing each other across a small
    penetration: a non-mortar grid and a finer, shifted mortar grid, every node displaced at random
    so that the surfaces are tilted and curved and no facet is parallel to another. Small enough
    that a finite-difference tangent over every degree of freedom stays cheap."""

    @staticmethod
    def build(elType: str, nNonMortar: int = 1, nMortar: int = 2, perturbation: float = 0.03, seed: int = 0):
        from edelweissfe.config.elementlibrary import getElementClass
        from edelweissfe.models.femodel import FEModel
        from edelweissfe.points.node import Node
        from edelweissfe.sets.elementset import ElementSet
        from edelweissfe.variables.fieldvariable import FieldVariable

        dim = 2 if "LINE" in elType else 3
        model = FEModel(dim)
        elementClass = getElementClass(elType, "edelweiss")
        natural = np.array(elementClass(elType, 0).getNodalNaturalCoordinates(), dtype=float)

        def surface(origin, n, height, mortar, firstLabel):
            pool, facets, size = {}, [], 1.0 / n

            def node(x):
                key = tuple(np.round(x, 9))
                if key not in pool:
                    nd = Node(1000 + len(model.nodes), np.array(x, dtype=float))
                    nd.fields["displacement"] = FieldVariable(nd, "displacement")
                    model.nodes[nd.label] = nd
                    pool[key] = nd
                return pool[key]

            def add(points):
                element = elementClass(elType, firstLabel + len(facets))
                element.setNodes([node(x) for x in points])
                model.elements[element.elNumber] = element
                facets.append(element)

            for i in range(n):
                if dim == 2:
                    # The non-mortar line runs along -x, so that its normal points up (+y) at the
                    # mortar line above it, which runs along +x and so faces down.
                    s = (natural[:, 0] + 1.0) / 2.0
                    u = s if mortar else 1.0 - s
                    add([[origin[0] + (i + a) * size, height] for a in u])
                    continue
                for j in range(n):
                    if "TRI" in elType:
                        cells = [natural, 1.0 - natural]
                    else:
                        cells = [(natural + 1.0) / 2.0]
                    for cell in cells:
                        # The mortar facets swap the in-plane axes, which turns their normal down.
                        uv = cell[:, ::-1] if mortar else cell
                        add([[origin[0] + (i + a) * size, origin[1] + (j + b) * size, height] for a, b in uv])
            return facets

        nonMortar = surface((0.0, 0.0), nNonMortar, 0.0, False, 1)
        mortar = surface((0.15, 0.1), nMortar, -0.01, True, 100)
        rng = np.random.default_rng(seed)
        for nd in model.nodes.values():
            nd.coordinates = nd.coordinates + perturbation * rng.standard_normal(dim)
        model.elementSets["nonMortar_facets"] = ElementSet("nonMortar_facets", nonMortar)
        model.elementSets["mortar_facets"] = ElementSet("mortar_facets", mortar)
        return model

    @staticmethod
    def constraint(model, **options):
        from edelweissfe.constraints.mortarcontact import Constraint as MortarContact
        from edelweissfe.journal.journal import Journal

        settings = dict(nonMortarSurface="nonMortar_facets", mortarSurface="mortar_facets")
        if options.get("formulation", "lagrange") == "lagrange":
            settings["cn"] = 1000.0
        else:
            settings["penaltyStiffness"] = 1000.0
        settings.update(options)
        return MortarContact("contact", model, Journal(verbose=False), **settings)


def _timeStep():
    from edelweissfe.timesteppers.timestep import TimeStep

    return TimeStep(
        number=1, stepProgressIncrement=1.0, stepProgress=1.0, timeIncrement=1.0, stepTime=1.0, totalTime=1.0
    )


ALL_TYPES = ("CONLINE2", "CONLINE3", "CONTRI3", "CONQUAD4", "CONTRI6", "CONQUAD8", "CONQUAD9")


class TestConsistentTangent(unittest.TestCase):
    """The tangent of ``geometryUpdate=consistent`` against central differences of the residual with
    the geometry REBUILT at every perturbation -- segmentation, coupling matrices and normals all
    follow the perturbed configuration. The active set is held fixed after one live assembly, since
    across a change of it there is no derivative; the layout of the segmentation does not change
    under perturbations of this size."""

    def _check(self, constraint, U, dU, tolerance=1e-6):
        n = constraint.nDof
        timeStep = _timeStep()
        constraint.applyConstraint(U, dU, np.zeros(n), np.zeros((n, n)), timeStep)
        self.assertGreater(int(np.sum(constraint.activeSet)), 0, "no node is in contact -- test is vacuous")
        constraint.useActiveSet = False
        K = np.zeros((n, n))
        constraint.applyConstraint(U, dU, np.zeros(n), K, timeStep)
        h = 1e-7
        numeric = np.zeros((n, n))
        for j in range(n):
            step = np.zeros(n)
            step[j] = h
            plus, minus = np.zeros(n), np.zeros(n)
            constraint.applyConstraint(U + step, dU + step, plus, np.zeros((n, n)), timeStep)
            constraint.applyConstraint(U - step, dU - step, minus, np.zeros((n, n)), timeStep)
            numeric[:, j] = -(plus - minus) / (2.0 * h)
        scale = float(np.max(np.abs(K)))
        self.assertLess(float(np.max(np.abs(K - numeric))) / scale, tolerance)
        return K, numeric

    def test_lagrange(self):
        for elType in ALL_TYPES:
            with self.subTest(elType):
                constraint = _FacetPair.constraint(_FacetPair.build(elType), geometryUpdate="consistent")
                U = np.zeros(constraint.nDof)
                # Non-zero multipliers, so that the geometric terms of the forces are exercised.
                U[constraint.sizeField * len(constraint.nodes) :] = 1.0 + 0.1 * np.arange(constraint.nMultipliers)
                self._check(constraint, U, np.zeros_like(U))

    def test_penalty(self):
        for elType in ALL_TYPES:
            with self.subTest(elType):
                constraint = _FacetPair.constraint(
                    _FacetPair.build(elType), formulation="penalty", geometryUpdate="consistent"
                )
                U = np.zeros(constraint.nDof)
                self._check(constraint, U, U)

    def test_the_frozen_tangent_misses_the_geometric_terms(self):
        """The same comparison for the frozen tangent fails by orders of magnitude: what the
        consistent tangent adds is not round-off."""
        constraint = _FacetPair.constraint(
            _FacetPair.build("CONQUAD4"), formulation="penalty", geometryUpdate="consistent"
        )
        constraint._geometricTangent = lambda *args, **kwargs: np.zeros(1)
        with self.assertRaises(AssertionError):
            self._check(constraint, np.zeros(constraint.nDof), np.zeros(constraint.nDof), tolerance=1e-4)

    def _frictionCase(self, elType, slide, mu, **options):
        constraint = _FacetPair.constraint(
            _FacetPair.build(elType), formulation="penalty", mu=mu, geometryUpdate="consistent", **options
        )
        sf, nSlave = constraint.sizeField, constraint.nNonMortarNodes
        U = np.zeros(constraint.nDof)
        for J in range(constraint.nMortarNodes):
            U[sf * (nSlave + J)] = slide
            if sf == 3:
                U[sf * (nSlave + J) + 1] = 0.5 * slide
        U += 1e-3 * np.random.default_rng(1).standard_normal(len(U))
        return constraint, U

    def test_friction_slip(self):
        for elType in ("CONLINE3", "CONQUAD4", "CONQUAD8"):
            with self.subTest(elType):
                constraint, U = self._frictionCase(elType, 0.02, 0.3)
                constraint._friction.tractionConverged[:] = 20.0 * np.random.default_rng(2).standard_normal(
                    constraint._friction.tractionConverged.shape
                )
                self._check(constraint, U, U.copy())
                self.assertTrue(np.any(constraint.getSlipStates() == 2))

    def test_friction_stick(self):
        for elType in ("CONLINE2", "CONQUAD4", "CONTRI6"):
            for options in ({}, {"slipTolerance": 0.01}):
                with self.subTest(elType, **options):
                    constraint, U = self._frictionCase(elType, 1e-5, 2.0, **options)
                    self._check(constraint, U, U.copy())
                    self.assertTrue(np.any(constraint.getSlipStates() == 1))


if __name__ == "__main__":
    unittest.main()
