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
"""
Geometry-only contact elements (``CONLINE2``, ``CONLINE3``, ``CONQUAD4``, ``CONQUAD8``, ``CONQUAD9``,
``CONTRI3``, ``CONTRI6``): whole element faces of a solid, carrying neither degrees of freedom nor
stiffness of their own. They describe the contact surfaces of
:mod:`~edelweissfe.constraints.mortarcontact` and, as whole faces, may also be given to the node- and
segment-based contact constraints. Besides the topology they provide the face's shape functions,
quadrature rules and, for the mortar method, the basis transformation and the reference-element
dual basis.
"""

import numpy as np

from edelweissfe.elements.base.baseelement import BaseElement
from edelweissfe.elements.library import elLibrary
from edelweissfe.points.node import Node

#: 3-point Gauss-Legendre rule on [-1, 1] (exact to degree 5).
lineGaussPoints = np.array(
    [
        [-np.sqrt(0.6)],
        [0.0],
        [np.sqrt(0.6)],
    ]
)
lineGaussWeights = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])

# 7-point symmetric rule on the reference triangle (exact to degree 5), Dunavant (1985), p = 5,
# n_g = 7, in closed form. Weights sum to 1/2, the area of the reference triangle.
_triangleGaussA = (6.0 - np.sqrt(15.0)) / 21.0
_triangleGaussB = (6.0 + np.sqrt(15.0)) / 21.0
#: The points of the 7-point triangle rule, in the natural coordinates (r, s).
triangleGaussPoints = np.array(
    [
        [1.0 / 3.0, 1.0 / 3.0],
        [_triangleGaussA, _triangleGaussA],
        [_triangleGaussA, 1.0 - 2.0 * _triangleGaussA],
        [1.0 - 2.0 * _triangleGaussA, _triangleGaussA],
        [_triangleGaussB, _triangleGaussB],
        [_triangleGaussB, 1.0 - 2.0 * _triangleGaussB],
        [1.0 - 2.0 * _triangleGaussB, _triangleGaussB],
    ]
)
#: The weights of the 7-point triangle rule.
triangleGaussWeights = np.array(
    [
        9.0 / 80.0,
        (155.0 - np.sqrt(15.0)) / 2400.0,
        (155.0 - np.sqrt(15.0)) / 2400.0,
        (155.0 - np.sqrt(15.0)) / 2400.0,
        (155.0 + np.sqrt(15.0)) / 2400.0,
        (155.0 + np.sqrt(15.0)) / 2400.0,
        (155.0 + np.sqrt(15.0)) / 2400.0,
    ]
)

# Shape function derivatives at the element's own nodes, per element type: they do not depend on the
# geometry, so one table per type serves every element (see getShapeFunctionDerivativesAtNodes).
_shapeFunctionDerivativesAtNodes: dict[str, np.ndarray] = {}


class ContactElement(BaseElement):
    """A geometry-only element for a whole face of a contact surface."""

    def __init__(self, elementType: str, elNumber: int):
        self._elType = elementType
        self._elNumber = elNumber
        self._nodes = []
        self._properties = np.array([])

        properties = elLibrary[elementType]
        self._nNodes = properties["nNodes"]
        self._nDof = properties["nDof"]
        self._dofIndices = properties["dofIndices"]
        self._ensightType = properties["ensightType"]
        self.nSpatialDimensions = properties["nSpatialDimensions"]

        if self._elType.upper() in ("CONLINE2", "CONLINE3"):
            self.nLocalDim = 1
        else:
            self.nLocalDim = 2

        self._fields = [["displacement"] for _ in range(self._nNodes)]

    @property
    def elNumber(self) -> int:
        return self._elNumber

    @property
    def elType(self) -> str:
        return self._elType

    @property
    def nNodes(self) -> int:
        return self._nNodes

    @property
    def nodes(self) -> list[Node]:
        return self._nodes

    @property
    def hasMaterial(self) -> bool:
        # The model requires every element to report a material; this one needs none.
        return True

    @property
    def fields(self) -> list[list[str]]:
        return self._fields

    @property
    def dofIndicesPermutation(self) -> np.ndarray:
        return self._dofIndices

    @property
    def ensightType(self) -> str:
        return self._ensightType

    @property
    def visualizationNodes(self) -> list[Node]:
        return self._nodes

    @property
    def nDof(self) -> int:
        return self._nDof

    def setNodes(self, nodes: list[Node]):
        if len(nodes) != self._nNodes:
            raise ValueError(f"Element {self._elType} requires {self._nNodes} nodes, got {len(nodes)}.")
        self._nodes = nodes

    def setProperties(self, elementProperties: np.ndarray):
        self._properties = elementProperties

    def initializeElement(self):
        pass

    def setMaterial(self, materialName: str, materialProperties: np.ndarray = None):
        pass

    def setInitialCondition(self, stateType: str, values: np.ndarray):
        pass

    def computeDistributedLoad(
        self,
        loadType: str,
        P: np.ndarray,
        K: np.ndarray,
        faceID: int,
        load: np.ndarray,
        U: np.ndarray,
        time: float,
        dT: float,
    ):
        pass

    def computeKernels(
        self,
        P: np.ndarray,
        K: np.ndarray,
        U: np.ndarray,
        dU: np.ndarray,
        time: float,
        dT: float,
    ):
        pass

    # Abstract methods from BaseNodeCouplingEntity & VIJEntityBase
    def acceptLastState(self):
        pass

    def resetToLastValidState(self):
        pass

    def getStateVars(self) -> np.ndarray:
        """This element carries no state from one increment to the next.

        Returns
        -------
        np.ndarray
            An empty buffer.
        """

        return np.empty(0)

    def setStateVars(self, values: np.ndarray):
        """Nothing to restore: this element carries no state.

        Parameters
        ----------
        values
            The empty buffer :meth:`getStateVars` returned.
        """

    def computeBodyForce(self, P, load, U, time, dT):
        pass

    def computeCriticalTimeStepForExplicitDynamics(self, Q):
        # No stiffness, no stability limit.
        return np.inf

    def computeInternalEnergy(self, U=None, dU=None):
        return 0.0

    def computeKernelsExplicit(self, P, U, dU, time, dT):
        pass

    def computeLumpedInertia(self, M_diag, U=None, time=None):
        pass

    def getCoordinatesAtCenter(self) -> np.ndarray:
        if not self._nodes:
            return np.zeros(self.nSpatialDimensions)
        coords = np.array([node.coordinates for node in self._nodes])
        return np.mean(coords, axis=0)

    def getQuadraturePoints(self) -> tuple[list[np.ndarray], list[float]]:
        """The natural coordinates and weights of the element's quadrature rule: Gauss-Legendre with 2
        (linear) or 3 (quadratic) points per direction, and on triangles the 3-point rule (linear) or
        the 7-point rule (quadratic)."""

        elType = self._elType.upper()
        if elType == "CONLINE2":
            points = [np.array([-1.0 / np.sqrt(3.0)]), np.array([1.0 / np.sqrt(3.0)])]
            weights = [1.0, 1.0]
        elif elType == "CONLINE3":
            points, weights = list(lineGaussPoints), list(lineGaussWeights)
        elif elType == "CONQUAD4":
            gp = 1.0 / np.sqrt(3.0)
            points = [np.array([-gp, -gp]), np.array([gp, -gp]), np.array([gp, gp]), np.array([-gp, gp])]
            weights = [1.0, 1.0, 1.0, 1.0]
        elif elType in ("CONQUAD8", "CONQUAD9"):
            linePoints, lineWeights = lineGaussPoints[:, 0], lineGaussWeights
            points = []
            weights = []
            for i in range(3):
                for j in range(3):
                    points.append(np.array([linePoints[i], linePoints[j]]))
                    weights.append(lineWeights[i] * lineWeights[j])
        elif elType == "CONTRI3":
            points = [
                np.array([1.0 / 6.0, 1.0 / 6.0]),
                np.array([2.0 / 3.0, 1.0 / 6.0]),
                np.array([1.0 / 6.0, 2.0 / 3.0]),
            ]
            weights = [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]
        elif elType == "CONTRI6":
            points, weights = list(triangleGaussPoints), list(triangleGaussWeights)
        else:
            raise NotImplementedError(f"Quadrature points not defined for element type '{elType}'")
        return points, weights

    def getCoordinatesAtQuadraturePoints(self, coords: np.ndarray = None) -> np.ndarray:
        if coords is None:
            if not self._nodes:
                return np.array([])
            coords = np.array([node.coordinates for node in self._nodes])
        points, _ = self.getQuadraturePoints()
        return np.array([self.getShapeFunctions(xi) @ coords for xi in points])

    def getNumberOfQuadraturePoints(self) -> int:
        points, _ = self.getQuadraturePoints()
        return len(points)

    def getShapeFunctionDerivativesAtNodes(self) -> np.ndarray:
        """The shape function derivatives at the element's own nodes, shape (nNodes, nLocalDim,
        nNodes): entry [a] is ``getShapeFunctionDerivatives(getNodalNaturalCoordinates()[a])``.
        They depend on the element type only and are computed once per type."""

        cached = _shapeFunctionDerivativesAtNodes.get(self._elType.upper())
        if cached is None:
            cached = np.array([self.getShapeFunctionDerivatives(xi) for xi in self.getNodalNaturalCoordinates()])
            cached.flags.writeable = False
            _shapeFunctionDerivativesAtNodes[self._elType.upper()] = cached
        return cached

    def getNodalNaturalCoordinates(self) -> np.ndarray:
        """The natural coordinates of the element's own nodes, in node order, shape (nNodes,
        nLocalDim). The order is that of :meth:`getShapeFunctions`: ``N_a`` at the coordinate of node
        ``b`` is ``delta_ab``."""

        elType = self._elType.upper()
        if elType == "CONLINE2":
            return np.array([[-1.0], [1.0]])
        if elType == "CONLINE3":
            # node order is [end, end, mid]
            return np.array([[-1.0], [1.0], [0.0]])
        corners = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
        mids = [[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
        if elType == "CONQUAD4":
            return np.array(corners)
        if elType == "CONQUAD8":
            return np.array(corners + mids)
        if elType == "CONQUAD9":
            return np.array(corners + mids + [[0.0, 0.0]])
        if elType == "CONTRI3":
            return np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        if elType == "CONTRI6":
            return np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
        raise NotImplementedError(f"Nodal natural coordinates not defined for element type '{elType}'")

    def getShapeFunctions(self, localCoords: np.ndarray) -> np.ndarray:
        """The standard shape functions at the natural coordinates ``localCoords``."""

        elType = self._elType.upper()
        if elType == "CONLINE2":
            xi = localCoords[0]
            return np.array([0.5 * (1.0 - xi), 0.5 * (1.0 + xi)])
        elif elType == "CONLINE3":
            xi = localCoords[0]
            return np.array([0.5 * xi * (xi - 1.0), 0.5 * xi * (xi + 1.0), 1.0 - xi * xi])
        elif elType == "CONQUAD4":
            xi, eta = localCoords[0], localCoords[1]
            return np.array(
                [
                    0.25 * (1.0 - xi) * (1.0 - eta),
                    0.25 * (1.0 + xi) * (1.0 - eta),
                    0.25 * (1.0 + xi) * (1.0 + eta),
                    0.25 * (1.0 - xi) * (1.0 + eta),
                ]
            )
        elif elType == "CONQUAD8":
            xi, eta = localCoords[0], localCoords[1]
            return np.array(
                [
                    0.25 * (1.0 - xi) * (1.0 - eta) * (-xi - eta - 1.0),
                    0.25 * (1.0 + xi) * (1.0 - eta) * (xi - eta - 1.0),
                    0.25 * (1.0 + xi) * (1.0 + eta) * (xi + eta - 1.0),
                    0.25 * (1.0 - xi) * (1.0 + eta) * (-xi + eta - 1.0),
                    0.5 * (1.0 - xi * xi) * (1.0 - eta),
                    0.5 * (1.0 + xi) * (1.0 - eta * eta),
                    0.5 * (1.0 - xi * xi) * (1.0 + eta),
                    0.5 * (1.0 - xi) * (1.0 - eta * eta),
                ]
            )
        elif elType == "CONQUAD9":
            xi, eta = localCoords[0], localCoords[1]
            l0_xi, l1_xi, l2_xi = 0.5 * xi * (xi - 1.0), 1.0 - xi * xi, 0.5 * xi * (xi + 1.0)
            l0_eta, l1_eta, l2_eta = 0.5 * eta * (eta - 1.0), 1.0 - eta * eta, 0.5 * eta * (eta + 1.0)
            return np.array(
                [
                    l0_xi * l0_eta,
                    l2_xi * l0_eta,
                    l2_xi * l2_eta,
                    l0_xi * l2_eta,
                    l1_xi * l0_eta,
                    l2_xi * l1_eta,
                    l1_xi * l2_eta,
                    l0_xi * l1_eta,
                    l1_xi * l1_eta,
                ]
            )
        elif elType == "CONTRI3":
            r, s = localCoords[0], localCoords[1]
            return np.array([1.0 - r - s, r, s])
        elif elType == "CONTRI6":
            r, s = localCoords[0], localCoords[1]
            L1 = 1.0 - r - s
            return np.array(
                [
                    L1 * (2.0 * L1 - 1.0),
                    r * (2.0 * r - 1.0),
                    s * (2.0 * s - 1.0),
                    4.0 * r * L1,
                    4.0 * r * s,
                    4.0 * s * L1,
                ]
            )
        else:
            raise NotImplementedError(f"Shape functions not defined for element type '{elType}'")

    def getShapeFunctionDerivatives(self, localCoords: np.ndarray) -> np.ndarray:
        """The derivatives of the standard shape functions with respect to the natural coordinates at
        ``localCoords``, shape (nLocalDim, nNodes)."""

        elType = self._elType.upper()
        if elType == "CONLINE2":
            return np.array([[-0.5, 0.5]])
        elif elType == "CONLINE3":
            xi = localCoords[0]
            return np.array([[xi - 0.5, xi + 0.5, -2.0 * xi]])
        elif elType == "CONQUAD4":
            xi, eta = localCoords[0], localCoords[1]
            return np.array(
                [
                    [-0.25 * (1.0 - eta), 0.25 * (1.0 - eta), 0.25 * (1.0 + eta), -0.25 * (1.0 + eta)],
                    [-0.25 * (1.0 - xi), -0.25 * (1.0 + xi), 0.25 * (1.0 + xi), 0.25 * (1.0 - xi)],
                ]
            )
        elif elType == "CONQUAD8":
            xi, eta = localCoords[0], localCoords[1]
            return np.array(
                [
                    [
                        0.25 * (1.0 - eta) * (2.0 * xi + eta),
                        0.25 * (1.0 - eta) * (2.0 * xi - eta),
                        0.25 * (1.0 + eta) * (2.0 * xi + eta),
                        0.25 * (1.0 + eta) * (2.0 * xi - eta),
                        -xi * (1.0 - eta),
                        0.5 * (1.0 - eta * eta),
                        -xi * (1.0 + eta),
                        -0.5 * (1.0 - eta * eta),
                    ],
                    [
                        0.25 * (1.0 - xi) * (xi + 2.0 * eta),
                        0.25 * (1.0 + xi) * (-xi + 2.0 * eta),
                        0.25 * (1.0 + xi) * (xi + 2.0 * eta),
                        0.25 * (1.0 - xi) * (-xi + 2.0 * eta),
                        -0.5 * (1.0 - xi * xi),
                        -eta * (1.0 + xi),
                        0.5 * (1.0 - xi * xi),
                        -eta * (1.0 - xi),
                    ],
                ]
            )
        elif elType == "CONQUAD9":
            xi, eta = localCoords[0], localCoords[1]
            l0_xi, l1_xi, l2_xi = 0.5 * xi * (xi - 1.0), 1.0 - xi * xi, 0.5 * xi * (xi + 1.0)
            l0_eta, l1_eta, l2_eta = 0.5 * eta * (eta - 1.0), 1.0 - eta * eta, 0.5 * eta * (eta + 1.0)
            dl0_xi, dl1_xi, dl2_xi = xi - 0.5, -2.0 * xi, xi + 0.5
            dl0_eta, dl1_eta, dl2_eta = eta - 0.5, -2.0 * eta, eta + 0.5
            return np.array(
                [
                    [
                        dl0_xi * l0_eta,
                        dl2_xi * l0_eta,
                        dl2_xi * l2_eta,
                        dl0_xi * l2_eta,
                        dl1_xi * l0_eta,
                        dl2_xi * l1_eta,
                        dl1_xi * l2_eta,
                        dl0_xi * l1_eta,
                        dl1_xi * l1_eta,
                    ],
                    [
                        l0_xi * dl0_eta,
                        l2_xi * dl0_eta,
                        l2_xi * dl2_eta,
                        l0_xi * dl2_eta,
                        l1_xi * dl0_eta,
                        l2_xi * dl1_eta,
                        l1_xi * dl2_eta,
                        l0_xi * dl1_eta,
                        l1_xi * dl1_eta,
                    ],
                ]
            )
        elif elType == "CONTRI3":
            return np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
        elif elType == "CONTRI6":
            r, s = localCoords[0], localCoords[1]
            return np.array(
                [
                    [4.0 * (r + s) - 3.0, 4.0 * r - 1.0, 0.0, 4.0 * (1.0 - 2.0 * r - s), 4.0 * s, -4.0 * s],
                    [4.0 * (r + s) - 3.0, 0.0, 4.0 * s - 1.0, -4.0 * r, 4.0 * r, 4.0 * (1.0 - r - 2.0 * s)],
                ]
            )
        else:
            raise NotImplementedError(f"Shape function derivatives not defined for element type '{elType}'")

    def getJacobianAndAreaWeight(self, localCoords: np.ndarray, coords: np.ndarray) -> float:
        """The surface Jacobian ``|x,xi x x,eta|`` (3D) or the length element ``|x,xi|`` (2D) at
        ``localCoords``, with the node coordinates ``coords`` of shape (nNodes, dim)."""

        t = self.getShapeFunctionDerivatives(localCoords) @ coords
        if self.nLocalDim == 1:
            return float(np.linalg.norm(t[0]))
        return float(np.linalg.norm(np.cross(t[0], t[1])))

    def getBasisTransformation(self) -> np.ndarray:
        """The basis transformation T_e for the dual basis of second-order elements, ``N~ = T_e N``.

        The contributions of the mid-side nodes are shifted to their adjacent corner nodes,

            N~_corner = N_corner + alpha * (N of the adjacent mid-side nodes)
            N~_mid    = (1 - 2 alpha) * N_mid ,

        which keeps the partition of unity and makes the integrals of the corner functions positive,
        as the dual basis requires (Popp, Wohlmuth, Gee & Wall 2012, Section 4.4.1, Eqs. (4.6), (4.9)).
        Without it the corner integrals are ``-A/12`` (CONQUAD8) and 0 (CONTRI6). CONLINE3 has positive
        integrals already; the transformation makes its corner functions pointwise non-negative, which
        matters where a facet is covered only in part.

        alpha = 1/3, the value of Farah (2018), Section 6.2.3.2, Eq. (6.20), for the same transformation
        (admissible for alpha < 1/2), where Popp et al. (2012) suggest 1/5. Both satisfy the
        conditions alpha > 0 (tri6) and alpha > 1/8 (quad8); 1/3 is taken because the minimum of the
        transformed quad8 corner function over the element is -1/324 at alpha = 1/3 against -0.0706
        at alpha = 1/5, which matters for partially covered facets.

        CONQUAD9 and the linear types get the identity: the quad9 shape functions already satisfy the
        positivity condition (Popp et al. 2012). Under partial coverage the nodal weights of CONQUAD9
        can still turn negative, which the mortar constraint handles through ``sgn(D_II)``.
        """

        n = self._nNodes
        T_e = np.eye(n)
        alpha = 1.0 / 3.0

        elType = self._elType.upper()
        if elType == "CONQUAD8":
            midToCorners = {4: (0, 1), 5: (1, 2), 6: (2, 3), 7: (3, 0)}
        elif elType == "CONTRI6":
            midToCorners = {3: (0, 1), 4: (1, 2), 5: (2, 0)}
        elif elType == "CONLINE3":
            midToCorners = {2: (0, 1)}
        else:
            return T_e

        for mid, (c1, c2) in midToCorners.items():
            T_e[mid, mid] = 1.0 - 2.0 * alpha
            T_e[c1, mid] = alpha
            T_e[c2, mid] = alpha
        return T_e

    def computeLocalMassMatrices(self, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The transformed mass matrix ``M_e[a,b] = int N~_a N~_b``, the diagonal ``D_e[a,a] = int N~_a``
        and the dual coefficients ``A_e = D_e M_e^-1 T_e`` over the whole element, with the node
        coordinates ``coords`` of shape (nNodes, dim). The dual shape functions ``Phi = A_e N`` are
        biorthogonal to the transformed basis, ``int Phi_a N~_b = delta_ab int N~_a`` (Wohlmuth 2000;
        Popp, Wohlmuth, Gee & Wall 2012, Section 4). The mortar coupling uses it where the system
        of the integrated part of an element is singular."""

        n = self._nNodes
        M_e = np.zeros((n, n))
        D_e = np.zeros((n, n))

        T_e = self.getBasisTransformation()
        points, weights = self.getQuadraturePoints()

        for localCoords, w in zip(points, weights):
            N_t = T_e @ self.getShapeFunctions(localCoords)
            dGamma = self.getJacobianAndAreaWeight(localCoords, coords) * w

            M_e += np.outer(N_t, N_t) * dGamma
            for i in range(n):
                D_e[i, i] += N_t[i] * dGamma

        try:
            invM = np.linalg.inv(M_e)
        except np.linalg.LinAlgError:
            invM = np.linalg.pinv(M_e)

        A_e = D_e @ invM @ T_e
        return M_e, D_e, A_e

    def getResultArray(self, name: str, U: np.ndarray) -> np.ndarray:
        return np.array([])
