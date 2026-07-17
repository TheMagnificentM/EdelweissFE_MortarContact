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

import numpy as np

from edelweissfe.elements.base.baseelement import BaseElement
from edelweissfe.elements.library import elLibrary
from edelweissfe.points.node import Node


class ContactElement(BaseElement):
    """A geometric placeholder element for contact boundary surfaces."""

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

    def computeBodyForce(self, P, load, U, time, dT):
        pass

    def computeCriticalTimeStepForExplicitDynamics(self, U, time=None):
        return 1.0e10

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

    def getCoordinatesAtCenter(self) -> np.ndarray:
        if not self._nodes:
            return np.zeros(self.nSpatialDimensions)
        coords = np.array([node.coordinates for node in self._nodes])
        return np.mean(coords, axis=0)

    def getQuadraturePoints(self) -> tuple[list[np.ndarray], list[float]]:
        """Return the local coordinates and weights of the quadrature points."""
        el_type = self._elType.upper()
        if el_type == "CONLINE2":
            points = [np.array([-1.0 / np.sqrt(3.0)]), np.array([1.0 / np.sqrt(3.0)])]
            weights = [1.0, 1.0]
        elif el_type == "CONLINE3":
            points = [np.array([-np.sqrt(0.6)]), np.array([0.0]), np.array([np.sqrt(0.6)])]
            weights = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]
        elif el_type == "CONQUAD4":
            gp = 1.0 / np.sqrt(3.0)
            points = [
                np.array([-gp, -gp]),
                np.array([gp, -gp]),
                np.array([gp, gp]),
                np.array([-gp, gp])
            ]
            weights = [1.0, 1.0, 1.0, 1.0]
        elif el_type in ("CONQUAD8", "CONQUAD9"):
            g_pts = [-np.sqrt(0.6), 0.0, np.sqrt(0.6)]
            g_w = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]
            points = []
            weights = []
            for i in range(3):
                for j in range(3):
                    points.append(np.array([g_pts[i], g_pts[j]]))
                    weights.append(g_w[i] * g_w[j])
        elif el_type == "CONTRI3":
            points = [
                np.array([1.0 / 6.0, 1.0 / 6.0]),
                np.array([2.0 / 3.0, 1.0 / 6.0]),
                np.array([1.0 / 6.0, 2.0 / 3.0])
            ]
            weights = [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]
        elif el_type == "CONTRI6":
            a = (6.0 - np.sqrt(15.0)) / 21.0
            b = (6.0 + np.sqrt(15.0)) / 21.0
            w_a = (155.0 - np.sqrt(15.0)) / 2400.0
            w_b = (155.0 + np.sqrt(15.0)) / 2400.0
            points = [
                np.array([1.0 / 3.0, 1.0 / 3.0]),
                np.array([a, a]),
                np.array([a, 1.0 - 2.0 * a]),
                np.array([1.0 - 2.0 * a, a]),
                np.array([b, b]),
                np.array([b, 1.0 - 2.0 * b]),
                np.array([1.0 - 2.0 * b, b])
            ]
            weights = [
                9.0 / 80.0,
                w_a, w_a, w_a,
                w_b, w_b, w_b
            ]
        else:
            raise NotImplementedError(f"Quadrature points not defined for element type '{el_type}'")
        return points, weights

    def getCoordinatesAtQuadraturePoints(self, coords: np.ndarray = None) -> np.ndarray:
        if coords is None:
            if not self._nodes:
                return np.array([])
            coords = np.array([node.coordinates for node in self._nodes])
        points, _ = self.getQuadraturePoints()
        quad_coords = []
        for local_coords in points:
            N = self.getShapeFunctions(local_coords)
            quad_coords.append(N @ coords)
        return np.array(quad_coords)

    def getNumberOfQuadraturePoints(self) -> int:
        points, _ = self.getQuadraturePoints()
        return len(points)

    def getShapeFunctions(self, local_coords: np.ndarray) -> np.ndarray:
        """Evaluate standard shape functions at the given local coordinates."""
        el_type = self._elType.upper()
        if el_type == "CONLINE2":
            xi = local_coords[0]
            return np.array([0.5 * (1.0 - xi), 0.5 * (1.0 + xi)])
        elif el_type == "CONLINE3":
            xi = local_coords[0]
            return np.array([0.5 * xi * (xi - 1.0), 0.5 * xi * (xi + 1.0), 1.0 - xi * xi])
        elif el_type == "CONQUAD4":
            xi, eta = local_coords[0], local_coords[1]
            return np.array([
                0.25 * (1.0 - xi) * (1.0 - eta),
                0.25 * (1.0 + xi) * (1.0 - eta),
                0.25 * (1.0 + xi) * (1.0 + eta),
                0.25 * (1.0 - xi) * (1.0 + eta)
            ])
        elif el_type == "CONQUAD8":
            xi, eta = local_coords[0], local_coords[1]
            return np.array([
                0.25 * (1.0 - xi) * (1.0 - eta) * (-xi - eta - 1.0),
                0.25 * (1.0 + xi) * (1.0 - eta) * (xi - eta - 1.0),
                0.25 * (1.0 + xi) * (1.0 + eta) * (xi + eta - 1.0),
                0.25 * (1.0 - xi) * (1.0 + eta) * (-xi + eta - 1.0),
                0.5 * (1.0 - xi * xi) * (1.0 - eta),
                0.5 * (1.0 + xi) * (1.0 - eta * eta),
                0.5 * (1.0 - xi * xi) * (1.0 + eta),
                0.5 * (1.0 - xi) * (1.0 - eta * eta)
            ])
        elif el_type == "CONQUAD9":
            xi, eta = local_coords[0], local_coords[1]
            l0_xi, l1_xi, l2_xi = 0.5 * xi * (xi - 1.0), 1.0 - xi * xi, 0.5 * xi * (xi + 1.0)
            l0_eta, l1_eta, l2_eta = 0.5 * eta * (eta - 1.0), 1.0 - eta * eta, 0.5 * eta * (eta + 1.0)
            return np.array([
                l0_xi * l0_eta,
                l2_xi * l0_eta,
                l2_xi * l2_eta,
                l0_xi * l2_eta,
                l1_xi * l0_eta,
                l2_xi * l1_eta,
                l1_xi * l2_eta,
                l0_xi * l1_eta,
                l1_xi * l1_eta
            ])
        elif el_type == "CONTRI3":
            r, s = local_coords[0], local_coords[1]
            return np.array([1.0 - r - s, r, s])
        elif el_type == "CONTRI6":
            r, s = local_coords[0], local_coords[1]
            L1 = 1.0 - r - s
            return np.array([
                L1 * (2.0 * L1 - 1.0),
                r * (2.0 * r - 1.0),
                s * (2.0 * s - 1.0),
                4.0 * r * L1,
                4.0 * r * s,
                4.0 * s * L1
            ])
        else:
            raise NotImplementedError(f"Shape functions not defined for element type '{el_type}'")

    def getShapeFunctionDerivatives(self, local_coords: np.ndarray) -> np.ndarray:
        """Evaluate local derivatives of standard shape functions at the given local coordinates."""
        el_type = self._elType.upper()
        if el_type == "CONLINE2":
            return np.array([[-0.5, 0.5]])
        elif el_type == "CONLINE3":
            xi = local_coords[0]
            return np.array([[xi - 0.5, xi + 0.5, -2.0 * xi]])
        elif el_type == "CONQUAD4":
            xi, eta = local_coords[0], local_coords[1]
            return np.array([
                [
                    -0.25 * (1.0 - eta),
                     0.25 * (1.0 - eta),
                     0.25 * (1.0 + eta),
                    -0.25 * (1.0 + eta)
                ],
                [
                    -0.25 * (1.0 - xi),
                    -0.25 * (1.0 + xi),
                     0.25 * (1.0 + xi),
                     0.25 * (1.0 - xi)
                ]
            ])
        elif el_type == "CONQUAD8":
            xi, eta = local_coords[0], local_coords[1]
            return np.array([
                [
                    0.25 * (1.0 - eta) * (2.0 * xi + eta),
                    0.25 * (1.0 - eta) * (2.0 * xi - eta),
                    0.25 * (1.0 + eta) * (2.0 * xi + eta),
                    0.25 * (1.0 + eta) * (2.0 * xi - eta),
                    -xi * (1.0 - eta),
                    0.5 * (1.0 - eta * eta),
                    -xi * (1.0 + eta),
                    -0.5 * (1.0 - eta * eta)
                ],
                [
                    0.25 * (1.0 - xi) * (xi + 2.0 * eta),
                    0.25 * (1.0 + xi) * (-xi + 2.0 * eta),
                    0.25 * (1.0 + xi) * (xi + 2.0 * eta),
                    0.25 * (1.0 - xi) * (-xi + 2.0 * eta),
                    -0.5 * (1.0 - xi * xi),
                    -eta * (1.0 + xi),
                    0.5 * (1.0 - xi * xi),
                    -eta * (1.0 - xi)
                ]
            ])
        elif el_type == "CONQUAD9":
            xi, eta = local_coords[0], local_coords[1]
            l0_xi, l1_xi, l2_xi = 0.5 * xi * (xi - 1.0), 1.0 - xi * xi, 0.5 * xi * (xi + 1.0)
            l0_eta, l1_eta, l2_eta = 0.5 * eta * (eta - 1.0), 1.0 - eta * eta, 0.5 * eta * (eta + 1.0)
            dl0_xi, dl1_xi, dl2_xi = xi - 0.5, -2.0 * xi, xi + 0.5
            dl0_eta, dl1_eta, dl2_eta = eta - 0.5, -2.0 * eta, eta + 0.5
            return np.array([
                [
                    dl0_xi * l0_eta,
                    dl2_xi * l0_eta,
                    dl2_xi * l2_eta,
                    dl0_xi * l2_eta,
                    dl1_xi * l0_eta,
                    dl2_xi * l1_eta,
                    dl1_xi * l2_eta,
                    dl0_xi * l1_eta,
                    dl1_xi * l1_eta
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
                    l1_xi * dl1_eta
                ]
            ])
        elif el_type == "CONTRI3":
            return np.array([
                [-1.0, 1.0, 0.0],
                [-1.0, 0.0, 1.0]
            ])
        elif el_type == "CONTRI6":
            r, s = local_coords[0], local_coords[1]
            return np.array([
                [
                    4.0 * (r + s) - 3.0,
                    4.0 * r - 1.0,
                    0.0,
                    4.0 * (1.0 - 2.0 * r - s),
                    4.0 * s,
                    -4.0 * s
                ],
                [
                    4.0 * (r + s) - 3.0,
                    0.0,
                    4.0 * s - 1.0,
                    -4.0 * r,
                    4.0 * r,
                    4.0 * (1.0 - r - 2.0 * s)
                ]
            ])
        else:
            raise NotImplementedError(f"Shape function derivatives not defined for element type '{el_type}'")

    def getJacobianAndAreaWeight(self, local_coords: np.ndarray, coords: np.ndarray) -> float:
        """Compute the Jacobian determinant for the area mapping at a given local coordinate.
        
        coords: numpy array of shape (nNodes, dim) containing the current coordinates of the element's nodes.
        """
        dN = self.getShapeFunctionDerivatives(local_coords)
        t = dN @ coords  # shape: (nLocalDim, dim)
        
        if self.nLocalDim == 1:
            # 1D line in 2D space
            return float(np.linalg.norm(t[0]))
        else:
            # 2D surface (triangle or quad) in 3D space
            t1 = t[0]
            t2 = t[1]
            J = np.cross(t1, t2)
            return float(np.linalg.norm(J))

    def getBasisTransformation(self) -> np.ndarray:
        """Return the basis transformation matrix T_e for the construction of dual
        shape functions on second-order elements.

        For quadratic elements the weighted integrals int(N_a) dGamma of the corner
        node shape functions are negative or zero (e.g. exactly -1/12 * A for the
        corner nodes of an undistorted CONQUAD8), so the dual weights D_II would
        lose their meaning as positive area weights. Following Popp, Wohlmuth,
        Gee & Wall (2012) and Farah (2018), Sec. 6.2.3.2, shape function
        contributions of the mid-side nodes are shifted to their adjacent corner
        nodes with the factor alpha = 1/3:

            N_tilde_corner = N_corner + alpha * (N_adjacent mid-side nodes)
            N_tilde_mid    = (1 - 2*alpha) * N_mid

        This guarantees strictly positive integrals int(N_tilde_a) dGamma while
        preserving the partition of unity. For linear element types the identity
        matrix is returned.
        """
        n = self._nNodes
        T_e = np.eye(n)
        alpha = 1.0 / 3.0

        el_type = self._elType.upper()
        if el_type in ("CONQUAD8", "CONQUAD9"):
            mid_to_corners = {4: (0, 1), 5: (1, 2), 6: (2, 3), 7: (3, 0)}
        elif el_type == "CONTRI6":
            mid_to_corners = {3: (0, 1), 4: (1, 2), 5: (2, 0)}
        elif el_type == "CONLINE3":
            mid_to_corners = {2: (0, 1)}
        else:
            return T_e

        for mid, (c1, c2) in mid_to_corners.items():
            T_e[mid, mid] = 1.0 - 2.0 * alpha
            T_e[c1, mid] = alpha
            T_e[c2, mid] = alpha
        return T_e

    def computeLocalMassMatrices(self, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the transformed local mass matrix M_e, the diagonal D_e, and the dual
        coefficient matrix A_e, such that the dual shape functions are Phi = A_e * N.

        The biorthogonality is enforced with respect to the transformed basis
        N_tilde = T_e * N (identity transformation for linear elements):

            int(Phi_a * N_tilde_b) dGamma = delta_ab * int(N_tilde_a) dGamma

        with M_e[a,b] = int(N_tilde_a * N_tilde_b) dGamma and
        D_e[a,a] = int(N_tilde_a) dGamma > 0, so that
        A_e = D_e * inv(M_e) * T_e maps STANDARD shape function values N to the
        dual shape function values Phi (Popp et al. 2012; Farah 2018, Sec. 6.2.3).

        coords: numpy array of shape (nNodes, dim) containing the current coordinates of the element's nodes.
        """
        n = self._nNodes
        M_e = np.zeros((n, n))
        D_e = np.zeros((n, n))

        T_e = self.getBasisTransformation()
        points, weights = self.getQuadraturePoints()

        for local_coords, w in zip(points, weights):
            N_t = T_e @ self.getShapeFunctions(local_coords)
            jac = self.getJacobianAndAreaWeight(local_coords, coords)
            dGamma = jac * w

            M_e += np.outer(N_t, N_t) * dGamma
            for i in range(n):
                D_e[i, i] += N_t[i] * dGamma

        try:
            inv_M_e = np.linalg.inv(M_e)
        except np.linalg.LinAlgError:
            inv_M_e = np.linalg.pinv(M_e)

        A_e = D_e @ inv_M_e @ T_e
        return M_e, D_e, A_e

    def getResultArray(self, name: str, U: np.ndarray) -> np.ndarray:
        return np.array([])
