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
"""
The contact points of an integrated (Gauss-point-to-segment) contact formulation: quadrature points
on the curved parent faces of a slave surface.

Shared by every constraint that integrates contact over a slave surface, whatever the master side
is -- a deformable surface or a rigid body. See
:mod:`~edelweissfe.constraints.surfacetodeformablesurfacepenalty` for why the pressure must be
distributed with the *parent* face's shape functions.
"""

import numpy as np

from edelweissfe.models.femodel import FEModel
from edelweissfe.models.modelchange import ModelChange
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.utils.facetcontactgeometry import facetNormalAndMeasure
from edelweissfe.utils.meshtools import currentNodeCoordinates
from edelweissfe.utils.parentfacegeometry import (
    PARENT_FACE_LOCAL_DIMENSION,
    facetQuadratureRule,
    parentFaceAreaElement,
    parentFaceGaussRule,
    parentFaceShapeFunctions,
    parentFaceShapeFunctionsAt,
    parentFaceTangents,
)


def checkFacetsCarryParentFaces(facets: list, side: str, setName: str, constraintName: str):
    """Reject contact facets that cannot be used with parent-face shape functions.

    Parameters
    ----------
    facets
        The contact facet elements.
    side
        'slave' or 'master', for the error message.
    setName
        The name of the facet element set, for the error message.
    constraintName
        The name of the constraint, for the error message.
    """

    unstamped = [facet.elNumber for facet in facets if facet.parentFaceType is None]
    if unstamped:
        raise ValueError(
            f"Constraint '{constraintName}': {side} facet {unstamped[0]} carries no parent face. "
            "The integrated contact formulation distributes the contact pressure with the "
            "parent element face's shape functions, so the facets must come from the "
            "surface element generator, which stamps them."
        )

    # A per-node weighting has no meaning in this formulation and, unlike the node-based
    # constraint, there is no code path that could apply one: the pressure is distributed with
    # the parent face's own shape functions, evaluated at the quadrature points. Accepting a
    # surface stamped with a weight transform would silently give a different answer from the
    # sibling constraint on the same input, which is the one outcome worth refusing outright --
    # the whole point of the parent-face basis is that it fixes the corner mismatch the
    # weighting only minimises.
    weighted = [facet.elNumber for facet in facets if facet.weightTransform is not None]
    if weighted:
        raise ValueError(
            f"Constraint '{constraintName}': {side} surface '{setName}' was generated with "
            f"nodalWeights='serendipityOptimal' (facet {weighted[0]} and "
            f"{len(weighted) - 1} more carry a weight transform), which this constraint "
            "cannot honour -- it distributes the contact pressure with the parent face's "
            "shape functions, not with per-node weights, and needs no corner reweighting "
            "to begin with. Generate these facets with the default "
            "nodalWeights='facetConsistent'."
        )


def groupFacetsByParentFace(facets: list) -> dict[tuple, list]:
    """The facets grouped by their parent element face, in first-encounter order.

    Parameters
    ----------
    facets
        Contact facet elements carrying a parent face.

    Returns
    -------
    dict[tuple, list]
        Per parent face, keyed by its nodes in canonical order, the facets cut from it.
    """

    groups = {}
    for facet in facets:
        groups.setdefault(tuple(facet.parentFaceNodes), []).append(facet)
    return groups


def checkFacetsTileParentFace(faceFacets: list, setName: str, constraintName: str):
    """Reject a parent face that the facets of a surface cover only in part.

    Integrating over the whole parent face is only right if the surface contains all of it. The
    check is made in the face's parametric domain, where the facets' parametric areas (lengths) sum
    to that of the domain exactly, however the face is curved.

    Parameters
    ----------
    faceFacets
        The facets of one parent face.
    setName
        The name of the facet element set, for the error message.
    constraintName
        The name of the constraint, for the error message.
    """

    face = faceFacets[0]
    localDimension = PARENT_FACE_LOCAL_DIMENSION[face.parentFaceType]
    covered = 0.0
    for facet in faceFacets:
        Xi = np.asarray(facet.vertexParametricCoords, dtype=float)
        if localDimension == 2:
            e1, e2 = Xi[1] - Xi[0], Xi[2] - Xi[0]
            covered += 0.5 * abs(e1[0] * e2[1] - e1[1] * e2[0])
        else:
            covered += abs(Xi[1, 0] - Xi[0, 0])
    domain = 2.0**localDimension
    if abs(covered - domain) > 1e-10 * domain:
        raise ValueError(
            f"Constraint '{constraintName}': the facets of surface '{setName}' cover only "
            f"{covered / domain:.1%} of the parent face of facet {face.elNumber}; "
            "quadrature='parentFace' integrates over whole parent faces."
        )


#: The ways of integrating over the slave surface, see :class:`ContactPointsOnSlaveSurface`.
SLAVE_QUADRATURES = ("facet", "parentFace")


class ContactPointsOnSlaveSurface:
    """The quadrature points of a slave surface, used as contact points.

    The points are grouped into *integration cells*, whose kind is set by ``quadrature``:

    * ``'facet'``: each slave facet is a cell and carries an ``nQuadraturePoints``-point rule. A
      point at facet-barycentric coordinates :math:`b` sits at parent-face parametric coordinates
      :math:`\\xi = b \\cdot \\Xi` (with :math:`\\Xi` the facet vertices' own parametric coordinates
      in the parent face, stamped by the surface element generator), and carries the constant weight
      :math:`J_q = A_{\\text{facet}}^{\\text{ref}} w_q`. This is exact where the parent face's map is
      affine on each facet.
    * ``'parentFace'``: each parent element face is a cell and carries the native Gauss-Legendre rule
      of ``gaussPointsPerDirection`` points per direction, with the weight
      :math:`J_q = w_q \\, |X_{,\\xi} \\times X_{,\\eta}|(\\xi_q)` (3D) or :math:`w_q |X_{,\\xi}|(\\xi_q)`
      (2D) from the parent face's own surface Jacobian in the reference configuration -- Gauss points
      defined on the slave segments, as in Wriggers 2006, Section 9.5.3, p. 257 and Fig. 9.14. The
      facets then play no part in the integration; they only identify the surface.

    Either way, a point's position follows the *curved* parent surface, :math:`x_q = N^s(\\xi_q)
    \\cdot x^s`, and the contact points are numbered by cell and, within a cell, by quadrature point.

    Parameters
    ----------
    slaveSurface
        The element set of contact facet elements forming the slave surface.
    nQuadraturePoints
        The number of quadrature points per slave facet, for ``quadrature='facet'``.
    constraintName
        The name of the owning constraint, for error messages.
    quadrature
        One of :data:`SLAVE_QUADRATURES`.
    gaussPointsPerDirection
        The number of Gauss points per parametric direction, for ``quadrature='parentFace'``.

    Attributes
    ----------
    facets : list
        The slave contact facet elements.
    quadrature : str
        The kind of integration cell, see above.
    nCells : int
        The number of integration cells.
    pointsPerCell : int
        The number of quadrature points per cell.
    cellKeys : list[tuple]
        Per cell, the nodes identifying it across a retiling: the facet's nodes, or the parent face's.
    cellFaceTypes : list[str]
        Per cell, its parent face type.
    parentNodes : list[list]
        Per cell, the nodes of its parent element face.
    parentReferenceCoordinates : list[numpy.ndarray]
        Per cell, the reference coordinates of its parent face nodes.
    parametricCoordinates : list[numpy.ndarray]
        Per cell, the parent-face parametric coordinates of its quadrature points.
    shapeFunctions : list[numpy.ndarray]
        Per cell, the parent-face shape functions at each of its quadrature points, of shape
        (pointsPerCell, nParentNodes). Constant, since the parametric locations are fixed.
    integrationWeights : list[numpy.ndarray]
        Per cell, the integration weight of each of its quadrature points in the reference
        configuration (consistent with the small-deformation setting, as the node-based
        constraint's tributary areas are).
    facetMeasure : float
        The summed reference measure of the slave facets, whatever the cells -- the basis of a
        characteristic facet size that does not depend on the quadrature.
    pointCell : numpy.ndarray
        The cell index of each contact point.
    pointQuadraturePoint : numpy.ndarray
        The quadrature point index of each contact point within its cell.
    nPoints : int
        The number of contact points.
    surfaceNodes : list
        The unique nodes of the slave facets, in the same first-encounter order the surface element
        generator uses for its '<prefix>_nodes' node set -- so a fromExpression field output over
        that set lines up with :meth:`nodalNormalForces`.
    """

    def __init__(
        self,
        slaveSurface: ElementSet,
        nQuadraturePoints: int,
        constraintName: str,
        quadrature: str = "facet",
        gaussPointsPerDirection: int = 3,
    ):
        self.setName = slaveSurface.name
        self.facets = list(slaveSurface)
        if quadrature not in SLAVE_QUADRATURES:
            raise ValueError(
                f"Constraint '{constraintName}': unknown quadrature '{quadrature}'; "
                f"choose one of {SLAVE_QUADRATURES}."
            )
        self.quadrature = quadrature

        checkFacetsCarryParentFaces(self.facets, "slave", self.setName, constraintName)

        facetMeasures = []
        for facet in self.facets:
            _, measure = facetNormalAndMeasure(np.array([n.coordinates for n in facet.nodes]))
            facetMeasures.append(measure)
        _, facetWeights = facetQuadratureRule(self.facets[0].nNodes, nQuadraturePoints) if self.facets else (0, [])
        self.facetMeasure = float(sum(np.sum(measure * facetWeights) for measure in facetMeasures))

        self.cellKeys = []
        self.cellFaceTypes = []
        self.parentNodes = []
        self.parentReferenceCoordinates = []
        self.parametricCoordinates = []
        self.shapeFunctions = []
        self.integrationWeights = []

        if quadrature == "facet":
            self.pointsPerCell = nQuadraturePoints
            for facet, measure in zip(self.facets, facetMeasures):
                barycentric, weights = facetQuadratureRule(facet.nNodes, nQuadraturePoints)
                parametric = barycentric @ facet.vertexParametricCoords
                self._addCell(
                    tuple(facet.nodes),
                    facet,
                    parametric,
                    np.array([parentFaceShapeFunctions(facet.parentFaceType, xi) for xi in parametric]),
                    measure * weights,
                )
        else:
            self.pointsPerCell = None
            for faceFacets in groupFacetsByParentFace(self.facets).values():
                face = faceFacets[0]
                if face.parentFaceType not in PARENT_FACE_LOCAL_DIMENSION:
                    raise ValueError(
                        f"Constraint '{constraintName}': quadrature='parentFace' has no native rule for "
                        f"parent face type '{face.parentFaceType}' (slave facet {face.elNumber})."
                    )
                checkFacetsTileParentFace(faceFacets, self.setName, constraintName)
                parametric, gaussWeights = parentFaceGaussRule(face.parentFaceType, gaussPointsPerDirection)
                referenceCoordinates = np.array([n.coordinates for n in face.parentFaceNodes])
                tangents = parentFaceTangents(face.parentFaceType, parametric, referenceCoordinates)
                self._addCell(
                    tuple(face.parentFaceNodes),
                    face,
                    parametric,
                    parentFaceShapeFunctionsAt(face.parentFaceType, parametric),
                    gaussWeights * parentFaceAreaElement(tangents),
                )
                self.pointsPerCell = len(gaussWeights)
            if self.pointsPerCell is None:
                self.pointsPerCell = 0

        self.nCells = len(self.cellKeys)
        self.pointCell = np.repeat(np.arange(self.nCells), self.pointsPerCell)
        self.pointQuadraturePoint = np.tile(np.arange(self.pointsPerCell), self.nCells)
        self.nPoints = len(self.pointCell)

        self.surfaceNodes = list(dict.fromkeys(node for facet in self.facets for node in facet.nodes))

    def _addCell(self, key: tuple, facet, parametric: np.ndarray, shapeFunctions: np.ndarray, weights: np.ndarray):
        """Append one integration cell, whose parent face is that of ``facet``."""

        self.cellKeys.append(key)
        self.cellFaceTypes.append(facet.parentFaceType)
        self.parentNodes.append(list(facet.parentFaceNodes))
        self.parentReferenceCoordinates.append(np.array([n.coordinates for n in facet.parentFaceNodes]))
        self.parametricCoordinates.append(parametric)
        self.shapeFunctions.append(shapeFunctions)
        self.integrationWeights.append(weights)

    def referencePointCoordinates(self) -> np.ndarray:
        """The reference positions of all contact points, of shape (nPoints, nDim)."""

        if not self.nCells:
            nDim = len(self.facets[0].nodes[0].coordinates) if self.facets else 0
            return np.zeros((0, nDim))
        return np.concatenate([N @ X for N, X in zip(self.shapeFunctions, self.parentReferenceCoordinates)])

    def allParentNodes(self) -> set:
        """The set of all parent face nodes of the slave surface."""

        return {node for nodes in self.parentNodes for node in nodes}

    def currentPointCoordinates(self, model: FEModel) -> np.ndarray:
        """The current positions of all contact points, on the *curved* slave parent surface.

        Parameters
        ----------
        model
            The model tree, holding the displacement field.

        Returns
        -------
        numpy.ndarray
            The coordinates, of shape (nPoints, nDim).
        """

        nDim = model.domainSize
        coords = np.empty((self.nPoints, nDim))
        for c, nodes in enumerate(self.parentNodes):
            currentCoords = currentNodeCoordinates(nodes, model, self.parentReferenceCoordinates[c])
            first = c * self.pointsPerCell
            coords[first : first + self.pointsPerCell] = self.shapeFunctions[c] @ currentCoords
        return coords

    def normalPressures(self, normalForces: np.ndarray) -> np.ndarray:
        """Convert the normal forces of the contact points into pressures (positive in compression).

        Parameters
        ----------
        normalForces
            The normal force of each contact point (negative in compression).

        Returns
        -------
        numpy.ndarray
            The normal pressure of each contact point.
        """

        weights = np.concatenate(self.integrationWeights) if self.nPoints else np.zeros(0)
        return np.divide(-normalForces, weights, out=np.zeros_like(normalForces), where=weights > 0)

    def nodalNormalForces(self, normalForces: np.ndarray) -> np.ndarray:
        """Distribute the normal forces of the contact points to the slave surface nodes.

        Positive means repelled; **negative means pulled**, which is exactly the tensile corner load
        a serendipity face requires and a node-based scheme cannot produce. Ordered like
        :attr:`surfaceNodes`. A parent-face node that is not itself a facet node -- possible only
        with ``triangulation=corner`` -- receives contact force but has no slot there, and is
        omitted.

        Parameters
        ----------
        normalForces
            The normal force of each contact point (negative in compression, zero if inactive).

        Returns
        -------
        numpy.ndarray
            The normal force at each slave surface node.
        """

        forceOfNode = dict.fromkeys(self.surfaceNodes, 0.0)
        for p in range(self.nPoints):
            if normalForces[p] == 0.0:
                continue
            c = self.pointCell[p]
            for node, shapeFunction in zip(self.parentNodes[c], self.shapeFunctions[c][self.pointQuadraturePoint[p]]):
                if node in forceOfNode:
                    forceOfNode[node] += -normalForces[p] * shapeFunction
        return np.array([forceOfNode[node] for node in self.surfaceNodes])

    def isTouchedBy(self, model: FEModel, change: ModelChange) -> bool:
        """Whether a mesh change touched the source elements of the slave surface, so the contact
        points must be rebuilt from the regenerated facets.

        Parameters
        ----------
        model
            The model tree, holding the recipes of the generated contact facet sets.
        change
            The mesh change.

        Returns
        -------
        bool
            True if the slave surface must be rebuilt.
        """

        recipe = model.contactFacetRecipes.get(self.setName)
        return recipe is not None and change.touchesSurface(recipe[0])
