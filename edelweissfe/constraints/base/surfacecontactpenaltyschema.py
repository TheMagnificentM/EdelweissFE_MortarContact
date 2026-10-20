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
The options shared by the integrated (surface-to-surface) penalty contact constraints.
"""

from dataclasses import dataclass

from edelweissfe.utils.schema import schemaField


@dataclass(frozen=True)
class SurfaceContactPenaltySchema:
    """The options of the slave side and the penalty law, shared by
    :mod:`~edelweissfe.constraints.surfacetodeformablesurfacepenalty` and
    :mod:`~edelweissfe.constraints.surfacetodiscreterigidbodypenalty`, which add their master side.

    The update-type option is spelled ``type`` in the input file but the field is named
    ``contactType`` here -- a dataclass field literally called ``type`` would shadow the builtin,
    which this project's conventions avoid. ``penalty`` is an interface stiffness modulus per unit
    area, multiplied by each contact point's quadrature weight.
    """

    slaveSurface: str | None = schemaField(
        description="The element set of contact facet elements (Tria3ContactFacet/Line2ContactFacet) "
        "forming the slave surface; contact is integrated over these facets at quadrature points. "
        "The facets must come from the surface element generator, which stamps their parent faces.",
        dtype=str,
        default=None,
        required=True,
    )
    penalty: float | None = schemaField(
        description="The numerical penalty value, an interface stiffness modulus per unit slave " "surface area.",
        dtype=float,
        default=None,
        required=True,
    )
    contactType: str = schemaField(
        description="The formulation type: 'linear' (linear force, constant stiffness with jump) "
        "or 'quadratic' (quadratic force, linear stiffness).",
        dtype=str,
        default="linear",
        optionName="type",
    )
    quadrature: str = schemaField(
        description="How the slave surface is integrated: 'facet' (an nQuadraturePoints-point rule on "
        "every flat facet, weighted with the facet's reference area), 'subcells' (the same points, the "
        "facets taken as sub-cells of the parent face and weighted with the parent face's own surface "
        "Jacobian; a composite rule, exact for curved and distorted faces up to the order of the facet "
        "rule) or 'parentFace' (the native "
        "Gauss rule of gaussPointsPerDirection points per direction on every parent element face, "
        "weighted with the face's own surface Jacobian; exact on curved and distorted faces up to "
        "the order of the rule, and independent of the triangulation).",
        dtype=str,
        default="facet",
    )
    nQuadraturePoints: int = schemaField(
        description="For quadrature='facet' and 'subcells', the number of quadrature points per slave "
        "facet: 1, 3 or "
        "6 for a Tria3 facet (3D), 1, 2 or 3 for a Line2 facet (2D). The 3-point rule is exact to "
        "degree 2 only: on a single facet of a serendipity face it does not integrate the cubic "
        "terms of the shape functions exactly, and over a whole flat, undistorted face the errors "
        "cancel, which is why the consistent nodal loads -- including the negative corner loads -- "
        "still come out exactly there. With the facet's constant reference area as the weight, a "
        "facet rule is exact only where the parent face's map is affine on each facet.",
        dtype=int,
        default=3,
    )
    gaussPointsPerDirection: int = schemaField(
        description="For quadrature='parentFace', the number of Gauss points per parametric direction "
        "of a parent face (1 to 4): n x n on a quadrilateral face, n on an edge face. 2 integrates "
        "the consistent nodal loads of a constant pressure exactly on a flat parallelogram face; 3, "
        "the default, also covers distorted and curved faces to higher order.",
        dtype=int,
        default=3,
    )
    searchDistance: float | None = schemaField(
        description="An optional broadphase distance for the per-increment candidate-facet search. "
        "If not given, every contact point is always assigned its single closest master facet, "
        "without a distance gate.",
        dtype=float,
        default=None,
    )
    sliding: str = schemaField(
        description="The kinematic treatment of the contact geometry. Only 'small' (Abaqus-style "
        "small sliding: the closest-point projection -- master facet, parametric location and "
        "normal -- is frozen at every contact search from the last converged configuration, making the "
        "gap linear in the displacement DOFs) is implemented; 'finite' is rejected.",
        dtype=str,
        default="small",
    )
