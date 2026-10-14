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
"""The gap of a point whose closest point lies on an edge between two master facets.

Both facets are then exactly equally close, the projection is not unique (Wriggers 2006, Remark 4.1
and Fig. 10.8), and with the facet normal each facet would give a gap of its own -- a choice that is
decided by rounding. Along the distance vector (Wriggers 2006, Eqs. 9.123 and 9.131) the gap is the
signed distance, the same for both facets and continuous across the edge. These tests pin that on a
valley of two inclined triangles, and that a flat surface and the surface boundary keep the facet
normal.
"""

import numpy as np
import pytest

from edelweissfe.utils.facetcontactgeometry import (
    closestFacetsWithTies,
    edgeOrVertexNormal,
    facetNormalAndMeasure,
    facetSurfaceBoundary,
    liesOnSurfaceBoundary,
)

tolerance = 1e-10


def _closest(point, order, facets):
    """The closest of the facets in ``order`` to ``point``, its weights, and whether it is a tie."""

    facet, weights, _, tied = closestFacetsWithTies(point[None], facets, [order], tolerance)
    return int(facet[0]), weights[0], bool(tied[0])


def _valley(slope: float) -> list:
    """Two triangles sharing the edge (0,0,0)-(0,1,0), rising with ``slope`` to either side, both
    with an upward normal."""

    left = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.5, slope]])
    right = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, slope], [0.0, 1.0, 0.0]])
    facets = []
    for facet in (left, right):
        normal, _ = facetNormalAndMeasure(facet)
        facets.append(facet if normal[2] > 0 else facet[[0, 2, 1]])
    return facets


def _gap(point, facets, order, edgeNormal=True):
    facet, weights, tied = _closest(point, order, facets)
    closestPoint = weights @ facets[facet]
    normal, _ = facetNormalAndMeasure(facets[facet])
    if tied and edgeNormal:
        normal = edgeOrVertexNormal(point, closestPoint, normal, tolerance)
    return normal @ (point - closestPoint), tied


def test_a_point_below_the_edge_is_equally_close_to_both_facets():
    facets = _valley(0.2)
    _, tied = _gap(np.array([0.02, 0.5, -0.3]), facets, [0, 1])
    assert tied


def test_the_gap_on_the_edge_does_not_depend_on_which_facet_is_chosen():
    facets = _valley(0.2)
    point = np.array([0.02, 0.5, -0.3])
    first, _ = _gap(point, facets, [0, 1])
    second, _ = _gap(point, facets, [1, 0])
    assert first == pytest.approx(second, abs=1e-15)
    assert first == pytest.approx(-np.linalg.norm(point - np.array([0.0, 0.5, 0.0])), abs=1e-15)

    # the facet normal is what made it depend on the choice
    first, _ = _gap(point, facets, [0, 1], edgeNormal=False)
    second, _ = _gap(point, facets, [1, 0], edgeNormal=False)
    assert abs(first - second) > 1e-3


def test_the_gap_is_continuous_across_the_edge():
    facets = _valley(0.2)
    xs = np.linspace(-0.3, 0.3, 601)
    gaps = np.array([_gap(np.array([x, 0.5, -0.3]), facets, [0, 1])[0] for x in xs])
    assert np.max(np.abs(np.diff(gaps))) < 2.0 * (xs[1] - xs[0])

    facetGaps = np.array([_gap(np.array([x, 0.5, -0.3]), facets, [0, 1], edgeNormal=False)[0] for x in xs])
    assert np.max(np.abs(np.diff(facetGaps))) > 10.0 * (xs[1] - xs[0]), "the counter-check lost its jump"


def test_a_flat_surface_keeps_the_facet_normal_bitwise():
    facets = _valley(0.0)
    point = np.array([0.0, 0.5, -0.3])
    _, weights, tied = _closest(point, [0, 1], facets)
    normal, _ = facetNormalAndMeasure(facets[0])
    assert tied
    assert edgeOrVertexNormal(point, weights @ facets[0], normal, tolerance) is normal


def test_the_surface_boundary_is_the_edges_of_one_facet():
    a, b, c, d = "a", "b", "c", "d"
    boundaryEdges, boundaryNodes = facetSurfaceBoundary([[a, b, c], [a, c, d]])
    assert frozenset((a, c)) not in boundaryEdges
    assert boundaryEdges == {frozenset(e) for e in [(a, b), (b, c), (c, d), (d, a)]}
    assert boundaryNodes == {a, b, c, d}

    boundary = (boundaryEdges, boundaryNodes)
    assert liesOnSurfaceBoundary([a, b, c], np.array([0.5, 0.5, 0.0]), boundary)
    assert not liesOnSurfaceBoundary([a, b, c], np.array([0.5, 0.0, 0.5]), boundary)
    assert liesOnSurfaceBoundary([a, b, c], np.array([0.0, 1.0, 0.0]), boundary)
    assert not liesOnSurfaceBoundary([a, b, c], np.array([0.2, 0.3, 0.5]), boundary)
