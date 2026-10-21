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
"""Contact surfaces given as whole element faces (``CON*`` contact elements, from a mesh generator or
``facets=wholeFace``) are usable by every contact constraint: the node-based and the integrated
constraint cut them into exactly the facets the surface element generator cuts from the same faces,
and the integrated constraint integrates either input the way its ``integration`` option says."""

import unittest

import numpy as np
from _hexa20cube import twoBlockModel  # noqa: E402  (tests/ is on sys.path)

import edelweissfe.utils.inputfileparser  # noqa: F401 bootstrap input language
from edelweissfe.constraints.nodetodeformablesurfacepenalty import (
    Constraint as NodeBasedContact,
)
from edelweissfe.constraints.nodetodeformablesurfacepenalty import (
    NodeToDeformableSurfacePenaltySchema,
)
from edelweissfe.constraints.surfacetodeformablesurfacepenalty import (
    Constraint as IntegratedContact,
)
from edelweissfe.constraints.surfacetodeformablesurfacepenalty import (
    SurfaceToDeformableSurfacePenaltySchema,
)
from edelweissfe.generators.surfaceelementgenerator import (
    _FACE_TABLES,
    _MIDSIDE_FACE_TABLES,
    _WHOLE_FACE_FACETS,
    asFacets,
    canonicalParentFace,
)
from edelweissfe.journal.journal import Journal


def _mapping(x):
    """A curved and distorted interface, so that nothing agrees by symmetry alone."""

    return np.array([x[0] + 0.1 * x[0] * (2.0 - x[0]), x[1] + 0.2 * (x[0] - 1.0) ** 2, x[2] * (1.0 + 0.05 * x[0])])


class TestFacetsOfWholeFaces(unittest.TestCase):
    def test_every_face_of_an_element_type_is_cut_alike(self):
        """The canonical facet pattern is derived from one face of each table; every other face of the
        same element type must give the same one, or a whole face would be cut differently from the
        solid face it stands for."""

        for table, ensightType, faceType in (
            (_FACE_TABLES, "hexa8", "quad4"),
            (_MIDSIDE_FACE_TABLES, "hexa20", "quad8"),
            (_FACE_TABLES, "quad4", "line2"),
            (_MIDSIDE_FACE_TABLES, "quad8", "line3"),
        ):
            for faceNumber, groups in table[ensightType].items():
                _, canonicalIndices = canonicalParentFace(ensightType, faceNumber)
                position = {localIndex: k for k, localIndex in enumerate(canonicalIndices)}
                pattern = tuple(tuple(position[i] for i in group) for group in groups)
                self.assertEqual(pattern, _WHOLE_FACE_FACETS[faceType], (ensightType, faceNumber))

    def test_cut_facets_equal_the_generators(self):
        _, generated, _ = twoBlockModel(Journal(), 0.0, mapping=_mapping)
        _, wholeFaces, _ = twoBlockModel(Journal(), 0.0, mapping=_mapping, facets="wholeFace")
        cut = asFacets(list(wholeFaces), "s")
        self.assertEqual(len(cut), len(generated))
        for a, b in zip(generated, cut):
            self.assertEqual([n.label for n in a.nodes], [n.label for n in b.nodes])
            self.assertEqual([n.label for n in a.parentFaceNodes], [n.label for n in b.parentFaceNodes])
            self.assertEqual(a.parentFaceType, b.parentFaceType)
            np.testing.assert_array_equal(a.vertexParametricCoords, b.vertexParametricCoords)
            np.testing.assert_array_equal(a.nodalAreaShares, b.nodalAreaShares)

    def test_mixed_surfaces_are_rejected(self):
        _, generated, _ = twoBlockModel(Journal(), 0.0)
        _, wholeFaces, _ = twoBlockModel(Journal(), 0.0, facets="wholeFace")
        with self.assertRaisesRegex(ValueError, "mixes whole element faces and facets"):
            asFacets(list(generated) + list(wholeFaces), "mixed")


class TestSameResultWhateverTheInput(unittest.TestCase):
    """The same model with its contact surfaces given as generator facets or as whole faces."""

    def _models(self):
        return (
            twoBlockModel(Journal(), 0.03, mapping=_mapping),
            twoBlockModel(Journal(), 0.03, mapping=_mapping, facets="wholeFace"),
        )

    @staticmethod
    def _evaluate(constraint, model):
        constraint.updateConnectivity(model)
        PExt = np.zeros(constraint.nDof)
        constraint.applyConstraint(np.zeros(constraint.nDof), np.zeros(constraint.nDof), PExt, None, None)
        return dict(zip([n.label for n in constraint.slaveSurfaceNodes], constraint.getSlaveNodalNormalForces()))

    def test_node_based_contact(self):
        results = []
        for model, slave, master in self._models():
            configuration = NodeToDeformableSurfacePenaltySchema(penalty=1e4, searchDistance=1.0, sliding="small")
            constraint = NodeBasedContact("c", model, slave, master, Journal(), configuration=configuration)
            constraint.updateConnectivity(model)
            PExt = np.zeros(constraint.nDof)
            constraint.applyConstraint(np.zeros(constraint.nDof), np.zeros(constraint.nDof), PExt, None, None)
            results.append(
                (dict(zip([n.label for n in constraint.slaveNodes], constraint.getNormalPressures())), constraint)
            )
        (facetPressures, facetConstraint), (wholePressures, wholeConstraint) = results
        self.assertEqual(facetPressures, wholePressures)
        self.assertGreater(max(facetPressures.values()), 0.0, "no contact -- test is vacuous")
        self.assertEqual(facetConstraint.totalNormalForce, wholeConstraint.totalNormalForce)

    def test_integrated_contact(self):
        for integration in ("triangles", "wholeFace"):
            results = []
            for model, slave, master in self._models():
                configuration = SurfaceToDeformableSurfacePenaltySchema(
                    penalty=1e4, searchDistance=1.0, integration=integration
                )
                constraint = IntegratedContact("c", model, slave, master, Journal(), configuration=configuration)
                results.append((self._evaluate(constraint, model), constraint.getNormalPressures()))
            (facetForces, facetPressures), (wholeForces, wholePressures) = results
            self.assertGreater(facetPressures.max(), 0.0, "no contact -- test is vacuous")
            # The slave integration is identical; the master search differs (flat facets or a
            # sampling of the faces) and converges to the same closest points.
            np.testing.assert_allclose(wholePressures, facetPressures, rtol=1e-9, err_msg=integration)
            for label, force in facetForces.items():
                np.testing.assert_allclose(wholeForces[label], force, rtol=1e-9, atol=1e-12, err_msg=integration)


if __name__ == "__main__":
    unittest.main()
