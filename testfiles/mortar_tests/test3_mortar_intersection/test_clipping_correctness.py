import os
import unittest
import numpy as np
from edelweissfe.constraints.utils.intersection import (
    project_point_to_plane,
    sutherland_hodgman_clip,
    triangulate_polygon,
    to_plane_coords,
    to_3d_coords,
    get_tangent_basis
)

class TestMortarIntersection(unittest.TestCase):
    def test_full_overlap(self):
        # Master and Slave match exactly: [0, 1]x[0, 1]
        slave = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        master = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        clip = sutherland_hodgman_clip(master, slave)
        self.assertEqual(len(clip), 4)
        for i in range(4):
            np.testing.assert_allclose(clip[i], slave[i], atol=1e-12)

    def test_no_overlap(self):
        # Master shifted completely out of slave
        slave = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        master = np.array([[2.0, 2.0], [3.0, 2.0], [3.0, 3.0], [2.0, 3.0]])
        clip = sutherland_hodgman_clip(master, slave)
        self.assertEqual(len(clip), 0)

    def test_partial_overlap_half(self):
        # Master covers right half of slave: [0.5, 1.5]x[0, 1]
        slave = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        master = np.array([[0.5, 0.0], [1.5, 0.0], [1.5, 1.0], [0.5, 1.0]])
        clip = sutherland_hodgman_clip(master, slave)
        self.assertEqual(len(clip), 4)
        expected = np.array([[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]])
        # Find corresponding vertices (due to start point alignment)
        for p in expected:
            dists = np.linalg.norm(clip - p, axis=1)
            self.assertTrue(np.any(dists < 1e-12))

    def test_triangulation_area(self):
        # Triangulate a pentagon of known area
        pentagon = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.5], [0.0, 1.0]])
        triangles = triangulate_polygon(pentagon)
        self.assertEqual(len(triangles), 3)
        total_area = 0.0
        for tri in triangles:
            v1 = tri[1] - tri[0]
            v2 = tri[2] - tri[0]
            area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
            total_area += area
        self.assertAlmostEqual(total_area, 1.25)

if __name__ == '__main__':
    unittest.main()
