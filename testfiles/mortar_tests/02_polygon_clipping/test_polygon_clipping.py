#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test 2: Polygon-Clipping-Korrektheit (Sutherland--Hodgman)
==========================================================

Verifiziert die geometrische Verschneidung zweier projizierter Facetten-Polygone
(Sutherland & Hodgman 1974): voller Überlapp, kein Überlapp, halber Überlapp und
verdreht/verzerrt. Grundlage der segmentbasierten Mortar-Integration.

Die visuelle Inspektion (flache/gekrümmte Projektion, VTK-Export für ParaView)
liegt im Unterordner ``visualization/``.
"""

import os
import unittest
import numpy as np
from edelweissfe.constraints.mortar_geom_utils import (
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
        print("\n--- Test Full Overlap ---")
        print("Slave nodes:\n", slave)
        print("Master nodes:\n", master)
        print("Clipped intersection nodes:\n", clip)
        self.assertEqual(len(clip), 4)
        for i in range(4):
            np.testing.assert_allclose(clip[i], slave[i], atol=1e-12)

    def test_no_overlap(self):
        # Master shifted completely out of slave
        slave = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        master = np.array([[2.0, 2.0], [3.0, 2.0], [3.0, 3.0], [2.0, 3.0]])
        clip = sutherland_hodgman_clip(master, slave)
        print("\n--- Test No Overlap ---")
        print("Slave nodes:\n", slave)
        print("Master nodes (shifted):\n", master)
        print("Clipped intersection nodes:\n", clip)
        self.assertEqual(len(clip), 0)

    def test_partial_overlap_half(self):
        # Master covers right half of slave: [0.5, 1.5]x[0, 1]
        slave = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        master = np.array([[0.5, 0.0], [1.5, 0.0], [1.5, 1.0], [0.5, 1.0]])
        clip = sutherland_hodgman_clip(master, slave)
        print("\n--- Test Partial Overlap (Right Half) ---")
        print("Slave nodes:\n", slave)
        print("Master nodes:\n", master)
        print("Clipped intersection nodes:\n", clip)
        self.assertEqual(len(clip), 4)
        expected = np.array([[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]])
        # Find corresponding vertices (due to start point alignment)
        for p in expected:
            dists = np.linalg.norm(clip - p, axis=1)
            self.assertTrue(np.any(dists < 1e-12))

    def test_skewed_rotated_overlap(self):
        # Slave element: Quad centered on a plane [0, 1]x[0, 1]
        slave = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        # Master element: Quad rotated by 45 degrees around center [0.5, 0.5]
        # and scaled slightly
        # Center is [0.5, 0.5], half-diagonal length ~ 0.5 * sqrt(2) * scale
        c = 0.5
        s = 0.8
        # Rotated coordinates of square
        master = np.array([
            [c - s * np.sqrt(2)/2, c],
            [c, c - s * np.sqrt(2)/2],
            [c + s * np.sqrt(2)/2, c],
            [c, c + s * np.sqrt(2)/2]
        ])
        
        # Clip
        clip = sutherland_hodgman_clip(master, slave)
        triangles = triangulate_polygon(clip)
        
        # Area via triangulation
        tri_area = 0.0
        for tri in triangles:
            v1 = tri[1] - tri[0]
            v2 = tri[2] - tri[0]
            tri_area += 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
            
        # Analytical area using Shoelace formula directly on clipped polygon
        x = clip[:, 0]
        y = clip[:, 1]
        shoelace_area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        
        print("\n--- Test Skewed Rotated Overlap ---")
        print("Slave nodes:\n", slave)
        print("Master nodes (rotated 45deg):\n", master)
        print("Clipped polygon vertices:\n", clip)
        print("Triangulated Area:", tri_area)
        print("Shoelace Area:", shoelace_area)
        
        self.assertAlmostEqual(tri_area, shoelace_area, places=12)
        # Ensure clipping produces a non-zero overlap area
        self.assertGreater(tri_area, 0.1)

    def test_triangulation_area(self):
        # Triangulate a pentagon of known area
        pentagon = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.5], [0.0, 1.0]])
        triangles = triangulate_polygon(pentagon)
        print("\n--- Test Triangulation Area ---")
        print("Pentagon coordinates:\n", pentagon)
        print("Triangulation sub-triangles (cells):\n", triangles)
        self.assertEqual(len(triangles), 3)
        total_area = 0.0
        for tri in triangles:
            v1 = tri[1] - tri[0]
            v2 = tri[2] - tri[0]
            area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
            total_area += area
        print("Summed area of sub-triangles:", total_area)
        self.assertAlmostEqual(total_area, 1.25)

    def test_nonconvex_clip_polygon_loses_area(self):
        """Sutherland-Hodgman setzt ein KONVEXES Clip-Polygon voraus.

        Der Algorithmus schneidet nacheinander gegen die HALBEBENE jeder
        Clip-Kante. Bei einem einspringenden (reflexen) Eckpunkt schneidet die
        zugehörige Halbebene echtes Gebiet weg, das zum Polygon gehört -- ohne
        Fehler, ohne Meldung. Dasselbe gilt für ``triangulate_polygon``, das als
        Dreiecksfächer ab Ecke 0 ebenfalls Konvexität voraussetzt.

        Dieser Test hält den Verlust FEST, statt ihn zu behaupten: er ist die
        dokumentierte Voraussetzung des Verfahrens, nicht ein Fehler der
        Implementierung. Praktisch relevant wird er nur, wenn eine Sub-Zelle
        einer quadratischen Facette nicht-konvex wird -- wie weit die Knoten
        dafür wandern müssen, misst 05_quadratic_segmentation.
        """
        # Mittel-Quad [4,5,6,7] eines CONQUAD8, dessen unterer Mittelknoten über
        # den Elementmittelpunkt hinaus verschoben wurde: Ecke 0 ist reflex.
        nonconvex = np.array([[0.5, 0.75], [1.0, 0.5], [0.5, 1.0], [0.0, 0.5]])
        covering = np.array([[-2.0, -2.0], [3.0, -2.0], [3.0, 3.0], [-2.0, 3.0]])

        def shoelace(p):
            x, y = p[:, 0], p[:, 1]
            return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

        true_area = shoelace(nonconvex)
        clip = sutherland_hodgman_clip(covering, nonconvex)
        clipped_area = shoelace(clip) if len(clip) >= 3 else 0.0

        print("\n--- Test Non-Convex Clip Polygon ---")
        print("Non-convex clip polygon:\n", nonconvex)
        print("True area (shoelace):", true_area)
        print("Clipped area:", clipped_area)
        print("Lost fraction:", 1.0 - clipped_area / true_area)

        self.assertAlmostEqual(true_area, 0.125, places=12)
        # Der Verlust ist gross, nicht marginal -- deshalb ist die Voraussetzung
        # kein Detail: hier gehen zwei Drittel der Flaeche verloren.
        self.assertLess(clipped_area, 0.5 * true_area)

        # Gegenprobe: dasselbe Polygon konvex gemacht -> exakt erhalten.
        convex = np.array([[0.5, 0.0], [1.0, 0.5], [0.5, 1.0], [0.0, 0.5]])
        clip_c = sutherland_hodgman_clip(covering, convex)
        self.assertAlmostEqual(shoelace(clip_c), shoelace(convex), places=12)


if __name__ == '__main__':
    unittest.main()
