from __future__ import annotations

import unittest

import numpy as np
from shapely.geometry import Point

from tools.rooms.room_screening.geometry import (
    classify_category,
    conservative_projected_footprint,
    projected_unmapped_surface_union,
    shape_preserving_projected_footprint,
)


class FurnitureObstacleGeometryTests(unittest.TestCase):
    def test_category_families(self):
        self.assertEqual(classify_category("dining table"), "blocker")
        self.assertEqual(classify_category("l-shaped sofa"), "blocker")
        self.assertEqual(classify_category("armchair"), "blocker")
        self.assertEqual(classify_category("sunbed"), "blocker")
        self.assertEqual(classify_category("sun lounger"), "blocker")
        self.assertEqual(classify_category("recliner"), "blocker")
        self.assertEqual(classify_category("chaise lounge"), "blocker")
        self.assertEqual(classify_category("daybed"), "blocker")
        self.assertEqual(classify_category("rug"), "ground")
        self.assertEqual(classify_category("shower floor"), "ground")
        self.assertEqual(classify_category("bathroom floor"), "ground")
        self.assertEqual(classify_category("bath mat"), "ground")
        self.assertEqual(classify_category("shower mat"), "ground")
        self.assertEqual(classify_category("wall"), "structural")
        self.assertEqual(classify_category("ceiling"), "structural")
        self.assertEqual(classify_category("decorative object"), "review")

    def test_mesh_outside_agent_height_has_no_footprint(self):
        triangle = np.asarray([
            [[0, 1.6, 0], [1, 1.6, 0], [0, 1.6, 1]],
        ], dtype=float)
        result = conservative_projected_footprint(triangle, floor_y=0, agent_height=1.5)
        self.assertTrue(result.is_empty)

    def test_triangle_clips_at_agent_height(self):
        triangle = np.asarray([
            [[0, 0.5, 0], [2, 1.8, 0], [0, 0.5, 2]],
        ], dtype=float)
        result = conservative_projected_footprint(triangle, floor_y=0, agent_height=1.5)
        self.assertAlmostEqual(result.bounds[2], 1.53846153846, places=6)
        self.assertAlmostEqual(result.bounds[3], 2.0, places=6)
        self.assertGreater(result.area, 0)

    def test_convex_hull_fills_missing_surface_triangles(self):
        # Two separated facets from one semantic instance still yield a single
        # conservative footprint, avoiding holes caused by incomplete scans.
        triangles = np.asarray([
            [[0, 0.5, 0], [1, 0.5, 0], [0, 0.5, 1]],
            [[2, 0.5, 2], [3, 0.5, 2], [3, 0.5, 3]],
        ], dtype=float)
        result = conservative_projected_footprint(triangles, floor_y=0, agent_height=1.5)
        self.assertEqual(result.geom_type, "Polygon")
        self.assertAlmostEqual(result.area, 4.0)

    def test_tabletop_perimeter_mesh_blocks_its_interior(self):
        # A scanned tabletop may only retain perimeter strips or separated
        # facets. Its instance hull must still cover the tabletop interior.
        triangles = np.asarray([
            [[0, 0.75, 0], [2, 0.75, 0], [0, 0.75, 0.1]],
            [[2, 0.75, 0], [2, 0.75, 0.1], [0, 0.75, 0.1]],
            [[0, 0.75, 1.9], [2, 0.75, 1.9], [0, 0.75, 2]],
            [[2, 0.75, 1.9], [2, 0.75, 2], [0, 0.75, 2]],
            [[0, 0.75, 0], [0.1, 0.75, 0], [0, 0.75, 2]],
            [[1.9, 0.75, 0], [2, 0.75, 0], [2, 0.75, 2]],
        ], dtype=float)
        result = conservative_projected_footprint(triangles, floor_y=0, agent_height=1.5)
        self.assertAlmostEqual(result.area, 4.0)
        self.assertTrue(result.covers(Point(1.0, 1.0)))

    def test_shape_preserving_footprint_keeps_l_shaped_floor_out_of_blocker(self):
        # Projected vertices form an L-shaped object rather than a filled square.
        triangles = np.asarray([
            [[0, 0.5, 0], [2, 0.5, 0], [2, 0.5, 1]],
            [[0, 0.5, 0], [2, 0.5, 1], [0, 0.5, 1]],
            [[0, 0.5, 1], [1, 0.5, 1], [1, 0.5, 2]],
            [[0, 0.5, 1], [1, 0.5, 2], [0, 0.5, 2]],
        ], dtype=float)
        result = shape_preserving_projected_footprint(triangles, floor_y=0, agent_height=1.5)
        self.assertLess(result.area, conservative_projected_footprint(triangles, 0, 1.5).area)
        self.assertTrue(result.covers(Point(0.5, 1.5)))
        self.assertFalse(result.covers(Point(1.5, 1.5)))

    def test_shape_preserving_footprint_still_fills_tabletop_interior(self):
        triangles = np.asarray([
            [[0, 0.75, 0], [2, 0.75, 0], [0, 0.75, 0.1]],
            [[2, 0.75, 0], [2, 0.75, 0.1], [0, 0.75, 0.1]],
            [[0, 0.75, 1.9], [2, 0.75, 1.9], [0, 0.75, 2]],
            [[2, 0.75, 1.9], [2, 0.75, 2], [0, 0.75, 2]],
            [[0, 0.75, 0], [0.1, 0.75, 0], [0, 0.75, 2]],
            [[1.9, 0.75, 0], [2, 0.75, 0], [2, 0.75, 2]],
        ], dtype=float)
        result = shape_preserving_projected_footprint(triangles, floor_y=0, agent_height=1.5)
        self.assertTrue(result.covers(Point(1.0, 1.0)))

    def test_unmapped_faces_are_not_joined_into_one_scene_hull(self):
        triangles = np.asarray([
            [[0, 0.5, 0], [1, 0.5, 0], [1, 0.5, 1]],
            [[0, 0.5, 0], [1, 0.5, 1], [0, 0.5, 1]],
            [[3, 0.5, 0], [4, 0.5, 0], [4, 0.5, 1]],
            [[3, 0.5, 0], [4, 0.5, 1], [3, 0.5, 1]],
        ], dtype=float)
        result = projected_unmapped_surface_union(triangles, floor_y=0, agent_height=1.5)
        self.assertAlmostEqual(result.area, 2.0)
        self.assertFalse(result.covers(Point(2.0, 0.5)))


if __name__ == "__main__":
    unittest.main()
