"""Tests for the uniform location grid (k-nearest search, updates, edge cases)."""

import math
import random
import unittest

from dispatch import SpatialGrid


class SpatialGridTests(unittest.TestCase):
    def nearest_by_scanning_all(self, points, x, y, k, max_distance=math.inf):
        ranked = sorted((math.hypot(px - x, py - y), name) for name, (px, py) in points.items())
        return [entry for entry in ranked if entry[0] <= max_distance][:k]

    def test_empty_grid(self):
        self.assertEqual(SpatialGrid(10).nearest(0, 0, k=3), [])

    def test_rejects_bad_cell_size_and_duplicates(self):
        with self.assertRaises(ValueError):
            SpatialGrid(0)
        grid = SpatialGrid(10)
        grid.insert("a", 1, 1)
        with self.assertRaises(KeyError):
            grid.insert("a", 2, 2)

    def test_agrees_with_full_scan_on_random_points(self):
        rng = random.Random(7)
        for cell_size in (5, 50, 500):
            grid = SpatialGrid(cell_size)
            points = {}
            for n in range(300):
                points[f"p{n}"] = (rng.uniform(-1000, 1000), rng.uniform(-1000, 1000))
                grid.insert(f"p{n}", *points[f"p{n}"])
            for _ in range(50):
                x, y = rng.uniform(-1200, 1200), rng.uniform(-1200, 1200)
                k = rng.randint(1, 10)
                with self.subTest(cell_size=cell_size, x=x, y=y, k=k):
                    self.assertEqual(grid.nearest(x, y, k),
                                     self.nearest_by_scanning_all(points, x, y, k))

    def test_max_distance(self):
        grid = SpatialGrid(10)
        grid.insert("near", 3, 4)
        grid.insert("far", 300, 400)
        self.assertEqual(grid.nearest(0, 0, k=5, max_distance=10), [(5.0, "near")])

    def test_move_and_delete(self):
        grid = SpatialGrid(10)
        grid.insert("a", 0, 0)
        grid.insert("b", 100, 100)
        grid.move("a", 95, 95)
        self.assertEqual(grid.nearest(100, 100, k=1)[0][1], "b")
        grid.delete("b")
        self.assertEqual(grid.nearest(100, 100, k=1)[0][1], "a")
        grid.delete("a")
        self.assertEqual(len(grid), 0)
        self.assertEqual(grid._cells, {})  # empty cells are dropped

    # ----- added in Phase 2 -------------------------------------------

    def test_k_bigger_than_grid_and_negative_coords(self):
        grid = SpatialGrid(50)
        grid.insert("west", -120, -30)
        grid.insert("east", 80, 10)
        self.assertEqual([name for _, name in grid.nearest(-100, -30, k=10)], ["west", "east"])
        self.assertEqual(grid.nearest(0, 0, k=0), [])

    def test_move_inside_same_cell_keeps_one_cell(self):
        grid = SpatialGrid(100)
        grid.insert("a", 10, 10)
        grid.move("a", 90, 90)
        self.assertEqual(grid._cells, {(0, 0): {"a"}})
        self.assertEqual(grid.coords_of("a"), (90, 90))

    def test_missing_id_errors(self):
        grid = SpatialGrid(10)
        with self.assertRaises(KeyError):
            grid.delete("ghost")
        with self.assertRaises(KeyError):
            grid.move("ghost", 1, 1)
        self.assertNotIn("ghost", grid)

    def test_random_moves_still_agree_with_full_scan(self):
        rng = random.Random(3)
        grid, points = SpatialGrid(40), {}
        for n in range(200):
            points[n] = (rng.uniform(0, 800), rng.uniform(0, 800))
            grid.insert(n, *points[n])
        for _ in range(2000):
            n = rng.randrange(200)
            points[n] = (rng.uniform(0, 800), rng.uniform(0, 800))
            grid.move(n, *points[n])
        for _ in range(30):
            x, y = rng.uniform(0, 800), rng.uniform(0, 800)
            self.assertEqual(grid.nearest(x, y, 5), self.nearest_by_scanning_all(points, x, y, 5))


if __name__ == "__main__":
    unittest.main()
