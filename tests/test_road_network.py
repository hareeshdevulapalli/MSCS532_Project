"""Tests for the road graph (Dijkstra, one-way and closed roads, speed bound)."""

import math
import unittest

from dispatch import RoadNetwork, build_grid_city


class RoadNetworkTests(unittest.TestCase):
    def test_shortest_path_takes_quicker_detour(self):
        graph = RoadNetwork()
        for name in "ABCD":
            graph.add_node(name, 0, 0)
        graph.add_street("A", "D", 100)
        graph.add_street("A", "B", 20)
        graph.add_street("B", "C", 20)
        graph.add_street("C", "D", 20)
        self.assertEqual(graph.shortest_path("A", "D"), (60, ["A", "B", "C", "D"]))

    def test_one_way_and_closed_streets(self):
        graph = RoadNetwork()
        for name in "AB":
            graph.add_node(name, 0, 0)
        graph.add_street("A", "B", 10, two_way=False)
        self.assertEqual(graph.shortest_path("A", "B")[0], 10)
        self.assertEqual(graph.shortest_path("B", "A"), (math.inf, []))
        graph.close_street("A", "B", two_way=False)
        self.assertEqual(graph.shortest_path("A", "B"), (math.inf, []))

    def test_bad_input(self):
        graph = RoadNetwork()
        graph.add_node("A", 0, 0)
        with self.assertRaises(KeyError):
            graph.add_street("A", "Z", 5)
        graph.add_node("B", 0, 0)
        with self.assertRaises(ValueError):
            graph.add_street("A", "B", -1)

    def test_times_to_target_agrees_with_shortest_path(self):
        city = build_grid_city(8, 8)
        starts = [(0, 0), (7, 7), (3, 5), (4, 4)]
        times = city.times_to_target((4, 4), starts)
        for start in starts:
            self.assertEqual(times[start], city.shortest_path(start, (4, 4))[0])
        self.assertEqual(set(city.times_to_target((4, 4), starts, limit=100)), {(4, 4), (3, 5)})

    # ----- added in Phase 2 -------------------------------------------

    def test_max_speed_and_lower_bound(self):
        city = build_grid_city(3, 3, block_m=200.0, block_secs=20.0)
        self.assertAlmostEqual(city.max_speed, 10.0)
        self.assertAlmostEqual(city.lower_bound_time(500.0), 50.0)
        self.assertEqual(city.lower_bound_time(0), 0.0)
        city.add_street((0, 0), (1, 0), 100.0)          # slowing a road keeps the old top speed
        self.assertAlmostEqual(city.max_speed, 10.0)
        city.add_street((0, 0), (1, 0), 10.0)           # a faster road raises it
        self.assertAlmostEqual(city.max_speed, 20.0)

    def test_bound_never_exceeds_true_driving_time(self):
        city = build_grid_city(6, 6)
        city.add_street((0, 0), (5, 5), 60.0, two_way=False)   # a fast diagonal highway
        for start in city.nodes():
            for end in [(5, 5), (2, 3)]:
                seconds, _ = city.shortest_path(start, end)
                sx, sy = city.coords(start)
                ex, ey = city.coords(end)
                self.assertLessEqual(city.lower_bound_time(math.hypot(ex - sx, ey - sy)),
                                     seconds + 1e-9)

    def test_zero_time_street_and_empty_network_bounds(self):
        graph = RoadNetwork()
        self.assertEqual(graph.lower_bound_time(100), math.inf)   # no roads at all
        graph.add_node("A", 0, 0)
        graph.add_node("B", 100, 0)
        graph.add_street("A", "B", 0)
        self.assertEqual(graph.lower_bound_time(100), 0.0)

    def test_missing_streets_and_nodes_raise(self):
        city = build_grid_city(2, 2)
        with self.assertRaises(KeyError):
            city.close_street((0, 0), (1, 1))
        with self.assertRaises(KeyError):
            city.shortest_path((0, 0), (9, 9))
        with self.assertRaises(KeyError):
            city.times_to_target((9, 9), [(0, 0)])
        with self.assertRaises(KeyError):
            city.add_node((0, 0), 0, 0)
        self.assertEqual(city.street_time((0, 0), (1, 0)), 30.0)


if __name__ == "__main__":
    unittest.main()
