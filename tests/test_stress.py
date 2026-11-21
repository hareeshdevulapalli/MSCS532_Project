"""Stress tests and odd input.

By default the sizes are chosen so the whole file runs in a few seconds.
Set STRESS=1 to run the same tests ten times bigger:

    STRESS=1 python3 -m unittest tests.test_stress -v
"""

import math
import os
import random
import time
import unittest

from dispatch import (Dispatcher, IndexedMinHeap, RoadNetwork, SpatialGrid, add_river_barrier,
                      build_grid_city)
from dispatch.simulation import SimSettings, simulate

SCALE = 10 if os.environ.get("STRESS") else 1


class LargeInputTests(unittest.TestCase):
    def test_grid_with_many_drivers(self):
        rng = random.Random(1)
        n = 20_000 * SCALE
        grid = SpatialGrid(250)
        for i in range(n):
            grid.insert(i, rng.uniform(0, 30_000), rng.uniform(0, 30_000))
        start = time.perf_counter()
        for _ in range(500):
            x, y = rng.uniform(0, 30_000), rng.uniform(0, 30_000)
            result = grid.nearest(x, y, k=8)
            self.assertEqual(len(result), 8)
            self.assertEqual(result, sorted(result))
        per_query = (time.perf_counter() - start) / 500
        self.assertLess(per_query, 0.005)   # well under 5 ms even with 200k points

    def test_heap_with_many_operations(self):
        rng = random.Random(2)
        heap, mirror = IndexedMinHeap(), {}
        for step in range(50_000 * SCALE):
            roll = rng.random()
            if roll < 0.5 or not mirror:
                prio = (rng.randint(0, 2), rng.random())
                heap.push(step, prio)
                mirror[step] = prio
            elif roll < 0.7:
                key, prio = heap.pop()
                if step % 997 == 0:     # checking every pop would make the test O(n^2)
                    self.assertEqual(prio, min(mirror.values()))
                del mirror[key]
            elif roll < 0.85:
                key = next(iter(mirror))
                heap.remove(key)
                del mirror[key]
            else:
                key = next(iter(mirror))
                mirror[key] = (rng.randint(0, 2), rng.random())
                heap.change_priority(key, mirror[key])
        drained = [heap.pop()[1] for _ in range(len(heap))]
        self.assertEqual(drained, sorted(mirror.values()))

    def test_big_city_search(self):
        side = 120 if SCALE == 1 else 300
        city = build_grid_city(side, side)
        seconds, path = city.shortest_path((0, 0), (side - 1, side - 1))
        self.assertEqual(seconds, 2 * (side - 1) * 30)
        self.assertEqual(len(path), 2 * side - 1)

    def test_dispatcher_with_thousands_of_drivers_and_requests(self):
        rng = random.Random(3)
        city = build_grid_city(80, 80)
        dispatcher = Dispatcher(city, cell_size=400.0, pickup_limit=900, promote_after=120)
        nodes = list(city.nodes())
        for n in range(2_000 * SCALE):
            dispatcher.add_driver(f"D{n}", rng.choice(nodes))
        for n in range(3_000 * SCALE):
            dispatcher.new_request(f"R{n}", rng.choice(nodes), rng.choice(nodes),
                                   rng.choice(["standard", "premium"]), float(n % 600))
        matches = dispatcher.run_round(now=600.0)
        self.assertEqual(len(matches), 2_000 * SCALE)          # every driver used
        self.assertEqual(dispatcher.count_free_drivers(), 0)
        self.assertEqual(dispatcher.count_waiting(), 1_000 * SCALE)
        self.assertTrue(all(m.pickup_time <= 900 for m in matches))

    def test_large_simulation_finishes(self):
        report = simulate(SimSettings(cols=60, rows=60, drivers=500 * SCALE,
                                      requests=4_000 * SCALE, hotspot_share=0.3,
                                      incidents=50, shift_changes=50, keep_matches=False))
        self.assertEqual(report.served + report.cancelled + report.still_waiting,
                         4_000 * SCALE)


class ExtremeConditionTests(unittest.TestCase):
    def test_every_driver_on_one_corner(self):
        dispatcher = Dispatcher(build_grid_city(10, 10))
        for n in range(500):
            dispatcher.add_driver(f"D{n:03d}", (5, 5))
        dispatcher.new_request("R", (5, 6), (0, 0), "standard", 0.0)
        match = dispatcher.assign_top()
        self.assertEqual((match.driver_id, match.pickup_time), ("D000", 30))

    def test_every_request_on_one_corner(self):
        dispatcher = Dispatcher(build_grid_city(10, 10), pickup_limit=10_000)
        for n in range(50):
            dispatcher.add_driver(f"D{n}", (n % 10, n // 10))
        for n in range(200):
            dispatcher.new_request(f"R{n:03d}", (0, 0), (9, 9), "standard", float(n))
        matches = dispatcher.run_round(now=300.0)
        self.assertEqual(len(matches), 50)
        times = [m.pickup_time for m in matches]
        self.assertEqual(times, sorted(times))     # nearest drivers go first

    def test_island_nobody_can_reach(self):
        city = build_grid_city(12, 12)
        add_river_barrier(city, west_col=5, rows=12, bridge_rows=set())
        dispatcher = Dispatcher(city, pickup_limit=100_000)
        for n in range(30):
            dispatcher.add_driver(f"D{n}", (n % 6, n // 6))
        for n in range(300):
            dispatcher.new_request(f"R{n}", (11, n % 12), (11, 0), "standard", 0.0)
        for round_no in range(20):
            self.assertEqual(dispatcher.run_round(now=float(round_no)), [])
        # 12 different pickups, each searched once; every retry is a cache hit
        self.assertLessEqual(dispatcher.stats["searches"], 12)
        self.assertEqual(dispatcher.count_waiting(), 300)

    def test_all_streets_closed(self):
        city = build_grid_city(4, 4)
        for c in range(4):
            for r in range(4):
                if c + 1 < 4:
                    city.close_street((c, r), (c + 1, r))
                if r + 1 < 4:
                    city.close_street((c, r), (c, r + 1))
        dispatcher = Dispatcher(city)
        dispatcher.add_driver("D", (0, 0))
        dispatcher.new_request("same_spot", (0, 0), (1, 1), "standard", 0.0)
        dispatcher.new_request("elsewhere", (3, 3), (1, 1), "premium", 0.0)
        matches = dispatcher.run_round(now=1.0)
        self.assertEqual([(m.request_id, m.pickup_time) for m in matches], [("same_spot", 0)])

    def test_zero_second_streets(self):
        city = build_grid_city(5, 5)
        city.add_street((0, 0), (1, 0), 0)          # teleport-fast street
        self.assertEqual(city.max_speed, math.inf)
        self.assertEqual(city.lower_bound_time(1000), 0.0)
        self.assertEqual(city.shortest_path((0, 0), (1, 0))[0], 0)
        dispatcher = Dispatcher(city)
        dispatcher.add_driver("B", (1, 0))
        dispatcher.add_driver("A", (0, 0))
        dispatcher.new_request("R", (0, 0), (4, 4), "standard", 0.0)
        self.assertEqual(dispatcher.assign_top().driver_id, "A")   # tie at 0 s -> smallest id

    def test_huge_and_negative_coordinates(self):
        grid = SpatialGrid(1.0)
        grid.insert("far", 1e12, -1e12)
        grid.insert("near", -3.0, 4.0)
        self.assertEqual(grid.nearest(0, 0, k=1), [(5.0, "near")])
        self.assertEqual(grid.nearest(1e12, -1e12, k=1)[0][1], "far")

    def test_cancel_storm(self):
        dispatcher = Dispatcher(build_grid_city(5, 5), promote_after=10)
        for n in range(10_000):
            dispatcher.new_request(f"R{n}", (0, 0), (4, 4), "standard", float(n))
        for n in range(0, 10_000, 2):
            dispatcher.cancel_request(f"R{n}")
        dispatcher.age_requests(20_000.0)
        self.assertEqual(dispatcher.count_waiting(), 5_000)
        self.assertEqual(set(dispatcher._queue.all_keys()), set(dispatcher.waiting_requests))

    def test_rapid_gps_updates(self):
        rng = random.Random(6)
        city = build_grid_city(20, 20)
        dispatcher = Dispatcher(city, cell_size=300.0)
        nodes = list(city.nodes())
        for n in range(200):
            dispatcher.add_driver(f"D{n}", rng.choice(nodes))
        for _ in range(20_000):
            dispatcher.move_driver(f"D{rng.randrange(200)}", rng.choice(nodes))
        for spot, ids in dispatcher._free_at.items():
            for driver_id in ids:
                self.assertEqual(dispatcher.drivers[driver_id].location, spot)
        self.assertEqual(sum(map(len, dispatcher._free_at.values())), 200)


class UnexpectedInputTests(unittest.TestCase):
    def test_nan_and_infinite_coordinates_are_rejected(self):
        grid = SpatialGrid(10)
        for bad in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                grid.insert("p", bad, 0)
        self.assertEqual(len(grid), 0)                # nothing half-inserted
        grid.insert("p", 1, 1)
        with self.assertRaises(ValueError):
            grid.move("p", 0, math.nan)
        self.assertEqual(grid.coords_of("p"), (1, 1))  # old position kept
        with self.assertRaises(ValueError):
            grid.nearest(math.nan, 0)

    def test_bad_cell_sizes(self):
        for bad in (0, -1, math.nan, math.inf, "10"):
            with self.assertRaises(ValueError):
                SpatialGrid(bad)

    def test_bad_road_values(self):
        city = RoadNetwork()
        with self.assertRaises(ValueError):
            city.add_node("a", math.nan, 0)
        city.add_node("a", 0, 0)
        city.add_node("b", 1, 0)
        for bad in (math.nan, math.inf, -1):
            with self.assertRaises(ValueError):
                city.add_street("a", "b", bad)
        with self.assertRaises(ValueError):
            RoadNetwork(route_cache_size=-1)

    def test_bad_request_time(self):
        dispatcher = Dispatcher(build_grid_city(3, 3))
        with self.assertRaises(ValueError):
            dispatcher.new_request("R", (0, 0), (1, 1), "standard", math.nan)
        self.assertEqual(dispatcher.count_waiting(), 0)

    def test_bad_dispatcher_settings(self):
        city = build_grid_city(3, 3)
        for bad in ({"pickup_limit": math.nan}, {"promote_after": 0}, {"cell_size": -5}):
            with self.subTest(**bad), self.assertRaises(ValueError):
                Dispatcher(city, **bad)

    def test_empty_structures(self):
        self.assertEqual(SpatialGrid(5).nearest(0, 0, k=10), [])
        heap = IndexedMinHeap()
        heap.push_many([])
        with self.assertRaises(IndexError):
            heap.pop()
        dispatcher = Dispatcher(build_grid_city(2, 2))
        self.assertEqual(dispatcher.run_round(now=0.0), [])
        self.assertEqual(dispatcher.age_requests(1e9), 0)
        report = simulate(SimSettings(requests=0, drivers=0))
        self.assertEqual(report.served, 0)

    def test_negative_simulation_sizes(self):
        with self.assertRaises(ValueError):
            simulate(SimSettings(requests=-1))


if __name__ == "__main__":
    unittest.main()
