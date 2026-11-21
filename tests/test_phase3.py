"""Phase 3: the optimized structures must give the same answers as Phase 2.

Each test runs the new code and the frozen Phase 2 code (``dispatch.phase2``)
or a brute-force answer key side by side on random input.
"""

import heapq
import math
import random
import unittest

from dispatch import (Dispatcher, IndexedMinHeap, RoadNetwork, SpatialGrid, add_river_barrier,
                      build_grid_city, phase2)
from dispatch.dispatcher import slow_best_time
from dispatch.simulation import SimSettings, arrival_times, simulate


def random_city(rng, cols=8, rows=8, network_cls=RoadNetwork, closures=25):
    city = build_grid_city(cols, rows, network_cls=network_cls)
    for _ in range(closures):
        col, row = rng.randrange(cols - 1), rng.randrange(rows)
        try:
            if rng.random() < 0.6:
                city.close_street((col, row), (col + 1, row))
            else:
                city.add_street((col, row), (col + 1, row), rng.choice([10, 60, 120, 300]))
        except KeyError:
            pass
    return city


class GridEquivalenceTests(unittest.TestCase):
    def test_same_answers_as_phase2_under_churn(self):
        rng = random.Random(31)
        for cell_size in (3, 40, 400, 5000):
            new, old, points = SpatialGrid(cell_size), phase2.SpatialGrid(cell_size), {}
            for step in range(2500):
                roll = rng.random()
                if roll < 0.4 or not points:
                    name = f"p{step}"
                    points[name] = (rng.uniform(-2000, 2000), rng.uniform(-2000, 2000))
                    new.insert(name, *points[name])
                    old.insert(name, *points[name])
                elif roll < 0.6:
                    name = rng.choice(list(points))
                    del points[name]
                    new.delete(name)
                    old.delete(name)
                elif roll < 0.8:
                    name = rng.choice(list(points))
                    points[name] = (rng.uniform(-2000, 2000), rng.uniform(-2000, 2000))
                    new.move(name, *points[name])
                    old.move(name, *points[name])
                else:
                    x, y = rng.uniform(-3000, 3000), rng.uniform(-3000, 3000)
                    k = rng.randint(1, 12)
                    cap = rng.choice([math.inf, 100.0, 800.0])
                    with self.subTest(cell_size=cell_size, step=step):
                        self.assertEqual(new.nearest(x, y, k, cap), old.nearest(x, y, k, cap))

    def test_bounding_box_always_covers_every_cell(self):
        rng = random.Random(4)
        grid = SpatialGrid(10)
        for n in range(500):
            grid.insert(n, rng.uniform(-500, 500), rng.uniform(-500, 500))
        for n in range(480):
            grid.delete(n)
            box = grid._box
            for col, row in grid._cells:
                self.assertTrue(box[0] <= col <= box[1] and box[2] <= row <= box[3])

    def test_within_matches_brute_force(self):
        rng = random.Random(17)
        for cell_size in (5, 100, 2000):
            grid, points = SpatialGrid(cell_size), {}
            for n in range(400):
                points[n] = (rng.uniform(-1000, 1000), rng.uniform(-1000, 1000))
                grid.insert(n, *points[n])
            for _ in range(60):
                x, y, r = rng.uniform(-1200, 1200), rng.uniform(-1200, 1200), rng.uniform(0, 900)
                expected = {n for n, (px, py) in points.items() if math.hypot(px - x, py - y) <= r}
                self.assertEqual(set(grid.within(x, y, r)), expected)
        self.assertEqual(SpatialGrid(5).within(0, 0, 10), [])

    def test_ties_go_to_smallest_id(self):
        grid = SpatialGrid(1)
        for name in ("c", "a", "b"):
            grid.insert(name, 5, 0)
        grid.insert("z", 100, 100)
        self.assertEqual([i for _, i in grid.nearest(0, 0, k=2)], ["a", "b"])


class HeapEquivalenceTests(unittest.TestCase):
    def test_same_output_as_phase2_heap(self):
        rng = random.Random(5)
        new, old = IndexedMinHeap(), phase2.IndexedMinHeap()
        live = set()
        for step in range(6000):
            roll = rng.random()
            if roll < 0.35:
                key = f"k{step}"
                prio = (rng.randint(0, 3), rng.randint(0, 50))
                new.push(key, prio)
                old.push(key, prio)
                live.add(key)
            elif roll < 0.5 and live:
                key, _ = new.pop()
                self.assertEqual((key, _), old.pop())
                live.discard(key)
            elif roll < 0.7 and live:
                key = rng.choice(sorted(live))
                new.remove(key)
                old.remove(key)
                live.discard(key)
            elif roll < 0.9 and live:
                key = rng.choice(sorted(live))
                prio = (rng.randint(0, 3), rng.randint(0, 50))
                new.change_priority(key, prio)
                old.change_priority(key, prio)
            elif live:
                self.assertEqual(new.peek(), old.peek())
            self.assertEqual(len(new), len(old))
        while old:
            self.assertEqual(new.pop(), old.pop())

    def test_push_many_matches_single_pushes(self):
        rng = random.Random(8)
        for start, batch in ((0, 50), (40, 5), (10, 200)):
            bulk, single = IndexedMinHeap(), IndexedMinHeap()
            for n in range(start):
                prio = (rng.random(),)
                bulk.push(f"s{n}", prio)
                single.push(f"s{n}", prio)
            items = [(f"b{n}", (rng.random(),)) for n in range(batch)]
            bulk.push_many(items)
            for key, prio in items:
                single.push(key, prio)
            out_bulk = [bulk.pop() for _ in range(len(bulk))]
            out_single = [single.pop() for _ in range(len(single))]
            self.assertEqual(out_bulk, out_single)

    def test_push_many_rejects_duplicates(self):
        heap = IndexedMinHeap()
        heap.push("a", 1)
        with self.assertRaises(KeyError):
            heap.push_many([("a", 2)])
        with self.assertRaises(KeyError):
            heap.push_many([("b", 2), ("b", 3)])
        self.assertEqual(len(heap), 1)


class RoadSearchTests(unittest.TestCase):
    def test_astar_matches_phase2_dijkstra(self):
        rng = random.Random(77)
        for trial in range(40):
            seed = rng.random()
            new = random_city(random.Random(seed))
            old = random_city(random.Random(seed), network_cls=phase2.RoadNetwork)
            nodes = list(new.nodes())
            for _ in range(10):
                a, b = rng.choice(nodes), rng.choice(nodes)
                with self.subTest(trial=trial, a=a, b=b):
                    t_new, path = new.shortest_path(a, b)
                    t_old, _ = old.shortest_path(a, b)
                    self.assertEqual(t_new, t_old)
                    if path:   # the path really takes that long
                        self.assertEqual(sum(new.street_time(p, q) for p, q in zip(path, path[1:])),
                                         t_new)

    def test_astar_settles_fewer_nodes_than_dijkstra(self):
        city = build_grid_city(40, 40)
        city.shortest_path((0, 0), (39, 0))
        astar = city.last_settled
        self.assertLess(astar, 100)    # plain Dijkstra settles ~800 nodes here

    def test_nearest_source_matches_brute_force(self):
        rng = random.Random(12)
        for trial in range(60):
            city = random_city(rng)
            nodes = list(city.nodes())
            sources = set(rng.sample(nodes, rng.randint(1, 6)))
            target = rng.choice(nodes)
            limit = rng.choice([math.inf, 200, 500])
            best, spots = city.nearest_source(target, sources, limit)
            times = {s: city.shortest_path(s, target)[0] for s in sources}
            in_range = {s: t for s, t in times.items() if t <= limit}
            with self.subTest(trial=trial):
                if not in_range:
                    self.assertIsNone(best)
                else:
                    self.assertEqual(best, min(in_range.values()))
                    self.assertEqual(set(spots), {s for s, t in in_range.items() if t == best})

    def test_route_cache_is_cleared_when_a_street_changes(self):
        city = build_grid_city(5, 1)
        self.assertEqual(city.travel_time((0, 0), (4, 0)), 120)
        self.assertEqual(city.travel_time((0, 0), (4, 0)), 120)
        self.assertEqual((city.cache_hits, city.cache_misses), (1, 1))
        city.close_street((2, 0), (3, 0))
        self.assertEqual(city.travel_time((0, 0), (4, 0)), math.inf)
        city.add_street((2, 0), (3, 0), 5)
        self.assertEqual(city.travel_time((0, 0), (4, 0)), 95)

    def test_route_cache_has_a_size_limit(self):
        city = build_grid_city(6, 6, network_cls=lambda: RoadNetwork(route_cache_size=10))
        nodes = list(city.nodes())
        for a in nodes:
            city.travel_time(a, (0, 0))
        self.assertEqual(len(city._route_cache), 10)


class LandmarkTests(unittest.TestCase):
    def test_landmark_astar_is_exact_on_random_cities(self):
        rng = random.Random(55)
        for trial in range(25):
            seed = rng.random()
            city = random_city(random.Random(seed), 10, 10, closures=35)
            plain = random_city(random.Random(seed), 10, 10, network_cls=phase2.RoadNetwork,
                                closures=35)
            city.prepare_landmarks(rng.randint(1, 6))
            nodes = list(city.nodes())
            for _ in range(15):
                a, b = rng.choice(nodes), rng.choice(nodes)
                with self.subTest(trial=trial, a=a, b=b):
                    self.assertEqual(city.shortest_path(a, b)[0], plain.shortest_path(a, b)[0])

    def test_landmarks_stay_exact_through_closures_and_reopenings(self):
        rng = random.Random(66)
        city = build_grid_city(12, 12)
        plain = phase2.build_grid_city(12, 12)
        city.prepare_landmarks(4)
        closed = []
        nodes = list(city.nodes())
        for step in range(300):
            if rng.random() < 0.6 or not closed:
                a = rng.choice(nodes)
                options = [b for b in city._out_edges[a] if a in city._out_edges[b]]
                if options:
                    b = rng.choice(options)
                    closed.append((a, b, city.street_time(a, b)))
                    city.close_street(a, b)
                    plain.close_street(a, b)
            else:
                a, b, secs = closed.pop(rng.randrange(len(closed)))
                slower = secs + rng.choice([0, 30])
                city.add_street(a, b, slower)
                plain.add_street(a, b, slower)
            x, y = rng.choice(nodes), rng.choice(nodes)
            self.assertEqual(city.shortest_path(x, y)[0], plain.shortest_path(x, y)[0])
        self.assertTrue(city.has_landmarks)     # nothing ever got faster

    def test_a_faster_street_switches_landmarks_off(self):
        city = build_grid_city(6, 6)
        city.prepare_landmarks(3)
        city.add_street((0, 0), (1, 0), 30)     # same time: still fine
        self.assertTrue(city.has_landmarks)
        city.add_street((0, 0), (1, 0), 5)      # faster: tables could now overestimate
        self.assertFalse(city.has_landmarks)
        self.assertEqual(city.shortest_path((0, 0), (5, 0))[0], 125)

    def test_landmarks_settle_fewer_nodes(self):
        city = build_grid_city(60, 60)
        add_river_barrier(city, 29, 60, bridge_rows={15, 45})
        city.shortest_path((28, 30), (31, 30))
        plain = city.last_settled
        city.prepare_landmarks(8)
        self.assertEqual(city.shortest_path((28, 30), (31, 30))[0], 32 * 30 - 30 * 2 + 30 * 3)
        self.assertLess(city.last_settled * 2, plain)   # 173 vs 407 here

    def test_landmark_memory_is_reported(self):
        city = build_grid_city(10, 10)
        self.assertEqual(city.landmark_memory(), 0)
        city.prepare_landmarks(4)
        self.assertEqual(city.landmark_memory(), 4 * 2 * 100 * 8)


class DispatcherEquivalenceTests(unittest.TestCase):
    def test_pick_driver_is_optimal_on_random_cities(self):
        rng = random.Random(2027)
        for trial in range(80):
            city = random_city(rng, closures=rng.randint(0, 40))
            dispatcher = Dispatcher(city, cell_size=rng.choice([150, 400, 1000]),
                                    pickup_limit=rng.choice([200, 600, 1200]))
            nodes = list(city.nodes())
            for n in range(rng.randint(1, 15)):
                dispatcher.add_driver(f"D{n:02d}", rng.choice(nodes))
            dispatcher.new_request("R", rng.choice(nodes), (0, 0), "standard", 0.0)
            req = dispatcher.waiting_requests["R"]
            expected = slow_best_time(dispatcher, req)
            found = dispatcher.pick_driver(req)
            with self.subTest(trial=trial):
                if expected == math.inf:
                    self.assertIsNone(found)
                else:
                    self.assertEqual(found[1], expected)
                    # and the winner is the smallest id among drivers that tie
                    tied = [d.driver_id for d in dispatcher.drivers.values()
                            if city.shortest_path(d.location, req.pickup)[0] == expected]
                    self.assertEqual(found[0], min(tied))

    def test_same_matches_as_phase2_over_random_operations(self):
        rng = random.Random(404)
        for trial in range(6):
            seed = rng.random()
            new_city = random_city(random.Random(seed), 10, 10, closures=30)
            old_city = random_city(random.Random(seed), 10, 10, phase2.RoadNetwork, closures=30)
            new = Dispatcher(new_city, cell_size=300, promote_after=90, pickup_limit=500,
                             keep_tier_order=False)
            old = phase2.Dispatcher(old_city, cell_size=300, promote_after=90, pickup_limit=500)
            ops = random.Random(seed)
            nodes = list(new_city.nodes())
            for n in range(10):
                spot = ops.choice(nodes)
                new.add_driver(f"D{n}", spot)
                old.add_driver(f"D{n}", spot)
            now = 0.0
            for step in range(600):
                now += ops.uniform(0, 8)
                roll = ops.random()
                if roll < 0.35:
                    args = (f"R{step}", ops.choice(nodes), ops.choice(nodes),
                            ops.choice(["standard", "premium", "accessible"]), now)
                    new.new_request(*args)
                    old.new_request(*args)
                elif roll < 0.45 and old.waiting_requests:
                    rid = ops.choice(sorted(old.waiting_requests))
                    new.cancel_request(rid)
                    old.cancel_request(rid)
                elif roll < 0.65 and old.trips_in_progress:
                    did = ops.choice(sorted(old.trips_in_progress))
                    new.complete_trip(did)
                    old.complete_trip(did)
                elif roll < 0.75:
                    did, spot = ops.choice(sorted(old.drivers)), ops.choice(nodes)
                    new.move_driver(did, spot)
                    old.move_driver(did, spot)
                else:
                    a = [(m.request_id, m.pickup_time, m.waited) for m in new.run_round(now)]
                    b = [(m.request_id, m.pickup_time, m.waited) for m in old.run_round(now)]
                    with self.subTest(trial=trial, step=step):
                        self.assertEqual(a, b)
                self.assertEqual(set(new.waiting_requests), set(old.waiting_requests))

    def test_aging_matches_phase2(self):
        rng = random.Random(9)
        city, old_city = build_grid_city(4, 4), phase2.build_grid_city(4, 4)
        new = Dispatcher(city, promote_after=60, keep_tier_order=False)
        old = phase2.Dispatcher(old_city, promote_after=60)
        now = 0.0
        for n in range(400):
            now += rng.uniform(0, 20)
            args = (f"R{n}", (0, 0), (1, 1), rng.choice(["standard", "premium", "accessible"]), now)
            new.new_request(*args)
            old.new_request(*args)
            if rng.random() < 0.3:
                rid = rng.choice(sorted(old.waiting_requests))
                new.cancel_request(rid)
                old.cancel_request(rid)
            if rng.random() < 0.2:
                self.assertEqual(new.age_requests(now), old.age_requests(now))
                for rid, req in old.waiting_requests.items():
                    self.assertEqual(new.waiting_requests[rid].bumps, req.bumps)
                    self.assertEqual(new._queue.get_priority(rid), old._queue.get_priority(rid))

    def test_bump_heap_stays_small_with_many_cancellations(self):
        dispatcher = Dispatcher(build_grid_city(3, 3), promote_after=30)
        for n in range(5000):
            dispatcher.new_request(f"R{n}", (0, 0), (1, 1), "standard", float(n))
            if n > 10:
                dispatcher.cancel_request(f"R{n - 10}")
            if n % 50 == 0:
                dispatcher.age_requests(float(n))
        self.assertLess(len(dispatcher._bumps_due), 2 * dispatcher.count_waiting() + 200)

    def test_accessible_rider_stays_ahead_of_bumped_standard(self):
        dispatcher = Dispatcher(build_grid_city(10, 10), cell_size=400.0, promote_after=100)
        dispatcher.add_driver("D1", (5, 5))
        dispatcher.new_request("old_standard", (4, 4), (0, 0), "standard", 0.0)
        dispatcher.new_request("accessible", (6, 6), (9, 9), "accessible", 250.0)
        # at t=260 the standard rider has been bumped twice (level 0), like accessible
        self.assertEqual(dispatcher.run_round(now=260.0)[0].request_id, "accessible")
        # Phase 2 ordering (by request time inside a level) would have served the old one
        old_order = Dispatcher(build_grid_city(10, 10), cell_size=400.0, promote_after=100,
                               keep_tier_order=False)
        old_order.add_driver("D1", (5, 5))
        old_order.new_request("old_standard", (4, 4), (0, 0), "standard", 0.0)
        old_order.new_request("accessible", (6, 6), (9, 9), "accessible", 250.0)
        self.assertEqual(old_order.run_round(now=260.0)[0].request_id, "old_standard")


class DeadEndCacheTests(unittest.TestCase):
    def build(self):
        city = build_grid_city(10, 10)
        add_river_barrier(city, west_col=4, rows=10, bridge_rows=set())   # no bridge at all
        dispatcher = Dispatcher(city, cell_size=400.0, pickup_limit=10_000)
        dispatcher.add_driver("west", (0, 0))
        dispatcher.new_request("east_rider", (9, 9), (9, 0), "standard", 0.0)
        return city, dispatcher

    def test_unreachable_pickup_is_searched_once(self):
        _, dispatcher = self.build()
        for round_no in range(5):
            self.assertEqual(dispatcher.run_round(now=float(round_no)), [])
        self.assertEqual(dispatcher.stats["searches"], 1)
        self.assertEqual(dispatcher.stats["dead_end_hits"], 4)

    def test_cache_forgets_when_a_driver_frees_up_or_a_road_opens(self):
        city, dispatcher = self.build()
        dispatcher.run_round(now=0.0)
        city.add_street((4, 3), (5, 3), 30)          # a bridge opens
        self.assertEqual(len(dispatcher.run_round(now=1.0)), 1)

        _, dispatcher = self.build()
        dispatcher.run_round(now=0.0)
        dispatcher.add_driver("east", (8, 8))        # a driver appears on the east side
        self.assertEqual(dispatcher.run_round(now=1.0)[0].driver_id, "east")

        _, dispatcher = self.build()
        dispatcher.run_round(now=0.0)
        dispatcher.move_driver("west", (7, 7))       # the driver crosses somehow
        self.assertEqual(dispatcher.run_round(now=1.0)[0].driver_id, "west")

    def test_cache_survives_a_driver_getting_busy(self):
        _, dispatcher = self.build()
        dispatcher.add_driver("west2", (1, 1))
        dispatcher.run_round(now=0.0)
        dispatcher.new_request("west_rider", (0, 1), (0, 2), "premium", 1.0)
        dispatcher.run_round(now=1.0)                # west rider served, east still stuck
        self.assertEqual(dispatcher.stats["dead_end_hits"], 1)

    def test_closing_a_street_keeps_the_cache(self):
        city, dispatcher = self.build()
        dispatcher.run_round(now=0.0)
        city.close_street((0, 5), (0, 6))            # closures can only make it worse
        dispatcher.run_round(now=1.0)
        self.assertEqual(dispatcher.stats["dead_end_hits"], 1)

    def test_a_far_away_driver_freeing_up_keeps_the_cache(self):
        city = build_grid_city(40, 40)
        add_river_barrier(city, west_col=4, rows=40, bridge_rows=set())
        dispatcher = Dispatcher(city, cell_size=400.0, pickup_limit=300)
        dispatcher.add_driver("west", (0, 0))
        dispatcher.new_request("east_rider", (5, 0), (9, 0), "standard", 0.0)
        dispatcher.run_round(now=0.0)
        dispatcher.add_driver("far_away", (39, 39))  # 7+ km away: can't reach in 300 s
        dispatcher.run_round(now=1.0)
        self.assertEqual(dispatcher.stats["dead_end_hits"], 1)
        dispatcher.add_driver("close_by", (8, 2))    # within reach: cache entry dropped
        self.assertEqual(dispatcher.run_round(now=2.0)[0].driver_id, "close_by")

    def test_straight_line_check_skips_the_road_search(self):
        dispatcher = Dispatcher(build_grid_city(30, 30), cell_size=400.0, pickup_limit=100)
        dispatcher.add_driver("far", (0, 0))
        dispatcher.new_request("R", (29, 29), (0, 0), "standard", 0.0)
        self.assertEqual(dispatcher.run_round(now=0.0), [])
        self.assertEqual((dispatcher.stats["searches"], dispatcher.stats["precheck_skips"]), (0, 1))

    def test_turning_the_cache_off_gives_same_matches(self):
        outcomes = []
        for remember in (True, False):
            rng = random.Random(13)
            city = random_city(random.Random(5), 10, 10, closures=45)
            dispatcher = Dispatcher(city, cell_size=300, pickup_limit=400,
                                    remember_dead_ends=remember)
            nodes = list(city.nodes())
            for n in range(6):
                dispatcher.add_driver(f"D{n}", rng.choice(nodes))
            log = []
            for step in range(800):
                roll = rng.random()
                if roll < 0.3:
                    dispatcher.new_request(f"R{step}", rng.choice(nodes), rng.choice(nodes),
                                           "standard", float(step))
                elif roll < 0.4 and dispatcher.trips_in_progress:
                    dispatcher.complete_trip(rng.choice(sorted(dispatcher.trips_in_progress)))
                elif roll < 0.45:
                    a = rng.choice(nodes)
                    for b in list(city._out_edges[a]):
                        if a in city._out_edges[b] and rng.random() < 0.5:
                            city.close_street(a, b)
                            break
                    else:
                        if a[0] + 1 < 10:
                            city.add_street(a, (a[0] + 1, a[1]), 30)
                else:
                    log += [(m.request_id, m.driver_id, m.pickup_time)
                            for m in dispatcher.run_round(float(step))]
            outcomes.append(log)
        self.assertEqual(outcomes[0], outcomes[1])
        self.assertGreater(len(outcomes[0]), 50)


class SimulationPhase3Tests(unittest.TestCase):
    def test_both_engines_agree_on_the_same_workload(self):
        for seed in (1, 2, 3):
            base = dict(requests=400, drivers=35, seed=seed, keep_tier_order=False)
            old = simulate(SimSettings(engine="phase2", **base))
            new = simulate(SimSettings(engine="phase3", **base))
            with self.subTest(seed=seed):
                self.assertEqual([(m.request_id, m.driver_id, m.pickup_time) for m in old.matches],
                                 [(m.request_id, m.driver_id, m.pickup_time) for m in new.matches])

    def test_arrival_stream_is_sorted_complete_and_uniform(self):
        times = list(arrival_times(20_000, 3600.0, random.Random(1)))
        self.assertEqual(len(times), 20_000)
        self.assertEqual(times, sorted(times))
        self.assertTrue(0 <= times[0] and times[-1] <= 3600.0)
        # roughly a quarter of the arrivals in each quarter of the hour
        for q in range(4):
            share = sum(q * 900 <= t < (q + 1) * 900 for t in times) / len(times)
            self.assertAlmostEqual(share, 0.25, delta=0.02)

    def test_event_heap_does_not_grow_with_the_shift(self):
        small = simulate(SimSettings(requests=500, drivers=40, keep_matches=False))
        large = simulate(SimSettings(requests=500, drivers=40, duration=14_400.0,
                                     keep_matches=False))
        # four times as long, same number of requests: fewer live at once
        self.assertLessEqual(large.peak_events, small.peak_events)

    def test_busy_scenario_keeps_books_balanced(self):
        report = simulate(SimSettings(requests=600, drivers=40, hotspot_share=0.5,
                                      incidents=40, shift_changes=15, seed=21))
        self.assertEqual(report.served + report.cancelled + report.still_waiting, 600)
        self.assertTrue(all(m.waited <= 600 for m in report.matches))

    def test_unknown_engine(self):
        with self.assertRaises(ValueError):
            simulate(SimSettings(engine="phase9"))


if __name__ == "__main__":
    unittest.main()
