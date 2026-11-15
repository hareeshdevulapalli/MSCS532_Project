"""Tests for the Dispatcher: matching, widened search, queue fairness, driver states."""

import math
import random
import unittest

from dispatch import DriverState, Dispatcher, add_river_barrier, build_grid_city
from dispatch.dispatcher import slow_best_time


def build_river_city():
    """10 x 10 grid with a river between columns 4 and 5 and one bridge in row 9."""
    city = build_grid_city(10, 10)
    add_river_barrier(city, west_col=4, rows=10, bridge_rows={9})
    return city


class MatchingTests(unittest.TestCase):
    def setUp(self):
        self.city = build_grid_city(10, 10)
        self.dispatcher = Dispatcher(self.city, cell_size=400.0, candidate_count=3)

    def test_tier_first_then_request_time(self):
        dispatcher = self.dispatcher
        dispatcher.add_driver("D1", (0, 0))
        dispatcher.add_driver("D2", (9, 9))
        dispatcher.new_request("late_premium", (1, 1), (2, 2), "premium", 50.0)
        dispatcher.new_request("early_standard", (8, 8), (2, 2), "standard", 0.0)
        self.assertEqual(dispatcher.assign_top().request_id, "late_premium")
        self.assertEqual(dispatcher.assign_top().request_id, "early_standard")

    def test_chooses_by_driving_time_not_straight_line(self):
        dispatcher = self.dispatcher
        dispatcher.add_driver("across_river", (5, 1))
        dispatcher.add_driver("same_bank", (3, 4))
        for row in range(9):
            self.city.close_street((4, row), (5, row))
        dispatcher.new_request("R", (4, 1), (0, 0), "standard", 0.0)
        match = dispatcher.assign_top()
        self.assertEqual((match.driver_id, match.pickup_time), ("same_bank", 120))

    def test_drivers_at_same_corner_break_ties_by_id(self):
        dispatcher = self.dispatcher
        for driver_id in ("D9", "D2", "D5"):
            dispatcher.add_driver(driver_id, (3, 3))
        dispatcher.new_request("R", (3, 4), (0, 0), "standard", 0.0)
        self.assertEqual(dispatcher.assign_top().driver_id, "D2")

    def test_wait_time_is_recorded(self):
        self.dispatcher.add_driver("D1", (0, 0))
        self.dispatcher.new_request("R", (0, 1), (1, 1), "standard", 12.0)
        self.assertEqual(self.dispatcher.run_round(now=20.0)[0].waited, 8.0)


class WidenedSearchTests(unittest.TestCase):
    """Phase 2: the candidate list keeps growing until the speed bound says stop."""

    def build(self, widen):
        dispatcher = Dispatcher(build_river_city(), cell_size=400.0, candidate_count=3, widen_search=widen)
        for driver_id, spot in [("across_1", (5, 1)), ("across_2", (5, 0)), ("across_3", (5, 2)),
                                ("across_4", (6, 1)), ("same_bank", (1, 4))]:
            dispatcher.add_driver(driver_id, spot)
        dispatcher.new_request("R", (4, 1), (0, 0), "standard", 0.0)
        return dispatcher

    def test_fixed_list_misses_the_quickest_driver(self):
        match = self.build(widen=False).assign_top()
        self.assertNotEqual(match.driver_id, "same_bank")   # this is the Phase 1 problem
        self.assertGreater(match.pickup_time, 180)

    def test_widened_search_finds_the_quickest_driver(self):
        dispatcher = self.build(widen=True)
        match = dispatcher.assign_top()
        self.assertEqual((match.driver_id, match.pickup_time), ("same_bank", 180))
        self.assertGreaterEqual(dispatcher.stats["widenings"], 1)

    def test_no_widening_needed_on_open_grid(self):
        dispatcher = Dispatcher(build_grid_city(10, 10), cell_size=400.0, candidate_count=2)
        for n, spot in enumerate([(2, 2), (2, 3), (8, 8), (9, 0), (0, 9)]):
            dispatcher.add_driver(f"D{n}", spot)
        dispatcher.new_request("R", (2, 2), (5, 5), "standard", 0.0)
        dispatcher.assign_top()
        self.assertEqual((dispatcher.stats["searches"], dispatcher.stats["widenings"]), (1, 0))

    def test_matches_slow_search_on_random_cities(self):
        rng = random.Random(2026)
        for trial in range(40):
            city = build_grid_city(8, 8)
            for _ in range(25):                    # random closures and slow roads
                col, row = rng.randrange(7), rng.randrange(8)
                try:
                    if rng.random() < 0.6:
                        city.close_street((col, row), (col + 1, row))
                    else:
                        city.add_street((col, row), (col + 1, row), rng.choice([60, 120, 300]))
                except KeyError:
                    pass                           # road already closed
            dispatcher = Dispatcher(city, cell_size=rng.choice([150, 400, 1000]),
                                        candidate_count=rng.randint(1, 4), pickup_limit=1200)
            corners = list(city.nodes())
            for n in range(rng.randint(1, 15)):
                dispatcher.add_driver(f"D{n}", rng.choice(corners))
            dispatcher.new_request("R", rng.choice(corners), (0, 0), "standard", 0.0)
            request = dispatcher.waiting_requests["R"]
            expected = slow_best_time(dispatcher, request)
            found = dispatcher.pick_driver(request)
            with self.subTest(trial=trial):
                if expected == math.inf:
                    self.assertIsNone(found)
                else:
                    self.assertEqual(found[1], expected)


class QueueFairnessTests(unittest.TestCase):
    def setUp(self):
        self.city = build_grid_city(10, 10)

    def blocked_dispatcher(self):
        dispatcher = Dispatcher(self.city, cell_size=400.0, pickup_limit=300)
        dispatcher.add_driver("D1", (0, 0))
        dispatcher.add_driver("D2", (1, 0))
        dispatcher.new_request("far_premium", (9, 9), (0, 0), "premium", 0.0)
        dispatcher.new_request("near_standard", (0, 2), (5, 5), "standard", 1.0)
        return dispatcher

    def test_assign_top_is_blocked_by_unreachable_front(self):
        dispatcher = self.blocked_dispatcher()
        self.assertIsNone(dispatcher.assign_top())
        self.assertEqual(dispatcher.count_waiting(), 2)

    def test_run_round_skips_and_requeues_unreachable_request(self):
        dispatcher = self.blocked_dispatcher()
        matches = dispatcher.run_round(now=5.0)
        self.assertEqual([m.request_id for m in matches], ["near_standard"])
        self.assertEqual(dispatcher._queue.peek(), ("far_premium", (1, 0.0)))
        self.assertEqual(dispatcher.stats["skipped"], 1)

    def test_round_without_drivers_changes_nothing(self):
        dispatcher = Dispatcher(self.city)
        dispatcher.new_request("R", (0, 0), (1, 1), "standard", 0.0)
        self.assertEqual(dispatcher.run_round(now=1.0), [])
        self.assertEqual(dispatcher.count_waiting(), 1)

    def test_bumping_lets_old_standard_beat_new_premium(self):
        dispatcher = Dispatcher(self.city, cell_size=400.0, promote_after=120)
        dispatcher.add_driver("D1", (5, 5))
        dispatcher.new_request("old_standard", (4, 4), (0, 0), "standard", 0.0)
        dispatcher.new_request("new_premium", (6, 6), (9, 9), "premium", 290.0)
        self.assertEqual(dispatcher.run_round(now=300.0)[0].request_id, "old_standard")

    def test_no_bumping_keeps_strict_tiers(self):
        dispatcher = Dispatcher(self.city, cell_size=400.0)
        dispatcher.add_driver("D1", (5, 5))
        dispatcher.new_request("old_standard", (4, 4), (0, 0), "standard", 0.0)
        dispatcher.new_request("new_premium", (6, 6), (9, 9), "premium", 290.0)
        self.assertEqual(dispatcher.age_requests(10_000), 0)
        self.assertEqual(dispatcher.run_round(now=300.0)[0].request_id, "new_premium")

    def test_bumps_stop_at_top_level(self):
        dispatcher = Dispatcher(self.city, promote_after=60)
        dispatcher.new_request("R", (0, 0), (1, 1), "standard", 0.0)
        self.assertEqual(dispatcher.age_requests(60), 1)
        self.assertEqual(dispatcher.waiting_requests["R"].level, 1)
        dispatcher.age_requests(10_000)
        self.assertEqual(dispatcher.waiting_requests["R"].level, 0)
        self.assertEqual(dispatcher._queue.get_priority("R"), (0, 0.0))
        self.assertEqual(dispatcher.age_requests(20_000), 0)   # nothing left to promote


class LifeCycleAndErrorTests(unittest.TestCase):
    def setUp(self):
        self.city = build_grid_city(10, 10)
        self.dispatcher = Dispatcher(self.city, cell_size=400.0, candidate_count=3)

    def test_driver_trip_cycle(self):
        dispatcher = self.dispatcher
        dispatcher.add_driver("D1", (0, 0))
        dispatcher.new_request("R1", (0, 1), (5, 5), "standard", 0.0)
        match = dispatcher.assign_top()
        self.assertEqual((match.driver_id, match.pickup_time), ("D1", 30))
        self.assertEqual(dispatcher.drivers["D1"].state, DriverState.BUSY)
        self.assertEqual(dispatcher.count_free_drivers(), 0)
        with self.assertRaises(KeyError):
            dispatcher.move_driver("D1", (99, 99))
        with self.assertRaises(ValueError):
            dispatcher.go_offline("D1")
        self.assertEqual(dispatcher.complete_trip("D1").request_id, "R1")
        self.assertEqual(dispatcher.drivers["D1"].location, (5, 5))
        self.assertEqual(dispatcher.drivers["D1"].trips_done, 1)
        with self.assertRaises(ValueError):
            dispatcher.complete_trip("D1")

    def test_offline_and_online(self):
        dispatcher = self.dispatcher
        dispatcher.add_driver("D1", (0, 0))
        dispatcher.go_offline("D1")
        dispatcher.new_request("R", (0, 1), (2, 2), "standard", 0.0)
        self.assertIsNone(dispatcher.assign_top())
        with self.assertRaises(ValueError):
            dispatcher.complete_trip("D1")
        dispatcher.move_driver("D1", (7, 7))   # allowed while offline
        dispatcher.go_online("D1", (0, 2))
        self.assertEqual(dispatcher.assign_top().pickup_time, 30)
        with self.assertRaises(ValueError):
            dispatcher.go_online("D1")

    def test_invalid_requests_and_ids(self):
        dispatcher = self.dispatcher
        with self.assertRaises(ValueError):
            dispatcher.new_request("R", (0, 0), (1, 1), "luxury", 0.0)
        with self.assertRaises(KeyError):
            dispatcher.new_request("R", (0, 0), (50, 50), "standard", 0.0)
        dispatcher.new_request("R", (0, 0), (1, 1), "standard", 0.0)
        with self.assertRaises(KeyError):
            dispatcher.new_request("R", (0, 0), (1, 1), "standard", 0.0)
        with self.assertRaises(KeyError):
            dispatcher.cancel_request("missing")
        with self.assertRaises(KeyError):
            dispatcher.go_offline("nobody")
        with self.assertRaises(KeyError):
            dispatcher.add_driver("D1", (42, 42))

    def test_constructor_validation(self):
        with self.assertRaises(TypeError):
            Dispatcher("not a graph")
        for bad in ({"candidate_count": 0}, {"pickup_limit": 0}, {"promote_after": -5}):
            with self.subTest(**bad), self.assertRaises(ValueError):
                Dispatcher(self.city, **bad)

    def test_invariants_hold_under_random_operations(self):
        rng = random.Random(99)
        dispatcher = Dispatcher(self.city, cell_size=300.0, promote_after=100,
                                    pickup_limit=400)
        corners = list(self.city.nodes())
        for n in range(12):
            dispatcher.add_driver(f"D{n}", rng.choice(corners))
        now, next_request = 0.0, 0
        for _ in range(1500):
            now += rng.uniform(0, 10)
            roll = rng.random()
            if roll < 0.35:
                dispatcher.new_request(f"R{next_request}", rng.choice(corners),
                                        rng.choice(corners), rng.choice(["standard", "premium"]), now)
                next_request += 1
            elif roll < 0.45 and dispatcher.waiting_requests:
                dispatcher.cancel_request(rng.choice(list(dispatcher.waiting_requests)))
            elif roll < 0.65 and dispatcher.trips_in_progress:
                dispatcher.complete_trip(rng.choice(list(dispatcher.trips_in_progress)))
            elif roll < 0.80:
                dispatcher.move_driver(rng.choice(list(dispatcher.drivers)),
                                                  rng.choice(corners))
            else:
                dispatcher.run_round(now)
            free = {d for d, rec in dispatcher.drivers.items() if rec.is_free}
            busy = {d for d, rec in dispatcher.drivers.items() if rec.state is DriverState.BUSY}
            self.assertEqual(free, set(dispatcher._free_grid._coords))
            self.assertEqual(busy, set(dispatcher.trips_in_progress))
            self.assertEqual(set(dispatcher.waiting_requests), set(dispatcher._queue.all_keys()))


if __name__ == "__main__":
    unittest.main()
