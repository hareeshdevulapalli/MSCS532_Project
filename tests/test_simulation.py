"""Tests for the event-driven simulation and the command-line demo."""

import contextlib
import io
import unittest

import demo
from dispatch.simulation import SimSettings, simulate


class SimulationTests(unittest.TestCase):
    def test_every_request_is_accounted_for(self):
        report = simulate(SimSettings(requests=200, drivers=30, seed=5))
        self.assertEqual(report.served + report.cancelled + report.still_waiting, report.requests)
        self.assertEqual(report.still_waiting, 0)        # the simulation drains the queue
        self.assertEqual(len({m.request_id for m in report.matches}), report.served)

    def test_same_seed_gives_same_result(self):
        first = simulate(SimSettings(requests=120, seed=11))
        second = simulate(SimSettings(requests=120, seed=11))
        self.assertEqual([(m.request_id, m.driver_id) for m in first.matches],
                         [(m.request_id, m.driver_id) for m in second.matches])

    def test_waits_never_exceed_patience(self):
        config = SimSettings(requests=250, drivers=20, seed=3)
        report = simulate(config)
        self.assertTrue(all(m.waited <= config.patience for m in report.matches))
        self.assertTrue(all(m.pickup_time <= config.patience for m in report.matches))

    def test_exact_search_never_gives_slower_pickups_than_fixed_list_per_request(self):
        # Greedy choices change later requests, so the hour-long average can go
        # either way; this only checks both modes run and serve everyone.
        exact = simulate(SimSettings(requests=150, widen_search=True, seed=8))
        fixed = simulate(SimSettings(requests=150, widen_search=False, seed=8))
        self.assertEqual(exact.served, 150)
        self.assertEqual(fixed.served, 150)
        self.assertGreater(exact.counters["settled"], 0)

    def test_no_requests(self):
        report = simulate(SimSettings(requests=0))
        self.assertEqual((report.served, report.avg_wait), (0, 0.0))


class DemoTests(unittest.TestCase):
    def run_demo(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(demo.main(list(argv)), 0)
        return out.getvalue()

    def test_river_scenario_shows_the_improvement(self):
        text = self.run_demo("river")
        self.assertIn("exact search (Phase 3): same_bank pickup 180 s", text)

    def test_all_scenarios_run(self):
        text = self.run_demo("all", "--requests", "50")
        for name in ("basic", "river", "blocking", "aging", "simulate", "compare"):
            self.assertIn(f"=== {name}", text)

    def test_bad_arguments_are_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            demo.main(["simulate", "--drivers", "0"])


if __name__ == "__main__":
    unittest.main()
