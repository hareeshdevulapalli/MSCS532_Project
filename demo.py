"""Command-line demo for the Phase 2 proof of concept.

How to run it:
    python3 demo.py                 # every scenario
    python3 demo.py river           # just one
    python3 demo.py simulate --drivers 60 --requests 500 --seed 7

Scenarios:
    basic     add drivers, create and cancel requests, run a round, end trips
    river     fixed candidate list vs widened search when a river is in the way
    blocking  a request nobody can reach doesn't hold up the others any more
    aging     a standard ride that has waited a long time beats a new premium one
    simulate  one simulated hour on a 20 x 20 street grid
"""

import argparse
import sys

from dispatch import Dispatcher, add_river_barrier, build_grid_city
from dispatch.simulation import SimSettings, simulate


def title(text):
    print(f"\n=== {text} " + "=" * max(0, 60 - len(text)))


def print_matches(matches):
    for m in matches:
        print(f"  {m.request_id} -> {m.driver_id}  pickup {m.pickup_time:>4.0f} s"
              f"  waited {m.waited:>4.0f} s")


def demo_basic():
    title("basic: a few rides from start to finish")
    city = build_grid_city(10, 10)
    disp = Dispatcher(city, cell_size=400.0, candidate_count=3)
    for driver_id, start in [("D1", (0, 0)), ("D2", (5, 5)), ("D3", (9, 9)), ("D4", (2, 7))]:
        disp.add_driver(driver_id, start)
    disp.new_request("R1", (4, 4), (8, 1), "standard", requested_at=0.0)
    disp.new_request("R2", (1, 1), (6, 6), "standard", requested_at=5.0)
    disp.new_request("R3", (9, 8), (0, 9), "accessible", requested_at=9.0)
    disp.new_request("R4", (3, 3), (3, 9), "premium", requested_at=12.0)
    disp.cancel_request("R2")
    print(f"  {disp.count_free_drivers()} free drivers, {disp.count_waiting()} waiting"
          " (rider cancelled R2)")
    disp.move_driver("D4", (3, 6))
    print("  D4 GPS update: (2, 7) -> (3, 6)")
    print("  round at t = 15 s (accessible first, then premium, then standard):")
    print_matches(disp.run_round(now=15.0))
    print(f"  free drivers left: {disp.count_free_drivers()},"
          f" trips going: {sorted(disp.trips_in_progress)}")
    trip = disp.complete_trip("D3")
    print(f"  D3 finished {trip.request_id} and is free again at {disp.drivers['D3'].location}")
    disp.go_offline("D1")
    print(f"  D1 logged off; free drivers: {disp.count_free_drivers()}")
    secs, path = city.shortest_path((0, 0), (3, 2))
    print(f"  quickest route (0,0) -> (3,2): {secs:.0f} s through {len(path)} intersections")


def demo_river():
    title("river: fixed list (Phase 1) vs widened search (Phase 2)")
    print("  A river runs between columns 4 and 5 and the only bridge is in row 9.")
    print("  The pickup is at (4, 1). Four drivers are just across the water and")
    print("  one is further away in a straight line but on the same side, at (1, 4).")
    for widen in (False, True):
        city = build_grid_city(10, 10)
        add_river_barrier(city, west_col=4, rows=10, bridge_rows={9})
        disp = Dispatcher(city, cell_size=400.0, candidate_count=3, widen_search=widen)
        for driver_id, spot in [("across_1", (5, 1)), ("across_2", (5, 0)), ("across_3", (5, 2)),
                                ("across_4", (6, 1)), ("same_bank", (1, 4))]:
            disp.add_driver(driver_id, spot)
        disp.new_request("R", (4, 1), (0, 0), "standard", 0.0)
        match = disp.assign_top(now=0.0)
        label = "widened (Phase 2)   " if widen else "fixed list (Phase 1)"
        print(f"  {label}: {match.driver_id:<9} pickup {match.pickup_time:.0f} s"
              f"  [{disp.stats['searches']} search(es), "
              f"{disp.stats['widenings']} widening(s)]")


def demo_blocking():
    title("blocking: request nobody can reach is at the front")
    for method in ("assign_top", "run_round"):
        city = build_grid_city(10, 10)
        disp = Dispatcher(city, cell_size=400.0, pickup_limit=300)
        disp.add_driver("D1", (0, 0))
        disp.add_driver("D2", (1, 0))
        disp.new_request("far_premium", (9, 9), (0, 0), "premium", 0.0)   # 540 s away
        disp.new_request("near_standard", (0, 2), (5, 5), "standard", 1.0)
        if method == "assign_top":
            served = [m for m in [disp.assign_top(now=10.0)] if m]
        else:
            served = disp.run_round(now=10.0)
        names = [m.request_id for m in served] or ["nobody"]
        print(f"  {method:<15} served {', '.join(names):<15}"
              f" still waiting: {sorted(disp.waiting_requests)}")


def demo_aging():
    title("aging: being fair to standard riders")
    for promote_after in (None, 120.0):
        city = build_grid_city(10, 10)
        disp = Dispatcher(city, cell_size=400.0, promote_after=promote_after)
        disp.add_driver("D1", (5, 5))
        disp.new_request("old_standard", (4, 4), (0, 0), "standard", requested_at=0.0)
        disp.new_request("new_premium", (6, 6), (9, 9), "premium", requested_at=290.0)
        first = disp.run_round(now=300.0)[0]
        label = "off" if promote_after is None else f"every {promote_after:.0f} s"
        print(f"  bumping {label:<12} -> served first: {first.request_id}"
              f" (waited {first.waited:.0f} s)")


def demo_simulate(args):
    title("simulate: one hour on a 20 x 20 grid split by a river")
    settings = SimSettings(drivers=args.drivers, requests=args.requests, seed=args.seed)
    print(f"  {settings.drivers} drivers, {settings.requests} requests over"
          f" {settings.duration / 60:.0f} min, a round every {settings.round_every:.0f} s")
    for line in simulate(settings).report_lines():
        print("  " + line)


DEMOS = {
    "basic": demo_basic,
    "river": demo_river,
    "blocking": demo_blocking,
    "aging": demo_aging,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scenario", nargs="?", default="all",
                        choices=["all", *DEMOS, "simulate"])
    parser.add_argument("--drivers", type=int, default=60)
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.drivers < 1 or args.requests < 0:
        parser.error("you need at least one driver and zero or more requests")

    picked = list(DEMOS) + ["simulate"] if args.scenario == "all" else [args.scenario]
    for name in picked:
        if name == "simulate":
            demo_simulate(args)
        else:
            DEMOS[name]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
