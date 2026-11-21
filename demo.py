"""Command-line demo for the ride-hailing dispatcher (Phase 3).

How to run it:
    python3 demo.py                 # every scenario
    python3 demo.py river           # just one
    python3 demo.py simulate --drivers 60 --requests 500 --seed 7
    python3 demo.py simulate --size 60 --drivers 600 --requests 6000 --busy
    python3 demo.py compare --size 40 --drivers 300 --requests 3000

Scenarios:
    basic     add drivers, create and cancel requests, run a round, end trips
    river     fixed candidate list vs widened search when a river is in the way
    blocking  a request nobody can reach doesn't hold up the others any more
    aging     a standard ride that has waited a long time beats a new premium one
    simulate  one simulated hour on a street grid (--busy adds hotspots,
              road incidents and driver shift changes)
    compare   the same workload through the Phase 2 and Phase 3 dispatchers
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
    title("river: fixed list (Phase 1) vs exact search (Phase 3)")
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
        label = "exact search (Phase 3)" if widen else "fixed list (Phase 1)  "
        print(f"  {label}: {match.driver_id:<9} pickup {match.pickup_time:.0f} s"
              f"  [{disp.stats['searches']} search(es), "
              f"{disp.stats['settled']} intersections settled]")


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


def settings_from(args, **extra):
    busy = dict(hotspot_share=0.4, incidents=args.size * 2, shift_changes=args.drivers // 5)
    return SimSettings(cols=args.size, rows=args.size, drivers=args.drivers,
                       requests=args.requests, seed=args.seed,
                       **(busy if args.busy else {}), **extra)


def demo_simulate(args):
    title(f"simulate: one hour on a {args.size} x {args.size} grid split by a river")
    settings = settings_from(args)
    print(f"  {settings.drivers} drivers, {settings.requests} requests over"
          f" {settings.duration / 60:.0f} min, a round every {settings.round_every:.0f} s")
    if args.busy:
        print(f"  busy mode: {settings.hotspot_share:.0%} of pickups near {settings.hotspots}"
              f" hotspots, {settings.incidents} road incidents,"
              f" {settings.shift_changes} driver shift changes")
    for line in simulate(settings).report_lines():
        print("  " + line)


def demo_compare(args):
    title("compare: same workload, Phase 2 vs Phase 3 dispatcher")
    results = {}
    for engine in ("phase2", "phase3"):
        # Phase 2's queue key is used for both so the outcomes can be compared.
        results[engine] = simulate(settings_from(args, engine=engine, keep_tier_order=False))
    old, new = results["phase2"], results["phase3"]
    print(f"  {args.size} x {args.size} grid, {args.drivers} drivers, {args.requests} requests")
    print(f"  {'':<22}{'Phase 2':>12}{'Phase 3':>12}")
    for label, a, b in [("served", old.served, new.served),
                        ("average wait (s)", f"{old.avg_wait:.1f}", f"{new.avg_wait:.1f}"),
                        ("average pickup (s)", f"{old.avg_pickup:.1f}", f"{new.avg_pickup:.1f}"),
                        ("road searches", old.counters["searches"], new.counters["searches"]),
                        ("run time (s)", f"{old.elapsed:.3f}", f"{new.elapsed:.3f}")]:
        print(f"  {label:<22}{a!s:>12}{b!s:>12}")
    same = [(m.request_id, m.driver_id) for m in old.matches] == \
           [(m.request_id, m.driver_id) for m in new.matches]
    print(f"  speed-up {old.elapsed / max(new.elapsed, 1e-9):.1f}x;"
          f" identical matches: {'yes' if same else 'no'}")


DEMOS = {
    "basic": demo_basic,
    "river": demo_river,
    "blocking": demo_blocking,
    "aging": demo_aging,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scenario", nargs="?", default="all",
                        choices=["all", *DEMOS, "simulate", "compare"])
    parser.add_argument("--size", type=int, default=20, help="streets per side of the grid city")
    parser.add_argument("--busy", action="store_true",
                        help="add hotspots, road incidents and driver shift changes")
    parser.add_argument("--drivers", type=int, default=60)
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.drivers < 1 or args.requests < 0:
        parser.error("you need at least one driver and zero or more requests")
    if args.size < 2:
        parser.error("the city needs at least 2 streets per side")

    picked = list(DEMOS) + ["simulate", "compare"] if args.scenario == "all" else [args.scenario]
    for name in picked:
        if name == "simulate":
            demo_simulate(args)
        elif name == "compare":
            demo_compare(args)
        else:
            DEMOS[name]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
