"""Phase 3 benchmarks: Phase 2 (frozen copy) vs the optimized package.

    python3 benchmarks/run_benchmarks.py            # full run, a few minutes
    python3 benchmarks/run_benchmarks.py --quick    # smaller sizes, ~30 s

Writes results/benchmarks.json and prints a summary. benchmarks/make_figures.py
turns the JSON into the PNG graphs used in the report.

Every timing is the best of several repeats with time.perf_counter, so
background noise on the laptop pushes numbers up, never down.
"""

import argparse
import gc
import heapq
import json
import math
import os
import platform
import random
import sys
import time
import tracemalloc
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dispatch import Dispatcher, IndexedMinHeap, RoadNetwork, SpatialGrid, phase2  # noqa: E402
from dispatch.city_builder import add_river_barrier, build_grid_city  # noqa: E402
from dispatch.models import TripRequest  # noqa: E402
from dispatch.simulation import SimSettings, simulate  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def best_of(fn, repeats=3):
    best = math.inf
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


# ----- 1. grid nearest-k ----------------------------------------------------

def bench_grid(sizes):
    """Constant driver density: the city grows with the fleet (one driver per ~0.1 km^2)."""
    rows = []
    for n in sizes:
        side = math.sqrt(n) * 300.0
        rng = random.Random(n)
        points = [(rng.uniform(0, side), rng.uniform(0, side)) for _ in range(n)]
        queries = [(rng.uniform(0, side), rng.uniform(0, side)) for _ in range(400)]
        row = {"n": n}
        for label, cls in (("phase2", phase2.SpatialGrid), ("phase3", SpatialGrid)):
            grid = cls(500.0)
            for i, (x, y) in enumerate(points):
                grid.insert(i, x, y)
            # churn so Phase 3's bounding box has seen deletes too
            for i in range(0, n, 10):
                grid.delete(i)
                grid.insert(i, *points[i])
            row[label] = best_of(lambda: [grid.nearest(x, y, k=5) for x, y in queries]) / len(queries)
        rows.append(row)
        print(f"  grid n={n:>7}: phase2 {row['phase2'] * 1e6:8.1f} us  "
              f"phase3 {row['phase3'] * 1e6:8.1f} us  x{row['phase2'] / row['phase3']:.1f}")
    return rows


def bench_cell_size(n=20_000):
    """Same fleet, different cell sizes: shows the tuning trade-off."""
    side = math.sqrt(n) * 300.0
    rng = random.Random(1)
    points = [(rng.uniform(0, side), rng.uniform(0, side)) for _ in range(n)]
    queries = [(rng.uniform(0, side), rng.uniform(0, side)) for _ in range(400)]
    rows = []
    for cell in (50, 100, 200, 400, 800, 1600, 3200, 6400):
        grid = SpatialGrid(cell)
        for i, (x, y) in enumerate(points):
            grid.insert(i, x, y)
        t = best_of(lambda: [grid.nearest(x, y, k=5) for x, y in queries]) / len(queries)
        rows.append({"cell": cell, "phase3": t, "cells": len(grid._cells)})
    print("  cell size: " + ", ".join(f"{r['cell']}m {r['phase3'] * 1e6:.0f}us" for r in rows))
    return rows


# ----- 2. indexed heap ------------------------------------------------------

def heap_workload(n, seed):
    rng = random.Random(seed)
    ops = []
    for i in range(n):                       # fill
        ops.append(("push", i, (rng.randint(0, 2), rng.random())))
    for _ in range(n):                       # then a realistic mix
        roll = rng.random()
        if roll < 0.3:
            ops.append(("push", None, (rng.randint(0, 2), rng.random())))
        elif roll < 0.55:
            ops.append(("pop", None, None))
        elif roll < 0.8:
            ops.append(("change", None, (rng.randint(0, 2), rng.random())))
        else:
            ops.append(("remove", None, None))
    return ops


def run_heap(cls, ops):
    heap = cls()
    live, next_key = [], 0
    pick = random.Random(0)
    for op, key, prio in ops:
        if op == "push":
            heap.push(next_key, prio)
            live.append(next_key)
            next_key += 1
        elif not live:
            continue
        elif op == "pop":
            k, _ = heap.pop()
        elif op == "change":
            k = live[pick.randrange(len(live))]
            if k in heap:
                heap.change_priority(k, prio)
        else:
            i = pick.randrange(len(live))
            k = live[i]
            live[i] = live[-1]
            live.pop()
            if k in heap:
                heap.remove(k)


def bench_heap(sizes):
    rows = []
    for n in sizes:
        ops = heap_workload(n, n)
        row = {"n": n}
        for label, cls in (("phase2", phase2.IndexedMinHeap), ("phase3", IndexedMinHeap)):
            row[label] = best_of(lambda: run_heap(cls, ops)) / len(ops)
        rows.append(row)
        print(f"  heap n={n:>7}: phase2 {row['phase2'] * 1e6:6.2f} us/op  "
              f"phase3 {row['phase3'] * 1e6:6.2f} us/op  x{row['phase2'] / row['phase3']:.1f}")
    return rows


# ----- 3. aging -------------------------------------------------------------

def bench_aging(sizes):
    """n waiting requests; one round where about 1% of them are due a bump."""
    rows = []
    for n in sizes:
        row = {"n": n}
        for label in ("phase2", "phase3"):
            if label == "phase2":
                disp = phase2.Dispatcher(phase2.build_grid_city(3, 3), promote_after=180)
            else:
                disp = Dispatcher(build_grid_city(3, 3), promote_after=180, keep_tier_order=False)
            # requests spread over 18,000 s, so a 5 s round makes ~n * 5 / 180 / 100 due
            for i in range(n):
                disp.new_request(f"R{i}", (0, 0), (1, 1), "standard", 18_000.0 * i / n)
            disp.age_requests(18_000.0)
            now = [18_000.0]

            def one_round():
                now[0] += 5.0
                disp.age_requests(now[0])
            row[label] = best_of(one_round, repeats=5)
        rows.append(row)
        print(f"  aging n={n:>7}: phase2 {row['phase2'] * 1e3:7.3f} ms  "
              f"phase3 {row['phase3'] * 1e3:7.3f} ms  x{row['phase2'] / row['phase3']:.0f}")
    return rows


# ----- 4. driver search -----------------------------------------------------

class CountingPhase2Roads(phase2.RoadNetwork):
    """Phase 2 road network that counts settled intersections (for counts only)."""

    settled = 0

    def times_to_target(self, target, sources, limit=math.inf):
        remaining, result = set(sources), {}
        dist, frontier = {target: 0}, [(0, target)]
        while frontier and remaining:
            t, node = heapq.heappop(frontier)
            if t > dist[node]:
                continue
            if t > limit:
                break
            CountingPhase2Roads.settled += 1
            if node in remaining:
                remaining.discard(node)
                result[node] = t
            for before, seconds in self._in_edges[node].items():
                arrive = t + seconds
                if arrive < dist.get(before, math.inf):
                    dist[before] = arrive
                    heapq.heappush(frontier, (arrive, before))
        return result


def search_setup(side, label, river, count=False):
    rng = random.Random(side)
    if label == "phase2":
        cls = CountingPhase2Roads if count else phase2.RoadNetwork
        city = _grid(side, cls)
    else:
        city = _grid(side, RoadNetwork)
    if river:
        add_river_barrier(city, side // 2 - 1, side, bridge_rows={side // 4, 3 * side // 4})
    common = dict(cell_size=400.0, candidate_count=5, pickup_limit=900)
    disp = (phase2.Dispatcher(city, **common) if label == "phase2"
            else Dispatcher(city, **common))
    nodes = list(city.nodes())
    for i in range(max(side * side // 10, 5)):          # one free driver per 10 corners
        disp.add_driver(f"D{i}", rng.choice(nodes))
    reqs = [TripRequest(f"R{i}", rng.choice(nodes), nodes[0], "standard", 0.0) for i in range(300)]
    return disp, reqs


def _grid(side, cls):
    city = cls()
    for c in range(side):
        for r in range(side):
            city.add_node((c, r), c * 200.0, r * 200.0)
    for c in range(side):
        for r in range(side):
            if c + 1 < side:
                city.add_street((c, r), (c + 1, r), 30)
            if r + 1 < side:
                city.add_street((c, r), (c, r + 1), 30)
    return city


def bench_search(sides, river=True):
    rows = []
    for side in sides:
        row = {"side": side, "nodes": side * side, "drivers": max(side * side // 10, 5)}
        for label in ("phase2", "phase3"):
            disp, reqs = search_setup(side, label, river)
            row[label] = best_of(lambda: [disp.pick_driver(r) for r in reqs]) / len(reqs)
        # settled-intersection counts (separate, instrumented run for Phase 2)
        CountingPhase2Roads.settled = 0
        disp, reqs = search_setup(side, "phase2", river, count=True)
        for r in reqs:
            disp.pick_driver(r)
        row["phase2_settled"] = CountingPhase2Roads.settled / len(reqs)
        disp, reqs = search_setup(side, "phase3", river)
        for r in reqs:
            disp.pick_driver(r)
        row["phase3_settled"] = disp.stats["settled"] / len(reqs)
        rows.append(row)
        print(f"  search {side:>3}x{side:<3}: phase2 {row['phase2'] * 1e6:8.1f} us "
              f"({row['phase2_settled']:6.1f} settled)  phase3 {row['phase3'] * 1e6:8.1f} us "
              f"({row['phase3_settled']:6.1f} settled)  x{row['phase2'] / row['phase3']:.1f}")
    return rows


# ----- 4b. route queries (trip durations) -------------------------------------

def bench_routes(sides, landmark_counts=(2, 4, 8, 16)):
    rows = []
    for side in sides:
        settings = SimSettings(cols=side, rows=side)
        from dispatch.simulation import build_city
        old, new = build_city(settings, phase2.RoadNetwork), build_city(settings)
        rng = random.Random(side)
        nodes = list(new.nodes())
        pairs = [(rng.choice(nodes), rng.choice(nodes)) for _ in range(200)]

        def run(city):
            return best_of(lambda: [city.shortest_path(a, b) for a, b in pairs]) / len(pairs)

        def settled(city):
            total = 0
            for a, b in pairs:
                city.shortest_path(a, b)
                total += city.last_settled
            return total / len(pairs)
        row = {"side": side, "dijkstra": run(old), "astar": run(new), "astar_settled": settled(new)}
        for k in landmark_counts:
            t0 = time.perf_counter()
            new.prepare_landmarks(k)
            row[f"alt{k}_setup"] = time.perf_counter() - t0
            row[f"alt{k}"] = run(new)
            row[f"alt{k}_settled"] = settled(new)
            row[f"alt{k}_bytes"] = new.landmark_memory()
        rows.append(row)
        print(f"  routes {side:>3}x{side:<3}: dijkstra {row['dijkstra'] * 1e3:6.3f} ms  "
              f"A* {row['astar'] * 1e3:6.3f} ms  ALT8 {row['alt8'] * 1e3:6.3f} ms "
              f"({row['alt8_settled']:.0f} settled, {row['alt8_bytes'] / 1024:.0f} KiB, "
              f"setup {row['alt8_setup']:.2f} s)")
    return rows


# ----- 5. end-to-end simulation ------------------------------------------------

def sim_settings(side, busy, **extra):
    drivers = side * side * 3 // 20          # 0.15 drivers per corner
    # Trips get longer as the city grows, so each driver manages fewer per
    # hour; this keeps the fleet busy but not hopelessly overloaded.
    requests = drivers * 100 // side
    busy_opts = dict(hotspot_share=0.4, incidents=side * 2, shift_changes=drivers // 5) if busy else {}
    options = dict(cols=side, rows=side, drivers=drivers, requests=requests,
                   keep_tier_order=False, keep_matches=False, **busy_opts)
    options.update(extra)
    return SimSettings(**options)


def bench_sim(sides, busy):
    rows = []
    for side in sides:
        row = {"side": side}
        out = {}
        for engine in ("phase2", "phase3"):
            settings = sim_settings(side, busy, engine=engine, keep_matches=True)
            best = None
            for _ in range(2 if side <= 80 else 1):
                gc.collect()
                result = simulate(settings)
                best = result if best is None or result.elapsed < best.elapsed else best
            out[engine] = best
            row[engine] = best.elapsed
        a, b = out["phase2"], out["phase3"]
        row["setup"] = b.setup
        row.update(drivers=sim_settings(side, busy).drivers,
                   requests=sim_settings(side, busy).requests,
                   served=b.served, served_phase2=a.served,
                   avg_wait=b.avg_wait, avg_pickup=b.avg_pickup,
                   identical=[(m.request_id, m.driver_id) for m in a.matches]
                   == [(m.request_id, m.driver_id) for m in b.matches],
                   searches_phase2=a.counters["searches"], searches_phase3=b.counters["searches"],
                   dead_end_skips=b.counters["dead_end_hits"] + b.counters["precheck_skips"],
                   peak_events=b.peak_events)
        rows.append(row)
        print(f"  sim {'busy' if busy else 'calm'} {side:>3}x{side:<3} ({row['drivers']} drivers, "
              f"{row['requests']} req): phase2 {a.elapsed:7.2f} s  phase3 {b.elapsed:6.2f} s"
              f" (+{b.setup:.2f} setup)  x{a.elapsed / (b.elapsed + b.setup):.1f}  "
              f"served {b.served}/{a.served}  same={row['identical']}")
    return rows


def bench_memory(sides):
    """Peak traced memory of a whole simulation (tracemalloc slows runs, so separate)."""
    rows = []
    for side in sides:
        row = {"side": side}
        for engine in ("phase2", "phase3"):
            settings = sim_settings(side, False, engine=engine)
            if engine == "phase2":
                settings.keep_matches = True     # Phase 2 always kept every match
            gc.collect()
            tracemalloc.start()
            simulate(settings)
            row[engine] = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
        rows.append(row)
        print(f"  memory {side:>3}x{side:<3}: phase2 {row['phase2'] / 2**20:6.2f} MiB  "
              f"phase3 {row['phase3'] / 2**20:6.2f} MiB")
    return rows


def bench_long_shift():
    """Same request rate, longer shift: Phase 2's event heap holds every future request."""
    rows = []
    for hours in (1, 2, 4, 8):
        row = {"hours": hours}
        for engine in ("phase2", "phase3"):
            if engine == "phase2":
                settings = phase2.SimSettings(cols=40, rows=40, drivers=240,
                                              requests=1200 * hours, duration=3600.0 * hours)
                run = phase2.simulate
            else:
                settings = SimSettings(cols=40, rows=40, drivers=240, requests=1200 * hours,
                                       duration=3600.0 * hours, keep_matches=False)
                run = simulate
            gc.collect()
            tracemalloc.start()
            run(settings)
            row[engine] = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
        rows.append(row)
        print(f"  shift {hours}h: phase2 {row['phase2'] / 2**20:6.2f} MiB  "
              f"phase3 {row['phase3'] / 2**20:6.2f} MiB")
    return rows


def bench_object_memory(n=100_000):
    """Bytes per driver/request record and per grid entry."""
    from dispatch import models as new_models
    from dispatch.phase2 import models as old_models
    rows = {}
    for label, m in (("phase2", old_models), ("phase3", new_models)):
        gc.collect()
        tracemalloc.start()
        drivers = [m.Driver(f"D{i}", (i % 100, i // 100)) for i in range(n)]
        reqs = [m.TripRequest(f"R{i}", (0, 0), (1, 1), "standard", float(i)) for i in range(n)]
        used = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        rows[label] = used / (2 * n)
        del drivers, reqs
    print(f"  record size: phase2 {rows['phase2']:.0f} B  phase3 {rows['phase3']:.0f} B")
    return rows


# ----- 6. accuracy vs speed ---------------------------------------------------

def bench_tradeoff(side=60):
    """Fixed list of k drivers (Phase 1 style, fast but can miss) vs exact search."""
    rows = []
    rng = random.Random(5)
    city = _grid(side, RoadNetwork)
    add_river_barrier(city, side // 2 - 1, side, bridge_rows={side // 4, 3 * side // 4})
    nodes = list(city.nodes())
    spots = [rng.choice(nodes) for _ in range(side * side // 10)]
    reqs = [TripRequest(f"R{i}", rng.choice(nodes), nodes[0], "standard", 0.0) for i in range(400)]
    exact = Dispatcher(city, cell_size=400.0, pickup_limit=1800)
    for i, s in enumerate(spots):
        exact.add_driver(f"D{i}", s)
    best = [exact.pick_driver(r) for r in reqs]
    t_exact = best_of(lambda: [exact.pick_driver(r) for r in reqs]) / len(reqs)
    rows.append({"mode": "exact", "k": None, "time": t_exact, "missed": 0.0, "extra": 0.0})
    for k in (1, 2, 4, 8, 16):
        fixed = Dispatcher(city, cell_size=400.0, pickup_limit=1800, candidate_count=k,
                           widen_search=False)
        for i, s in enumerate(spots):
            fixed.add_driver(f"D{i}", s)
        got = [fixed.pick_driver(r) for r in reqs]
        t = best_of(lambda: [fixed.pick_driver(r) for r in reqs]) / len(reqs)
        worse = [(g[1] if g else 1800) - b[1] for g, b in zip(got, best) if b]
        rows.append({"mode": "fixed", "k": k, "time": t,
                     "missed": sum(w > 0 for w in worse) / len(worse),
                     "extra": sum(worse) / len(worse)})
    for r in rows:
        print(f"  {r['mode']:<5} k={r['k']!s:<4} {r['time'] * 1e6:7.1f} us  "
              f"suboptimal {r['missed']:5.1%}  extra pickup {r['extra']:5.1f} s")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    q = args.quick
    results = {"machine": f"{platform.machine()} {platform.system()} Python {platform.python_version()}",
               "quick": q}
    print("1. grid nearest-5 (constant density)")
    results["grid"] = bench_grid([1_000, 4_000, 16_000, 64_000] if q else
                                 [1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000, 128_000])
    results["cell_size"] = bench_cell_size()
    print("2. indexed heap, mixed operations")
    results["heap"] = bench_heap([1_000, 10_000, 100_000] if q else
                                 [1_000, 4_000, 16_000, 64_000, 256_000])
    print("3. age_requests, one round")
    results["aging"] = bench_aging([1_000, 8_000, 64_000] if q else
                                   [500, 1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000])
    print("4. pick_driver on a river city")
    results["search"] = bench_search([20, 40, 80] if q else [20, 40, 60, 80, 120, 160])
    results["search_open"] = bench_search([40, 120] if q else [20, 40, 80, 160], river=False)
    print("4b. trip-duration routes")
    results["routes"] = bench_routes([40, 80] if q else [20, 40, 80, 120, 160])
    print("5. whole simulation")
    results["sim"] = bench_sim([20, 40, 60] if q else [20, 40, 60, 80, 100, 120], busy=False)
    results["sim_busy"] = bench_sim([20, 40] if q else [20, 40, 60, 80, 100], busy=True)
    print("6. memory")
    results["memory"] = bench_memory([20, 40] if q else [20, 40, 60, 80, 100])
    results["long_shift"] = bench_long_shift()
    results["record_bytes"] = bench_object_memory()
    print("7. accuracy vs speed")
    results["tradeoff"] = bench_tradeoff()
    os.makedirs(OUT, exist_ok=True)
    name = "benchmarks_quick.json" if q else "benchmarks.json"
    with open(os.path.join(OUT, name), "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"saved results/{name}")


if __name__ == "__main__":
    main()
