# MSCS532 Project: Real-Time Ride-Hailing Dispatch

When a rider taps "request", a ride-hailing app has a few seconds to pick
which free driver should go. This project builds the data structures behind
that choice in plain Python.

* **Phase 1** was the design (`Deliverable1_Report.docx`; the original code
  is in `MSCS532_Project_Code.zip`).
* **Phase 2** turned that code into a package and a working proof of
  concept: an exact driver search, a queue that one unreachable rider can't
  block, tier bumps, driver states, a simulation, a demo and 51 tests
  (`Deliverable2_Report.docx`). A frozen copy lives in `dispatch/phase2/`
  so Phase 3 can be compared against it.
* **Phase 3** (this version) optimises and scales everything, adds stress
  tests and benchmarks, and gives the same matches as Phase 2 on the same
  workload (`Deliverable3_Report.docx`).

## Layout

```
dispatch/
  spatial_grid.py    hash grid of free drivers, nearest-k search
  indexed_heap.py    binary min-heap with a key index, for waiting requests
  road_network.py    directed weighted road graph, Dijkstra, speed bound
  models.py          Driver, TripRequest, Assignment, DriverState, TIER_PRIORITY
  dispatcher.py      Dispatcher: ties the structures together
  city_builder.py    made-up street grids (and rivers) for the demo and tests
  simulation.py      event-driven simulation (streaming arrivals, hotspots,
                     road incidents, driver shift changes, either engine)
  phase2/            frozen Phase 2 code, the baseline for comparisons
demo.py              command-line demo of every scenario
benchmarks/
  run_benchmarks.py  Phase 2 vs Phase 3 timings and memory -> results/benchmarks.json
  make_figures.py    draws figures/*.png from that JSON (needs matplotlib)
tests/               unittest suite: one file per module, plus
  test_phase3.py     new code vs Phase 2 vs brute force on random input
  test_stress.py     large inputs, extreme conditions, bad input (STRESS=1 = 10x)
results/             saved demo, test and benchmark output
figures/             benchmark graphs used in the report
screenshots/         terminal captures
```

| Module | Data structure | What it does |
| --- | --- | --- |
| `spatial_grid.py` | Dict of grid cells → sets of ids | Tracks free drivers and finds the k closest to a pickup |
| `indexed_heap.py` | Binary min-heap + position dict | Orders waiting requests by (tier, request time); remove and re-prioritise in O(log n) |
| `road_network.py` | Adjacency dicts (in and out edges) | Drive times with Dijkstra; one backward search ranks several drivers |
| `dispatcher.py` | Dicts of drivers, requests, trips + the above | Gives each request to the driver who can get there first |

## What changed in Phase 3

| Bottleneck in Phase 2 | Phase 3 fix | Effect |
| --- | --- | --- |
| `SpatialGrid.nearest` scanned every occupied cell on each query, then sorted every point it saw | bounding box of occupied cells (O(1) amortised), fall back to a direct scan when rings would be mostly empty, keep only possible top-k points | 165x faster at 128,000 drivers |
| `pick_driver` doubled k and restarted the grid query and Dijkstra each time | one backward Dijkstra that stops at the first intersection with a free driver (`RoadNetwork.nearest_source`) + per-intersection free-driver index | 101 → 14 intersections settled, 17x faster (160x160 river city) |
| unreachable requests re-ran a full search every round | straight-line "anyone in range?" check, and a dead-end cache that only resets when a driver frees up / moves or a street changes | busy 100x100 simulation 6.9 s → 3.1 s |
| `age_requests` looked at every waiting request, O(n log n) per round | heap of next-bump times with lazy deletion; O(p log n) for p bumps | 123x faster at 64,000 waiting |
| heap compared list slices and swapped entries | tuple entries, hole-based sifting, `push_many` with O(n) heapify | 1.5-2.2x faster per operation |
| trip routes used plain Dijkstra | A* with straight-line bound or ALT landmark tables, ties broken towards larger g, LRU route cache cleared on any street change | 14x faster than Dijkstra on 160x160 |
| all requests generated up front onto the event heap | streaming sorted-uniform arrivals, `array('d')` waits, slotted records | 1 → 8 h shift: +0.7 MiB vs +2.0 MiB |
| bumped standard riders overtook accessible riders | queue key (level, original tier, time); `keep_tier_order=False` restores Phase 2 | fairness fix |

Whole simulations run 2-14x faster than Phase 2 and, on 10 of 11 benchmark
runs, make identical matches (the 11th differs only in which of two drivers
tied at 150 s gets picked). The cost is about 31% more peak memory for the
extra indexes and landmark tables. 105 tests pass; see `results/`.

## What changed in Phase 2

| Problem in Phase 1 | What Phase 2 does |
| --- | --- |
| Only the k closest drivers in a straight line were checked, so a quicker driver by road could be missed | `pick_driver` doubles k until `RoadNetwork.lower_bound_time` shows nobody further away could be faster |
| `assign_top` gave up at the first request nobody could reach | `run_round` skips it, serves the rest, and puts it back with the same priority |
| Standard riders could wait forever behind premium ones | `age_requests` bumps a request up one tier every `promote_after` seconds using `change_priority` |
| Drivers were only free or busy | `DriverState` adds OFFLINE; `go_offline` / `go_online`; trips kept in `trips_in_progress` |

## Requirements

Python 3.10 or newer (slotted dataclasses). No third-party packages for the
package, demo and tests; `benchmarks/make_figures.py` needs matplotlib.

## Running

```
python3 demo.py                 # every scenario
python3 demo.py river           # one of: basic | river | blocking | aging | simulate
python3 demo.py simulate --drivers 40 --requests 300 --seed 7
python3 demo.py simulate --size 60 --drivers 540 --requests 900 --busy
python3 demo.py compare --size 60 --drivers 540 --requests 900   # Phase 2 vs Phase 3
python3 benchmarks/run_benchmarks.py            # ~15 min; --quick for a short run
python3 benchmarks/make_figures.py
```

Tests:

```
python3 -m unittest discover -s tests -t . -v
STRESS=1 python3 -m unittest tests.test_stress -v     # stress tests at 10x size
```

A lot of the tests compare the fast code with a slow version that is easy
to trust: the grid against checking every point, the heap against a plain
dict after thousands of random operations, and the driver search against
running Dijkstra from every driver on 40 random cities with closed and
slowed streets. There is also a test that does 1,500 random dispatcher
operations and checks the grid, queue and trip records still agree after
each one. Edge cases covered include empty structures, duplicate ids,
pickups nobody can reach, closed streets, offline drivers and bad arguments.
