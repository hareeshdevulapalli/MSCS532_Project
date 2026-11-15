# MSCS532 Project: Real-Time Ride-Hailing Dispatch

When a rider taps "request", a ride-hailing app has a few seconds to pick
which free driver should go. This project builds the data structures behind
that choice in plain Python.

* **Phase 1** was the design (`Deliverable1_Report.docx`; the original code
  is in `MSCS532_Project_Code.zip`).
* **Phase 2** (this version) turns that code into a package and adds a
  working proof of concept: a driver search that is guaranteed to find the
  quickest driver, a queue that one unreachable rider can't block, tier
  bumps for riders who wait too long, drivers going on and offline, a small
  simulation, a command-line demo and 51 unit tests
  (`Deliverable2_Report.docx`).

## Layout

```
dispatch/
  spatial_grid.py    hash grid of free drivers, nearest-k search
  indexed_heap.py    binary min-heap with a key index, for waiting requests
  road_network.py    directed weighted road graph, Dijkstra, speed bound
  models.py          Driver, TripRequest, Assignment, DriverState, TIER_PRIORITY
  dispatcher.py      Dispatcher: ties the structures together
  city_builder.py    made-up street grids (and rivers) for the demo and tests
  simulation.py      event-driven simulation of one hour of requests
demo.py              command-line demo of every scenario
tests/               unittest suite, one file per module
results/             saved demo and test output
screenshots/         terminal captures of the demo, simulation and test runs
```

| Module | Data structure | What it does |
| --- | --- | --- |
| `spatial_grid.py` | Dict of grid cells → sets of ids | Tracks free drivers and finds the k closest to a pickup |
| `indexed_heap.py` | Binary min-heap + position dict | Orders waiting requests by (tier, request time); remove and re-prioritise in O(log n) |
| `road_network.py` | Adjacency dicts (in and out edges) | Drive times with Dijkstra; one backward search ranks several drivers |
| `dispatcher.py` | Dicts of drivers, requests, trips + the above | Gives each request to the driver who can get there first |

## What changed since Phase 1

| Problem in Phase 1 | What Phase 2 does |
| --- | --- |
| Only the k closest drivers in a straight line were checked, so a quicker driver by road could be missed | `pick_driver` doubles k until `RoadNetwork.lower_bound_time` shows nobody further away could be faster |
| `assign_top` gave up at the first request nobody could reach | `run_round` skips it, serves the rest, and puts it back with the same priority |
| Standard riders could wait forever behind premium ones | `age_requests` bumps a request up one tier every `promote_after` seconds using `change_priority` |
| Drivers were only free or busy | `DriverState` adds OFFLINE; `go_offline` / `go_online`; trips kept in `trips_in_progress` |

## Requirements

Python 3.8 or newer. No third-party packages.

## Running

```
python3 demo.py                 # every scenario
python3 demo.py river           # one of: basic | river | blocking | aging | simulate
python3 demo.py simulate --drivers 40 --requests 300 --seed 7
```

Tests:

```
python3 -m unittest discover -s tests -t . -v
```

A lot of the tests compare the fast code with a slow version that is easy
to trust: the grid against checking every point, the heap against a plain
dict after thousands of random operations, and the driver search against
running Dijkstra from every driver on 40 random cities with closed and
slowed streets. There is also a test that does 1,500 random dispatcher
operations and checks the grid, queue and trip records still agree after
each one. Edge cases covered include empty structures, duplicate ids,
pickups nobody can reach, closed streets, offline drivers and bad arguments.
