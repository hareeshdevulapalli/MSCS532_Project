"""The dispatcher: matches waiting riders with free drivers.

How the pieces fit together:

* ``drivers``, ``waiting_requests`` and ``trips_in_progress`` are plain
  dictionaries keyed by id, so finding any record is O(1) on average.
* ``SpatialGrid`` only holds drivers who are free right now and answers
  "who is closest to this pickup?".
* ``IndexedMinHeap`` orders waiting requests by (tier level, request time)
  and lets a rider cancel, or a request be bumped, in O(log n).
* ``RoadNetwork`` turns the straight-line candidates into real drive times.

What I changed in Phase 2
-------------------------
1. Picking the right driver. In Phase 1 I only looked at the k drivers
   closest in a straight line, so a driver a bit further away but much
   quicker by road could be missed. ``pick_driver`` now keeps doubling k
   until the road network's speed bound shows nobody outside the
   candidate list could get there faster.
2. One stuck request no longer holds up the queue. Phase 1's ``assign_top``
   gave up as soon as the first request had no driver in range.
   ``run_round`` puts those requests to one side, serves everyone else, and
   then pushes the skipped ones back with the same priority they had.
3. Bumping old requests. With fixed tiers a standard rider could wait
   forever behind premium riders. ``age_requests`` moves a request up one
   tier for every ``promote_after`` seconds it has waited, using the heap's
   O(log n) ``change_priority``.
4. Driver states. Drivers can log off and back on, and every trip is kept
   in ``trips_in_progress`` from the moment it is assigned until
   ``complete_trip`` is called.
"""

import math

from .indexed_heap import IndexedMinHeap
from .models import TIER_PRIORITY, Assignment, Driver, DriverState, TripRequest
from .road_network import RoadNetwork
from .spatial_grid import SpatialGrid


class Dispatcher:
    """Gives each waiting rider to the free driver who can reach them first."""

    def __init__(self, roads, cell_size=500.0, candidate_count=5,
                 pickup_limit=900, promote_after=None, widen_search=True):
        if not isinstance(roads, RoadNetwork):
            raise TypeError("roads has to be a RoadNetwork")
        if candidate_count < 1:
            raise ValueError("candidate_count has to be at least 1")
        if pickup_limit <= 0:
            raise ValueError("pickup_limit has to be positive")
        if promote_after is not None and promote_after <= 0:
            raise ValueError("promote_after has to be positive (or None)")
        self.roads = roads
        self.candidate_count = candidate_count
        self.pickup_limit = pickup_limit
        self.promote_after = promote_after
        self.widen_search = widen_search
        self.drivers = {}             # driver id -> Driver
        self.waiting_requests = {}    # request id -> TripRequest not yet assigned
        self.trips_in_progress = {}   # driver id -> TripRequest they are serving
        self.stats = {"searches": 0, "widenings": 0, "skipped": 0, "promotions": 0}
        self._free_grid = SpatialGrid(cell_size)
        self._queue = IndexedMinHeap()

    # ----- drivers -----------------------------------------------------

    def add_driver(self, driver_id, location):
        if driver_id in self.drivers:
            raise KeyError(f"driver {driver_id!r} is already registered")
        self._require_node(location)
        self.drivers[driver_id] = Driver(driver_id, location)
        self._free_grid.insert(driver_id, *self.roads.coords(location))

    def move_driver(self, driver_id, location):
        """GPS update. If the driver is free, their grid cell is updated too."""
        self._require_node(location)
        driver = self._get_driver(driver_id)
        driver.location = location
        if driver.is_free:
            self._free_grid.move(driver_id, *self.roads.coords(location))

    def go_offline(self, driver_id):
        """Log a free driver off. Someone on a trip has to finish it first."""
        driver = self._get_driver(driver_id)
        if driver.state is DriverState.BUSY:
            raise ValueError(f"driver {driver_id!r} is in the middle of a trip")
        if driver.is_free:
            self._free_grid.delete(driver_id)
        driver.state = DriverState.OFFLINE

    def go_online(self, driver_id, location=None):
        """Bring an offline driver back, at a new spot if one is given."""
        driver = self._get_driver(driver_id)
        if driver.state is not DriverState.OFFLINE:
            raise ValueError(f"driver {driver_id!r} isn't offline")
        if location is not None:
            self._require_node(location)
            driver.location = location
        driver.state = DriverState.FREE
        self._free_grid.insert(driver_id, *self.roads.coords(driver.location))

    def complete_trip(self, driver_id):
        """Rider dropped off; the driver becomes free at the drop-off point."""
        driver = self._get_driver(driver_id)
        if driver.state is not DriverState.BUSY:
            raise ValueError(f"driver {driver_id!r} isn't on a trip")
        trip = self.trips_in_progress.pop(driver_id)
        driver.location = trip.dropoff
        driver.state = DriverState.FREE
        driver.trips_done += 1
        self._free_grid.insert(driver_id, *self.roads.coords(driver.location))
        return trip

    def count_free_drivers(self):
        return len(self._free_grid)

    # ----- requests ----------------------------------------------------

    def new_request(self, request_id, pickup, dropoff, tier, requested_at):
        if request_id in self.waiting_requests:
            raise KeyError(f"request {request_id!r} already exists")
        if tier not in TIER_PRIORITY:
            raise ValueError(f"unknown tier {tier!r}")
        self._require_node(pickup)
        self._require_node(dropoff)
        req = TripRequest(request_id, pickup, dropoff, tier, requested_at)
        self.waiting_requests[request_id] = req
        self._queue.push(request_id, (req.level, requested_at))

    def cancel_request(self, request_id):
        if request_id not in self.waiting_requests:
            raise KeyError(f"no waiting request {request_id!r}")
        self._queue.remove(request_id)
        del self.waiting_requests[request_id]

    def count_waiting(self):
        return len(self._queue)

    def age_requests(self, now):
        """Bump each request up one tier for every ``promote_after`` seconds waited.

        Returns how many requests moved. In this version it looks at every
        waiting request, so it is O(n log n) (see next steps in the report).
        """
        if self.promote_after is None:
            return 0
        moved = 0
        for request_id in self._queue.all_keys():
            req = self.waiting_requests[request_id]
            earned = int((now - req.requested_at) // self.promote_after)
            earned = min(earned, TIER_PRIORITY[req.tier])   # can't go past level 0
            if earned > req.bumps:
                req.bumps = earned
                self._queue.change_priority(request_id, (req.level, req.requested_at))
                moved += 1
        self.stats["promotions"] += moved
        return moved

    # ----- matching ----------------------------------------------------

    def pick_driver(self, req):
        """Return (driver_id, pickup_time) for the quickest free driver, or None.

        Drivers who would take longer than ``pickup_limit`` don't count.
        With ``widen_search=False`` it only checks the first candidate list,
        which is how Phase 1 worked.
        """
        px, py = self.roads.coords(req.pickup)
        free_count = len(self._free_grid)
        k = min(self.candidate_count, free_count)
        while k > 0:
            candidates = self._free_grid.nearest(px, py, k=k)
            by_spot = {}
            for _, driver_id in candidates:
                by_spot.setdefault(self.drivers[driver_id].location, []).append(driver_id)
            times = self.roads.times_to_target(req.pickup, by_spot, limit=self.pickup_limit)
            self.stats["searches"] += 1

            best = None
            if times:
                spot = min(times, key=lambda s: (times[s], min(by_spot[s])))
                best = (min(by_spot[spot]), times[spot])

            if not self.widen_search or k >= free_count:
                return best
            # Every driver we haven't checked is at least this far away in a
            # straight line, so they need at least ``floor`` seconds by road.
            floor = self.roads.lower_bound_time(candidates[-1][0])
            if floor > self.pickup_limit:
                return best
            if best is not None and best[1] <= floor:
                return best
            k = min(2 * k, free_count)
            self.stats["widenings"] += 1
        return None

    def assign_top(self, now=None):
        """Try to match only the request at the front (the Phase 1 way)."""
        if not self._queue:
            return None
        request_id, _ = self._queue.peek()
        req = self.waiting_requests[request_id]
        found = self.pick_driver(req)
        if found is None:
            return None
        self._queue.pop()
        return self._make_match(req, *found, now)

    def run_round(self, now):
        """Match as many waiting requests as we can, most urgent first.

        Requests that nobody can reach are put aside instead of blocking the
        rest, then pushed back at the end with their priority unchanged.
        """
        self.age_requests(now)
        matches, skipped = [], []
        while self._queue and self._free_grid:
            request_id, priority = self._queue.pop()
            req = self.waiting_requests[request_id]
            found = self.pick_driver(req)
            if found is None:
                skipped.append((request_id, priority))
                continue
            matches.append(self._make_match(req, *found, now))
        for request_id, priority in skipped:
            self._queue.push(request_id, priority)
        self.stats["skipped"] += len(skipped)
        return matches

    # ----- helpers -----------------------------------------------------

    def _make_match(self, req, driver_id, pickup_time, now):
        """Record a match. The request has to be out of the queue already."""
        driver = self.drivers[driver_id]
        driver.state = DriverState.BUSY
        self._free_grid.delete(driver_id)
        del self.waiting_requests[req.request_id]
        self.trips_in_progress[driver_id] = req
        waited = 0.0 if now is None else now - req.requested_at
        return Assignment(req.request_id, driver_id, pickup_time, waited)

    def _get_driver(self, driver_id):
        try:
            return self.drivers[driver_id]
        except KeyError:
            raise KeyError(f"unknown driver {driver_id!r}") from None

    def _require_node(self, node):
        if node not in self.roads:
            raise KeyError(f"unknown intersection {node!r}")


def slow_best_time(dispatcher, req):
    """Answer key for the tests: drive time of the best free driver (or inf).

    It runs a full shortest_path from every free driver. Much too slow for
    real use, but easy to trust.
    """
    best = math.inf
    for driver in dispatcher.drivers.values():
        if driver.is_free:
            seconds, _ = dispatcher.roads.shortest_path(driver.location, req.pickup)
            if seconds <= dispatcher.pickup_limit:
                best = min(best, seconds)
    return best
