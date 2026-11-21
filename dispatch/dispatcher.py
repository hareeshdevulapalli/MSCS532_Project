"""The dispatcher: matches waiting riders with free drivers.

How the pieces fit together:

* ``drivers``, ``waiting_requests`` and ``trips_in_progress`` are plain
  dictionaries keyed by id, so finding any record is O(1) on average.
* ``SpatialGrid`` only holds drivers who are free right now and answers
  "who is closest to this pickup in a straight line?".
* ``_free_at`` maps an intersection to the set of free drivers standing
  there. The backward road search checks it to know when it has hit a
  driver.
* ``IndexedMinHeap`` orders waiting requests and lets a rider cancel, or a
  request be bumped, in O(log n).
* ``RoadNetwork`` turns everything into real drive times.

Phase 2 recap: the search is exact (it never misses a quicker driver behind
a river), one unreachable rider can't block the queue, old requests move up
a tier, and drivers can go offline.

What changed in Phase 3
-----------------------
1. One search per request instead of a doubling loop. ``pick_driver`` runs a
   single backward Dijkstra from the pickup and stops at the first
   intersection with a free driver (``RoadNetwork.nearest_source``). Phase 2
   asked the grid for k drivers, searched until it had reached all of them,
   and started over with 2k if the speed bound said someone further away
   might still win.
2. A cheap "anyone in range?" check first. Before searching the roads, the
   grid is asked for the closest free driver in a straight line within
   ``pickup_limit * max_speed``. If there isn't one, nobody can make it in
   time, so the road search is skipped.
3. Remembering dead ends (memoisation). When nobody can reach a pickup, the
   pickup is stored in ``_dead_ends`` (a dict) and in ``_dead_end_grid`` (a
   second SpatialGrid). A driver getting busy, or a street closing, can only
   make things worse, so neither touches the cache. When a driver becomes
   free (or moves), only the dead ends within that driver's straight-line
   reach (``pickup_limit * max_speed``) are forgotten, found with one range
   query on the grid. Opening or speeding up a street clears it all. Phase 2
   re-ran the full search for every stuck request in every round.
4. A second heap for bumps. ``age_requests`` used to look at every waiting
   request each round (O(n log n)). Now a heap ordered by "next bump time"
   is popped only for requests that are actually due, so a round costs
   O(p log n) for p bumps. Entries for riders who already left are skipped
   when they come up, and the heap is rebuilt if they pile up.
5. Fairer order after bumps (Phase 2 finding). The queue key is now
   (level, original tier, request time), so a bumped standard rider no
   longer jumps ahead of an accessible rider who has the same level.
   ``keep_tier_order=False`` gives the Phase 2 key back for comparisons.
"""

import heapq
import itertools
import math

from .indexed_heap import IndexedMinHeap
from .models import TIER_PRIORITY, Assignment, Driver, DriverState, TripRequest
from .road_network import RoadNetwork
from .spatial_grid import SpatialGrid


class Dispatcher:
    """Gives each waiting rider to the free driver who can reach them first."""

    def __init__(self, roads, cell_size=500.0, candidate_count=5,
                 pickup_limit=900, promote_after=None, widen_search=True,
                 keep_tier_order=True, remember_dead_ends=True):
        if not isinstance(roads, RoadNetwork):
            raise TypeError("roads has to be a RoadNetwork")
        if candidate_count < 1:
            raise ValueError("candidate_count has to be at least 1")
        if not pickup_limit > 0:
            raise ValueError("pickup_limit has to be positive")
        if promote_after is not None and not promote_after > 0:
            raise ValueError("promote_after has to be positive (or None)")
        self.roads = roads
        self.candidate_count = candidate_count
        self.pickup_limit = pickup_limit
        self.promote_after = promote_after
        self.widen_search = widen_search          # False = Phase 1 fixed list
        self.keep_tier_order = keep_tier_order
        self.remember_dead_ends = remember_dead_ends
        self.drivers = {}             # driver id -> Driver
        self.waiting_requests = {}    # request id -> TripRequest not yet assigned
        self.trips_in_progress = {}   # driver id -> TripRequest they are serving
        self.stats = {"searches": 0, "settled": 0, "precheck_skips": 0,
                      "dead_end_hits": 0, "skipped": 0, "promotions": 0}
        self._free_grid = SpatialGrid(cell_size)
        self._free_at = {}            # intersection -> set of free driver ids there
        self._queue = IndexedMinHeap()
        self._dead_ends = {}          # pickup -> roads.open_version when nobody could reach it
        self._dead_end_grid = SpatialGrid(cell_size)   # the same pickups, by position
        self._bumps_due = []          # heap of (due time, n, request id, bumps when scheduled)
        self._bump_counter = itertools.count()

    # ----- free-driver bookkeeping (grid + per-intersection index) -------

    def _add_free(self, driver):
        self._free_grid.insert(driver.driver_id, *self.roads.coords(driver.location))
        self._free_at.setdefault(driver.location, set()).add(driver.driver_id)
        self._forget_dead_ends_near(driver.location)

    def _forget_dead_ends_near(self, location):
        """A driver just became free here; pickups they might reach aren't dead ends now."""
        if not self._dead_ends:
            return
        x, y = self.roads.coords(location)
        for pickup in self._dead_end_grid.within(x, y, self.roads.reach(self.pickup_limit)):
            self._dead_end_grid.delete(pickup)
            del self._dead_ends[pickup]

    def _remove_free(self, driver):
        self._free_grid.delete(driver.driver_id)
        here = self._free_at[driver.location]
        here.discard(driver.driver_id)
        if not here:
            del self._free_at[driver.location]

    # ----- drivers -----------------------------------------------------

    def add_driver(self, driver_id, location):
        if driver_id in self.drivers:
            raise KeyError(f"driver {driver_id!r} is already registered")
        self._require_node(location)
        driver = Driver(driver_id, location)
        self.drivers[driver_id] = driver
        self._add_free(driver)

    def move_driver(self, driver_id, location):
        """GPS update. If the driver is free, the grid and index move too."""
        self._require_node(location)
        driver = self._get_driver(driver_id)
        if driver.is_free:
            self._remove_free(driver)
            driver.location = location
            self._add_free(driver)
        else:
            driver.location = location

    def go_offline(self, driver_id):
        """Log a free driver off. Someone on a trip has to finish it first."""
        driver = self._get_driver(driver_id)
        if driver.state is DriverState.BUSY:
            raise ValueError(f"driver {driver_id!r} is in the middle of a trip")
        if driver.is_free:
            self._remove_free(driver)
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
        self._add_free(driver)

    def complete_trip(self, driver_id):
        """Rider dropped off; the driver becomes free at the drop-off point."""
        driver = self._get_driver(driver_id)
        if driver.state is not DriverState.BUSY:
            raise ValueError(f"driver {driver_id!r} isn't on a trip")
        trip = self.trips_in_progress.pop(driver_id)
        driver.location = trip.dropoff
        driver.state = DriverState.FREE
        driver.trips_done += 1
        self._add_free(driver)
        return trip

    def count_free_drivers(self):
        return len(self._free_grid)

    # ----- requests ----------------------------------------------------

    def _priority(self, req):
        if self.keep_tier_order:
            return (req.level, TIER_PRIORITY[req.tier], req.requested_at)
        return (req.level, req.requested_at)

    def _schedule_bump(self, req):
        if self.promote_after is None or req.bumps >= req.max_bumps:
            return
        due = req.requested_at + (req.bumps + 1) * self.promote_after
        heapq.heappush(self._bumps_due, (due, next(self._bump_counter), req.request_id, req.bumps))

    def new_request(self, request_id, pickup, dropoff, tier, requested_at):
        if request_id in self.waiting_requests:
            raise KeyError(f"request {request_id!r} already exists")
        if tier not in TIER_PRIORITY:
            raise ValueError(f"unknown tier {tier!r}")
        if not math.isfinite(requested_at):
            raise ValueError("requested_at has to be a finite number")
        self._require_node(pickup)
        self._require_node(dropoff)
        req = TripRequest(request_id, pickup, dropoff, tier, requested_at)
        self.waiting_requests[request_id] = req
        self._queue.push(request_id, self._priority(req))
        self._schedule_bump(req)

    def cancel_request(self, request_id):
        if request_id not in self.waiting_requests:
            raise KeyError(f"no waiting request {request_id!r}")
        self._queue.remove(request_id)
        del self.waiting_requests[request_id]
        # Its entry in _bumps_due is left behind and skipped later (lazy delete).

    def count_waiting(self):
        return len(self._queue)

    def age_requests(self, now):
        """Bump each request up one tier for every ``promote_after`` seconds waited.

        Only requests whose next bump is due get looked at, so this is
        O(p log n) for p bumps instead of Phase 2's O(n log n). Returns how
        many requests moved.
        """
        if self.promote_after is None:
            return 0
        due_heap, waiting = self._bumps_due, self.waiting_requests
        moved, recheck = 0, []
        # The heap says "due by time", but the bump itself is decided with the
        # same formula as Phase 2 so rounding can never make them disagree.
        slack = 1e-9 * max(1.0, abs(now))
        while due_heap and due_heap[0][0] <= now + slack:
            entry = heapq.heappop(due_heap)
            _, _, request_id, bumps_then = entry
            req = waiting.get(request_id)
            if req is None or req.bumps != bumps_then:
                continue   # rider left or got matched; nothing to do
            earned = min(int((now - req.requested_at) // self.promote_after), req.max_bumps)
            if earned <= req.bumps:
                recheck.append(entry)   # due only by a rounding hair; look again next time
                continue
            req.bumps = earned
            self._queue.change_priority(request_id, self._priority(req))
            moved += 1
            self._schedule_bump(req)
        for entry in recheck:
            heapq.heappush(due_heap, entry)
        # Lazy deletes leave old entries behind; rebuild if they outnumber live ones.
        if len(due_heap) > 2 * len(waiting) + 64:
            self._bumps_due = []
            for req in waiting.values():
                self._schedule_bump(req)
        self.stats["promotions"] += moved
        return moved

    # ----- matching ----------------------------------------------------

    def pick_driver(self, req):
        """Return (driver_id, pickup_time) for the quickest free driver, or None.

        Drivers who would take longer than ``pickup_limit`` don't count. Ties
        on time go to the smallest driver id. With ``widen_search=False`` it
        only checks the ``candidate_count`` closest drivers in a straight
        line, which is how Phase 1 worked.
        """
        if not self._free_at:
            return None
        if not self.widen_search:
            return self._pick_from_fixed_list(req)
        roads, pickup = self.roads, req.pickup
        if self.remember_dead_ends and pickup in self._dead_ends:
            if self._dead_ends[pickup] == roads.open_version:
                self.stats["dead_end_hits"] += 1
                return None
            self._forget_all_dead_ends()          # a street opened since; start over
        # Straight-line check: is any free driver close enough to possibly make it?
        px, py = roads.coords(pickup)
        if not self._free_grid.nearest(px, py, k=1, max_distance=roads.reach(self.pickup_limit)):
            self.stats["precheck_skips"] += 1
            self._remember_dead_end(pickup)
            return None
        seconds, spots = roads.nearest_source(pickup, self._free_at, limit=self.pickup_limit)
        self.stats["searches"] += 1
        self.stats["settled"] += roads.last_settled
        if seconds is None:
            self._remember_dead_end(pickup)
            return None
        driver_id = min(min(self._free_at[spot]) for spot in spots)
        return driver_id, seconds

    def _remember_dead_end(self, pickup):
        if not self.remember_dead_ends:
            return
        if pickup not in self._dead_ends:
            self._dead_end_grid.insert(pickup, *self.roads.coords(pickup))
        self._dead_ends[pickup] = self.roads.open_version

    def _forget_all_dead_ends(self):
        self._dead_ends = {}
        self._dead_end_grid = SpatialGrid(self._dead_end_grid.cell_size)

    def _pick_from_fixed_list(self, req):
        px, py = self.roads.coords(req.pickup)
        candidates = self._free_grid.nearest(px, py, k=self.candidate_count)
        by_spot = {}
        for _, driver_id in candidates:
            by_spot.setdefault(self.drivers[driver_id].location, []).append(driver_id)
        times = self.roads.times_to_target(req.pickup, by_spot, limit=self.pickup_limit)
        self.stats["searches"] += 1
        self.stats["settled"] += self.roads.last_settled
        if not times:
            return None
        spot = min(times, key=lambda s: (times[s], min(by_spot[s])))
        return min(by_spot[spot]), times[spot]

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
        queue, free_at, waiting = self._queue, self._free_at, self.waiting_requests
        while queue and free_at:
            request_id, priority = queue.pop()
            req = waiting[request_id]
            found = self.pick_driver(req)
            if found is None:
                skipped.append((request_id, priority))
                continue
            matches.append(self._make_match(req, *found, now))
        queue.push_many(skipped)
        self.stats["skipped"] += len(skipped)
        return matches

    # ----- helpers -----------------------------------------------------

    def _make_match(self, req, driver_id, pickup_time, now):
        """Record a match. The request has to be out of the queue already."""
        driver = self.drivers[driver_id]
        self._remove_free(driver)
        driver.state = DriverState.BUSY
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
