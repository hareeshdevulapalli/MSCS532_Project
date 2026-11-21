"""Event-driven simulation of one dispatcher shift (Phase 3 version).

Every event (a request coming in, a rider giving up, a trip ending, a
dispatch round, a road closing or opening, a driver logging off or on) goes
into a binary heap ordered by time, so getting the next one is O(log n).
Dispatch runs in rounds every few seconds, which is roughly how real
systems batch their decisions.

What changed in Phase 3
-----------------------
1. Streaming arrivals. Phase 2 created every request up front and pushed
   them all onto the event heap, so memory grew with the length of the
   shift. Now only the *next* arrival is on the heap. Arrival times are
   drawn one at a time as sorted uniform values (each new time is the
   minimum of the uniforms still to come), which gives exactly ``requests``
   arrivals spread evenly over ``duration`` using O(1) memory for the
   arrival stream. The heap only holds events for riders and trips that are
   live right now.
2. Ride times come from ``RoadNetwork.travel_time`` (A* with landmark
   bounds, plus an LRU cache) instead of a fresh Dijkstra for every trip.
   Building the landmark tables is timed separately as ``setup``.
3. Harder scenarios: rush-hour hotspots (a share of pickups near a few busy
   corners), road incidents (random streets close for a while and reopen),
   and driver shift changes (free drivers log off and come back later).
4. Either dispatcher can be plugged in (``engine="phase3"`` or
   ``"phase2"``), so both can be timed on exactly the same workload.
5. Wait times are kept in a compact ``array('d')``, and the list of
   Assignment objects is only kept if ``keep_matches`` is on.
"""

import heapq
import itertools
import math
import random
import statistics
import time
from array import array
from dataclasses import dataclass, field

from . import phase2
from .city_builder import add_river_barrier, build_grid_city
from .dispatcher import Dispatcher
from .road_network import RoadNetwork

NEW_REQUEST, RIDER_LEAVES, TRIP_ENDS, ROUND = "new_request", "rider_leaves", "trip_ends", "round"
ROAD_CLOSES, ROAD_OPENS, DRIVER_OFF, DRIVER_ON = "road_closes", "road_opens", "driver_off", "driver_on"
TIER_SHARE = (("accessible", 0.05), ("premium", 0.20), ("standard", 0.75))


@dataclass
class SimSettings:
    cols: int = 20
    rows: int = 20
    drivers: int = 60
    requests: int = 300
    duration: float = 3600.0        # requests come in during this many seconds
    round_every: float = 5.0        # seconds between dispatch rounds
    patience: float = 600.0         # a rider cancels after waiting this long
    promote_after: float = 180.0
    candidate_count: int = 5
    widen_search: bool = True
    river: bool = True              # river down the middle with two bridges
    seed: int = 42
    cell_size: float = 400.0
    # Phase 3 additions -------------------------------------------------
    engine: str = "phase3"          # "phase3" or "phase2" dispatcher
    keep_tier_order: bool = True    # Phase 3 queue key (ignored by phase2)
    hotspot_share: float = 0.0      # share of pickups near a hotspot corner
    hotspots: int = 3
    incidents: int = 0              # streets that close for a while during the shift
    incident_length: float = 900.0
    shift_changes: int = 0          # free drivers that log off and come back
    keep_matches: bool = True
    landmarks: int = 8              # 0 = plain A* with the straight-line bound


@dataclass
class SimResult:
    requests: int
    served: int
    cancelled: int
    still_waiting: int
    avg_wait: float
    p95_wait: float
    avg_pickup: float
    wait_per_tier: dict
    counters: dict
    elapsed: float
    setup: float = 0.0
    events: int = 0
    peak_events: int = 0
    matches: list = field(default_factory=list, repr=False)

    def report_lines(self):
        lines = [
            f"requests            {self.requests}",
            f"served              {self.served} ({100 * self.served / max(self.requests, 1):.1f}%)",
            f"rider cancelled     {self.cancelled}",
            f"still waiting       {self.still_waiting}",
            f"average wait        {self.avg_wait:.1f} s",
            f"95th pct wait       {self.p95_wait:.1f} s",
            f"average pickup      {self.avg_pickup:.1f} s",
        ]
        for tier, wait in self.wait_per_tier.items():
            lines.append(f"  {tier + ' wait':<18}{wait:.1f} s")
        c = self.counters
        lines.append(f"road searches       {c.get('searches', 0)}"
                     f" ({c.get('settled', 0)} intersections settled)")
        if "dead_end_hits" in c:
            lines.append(f"dead ends skipped   {c['dead_end_hits'] + c['precheck_skips']}"
                         f" (cache {c['dead_end_hits']}, straight-line check {c['precheck_skips']})")
        lines += [
            f"skipped, retried    {c.get('skipped', 0)}",
            f"tier bumps          {c.get('promotions', 0)}",
            f"events handled      {self.events} (at most {self.peak_events} queued at once)",
            f"run time            {self.elapsed:.3f} s"
            + (f" (+ {self.setup:.3f} s building landmarks)" if self.setup else ""),
        ]
        return lines


def arrival_times(count, duration, rng):
    """Yield ``count`` sorted times uniform on [0, duration], one at a time.

    The smallest of m uniform values on [a, b] is a + (b - a)(1 - U^(1/m)),
    so each time can be drawn from the previous one without storing the rest.
    """
    now = 0.0
    for remaining in range(count, 0, -1):
        now += (duration - now) * (1.0 - rng.random() ** (1.0 / remaining))
        yield now


def build_city(settings, network_cls=RoadNetwork):
    city = build_grid_city(settings.cols, settings.rows, network_cls=network_cls)
    if settings.river and settings.cols > 1:
        middle = settings.cols // 2 - 1
        add_river_barrier(city, middle, settings.rows,
                          bridge_rows={settings.rows // 4, 3 * settings.rows // 4})
    return city


def make_dispatcher(settings, city):
    common = dict(cell_size=settings.cell_size, candidate_count=settings.candidate_count,
                  pickup_limit=settings.patience, promote_after=settings.promote_after,
                  widen_search=settings.widen_search)
    if settings.engine == "phase2":
        return phase2.Dispatcher(city, **common)
    if settings.engine == "phase3":
        return Dispatcher(city, keep_tier_order=settings.keep_tier_order, **common)
    raise ValueError(f"unknown engine {settings.engine!r}")


def simulate(settings=None):
    """Run one simulated shift and return a SimResult."""
    settings = settings or SimSettings()
    if settings.requests < 0 or settings.drivers < 0:
        raise ValueError("drivers and requests can't be negative")
    rng = random.Random(settings.seed)
    phase2_engine = settings.engine == "phase2"
    city = build_city(settings, phase2.RoadNetwork if phase2_engine else RoadNetwork)
    dispatcher = make_dispatcher(settings, city)
    corners = list(city.nodes())

    for n in range(settings.drivers):
        dispatcher.add_driver(f"D{n:05d}", rng.choice(corners))

    setup = 0.0
    if phase2_engine:
        def ride_time(a, b):
            return city.shortest_path(a, b)[0]
    else:
        ride_time = city.travel_time
        if settings.landmarks:
            t_setup = time.perf_counter()
            city.prepare_landmarks(settings.landmarks)
            setup = time.perf_counter() - t_setup

    hot = rng.sample(corners, min(settings.hotspots, len(corners))) if settings.hotspot_share else []

    def pick_pickup():
        if hot and rng.random() < settings.hotspot_share:
            col, row = rng.choice(hot)
            col = min(max(col + rng.randint(-2, 2), 0), settings.cols - 1)
            row = min(max(row + rng.randint(-2, 2), 0), settings.rows - 1)
            return (col, row)
        return rng.choice(corners)

    timeline, tiebreak = [], itertools.count()   # tiebreak keeps equal times in order
    peak = 0

    def schedule(at, kind, data=None):
        nonlocal peak
        heapq.heappush(timeline, (at, next(tiebreak), kind, data))
        if len(timeline) > peak:
            peak = len(timeline)

    tiers, shares = zip(*TIER_SHARE)
    arrivals = arrival_times(settings.requests, settings.duration, rng)
    next_number = itertools.count()

    def schedule_next_arrival():
        at = next(arrivals, None)
        if at is not None:
            schedule(at, NEW_REQUEST, next(next_number))

    schedule_next_arrival()
    schedule(0.0, ROUND)
    for _ in range(settings.incidents):
        schedule(rng.uniform(0, settings.duration), ROAD_CLOSES)
    for _ in range(settings.shift_changes):
        schedule(rng.uniform(0, settings.duration), DRIVER_OFF)

    tier_of, matches, per_tier = {}, [], {}   # per_tier: tier -> [total wait, count]
    waits, pickups = array("d"), array("d")
    cancelled = handled = 0
    t0 = time.perf_counter()
    while timeline:
        now, _, kind, data = heapq.heappop(timeline)
        handled += 1
        if kind == NEW_REQUEST:
            request_id = f"R{data:06d}"
            pickup = pick_pickup()
            dropoff = rng.choice(corners)
            while dropoff == pickup and len(corners) > 1:
                dropoff = rng.choice(corners)
            tier = rng.choices(tiers, shares)[0]
            dispatcher.new_request(request_id, pickup, dropoff, tier, now)
            tier_of[request_id] = tier
            schedule(now + settings.patience, RIDER_LEAVES, request_id)
            schedule_next_arrival()
        elif kind == RIDER_LEAVES:
            if data in dispatcher.waiting_requests:
                dispatcher.cancel_request(data)
                cancelled += 1
                tier_of.pop(data, None)
        elif kind == TRIP_ENDS:
            dispatcher.complete_trip(data)
        elif kind == ROAD_CLOSES:
            a = rng.choice(corners)
            open_streets = [b for b in city._out_edges[a] if a in city._out_edges[b]]
            if open_streets:
                b = rng.choice(open_streets)
                seconds = city.street_time(a, b)
                city.close_street(a, b)
                schedule(now + settings.incident_length, ROAD_OPENS, (a, b, seconds))
        elif kind == ROAD_OPENS:
            a, b, seconds = data
            city.add_street(a, b, seconds)
        elif kind == DRIVER_OFF:
            free = [d for d in dispatcher.drivers.values() if d.is_free]
            if free:
                driver = rng.choice(free)
                dispatcher.go_offline(driver.driver_id)
                schedule(now + rng.uniform(300, 1800), DRIVER_ON, driver.driver_id)
        elif kind == DRIVER_ON:
            dispatcher.go_online(data, rng.choice(corners))
        elif kind == ROUND:
            for match in dispatcher.run_round(now):
                trip = dispatcher.trips_in_progress[match.driver_id]
                ride_secs = ride_time(trip.pickup, trip.dropoff)
                if math.isinf(ride_secs):
                    ride_secs = 0.0   # an incident cut the route; treat as a no-show
                schedule(now + match.pickup_time + ride_secs, TRIP_ENDS, match.driver_id)
                waits.append(match.waited)
                pickups.append(match.pickup_time)
                tier = tier_of.pop(match.request_id)
                total = per_tier.setdefault(tier, [0.0, 0])
                total[0] += match.waited
                total[1] += 1
                if settings.keep_matches:
                    matches.append(match)
            # Keep running rounds while requests can still arrive or are waiting.
            if now < settings.duration or dispatcher.count_waiting():
                schedule(now + settings.round_every, ROUND)
    elapsed = time.perf_counter() - t0

    sorted_waits = sorted(waits)
    counters = dict(dispatcher.stats)
    if not phase2_engine:
        counters["route_cache_hits"] = city.cache_hits
        counters["route_cache_misses"] = city.cache_misses
        counters["landmarks_still_valid"] = city.has_landmarks
    return SimResult(
        requests=settings.requests,
        served=len(waits),
        cancelled=cancelled,
        still_waiting=dispatcher.count_waiting(),
        avg_wait=statistics.fmean(waits) if waits else 0.0,
        p95_wait=sorted_waits[int(0.95 * (len(sorted_waits) - 1))] if waits else 0.0,
        avg_pickup=statistics.fmean(pickups) if pickups else 0.0,
        wait_per_tier={t: per_tier[t][0] / per_tier[t][1] for t in tiers if t in per_tier},
        counters=counters,
        elapsed=elapsed,
        setup=setup,
        events=handled,
        peak_events=peak,
        matches=matches,
    )
