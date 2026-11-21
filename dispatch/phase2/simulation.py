"""A small event-driven simulation of one dispatcher shift.

Every event (a request coming in, a rider giving up, a trip ending, a
dispatch round) goes into a binary heap ordered by time, so getting the
next one is O(log n). Dispatch runs in rounds every few seconds, which is
roughly how real systems batch their decisions.

This is kept simple on purpose. Its job is to throw a realistic mix of
operations at the data structures and to give Phase 3 a workload I can
repeat when benchmarking.
"""

import heapq
import itertools
import random
import statistics
import time
from dataclasses import dataclass, field

from .city_builder import add_river_barrier, build_grid_city
from .dispatcher import Dispatcher

NEW_REQUEST, RIDER_LEAVES, TRIP_ENDS, ROUND = "new_request", "rider_leaves", "trip_ends", "round"
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
    events: int = 0
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
        lines += [
            f"Dijkstra searches   {c['searches']} (search widened {c['widenings']} times)",
            f"skipped, retried    {c['skipped']}",
            f"tier bumps          {c['promotions']}",
            f"events handled      {self.events}",
            f"run time            {self.elapsed:.3f} s",
        ]
        return lines


def simulate(settings=None):
    """Run one simulated shift and return a SimResult."""
    settings = settings or SimSettings()
    rng = random.Random(settings.seed)
    city = build_grid_city(settings.cols, settings.rows)
    if settings.river and settings.cols > 1:
        middle = settings.cols // 2 - 1
        add_river_barrier(city, middle, settings.rows,
                          bridge_rows={settings.rows // 4, 3 * settings.rows // 4})
    dispatcher = Dispatcher(
        city, cell_size=400.0, candidate_count=settings.candidate_count,
        pickup_limit=settings.patience, promote_after=settings.promote_after,
        widen_search=settings.widen_search)
    corners = list(city.nodes())

    for n in range(settings.drivers):
        dispatcher.add_driver(f"D{n:03d}", rng.choice(corners))

    timeline, tiebreak = [], itertools.count()   # tiebreak keeps equal times in order

    def schedule(at, kind, data=None):
        heapq.heappush(timeline, (at, next(tiebreak), kind, data))

    tiers, shares = zip(*TIER_SHARE)
    for n in range(settings.requests):
        at = rng.uniform(0, settings.duration)
        pickup, dropoff = rng.sample(corners, 2)
        tier = rng.choices(tiers, shares)[0]
        schedule(at, NEW_REQUEST, (f"R{n:04d}", pickup, dropoff, tier))
    schedule(0.0, ROUND)

    tier_of, matches, cancelled, handled = {}, [], 0, 0
    t0 = time.perf_counter()
    while timeline:
        now, _, kind, data = heapq.heappop(timeline)
        handled += 1
        if kind == NEW_REQUEST:
            request_id, pickup, dropoff, tier = data
            dispatcher.new_request(request_id, pickup, dropoff, tier, now)
            tier_of[request_id] = tier
            schedule(now + settings.patience, RIDER_LEAVES, request_id)
        elif kind == RIDER_LEAVES:
            if data in dispatcher.waiting_requests:
                dispatcher.cancel_request(data)
                cancelled += 1
        elif kind == TRIP_ENDS:
            dispatcher.complete_trip(data)
        elif kind == ROUND:
            for match in dispatcher.run_round(now):
                trip = dispatcher.trips_in_progress[match.driver_id]
                ride_secs, _ = city.shortest_path(trip.pickup, trip.dropoff)
                schedule(now + match.pickup_time + ride_secs, TRIP_ENDS, match.driver_id)
                matches.append(match)
            # Keep running rounds while requests can still arrive or are waiting.
            if now < settings.duration or dispatcher.count_waiting():
                schedule(now + settings.round_every, ROUND)
    elapsed = time.perf_counter() - t0

    waits = sorted(m.waited for m in matches)
    per_tier = {}
    for m in matches:
        per_tier.setdefault(tier_of[m.request_id], []).append(m.waited)
    return SimResult(
        requests=settings.requests,
        served=len(matches),
        cancelled=cancelled,
        still_waiting=dispatcher.count_waiting(),
        avg_wait=statistics.fmean(waits) if waits else 0.0,
        p95_wait=waits[int(0.95 * (len(waits) - 1))] if waits else 0.0,
        avg_pickup=statistics.fmean(m.pickup_time for m in matches) if matches else 0.0,
        wait_per_tier={t: statistics.fmean(per_tier[t]) for t in tiers if t in per_tier},
        counters=dict(dispatcher.stats),
        elapsed=elapsed,
        events=handled,
        matches=matches,
    )
