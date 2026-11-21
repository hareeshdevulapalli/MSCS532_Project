"""Directed road network with travel times, searched with Dijkstra and A*.

Intersections are nodes and each street segment is a one-direction edge
weighted by how many seconds it takes to drive. City road maps are sparse
(each corner only touches a few streets), so I store edges in adjacency
dictionaries: O(V + E) memory, and adding, re-timing or closing a street is
O(1) on average.

I keep the incoming edges as well as the outgoing ones so the dispatcher
can search backwards from a pickup towards the drivers.

The network remembers the highest speed (m/s) of any street it has been
given. A straight-line distance divided by that speed is a lower bound on
the driving time.

What changed in Phase 3
-----------------------
1. ``nearest_source``: a backward Dijkstra from the pickup that stops at the
   first intersection holding a free driver. Phase 2 had to reach *every*
   candidate the grid suggested (and start again from scratch each time the
   list was doubled); this settles only the intersections that are quicker
   to reach than the winning driver.
2. ``shortest_path`` is now A*, using the speed bound as the estimate. The
   estimate never overshoots and obeys the triangle inequality, so A* still
   returns the exact quickest route but looks at far fewer intersections.
3. ``travel_time`` remembers recent answers in a small LRU cache. Every
   change to a street bumps ``version`` and empties the cache, so an old
   answer is never served after a closure. ``open_version`` only moves when
   a street is added or re-timed; closures can only make trips longer, so
   "nobody can get here" answers stay true through them.
4. Landmarks (ALT). ``prepare_landmarks`` runs a full Dijkstra to and from
   a handful of far-apart intersections and stores the times in flat
   ``array('d')`` tables. By the triangle inequality,
   d(v, t) >= d(L, t) - d(L, v) and d(v, t) >= d(v, L) - d(t, L), which is a
   much tighter A* estimate than straight-line distance on a street grid
   (it "knows" about the river). It costs 2 x landmarks x V floats of
   memory - a deliberate space-for-time trade.
   Closing a street only makes trips longer, so the bounds stay valid
   through closures; re-opening a street is fine as long as it is not faster
   than it was when the tables were built. Any other street change switches
   the landmarks off until ``prepare_landmarks`` is run again.
5. Node coordinates and travel times are checked for NaN / infinity.
"""

import heapq
import math
from array import array
from collections import OrderedDict

INF = math.inf


class RoadNetwork:
    """Sparse directed graph of intersections joined by timed streets."""

    __slots__ = ("_coords", "_out_edges", "_in_edges", "_max_speed", "version",
                 "last_settled", "_route_cache", "_cache_size", "cache_hits", "cache_misses",
                 "_landmarks", "_node_index", "_closed_since_landmarks", "open_version")

    def __init__(self, route_cache_size=4096):
        if route_cache_size < 0:
            raise ValueError("route_cache_size can't be negative")
        self._coords = {}      # node -> (x, y)
        self._out_edges = {}   # node -> {next node: seconds}
        self._in_edges = {}    # node -> {previous node: seconds}
        self._max_speed = 0.0  # fastest m/s seen on any street so far
        self.version = 0       # goes up on every street change
        self.open_version = 0  # goes up only when a street is added or re-timed
        self.last_settled = 0  # intersections settled by the most recent search
        self._route_cache = OrderedDict()   # (a, b) -> seconds, most recent last
        self._cache_size = route_cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        self._landmarks = None              # list of (from_L, to_L) arrays, see prepare_landmarks
        self._node_index = None             # node -> position in those arrays
        self._closed_since_landmarks = {}   # (a, b) -> seconds it had when landmarks were made

    def __contains__(self, node):
        return node in self._coords

    def __len__(self):
        return len(self._coords)

    def nodes(self):
        return self._coords.keys()

    def coords(self, node):
        return self._coords[node]

    def street_count(self):
        return sum(len(out) for out in self._out_edges.values())

    # ----- building and changing the map ----------------------------------

    def _changed(self):
        self.version += 1
        self._route_cache.clear()

    def add_node(self, node, x, y):
        if node in self._coords:
            raise KeyError(f"node {node!r} already exists")
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"coordinates have to be finite numbers, got ({x!r}, {y!r})")
        self._coords[node] = (x, y)
        self._out_edges[node] = {}
        self._in_edges[node] = {}

    def add_street(self, a, b, seconds, two_way=True):
        """Add (or re-time) the street a -> b, and b -> a too if ``two_way``."""
        if a not in self._coords or b not in self._coords:
            raise KeyError("both nodes have to exist before adding a street")
        if not math.isfinite(seconds):
            raise ValueError("travel time has to be a finite number")
        if seconds < 0:
            raise ValueError("travel time can't be negative")
        if self._landmarks is not None:
            self._check_landmarks_after_add(a, b, seconds)
            if two_way:
                self._check_landmarks_after_add(b, a, seconds)
        self._out_edges[a][b] = seconds
        self._in_edges[b][a] = seconds
        self._track_speed(a, b, seconds)
        if two_way:
            self._out_edges[b][a] = seconds
            self._in_edges[a][b] = seconds
        self.open_version += 1
        self._changed()

    def _check_landmarks_after_add(self, a, b, seconds):
        """Keep the landmark tables only if no trip can get quicker than they assume."""
        current = self._out_edges[a].get(b)
        if current is not None and seconds >= current:
            return                                     # same street, not faster
        before = self._closed_since_landmarks.get((a, b))
        if before is not None and seconds >= before:
            del self._closed_since_landmarks[(a, b)]   # re-opened, not faster
            return
        self.drop_landmarks()

    def street_time(self, a, b):
        """Seconds to drive a -> b; KeyError if there is no such street."""
        return self._out_edges[a][b]

    def _track_speed(self, a, b, seconds):
        (x1, y1), (x2, y2) = self._coords[a], self._coords[b]
        length = math.hypot(x2 - x1, y2 - y1)
        if length == 0:
            return
        speed = math.inf if seconds == 0 else length / seconds
        self._max_speed = max(self._max_speed, speed)

    @property
    def max_speed(self):
        """Highest m/s of any street added so far.

        This never goes down when a street gets slower or is closed. That is
        fine: a speed that is too high only makes ``lower_bound_time``
        smaller, so the bound is still a valid lower bound.
        """
        return self._max_speed

    def lower_bound_time(self, distance):
        """Smallest number of seconds it could take to cover a straight-line distance."""
        if distance <= 0 or self._max_speed == math.inf:
            return 0.0
        if self._max_speed == 0:
            return math.inf   # there are no streets with any length yet
        return distance / self._max_speed

    def reach(self, seconds):
        """Furthest straight-line distance anyone could cover in ``seconds``."""
        if self._max_speed == math.inf:
            return math.inf
        return seconds * self._max_speed

    def close_street(self, a, b, two_way=True):
        """Remove the street a -> b (crash, road works, etc.)."""
        if b not in self._out_edges.get(a, {}):
            raise KeyError(f"no street {a!r} -> {b!r}")
        if two_way and a not in self._out_edges.get(b, {}):
            raise KeyError(f"no street {b!r} -> {a!r}")
        if self._landmarks is not None:
            self._closed_since_landmarks.setdefault((a, b), self._out_edges[a][b])
            if two_way:
                self._closed_since_landmarks.setdefault((b, a), self._out_edges[b][a])
        del self._out_edges[a][b]
        del self._in_edges[b][a]
        if two_way:
            del self._out_edges[b][a]
            del self._in_edges[a][b]
        self._changed()

    # ----- landmarks (ALT) -------------------------------------------------

    def _all_times(self, start, edges):
        """Full Dijkstra from ``start`` over ``edges``; returns {node: seconds}."""
        dist = {start: 0}
        frontier = [(0, start)]
        pop, push = heapq.heappop, heapq.heappush
        while frontier:
            t, node = pop(frontier)
            if t > dist[node]:
                continue
            for nxt, seconds in edges[node].items():
                arrive = t + seconds
                if arrive < dist.get(nxt, INF):
                    dist[nxt] = arrive
                    push(frontier, (arrive, nxt))
        return dist

    def prepare_landmarks(self, count=8, seed_node=None):
        """Pick ``count`` far-apart landmarks and store times to and from each.

        Landmarks are chosen greedily: each new one is the intersection
        furthest (by drive time) from the ones already chosen. Cost is
        2 x count full Dijkstra runs, O(count (V + E) log V) time and
        2 x count x V floats of memory.
        """
        if count < 1:
            raise ValueError("count has to be at least 1")
        if not self._coords:
            return []
        nodes = list(self._coords)
        index = {node: i for i, node in enumerate(nodes)}
        start = seed_node if seed_node is not None else nodes[0]
        # first landmark: whatever is furthest from the starting node
        first = self._all_times(start, self._out_edges)
        chosen = [max(first, key=first.get)]
        nearest_landmark = None
        tables = []
        while True:
            landmark = chosen[-1]
            from_l = self._all_times(landmark, self._out_edges)
            to_l = self._all_times(landmark, self._in_edges)
            tables.append((array("d", (from_l.get(n, INF) for n in nodes)),
                           array("d", (to_l.get(n, INF) for n in nodes))))
            if len(chosen) == count:
                break
            if nearest_landmark is None:
                nearest_landmark = dict(from_l)
            else:
                for node, t in from_l.items():
                    if t < nearest_landmark.get(node, INF):
                        nearest_landmark[node] = t
            candidates = [n for n in nearest_landmark if n not in chosen]
            if not candidates:
                break
            chosen.append(max(candidates, key=nearest_landmark.get))
        self._landmarks = tables
        self._node_index = index
        self._closed_since_landmarks = {}
        return chosen

    def drop_landmarks(self):
        self._landmarks = None
        self._node_index = None
        self._closed_since_landmarks = {}

    @property
    def has_landmarks(self):
        return self._landmarks is not None

    def landmark_memory(self):
        """Bytes used by the landmark tables (0 if there are none)."""
        if self._landmarks is None:
            return 0
        return sum(a.itemsize * len(a) + b.itemsize * len(b) for a, b in self._landmarks)

    def _landmark_estimate(self, goal):
        """Return h(node) using the landmark tables, for one goal."""
        index = self._node_index
        g = index[goal]
        terms = [(f, b, f[g], b[g]) for f, b in self._landmarks]

        def h(node):
            i = index[node]
            best = 0.0
            for from_l, to_l, f_goal, b_goal in terms:
                f_node = from_l[i]
                if f_node != INF:
                    if f_goal == INF:
                        return INF    # L reaches node but not goal, so node can't reach goal
                    if f_goal - f_node > best:
                        best = f_goal - f_node
                b_node = to_l[i]
                if b_goal != INF:
                    if b_node == INF:
                        return INF    # goal reaches L but node doesn't, so node can't reach goal
                    if b_node - b_goal > best:
                        best = b_node - b_goal
            return best
        return h

    # ----- searches -----------------------------------------------------

    def shortest_path(self, start, goal):
        """Return (seconds, [nodes]) for the quickest route, or (inf, []) if none.

        A* with h(n) = straight-line distance to ``goal`` / max_speed, or the
        landmark bound if ``prepare_landmarks`` has been run. Both never
        overestimate and are consistent (h(u) <= time(u, v) + h(v)), so the
        first time ``goal`` comes off the heap its time is exact. Worst case
        is still O((V + E) log V).
        """
        if start not in self._coords or goal not in self._coords:
            raise KeyError("both nodes have to exist")
        coords, out_edges = self._coords, self._out_edges
        gx, gy = coords[goal]
        speed = self._max_speed
        use_h = 0 < speed < math.inf
        hypot = math.hypot

        if self._landmarks is not None:
            h = self._landmark_estimate(goal)   # ALT: tighter than straight-line
        elif use_h:
            def h(node):
                x, y = coords[node]
                return hypot(gx - x, gy - y) / speed
        else:
            def h(node):
                return 0.0

        dist = {start: 0}
        prev = {start: None}
        # Ties on f are broken towards the node that has come further (-g). On a
        # street grid many routes tie, and preferring small g would flood the
        # whole rectangle between start and goal.
        frontier = [(h(start), 0, start)]
        pop, push = heapq.heappop, heapq.heappush
        settled = 0
        while frontier:
            _, neg_t, node = pop(frontier)
            t = -neg_t
            if t > dist[node]:
                continue  # stale entry, we already found a quicker way here
            settled += 1
            if node == goal:
                self.last_settled = settled
                path = []
                while node is not None:
                    path.append(node)
                    node = prev[node]
                return t, path[::-1]
            for nxt, seconds in out_edges[node].items():
                arrive = t + seconds
                if arrive < dist.get(nxt, INF):
                    estimate = h(nxt)
                    if estimate == INF:
                        continue      # the landmarks prove nxt can't reach the goal
                    dist[nxt] = arrive
                    prev[nxt] = node
                    push(frontier, (arrive + estimate, -arrive, nxt))
        self.last_settled = settled
        return math.inf, []

    def travel_time(self, start, goal):
        """Seconds for the quickest route, using a small LRU cache of answers."""
        key = (start, goal)
        cache = self._route_cache
        if key in cache:
            cache.move_to_end(key)
            self.cache_hits += 1
            return cache[key]
        self.cache_misses += 1
        seconds, _ = self.shortest_path(start, goal)
        if self._cache_size:
            cache[key] = seconds
            if len(cache) > self._cache_size:
                cache.popitem(last=False)
        return seconds

    def times_to_target(self, target, sources, limit=math.inf):
        """Return {source: seconds to drive to ``target``} for the sources we reach.

        One Dijkstra search over the incoming edges, starting at ``target``.
        It stops once every source has been reached or the times go past
        ``limit``. (Still used by the fixed-list mode that mimics Phase 1.)
        """
        if target not in self._coords:
            raise KeyError(f"unknown node {target!r}")
        remaining = set(sources)
        result = {}
        dist = {target: 0}
        frontier = [(0, target)]
        in_edges = self._in_edges
        pop, push = heapq.heappop, heapq.heappush
        settled = 0
        while frontier and remaining:
            t, node = pop(frontier)
            if t > dist[node]:
                continue
            if t > limit:
                break
            settled += 1
            if node in remaining:
                remaining.discard(node)
                result[node] = t
            for before, seconds in in_edges[node].items():
                arrive = t + seconds
                if arrive < dist.get(before, math.inf):
                    dist[before] = arrive
                    push(frontier, (arrive, before))
        self.last_settled = settled
        return result

    def nearest_source(self, target, sources, limit=math.inf):
        """Quickest time from any node in ``sources`` to ``target``.

        ``sources`` only needs to support ``in`` (the dispatcher passes its
        dict of intersections that have a free driver). Returns
        (seconds, [nodes that tie for that time]) or (None, []) if nobody is
        within ``limit``. Backward Dijkstra that stops as soon as the times
        go past the first hit, so it only settles intersections that are
        quicker to reach than the answer.
        """
        if target not in self._coords:
            raise KeyError(f"unknown node {target!r}")
        in_edges = self._in_edges
        dist = {target: 0}
        frontier = [(0, target)]
        pop, push = heapq.heappop, heapq.heappush
        best, hits, settled = None, [], 0
        while frontier:
            t, node = pop(frontier)
            if t > dist[node]:
                continue
            if t > limit or (best is not None and t > best):
                break
            settled += 1
            if node in sources:
                best = t
                hits.append(node)
            # Keep expanding: a zero-second street could still tie with best.
            for before, seconds in in_edges[node].items():
                arrive = t + seconds
                if arrive < dist.get(before, math.inf):
                    dist[before] = arrive
                    push(frontier, (arrive, before))
        self.last_settled = settled
        return best, hits
