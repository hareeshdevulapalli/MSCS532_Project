"""Directed road network with travel times, searched with Dijkstra.

Intersections are nodes and each street segment is a one-direction edge
weighted by how many seconds it takes to drive. City road maps are sparse
(each corner only touches a few streets), so I store edges in adjacency
dictionaries: O(V + E) memory, and adding, re-timing or closing a street is
O(1) on average.

I keep the incoming edges as well as the outgoing ones. When a few nearby
drivers are all candidates for the same pickup, the dispatcher can run one
Dijkstra search backwards from the pickup until it has reached all of them,
rather than one search per driver.

Added in Phase 2: the network remembers the highest speed (m/s) of any
street it has been given. A straight-line distance divided by that speed is
a lower bound on the driving time, and the dispatcher uses it to know when
no driver further away could possibly beat the best one it has found.
"""

import heapq
import math


class RoadNetwork:
    """Sparse directed graph of intersections joined by timed streets."""

    def __init__(self):
        self._coords = {}      # node -> (x, y)
        self._out_edges = {}   # node -> {next node: seconds}
        self._in_edges = {}    # node -> {previous node: seconds}
        self._max_speed = 0.0  # fastest m/s seen on any street so far

    def __contains__(self, node):
        return node in self._coords

    def __len__(self):
        return len(self._coords)

    def nodes(self):
        return self._coords.keys()

    def coords(self, node):
        return self._coords[node]

    def add_node(self, node, x, y):
        if node in self._coords:
            raise KeyError(f"node {node!r} already exists")
        self._coords[node] = (x, y)
        self._out_edges[node] = {}
        self._in_edges[node] = {}

    def add_street(self, a, b, seconds, two_way=True):
        """Add (or re-time) the street a -> b, and b -> a too if ``two_way``."""
        if a not in self._coords or b not in self._coords:
            raise KeyError("both nodes have to exist before adding a street")
        if seconds < 0:
            raise ValueError("travel time can't be negative")
        self._out_edges[a][b] = seconds
        self._in_edges[b][a] = seconds
        self._track_speed(a, b, seconds)
        if two_way:
            self._out_edges[b][a] = seconds
            self._in_edges[a][b] = seconds

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

    def close_street(self, a, b, two_way=True):
        """Remove the street a -> b (crash, road works, etc.)."""
        if b not in self._out_edges.get(a, {}):
            raise KeyError(f"no street {a!r} -> {b!r}")
        if two_way and a not in self._out_edges.get(b, {}):
            raise KeyError(f"no street {b!r} -> {a!r}")
        del self._out_edges[a][b]
        del self._in_edges[b][a]
        if two_way:
            del self._out_edges[b][a]
            del self._in_edges[a][b]

    def shortest_path(self, start, goal):
        """Return (seconds, [nodes]) for the quickest route, or (inf, []) if none.

        Plain Dijkstra on a binary heap. Old heap entries are skipped when
        popped instead of being deleted. Worst case O((V + E) log V), but it
        stops as soon as ``goal`` comes off the heap.
        """
        if start not in self._coords or goal not in self._coords:
            raise KeyError("both nodes have to exist")
        dist = {start: 0}
        prev = {start: None}
        frontier = [(0, start)]
        while frontier:
            t, node = heapq.heappop(frontier)
            if node == goal:
                path = []
                while node is not None:
                    path.append(node)
                    node = prev[node]
                return t, path[::-1]
            if t > dist[node]:
                continue  # stale entry, we already found a quicker way here
            for nxt, seconds in self._out_edges[node].items():
                arrive = t + seconds
                if arrive < dist.get(nxt, math.inf):
                    dist[nxt] = arrive
                    prev[nxt] = node
                    heapq.heappush(frontier, (arrive, nxt))
        return math.inf, []

    def times_to_target(self, target, sources, limit=math.inf):
        """Return {source: seconds to drive to ``target``} for the sources we reach.

        One Dijkstra search over the incoming edges, starting at ``target``.
        It stops once every source has been reached or the times go past
        ``limit``.
        """
        if target not in self._coords:
            raise KeyError(f"unknown node {target!r}")
        remaining = set(sources)
        result = {}
        dist = {target: 0}
        frontier = [(0, target)]
        while frontier and remaining:
            t, node = heapq.heappop(frontier)
            if t > dist[node]:
                continue
            if t > limit:
                break
            if node in remaining:
                remaining.discard(node)
                result[node] = t
            for before, seconds in self._in_edges[node].items():
                arrive = t + seconds
                if arrive < dist.get(before, math.inf):
                    dist[before] = arrive
                    heapq.heappush(frontier, (arrive, before))
        return result
