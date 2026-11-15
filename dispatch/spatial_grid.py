"""A simple grid that lets us look up the free drivers closest to a point.

I split the map into square cells (``cell_size`` metres on each side).
``_cells`` maps a (col, row) cell to the set of ids sitting in it, and
``_coords`` maps each id to its exact (x, y). Drivers send a new GPS fix
every few seconds, so moving one has to be cheap - here it is just a couple
of dictionary operations.

For the nearest-k search I start at the cell the point falls in and then
walk outward one square ring of cells at a time. After ring r, anything I
have not looked at yet is at least r * cell_size away, so I can stop once
the k-th best distance is inside that radius. Empty cells are deleted, so
memory depends on how many drivers there are, not on how big the city is.
"""

import heapq
import math


class SpatialGrid:
    """Grid of hash buckets holding (x, y) points by id."""

    def __init__(self, cell_size):
        if cell_size <= 0:
            raise ValueError("cell_size has to be greater than zero")
        self.cell_size = cell_size
        self._cells = {}    # (col, row) -> set of ids in that cell
        self._coords = {}   # id -> (x, y)

    def __len__(self):
        return len(self._coords)

    def __contains__(self, item_id):
        return item_id in self._coords

    def _cell_for(self, x, y):
        return (math.floor(x / self.cell_size), math.floor(y / self.cell_size))

    def coords_of(self, item_id):
        """Give back the stored (x, y) for an id (KeyError if it isn't here)."""
        return self._coords[item_id]

    def insert(self, item_id, x, y):
        """Put a new id on the grid. O(1) on average."""
        if item_id in self._coords:
            raise KeyError(f"{item_id!r} is already on the grid")
        self._coords[item_id] = (x, y)
        self._cells.setdefault(self._cell_for(x, y), set()).add(item_id)

    def _drop_from_cell(self, item_id, cell):
        members = self._cells[cell]
        members.discard(item_id)
        if not members:
            del self._cells[cell]   # don't keep empty cells around

    def delete(self, item_id):
        """Take an id off the grid. O(1) on average."""
        x, y = self._coords.pop(item_id)
        self._drop_from_cell(item_id, self._cell_for(x, y))

    def move(self, item_id, x, y):
        """Update where an id is. O(1) on average."""
        old_x, old_y = self._coords[item_id]
        old_cell, new_cell = self._cell_for(old_x, old_y), self._cell_for(x, y)
        self._coords[item_id] = (x, y)
        if old_cell != new_cell:
            self._drop_from_cell(item_id, old_cell)
            self._cells.setdefault(new_cell, set()).add(item_id)

    def _cells_in_ring(self, mid_col, mid_row, ring):
        """Yield the cells along the edge of the square that is ``ring`` cells out."""
        if ring == 0:
            yield (mid_col, mid_row)
            return
        for i in range(-ring, ring + 1):
            yield (mid_col + i, mid_row - ring)
            yield (mid_col + i, mid_row + ring)
        for i in range(-ring + 1, ring):
            yield (mid_col - ring, mid_row + i)
            yield (mid_col + ring, mid_row + i)

    def nearest(self, x, y, k=1, max_distance=math.inf):
        """Return up to k (distance, id) pairs closest to (x, y), nearest first.

        Anything further than ``max_distance`` is skipped. How long this takes
        depends on how many cells get visited, which stays small as long as
        the cell size is a reasonable match for the search radius.
        """
        if k <= 0 or not self._coords:
            return []
        mid_col, mid_row = self._cell_for(x, y)
        # Rings past the furthest occupied cell (or past the distance cap)
        # can't hold anything, so this is where the loop has to end.
        last_ring = max(max(abs(col - mid_col), abs(row - mid_row))
                        for col, row in self._cells)
        if max_distance != math.inf:
            last_ring = min(last_ring, int(max_distance // self.cell_size) + 1)

        found = []       # every (distance, id) inside the cap that we looked at
        kth_best = []    # max-heap (as negatives) of the k smallest distances so far
        for ring in range(last_ring + 1):
            for cell in self._cells_in_ring(mid_col, mid_row, ring):
                for item_id in self._cells.get(cell, ()):
                    px, py = self._coords[item_id]
                    dist = math.hypot(px - x, py - y)
                    if dist > max_distance:
                        continue
                    found.append((dist, item_id))
                    if len(kth_best) < k:
                        heapq.heappush(kth_best, -dist)
                    elif dist < -kth_best[0]:
                        heapq.heapreplace(kth_best, -dist)
            # Whatever is outside rings 0..ring is at least ring * cell_size away.
            if len(kth_best) == k and -kth_best[0] <= ring * self.cell_size:
                break
        # Sorting by (distance, id) means ties always come out the same way,
        # no matter what order the sets handed the ids to us.
        found.sort()
        return found[:k]
