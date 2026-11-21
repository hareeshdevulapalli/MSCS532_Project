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

What changed in Phase 3
-----------------------
1. Phase 2 worked out how many rings it might need by looking at every
   occupied cell on *every* query, which made each search O(cells) before
   it even started. The grid now keeps a bounding box of the occupied cells.
   Inserts grow it in O(1). Deletes only mark it as possibly too big, and it
   is shrunk back once the number of deletes passes the number of cells, so
   that cost is O(1) amortised.
2. When the rings would visit more empty cells than there are occupied ones
   (few drivers spread over a big city), the search switches to scanning the
   drivers directly, so a query never costs more than O(n) distance checks.
3. Phase 2 kept every point it looked at and sorted all of them at the end.
   Now a point is only kept if it could still be in the top k, so the final
   sort is over about k items.
4. Coordinates are checked: NaN or infinite values are rejected up front
   instead of silently landing in a broken cell.
"""

import heapq
import math


class SpatialGrid:
    """Grid of hash buckets holding (x, y) points by id."""

    __slots__ = ("cell_size", "_cells", "_coords", "_box", "_deletes_since_fit")

    def __init__(self, cell_size):
        if not (isinstance(cell_size, (int, float)) and math.isfinite(cell_size)) or cell_size <= 0:
            raise ValueError("cell_size has to be a finite number greater than zero")
        self.cell_size = float(cell_size)
        self._cells = {}    # (col, row) -> set of ids in that cell
        self._coords = {}   # id -> (x, y)
        self._box = None    # [min_col, max_col, min_row, max_row]; may be loose after deletes
        self._deletes_since_fit = 0

    def __len__(self):
        return len(self._coords)

    def __contains__(self, item_id):
        return item_id in self._coords

    def _cell_for(self, x, y):
        return (math.floor(x / self.cell_size), math.floor(y / self.cell_size))

    @staticmethod
    def _check(x, y):
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"coordinates have to be finite numbers, got ({x!r}, {y!r})")

    def coords_of(self, item_id):
        """Give back the stored (x, y) for an id (KeyError if it isn't here)."""
        return self._coords[item_id]

    # ----- bounding box of the occupied cells ---------------------------

    def _grow_box(self, cell):
        col, row = cell
        box = self._box
        if box is None:
            self._box = [col, col, row, row]
            return
        if col < box[0]:
            box[0] = col
        elif col > box[1]:
            box[1] = col
        if row < box[2]:
            box[2] = row
        elif row > box[3]:
            box[3] = row

    def _refit_box(self):
        """Shrink the box to the cells really in use. O(cells), but rare."""
        self._deletes_since_fit = 0
        if not self._cells:
            self._box = None
            return
        cols = [c for c, _ in self._cells]
        rows = [r for _, r in self._cells]
        self._box = [min(cols), max(cols), min(rows), max(rows)]

    # ----- updates ------------------------------------------------------

    def insert(self, item_id, x, y):
        """Put a new id on the grid. O(1) on average."""
        if item_id in self._coords:
            raise KeyError(f"{item_id!r} is already on the grid")
        self._check(x, y)
        cell = self._cell_for(x, y)
        self._coords[item_id] = (x, y)
        members = self._cells.get(cell)
        if members is None:
            self._cells[cell] = {item_id}
            self._grow_box(cell)
        else:
            members.add(item_id)

    def _drop_from_cell(self, item_id, cell):
        members = self._cells[cell]
        members.discard(item_id)
        if not members:
            del self._cells[cell]   # don't keep empty cells around
            self._deletes_since_fit += 1
            if self._deletes_since_fit > len(self._cells) + 16:
                self._refit_box()

    def delete(self, item_id):
        """Take an id off the grid. O(1) amortised."""
        x, y = self._coords.pop(item_id)
        self._drop_from_cell(item_id, self._cell_for(x, y))

    def move(self, item_id, x, y):
        """Update where an id is. O(1) amortised."""
        old_x, old_y = self._coords[item_id]
        self._check(x, y)
        old_cell, new_cell = self._cell_for(old_x, old_y), self._cell_for(x, y)
        self._coords[item_id] = (x, y)
        if old_cell != new_cell:
            self._drop_from_cell(item_id, old_cell)
            members = self._cells.get(new_cell)
            if members is None:
                self._cells[new_cell] = {item_id}
                self._grow_box(new_cell)
            else:
                members.add(item_id)

    # ----- search -------------------------------------------------------

    @staticmethod
    def _cells_in_ring(mid_col, mid_row, ring):
        """Yield the cells along the edge of the square that is ``ring`` cells out."""
        if ring == 0:
            yield (mid_col, mid_row)
            return
        top, bottom = mid_row - ring, mid_row + ring
        for col in range(mid_col - ring, mid_col + ring + 1):
            yield (col, top)
            yield (col, bottom)
        left, right = mid_col - ring, mid_col + ring
        for row in range(mid_row - ring + 1, mid_row + ring):
            yield (left, row)
            yield (right, row)

    def _scan_all(self, x, y, k, max_distance):
        """Check every point. Used when the rings would mostly hit empty cells."""
        hypot = math.hypot
        inside = ((hypot(px - x, py - y), item_id) for item_id, (px, py) in self._coords.items())
        if max_distance != math.inf:
            inside = (entry for entry in inside if entry[0] <= max_distance)
        return heapq.nsmallest(k, inside)

    def within(self, x, y, radius):
        """Return the ids of every point at most ``radius`` from (x, y), in no order.

        Looks at the square of cells covering the circle, or at every occupied
        cell if that is fewer.
        """
        if not self._coords or radius < 0:
            return []
        self._check(x, y)
        if radius == math.inf:
            return list(self._coords)
        size, cells, coords = self.cell_size, self._cells, self._coords
        lo_col, lo_row = self._cell_for(x - radius, y - radius)
        hi_col, hi_row = self._cell_for(x + radius, y + radius)
        hypot = math.hypot
        if (hi_col - lo_col + 1) * (hi_row - lo_row + 1) > len(cells):
            look = (members for cell, members in cells.items()
                    if lo_col <= cell[0] <= hi_col and lo_row <= cell[1] <= hi_row)
        else:
            look = (cells[(c, r)] for c in range(lo_col, hi_col + 1)
                    for r in range(lo_row, hi_row + 1) if (c, r) in cells)
        found = []
        for members in look:
            for item_id in members:
                px, py = coords[item_id]
                if hypot(px - x, py - y) <= radius:
                    found.append(item_id)
        return found

    def nearest(self, x, y, k=1, max_distance=math.inf):
        """Return up to k (distance, id) pairs closest to (x, y), nearest first.

        Anything further than ``max_distance`` is skipped. Ties on distance
        are broken by id, so the answer never depends on set order.
        Cost: the cells visited, capped at about O(n) point checks.
        """
        if k <= 0 or not self._coords:
            return []
        self._check(x, y)
        cells, coords, size = self._cells, self._coords, self.cell_size
        if k >= len(coords):
            return self._scan_all(x, y, k, max_distance)

        mid_col, mid_row = self._cell_for(x, y)
        min_col, max_col, min_row, max_row = self._box
        last_ring = max(mid_col - min_col, max_col - mid_col, mid_row - min_row, max_row - mid_row, 0)
        if max_distance != math.inf:
            last_ring = min(last_ring, int(max_distance // size) + 1)
        # Visiting a cell costs about as much as checking a point, so give up on
        # the rings once they have cost more than a plain scan would.
        budget = len(cells) + len(coords)

        hypot = math.hypot
        found = []      # (distance, id) of points that might still be in the top k
        kth_best = []   # max-heap (as negatives) of the k smallest distances so far
        visited = 0
        for ring in range(last_ring + 1):
            for cell in self._cells_in_ring(mid_col, mid_row, ring):
                visited += 1
                members = cells.get(cell)
                if not members:
                    continue
                for item_id in members:
                    px, py = coords[item_id]
                    dist = hypot(px - x, py - y)
                    if dist > max_distance:
                        continue
                    if len(kth_best) < k:
                        heapq.heappush(kth_best, -dist)
                        found.append((dist, item_id))
                    elif dist <= -kth_best[0]:
                        if dist < -kth_best[0]:
                            heapq.heapreplace(kth_best, -dist)
                        found.append((dist, item_id))
            # Whatever is outside rings 0..ring is at least ring * cell_size away.
            if len(kth_best) == k and -kth_best[0] <= ring * size:
                break
            if visited > budget:
                return self._scan_all(x, y, k, max_distance)
        found.sort()
        return found[:k]
