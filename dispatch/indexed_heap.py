"""Binary min-heap with a lookup table, used for the waiting ride requests.

Python's ``heapq`` can't remove or re-prioritise something in the middle of
the heap without scanning the whole list. Riders cancel all the time, and I
also want to bump old requests up, so next to the heap list I keep a
dictionary that says where each key sits. With that, push, pop, remove and
change_priority are all O(log n) and reading a key's priority is O(1).

Priorities are compared as tuples, e.g. (tier, request time). If two keys
have the same priority, the one that came in first comes out first.

What changed in Phase 3 (same O(log n) bounds, smaller constants)
------------------------------------------------------------------
1. Each entry is an immutable tuple (priority, arrival number, key). The
   arrival number is unique, so comparing two whole tuples never gets as
   far as the key. Phase 2 compared ``entry[:2]`` slices, which built two
   new lists on every comparison.
2. Sifting moves a "hole" instead of swapping: the entry being placed is
   held aside, the others slide into the gap, and it is written once at the
   end. That is about half the list writes and index updates of swapping.
3. ``push_many`` adds a batch. For a big batch it rebuilds the heap
   bottom-up in O(n), which is cheaper than n separate pushes.
"""

import itertools


class IndexedMinHeap:
    """Min-heap where each entry can be found and changed by its key."""

    __slots__ = ("_heap", "_index", "_counter")

    def __init__(self):
        self._heap = []      # list of (priority, arrival number, key)
        self._index = {}     # key -> position of that entry in self._heap
        self._counter = itertools.count()

    def __len__(self):
        return len(self._heap)

    def __bool__(self):
        return bool(self._heap)

    def __contains__(self, key):
        return key in self._index

    def all_keys(self):
        """List of every key currently in the heap (order is not meaningful)."""
        return list(self._index)

    def get_priority(self, key):
        """Current priority of a key; KeyError if it isn't in the heap."""
        return self._heap[self._index[key]][0]

    def push(self, key, priority):
        """Add a key with the given priority. O(log n)."""
        if key in self._index:
            raise KeyError(f"{key!r} is already in the heap")
        self._heap.append((priority, next(self._counter), key))
        self._sift_up(len(self._heap) - 1)

    def push_many(self, items):
        """Add several (key, priority) pairs at once.

        Small batches are pushed one by one. If the batch is at least as big
        as the heap already is, everything is rebuilt bottom-up in O(n).
        """
        items = list(items)
        keys = [key for key, _ in items]
        if len(set(keys)) != len(keys) or any(key in self._index for key in keys):
            raise KeyError("push_many got a key that is repeated or already in the heap")
        if len(items) < len(self._heap):
            for key, priority in items:
                self.push(key, priority)
            return
        heap, index = self._heap, self._index
        for key, priority in items:
            heap.append((priority, next(self._counter), key))
        for pos, entry in enumerate(heap):
            index[entry[2]] = pos
        for pos in range(len(heap) // 2 - 1, -1, -1):
            self._sift_down(pos)

    def peek(self):
        """(key, priority) of the smallest entry, without taking it out. O(1)."""
        if not self._heap:
            raise IndexError("peek on an empty heap")
        priority, _, key = self._heap[0]
        return key, priority

    def pop(self):
        """Take out and return (key, priority) of the smallest entry. O(log n)."""
        heap = self._heap
        if not heap:
            raise IndexError("pop from an empty heap")
        priority, _, key = heap[0]
        del self._index[key]
        last = heap.pop()
        if heap:
            heap[0] = last
            self._sift_down(0)
        return key, priority

    def remove(self, key):
        """Delete a key from wherever it is in the heap. O(log n)."""
        pos = self._index.pop(key)
        heap = self._heap
        last = heap.pop()
        if pos < len(heap):
            # The last entry fills the gap; it may belong higher or lower.
            heap[pos] = last
            if pos > 0 and last < heap[(pos - 1) >> 1]:
                self._sift_up(pos)
            else:
                self._sift_down(pos)

    def change_priority(self, key, priority):
        """Give a key a new priority (higher or lower). O(log n)."""
        pos = self._index[key]
        old_priority, arrival, _ = self._heap[pos]
        self._heap[pos] = (priority, arrival, key)
        if priority < old_priority:
            self._sift_up(pos)
        elif old_priority < priority:
            self._sift_down(pos)

    def _sift_up(self, pos):
        heap, index = self._heap, self._index
        entry = heap[pos]
        while pos > 0:
            parent = (pos - 1) >> 1
            above = heap[parent]
            if not entry < above:
                break
            heap[pos] = above
            index[above[2]] = pos
            pos = parent
        heap[pos] = entry
        index[entry[2]] = pos

    def _sift_down(self, pos):
        heap, index = self._heap, self._index
        size = len(heap)
        entry = heap[pos]
        child = 2 * pos + 1
        while child < size:
            right = child + 1
            if right < size and heap[right] < heap[child]:
                child = right
            below = heap[child]
            if not below < entry:
                break
            heap[pos] = below
            index[below[2]] = pos
            pos = child
            child = 2 * pos + 1
        heap[pos] = entry
        index[entry[2]] = pos
