"""Binary min-heap with a lookup table, used for the waiting ride requests.

Python's ``heapq`` can't remove or re-prioritise something in the middle of
the heap without scanning the whole list. Riders cancel all the time, and I
also want to bump old requests up, so next to the heap list I keep a
dictionary that says where each key sits. With that, push, pop, remove and
change_priority are all O(log n) and reading a key's priority is O(1).

Priorities are compared as tuples, e.g. (tier, request time). If two keys
have the same priority, the one that came in first comes out first.
"""

import itertools


class IndexedMinHeap:
    """Min-heap where each entry can be found and changed by its key."""

    def __init__(self):
        self._heap = []      # list of [priority, arrival number, key]
        self._index = {}     # key -> position of that entry in self._heap
        self._counter = itertools.count()

    def __len__(self):
        return len(self._heap)

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
        self._heap.append([priority, next(self._counter), key])
        self._index[key] = len(self._heap) - 1
        self._sift_up(len(self._heap) - 1)

    def peek(self):
        """(key, priority) of the smallest entry, without taking it out. O(1)."""
        if not self._heap:
            raise IndexError("peek on an empty heap")
        priority, _, key = self._heap[0]
        return key, priority

    def pop(self):
        """Take out and return (key, priority) of the smallest entry. O(log n)."""
        if not self._heap:
            raise IndexError("pop from an empty heap")
        priority, _, key = self._heap[0]
        self._remove_at(0)
        return key, priority

    def remove(self, key):
        """Delete a key from wherever it is in the heap. O(log n)."""
        self._remove_at(self._index[key])

    def change_priority(self, key, priority):
        """Give a key a new priority (higher or lower). O(log n)."""
        pos = self._index[key]
        old = self._heap[pos][0]
        self._heap[pos][0] = priority
        if priority < old:
            self._sift_up(pos)
        else:
            self._sift_down(pos)

    def _remove_at(self, pos):
        last = len(self._heap) - 1
        if pos != last:
            self._swap(pos, last)
        _, _, key = self._heap.pop()
        del self._index[key]
        if pos < len(self._heap):
            # The entry that got moved into the gap might belong higher or
            # lower; only one of these two calls will actually move it.
            self._sift_up(pos)
            self._sift_down(pos)

    def _less(self, a, b):
        return self._heap[a][:2] < self._heap[b][:2]

    def _swap(self, a, b):
        heap = self._heap
        heap[a], heap[b] = heap[b], heap[a]
        self._index[heap[a][2]] = a
        self._index[heap[b][2]] = b

    def _sift_up(self, pos):
        while pos > 0:
            parent = (pos - 1) // 2
            if not self._less(pos, parent):
                break
            self._swap(pos, parent)
            pos = parent

    def _sift_down(self, pos):
        size = len(self._heap)
        while True:
            smallest = pos
            for child in (2 * pos + 1, 2 * pos + 2):
                if child < size and self._less(child, smallest):
                    smallest = child
            if smallest == pos:
                break
            self._swap(pos, smallest)
            pos = smallest
