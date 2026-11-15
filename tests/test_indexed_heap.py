"""Tests for the indexed priority queue (heap order, index, cancel, re-rank)."""

import random
import unittest

from dispatch import IndexedMinHeap


class IndexedMinHeapTests(unittest.TestCase):
    def test_pop_order_and_arrival_ties(self):
        queue = IndexedMinHeap()
        for key, priority in [("c", 3), ("a", 1), ("b1", 2), ("b2", 2)]:
            queue.push(key, priority)
        self.assertEqual([queue.pop()[0] for _ in range(4)], ["a", "b1", "b2", "c"])

    def test_empty_heap_errors(self):
        queue = IndexedMinHeap()
        with self.assertRaises(IndexError):
            queue.pop()
        with self.assertRaises(IndexError):
            queue.peek()
        with self.assertRaises(KeyError):
            queue.remove("missing")

    def test_duplicate_key_rejected(self):
        queue = IndexedMinHeap()
        queue.push("a", 1)
        with self.assertRaises(KeyError):
            queue.push("a", 2)

    def test_random_operations_match_plain_dict(self):
        rng = random.Random(11)
        queue, expected = IndexedMinHeap(), {}
        for step in range(3000):
            roll = rng.random()
            if roll < 0.45 or not expected:
                key = f"k{step}"
                priority = rng.randint(0, 100)
                queue.push(key, priority)
                expected[key] = priority
            elif roll < 0.65:
                key = rng.choice(list(expected))
                queue.remove(key)
                del expected[key]
            elif roll < 0.85:
                key = rng.choice(list(expected))
                expected[key] = rng.randint(0, 100)
                queue.change_priority(key, expected[key])
            else:
                key, priority = queue.pop()
                self.assertEqual(priority, min(expected.values()))
                self.assertEqual(expected.pop(key), priority)
            self.assertEqual(len(queue), len(expected))
            for child in range(1, len(queue._heap)):
                self.assertLessEqual(queue._heap[(child - 1) // 2][:2], queue._heap[child][:2])
            for key, slot in queue._index.items():
                self.assertEqual(queue._heap[slot][2], key)

    # ----- added in Phase 2 -------------------------------------------

    def test_single_entry_and_all_keys(self):
        queue = IndexedMinHeap()
        queue.push("only", (2, 0.0))
        self.assertEqual(queue.peek(), ("only", (2, 0.0)))
        self.assertEqual(queue.all_keys(), ["only"])
        self.assertEqual(queue.pop(), ("only", (2, 0.0)))
        self.assertEqual(len(queue), 0)
        self.assertNotIn("only", queue)

    def test_change_priority_moves_entry_both_ways(self):
        queue = IndexedMinHeap()
        for key, priority in [("a", 5), ("b", 6), ("c", 7)]:
            queue.push(key, priority)
        queue.change_priority("c", 1)
        self.assertEqual(queue.peek()[0], "c")
        queue.change_priority("c", 9)
        self.assertEqual([queue.pop()[0] for _ in range(3)], ["a", "b", "c"])
        with self.assertRaises(KeyError):
            queue.change_priority("c", 1)

    def test_tuple_priorities_order_tier_then_time(self):
        queue = IndexedMinHeap()
        queue.push("standard_early", (2, 0.0))
        queue.push("premium_late", (1, 50.0))
        queue.push("premium_early", (1, 10.0))
        self.assertEqual([queue.pop()[0] for _ in range(3)],
                         ["premium_early", "premium_late", "standard_early"])


if __name__ == "__main__":
    unittest.main()
