"""Frozen copy of the Phase 2 proof of concept, kept only as a baseline.

Phase 3 optimises the main ``dispatch`` package. These files are the exact
Phase 2 versions (same code as commit c81a040) so the benchmarks and the
equivalence tests can compare old and new side by side. Nothing outside the
tests and the benchmarks should import from here.
"""

from .city_builder import add_river_barrier, build_grid_city
from .dispatcher import Dispatcher, slow_best_time
from .indexed_heap import IndexedMinHeap
from .road_network import RoadNetwork
from .simulation import SimSettings, simulate
from .spatial_grid import SpatialGrid

__all__ = ["SpatialGrid", "IndexedMinHeap", "RoadNetwork", "Dispatcher", "slow_best_time",
           "build_grid_city", "add_river_barrier", "SimSettings", "simulate"]
