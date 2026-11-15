"""Data structures and dispatch logic for a real-time ride-hailing service.

Modules
-------
spatial_grid    hash grid of free drivers with nearest-k search
indexed_heap    binary min-heap with a key index, for waiting requests
road_network    directed weighted road graph searched with Dijkstra
models          Driver / TripRequest / Assignment records
dispatcher      Dispatcher, which ties the structures together
city_builder    made-up street grids for the demo and the tests
simulation      small event-driven simulation of an hour of requests
"""

from .city_builder import add_river_barrier, build_grid_city
from .dispatcher import Dispatcher
from .indexed_heap import IndexedMinHeap
from .models import TIER_PRIORITY, Assignment, Driver, DriverState, TripRequest
from .road_network import RoadNetwork
from .spatial_grid import SpatialGrid

__all__ = [
    "SpatialGrid", "IndexedMinHeap", "RoadNetwork", "Dispatcher",
    "Driver", "DriverState", "TripRequest", "Assignment", "TIER_PRIORITY",
    "build_grid_city", "add_river_barrier",
]
