"""Small made-up road networks used by the demo, the tests and the simulation."""

from .road_network import RoadNetwork


def build_grid_city(cols, rows, block_m=200.0, block_secs=30.0, network_cls=RoadNetwork):
    """Make a ``cols`` x ``rows`` street grid as a RoadNetwork.

    ``network_cls`` lets the benchmarks build the same city with the frozen
    Phase 2 RoadNetwork class.

    Nodes are named (col, row) and are ``block_m`` metres apart. Each block
    takes ``block_secs`` seconds to drive, in either direction.
    """
    if cols <= 0 or rows <= 0:
        raise ValueError("the city needs at least one column and one row")
    city = network_cls()
    for c in range(cols):
        for r in range(rows):
            city.add_node((c, r), c * block_m, r * block_m)
    for c in range(cols):
        for r in range(rows):
            if c + 1 < cols:
                city.add_street((c, r), (c + 1, r), block_secs)
            if r + 1 < rows:
                city.add_street((c, r), (c, r + 1), block_secs)
    return city


def add_river_barrier(city, west_col, rows, bridge_rows):
    """Close the east-west streets between ``west_col`` and the column after it.

    Only the rows in ``bridge_rows`` stay open, so anyone crossing has to go
    round to a bridge. This is where straight-line distance and real driving
    time disagree the most.
    """
    for r in range(rows):
        if r not in bridge_rows:
            city.close_street((west_col, r), (west_col + 1, r))
