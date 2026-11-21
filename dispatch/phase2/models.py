"""Plain data records used by the dispatcher and the simulation.

I put these in their own file so the data structure modules never need to
import the dispatcher, and so new fields (rating, car type, surge zone) can
be added later without touching the structures.
"""

from dataclasses import dataclass
from enum import Enum

# Lower number = served earlier.
TIER_PRIORITY = {"accessible": 0, "premium": 1, "standard": 2}


class DriverState(Enum):
    FREE = "free"          # online and available; kept in the grid
    BUSY = "busy"          # heading to a pickup or driving a rider
    OFFLINE = "offline"    # logged out; matching ignores them


@dataclass
class Driver:
    driver_id: str
    location: object
    state: DriverState = DriverState.FREE
    trips_done: int = 0

    @property
    def is_free(self):
        return self.state is DriverState.FREE


@dataclass
class TripRequest:
    request_id: str
    pickup: object
    dropoff: object
    tier: str
    requested_at: float
    bumps: int = 0    # how many tiers the request has moved up from waiting

    @property
    def level(self):
        """Tier number after bumps; 0 is the most urgent."""
        return max(TIER_PRIORITY[self.tier] - self.bumps, 0)


@dataclass
class Assignment:
    request_id: str
    driver_id: str
    pickup_time: float
    waited: float = 0.0   # seconds between the request and the assignment
