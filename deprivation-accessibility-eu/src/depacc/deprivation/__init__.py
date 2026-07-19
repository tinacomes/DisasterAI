"""Deprivation surfaces: pluggable DLF/DCF forms, soft-min reducer, 2SFCA
congestion inflation, nearest-facility baseline."""

from depacc.deprivation.functions import DeprivationFunction  # noqa: F401
from depacc.deprivation.softmin import softmin, grouped_softmin  # noqa: F401
from depacc.deprivation.catchment import (  # noqa: F401
    supply_demand_ratio,
    congestion_factor,
)
from depacc.deprivation.surfaces import (  # noqa: F401
    everyday_surface,
    emergency_surface,
)
