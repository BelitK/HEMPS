# this is for forecasting and load profile usage within agents
# this can be changed to any forecasting method as needed or even live time systems can be added


import math
from typing import Iterable, List

def sinusoidal_prices(
    t: Iterable[float],
    base_price: float = 50.0,
    amplitude: float = 10.0,
    period: float = 24.0,
    phase: float = 0.0,
) -> List[float]:
    """
    Generate sinusoidal price values.

    Parameters:
    - t: iterable of time values (e.g. hours)
    - base_price: average price level
    - amplitude: peak deviation from base_price
    - period: length of one full cycle (same unit as t)
    - phase: phase shift (radians)

    Returns:
    - list of price values
    """
    omega = 2 * math.pi / period
    return [
        base_price + amplitude * math.sin(omega * ti + phase)
        for ti in t
    ]
