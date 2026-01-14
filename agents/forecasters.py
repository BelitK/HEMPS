# this is for forecasting and load profile usage within agents
# this can be changed to any forecasting method as needed or even live time systems can be added


import math
import random
from typing import Iterable, List

import pandas as pd

from .dynamic_agent import DynamicAgent


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



class PVForecastAgent(DynamicAgent):
    # Catalog metadata (required for agent factory & catalog)
    TYPE = "pv forecast"
    LABEL = "PV Forecast Agent"
    DEFAULT_PERSONA = "This Agent can retreive a timeseries for available capacity from the PV-System (PV-Agent) from a database. The timeseries have 15-min resolution"
    DEFAULT_USAGE = "Supply forecast timeseries for PV production."
    CAPABILITIES = ["get forecast", "refresh forecast"]
    
    
    # "database connection" (file location)
    PARQUET_PATH = "forecast_database/slp_pv.parquet"
    
    
    def __init__(
        self,
        name: str = "forecast agent",
        persona: str | None = None,
        usage: str | None = None,
        value_column: str = "value",
    ):
        super().__init__(
            name=name,
            persona=persona or self.DEFAULT_PERSONA,
            usage=usage or self.DEFAULT_USAGE,
        )

        self.forecast: list[int]  # exactly 96 values (15-min day)

        df = pd.read_parquet(self.PARQUET_PATH, engine="pyarrow")

        # derive day key WITHOUT pandas .dt helpers
        df["day"] = df["timestamp"].values.astype("datetime64[D]")

        full_days = [g for _, g in df.groupby("day") if len(g) == 96]
        if not full_days:
            raise ValueError("No complete 15-min days found")

        day_df = random.choice(full_days)

        self.forecast = day_df["Profilwert"].fillna(0).astype(int).tolist()
        self._df = pd.read_parquet(self.PARQUET_PATH, engine="pyarrow")

        
    def get_forecast(self) -> list[int]:
        return self.forecast



    def refresh_forecast(self) -> None:
        df = self._df  # loaded once in __init__

        df["day"] = df["timestamp"].values.astype("datetime64[D]")

        full_days = [g for _, g in df.groupby("day") if len(g) == 96]
        if not full_days:
            raise ValueError("No complete 15-min days found")

        day_df = random.choice(full_days)

        self.forecast = day_df["Profilwert"].to_numpy(dtype=int).tolist()




