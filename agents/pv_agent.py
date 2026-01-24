from typing import Any, Dict, List, Optional
import math

from .dynamic_agent import DynamicAgent
from .forecasters import sinusoidal_prices  # Reuse for solar curve simulation


class PVAgent(DynamicAgent):
    """
    PV Agent - Manages the Photovoltaic (Solar Panel) system.
    
    Responsibilities:
    1. Track current power generation
    2. Provide generation forecasts
    3. Report panel status and efficiency
    """
    
    # Catalog metadata
    TYPE = "solar"
    LABEL = "PV Agent"
    DEFAULT_PERSONA = "Monitors solar panel generation and provides power forecasts."
    DEFAULT_USAGE = "Reports real-time solar output and predicts future generation."
    CAPABILITIES = ["generation", "forecast", "status", "efficiency"]
    
    def __init__(
        self,
        name: str = "pv_agent",
        persona: str | None = None,
        usage: str | None = None,
        peak_capacity_kw: float = 5.0,
        efficiency: float = 0.85,
    ):
        super().__init__(
            name=name,
            persona=persona or self.DEFAULT_PERSONA,
            usage=usage or self.DEFAULT_USAGE,
        )
        # PV system parameters
        self.peak_capacity_kw = peak_capacity_kw
        self.efficiency = efficiency
        self.current_output_kw = 0.0
        self.panel_status = "operational"
    
    def get_current_output(self) -> float:
        """Get current power output in kW."""
        return self.current_output_kw
    
    def set_current_output(self, output_kw: float):
        """Set current power output (called by simulation or real sensor)."""
        self.current_output_kw = min(output_kw, self.peak_capacity_kw)
    
    def get_generation_forecast(self, hours: int = 24) -> List[float]:
        """
        Generate a solar power forecast for the next N hours.
        Uses a sinusoidal model peaking at noon.
        """
        # Solar generation follows daylight - peak at hour 12
        time_values = list(range(hours))
        
        # Generate using sinusoidal pattern (peak at noon)
        forecast = []
        for hour in time_values:
            # Solar curve: 0 at night, peak at noon
            # Shift phase so peak is at hour 12
            hour_of_day = hour % 24
            if 6 <= hour_of_day <= 18:  # Daylight hours
                # Sine wave from 6am to 6pm
                solar_factor = math.sin(math.pi * (hour_of_day - 6) / 12)
                output = self.peak_capacity_kw * self.efficiency * solar_factor
            else:
                output = 0.0
            forecast.append(round(output, 2))
        
        return forecast
    
    def get_daily_energy_estimate(self) -> float:
        """Estimate total daily energy production in kWh."""
        forecast = self.get_generation_forecast(24)
        return round(sum(forecast), 2)
    
    def info(self) -> Dict[str, Any]:
        """Return agent info for LLM/catalog."""
        base_info = super().info()
        base_info.update({
            "type": self.TYPE,
            "peak_capacity_kw": self.peak_capacity_kw,
            "efficiency": self.efficiency,
            "current_output_kw": self.current_output_kw,
            "panel_status": self.panel_status,
            "daily_estimate_kwh": self.get_daily_energy_estimate(),
        })
        return base_info
    
    def handle_message(self, content, meta):
        """Handle incoming messages."""
        c = str(content).lower()
        
        if "forecast" in c or "predict" in c:
            forecast = self.get_generation_forecast(24)
            print(f"[{self.name}] Next 24h forecast (kW): {forecast}")
            return {"forecast": forecast}
        
        elif "output" in c or "current" in c or "generation" in c:
            output = self.get_current_output()
            print(f"[{self.name}] Current output: {output} kW")
            return {"current_output_kw": output}
        
        elif "status" in c:
            print(f"[{self.name}] Panel status: {self.panel_status}")
            return {"status": self.panel_status}
        
        elif "info" in c:
            info = self.info()
            print(f"[{self.name}] Full info: {info}")
            return info
        
        else:
            super().handle_message(content, meta)
