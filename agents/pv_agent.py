from typing import Any, Dict, List, Optional

from .dynamic_agent import DynamicAgent
from .forecasters import solar_forecast


class PVAgent(DynamicAgent):
    """
    PV Agent - Manages the Photovoltaic (Solar Panel) system.
    
    Responsibilities:
    1. Track current power generation
    2. Provide generation forecasts (via forecasters module)
    3. Report panel status and efficiency
    
    CAPABILITIES match method names for LLM invocation.
    """
    
    # Catalog metadata
    TYPE = "pv"
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
        self._efficiency = efficiency
        self.current_output_kw = 0.0
        self.panel_status = "operational"
    
    # ==========================================
    # CAPABILITY METHODS (match CAPABILITIES list)
    # ==========================================
    
    def generation(self) -> Dict[str, Any]:
        """Get current power generation."""
        return {
            "current_output_kw": self.current_output_kw,
            "peak_capacity_kw": self.peak_capacity_kw,
            "utilization_percent": round((self.current_output_kw / self.peak_capacity_kw) * 100, 1) if self.peak_capacity_kw > 0 else 0,
        }
    
    def forecast(self, hours: int = 24) -> Dict[str, Any]:
        """
        Generate a solar power forecast using the forecasters module.
        Delegates to solar_forecast() with this agent's parameters.
        """
        # Use the centralized forecaster function
        result = solar_forecast(
            hours=hours,
            peak_capacity_kw=self.peak_capacity_kw,
            efficiency=self._efficiency,
        )
        return result
    
    def status(self) -> Dict[str, Any]:
        """Get panel status and health."""
        return {
            "panel_status": self.panel_status,
            "current_output_kw": self.current_output_kw,
            "is_generating": self.current_output_kw > 0,
        }
    
    def efficiency(self) -> Dict[str, Any]:
        """Get efficiency metrics."""
        return {
            "efficiency_percent": round(self._efficiency * 100, 1),
            "peak_capacity_kw": self.peak_capacity_kw,
            "effective_capacity_kw": round(self.peak_capacity_kw * self._efficiency, 2),
        }


    # ==========================================
                # Helper methods 
    # ==========================================
    def set_output(self, output_kw: float):
        """Set current power output (called by simulation or real sensor)."""
        self.current_output_kw = min(output_kw, self.peak_capacity_kw)
    
    def info(self) -> Dict[str, Any]:
        """Return agent info for LLM/catalog."""
        base_info = super().info()
        base_info.update({
            "type": self.TYPE,
            "peak_capacity_kw": self.peak_capacity_kw,
            "efficiency_percent": round(self._efficiency * 100, 1),
            "current_output_kw": self.current_output_kw,
            "panel_status": self.panel_status,
            "daily_estimate_kwh": self.forecast(24)["total_energy_kwh"],
        })
        return base_info
    
    def handle_message(self, content, meta):
        """Handle incoming messages by routing to capability methods."""
        c = str(content).lower()
        
        if "forecast" in c:
            result = self.forecast()
            print(f"[{self.name}] Forecast: {result}")
            return result
        
        elif "generation" in c or "output" in c:
            result = self.generation()
            print(f"[{self.name}] Generation: {result}")
            return result
        
        elif "status" in c:
            result = self.status()
            print(f"[{self.name}] Status: {result}")
            return result
        
        elif "efficiency" in c:
            result = self.efficiency()
            print(f"[{self.name}] Efficiency: {result}")
            return result
        
        elif "info" in c:
            result = self.info()
            print(f"[{self.name}] Info: {result}")
            return result
        
        else:
            super().handle_message(content, meta)

