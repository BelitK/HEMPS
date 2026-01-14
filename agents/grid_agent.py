from typing import Any, Dict, List, Optional

from .dynamic_agent import DynamicAgent
from .forecasters import price_forecast, load_household_data


class GridAgent(DynamicAgent):
    """
    Grid Agent - Manages grid connection and electricity pricing.
    
    Responsibilities:
    1. Provide electricity price forecasts
    2. Track grid import/export
    3. Report grid status and tariffs
    
    CAPABILITIES match method names for LLM invocation.
    """
    
    # Catalog metadata
    TYPE = "grid"
    LABEL = "Grid Agent"
    DEFAULT_PERSONA = "Manages grid interactions, pricing, and import/export tracking."
    DEFAULT_USAGE = "Provides price forecasts and tracks energy flow with the utility grid."
    CAPABILITIES = ["pricing", "forecast", "status", "tariff"]
    
    def __init__(
        self,
        name: str = "grid_agent",
        persona: str | None = None,
        usage: str | None = None,
        base_price_cents: float = 50.0,
        feed_in_tariff_cents: float = 8.0,
    ):
        super().__init__(
            name=name,
            persona=persona or self.DEFAULT_PERSONA,
            usage=usage or self.DEFAULT_USAGE,
        )
        # Grid parameters
        self.base_price_cents = base_price_cents
        self.feed_in_tariff_cents = feed_in_tariff_cents
        self.current_import_kw = 0.0
        self.current_export_kw = 0.0
        self.grid_status = "connected"
        self.total_import_kwh = 0.0
        self.total_export_kwh = 0.0
    

    def pricing(self) -> Dict[str, Any]:
        """Get current pricing information."""
        return {
            "current_import_price_cents": self.base_price_cents,
            "feed_in_tariff_cents": self.feed_in_tariff_cents,
            "currency": "cents/kWh",
        }
    
    def forecast(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get electricity price forecast using forecasters module.
        """
        result = price_forecast(
            hours=hours,
            base_price=self.base_price_cents,
        )
        return result
    
    def status(self) -> Dict[str, Any]:
        """Get grid connection status."""
        return {
            "grid_status": self.grid_status,
            "current_import_kw": self.current_import_kw,
            "current_export_kw": self.current_export_kw,
            "total_import_kwh": round(self.total_import_kwh, 2),
            "total_export_kwh": round(self.total_export_kwh, 2),
            "is_importing": self.current_import_kw > 0,
            "is_exporting": self.current_export_kw > 0,
        }
    
    def tariff(self) -> Dict[str, Any]:
        """Get tariff structure."""
        # Get forecast to find cheapest/most expensive hours
        fc = self.forecast(24)
        return {
            "base_price_cents": self.base_price_cents,
            "feed_in_tariff_cents": self.feed_in_tariff_cents,
            "cheapest_hour": fc["cheapest_hour"],
            "cheapest_price": fc["min_price"],
            "peak_hour": fc["most_expensive_hour"],
            "peak_price": fc["max_price"],
        }
    
    # ==============
    # Helper methods 
    # ==============
    
    def set_import(self, power_kw: float):
        """Set current import from grid."""
        self.current_import_kw = max(0, power_kw)
        self.current_export_kw = 0.0
    
    def set_export(self, power_kw: float):
        """Set current export to grid."""
        self.current_export_kw = max(0, power_kw)
        self.current_import_kw = 0.0
    
    def record_energy(self, hours: float = 1.0):
        """Record energy imported/exported over time period."""
        self.total_import_kwh += self.current_import_kw * hours
        self.total_export_kwh += self.current_export_kw * hours
    
    def info(self) -> Dict[str, Any]:
        """Return agent info for LLM/catalog."""
        base_info = super().info()
        base_info.update({
            "type": self.TYPE,
            "grid_status": self.grid_status,
            "base_price_cents": self.base_price_cents,
            "feed_in_tariff_cents": self.feed_in_tariff_cents,
            "current_import_kw": self.current_import_kw,
            "current_export_kw": self.current_export_kw,
            "total_import_kwh": round(self.total_import_kwh, 2),
            "total_export_kwh": round(self.total_export_kwh, 2),
        })
        return base_info
    
    def handle_message(self, content, meta):
        """Handle incoming messages by routing to capability methods."""
        c = str(content).lower()
        
        if "forecast" in c or "price forecast" in c:
            result = self.forecast()
            print(f"[{self.name}] Forecast: {result}")
            return result
        
        elif "pricing" in c or "price" in c:
            result = self.pricing()
            print(f"[{self.name}] Pricing: {result}")
            return result
        
        elif "status" in c:
            result = self.status()
            print(f"[{self.name}] Status: {result}")
            return result
        
        elif "tariff" in c:
            result = self.tariff()
            print(f"[{self.name}] Tariff: {result}")
            return result
        
        elif "info" in c:
            result = self.info()
            print(f"[{self.name}] Info: {result}")
            return result
        
        else:
            super().handle_message(content, meta)
