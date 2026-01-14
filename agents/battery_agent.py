from typing import Any, Dict, List, Optional

from .dynamic_agent import DynamicAgent


class BatteryAgent(DynamicAgent):
    """
    Battery Agent - Manages home battery storage system.
    
    Responsibilities:
    1. Track State of Charge (SoC)
    2. Handle charge/discharge commands
    3. Enforce power and capacity limits
    4. Report battery status and health
    """
    
    # Catalog metadata
    TYPE = "battery"
    LABEL = "Battery Agent"
    DEFAULT_PERSONA = "Manages home battery storage, tracks charge levels and handles energy flow."
    DEFAULT_USAGE = "Responds to charge/discharge commands and reports energy storage status."
    CAPABILITIES = ["charge", "discharge", "soc", "status", "schedule"]
    
    def __init__(
        self,
        name: str = "battery_agent",
        persona: str | None = None,
        usage: str | None = None,
        capacity_kwh: float = 13.5,
        max_charge_kw: float = 5.0,
        max_discharge_kw: float = 5.0,
        initial_soc: float = 0.5,
        min_soc: float = 0.1,
        max_soc: float = 0.95,
    ):
        super().__init__(
            name=name,
            persona=persona or self.DEFAULT_PERSONA,
            usage=usage or self.DEFAULT_USAGE,
        )
        # Battery parameters
        self.capacity_kwh = capacity_kwh
        self.max_charge_kw = max_charge_kw
        self.max_discharge_kw = max_discharge_kw
        self.min_soc = min_soc
        self.max_soc = max_soc
        
        # Current state
        self.current_soc = initial_soc
        self.current_power_kw = 0.0  # Positive = charging, Negative = discharging
        self.battery_status = "idle"  # idle, charging, discharging
        self.health_percent = 100.0
        self.cycle_count = 0
    
    def get_soc(self) -> float:
        """Get current State of Charge (0.0 to 1.0)."""
        return self.current_soc
    
    def get_soc_percent(self) -> float:
        """Get current State of Charge as percentage."""
        return round(self.current_soc * 100, 1)
    
    def get_available_energy_kwh(self) -> float:
        """Get available energy above minimum SoC."""
        usable_soc = self.current_soc - self.min_soc
        return round(max(0, usable_soc * self.capacity_kwh), 2)
    
    def get_available_capacity_kwh(self) -> float:     
        """Get remaining capacity below maximum SoC."""
        headroom_soc = self.max_soc - self.current_soc
        return round(max(0, headroom_soc * self.capacity_kwh), 2)
    
    def charge(self, power_kw: float, duration_hours: float = 1.0) -> Dict[str, Any]:
        """
        Charge the battery.
        Returns result with actual energy transferred.
        """
        # Limit to max charge rate
        actual_power = min(power_kw, self.max_charge_kw)
        
        # Calculate energy to add
        energy_kwh = actual_power * duration_hours
        available_capacity = self.get_available_capacity_kwh()
        actual_energy = min(energy_kwh, available_capacity)
        
        # Update SoC
        soc_increase = actual_energy / self.capacity_kwh
        self.current_soc = min(self.max_soc, self.current_soc + soc_increase)
        
        # Update state
        self.current_power_kw = actual_power
        self.battery_status = "charging" if actual_energy > 0 else "full"
        
        result = {
            "action": "charge",
            "requested_kw": power_kw,
            "actual_kw": actual_power,
            "energy_added_kwh": round(actual_energy, 2),
            "new_soc_percent": self.get_soc_percent(),
        }
        print(f"[{self.name}] Charged: {result}")
        return result
    
    def discharge(self, power_kw: float, duration_hours: float = 1.0) -> Dict[str, Any]:
        """
        Discharge the battery.
        Returns result with actual energy transferred.
        """
        # Limit to max discharge rate
        actual_power = min(power_kw, self.max_discharge_kw)
        
        # Calculate energy to remove
        energy_kwh = actual_power * duration_hours
        available_energy = self.get_available_energy_kwh()
        actual_energy = min(energy_kwh, available_energy)
        
        # Update SoC
        soc_decrease = actual_energy / self.capacity_kwh
        self.current_soc = max(self.min_soc, self.current_soc - soc_decrease)
        
        # Update state
        self.current_power_kw = -actual_power
        self.battery_status = "discharging" if actual_energy > 0 else "empty"
        self.cycle_count += actual_energy / self.capacity_kwh  # Partial cycle count
        
        result = {
            "action": "discharge",
            "requested_kw": power_kw,
            "actual_kw": actual_power,
            "energy_delivered_kwh": round(actual_energy, 2),
            "new_soc_percent": self.get_soc_percent(),
        }
        print(f"[{self.name}] Discharged: {result}")
        return result
    
    def stop(self) -> Dict[str, Any]:
        """Stop charging/discharging."""
        self.current_power_kw = 0.0
        self.battery_status = "idle"
        return {"action": "stop", "status": "idle"}
    
    def info(self) -> Dict[str, Any]:
        """Return agent info for LLM/catalog."""
        base_info = super().info()
        base_info.update({
            "type": self.TYPE,
            "capacity_kwh": self.capacity_kwh,
            "soc_percent": self.get_soc_percent(),
            "available_energy_kwh": self.get_available_energy_kwh(),
            "available_capacity_kwh": self.get_available_capacity_kwh(),
            "current_power_kw": self.current_power_kw,
            "battery_status": self.battery_status,
            "health_percent": self.health_percent,
            "cycle_count": round(self.cycle_count, 1),
            "limits": {
                "max_charge_kw": self.max_charge_kw,
                "max_discharge_kw": self.max_discharge_kw,
                "min_soc": self.min_soc,
                "max_soc": self.max_soc,
            },
        })
        return base_info
    
    def handle_message(self, content, meta):
        """Handle incoming messages."""
        c = str(content).lower()
        
        if "charge" in c and "discharge" not in c:
            # Extract power if specified, default to max
            return self.charge(self.max_charge_kw)
        
        elif "discharge" in c:
            return self.discharge(self.max_discharge_kw)
        
        elif "stop" in c or "idle" in c:
            return self.stop()
        
        elif "soc" in c or "level" in c or "percent" in c:
            soc = self.get_soc_percent()
            print(f"[{self.name}] Current SoC: {soc}%")
            return {"soc_percent": soc}
        
        elif "status" in c or "info" in c:
            info = self.info()
            print(f"[{self.name}] Status: {info}")
            return info
        
        else:
            super().handle_message(content, meta)
