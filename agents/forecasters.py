"""
Forecasting utilities for HEMS agents.

This module provides:
1. Mock/simulated forecasting functions (sinusoidal models)
2. Data loading from CSV files
3. Combined forecasts for optimization

Can be extended with real APIs or ML models as needed.
"""

import csv
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from datetime import datetime


# ==========================================
# DATA LOADING FROM CSV
# ==========================================

def load_timeseries_csv(
    filepath: str,
    timestamp_col: str = "timestamp",
) -> Dict[str, Any]:
    """
    Load timeseries data from a CSV file.
    
    Args:
        filepath: Path to CSV file
        timestamp_col: Name of timestamp column
    
    Returns:
        Dict with columns as lists
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    data: Dict[str, List] = {}
    timestamps: List[str] = []
    
    with open(filepath, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            for key, value in row.items():
                if key not in data:
                    data[key] = []
                
                # Try to convert to float, keep as string if fails
                try:
                    data[key].append(float(value))
                except (ValueError, TypeError):
                    data[key].append(value)
    
    hours = len(data.get(timestamp_col, []))
    
    return {
        "type": "csv_data",
        "filepath": filepath,
        "hours": hours,
        "columns": list(data.keys()),
        "data": data,
    }


def load_household_data(filepath: str = None) -> Dict[str, Any]:
    """
    Load household energy data from CSV.
    
    Expected columns: timestamp, load_kw, pv_generation_kw, grid_price_cents, temperature_c
    
    Args:
        filepath: Path to CSV (defaults to sample data)
    
    Returns:
        Structured forecast-like dict
    """
    if filepath is None:
        # Default to sample data in project
        base_dir = Path(__file__).parent.parent
        filepath = str(base_dir / "data" / "sample_household_24h.csv")
    
    raw = load_timeseries_csv(filepath)
    data = raw["data"]
    
    return {
        "type": "household_data",
        "source": filepath,
        "hours": raw["hours"],
        "load_kw": data.get("load_kw", []),
        "pv_generation_kw": data.get("pv_generation_kw", []),
        "grid_price_cents": data.get("grid_price_cents", []),
        "temperature_c": data.get("temperature_c", []),
        "timestamps": data.get("timestamp", []),
        "summary": {
            "total_load_kwh": round(sum(data.get("load_kw", [])), 2),
            "total_pv_kwh": round(sum(data.get("pv_generation_kw", [])), 2),
            "avg_price": round(sum(data.get("grid_price_cents", [])) / max(1, len(data.get("grid_price_cents", []))), 2),
            "peak_load_kw": max(data.get("load_kw", [0])),
            "peak_pv_kw": max(data.get("pv_generation_kw", [0])),
        },
    }


# ==========================================
# ELECTRICITY PRICE FORECASTING
# ==========================================

def price_forecast(
    hours: int = 24,
    base_price: float = 50.0,
    amplitude: float = 15.0,
    period: float = 24.0,
    phase: float = 0.0,
) -> Dict[str, Any]:
    """
    Generate electricity price forecast.
    
    Uses sinusoidal pattern - cheap at night, expensive during day.
    
    Args:
        hours: Number of hours to forecast
        base_price: Average price (cents/kWh)
        amplitude: Peak deviation from base
        period: Cycle length in hours
        phase: Phase shift (radians)
    
    Returns:
        Dict with forecast data
    """
    omega = 2 * math.pi / period
    prices = [
        round(base_price + amplitude * math.sin(omega * t + phase), 2)
        for t in range(hours)
    ]
    
    return {
        "type": "price",
        "hours": hours,
        "prices_cents_kwh": prices,
        "min_price": min(prices),
        "max_price": max(prices),
        "avg_price": round(sum(prices) / len(prices), 2),
        "cheapest_hour": prices.index(min(prices)),
        "most_expensive_hour": prices.index(max(prices)),
    }


def sinusoidal_prices(
    t: Iterable[float],
    base_price: float = 50.0,
    amplitude: float = 10.0,
    period: float = 24.0,
    phase: float = 0.0,
) -> List[float]:
    """
    Generate sinusoidal price values (legacy function).
    
    Parameters:
        t: iterable of time values (e.g. hours)
        base_price: average price level
        amplitude: peak deviation from base_price
        period: length of one full cycle (same unit as t)
        phase: phase shift (radians)
    
    Returns:
        list of price values
    """
    omega = 2 * math.pi / period
    return [
        base_price + amplitude * math.sin(omega * ti + phase)
        for ti in t
    ]


# ==========================================
# SOLAR (PV) GENERATION FORECASTING
# ==========================================

def solar_forecast(
    hours: int = 24,
    peak_capacity_kw: float = 5.0,
    efficiency: float = 0.85,
    cloud_factor: float = 1.0,
    sunrise_hour: int = 6,
    sunset_hour: int = 18,
) -> Dict[str, Any]:
    """
    Generate solar power generation forecast.
    
    Uses sinusoidal pattern during daylight hours, zero at night.
    
    Args:
        hours: Number of hours to forecast
        peak_capacity_kw: Maximum panel output
        efficiency: Panel efficiency (0-1)
        cloud_factor: Weather adjustment (0-1, 1=clear sky)
        sunrise_hour: Hour when generation starts
        sunset_hour: Hour when generation ends
    
    Returns:
        Dict with forecast data
    """
    generation = []
    
    for hour in range(hours):
        hour_of_day = hour % 24
        
        if sunrise_hour <= hour_of_day <= sunset_hour:
            # Sine wave during daylight
            daylight_duration = sunset_hour - sunrise_hour
            normalized_hour = (hour_of_day - sunrise_hour) / daylight_duration
            solar_factor = math.sin(math.pi * normalized_hour)
            output = peak_capacity_kw * efficiency * cloud_factor * solar_factor
        else:
            output = 0.0
        
        generation.append(round(output, 2))
    
    total_energy = round(sum(generation), 2)
    
    return {
        "type": "solar",
        "hours": hours,
        "generation_kw": generation,
        "total_energy_kwh": total_energy,
        "peak_output_kw": max(generation),
        "peak_hour": generation.index(max(generation)) if max(generation) > 0 else None,
        "parameters": {
            "peak_capacity_kw": peak_capacity_kw,
            "efficiency": efficiency,
            "cloud_factor": cloud_factor,
        },
    }


# ==========================================
# HOUSEHOLD LOAD PROFILE
# ==========================================

def load_forecast(
    hours: int = 24,
    base_load_kw: float = 0.5,
    morning_peak_kw: float = 2.0,
    evening_peak_kw: float = 3.0,
    morning_peak_hour: int = 7,
    evening_peak_hour: int = 19,
) -> Dict[str, Any]:
    """
    Generate household load (demand) forecast.
    
    Typical residential pattern: morning peak, low midday, evening peak.
    
    Args:
        hours: Number of hours to forecast
        base_load_kw: Minimum constant load
        morning_peak_kw: Morning peak demand
        evening_peak_kw: Evening peak demand
        morning_peak_hour: Hour of morning peak
        evening_peak_hour: Hour of evening peak
    
    Returns:
        Dict with forecast data
    """
    load = []
    
    for hour in range(hours):
        hour_of_day = hour % 24
        
        # Morning peak (Gaussian-like)
        morning_factor = math.exp(-0.5 * ((hour_of_day - morning_peak_hour) / 1.5) ** 2)
        morning_load = (morning_peak_kw - base_load_kw) * morning_factor
        
        # Evening peak (Gaussian-like, wider)
        evening_factor = math.exp(-0.5 * ((hour_of_day - evening_peak_hour) / 2.0) ** 2)
        evening_load = (evening_peak_kw - base_load_kw) * evening_factor
        
        # Night reduction (lower base load)
        if 0 <= hour_of_day < 6:
            night_factor = 0.7
        else:
            night_factor = 1.0
        
        total = base_load_kw * night_factor + morning_load + evening_load
        load.append(round(total, 2))
    
    total_energy = round(sum(load), 2)
    
    return {
        "type": "load",
        "hours": hours,
        "load_kw": load,
        "total_energy_kwh": total_energy,
        "peak_load_kw": max(load),
        "min_load_kw": min(load),
        "peak_hour": load.index(max(load)),
    }


# ==========================================
# WEATHER FORECAST
# ==========================================

def weather_forecast(
    hours: int = 24,
    base_temp_c: float = 20.0,
    temp_amplitude: float = 8.0,
    cloud_cover_base: float = 0.3,
) -> Dict[str, Any]:
    """
    Generate mock weather forecast.
    
    Args:
        hours: Number of hours to forecast
        base_temp_c: Average temperature
        temp_amplitude: Temperature variation
        cloud_cover_base: Base cloud cover (0-1)
    
    Returns:
        Dict with weather data
    """
    temps = []
    cloud_cover = []
    
    for hour in range(hours):
        hour_of_day = hour % 24
        
        # Temperature: coldest at 5am, warmest at 3pm
        temp_phase = -math.pi / 2 + (hour_of_day - 5) * math.pi / 12
        temp = base_temp_c + temp_amplitude * math.sin(temp_phase)
        temps.append(round(temp, 1))
        
        # Cloud cover: random-ish variation
        cloud_var = 0.2 * math.sin(hour * 0.7)
        cloud = max(0, min(1, cloud_cover_base + cloud_var))
        cloud_cover.append(round(cloud, 2))
    
    # Clear sky factor (inverse of cloud cover)
    clear_sky = [round(1 - c, 2) for c in cloud_cover]
    
    return {
        "type": "weather",
        "hours": hours,
        "temperature_c": temps,
        "cloud_cover": cloud_cover,
        "clear_sky_factor": clear_sky,
        "avg_temp_c": round(sum(temps) / len(temps), 1),
        "avg_cloud_cover": round(sum(cloud_cover) / len(cloud_cover), 2),
    }


# ==========================================
# EV CHARGING SCHEDULE
# ==========================================

def ev_schedule(
    current_soc: float = 0.3,
    target_soc: float = 0.8,
    departure_hour: int = 8,
    battery_capacity_kwh: float = 60.0,
    max_charge_kw: float = 7.0,
    current_hour: int = 22,
) -> Dict[str, Any]:
    """
    Calculate EV charging schedule.
    
    Determines optimal charging window to reach target SoC by departure.
    
    Args:
        current_soc: Current state of charge (0-1)
        target_soc: Target state of charge (0-1)
        departure_hour: Hour when EV must be ready
        battery_capacity_kwh: EV battery capacity
        max_charge_kw: Maximum charging power
        current_hour: Current hour
    
    Returns:
        Dict with charging schedule
    """
    # Energy needed
    energy_needed_kwh = (target_soc - current_soc) * battery_capacity_kwh
    
    if energy_needed_kwh <= 0:
        return {
            "type": "ev_schedule",
            "charging_needed": False,
            "current_soc": current_soc,
            "target_soc": target_soc,
            "message": "Already at or above target SoC",
        }
    
    # Hours until departure
    if departure_hour > current_hour:
        hours_available = departure_hour - current_hour
    else:
        hours_available = 24 - current_hour + departure_hour
    
    # Time needed to charge
    hours_to_charge = math.ceil(energy_needed_kwh / max_charge_kw)
    
    # Can we make it?
    feasible = hours_to_charge <= hours_available
    
    # Optimal start time (charge as late as possible for grid flexibility)
    if feasible:
        start_hour = (departure_hour - hours_to_charge) % 24
    else:
        start_hour = current_hour  # Start immediately
    
    return {
        "type": "ev_schedule",
        "charging_needed": True,
        "feasible": feasible,
        "current_soc": current_soc,
        "target_soc": target_soc,
        "energy_needed_kwh": round(energy_needed_kwh, 2),
        "hours_to_charge": hours_to_charge,
        "hours_available": hours_available,
        "recommended_start_hour": start_hour,
        "departure_hour": departure_hour,
        "charge_power_kw": max_charge_kw,
    }


# ==========================================
# COMBINED FORECAST
# ==========================================

def combined_forecast(
    hours: int = 24,
    pv_capacity_kw: float = 5.0,
    battery_capacity_kwh: float = 13.5,
) -> Dict[str, Any]:
    """
    Generate combined forecast for HEMS optimization.
    
    Returns all forecasts in one call for the optimizer.
    """
    price = price_forecast(hours=hours)
    solar = solar_forecast(hours=hours, peak_capacity_kw=pv_capacity_kw)
    load = load_forecast(hours=hours)
    weather = weather_forecast(hours=hours)
    
    # Calculate net load (load - solar)
    net_load = [
        round(load["load_kw"][i] - solar["generation_kw"][i], 2)
        for i in range(hours)
    ]
    
    return {
        "type": "combined",
        "hours": hours,
        "price": price,
        "solar": solar,
        "load": load,
        "weather": weather,
        "net_load_kw": net_load,
        "summary": {
            "total_solar_kwh": solar["total_energy_kwh"],
            "total_load_kwh": load["total_energy_kwh"],
            "net_import_kwh": round(sum(max(0, nl) for nl in net_load), 2),
            "net_export_kwh": round(sum(abs(min(0, nl)) for nl in net_load), 2),
            "cheapest_hour": price["cheapest_hour"],
            "peak_solar_hour": solar["peak_hour"],
            "peak_load_hour": load["peak_hour"],
        },
    }

