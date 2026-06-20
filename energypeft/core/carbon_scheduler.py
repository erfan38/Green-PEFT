# energypeft/core/carbon_scheduler.py
"""
Carbon Intensity Scheduler with Provider Pattern

Providers:
  - UKNationalGridProvider: Free UK API (no key needed)
  - ElectricityMapsProvider: Uses ELECTRICITY_MAPS_API_KEY env var
  - StaticFallbackProvider: Regional averages as fallback

Features:
  - Caching (default 5 minutes) to reduce API calls
  - Fallback chain: tries providers in order
  - Clean API unchanged: get_carbon_intensity(), wait_for_green_grid()
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests


# ============================================================
# Provider Interface
# ============================================================

class CarbonIntensityProvider(ABC):
    """Abstract base class for carbon intensity data sources."""

    @abstractmethod
    def get_intensity(self, zone: str) -> Optional[float]:
        """Return carbon intensity in gCO2/kWh, or None if unavailable."""
        pass

    @abstractmethod
    def supports_zone(self, zone: str) -> bool:
        """Return True if this provider can serve data for the given zone."""
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ============================================================
# Providers
# ============================================================

class UKNationalGridProvider(CarbonIntensityProvider):
    """Free UK National Grid Carbon Intensity API (no key required)."""

    API_URL = "https://api.carbonintensity.org.uk/intensity"

    def supports_zone(self, zone: str) -> bool:
        return zone.upper() == "GB"

    def get_intensity(self, zone: str) -> Optional[float]:
        if not self.supports_zone(zone):
            return None
        try:
            resp = requests.get(self.API_URL, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return float(data["data"][0]["intensity"]["actual"])
        except Exception:
            pass
        return None


class ElectricityMapsProvider(CarbonIntensityProvider):
    """
    Electricity Maps API (https://www.electricitymaps.com/)
    
    Requires API key in env var: ELECTRICITY_MAPS_API_KEY
    Free tier: 50 requests/hour, 1 zone
    """

    API_URL = "https://api.electricitymap.org/v3/carbon-intensity/latest"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ELECTRICITY_MAPS_API_KEY")

    def supports_zone(self, zone: str) -> bool:
        # Electricity Maps supports 50+ zones, we allow any if key is present
        return self.api_key is not None

    def get_intensity(self, zone: str) -> Optional[float]:
        if not self.api_key:
            return None
        try:
            resp = requests.get(
                self.API_URL,
                params={"zone": zone},
                headers={"auth-token": self.api_key},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                return float(data.get("carbonIntensity", 0))
        except Exception:
            pass
        return None


class StaticFallbackProvider(CarbonIntensityProvider):
    """
    Hardcoded regional averages as fallback.
    
    Sources: IEA, Electricity Maps historical averages
    """

    # gCO2/kWh averages (approximate)
    STATIC_VALUES: Dict[str, float] = {
        # Canada
        "CA-QC": 20.0,   # Quebec: 99% hydro
        "CA-ON": 40.0,   # Ontario: nuclear + hydro + gas
        "CA-BC": 15.0,   # BC: mostly hydro
        "CA-AB": 450.0,  # Alberta: coal + gas
        # Europe
        "FR": 50.0,      # France: nuclear
        "DE": 350.0,     # Germany: coal + renewables
        "NO": 20.0,      # Norway: hydro
        "SE": 25.0,      # Sweden: hydro + nuclear
        "PL": 650.0,     # Poland: coal
        "GB": 200.0,     # UK: mix (fallback if API fails)
        # US
        "US-CA": 200.0,  # California: renewables + gas
        "US-TX": 400.0,  # Texas: gas + wind
        "US-NY": 150.0,  # New York: nuclear + hydro
        "US-WA": 80.0,   # Washington: hydro
    }

    DEFAULT_DIRTY = 400.0  # Assume moderately dirty if unknown

    def supports_zone(self, zone: str) -> bool:
        return True  # Always supports (fallback)

    def get_intensity(self, zone: str) -> Optional[float]:
        zone_upper = zone.upper()
        return self.STATIC_VALUES.get(zone_upper, self.DEFAULT_DIRTY)


# ============================================================
# Caching Wrapper
# ============================================================

@dataclass
class CachedValue:
    value: float
    timestamp: float


class CachingProvider(CarbonIntensityProvider):
    """Wraps a provider with time-based caching."""

    def __init__(
        self,
        provider: CarbonIntensityProvider,
        ttl_sec: float = 300.0,  # 5 minutes
    ):
        self.provider = provider
        self.ttl_sec = ttl_sec
        self._cache: Dict[str, CachedValue] = {}

    @property
    def name(self) -> str:
        return f"Cached({self.provider.name})"

    def supports_zone(self, zone: str) -> bool:
        return self.provider.supports_zone(zone)

    def get_intensity(self, zone: str) -> Optional[float]:
        now = time.time()
        zone_key = zone.upper()

        # Check cache
        if zone_key in self._cache:
            cached = self._cache[zone_key]
            if (now - cached.timestamp) < self.ttl_sec:
                return cached.value

        # Fetch fresh
        value = self.provider.get_intensity(zone)
        if value is not None:
            self._cache[zone_key] = CachedValue(value=value, timestamp=now)
        return value


# ============================================================
# Fallback Chain
# ============================================================

class FallbackChainProvider(CarbonIntensityProvider):
    """Tries providers in order until one succeeds."""

    def __init__(self, providers: List[CarbonIntensityProvider]):
        if not providers:
            raise ValueError("At least one provider required")
        self.providers = providers

    @property
    def name(self) -> str:
        names = [p.name for p in self.providers]
        return f"Chain({' -> '.join(names)})"

    def supports_zone(self, zone: str) -> bool:
        return any(p.supports_zone(zone) for p in self.providers)

    def get_intensity(self, zone: str) -> Optional[float]:
        for provider in self.providers:
            if provider.supports_zone(zone):
                value = provider.get_intensity(zone)
                if value is not None:
                    return value
        return None


# ============================================================
# Default Provider Chain (module-level singleton)
# ============================================================

def _make_default_chain() -> FallbackChainProvider:
    """Create the default provider chain with caching."""
    ttl = float(os.environ.get("CARBON_CACHE_TTL_SEC", "300"))

    providers: List[CarbonIntensityProvider] = []

    # 1. Electricity Maps (if API key available)
    em = ElectricityMapsProvider()
    if em.api_key:
        providers.append(CachingProvider(em, ttl_sec=ttl))

    # 2. UK National Grid (free, no key)
    providers.append(CachingProvider(UKNationalGridProvider(), ttl_sec=ttl))

    # 3. Static fallback (always works)
    providers.append(StaticFallbackProvider())

    return FallbackChainProvider(providers)


_default_provider: Optional[FallbackChainProvider] = None


def get_default_provider() -> FallbackChainProvider:
    """Get or create the default provider chain."""
    global _default_provider
    if _default_provider is None:
        _default_provider = _make_default_chain()
    return _default_provider


# ============================================================
# Public API (unchanged interface)
# ============================================================

def get_carbon_intensity(
    region: str = "CA-QC",
    api_key: Optional[str] = None,
) -> Optional[float]:
    """
    Get carbon intensity for a region in gCO2/kWh.

    Args:
        region: Zone code (e.g., 'CA-QC', 'GB', 'DE', 'US-CA')
        api_key: Optional Electricity Maps API key (overrides env var)

    Returns:
        Carbon intensity in gCO2/kWh, or None if unavailable.
    """
    # If explicit API key provided, use dedicated ElectricityMaps call
    if api_key:
        em = ElectricityMapsProvider(api_key=api_key)
        result = em.get_intensity(region)
        if result is not None:
            return result

    # Otherwise use default chain
    return get_default_provider().get_intensity(region)


def wait_for_green_grid(
    max_intensity: float = 200.0,
    check_interval_sec: float = 60.0,
    region: str = "CA-QC",
) -> None:
    """
    Block until carbon intensity drops below threshold.

    Args:
        max_intensity: Maximum acceptable gCO2/kWh
        check_interval_sec: Seconds between checks
        region: Zone code to monitor
    """
    print(f"🌍 Green PEFT Scheduler: Checking grid in {region}...")

    while True:
        intensity = get_carbon_intensity(region=region)

        if intensity is None:
            print("⚠️ Could not fetch intensity. Proceeding with caution.")
            break

        print(f"📉 Current Intensity: {intensity:.0f} gCO2/kWh (Threshold: {max_intensity:.0f})")

        if intensity <= max_intensity:
            print(f"✅ Grid is Green ({region})! Starting training.")
            break
        else:
            print(f"🛑 Grid is Dirty. Waiting {check_interval_sec:.0f}s...")
            time.sleep(check_interval_sec)


# ============================================================
# CLI Test
# ============================================================

if __name__ == "__main__":
    print("Testing Carbon Scheduler Providers\n")

    # Show which providers are active
    provider = get_default_provider()
    print(f"Active chain: {provider.name}\n")

    # Test a few zones
    for zone in ["CA-QC", "GB", "DE", "US-CA"]:
        intensity = get_carbon_intensity(zone)
        print(f"  {zone}: {intensity} gCO2/kWh")

    print("\n✅ Done")