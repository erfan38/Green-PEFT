# energypeft/core/carbon_monitor.py
"""
CarbonIntensityMonitor — real-time carbon intensity monitor for carbon-aware training.

Wraps carbon_scheduler.py provider chain with:
  - Background polling thread (refresh every poll_interval_s)
  - Aggressiveness factor: scales T1/T2 skip thresholds based on current carbon intensity
      High carbon → factor > 1.0 → more aggressive skipping → less energy consumed
      Low carbon  → factor < 1.0 → less aggressive skipping → better model quality
  - Best training window recommendation (requires Electricity Maps forecast API key)

The aggressiveness factor is a linear interpolation between:
  - low_carbon_g_per_kwh  → min_aggressiveness (e.g. 0.7 — relax skipping, cleaner grid)
  - high_carbon_g_per_kwh → max_aggressiveness (e.g. 1.3 — tighten skipping, dirty grid)

Usage:
    monitor = CarbonIntensityMonitor(zone="CA-QC", api_key="...", fallback_intensity_g_per_kwh=50.0)
    monitor.start_background_polling()

    # inside training loop:
    factor = monitor.get_aggressiveness_factor()  # ∈ [0.7, 1.3]

    monitor.stop_background_polling()
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from .carbon_scheduler import get_carbon_intensity

logger = logging.getLogger(__name__)


class CarbonIntensityMonitor:
    """
    Polls carbon intensity in the background and exposes an aggressiveness
    factor to scale GreenTrainer2's T1/T2 skip thresholds dynamically.

    Aggressiveness factor ∈ [min_aggressiveness=1.0, max_aggressiveness=1.3]:
        Clean grid (≤ low_carbon_g_per_kwh)  → factor = 1.0 → base T1/T2 behavior unchanged
        Dirty grid (≥ high_carbon_g_per_kwh) → factor = 1.3 → 30% more aggressive skipping

    Linear interpolation:
        t = clamp((intensity - low_carbon) / (high_carbon - low_carbon), 0, 1)
        factor = min_aggressiveness + t * (max_aggressiveness - min_aggressiveness)
    """

    def __init__(
        self,
        zone: str = "CA-QC",
        api_key: Optional[str] = None,
        fallback_intensity_g_per_kwh: float = 50.0,
        poll_interval_s: int = 300,             # re-query every 5 minutes
        low_carbon_g_per_kwh: float = 100.0,    # "clean" grid: neutral skipping
        high_carbon_g_per_kwh: float = 400.0,   # "dirty" grid: max skipping
        min_aggressiveness: float = 1.0,         # clean grid = neutral (no suppression)
        max_aggressiveness: float = 1.3,         # dirty grid = 30% more aggressive
    ):
        self.zone = zone
        self.api_key = api_key
        self.fallback_intensity = float(fallback_intensity_g_per_kwh)
        self.poll_interval_s = max(60, int(poll_interval_s))
        self.low_carbon = float(low_carbon_g_per_kwh)
        self.high_carbon = float(high_carbon_g_per_kwh)
        self.min_aggressiveness = float(min_aggressiveness)
        self.max_aggressiveness = float(max_aggressiveness)

        self._current_intensity: float = fallback_intensity_g_per_kwh
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._api_live: bool = False

        # Attempt an initial fetch on construction
        fetched = get_carbon_intensity(region=zone, api_key=api_key)
        if fetched is not None:
            with self._lock:
                self._current_intensity = float(fetched)
            self._api_live = True
            print(
                f"🌍 CarbonMonitor: live | zone={zone} | "
                f"{fetched:.1f} g CO\u2082/kWh | "
                f"aggressiveness={self.get_aggressiveness_factor():.2f}x"
            )
        else:
            print(
                f"🌍 CarbonMonitor: API unavailable — using static fallback "
                f"{fallback_intensity_g_per_kwh:.1f} g CO\u2082/kWh | "
                f"aggressiveness={self.get_aggressiveness_factor():.2f}x"
            )

    # ------------------------------------------------------------------
    # Background polling
    # ------------------------------------------------------------------

    def start_background_polling(self) -> None:
        """Start a daemon thread that re-fetches carbon intensity periodically."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="CarbonMonitorPoller",
        )
        self._thread.start()
        logger.info(f"CarbonMonitor polling started (interval={self.poll_interval_s}s).")

    def stop_background_polling(self) -> None:
        """Stop the background polling thread gracefully."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None

    def _poll_loop(self) -> None:
        while not self._stop_event.wait(timeout=self.poll_interval_s):
            try:
                fetched = get_carbon_intensity(region=self.zone, api_key=self.api_key)
                if fetched is not None:
                    with self._lock:
                        self._current_intensity = float(fetched)
                    logger.info(
                        f"CarbonMonitor updated: {fetched:.1f} g CO\u2082/kWh "
                        f"(factor={self.get_aggressiveness_factor():.2f}x)"
                    )
            except Exception as exc:
                logger.debug(f"CarbonMonitor poll failed: {exc}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_current_intensity(self) -> float:
        """Returns current carbon intensity in g CO₂/kWh (thread-safe)."""
        with self._lock:
            return self._current_intensity

    def get_aggressiveness_factor(self) -> float:
        """
        Returns a multiplier ∈ [min_aggressiveness, max_aggressiveness].

        Used by GreenTrainer2 to scale T1 skip_fraction and T2 convergence_threshold:
            effective_T1_fraction      = base_fraction  * factor
            effective_T2_threshold_ema = ema_threshold  * factor

        High carbon (dirty grid): factor > 1.0 → thresholds increase → more skipping.
        Low carbon  (clean grid): factor < 1.0 → thresholds decrease → fewer skips.
        """
        intensity = self.get_current_intensity()
        t = (intensity - self.low_carbon) / max(self.high_carbon - self.low_carbon, 1.0)
        t = max(0.0, min(1.0, t))
        return self.min_aggressiveness + t * (self.max_aggressiveness - self.min_aggressiveness)

    def summary(self) -> dict:
        """Returns a serializable summary for inclusion in run reports."""
        return {
            "zone": self.zone,
            "api_live": self._api_live,
            "current_intensity_g_per_kwh": round(self.get_current_intensity(), 2),
            "aggressiveness_factor": round(self.get_aggressiveness_factor(), 3),
            "low_carbon_threshold_g_per_kwh": self.low_carbon,
            "high_carbon_threshold_g_per_kwh": self.high_carbon,
        }
