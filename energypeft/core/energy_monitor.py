"""
Improved EnergyMonitor

- Uses monotonic clock (time.perf_counter) for stable deltas
- Thread-safe updates via Lock
- Avoids double-integration (only background thread integrates Wh)
- Tracks GPU and CPU energy separately + total
- Stores structured time series (timestamp, gpu_w, cpu_w, total_w, total_wh)
- Optional CPU power backends:
    * "tdp"  : cpu_percent * cpu_tdp_w (configurable)
    * "rapl" : Linux Intel RAPL if available (best-effort; no extra deps)
- Adds step-aligned snapshots: log_step(step_id) -> stores {step_id: total_wh}
- Cleaner NVML init + optional shutdown
"""

from __future__ import annotations

import json # saving logs
import os
import platform  # Detect macOS
import subprocess  # For powermetrics on Mac
import threading # Run energy monitoring in a background thread
import time
from collections import deque # Keep a fixed-size history of recent power samples
from dataclasses import dataclass, asdict
from typing import Deque, Dict, List, Optional, Tuple

import psutil #Read CPU utilization and system stats.
# NVML (NVIDIA Management Library): read GPU telemetry such as power draw (Watts)
# Note: we filter the pynvml deprecation warning as nvidia-ml-py might still trigger it
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pynvml")

try:
    import pynvml
    _NVML_AVAILABLE = True
except ImportError:
    pynvml = None  # type: ignore
    _NVML_AVAILABLE = False
except Exception:
    pynvml = None
    _NVML_AVAILABLE = False

# Watt (W) = how fast energy is being used right now
#Watt-hour (Wh) = how much energy has been used in total
@dataclass
class EnergyMetrics:
    total_energy_wh: float = 0.0
    gpu_energy_wh: float = 0.0
    cpu_energy_wh: float = 0.0
    current_power_w: float = 0.0
    gpu_power_w: float = 0.0
    cpu_power_w: float = 0.0
    budget_used_percent: float = 0.0
    timestamp_wall: float = 0.0  # time.time()
    timestamp_mono: float = 0.0  # time.perf_counter()


class EnergyMonitor:
    """Real-time energy monitoring for training loops."""

    def __init__(
        self,
        energy_budget_wh: float = 100.0,
        sampling_interval: float = 1.0,
        power_history_len: int = 60,
        cpu_backend: str = "auto",  # "auto", "tdp", "rapl", or "powermetrics" (Mac)
        cpu_tdp_w: float = 100.0,
        nvml_shutdown_on_stop: bool = False,
    ):
        if energy_budget_wh <= 0:
            raise ValueError("energy_budget_wh must be > 0")
        if sampling_interval <= 0:
            raise ValueError("sampling_interval must be > 0")
        
        # Auto-detect best CPU backend
        if cpu_backend == "auto":
            if platform.system() == "Darwin":  # macOS
                cpu_backend = "powermetrics"
            elif os.path.exists("/sys/class/powercap/intel-rapl:0"):
                cpu_backend = "rapl"
            else:
                cpu_backend = "tdp"
        
        if cpu_backend not in ("tdp", "rapl", "powermetrics"):
            raise ValueError("cpu_backend must be 'auto', 'tdp', 'rapl', or 'powermetrics'")

        self.energy_budget_wh = float(energy_budget_wh)
        self.sampling_interval = float(sampling_interval)
        self.cpu_backend = cpu_backend
        self.cpu_tdp_w = float(cpu_tdp_w)
        self.nvml_shutdown_on_stop = nvml_shutdown_on_stop

        # Threading
        self._lock = threading.Lock()
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None

        # Timekeeping
        self._start_mono = time.perf_counter()
        self._start_wall = time.time()
        self._last_update_mono = self._start_mono

        # Energy accumulators
        self._total_energy_wh = 0.0
        self._gpu_energy_wh = 0.0
        self._cpu_energy_wh = 0.0

        # Latest power snapshot
        self._last_gpu_w = 0.0
        self._last_cpu_w = 0.0
        self._last_total_w = 0.0

        # History
        self.power_history: Deque[float] = deque(maxlen=power_history_len)
        self.series: Deque[Tuple[float, float, float, float, float]] = deque(
            maxlen=max(300, power_history_len * 10)
        )
        # series entries: (mono_ts, gpu_w, cpu_w, total_w, total_wh)

        # Step-aligned snapshots
        self.step_energy_wh: Dict[int, float] = {}

        # NVML init
        self.gpu_count = 0
        self.gpu_handles: List[object] = []
        self._nvml_ok = False
        if _NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                self.gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.gpu_count)]
                self._nvml_ok = True
            except Exception:
                self.gpu_count = 0
                self.gpu_handles = []
                self._nvml_ok = False

        # RAPL init (best-effort)
        self._rapl_paths = self._discover_rapl_paths() if cpu_backend == "rapl" else []
        self._rapl_prev_uj: Optional[int] = None
        self._rapl_prev_mono: Optional[float] = None
        
        # Mac powermetrics calibration cache
        # Powermetrics requires sudo and is slow, so we cache the result
        self._powermetrics_cached_cpu_w: Optional[float] = None
        self._powermetrics_cached_gpu_w: Optional[float] = None
        self._powermetrics_cached_ane_w: Optional[float] = None
        self._powermetrics_cache_time: float = 0.0
        self._powermetrics_cache_ttl: float = 30.0  # Re-calibrate every 30 seconds

    # ----------------------------
    # Public API
    # ----------------------------

    def start_monitoring(self) -> None:
        """Start background monitoring (integration happens only in background thread)."""
        with self._lock:
            if self._monitoring:
                return
            self._monitoring = True
            # Reset timing baseline so energy doesn't include time spent before start
            now = time.perf_counter()
            self._last_update_mono = now
            self._start_mono = now
            self._start_wall = time.time()

            # Initialize RAPL baseline if needed
            if self.cpu_backend == "rapl":
                self._rapl_prev_uj = self._read_rapl_uj_total()
                self._rapl_prev_mono = now

        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self) -> EnergyMetrics:
        """Stop monitoring and return final metrics."""
        with self._lock:
            self._monitoring = False

        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)

        if self.nvml_shutdown_on_stop and self._nvml_ok:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

        return self.get_current_metrics()

    def get_current_metrics(self) -> EnergyMetrics:
        """
        Return snapshot WITHOUT integrating additional energy.
        Integration happens only in the background thread to avoid double counting.
        """
        with self._lock:
            total_wh = self._total_energy_wh
            gpu_wh = self._gpu_energy_wh
            cpu_wh = self._cpu_energy_wh
            total_w = self._last_total_w
            gpu_w = self._last_gpu_w
            cpu_w = self._last_cpu_w

        return EnergyMetrics(
            total_energy_wh=total_wh,
            gpu_energy_wh=gpu_wh,
            cpu_energy_wh=cpu_wh,
            current_power_w=total_w,
            gpu_power_w=gpu_w,
            cpu_power_w=cpu_w,
            budget_used_percent=min(max((total_wh / self.energy_budget_wh) * 100.0, 0.0), 100.0) if self.energy_budget_wh > 0 else 0.0,
            timestamp_wall=time.time(),
            timestamp_mono=time.perf_counter(),
        )

    def has_energy_remaining(self, threshold_percent: float = 95.0) -> bool:
        """Check if budget allows continued training."""
        m = self.get_current_metrics()
        return m.budget_used_percent < threshold_percent

    def has_energy(self, threshold_percent: float = 95.0) -> bool:
        """Alias for API consistency."""
        return self.has_energy_remaining(threshold_percent)

    def get_remaining_energy(self) -> float:
        """Remaining energy in Wh (floor at 0)."""
        with self._lock:
            return max(0.0, self.energy_budget_wh - self._total_energy_wh)

    @property
    def total_energy_wh(self) -> float:
        with self._lock:
            return self._total_energy_wh

    @property
    def total_energy_consumed(self) -> float:
        """Alias for API consistency."""
        return self.total_energy_wh

    @total_energy_consumed.setter
    def total_energy_consumed(self, value: float) -> None:
        """Allow setting in tests."""
        with self._lock:
            self._total_energy_wh = float(value)

    @property
    def is_monitoring(self) -> bool:
        with self._lock:
            return self._monitoring

    def log_step(self, step_id: Optional[int] = None) -> None:
        """
        Snapshot the current cumulative energy for alignment with training logs.
        Provide step_id (global_step) for reliable mapping.
        """
        m = self.get_current_metrics()
        if step_id is None:
            # If caller doesn't provide step_id, do nothing except keep API compatibility
            return
        with self._lock:
            self.step_energy_wh[int(step_id)] = float(m.total_energy_wh)

    def save_energy_log(self, filepath: str) -> None:
        """Save a structured JSON log with summary + time series + step map."""
        m = self.get_current_metrics()
        with self._lock:
            series_list = list(self.series)
            power_hist = list(self.power_history)
            step_map = dict(self.step_energy_wh)

        log_data = {
            "energy_budget_wh": self.energy_budget_wh,
            "total_energy_wh": m.total_energy_wh,
            "gpu_energy_wh": m.gpu_energy_wh,
            "cpu_energy_wh": m.cpu_energy_wh,
            "budget_used_percent": m.budget_used_percent,
            "duration_sec": max(0.0, m.timestamp_mono - self._start_mono),
            "avg_power_w": (sum(power_hist) / len(power_hist)) if power_hist else 0.0,
            "last_power_w": m.current_power_w,
            "backend": {
                "gpu": "nvml" if self._nvml_ok else "unavailable",
                "cpu": self.cpu_backend,
                "cpu_tdp_w": self.cpu_tdp_w if self.cpu_backend == "tdp" else None,
                "rapl_paths": self._rapl_paths if self.cpu_backend == "rapl" else None,
            },
            # Time series: (mono_ts, gpu_w, cpu_w, total_w, total_wh)
            "series": series_list,
            "step_energy_wh": step_map,
        }

        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(log_data, f, indent=2)

    # ----------------------------
    # Background loop + integration
    # ----------------------------

    def _monitor_loop(self) -> None:
        while True:
            with self._lock:
                if not self._monitoring:
                    break
            self._sample_and_integrate_once()
            time.sleep(self.sampling_interval)

    def _sample_and_integrate_once(self) -> None:
        """Read power, integrate energy (Wh) based on elapsed monotonic time, update histories."""
        now_mono = time.perf_counter()

        # Read power outside lock where possible (to reduce contention)
        gpu_w = self._get_gpu_power()
        cpu_w, cpu_wh_increment = self._get_cpu_power_and_optional_energy(now_mono)

        with self._lock:
            dt = now_mono - self._last_update_mono
            if dt < 0:
                dt = 0.0

            total_w = max(0.0, gpu_w + cpu_w)

            # Integrate GPU energy via power*dt
            gpu_wh_inc = max(0.0, gpu_w) * (dt / 3600.0)

            # CPU energy:
            # - if backend="rapl": use measured increment if available (cpu_wh_increment)
            # - else integrate cpu_w like gpu_w
            if self.cpu_backend == "rapl" and cpu_wh_increment is not None:
                cpu_wh_inc = max(0.0, cpu_wh_increment)
            else:
                cpu_wh_inc = max(0.0, cpu_w) * (dt / 3600.0)

            self._gpu_energy_wh += gpu_wh_inc
            self._cpu_energy_wh += cpu_wh_inc
            self._total_energy_wh += (gpu_wh_inc + cpu_wh_inc)

            self._last_update_mono = now_mono
            self._last_gpu_w = gpu_w
            self._last_cpu_w = cpu_w
            self._last_total_w = total_w

            self.power_history.append(total_w)
            self.series.append((now_mono, gpu_w, cpu_w, total_w, self._total_energy_wh))

    # ----------------------------
    # Power backends
    # ----------------------------

    def _get_gpu_power(self) -> float:
        """Total GPU power in watts (NVML for NVIDIA, powermetrics for Apple Silicon)."""
        if platform.system() == "Darwin" and self.cpu_backend == "powermetrics":
            # On Mac, ensure cache is updated, then return cached GPU power.
            # _get_cpu_power_and_optional_energy is called right after this in the loop,
            # which will refresh the cache. For now, just trigger a refresh if stale.
            self._refresh_powermetrics()
            return self._powermetrics_cached_gpu_w or 0.0

        if not self._nvml_ok:
            return 0.0
        total = 0.0
        for h in self.gpu_handles:
            try:
                mw = pynvml.nvmlDeviceGetPowerUsage(h)
                total += mw / 1000.0
            except Exception:
                continue
        return max(0.0, total)

    def _get_cpu_power_and_optional_energy(self, now_mono: float) -> Tuple[float, Optional[float]]:
        """
        Returns:
          (cpu_power_w, cpu_wh_increment_if_measured_else_None)

        - backend="tdp": cpu_power_w from cpu_percent * cpu_tdp_w, energy increment None (integrate later)
        - backend="rapl": best-effort read of RAPL energy counter (uj); returns measured cpu Wh increment
        - backend="powermetrics": Mac-specific, reads from powermetrics (cached)
        """
        if self.cpu_backend == "rapl":
            wh_inc = self._read_rapl_wh_increment(now_mono)
            # If RAPL gives measured energy, we can still return a rough power for logging:
            # power ~= dE/dt (if dt > 0)
            if wh_inc is not None and self._rapl_prev_mono is not None:
                dt = now_mono - self._rapl_prev_mono
                cpu_w = (wh_inc * 3600.0 / dt) if dt > 0 else 0.0
                return max(0.0, cpu_w), wh_inc
            # fallback if rapl unavailable
            return self._cpu_power_tdp(), None
        
        # Mac powermetrics backend
        if self.cpu_backend == "powermetrics":
            self._refresh_powermetrics()
            cpu_w = self._powermetrics_cached_cpu_w or 0.0
            # To capture full SoC power (CPU + GPU + ANE), the GPU is handled by _get_gpu_power.
            # We can optionally add ANE power to CPU power here so it isn't "lost".
            ane_w = self._powermetrics_cached_ane_w or 0.0
            return cpu_w + ane_w, None

        # Default "tdp" backend
        return self._cpu_power_tdp(), None

    def _cpu_power_tdp(self) -> float:
        """Heuristic CPU power: utilization * TDP."""
        # psutil.cpu_percent() can be blocking on first call; it's okay in a monitor thread.
        cpu_pct = psutil.cpu_percent(interval=None)
        return max(0.0, (cpu_pct / 100.0) * self.cpu_tdp_w)
    
    def _refresh_powermetrics(self) -> None:
        """
        Mac-specific: Read CPU, GPU, and ANE power from powermetrics.
        Updates internal cache.
        """
        now = time.time()
        if (self._powermetrics_cached_cpu_w is not None and 
            now - self._powermetrics_cache_time < self._powermetrics_cache_ttl):
            return

        try:
            # -n 1: one sample, -i 100: 100ms interval
            # Note: ANE doesn't have a standalone sampler flag in all macOS versions,
            # but usually appears under cpu_power on M-series chips depending on the OS version.
            result = subprocess.run(
                ["sudo", "-n", "powermetrics", "-n", "1", "-i", "100", "--samplers", "cpu_power,gpu_power"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                cpu_w, gpu_w, ane_w = 0.0, 0.0, 0.0
                found_any = False
                for line in result.stdout.split("\n"):
                    lower_line = line.lower()
                    if "cpu power:" in lower_line:
                        parts = line.split(":")
                        if len(parts) >= 2:
                            val = parts[1].strip().lower().replace("w", "").replace("m", "")
                            try:
                                # if the output says "mW", divide by 1000
                                p = float(val)
                                if "mw" in lower_line: p /= 1000.0
                                cpu_w = p
                                found_any = True
                            except ValueError: pass
                    elif "gpu power:" in lower_line:
                        parts = line.split(":")
                        if len(parts) >= 2:
                            val = parts[1].strip().lower().replace("w", "").replace("m", "")
                            try:
                                p = float(val)
                                if "mw" in lower_line: p /= 1000.0
                                gpu_w = p
                                found_any = True
                            except ValueError: pass
                    elif "ane power:" in lower_line:
                        parts = line.split(":")
                        if len(parts) >= 2:
                            val = parts[1].strip().lower().replace("w", "").replace("m", "")
                            try:
                                p = float(val)
                                if "mw" in lower_line: p /= 1000.0
                                ane_w = p
                                found_any = True
                            except ValueError: pass
                
                if found_any:
                    self._powermetrics_cached_cpu_w = cpu_w
                    self._powermetrics_cached_gpu_w = gpu_w
                    self._powermetrics_cached_ane_w = ane_w
                    self._powermetrics_cache_time = now
                    return
        except Exception:
            pass

        # If powermetrics fails, we fall back to TDP heuristic for CPU
        if self._powermetrics_cached_cpu_w is None:
            self._powermetrics_cached_cpu_w = self._cpu_power_tdp()
            self._powermetrics_cached_gpu_w = 0.0
            self._powermetrics_cached_ane_w = 0.0
            self._powermetrics_cache_time = now

    # ----------------------------
    # RAPL (Linux Intel) best-effort
    # ----------------------------

    def _discover_rapl_paths(self) -> List[str]:
        """
        Find Intel RAPL energy_uj files.
        Typical path:
          /sys/class/powercap/intel-rapl:0/energy_uj
        """
        base = "/sys/class/powercap"
        paths: List[str] = []
        if not os.path.isdir(base):
            return paths

        for root, dirs, files in os.walk(base):
            if "energy_uj" in files and "intel-rapl" in root:
                paths.append(os.path.join(root, "energy_uj"))
        return sorted(paths)

    def _read_rapl_uj_total(self) -> Optional[int]:
        """Sum energy_uj across all rapl domains (microjoules)."""
        if not self._rapl_paths:
            return None
        total_uj = 0
        ok = False
        for p in self._rapl_paths:
            try:
                with open(p, "r") as f:
                    total_uj += int(f.read().strip())
                    ok = True
            except Exception:
                continue
        return total_uj if ok else None

    def _read_rapl_wh_increment(self, now_mono: float) -> Optional[float]:
        """
        Compute incremental CPU energy (Wh) from RAPL counters.
        Handles counter reset/overflow by returning None (fallback to heuristic integration).
        """
        current_uj = self._read_rapl_uj_total()
        if current_uj is None:
            return None

        if self._rapl_prev_uj is None or self._rapl_prev_mono is None:
            self._rapl_prev_uj = current_uj
            self._rapl_prev_mono = now_mono
            return None

        delta_uj = current_uj - self._rapl_prev_uj
        if delta_uj < 0:
            # counter reset/overflow; reset baseline
            self._rapl_prev_uj = current_uj
            self._rapl_prev_mono = now_mono
            return None

        self._rapl_prev_uj = current_uj
        self._rapl_prev_mono = now_mono

        # microjoules -> joules: 1e-6; joules -> Wh: /3600
        wh = (delta_uj * 1e-6) / 3600.0
        return max(0.0, wh)