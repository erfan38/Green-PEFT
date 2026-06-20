# energypeft/core/__init__.py
"""
Core components for energy-aware training.
"""

from .energy_monitor import EnergyMonitor, EnergyMetrics
from .carbon_monitor import CarbonIntensityMonitor
from .efficient_training import (
    LossEfficiencyTracker,
    EnergyAwareBatchController,
    EnergyAwareBatchSampler,
    EnergyAwareTrainingController,
    EarlyStoppingConfig,
    EnergyAwareEarlyStopper,
    attach_indices_collate,
)

# Backwards compatibility aliases
EnergyAwareBatcher = EnergyAwareBatchController
EnergyAwareSampler = EnergyAwareBatchSampler

__all__ = [
    "EnergyMonitor",
    "EnergyMetrics",
    "CarbonIntensityMonitor",
    "EnergyAwareBatchController",
    "EnergyAwareBatchSampler", 
    "EnergyAwareTrainingController",
    "EarlyStoppingConfig",
    "EnergyAwareEarlyStopper",
    "LossEfficiencyTracker",
    "attach_indices_collate",
    # Backwards compatibility
    "EnergyAwareBatcher",
    "EnergyAwareSampler",
]