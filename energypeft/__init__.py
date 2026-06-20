# energypeft/__init__.py

"""
Green PEFT: energy-aware, carbon-aware parameter-efficient fine-tuning.

Public API exports:
- Core (always available): EnergyMonitor, Carbon Scheduler, Efficient Training Controller
- Trainer (requires transformers): GreenTrainer
- Backwards-compatibility aliases

Note: Core components work without HuggingFace transformers installed.
      GreenTrainer requires transformers and will be None if not installed.
"""

# Core imports (no HF dependency)
from .core.energy_monitor import EnergyMonitor, EnergyMetrics
from .core.carbon_scheduler import get_carbon_intensity, wait_for_green_grid
from .core.efficient_training import (
    LossEfficiencyTracker,
    EnergyAwareBatchController,
    EnergyAwareBatchSampler,
    EnergyAwareTrainingController,
    EarlyStoppingConfig,
    EnergyAwareEarlyStopper,
    attach_indices_collate,
)

__version__ = "0.1.0"

# Backwards compatibility aliases
EnergyAwareBatcher = EnergyAwareBatchController
EnergyAwareSampler = EnergyAwareBatchSampler

# Optional HF Trainer import (lazy/optional)
try:
    from .trainers.green_trainer import GreenTrainer
except ImportError:
    GreenTrainer = None  # type: ignore


class EnergyPEFT:
    """
    Convenience wrapper for quickly instantiating core components and a GreenTrainer.
    
    Note: wrap_trainer() requires transformers to be installed.
    """

    def __init__(self, energy_budget_wh: float = 100.0, base_batch_size: int = 32):
        self.energy_budget_wh = float(energy_budget_wh)
        self.base_batch_size = int(base_batch_size)
        self.energy_monitor = EnergyMonitor(self.energy_budget_wh)
        # Alias for backwards compatibility
        self.monitor = self.energy_monitor

    def wrap_trainer(self, trainer_type: str = "huggingface", **kwargs):
        if trainer_type == "huggingface":
            if GreenTrainer is None:
                raise ImportError(
                    "GreenTrainer requires 'transformers' package. "
                    "Install with: pip install transformers"
                )
            return GreenTrainer(
                energy_budget_wh=self.energy_budget_wh,
                base_batch_size=self.base_batch_size,
                **kwargs,
            )
        raise ValueError(f"Unsupported trainer type: {trainer_type}")


__all__ = [
    # Convenience
    "EnergyPEFT",
    "__version__",
    # Core (always available)
    "EnergyMonitor",
    "EnergyMetrics",
    "get_carbon_intensity",
    "wait_for_green_grid",
    "LossEfficiencyTracker",
    "EnergyAwareBatchController",
    "EnergyAwareBatchSampler",
    "EnergyAwareTrainingController",
    "EarlyStoppingConfig",
    "EnergyAwareEarlyStopper",
    "attach_indices_collate",
    # Trainer (optional, requires transformers)
    "GreenTrainer",
    # Backwards compatibility
    "EnergyAwareBatcher",
    "EnergyAwareSampler",
]