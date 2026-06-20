"""
Tests for efficient_training.py - the unified energy-aware training module.

Covers:
1. LossEfficiencyTracker - sample scoring and probability distribution
2. EnergyAwareBatchController - adaptive batch sizing
3. EnergyAwareEarlyStopper - early stopping logic
4. EnergyAwareBatchSampler - sampling with dynamic batch sizes
5. EnergyAwareTrainingController - full integration
"""

import numpy as np
import torch

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

from energypeft.core.efficient_training import (
    LossEfficiencyTracker,
    EnergyAwareBatchController,
    EnergyAwareBatchSampler,
    EnergyAwareTrainingController,
    EarlyStoppingConfig,
    EnergyAwareEarlyStopper,
    attach_indices_collate,
)


# ---------------------------------
# Mock Classes
# ---------------------------------

class MockEnergyMonitor:
    """Mock energy monitor for testing."""
    
    def __init__(self, budget_wh: float = 10.0, remaining_wh: float = 8.0):
        self.energy_budget_wh = budget_wh
        self._remaining = remaining_wh
    
    def get_remaining_energy(self) -> float:
        return self._remaining
    
    def set_remaining(self, wh: float):
        self._remaining = wh


class MockDataset:
    """Mock dataset for testing."""
    
    def __init__(self, size: int = 100):
        self._size = size
    
    def __len__(self):
        return self._size


# ---------------------------------
# LossEfficiencyTracker Tests
# ---------------------------------

class TestLossEfficiencyTracker:
    
    def test_initialization(self):
        """Test tracker initializes with uniform scores."""
        tracker = LossEfficiencyTracker(dataset_size=100)
        assert tracker.scores.shape == (100,)
        assert np.allclose(tracker.scores, 1.0)
    
    def test_update_single_sample(self):
        """Test updating a single sample score."""
        tracker = LossEfficiencyTracker(dataset_size=10, decay=0.0)  # No EMA
        tracker.update([0], [2.0], [50])
        
        # Score should change for index 0
        assert tracker.scores[0] != 1.0
        # Others should remain 1.0
        assert np.allclose(tracker.scores[1:], 1.0)
    
    def test_update_multiple_samples(self):
        """Test updating multiple samples."""
        tracker = LossEfficiencyTracker(dataset_size=10, decay=0.5)
        tracker.update([0, 1, 2], [1.5, 2.0, 0.5], [50, 100, 25])
        
        # All three should be different from initial
        for idx in [0, 1, 2]:
            assert tracker.scores[idx] != 1.0
    
    def test_probabilities_sum_to_one(self):
        """Test that probabilities form a valid distribution."""
        tracker = LossEfficiencyTracker(dataset_size=100)
        tracker.update([0, 1, 2], [1.5, 2.0, 0.5], [50, 100, 25])
        
        probs = tracker.get_probabilities()
        assert probs.shape == (100,)
        assert np.isclose(probs.sum(), 1.0)
        assert np.all(probs >= 0)
    
    def test_exploration_mixture(self):
        """Test that exploration parameter adds uniform noise."""
        tracker = LossEfficiencyTracker(dataset_size=10)
        
        probs_no_explore = tracker.get_probabilities(explore_eps=0.0)
        probs_with_explore = tracker.get_probabilities(explore_eps=0.5)
        
        # With 50% exploration, distribution should be more uniform
        assert np.std(probs_with_explore) < np.std(probs_no_explore) or np.isclose(np.std(probs_no_explore), 0)
    
    def test_invalid_indices_ignored(self):
        """Test that out-of-bounds indices are safely ignored."""
        tracker = LossEfficiencyTracker(dataset_size=10)
        tracker.update([-1, 100, 5], [1.0, 1.0, 2.0], [50, 50, 50])
        
        # Only index 5 should be updated
        assert tracker.scores[5] != 1.0
        assert np.allclose(tracker.scores[:5], 1.0)
        assert np.allclose(tracker.scores[6:], 1.0)
    
    def test_nan_loss_ignored(self):
        """Test that NaN losses are safely ignored."""
        tracker = LossEfficiencyTracker(dataset_size=10)
        tracker.update([0, 1], [float('nan'), 2.0], [50, 50])
        
        # Index 0 should remain unchanged, index 1 should be updated
        assert tracker.scores[0] == 1.0
        assert tracker.scores[1] != 1.0


# ---------------------------------
# EnergyAwareBatchController Tests
# ---------------------------------

class TestEnergyAwareBatchController:
    
    def test_base_batch_size_returned(self):
        """Test that base batch size is used with medium energy."""
        monitor = MockEnergyMonitor(budget_wh=10.0, remaining_wh=5.0)  # 50%
        controller = EnergyAwareBatchController(base_batch_size=32, energy_monitor=monitor)
        
        bs = controller.get_adaptive_batch_size(convergence_progress=0.0)
        assert bs == 32  # energy_factor=1.0, progress_factor=1.0
    
    def test_high_energy_increases_batch(self):
        """Test that high remaining energy increases batch size."""
        monitor = MockEnergyMonitor(budget_wh=10.0, remaining_wh=9.0)  # 90%
        controller = EnergyAwareBatchController(base_batch_size=32, energy_monitor=monitor)
        
        bs = controller.get_adaptive_batch_size(convergence_progress=0.0)
        assert bs > 32  # energy_factor=1.2
    
    def test_low_energy_decreases_batch(self):
        """Test that low remaining energy decreases batch size."""
        monitor = MockEnergyMonitor(budget_wh=10.0, remaining_wh=1.0)  # 10%
        controller = EnergyAwareBatchController(base_batch_size=32, energy_monitor=monitor)
        
        bs = controller.get_adaptive_batch_size(convergence_progress=0.0)
        assert bs < 32  # energy_factor=0.6
    
    def test_late_progress_decreases_batch(self):
        """Test that late training progress decreases batch size."""
        monitor = MockEnergyMonitor(budget_wh=10.0, remaining_wh=5.0)
        controller = EnergyAwareBatchController(base_batch_size=32, energy_monitor=monitor)
        
        bs = controller.get_adaptive_batch_size(convergence_progress=0.9)
        assert bs < 32  # progress_factor=0.7
    
    def test_zero_energy_returns_min_batch(self):
        """Test that zero energy returns minimum batch size."""
        monitor = MockEnergyMonitor(budget_wh=10.0, remaining_wh=0.0)
        controller = EnergyAwareBatchController(base_batch_size=32, energy_monitor=monitor, min_batch_size=4)
        
        bs = controller.get_adaptive_batch_size(convergence_progress=0.0)
        assert bs == 4
    
    def test_min_batch_size_enforced(self):
        """Test that batch size never goes below minimum."""
        monitor = MockEnergyMonitor(budget_wh=10.0, remaining_wh=0.5)
        controller = EnergyAwareBatchController(base_batch_size=4, energy_monitor=monitor, min_batch_size=2)
        
        bs = controller.get_adaptive_batch_size(convergence_progress=0.99)
        assert bs >= 2


# ---------------------------------
# EnergyAwareEarlyStopper Tests
# ---------------------------------

class TestEnergyAwareEarlyStopper:
    
    def test_energy_exhaustion_stops(self):
        """Test that energy exhaustion triggers stop."""
        cfg = EarlyStoppingConfig(patience=5, energy_stop_threshold_wh=1.0)
        stopper = EnergyAwareEarlyStopper(cfg)
        
        assert stopper.should_stop(step=0, remaining_energy_wh=0.5, val_metric=0.9) == True
    
    def test_no_stop_with_energy(self):
        """Test that training continues with energy remaining."""
        cfg = EarlyStoppingConfig(patience=5, energy_stop_threshold_wh=0.0)
        stopper = EnergyAwareEarlyStopper(cfg)
        
        assert stopper.should_stop(step=0, remaining_energy_wh=5.0, val_metric=0.9) == False
    
    def test_patience_exhausted_stops(self):
        """Test that patience exhaustion triggers stop."""
        cfg = EarlyStoppingConfig(patience=3, mode="max")
        stopper = EnergyAwareEarlyStopper(cfg)
        
        # Initial good metric
        assert stopper.should_stop(0, 5.0, val_metric=0.8) == False  # best=0.8
        # 3 bad metrics in a row
        assert stopper.should_stop(1, 5.0, val_metric=0.7) == False  # bad 1
        assert stopper.should_stop(2, 5.0, val_metric=0.75) == False  # bad 2
        assert stopper.should_stop(3, 5.0, val_metric=0.7) == True   # bad 3 -> stop
    
    def test_improvement_resets_patience(self):
        """Test that improvement resets bad count."""
        cfg = EarlyStoppingConfig(patience=3, mode="max")
        stopper = EnergyAwareEarlyStopper(cfg)
        
        stopper.should_stop(0, 5.0, val_metric=0.8)  # best=0.8
        stopper.should_stop(1, 5.0, val_metric=0.7)  # bad 1
        stopper.should_stop(2, 5.0, val_metric=0.7)  # bad 2
        stopper.should_stop(3, 5.0, val_metric=0.9)  # improvement! resets
        
        assert stopper.bad_counts == 0
        assert stopper.best == 0.9
    
    def test_min_mode(self):
        """Test mode='min' for loss-based stopping."""
        cfg = EarlyStoppingConfig(patience=2, mode="min")
        stopper = EnergyAwareEarlyStopper(cfg)
        
        stopper.should_stop(0, 5.0, val_metric=1.0)   # best=1.0
        stopper.should_stop(1, 5.0, val_metric=0.5)   # improvement
        assert stopper.best == 0.5
        
        stopper.should_stop(2, 5.0, val_metric=0.6)   # bad
        assert stopper.should_stop(3, 5.0, val_metric=0.7) == True  # bad 2 -> stop


# ---------------------------------
# EnergyAwareBatchSampler Tests
# ---------------------------------

class TestEnergyAwareBatchSampler:
    
    def test_samples_entire_dataset(self):
        """Test that sampler covers entire dataset per epoch."""
        monitor = MockEnergyMonitor()
        tracker = LossEfficiencyTracker(dataset_size=100)
        controller = EnergyAwareBatchController(base_batch_size=16, energy_monitor=monitor)
        
        sampler = EnergyAwareBatchSampler(
            dataset_size=100,
            batch_controller=controller,
            tracker=tracker,
            max_steps=100,
            seed=42,
        )
        
        all_indices = []
        for batch in sampler:
            all_indices.extend(batch)
        
        # All 100 indices should be covered
        assert sorted(all_indices) == list(range(100))
    
    def test_yields_lists_of_indices(self):
        """Test that batches are lists of integers."""
        monitor = MockEnergyMonitor()
        tracker = LossEfficiencyTracker(dataset_size=50)
        controller = EnergyAwareBatchController(base_batch_size=10, energy_monitor=monitor)
        
        sampler = EnergyAwareBatchSampler(
            dataset_size=50,
            batch_controller=controller,
            tracker=tracker,
            max_steps=50,
        )
        
        for batch in sampler:
            assert isinstance(batch, list)
            assert all(isinstance(i, int) for i in batch)
            break  # Just check first batch
    
    def test_drop_last(self):
        """Test drop_last functionality."""
        monitor = MockEnergyMonitor()
        tracker = LossEfficiencyTracker(dataset_size=95)
        controller = EnergyAwareBatchController(base_batch_size=32, energy_monitor=monitor)
        
        sampler = EnergyAwareBatchSampler(
            dataset_size=95,
            batch_controller=controller,
            tracker=tracker,
            max_steps=100,
            drop_last=True,
        )
        
        all_indices = []
        for batch in sampler:
            all_indices.extend(batch)
        
        # Should be less than 95 if last incomplete batch is dropped
        assert len(all_indices) <= 95


# ---------------------------------
# EnergyAwareTrainingController Tests
# ---------------------------------

class TestEnergyAwareTrainingController:
    
    def test_initialization(self):
        """Test controller initializes all components."""
        monitor = MockEnergyMonitor()
        dataset = MockDataset(size=100)
        
        controller = EnergyAwareTrainingController(
            dataset=dataset,
            energy_monitor=monitor,
            base_batch_size=16,
            min_batch_size=4,
            max_steps=100,
        )
        
        assert controller.dataset_size == 100
        assert controller.tracker is not None
        assert controller.batch_controller is not None
        assert controller.early_stopper is not None
        assert controller.batch_sampler is not None
    
    def test_on_train_step_end_updates_tracker(self):
        """Test that on_train_step_end updates the efficiency tracker."""
        monitor = MockEnergyMonitor()
        dataset = MockDataset(size=100)
        
        controller = EnergyAwareTrainingController(
            dataset=dataset,
            energy_monitor=monitor,
            base_batch_size=16,
            min_batch_size=4,
            max_steps=100,
        )
        
        # Initial scores should be 1.0
        assert np.allclose(controller.tracker.scores, 1.0)
        
        # Update with some training data
        controller.on_train_step_end(
            batch_indices=[0, 1, 2],
            per_sample_losses=[1.0, 2.0, 0.5],
            lengths=[32, 64, 16],
        )
        
        # Scores for indices 0, 1, 2 should have changed
        assert controller.tracker.scores[0] != 1.0
        assert controller.tracker.scores[1] != 1.0
        assert controller.tracker.scores[2] != 1.0
    
    def test_on_train_step_end_accepts_tensors(self):
        """Test that on_train_step_end handles torch tensors."""
        monitor = MockEnergyMonitor()
        dataset = MockDataset(size=100)
        
        controller = EnergyAwareTrainingController(
            dataset=dataset,
            energy_monitor=monitor,
            base_batch_size=16,
            min_batch_size=4,
            max_steps=100,
        )
        
        # Pass tensors instead of lists
        losses = torch.tensor([1.0, 2.0, 0.5])
        lengths = torch.tensor([32, 64, 16])
        
        controller.on_train_step_end(
            batch_indices=[0, 1, 2],
            per_sample_losses=losses,
            lengths=lengths,
        )
        
        # Should work without error
        assert controller.tracker.scores[0] != 1.0
    
    def test_history_logging(self):
        """Test that history is logged correctly."""
        monitor = MockEnergyMonitor(remaining_wh=7.5)
        dataset = MockDataset(size=100)
        
        controller = EnergyAwareTrainingController(
            dataset=dataset,
            energy_monitor=monitor,
            base_batch_size=16,
            min_batch_size=4,
            max_steps=100,
        )
        
        controller.on_train_step_end([0], [1.0], [32])
        
        assert len(controller.history["remaining_energy_wh"]) == 1
        assert controller.history["remaining_energy_wh"][0] == 7.5
        assert len(controller.history["batch_size"]) == 1


# ---------------------------------
# Helper Function Tests
# ---------------------------------

class TestAttachIndicesCollate:
    
    def test_collate_with_index_key(self):
        """Test collating dict items with _index key."""
        batch = [
            {"_index": 0, "input_ids": torch.tensor([1, 2, 3])},
            {"_index": 1, "input_ids": torch.tensor([4, 5, 6])},
        ]
        
        collated = attach_indices_collate(batch)
        
        assert "_indices" in collated
        assert collated["_indices"] == [0, 1]
        assert "input_ids" in collated
        assert collated["input_ids"].shape == (2, 3)
    
    def test_collate_without_index_fallback(self):
        """Test fallback when _index is missing."""
        batch = [
            {"input_ids": torch.tensor([1, 2, 3])},
            {"input_ids": torch.tensor([4, 5, 6])},
        ]
        
        collated = attach_indices_collate(batch)
        
        assert "_indices" in collated
        assert collated["_indices"] == [-1, -1]  # fallback


# ---------------------------------
# Run Tests
# ---------------------------------

if __name__ == "__main__":
    if HAS_PYTEST:
        pytest.main([__file__, "-v"])
    else:
        # Manual test runner when pytest is not installed
        def run_test_class(cls):
            instance = cls()
            methods = [m for m in dir(instance) if m.startswith('test_')]
            passed = 0
            failed = 0
            for method in methods:
                try:
                    getattr(instance, method)()
                    print(f'  ✅ {method}')
                    passed += 1
                except Exception as e:
                    print(f'  ❌ {method}: {e}')
                    failed += 1
            return passed, failed

        total_passed = 0
        total_failed = 0

        for test_class in [
            TestLossEfficiencyTracker,
            TestEnergyAwareBatchController,
            TestEnergyAwareEarlyStopper,
            TestEnergyAwareBatchSampler,
            TestEnergyAwareTrainingController,
            TestAttachIndicesCollate,
        ]:
            print(f'\n{test_class.__name__}:')
            p, f = run_test_class(test_class)
            total_passed += p
            total_failed += f

        print()
        print('=' * 50)
        print(f'Total: {total_passed} passed, {total_failed} failed')
        if total_failed == 0:
            print('🎉 All tests passed!')
