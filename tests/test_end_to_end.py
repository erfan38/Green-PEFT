# test_end_to_end.py
"""
End-to-end workflow tests for EnergyPEFT
"""

import unittest
import tempfile
import json
import os
from unittest.mock import Mock, patch, MagicMock
import sys

# Setup path for importing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import energypeft
from energypeft.core.energy_monitor import EnergyMonitor
from energypeft.core.smart_sampler import EnergyAwareSampler  
from energypeft.core.adaptive_batcher import EnergyAwareBatcher
from energypeft.integrations.llamafactory import LlamaFactoryEnergyWrapper


class TestEndToEndWorkflows(unittest.TestCase):
    """Test complete end-to-end workflows"""
    
    def setUp(self):
        """Set up end-to-end test fixtures"""
        self.energy_budget = 50.0
        self.base_batch_size = 16
        self.dataset_size = 100
    
    def test_complete_energypeft_workflow(self):
        """Test complete EnergyPEFT framework workflow"""
        # Initialize EnergyPEFT framework
        energy_peft = energypeft.EnergyPEFT(
            energy_budget_wh=self.energy_budget,
            base_batch_size=self.base_batch_size,
            importance_weight=0.7
        )
        
        # Verify all components are initialized
        self.assertIsNotNone(energy_peft.energy_monitor)
        self.assertEqual(energy_peft.energy_monitor.energy_budget_wh, self.energy_budget)
        self.assertEqual(energy_peft.base_batch_size, self.base_batch_size)
        self.assertEqual(energy_peft.importance_weight, 0.7)
    
    @patch('energypeft.integrations.llamafactory.subprocess.run')
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    def test_llamafactory_end_to_end(self, mock_open, mock_subprocess):
        """Test complete LlamaFactory integration workflow"""
        # Setup successful subprocess
        mock_subprocess.return_value = Mock(returncode=0)
        
        # Initialize framework
        energy_peft = energypeft.EnergyPEFT(energy_budget_wh=30.0)
        
        # Create LlamaFactory wrapper
        wrapper = LlamaFactoryEnergyWrapper(
            energy_framework=energy_peft,
            llamafactory_config={'custom_param': 'test'}
        )
        
        # Create energy config
        config = wrapper.create_energy_config(
            model_name="test-model",
            dataset="test-dataset"
        )
        
        # Verify config contains energy-aware settings
        self.assertEqual(config['per_device_train_batch_size'], 2)
        self.assertEqual(config['quantization_bit'], 4)
        self.assertEqual(config['custom_param'], 'test')
        
        # Mock energy monitoring
        with patch.object(energy_peft.energy_monitor, 'start_monitoring'), \
             patch.object(energy_peft.energy_monitor, 'stop_monitoring') as mock_stop, \
             patch.object(energy_peft.energy_monitor, 'save_energy_log'):
            
            mock_stop.return_value = Mock(
                total_energy_wh=15.0,
                budget_used_percent=50.0
            )
            
            # Run training
            result = wrapper.train_with_energy_monitoring(config=config)
            
            # Verify results
            self.assertIn('energy_consumed_wh', result)
            self.assertIn('budget_utilization', result)
            self.assertEqual(result['energy_consumed_wh'], 15.0)
            self.assertEqual(result['budget_utilization'], 50.0)
    
    def test_energy_monitoring_lifecycle(self):
        """Test complete energy monitoring lifecycle"""
        monitor = EnergyMonitor(energy_budget_wh=25.0)
        
        # Test initial state
        self.assertEqual(monitor.total_energy_consumed, 0.0)
        self.assertFalse(monitor.is_monitoring)
        
        # Start monitoring
        monitor.start_monitoring()
        self.assertTrue(monitor.is_monitoring)
        
        # Simulate some energy consumption
        import time
        time.sleep(0.1)
        monitor.log_step()
        
        # Stop monitoring
        metrics = monitor.stop_monitoring()
        
        # Verify final state
        self.assertFalse(monitor.is_monitoring)
        self.assertGreater(metrics.total_energy_wh, 0)
        self.assertGreater(monitor.total_energy_consumed, 0)
    
    def test_smart_sampling_workflow(self):
        """Test complete smart sampling workflow"""
        mock_energy_monitor = Mock()
        mock_energy_monitor.get_remaining_energy.return_value = 40.0
        
        sampler = EnergyAwareSampler(
            dataset_size=self.dataset_size,
            energy_monitor=mock_energy_monitor,
            base_batch_size=10,
            importance_weight=0.8
        )
        
        # Initial sampling
        batch1 = sampler.sample_batch(target_batch_size=8)
        self.assertLessEqual(len(batch1), 8)
        self.assertGreater(len(batch1), 0)
        
        # Simulate gradient update
        import torch
        gradients = torch.randn(len(batch1))
        sampler.update_gradient_importance(batch1, gradients)
        
        # Sample with importance scores
        importance_scores = sampler.get_importance_scores()
        batch2 = sampler.sample_batch(
            target_batch_size=8,
            importance_scores=importance_scores
        )
        
        # Verify without-replacement sampling
        self.assertEqual(len(set(batch1).intersection(set(batch2))), 0)
    
    def test_adaptive_batching_workflow(self):
        """Test complete adaptive batching workflow"""
        mock_energy_monitor = Mock()
        mock_energy_monitor.energy_budget_wh = 100.0
        
        batcher = EnergyAwareBatcher(
            base_batch_size=32,
            energy_monitor=mock_energy_monitor,
            min_batch_size=2
        )
        
        # Simulate training progression with decreasing energy
        scenarios = [
            (90.0, 0.1),  # High energy, early training
            (60.0, 0.4),  # Medium energy, mid training  
            (25.0, 0.8),  # Low energy, late training
            (5.0, 0.95)   # Very low energy, final phase
        ]
        
        batch_sizes = []
        for remaining_energy, progress in scenarios:
            mock_energy_monitor.get_remaining_energy.return_value = remaining_energy
            
            batch_size = batcher.get_adaptive_batch_size(convergence_progress=progress)
            batch_sizes.append(batch_size)
            
            # Verify valid batch size
            self.assertGreaterEqual(batch_size, 2)
            self.assertIsInstance(batch_size, int)
        
        # Verify general trend of decreasing batch sizes
        self.assertGreaterEqual(batch_sizes[0], batch_sizes[-1])
    
    def test_energy_budget_enforcement(self):
        """Test energy budget enforcement across components"""
        energy_budget = 20.0
        monitor = EnergyMonitor(energy_budget_wh=energy_budget)
        
        # Initially should have energy
        self.assertTrue(monitor.has_energy())
        self.assertEqual(monitor.get_remaining_energy(), energy_budget)
        
        # Simulate energy consumption beyond budget
        monitor.total_energy_consumed = energy_budget + 5.0
        
        # Should report no energy remaining
        self.assertFalse(monitor.has_energy())
        self.assertLessEqual(monitor.get_remaining_energy(), 0)
        
        # Components should respect this
        mock_energy_monitor = Mock()
        mock_energy_monitor.get_remaining_energy.return_value = 0.0
        
        sampler = EnergyAwareSampler(
            dataset_size=100,
            energy_monitor=mock_energy_monitor,
            base_batch_size=16
        )
        
        # Should return empty batch when no energy
        batch = sampler.sample_batch(target_batch_size=10)
        self.assertEqual(len(batch), 0)
    
    @patch('energypeft.core.energy_monitor.json.dump')
    def test_energy_reporting_workflow(self, mock_json_dump):
        """Test complete energy reporting workflow"""
        monitor = EnergyMonitor(energy_budget_wh=40.0)
        
        # Start monitoring
        monitor.start_monitoring()
        
        # Simulate training steps
        import time
        for _ in range(3):
            time.sleep(0.05)  # Small delay
            monitor.log_step()
        
        # Stop monitoring
        metrics = monitor.stop_monitoring()
        
        # Generate report
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            monitor.save_energy_log(temp_file)
            
            # Verify file was created
            self.assertTrue(os.path.exists(temp_file))
            mock_json_dump.assert_called_once()
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestIntegrationScenarios(unittest.TestCase):
    """Test realistic integration scenarios"""
    
    def test_low_energy_scenario(self):
        """Test behavior when energy budget is very low"""
        energy_peft = energypeft.EnergyPEFT(
            energy_budget_wh=2.0,  # Very low budget
            base_batch_size=32
        )
        
        # Components should adapt to low energy
        self.assertEqual(energy_peft.energy_monitor.energy_budget_wh, 2.0)
        self.assertTrue(energy_peft.energy_monitor.has_energy())
    
    def test_high_performance_scenario(self):
        """Test behavior with high-performance settings"""
        energy_peft = energypeft.EnergyPEFT(
            energy_budget_wh=200.0,  # High budget
            base_batch_size=64,      # Large batches
            importance_weight=0.9    # High importance weight
        )
        
        # Should handle high-performance requirements
        self.assertEqual(energy_peft.energy_monitor.energy_budget_wh, 200.0)
        self.assertEqual(energy_peft.base_batch_size, 64)
        self.assertEqual(energy_peft.importance_weight, 0.9)
    
    def test_error_recovery_scenario(self):
        """Test error recovery in end-to-end workflows"""
        energy_peft = energypeft.EnergyPEFT(energy_budget_wh=30.0)
        
        # Test handling of various error conditions
        with patch.object(energy_peft.energy_monitor, 'start_monitoring', 
                         side_effect=Exception("Hardware error")):
            
            # Should handle monitoring errors gracefully
            try:
                wrapper = LlamaFactoryEnergyWrapper(energy_framework=energy_peft)
                # Should create wrapper even if monitoring fails
                self.assertIsNotNone(wrapper)
            except Exception as e:
                # Or should raise appropriate error
                self.assertIn("error", str(e).lower())


class TestPerformanceBenchmarks(unittest.TestCase):
    """Test performance characteristics of end-to-end workflows"""
    
    def test_large_dataset_handling(self):
        """Test handling of large datasets"""
        large_dataset_size = 10000
        
        mock_energy_monitor = Mock()
        mock_energy_monitor.get_remaining_energy.return_value = 100.0
        
        sampler = EnergyAwareSampler(
            dataset_size=large_dataset_size,
            energy_monitor=mock_energy_monitor,
            base_batch_size=64
        )
        
        # Should handle large datasets efficiently
        batch = sampler.sample_batch(target_batch_size=50)
        self.assertLessEqual(len(batch), 50)
        
        # All indices should be valid
        self.assertTrue(all(0 <= idx < large_dataset_size for idx in batch))
    
    def test_memory_efficiency(self):
        """Test memory efficiency of components"""
        # Test that components don't consume excessive memory
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create multiple components
        components = []
        for _ in range(10):
            energy_peft = energypeft.EnergyPEFT(energy_budget_wh=50.0)
            components.append(energy_peft)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for 10 instances)
        self.assertLess(memory_increase, 100 * 1024 * 1024)


if __name__ == '__main__':
    unittest.main()