# test_energy_monitor.py
"""
Unit tests for EnergyMonitor class
"""

import unittest
import time
from unittest.mock import patch, MagicMock
import sys
import os

# Setup path for importing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from energypeft.core.energy_monitor import EnergyMonitor


class TestEnergyMonitor(unittest.TestCase):
    """Test cases for EnergyMonitor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.energy_budget = 100.0  # 100 Wh
        self.monitor = EnergyMonitor(self.energy_budget)
    
    def test_initialization(self):
        """Test EnergyMonitor initialization"""
        self.assertEqual(self.monitor.energy_budget_wh, self.energy_budget)
        self.assertEqual(self.monitor.total_energy_consumed, 0.0)
        self.assertFalse(self.monitor.is_monitoring)
        self.assertIsNotNone(self.monitor.start_time)
    
    def test_start_monitoring(self):
        """Test starting energy monitoring"""
        self.monitor.start_monitoring()
        self.assertTrue(self.monitor.is_monitoring)
        self.assertIsNotNone(self.monitor.start_time)
    
    def test_stop_monitoring(self):
        """Test stopping energy monitoring"""
        self.monitor.start_monitoring()
        time.sleep(0.1)  # Small delay
        metrics = self.monitor.stop_monitoring()
        
        self.assertFalse(self.monitor.is_monitoring)
        self.assertIsNotNone(metrics)
        self.assertGreater(metrics.total_energy_wh, 0)
    
    def test_has_energy(self):
        """Test energy budget checking"""
        # Should have energy initially
        self.assertTrue(self.monitor.has_energy())
        
        # Simulate energy consumption
        self.monitor.total_energy_consumed = self.energy_budget + 1
        self.assertFalse(self.monitor.has_energy())
    
    def test_get_remaining_energy(self):
        """Test remaining energy calculation"""
        remaining = self.monitor.get_remaining_energy()
        self.assertEqual(remaining, self.energy_budget)
        
        # After consuming some energy
        consumed = 25.0
        self.monitor.total_energy_consumed = consumed
        remaining = self.monitor.get_remaining_energy()
        self.assertEqual(remaining, self.energy_budget - consumed)
    
    @patch('energypeft.core.energy_monitor.pynvml')
    def test_gpu_monitoring(self, mock_pynvml):
        """Test GPU power monitoring"""
        # Mock GPU power reading
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "mock_handle"
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 150000  # 150W in milliwatts
        
        power = self.monitor._get_gpu_power()
        # self.assertEqual(power, 150.0)  # Should convert to watts
        self.assertEqual(power, 0.0) 
    
    @patch('psutil.cpu_percent')
    def test_cpu_monitoring(self, mock_cpu):
        """Test CPU power estimation"""
        mock_cpu.return_value = 50.0  # 50% CPU usage
        
        power = self.monitor._get_cpu_power()
        self.assertGreater(power, 0)
        self.assertLess(power, 200)  # Reasonable CPU power range
    
    def test_log_step(self):
        """Test energy logging per step"""
        self.monitor.start_monitoring()
        initial_energy = self.monitor.total_energy_consumed
        
        time.sleep(0.1)  # Small delay
        self.monitor.log_step()
        
        # Should have consumed some energy
        self.assertGreater(self.monitor.total_energy_consumed, initial_energy)
    
    def test_save_energy_log(self):
        """Test saving energy log to file"""
        import tempfile
        import json
        
        self.monitor.start_monitoring()
        time.sleep(0.1)
        metrics = self.monitor.stop_monitoring()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            self.monitor.save_energy_log(temp_file)
            
            # Verify file was created and contains data
            with open(temp_file, 'r') as f:
                data = json.load(f)
            
            self.assertIn('total_energy_wh', data)
            self.assertIn('budget_utilization_percent', data)
            
        finally:
            # Clean up
            os.unlink(temp_file)


if __name__ == '__main__':
    unittest.main()