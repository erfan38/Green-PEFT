import sys
import os
import platform
from unittest.mock import patch, MagicMock
from energypeft.core.energy_monitor import EnergyMonitor

def test_mac_powermetrics_parsing():
    # Only run logic if on mac for testing, but we can mock it anyway
    monitor = EnergyMonitor(energy_budget_wh=100.0, cpu_backend="powermetrics")
    
    # Mock subprocess.run to return a fake powermetrics output
    fake_output = """
*** Sampled system activity (100.04 ms) (Wed Feb 19 12:00:00 2026) ***

Machine model: Mac14,2
OS version: 23A344
Boot arguments: 

**** Processor usage ****
CPU Power: 1200 mW
GPU Power: 5300 mW
ANE Power: 0 mW
Combined Power: 6500 mW
    """
    
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = fake_output
    
    with patch("subprocess.run", return_value=mock_result), \
         patch("time.time", return_value=1000.0):
        
        # This should trigger the refresh
        gpu_power = monitor._get_gpu_power()
        cpu_power, _ = monitor._get_cpu_power_and_optional_energy(now_mono=1000.0)
        
        print(f"Parsed GPU Power: {gpu_power} W")
        print(f"Parsed CPU+ANE Power: {cpu_power} W")
        
        assert gpu_power == 5.3, f"Expected 5.3 W for GPU, got {gpu_power}"
        assert cpu_power == 1.2, f"Expected 1.2 W for CPU+ANE, got {cpu_power}"
        print("✅ Parsing test passed!")

if __name__ == "__main__":
    test_mac_powermetrics_parsing()
