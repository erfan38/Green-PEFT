# tests/__init__.py
"""
EnergyPEFT Test Suite

This package contains comprehensive tests for all EnergyPEFT components.
Run with: pytest tests/
"""

__version__ = "0.1.0"

# Test configuration
import sys
import os

# Add the parent directory to the path so we can import energypeft
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Common test utilities
def setup_test_environment():
    """Setup common test environment"""
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
def create_mock_model():
    """Create a mock model for testing"""
    import torch
    import torch.nn as nn
    
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 2)
            
        def forward(self, x):
            return self.linear(x)
    
    return MockModel()

def create_mock_dataset():
    """Create a mock dataset for testing"""
    import torch
    from torch.utils.data import Dataset
    
    class MockDataset(Dataset):
        def __init__(self, size=100):
            self.size = size
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            return {
                'input_ids': torch.randint(0, 1000, (10,)),
                'labels': torch.randint(0, 2, (1,))
            }
    
    return MockDataset()

# Export test utilities
__all__ = [
    "setup_test_environment",
    "create_mock_model", 
    "create_mock_dataset"
]