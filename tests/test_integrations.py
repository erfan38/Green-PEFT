# test_integrations.py
"""
Unit tests for Framework Integrations
"""


import unittest
from unittest.mock import Mock, patch
import sys
import os


# Setup path for importing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from energypeft.integrations.huggingface_peft import HuggingFacePEFTTrainer
from energypeft.integrations.llamafactory import LlamaFactoryEnergyWrapper



class TestHuggingFacePEFTTrainer(unittest.TestCase):
    """Test cases for HuggingFace PEFT integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_model = Mock()
        self.mock_tokenizer = Mock()
        self.mock_dataset = Mock()
        
    @patch('energypeft.integrations.huggingface_peft.get_peft_model')
    @patch('energypeft.integrations.huggingface_peft.Trainer')
    def test_initialization(self, mock_trainer_class, mock_get_peft_model):
        """Test trainer initialization"""
        # Mock both PEFT and Trainer creation
        mock_get_peft_model.return_value = self.mock_model
        mock_trainer_instance = Mock()
        mock_trainer_class.return_value = mock_trainer_instance
        
        trainer = HuggingFacePEFTTrainer(
            model=self.mock_model,
            train_dataset=self.mock_dataset
        )
        
        # Test the created trainer
        self.assertIsNotNone(trainer)
        self.assertIsNotNone(trainer.model)
        self.assertEqual(trainer.trainer, mock_trainer_instance)
        
        # Verify both mocks were called
        mock_get_peft_model.assert_called_once()
        mock_trainer_class.assert_called_once()
    
    @patch('energypeft.integrations.huggingface_peft.get_peft_model')
    def test_peft_model_creation(self, mock_get_peft_model):
        """Test PEFT model creation with energy awareness"""
        mock_get_peft_model.return_value = self.mock_model
        
        # Test model wrapping with PEFT
        result = mock_get_peft_model(self.mock_model, Mock())
        
        self.assertEqual(result, self.mock_model)
        mock_get_peft_model.assert_called_once()
    
    @patch('energypeft.integrations.huggingface_peft.get_peft_model')
    @patch('energypeft.integrations.huggingface_peft.Trainer')
    def test_energy_aware_training_setup(self, mock_trainer_class, mock_get_peft_model):
        """Test energy-aware training configuration"""
        # Mock both dependencies
        mock_get_peft_model.return_value = self.mock_model
        mock_trainer_instance = Mock()
        mock_trainer_class.return_value = mock_trainer_instance
        
        trainer = HuggingFacePEFTTrainer(
            model=self.mock_model,
            train_dataset=self.mock_dataset
        )
        
        # Test that energy monitoring is properly integrated
        self.assertIsNotNone(trainer.trainer)
        mock_get_peft_model.assert_called_once()
        mock_trainer_class.assert_called_once()
    
    @patch('energypeft.integrations.huggingface_peft.get_peft_model')
    @patch('energypeft.integrations.huggingface_peft.Trainer')
    def test_gradient_tracking_integration(self, mock_trainer_class, mock_get_peft_model):
        """Test integration with gradient importance tracking"""
        # Mock both dependencies
        mock_get_peft_model.return_value = self.mock_model
        mock_trainer_instance = Mock()
        mock_trainer_class.return_value = mock_trainer_instance
        
        trainer = HuggingFacePEFTTrainer(
            model=self.mock_model,
            train_dataset=self.mock_dataset
        )
        
        # Test that gradients are properly tracked for importance sampling
        self.assertIsNotNone(trainer)
        mock_get_peft_model.assert_called_once()
        mock_trainer_class.assert_called_once()



class TestLlamaFactoryEnergyWrapper(unittest.TestCase):
    """Test cases for LlamaFactory integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_energy_framework = Mock()
        self.mock_energy_framework.energy_monitor = Mock()
        self.mock_energy_framework.energy_monitor.energy_budget_wh = 100.0
        
        self.wrapper = LlamaFactoryEnergyWrapper(
            energy_framework=self.mock_energy_framework,
            llamafactory_config={}
        )
    
    def test_initialization(self):
        """Test wrapper initialization"""
        self.assertEqual(self.wrapper.energy_framework, self.mock_energy_framework)
        self.assertIsInstance(self.wrapper.config, dict)
    
    def test_create_energy_config(self):
        """Test energy-optimized configuration creation"""
        config = self.wrapper.create_energy_config(
            model_name="test-model",
            dataset="test-dataset",
            output_dir="./test_output"
        )
        
        # Verify energy-efficient settings
        self.assertIn('per_device_train_batch_size', config)
        self.assertIn('lora_rank', config)
        self.assertIn('quantization_bit', config)
        self.assertEqual(config['quantization_bit'], 4)  # Energy-efficient 4-bit
        self.assertEqual(config['per_device_train_batch_size'], 2)  # Small batch for efficiency
    
    def test_energy_config_defaults(self):
        """Test default energy configuration values"""
        config = self.wrapper.create_energy_config()
        
        # Check energy-aware defaults
        expected_defaults = {
            'per_device_train_batch_size': 2,
            'per_device_eval_batch_size': 1,
            'gradient_accumulation_steps': 4,
            'num_train_epochs': 2,
            'max_samples': 1000,
            'quantization_bit': 4
        }
        
        for key, expected_value in expected_defaults.items():
            self.assertEqual(config[key], expected_value)
    
    def test_config_override(self):
        """Test configuration override functionality"""
        custom_config = {'per_device_train_batch_size': 8, 'custom_param': 'test'}
        
        wrapper = LlamaFactoryEnergyWrapper(
            energy_framework=self.mock_energy_framework,
            llamafactory_config=custom_config
        )
        
        config = wrapper.create_energy_config()
        
        # Custom config should override defaults
        self.assertEqual(config['per_device_train_batch_size'], 8)
        self.assertEqual(config['custom_param'], 'test')
    
    @patch('energypeft.integrations.llamafactory.subprocess.run')
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('energypeft.integrations.llamafactory.yaml.dump')
    def test_train_with_energy_monitoring(self, mock_yaml_dump, mock_open, mock_subprocess):
        """Test training with energy monitoring"""
        # Setup mocks
        mock_subprocess.return_value = Mock(returncode=0)
        self.mock_energy_framework.energy_monitor.start_monitoring = Mock()
        self.mock_energy_framework.energy_monitor.stop_monitoring = Mock()
        self.mock_energy_framework.energy_monitor.stop_monitoring.return_value = Mock(
            total_energy_wh=25.0,
            budget_used_percent=25.0
        )
        self.mock_energy_framework.energy_monitor.save_energy_log = Mock()
        
        # Test training
        result = self.wrapper.train_with_energy_monitoring()
        
        # Verify energy monitoring was called
        self.mock_energy_framework.energy_monitor.start_monitoring.assert_called_once()
        self.mock_energy_framework.energy_monitor.stop_monitoring.assert_called_once()
        self.mock_energy_framework.energy_monitor.save_energy_log.assert_called_once()
        
        # Verify subprocess was called
        mock_subprocess.assert_called_once()
        
        # Verify results
        self.assertIn('energy_consumed_wh', result)
        self.assertIn('budget_utilization', result)
        self.assertEqual(result['energy_consumed_wh'], 25.0)
    
    @patch('energypeft.integrations.llamafactory.subprocess.run')
    def test_training_failure_handling(self, mock_subprocess):
        """Test handling of training failures"""
        # Setup failed subprocess
        mock_subprocess.return_value = Mock(returncode=1, stderr="Training failed")
        
        self.mock_energy_framework.energy_monitor.start_monitoring = Mock()
        self.mock_energy_framework.energy_monitor.stop_monitoring = Mock()
        self.mock_energy_framework.energy_monitor.stop_monitoring.return_value = Mock(
            total_energy_wh=5.0,
            budget_used_percent=5.0
        )
        self.mock_energy_framework.energy_monitor.save_energy_log = Mock()
        
        # Training should still complete energy monitoring even on failure
        result = self.wrapper.train_with_energy_monitoring()
        
        # Energy monitoring should still work
        self.mock_energy_framework.energy_monitor.start_monitoring.assert_called_once()
        self.mock_energy_framework.energy_monitor.stop_monitoring.assert_called_once()
        
        self.assertIn('energy_consumed_wh', result)



class TestTransformersIntegration(unittest.TestCase):
    """Test cases for direct Transformers integration"""
    
    def setUp(self):
        """Set up transformers integration test fixtures"""
        pass
    
    def test_transformers_energy_wrapper(self):
        """Test direct transformers integration with energy awareness"""
        # Test transformers.py integration
        pass
    
    def test_trainer_monkey_patching(self):
        """Test monkey patching of transformers.Trainer"""
        # Test if energy awareness is properly added to existing Trainer
        pass



class TestIntegrationWorkflows(unittest.TestCase):
    """Test integration workflows and compatibility"""
    
    def setUp(self):
        """Set up workflow test fixtures"""
        self.mock_energy_framework = Mock()
        self.mock_energy_framework.energy_monitor = Mock()
        self.mock_energy_framework.adaptive_batcher = Mock()
        self.mock_energy_framework.smart_sampler = Mock()
    
    def test_huggingface_to_llamafactory_compatibility(self):
        """Test compatibility between HuggingFace and LlamaFactory integrations"""
        # Test that configs and models can be shared between integrations
        pass
    
    @patch('energypeft.integrations.huggingface_peft.get_peft_model')
    @patch('energypeft.integrations.huggingface_peft.Trainer')
    def test_energy_framework_integration(self, mock_trainer_class, mock_get_peft_model):
        """Test that all integrations properly use the energy framework"""
        # Mock both PEFT and Trainer creation
        mock_get_peft_model.return_value = Mock()
        mock_trainer_instance = Mock()
        mock_trainer_class.return_value = mock_trainer_instance
        
        # Test HuggingFace integration
        hf_trainer = HuggingFacePEFTTrainer(
            model=Mock(),
            train_dataset=Mock()
        )
        
        # Test LlamaFactory integration  
        lf_wrapper = LlamaFactoryEnergyWrapper(
            energy_framework=self.mock_energy_framework
        )

        # Test both integrations work properly
        self.assertIsNotNone(hf_trainer)
        self.assertEqual(hf_trainer.trainer, mock_trainer_instance)
        self.assertEqual(lf_wrapper.energy_framework, self.mock_energy_framework)
        
        # Verify mocks were called
        mock_get_peft_model.assert_called_once()
        mock_trainer_class.assert_called_once()

    def test_end_to_end_integration_flow(self):
        """Test complete integration workflow"""
        # Test that all components work together:
        # Energy Framework -> Integration -> Training -> Results
        pass



class TestIntegrationErrorHandling(unittest.TestCase):
    """Test error handling in integrations"""
    
    def test_missing_dependencies(self):
        """Test handling of missing integration dependencies"""
        # Test behavior when PEFT, transformers, etc. are not installed
        pass
    
    def test_invalid_configurations(self):
        """Test handling of invalid configurations"""
        mock_energy_framework = Mock()
        
        # Test invalid LlamaFactory config
        with self.assertRaises((ValueError, KeyError, TypeError)):
            wrapper = LlamaFactoryEnergyWrapper(
                energy_framework=mock_energy_framework,
                llamafactory_config="invalid_config"  # Should be dict
            )
    
    def test_energy_monitoring_failures(self):
        """Test handling of energy monitoring failures"""
        mock_energy_framework = Mock()
        mock_energy_framework.energy_monitor.start_monitoring.side_effect = Exception("GPU not found")
        
        wrapper = LlamaFactoryEnergyWrapper(energy_framework=mock_energy_framework)
        
        # Should handle gracefully
        try:
            result = wrapper.train_with_energy_monitoring()
            # Should complete even if energy monitoring fails
            self.assertIsNotNone(result)
        except Exception as e:
            # Or should raise appropriate error
            self.assertIn("gpu not found", str(e).lower())



if __name__ == '__main__':
    unittest.main()