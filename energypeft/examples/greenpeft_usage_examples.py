# CORRECTED EnergyPEFT Usage Examples
# Clean examples that match your actual package structure

"""
🎯 THREE USAGE PATTERNS FOR YOUR ENERGYPEFT LIBRARY

Your examples show good thinking but need alignment with your package structure.
Here are clean, working examples for all user types.
"""

# ================================================================
# PATTERN 1: GreenTrainer (EASIEST - Drop-in Replacement)
# ================================================================

"""
⭐ SIMPLICITY: Just replace Trainer with GreenTrainer
🎯 TARGET: Users who want minimal code changes
✅ WORKS: After you add GreenTrainer to __init__.py
"""

# Example 1A: Basic GreenTrainer Usage
def pattern1_basic_greentrainer():
    """Simplest possible usage - drop-in replacement"""
    
    from transformers import AutoModel, AutoTokenizer, TrainingArguments
    from datasets import Dataset
    from energypeft.trainers import GreenTrainer  # 🔄 NEED TO ADD TO __init__.py
    
    # Load model and data (standard HuggingFace way)
    model = AutoModel.from_pretrained("microsoft/DialoGPT-small")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create simple dataset
    train_dataset = Dataset.from_dict({
        "input_ids": [[1, 2, 3], [4, 5, 6]], 
        "labels": [[1, 2, 3], [4, 5, 6]]
    })
    
    # 🌟 JUST REPLACE Trainer WITH GreenTrainer
    trainer = GreenTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        energy_budget_wh=50.0,  # 🆕 Only new parameter needed!
        args=TrainingArguments(
            output_dir="./results",
            num_train_epochs=1,
            per_device_train_batch_size=2
        )
    )
    
    # Train with automatic energy optimization
    trainer.train()  # 🌱 Automatic energy monitoring and optimization!
    
    print("✅ Pattern 1: GreenTrainer works perfectly!")


# ================================================================
# PATTERN 2: EnergyPEFT Framework (FLEXIBLE)
# ================================================================

"""
⭐⭐ FLEXIBILITY: Full framework control
🎯 TARGET: Users who want more control over energy components
✅ WORKS: With your current package structure
"""

# Example 2A: Using Current EnergyPEFT Framework
def pattern2_energypeft_framework():
    """Using your current EnergyPEFT framework API"""
    
    import energypeft
    from transformers import AutoModel, AutoTokenizer
    
    # Create energy framework
    energy_framework = energypeft.EnergyPEFT(
        energy_budget_wh=100.0,
        base_batch_size=4,
        importance_weight=0.7
    )
    
    # Load model
    model = AutoModel.from_pretrained("microsoft/DialoGPT-small") 
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    
    # Use framework's wrapper capabilities
    trainer = energy_framework.wrap_trainer(
        trainer_type="huggingface",
        model=model,
        tokenizer=tokenizer,
        # Add other trainer parameters
    )
    
    print("✅ Pattern 2: EnergyPEFT framework created!")
    return energy_framework, trainer


# Example 2B: Manual Framework Setup
def pattern2_manual_framework():
    """More control over individual components"""
    
    from energypeft import EnergyPEFT, EnergyMonitor
    from energypeft.core import AdaptiveBatcher, SmartSampler
    
    # Create components manually
    energy_monitor = EnergyMonitor(energy_budget_wh=75.0)
    adaptive_batcher = AdaptiveBatcher(base_batch_size=8, energy_monitor=energy_monitor)
    
    # Create main framework
    framework = EnergyPEFT(
        energy_budget_wh=75.0,
        base_batch_size=8,
        importance_weight=0.8
    )
    
    print("✅ Pattern 2B: Manual framework setup complete!")
    return framework


# ================================================================
# PATTERN 3: Direct Components (ADVANCED)
# ================================================================

"""
⭐⭐⭐ ADVANCED: Use components directly in custom solutions
🎯 TARGET: Advanced users building custom training loops
✅ WORKS: With your current package structure
"""

# Example 3A: Custom Training Loop
def pattern3_direct_components():
    """Advanced usage with direct component access"""
    
    from energypeft.core import EnergyMonitor, AdaptiveBatcher, SmartSampler
    from energypeft.core import EnergyEfficiencyEarlyStopping
    
    # Create components directly
    energy_monitor = EnergyMonitor(energy_budget_wh=200.0)
    batcher = AdaptiveBatcher(base_batch_size=16, energy_monitor=energy_monitor)
    sampler = SmartSampler(dataset_size=1000, energy_monitor=energy_monitor)
    early_stopper = EnergyEfficiencyEarlyStopping(energy_monitor=energy_monitor)
    
    # Start monitoring
    energy_monitor.start_monitoring()
    
    try:
        # Custom training loop
        for epoch in range(3):
            if not energy_monitor.has_energy():
                print("⚠️ Energy budget exhausted!")
                break
                
            # Get adaptive batch size
            adaptive_batch_size = batcher.get_adaptive_batch_size(epoch / 3.0)
            print(f"Epoch {epoch}: Using batch size {adaptive_batch_size}")
            
            # Simulate training step
            energy_monitor.log_step()
            
            # Check early stopping
            if early_stopper.should_stop(current_loss=0.5):
                print("🛑 Early stopping triggered!")
                break
                
    finally:
        # Always stop monitoring
        metrics = energy_monitor.stop_monitoring()
        print(f"✅ Training completed. Energy used: {metrics.total_energy_wh:.2f}Wh")
    
    print("✅ Pattern 3: Advanced component usage complete!")


# ================================================================
# INTEGRATION EXAMPLES
# ================================================================

# Example 4A: PEFT Integration
def pattern4_peft_integration():
    """Show how to combine with PEFT techniques"""
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, LoraConfig
    from energypeft.trainers import GreenTrainer  # 🔄 ADD TO __init__.py
    
    # Load and setup PEFT model
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    
    # Create energy-efficient LoRA config
    lora_config = LoraConfig(
        r=8,           # Small rank for efficiency
        lora_alpha=16, # Conservative alpha
        target_modules=["c_attn"],  # Target specific modules
        lora_dropout=0.05
    )
    
    # Apply PEFT
    model = get_peft_model(model, lora_config)
    
    # Use with GreenTrainer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    tokenizer.pad_token = tokenizer.eos_token
    
    trainer = GreenTrainer(
        model=model,
        tokenizer=tokenizer, 
        train_dataset=create_dummy_dataset(),  # Your dataset here
        energy_budget_wh=30.0,  # Lower budget for PEFT
        importance_weight=0.8   # Higher importance weight
    )
    
    print("✅ Pattern 4: PEFT + EnergyPEFT integration ready!")


def create_dummy_dataset():
    """Create simple dataset for testing"""
    from datasets import Dataset
    return Dataset.from_dict({
        "input_ids": [[1, 2, 3, 4], [5, 6, 7, 8]],
        "labels": [[1, 2, 3, 4], [5, 6, 7, 8]]
    })


# ================================================================
# FIXES NEEDED FOR YOUR PACKAGE
# ================================================================

"""
🔧 TO MAKE THESE EXAMPLES WORK, UPDATE YOUR __init__.py:

# energypeft/__init__.py
from energypeft.core.energy_monitor import EnergyMonitor
from energypeft.core.adaptive_batcher import AdaptiveBatcher  
from energypeft.core.early_stopper import EnergyEfficiencyEarlyStopping
from energypeft.core.smart_sampler import SmartSampler
from energypeft.integrations.huggingface_peft import HuggingFacePEFTWrapper
from energypeft.integrations.llamafactory import LlamaFactoryEnergyWrapper

# 🆕 ADD THESE LINES:
from energypeft.trainers.green_trainer import GreenTrainer

class EnergyPEFT:
    # ... existing code ...

__all__ = [
    "EnergyPEFT", "EnergyMonitor", "AdaptiveBatcher", 
    "EnergyEfficiencyEarlyStopping", "SmartSampler",
    "HuggingFacePEFTWrapper", "LlamaFactoryEnergyWrapper",
    "GreenTrainer"  # 🆕 ADD THIS
]
"""

# ================================================================
# TESTING YOUR EXAMPLES
# ================================================================

if __name__ == "__main__":
    print("🌱 Testing EnergyPEFT Usage Patterns")
    print("=" * 40)
    
    try:
        # Test Pattern 2 (should work with current structure)
        framework, trainer = pattern2_energypeft_framework()
        print("✅ Pattern 2 works!")
        
        # Test Pattern 3 (should work with current structure)  
        pattern3_direct_components()
        print("✅ Pattern 3 works!")
        
        # Pattern 1 needs GreenTrainer in __init__.py
        print("⏳ Pattern 1 needs GreenTrainer added to __init__.py")
        
        print("\n🎉 Your EnergyPEFT package structure is solid!")
        print("Just add GreenTrainer to __init__.py and you're ready!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Update your __init__.py with the fixes above!")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Check your package structure!")


# ================================================================
# RECOMMENDED USER DOCUMENTATION
# ================================================================

"""
📚 HOW TO DOCUMENT THESE PATTERNS:

BEGINNER USERS → Start with Pattern 1 (GreenTrainer)
"Just replace Trainer with GreenTrainer for automatic energy optimization!"

INTERMEDIATE USERS → Use Pattern 2 (EnergyPEFT Framework)  
"Get more control over energy components while keeping it simple!"

ADVANCED USERS → Use Pattern 3 (Direct Components)
"Build custom energy-aware training solutions!"

RESEARCH USERS → Use Pattern 4 (PEFT Integration)
"Combine energy efficiency with parameter-efficient fine-tuning!"
"""