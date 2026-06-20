# energypeft/integrations/huggingface_peft.py
"""
HuggingFace PEFT integration with energy-aware training.

This module wraps PEFT (Parameter-Efficient Fine-Tuning) with energy monitoring.
"""

from typing import Optional, List

from peft import get_peft_model, LoraConfig
from transformers import Trainer, TrainingArguments


def get_default_lora_config(
    target_modules: Optional[List[str]] = None,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
) -> LoraConfig:
    """
    Create a sensible default LoRA config.
    
    Args:
        target_modules: List of module names to apply LoRA to.
                        If None, uses common defaults for transformer models.
        r: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        lora_dropout: Dropout probability
    
    Returns:
        LoraConfig with valid settings
    """
    if target_modules is None:
        # Common defaults that work for most transformer models
        target_modules = ["q_proj", "v_proj"]
    
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )


class HuggingFacePEFTTrainer:
    """
    Simple PEFT trainer wrapper.
    
    Note: For energy-aware training, use GreenTrainer instead. This class
    is a basic utility for applying LoRA and training without energy features.
    """

    def __init__(
        self,
        model,
        train_dataset,
        eval_dataset=None,
        peft_config: Optional[LoraConfig] = None,
        training_args: Optional[TrainingArguments] = None,
        target_modules: Optional[List[str]] = None,
    ):
        """
        Args:
            model: Base model to apply PEFT to
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            peft_config: LoRA config. If None, creates default with target_modules.
            training_args: HF TrainingArguments. If None, uses minimal defaults.
            target_modules: If peft_config is None, use these target modules for LoRA.
        """
        # Create PEFT config if not provided
        if peft_config is None:
            peft_config = get_default_lora_config(target_modules=target_modules)
        
        # Apply PEFT
        self.model = get_peft_model(model, peft_config)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Create training args if not provided
        if training_args is None:
            training_args = TrainingArguments(
                output_dir="./peft_results",
                num_train_epochs=1,
                per_device_train_batch_size=4,
                logging_steps=10,
            )
        self.training_args = training_args
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
        )

    def train(self):
        """Run training and return results."""
        return self.trainer.train()

    def evaluate(self):
        """Run evaluation and return metrics."""
        return self.trainer.evaluate()

    def save_model(self, output_dir: str):
        """Save the PEFT adapter weights."""
        self.model.save_pretrained(output_dir)
