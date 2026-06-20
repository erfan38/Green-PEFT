# energypeft/integrations/llamafactory.py
"""
LlamaFactory integration with energy-aware training.

Wraps LlamaFactory CLI with energy monitoring and provides a clean Python API.
"""

import os
import shutil
import subprocess
import sys
from typing import Any, Dict, Optional

import yaml


class LlamaFactoryNotFoundError(Exception):
    """Raised when llamafactory-cli is not installed."""
    pass


class LlamaFactoryEnergyWrapper:
    """Energy-aware wrapper for LlamaFactory training."""

    def __init__(self, energy_framework, llamafactory_config: Optional[Dict] = None):
        """
        Args:
            energy_framework: EnergyPEFT instance with energy_monitor
            llamafactory_config: Optional base configuration dict
        """
        self.energy_framework = energy_framework

        if llamafactory_config is not None and not isinstance(llamafactory_config, dict):
            raise TypeError("llamafactory_config must be a dictionary")

        self.config = llamafactory_config or {}
        
        # Validate that llamafactory-cli exists
        self._cli_path = self._find_cli()

    def _find_cli(self) -> str:
        """Find llamafactory-cli or raise helpful error."""
        cli = shutil.which("llamafactory-cli")
        if cli is None:
            raise LlamaFactoryNotFoundError(
                "llamafactory-cli not found in PATH. "
                "Install with: pip install llamafactory"
            )
        return cli

    def create_energy_config(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        dataset: str = "alpaca_gpt4_en",
        output_dir: str = "./energy_efficient_llama",
    ) -> Dict[str, Any]:
        """Generate energy-optimized LlamaFactory config."""

        energy_config = {
            # Model settings
            "model_name_or_path": model_name,
            "template": "llama3",
            # Dataset settings
            "dataset": dataset,
            "cutoff_len": 1024,
            # PEFT settings
            "finetuning_type": "lora",
            "lora_target": "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj",
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            # Energy-aware training settings
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "num_train_epochs": 2,
            "max_samples": 1000,
            # Optimization settings
            "learning_rate": 1e-4,
            "lr_scheduler_type": "cosine",
            "warmup_steps": 20,
            "max_grad_norm": 1.0,
            # Logging settings
            "logging_steps": 10,
            "save_steps": 100,
            "eval_steps": 50,
            "evaluation_strategy": "steps",
            "load_best_model_at_end": True,
            "save_total_limit": 2,
            # Output settings
            "output_dir": output_dir,
            "overwrite_output_dir": True,
            "overwrite_cache": True,
            # Hardware settings
            "quantization_bit": 4,
            "report_to": "tensorboard",
        }

        # Merge with user config
        energy_config.update(self.config)

        return energy_config

    def train_with_energy_monitoring(
        self,
        config: Optional[Dict] = None,
        config_file: str = "energy_config.yaml",
        stream_logs: bool = True,
    ) -> Dict[str, Any]:
        """
        Run LlamaFactory training with energy monitoring.

        Args:
            config: Training config dict. If None, uses create_energy_config().
            config_file: Path to save YAML config
            stream_logs: If True, stream training logs to stdout in real-time

        Returns:
            Dict with energy_consumed_wh, budget_utilization, and config_used
        """
        if config is None:
            config = self.create_energy_config()

        # Save config to file
        with open(config_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        # Start energy monitoring
        self.energy_framework.energy_monitor.start_monitoring()

        budget = self.energy_framework.energy_monitor.energy_budget_wh
        print(f"🌱 Starting energy-aware training with {budget}Wh budget")
        print(f"📁 Config: {config_file}")

        try:
            # Build command as list (safer than shell=True)
            cmd = [self._cli_path, "train", config_file]

            if stream_logs:
                # Stream output in real-time
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,  # Line buffered
                )

                # Stream stdout line by line
                for line in iter(process.stdout.readline, ""):
                    sys.stdout.write(line)
                    sys.stdout.flush()

                process.wait()
                returncode = process.returncode
            else:
                # Capture output (original behavior)
                result = subprocess.run(cmd, capture_output=True, text=True)
                returncode = result.returncode
                if returncode != 0:
                    print(f"Training failed: {result.stderr}")

            if returncode != 0:
                print(f"⚠️ Training exited with code {returncode}")

        finally:
            # Stop monitoring and get results
            metrics = self.energy_framework.energy_monitor.stop_monitoring()

            # Save energy report
            self.energy_framework.energy_monitor.save_energy_log("energy_report.json")

            print(f"\n🎉 Training completed!")
            print(f"⚡ Energy consumed: {metrics.total_energy_wh:.2f} Wh")
            print(f"📊 Budget utilization: {metrics.budget_used_percent:.1f}%")

            return {
                "energy_consumed_wh": metrics.total_energy_wh,
                "budget_utilization": metrics.budget_used_percent,
                "config_used": config,
            }
