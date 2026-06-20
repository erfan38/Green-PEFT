# energypeft/trainers/green_trainer.py

import json
from dataclasses import asdict
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer

from ..core.energy_monitor import EnergyMonitor
from ..core.efficient_training import EnergyAwareTrainingController

class _HFDataCollatorWithIndices:
    def __init__(self, base_collator, device=None):
        self.base = base_collator
        self.device = device  # model device; tensors are moved here immediately

    def __call__(self, features):
        feats = [dict(f) for f in features]  # robust conversion
        indices = [int(f.get("_index", -1)) for f in feats]
        for f in feats:
            f.pop("_index", None)

        batch = self.base(feats)
        batch["_indices"] = indices

        # Move all tensors to the model device so inputs never arrive on CPU
        # when the model is on CUDA/MPS. Without this, every training step
        # pays a full host→device copy inside compute_loss, which also corrupts
        # the loss values fed to LossEfficiencyTracker.
        if self.device is not None and str(self.device).startswith("mps"):
            batch = {
                k: v.to(self.device) if hasattr(v, "to") else v
                for k, v in batch.items()
            }
        return batch
        
class _IndexWrapper(Dataset):
    """
    Wraps any HF-style dataset (or list-like dataset) and injects `_index`
    so the efficient training controller can update per-sample scores.
    """

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.base[idx]
        # Convert to dict if needed
        if not isinstance(item, dict):
            raise TypeError("Dataset items must be dict-like for GreenTrainer.")
        item = dict(item)
        item["_index"] = idx
        return item


def _per_sample_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute per-sample (per-sequence) loss for causal LM.

    logits: [B, T, V]
    labels: [B, T]  (can include -100)
    attention_mask: [B, T] (1 real, 0 pad). If None, assume all tokens are real.

    Returns:
        loss_per_sample: [B]
    """
    # Shift for causal LM
    shift_logits = logits[:, :-1, :].contiguous()  # [B, T-1, V]
    shift_labels = labels[:, 1:].contiguous()      # [B, T-1]

    if attention_mask is None:
        shift_mask = torch.ones_like(shift_labels, dtype=torch.float)
    else:
        shift_mask = attention_mask[:, 1:].contiguous().float()

    # Mask out ignore index (-100) as well
    valid = (shift_labels != -100).float()
    shift_mask = shift_mask * valid

    # For CrossEntropy, replace -100 with any valid label (it will be masked anyway)
    safe_labels = shift_labels.clone()
    safe_labels[safe_labels == -100] = 0

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    token_loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        safe_labels.view(-1),
    ).view_as(safe_labels)  # [B, T-1]

    token_loss = token_loss * shift_mask

    denom = shift_mask.sum(dim=1).clamp(min=1.0)
    loss_per_sample = token_loss.sum(dim=1) / denom
    return loss_per_sample


class GreenTrainer(Trainer):
    """
    GreenTrainer integrates:
      - EnergyMonitor (energy tracking + budget)
      - EnergyAwareTrainingController (loss/length sampling + adaptive batch + early stop)

    IMPORTANT:
      - This Trainer overrides get_train_dataloader() to use a BatchSampler,
        which enables dynamic batch sizes and smart sampling.
      - It computes per-sample loss for causal LM and feeds it back to the controller.
    """

    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        eval_dataset=None,
        energy_budget_wh: float = 100.0,
        base_batch_size: Optional[int] = None,
        min_batch_size: int = 1,
        max_steps_for_progress: Optional[int] = None,
        region: Optional[str] = None,
        carbon_intensity: Optional[float] = None,
        energy_monitor: Optional[EnergyMonitor] = None,
        use_smart_sampling: bool = True,
        **kwargs,
    ):
        # Transformers 5.0+ rebranded tokenizer to processing_class
        import inspect
        sig = inspect.signature(Trainer.__init__)
        trainer_kwargs = {
            "model": model,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
        }
        
        if "processing_class" in sig.parameters:
            trainer_kwargs["processing_class"] = tokenizer
        else:
            trainer_kwargs["tokenizer"] = tokenizer
            
        super().__init__(**trainer_kwargs, **kwargs)

        # Bug 7 Fix: Don't disable remove_unused_columns. The Trainer handles standard dict-based datasets cleanly.

        self.energy_budget_wh = float(energy_budget_wh)
        # Internal Monitoring & Control
        if energy_monitor is None:
            raise ValueError("GreenTrainer requires a shared EnergyMonitor instance to be provided.")
        
        self.energy_monitor = energy_monitor
        self.energy_monitor.energy_budget_wh = self.energy_budget_wh
            
        if region:
            self.energy_monitor.region = region
        if carbon_intensity:
            self.energy_monitor.carbon_intensity = carbon_intensity
        
        # Wrap dataset to inject `_index` per example
        if self.train_dataset is None:
            raise ValueError("GreenTrainer requires a train_dataset.")
        self.train_dataset = _IndexWrapper(self.train_dataset)

        # Batch sizing defaults
        if base_batch_size is None:
            base_batch_size = int(getattr(self.args, "per_device_train_batch_size", 8) or 8)

        if max_steps_for_progress is None:
            # Use HF max_steps if set; else approximate via epochs * steps is hard here
            max_steps_for_progress = int(self.args.max_steps) if self.args.max_steps and self.args.max_steps > 0 else 100000

        # Controller: sampler + adaptive batch size + early stopping
        self.controller = EnergyAwareTrainingController(
            dataset=self.train_dataset,
            energy_monitor=self.energy_monitor,
            base_batch_size=base_batch_size,
            min_batch_size=min_batch_size,
            max_steps=max_steps_for_progress,
        )

        # Keep last validation metric for plateau stopping (optional)
        self._last_val_metric: Optional[float] = None

        # When False, use standard shuffle + fixed batch size (same as baseline) for fair comparison.
        self.use_smart_sampling = bool(use_smart_sampling)

        print(f"🌱 GreenTrainer initialized | budget={self.energy_budget_wh:.2f} Wh | base_batch={base_batch_size} | smart_sampling={self.use_smart_sampling}")

    # ---------------------------
    # Dataloader: use BatchSampler or standard shuffle
    # ---------------------------
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        device = next(self.model.parameters()).device
        collator = _HFDataCollatorWithIndices(self.data_collator, device=device)
        num_workers = getattr(self.args, "dataloader_num_workers", 0)
        pin_memory = False

        if self.use_smart_sampling:
            # Importance sampling + dynamic batch size
            return DataLoader(
                self.train_dataset,
                batch_sampler=self.controller.batch_sampler,
                collate_fn=collator,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        # Same as baseline: fixed batch size, shuffle (fair comparison)
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=getattr(self.args, "dataloader_drop_last", False),
        )

    # ---------------------------
    # Training lifecycle
    # ---------------------------
    def train(self, resume_from_checkpoint=None, trial=None, **kwargs):
        self.energy_monitor.start_monitoring()
        print("⚡ Energy monitoring started.")

        try:
            result = super().train(resume_from_checkpoint=resume_from_checkpoint, trial=trial, **kwargs)
            return result
        finally:
            # stop_monitoring() returns the final EnergyMetrics snapshot.
            # We store it on self so that fine-tune.py's TieredCarbonTracker can
            # retrieve it via get_current_metrics() AFTER training completes.
            # Crucially we call stop first, THEN save — so the snapshot is frozen.
            self._final_energy_metrics = self.energy_monitor.stop_monitoring()
            self._save_energy_report(self._final_energy_metrics)

    # ---------------------------
    # Loss: compute per-sample + feedback
    # ---------------------------
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        """
        Computes mean loss for backprop (as HF expects), but also computes:
          - per-sample loss vector
          - lengths vector (token cost proxy)
          - batch indices

        Then sends feedback to controller to update sampling distribution.
        """
        # Extract indices saved by attach_indices_collate
        batch_indices = inputs.get("_indices", None)
        if batch_indices is None and self.use_smart_sampling:
            print("WARNING: _indices missing from batch — smart sampling disabled this step. "
                  "Check that remove_unused_columns=False and _HFDataCollatorWithIndices is active.")

        # One forward pass WITH labels — model computes loss natively (identical to baseline)
        clean_inputs = {k: v for k, v in inputs.items() if k != "_indices"}

        # Transformers 5.0: if the model accepts num_items_in_batch (model_accepts_loss_kwargs=True),
        # training_step will NOT divide by gradient_accumulation_steps — it expects compute_loss to
        # pass num_items_in_batch to the model so the model normalises across all accum tokens.
        # Without this, GreenTrainer returns mean loss (~2.15) and it gets accumulated raw × grad_accum.
        if num_items_in_batch is not None and getattr(self, "model_accepts_loss_kwargs", False):
            clean_inputs["num_items_in_batch"] = num_items_in_batch

        outputs = model(**clean_inputs)

        logits = outputs["logits"] if isinstance(outputs, dict) and "logits" in outputs else getattr(outputs, "logits", None)
        attention_mask = inputs.get("attention_mask", None)
        labels = clean_inputs.get("labels", None)

        # Use the model's native loss for backprop — no extra forward pass, same as baseline
        loss = outputs["loss"] if isinstance(outputs, dict) and "loss" in outputs else getattr(outputs, "loss", None)

        if self.use_smart_sampling and logits is not None and labels is not None and batch_indices is not None:
            # Per-sample losses from already-computed logits — no extra forward pass
            with torch.no_grad():
                per_sample_losses = _per_sample_causal_lm_loss(
                    logits=logits,
                    labels=labels,
                    attention_mask=attention_mask,
                )
                if attention_mask is not None:
                    lengths = attention_mask.sum(dim=1).clamp(min=1)
                else:
                    lengths = torch.full((per_sample_losses.size(0),), labels.size(1), device=labels.device, dtype=torch.long)

            self.controller.on_train_step_end(
                batch_indices=batch_indices,
                per_sample_losses=per_sample_losses,
                lengths=lengths,
                global_step=int(getattr(self.state, "global_step", 0)),
            )

        if loss is None:
            if logits is not None and labels is not None:
                per_sample_losses = _per_sample_causal_lm_loss(logits=logits, labels=labels, attention_mask=attention_mask)
                loss = per_sample_losses.mean()
            else:
                loss = torch.tensor(0.0, device=next(model.parameters()).device, requires_grad=True)

        # Inject loss back into outputs if requested (HF sometimes expects outputs.loss)
        if isinstance(outputs, dict):
            outputs["loss"] = loss
        elif hasattr(outputs, "loss"):
            outputs.loss = loss

        return (loss, outputs) if return_outputs else loss

    # ---------------------------
    # Step-level budget stop + optional plateau stop
    # ---------------------------
    def training_step(self, model, inputs, num_items_in_batch=None):
        # Hard stop if energy budget is exhausted
        if not self.energy_monitor.has_energy_remaining():
            print("⚠️ Energy budget exhausted. Stopping training.")
            self.control.should_training_stop = True
            return torch.tensor(0.0, device=next(model.parameters()).device, requires_grad=True)

        self.energy_monitor.log_step(getattr(self.state, "global_step", None))

        if self._last_val_metric is not None:
            if self.controller.early_stopper.should_stop(
                step=int(getattr(self.state, "global_step", 0)),
                val_metric=float(self._last_val_metric),
                remaining_energy_wh=float(self.energy_monitor.get_remaining_energy()),
            ):
                print("✋ Early stopping triggered.")
                self.control.should_training_stop = True

        return super().training_step(model, inputs, num_items_in_batch)

    def evaluate(self, eval_dataset=None, **kwargs):
        """
        Capture a validation metric for plateau-based early stopping.
        Preference order:
          1) args.metric_for_best_model if present in metrics
          2) eval_loss if present
        """
        metrics = super().evaluate(eval_dataset=eval_dataset, **kwargs)

        metric_name = getattr(self.args, "metric_for_best_model", None)
        if metric_name and metric_name in metrics:
            self._last_val_metric = float(metrics[metric_name])
        elif "eval_loss" in metrics:
            # For loss, lower is better; the early stopper inside controller should be configured accordingly
            self._last_val_metric = float(metrics["eval_loss"])
        else:
            self._last_val_metric = None

        return metrics

    # ---------------------------
    # Reporting
    # ---------------------------
    def _save_energy_report(self, metrics):
        """
        Save a JSON report with energy + rough CO2 estimate (replace factor if needed).
        """
        # Example global avg factor; replace with region-specific if desired
        co2_factor_kg_per_kwh = 0.4
        co2_emissions_kg = (metrics.total_energy_wh / 1000.0) * co2_factor_kg_per_kwh

        report = {
            "total_energy_wh": metrics.total_energy_wh,
            "cpu_energy_wh": getattr(metrics, "cpu_energy_wh", None),
            "gpu_energy_wh": getattr(metrics, "gpu_energy_wh", None),
            "budget_used_percent": getattr(metrics, "budget_used_percent", None),
            "remaining_energy_wh": self.energy_monitor.get_remaining_energy(),
            "co2_emissions_kg_est": co2_emissions_kg,
        }

        # If metrics is a dataclass, you can store full snapshot too
        try:
            report["metrics_raw"] = asdict(metrics)
        except Exception:
            pass

        # Save to output_dir with a unique filename (timestamp + optional rank)
        import time
        import os
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"green_training_report_{timestamp}.json"
        
        # Use output_dir from TrainingArguments if available
        output_dir = getattr(self.args, "output_dir", None) or "."
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, filename)
        
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)

        print("\n🌱 GREEN TRAINING COMPLETED")
        print(f"⚡ Energy used: {metrics.total_energy_wh:.4f} Wh")
        print(f"🌍 CO2 estimate: {co2_emissions_kg:.6f} kg")
        print(f"📄 Report saved: {out_path}")