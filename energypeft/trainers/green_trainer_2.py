# energypeft/trainers/green_trainer_2.py
#
# GreenTrainer2 — extends GreenTrainer with two additional energy-saving techniques:
#   Technique 1: Adaptive loss-threshold early exit (skip entire forward+backward for easy batches)
#   Technique 2: Gradient-accumulation early exit (force optimizer step when loss converges early)
#
# All Improvement-2 fixes are preserved:
#   - num_items_in_batch passthrough for Transformers 5.0 token-level normalization

import json
from dataclasses import asdict
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer

from ..core.energy_monitor import EnergyMonitor
from ..core.efficient_training import EnergyAwareTrainingController
from ..core.carbon_monitor import CarbonIntensityMonitor


class _HFDataCollatorWithIndices:
    def __init__(self, base_collator, device=None):
        self.base = base_collator
        self.device = device

    def __call__(self, features):
        feats = [dict(f) for f in features]
        indices = [int(f.get("_index", -1)) for f in feats]
        for f in feats:
            f.pop("_index", None)

        batch = self.base(feats)
        batch["_indices"] = indices

        # Move tensors to model device for both MPS and CUDA.
        # Without this, tensors stay on CPU after the collator when using a
        # custom batch_sampler DataLoader, and Accelerate's auto-placement may
        # not fire reliably in T5.0 with a custom BatchSampler.
        if self.device is not None and (
            str(self.device).startswith("mps") or str(self.device).startswith("cuda")
        ):
            batch = {
                k: v.to(self.device) if hasattr(v, "to") else v
                for k, v in batch.items()
            }
        return batch


class _IndexWrapper(Dataset):
    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.base[idx]
        if not isinstance(item, dict):
            raise TypeError("Dataset items must be dict-like for GreenTrainer2.")
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
    Returns: loss_per_sample [B]
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    if attention_mask is None:
        shift_mask = torch.ones_like(shift_labels, dtype=torch.float)
    else:
        shift_mask = attention_mask[:, 1:].contiguous().float()

    valid = (shift_labels != -100).float()
    shift_mask = shift_mask * valid

    safe_labels = shift_labels.clone()
    safe_labels[safe_labels == -100] = 0

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    token_loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        safe_labels.view(-1),
    ).view_as(safe_labels)

    token_loss = token_loss * shift_mask

    denom = shift_mask.sum(dim=1).clamp(min=1.0)
    return token_loss.sum(dim=1) / denom


class GreenTrainer2(Trainer):
    """
    GreenTrainer2 extends GreenTrainer (Improvement-2) with two energy-saving techniques:

    Technique 1 — Adaptive loss-threshold early exit (training_step level):
        Maintains a slow EMA of batch losses. Before each micro-batch, checks whether
        the PREVIOUS step's loss was below (ema * skip_fraction). If so, skips the
        ENTIRE micro-batch (forward + backward), saving ~100% of that step's compute.

        The skip_fraction anneals from loss_threshold_start to loss_threshold_end over
        training. Both are fractions of the running EMA (e.g., 0.85 = skip if loss is
        below 85% of running average). This adapts to any loss scale automatically.

    Technique 2 — Gradient-accumulation early exit:
        During grad_accum micro-batches, track running average loss.
        Force an optimizer step early if the running average drops below
        (loss_ema * convergence_threshold). convergence_threshold is an EMA
        fraction (e.g. 0.90 = fire when window average < 90% of running EMA).
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
        # Technique 1 — EMA-relative fractions
        # skip_fraction = loss_threshold_start (anneals to loss_threshold_end over training)
        # A micro-batch is skipped when: prev_loss < ema_loss * skip_fraction
        # e.g. 0.85 = skip if loss is below 85% of running average
        use_loss_threshold: bool = True,
        loss_threshold_start: float = 0.85,
        loss_threshold_end: float = 0.70,
        # Technique 2
        # convergence_threshold: EMA fraction — fire early exit when window
        # running average < loss_ema * convergence_threshold.
        # e.g. 0.90 = fire when window average is below 90% of the running EMA.
        use_early_accum_exit: bool = True,
        convergence_threshold: float = 0.90,
        # Carbon-aware scaling: dynamically adjusts T1/T2 thresholds based on
        # real-time grid carbon intensity (via CarbonIntensityMonitor).
        # High carbon intensity → aggressiveness_factor > 1.0 → more skipping.
        # Low carbon intensity  → aggressiveness_factor < 1.0 → fewer skips.
        carbon_monitor: Optional[CarbonIntensityMonitor] = None,
        carbon_update_interval: int = 50,   # re-read carbon factor every N steps
        **kwargs,
    ):
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

        self.energy_budget_wh = float(energy_budget_wh)

        if energy_monitor is None:
            raise ValueError("GreenTrainer2 requires a shared EnergyMonitor instance.")

        self.energy_monitor = energy_monitor
        self.energy_monitor.energy_budget_wh = self.energy_budget_wh

        if region:
            self.energy_monitor.region = region
        if carbon_intensity:
            self.energy_monitor.carbon_intensity = carbon_intensity

        if self.train_dataset is None:
            raise ValueError("GreenTrainer2 requires a train_dataset.")
        self.train_dataset = _IndexWrapper(self.train_dataset)

        if base_batch_size is None:
            base_batch_size = int(getattr(self.args, "per_device_train_batch_size", 8) or 8)

        if max_steps_for_progress is None:
            max_steps_for_progress = (
                int(self.args.max_steps)
                if self.args.max_steps and self.args.max_steps > 0
                else 100000
            )

        self.controller = EnergyAwareTrainingController(
            dataset=self.train_dataset,
            energy_monitor=self.energy_monitor,
            base_batch_size=base_batch_size,
            min_batch_size=min_batch_size,
            max_steps=max_steps_for_progress,
        )

        self._last_val_metric: Optional[float] = None
        self.use_smart_sampling = bool(use_smart_sampling)

        # Technique 1 — EMA-relative adaptive threshold
        self.use_loss_threshold = bool(use_loss_threshold)
        self.loss_threshold_start = float(loss_threshold_start)
        self.loss_threshold_end = float(loss_threshold_end)
        # _last_step_loss: natural-scale loss from previous non-skipped step
        # _loss_ema: slow exponential moving average of natural-scale batch losses
        self._loss_ema: Optional[float] = None
        self._ema_alpha: float = 0.05  # ≈20-step memory; slow to avoid noise

        # Technique 2 — gradient-accumulation early exit
        self.use_early_accum_exit = bool(use_early_accum_exit)
        self.convergence_threshold = float(convergence_threshold)
        self._accum_losses = []

        # Per-window tracking: ensures the FIRST micro-batch of every grad_accum
        # window always runs so each optimizer step gets at least one real gradient.
        # Without this, T1 can skip all 8 micro-batches in a window → grad_norm=0.
        self._real_microbatches_this_window = 0
        self._last_global_step_seen = -1

        # Carbon-aware scaling
        self.carbon_monitor = carbon_monitor
        self.carbon_update_interval = max(1, int(carbon_update_interval))
        # _carbon_aggressiveness: multiplier applied to T1 skip_fraction and T2 threshold.
        # Starts at 1.0 (neutral); updated every carbon_update_interval steps.
        self._carbon_aggressiveness: float = (
            carbon_monitor.get_aggressiveness_factor()
            if carbon_monitor is not None
            else 1.0
        )

        # Metrics
        self._skipped_samples = 0
        self._total_samples_seen = 0
        self._skipped_steps = 0
        self._early_accum_exits = 0
        self._micro_batches_saved = 0

        print(
            f"🌱 GreenTrainer2 initialized | budget={self.energy_budget_wh:.2f} Wh "
            f"| base_batch={base_batch_size} | smart_sampling={self.use_smart_sampling} "
            f"| loss_threshold={self.use_loss_threshold} (EMA fractions "
            f"{self.loss_threshold_start:.2f}→{self.loss_threshold_end:.2f}) "
            f"| early_accum_exit={self.use_early_accum_exit}"
        )

    # ---------------------------
    # Dataloader
    # ---------------------------
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        device = next(self.model.parameters()).device
        collator = _HFDataCollatorWithIndices(self.data_collator, device=device)
        num_workers = getattr(self.args, "dataloader_num_workers", 0)
        pin_memory = False

        if self.use_smart_sampling:
            return DataLoader(
                self.train_dataset,
                batch_sampler=self.controller.batch_sampler,
                collate_fn=collator,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
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

        if self.carbon_monitor is not None:
            self.carbon_monitor.start_background_polling()
            print(
                f"🌍 Carbon-aware training enabled | zone={self.carbon_monitor.zone} | "
                f"intensity={self.carbon_monitor.get_current_intensity():.1f} g CO\u2082/kWh | "
                f"initial aggressiveness={self._carbon_aggressiveness:.2f}x"
            )

        try:
            result = super().train(
                resume_from_checkpoint=resume_from_checkpoint,
                trial=trial,
                **kwargs,
            )
            return result
        finally:
            if self.carbon_monitor is not None:
                self.carbon_monitor.stop_background_polling()
            self._final_energy_metrics = self.energy_monitor.stop_monitoring()
            self._save_energy_report(self._final_energy_metrics)

    # ---------------------------
    # compute_loss: T5.0 fix + single per-sample computation for tracker
    # ---------------------------
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        """
        Computes loss for backprop.

        Technique 1 is applied in training_step BEFORE this method is called,
        so skipped micro-batches never reach compute_loss — the forward pass
        is avoided entirely for easy batches.

        Per-sample losses are computed once (no_grad) for tracker feedback only.
        """
        batch_indices = inputs.get("_indices", None)
        if batch_indices is None and self.use_smart_sampling:
            print(
                "WARNING: _indices missing from batch — smart sampling disabled this step. "
                "Check that remove_unused_columns=False and _HFDataCollatorWithIndices is active."
            )

        clean_inputs = {k: v for k, v in inputs.items() if k != "_indices"}

        # Transformers 5.0: pass num_items_in_batch to model when it accepts it,
        # so training_step (which skips the /grad_accum division) gets a correctly
        # normalised loss.
        if num_items_in_batch is not None and getattr(self, "model_accepts_loss_kwargs", False):
            clean_inputs["num_items_in_batch"] = num_items_in_batch

        outputs = model(**clean_inputs)

        logits = (
            outputs["logits"]
            if isinstance(outputs, dict) and "logits" in outputs
            else getattr(outputs, "logits", None)
        )
        attention_mask = inputs.get("attention_mask", None)
        labels = clean_inputs.get("labels", None)

        loss = (
            outputs["loss"]
            if isinstance(outputs, dict) and "loss" in outputs
            else getattr(outputs, "loss", None)
        )

        # ── Single per-sample computation (no_grad) for tracker feedback ────
        if logits is not None and labels is not None and self.use_smart_sampling and batch_indices is not None:
            with torch.no_grad():
                per_sample = _per_sample_causal_lm_loss(logits, labels, attention_mask)
                if attention_mask is not None:
                    lengths = attention_mask.sum(dim=1).clamp(min=1)
                else:
                    lengths = torch.full(
                        (per_sample.size(0),),
                        labels.size(1),
                        device=labels.device,
                        dtype=torch.long,
                    )
            self.controller.on_train_step_end(
                batch_indices=batch_indices,
                per_sample_losses=per_sample,
                lengths=lengths,
                global_step=int(getattr(self.state, "global_step", 0)),
            )

        if loss is None:
            if logits is not None and labels is not None:
                per_sample_grad = _per_sample_causal_lm_loss(logits, labels, attention_mask)
                loss = per_sample_grad.mean()
            else:
                loss = torch.tensor(
                    0.0,
                    device=next(model.parameters()).device,
                    requires_grad=True,
                )

        if isinstance(outputs, dict):
            outputs["loss"] = loss
        elif hasattr(outputs, "loss"):
            outputs.loss = loss

        return (loss, outputs) if return_outputs else loss

    # ---------------------------
    # training_step: Technique 1 + Technique 2
    # ---------------------------
    def training_step(self, model, inputs, num_items_in_batch=None):
        # Hard stop if energy budget is exhausted
        if not self.energy_monitor.has_energy_remaining():
            print("⚠️ Energy budget exhausted. Stopping training.")
            self.control.should_training_stop = True
            return torch.tensor(
                0.0,
                device=next(model.parameters()).device,
                requires_grad=True,
            )

        self.energy_monitor.log_step(getattr(self.state, "global_step", None))

        # Carbon-aware: periodically refresh the aggressiveness factor so T1/T2
        # thresholds adapt to real-time grid carbon intensity.
        current_step = getattr(self.state, "global_step", 0)
        if (
            self.carbon_monitor is not None
            and current_step % self.carbon_update_interval == 0
        ):
            self._carbon_aggressiveness = self.carbon_monitor.get_aggressiveness_factor()

        if self._last_val_metric is not None:
            if self.controller.early_stopper.should_stop(
                step=int(getattr(self.state, "global_step", 0)),
                val_metric=float(self._last_val_metric),
                remaining_energy_wh=float(self.energy_monitor.get_remaining_energy()),
            ):
                print("✋ Early stopping triggered.")
                self.control.should_training_stop = True

        # ── Technique 1: skip entire micro-batch (forward+backward) ─────────
        # Uses EMA-relative threshold so it adapts to the actual loss scale:
        #   threshold = _loss_ema * skip_fraction
        # where skip_fraction anneals from loss_threshold_start to loss_threshold_end.
        # A batch is skipped when its loss was below (EMA * fraction), i.e. it is
        # an "easy" batch that contributes little to further learning.
        #
        # IMPORTANT: we track which micro-batch position we are in the current
        # grad_accum window. The FIRST micro-batch of every window always runs so
        # each optimizer step gets at least one real gradient update (grad_norm > 0).
        # global_step is constant across all micro-batches within a single window
        # and increments only after the optimizer steps.
        current_global_step = getattr(self.state, "global_step", 0)
        if current_global_step != self._last_global_step_seen:
            self._real_microbatches_this_window = 0
            self._last_global_step_seen = current_global_step

        if (
            self.use_loss_threshold
            and hasattr(self, "_last_step_loss")
            and self._loss_ema is not None
            and self._real_microbatches_this_window >= 1  # never skip the first in a window
        ):
            progress = getattr(self.state, "global_step", 0) / max(
                self.controller.max_steps, 1
            )
            skip_fraction = (
                self.loss_threshold_start * (1.0 - progress)
                + self.loss_threshold_end * progress
            )
            # Carbon scaling: dirty grid → higher skip_fraction → more micro-batches skipped.
            # Cap at 0.99 so we never guarantee a skip even when grid is very dirty.
            skip_fraction = min(skip_fraction * self._carbon_aggressiveness, 0.99)
            threshold = self._loss_ema * skip_fraction
            if self._last_step_loss < threshold:
                self._skipped_steps += 1
                batch_size = next(
                    (v.size(0) for v in inputs.values() if hasattr(v, "size") and v.dim() > 0),
                    1,
                )
                self._skipped_samples += batch_size
                self._total_samples_seen += batch_size
                # Return the cached loss at HF-normalised scale (loss/grad_accum)
                # instead of 0.0 so the trainer's logged train_loss stays meaningful
                # and is comparable to baseline. Using 0.0 would deflate the reported
                # average because HF sums all returned values across all steps.
                grad_accum = self.args.gradient_accumulation_steps
                cached_normalised = self._last_step_loss / max(grad_accum, 1)
                return torch.tensor(
                    cached_normalised,
                    device=next(model.parameters()).device,
                    dtype=torch.float32,
                )

        loss = super().training_step(model, inputs, num_items_in_batch)

        # Cache loss at natural scale and update EMA for next step's T1 decision.
        # super().training_step() returns loss/grad_accum for T5.0 (Qwen2.5).
        # Multiply back to natural scale (~2.15) so EMA and threshold are meaningful.
        if self.use_loss_threshold:
            self._real_microbatches_this_window += 1  # this window now has a real gradient
            batch_size = next(
                (v.size(0) for v in inputs.values() if hasattr(v, "size") and v.dim() > 0),
                1,
            )
            self._total_samples_seen += batch_size
            grad_accum = self.args.gradient_accumulation_steps
            natural_loss = (loss.item() if hasattr(loss, "item") else float(loss)) * grad_accum
            self._last_step_loss = natural_loss
            if self._loss_ema is None:
                self._loss_ema = natural_loss
            else:
                self._loss_ema = (
                    self._ema_alpha * natural_loss
                    + (1.0 - self._ema_alpha) * self._loss_ema
                )

        # ── Technique 2: gradient-accumulation early exit ────────────────────
        if self.use_early_accum_exit:
            if not hasattr(self, "_accum_losses"):
                self._accum_losses = []

            grad_accum = self.args.gradient_accumulation_steps
            # Multiply back to natural scale for comparison with convergence_threshold.
            raw_loss = (loss.item() if hasattr(loss, "item") else float(loss)) * grad_accum
            self._accum_losses.append(raw_loss)
            micro_step = len(self._accum_losses)

            window_avg = sum(self._accum_losses) / micro_step
            # convergence_threshold is an EMA fraction, not an absolute value.
            # Fires when the window's running average drops below
            # (EMA * convergence_threshold), i.e. the window is significantly
            # easier than the long-run training average.
            # Carbon scaling: dirty grid → higher threshold → T2 fires more readily.
            ema_threshold = (
                self._loss_ema * self.convergence_threshold * self._carbon_aggressiveness
                if self._loss_ema is not None else float("inf")
            )
            if (
                micro_step >= 2
                and micro_step < grad_accum
                and window_avg < ema_threshold
            ):
                self._early_accum_exits = getattr(self, "_early_accum_exits", 0) + 1
                self._micro_batches_saved = (
                    getattr(self, "_micro_batches_saved", 0) + (grad_accum - micro_step)
                )
                self._accum_losses = []
                # Signal Accelerate to treat this as the last micro-batch in the
                # accumulation window so it syncs gradients and runs the optimizer.
                try:
                    self.accelerator.gradient_state._remainder = 0
                except Exception:
                    pass

            elif micro_step >= grad_accum:
                self._accum_losses = []

        return loss

    # ---------------------------
    # Evaluation (plateau early stop)
    # ---------------------------
    def evaluate(self, eval_dataset=None, **kwargs):
        metrics = super().evaluate(eval_dataset=eval_dataset, **kwargs)

        metric_name = getattr(self.args, "metric_for_best_model", None)
        if metric_name and metric_name in metrics:
            self._last_val_metric = float(metrics[metric_name])
        elif "eval_loss" in metrics:
            self._last_val_metric = float(metrics["eval_loss"])
        else:
            self._last_val_metric = None

        return metrics

    # ---------------------------
    # Reporting
    # ---------------------------
    def _save_energy_report(self, metrics):
        co2_factor_kg_per_kwh = 0.4
        co2_emissions_kg = (metrics.total_energy_wh / 1000.0) * co2_factor_kg_per_kwh

        skipped_samples = getattr(self, "_skipped_samples", 0)
        total_samples = getattr(self, "_total_samples_seen", 0)

        report = {
            "total_energy_wh": metrics.total_energy_wh,
            "cpu_energy_wh": getattr(metrics, "cpu_energy_wh", None),
            "gpu_energy_wh": getattr(metrics, "gpu_energy_wh", None),
            "budget_used_percent": getattr(metrics, "budget_used_percent", None),
            "remaining_energy_wh": self.energy_monitor.get_remaining_energy(),
            "co2_emissions_kg_est": co2_emissions_kg,
            # Technique 1 metrics
            "skipped_steps": getattr(self, "_skipped_steps", 0),
            "skipped_samples": skipped_samples,
            "total_samples_seen": total_samples,
            "skip_rate_pct": round(
                100 * skipped_samples / max(total_samples, 1), 2
            ),
            "loss_ema_final": round(self._loss_ema, 4) if self._loss_ema is not None else None,
            # Technique 2 metrics
            "early_accum_exits": getattr(self, "_early_accum_exits", 0),
            "micro_batches_saved": getattr(self, "_micro_batches_saved", 0),
        }

        try:
            report["metrics_raw"] = asdict(metrics)
        except Exception:
            pass

        if self.carbon_monitor is not None:
            report["carbon_monitor"] = self.carbon_monitor.summary()

        import time
        import os

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        # Use "detail" (not "report") so compare_2.py's *_report_*.json glob
        # does NOT pick this file up and instead reads the unified green_run_report
        # saved by fine-tune_2.py, which has time_seconds_wall, gpu_power_w,
        # co2_kg_operational_x_pue, and the other fields needed for comparison.
        filename = f"green_training_detail_{timestamp}.json"

        output_dir = getattr(self.args, "output_dir", None) or "."
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, filename)

        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)

        print("\n🌱 GREEN TRAINING COMPLETED (GreenTrainer2)")
        print(f"⚡ Energy used: {metrics.total_energy_wh:.4f} Wh")
        print(f"🌍 CO2 estimate: {co2_emissions_kg:.6f} kg")
        print(f"⏭️  Skipped steps: {report['skipped_steps']} | Skip rate: {report['skip_rate_pct']}%")
        print(f"📉 Loss EMA final: {report['loss_ema_final']}")
        print(f"⚡ Early accum exits: {report['early_accum_exits']} | Micro-batches saved: {report['micro_batches_saved']}")
        print(f"📄 Detail saved: {out_path}")
