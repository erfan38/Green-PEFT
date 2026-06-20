#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fine-tune.py

Run ONE fine-tuning job per invocation:
  python fine-tune.py --mode green
  python fine-tune.py --mode baseline

Goals:
- Same config for both modes (model/dataset/LoRA/training args).
- Timestamped output directory per run (never overwrite).
- Robust energy + carbon + time tracking (same monitoring pattern for both modes):
  Priority:
    1) CodeCarbon (if installed and enabled)
    2) energypeft.core.energy_monitor.EnergyMonitor
    3) Proxy (GPU-hours × TDP) if measurement unavailable
- Unified JSON report schema written by this script for BOTH modes.

Important:
- For FIXED-WORK comparisons, ensure green does NOT stop early:
  set --energy_budget_wh very large (e.g., 10000) so budget never triggers.
- If your GreenTrainer still starts/stops its own EnergyMonitor internally,
  you should disable that in GreenTrainer (recommended) to avoid double-monitor overhead.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, List

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)

from energypeft import GreenTrainer  # your GreenTrainer class
from energypeft.core.energy_monitor import EnergyMonitor


# =========================
# Utilities
# =========================
def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, payload: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def hours(seconds: float) -> float:
    return max(0.0, seconds) / 3600.0


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# =========================
# Carbon tracker (tiered)
# =========================
@dataclass
class CarbonConfig:
    carbon_intensity_g_per_kwh: float
    pue: float
    include_embodied: bool
    gpu_tdp_w: float
    embodied_gpu_kg_per_gpu_hour: float
    embodied_server_kg_per_server_hour: float


@dataclass
class CarbonResult:
    method: str  # "codecarbon" | "energymonitor" | "tdp_proxy"
    energy_kwh: float
    energy_wh: float
    gpu_energy_wh: float
    cpu_energy_wh: float
    current_power_w: float
    gpu_power_w: float
    cpu_power_w: float
    budget_used_percent: float
    timestamp_wall: float
    timestamp_mono: float
    co2_kg_operational: float
    co2_kg_pue_adjusted: float
    co2_kg_embodied: float
    co2_kg_total_pue_plus_embodied: float
    notes: str


class TieredCarbonTracker:
    """
    Priority:
      1) CodeCarbon (if installed) -> CO2 (kg) and sometimes energy (kWh)
      2) EnergyMonitor -> energy (Wh), CO2 (kg) optional
      3) TDP proxy -> GPU-hours × TDP
    """

    def __init__(self, cfg: CarbonConfig, allow_codecarbon: bool = True, energy_monitor: Optional[EnergyMonitor] = None, manage_monitor: bool = True):
        self.cfg = cfg
        self.allow_codecarbon = allow_codecarbon

        self._start_time = 0.0
        self._end_time = 0.0

        self._energy_monitor = energy_monitor
        self.manage_monitor = manage_monitor
        self._cc_tracker = None
        self._cc_energy_kwh = 0.0
        self._cc_co2_kg = 0.0

        self._em_notes = ""

    # ---------- EnergyMonitor compatibility layer ----------
    @staticmethod
    def _em_start(m: Any) -> bool:
        for fn in ("start_monitoring", "start", "begin"):
            if hasattr(m, fn):
                getattr(m, fn)()
                return True
        return False

    @staticmethod
    def _em_extract(m: Any, stop: bool = True) -> Tuple[float, float, str, Dict[str, float]]:
        """
        Returns (energy_wh, co2_kg, notes, detailed_metrics)
        Supports:
          - stop_monitoring()/stop()/end()/finish() returning dict or object
          - to_dict()
        """
        energy_wh = 0.0
        co2_kg = 0.0
        notes = ""
        detailed = {
            "gpu_energy_wh": 0.0,
            "cpu_energy_wh": 0.0,
            "current_power_w": 0.0,
            "gpu_power_w": 0.0,
            "cpu_power_w": 0.0,
            "budget_used_percent": 0.0,
            "timestamp_wall": 0.0,
            "timestamp_mono": 0.0,
        }

        result = None
        if stop:
            for fn in ("stop_monitoring", "stop", "end", "finish"):
                if hasattr(m, fn):
                    try:
                        result = getattr(m, fn)()
                    except Exception:
                        result = None
                    break
        else:
            if hasattr(m, "get_current_metrics"):
                result = m.get_current_metrics()

        if result is not None:
            if hasattr(result, "total_energy_wh"):
                energy_wh = safe_float(getattr(result, "total_energy_wh", 0.0), 0.0)
            if hasattr(result, "co2_emissions_kg_est"):
                co2_kg = safe_float(getattr(result, "co2_emissions_kg_est", 0.0), 0.0)

            for k in detailed.keys():
                if hasattr(result, k):
                    detailed[k] = safe_float(getattr(result, k, 0.0), 0.0)

            notes = "EnergyMonitor: stop_* returned object."
            if energy_wh > 0.0 or co2_kg > 0.0:
                return energy_wh, co2_kg, notes, detailed

        if hasattr(m, "to_dict"):
            try:
                d = m.to_dict()
                if isinstance(d, dict):
                    energy_wh = safe_float(d.get("total_energy_wh", 0.0), 0.0)
                    co2_kg = safe_float(d.get("co2_emissions_kg_est", 0.0), 0.0)
                    for k in detailed.keys():
                        detailed[k] = safe_float(d.get(k, 0.0), 0.0)
                    notes = "EnergyMonitor: to_dict() used."
                    return energy_wh, co2_kg, notes, detailed
            except Exception:
                pass

        return (
            0.0,
            0.0,
            "EnergyMonitor: no metrics extracted (API unsupported or platform power unavailable).",
            detailed,
        )

    # ---------- CodeCarbon helper ----------
    @staticmethod
    def _codecarbon_energy_kwh_from_tracker(tracker: Any) -> float:
        for attr in ("_last_emissions", "final_emissions_data", "_emissions"):
            try:
                data = getattr(tracker, attr, None)
                if data is None:
                    continue
                if hasattr(data, "energy_consumed"):
                    return safe_float(getattr(data, "energy_consumed"), 0.0)
                if isinstance(data, dict) and "energy_consumed" in data:
                    return safe_float(data.get("energy_consumed"), 0.0)
            except Exception:
                continue
        return 0.0

    def start(self) -> None:
        self._start_time = time.time()

        # CodeCarbon (optional)
        self._cc_tracker = None
        if self.allow_codecarbon:
            try:
                from codecarbon import EmissionsTracker  # type: ignore

                self._cc_tracker = EmissionsTracker(
                    measure_power_secs=5,
                    save_to_file=False,
                    log_level="error",
                )
                self._cc_tracker.start()
            except Exception:
                self._cc_tracker = None

        # EnergyMonitor (attempt)
        if self.manage_monitor:
            if self._energy_monitor is None:
                try:
                    self._energy_monitor = EnergyMonitor(energy_budget_wh=10_000.0, cpu_backend="auto")
                except Exception:
                    self._energy_monitor = None
                    self._em_notes = "EnergyMonitor: failed to initialize."
            
            if self._energy_monitor is not None:
                ok = self._em_start(self._energy_monitor)
                if not ok:
                    self._em_notes = "EnergyMonitor: no start method found."

    def stop(self, gpu_count: int = 1, server_count: int = 1) -> CarbonResult:
        self._end_time = time.time()
        runtime_s = max(0.0, self._end_time - self._start_time)

        # Stop CodeCarbon
        cc_energy_kwh = 0.0
        cc_co2_kg = 0.0
        if self._cc_tracker is not None:
            try:
                cc_co2_kg = safe_float(self._cc_tracker.stop(), 0.0)
            except Exception:
                cc_co2_kg = 0.0
            try:
                cc_energy_kwh = self._codecarbon_energy_kwh_from_tracker(self._cc_tracker)
            except Exception:
                cc_energy_kwh = 0.0

        # Stop EnergyMonitor
        em_energy_wh = 0.0
        em_co2_kg = 0.0
        detailed_em_metrics: Dict[str, float] = {}
        em_notes = self._em_notes

        # Green mode: GreenTrainer pre-stopped the monitor and injected the frozen snapshot.
        injected = getattr(self, "_injected_em_metrics", None)
        if injected is not None:
            em_energy_wh = safe_float(getattr(injected, "total_energy_wh", 0.0), 0.0)
            em_notes = "EnergyMonitor: metrics injected from GreenTrainer._final_energy_metrics."
            detailed_em_metrics = {
                "gpu_energy_wh": safe_float(getattr(injected, "gpu_energy_wh", 0.0), 0.0),
                "cpu_energy_wh": safe_float(getattr(injected, "cpu_energy_wh", 0.0), 0.0),
                "current_power_w": safe_float(getattr(injected, "current_power_w", 0.0), 0.0),
                "gpu_power_w": safe_float(getattr(injected, "gpu_power_w", 0.0), 0.0),
                "cpu_power_w": safe_float(getattr(injected, "cpu_power_w", 0.0), 0.0),
                "budget_used_percent": safe_float(getattr(injected, "budget_used_percent", 0.0), 0.0),
                "timestamp_wall": safe_float(getattr(injected, "timestamp_wall", 0.0), 0.0),
                "timestamp_mono": safe_float(getattr(injected, "timestamp_mono", 0.0), 0.0),
            }
        elif self._energy_monitor is not None:
            try:
                ewh, ckg, note, detailed = self._em_extract(self._energy_monitor, stop=self.manage_monitor)
                em_energy_wh = ewh
                em_co2_kg = ckg
                em_notes = note
                detailed_em_metrics = detailed
            except Exception:
                em_energy_wh = 0.0
                em_co2_kg = 0.0
                em_notes = "EnergyMonitor: stop/extract failed."

        # Choose best channel
        if (cc_energy_kwh > 0.0) or (cc_co2_kg > 0.0):
            energy_kwh = cc_energy_kwh
            if cc_co2_kg > 0.0:
                co2_oper = cc_co2_kg
                notes = "CodeCarbon used (CO2 reported by CodeCarbon)."
            else:
                co2_oper = self._co2_from_energy(energy_kwh)
                notes = "CodeCarbon energy used; CO2 computed from provided carbon intensity."
            method = "codecarbon"

        elif em_energy_wh > 0.0:
            energy_kwh = em_energy_wh / 1000.0
            if em_co2_kg > 0.0:
                co2_oper = em_co2_kg
                notes = f"EnergyMonitor used (energy+CO2). {em_notes}"
            else:
                co2_oper = self._co2_from_energy(energy_kwh)
                notes = f"EnergyMonitor used (energy). CO2 computed from carbon intensity. {em_notes}"
            method = "energymonitor"

        else:
            gpu_hours = hours(runtime_s) * max(1, gpu_count)
            energy_kwh = gpu_hours * (self.cfg.gpu_tdp_w / 1000.0)
            co2_oper = self._co2_from_energy(energy_kwh)
            notes = f"TDP proxy used: GPU_hours × TDP. EnergyMonitor notes: {em_notes or 'n/a'}"
            method = "tdp_proxy"

        # PUE-adjusted operational
        co2_pue = co2_oper * max(1.0, self.cfg.pue)

        # Embodied (optional)
        co2_emb = 0.0
        if self.cfg.include_embodied:
            run_h = hours(runtime_s)
            co2_emb = (
                run_h * max(1, gpu_count) * max(0.0, self.cfg.embodied_gpu_kg_per_gpu_hour)
                + run_h * max(1, server_count) * max(0.0, self.cfg.embodied_server_kg_per_server_hour)
            )

        total = co2_pue + co2_emb

        return CarbonResult(
            method=method,
            energy_kwh=energy_kwh,
            energy_wh=energy_kwh * 1000.0,
            gpu_energy_wh=detailed_em_metrics.get("gpu_energy_wh", 0.0),
            cpu_energy_wh=detailed_em_metrics.get("cpu_energy_wh", 0.0),
            current_power_w=detailed_em_metrics.get("current_power_w", 0.0),
            gpu_power_w=detailed_em_metrics.get("gpu_power_w", 0.0),
            cpu_power_w=detailed_em_metrics.get("cpu_power_w", 0.0),
            budget_used_percent=detailed_em_metrics.get("budget_used_percent", 0.0),
            timestamp_wall=detailed_em_metrics.get("timestamp_wall", 0.0),
            timestamp_mono=detailed_em_metrics.get("timestamp_mono", 0.0),
            co2_kg_operational=co2_oper,
            co2_kg_pue_adjusted=co2_pue,
            co2_kg_embodied=co2_emb,
            co2_kg_total_pue_plus_embodied=total,
            notes=notes,
        )

    def _co2_from_energy(self, energy_kwh: float) -> float:
        intensity_kg_per_kwh = self.cfg.carbon_intensity_g_per_kwh / 1000.0
        return max(0.0, energy_kwh) * max(0.0, intensity_kg_per_kwh)


# =========================
# Training pipeline
# =========================
def build_model_and_tokenizer(
    model_name: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: List[str],
    device: str,
) -> Tuple[Any, Any]:
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # dtype selection: conservative for MPS
    if device == "cuda":
        dtype = torch.float16
    elif device == "mps":
        # MPS can be sensitive; float16 is faster but may fail for some ops/models.
        # Use float16 if you know your model supports it; otherwise switch to float32.
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)

    # Move model to device (avoid device_map="auto" on MPS)
    model.to(device)

    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_cfg)
    return model, tok


def prepare_dataset(
    tokenizer: Any,
    dataset_name: str,
    split: str,
    max_length: int,
) -> Any:
    ds = load_dataset(dataset_name, split=split)

    def fmt(ex):
        out = tokenizer(
            ex["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        out["labels"] = out["input_ids"].copy()
        return out

    tokenized = ds.map(fmt, remove_columns=ds.column_names)
    tokenized.set_format("torch")
    return tokenized


def build_baseline_trainer(
    model: Any,
    tok: Any,
    args: TrainingArguments,
    train_dataset: Any,
    collator: Any,
) -> Trainer:
    sig = inspect.signature(Trainer.__init__)
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "data_collator": collator,
    }
    if "processing_class" in sig.parameters:
        trainer_kwargs["processing_class"] = tok
    else:
        trainer_kwargs["tokenizer"] = tok
    return Trainer(**trainer_kwargs)


def run_training(
    mode: str,
    output_root: str,
    model_name: str,
    dataset_name: str,
    split: str,
    max_steps: int,
    num_train_epochs: float,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    logging_steps: int,
    save_steps: int,
    seed: int,
    max_length: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: List[str],
    energy_budget_wh: float,
    carbon_intensity_g_per_kwh: float,
    pue: float,
    include_embodied: bool,
    gpu_tdp_w: float,
    embodied_gpu_kg_per_gpu_hour: float,
    embodied_server_kg_per_server_hour: float,
    allow_codecarbon: bool,
    use_smart_sampling: bool = True,
) -> str:
    tag = now_tag()
    run_dir = os.path.join(output_root, f"{mode}_{tag}")
    ensure_dir(run_dir)
    set_seed(seed)

    device = get_device()

    model, tok = build_model_and_tokenizer(
        model_name=model_name,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        device=device,
    )

    tokenized = prepare_dataset(
        tokenizer=tok,
        dataset_name=dataset_name,
        split=split,
        max_length=max_length,
    )

    # Both modes use the same precision for a fair comparison.
    # Green disables remove_unused_columns so _index survives to compute_loss.
    use_fp16 = False
    args = TrainingArguments(
        output_dir=run_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        max_steps=max_steps,
        num_train_epochs=num_train_epochs,
        report_to="none",
        logging_strategy="steps",
        save_strategy="steps",
        save_steps=save_steps,
        seed=seed,
        data_seed=seed,
        fp16=use_fp16,
        bf16=False,
        remove_unused_columns=(mode == "baseline"),
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    # Tracker config (same for both modes)
    c_cfg = CarbonConfig(
        carbon_intensity_g_per_kwh=carbon_intensity_g_per_kwh,
        pue=pue,
        include_embodied=include_embodied,
        gpu_tdp_w=gpu_tdp_w,
        embodied_gpu_kg_per_gpu_hour=embodied_gpu_kg_per_gpu_hour,
        embodied_server_kg_per_server_hour=embodied_server_kg_per_server_hour,
    )
    # Shared Energy Monitor
    shared_energy_monitor = None
    try:
        em_budget = energy_budget_wh if mode == "green" else 10_000.0
        # On Mac, powermetrics requires passwordless sudo and silently produces 0W
        # when it fails. Force the tdp backend so we always get non-zero readings
        # based on cpu_percent * TDP. This is a proxy but consistent across both modes.
        import platform as _platform
        cpu_backend = "tdp" if _platform.system() == "Darwin" else "auto"
        shared_energy_monitor = EnergyMonitor(energy_budget_wh=em_budget, cpu_backend=cpu_backend)
    except Exception as e:
        print(f"Warning: Failed to initialize single EnergyMonitor: {e}")

    # Let GreenTrainer manage its monitor to avoid double tracking startup
    manage_em = (mode == "baseline")
    tracker = TieredCarbonTracker(cfg=c_cfg, allow_codecarbon=allow_codecarbon, energy_monitor=shared_energy_monitor, manage_monitor=manage_em)

    # Trainer selection
    if mode == "green":
        # Compute actual number of optimizer steps so the progress-based
        # adaptive batch controller works correctly (not stuck at 0.00047).
        steps_per_epoch = len(tokenized) // (per_device_train_batch_size * gradient_accumulation_steps)
        if max_steps and max_steps > 0:
            actual_max_steps = max_steps
        else:
            actual_max_steps = max(1, int(steps_per_epoch * num_train_epochs))

        trainer = GreenTrainer(
            model=model,
            tokenizer=tok,
            args=args,
            train_dataset=tokenized,
            data_collator=collator,
            energy_budget_wh=energy_budget_wh,
            base_batch_size=per_device_train_batch_size,
            min_batch_size=1,
            energy_monitor=shared_energy_monitor,
            use_smart_sampling=use_smart_sampling,
            max_steps_for_progress=actual_max_steps,
        )
    elif mode == "baseline":
        trainer = build_baseline_trainer(
            model=model,
            tok=tok,
            args=args,
            train_dataset=tokenized,
            collator=collator,
        )
        # ---- DEBUG DEVICE CHECK (add here) ----
        orig_training_step = trainer.training_step

        def debug_training_step(model, inputs, num_items_in_batch=None):
            if not hasattr(trainer, "_printed_device_info"):
                print("MODEL DEVICE:", next(model.parameters()).device)
                if "input_ids" in inputs:
                    print("INPUT DEVICE:", inputs["input_ids"].device)
                trainer._printed_device_info = True
            return orig_training_step(model, inputs, num_items_in_batch)

        trainer.training_step = debug_training_step
    # ---------------------------------------
    else:
        raise ValueError("mode must be 'green' or 'baseline'")

    gpu_count = 1
    server_count = 1

    tracker.start()
    wall_start = time.time()
    train_result = trainer.train()
    wall_end = time.time()

    # For green mode: GreenTrainer.train() stops the EnergyMonitor internally in its
    # finally block and stores the frozen snapshot on trainer._final_energy_metrics.
    # Inject it back into the tracker so tracker.stop() reads real numbers instead of
    # calling get_current_metrics() on an already-stopped (zeroed-out on Mac) monitor.
    if mode == "green" and hasattr(trainer, "_final_energy_metrics"):
        tracker._energy_monitor = None          # prevent re-read of stopped monitor
        tracker._injected_em_metrics = trainer._final_energy_metrics

    carbon = tracker.stop(gpu_count=gpu_count, server_count=server_count)

    # Trainer state path (if present)
    trainer_state_path = None
    direct_ts = os.path.join(run_dir, "trainer_state.json")
    if os.path.exists(direct_ts):
        trainer_state_path = direct_ts
    else:
        for root, _, files in os.walk(run_dir):
            if "trainer_state.json" in files:
                trainer_state_path = os.path.join(root, "trainer_state.json")
                break

    report = {
        "run_type": mode,
        "timestamp": tag,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": device,
        },
        "model": model_name,
        "dataset": dataset_name,
        "split": split,
        "max_steps": max_steps,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "seed": seed,
        "max_length": max_length,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": target_modules,
        "time_seconds_wall": float(wall_end - wall_start),
        # Disaggregated carbon/energy
        "energy_kwh": carbon.energy_kwh,
        "total_energy_wh": carbon.energy_wh,
        "gpu_energy_wh": carbon.gpu_energy_wh,
        "cpu_energy_wh": carbon.cpu_energy_wh,
        "current_power_w": carbon.current_power_w,
        "gpu_power_w": carbon.gpu_power_w,
        "cpu_power_w": carbon.cpu_power_w,
        "budget_used_percent": carbon.budget_used_percent,
        "timestamp_wall": carbon.timestamp_wall,
        "timestamp_mono": carbon.timestamp_mono,
        "carbon_intensity_g_per_kwh": carbon_intensity_g_per_kwh,
        "pue": pue,
        "co2_kg_operational": carbon.co2_kg_operational,
        "co2_kg_operational_x_pue": carbon.co2_kg_pue_adjusted,
        "co2_kg_embodied_amortized": carbon.co2_kg_embodied,
        "co2_kg_total_pue_plus_embodied": carbon.co2_kg_total_pue_plus_embodied,
        "carbon_method": carbon.method,
        "carbon_notes": carbon.notes,
        "output_dir": run_dir,
        "trainer_state_path": trainer_state_path,
        "trainer_metrics": getattr(train_result, "metrics", None),
        "warning": (
            "If GreenTrainer also starts/stops its own EnergyMonitor internally, "
            "Green may have double-monitor overhead. Disable internal monitoring in GreenTrainer for strict fairness."
            if mode == "green" else ""
        ),
    }

    out_path = os.path.join(run_dir, f"{mode}_run_report_{tag}.json")
    save_json(out_path, report)
    print(f"✅ Saved unified report: {out_path}")

    if carbon.method == "tdp_proxy":
        print("⚠️ EnergyMonitor/CodeCarbon did not provide energy on this platform; proxy used.")

    return run_dir


# =========================
# CLI
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--mode", choices=["green", "baseline"], required=True)

    # Shared training config
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--dataset", default="mlabonne/guanaco-llama2-1k")
    p.add_argument("--split", default="train[:100]")
    p.add_argument("--max_steps", type=int, default=50)
    p.add_argument("--num_train_epochs", type=float, default=0.0)  # 0 => ignored
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_length", type=int, default=128)

    # LoRA
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--target_modules", default="q_proj,v_proj")

    # Output roots
    p.add_argument("--outroot_green", default="./green_peft_test_results")
    p.add_argument("--outroot_baseline", default="./baseline_training_results")

    # Green-specific (and also used by monitor)
    p.add_argument("--energy_budget_wh", type=float, default=10000.0)

    # Carbon accounting
    p.add_argument("--carbon_intensity_g_per_kwh", type=float, default=100.0)
    p.add_argument("--pue", type=float, default=1.0)
    p.add_argument("--include_embodied", action="store_true")

    # Proxy + embodied parameters
    p.add_argument("--gpu_tdp_w", type=float, default=400.0)
    p.add_argument("--embodied_gpu_kg_per_gpu_hour", type=float, default=0.003)
    p.add_argument("--embodied_server_kg_per_server_hour", type=float, default=0.056)

    # CodeCarbon toggle
    p.add_argument("--no_codecarbon", action="store_true")

    # Green: disable importance sampling (use same data order as baseline for fair comparison)
    p.add_argument("--no_smart_sampling", action="store_true", help="Green only: use standard shuffle + fixed batch size (same as baseline)")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    outroot = args.outroot_green if args.mode == "green" else args.outroot_baseline
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]

    # Fixed step logic: if max_steps > 0, epochs just needs to be valid (>0)
    if args.max_steps and args.max_steps > 0:
        num_train_epochs = 1.0
    else:
        num_train_epochs = float(args.num_train_epochs) if args.num_train_epochs and args.num_train_epochs > 0 else 1.0

    print("\n==============================")
    print(f"Mode: {args.mode}")
    print(f"Output root: {outroot}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset} ({args.split})")
    print(f"Steps: {args.max_steps} | Epochs: {num_train_epochs}")
    print(f"Batch: {args.batch} | GradAccum: {args.grad_accum} | LR: {args.lr}")
    print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}, targets={target_modules}")
    print(f"Carbon intensity: {args.carbon_intensity_g_per_kwh} g/kWh | PUE: {args.pue} | Embodied: {args.include_embodied}")
    print(f"Energy budget (Wh): {args.energy_budget_wh}")
    print(f"CodeCarbon: {'disabled' if args.no_codecarbon else 'enabled (if installed)'}")
    if args.mode == "green":
        print(f"Green smart sampling: {'disabled (baseline-like order)' if args.no_smart_sampling else 'enabled'}")
    print("==============================\n")

    run_training(
        mode=args.mode,
        output_root=outroot,
        model_name=args.model,
        dataset_name=args.dataset,
        split=args.split,
        max_steps=args.max_steps,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        seed=args.seed,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        energy_budget_wh=args.energy_budget_wh,
        carbon_intensity_g_per_kwh=args.carbon_intensity_g_per_kwh,
        pue=args.pue,
        include_embodied=args.include_embodied,
        gpu_tdp_w=args.gpu_tdp_w,
        embodied_gpu_kg_per_gpu_hour=args.embodied_gpu_kg_per_gpu_hour,
        embodied_server_kg_per_server_hour=args.embodied_server_kg_per_server_hour,
        allow_codecarbon=(not args.no_codecarbon),
        use_smart_sampling=not getattr(args, "no_smart_sampling", False),
    )


if __name__ == "__main__":
    main()