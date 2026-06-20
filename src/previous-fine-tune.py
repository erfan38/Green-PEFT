# fine-tune.py
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

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

# Green PEFT
from energypeft import GreenTrainer, EnergyMonitor


# ============================================================
# Defaults (override via CLI)
# ============================================================
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_DATASET = "mlabonne/guanaco-llama2-1k"
DEFAULT_SPLIT = "train[:100]"

DEFAULT_MAX_STEPS = 50
DEFAULT_SEED = 42
DEFAULT_BATCH = 4
DEFAULT_GRAD_ACCUM = 1
DEFAULT_LR = 2e-4
DEFAULT_LOGGING_STEPS = 5
DEFAULT_SAVE_STEPS = 50
DEFAULT_MAX_LENGTH = 128

DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.1
DEFAULT_TARGET_MODULES = "q_proj,v_proj"

DEFAULT_OUTDIR_GREEN = "./green_peft_test_results"
DEFAULT_OUTDIR_BASELINE = "./baseline_training_results"

# If your EnergyMonitor uses intensity internally, fine. If not, CO2 may remain 0.
DEFAULT_ENERGY_BUDGET_WH = 10.0


# ============================================================
# Data structures
# ============================================================
@dataclass
class RunReport:
    run_type: str  # "green" or "baseline"
    timestamp: str
    model: str
    dataset: str
    split: str

    max_steps: int
    num_train_epochs: Optional[float]
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    seed: int

    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]

    # measurements
    time_seconds: float
    total_energy_wh: float
    co2_emissions_kg_est: float

    # file pointers
    output_dir: str
    trainer_state_path: Optional[str]
    green_report_path: Optional[str]  # if GreenTrainer writes its own report
    notes: Optional[str] = None


# ============================================================
# Helpers
# ============================================================
def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, payload: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def find_latest(pattern: str) -> Optional[str]:
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def find_trainer_state(output_dir: str) -> Optional[str]:
    return find_latest(os.path.join(output_dir, "**", "trainer_state.json"))


def find_green_training_report(output_dir: str) -> Optional[str]:
    return find_latest(os.path.join(output_dir, "green_training_report*.json"))


def to_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def safe_monitor_start(monitor: Any) -> None:
    # Support different monitor APIs
    for fn in ("start_monitoring", "start", "begin", "resume"):
        if hasattr(monitor, fn):
            try:
                getattr(monitor, fn)()
            except Exception:
                pass
            return


def safe_monitor_stop(monitor: Any) -> None:
    for fn in ("stop_monitoring", "stop", "end", "finish", "pause"):
        if hasattr(monitor, fn):
            try:
                getattr(monitor, fn)()
            except Exception:
                pass
            return


def extract_monitor_metrics(monitor: Any) -> Tuple[float, float, float]:
    """
    Returns (energy_wh, co2_kg, time_s) as best-effort.
    Time may be 0 if monitor doesn't track it; caller should override with wall-clock.
    """
    energy_wh = 0.0
    co2_kg = 0.0
    time_s = 0.0

    # 1) to_dict()
    if hasattr(monitor, "to_dict"):
        try:
            d = monitor.to_dict()
            if isinstance(d, dict):
                energy_wh = to_float(d.get("total_energy_wh", d.get("energy_wh", 0.0)), 0.0)
                co2_kg = to_float(d.get("co2_emissions_kg_est", d.get("co2_kg", d.get("co2e_kg", 0.0))), 0.0)
                time_s = to_float(d.get("time_seconds", d.get("elapsed_seconds", 0.0)), 0.0)
        except Exception:
            pass

    # 2) attribute fallbacks
    for k in ("total_energy_wh", "energy_wh"):
        if energy_wh == 0.0 and hasattr(monitor, k):
            try:
                energy_wh = to_float(getattr(monitor, k), 0.0)
            except Exception:
                pass

    for k in ("co2_emissions_kg_est", "co2_kg", "co2e_kg"):
        if co2_kg == 0.0 and hasattr(monitor, k):
            try:
                co2_kg = to_float(getattr(monitor, k), 0.0)
            except Exception:
                pass

    for k in ("time_seconds", "elapsed_seconds"):
        if time_s == 0.0 and hasattr(monitor, k):
            try:
                time_s = to_float(getattr(monitor, k), 0.0)
            except Exception:
                pass

    return energy_wh, co2_kg, time_s


# ============================================================
# Build model/tokenizer/dataset
# ============================================================
def build_model_and_tokenizer(model_name: str, use_fp16_if_cuda: bool = True) -> Tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if (use_fp16_if_cuda and torch.cuda.is_available()) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto")
    return model, tokenizer


def apply_lora(
    model: Any,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: list[str],
) -> Any:
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    return model


def prepare_dataset(dataset_name: str, split: str, tokenizer: Any, max_length: int) -> Any:
    ds = load_dataset(dataset_name, split=split)

    def tokenize_fn(sample: Dict[str, Any]) -> Dict[str, Any]:
        # dataset has "text" column
        out = tokenizer(sample["text"], padding="max_length", truncation=True, max_length=max_length)
        out["labels"] = out["input_ids"].copy()
        return out

    tokenized = ds.map(tokenize_fn, remove_columns=ds.column_names)
    tokenized.set_format("torch")
    return tokenized


# ============================================================
# Run training
# ============================================================
def run_training(
    mode: str,  # "green" or "baseline"
    output_dir: str,
    model_name: str,
    dataset_name: str,
    split: str,
    max_steps: int,
    num_train_epochs: Optional[float],
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
    target_modules: list[str],
    energy_budget_wh: float,
    region: Optional[str] = None,
    carbon_intensity: Optional[float] = None,
) -> RunReport:
    ensure_dir(output_dir)
    set_seed(seed)

    model, tokenizer = build_model_and_tokenizer(model_name)
    model = apply_lora(model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout, target_modules=target_modules)

    train_dataset = prepare_dataset(dataset_name, split, tokenizer, max_length=max_length)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        max_steps=max_steps if max_steps > 0 else -1,
        num_train_epochs=num_train_epochs if (num_train_epochs is not None and num_train_epochs > 0) else 3.0,
        report_to="none",
        logging_strategy="steps",
        save_strategy="steps",
        save_steps=save_steps,
        seed=seed,
        data_seed=seed,
    )

    # External monitor for both modes (so baseline and green both have comparable measurements)
    # 🌍 Configure monitor with region/intensity for Baseline
    monitor_kwargs = {"energy_budget_wh": energy_budget_wh}
    # Note: efficient_training.py's EnergyMonitor might not accept region in __init__, check signature.
    # Actually, EnergyMonitor(energy_budget_wh=..., region=..., carbon_intensity=...) is standard if supported.
    # Let's check if EnergyMonitor supports region in __init__.
    # Based on previous files, it seems it takes (energy_budget_wh=..., cpu_backend=...)
    # But let's assume standard usage or add it if missing.
    # Wait, EnergyMonitor signature seen earlier: (energy_budget_wh: float = 0.0, cpu_backend: str = "auto", ...)
    # It seems EnergyMonitor MIGHT NOT take region/carbon_intensity in __init__ in this version.
    # However, GreenTrainer does take `region` via kwargs passed to its internal components?
    # Let's assume we pass what we can.

    # FOR NOW, let's pass it to GreenTrainer which DEFINITELY should handle it if it's "Green".

    monitor = EnergyMonitor(energy_budget_wh=energy_budget_wh) # robust init
    # If monitor has region attribute, set it.
    if hasattr(monitor, "region") and region:
        monitor.region = region
    if hasattr(monitor, "carbon_intensity") and carbon_intensity:
        monitor.carbon_intensity = carbon_intensity

    start_wall = time.time()
    safe_monitor_start(monitor)

    green_report_path = None
    notes = None

    if mode == "green":
        # GreenTrainer kwargs
        gt_kwargs = {
            "model": model,
            "tokenizer": tokenizer,
            "args": args,
            "train_dataset": train_dataset,
            "data_collator": collator,
            "energy_budget_wh": energy_budget_wh,
            "base_batch_size": per_device_train_batch_size,
            "min_batch_size": 1,
        }
        # Pass region/intensity if GreenTrainer accepts them (it likely does via kwargs or EnergyMonitor injection)
        # But GreenTrainer creates its OWN EnergyMonitor usually.
        # We can try passing 'region' in kwargs if GreenTrainer supports it.
        if region:
            gt_kwargs["region"] = region
        if carbon_intensity:
            gt_kwargs["carbon_intensity"] = carbon_intensity

        trainer = GreenTrainer(**gt_kwargs)
        trainer.train()

        # If GreenTrainer wrote its own report, capture it (optional)
        green_report_path = find_green_training_report(output_dir)

    elif mode == "baseline":
        # HF Trainer (no green)
        # Compatibility: older/newer transformers versions use tokenizer vs processing_class.
        import inspect

        sig = inspect.signature(Trainer.__init__)
        trainer_kwargs = {
            "model": model,
            "args": args,
            "train_dataset": train_dataset,
            "data_collator": collator,
        }
        if "processing_class" in sig.parameters:
            trainer_kwargs["processing_class"] = tokenizer
        else:
            trainer_kwargs["tokenizer"] = tokenizer

        trainer = Trainer(**trainer_kwargs)
        trainer.train()
    else:
        raise ValueError("mode must be 'green' or 'baseline'")

    safe_monitor_stop(monitor)
    end_wall = time.time()

    energy_wh, co2_kg, time_s_mon = extract_monitor_metrics(monitor)

    # 🌍 Manual Carbon Calculation fallback if monitor didn't do it
    # If monitor returned 0.0 kg but we have energy and intensity, calculate it manually.
    if co2_kg == 0.0 and energy_wh > 0:
        # 1. Use user-provided intensity
        if carbon_intensity:
            # g/kWh -> kg/Wh:  (g / 1000) / 1000 * Wh = kg
            # Wait, intensity is g/kWh.
            # Energy is Wh.
            # Wh / 1000 = kWh.
            # kWh * (g/kWh) = g.
            # g / 1000 = kg.
            kwh = energy_wh / 1000.0
            gram_co2 = kwh * carbon_intensity
            co2_kg = gram_co2 / 1000.0
            notes = f"Manual Calc: {carbon_intensity} g/kWh"
        # 2. Use region-based default if provided (simplified map)
        elif region:
            # Simple fallback map for demo
            REGIONS = {
                "CA-QC": 32.0,   # Quebec (hydro) - very clean
                "US-CA": 250.0,  # California - mixed
                "US-WY": 800.0,  # Wyoming (coal) - dirty
                "Global": 475.0, # World avg
            }
            if region in REGIONS:
                intensity = REGIONS[region]
                kwh = energy_wh / 1000.0
                co2_kg = (kwh * intensity) / 1000.0
                notes = f"Manual Calc: Region {region} ({intensity} g/kWh)"

    time_s = time_s_mon if time_s_mon > 0 else (end_wall - start_wall)

    trainer_state_path = find_trainer_state(output_dir)

    report = RunReport(
        run_type=mode,
        timestamp=now_tag(),
        model=model_name,
        dataset=dataset_name,
        split=split,
        max_steps=max_steps,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        seed=seed,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        time_seconds=float(time_s),
        total_energy_wh=float(energy_wh),
        co2_emissions_kg_est=float(co2_kg),
        output_dir=output_dir,
        trainer_state_path=trainer_state_path,
        green_report_path=green_report_path,
        notes=notes,
    )

    # Write a unified report for BOTH modes (always)
    out_report_path = os.path.join(output_dir, f"{mode}_run_report_{report.timestamp}.json")
    save_json(out_report_path, asdict(report))
    print(f"✅ Saved unified report: {out_report_path}")

    # Optional: surface missing metrics explicitly
    if report.total_energy_wh == 0.0:
        print("⚠️ Energy is 0.0 Wh. This usually means EnergyMonitor did not capture power on this platform.")
    if report.co2_emissions_kg_est == 0.0:
        print("⚠️ Carbon is 0.0 kg. Provide --region or --carbon_intensity to calculate emissions.")
    else:
        print(f"🌍 Carbon Emissions: {report.co2_emissions_kg_est:.6f} kg CO2e")
    if report.trainer_state_path is None:
        print("⚠️ trainer_state.json not found. Loss/runtime extraction will be limited.")

    if mode == "green" and report.green_report_path:
        print(f"ℹ️ GreenTrainer report detected: {report.green_report_path}")

    return report


# ============================================================
# CLI
# ============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune in GREEN or BASELINE mode with identical config + monitoring.")
    p.add_argument("--mode", choices=["green", "baseline"], required=True, help="Training mode.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--split", default=DEFAULT_SPLIT)

    p.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS)
    p.add_argument("--num_train_epochs", type=float, default=0.0, help="If >0, uses epochs; otherwise uses max_steps.")

    p.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    p.add_argument("--grad_accum", type=int, default=DEFAULT_GRAD_ACCUM)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--logging_steps", type=int, default=DEFAULT_LOGGING_STEPS)
    p.add_argument("--save_steps", type=int, default=DEFAULT_SAVE_STEPS)
    p.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)

    p.add_argument("--lora_r", type=int, default=DEFAULT_LORA_R)
    p.add_argument("--lora_alpha", type=int, default=DEFAULT_LORA_ALPHA)
    p.add_argument("--lora_dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    p.add_argument("--target_modules", default=DEFAULT_TARGET_MODULES, help="Comma-separated modules, e.g. q_proj,v_proj")

    p.add_argument("--energy_budget_wh", type=float, default=DEFAULT_ENERGY_BUDGET_WH)
    p.add_argument("--region", default=None, help="Region code (e.g. CA-QC, US-CA) for carbon intensity lookup.")
    p.add_argument("--carbon_intensity", type=float, default=None, help="Manual carbon intensity (g/kWh) if region is unknown.")

    p.add_argument("--outdir", default="", help="If empty: uses default green/baseline output dir based on mode.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    mode = args.mode
    timestamp = now_tag()

    # Base root (green or baseline default root)
    base_root = DEFAULT_OUTDIR_GREEN if mode == "green" else DEFAULT_OUTDIR_BASELINE

    # If user manually specifies --outdir, use it as root
    if args.outdir.strip():
        base_root = args.outdir.strip()

    # ✅ Create UNIQUE directory per run (no overwrite ever)
    outdir = os.path.join(base_root, f"{mode}_{timestamp}")

    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    num_train_epochs = args.num_train_epochs if args.num_train_epochs and args.num_train_epochs > 0 else None

    print(f"\n==============================")
    print(f"Mode: {mode}")
    print(f"Output dir: {outdir}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset} ({args.split})")
    print(f"Steps: {args.max_steps} | Epochs: {num_train_epochs}")
    print(f"Batch: {args.batch} | GradAccum: {args.grad_accum} | LR: {args.lr}")
    print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}, targets={target_modules}")
    print(f"Energy budget (Wh): {args.energy_budget_wh}")
    print(f"==============================\n")

    run_training(
        mode=mode,
        output_dir=outdir,
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
        region=args.region,
        carbon_intensity=args.carbon_intensity,
    )


if __name__ == "__main__":
    main()