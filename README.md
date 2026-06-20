# GreenPEFT: Energy- and Carbon-Aware Parameter-Efficient Fine-Tuning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Sustainable AI](https://img.shields.io/badge/AI-Energy%20%26%20Carbon%20Aware-green.svg)](https://github.com/erfan38/Green-PEFT)

**GreenPEFT** is an energy- and carbon-aware framework for parameter-efficient
fine-tuning (PEFT) of large language models. Standard PEFT methods such as LoRA
reduce the number of *trainable parameters*, but they still run a full forward
and backward pass through the frozen backbone on every step, so energy
consumption keeps scaling with the number of training steps. GreenPEFT instead
targets the **redundant computation of fine-tuning itself**, reducing energy,
wall-clock time, and operational carbon emissions while preserving downstream
model quality.

> This repository accompanies the paper *"Green Parameter-Efficient Fine-Tuning:
> Sustainable PEFT for Large Language Models."* It is released so reviewers and
> practitioners can inspect the implementation and reproduce the reported
> results. GreenPEFT is built on top of LoRA and is distributed as the
> `energypeft` Python package.

## 🌱 What GreenPEFT does

GreenPEFT introduces **two novel compute-adaptive techniques** that decide *when*
and *how much* computation to spend during training, plus four supporting
components.

### Core techniques (the contribution)

| # | Technique | Idea |
|---|-----------|------|
| **1** | **Adaptive Loss-Threshold Micro-Batch Skipping** | Skips an entire micro-batch (forward **and** backward) when the most recent loss falls below an EMA-relative, annealing threshold. A safety rule guarantees at least one real gradient per accumulation window. |
| **2** | **Gradient-Accumulation Early Exit** | Triggers the optimizer step *before* the accumulation window finishes once the running window loss signals early convergence, saving the remaining micro-batches. |

### Supporting components

| Component | Description |
|-----------|-------------|
| **Loss–Length Importance Sampling** | Prioritizes high-value-per-watt samples via `score = (loss/length)^α / length^β`. |
| **Energy-Aware Adaptive Batching** | Modulates the effective batch size from the remaining energy budget and training progress. |
| **Real-Time Energy Monitoring** | Per-step GPU/CPU energy via NVML (NVIDIA), RAPL (Intel), with a TDP fallback. |
| **Carbon-Aware Scheduling** | Scales skipping aggressiveness with real-time grid carbon intensity (Electricity Maps / UK National Grid). |

## 📊 Headline results

Evaluated with LoRA across three GPU classes (T4, L4, A100), three model scales
(0.5–1.7B parameters), and two datasets (full numbers in
[`results/RESULTS_SUMMARY_june19.md`](results/RESULTS_SUMMARY_june19.md)):

- **Up to 24.2%** lower energy and operational CO₂ on the primary
  Qwen2.5-0.5B × Guanaco configuration, with **24.9%** less wall-clock time.
- **19.8–47.0%** energy savings across the generalization study (largest where
  the data contain the most redundant computation).
- **No measurable quality loss**: the GreenPEFT adapter stays within **0.75
  points** of the LoRA baseline across ARC-Easy, ARC-Challenge, HellaSwag, and
  WinoGrande (zero-shot).

## 📁 Project structure

```
Green-PEFT/
├── setup.py                          # Package installation
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
│
├── energypeft/                       # Main Python package (imported as `energypeft`)
│   ├── __init__.py                   # Public API exports
│   ├── core/
│   │   ├── energy_monitor.py         # Real-time GPU/CPU energy tracking (NVML/RAPL/TDP)
│   │   ├── efficient_training.py     # Importance sampling + adaptive batch + early stopping
│   │   ├── carbon_scheduler.py       # Carbon-aware scheduling / grid intensity
│   │   └── carbon_monitor.py         # Background carbon-intensity polling
│   ├── trainers/
│   │   ├── green_trainer.py          # Drop-in HF Trainer subclass
│   │   └── green_trainer_2.py        # Trainer with Techniques 1 & 2 (used in the paper)
│   ├── integrations/                 # huggingface_peft / llamafactory / transformers hooks
│   └── examples/                     # API usage snippets
│
├── src/                              # Experiment scripts (reproduce the paper)
│   ├── fine-tune_2.py                # Main training entry point (baseline & green modes)
│   ├── ablation_runner.py            # Ablation study (configs C0–C5)
│   ├── compare.py / compare_2.py     # Baseline-vs-GreenPEFT comparison reports
│   └── green_clean.ipynb             # End-to-end Colab notebook
│
├── results/                          # Reviewer-facing summary results
│   ├── RESULTS_SUMMARY_june19.md     # Locked headline numbers
│   ├── generalization_summary.csv    # 3-model × dataset energy/time table
│   ├── quality_eval.csv              # Downstream accuracy table
│   └── comparison_pdfs/              # Energy / CO₂ comparison charts
│
├── examples/                         # Standalone usage examples
└── tests/                            # Unit + end-to-end tests
```

## 🚀 Quick start

### Installation

```bash
git clone https://github.com/erfan38/Green-PEFT.git
cd Green-PEFT
pip install -e .
```

### Reproduce the paper's training runs

Training is driven by `src/fine-tune_2.py`, which supports a `baseline` LoRA mode
and a `green` mode that enables Techniques 1 & 2:

```bash
# Baseline LoRA
python src/fine-tune_2.py --mode baseline \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset mlabonne/guanaco-llama2-1k \
  --num_train_epochs 3 --batch 8 --grad_accum 8 --lr 2e-4 --max_length 256

# GreenPEFT (loss-threshold skipping + early exit + carbon-aware)
python src/fine-tune_2.py --mode green \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset mlabonne/guanaco-llama2-1k \
  --num_train_epochs 3 --batch 8 --grad_accum 8 --lr 2e-4 --max_length 256 \
  --energy_budget_wh 400 \
  --use_loss_threshold \
  --use_early_accum_exit --convergence_threshold 0.90 \
  --use_carbon_aware --carbon_zone CA-QC --carbon_update_interval 50
```

Each run writes a report JSON with energy, time, CO₂, skip rate, and early-exit
counts. See [`src/`](src/) and the SLURM wrappers (`run*.sh`) for the full set of
flags, and [`src/green_clean.ipynb`](src/green_clean.ipynb) for an end-to-end
Colab walkthrough.

### Option A — `GreenTrainer` (Hugging Face integration)

```python
from energypeft import GreenTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

args = TrainingArguments(output_dir="./results", num_train_epochs=3)

trainer = GreenTrainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    train_dataset=your_dataset,
    energy_budget_wh=400.0,
)

trainer.train()   # writes a training report with energy/carbon metrics
```

### Option B — standalone training loop

```python
from torch.utils.data import DataLoader
from energypeft import EnergyMonitor, EnergyAwareTrainingController, attach_indices_collate

monitor = EnergyMonitor(energy_budget_wh=400.0)
controller = EnergyAwareTrainingController(
    dataset=dataset, energy_monitor=monitor, base_batch_size=8, max_steps=1000,
)
loader = DataLoader(dataset, batch_sampler=controller.batch_sampler,
                    collate_fn=attach_indices_collate)

monitor.start_monitoring()
for step, batch in enumerate(loader):
    loss = model(batch).loss
    loss.backward(); optimizer.step()
    controller.on_train_step_end(
        batch_indices=batch["_indices"], per_sample_losses=per_sample_losses,
        lengths=lengths, global_step=step,
    )
    if controller.early_stopper.should_stop(
        step=step, remaining_energy_wh=monitor.get_remaining_energy(), val_metric=val_metric,
    ):
        break
metrics = monitor.stop_monitoring()
monitor.save_energy_log("energy_log.json")
```

## 📊 Core components

### Energy monitor (`core/energy_monitor.py`)

| Backend | Platform | Method |
|---------|----------|--------|
| NVML | NVIDIA GPU | Direct power reading |
| RAPL | Linux Intel | Hardware energy counters |
| powermetrics | macOS | System power metrics |
| TDP | Fallback | CPU utilization × TDP estimate |

GPU and CPU energy are tracked separately, aggregated into total consumed energy,
and compared against a user-defined budget to maintain the remaining-energy ratio
used by the two training techniques at runtime.

### Carbon scheduler (`core/carbon_scheduler.py`)

```python
from energypeft import wait_for_green_grid
wait_for_green_grid(max_intensity=250, region="CA-QC")  # block until grid is clean enough
```

During training, the live carbon intensity is mapped to an aggressiveness factor
that tightens the skip threshold (Technique 1) and the early-exit threshold
(Technique 2) when the grid is dirtier — removing more low-utility computation
exactly when each unit of energy carries a higher carbon cost.

## 📋 Requirements

- Python ≥ 3.8, PyTorch ≥ 2.0, Transformers ≥ 4.30, PEFT ≥ 0.4
- `numpy`, `pandas`, `psutil`, `tqdm`, `datasets`, `accelerate`
- `pynvml` (optional, NVIDIA GPU monitoring)

Install everything with `pip install -e .` (pulls from `requirements.txt`).

## 🧪 Testing

```bash
PYTHONPATH=. python -m pytest tests/        # run the test suite
python -c "from energypeft import GreenTrainer; print('OK')"   # quick import check
```

## 📄 License

Released under the MIT License — see the `LICENSE` file.

## 📚 Citation

If you use GreenPEFT, please cite the paper:

```bibtex
Comming soon!
```

---

**Making AI fine-tuning sustainable — one micro-batch at a time.** 🌱
