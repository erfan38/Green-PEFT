# GreenPEFT — Results Summary (Colab run, 2026-06-19)

This file locks in the headline numbers from the clean 4-cell Colab run so the paper
can be written directly from it. All numbers are read from the run-report JSONs and
`trainer_state.json` files in this folder.

---

## 1. Experimental setup (shared by all runs unless noted)

| Item | Value |
|---|---|
| Method | LoRA (PEFT) |
| LoRA config | `r=8`, `alpha=32`, `dropout=0.1`, targets `q_proj, v_proj` |
| Epochs | 3 |
| Per-device batch | 8 |
| Gradient accumulation | 8 (effective batch 64) |
| Max sequence length | 256 |
| Learning rate | 2e-4 |
| Carbon accounting | intensity 50 g CO₂/kWh, PUE 1.2 (carbon-aware run uses zone **CA-QC**) |
| Energy/CO₂ source | `EnergyMonitor` (GPU+CPU); CodeCarbon disabled |
| Green levers | energy budget 400 Wh, loss-threshold skip, early grad-accum exit, convergence threshold 0.90 |
| Platform | Google Colab, T4 GPU (~15 GiB), Python 3.12 |

---

## 2. Core experiment — Qwen2.5-0.5B-Instruct × guanaco-llama2-1k

**The headline result.** Run folders used:
- Baseline → `fair/baseline/baseline_20260619_034636/`
- Green (no carbon) → `fair/green_no_carbon/green_20260619_034921/`
- Green (carbon-aware) → `fair/green_carbon/green_20260619_035153/`

| Mode | Energy (Wh) | Time (s) | CO₂×PUE (kg) | Skip % | Final loss |
|---|---|---|---|---|---|
| Baseline | 3.3615 | 145.68 | 2.017e-4 | 0.0 | 1.8991 |
| Green (no carbon) | 3.0359 | 136.24 | 1.822e-4 | 21.37 | 1.8452 |
| Green (carbon-aware) | 3.0658 | 138.22 | 1.839e-4 | 21.37 | 1.8452 |

**Green vs Baseline (no-carbon):**
- Energy: **−9.7%** (3.3615 → 3.0359 Wh)
- CO₂: **−9.7%**
- Time: **−6.5%** (145.7 → 136.2 s)
- Final loss: **1.845 vs 1.899 — green is slightly *lower* (better)**

Takeaway: green skips ~21% of micro-batches, cutting energy/CO₂ ~10% with **no quality
penalty** (loss even improves). Carbon-aware adds grid-aware scheduling at essentially
the same energy.

---

## 3. Quality / downstream accuracy (lm-evaluation-harness)

Backend: EleutherAI lm-eval-harness, **0-shot**, `limit=200` per task (smoke-test scale;
re-run with `limit=None` and few-shot for final paper numbers). Accuracy in %.

| Model | arc_easy | arc_challenge | hellaswag | winogrande | **avg** |
|---|---|---|---|---|---|
| base (no FT) | 56.5 | 36.0 | 48.5 | 59.5 | 50.12 |
| baseline-LoRA | 62.0 | 34.5 | 49.5 | 58.0 | 51.00 |
| green-LoRA | 59.5 | 35.5 | 48.5 | 57.5 | 50.25 |

**Key check:** green-LoRA avg (50.25) is within **0.75 pt** of baseline-LoRA (51.00), and
both beat the un-fine-tuned base (50.12). → **Quality is preserved** while saving energy.

> Caveat for the paper: these are `limit=200`, 0-shot numbers (quick eval). For the final
> table, re-run with full benchmark + standard few-shot (ARC 25-shot, MMLU 5-shot).

---

## 4. Generalization — 3 models × dataset (multi-eval)

Source: `multi_eval/comparisons/summary.csv`.

| Model × dataset | Baseline Wh | Green Wh | Energy saving | Baseline s | Green s | Time saving | Skip % |
|---|---|---|---|---|---|---|---|
| tinyllama-1.1b × guanaco | 6.2282 | 4.8265 | **−22.5%** | 275.9 | 215.6 | −21.9% | 22.73 |
| smollm2-1.7b × guanaco | 9.4283 | 7.3957 | **−21.6%** | 417.6 | 325.3 | −22.1% | 20.47 |
| qwen-0.5b × dolly | 3.5750 | 1.8671 | **−47.8%** | 160.2 | 85.8 | −46.4% | 49.70 |

Takeaway: energy savings hold across model sizes and datasets, ranging from ~22% up to
~48% (qwen×dolly, where nearly half the micro-batches were skippable).

---

## 5. Artifacts in this folder

| Path | What |
|---|---|
| `fair/baseline/…_034636/` | final baseline run (checkpoint-48) |
| `fair/green_no_carbon/…_034921/` | final green run (checkpoints 5–40) |
| `fair/green_carbon/…_035153/` | final green carbon-aware run (checkpoints 5–40) |
| `fair/compare_green/`, `fair/compare_carbon/` | comparison PDFs + charts (energy/CO₂/time/loss) |
| `multi_eval/` | 3-model generalization runs + comparison PDFs + `summary.csv` |
| `quality_eval_20260619_035950 (1).csv` | accuracy table (section 3) |

### Ignore / stale (not for the paper)
- `fair/green_*/…_0253xx/` and `…_0255xx/` — early green runs **without checkpoints**
  (before the `--save_steps` fix); JSON reports only.
- `fair/baseline/…_034126/` — aborted (CUDA OOM) run.
- `sample_data/` — Colab default sample data, copied in by accident; safe to delete.
- `Comparison_Report (N) copy.pdf` — duplicate downloads.

---

*Generated 2026-06-19 from the downloaded Colab results in `june-18-results-4cells/`.*
