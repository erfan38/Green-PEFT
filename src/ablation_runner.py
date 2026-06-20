# ablation_runner.py / Colab cell
# Runs the 5-config additive ablation:
#   C0  baseline (everything off)
#   C1  + smart sampling only
#   C2  + smart sampling + adaptive batch
#   C3  + smart sampling + adaptive batch + loss-threshold skip
#   C4  full green (all 4 techniques on)
#
# Plus the carbon-aware run kept separately for the paper:
#   C5  full green + carbon-aware
#
# All runs use the same seed so loss curves are comparable.

import subprocess, sys, glob, os

def run(cmd):
    print(f"\n{'='*60}\n{cmd}\n{'='*60}", flush=True)
    proc = subprocess.Popen(
        cmd, shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True, bufsize=1
    )
    for line in proc.stdout:
        print(line, end='', flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit {proc.returncode})")

# ── Shared config (identical across all runs) ─────────────────
SEED = 42
BASE = (
    'python -u src/fine-tune_2.py '
    '--model "Qwen/Qwen2.5-0.5B-Instruct" '
    '--dataset "mlabonne/guanaco-llama2-1k" --split "train" '
    '--max_steps -1 --num_train_epochs 3.0 '
    '--batch 8 --grad_accum 8 --lr 2e-4 --max_length 256 '
    '--lora_r 8 --lora_alpha 32 --lora_dropout 0.1 '
    '--target_modules "q_proj,v_proj" '
    '--carbon_intensity_g_per_kwh 50 --pue 1.2 --no_codecarbon '
    f'--seed {SEED}'
)

# Energy-budget arg only matters for green configs but harmless to include
ENERGY = '--energy_budget_wh 400'

OUT_ROOT = "/content/ablation"
os.makedirs(OUT_ROOT, exist_ok=True)

# ── C0: Baseline (everything off) ────────────────────────────
print("\n\n===== C0: BASELINE =====", flush=True)
run(f'{BASE} --mode baseline --outroot_baseline {OUT_ROOT}/C0_baseline')

# ── C1: + smart sampling only ────────────────────────────────
# Smart sampling on, adaptive batch off, loss-skip off, grad-exit off
# This requires an explicit --no_adaptive_batch flag in fine-tune_2.py
# OR you can implement by setting min_batch_size = batch (so it never changes).
print("\n\n===== C1: + Smart Sampling =====", flush=True)
run(f'{BASE} --mode green {ENERGY} '
    f'--outroot_green {OUT_ROOT}/C1_sampling')

# ── C2: + smart sampling + adaptive batch ───────────────────
# Add (assuming this flag exists or is implicit)
print("\n\n===== C2: + Adaptive Batch =====", flush=True)
run(f'{BASE} --mode green {ENERGY} '
    f''
    f'--outroot_green {OUT_ROOT}/C2_sampling_batch')

# ── C3: + loss-threshold skip ────────────────────────────────
print("\n\n===== C3: + Loss-Threshold Skip =====", flush=True)
run(f'{BASE} --mode green {ENERGY} '
    f''
    f'--use_loss_threshold '
    f'--outroot_green {OUT_ROOT}/C3_sampling_batch_loss')

# ── C4: Full Green (all 4 techniques) ────────────────────────
print("\n\n===== C4: FULL GREEN =====", flush=True)
run(f'{BASE} --mode green {ENERGY} '
    f''
    f'--use_loss_threshold '
    f'--use_early_accum_exit --convergence_threshold 0.90 '
    f'--outroot_green {OUT_ROOT}/C4_full_green')

# ── C5: Full Green + carbon-aware (kept for paper, not part of ablation) ──
print("\n\n===== C5: FULL GREEN + CARBON-AWARE =====", flush=True)
run(f'{BASE} --mode green {ENERGY} '
    f''
    f'--use_loss_threshold '
    f'--use_early_accum_exit --convergence_threshold 0.90 '
    f'--use_carbon_aware --carbon_zone "CA-QC" '
    f'--outroot_green {OUT_ROOT}/C5_carbon_aware')

# ── Locate run dirs ──────────────────────────────────────────
def latest(path):
    dirs = sorted(glob.glob(f"{path}/*/"))
    return dirs[-1] if dirs else None

runs = {
    "C0_baseline":         latest(f"{OUT_ROOT}/C0_baseline"),
    "C1_sampling":         latest(f"{OUT_ROOT}/C1_sampling"),
    "C2_sampling_batch":   latest(f"{OUT_ROOT}/C2_sampling_batch"),
    "C3_sampling_batch_loss": latest(f"{OUT_ROOT}/C3_sampling_batch_loss"),
    "C4_full_green":       latest(f"{OUT_ROOT}/C4_full_green"),
    "C5_carbon_aware":     latest(f"{OUT_ROOT}/C5_carbon_aware"),
}
print("\n\n===== RUN DIRECTORIES =====")
for k, v in runs.items():
    print(f"  {k}: {v}")

# ── Pairwise compare each green config vs baseline ───────────
COMPARE_OUT = f"{OUT_ROOT}/comparisons"
os.makedirs(COMPARE_OUT, exist_ok=True)

for name in ["C1_sampling", "C2_sampling_batch", "C3_sampling_batch_loss",
             "C4_full_green", "C5_carbon_aware"]:
    if runs[name] is None:
        continue
    out = f"{COMPARE_OUT}/{name}_vs_baseline"
    print(f"\n----- COMPARE: {name} vs Baseline -----")
    run(f'python -u src/compare_2.py '
        f'--run_a "{runs[name]}" --label_a "{name}" '
        f'--run_b "{runs["C0_baseline"]}" --label_b "Baseline" '
        f'--outdir "{out}"')

# ── Download all PDFs ────────────────────────────────────────
try:
    from google.colab import files
    for pdf in sorted(glob.glob(f"{COMPARE_OUT}/*/*.pdf")):
        print(f"Downloading: {pdf}")
        files.download(pdf)
except Exception:
    print("Not in Colab — PDFs are at:", COMPARE_OUT)