#!/usr/bin/env bash
#SBATCH --account=def-mabell
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=greenpeft
#SBATCH --output=greenpeft_%j.out
#SBATCH --error=greenpeft_%j.err

set -euo pipefail
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

export HF_HOME="${SCRATCH}/hf"
export TRANSFORMERS_CACHE="${SCRATCH}/hf/transformers"
export HF_DATASETS_CACHE="${SCRATCH}/hf/datasets"
mkdir -p "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"
export PYTHONNOUSERSITE=1

module --force purge
module load StdEnv/2023
module load gcc/12.3
module load arrow/23.0.1
module load python/3.11
module load cuda/12.2

source "${HOME}/scratch/greenpeft_env_py311/bin/activate"

cd "${HOME}/scratch/greenpeft/repos/GreenPEFT"
export PYTHONPATH="${HOME}/scratch/greenpeft/repos/GreenPEFT:${PYTHONPATH:-}"

python -c "import sys; print('Python:', sys.executable); import energypeft; print('energypeft OK:', energypeft.__file__)"
python -c "import transformers; print('transformers:', transformers.__version__)"
echo "Host: $(hostname) | Job: ${SLURM_JOB_ID:-NA}"
python -c "import torch; print('Torch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo

BASE_DIR="${SCRATCH}/greenpeft_runs"
OUT_GREEN="${BASE_DIR}/green"
OUT_BASE="${BASE_DIR}/baseline"
OUT_COMPARE="${BASE_DIR}/compare"
mkdir -p "${OUT_GREEN}" "${OUT_BASE}" "${OUT_COMPARE}"

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
DATASET="mlabonne/guanaco-llama2-1k"
SPLIT="train[:75%]"
BATCH=8
GRAD_ACCUM=8
LR=2e-4
MAX_LENGTH=256
LORA_R=8
LORA_ALPHA=32
LORA_DROPOUT=0.1
TARGET_MODULES="q_proj,v_proj"
ENERGY_BUDGET_WH=400     # realistic for A100: ~300W × ~25min/epoch × 3 epochs ≈ 375 Wh
                         # ratio drops meaningfully → adaptive controller + early stopper engage
CARBON_INTENSITY=50      # gCO2/kWh — Quebec grid
PUE=1.2                  # Narval datacenter
GPU_TDP_W=400            # A100 TDP (fallback only; NVML gives real readings on Narval)

echo "=== GREEN MODE ==="
python -u src/fine-tune.py \
  --mode green \
  --model "${MODEL}" \
  --dataset "${DATASET}" \
  --split "${SPLIT}" \
  --max_steps -1 \
  --num_train_epochs 3.0 \
  --batch "${BATCH}" \
  --grad_accum "${GRAD_ACCUM}" \
  --lr "${LR}" \
  --max_length "${MAX_LENGTH}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --target_modules "${TARGET_MODULES}" \
  --outroot_green "${OUT_GREEN}" \
  --energy_budget_wh "${ENERGY_BUDGET_WH}" \
  --carbon_intensity_g_per_kwh "${CARBON_INTENSITY}" \
  --pue "${PUE}" \
  --gpu_tdp_w "${GPU_TDP_W}" \
  --no_codecarbon

echo "=== BASELINE MODE ==="
python -u src/fine-tune.py \
  --mode baseline \
  --model "${MODEL}" \
  --dataset "${DATASET}" \
  --split "${SPLIT}" \
  --max_steps -1 \
  --num_train_epochs 3.0 \
  --batch "${BATCH}" \
  --grad_accum "${GRAD_ACCUM}" \
  --lr "${LR}" \
  --max_length "${MAX_LENGTH}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --target_modules "${TARGET_MODULES}" \
  --outroot_baseline "${OUT_BASE}" \
  --carbon_intensity_g_per_kwh "${CARBON_INTENSITY}" \
  --pue "${PUE}" \
  --gpu_tdp_w "${GPU_TDP_W}" \
  --no_codecarbon

LATEST_GREEN="$(ls -td "${OUT_GREEN}"/* 2>/dev/null | head -n 1 || true)"
LATEST_BASE="$(ls -td "${OUT_BASE}"/* 2>/dev/null | head -n 1 || true)"

if [[ -z "${LATEST_GREEN}" || -z "${LATEST_BASE}" ]]; then
  echo "ERROR: Could not find latest run directories."
  ls -la "${OUT_GREEN}" || true
  ls -la "${OUT_BASE}" || true
  exit 1
fi

echo "Latest GREEN: ${LATEST_GREEN}"
echo "Latest BASE:  ${LATEST_BASE}"

python -u src/compare.py \
  --run_a "${LATEST_GREEN}" \
  --run_b "${LATEST_BASE}" \
  --label_a Green \
  --label_b Baseline \
  --outdir "${OUT_COMPARE}"

echo "=== DONE ==="