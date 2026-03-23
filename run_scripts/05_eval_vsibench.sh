#!/bin/bash
# =============================================================================
# Step 5: VSI-Bench Evaluation
# =============================================================================
# Evaluates DiSPider on the VSI-Bench benchmark.
#
# Submit via SLURM:
#   sbatch --account=bfhg-delta-gpu --partition=gpuA100x4 \
#          --mem=80g --nodes=1 --gpus-per-node=1 \
#          --cpus-per-task=8 --time=0-04:00:00 \
#          run_scripts/05_eval_vsibench.sh
#
# Or for Stage 1 finetuned compressor variant:
#   MODE=stage1_ft sbatch ...  run_scripts/05_eval_vsibench.sh
#
# Environment variables (with defaults):
#   MODE            baseline | stage1_ft | stage2_ft  (default: baseline)
#   MODEL_PATH      HF Hub ID or local path (default: Mar2Ding/Dispider)
#   STAGE1_CKPT     path to finetuned Stage 1 ckpt (default: auto)
#   DATA_PATH       path to test_combined.json
#   VIDEO_ROOT      path to scannet videos
#   OUTPUT_DIR      base output directory
# =============================================================================

source ~/.bashrc
conda activate dispider

set -e
cd /u/hyin2/Dispider

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Config ---
MODE="${MODE:-stage1_ft}"
MODEL_PATH="${MODEL_PATH:-Mar2Ding/Dispider}"
STAGE1_CKPT="${STAGE1_CKPT:-outputs/dispider_finetune/stage1/checkpoint-4143}"
DATA_PATH="${DATA_PATH:-datasets/vsi-bench/my_qa/test/test_combined.json}"
VIDEO_ROOT="${VIDEO_ROOT:-datasets/vsi-bench/scannet}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/eval_vsibench}"

NUM_FRAMES="${NUM_FRAMES:-16}"
MAX_CLIPS="${MAX_CLIPS:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"

echo "=== VSI-Bench Evaluation ==="
echo "  Mode        : $MODE"
echo "  Model       : $MODEL_PATH"
echo "  Data        : $DATA_PATH"
echo "  Video root  : $VIDEO_ROOT"
echo "  Output dir  : $OUTPUT_DIR/$MODE"
echo ""

EXTRA_ARGS=""
if [ "$MODE" = "stage1_ft" ]; then
    EXTRA_ARGS="--compressor_override $STAGE1_CKPT"
    echo "  Compressor override: $STAGE1_CKPT"
fi

python eval/eval_vsibench.py \
    --model_path     "$MODEL_PATH"                    \
    --data_path      "$DATA_PATH"                     \
    --video_root     "$VIDEO_ROOT"                    \
    --output_path    "$OUTPUT_DIR/$MODE/results.json" \
    --num_frames     "$NUM_FRAMES"                    \
    --max_clips      "$MAX_CLIPS"                     \
    --max_new_tokens "$MAX_NEW_TOKENS"                \
    $EXTRA_ARGS                                        \
    --resume                                          \


echo ""
echo "Done. Results in $OUTPUT_DIR/$MODE/"
