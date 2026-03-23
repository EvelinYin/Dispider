#!/usr/bin/env bash
# =============================================================================
# run_eval_vsibench.sh
# VSI-Bench evaluation runner for DiSPider
#
# Run from the repo root:
#   cd /u/hyin2/Dispider
#   bash eval/run_eval_vsibench.sh [baseline|stage1_ft|smoke]
#
# Modes
# -----
#   baseline   -- evaluate published Mar2Ding/Dispider weights (default)
#   stage1_ft  -- evaluate with your finetuned Stage 1 compressor substituted in
#   smoke      -- quick 3-sample sanity check (no GPU required for logic check)
# =============================================================================

set -euo pipefail

MODE="${1:-baseline}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_PATH="Mar2Ding/Dispider"
DATA_PATH="datasets/vsi-bench/my_qa/test/test_combined.json"
VIDEO_ROOT="datasets/vsi-bench/scannet"
STAGE1_CKPT="outputs/dispider_finetune/stage1/checkpoint-4143"

NUM_FRAMES=16
MAX_CLIPS=32
MAX_NEW_TOKENS=64

echo "=================================================="
echo "DiSPider  VSI-Bench Evaluation"
echo "Mode     : ${MODE}"
echo "Repo root: ${REPO_ROOT}"
echo "=================================================="

cd "${REPO_ROOT}"

case "${MODE}" in

  baseline)
    OUTPUT_PATH="outputs/eval_vsibench/baseline/results.json"
    python eval/eval_vsibench.py \
        --model_path      "${MODEL_PATH}"  \
        --data_path       "${DATA_PATH}"   \
        --video_root      "${VIDEO_ROOT}"  \
        --output_path     "${OUTPUT_PATH}" \
        --num_frames      "${NUM_FRAMES}"  \
        --max_clips       "${MAX_CLIPS}"   \
        --max_new_tokens  "${MAX_NEW_TOKENS}" \
        --resume
    ;;

  stage1_ft)
    OUTPUT_PATH="outputs/eval_vsibench/stage1_ft/results.json"
    python eval/eval_vsibench.py \
        --model_path          "${MODEL_PATH}"  \
        --compressor_override "${STAGE1_CKPT}" \
        --data_path           "${DATA_PATH}"   \
        --video_root          "${VIDEO_ROOT}"  \
        --output_path         "${OUTPUT_PATH}" \
        --num_frames          "${NUM_FRAMES}"  \
        --max_clips           "${MAX_CLIPS}"   \
        --max_new_tokens      "${MAX_NEW_TOKENS}" \
        --resume
    ;;

  smoke)
    # 3 samples per task, no compressor override — just confirm the pipeline runs
    OUTPUT_PATH="outputs/eval_vsibench/smoke/results.json"
    python eval/eval_vsibench.py \
        --model_path      "${MODEL_PATH}"  \
        --data_path       "${DATA_PATH}"   \
        --video_root      "${VIDEO_ROOT}"  \
        --output_path     "${OUTPUT_PATH}" \
        --num_frames      "${NUM_FRAMES}"  \
        --max_clips       "${MAX_CLIPS}"   \
        --max_new_tokens  "${MAX_NEW_TOKENS}" \
        --max_samples     3
    ;;

  *)
    echo "Unknown mode '${MODE}'. Choose: baseline | stage1_ft | smoke"
    exit 1
    ;;
esac
