#!/bin/bash
# =============================================================================
# Stage 2 — Single-GPU correctness test
# Uses Adafactor optimizer to fit Qwen2-7B on one 40 GB GPU without DeepSpeed.
# Adafactor: ~2 GB optimizer states vs ~57 GB for Adam — no CUDA compilation needed.
# Runs only 10 steps — enough to verify the forward/backward pass works.
# =============================================================================

set -e
cd "$(dirname "$0")/.."

MODEL_PATH="${MODEL_PATH:-Mar2Ding/Dispider}"
STAGE1_CKPT="${STAGE1_CKPT:-outputs/dispider_finetune/stage1/checkpoint-4143}"
VIDEO_ROOT="${VIDEO_ROOT:-datasets/vlm3r_videos/scannet/videos_2fps_max384}"
DATA_PATH="${DATA_PATH:-datasets/train_curated.json}"
SCENE_SEP="${SCENE_SEP:-datasets/scene_sep.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/dispider_finetune/stage2_test}"
MASTER_PORT=${MASTER_PORT:-29501}

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Stage 2 Single-GPU Correctness Test (Adafactor, no DeepSpeed) ==="
echo "  Model:        $MODEL_PATH"
echo "  Stage 1 ckpt: $STAGE1_CKPT"
echo "  Output:       $OUTPUT_DIR"
echo ""

python train.py \
    --model_name_or_path "$MODEL_PATH" \
    --compressor         "$STAGE1_CKPT" \
    --data_path          "$DATA_PATH" \
    --video_root         "$VIDEO_ROOT" \
    --scene_sep_json     "$SCENE_SEP" \
    --stage              2 \
    --output_dir         "$OUTPUT_DIR" \
    --num_frames         16 \
    --max_clips          15 \
    --max_length         512 \
    --bf16               True \
    --max_steps          10 \
    --per_device_train_batch_size   1 \
    --gradient_accumulation_steps   1 \
    --optim              adafactor \
    --learning_rate      1e-5 \
    --weight_decay       0.0 \
    --warmup_ratio       0.0 \
    --lr_scheduler_type  constant \
    --logging_steps      1 \
    --save_steps         999999 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --freeze_compressor  True \
    --report_to          none

echo ""
echo "=== Test complete — if no crash, Stage 2 code is correct ==="
