#!/bin/bash

source ~/.bashrc
conda activate dispider
# =============================================================================
# Step 4: Stage 2 Training — Reaction LLM
# =============================================================================
# Trains Qwen2-7B backbone + projectors (compress_projector, clip_projector).
# Compressor (from Stage 1) is frozen.
# Loss: Cross-entropy (next-token prediction)
#
# NOTE: Uses torchrun DDP (not DeepSpeed ZeRO-3) + Adafactor optimizer.
# ZeRO-3 deadlocks because the frozen compressor's forward_token_stream makes
# a variable number of self.model() calls per sample (2 + len(ans_position)).
# Different ranks hold different samples → different call counts → all-gather
# deadlock. DDP keeps full model params on each GPU, so no collective ops are
# needed during the forward pass — zero deadlock risk.
# Adafactor stores factored second moments (~2 GB) vs Adam fp32 (~57 GB),
# keeping total GPU memory ~35 GB/GPU, well within the 40 GB A40 limit.
# =============================================================================

set -e
cd /u/hyin2/Dispider

export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=600
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH="${MODEL_PATH:-Mar2Ding/Dispider}"
STAGE1_CKPT="${STAGE1_CKPT:-outputs/dispider_finetune/stage1/checkpoint-4143}"
VIDEO_ROOT="${VIDEO_ROOT:-datasets/vlm3r_videos/scannet/videos_2fps_max384}"
DATA_PATH="${DATA_PATH:-datasets/train_curated.json}"
SCENE_SEP="${SCENE_SEP:-datasets/scene_sep.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/dispider_finetune/stage2}"

NUM_GPUS=${GPUS:-1}
MASTER_PORT=${MASTER_PORT:-29500}

echo "=== Step 4: Stage 2 Training ==="
echo "  Model:        $MODEL_PATH"
echo "  Stage 1 ckpt: $STAGE1_CKPT"
echo "  Data:         $DATA_PATH"
echo "  Output:       $OUTPUT_DIR"
echo "  GPUs:         $NUM_GPUS"
echo ""

torchrun --nproc_per_node $NUM_GPUS --master_port $MASTER_PORT \
    train.py \
    --model_name_or_path "$MODEL_PATH" \
    --compressor         "$STAGE1_CKPT" \
    --data_path          "$DATA_PATH" \
    --video_root         "$VIDEO_ROOT" \
    --scene_sep_json     "$SCENE_SEP" \
    --stage              2 \
    --output_dir         "$OUTPUT_DIR" \
    --num_frames         16 \
    --max_clips          15 \
    --max_length         2048 \
    --bf16               True \
    --num_train_epochs   2 \
    --per_device_train_batch_size   1 \
    --gradient_accumulation_steps   8 \
    --optim              adafactor \
    --learning_rate      1e-5 \
    --weight_decay       0.0 \
    --warmup_ratio       0.03 \
    --lr_scheduler_type  cosine \
    --logging_steps      10 \
    --save_steps         500 \
    --save_total_limit   3 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --freeze_compressor  True \
    --report_to          tensorboard

echo ""
echo "Done. Stage 2 checkpoint saved to $OUTPUT_DIR"
