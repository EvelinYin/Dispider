#!/bin/bash

source ~/.bashrc
conda activate dispider


# =============================================================================
# Step 3: Stage 1 Training — Decision / Perception Module
# =============================================================================
# Trains the compact LLM (Qwen2-1.5B) + silent_head decision module.
# Vision encoder (SigLIP) is frozen.
# Loss: BCE (decision) + KL (temporal retrieval)
# =============================================================================

set -e
cd /u/hyin2/Dispider

export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=600
# export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True



MODEL_PATH="${MODEL_PATH:-Mar2Ding/Dispider}"
VIDEO_ROOT="${VIDEO_ROOT:-datasets/vlm3r_videos/scannet/videos_2fps_max384}"
DATA_PATH="${DATA_PATH:-datasets/train_curated.json}"
SCENE_SEP="${SCENE_SEP:-datasets/scene_sep.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/dispider_finetune/stage1}"

NUM_GPUS=${GPUS:-4}
MASTER_PORT=${MASTER_PORT:-29500}

echo "=== Step 3: Stage 1 Training ==="
echo "  Compressor: $MODEL_PATH"
echo "  Data:       $DATA_PATH"
echo "  Output:     $OUTPUT_DIR"
echo "  GPUs:       $NUM_GPUS"
echo ""

torchrun --nproc_per_node $NUM_GPUS --master_port $MASTER_PORT \
    train.py \
    --compressor         "$MODEL_PATH" \
    --data_path          "$DATA_PATH" \
    --video_root         "$VIDEO_ROOT" \
    --scene_sep_json     "$SCENE_SEP" \
    --stage              1 \
    --output_dir         "$OUTPUT_DIR" \
    --num_frames         16 \
    --max_clips          15 \
    --bf16               True \
    --num_train_epochs   1 \
    --per_device_train_batch_size   1 \
    --gradient_accumulation_steps   8 \
    --learning_rate      2e-5 \
    --weight_decay       0.01 \
    --warmup_ratio       0.03 \
    --lr_scheduler_type  cosine \
    --logging_steps      10 \
    --save_steps         500 \
    --save_total_limit   1 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --freeze_vision_encoder True \
    --report_to          tensorboard \
    # --resume_from_checkpoint True

echo ""
echo "Done. Stage 1 checkpoint saved to $OUTPUT_DIR"
