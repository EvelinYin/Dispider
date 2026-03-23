#!/bin/bash
# =============================================================================
# Dispider Fine-tuning Launch Script
# =============================================================================
# Two-stage training:
#   Stage 1 — trains compact LLM + decision module (CLIP frozen)
#   Stage 2 — trains Qwen2-7B backbone + projectors (compressor frozen)
#
# Preparation steps (run once before this script):
#   1. python curate_dataset.py \
#          --input  /u/hyin2/videollm-online/datasets/vlm3r_ours/my_qa/train/train_combined.json \
#          --output /u/hyin2/Dispider/datasets/train_curated.json \
#          --video_root $VIDEO_ROOT --validate
#
#   2. python precompute_scene_sep.py \
#          --model_path $MODEL_PATH \
#          --data_path  /u/hyin2/Dispider/datasets/train_curated.json \
#          --video_root $VIDEO_ROOT \
#          --output     /u/hyin2/Dispider/datasets/scene_sep.json
# =============================================================================

set -e

# --------------- Paths (edit these) ---------------
MODEL_PATH="YOUR_DISPIDER_CKPT_PATH"
DATA_PATH="/u/hyin2/Dispider/datasets/train_curated.json"
SCENE_SEP="/u/hyin2/Dispider/datasets/scene_sep.json"
VIDEO_ROOT="/work/nvme/bfhg/hyin2/datasets/vlm3r_videos/scannet/videos_2fps_max384"
OUTPUT_DIR="./outputs/dispider_finetune"

# --------------- Hardware ---------------
NUM_GPUS=${GPUS:-8}
MASTER_PORT=${MASTER_PORT:-29500}

# =============================================================================
# Stage 1: Decision / Perception Module Training
# =============================================================================
echo "============================================"
echo "  Stage 1: Training Decision Module"
echo "============================================"

deepspeed --num_gpus $NUM_GPUS --master_port $MASTER_PORT \
    train.py \
    --compressor         $MODEL_PATH \
    --data_path          $DATA_PATH \
    --video_root         $VIDEO_ROOT \
    --scene_sep_json     $SCENE_SEP \
    --stage              1 \
    --output_dir         ${OUTPUT_DIR}/stage1 \
    --num_frames         16 \
    --max_clips          100 \
    --bf16               True \
    --num_train_epochs   3 \
    --per_device_train_batch_size   1 \
    --gradient_accumulation_steps   8 \
    --learning_rate      2e-5 \
    --weight_decay       0.01 \
    --warmup_ratio       0.03 \
    --lr_scheduler_type  cosine \
    --logging_steps      10 \
    --save_steps         500 \
    --save_total_limit   3 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --freeze_vision_encoder True \
    --deepspeed          scripts/ds_config_zero3.json \
    --report_to          tensorboard

# =============================================================================
# Stage 2: Reaction / LLM Training
# =============================================================================
echo "============================================"
echo "  Stage 2: Training Reaction LLM"
echo "============================================"

STAGE1_CKPT="${OUTPUT_DIR}/stage1"

deepspeed --num_gpus $NUM_GPUS --master_port $MASTER_PORT \
    train.py \
    --model_name_or_path $MODEL_PATH \
    --compressor         $STAGE1_CKPT \
    --data_path          $DATA_PATH \
    --video_root         $VIDEO_ROOT \
    --scene_sep_json     $SCENE_SEP \
    --stage              2 \
    --output_dir         ${OUTPUT_DIR}/stage2 \
    --num_frames         16 \
    --max_clips          100 \
    --max_length         2048 \
    --bf16               True \
    --num_train_epochs   2 \
    --per_device_train_batch_size   1 \
    --gradient_accumulation_steps   8 \
    --learning_rate      1e-5 \
    --weight_decay       0.01 \
    --warmup_ratio       0.03 \
    --lr_scheduler_type  cosine \
    --logging_steps      10 \
    --save_steps         500 \
    --save_total_limit   3 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --freeze_compressor  True \
    --deepspeed          scripts/ds_config_zero3.json \
    --report_to          tensorboard

echo "============================================"
echo "  Training complete!"
echo "  Stage 1 checkpoint: ${OUTPUT_DIR}/stage1"
echo "  Stage 2 checkpoint: ${OUTPUT_DIR}/stage2"
echo "============================================"
