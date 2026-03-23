#!/bin/bash
#SBATCH --account=bfhg-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --mem=240g
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --job-name=dispider_stage2
#SBATCH --dependency=singleton
#SBATCH --open-mode=append

# =============================================================================
# Stage 2 SLURM wrapper
# =============================================================================
# Submit multiple times upfront to cover total training time across walltimes:
#   sbatch run_scripts/slurm_stage2.sh   # repeat as needed (e.g. 2x for ~4 days)
# --dependency=singleton ensures jobs run sequentially, never concurrently.
# On first job: starts training from scratch.
# On subsequent jobs: resumes from the latest saved checkpoint automatically.
# On clean finish: cancels any remaining queued jobs automatically.
# =============================================================================

source ~/.bashrc
conda activate dispider

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
NUM_GPUS=${GPUS:-4}
MASTER_PORT=${MASTER_PORT:-29500}

# Auto-detect latest checkpoint for seamless resume.
LATEST_CKPT=$(ls -td "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | head -1)
RESUME_ARG=""
if [ -n "$LATEST_CKPT" ]; then
    echo "=== Resuming from checkpoint: $LATEST_CKPT ==="
    RESUME_ARG="--resume_from_checkpoint $LATEST_CKPT"
else
    echo "=== Starting Stage 2 training from scratch ==="
fi

echo "  Model:        $MODEL_PATH"
echo "  Stage 1 ckpt: $STAGE1_CKPT"
echo "  Data:         $DATA_PATH"
echo "  Output:       $OUTPUT_DIR"
echo "  GPUs:         $NUM_GPUS"
echo "  Job ID:       ${SLURM_JOB_ID:-<interactive>}"
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
    --report_to          tensorboard \
    $RESUME_ARG
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=== Training complete. Stage 2 checkpoint saved to $OUTPUT_DIR ==="
    echo "=== Cancelling any queued jobs with name dispider_stage2 ==="
    scancel --jobname=dispider_stage2 --state=PENDING
else
    echo ""
    echo "=== Training exited with code $EXIT_CODE ==="
fi
