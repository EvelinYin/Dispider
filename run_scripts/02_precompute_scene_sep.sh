#!/bin/bash
# =============================================================================
# Step 2: Precompute Scene Boundaries
# =============================================================================
# Uses SigLIP embeddings to detect scene cuts in each video.
# Requires GPU and a Dispider checkpoint.
# Saves incrementally every 50 videos (resumable if interrupted).
#
# Input:  datasets/train_curated.json + video files
# Output: datasets/scene_sep.json
# =============================================================================

set -e
cd "$(dirname "$0")/.."

VIDEO_ROOT="${VIDEO_ROOT:-datasets/vlm3r_videos/scannet/videos_2fps_max384}"
DATA_PATH="${DATA_PATH:-datasets/train_curated.json}"
OUTPUT="${OUTPUT:-datasets/scene_sep.json}"

echo "=== Step 2: Precomputing scene boundaries ==="
echo "  Vision:     google/siglip-large-patch16-384 (from HF cache)"
echo "  Data:       $DATA_PATH"
echo "  Video root: $VIDEO_ROOT"
echo "  Output:     $OUTPUT"
echo ""

python precompute_scene_sep.py \
    --data_path    "$DATA_PATH" \
    --video_root   "$VIDEO_ROOT" \
    --output       "$OUTPUT" \
    --threshold    0.85 \
    --sample_fps   1.0 \
    --min_clip_sec 4.0

echo ""
echo "Done. Scene boundaries saved to $OUTPUT"
