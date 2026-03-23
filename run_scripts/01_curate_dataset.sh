#!/bin/bash
# =============================================================================
# Step 1: Curate Dataset
# =============================================================================
# Flattens the nested vlm3r_ours train_combined.json (5 task groups) into
# the flat format required by Dispider training.
#
# Input:  datasets/vlm3r_ours/my_qa/train/train_combined.json
# Output: datasets/train_curated.json
# =============================================================================

set -e
cd "$(dirname "$0")/.."

VIDEO_ROOT="${VIDEO_ROOT:-datasets/vlm3r_videos/scannet/videos_2fps_max384}"
INPUT="${INPUT:-datasets/vlm3r_ours/my_qa/train/train_combined.json}"
OUTPUT="${OUTPUT:-datasets/train_curated.json}"

echo "=== Step 1: Curating dataset ==="
echo "  Input:      $INPUT"
echo "  Output:     $OUTPUT"
echo "  Video root: $VIDEO_ROOT"
echo ""

python curate_dataset.py \
    --input      "$INPUT" \
    --output     "$OUTPUT" \
    --video_root "$VIDEO_ROOT" \
    --validate

echo ""
echo "Done. Curated dataset saved to $OUTPUT"
