#!/bin/bash
# =============================================================================
# Step 0: Download Dispider Checkpoint from HuggingFace
# =============================================================================
# Pre-downloads Mar2Ding/Dispider so subsequent steps don't download on the fly.
# Optional — the training scripts auto-download if the path doesn't exist locally.
# =============================================================================

set -e

REPO_ID="${REPO_ID:-Mar2Ding/Dispider}"

echo "=== Step 0: Pre-downloading Dispider checkpoint ==="
echo "  Repo: $REPO_ID"
echo ""

python -c "
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id='${REPO_ID}')
print(f'Downloaded to: {path}')
"

echo ""
echo "Done. The checkpoint is cached by huggingface_hub."
echo "Use --model_name_or_path Mar2Ding/Dispider or --compressor Mar2Ding/Dispider"
echo "in training scripts — they will resolve the cache path automatically."
