"""
precompute_scene_sep.py
=======================
Pre-compute SigLIP-based scene boundaries for all unique videos in a
curated dataset JSON and save them to a lookup file consumed by
DispiderStage1Dataset / DispiderStage2Dataset at training time.

This implements the Scene-based Perception Module from the Dispider paper
(arXiv:2501.03218, Section 3.1):
  1. Sample frames at a regular interval.
  2. Extract L2-normalised embeddings with the pre-trained SigLIP vision tower.
  3. Compute cosine similarity between consecutive frame embeddings.
  4. Mark a scene boundary where similarity < threshold.
  5. Apply an exclusion window so clips are not excessively short.

Usage
-----
    python precompute_scene_sep.py \\
        --model_path  /path/to/dispider-checkpoint \\
        --data_path   /u/hyin2/Dispider/datasets/train_curated.json \\
        --video_root  /work/nvme/bfhg/hyin2/datasets/vlm3r_videos/scannet/videos_2fps_max384 \\
        --output      /u/hyin2/Dispider/datasets/scene_sep.json \\
        --threshold   0.85 \\
        --sample_fps  1.0 \\
        --min_clip_sec 4.0

Output
------
A JSON file mapping each video filename to a list of boundary timestamps:
    {
        "scene0191_00.mp4": [12.5, 28.0, 41.5],
        ...
    }
Pass this file to the dataset as ``scene_sep_json=<output>``.
"""

import os
import json
import argparse
import logging

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from decord import VideoReader

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def load_vision_tower_and_processor(model_path: str):
    """Load the SigLIP vision tower and its image processor.

    Only the vision encoder is needed for scene boundary detection, so we
    load SigLIP directly instead of instantiating the full 10B Dispider model.
    This avoids network calls for other sub-models (e.g. CLIP) and large
    memory overhead.
    """
    from transformers import AutoModel, AutoImageProcessor

    siglip_id = "google/siglip-large-patch16-384"
    logger.info(f"Loading SigLIP vision tower from {siglip_id} …")
    vision_tower = AutoModel.from_pretrained(
        siglip_id, torch_dtype=torch.float16
    ).vision_model.cuda().eval()
    image_processor = AutoImageProcessor.from_pretrained(siglip_id)

    return vision_tower, image_processor


@torch.no_grad()
def detect_scene_boundaries_siglip(
    video_path: str,
    vision_tower,
    image_processor,
    similarity_threshold: float = 0.85,
    sample_fps: float = 1.0,
    min_clip_seconds: float = 4.0,
    device: str = "cuda",
) -> list:
    """Detect scene cut timestamps using SigLIP embeddings.

    See dataset.py:detect_scene_boundaries_siglip() for full documentation.
    This version runs on GPU for efficiency during bulk preprocessing.
    """
    dtype = next(vision_tower.parameters()).dtype

    vr = VideoReader(video_path, num_threads=1)
    total_frames = len(vr)
    video_fps = vr.get_avg_fps()
    if video_fps <= 0 or total_frames < 2:
        return []

    step = max(int(round(video_fps / sample_fps)), 1)
    sample_indices = list(range(0, total_frames, step))
    if len(sample_indices) < 2:
        return []

    frames_np = vr.get_batch(sample_indices).asnumpy()
    pil_frames = [Image.fromarray(frames_np[i]) for i in range(len(sample_indices))]

    pixel_values = torch.cat([
        image_processor(img, return_tensors="pt")["pixel_values"]
        for img in pil_frames
    ], dim=0).to(device=device, dtype=dtype)

    out = vision_tower(pixel_values)
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        embeddings = out.pooler_output
    elif hasattr(out, "last_hidden_state"):
        embeddings = out.last_hidden_state[:, 0]
    else:
        embeddings = out[0][:, 0]

    embeddings = F.normalize(embeddings.float(), dim=-1)
    sim = (embeddings[:-1] * embeddings[1:]).sum(dim=-1).cpu().numpy()

    boundaries = []
    last_accepted = -min_clip_seconds
    for k, s in enumerate(sim):
        if s < similarity_threshold:
            t = sample_indices[k + 1] / video_fps
            if t - last_accepted >= min_clip_seconds:
                boundaries.append(round(t, 3))
                last_accepted = t

    return boundaries


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute SigLIP scene boundaries for Dispider training."
    )
    parser.add_argument("--model_path", default=None,
                        help="(Unused, kept for CLI compatibility.) "
                             "SigLIP is loaded directly from HF cache.")
    parser.add_argument("--data_path", required=True,
                        help="Path to train_curated.json.")
    parser.add_argument("--video_root", required=True,
                        help="Root directory containing {video}.mp4 files.")
    parser.add_argument("--output", required=True,
                        help="Output path for scene_sep.json.")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="Cosine similarity threshold for scene cut detection.")
    parser.add_argument("--sample_fps", type=float, default=1.0,
                        help="Frame sampling rate (fps) for boundary detection.")
    parser.add_argument("--min_clip_sec", type=float, default=4.0,
                        help="Minimum clip duration (exclusion window, seconds).")
    args = parser.parse_args()

    # Load unique video filenames from curated JSON
    with open(args.data_path, "r") as f:
        data = json.load(f)
    unique_videos = sorted({item["video"] for item in data})
    logger.info(f"Found {len(unique_videos)} unique videos in {args.data_path}")

    # Load SigLIP vision tower (from HF cache)
    vision_tower, image_processor = load_vision_tower_and_processor(args.model_path)
    device = next(vision_tower.parameters()).device

    # Load previously computed results if output exists (for resumption)
    scene_sep_map = {}
    if os.path.isfile(args.output):
        with open(args.output, "r") as f:
            scene_sep_map = json.load(f)
        logger.info(f"Loaded {len(scene_sep_map)} existing entries from {args.output}")

    skipped = 0
    processed = 0
    for video_filename in tqdm(unique_videos, desc="Computing scene boundaries"):
        if video_filename in scene_sep_map:
            skipped += 1
            continue

        video_path = os.path.join(args.video_root, video_filename)
        if not os.path.isfile(video_path):
            logger.warning(f"Video not found, skipping: {video_path}")
            scene_sep_map[video_filename] = []
            continue

        try:
            boundaries = detect_scene_boundaries_siglip(
                video_path,
                vision_tower,
                image_processor,
                similarity_threshold=args.threshold,
                sample_fps=args.sample_fps,
                min_clip_seconds=args.min_clip_sec,
                device=str(device),
            )
            scene_sep_map[video_filename] = boundaries
            processed += 1
        except Exception as e:
            logger.warning(f"Failed on {video_filename}: {e}")
            scene_sep_map[video_filename] = []

        # Save incrementally every 50 videos
        if (processed % 50) == 0 and processed > 0:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(scene_sep_map, f, indent=2)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(scene_sep_map, f, indent=2)

    logger.info(
        f"Done. Processed {processed} videos, skipped {skipped} (already cached)."
    )
    logger.info(f"Scene boundaries saved to {args.output}")

    # Quick stats
    n_with_cuts = sum(1 for v in scene_sep_map.values() if len(v) > 0)
    avg_cuts = np.mean([len(v) for v in scene_sep_map.values()])
    logger.info(
        f"Videos with ≥1 cut: {n_with_cuts}/{len(scene_sep_map)} "
        f"(avg {avg_cuts:.1f} cuts per video)"
    )


if __name__ == "__main__":
    main()
