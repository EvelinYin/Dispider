"""
curate_dataset.py
=================
Convert the nested vlm3r_ours/my_qa training JSON into the flat format
required by DispiderStage1Dataset and DispiderStage2Dataset.

Input schema (train_combined.json):
  list[5 task groups]
  Each group: list[samples]
  Each sample:
    {
      "video_uid": "scene0191_00",
      "conversation": [
        {"role": "user",      "content": "<question>", "time": 19.87},
        {"role": "assistant", "content": "<answer>",   "time": 28.39}
      ]
    }

Output schema (train_curated.json):
  list[samples]
  Each sample:
    {
      "video":         "scene0191_00.mp4",   # filename only; joined with video_root at load time
      "question":      "<question text>",
      "answer":        "<answer text>",
      "question_time": 19.87,               # seconds — when the question is posed
      "answer_time":   28.39               # seconds — when the answer should be given
    }

Usage:
  python curate_dataset.py \
      --input  /u/hyin2/videollm-online/datasets/vlm3r_ours/my_qa/train/train_combined.json \
      --output /u/hyin2/Dispider/datasets/train_curated.json \
      --video_root /work/nvme/bfhg/hyin2/datasets/vlm3r_videos/scannet/videos_2fps_max384 \
      --validate
"""

import os
import json
import argparse
import logging
from typing import List, Dict, Any

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def curate(
    input_path: str,
    video_root: str,
    validate: bool = False,
) -> List[Dict[str, Any]]:
    """Flatten nested JSON and convert to Dispider training format."""

    with open(input_path, "r") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError(f"Expected a list at top level, got {type(raw)}")

    # Determine if this is the nested (5-group) format or already flat
    first = raw[0]
    if isinstance(first, list):
        # Nested: list[groups] where each group is list[samples]
        all_samples = [sample for group in raw for sample in group]
        logger.info(f"Flattened {len(raw)} groups → {len(all_samples)} total samples")
    elif isinstance(first, dict):
        # Already flat
        all_samples = raw
        logger.info(f"Input is already flat: {len(all_samples)} samples")
    else:
        raise ValueError(f"Unexpected element type in top-level list: {type(first)}")

    curated: List[Dict[str, Any]] = []
    skipped_bad_conv = 0
    skipped_no_video = 0

    for sample in all_samples:
        video_uid = sample.get("video_uid", "")
        conversation = sample.get("conversation", [])

        # Expect exactly [user_turn, assistant_turn]
        if len(conversation) < 2:
            skipped_bad_conv += 1
            continue

        user_turn = conversation[0]
        asst_turn = conversation[1]

        if user_turn.get("role") != "user" or asst_turn.get("role") != "assistant":
            skipped_bad_conv += 1
            continue

        question = user_turn.get("content", "").strip()
        answer = asst_turn.get("content", "").strip()
        def _parse_time(val, default=0.0):
            s = str(val).strip().rstrip("s")
            try:
                return float(s)
            except (ValueError, TypeError):
                return float(default)

        question_time = _parse_time(user_turn.get("time", 0.0))
        answer_time = _parse_time(asst_turn.get("time", question_time), default=question_time)

        if not question or not answer or not video_uid:
            skipped_bad_conv += 1
            continue

        video_filename = video_uid + ".mp4"

        if validate:
            video_path = os.path.join(video_root, video_filename)
            if not os.path.exists(video_path):
                skipped_no_video += 1
                continue

        curated.append(
            {
                "video": video_filename,
                "question": question,
                "answer": answer,
                "question_time": question_time,
                "answer_time": answer_time,
            }
        )

    logger.info(f"Curated {len(curated)} samples")
    if skipped_bad_conv:
        logger.warning(f"Skipped {skipped_bad_conv} samples with malformed conversations")
    if skipped_no_video:
        logger.warning(f"Skipped {skipped_no_video} samples whose video files were not found on disk")

    # Statistics
    same_time = sum(1 for s in curated if s["question_time"] == s["answer_time"])
    diff_time = len(curated) - same_time
    logger.info(
        f"Timestamp breakdown: {same_time} retrospective (q_time==a_time), "
        f"{diff_time} forward-looking (q_time<a_time)"
    )

    return curated


def main():
    parser = argparse.ArgumentParser(
        description="Convert vlm3r_ours training JSON to Dispider flat format."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to train_combined.json (nested format).",
    )
    parser.add_argument(
        "--output",
        default="/u/hyin2/Dispider/datasets/train_curated.json",
        help="Output path for the curated JSON file.",
    )
    parser.add_argument(
        "--video_root",
        default="/work/nvme/bfhg/hyin2/datasets/vlm3r_videos/scannet/videos_2fps_max384",
        help="Root directory containing {video_uid}.mp4 files.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="If set, skip samples whose video file is not found under --video_root.",
    )
    args = parser.parse_args()

    curated = curate(args.input, args.video_root, validate=args.validate)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(curated, f, indent=2)

    logger.info(f"Saved {len(curated)} curated samples → {args.output}")

    # Print 3 samples for sanity check
    print("\n=== Sample output (first 3) ===")
    for s in curated[:3]:
        print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
