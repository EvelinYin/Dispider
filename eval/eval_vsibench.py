#!/usr/bin/env python3
"""
VSI-Bench evaluation script for DiSPider.

Runs inference on all 5 VSI-Bench task types and reports task-specific metrics:
  - Categorical tasks (camera_movement_direction, camera_obj_rel_dist_v2,
    obj_obj_relative_pos_nf): accuracy
  - Numeric tasks (camera_displacement, camera_obj_abs_dist): mean relative
    error and accuracy within 10/20/50% thresholds

Supports two evaluation modes:
  1. Baseline  -- use the published Mar2Ding/Dispider weights unchanged
  2. Stage 1 finetuned -- load the full model but swap in your finetuned
     Decision Module via --compressor_override

Usage examples
--------------
# Baseline (paper model):
python eval/eval_vsibench.py \\
    --model_path Mar2Ding/Dispider \\
    --data_path  datasets/vsi-bench/my_qa/test/test_combined.json \\
    --video_root datasets/vsi-bench/scannet \\
    --output_path outputs/eval_vsibench/baseline/results.json

# With finetuned Stage 1 compressor:
python eval/eval_vsibench.py \\
    --model_path Mar2Ding/Dispider \\
    --compressor_override outputs/dispider_finetune/stage1/checkpoint-4143 \\
    --data_path  datasets/vsi-bench/my_qa/test/test_combined.json \\
    --video_root datasets/vsi-bench/scannet \\
    --output_path outputs/eval_vsibench/stage1_ft/results.json

# Quick smoke-test (3 samples per task):
python eval/eval_vsibench.py \\
    --model_path Mar2Ding/Dispider \\
    --max_samples 3 \\
    --output_path outputs/eval_vsibench/smoke/results.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Make sure the repo root is on sys.path regardless of cwd
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from PIL import Image
from decord import VideoReader
from transformers import StoppingCriteria, StoppingCriteriaList

from dispider.constants import (
    DEFAULT_ANS_TOKEN,
    DEFAULT_TODO_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from dispider.conversation import conv_templates
from dispider.model.builder import load_pretrained_model
from dispider.mm_utils import tokenizer_image_token, get_model_name_from_path
from dispider.utils import disable_torch_init


# ---------------------------------------------------------------------------
# Task metadata
# (For reference only; metric routing is determined by answer format, not task name)
# ---------------------------------------------------------------------------

NUMERIC_TASKS = frozenset({"camera_displacement", "camera_obj_abs_dist"})
CATEGORICAL_TASKS = frozenset(
    {"camera_movement_direction", "camera_obj_rel_dist_v2", "obj_obj_relative_pos_nf"}
)

# ---------------------------------------------------------------------------
# Official VSI-Bench metric helpers  (ported from livechat.py)
# ---------------------------------------------------------------------------

def _is_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def _fuzzy_match_pred(pred: str) -> str:
    """Extract first word and strip trailing period — matches VLM-3R / VSI-Bench logic."""
    return pred.strip().split(" ")[0].rstrip(".").strip()


def _exact_match(pred: str, target: str) -> float:
    """Case-insensitive exact match for MCA questions."""
    return 1.0 if pred.lower() == target.lower() else 0.0


def _mean_relative_accuracy(pred: float, target: float,
                             start: float = 0.5, end: float = 0.95,
                             interval: float = 0.05) -> float:
    """
    MRA metric for numerical answers — matches VLM-3R / VSI-Bench logic.

    Averages binary accuracy indicators across confidence thresholds
    [start, ..., end] (inclusive) at step `interval`.
    At threshold t the model is correct if |pred-target|/target <= (1-t).
    """
    num_pts = int((end - start) / interval + 2)
    conf_intervs = np.linspace(start, end, num_pts)
    if target == 0:
        score = 1.0 if pred == 0.0 else 0.0
        return float(score)
    abs_norm = abs(pred - target) / abs(target)
    accuracy = (abs_norm <= (1 - conf_intervs))
    return float(accuracy.mean())


# ---------------------------------------------------------------------------
# Stopping criteria
# ---------------------------------------------------------------------------

class _StopOnToken(StoppingCriteria):  # noqa: E302
    def __init__(self, stop_seqs):
        self.stop_seqs = stop_seqs  # list of 1-D tensors

    def __call__(self, input_ids, scores, **kwargs):
        for seq in self.stop_seqs:
            if torch.all(seq == input_ids[0, -len(seq):]):
                return True
        return False


# ---------------------------------------------------------------------------
# Video loading helpers (adapted from inference.py / model_videomme_long.py)
# ---------------------------------------------------------------------------

def _get_seq_frames(total, desired):
    seg = float(total - 1) / desired
    return [int((np.round(seg * i) + np.round(seg * (i + 1))) // 2) for i in range(desired)]


def _get_seq_time(vr, frame_idx, num_clip):
    fpc = len(frame_idx) // num_clip
    key_frames = [
        [frame_idx[i * fpc], frame_idx[i * fpc + fpc - 1]]
        for i in range(num_clip)
    ]
    ts = vr.get_frame_timestamp(key_frames)
    return np.hstack([ts[:, 0, 0], ts[:, 1, 1]])


def load_video(vis_path, num_frm=16, max_clip=32):
    """
    Returns
    -------
    frames : list[PIL.Image]  length = num_clip * num_frm
    time_idx : np.ndarray     shape [2 * num_clip]
    num_clip : int
    """
    vr = VideoReader(vis_path, num_threads=1)
    total_frames = len(vr)
    fps = vr.get_avg_fps()
    total_time = total_frames / fps

    num_clip = max(1, min(max_clip, int(round(total_time / num_frm))))
    total_num_frm = num_frm * num_clip
    frame_idx = _get_seq_frames(total_frames, total_num_frm)
    time_idx = _get_seq_time(vr, frame_idx, num_clip)

    img_array = vr.get_batch(frame_idx).asnumpy()          # [T, H, W, 3]
    H, W = img_array.shape[1], img_array.shape[2]
    if H != W:
        t = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
        s = min(H, W)
        t = F.interpolate(t, size=(s, s))
        img_array = t.permute(0, 2, 3, 1).to(torch.uint8).numpy()

    frames = [Image.fromarray(img_array[j]) for j in range(total_num_frm)]
    return frames, time_idx, num_clip


# ---------------------------------------------------------------------------
# Text preprocessing helpers (same as inference.py)
# ---------------------------------------------------------------------------

def _preprocess_time(time_idx, num_clip, time_tokenizer):
    """Build per-clip time-description token tensors."""
    t = time_idx.reshape(2, num_clip)
    seqs = []
    for i in range(num_clip):
        s, e = int(round(t[0, i])), int(round(t[1, i]))
        sentence = (
            f"This contains a clip sampled in {s} to {e} seconds"
            + DEFAULT_IMAGE_TOKEN
        )
        seqs.append(tokenizer_image_token(sentence, time_tokenizer, return_tensors="pt"))
    return seqs


def _preprocess_question(question, time_tokenizer):
    """Tokenise question + <to_do> for the compressor."""
    sentence = tokenizer_image_token(
        question + DEFAULT_TODO_TOKEN, time_tokenizer, return_tensors="pt"
    )
    return sentence.unsqueeze(0)  # [1, L]


def _timestamp_to_clip(timestamp: float, time_idx: np.ndarray, num_clips: int) -> int:
    """Map a question timestamp (seconds) to the clip index it falls in.

    Identical to dataset.py::timestamp_to_clip so eval matches training.
    """
    t = time_idx.reshape(2, num_clips)
    for i in range(num_clips):
        if timestamp <= t[1, i]:
            return i
    return num_clips - 1


# ---------------------------------------------------------------------------
# Single-sample inference
# ---------------------------------------------------------------------------

def run_inference(model, tokenizer, image_processor, time_tokenizer,
                  video_path, question, question_time, stopping_criteria, args):
    """
    Run the full DiSPider pipeline on one (video, question) pair.

    question_time : float  — seconds in the video when the question is asked.
                             Used to compute insert_position (matches training).
    Returns the raw decoded string from the reaction LLM.
    """
    frames, time_idx, num_clips = load_video(
        video_path, num_frm=args.num_frames, max_clip=args.max_clips
    )

    # Pixel values — same processor for small (compressor) and large (first frame)
    video = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
    video = video.view(num_clips, args.num_frames, *video.shape[1:])

    video_large = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
    video_large = (
        video_large.view(num_clips, args.num_frames, *video_large.shape[1:])[:, :1]
        .contiguous()
    )

    # Per-clip time descriptions for the compressor
    seqs = _preprocess_time(time_idx, num_clips, time_tokenizer)
    seqs = torch.nn.utils.rnn.pad_sequence(
        seqs, batch_first=True, padding_value=time_tokenizer.pad_token_id
    )
    compress_mask = seqs.ne(time_tokenizer.pad_token_id)

    # Question tokens for the compressor
    qs_t = _preprocess_question(question, time_tokenizer)
    qs_mask = qs_t.ne(time_tokenizer.pad_token_id)

    # Full prompt for the reaction LLM (Qwen chat template)
    qs_full = DEFAULT_IMAGE_TOKEN + "\n" + question
    conv = conv_templates["qwen"].copy()
    conv.append_message(conv.roles[0], qs_full)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids.to("cuda"),
            images=video.to(dtype=torch.float16, device="cuda"),
            images_large=video_large.to(dtype=torch.float16, device="cuda"),
            seqs=seqs.to("cuda"),
            compress_mask=compress_mask.to("cuda"),
            qs=qs_t.to("cuda"),
            qs_mask=qs_mask.to("cuda"),
            ans_token=time_tokenizer(DEFAULT_ANS_TOKEN, return_tensors="pt")
            .input_ids.to("cuda"),
            todo_token=time_tokenizer(DEFAULT_TODO_TOKEN, return_tensors="pt")
            .input_ids.to("cuda"),
            insert_position=_timestamp_to_clip(question_time, time_idx, num_clips),
            ans_position=[],
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
            use_cache=True,
        )

    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return output


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _extract_letter(text: str) -> str | None:
    """Extract the first A/B/C/D letter from a response string."""
    m = re.search(r"\b([A-D])\b", text)
    return m.group(1) if m else None


def _extract_number(text: str) -> float | None:
    """Extract the first numeric value (possibly decimal) from a response string."""
    m = re.search(r"[-+]?\d*\.?\d+", text)
    return float(m.group()) if m else None


def compute_metrics(results: list[dict]) -> dict:
    """
    Compute official VSI-Bench metrics, matching the logic in livechat.py.

    Parameters
    ----------
    results : list of dicts with keys
        video_uid, question_type, question, ground_truth, prediction

    Returns
    -------
    metrics : dict with keys
        mca_accuracy       -- MCA accuracy (%) across categorical tasks
        na_mra             -- Mean Relative Accuracy (%) across numeric tasks
        overall            -- unweighted average of the two above
        cat_{task}         -- per-task score (%)
        cat_{task}_n       -- sample count for that task
    """
    from collections import defaultdict

    mca_scores: list[float] = []
    na_scores: list[float] = []
    category_data: dict[str, list[float]] = defaultdict(list)

    for r in results:
        label = str(r["ground_truth"]).strip()
        pred_raw = r["prediction"]
        task = r["question_type"]

        if _is_numeric(label):
            # Numerical answer → MRA
            pred_text = _fuzzy_match_pred(pred_raw)
            try:
                pred_val = float(pred_text)
                target_val = float(label)
                score = _mean_relative_accuracy(pred_val, target_val)
            except (ValueError, TypeError):
                score = 0.0
            na_scores.append(score)
        else:
            # Multiple-choice answer → exact match (first word, case-insensitive)
            pred_text = _fuzzy_match_pred(pred_raw)
            score = _exact_match(pred_text, label)
            mca_scores.append(score)

        category_data[task].append(score)

    metrics: dict = {}
    if mca_scores:
        metrics["mca_accuracy"] = float(np.mean(mca_scores) * 100.0)
    if na_scores:
        metrics["na_mra"] = float(np.mean(na_scores) * 100.0)

    # Overall: unweighted average of MCA accuracy and NA MRA (matches VLM-3R)
    agg = [metrics[k] for k in ("mca_accuracy", "na_mra") if k in metrics]
    metrics["overall"] = float(np.mean(agg)) if agg else 0.0

    # Per-task breakdown
    for cat, scores in sorted(category_data.items()):
        metrics[f"cat_{cat}"] = float(np.mean(scores) * 100.0)
        metrics[f"cat_{cat}_n"] = len(scores)

    metrics["n_total"] = len(results)

    # time_diff: mean (answer_time - question_time) across samples — measures
    # how much temporal look-ahead context each question requires.
    time_diffs = [
        r["answer_time"] - r["question_time"]
        for r in results
        if "answer_time" in r and "question_time" in r
    ]
    if time_diffs:
        metrics["time_diff"] = float(np.mean(time_diffs))

    return metrics


def _print_metrics(metrics: dict) -> None:
    width = 60
    print("\n" + "=" * width)
    print("VSI-Bench Evaluation Results")
    print("=" * width)

    # Top-level summary
    top_keys = ["overall", "mca_accuracy", "na_mra", "time_diff", "n_total"]
    for k in top_keys:
        if k in metrics:
            v = metrics[k]
            if isinstance(v, float):
                print(f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v}")

    # Per-task
    print("\nPer-task breakdown:")
    for k, v in sorted(metrics.items()):
        if k.startswith("cat_") and not k.endswith("_n"):
            task = k[4:]
            n_key = f"cat_{task}_n"
            n = metrics.get(n_key, "?")
            if isinstance(v, float):
                print(f"  {task:40s}  {v:6.2f}%   (n={n})")
            else:
                print(f"  {task:40s}  {v}   (n={n})")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DiSPider on VSI-Bench",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path", default="Mar2Ding/Dispider",
        help="Full DiSPider model — local path or HuggingFace Hub ID",
    )
    parser.add_argument(
        "--compressor_override", default=None,
        help=(
            "Path to a finetuned Stage 1 checkpoint. When set, the model is loaded "
            "normally and then the embedded compressor is replaced with these weights."
        ),
    )
    parser.add_argument(
        "--data_path",
        default="datasets/vsi-bench/my_qa/test/test_combined.json",
    )
    parser.add_argument(
        "--video_root",
        default="datasets/vsi-bench/scannet",
    )
    parser.add_argument(
        "--output_path",
        default="outputs/eval_vsibench/results.json",
    )
    parser.add_argument("--num_frames", type=int, default=16,
                        help="Frames per clip")
    parser.add_argument("--max_clips", type=int, default=32,
                        help="Max clips per video")
    parser.add_argument("--max_new_tokens", type=int, default=64,
                        help="Max generation tokens per answer")
    parser.add_argument("--task", default=None,
                        help="Evaluate only this task type (omit for all)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit samples per task (useful for quick testing)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from partial results already written to output_path")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(f"Loading model from: {model_path}")
    # device_map=None avoids infer_auto_device_map which crashes on tied-weight
    # models; we move the model to GPU manually after loading.
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name, device_map=None
    )
    image_processor, time_tokenizer = image_processor
    if time_tokenizer.pad_token is None:
        time_tokenizer.pad_token = "<pad>"
    model.eval().cuda()

    # ------------------------------------------------------------------
    # Optionally swap in finetuned Stage 1 compressor
    # ------------------------------------------------------------------
    if args.compressor_override:
        ckpt_path = os.path.expanduser(args.compressor_override)
        print(f"Replacing compressor with finetuned Stage 1 from: {ckpt_path}")
        # TrainableStreamCompressor IS a StreamGroundQwenForCausalLM subclass,
        # so AutoModelForCausalLM resolves it correctly via the registered class.
        from dispider.model.language_model.trainable_compressor import (
            TrainableStreamCompressor,  # registers the class
        )
        from transformers import AutoModelForCausalLM

        stage1 = AutoModelForCausalLM.from_pretrained(
            ckpt_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=None,
        )
        # The compressor in the full model lives at model.model.compressor.compressor
        model.model.compressor.compressor = stage1.cuda()
        del stage1
        torch.cuda.empty_cache()
        print("Compressor replaced.")

    # ------------------------------------------------------------------
    # Stopping criteria
    # ------------------------------------------------------------------
    stop_ids = [torch.tensor(tokenizer("<|im_end|>").input_ids).cuda()]
    stopping_criteria = StoppingCriteriaList([_StopOnToken(stop_ids)])

    # ------------------------------------------------------------------
    # Load VSI-Bench data
    # ------------------------------------------------------------------
    data_path = os.path.join(_repo_root, args.data_path) if not os.path.isabs(args.data_path) else args.data_path
    with open(data_path) as f:
        raw = json.load(f)

    # raw is [[task1_items], [task2_items], ...]
    all_items = []
    for sublist in raw:
        for item in sublist:
            if args.task is None or item["question_type"] == args.task:
                all_items.append(item)
    print(f"Total items to evaluate: {len(all_items)}")

    # ------------------------------------------------------------------
    # Resume logic
    # ------------------------------------------------------------------
    output_path = (
        os.path.join(_repo_root, args.output_path)
        if not os.path.isabs(args.output_path)
        else args.output_path
    )
    results = []
    done_keys: set[tuple] = set()
    if args.resume and os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        for r in results:
            done_keys.add((r["video_uid"], r["question_type"], r["question"][:80]))
        print(f"Resuming: {len(done_keys)} items already done.")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    video_root = (
        os.path.join(_repo_root, args.video_root)
        if not os.path.isabs(args.video_root)
        else args.video_root
    )
    task_counts: dict[str, int] = {}

    for item in tqdm(all_items, desc="Evaluating"):
        task = item["question_type"]
        video_uid = item["video_uid"]
        question = item["conversation"][0]["content"]
        question_time = float(item["conversation"][0]["time"])
        gt_answer = item["conversation"][1]["content"]

        key = (video_uid, task, question[:80])
        if key in done_keys:
            continue

        # Per-task sample limit
        if args.max_samples is not None:
            task_counts.setdefault(task, 0)
            if task_counts[task] >= args.max_samples:
                continue
            task_counts[task] += 1

        video_path = os.path.join(video_root, video_uid + ".mp4")
        if not os.path.exists(video_path):
            print(f"[WARN] Video not found: {video_path}")
            continue

        try:
            prediction = run_inference(
                model, tokenizer, image_processor, time_tokenizer,
                video_path, question, question_time, stopping_criteria, args,
            )
        except Exception as exc:
            print(f"[ERROR] {video_uid} / {task}: {exc}")
            prediction = ""

        results.append(
            {
                "video_uid": video_uid,
                "question_type": task,
                "question": question,
                "question_time": question_time,
                "answer_time": float(item["conversation"][1]["time"]),
                "ground_truth": gt_answer,
                "prediction": prediction,
            }
        )
        done_keys.add(key)

        # Incremental save after every sample
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    # ------------------------------------------------------------------
    # Compute and report metrics
    # ------------------------------------------------------------------
    metrics = compute_metrics(results)
    _print_metrics(metrics)

    metrics_path = output_path.replace(".json", "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Raw predictions : {output_path}")
    print(f"Metrics         : {metrics_path}")


if __name__ == "__main__":
    main()
