"""
Streaming-aware dataset for Dispider training.
Handles the curated flat-JSON format produced by curate_dataset.py for both
Stage 1 (Decision/Perception) and Stage 2 (Reaction/LLM).

Curated JSON schema (train_curated.json):
  list[{
      "video":         "scene0191_00.mp4",
      "question":      "...",
      "answer":        "...",
      "question_time": 19.87,
      "answer_time":   28.39
  }]

Scene boundary pre-computation
-------------------------------
This module provides detect_scene_boundaries_siglip() which implements the
paper's scene segmentation approach (SigLIP embeddings + cosine similarity).
Because running a vision forward pass inside __getitem__ is too slow for
training, boundaries should be pre-computed once with the helper script:

    python precompute_scene_sep.py \\
        --model_path  <dispider-checkpoint> \\
        --data_path   datasets/train_curated.json \\
        --video_root  <video-root> \\
        --output      datasets/scene_sep.json

Both DispiderStage1Dataset and DispiderStage2Dataset accept a scene_sep_json
argument.  If that file is present each video's boundaries are looked up from
it; otherwise uniform temporal sampling is used as a fallback.
"""

import os
import json
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from decord import VideoReader

from dispider.constants import (
    IGNORE_INDEX, IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
    DEFAULT_ANS_TOKEN, DEFAULT_TODO_TOKEN, DEFAULT_SILENT_TOKEN,
)
from dispider.conversation import conv_templates
from dispider.mm_utils import tokenizer_image_token


# ---------------------------------------------------------------------------
# Scene boundary detection  (mirrors paper Section 3.1)
# ---------------------------------------------------------------------------

def detect_scene_boundaries_siglip(
    video_path: str,
    vision_tower,
    image_processor,
    similarity_threshold: float = 0.85,
    sample_fps: float = 1.0,
    min_clip_seconds: float = 4.0,
) -> List[float]:
    """Detect scene-cut timestamps using SigLIP embeddings.

    Implements the approach described in the paper:
    1. Sample frames at ``sample_fps`` (paper: regular interval).
    2. Extract L2-normalised feature embeddings using the pre-trained SigLIP
       model (``vision_tower``).
    3. Compute cosine similarity between consecutive frame embeddings.
    4. Mark a scene boundary wherever similarity < ``similarity_threshold``.
    5. Apply an exclusion window: remove boundaries that are closer than
       ``min_clip_seconds`` to the previous accepted boundary, so that
       resulting clips are not excessively short.

    Args:
        video_path:           Path to the video file.
        vision_tower:         The SigLIP / CLIP vision encoder from the
                              Dispider model (must support a forward call that
                              returns per-image features, e.g. CLIPVisionModel).
        image_processor:      Matching HF image processor / feature extractor.
        similarity_threshold: Cosine-similarity threshold below which a frame
                              transition is considered a scene cut (default 0.85).
        sample_fps:           Frame sampling rate for boundary detection.
        min_clip_seconds:     Minimum duration (seconds) between two accepted
                              boundaries (exclusion window, default 4 s).

    Returns:
        List of scene boundary timestamps in seconds, suitable for
        ``load_video(..., scene_sep=...)``.  Empty list = no detected cuts.
    """
    device = next(vision_tower.parameters()).device
    dtype = next(vision_tower.parameters()).dtype

    vr = VideoReader(video_path, num_threads=1)
    total_frames = len(vr)
    video_fps = vr.get_avg_fps()
    if video_fps <= 0 or total_frames < 2:
        return []

    # Sample one frame every (video_fps / sample_fps) original frames
    step = max(int(round(video_fps / sample_fps)), 1)
    sample_indices = list(range(0, total_frames, step))
    if len(sample_indices) < 2:
        return []

    frames_np = vr.get_batch(sample_indices).asnumpy()  # (N, H, W, 3)

    # Process frames through the SigLIP image processor
    pil_frames = [Image.fromarray(frames_np[i]) for i in range(len(sample_indices))]
    pixel_values = image_processor.preprocess(
        pil_frames, return_tensors="pt"
    )["pixel_values"].to(device=device, dtype=dtype)

    # Extract L2-normalised embeddings
    with torch.no_grad():
        # vision_tower returns (features,) or a BaseModelOutput — take pooled/last token
        out = vision_tower(pixel_values)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            embeddings = out.pooler_output  # (N, D)
        elif hasattr(out, "last_hidden_state"):
            embeddings = out.last_hidden_state[:, 0]  # CLS token, (N, D)
        else:
            embeddings = out[0][:, 0]

    embeddings = F.normalize(embeddings.float(), dim=-1)  # L2 normalise

    # Cosine similarity between consecutive sampled frames
    sim = (embeddings[:-1] * embeddings[1:]).sum(dim=-1).cpu().numpy()  # (N-1,)

    # Find cuts: frames where similarity drops below threshold
    cut_mask = sim < similarity_threshold  # True where there is a scene change

    # Convert sample indices to timestamps; apply exclusion window
    boundaries: List[float] = []
    last_accepted = -min_clip_seconds  # allow first boundary at t=0+
    for k, is_cut in enumerate(cut_mask):
        if not is_cut:
            continue
        # Timestamp corresponds to the start of the *next* sampled frame
        t = sample_indices[k + 1] / video_fps
        if t - last_accepted >= min_clip_seconds:
            boundaries.append(t)
            last_accepted = t

    return boundaries


# ---------------------------------------------------------------------------
# Video loading utilities  (mirrors inference.py)
# ---------------------------------------------------------------------------

def get_seq_frames(total_num_frames: int, desired_num_frames: int) -> List[int]:
    seg_size = float(total_num_frames - 1) / desired_num_frames
    seq = []
    for i in range(desired_num_frames):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        seq.append((start + end) // 2)
    return seq


def get_seq_time(vr, frame_idx: List[int], num_clip: int) -> np.ndarray:
    frm_per_clip = len(frame_idx) // num_clip
    key_frame = [
        [frame_idx[i * frm_per_clip], frame_idx[i * frm_per_clip + frm_per_clip - 1]]
        for i in range(num_clip)
    ]
    time = vr.get_frame_timestamp(key_frame)
    return np.hstack([time[:, 0, 0], time[:, 1, 1]])


def calculate_diff(scene_sep: List[int], start_frame: int) -> List[int]:
    diff = [scene_sep[0] - start_frame]
    for i in range(len(scene_sep) - 1):
        diff.append(scene_sep[i + 1] - scene_sep[i])
    return diff


def load_video(
    vis_path: str,
    scene_sep: List[float],
    num_frm: int = 16,
    max_clip: int = 100,
    sample_frame=None,
):
    """Load video frames and temporal metadata.  Mirrors inference.py load_video.

    Args:
        vis_path:     Path to the video file.
        scene_sep:    Scene boundary timestamps (seconds) from
                      detect_scene_boundaries_siglip().  Pass [] for uniform sampling.
        num_frm:      Frames per clip.
        max_clip:     Maximum number of clips.
        sample_frame: Optional [(start_frame, end_frame)] range restriction.

    Returns:
        clip_imgs  – list of PIL Images (total_num_frm images)
        time_idx   – np.ndarray of shape (2*num_clip,) [start_times … end_times]
        num_clip   – int
        num_frm    – int (same as input, returned for convenience)
    """
    block_size = 1
    vr = VideoReader(vis_path, num_threads=1)
    total_frame_num = (
        len(vr) if sample_frame is None else (sample_frame[0][1] - sample_frame[0][0])
    )
    fps = vr.get_avg_fps()
    total_time = total_frame_num / fps

    frame_idx: List[int] = []

    if len(scene_sep) == 0:
        # Uniform temporal sampling (no scene boundaries)
        num_clip = total_time / num_frm
        num_clip = (
            int(block_size * np.round(num_clip / block_size))
            if num_clip > block_size
            else int(np.round(num_clip))
        )
        num_clip = max(num_clip, 1)
        num_clip = min(num_clip, max_clip)
        total_num_frm = num_frm * num_clip
        frame_idx = get_seq_frames(total_frame_num, total_num_frm)
    else:
        # Scene-boundary-aware clip segmentation
        end_frame = total_frame_num if sample_frame is None else sample_frame[0][1]
        scene_sep_frames: List[int] = []
        for ele in scene_sep:
            sep = int(fps * (ele + 1))
            sep = min(sep, end_frame - 1)
            scene_sep_frames.append(sep)
        scene_sep_frames.append(end_frame - 1)

        if len(scene_sep_frames) > max_clip:
            diff = calculate_diff(scene_sep_frames, start_frame=0)
            min_idx = np.argsort(diff[:-1])[: len(scene_sep_frames) - max_clip]
            for i in np.sort(min_idx)[::-1]:
                del scene_sep_frames[i]

        num_clip = len(scene_sep_frames)
        total_num_frm = num_frm * num_clip
        start_ = 0
        for end_f in scene_sep_frames:
            idx_list = np.linspace(start_, end_f, num=num_frm, endpoint=False)
            frame_idx.extend([int(id_) for id_ in idx_list])
            start_ = end_f

    time_idx = get_seq_time(vr, frame_idx, num_clip)
    img_array = vr.get_batch(frame_idx).asnumpy()  # (total_num_frm, H, W, 3)

    a, H, W, _ = img_array.shape
    if H != W:
        img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
        img_array = torch.nn.functional.interpolate(
            img_array, size=(min(H, W), min(H, W))
        )
        img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()

    clip_imgs = [Image.fromarray(img_array[j]) for j in range(total_num_frm)]
    return clip_imgs, time_idx, num_clip, num_frm


# ---------------------------------------------------------------------------
# Temporal helpers
# ---------------------------------------------------------------------------

def timestamp_to_clip(timestamp: float, time_idx: np.ndarray, num_clips: int) -> int:
    """Map a timestamp (seconds) to its containing clip index."""
    time = time_idx.reshape(2, num_clips)
    for i in range(num_clips):
        if timestamp <= time[1, i]:
            return i
    return num_clips - 1


# ---------------------------------------------------------------------------
# Sequence preprocessing  (mirrors inference.py exactly)
# ---------------------------------------------------------------------------

def preprocess_time(time: np.ndarray, num_clip: int, tokenizer) -> List[torch.Tensor]:
    """Tokenize temporal descriptions for each clip."""
    time = time.reshape(2, num_clip)
    seq = []
    for i in range(num_clip):
        start = int(np.round(time[0, i]))
        end = int(np.round(time[1, i]))
        sentence = (
            f"This contains a clip sampled in {start} to {end} seconds"
            + DEFAULT_IMAGE_TOKEN
        )
        sentence = tokenizer_image_token(sentence, tokenizer, return_tensors="pt")
        seq.append(sentence)
    return seq


def preprocess_question(questions: List[str], tokenizer) -> List[torch.Tensor]:
    """Tokenize questions with <to_do> appended."""
    seq = []
    for q in questions:
        sentence = tokenizer_image_token(
            q + DEFAULT_TODO_TOKEN, tokenizer, return_tensors="pt"
        )
        seq.append(sentence)
    return seq


# ---------------------------------------------------------------------------
# Conversation prompt builder  (mirrors inference.py process_data)
# ---------------------------------------------------------------------------

def build_conversation_prompt(
    question: str, answer: Optional[str], model_config
) -> str:
    """Build a Qwen chat-template prompt.  answer=None yields the partial
    prompt (instruction only) used for label masking."""
    if getattr(model_config, "mm_use_im_start_end", False):
        qs = (
            DEFAULT_IM_START_TOKEN
            + DEFAULT_IMAGE_TOKEN
            + DEFAULT_IM_END_TOKEN
            + "\n"
            + question
        )
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + question

    conv = conv_templates["qwen"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], answer)
    return conv.get_prompt()


# ---------------------------------------------------------------------------
# Dataset: Stage 1 – Decision / Perception Training
# ---------------------------------------------------------------------------


class DispiderStage1Dataset(Dataset):
    """
    Stage 1 trains the compact LLM compressor + silent_head decision module.
    Reads the curated flat JSON and synthesises all temporal training labels
    from question_time / answer_time at load time.

    Scene boundaries
    ----------------
    Pass ``scene_sep_json`` (path to a pre-computed JSON produced by
    precompute_scene_sep.py) to use SigLIP-based scene segmentation as
    described in the paper.  Without it, uniform temporal sampling is used.

    Synthesised labels
    ------------------
    insert_position  — clip index when the question is first heard
    ans_position     — [clip index when the answer should be triggered]
    silent_position  — clip indices between question and answer (model stays silent)
    time_labels      — uniform distribution over [0 … ans_clip], shape (1, num_clips)
    """

    def __init__(
        self,
        data_path: str,
        video_root: str,
        image_processor,
        time_tokenizer,
        tokenizer,
        model_config,
        num_frames: int = 16,
        max_clips: int = 100,
        scene_sep_json: Optional[str] = None,
    ):
        super().__init__()
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.video_root = video_root
        self.image_processor = image_processor
        self.time_tokenizer = time_tokenizer
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.num_frames = num_frames
        self.max_clips = max_clips

        # Load precomputed scene boundaries if provided
        self._scene_sep: Dict[str, List[float]] = {}
        if scene_sep_json and os.path.isfile(scene_sep_json):
            with open(scene_sep_json, "r") as f:
                self._scene_sep = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict:
        item = self.data[idx]
        video_filename = item["video"]
        video_path = os.path.join(self.video_root, video_filename)

        # Look up precomputed scene boundaries; fall back to uniform sampling
        scene_sep = self._scene_sep.get(video_filename, [])
        clip_imgs, time_idx, num_clips, num_frm = load_video(
            video_path,
            scene_sep=scene_sep,
            num_frm=self.num_frames,
            max_clip=self.max_clips,
        )

        # Process frames (same processor for small and large paths, mirrors inference.py)
        video = self.image_processor.preprocess(clip_imgs, return_tensors="pt")[
            "pixel_values"
        ]
        video = video.view(num_clips, num_frm, *video.shape[1:])

        # Time sequences for compressor
        seqs = preprocess_time(time_idx, num_clips, self.time_tokenizer)
        seqs = torch.nn.utils.rnn.pad_sequence(
            seqs,
            batch_first=True,
            padding_value=self.time_tokenizer.pad_token_id,
        )
        compress_mask = seqs.ne(self.time_tokenizer.pad_token_id)

        # Question tokens for decision module
        questions = preprocess_question([item["question"]], self.time_tokenizer)
        questions = torch.nn.utils.rnn.pad_sequence(
            questions,
            batch_first=True,
            padding_value=self.time_tokenizer.pad_token_id,
        )
        qs_mask = questions.ne(self.time_tokenizer.pad_token_id)

        # --- Synthesise temporal training labels from timestamps ---
        question_time = float(item.get("question_time", 0.0))
        answer_time = float(item.get("answer_time", question_time))

        insert_position = timestamp_to_clip(question_time, time_idx, num_clips)
        ans_clip = timestamp_to_clip(answer_time, time_idx, num_clips)

        # First-turn single-QA: no prior answers → ans_position is empty.
        # forward_grounding_stream() handles ans_position=[] correctly.
        ans_position = []

        # silent_label: shape [num_clips - insert_position + 1].
        # Convention (matching forward_grounding_stream in modeling_qwen.py):
        #   1.0 = "respond" (model should answer here; KL loss is also computed)
        #   0.0 = "stay silent"
        n_eval = num_clips - insert_position + 1
        silent_label_vec = np.zeros(n_eval, dtype=np.float32)
        respond_idx = ans_clip - insert_position
        if 0 <= respond_idx < n_eval:
            silent_label_vec[respond_idx] = 1.0
        silent_label = torch.tensor(silent_label_vec, dtype=torch.float32)

        # time_labels: uniform distribution over [0 … ans_clip].  Shape [1, num_clips].
        time_label_vec = np.zeros(num_clips, dtype=np.float32)
        time_label_vec[: ans_clip + 1] = 1.0
        if time_label_vec.sum() > 0:
            time_label_vec /= time_label_vec.sum()
        time_labels = torch.tensor(time_label_vec, dtype=torch.float32).unsqueeze(0)

        # Stage 1 does NOT train the reaction LLM — all labels are IGNORE
        prompt = build_conversation_prompt(item["question"], None, self.model_config)
        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        labels = torch.full_like(input_ids, IGNORE_INDEX)

        # Special tokens (same as inference.py)
        ans_token = self.time_tokenizer(DEFAULT_ANS_TOKEN, return_tensors="pt").input_ids
        todo_token = self.time_tokenizer(DEFAULT_TODO_TOKEN, return_tensors="pt").input_ids

        return dict(
            input_ids=input_ids,
            labels=labels,
            images=video,
            images_large=video[:, :1].contiguous(),
            seqs=seqs,
            compress_mask=compress_mask,
            qs=questions,
            qs_mask=qs_mask,
            time_labels=time_labels,
            ans_token=ans_token,
            todo_token=todo_token,
            insert_position=insert_position,
            ans_position=ans_position,
            silent_label=silent_label,
        )


# ---------------------------------------------------------------------------
# Dataset: Stage 2 – Reaction / LLM Training
# ---------------------------------------------------------------------------


class DispiderStage2Dataset(Dataset):
    """
    Stage 2 freezes the compressor + decision module and trains the Qwen2-7B backbone.
    Reads the same curated flat JSON format as Stage 1.

    Pass ``scene_sep_json`` (produced by precompute_scene_sep.py) for
    SigLIP-based scene segmentation; falls back to uniform sampling otherwise.
    """

    def __init__(
        self,
        data_path: str,
        video_root: str,
        image_processor,
        time_tokenizer,
        tokenizer,
        model_config,
        num_frames: int = 16,
        max_clips: int = 100,
        max_length: int = 2048,
        scene_sep_json: Optional[str] = None,
    ):
        super().__init__()
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.video_root = video_root
        self.image_processor = image_processor
        self.time_tokenizer = time_tokenizer
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.num_frames = num_frames
        self.max_clips = max_clips
        self.max_length = max_length

        self._scene_sep: Dict[str, List[float]] = {}
        if scene_sep_json and os.path.isfile(scene_sep_json):
            with open(scene_sep_json, "r") as f:
                self._scene_sep = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict:
        item = self.data[idx]
        video_filename = item["video"]
        video_path = os.path.join(self.video_root, video_filename)

        scene_sep = self._scene_sep.get(video_filename, [])
        clip_imgs, time_idx, num_clips, num_frm = load_video(
            video_path,
            scene_sep=scene_sep,
            num_frm=self.num_frames,
            max_clip=self.max_clips,
        )

        # Process frames
        video = self.image_processor.preprocess(clip_imgs, return_tensors="pt")[
            "pixel_values"
        ]
        video = video.view(num_clips, num_frm, *video.shape[1:])

        # Time sequences
        seqs = preprocess_time(time_idx, num_clips, self.time_tokenizer)
        seqs = torch.nn.utils.rnn.pad_sequence(
            seqs,
            batch_first=True,
            padding_value=self.time_tokenizer.pad_token_id,
        )
        compress_mask = seqs.ne(self.time_tokenizer.pad_token_id)

        question = item["question"]
        answer = item["answer"]
        question_time = float(item.get("question_time", 0.0))

        insert_position = timestamp_to_clip(question_time, time_idx, num_clips)

        # Question tokens for compressor decision module
        questions = preprocess_question([question], self.time_tokenizer)
        questions = torch.nn.utils.rnn.pad_sequence(
            questions,
            batch_first=True,
            padding_value=self.time_tokenizer.pad_token_id,
        )
        qs_mask = questions.ne(self.time_tokenizer.pad_token_id)

        # Build full prompt (with answer) and partial prompt (instruction only).
        # Tokenising both gives us a tokeniser-accurate instruction boundary
        # so that label masking is correct regardless of subword tokenisation.
        full_prompt = build_conversation_prompt(question, answer, self.model_config)
        partial_prompt = build_conversation_prompt(question, None, self.model_config)

        input_ids = tokenizer_image_token(
            full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        partial_ids = tokenizer_image_token(
            partial_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )

        # Mask instruction; supervise answer tokens only
        labels = input_ids.clone()
        labels[: partial_ids.shape[0]] = IGNORE_INDEX

        # Truncate to max_length
        if input_ids.shape[0] > self.max_length:
            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]

        # time_labels: zeros for Stage 2 (decision module is frozen, not trained)
        time_labels = torch.zeros(1, num_clips, dtype=torch.float32)

        # Special tokens
        ans_token = self.time_tokenizer(DEFAULT_ANS_TOKEN, return_tensors="pt").input_ids
        todo_token = self.time_tokenizer(DEFAULT_TODO_TOKEN, return_tensors="pt").input_ids

        return dict(
            input_ids=input_ids,
            labels=labels,
            images=video,
            images_large=video[:, :1].contiguous(),
            seqs=seqs,
            compress_mask=compress_mask,
            qs=questions,
            qs_mask=qs_mask,
            time_labels=time_labels,
            ans_token=ans_token,
            todo_token=todo_token,
            insert_position=insert_position,
            ans_position=[],
            silent_position=[],
            # No silent_label for Stage 2 — compressor is frozen, decision
            # module is not trained.  LongQwen2ForCausalLM.forward() never
            # invokes forward_grounding_stream() when silent_label is absent.
        )


# ---------------------------------------------------------------------------
# Collator – handles variable-length padding for the DataLoader
# ---------------------------------------------------------------------------


class DispiderDataCollator:
    """Collate function for Dispider datasets.
    input_ids and labels receive an unsqueeze(0) batch dimension;
    everything else passes through as-is.
    """

    def __init__(self, tokenizer, time_tokenizer):
        self.tokenizer = tokenizer
        self.time_tokenizer = time_tokenizer

    def __call__(self, batch: List[Dict]) -> Dict:
        assert len(batch) == 1, (
            "Dispider training expects per_device_train_batch_size=1 "
            "(use gradient_accumulation_steps for effective batch size)."
        )
        item = batch[0]
        result = {}
        for key, val in item.items():
            if isinstance(val, torch.Tensor) and key in ("input_ids", "labels"):
                result[key] = val.unsqueeze(0)
            else:
                result[key] = val
        return result
