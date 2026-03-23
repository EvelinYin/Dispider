"""
trainable_compressor.py
=======================
Stage 1 training wrapper for Dispider's compact decision module.

This file introduces TrainableStreamCompressor, a thin subclass of the
official StreamGroundQwenForCausalLM that:

  1. Re-maps the Dispider dataset keys (images, seqs, qs, silent_label, ...)
     to the correct positional/keyword arguments expected by the official
     forward_grounding_stream() loss function.
  2. Returns a HuggingFace CausalLMOutputWithPast whose .loss field is the
     combined BCE (decision) + KL (temporal retrieval) scalar already
     computed inside the official model code — so no official files are
     touched.

Design contract
---------------
* Do NOT import or call anything from long_qwen.py / long_arch.py here.
  Stage 1 trains the 1.5 B compact LLM directly; the 7 B reaction model
  (LongQwen2ForCausalLM) is only involved in Stage 2.
* This file only imports from official files that already exist in the repo.
  It adds zero new training logic — all loss computation is delegated to
  the official forward_grounding_stream() / forward_grounding_hm() path.
"""

from __future__ import annotations

from typing import List, Optional, Union

import torch

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

# Official model — not modified
from .stream_grounding_qwen import StreamGroundQwenForCausalLM, StreamGroundQwenConfig


class TrainableStreamCompressorConfig(StreamGroundQwenConfig):
    model_type = "trainable_stream_compressor"


class TrainableStreamCompressor(StreamGroundQwenForCausalLM):
    """
    HuggingFace-Trainer-compatible wrapper for Stage 1 Dispider training.

    The official StreamGroundQwenForCausalLM.forward() already computes the
    correct losses (BCE + KL) via forward_grounding_stream(), but its
    parameter names differ from the keys produced by DispiderStage1Dataset.
    This subclass bridges that gap without touching any official code.

    Dataset key → official parameter mapping
    ----------------------------------------
    seqs          → input_ids        (time-description token ids, shape [C, L])
    compress_mask → attention_mask   (padding mask for seqs)
    qs            → qs_ids           (question token ids, shape [1, Q])
    qs_mask       → qs_mask          (question padding mask)
    images        → images           (video frames, shape [C, F, 3, H, W])
    time_labels   → time_labels      (KL target distribution, shape [1, C])
    ans_token     → ans_token        (<has_answer> token ids)
    todo_token    → todo_token       (<to_do> token ids)
    insert_position → insert_position  (int, clip index of question onset)
    ans_position  → ans_position     (list; [] for first-turn single-QA)
    silent_label  → silent_label     (binary tensor, shape [C - insert_pos + 1]
                                      1 = respond, 0 = stay silent)

    Keys ignored in Stage 1 (produced by collator but unused here):
        input_ids, labels, images_large, silent_position
    """

    config_class = TrainableStreamCompressorConfig

    def forward(
        self,
        # ---- Stage-1 dataset keys (primary) ----
        seqs: Optional[torch.LongTensor] = None,
        compress_mask: Optional[torch.Tensor] = None,
        qs: Optional[torch.LongTensor] = None,
        qs_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.FloatTensor] = None,
        time_labels: Optional[torch.FloatTensor] = None,
        ans_token: Optional[torch.LongTensor] = None,
        todo_token: Optional[torch.LongTensor] = None,
        insert_position: Optional[int] = None,
        ans_position: Optional[list] = None,
        silent_label: Optional[torch.FloatTensor] = None,
        # ---- HF Trainer standard keys (accepted but ignored in Stage 1) ----
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        images_large: Optional[torch.FloatTensor] = None,
        silent_position: Optional[list] = None,
        # ---- Pass-through kwargs ----
        **kwargs,
    ) -> Union[CausalLMOutputWithPast, torch.FloatTensor]:
        """
        Delegates to StreamGroundQwenForCausalLM.forward() with the correct
        argument names, triggering forward_grounding_stream() when
        insert_position and silent_label are provided.
        """
        return super().forward(
            # seqs are the per-clip time-description sequences fed to the
            # compact LLM — they become input_ids for the compressor.
            input_ids=seqs,
            attention_mask=compress_mask,
            qs_ids=qs,
            qs_mask=qs_mask,
            images=images,
            time_labels=time_labels,
            ans_token=ans_token,
            todo_token=todo_token,
            insert_position=insert_position,
            # ans_position=[] for first-turn single-QA (no prior responses).
            # forward_grounding_stream() handles the empty-list case correctly.
            ans_position=ans_position if ans_position is not None else [],
            silent_label=silent_label,
            # The following are either unused by StreamGroundQwenForCausalLM
            # or delegated to prepare_inputs_labels_for_multimodal():
            labels=None,   # loss is computed internally via BCE+KL
            return_dict=True,
        )


# Register so that AutoModel/AutoConfig work with the checkpoint's config type.
AutoConfig.register("trainable_stream_compressor", TrainableStreamCompressorConfig)
AutoModelForCausalLM.register(
    TrainableStreamCompressorConfig, TrainableStreamCompressor
)
