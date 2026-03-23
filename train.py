"""
Dispider Training Script
========================
Two-stage training for the Dispider video LLM:
  Stage 1: Train the compact LLM (compressor) + decision module directly.
           Uses TrainableStreamCompressor (subclass of StreamGroundQwenForCausalLM)
           which routes to forward_grounding_stream() for proper BCE + KL loss.
           Vision encoder (SigLIP) is frozen.
  Stage 2: Freeze the compressor + decision module, train the reaction LLM (Qwen2-7B).
           Uses LongQwen2ForCausalLM (the full Dispider model).

Usage:
    deepspeed train.py --stage 1 --compressor <path/to/stream_compressor> \
        --deepspeed scripts/ds_config_zero2.json ...
    deepspeed train.py --stage 2 --model_name_or_path <path/to/full_dispider> \
        --deepspeed scripts/ds_config_zero2.json ...
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
    set_seed,
)
from transformers import DynamicCache

# Patch DynamicCache API: Dispider's Qwen2 code calls get_usable_length()
# (old transformers Cache API) but newer transformers renamed it get_seq_length().
# get_usable_length(new_seq_len, layer_idx) returns the existing cache length
# (same as get_seq_length for DynamicCache which has no size limit).
if not hasattr(DynamicCache, "get_usable_length"):
    DynamicCache.get_usable_length = lambda self, new_seq_length, layer_idx=0: self.get_seq_length(layer_idx)

# Register model types before loading
from dispider.model.language_model.long_qwen import (
    LongConfig,
    LongQwen2ForCausalLM,
)
from dispider.model.language_model.llava_qwen import LlavaQwenConfig, LlavaQwenForCausalLM
from dispider.model.language_model.stream_grounding_qwen import (
    StreamGroundQwenConfig,
    StreamGroundQwenForCausalLM,
)
from dispider.model.language_model.trainable_compressor import (
    TrainableStreamCompressorConfig,
    TrainableStreamCompressor,
)
from dispider.constants import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_ANS_TOKEN,
    DEFAULT_TODO_TOKEN,
    DEFAULT_SILENT_TOKEN,
)
from dataset import (
    DispiderStage1Dataset,
    DispiderStage2Dataset,
    DispiderDataCollator,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HuggingFace model resolution helpers
# ---------------------------------------------------------------------------

def resolve_model_path(name_or_path: str) -> str:
    """Return a local directory for *name_or_path*.

    If it already exists on disk, return it as-is.  Otherwise treat it as a
    HuggingFace Hub repo id and download the full snapshot (including
    sub-folders like ``stream_compressor/``).
    """
    if os.path.exists(name_or_path):
        return name_or_path
    from huggingface_hub import snapshot_download
    logger.info(f"Downloading {name_or_path} from HuggingFace Hub …")
    local_dir = snapshot_download(repo_id=name_or_path)
    return local_dir


def fix_compressor_path_in_config(config, model_dir: str):
    """Rewrite ``config.mm_compressor`` so it points to the local
    ``stream_compressor/`` sub-folder inside *model_dir* rather than the
    author's original absolute path."""
    mm_comp = getattr(config, "mm_compressor", None)
    if mm_comp is None:
        return
    local_comp = os.path.join(model_dir, "stream_compressor")
    if os.path.isdir(local_comp) and mm_comp != local_comp:
        logger.info(f"Fixing mm_compressor path: {mm_comp} → {local_comp}")
        config.mm_compressor = local_comp


# ---------------------------------------------------------------------------
# Argument dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="",
        metadata={"help": "Path to the full Dispider checkpoint (Stage 2)."},
    )
    compressor: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Stage 1: path to the base stream_compressor checkpoint. "
                "Stage 2: optional path to a Stage-1 trained compressor "
                "checkpoint whose weights will be loaded into the full model's "
                "compressor sub-module."
            )
        },
    )
    compress_projector_type: str = field(default="mlp2x_gelu")
    clip_projector_type: str = field(default="mlp2x_gelu")
    pretrain_compress_mlp_adapter: Optional[str] = field(default=None)
    pretrain_clip_mlp_adapter: Optional[str] = field(default=None)
    pretrain_dual_mlp_adapter: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: int = field(default=-1)
    mm_vision_select_feature: str = field(default="patch")
    mm_projector_type: str = field(default="linear")


@dataclass
class DataArguments:
    data_path: str = field(metadata={"help": "Path to training annotation JSON."})
    video_root: str = field(metadata={"help": "Root directory containing video files."})
    num_frames: int = field(default=16)
    max_clips: int = field(default=100)
    max_length: int = field(default=2048)
    scene_sep_json: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to the precomputed scene boundaries JSON "
                "(produced by precompute_scene_sep.py).  "
                "If not provided, uniform temporal sampling is used."
            )
        },
    )


@dataclass
class DispiderTrainingArguments(TrainingArguments):
    stage: int = field(
        default=2,
        metadata={"help": "Training stage: 1 (Decision/Perception) or 2 (Reaction/LLM)."},
    )
    freeze_vision_encoder: bool = field(default=True)
    freeze_compressor: bool = field(default=False)
    freeze_compress_projector: bool = field(default=False)
    freeze_clip_projector: bool = field(default=False)
    freeze_llm_backbone: bool = field(default=False)


# ---------------------------------------------------------------------------
# Module freeze/unfreeze helpers
# ---------------------------------------------------------------------------

def _numel(p: torch.nn.Parameter) -> int:
    """Return real element count, even under DeepSpeed ZeRO-3 partitioning."""
    return getattr(p, "ds_numel", p.numel())


def freeze_module(module: torch.nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_module(module: torch.nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def configure_stage1_params(model: TrainableStreamCompressor):
    """
    Stage 1: Train the compact LLM (Qwen2-1.5B) + silent_head.
    Freeze the vision encoder (SigLIP) only.
    """
    unfreeze_module(model)
    vt = model.get_vision_tower()
    if vt is not None:
        freeze_module(vt)

    total_params = sum(_numel(p) for p in model.parameters())
    trainable_params = sum(_numel(p) for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Stage 1: {trainable_params:,} / {total_params:,} params trainable "
        f"({100 * trainable_params / total_params:.2f}%)"
    )


def configure_stage2_params(model: LongQwen2ForCausalLM, args: DispiderTrainingArguments):
    """
    Stage 2: Freeze compressor + decision module.
    Train reaction LLM (Qwen2-7B backbone) + projectors.
    """
    freeze_module(model)

    # Unfreeze main Qwen2-7B backbone + lm_head
    unfreeze_module(model.model)
    unfreeze_module(model.lm_head)

    # Keep compressor frozen and permanently in eval mode.
    # The compressor's forward_token_stream has a variable number of self.model()
    # calls in its training branch (2 + len(ans_position)), which differs across
    # samples on different GPUs. Under ZeRO-3 this causes a deadlock because all
    # ranks must participate in every parameter all-gather synchronously.
    # The eval branch always makes exactly 3 forward calls regardless of sample
    # structure, so it is safe for ZeRO-3. We override .train() to be a no-op so
    # that HF Trainer's model.train() call at each step cannot flip it back.
    compressor = model.get_compressor()
    if compressor is not None:
        freeze_module(compressor)
        compressor.eval()
        compressor.train = lambda *args, **kwargs: compressor

    # Unfreeze projectors
    inner = model.get_model()
    if hasattr(inner, "compress_projector"):
        unfreeze_module(inner.compress_projector)
    if hasattr(inner, "clip_projector"):
        unfreeze_module(inner.clip_projector)

    # Apply manual flag overrides from CLI
    if args.freeze_compressor and compressor is not None:
        freeze_module(compressor)
    if args.freeze_compress_projector and hasattr(inner, "compress_projector"):
        freeze_module(inner.compress_projector)
    if args.freeze_clip_projector and hasattr(inner, "clip_projector"):
        freeze_module(inner.clip_projector)
    if args.freeze_llm_backbone:
        for name, param in model.named_parameters():
            if "compressor" not in name and "projector" not in name:
                param.requires_grad = False

    total_params = sum(_numel(p) for p in model.parameters())
    trainable_params = sum(_numel(p) for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Stage 2: {trainable_params:,} / {total_params:,} params trainable "
        f"({100 * trainable_params / total_params:.2f}%)"
    )


# ---------------------------------------------------------------------------
# Custom Trainers for Dispider
# ---------------------------------------------------------------------------

class DispiderStage1Trainer(Trainer):
    """
    Stage 1 Trainer for TrainableStreamCompressor.

    The model's forward() remaps dataset keys and delegates to
    StreamGroundQwenForCausalLM.forward() → forward_grounding_stream(),
    which computes the BCE (decision) + KL (temporal retrieval) loss.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # HF Trainer._prepare_inputs already moves all tensors to the correct
        # device before compute_loss is called.  We only need to cast the
        # image tensor to the training dtype (float32 → bf16/fp16).
        if self.args.bf16:
            target_dtype = torch.bfloat16
        elif self.args.fp16:
            target_dtype = torch.float16
        else:
            target_dtype = torch.float32

        outputs = model(
            seqs=inputs["seqs"],
            compress_mask=inputs["compress_mask"],
            qs=inputs["qs"],
            qs_mask=inputs["qs_mask"],
            images=inputs["images"].to(dtype=target_dtype),
            time_labels=inputs["time_labels"],
            ans_token=inputs["ans_token"],
            todo_token=inputs["todo_token"],
            insert_position=inputs["insert_position"],
            ans_position=inputs.get("ans_position", []),
            silent_label=inputs["silent_label"],
        )

        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


class DispiderStage2Trainer(Trainer):
    """
    Stage 2 Trainer for LongQwen2ForCausalLM (full Dispider model).
    Compressor is frozen; only the reaction LLM is trained with LM loss.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # The compressor is frozen. Trainer.train() resets all submodules to
        # training mode each step, but forward_token_stream branches on
        # self.training: training=True with ans_position=[] (Stage 2) enters
        # the wrong branch and crashes. Keep compressor in eval so it uses the
        # correct inference path (not self.training and insert_position is not None).
        unwrapped = model.module if hasattr(model, "module") else model
        unwrapped.get_compressor().eval()

        input_ids = inputs["input_ids"]
        labels = inputs["labels"]

        _tokenizer = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        attention_mask = input_ids.ne(_tokenizer.pad_token_id).long()

        if self.args.bf16:
            target_dtype = torch.bfloat16
        elif self.args.fp16:
            target_dtype = torch.float16
        else:
            target_dtype = torch.float32

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            images=inputs["images"].to(dtype=target_dtype),
            images_large=inputs["images_large"].to(dtype=target_dtype),
            seqs=inputs["seqs"],
            compress_mask=inputs["compress_mask"],
            qs=inputs["qs"],
            qs_mask=inputs["qs_mask"],
            time_labels=inputs["time_labels"],
            ans_token=inputs["ans_token"],
            todo_token=inputs["todo_token"],
            insert_position=inputs.get("insert_position", 0),
            ans_position=inputs.get("ans_position", []),
            silent_position=inputs.get("silent_position", []),
        )

        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, DispiderTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO if training_args.local_rank in (-1, 0) else logging.WARN,
    )
    set_seed(training_args.seed)

    # ==================================================================
    # Stage 1: Train compact LLM (TrainableStreamCompressor) directly
    # ==================================================================
    if training_args.stage == 1:
        compressor_path = model_args.compressor
        if not compressor_path:
            raise ValueError(
                "Stage 1 requires --compressor pointing to the stream_compressor "
                "checkpoint directory or a HuggingFace repo id "
                "(e.g., Mar2Ding/Dispider/stream_compressor)."
            )
        compressor_path = resolve_model_path(compressor_path)
        # If pointed at the full Dispider repo, use the stream_compressor subfolder
        sub = os.path.join(compressor_path, "stream_compressor")
        if os.path.isdir(sub):
            compressor_path = sub
        logger.info(f"Stage 1: Loading compressor from {compressor_path}")

        config = AutoConfig.from_pretrained(compressor_path, trust_remote_code=True)
        # Keep the original model_type ("stream_ground_qwen") so from_pretrained
        # correctly matches the weight keys.  TrainableStreamCompressor inherits
        # from StreamGroundQwenForCausalLM, so the architecture is compatible.

        try:
            model = TrainableStreamCompressor.from_pretrained(
                compressor_path,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2",
            )
            logger.info("Using flash_attention_2")
        except (ImportError, ValueError) as e:
            logger.warning(f"flash_attention_2 unavailable ({e}), falling back to eager")
            model = TrainableStreamCompressor.from_pretrained(
                compressor_path,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )

        # Tokenizer and image processor come from the compressor itself
        time_tokenizer = AutoTokenizer.from_pretrained(
            compressor_path, use_fast=False, trust_remote_code=True
        )
        if time_tokenizer.pad_token is None:
            time_tokenizer.pad_token = "<pad>"

        image_processor = model.get_vision_tower().image_processor

        # For Stage 1, the main tokenizer (used only by the dataset to produce
        # dummy input_ids/labels that TrainableStreamCompressor ignores) can be
        # the same compressor tokenizer.
        tokenizer = time_tokenizer

        configure_stage1_params(model)

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            # Disable gradient checkpointing on the frozen vision tower
            # to avoid wasted re-forward passes and "requires_grad=True" warnings.
            vt = model.get_vision_tower()
            if vt is not None:
                if hasattr(vt, "gradient_checkpointing_disable"):
                    vt.gradient_checkpointing_disable()
                for module in vt.modules():
                    if hasattr(module, "gradient_checkpointing"):
                        module.gradient_checkpointing = False

        train_dataset = DispiderStage1Dataset(
            data_path=data_args.data_path,
            video_root=data_args.video_root,
            image_processor=image_processor,
            time_tokenizer=time_tokenizer,
            tokenizer=tokenizer,
            model_config=config,
            num_frames=data_args.num_frames,
            max_clips=data_args.max_clips,
            scene_sep_json=data_args.scene_sep_json,
        )
        data_collator = DispiderDataCollator(tokenizer, time_tokenizer)

        trainer = DispiderStage1Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

    # ==================================================================
    # Stage 2: Train reaction LLM (LongQwen2ForCausalLM)
    # ==================================================================
    else:
        if not model_args.model_name_or_path:
            raise ValueError(
                "Stage 2 requires --model_name_or_path pointing to the full "
                "Dispider checkpoint or a HuggingFace repo id (e.g., Mar2Ding/Dispider)."
            )
        model_path = resolve_model_path(model_args.model_name_or_path)
        logger.info(f"Stage 2: Loading model from {model_path}")

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        fix_compressor_path_in_config(config, model_path)

        try:
            model = LongQwen2ForCausalLM.from_pretrained(
                model_path,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2",
            )
            logger.info("Using flash_attention_2")
        except (ImportError, ValueError) as e:
            logger.warning(f"flash_attention_2 unavailable ({e}), falling back to eager")
            model = LongQwen2ForCausalLM.from_pretrained(
                model_path,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})

        # Add multimodal special tokens to the reaction tokenizer
        mm_use_im_start_end = getattr(config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(config, "mm_use_im_patch_token", False)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        # Get compressor tokenizer and image processor
        compressor = model.get_compressor()
        if compressor is None:
            raise ValueError(
                "Model has no compressor. Ensure the checkpoint has mm_compressor configured."
            )

        # Optionally load Stage 1 trained compressor weights
        if model_args.compressor and os.path.exists(model_args.compressor):
            logger.info(
                f"Stage 2: Loading Stage-1 compressor weights from {model_args.compressor}"
            )
            safetensors_path = os.path.join(model_args.compressor, "model.safetensors")
            pytorch_bin_path = os.path.join(model_args.compressor, "pytorch_model.bin")
            if os.path.exists(safetensors_path):
                from safetensors.torch import load_file
                stage1_state = load_file(safetensors_path, device="cpu")
            elif os.path.exists(pytorch_bin_path):
                stage1_state = torch.load(pytorch_bin_path, map_location="cpu")
            else:
                raise FileNotFoundError(
                    f"No model weights found in {model_args.compressor}. "
                    f"Expected model.safetensors or pytorch_model.bin."
                )
            # TrainableStreamCompressor keys are prefixed the same as
            # StreamGroundQwenForCausalLM — load into compressor.compressor.
            # When DeepSpeed ZeRO-3 is active, TrainingArguments.__post_init__
            # activates deepspeed.zero.Init(), which makes all subsequently
            # created parameters empty ([0]-shaped) until gathered. We must
            # use GatheredParameters to materialize them before writing.
            from transformers.integrations import is_deepspeed_zero3_enabled
            if is_deepspeed_zero3_enabled():
                import deepspeed
                params = list(compressor.compressor.parameters())
                with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                    if training_args.local_rank in (-1, 0):
                        missing, unexpected = compressor.compressor.load_state_dict(
                            stage1_state, strict=False
                        )
                    else:
                        missing, unexpected = [], []
            else:
                missing, unexpected = compressor.compressor.load_state_dict(
                    stage1_state, strict=False
                )
            if missing:
                logger.warning(f"Stage-1 load — missing keys: {missing[:10]}...")
            if unexpected:
                logger.warning(f"Stage-1 load — unexpected keys: {unexpected[:10]}...")
            del stage1_state

        image_processor = compressor.compressor.get_vision_tower().image_processor
        time_tokenizer = compressor.tokenizer
        if time_tokenizer.pad_token is None:
            time_tokenizer.pad_token = "<pad>"

        configure_stage2_params(model, training_args)

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()

        train_dataset = DispiderStage2Dataset(
            data_path=data_args.data_path,
            video_root=data_args.video_root,
            image_processor=image_processor,
            time_tokenizer=time_tokenizer,
            tokenizer=tokenizer,
            model_config=config,
            num_frames=data_args.num_frames,
            max_clips=data_args.max_clips,
            max_length=data_args.max_length,
            scene_sep_json=data_args.scene_sep_json,
        )
        data_collator = DispiderDataCollator(tokenizer, time_tokenizer)

        trainer = DispiderStage2Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    logger.info(f"*** Starting Stage {training_args.stage} Training ***")
    trainer.train(
        resume_from_checkpoint=training_args.resume_from_checkpoint,
    )

    # Save
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()
