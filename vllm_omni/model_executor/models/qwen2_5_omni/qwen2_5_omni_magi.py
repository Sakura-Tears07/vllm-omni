# SPDX-License-Identifier: Apache-2.0
"""Optional MagiCompiler integration for Qwen2.5-Omni thinker, talker and Token2Wav DiT.

Integration surface (intended usage):

1. **Example entrypoint**: ``examples/offline_inference/qwen2_5_omni/magi_example.py``
2. **Model wiring**:
   - Stage0 thinker: ``wire_magi_into_qwen2_model``, ``wire_magi_into_qwen2_logits``, and
     ``wire_magi_into_qwen2_vision`` from ``qwen2_5_omni_thinker``
     (``language_model.model`` / ``language_model`` / ``visual``)
   - Stage1 talker: ``apply_magi_to_qwen2_decoder_layers`` from ``qwen2_old.Qwen2Model``;
     ``apply_magi_to_talker_logits_stack`` from ``qwen2_old.Qwen2ForCausalLM``
   - Stage2 code2wav: ``apply_magi_to_decoder_layers`` from ``qwen2_5_omni_token2wav``
3. **Definition**: stack modules and apply helpers in this module.

Set ``VLLM_OMNI_MAGI_COMPILER=1`` and install `MagiCompiler` (SandAI-org/MagiCompiler).
If it is not installed, ``magi_compile`` is a no-op (same pattern as ``magi_human_dit``).

Engine startup still calls :func:`merge_vllm_compilation_config_for_magi` from
``stage_init_utils`` so vLLM's compile stack stays off on thinker/talker/code2wav when Magi
is enabled (implementation detail, not part of the three-part API above).
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, is_dataclass
from types import MethodType
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.models.utils import cast_overflow_tensors
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)

try:
    from magi_compiler.utils.nvtx import add_nvtx_event as _magi_add_nvtx_event
except Exception:

    class _magi_add_nvtx_event:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

_HAS_MAGI_COMPILER = True
try:
    # Preferred import path in most MagiCompiler examples.
    from magi_compiler import magi_compile as _magi_compile
except Exception:
    try:
        # Backward-compatible path used by some versions.
        from magi_compiler.api import magi_compile as _magi_compile
    except Exception:  # pragma: no cover - optional dependency
        _HAS_MAGI_COMPILER = False

        def _magi_compile(*args, **kwargs):
            def decorator(cls_or_fn):
                return cls_or_fn

            return decorator


def is_magi_compiler_enabled() -> bool:
    return os.environ.get("VLLM_OMNI_MAGI_COMPILER", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _warn_if_magi_requested_but_unavailable() -> None:
    if is_magi_compiler_enabled() and not _HAS_MAGI_COMPILER:
        logger.warning(
            "[MagiCompiler] VLLM_OMNI_MAGI_COMPILER is enabled but magi_compiler "
            "cannot be imported. The decorator is a no-op and no Magi fusion will happen."
        )


# Align with LightX2V-MagiCompiler practice: use MagiCompiler as the single
# torch.compile / Inductor path; disable vLLM's own compilation stack on Magi
# stages to avoid double-compilation and conflicting graph passes.
_MAGI_VLLM_COMPILATION_OVERRIDES: dict[str, Any] = {
    # vllm.config.compilation.CompilationMode.NONE
    "mode": 0,
    "pass_config": {
        "fuse_norm_quant": False,
        "fuse_act_quant": False,
        "fuse_attn_quant": False,
    },
}


def _compilation_config_to_dict(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return copy.deepcopy(raw)
    if isinstance(raw, str):
        return json.loads(raw)
    model_dump = getattr(raw, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    if is_dataclass(raw):
        return asdict(raw)
    if hasattr(raw, "__dict__"):
        return dict(vars(raw))
    raise TypeError(f"Unsupported compilation_config type: {type(raw)!r}")


_MAGI_ENABLED_STAGES = frozenset({"thinker", "talker", "code2wav"})


def merge_vllm_compilation_config_for_magi(engine_args_dict: dict[str, Any]) -> None:
    """When MagiCompiler is enabled for Qwen2.5-Omni thinker/talker/code2wav, turn off vLLM compilation.

    vLLM's ``VLLM_COMPILE`` mode and fusion passes can fight with ``@magi_compile``
    (nested Dynamo / Inductor).  This forces ``CompilationMode.NONE`` and disables
    the main custom fusion flags for the thinker, talker and Token2Wav stages.
    """
    if not is_magi_compiler_enabled():
        return
    model_arch = engine_args_dict.get("model_arch")
    # Some stage-config paths do not populate model_arch. Only reject when an
    # explicit non-Qwen2.5 architecture is present.
    if model_arch is not None and model_arch != "Qwen2_5OmniForConditionalGeneration":
        return
    model_stage = engine_args_dict.get("model_stage")
    if model_stage not in _MAGI_ENABLED_STAGES:
        return

    merged = _compilation_config_to_dict(engine_args_dict.get("compilation_config"))
    magi_pc = copy.deepcopy(_MAGI_VLLM_COMPILATION_OVERRIDES["pass_config"])
    user_pc = merged.get("pass_config")
    if isinstance(user_pc, dict):
        # Mandatory disables win over user-enabled fusions for Magi compatibility.
        merged["pass_config"] = {**user_pc, **magi_pc}
    else:
        merged["pass_config"] = magi_pc
    merged["mode"] = _MAGI_VLLM_COMPILATION_OVERRIDES["mode"]
    engine_args_dict["compilation_config"] = merged
    logger.info(
        "[MagiCompiler] Disabled vLLM torch.compile stack for %s "
        "(mode=NONE, fusion passes off for norm/act/attn quant).",
        model_stage,
    )


class _Qwen2OmniDiTTransformerStack(nn.Module):
    """Runs ``DiTDecoderLayer`` blocks; compiled via ``magi_compile`` when enabled."""

    def __init__(self, blocks: nn.ModuleList):
        super().__init__()
        # Shares the same ModuleList instance as ``Qwen2_5OmniToken2WavDiTModel.transformer_blocks``.
        self.blocks = blocks

    def forward(
        self,
        hidden_states: torch.Tensor,
        time_embedding: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        block_diff: torch.Tensor,
    ) -> torch.Tensor:
        position_embeddings = (cos, sin)
        for transformer_block in self.blocks:
            hidden_states = transformer_block(
                hidden_states,
                time_embedding,
                position_embeddings=position_embeddings,
                block_diff=block_diff,
            )
        return hidden_states


_MAGI_DYNAMIC_ARG_DIMS: dict[str, int | list[int]] = {
    # [batch, seq, hidden] — seq varies across requests/steps
    "hidden_states": 1,
    "cos": 1,
    "sin": 1,
    # [batch, heads, seq, seq] from ``_create_block_diff`` — only mark the
    # attention map spatial dims (2, 3) as dynamic. Do not mark dim 0:
    # Dynamo/SDPA specialize batch to the warmup size (e.g. 2) and
    # ``mark_dynamic`` on dim 0 then fails with ConstraintViolationError.
    "block_diff": [2, 3],
}


class _Qwen2OmniTalkerDecoderStack(nn.Module):
    """Runs Qwen2 decoder layers for talker; compiled via ``magi_compile`` when enabled.

    When ``norm`` is set (last pipeline-parallel rank), final ``RMSNorm`` runs inside
    this stack so Magi can fuse it with the decoder loop under the same ``model_tag``.
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        start_layer: int,
        end_layer: int,
        norm: nn.Module | None = None,
    ):
        super().__init__()
        # Shares the same layer list as ``Qwen2Model.layers``.
        self.layers = layers
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.norm = norm

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, residual = layer(positions, hidden_states, residual)
        if self.norm is not None:
            hidden_states, _ = self.norm(hidden_states, residual)
            return hidden_states, None
        return hidden_states, residual


_MAGI_TALKER_DYNAMIC_ARG_DIMS: dict[str, int | list[int]] = {
    # Flattened token layout [num_tokens, hidden] — num_tokens varies per batch.
    "hidden_states": 0,
    "residual": 0,
    # positions is (seq_len,) or (3, seq_len) for MRoPE.
    "positions": -1,
}

_MAGI_QWEN2_LOGITS_DYNAMIC_ARG_DIMS: dict[str, int | list[int]] = {
    # Flattened token layout [num_tokens, hidden] — num_tokens varies per decode step.
    # Callers must skip Magi when shape[0] <= 1 (see wired compute_logits helpers).
    "hidden_states": 0,
}


class _Qwen2OmniQwen2LogitsStack(nn.Module):
    """Qwen2 lm_head path: ``LogitsProcessor(lm_head, hidden_states)``."""

    def __init__(self, lm_head: nn.Module, logits_processor: nn.Module):
        super().__init__()
        self.lm_head = lm_head
        self.logits_processor = logits_processor

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)


def apply_magi_to_qwen2_decoder_layers(
    layers: nn.ModuleList,
    start_layer: int,
    end_layer: int,
    norm: nn.Module | None = None,
    *,
    model_tag: str = "qwen2_5_omni_talker",
) -> nn.Module:
    """Wrap Qwen2 decoder layers for optional MagiCompiler compilation."""
    stack = _Qwen2OmniTalkerDecoderStack(layers, start_layer, end_layer, norm=norm)
    _warn_if_magi_requested_but_unavailable()
    if not is_magi_compiler_enabled():
        return stack
    if not _HAS_MAGI_COMPILER:
        return stack

    try:
        compiled_stack = _magi_compile(
            stack,
            model_tag=model_tag,
            dynamic_arg_dims=_MAGI_TALKER_DYNAMIC_ARG_DIMS,
        )
        logger.info(
            "[MagiCompiler] Compiled %s Qwen2 decoder stack via magi_compile.",
            model_tag,
        )
        return compiled_stack
    except Exception as exc:
        logger.warning(
            "[MagiCompiler] Failed to compile %s decoder stack, fallback to eager stack: %s",
            model_tag,
            exc,
        )
        return stack


def wire_magi_into_qwen2_model(
    model: nn.Module,
    *,
    model_tag: str,
) -> None:
    """Attach a Magi-aware decoder stack to an upstream vLLM ``Qwen2Model``.

    Replaces the per-layer forward loop with ``_Qwen2OmniTalkerDecoderStack`` (optionally
    compiled). Final ``RMSNorm`` on the last pipeline-parallel rank is included in the
    stack when present. Intended for Stage0 thinker ``language_model.model``; talker uses
    built-in wiring in ``qwen2_old.Qwen2Model`` instead.
    """
    if getattr(model, "_magi_wired_model_tag", None) == model_tag:
        return
    if hasattr(model, "_decoder_stack") and getattr(model, "_magi_wired_model_tag", None) is not None:
        raise RuntimeError(
            f"Qwen2Model already wired for Magi tag {model._magi_wired_model_tag!r}; "
            f"refusing to re-wire as {model_tag!r}."
        )

    norm_for_stack = model.norm if get_pp_group().is_last_rank else None
    model._decoder_stack_includes_norm = get_pp_group().is_last_rank
    model._decoder_stack = apply_magi_to_qwen2_decoder_layers(
        model.layers,
        model.start_layer,
        model.end_layer,
        norm=norm_for_stack,
        model_tag=model_tag,
    )
    model._magi_wired_model_tag = model_tag
    model.forward = MethodType(_magi_qwen2_model_forward, model)


def _magi_qwen2_model_forward(
    model: nn.Module,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor | IntermediateTensors:
    if get_pp_group().is_first_rank:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = model.embed_input_ids(input_ids)
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    hidden_states, residual = model._decoder_stack(positions, hidden_states, residual)
    if not get_pp_group().is_last_rank:
        return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})
    if not model._decoder_stack_includes_norm:
        hidden_states, _ = model.norm(hidden_states, residual)
    return hidden_states


def apply_magi_to_qwen2_logits_stack(
    lm_head: nn.Module,
    logits_processor: nn.Module,
    *,
    model_tag: str = "qwen2_5_omni_talker_logits",
) -> nn.Module:
    """Wrap Qwen2 ``lm_head`` + ``LogitsProcessor`` for optional MagiCompiler compilation."""
    stack = _Qwen2OmniQwen2LogitsStack(lm_head, logits_processor)
    _warn_if_magi_requested_but_unavailable()
    if not is_magi_compiler_enabled():
        return stack
    if not _HAS_MAGI_COMPILER:
        return stack

    try:
        compiled_stack = _magi_compile(
            stack,
            model_tag=model_tag,
            dynamic_arg_dims=_MAGI_QWEN2_LOGITS_DYNAMIC_ARG_DIMS,
        )
        logger.info(
            "[MagiCompiler] Compiled %s Qwen2 logits stack via magi_compile.",
            model_tag,
        )
        return compiled_stack
    except Exception as exc:
        logger.warning(
            "[MagiCompiler] Failed to compile %s logits stack, fallback to eager stack: %s",
            model_tag,
            exc,
        )
        return stack


def apply_magi_to_talker_logits_stack(
    lm_head: nn.Module,
    logits_processor: nn.Module,
) -> nn.Module:
    """Wrap talker ``lm_head`` + ``LogitsProcessor`` for optional MagiCompiler compilation."""
    return apply_magi_to_qwen2_logits_stack(
        lm_head,
        logits_processor,
        model_tag="qwen2_5_omni_talker_logits",
    )


def wire_magi_into_qwen2_logits(
    language_model: nn.Module,
    *,
    model_tag: str,
) -> None:
    """Attach a Magi-aware logits stack to upstream vLLM ``Qwen2ForCausalLM``.

    Patches ``compute_logits`` to route through ``_Qwen2OmniQwen2LogitsStack`` when
    ``hidden_states.shape[0] > 1``. Intended for Stage0 thinker ``language_model``.
    """
    if getattr(language_model, "_magi_logits_wired_model_tag", None) == model_tag:
        return

    if get_pp_group().is_last_rank:
        language_model._magi_logits_stack = apply_magi_to_qwen2_logits_stack(
            language_model.lm_head,
            language_model.logits_processor,
            model_tag=model_tag,
        )
    else:
        language_model._magi_logits_stack = None

    language_model._magi_logits_wired_model_tag = model_tag
    language_model.compute_logits = MethodType(_magi_qwen2_compute_logits, language_model)


def _magi_qwen2_compute_logits(
    language_model: nn.Module,
    hidden_states: torch.Tensor,
) -> torch.Tensor | None:
    # MagiCompiler cannot mark dim 0 dynamic when batch/token count is 0 or 1
    # (Dynamo zero/one specialization). vLLM profile_run with max_num_seqs=1
    # passes [1, hidden] here; keep that path eager and compile only when >= 2.
    if language_model._magi_logits_stack is not None and hidden_states.shape[0] > 1:
        return language_model._magi_logits_stack(hidden_states)
    return language_model.logits_processor(language_model.lm_head, hidden_states)


class _Qwen2OmniVisionTransformerStack(nn.Module):
    """Runs Qwen2.5-VL ``Qwen2_5_VisionBlock`` stack; compiled via ``magi_compile`` when enabled."""

    def __init__(
        self,
        blocks: nn.ModuleList,
        fullatt_block_indexes: tuple[int, ...],
    ):
        super().__init__()
        self.blocks = blocks
        self.fullatt_block_indexes = fullatt_block_indexes

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cu_window_seqlens: torch.Tensor,
        max_seqlen_full: torch.Tensor,
        max_seqlen_window: torch.Tensor,
    ) -> torch.Tensor:
        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
                max_seqlen_now = max_seqlen_full
            else:
                cu_seqlens_now = cu_window_seqlens
                max_seqlen_now = max_seqlen_window

            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
                max_seqlen=max_seqlen_now,
            )
        return hidden_states


_MAGI_VISION_DYNAMIC_ARG_DIMS: dict[str, int | list[int]] = {
    # [seq_len, 1, hidden] after window reorder + unsqueeze(1).
    "hidden_states": 0,
    "rotary_pos_emb_cos": 0,
    "rotary_pos_emb_sin": 0,
    # Variable-length video/image token counts per forward.
    "cu_seqlens": 0,
    "cu_window_seqlens": 0,
}


def apply_magi_to_vision_blocks(
    blocks: nn.ModuleList,
    fullatt_block_indexes: list[int] | tuple[int, ...],
    *,
    model_tag: str = "qwen2_5_omni_thinker_visual",
) -> nn.Module:
    """Wrap Qwen2.5-VL vision transformer blocks for optional MagiCompiler compilation."""
    indexes = tuple(int(i) for i in fullatt_block_indexes)
    stack = _Qwen2OmniVisionTransformerStack(blocks, indexes)
    _warn_if_magi_requested_but_unavailable()
    if not is_magi_compiler_enabled():
        return stack
    if not _HAS_MAGI_COMPILER:
        return stack

    try:
        compiled_stack = _magi_compile(
            stack,
            model_tag=model_tag,
            dynamic_arg_dims=_MAGI_VISION_DYNAMIC_ARG_DIMS,
        )
        logger.info(
            "[MagiCompiler] Compiled %s Qwen2.5-VL vision stack via magi_compile.",
            model_tag,
        )
        return compiled_stack
    except Exception as exc:
        logger.warning(
            "[MagiCompiler] Failed to compile %s vision stack, fallback to eager stack: %s",
            model_tag,
            exc,
        )
        return stack


def wire_magi_into_qwen2_vision(
    visual: nn.Module,
    *,
    model_tag: str = "qwen2_5_omni_thinker_visual",
) -> None:
    """Attach a Magi-aware vision block stack to vLLM ``Qwen2_5_VisionTransformer``.

    Replaces the per-block loop in ``forward`` with ``_Qwen2OmniVisionTransformerStack``
    (optionally compiled). Patch embed, window reorder and merger stay eager.
    """
    if getattr(visual, "_magi_wired_model_tag", None) == model_tag:
        return

    visual._visual_blocks_stack = apply_magi_to_vision_blocks(
        visual.blocks,
        visual.fullatt_block_indexes,
        model_tag=model_tag,
    )
    visual._magi_wired_model_tag = model_tag
    visual.forward = MethodType(_magi_vision_transformer_forward, visual)


def _magi_vision_transformer_forward(
    visual: nn.Module,
    x: torch.Tensor,
    grid_thw: list[list[int]],
) -> torch.Tensor:
    with _magi_add_nvtx_event("thinker.visual.forward"):
        seq_len, _ = x.size()
        rotary_pos_emb_cos = []
        rotary_pos_emb_sin = []
        window_index: list = []
        cu_window_seqlens: list = [torch.tensor([0], dtype=torch.int32)]
        cu_seqlens: list = []

        hidden_states = x.to(device=visual.device, dtype=visual.dtype)
        hidden_states = visual.patch_embed(hidden_states)

        window_index_id = 0
        cu_window_seqlens_last = 0
        for t, h, w in grid_thw:
            t, h, w = int(t), int(h), int(w)
            llm_h = h // visual.spatial_merge_size
            llm_w = w // visual.spatial_merge_size

            (
                cos_thw,
                sin_thw,
                window_index_thw,
                cu_seqlens_window_thw,
                cu_seqlens_thw,
            ) = visual.get_rope_by_thw(t, h, w)

            window_index.append(window_index_thw + window_index_id)
            window_index_id += t * llm_h * llm_w

            cu_seqlens_window_thw = cu_seqlens_window_thw + cu_window_seqlens_last
            cu_window_seqlens_last = cu_seqlens_window_thw[-1]
            cu_window_seqlens.append(cu_seqlens_window_thw)

            rotary_pos_emb_cos.append(cos_thw)
            rotary_pos_emb_sin.append(sin_thw)

            cu_seqlens.append(cu_seqlens_thw)

        rotary_pos_emb_cos = torch.cat(rotary_pos_emb_cos)
        rotary_pos_emb_sin = torch.cat(rotary_pos_emb_sin)
        window_index = torch.cat(window_index)
        reverse_indices = visual.invert_permutation(window_index)
        cu_window_seqlens = torch.cat(cu_window_seqlens)
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
        cu_seqlens = torch.cat(cu_seqlens)
        cu_seqlens = torch.cumsum(cu_seqlens, dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), "constant", 0)

        max_seqlen_full = visual.compute_attn_mask_seqlen(cu_seqlens)
        max_seqlen_window = visual.compute_attn_mask_seqlen(cu_window_seqlens)

        cu_seqlens = cu_seqlens.to(device=visual.device, non_blocking=True)
        cu_window_seqlens = cu_window_seqlens.to(device=visual.device, non_blocking=True)
        rotary_pos_emb_cos = rotary_pos_emb_cos.to(device=visual.device, non_blocking=True)
        rotary_pos_emb_sin = rotary_pos_emb_sin.to(device=visual.device, non_blocking=True)
        window_index = window_index.to(device=hidden_states.device, non_blocking=True)
        reverse_indices = reverse_indices.to(device=hidden_states.device, non_blocking=True)

        hidden_states = hidden_states.reshape(
            seq_len // visual.spatial_merge_unit,
            visual.spatial_merge_unit,
            -1,
        )
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        hidden_states = hidden_states.unsqueeze(1)

        with _magi_add_nvtx_event("thinker.visual.stack"):
            hidden_states = visual._visual_blocks_stack(
                hidden_states,
                rotary_pos_emb_cos,
                rotary_pos_emb_sin,
                cu_seqlens,
                cu_window_seqlens,
                max_seqlen_full,
                max_seqlen_window,
            )

        if hidden_states.dtype == torch.float16:
            hidden_states = cast_overflow_tensors(hidden_states)

        hidden_states = visual.merger(hidden_states)
        hidden_states = hidden_states[reverse_indices, :]
        return hidden_states


def apply_magi_to_decoder_layers(blocks: nn.ModuleList) -> nn.Module:
    """Wrap Token2Wav DiT decoder layers for optional MagiCompiler compilation.

    When ``VLLM_OMNI_MAGI_COMPILER`` is enabled and ``magi_compiler`` is installed,
    the returned module's forward is compiled with ``magi_compile``. Otherwise behaviour
    matches an eager loop over the same ``blocks``.

    Args:
        blocks: ``Qwen2_5OmniToken2WavDiTModel.transformer_blocks`` (shared reference).

    Returns:
        A module whose ``forward(hidden_states, time_embedding, cos, sin, block_diff)``
        runs all decoder layers.
    """
    stack = _Qwen2OmniDiTTransformerStack(blocks)
    _warn_if_magi_requested_but_unavailable()
    if not is_magi_compiler_enabled():
        return stack
    if not _HAS_MAGI_COMPILER:
        return stack

    try:
        compiled_stack = _magi_compile(
            stack,
            model_tag="qwen2_5_omni_token2wav_dit",
            dynamic_arg_dims=_MAGI_DYNAMIC_ARG_DIMS,
        )
        logger.info("[MagiCompiler] Compiled Token2Wav DiT transformer stack via magi_compile.")
        return compiled_stack
    except Exception as exc:
        logger.warning(
            "[MagiCompiler] Failed to compile Token2Wav DiT stack, fallback to eager stack: %s",
            exc,
        )
        return stack
