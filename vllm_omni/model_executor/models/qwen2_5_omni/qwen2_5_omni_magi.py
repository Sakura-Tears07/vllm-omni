# SPDX-License-Identifier: Apache-2.0
"""Optional MagiCompiler integration for Qwen2.5-Omni talker and Token2Wav DiT.

Integration surface (intended usage):

1. **Example entrypoint**: ``examples/offline_inference/qwen2_5_omni/magi_example.py``
2. **Model wiring**:
   - Stage1 talker: ``apply_magi_to_qwen2_decoder_layers`` from ``qwen2_old.Qwen2Model``
   - Stage2 code2wav: ``apply_magi_to_decoder_layers`` from ``qwen2_5_omni_token2wav``
3. **Definition**: stack modules and apply helpers in this module.

Set ``VLLM_OMNI_MAGI_COMPILER=1`` and install `MagiCompiler` (SandAI-org/MagiCompiler).
If it is not installed, ``magi_compile`` is a no-op (same pattern as ``magi_human_dit``).

Engine startup still calls :func:`merge_vllm_compilation_config_for_magi` from
``stage_init_utils`` so vLLM's compile stack stays off on talker/code2wav when Magi
is enabled (implementation detail, not part of the three-part API above).
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any

import torch
import torch.nn as nn

from vllm.logger import init_logger

logger = init_logger(__name__)

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


_MAGI_ENABLED_STAGES = frozenset({"talker", "code2wav"})


def merge_vllm_compilation_config_for_magi(engine_args_dict: dict[str, Any]) -> None:
    """When MagiCompiler is enabled for Qwen2.5-Omni talker/code2wav, turn off vLLM compilation.

    vLLM's ``VLLM_COMPILE`` mode and fusion passes can fight with ``@magi_compile``
    (nested Dynamo / Inductor).  This forces ``CompilationMode.NONE`` and disables
    the main custom fusion flags for the talker and Token2Wav stages.
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
    """Runs Qwen2 decoder layers for talker; compiled via ``magi_compile`` when enabled."""

    def __init__(self, layers: nn.ModuleList, start_layer: int, end_layer: int):
        super().__init__()
        # Shares the same layer list as ``Qwen2Model.layers``.
        self.layers = layers
        self.start_layer = start_layer
        self.end_layer = end_layer

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, residual = layer(positions, hidden_states, residual)
        return hidden_states, residual


_MAGI_TALKER_DYNAMIC_ARG_DIMS: dict[str, int | list[int]] = {
    # Flattened token layout [num_tokens, hidden] — num_tokens varies per batch.
    "hidden_states": 0,
    "residual": 0,
    # positions is (seq_len,) or (3, seq_len) for MRoPE.
    "positions": -1,
}


def apply_magi_to_qwen2_decoder_layers(
    layers: nn.ModuleList,
    start_layer: int,
    end_layer: int,
) -> nn.Module:
    """Wrap talker Qwen2 decoder layers for optional MagiCompiler compilation."""
    stack = _Qwen2OmniTalkerDecoderStack(layers, start_layer, end_layer)
    _warn_if_magi_requested_but_unavailable()
    if not is_magi_compiler_enabled():
        return stack
    if not _HAS_MAGI_COMPILER:
        return stack

    try:
        compiled_stack = _magi_compile(
            stack,
            model_tag="qwen2_5_omni_talker",
            dynamic_arg_dims=_MAGI_TALKER_DYNAMIC_ARG_DIMS,
        )
        logger.info("[MagiCompiler] Compiled talker Qwen2 decoder stack via magi_compile.")
        return compiled_stack
    except Exception as exc:
        logger.warning(
            "[MagiCompiler] Failed to compile talker decoder stack, fallback to eager stack: %s",
            exc,
        )
        return stack


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
