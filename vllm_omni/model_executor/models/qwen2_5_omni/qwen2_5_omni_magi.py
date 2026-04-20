# SPDX-License-Identifier: Apache-2.0
"""Optional MagiCompiler integration for Qwen2.5-Omni Token2Wav DiT.

Set ``VLLM_OMNI_MAGI_COMPILER=1`` and install `MagiCompiler` (see SandAI-org/MagiCompiler)
to compile the DiT transformer stack. If MagiCompiler is not installed, decorators fall
back to no-ops (same behaviour as ``magi_human_dit``).
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

try:
    from magi_compiler.api import magi_compile as _magi_compile
except Exception:  # pragma: no cover - optional dependency

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


# Align with LightX2V-MagiCompiler practice: use MagiCompiler as the single
# torch.compile / Inductor path; disable vLLM's own compilation stack on the
# code2wav stage to avoid double-compilation and conflicting graph passes.
_MAGI_CODE2WAV_VLLM_COMPILATION_OVERRIDES: dict[str, Any] = {
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


def merge_vllm_compilation_config_for_magi(engine_args_dict: dict[str, Any]) -> None:
    """When MagiCompiler is enabled for Qwen2.5-Omni code2wav, turn off vLLM compilation.

    vLLM's ``VLLM_COMPILE`` mode and fusion passes can fight with ``@magi_compile``
    (nested Dynamo / Inductor).  This forces ``CompilationMode.NONE`` and disables
    the main custom fusion flags for the Token2Wav stage only.
    """
    if not is_magi_compiler_enabled():
        return
    if engine_args_dict.get("model_arch") != "Qwen2_5OmniForConditionalGeneration":
        return
    if engine_args_dict.get("model_stage") != "code2wav":
        return

    merged = _compilation_config_to_dict(engine_args_dict.get("compilation_config"))
    magi_pc = copy.deepcopy(_MAGI_CODE2WAV_VLLM_COMPILATION_OVERRIDES["pass_config"])
    user_pc = merged.get("pass_config")
    if isinstance(user_pc, dict):
        # Mandatory disables win over user-enabled fusions for Magi compatibility.
        merged["pass_config"] = {**user_pc, **magi_pc}
    else:
        merged["pass_config"] = magi_pc
    merged["mode"] = _MAGI_CODE2WAV_VLLM_COMPILATION_OVERRIDES["mode"]
    engine_args_dict["compilation_config"] = merged
    logger.info(
        "[MagiCompiler] Disabled vLLM torch.compile stack for code2wav "
        "(mode=NONE, fusion passes off for norm/act/attn quant)."
    )


@_magi_compile(
    enable_if=is_magi_compiler_enabled,
    model_tag="qwen2_5_omni_token2wav_dit",
    dynamic_arg_dims={
        # [batch, seq, hidden] — seq varies across requests/steps
        "hidden_states": 1,
        "time_embedding": 0,
        "cos": 1,
        "sin": 1,
        # [batch, heads, seq, seq] from ``_create_block_diff`` — only mark the
        # attention map spatial dims (2, 3) as dynamic.  Do not mark dim 0:
        # Dynamo/SDPA specialize batch to the warmup size (e.g. 2) and
        # ``mark_dynamic`` on dim 0 then fails with ConstraintViolationError.
        "block_diff": [2, 3],
    },
)
class Qwen2OmniDiTTransformerStack(nn.Module):
    """Runs ``DiTDecoderLayer`` blocks; subject to MagiCompiler when enabled."""

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
