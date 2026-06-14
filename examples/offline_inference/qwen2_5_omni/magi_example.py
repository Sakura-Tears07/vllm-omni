# SPDX-License-Identifier: Apache-2.0
"""Example entrypoint for Qwen2.5-Omni offline inference with MagiCompiler.

This is **(1)** the dedicated Magi sample entry: it enables ``VLLM_OMNI_MAGI_COMPILER``
then delegates to ``end2end.py``. Model-side wiring is **(2)** talker
``apply_magi_to_qwen2_decoder_layers`` and code2wav ``apply_magi_to_decoder_layers``;
**(3)** the implementation lives in ``vllm_omni/.../qwen2_5_omni_magi.py``.

Prerequisites:

- Install ``magi_compiler`` (SandAI-org/MagiCompiler).
- Set model path via ``VLLM_OMNI_MODEL`` if not using the default HF id.

Usage (same flags as ``end2end.py``)::

    python magi_example.py -q text
    python magi_example.py -q use_image -i /path/to/image.jpg --deploy-config ...
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_END2END = _THIS_DIR / "end2end.py"


def main() -> int:
    os.environ.setdefault("VLLM_OMNI_MAGI_COMPILER", "1")
    if not _END2END.is_file():
        print(f"[magi_example] Missing {_END2END}", file=sys.stderr)
        return 1
    argv = [sys.executable, str(_END2END), *sys.argv[1:]]
    return subprocess.call(argv)


if __name__ == "__main__":
    raise SystemExit(main())
