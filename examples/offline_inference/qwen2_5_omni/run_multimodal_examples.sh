#!/usr/bin/env bash
# 批量跑 Qwen2.5-Omni end2end 多模态样例（默认使用 vLLM 内置音视频图素材）。
# 用法：
#   bash run_multimodal_examples.sh
#   OUTPUT_ROOT=/tmp/mm_demo PYTHON=python bash run_multimodal_examples.sh
# 可选：自行指定本地文件（覆盖内置素材）
#   VIDEO_PATH=/path/to.mp4 IMAGE_PATH=/path/to.jpg AUDIO_PATH=/path/to.wav bash run_multimodal_examples.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"
END2END="$ROOT/end2end.py"
OUT="${OUTPUT_ROOT:-$ROOT/mm_demo_outputs}"
mkdir -p "$OUT"

extra_image=()
extra_video=()
extra_audio=()
[[ -n "${IMAGE_PATH:-}" ]] && extra_image=(--image-path "$IMAGE_PATH")
[[ -n "${VIDEO_PATH:-}" ]] && extra_video=(--video-path "$VIDEO_PATH")
[[ -n "${AUDIO_PATH:-}" ]] && extra_audio=(--audio-path "$AUDIO_PATH")

run_one() {
  local qtype="$1"
  local subdir="$2"
  shift 2
  echo ""
  echo "========== ${qtype} -> ${OUT}/${subdir} =========="
  "$PY" "$END2END" --query-type "$qtype" --output-dir "$OUT/$subdir" --num-prompts 1 "$@"
}

# 单图（默认识图 cherry_blossom）
run_one use_image mm_image "${extra_image[@]}"

# 单视频（默认 baby_reading）
run_one use_video mm_video "${extra_video[@]}"

# 单段音频（默认 mary_had_lamb）
run_one use_audio mm_audio "${extra_audio[@]}"

# 两段音频对比（默认 winning_call + mary_had_lamb；若设 AUDIO_PATH 则第一段用你的、第二段仍用内置）
run_one use_multi_audios mm_multi_audios "${extra_audio[@]}"

# 视频 + 从视频轨抽音频（use_audio_in_video）
run_one use_audio_in_video mm_audio_in_video "${extra_video[@]}"

# 图 + 音 + 视频 齐上阵（默认三套内置素材）
run_one use_mixed_modalities mm_mixed \
  "${extra_image[@]}" "${extra_video[@]}" "${extra_audio[@]}"

echo ""
echo "全部完成。输出目录: $OUT"
