# Qwen2.5-Omni

## Setup

Please refer to the [stage configuration documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/configuration/stage_configs/) to configure memory allocation appropriately for your hardware setup.

Set `VLLM_OMNI_MODEL` when using a local checkpoint instead of the default Hugging Face model id.

## Run examples

From this directory (`examples/offline_inference/qwen2_5_omni`):

### Multiple prompts (text file + generator mode)

```bash
python end2end.py --output-wav output_audio \
                  --query-type text \
                  --txt-prompts ../qwen3_omni/text_prompts_10.txt \
                  --py-generator
```

### Single mixed-modality prompt (default assets)

```bash
python end2end.py --output-wav output_audio \
                  --query-type use_mixed_modalities
```

### MagiCompiler (optional, talker + code2wav DiT)

```bash
python magi_example.py --query-type text
```

Same CLI flags as `end2end.py`; sets `VLLM_OMNI_MAGI_COMPILER=1` before delegating.
Compiles Stage1 talker Qwen2 decoder layers and Stage2 Token2Wav DiT transformer stack.

### Modality control

To restrict outputs (e.g. text only):

```bash
python end2end.py --output-wav output_audio \
                  --query-type use_mixed_modalities \
                  --modalities text
```

### Local media files

Pass paths explicitly (when omitted, built-in assets are used):

```bash
python end2end.py --query-type use_image --image-path /path/to/image.jpg
python end2end.py --query-type use_video --video-path /path/to/video.mp4
python end2end.py --query-type use_audio --audio-path /path/to/audio.wav

python end2end.py --query-type mixed_modalities \
    --video-path /path/to/video.mp4 \
    --image-path /path/to/image.jpg \
    --audio-path /path/to/audio.wav

python end2end.py --query-type use_audio_in_video --video-path /path/to/video.mp4
```

On ~24 GB GPUs where thinker and code2wav share one device, consider:

`--deploy-config ../../../vllm_omni/deploy/qwen2_5_omni_colocate_24gb.yaml` (paths relative to this folder).

Supported query types include `use_image`, `use_video`, `use_audio`, `mixed_modalities`, `use_audio_in_video`, and `text`.
