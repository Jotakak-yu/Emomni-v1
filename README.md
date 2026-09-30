# Emomni

[![code](https://img.shields.io/badge/Github-Code-keygen.svg?logo=github)](https://github.com/Jotakak-yu/Emomni-v1) [![models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging_Face-Models-blue.svg)](https://huggingface.co/Jotakak/Emomni-v1) [![arXiv](https://img.shields.io/badge/arXiv-uploading-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/xxxx.xxxxx)

## Models

| Size | Original weights | 4-bit weights |
| --- | --- | --- |
| Base (default) | [Emomni-v1](https://huggingface.co/Jotakak/Emomni-v1) | [Emomni-v1-bnb-4bit](https://huggingface.co/Jotakak/Emomni-v1-bnb-4bit) |
| Small | [Emomni-v1-small](https://huggingface.co/Jotakak/Emomni-v1-small) | [Emomni-v1-small-bnb-4bit](https://huggingface.co/Jotakak/Emomni-v1-small-bnb-4bit) |

## Quick start

- **Platform:** Linux x86_64 with a BF16-capable NVIDIA CUDA GPU. On Windows, use CUDA-enabled WSL2; native Windows and macOS are unsupported by this server setup.
- **Environment:** Python 3.11 virtualenv or an existing Conda environment.
- **Setup:** The launcher installs missing dependencies and downloads models on first use. Install `ffmpeg` and `espeak-ng` with your system package manager if prompted.

```bash
git clone https://github.com/Jotakak-yu/Emomni-v1.git
cd Emomni-v1
python3.11 -m venv .venv
source .venv/bin/activate

bash scripts/run_serve.sh -s base
```

Open the [WebUI](http://localhost:7860). Remote microphone input requires HTTPS.

### Model selection

```bash
# Published models: base / small, each with optional 4-bit weights
bash scripts/run_serve.sh                    # base, original BF16
bash scripts/run_serve.sh -s base -q 4
bash scripts/run_serve.sh -s small
bash scripts/run_serve.sh -s small -q 4

# Custom checkpoint
bash scripts/run_serve.sh --model /path/to/model
```

### Speech replies (TTS)

Choose the backend before startup. Kokoro is the default.

```bash
bash scripts/run_serve.sh -s base --tts-mode CosyVoice3

bash scripts/run_serve.sh --no-tts                       # Text replies only
bash scripts/run_serve.sh --tts-url http://host:8880/v1  # Existing TTS service
```

- **Runtime:** Kokoro uses PyTorch; CosyVoice3 uses vLLM. Both warm up before serving and can share the main model's Python environment.
- **CosyVoice3:** Higher-quality speech, a bundled reference voice, and voice cloning via uploaded reference audio and transcript. Normal-speed output streams within sentences; other speeds use complete sentences.
- **Downloads:** CosyVoice3 dependencies and submodules are installed only when selected. Weights come from Hugging Face; its text-normalization assets come from ModelScope.

### Local weights and GPU

- `KOKORO_MODEL_DIR`: local Kokoro weights (`config.json`, `kokoro-v1_0.pth`, and `voices/*.pt` from `hexgrad/Kokoro-82M`).
- `COSYVOICE_MODEL_DIR`: local CosyVoice3 weights. Exports and normalization assets are cached under `HF_HOME`.
- `TTS_GPU`: TTS GPU; defaults to the first worker GPU.

### Service management

```bash
bash scripts/run_serve.sh --check-deps --model /path/to/model
bash scripts/run_serve.sh --logs all
bash scripts/run_serve.sh --stop
bash scripts/run_serve.sh --help
```

Use `--no-auto-install` to disable automatic dependency installation. See `--help` for multi-GPU and other service options.

## Docker

- **Requires:** Docker Compose and NVIDIA Container Toolkit.
- **Defaults:** base BF16, Kokoro, one GPU, localhost port 7860.
- **Storage:** model cache and logs persist in Docker volumes; first startup downloads weights.
- **GPU and ports:** Emomni and TTS share the selected GPU; only the WebUI port is published.

```bash
docker compose up --build -d
# Optional: small 4-bit on host GPU 2
GPU_ID=2 MODEL_SIZE=small MODEL_QUANTIZE=4 docker compose up --build -d
# Optional: CosyVoice3 (initialize submodules before building)
git submodule update --init --recursive
TTS_MODE=CosyVoice3 docker compose up --build -d

docker compose logs -f
docker compose down
```

## Acknowledgements

* [Qwen2.5](https://github.com/QwenLM/Qwen2.5)
* [blsp-emo](https://github.com/cwang621/blsp-emo)
* [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
* [cosyvoice-api](https://github.com/jianchang512/cosyvoice-api)
* [Kokoro](https://github.com/hexgrad/kokoro)
