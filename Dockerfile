FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 GRADIO_ANALYTICS_ENABLED=False HF_HOME=/cache/huggingface \
    EMOMNI_PID_DIR=/tmp/emomni-pids MODELSCOPE_CACHE=/cache/modelscope OMP_NUM_THREADS=2
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg espeak-ng libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir torch==2.7.1 -r requirements.txt
ARG TTS_MODE=Kokoro
ENV TTS_MODE=${TTS_MODE}
COPY serve/tts/requirements-cosyvoice3.txt ./serve/tts/requirements-cosyvoice3.txt
RUN if [ "$TTS_MODE" = CosyVoice3 ]; then \
         pip install --no-cache-dir -r requirements.txt -r serve/tts/requirements-cosyvoice3.txt; \
       elif [ "$TTS_MODE" != Kokoro ]; then exit 1; fi
ARG MODEL_QUANTIZE=
RUN if [ "$MODEL_QUANTIZE" = 4 ]; then pip install --no-cache-dir 'bitsandbytes>=0.43.2,<1'; \
    elif [ -n "$MODEL_QUANTIZE" ]; then exit 1; fi
COPY src ./src
COPY serve ./serve
COPY scripts/run_serve.sh ./scripts/run_serve.sh
COPY third_party/CosyVoice ./third_party/CosyVoice
RUN if [ "$TTS_MODE" = CosyVoice3 ]; then \
         test -f third_party/CosyVoice/third_party/Matcha-TTS/matcha/__init__.py; \
       else rm -rf third_party; fi \
    && useradd --create-home --uid 1000 emomni \
    && mkdir -p /cache/huggingface /app/logs/serve \
    && chown -R emomni:emomni /cache /app/logs
USER emomni
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=20m --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/', timeout=3); urllib.request.urlopen('http://127.0.0.1:21002/health', timeout=3)"
CMD ["bash", "scripts/run_serve.sh", "--foreground"]
