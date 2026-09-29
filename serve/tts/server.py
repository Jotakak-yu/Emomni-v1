"""Kokoro / CosyVoice3 service with an OpenAI-compatible speech endpoint."""
import argparse
import asyncio
import queue
from contextlib import asynccontextmanager
import io
import os
import threading
import tempfile
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException, File, Form, UploadFile
from fastapi.responses import Response, StreamingResponse
import numpy as np
from pydantic import BaseModel, Field
import soundfile as sf

from .kokoro import KokoroBackend


def load_backend(kind):
    if kind == "cosyvoice3":
        from .cosyvoice import CosyVoice3Backend
        return CosyVoice3Backend()

    from huggingface_hub import snapshot_download

    local = os.environ.get("KOKORO_MODEL_DIR")
    if not local:
        print("Preparing Kokoro (first run downloads model and voices to the HF cache)...", flush=True)
        local = snapshot_download(
            "hexgrad/Kokoro-82M",
            revision="f3ff3571791e39611d31c381e3a41a3af07b4987",
            allow_patterns=["config.json", "kokoro-v1_0.pth", "voices/[abz]*.pt"],
            local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1",
        )
    return KokoroBackend(Path(local))


class SpeechRequest(BaseModel):
    input: str = Field(min_length=1, max_length=12000)
    voice: str | None = None
    model: str = "kokoro"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    response_format: str = "wav"
    stream: bool = False


def create_app(kind="kokoro", backend=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.backend = backend if backend is not None else load_backend(kind)
        if backend is None:
            print(f"Warming up {kind}...", flush=True)
            voice = "zf_xiaoxiao" if kind == "kokoro" else "default"
            for _ in app.state.backend.synthesize("你好，很高兴与你交流。", voice, 1.0, stream=kind == "cosyvoice3"):
                pass
        yield

    app = FastAPI(lifespan=lifespan)
    slot = threading.Lock()

    @app.get("/health")
    def health():
        return {"status": "healthy", "backend": kind}

    @app.get("/v1/audio/voices")
    def voices():
        return {"backend": kind, "voices": app.state.backend.voices}

    def render(body, reference_samples=None, reference_text=None):
        if body.response_format not in {"wav", "pcm"}:
            raise HTTPException(400, "Only WAV and PCM output are supported")
        if body.stream and (body.response_format != "pcm" or body.speed != 1):
            raise HTTPException(400, "Streaming requires PCM output and speed=1")
        voice = body.voice or ("zf_xiaoxiao" if kind == "kokoro" else "default")
        if not body.input.strip() or voice not in app.state.backend.voices:
            raise HTTPException(400, "Invalid text or voice")
        cancelled = threading.Event()

        def synthesize():
            # Keep clone files alive until the producer has finished, including
            # when an HTTP client disconnects during synthesis.
            with slot, tempfile.TemporaryDirectory() as directory:
                if cancelled.is_set():
                    return
                kwargs = {}
                if reference_samples is not None:
                    wav = Path(directory) / "reference.wav"
                    sf.write(wav, reference_samples, 24000)
                    kwargs = dict(reference_audio=str(wav), reference_text=reference_text)
                yield from app.state.backend.synthesize(
                    body.input, voice, body.speed, stream=body.stream, cancelled=cancelled, **kwargs)

        def pcm(samples):
            return np.clip(np.asarray(samples) * 32768, -32768, 32767).astype("<i2").tobytes()

        if body.stream:
            chunks = queue.Queue(maxsize=2)

            def put(item):
                while not cancelled.is_set():
                    try:
                        chunks.put(item, timeout=0.1)
                        return
                    except queue.Full:
                        pass

            def produce():
                try:
                    # Resume the backend on disconnect so it can stop inference
                    # and release its session caches and CUDA thread.
                    for samples in synthesize():
                        if not cancelled.is_set() and len(samples):
                            put(pcm(samples))
                except Exception as exc:
                    put(exc)
                finally:
                    put(None)

            async def audio_stream():
                producer = threading.Thread(target=produce, daemon=True)
                producer.start()
                try:
                    while True:
                        try:
                            item = await asyncio.to_thread(chunks.get, True, 0.1)
                        except queue.Empty:
                            continue
                        if item is None:
                            break
                        if isinstance(item, Exception):
                            raise item
                        yield item
                finally:
                    cancelled.set()

            return StreamingResponse(audio_stream(), media_type="audio/pcm", headers={
                "X-Audio-Sample-Rate": str(app.state.backend.sample_rate),
                "X-Accel-Buffering": "no", "Cache-Control": "no-store",
            })

        chunks = list(synthesize())
        if not chunks:
            raise HTTPException(422, "No audio generated")
        samples = np.concatenate(chunks)
        if body.response_format == "pcm":
            return Response(pcm(samples), media_type="audio/pcm")
        output = io.BytesIO()
        sf.write(output, samples, app.state.backend.sample_rate, format="WAV", subtype="PCM_16")
        return Response(output.getvalue(), media_type="audio/wav")

    @app.post("/v1/audio/speech")
    def speech(body: SpeechRequest):
        return render(body)

    @app.post("/clone")
    def clone(text: str = Form(min_length=1, max_length=12000),
              reference_text: str = Form(min_length=1, max_length=12000),
              reference_audio: UploadFile = File(), speed: float = Form(1, ge=0.5, le=2),
              stream: bool = Form(False), response_format: str = Form("wav")):
        if kind != "cosyvoice3":
            raise HTTPException(400, "Voice cloning requires CosyVoice3")
        if not reference_text.strip():
            raise HTTPException(400, "Reference text is required")
        data = reference_audio.file.read(20 * 1024 * 1024 + 1)
        if len(data) > 20 * 1024 * 1024:
            raise HTTPException(413, "Reference audio exceeds 20 MB")
        from ..audio import load_audio
        # Decode uploads to a standard WAV before passing them to CosyVoice.
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "reference"
            source.write_bytes(data)
            try:
                samples = load_audio(source, 24000, normalize=False)
            except (ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                raise HTTPException(400, "Invalid reference audio") from exc
        return render(SpeechRequest(input=text, speed=speed, stream=stream, response_format=response_format),
                      reference_samples=samples, reference_text=reference_text)

    return app


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["kokoro", "cosyvoice3"], default="kokoro")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8880)
    args = parser.parse_args()
    uvicorn.run(create_app(args.backend), host=args.host, port=args.port)
