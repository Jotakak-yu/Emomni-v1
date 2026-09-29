"""CosyVoice3 inference using the bundled upstream runtime."""
import os
import hashlib
from pathlib import Path
import sys
import tempfile
import threading
from contextlib import closing
from types import MethodType


SOURCE = Path(__file__).resolve().parents[2] / "third_party/CosyVoice"


def load_inference_yaml(stream, overrides=None):
    from hyperpyyaml import load_hyperpyyaml
    from ruamel.yaml import YAML

    text = stream.read()
    # Upstream YAML also instantiates training-only dataset callbacks.
    ignored = {key.value: None for key, value in YAML(typ="base").compose(text).value
               if value.tag.startswith(("!name:cosyvoice.dataset.", "!new:cosyvoice.dataset."))}
    return load_hyperpyyaml(text, overrides={**ignored, **(overrides or {})})


def normalization_assets():
    """Fetch only the four pinned FST files; reuse them without Hub metadata calls."""
    import requests
    files = {
        "en/tn/tagger.fst": "245e2dc9174cdd007a8e9e50f3339773d1adbbc7535b71cf67478dc1683cc3ec",
        "en/tn/verbalizer.fst": "03155c88f317b2795969e264c19f87faf98b9853d5ac631e419bacdc3b3ee15a",
        "zh/tn/tagger.fst": "cf341314c51f7ce59049f3b2c42f0ce8fd71e6d08d4d6969613aad384a5e2ae8",
        "zh/tn/verbalizer.fst": "5a13cd679dd54637d12d2bd1bd33ee2165d91c867e14468c93195af02256e5da",
    }
    root = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "wetext"
    for name, digest in files.items():
        path = root / name
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
            continue
        with requests.get("https://modelscope.cn/models/pengzhendong/wetext/resolve/"
                          f"master/{name}",
                          timeout=(10, 120)) as response:
            response.raise_for_status()
            data = response.content
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError(f"Invalid WeText asset: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
    return root


def inference_vllm(llm, lm_input, sampling, min_len, max_len, uuid):
    import torch
    from vllm import SamplingParams

    # The service serializes synthesis. Honor engine completion as well as EOS;
    # upstream's queue loop can otherwise wait forever at the context limit.
    params = SamplingParams(top_k=sampling, stop_token_ids=llm.stop_token_ids,
                            min_tokens=min_len, max_tokens=max_len)
    llm.vllm.add_request(uuid, {"prompt_embeds": lm_input.squeeze(0).to(torch.bfloat16)}, params)
    offset = 0
    try:
        while llm.vllm.has_unfinished_requests():
            cancelled = getattr(llm, "cancelled", None)
            if cancelled is not None and cancelled.is_set():
                return
            for output in llm.vllm.step():
                tokens = output.outputs[0].token_ids
                for token in tokens[offset:]:
                    if token not in llm.stop_token_ids:
                        yield token
                offset = len(tokens)
                if output.finished:
                    return
    finally:
        llm.vllm.abort_request(uuid)


def load_vllm(model, root):
    from filelock import FileLock
    from cosyvoice.utils.file_utils import export_cosyvoice2_vllm

    os.environ["VLLM_USE_V1"] = "0"
    from vllm import EngineArgs, LLMEngine

    weights = Path(root).resolve() / "llm.pt"
    identity = f"{weights}:{weights.stat().st_size}:{weights.stat().st_mtime_ns}"
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "cosyvoice3-vllm"
    cache.mkdir(parents=True, exist_ok=True)
    exported = cache / hashlib.sha256(identity.encode()).hexdigest()[:16]
    with FileLock(str(exported) + ".lock"):
        if not exported.is_dir():
            with tempfile.TemporaryDirectory(dir=cache) as temporary:
                output = Path(temporary) / "model"
                export_cosyvoice2_vllm(model.llm, str(output), model.device)
                output.rename(exported)
    model.llm.vllm = LLMEngine.from_engine_args(EngineArgs(
        model=str(exported), skip_tokenizer_init=True, enable_prompt_embeds=True,
        gpu_memory_utilization=0.1, max_num_seqs=1,
        # Allow a 30-second reference plus a normalized text segment. A single
        # 4096-token cache needs only 48 MiB, not 10% of the entire GPU.
        max_model_len=4096, block_size=16, num_gpu_blocks_override=256, swap_space=0,
    ))
    model.llm.inference_wrapper = MethodType(inference_vllm, model.llm)
    del model.llm.llm.model.model.layers


class CosyVoice3Backend:
    voices = ["default"]

    def __init__(self):
        from huggingface_hub import snapshot_download

        sys.path[:0] = [str(SOURCE), str(SOURCE / "third_party/Matcha-TTS")]
        from cosyvoice.cli import cosyvoice
        import wetext

        root = os.environ.get("COSYVOICE_MODEL_DIR") or snapshot_download(
            "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
            revision="29e01c4e8d000f4bcd70751be16fa94bf3d85a18",
            allow_patterns=["cosyvoice3.yaml", "campplus.onnx", "speech_tokenizer_v3.onnx",
                            "llm.pt", "flow.pt", "hift.pt", "CosyVoice-BlankEN/*"],
        )
        original = cosyvoice.load_hyperpyyaml
        normalizer = wetext.Normalizer
        normalization = normalization_assets()

        def make_normalizer(**kwargs):
            lang = "zh" if "remove_erhua" in kwargs else "en"
            directory = normalization / lang / "tn"
            return normalizer(tagger_path=str(directory / "tagger.fst"),
                              verbalizer_path=str(directory / "verbalizer.fst"), lang=lang, **kwargs)

        try:
            cosyvoice.load_hyperpyyaml = load_inference_yaml
            wetext.Normalizer = make_normalizer
            self.model = cosyvoice.CosyVoice3(root, load_trt=False, load_vllm=False, fp16=False)
        finally:
            cosyvoice.load_hyperpyyaml = original
            wetext.Normalizer = normalizer
        if not self.model.frontend.text_frontend:
            raise RuntimeError("CosyVoice3 text normalization failed to initialize")
        load_vllm(self.model.model, root)
        self.generation_error = None
        self.generation_thread = None
        original_job = self.model.model.llm_job

        def llm_job(*args):
            self.generation_thread = threading.current_thread()
            try:
                original_job(*args)
            except Exception as exc:
                self.generation_error = exc
            finally:
                self.model.model.llm_end_dict[args[-1]] = True

        self.model.model.llm_job = llm_job
        self.sample_rate = self.model.sample_rate
        self.prompt = str(SOURCE / "asset/zero_shot_prompt.wav")
        self.prompt_text = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
        self.model.add_zero_shot_spk(self.prompt_text, self.prompt, "default")

    def synthesize(self, text, voice, speed, reference_audio=None, reference_text=None,
                   stream=False, cancelled=None):
        prompt_text = reference_text or self.prompt_text
        if "<|endofprompt|>" not in prompt_text:
            prompt_text = "You are a helpful assistant.<|endofprompt|>" + prompt_text
        model = self.model.model
        hop = model.token_hop_len
        for segment in self.model.frontend.text_normalize(text, split=True):
            if cancelled is not None and cancelled.is_set():
                break
            self.generation_error = self.generation_thread = None
            model.llm.cancelled = cancelled
            try:
                # Upstream grows token_hop_len on the shared model. Reset it for
                # every normalized segment so later requests retain low latency.
                model.token_hop_len = hop
                with closing(self.model.inference_zero_shot(
                    segment, prompt_text, reference_audio or self.prompt,
                    zero_shot_spk_id="" if reference_audio else "default",
                    stream=stream, speed=speed, text_frontend=False,
                )) as outputs:
                    for output in outputs:
                        if self.generation_error is not None:
                            raise self.generation_error
                        if cancelled is not None and cancelled.is_set():
                            break
                        yield output["tts_speech"].detach().cpu().numpy().reshape(-1)
                        if cancelled is not None and cancelled.is_set():
                            break
            finally:
                if self.generation_thread is not None:
                    self.generation_thread.join()
                model.llm.cancelled = None
                model.token_hop_len = hop
                for cache in (model.tts_speech_token_dict, model.llm_end_dict, model.hift_cache_dict):
                    cache.clear()
