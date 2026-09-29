"""Check only the selected services and install missing dependencies on first run."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "controller": {"fastapi", "uvicorn", "requests", "numpy"},
    "webui": {"gradio", "requests", "numpy", "soundfile"},
    "worker": {"torch", "transformers", "peft", "safetensors", "numpy", "fastapi", "uvicorn", "requests"},
    "kokoro": {"torch", "transformers", "numpy", "fastapi", "uvicorn", "soundfile",
               "kokoro", "misaki", "phonemizer-fork", "huggingface-hub"},
    "cosyvoice3": {"torch", "transformers", "numpy", "fastapi", "uvicorn", "soundfile"},
}


def requirements(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]


def missing_requirements(specs):
    try:
        from packaging.requirements import Requirement
    except ImportError:
        from pip._vendor.packaging.requirements import Requirement
    missing = []
    for spec in dict.fromkeys(specs):
        req = Requirement(spec)
        try:
            version = importlib.metadata.version(req.name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        if version is None or not req.specifier.contains(version):
            missing.append(spec)
        # Extras (notably misaki[zh]) must also be checked after a partial install.
        elif req.extras:
            for dependency in importlib.metadata.requires(req.name) or []:
                dep = Requirement(dependency)
                if dep.marker is None or any(dep.marker.evaluate({"extra": extra}) for extra in req.extras):
                    try:
                        valid = dep.specifier.contains(importlib.metadata.version(dep.name))
                    except importlib.metadata.PackageNotFoundError:
                        valid = False
                    if not valid:
                        missing.append(spec)
                        break
    return list(dict.fromkeys(missing))


def ensure_system_tools(services):
    tools = (["ffmpeg"] if {"webui", "worker", "cosyvoice3"} & set(services) else []) + (["espeak-ng"] if "kokoro" in services else [])
    missing = [name for name in tools if shutil.which(name) is None]
    if not missing:
        return
    print("缺少系统依赖: " + ", ".join(missing), flush=True)
    raise RuntimeError("请使用系统包管理器安装 " + " ".join(missing)
                       + "; Ubuntu: sudo apt-get install -y " + " ".join(missing))


def ensure_packages(specs, check):
    missing = missing_requirements(specs)
    if missing:
        print("缺少依赖或版本不匹配: " + ", ".join(missing), flush=True)
        if check:
            raise RuntimeError("检查未通过；正常启动将自动安装依赖")
        subprocess.run([sys.executable, "-m", "pip", "install", *specs], check=True)


def uses_bitsandbytes(model):
    path = Path(model) / "config.json"
    if not Path(model).is_dir():
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(model, "config.json"))
    config = json.loads(path.read_text()).get("quantization_config") or {}
    return config.get("quant_method") == "bitsandbytes" or bool(
        config.get("load_in_4bit") or config.get("load_in_8bit")
        or config.get("_load_in_4bit") or config.get("_load_in_8bit"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--services", nargs="+", choices=GROUPS, required=True)
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--quantized", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if sys.version_info < (3, 10):
        raise RuntimeError("需要 Python 3.10+，推荐 Python 3.11")
    import re
    names = set().union(*(GROUPS[name] for name in args.services))
    specs = [spec for spec in requirements(ROOT / "requirements.txt")
             if re.split(r"[<>=\[]", spec)[0] in names]
    print(f"服务环境: {sys.executable}", flush=True)
    ensure_system_tools(args.services)
    imports = {"misaki": "misaki.zh", "phonemizer-fork": "phonemizer",
               "huggingface-hub": "huggingface_hub"}
    modules = [imports.get(name, name) for name in sorted(names)]
    if "worker" in args.services:
        modules += ["src.modeling_emomni", "serve.audio"]
    for backend in ("kokoro", "cosyvoice3"):
        if backend not in args.services:
            continue
        specs.append("python-multipart>=0.0.18")
        if backend == "cosyvoice3":
            specs += requirements(ROOT / "serve/tts/requirements-cosyvoice3.txt")
            source = ROOT / "third_party/CosyVoice"
            if not (source / "third_party/Matcha-TTS/matcha").is_dir():
                if args.check:
                    raise RuntimeError("请运行 git submodule update --init --recursive")
                subprocess.run(["git", "submodule", "update", "--init", "--recursive", "third_party/CosyVoice"], check=True, cwd=ROOT)
            # vLLM pins matching torch/torchaudio wheels for this backend.
            modules += ["torchaudio", "hyperpyyaml", "vllm", "onnxruntime"]
        modules += ["serve.tts.server"]
    ensure_packages(specs, args.check)
    if "worker" in args.services and (args.quantized or any(uses_bitsandbytes(m) for m in args.models if m)):
        print("量化模型: 检查 bitsandbytes", flush=True)
        ensure_packages([*specs, "bitsandbytes>=0.43.2,<1"], args.check)
    subprocess.run([sys.executable, "-c", "import importlib;\n"
                    + "\n".join(f"importlib.import_module({module!r})" for module in modules)], check=True, cwd=ROOT)
    if {"worker", "kokoro", "cosyvoice3"} & set(args.services):
        subprocess.run([sys.executable, "-c", "import torch; assert torch.cuda.is_available(), 'Inference requires CUDA; check the driver and PyTorch installation'"], check=True)
    print("依赖检查通过。", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError, OSError, ValueError) as exc:
        sys.exit(f"启动前检查失败: {exc}")
