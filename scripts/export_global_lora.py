"""Export a complete, unquantized Emomni global-LoRA checkpoint and verify reload."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoTokenizer, WhisperFeatureExtractor
from src.modeling_emomni import EmomniModel


def digest(tensor):
    data = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(data.numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src', type=Path, required=True)
    parser.add_argument('--dst', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    src, dst = args.src.resolve(), args.dst.resolve()
    config = json.loads((src / 'config.json').read_text())
    if config.get('lora_scope') != 'global' or not config.get('lora_config'):
        parser.error('Expected an unmerged global-LoRA checkpoint')
    if config.get('quantization_config'):
        parser.error('Use the original unquantized checkpoint, not bnb weights')
    if dst.exists():
        parser.error(f'Destination already exists: {dst}')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA is unavailable')

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(src, trust_remote_code=True, local_files_only=True)
    extractor = WhisperFeatureExtractor.from_pretrained(src, local_files_only=True)
    # Preserve checkpoint parameter precision and FP32 RoPE buffers.
    model = EmomniModel.from_pretrained(str(src), torch_dtype='auto',
                                       local_files_only=True).to(args.device).eval()
    if getattr(model, 'is_quantized', False):
        raise ValueError('Quantized models cannot use this exporter')
    with torch.inference_mode():
        model.merge_lora()
    assert not model.config.lora_config
    assert not any('lora_A' in name or 'lora_B' in name for name, _ in model.named_parameters())

    staging = dst.with_name(dst.name + '.incomplete')
    staging.mkdir(parents=True, exist_ok=False)
    # Independent CPU storage avoids shared-tensor ambiguity in custom-model serialization.
    state = {name: value.detach().cpu().clone().contiguous()
             for name, value in model.state_dict().items()}
    fingerprints = {name: {'shape': list(value.shape), 'dtype': str(value.dtype),
                           'sha256': digest(value)} for name, value in state.items()}
    model.save_pretrained(staging, state_dict=state, safe_serialization=True,
                          max_shard_size='4GB')
    tokenizer.save_pretrained(staging)
    extractor.save_pretrained(staging)
    if (src / 'generation_config.json').is_file():
        shutil.copy2(src / 'generation_config.json', staging / 'generation_config.json')
    del state, model
    gc.collect()
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()

    reloaded = EmomniModel.from_pretrained(str(staging), torch_dtype='auto',
                                          local_files_only=True).to(args.device).eval()
    actual = reloaded.state_dict()
    assert set(actual) == set(fingerprints), 'Reload changed state-dict keys'
    for name, value in actual.items():
        reference = fingerprints[name]
        assert list(value.shape) == reference['shape'], name
        assert str(value.dtype) == reference['dtype'], name
        assert digest(value) == reference['sha256'], f'Reload changed weights: {name}'
    assert not reloaded.config.lora_config
    report = dict(source=src.name, destination=dst.name, scope='global',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  torch=torch.__version__, elapsed_seconds=time.perf_counter()-started,
                  tensors_verified=len(fingerprints), state=fingerprints,
                  reload_verified=True)
    (staging / 'merge-verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    if dst.exists():
        raise FileExistsError(dst)
    staging.rename(dst)
    print(json.dumps({k: v for k, v in report.items() if k != 'state'},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
