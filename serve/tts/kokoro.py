"""Kokoro inference with Chinese/English pronunciation on CUDA."""
import re
import torch


class KokoroBackend:
    sample_rate = 24000
    def __init__(self, root):
        from kokoro import KModel
        from misaki.zh import ZHG2P
        from phonemizer.backend import EspeakBackend
        if not torch.cuda.is_available():
            raise RuntimeError('Kokoro requires CUDA; check the driver and PyTorch installation')
        self.model = KModel(repo_id='hexgrad/Kokoro-82M', config=str(root / 'config.json'),
                            model=str(root / 'kokoro-v1_0.pth')).eval().cuda()
        self.vocab = self.model.vocab
        self.zh = ZHG2P(version=None)
        self.en = {lang: EspeakBackend(lang, preserve_punctuation=True, with_stress=True) for lang in ['en-us', 'en-gb']}
        self.styles = {p.stem: torch.load(p, map_location='cpu', weights_only=True).cuda()
                       for p in sorted((root / 'voices').glob('*.pt'))
                       if re.fullmatch(r'(?:[abz][fm])_[a-z0-9_]+', p.stem)}
        self.voices = list(self.styles)

    def synthesize(self, text, voice, speed, **kwargs):
        if voice not in self.voices:
            raise ValueError('invalid_voice')
        for language, segment, segment_voice in self.language_segments(text, voice):
            if language == 'zh':
                phonemes, _ = self.zh(segment)
            else:
                phonemes = self.en['en-gb' if segment_voice.startswith('b') else 'en-us'].phonemize([segment], strip=True)[0]
            yield from self.render_phonemes(phonemes, segment_voice, speed)

    @staticmethod
    def language_segments(text, voice):
        """Keep all characters; switch pronunciation and gender-matched voice at script boundaries."""
        # Digits and punctuation follow their surrounding language. English
        # abbreviations are passed intact to eSpeak rather than dropped by ZHG2P.
        parts = re.findall(r'[A-Za-z]+(?:[\s\d\-\x27.:/]+[A-Za-z\d]+)*|[^A-Za-z]+', text)
        result = []
        for part in parts:
            language = 'en' if re.search(r'[A-Za-z]', part) else 'zh' if re.search(r'[\u3400-\u9fff]', part) else (result[-1][0] if result else ('zh' if voice.startswith('z') else 'en'))
            if result and (not any(c.isalnum() for c in part) or result[-1][0] == language):
                result[-1][1] += part
            else:
                result.append([language, part])
        male = len(voice) > 1 and voice[1] == 'm'
        for language, part in result:
            selected = voice
            if language == 'zh' and not voice.startswith('z'):
                selected = 'zm_yunxi' if male else 'zf_xiaoxiao'
            elif language == 'en' and voice.startswith('z'):
                selected = 'am_adam' if male else 'af_heart'
            if any(c.isalnum() for c in part):
                yield language, part, selected

    @torch.inference_mode()
    def render_phonemes(self, phonemes, voice, speed):
        tokens = [self.vocab[c] for c in phonemes if c in self.vocab]
        if not tokens:
            raise ValueError('empty_phonemes')
        styles = self.styles[voice]
        # Never silently truncate a long utterance. Frontends normally split by sentence.
        for offset in range(0, len(tokens), 480):
            chunk = tokens[offset:offset + 480]
            audio, _ = self.model.forward_with_tokens(
                torch.tensor([[0, *chunk, 0]], device="cuda"),
                styles[min(len(chunk), len(styles)-1)], speed)
            yield audio.float().cpu().numpy().reshape(-1)
