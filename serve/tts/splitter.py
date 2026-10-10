"""Incremental speech clauses: a short first phrase, longer buffered continuations."""
import re
import time


def units(text):
    return len(re.findall(r"[\u3400-\u9fff]|[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*", text))


def boundaries(text, final=False):
    """Yield safe punctuation boundaries without splitting decimals or abbreviations."""
    depth = 0
    for i, char in enumerate(text):
        if char in '（(【[“「':
            depth += 1
        elif char in '）)】]”」':
            depth = max(0, depth - 1)
        if depth:
            continue
        strong = char in '。！？!?；;\n'
        weak = char in '，,：:'
        if char in ',:' and i + 1 == len(text) and not final:
            continue  # A number, time, or URL may continue in the next chunk.
        if char == ':' and text[i + 1:i + 3] == '//':
            continue
        if char == '.':
            if i + 1 == len(text) and not final:
                continue  # The next streamed character may complete a decimal.
            prefix = text[:i + 1]
            if i + 1 < len(text) and not text[i + 1].isspace():
                continue
            if re.search(r'\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|[A-Z])\.$', prefix, re.I):
                continue
            strong = True
        if weak and i and i + 1 < len(text) and text[i - 1].isdigit() and text[i + 1].isdigit():
            continue
        if strong or weak:
            yield i + 1, strong


class SmartSplitter:
    def __init__(self):
        self.buffer = ''
        self.first = True
        self.since = time.monotonic()

    def feed(self, text='', *, final=False, audio_ahead=0.0, now=None):
        now = time.monotonic() if now is None else now
        if not self.buffer:
            self.since = now
        self.buffer += text
        cuts = list(boundaries(self.buffer, final))
        cut = 0
        for end, strong in cuts:
            size = units(self.buffer[:end])
            if size == 0:
                continue
            if self.first:
                ready = strong or size >= 8
            else:
                # When playback has a cushion, preserve longer prosodic units.
                target = 36 if audio_ahead >= 2.5 else 16
                ready = strong and size >= target
                ready |= size >= 8 and now - self.since >= .65 and audio_ahead < 2.5
                ready |= size >= 80
            if ready:
                cut = end
                break
        if final and not cut:
            cut = len(self.buffer)
        if not cut and len(self.buffer) > 240:
            # Prefer punctuation or whitespace before the existing hard limit.
            safe = [end for end, _ in cuts if end <= 240]
            spaces = [m.end() for m in re.finditer(r'\s+', self.buffer[:240])]
            cut = safe[-1] if safe else spaces[-1] if spaces else 240
        if not cut:
            return []
        chunk, self.buffer = self.buffer[:cut], self.buffer[cut:]
        self.first = False
        self.since = now
        return [chunk]
