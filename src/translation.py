"""Portuguese -> English machine translation, free and local (CPU).

Model: unicamp-dl/translation-pt-en-t5 run through CTranslate2 (int8). We compared four free models on real Olist
reviews (see docs/translation.md): the Marian opus-mt models loop or produce unrelated text on short reviews,
NLLB-200-600M invents sentences, and this T5 stayed faithful. CTranslate2 makes it ~4x faster than plain PyTorch.

Guards around the model (T5 drops sentences from long inputs and mangles all-caps / emoji-only text):
  * long reviews are split into sentences and translated piece by piece,
  * ALL-CAPS text is lower-cased first, runs of "......" are collapsed, no-letter text (emoji, digits) is kept as is,
  * an output that is empty or absurdly longer than its input is rejected and the original text is kept.
"""
import json
import re
from pathlib import Path

from common import DATA_DIR, env, env_int, get_logger

log = get_logger("translation")

HF_MODEL = env("TRANSLATE_MODEL", "unicamp-dl/translation-pt-en-t5")
CT2_DIR = DATA_DIR / "english" / "ct2_model"
PREFIX = "translate Portuguese to English: "

_ENDS = re.compile(r"[.!?…]\s*$")
_HAS_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-Ý0-9])")
_RUNS = re.compile(r"([^\w\s])\1{2,}")  # "........" "!!!!!" -> "..." "!!!"

_BUNDLE = None  # (tokenizer, ctranslate2.Translator), cached per process


def _load():
    global _BUNDLE
    if _BUNDLE is None:
        import ctranslate2
        from transformers import T5Tokenizer
        if not (CT2_DIR / "model.bin").exists():
            log.info("converting %s to CTranslate2 int8 (one-time, ~1 min) -> %s", HF_MODEL, CT2_DIR)
            ctranslate2.converters.TransformersConverter(HF_MODEL).convert(str(CT2_DIR), quantization="int8", force=True)
        tok = T5Tokenizer.from_pretrained(HF_MODEL)
        tr = ctranslate2.Translator(str(CT2_DIR), device="cpu", inter_threads=1,
                                    intra_threads=env_int("TRANSLATE_THREADS", 4))
        _BUNDLE = (tok, tr)
    return _BUNDLE


def split_sentences(text: str, min_len: int = 25) -> list[str]:
    """Split long text on sentence ends (capital letter or digit follows). Short pieces are merged into the
    previous one, so abbreviations like 'mt. bom' or 'conf. do pgto' are not torn apart."""
    if len(text) < 80:
        return [text]
    parts: list[str] = []
    for p in _SENT_SPLIT.split(text):
        if parts and len(p) < min_len:
            parts[-1] += " " + p
        else:
            parts.append(p)
    return parts


def _prepare(seg: str) -> str:
    seg = _RUNS.sub(lambda m: m.group(1) * 3, seg)
    letters = [c for c in seg if c.isalpha()]
    if len(letters) > 3 and all(c.isupper() for c in letters):
        seg = seg.lower()
    return seg


def _run_model(segments: list[str], batch_size: int = 32) -> list[str]:
    tok, tr = _load()
    added = [not _ENDS.search(s) for s in segments]
    src = [PREFIX + (s + "." if a else s) for s, a in zip(segments, added)]
    toks = [tok.convert_ids_to_tokens(tok.encode(s, truncation=True, max_length=256)) for s in src]
    res = tr.translate_batch(toks, beam_size=1, max_decoding_length=200, max_batch_size=batch_size,
                             batch_type="examples", repetition_penalty=1.15, no_repeat_ngram_size=4)
    out = []
    for r, a in zip(res, added):
        t = tok.decode(tok.convert_tokens_to_ids(r.hypotheses[0]), skip_special_tokens=True).strip()
        out.append(t[:-1] if a and t.endswith(".") else t)
    return out


def translate_texts(texts: list[str]) -> tuple[list[str], int]:
    """Translate a list of distinct texts. Returns (english_texts, number_of_texts_kept_in_Portuguese)."""
    plan: list[list] = []          # per text: list of segments (str = translate me, ('keep', s) = leave alone)
    to_translate: list[str] = []
    for text in texts:
        pieces = []
        for seg in split_sentences(text):
            if not _HAS_LETTER.search(seg):
                pieces.append(("keep", seg))
            else:
                pieces.append(len(to_translate))
                to_translate.append(_prepare(seg))
        plan.append(pieces)

    # similar lengths per batch = less padding
    order = sorted(range(len(to_translate)), key=lambda i: len(to_translate[i]))
    done = [""] * len(to_translate)
    for s in range(0, len(order), 256):
        idx = order[s:s + 256]
        for i, t in zip(idx, _run_model([to_translate[i] for i in idx])):
            done[i] = t

    result, rejected = [], 0
    for text, pieces in zip(texts, plan):
        parts, bad = [], False
        for p in pieces:
            if isinstance(p, tuple):
                parts.append(p[1])
            else:
                src, out = to_translate[p], done[p]
                if not out or len(out) > 3 * len(src) + 20:
                    bad = True
                parts.append(out)
        if bad:
            rejected += 1
            result.append(text)
        else:
            result.append(" ".join(parts).strip())
    return result, rejected


def load_cache(path: Path) -> dict[str, str]:
    cache: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                cache[d["pt"]] = d["en"]
    return cache
