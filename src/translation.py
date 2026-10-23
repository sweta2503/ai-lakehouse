"""Local machine translation (Portuguese -> English) with a Marian model. Free, offline, CPU.

Display only: embeddings and the IVF index are built from the ORIGINAL text.
Quirk handled here: the Romance->English Marian model appends junk to short inputs without terminal
punctuation ("Muito bom" -> "Very good to the"), so we add a period before translating and strip it after.
"""
import re

import pandas as pd
from pyspark.sql.functions import pandas_udf

from common import TRANSLATE_MODEL, env_int

_BUNDLE = None  # (tokenizer, model) cached per Python process
_ENDS = re.compile(r"[.!?…]\s*$")


def _load():
    global _BUNDLE
    if _BUNDLE is None:
        import torch
        torch.set_num_threads(env_int("EMBED_TORCH_THREADS", 1))
        from transformers import MarianMTModel, MarianTokenizer
        _BUNDLE = (MarianTokenizer.from_pretrained(TRANSLATE_MODEL), MarianMTModel.from_pretrained(TRANSLATE_MODEL).eval())
    return _BUNDLE


def translate_texts(texts: list[str], batch_size: int = 32) -> list[str]:
    import torch
    tok, model = _load()
    added = [not _ENDS.search(t) for t in texts]
    src = [t + "." if a else t for t, a in zip(texts, added)]
    order = sorted(range(len(src)), key=lambda i: len(src[i]))  # similar lengths per batch = less padding
    out = [""] * len(src)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        enc = tok([src[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            gen = model.generate(**enc, num_beams=1, max_new_tokens=256)
        for i, t in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
            out[i] = t[:-1] if added[i] and t.endswith(".") else t
    return out


def translate_udf():
    """Factory (a module-level pandas_udf would need a live SparkSession at import time)."""
    @pandas_udf("string")
    def _udf(texts: pd.Series) -> pd.Series:
        return pd.Series(translate_texts(texts.fillna("").tolist()))
    return _udf
