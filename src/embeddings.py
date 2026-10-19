"""Embedding helpers: one cached model per Python process, plus the three UDF styles we benchmark.

Embeddings are L2-normalised (normalize_embeddings=True), so cosine similarity == dot product.
"""

import os
from typing import Iterator

from common import EMBED_BATCH_SIZE, EMBEDDING_MODEL, env_int

_MODEL = None  # per-process cache (each Spark Python worker has its own)


def get_model(threads: int | None = None):
    """Load the SentenceTransformer once per process and cache it."""
    global _MODEL
    if _MODEL is None:
        import torch
        torch.set_num_threads(threads or env_int("EMBED_TORCH_THREADS", 1))
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    return _MODEL


def encode_texts(texts: list[str], batch_size: int = EMBED_BATCH_SIZE, threads: int | None = None):
    """Driver-side encode (e.g. for the search query). Returns float32 ndarray [n, dim], unit norm."""
    import torch
    if threads is None:
        threads = os.cpu_count() or 4  # driver isn't competing with 10 workers here
    return get_model(threads).encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                     show_progress_bar=False).astype("float32")


# ---------------------------------------------------------------------------------------------
# UDF 1 — NAIVE: plain Python UDF, one row per call -> model.encode(one string).
# The model is cached per worker, so the comparison with the batched version is fair: the only
# difference left is "one row per call + per-row Python<->JVM serialization" vs "batches + Arrow".
# ---------------------------------------------------------------------------------------------
def naive_udf():
    from pyspark.sql import functions as F
    from pyspark.sql.types import ArrayType, FloatType

    def embed_one(text):
        if text is None:
            return None
        return get_model().encode(text, normalize_embeddings=True, show_progress_bar=False).tolist()

    return F.udf(embed_one, ArrayType(FloatType()))


# ---------------------------------------------------------------------------------------------
# UDF 2 — BATCHED: iterator-style pandas_udf. The model loads once per *task* (cached per worker),
# and each Arrow batch of rows goes through model.encode(..., batch_size=64) in one call.
# ---------------------------------------------------------------------------------------------
def batched_udf(batch_size: int = EMBED_BATCH_SIZE):
    import pandas as pd
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import ArrayType, FloatType

    @pandas_udf(ArrayType(FloatType()))
    def embed_batches(batches: Iterator[pd.Series]) -> Iterator[pd.Series]:
        model = get_model()  # once per worker process, not per batch
        for texts in batches:
            vecs = model.encode(texts.fillna("").tolist(), batch_size=batch_size,
                                normalize_embeddings=True, show_progress_bar=False)
            yield pd.Series(list(vecs.astype("float32")))

    return embed_batches


# ---------------------------------------------------------------------------------------------
# UDF 3 — ANTI-PATTERN: loads the model on EVERY row. Only ever run on a tiny N.
# ---------------------------------------------------------------------------------------------
def reload_each_row_udf():
    from pyspark.sql import functions as F
    from pyspark.sql.types import ArrayType, FloatType

    def embed_one_reload(text):
        import torch
        torch.set_num_threads(env_int("EMBED_TORCH_THREADS", 1))
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")  # <- the sin
        return model.encode(text, normalize_embeddings=True, show_progress_bar=False).tolist()

    return F.udf(embed_one_reload, ArrayType(FloatType()))
