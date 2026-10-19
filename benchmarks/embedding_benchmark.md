# Embedding benchmark: naive UDF vs batched pandas_udf

Model `paraphrase-multilingual-MiniLM-L12-v2` (CPU), 10 Spark cores, 1 torch thread(s) per worker.
Measured on this machine; the model is already loaded in every worker (warm-up excluded).

| Approach | Rows | Seconds | Rows/sec | Speedup vs naive |
|---|---:|---:|---:|---:|
| Naive `udf` (1 row per `encode`) | 20000 | 47.7 | 419 | 1.00x |
| Batched `pandas_udf` (batch_size=64) | 20000 | 21.7 | 920 | 2.19x |
| ANTI-PATTERN: load model every row | 100 | 42.4 | 2.36 | n/a (tiny N) |

The anti-pattern row ran on only 100 rows, because it is that slow; compare rows/sec, not seconds.
Batched vs the anti-pattern: 390.5x in rows/sec.
Naive and batched vectors agree (min cosine similarity 1.000000 over 200 rows).
