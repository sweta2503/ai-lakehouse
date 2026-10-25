# Embedding benchmark: naive UDF vs batched pandas_udf

Model `all-MiniLM-L6-v2` (CPU), 10 Spark cores, 1 torch thread(s) per worker.
Measured on this machine; the model is already loaded in every worker (warm-up excluded).

| Approach | Rows | Seconds | Rows/sec | Speedup vs naive |
|---|---:|---:|---:|---:|
| Naive `udf` (1 row per `encode`) | 20000 | 45.4 | 440 | 1.00x |
| Batched `pandas_udf` (batch_size=64) | 20000 | 19.1 | 1047 | 2.38x |
| ANTI-PATTERN: load model every row | 100 | 27.9 | 3.58 | n/a (tiny N) |

The anti-pattern row ran on only 100 rows, because it is that slow; compare rows/sec, not seconds.
Batched vs the anti-pattern: 292.5x in rows/sec.
Naive and batched vectors agree (min cosine similarity 1.000000 over 200 rows).
