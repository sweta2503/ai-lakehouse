# Embedding benchmark: naive UDF vs batched pandas_udf

Model `paraphrase-multilingual-MiniLM-L12-v2` (CPU), 10 Spark cores, 1 torch thread(s) per worker.
Measured on this machine; the model is already loaded in every worker (warm-up excluded).

| Approach | Rows | Seconds | Rows/sec | Speedup vs naive |
|---|---:|---:|---:|---:|
| Naive `udf` (1 row per `encode`) | 20000 | 45.4 | 441 | 1.00x |
| Batched `pandas_udf` (batch_size=64) | 20000 | 21.5 | 929 | 2.11x |
| ANTI-PATTERN: load model every row | 100 | 50.0 | 2.00 | n/a (tiny N) |

The anti-pattern row ran on only 100 rows, because it is that slow; compare rows/sec, not seconds.
Batched vs the anti-pattern: 464.0x in rows/sec.
Naive and batched vectors agree (min cosine similarity 1.000000 over 200 rows).
