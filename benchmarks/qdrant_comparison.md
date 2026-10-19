# Qdrant vs Iceberg IVF

35977 vectors, 50 queries, same SQL-style filters (date range + status) applied in both systems, top-10.
Ground truth: exact brute force over the Iceberg gold table. Iceberg rows are copied from `ivf_benchmark.json`
(same run as `09_benchmark.py`); Qdrant ran locally in Docker.

| System | mean ms | p95 ms | recall@10 (by id) | recall@10 (tie-aware) |
|---|---:|---:|---:|---:|
| Qdrant HNSW (default) | 2.56 | 3.53 | 0.784 | 1.000 |
| Qdrant exact | 2.29 | 2.77 | 0.784 | 1.000 |
| Iceberg brute force (Spark) | 335.3 | 482.6 | 1.000 | 1.000 |
| Iceberg IVF nprobe=1 (Spark) | 215.5 | 273.8 | 0.786 | 0.808 |
| Iceberg IVF nprobe=3 (Spark) | 252.3 | 328.5 | 0.936 | 0.966 |
| Iceberg IVF nprobe=5 (Spark) | 284.2 | 374.4 | 0.948 | 0.970 |
| Iceberg IVF nprobe=10 (Spark) | 349.9 | 483.4 | 0.986 | 0.996 |

Notes
- Qdrant latency is a client-side HTTP round trip to a purpose-built engine; the Iceberg numbers include a full
  Spark job per query. Latencies are not comparable as "which index is better", only as "what you pay for
  keeping vectors inside the lakehouse and querying them with Spark".
- "By id" recall counts a result only if it is the same review as the ground truth. The data has many identical
  texts (identical vectors); Qdrant breaks such ties arbitrarily, so the tie-aware column counts a result as correct
  when its score reaches the ground truth's 10th-best score.
