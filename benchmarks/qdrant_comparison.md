# Qdrant vs Iceberg IVF

35977 vectors, 50 queries, same SQL-style filters (date range + status) applied in both systems, top-10.
Ground truth: exact brute force over the Iceberg gold table. Iceberg rows are copied from `ivf_benchmark.json`
(same run as `09_benchmark.py`); Qdrant ran locally in Docker.

| System | mean ms | p95 ms | recall@10 (by id) | recall@10 (tie-aware) |
|---|---:|---:|---:|---:|
| Qdrant HNSW (default) | 2.63 | 4.69 | 0.782 | 1.000 |
| Qdrant exact | 2.34 | 3.39 | 0.782 | 1.000 |
| Iceberg brute force (Spark) | 349.7 | 514.3 | 1.000 | 1.000 |
| Iceberg IVF nprobe=1 (Spark) | 237.1 | 286.5 | 0.786 | 0.808 |
| Iceberg IVF nprobe=3 (Spark) | 264.6 | 341.1 | 0.936 | 0.966 |
| Iceberg IVF nprobe=5 (Spark) | 304.1 | 409.6 | 0.948 | 0.970 |
| Iceberg IVF nprobe=10 (Spark) | 362.9 | 519.7 | 0.986 | 0.996 |

Notes
- Qdrant latency is a client-side HTTP round trip to a purpose-built engine; the Iceberg numbers include a full
  Spark job per query. Latencies are not comparable as "which index is better", only as "what you pay for
  keeping vectors inside the lakehouse and querying them with Spark".
- "By id" recall counts a result only if it is the same review as the ground truth. The data has many identical
  texts (identical vectors); Qdrant breaks such ties arbitrarily, so the tie-aware column counts a result as correct
  when its score reaches the ground truth's 10th-best score.
