# Qdrant vs Iceberg IVF

42687 vectors, 50 queries, same SQL-style filters (date range + status) applied in both systems, top-10.
Ground truth: exact brute force over the Iceberg gold table. Iceberg rows are copied from `ivf_benchmark.json`
(same run as `09_benchmark.py`); Qdrant ran locally in Docker.

| System | mean ms | p95 ms | recall@10 (by id) | recall@10 (tie-aware) |
|---|---:|---:|---:|---:|
| Qdrant HNSW (default) | 3.1 | 5.62 | 0.992 | 1.000 |
| Qdrant exact | 2.59 | 3.32 | 0.992 | 1.000 |
| Iceberg brute force (Spark) | 456.3 | 707.6 | 1.000 | 1.000 |
| Iceberg IVF nprobe=1 (Spark) | 254.4 | 336.9 | 0.702 | 0.708 |
| Iceberg IVF nprobe=3 (Spark) | 285.8 | 354.9 | 0.876 | 0.876 |
| Iceberg IVF nprobe=5 (Spark) | 337.6 | 480.7 | 0.952 | 0.952 |
| Iceberg IVF nprobe=10 (Spark) | 426.4 | 638.4 | 0.990 | 0.992 |

Notes
- Qdrant latency is a client-side HTTP round trip to a purpose-built engine; the Iceberg numbers include a full
  Spark job per query. Latencies are not comparable as "which index is better", only as "what you pay for
  keeping vectors inside the lakehouse and querying them with Spark".
- "By id" recall counts a result only if it is the same review as the ground truth. The data has many identical
  texts (identical vectors); Qdrant breaks such ties arbitrarily, so the tie-aware column counts a result as correct
  when its score reaches the ground truth's 10th-best score.
