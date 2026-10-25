# Qdrant vs Iceberg IVF

42687 vectors, 50 queries, same SQL-style filters (date range + status) applied in both systems, top-10.
Ground truth: exact brute force over the Iceberg gold table. Iceberg rows are copied from `ivf_benchmark.json`
(same run as `09_benchmark.py`); Qdrant ran locally in Docker.

| System | mean ms | p95 ms | recall@10 (by id) | recall@10 (tie-aware) |
|---|---:|---:|---:|---:|
| Qdrant HNSW (default) | 3.53 | 5.86 | 0.988 | 1.000 |
| Qdrant exact | 2.51 | 2.88 | 0.988 | 1.000 |
| Iceberg brute force (Spark) | 487.4 | 750.1 | 1.000 | 1.000 |
| Iceberg IVF nprobe=1 (Spark) | 272.9 | 342.4 | 0.638 | 0.644 |
| Iceberg IVF nprobe=3 (Spark) | 311.6 | 414.9 | 0.860 | 0.864 |
| Iceberg IVF nprobe=5 (Spark) | 354.6 | 501.3 | 0.904 | 0.908 |
| Iceberg IVF nprobe=10 (Spark) | 460.4 | 669.2 | 0.954 | 0.958 |

Notes
- Qdrant latency is a client-side HTTP round trip to a purpose-built engine; the Iceberg numbers include a full
  Spark job per query. Latencies are not comparable as "which index is better", only as "what you pay for
  keeping vectors inside the lakehouse and querying them with Spark".
- "By id" recall counts a result only if it is the same review as the ground truth. The data has many identical
  texts (identical vectors); Qdrant breaks such ties arbitrarily, so the tie-aware column counts a result as correct
  when its score reaches the ground truth's 10th-best score.
