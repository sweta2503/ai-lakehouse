# Qdrant vs Iceberg IVF

3617 vectors, 50 queries, same SQL-style filters (date range + status) applied in both systems, top-10.
Ground truth: exact brute force over the Iceberg gold table. Iceberg rows are copied from `ivf_benchmark.json`
(same run as `09_benchmark.py`); Qdrant ran locally in Docker.

| System | mean ms | p95 ms | recall@10 (by id) | recall@10 (tie-aware) |
|---|---:|---:|---:|---:|
| Qdrant HNSW (default) | 1.3 | 1.96 | 0.862 | 1.000 |
| Qdrant exact | 1.17 | 1.31 | 0.862 | 1.000 |
| Iceberg brute force (Spark) | 139.9 | 163.4 | 1.000 | n/a (ties broken by review_id) |
| Iceberg IVF nprobe=1 (Spark) | 173.2 | 245.3 | 0.776 | n/a (ties broken by review_id) |
| Iceberg IVF nprobe=3 (Spark) | 161.6 | 218.7 | 0.942 | n/a (ties broken by review_id) |
| Iceberg IVF nprobe=5 (Spark) | 169.1 | 222.9 | 0.970 | n/a (ties broken by review_id) |
| Iceberg IVF nprobe=10 (Spark) | 172.3 | 252.8 | 0.980 | n/a (ties broken by review_id) |

Notes
- Qdrant latency is a client-side HTTP round trip to a purpose-built engine; the Iceberg numbers include a full
  Spark job per query. Latencies are not comparable as "which index is better", only as "what you pay for
  keeping vectors inside the lakehouse and querying them with Spark".
- "By id" recall counts a result only if it is the same review as the ground truth. The data has many identical
  texts (identical vectors); Qdrant breaks such ties arbitrarily, so the tie-aware column counts a result as correct
  when its score reaches the ground truth's 10th-best score.
