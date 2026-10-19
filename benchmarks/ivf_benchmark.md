# IVF benchmark (Iceberg + Spark)

50 hybrid queries (SQL filters on date/status + vector similarity), top-10; each query/config run
3x back to back and the median latency kept. Gold table: 42687 rows in
221 data files, k=50 clusters. Recall@10 is measured against exact brute force with the same filters.
"Data files scanned" and "rows read" come from Iceberg's scan metrics (they already include date-partition pruning,
which brute force benefits from too).

| Config | mean ms | p95 ms | speedup | data files scanned | % of brute files | rows read | recall@10 | recall@10 tie-aware |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| brute force | 456.3 | 707.6 | 1.0x | 116.4 | 100.0% | 22643 | 1.000 | 1.000 |
| IVF nprobe=1 | 254.4 | 336.9 | 1.79x | 11.1 | 9.5% | 2160 | 0.702 | 0.708 |
| IVF nprobe=3 | 285.8 | 354.9 | 1.6x | 28.2 | 24.2% | 5718 | 0.876 | 0.876 |
| IVF nprobe=5 | 337.6 | 480.7 | 1.35x | 42.4 | 36.4% | 8773 | 0.952 | 0.952 |
| IVF nprobe=10 | 426.4 | 638.4 | 1.07x | 66.7 | 57.3% | 13755 | 0.990 | 0.992 |

**Scale caveat:** this is a laptop-sized table. Every query pays a fixed Spark job + Iceberg planning cost
regardless of how many vectors it scores, so on a table this small that overhead can hide (or even outweigh) the
savings from scanning fewer files; the files-scanned and rows-read columns show what IVF skips independent of that
overhead. Latency wins from IVF should grow with table size, but this run does not measure that. The reviews
(especially synthetic ones) are also much more clusterable than messy real-world text, so recall here may be
optimistic. "recall@10 tie-aware" counts a result as correct if it scores as high as the exact 10th-best result,
because many reviews share identical text and therefore identical vectors.
