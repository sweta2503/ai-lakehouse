# IVF benchmark (Iceberg + Spark)

50 hybrid queries (SQL filters on date/status + vector similarity), top-10; each query/config run
3x back to back and the median latency kept. Gold table: 42687 rows in
224 data files, k=50 clusters. Recall@10 is measured against exact brute force with the same filters.
"Data files scanned" and "rows read" come from Iceberg's scan metrics (they already include date-partition pruning,
which brute force benefits from too).

| Config | mean ms | p95 ms | speedup | data files scanned | % of brute files | rows read | recall@10 | recall@10 tie-aware |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| brute force | 487.4 | 750.1 | 1.0x | 117.6 | 100.0% | 22643 | 1.000 | 1.000 |
| IVF nprobe=1 | 272.9 | 342.4 | 1.79x | 11.2 | 9.5% | 2118 | 0.638 | 0.644 |
| IVF nprobe=3 | 311.6 | 414.9 | 1.56x | 27.3 | 23.2% | 5442 | 0.860 | 0.864 |
| IVF nprobe=5 | 354.6 | 501.3 | 1.37x | 40.8 | 34.7% | 8329 | 0.904 | 0.908 |
| IVF nprobe=10 | 460.4 | 669.2 | 1.06x | 67.3 | 57.2% | 13647 | 0.954 | 0.958 |

**Scale caveat:** this is a laptop-sized table. Every query pays a fixed Spark job + Iceberg planning cost
regardless of how many vectors it scores, so on a table this small that overhead can hide (or even outweigh) the
savings from scanning fewer files; the files-scanned and rows-read columns show what IVF skips independent of that
overhead. Latency wins from IVF should grow with table size, but this run does not measure that. The reviews
(especially synthetic ones) are also much more clusterable than messy real-world text, so recall here may be
optimistic. "recall@10 tie-aware" counts a result as correct if it scores as high as the exact 10th-best result,
because many reviews share identical text and therefore identical vectors.
