# IVF benchmark (Iceberg + Spark)

50 hybrid queries (SQL filters on date/status + vector similarity), top-10; each query/config run
3x back to back and the median latency kept. Gold table: 35977 rows in
182 data files, k=50 clusters. Recall@10 is measured against exact brute force with the same filters.
"Data files scanned" and "rows read" come from Iceberg's scan metrics (they already include date-partition pruning,
which brute force benefits from too).

| Config | mean ms | p95 ms | speedup | data files scanned | % of brute files | rows read | recall@10 | recall@10 tie-aware |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| brute force | 352.9 | 518.1 | 1.0x | 94.7 | 100.0% | 18921 | 1.000 | 1.000 |
| IVF nprobe=1 | 231.1 | 284.1 | 1.53x | 9.9 | 10.5% | 2098 | 0.786 | 0.808 |
| IVF nprobe=3 | 264.4 | 319.6 | 1.33x | 26.4 | 27.9% | 5556 | 0.936 | 0.966 |
| IVF nprobe=5 | 302.7 | 414.3 | 1.17x | 39.7 | 41.9% | 8250 | 0.948 | 0.970 |
| IVF nprobe=10 | 364.4 | 504.7 | 0.97x | 58.4 | 61.7% | 12036 | 0.986 | 0.996 |

**Scale caveat:** this is a laptop-sized table. Every query pays a fixed Spark job + Iceberg planning cost
regardless of how many vectors it scores, so on a table this small that overhead can hide (or even outweigh) the
savings from scanning fewer files; the files-scanned and rows-read columns show what IVF skips independent of that
overhead. Latency wins from IVF should grow with table size, but this run does not measure that. The reviews
(especially synthetic ones) are also much more clusterable than messy real-world text, so recall here may be
optimistic. "recall@10 tie-aware" counts a result as correct if it scores as high as the exact 10th-best result,
because many reviews share identical text and therefore identical vectors.
