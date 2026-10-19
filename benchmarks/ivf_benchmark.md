# IVF benchmark (Iceberg + Spark)

50 hybrid queries (SQL filters on date/status + vector similarity), top-10. Gold table: 3617 rows in
23 data files, k=50 clusters. Recall@10 is measured against exact brute force with the same filters.
"Data files scanned" and "rows read" come from Iceberg's scan metrics (they already include date-partition pruning,
which brute force benefits from too).

| Config | mean ms | p95 ms | speedup | data files scanned | % of brute files | rows read | recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| brute force | 139.9 | 163.4 | 1.0x | 11.6 | 100.0% | 1892 | 1.000 |
| IVF nprobe=1 | 173.2 | 245.3 | 0.81x | 9.8 | 84.5% | 1699 | 0.776 |
| IVF nprobe=3 | 161.6 | 218.7 | 0.87x | 10.7 | 92.2% | 1839 | 0.942 |
| IVF nprobe=5 | 169.1 | 222.9 | 0.83x | 11.1 | 95.7% | 1867 | 0.970 |
| IVF nprobe=10 | 172.3 | 252.8 | 0.81x | 11.4 | 98.3% | 1886 | 0.980 |

**Scale caveat:** this is a laptop-sized table. Each query pays a roughly fixed Spark job + Iceberg planning
overhead (tens to hundreds of ms) that dwarfs the actual scoring work on a few thousand vectors, so latency
differences are small and mostly noise-bounded; the file/row counts are the more reliable signal of what IVF
saves, and that saving only turns into large latency wins when the table is much bigger than this. The reviews
(especially the synthetic ones) are also unusually clusterable, so recall here is likely higher than on messy
real-world data.
