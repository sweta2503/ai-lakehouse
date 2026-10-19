# Video notes: "Build an AI Lakehouse from Scratch (PySpark + Iceberg + Vectors)"

Run order = video order. Every command below was run for real while building the repo. "Expected output" describes the
**shape** of the output; for numbers, read the files in `benchmarks/` on the day you record (don't quote numbers from memory).

**Honesty line to say on camera early:** *Iceberg has no vector type and no ANN index. We store embeddings as `array<float>` and build the IVF index ourselves; Iceberg just gives us partitioning, sort order and file statistics to skip data.*

Suggested timing: a 12 to 18 minute video is realistic if M3's benchmark and M5's benchmark are shown from pre-run results, not waited on live (the benchmarks take minutes).

---

## Pre-record checklist
- `make setup` done, Docker running, `make up` has been run once (jars + model are downloaded: first Spark start downloads the jars; first embedding run downloads the model).
- Have a clean full run's output saved (`make all`) so you can cut to results instead of waiting.
- Decide: synthetic vs real Olist. If synthetic, say so on screen ("fake English reviews with the same columns").
- Terminal font big; split view: terminal + MinIO console (`http://localhost:9001`, login from `.env`).

## M1: Infrastructure
**Run:** `docker compose up -d` → `docker compose ps` → `make up` (or `python src/01_check_connection.py`)
**Expected:** `lakehouse-minio` healthy, `lakehouse-mc` Exited (0) ("bucket ready"), `lakehouse-rest` healthy. Check script prints namespaces `['bronze','gold','silver']`, two rows back, a data file path starting `s3://warehouse/…`, and `PASS`.
**Explain:** MinIO = the S3 stand-in; REST catalog = the single source of truth for "which metadata file is current"; Spark talks to the catalog for metadata and straight to S3 for data.
**Pre-write:** `docker-compose.yml`, `.env`, `wait_for_services.sh`.
**Type live:** the Spark session builder (the 8 or so `spark.sql.catalog.lake.*` lines) in `common.py::get_spark`.
**Flash the architecture diagram** ([architecture.md](architecture.md)) right here: MinIO → REST → Spark.
**Show in MinIO console:** the `warehouse` bucket filling up after the check script (then it's dropped).

## M2: Bronze + schema drift
**Run:** `make data` (shows the manual-download message + synthetic fallback), `make bronze`, `make drift`
**Expected (bronze):** log line `DATA SOURCE: SYNTHETIC` (or OLIST); row counts; a status × has_review_text table; one partition row per month; `format-version=2`.
**Expected (drift):** after ADD COLUMN + append, a `review_channel` breakdown with a big `NULL` group (old rows) and four channel values; `.snapshots` with two `append` rows; `VERSION AS OF <first id>` returns the old row count with no `review_channel` column; `TIMESTAMP AS OF` agrees.
**Explain:** hidden partitioning (`months(order_purchase_timestamp)`: filter on the raw timestamp); why empty review text is flagged, not dropped; ADD COLUMN is metadata-only because Iceberg tracks columns by ID.
**Pre-write:** `data_source.py` (CSV reading, quoting/multiline handling).
**Type live:** the `ALTER TABLE … ADD COLUMN` + `writeTo(...).append()`, then the `VERSION AS OF` query.
**Checkpoint:** flash the diagram's Bronze box.

## M3: Embeddings
**Run:** `make embed-bench` (pre-run it; ~minutes), then `make silver`
**Expected (embed-bench):** a markdown table with three rows: naive UDF, batched pandas_udf, anti-pattern (tiny N, clearly labeled). A line "vectors agree: min cosine … = 1.000000". Read the speedup from `benchmarks/embedding_benchmark.md`.
**Expected (silver):** `silver rows=… dim(min,max)=(384,384) L2 norm(min,max)=(1.00000,1.00000)`.
**Explain:** why per-row UDFs hurt (Python↔JVM serialization per row, tiny matrix ops); why `pandas_udf` + Arrow + `batch_size=64` helps; "load the model once per worker/task, not per row"; normalized vectors → cosine = dot product.
**Say the true numbers.** The measured batched-vs-naive speedup on a CPU laptop is a small multiple, **not** 10x. Present it honestly, and note the anti-pattern comparison (model load per row) is the dramatic one.
**Pre-write:** the naive UDF and the benchmark harness.
**Type live:** the iterator `pandas_udf` (`embeddings.py::batched_udf`): the heart of the video.
**Gotcha to keep:** `from __future__ import annotations` breaks the iterator `pandas_udf` type hints.

## M4: IVF index inside the lakehouse
**Run:** `make index`, `make gold` (optionally `python src/07_gold_write.py --fragment 3 --compact`, then `make gold` again to restore)
**Expected (index):** KMeans timing, cluster sizes (min/median/max), 50 centroids written.
**Expected (gold):** `table sort order: cluster_id ASC NULLS FIRST`, number of data files, a table showing per-file `cid_min`/`cid_max` (narrow ranges!), a `PRUNING PREVIEW` line, and a few sample reviews from the biggest clusters, which should visibly share a theme (delivery speed / damaged goods / quality).
**Expected (--compact):** file counts before/after and the "files a cluster_id IN (…) query must scan" before/after.
**Explain (the key idea):** IVF = cluster the vectors, only search the nearest clusters. Iceberg keeps min/max per column per file; sorting by `cluster_id` makes those ranges narrow, so `cluster_id IN (…)` lets the planner skip files. **Unsorted → every file spans every cluster → nothing skipped.** Not ANN, not an Iceberg feature.
**Pre-write:** `07` table DDL with its property comments.
**Type live:** `KMeans(...).fit(...)` and `ALTER TABLE … WRITE ORDERED BY cluster_id`.
**Flash the diagram** and highlight the `centroids` → `gold` arrows.

## M5: Hybrid query + benchmark
**Run:**
```bash
python src/08_hybrid_query.py --query "package arrived damaged and the seller never answered" \
    --since 2018-01-01 --status delivered --mode ivf --nprobe 3
python src/08_hybrid_query.py --query "package arrived damaged and the seller never answered" \
    --since 2018-01-01 --status delivered --mode brute
make bench      # pre-run; show benchmarks/ivf_benchmark.md
```
**Expected (query):** the nearest-3 clusters with cosine scores; filters line incl. `cluster_id IN [...]`; 10 results all `delivered`, all 2018+; `Iceberg scan metrics: scanned X of N data files …`; the EXPLAIN scan node showing `cluster_id IN (…)` pushed down. Run `--mode brute` to compare the scanned-files count.
**Expected (bench):** table brute vs nprobe 1/3/5/10 with mean/p95 latency, files scanned, rows read, recall@10 (plain and tie-aware).
**Explain:** the whole query is four steps (embed, nearest centroids in Python, SQL filter + dot product, order/limit); the recall ↔ files-scanned tradeoff of `nprobe`; why the **latency** gain may be small or even negative on a tiny table (fixed Spark job overhead) while **files/rows scanned** drop. **Say the caveat out loud**: laptop scale, clusterable (synthetic) data, so don't extrapolate.
**Type live:** the hybrid query (`search.py::build_query`): `cluster_id IN` + filters + `aggregate(zip_with(...))` + `orderBy/limit`.
**Pre-write:** CLI arg parsing, the benchmark loop.
**Checkpoint:** show the pruning preview and scan metrics side by side with the brute-force run.

## M6: Inspect the Iceberg metadata
**Run:** `make inspect`
**Expected:** metadata tables (`snapshots`, `history`, `manifests`, `files` with `cluster_min/max`, `partitions`), then physical objects: N × `*.metadata.json`, a `snap-*.avro` manifest list, `*-m0.avro` manifest file(s), the Parquet data files, then the decoded newest metadata JSON (format-version 2, partition spec `month(purchase_ts)`, sort order `identity(cluster_id) asc`).
**Explain, using the exact terms:** catalog → **table metadata JSON** → snapshot → **manifest list** → **manifest files** → **data files**. Open the MinIO console and click through `warehouse/gold/review_embeddings/{metadata,data}`. The `.manifests` table is the contents of the manifest list; `.files` is the manifest entries.
**Flash the diagram** one more time to tie the metadata chain back to pruning (min/max live in the manifests).

## Bonus: Qdrant (optional)
**Run:** `make bonus` (starts Qdrant under the `bonus` profile).
**Expected:** `benchmarks/qdrant_comparison.md` with Qdrant HNSW/exact vs the Iceberg configs.
**Explain:** different category of system; compare recall and operational cost (one copy of data in the lakehouse + SQL/time travel vs a second system to run and keep in sync), not milliseconds. Qdrant's "by id" recall is lowered by ties among duplicate vectors; that's why the table has a tie-aware column.

---

## Results (auto-filled from `benchmarks/`)

<!-- BENCH:headline:START -->
_Run `make embed-bench`._
<!-- BENCH:headline:END -->

<!-- BENCH:ivf:START -->
_Not generated yet: run `make bench`._
<!-- BENCH:ivf:END -->

---

## Likely errors worth keeping in the video (all of these really happened)
1. **`Unable to locate a Java Runtime`**: Homebrew's `openjdk@17` isn't on PATH. (Fix: `JAVA_HOME`; `common.py` finds it.)
2. **`pull access denied for minio/minio`**: the official image is gone; we pin `bitnamilegacy/minio`.
3. **`SQLITE_CANTOPEN` from the REST catalog**: volume permissions; run the container as root.
4. **S3 `301 … must be addressed using the specified endpoint`**: Spark is talking to real AWS because the client wasn't told the MinIO endpoint (this fixture doesn't send one). Great teaching moment on the client/server split of endpoint config.
5. **`UNSUPPORTED_SIGNATURE … Iterator[pd.Series]`**: `from __future__ import annotations` + `pandas_udf`.
6. **`TIMESTAMP AS OF` shows the wrong data**: local timezone vs Spark's UTC session.
7. **"Pruning does nothing"**: files were far too big (one file spans every cluster). File *count* is driven by the number of write tasks; show before/after of the `PRUNING PREVIEW` / scan metrics.

## Boilerplate vs live-typing summary
| Pre-write (paste, don't type) | Type live |
|---|---|
| docker-compose, `.env`, Makefile, data loading, CLI parsing, benchmark harness, metadata printing | Spark catalog config; iterator `pandas_udf`; KMeans + `WRITE ORDERED BY`; the hybrid query (`cluster_id IN` + filters + `aggregate(zip_with)`); `VERSION AS OF` |
