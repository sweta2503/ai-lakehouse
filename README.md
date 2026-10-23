# AI Lakehouse: PySpark + Apache Iceberg + Vectors

Make a lakehouse searchable **by meaning**, not just by filters, on a laptop.

One pipeline:

1. Land orders and customer reviews in **Apache Iceberg** tables on **MinIO** (Bronze).
2. Use **PySpark** to embed the review text with a sentence-transformer (Silver), comparing a slow row-by-row UDF with a batched `pandas_udf`.
3. Build **our own IVF-style vector index inside the lakehouse**: Spark MLlib `KMeans` clusters the embeddings, `cluster_id` is stored in the Iceberg table, and the table is *sorted* by it so Iceberg's file statistics skip files (Gold).
4. Run **hybrid queries**: SQL filters (date, status) + vector similarity restricted to the nearest clusters, benchmarked against brute force.
5. Inspect the Iceberg metadata (snapshots, manifest list, manifests, data files) to see how the table format works.
6. *Optional:* load the same vectors into **Qdrant** and compare.

> **Be clear about what is and isn't Iceberg.** Iceberg has **no native vector type and no ANN index.**
> Embeddings are stored as plain `array<float>`. The IVF index here is an *application-level technique*
> implemented in this repo (k-means, a `cluster_id` column, probing the nearest clusters). Iceberg only supplies
> the generic machinery it already has: partitioning, sort order, and per-file min/max column statistics.

Architecture diagram and layer descriptions: [docs/architecture.md](docs/architecture.md).

```
Olist CSV / synthetic ──► bronze.reviews ──► silver.review_embeddings ──► gold.review_embeddings (cluster_id, sorted)
                              (Iceberg v2)        (array<float>, 384-d)         ▲
                                                           └── KMeans ──► gold.centroids
query text ─► embed ─► nearest nprobe centroids ─► WHERE cluster_id IN (…) AND date/status ─► dot product ─► top-k
```

## Prerequisites

| Need | Notes |
|---|---|
| Docker + Compose v2 | MinIO, the Iceberg REST catalog, (bonus) Qdrant |
| Java 17 | `brew install openjdk@17`. It does not need to be on your `PATH`: `src/common.py` finds it |
| Python 3.11 | `python3.11` must exist; `make setup` builds `.venv` |
| ~8 GB free RAM | Spark driver defaults to 4 GB (`SPARK_DRIVER_MEMORY`); each Spark Python worker also holds its own copy of the embedding model |
| No GPU | Everything runs on CPU |
| Internet (first run) | Downloads Iceberg jars from Maven Central and the embedding model from Hugging Face |

## Quickstart

```bash
make setup           # venv + pinned deps + .env
make up              # MinIO, bucket, REST catalog, then runs the connection check
make all             # data -> bronze -> drift -> embed-bench -> silver -> index -> gold -> query -> bench -> inspect
make test            # smoke tests against the lakehouse you just built
```

`make all` uses a **synthetic** dataset unless the real Olist CSVs are in `data/` (see below), and logs which one it used.
For a fast pass use `SAMPLE_ROWS=5000 make all` (or `make smoke`, which also runs the tests).

Individual steps: `make bronze`, `drift`, `embed-bench`, `silver`, `index`, `gold`, `query`, `bench`, `inspect`, `bonus`, `down`, `clean`.
Every script in `src/` also runs on its own and prints what it is doing and how long it took.

### Using the real Olist data

Kaggle needs a login, so nothing is downloaded automatically without credentials. `make data` runs `scripts/download_data.sh`:
with the `kaggle` CLI and credentials it downloads the dataset; otherwise it prints manual instructions and generates the synthetic data.
To use the real data: download [Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and put
`olist_orders_dataset.csv` and `olist_order_reviews_dataset.csv` in `data/`. Reviews are in Portuguese; the multilingual model also lets you query in English.

The synthetic generator (`scripts/generate_synthetic_data.py`) produces the same columns with English themed reviews
(late delivery, damaged packaging, great quality, wrong item, refund requests, …), ~50k rows. **It is fake data**, and
easier to cluster than real reviews, so recall measured on it is probably optimistic.

### Configuration

Everything lives in `.env` (created from `.env.example`): model name, `IVF_K`, `SAMPLE_ROWS`, Spark memory/cores, target file size, endpoints.
Pinned versions are listed in [versions.md](versions.md); the Iceberg version is set once (`ICEBERG_VERSION`) and used by both the REST image tag and the Spark jars.

## What each milestone does

**M1 Infrastructure** (`docker-compose.yml`, `src/common.py`, `src/01_check_connection.py`).
MinIO (S3 API 9000, console 9001), a one-shot `mc` container that creates the `warehouse` bucket, and the `apache/iceberg-rest-fixture` REST catalog on 8181 backed by `s3://warehouse/` via `S3FileIO` with path-style access. Spark connects with catalog name `lake`, type `rest`, region set, and the Iceberg SQL extensions. The check script creates `bronze`/`silver`/`gold`, writes and reads a tiny table, and asserts the Parquet file really landed in S3.

**M2 Bronze + schema drift** (`02_bronze_ingest.py`, `03_schema_drift.py`).
Orders joined to reviews → `lake.bronze.reviews` (Iceberg v2, `months(order_purchase_timestamp)`). Title and message are combined into `review_text`; empty text becomes `NULL` with `has_review_text = false` (kept, but never embedded). The latest ~3% of orders are held back as a "next batch" and appended in `03` after `ALTER TABLE … ADD COLUMN review_channel`. The script shows old rows reading `NULL` for the new column, the `.snapshots` table, and `VERSION AS OF` / `TIMESTAMP AS OF` (old snapshots read with the *old* schema).

**M3 Embeddings** (`04_embed_naive_vs_batched.py`, `05_silver_embeddings.py`).
`04` times, on the same N rows (default 20,000), a naive Python `udf` (one `encode` call per row, model cached per worker so the comparison is fair) against an iterator-style `pandas_udf` (model loaded once per task, `batch_size=64`, `normalize_embeddings=True`), plus a clearly labelled run of the load-the-model-on-every-row anti-pattern on a tiny N. It checks the two vectors agree. `05` embeds all non-empty reviews with the batched UDF into `lake.silver.review_embeddings` (L2-normalised, so cosine similarity = dot product), one partition per core.

**M4 IVF index** (`06_build_ivf_index.py`, `07_gold_write.py`).
`06` converts `array<float>` to an ML vector (`array_to_vector`), trains `KMeans` (default k=50, fixed seed, cosine distance) and stores unit-norm centroids in `lake.gold.centroids`. `07` assigns each row to its nearest centroid (the same dot-product rule the query side uses) and writes `lake.gold.review_embeddings` partitioned by `months(purchase_ts)` with `WRITE ORDERED BY cluster_id`.
*Why sorting works:* each data file records min/max per column in its manifest entry. Sorted by `cluster_id`, each file covers a narrow cluster range, so `cluster_id IN (…)` lets the planner discard files without opening them. Unsorted, every file spans all clusters and nothing can be skipped.
`python src/07_gold_write.py --fragment 3 --compact` demonstrates the maintenance side: three separate sorted appends leave overlapping cluster ranges (worse pruning), then `rewrite_data_files(strategy => 'sort')` repairs it, with before/after file counts.

**M5 Hybrid query + benchmark** (`08_hybrid_query.py`, `09_benchmark.py`, `src/search.py`).
```bash
python src/08_hybrid_query.py --query "package arrived damaged" --since 2018-01-01 --status delivered --mode ivf --nprobe 3 --top-k 10
```
The query is embedded locally, the nearest `nprobe` centroids are found in Python, and Spark SQL filters `cluster_id IN (…)` plus date/status and scores with `aggregate(zip_with(embedding, query, …))`. The script prints the pushed-down predicates from `EXPLAIN` and Iceberg's own scan metrics (data files scanned/skipped, rows read). Results are ordered by `(score DESC, review_id, order_id)` so ties between identical vectors are deterministic. `09` runs ~50 queries against brute force and IVF at `nprobe` ∈ {1, 3, 5, 10}.

**M6 Metadata inspection** (`10_inspect_iceberg_metadata.py`).
Prints `snapshots`, `history`, `manifests`, `files`, `partitions` for the gold table, then lists the physical objects in MinIO with boto3: the **table metadata JSON** files, the **manifest list** (`snap-*.avro`), the **manifest files** (`*-m0.avro`) and the **Parquet data files**, and decodes the newest metadata JSON (format version, partition spec, sort order, properties).

## Optional: English translation of the reviews

Olist reviews are Portuguese. `make translate` (not part of `make all`) translates the distinct review texts with a free local model
(`Helsinki-NLP/opus-mt-ROMANCE-en`) into `lake.silver.review_translations`, a lookup table used **only for display**: the UI and
`make query` then show an English line under each Portuguese result. Embeddings, the index and all benchmarks use the original
text and are unaffected, and everything works without it. It is slow on a laptop (each Spark worker holds its own model copy;
on a memory-starved machine I measured about 15 texts/s on 4 workers, so tens of minutes for ~37k texts) and machine translation
of short slangy reviews is imperfect. Tune with `TRANSLATE_CORES`.

## Search UI (for demos)

```bash
make ui     # needs `make up` and a built gold table (make all)
```

Open http://localhost:8765. Type a query, choose IVF or brute force, move the `nprobe` slider, add date/status filters.
The page shows latency, **data files scanned of total** (Iceberg's own scan metrics), rows read, and a map of the
k clusters showing which were probed and which contain a result. **Compare IVF vs brute** runs both and shows this
query's recall (by id and tie-aware) next to both result lists. The server keeps one Spark session alive, so the
Spark UI (http://localhost:4040, SQL tab shows the `BatchScan` metrics) and MinIO console (http://localhost:9001) are
available alongside. Change the port with `UI_PORT`.

## Results

Everything below is copied from `benchmarks/` by `scripts/render_results.py` (run at the end of `make all`); nothing is typed by hand.
Re-run `make embed-bench` / `make bench` to regenerate on your machine.
The committed results were measured on the **real Olist dataset** (log line `DATA SOURCE: OLIST`), not the synthetic fallback.

### Embeddings: naive UDF vs batched `pandas_udf`

<!-- BENCH:headline:START -->
On this machine, batched `pandas_udf` was **2.11x** faster than the naive row-by-row UDF (929 vs 441 rows/s over 20000 rows).
<!-- BENCH:headline:END -->

<!-- BENCH:embedding:START -->
Model `paraphrase-multilingual-MiniLM-L12-v2` (CPU), 10 Spark cores, 1 torch thread(s) per worker.
Measured on this machine; the model is already loaded in every worker (warm-up excluded).

| Approach | Rows | Seconds | Rows/sec | Speedup vs naive |
|---|---:|---:|---:|---:|
| Naive `udf` (1 row per `encode`) | 20000 | 45.4 | 441 | 1.00x |
| Batched `pandas_udf` (batch_size=64) | 20000 | 21.5 | 929 | 2.11x |
| ANTI-PATTERN: load model every row | 100 | 50.0 | 2.00 | n/a (tiny N) |

The anti-pattern row ran on only 100 rows, because it is that slow; compare rows/sec, not seconds.
Batched vs the anti-pattern: 464.0x in rows/sec.
Naive and batched vectors agree (min cosine similarity 1.000000 over 200 rows).
<!-- BENCH:embedding:END -->

### IVF index vs brute force

<!-- BENCH:ivf:START -->
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
<!-- BENCH:ivf:END -->

## Limitations (read these before quoting any number)

- **Brute force vs ANN.** Brute force is exact. IVF is approximate: a true neighbour sitting in a cluster you did not probe is missed. More `nprobe` → higher recall, more files scanned. There is no `nprobe` that is "right"; it is a recall/cost dial.
- **Recall depends on the data.** IVF works when the data clusters. Short, templated, synthetic reviews cluster very cleanly and contain many exact-duplicate texts. Real reviews are messier, so expect lower recall at the same `nprobe`. Because of duplicates we report recall two ways: by review id, and "tie-aware" (a result counts if it scores as high as the exact 10th-best).
- **Small-data caveat.** The gold table is tens of thousands of vectors. Each query launches a Spark job with fixed planning/scheduling overhead that can hide, or even outweigh, the work saved by scanning fewer files, so the *files scanned / rows read* columns say more than the latency columns. This repo does not measure behaviour at millions of vectors.
- **Demo-scale tuning.** To make file-level pruning visible on small data, the gold write sets a small target file size, sizes the number of write tasks to produce ~256 KB files, and sets small read-split sizes on the table (applied identically to brute force and IVF). At real scale you would leave Iceberg's defaults (hundreds of MB per file) and get the same effect with more rows.
- **Not a vector database.** No ANN graph, no incremental index maintenance: new data gets a `cluster_id` against existing centroids; if the data drifts you must retrain the centroids and rewrite. Appends also fragment cluster ranges until you compact (see M4).
- **Filters and pruning.** Only `cluster_id` and `purchase_ts` (month partitions) prune well. `order_status` has very few distinct values spread across all files and will not skip files.
- **Quality of the embedding model** (`paraphrase-multilingual-MiniLM-L12-v2`, 384-d) limits search quality regardless of the index.
- The REST catalog image is the Iceberg **test fixture**; fine for a demo, not a production catalog.

## Troubleshooting

Entries marked ✅ were hit and fixed while building this repo; ⚠️ ones are things I could not reproduce here and are documented as guidance only.

**✅ `pull access denied for minio/minio` (or 401 from quay.io).**
The official `minio/minio` images were removed from Docker Hub/Quay. This repo uses the community fork `pgsty/minio` for the server and the frozen `bitnamilegacy/minio-client` for bucket creation (both pinned in `docker-compose.yml`). Fine for local demos; swap in your own S3-compatible store for anything else.

**✅ MinIO console (http://localhost:9001) does not load, though the S3 API on 9000 works.**
The first image I pinned, `bitnamilegacy/minio:2025.7.23`, is started with `--console-address :9001` but nothing listens on 9001 (I checked from inside the container), so the web UI simply isn't in that build. `pgsty/minio` serves it. Login: `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD` from `.env` (default `minioadmin` / `minioadmin`).

**✅ REST catalog exits immediately: `SQLITE_CANTOPEN: Unable to open the database file`.**
The fixture keeps its catalog in SQLite; with a persistent named volume mounted at `/catalog-data` the image's default user can't create the file. The compose file runs that one container as root (`user: "0:0"`). Without a persistent volume the catalog (the list of tables) is lost whenever the container is recreated, even though the data stays in MinIO.

**✅ Writes fail with `S3Exception: The bucket you are attempting to access must be addressed using the specified endpoint … Status Code: 301`.**
That is the AWS SDK talking to **real AWS S3** instead of MinIO because no S3 endpoint reached the client (I reproduced it by removing `s3.endpoint` from the Spark catalog config). The REST fixture does *not* send an endpoint to clients (its `loadTable` response has no `config`), and the server itself uses the in-network address `http://minio:9000` which the host can't resolve. So the Spark client sets `s3.endpoint=http://localhost:9000` and `s3.path-style-access=true` itself (see `get_spark()` in `src/common.py`). If you use a REST catalog that *does* vend `s3.endpoint=http://minio:9000`, the symptom is `UnknownHostException: minio`; I did not hit that with this fixture. The two fixes are a client-side override of that property, or a hosts entry: `echo "127.0.0.1 minio" | sudo tee -a /etc/hosts`.

**⚠️ "Unable to load region" / credentials errors.**
The AWS SDK v2 needs a region and credentials even for MinIO, so `client.region`, `s3.access-key-id`, `s3.secret-access-key` are set explicitly in the catalog config (`AWS_REGION=us-east-1`, `MINIO_ROOT_USER/PASSWORD`). I tested with no `~/.aws` directory and removing the region did not fail on this machine, so I can't show you that error; setting it explicitly is for machines where the SDK's default provider chain fails.

**✅ `java: Unable to locate a Java Runtime`.**
Homebrew's `openjdk@17` is keg-only, so `java` isn't on `PATH`. `ensure_java17()` in `src/common.py` locates it (via `/usr/libexec/java_home`, then Homebrew/Linux paths) and sets `JAVA_HOME` for Spark. Spark 3.5 supports Java 8/11/17; this repo standardises on 17.

**✅ `PySparkNotImplementedError: [UNSUPPORTED_SIGNATURE] … Iterator[pd.Series]`.**
Caused by `from __future__ import annotations` in the module that defines the iterator `pandas_udf`: the type hints become strings that PySpark can't resolve. Don't use that import in files that define `pandas_udf`s.

**✅ `TIMESTAMP AS OF` returns the wrong snapshot.**
A Python `datetime` read from `.snapshots` is converted to your local timezone, while the Spark session runs in UTC (`spark.sql.session.timeZone`). Format the timestamp inside Spark (`date_format(committed_at, …)`) as `03_schema_drift.py` does.

**✅ Time-travel demo shows extra snapshots after re-running `make bronze`.**
`createOrReplace` keeps the old table's snapshot history. `02_bronze_ingest.py` now does `DROP TABLE … PURGE` first.

**✅ Gold table has only a few large files, so pruning barely skips anything.**
Setting `write.target-file-size-bytes` alone gave ~780 KB average files against a 256 KB target in the first full run (`cluster_id IN (3 clusters)` could skip only ~42% of files, vs ~70% after the fix). The number of files is governed by the number of write tasks of the sorted write, so `07_gold_write.py` sets `spark.sql.shuffle.partitions` from the estimated size and disables AQE for that write.

**✅ Brute force reports recall@10 below 1.0 against itself (real Olist data only).**
`review_id` is not unique in the real dataset: 1,603 of the 99,224 review rows share a `review_id` with another row (the pair `review_id` + `order_id` is unique). Comparing top-10 results as sets of `review_id` therefore collapsed duplicates and gave brute force 0.988 against its own ground truth. The benchmark, the Qdrant comparison and the tests now key on `(review_id, order_id)`, and ties are broken by both columns. The synthetic data has unique ids, which is why this only showed up on real data.

**⚠️ Jar / version mismatches.**
I did not hit one with the pinned set (Spark 3.5.6 / Scala 2.12 / Iceberg 1.9.2 jars / REST fixture 1.9.2). The rules: the runtime jar must match the Spark minor and Scala version (`iceberg-spark-runtime-3.5_2.12`), and keep the client jars and the REST image on the same Iceberg version. `NoSuchMethodError`/`ClassNotFoundException` on startup usually means one of those drifted. Wipe `.ivy2/` to force a re-download.

**⚠️ Memory.**
I hit no out-of-memory errors on a 16 GB machine with 10 cores. Each Spark Python worker loads its own copy of the embedding model, so on smaller machines lower `SPARK_CORES` (fewer workers) and/or `SPARK_DRIVER_MEMORY` in `.env`. `EMBED_TORCH_THREADS=1` avoids every worker also spinning up many torch threads.

**Benign warnings.** `Set SPARK_LOCAL_IP if you need to bind to another address` and `Unable to load native-hadoop library` are expected on a laptop.

## Bonus: Qdrant

```bash
make bonus        # installs qdrant-client, starts Qdrant (docker compose --profile bonus), runs the comparison
```

Qdrant is not started by `docker compose up`; it sits behind the `bonus` profile. `src/bonus_qdrant_compare.py` loads the gold vectors plus `purchase_ts` / `order_status` payload into a cosine collection with the same 384 dims, runs the same queries with equivalent filters, and writes `benchmarks/qdrant_comparison.md`.
It is a different kind of system (a purpose-built engine over HTTP vs a Spark job over Iceberg files), so compare *recall* and *what you operate*, not raw milliseconds.
The core pipeline and the tests do not need it.

<!-- BENCH:qdrant:START -->
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
<!-- BENCH:qdrant:END -->

## Data and attribution

Benchmarks were measured on the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
(licensed CC BY-NC-SA 4.0, non-commercial). The data is **not** included in this repository (`data/` is gitignored); download it yourself
from Kaggle. The review text is Portuguese; the multilingual embedding model lets you query in English.

## Repository layout

```
docker-compose.yml  .env.example  Makefile  requirements.txt  versions.md
scripts/   download_data.sh  generate_synthetic_data.py  wait_for_services.sh  render_results.py
src/       common.py  data_source.py  embeddings.py  search.py  ui_server.py  ui/index.html  01_…10_*.py  bonus_qdrant_compare.py
tests/     test_smoke.py
benchmarks/  measured results (json + markdown)
docs/      architecture.md
```
