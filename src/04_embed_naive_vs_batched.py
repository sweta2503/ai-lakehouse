"""M3 (1/2): naive row-by-row UDF vs batched pandas_udf, on the SAME N rows. Real measured numbers only.

Writes benchmarks/embedding_benchmark.json and benchmarks/embedding_benchmark.md.
"""
import json
import sys
import time

import numpy as np
from pyspark.sql import functions as F

from common import (BENCH_DIR, EMBED_BATCH_SIZE, EMBEDDING_MODEL, T_BRONZE, env_int, get_logger, get_spark,
                    local_cores, section)
from embeddings import batched_udf, naive_udf, reload_each_row_udf

log = get_logger("embed-bench")


def run(df, col_expr):
    """Force full evaluation of the UDF without any sink cost (noop write). Returns seconds."""
    t0 = time.perf_counter()
    df.select(col_expr.alias("emb")).write.format("noop").mode("overwrite").save()
    return time.perf_counter() - t0


def main() -> int:
    section("M3: embeddings — naive UDF vs batched pandas_udf")
    spark = get_spark("04_embed_naive_vs_batched")
    cores = local_cores()
    n_req = env_int("EMBED_BENCH_ROWS", 20000)
    n_bad = env_int("EMBED_ANTIPATTERN_ROWS", 100)

    texts = (spark.table(T_BRONZE).where("has_review_text").select("review_text")
             .orderBy(F.rand(1)).limit(n_req).repartition(cores).cache())
    n = texts.count()  # materialise so both approaches read identical, already-loaded rows
    if n < n_req:
        log.warning("only %d rows with text available (asked for %d)", n, n_req)
    log.info("benchmark rows N=%d, partitions=%d, cores=%d", n, cores, cores)

    # Warm-up: get the model into every Python worker so we time steady-state throughput, not the
    # (one-off, per-worker) model load. Reported separately below.
    warm_df = texts.limit(cores * 4).repartition(cores)
    t_warm_naive = run(warm_df, naive_udf()(F.col("review_text")))
    t_warm_batched = run(warm_df, batched_udf()(F.col("review_text")))
    log.info("warm-up: 1st pass (includes model load in each worker) %.1fs, 2nd pass (model cached) %.1fs",
             t_warm_naive, t_warm_batched)

    col = F.col("review_text")
    log.info("running NAIVE (1 row per encode call)...")
    t_naive = run(texts, naive_udf()(col))
    log.info("naive:   %.1fs  (%.0f rows/s)", t_naive, n / t_naive)
    log.info("running BATCHED (pandas_udf, batch_size=%d)...", EMBED_BATCH_SIZE)
    t_batched = run(texts, batched_udf()(col))
    log.info("batched: %.1fs  (%.0f rows/s)", t_batched, n / t_batched)

    # Anti-pattern on a tiny sample (loads the model for every single row).
    bad = texts.limit(n_bad).repartition(cores)
    log.info("running ANTI-PATTERN (load model on every row) on only %d rows...", n_bad)
    t_bad = run(bad, reload_each_row_udf()(col))
    log.info("reload-per-row: %.1fs for %d rows (%.2f rows/s)", t_bad, n_bad, n_bad / t_bad)

    # Sanity: both approaches must produce (nearly) the same vectors.
    chk = texts.limit(200).repartition(cores)
    a = np.array([r.e for r in chk.select(naive_udf()(col).alias("e")).collect()], dtype="float32")
    b = np.array([r.e for r in chk.select(batched_udf()(col).alias("e")).collect()], dtype="float32")
    cos = float(np.min(np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))))
    log.info("vectors agree: min cosine(naive, batched) over 200 rows = %.6f ; dim=%d", cos, a.shape[1])
    assert cos > 0.999, "naive and batched embeddings disagree"

    result = {
        "model": EMBEDDING_MODEL, "rows": n, "batch_size": EMBED_BATCH_SIZE, "spark_cores": cores,
        "torch_threads_per_worker": env_int("EMBED_TORCH_THREADS", 1),
        "naive_seconds": round(t_naive, 2), "naive_rows_per_sec": round(n / t_naive, 1),
        "batched_seconds": round(t_batched, 2), "batched_rows_per_sec": round(n / t_batched, 1),
        "speedup_batched_vs_naive": round(t_naive / t_batched, 2),
        "antipattern_rows": n_bad, "antipattern_seconds": round(t_bad, 2),
        "antipattern_rows_per_sec": round(n_bad / t_bad, 2),
        "speedup_batched_vs_antipattern": round((n / t_batched) / (n_bad / t_bad), 1),
        "warmup_seconds_first_includes_model_load": round(t_warm_naive, 1),
        "warmup_seconds_second_model_already_cached": round(t_warm_batched, 1),
        "min_cosine_naive_vs_batched": round(cos, 6),
    }
    BENCH_DIR.mkdir(exist_ok=True)
    (BENCH_DIR / "embedding_benchmark.json").write_text(json.dumps(result, indent=2))
    md = f"""# Embedding benchmark: naive UDF vs batched pandas_udf

Model `{EMBEDDING_MODEL}` (CPU), {cores} Spark cores, {result['torch_threads_per_worker']} torch thread(s) per worker.
Measured on this machine; the model is already loaded in every worker (warm-up excluded).

| Approach | Rows | Seconds | Rows/sec | Speedup vs naive |
|---|---:|---:|---:|---:|
| Naive `udf` (1 row per `encode`) | {n} | {t_naive:.1f} | {n / t_naive:.0f} | 1.00x |
| Batched `pandas_udf` (batch_size={EMBED_BATCH_SIZE}) | {n} | {t_batched:.1f} | {n / t_batched:.0f} | {t_naive / t_batched:.2f}x |
| ANTI-PATTERN: load model every row | {n_bad} | {t_bad:.1f} | {n_bad / t_bad:.2f} | n/a (tiny N) |

The anti-pattern row ran on only {n_bad} rows, because it is that slow; compare rows/sec, not seconds.
Batched vs the anti-pattern: {result['speedup_batched_vs_antipattern']}x in rows/sec.
Naive and batched vectors agree (min cosine similarity {cos:.6f} over 200 rows).
"""
    (BENCH_DIR / "embedding_benchmark.md").write_text(md)
    print("\n" + md)
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
