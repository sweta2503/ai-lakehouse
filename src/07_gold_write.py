"""M4 (2/2): Gold layer — assign cluster_id to every row and write a table SORTED by it.

Why sorting by cluster_id makes Iceberg skip files
--------------------------------------------------
Every Iceberg data file records per-column min/max values in its manifest entry. If rows are written in
cluster_id order, each file holds a narrow cluster_id range (e.g. 12..17), so a predicate like
`cluster_id IN (14, 31)` lets the planner discard every file whose [min, max] range does not contain 14 or
31 — *before reading a single Parquet byte*. If rows were NOT sorted, every file would contain every
cluster, every file's range would be [0, 49], and nothing could be skipped.

(Iceberg does this with ordinary column statistics. It is not a vector index, and not ANN.)

Usage:  python 07_gold_write.py [--fragment N] [--compact]
  --fragment N  write the data as N separate random appends (simulates incremental loads: each append is
                sorted on its own, so cluster ranges overlap across files and pruning gets worse)
  --compact     afterwards run rewrite_data_files(strategy => 'sort') and show before/after file counts
"""
import argparse
import sys
from typing import Iterator

import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import IntegerType

from common import T_CENTROIDS, T_GOLD, T_SILVER, TARGET_FILE_SIZE_BYTES, get_logger, get_spark, section, timed

log = get_logger("gold")


def assign_udf(centroids: np.ndarray):
    """cluster_id = argmax_c dot(embedding, centroid_c). Same rule the query side uses to pick clusters."""
    C = centroids.astype("float32")

    @pandas_udf(IntegerType())
    def assign(batches: Iterator[pd.Series]) -> Iterator[pd.Series]:
        for s in batches:
            E = np.vstack(s.to_numpy()).astype("float32")
            yield pd.Series((E @ C.T).argmax(axis=1).astype("int32"))

    return assign


def gold_file_stats(spark) -> tuple[int, int, float]:
    r = spark.sql(f"SELECT count(*) n, sum(file_size_in_bytes) b FROM {T_GOLD}.files").first()
    return r.n, r.b, r.b / max(r.n, 1)


def files_for_clusters(spark, clusters: list[int]) -> tuple[int, int]:
    """(files whose cluster_id [min,max] could contain any of `clusters`, total files) — straight from metadata."""
    in_list = ",".join(map(str, clusters))
    # Overlap test written out explicitly: a file survives if some probed id lies inside its [lower, upper].
    conds = " OR ".join(f"({c} BETWEEN readable_metrics.cluster_id.lower_bound AND readable_metrics.cluster_id.upper_bound)"
                        for c in clusters)
    r = spark.sql(f"SELECT count(*) total, sum(CASE WHEN {conds} THEN 1 ELSE 0 END) hit FROM {T_GOLD}.files").first()
    return int(r.hit), int(r.total)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fragment", type=int, default=1)
    ap.add_argument("--compact", action="store_true")
    a = ap.parse_args()

    section("M4: Gold write (cluster_id + sorted Iceberg table)")
    spark = get_spark("07_gold_write")

    cent = spark.table(T_CENTROIDS).orderBy("cluster_id").collect()
    C = np.array([r.centroid for r in cent], dtype="float32")
    log.info("loaded %d centroids x %d dims from %s", *C.shape, T_CENTROIDS)

    df = spark.table(T_SILVER).withColumn("cluster_id", assign_udf(C)("embedding"))

    spark.sql(f"DROP TABLE IF EXISTS {T_GOLD} PURGE")
    spark.sql(f"""
        CREATE TABLE {T_GOLD} (
            review_id STRING, order_id STRING, purchase_ts TIMESTAMP, order_status STRING,
            review_score INT, review_text STRING, embedding ARRAY<FLOAT>, cluster_id INT)
        USING iceberg
        PARTITIONED BY (months(purchase_ts))
        TBLPROPERTIES (
            'format-version' = '2',
            -- small target size so demo-scale data still yields many files per month; at real scale you
            -- would leave Iceberg's 512 MB default and get the same effect with far more rows.
            'write.target-file-size-bytes' = '{TARGET_FILE_SIZE_BYTES}',
            -- no min/max stats for the 384-float array: useless for pruning and bloats manifests
            'write.metadata.metrics.column.embedding' = 'none')""")
    # Table-level sort order. Every write (Spark respects it) now sorts rows by cluster_id inside each partition.
    spark.sql(f"ALTER TABLE {T_GOLD} WRITE ORDERED BY cluster_id")
    log.info("table sort order: %s", spark.sql(f"SHOW TBLPROPERTIES {T_GOLD}").where("key = 'sort-order'").first().value)

    with timed(f"write gold ({a.fragment} append(s))", log):
        if a.fragment <= 1:
            df.writeTo(T_GOLD).append()
        else:
            df = df.cache()
            df.count()
            bucket = F.abs(F.hash("review_id")) % a.fragment
            for i in range(a.fragment):
                df.where(bucket == i).writeTo(T_GOLD).append()
            df.unpersist()

    gold = spark.table(T_GOLD)
    n = gold.count()
    nulls = gold.where("cluster_id IS NULL").count()
    log.info("gold rows=%d  rows missing cluster_id=%d", n, nulls)
    assert nulls == 0

    nf, nb, avg = gold_file_stats(spark)
    log.info("data files: %d  (total %.1f MB, avg %.0f KB)", nf, nb / 1e6, avg / 1e3)
    print("\nrows / files per month partition, and the cluster_id range inside each file:")
    spark.sql(f"""SELECT add_months(DATE'1970-01-01', partition.purchase_ts_month) AS month, record_count rows,
                         readable_metrics.cluster_id.lower_bound AS cid_min, readable_metrics.cluster_id.upper_bound AS cid_max,
                         round(file_size_in_bytes/1024) AS kb
                  FROM {T_GOLD}.files ORDER BY month, cid_min LIMIT 14""").show(truncate=False)

    probe = [int(c) for c in np.random.default_rng(0).choice(len(cent), size=3, replace=False)]
    hit, total = files_for_clusters(spark, probe)
    log.info("PRUNING PREVIEW: cluster_id IN %s -> Iceberg would scan %d of %d data files (%.0f%% skipped)",
             probe, hit, total, 100 * (1 - hit / total))

    # Cluster sanity: do clusters mean something? Show a couple of reviews from the biggest clusters.
    print("\nWhat do clusters look like? (3 biggest clusters, 2 sample reviews each)")
    top = gold.groupBy("cluster_id").count().orderBy(F.desc("count")).limit(3).collect()
    for r in top:
        print(f"\ncluster {r.cluster_id} ({r['count']} reviews)")
        for t in gold.where(F.col("cluster_id") == r.cluster_id).select("review_text").limit(2).collect():
            print("   -", t.review_text[:110])

    if a.compact:
        section("Compaction: rewrite_data_files with sort strategy")
        before = gold_file_stats(spark)[0]
        hit_b, tot_b = files_for_clusters(spark, probe)
        with timed("rewrite_data_files(strategy => 'sort')", log):
            res = spark.sql(f"""CALL lake.system.rewrite_data_files(
                table => 'gold.review_embeddings', strategy => 'sort', sort_order => 'cluster_id ASC NULLS LAST',
                options => map('target-file-size-bytes', '{TARGET_FILE_SIZE_BYTES}', 'rewrite-all', 'true',
                               'max-file-group-size-bytes', '1073741824'))""")
        res.show(truncate=False)
        after = gold_file_stats(spark)[0]
        hit_a, tot_a = files_for_clusters(spark, probe)
        print(f"data files before: {before}  after: {after}")
        print(f"files a cluster_id IN {probe} query must scan: before {hit_b}/{tot_b}  after {hit_a}/{tot_a}")
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
