"""M2 (1/2): Bronze ingest. orders JOIN reviews -> Iceberg v2 table lake.bronze.reviews.

* Partitioned by months(order_purchase_timestamp): Iceberg *hidden partitioning* — queries filter on the
  raw timestamp and Iceberg maps that to month partitions; no extra partition column to maintain.
* Rows without any review text are KEPT (they still carry a score/status) but flagged with
  has_review_text = false. Only has_review_text rows get embeddings later.
* To make the schema-drift demo (03) realistic, the most recent ~3% of orders are held back as the
  "next batch" in data/landing/ and appended later with a new column.
"""
import sys

from pyspark.sql import functions as F

from common import DATA_DIR, T_BRONZE, get_logger, get_spark, section, timed
from data_source import load_joined

log = get_logger("bronze")
HOLDBACK_FRACTION = 0.03


def main() -> int:
    section("M2: Bronze ingest")
    spark = get_spark("02_bronze_ingest")
    spark.sql("CREATE NAMESPACE IF NOT EXISTS lake.bronze")

    df, source = load_joined(spark)
    df = df.cache()
    total = df.count()

    # Time-based holdback: latest HOLDBACK_FRACTION of orders become the "next batch".
    q = df.select(F.col("order_purchase_timestamp").cast("long").alias("t")).approxQuantile(
        "t", [1 - HOLDBACK_FRACTION], 0.0)[0]
    batch1 = df.where(F.col("order_purchase_timestamp").cast("long") < q)
    batch2 = df.where(F.col("order_purchase_timestamp").cast("long") >= q)

    landing = DATA_DIR / "landing" / "reviews_batch2.parquet"
    with timed("stage held-back batch for 03_schema_drift", log):
        batch2.write.mode("overwrite").parquet(str(landing))
    log.info("held back %d rows -> %s", batch2.count(), landing)

    with timed("write lake.bronze.reviews", log):
        # DROP (not createOrReplace): a replace keeps the old snapshot history, which would make the
        # time-travel demo in 03 confusing on re-runs. PURGE also deletes the old files from MinIO.
        spark.sql(f"DROP TABLE IF EXISTS {T_BRONZE} PURGE")
        (batch1.writeTo(T_BRONZE).using("iceberg")
         .partitionedBy(F.months("order_purchase_timestamp"))
         .tableProperty("format-version", "2")
         .create())

    n = spark.table(T_BRONZE).count()
    log.info("source=%s  joined_rows=%d  bronze_rows=%d", source, total, n)
    print("\nBronze schema:")
    spark.table(T_BRONZE).printSchema()
    print("Rows with / without review text, and by status:")
    spark.sql(f"""SELECT order_status, has_review_text, count(*) AS n
                  FROM {T_BRONZE} GROUP BY 1, 2 ORDER BY n DESC""").show(20, truncate=False)
    print("Partitions (one per month):")
    # Iceberg stores months(...) as "months since 1970-01"; render as a date for humans.
    spark.sql(f"""SELECT add_months(DATE'1970-01-01', partition.order_purchase_timestamp_month) AS month,
                         record_count, file_count
                  FROM {T_BRONZE}.partitions ORDER BY month""").show(30, False)
    props = {r.key: r.value for r in spark.sql(f"SHOW TBLPROPERTIES {T_BRONZE}").collect()}
    log.info("format-version=%s", props.get("format-version"))
    assert props.get("format-version") == "2"
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
