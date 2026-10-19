"""M2 (2/2): schema evolution + time travel.

A "new batch" arrives with a column the table has never seen (review_channel). We:
  1. ALTER TABLE ... ADD COLUMN  (metadata-only; no data files are rewritten)
  2. append the batch staged by 02_bronze_ingest
  3. show that old rows read NULL for the new column, and that OLD SNAPSHOTS still read with the
     OLD schema (time travel by snapshot id and by timestamp).
"""
import sys
import time

from pyspark.sql import functions as F

from common import DATA_DIR, T_BRONZE, get_logger, get_spark, section

log = get_logger("drift")


def main() -> int:
    section("M2: schema drift + time travel")
    spark = get_spark("03_schema_drift")
    landing = DATA_DIR / "landing" / "reviews_batch2.parquet"
    if not landing.exists():
        log.error("%s missing. Run `make bronze` first.", landing)
        return 1
    if "review_channel" in spark.table(T_BRONZE).columns:
        log.error("review_channel already exists: drift was already applied. Re-run `make bronze` to reset.")
        return 1

    snaps0 = spark.sql(f"SELECT snapshot_id, committed_at FROM {T_BRONZE}.snapshots ORDER BY committed_at").collect()
    first = snaps0[0]
    rows_before = spark.table(T_BRONZE).count()
    log.info("before: %d rows, %d snapshot(s), %d columns", rows_before, len(snaps0), len(spark.table(T_BRONZE).columns))
    time.sleep(1.5)  # keep committed_at timestamps visibly distinct for the TIMESTAMP AS OF demo

    print("\n-- Step 1: ALTER TABLE ADD COLUMN (no data rewritten)")
    spark.sql(f"ALTER TABLE {T_BRONZE} ADD COLUMN review_channel STRING COMMENT 'new in batch 2: app|web|marketplace|email'")

    print("-- Step 2: append batch with the new column")
    channels = F.array(F.lit("app"), F.lit("web"), F.lit("marketplace"), F.lit("email"))
    batch2 = (spark.read.parquet(str(landing))
              .withColumn("review_channel", F.element_at(channels, (F.abs(F.hash("review_id")) % 4 + 1).cast("int"))))
    batch2.writeTo(T_BRONZE).append()

    cur = spark.table(T_BRONZE)
    n_after = cur.count()
    n_null = cur.where(F.col("review_channel").isNull()).count()
    n_new = cur.where(F.col("review_channel").isNotNull()).count()
    print(f"\nCurrent table: {n_after} rows | review_channel NULL (old rows): {n_null} | populated (new batch): {n_new}")
    assert n_null == rows_before and n_new == n_after - rows_before, "old rows should be NULL for new column"
    cur.groupBy("review_channel").count().orderBy("review_channel").show()

    print(f"-- Snapshots of {T_BRONZE}:")
    snaps = spark.sql(f"SELECT committed_at, snapshot_id, parent_id, operation FROM {T_BRONZE}.snapshots ORDER BY committed_at")
    snaps.show(truncate=False)
    snap_rows = snaps.collect()
    old_id, new_id = snap_rows[0].snapshot_id, snap_rows[-1].snapshot_id

    print(f"-- Time travel: VERSION AS OF {old_id} (the first snapshot)")
    old = spark.sql(f"SELECT * FROM {T_BRONZE} VERSION AS OF {old_id}")
    log.info("old snapshot: %d rows, has review_channel column? %s", old.count(), "review_channel" in old.columns)
    assert old.count() == rows_before and "review_channel" not in old.columns

    # Format in Spark (session tz = UTC) — a Python datetime would be converted to the local tz.
    ts = spark.sql(f"SELECT date_format(committed_at, 'yyyy-MM-dd HH:mm:ss.SSS') AS ts FROM {T_BRONZE}.snapshots "
                   f"WHERE snapshot_id = {old_id}").first().ts
    print(f"-- Time travel: TIMESTAMP AS OF '{ts}'")
    by_ts = spark.sql(f"SELECT count(*) AS n FROM {T_BRONZE} TIMESTAMP AS OF '{ts}'").first().n
    log.info("TIMESTAMP AS OF first commit -> %d rows", by_ts)
    assert by_ts == rows_before

    print(f"-- Current snapshot {new_id}: {n_after} rows, review_channel present")
    print("\nTakeaway: ADD COLUMN touched only table metadata. Old Parquet files were never rewritten; "
          "Iceberg tracks columns by ID, so old files simply read NULL for the new one.")
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
