"""M1: prove Spark <-> Iceberg REST <-> MinIO works end to end.

Creates namespaces bronze/silver/gold, writes a tiny Iceberg table, reads it back,
and confirms the Parquet file really landed in the MinIO bucket.
"""
import sys

from common import CATALOG, S3_ENDPOINT, CATALOG_URI, get_logger, get_spark, section, timed

log = get_logger("check")


def main() -> int:
    section("M1: connection check")
    log.info("catalog=%s rest=%s s3=%s", CATALOG, CATALOG_URI, S3_ENDPOINT)
    with timed("start Spark (first run downloads Iceberg jars)", log):
        spark = get_spark("01_check_connection")

    for ns in ("bronze", "silver", "gold"):
        spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {CATALOG}.{ns}")
    log.info("namespaces: %s", [r[0] for r in spark.sql(f"SHOW NAMESPACES IN {CATALOG}").collect()])

    t = f"{CATALOG}.bronze.connection_check"
    spark.sql(f"DROP TABLE IF EXISTS {t} PURGE")
    spark.sql(f"CREATE TABLE {t} (id INT, msg STRING) USING iceberg")
    spark.sql(f"INSERT INTO {t} VALUES (1, 'hello'), (2, 'lakehouse')")
    rows = spark.sql(f"SELECT * FROM {t} ORDER BY id").collect()
    print(rows)
    assert [(r.id, r.msg) for r in rows] == [(1, "hello"), (2, "lakehouse")], "round trip failed"

    files = spark.sql(f"SELECT file_path FROM {t}.files").collect()
    log.info("data file(s) written to: %s", [f.file_path for f in files])
    assert files and files[0].file_path.startswith("s3://"), "data file is not on S3/MinIO"

    spark.sql(f"DROP TABLE {t} PURGE")
    log.info("PASS: Spark -> REST catalog -> MinIO round trip works")
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
