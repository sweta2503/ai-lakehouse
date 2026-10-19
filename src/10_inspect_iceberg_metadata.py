"""M6: demystify the table format. Look at Iceberg's metadata tables AND the physical objects in MinIO.

The chain Iceberg follows to read a table:

    catalog (REST)  ->  table metadata JSON   (schema, partition spec, sort order, list of snapshots)
                          -> snapshot -> manifest list   (snap-<id>-....avro: one entry per manifest file)
                                -> manifest files        (*-m0.avro: one entry per data file, with partition
                                                          values + per-column min/max/null counts)
                                      -> data files      (Parquet, under data/)

Spark exposes those layers as metadata tables: <table>.snapshots, .history, .manifests (= contents of the
manifest list), .files (= entries across the manifests), .partitions.
"""
import json
import sys

import boto3
from botocore.config import Config

from common import AWS_REGION, S3_ACCESS_KEY, S3_ENDPOINT, S3_SECRET_KEY, T_GOLD, WAREHOUSE_BUCKET, get_logger, get_spark, section

log = get_logger("inspect")


def s3_client():
    return boto3.client("s3", endpoint_url=S3_ENDPOINT, aws_access_key_id=S3_ACCESS_KEY,
                        aws_secret_access_key=S3_SECRET_KEY, region_name=AWS_REGION,
                        config=Config(s3={"addressing_style": "path"}))


def list_prefix(s3, bucket, prefix):
    out, token = [], None
    while True:
        kw = dict(Bucket=bucket, Prefix=prefix, **({"ContinuationToken": token} if token else {}))
        r = s3.list_objects_v2(**kw)
        out += [(o["Key"], o["Size"]) for o in r.get("Contents", [])]
        token = r.get("NextContinuationToken")
        if not token:
            return out


def main() -> int:
    spark = get_spark("10_inspect_iceberg_metadata")
    t = T_GOLD
    section(f"Iceberg metadata tables for {t}")

    print("\n[snapshots]  one row per commit; each points to ONE manifest list")
    spark.sql(f"SELECT committed_at, snapshot_id, operation, manifest_list FROM {t}.snapshots").show(truncate=False)
    print("[history]  which snapshot was current, and when")
    spark.sql(f"SELECT * FROM {t}.history").show(truncate=False)
    print("[manifests]  = the entries of the current manifest list: one row per manifest file")
    spark.sql(f"""SELECT path, length, added_snapshot_id, added_data_files_count, existing_data_files_count,
                         deleted_data_files_count FROM {t}.manifests""").show(truncate=False)
    nfiles = spark.sql(f"SELECT count(*) n FROM {t}.files").first().n
    print(f"[files]  data files tracked by the manifests: {nfiles}. First few, with the per-file cluster_id min/max "
          "that makes pruning possible:")
    spark.sql(f"""SELECT file_path, record_count, file_size_in_bytes,
                         readable_metrics.cluster_id.lower_bound AS cluster_min,
                         readable_metrics.cluster_id.upper_bound AS cluster_max
                  FROM {t}.files ORDER BY file_path LIMIT 6""").show(truncate=60)
    print("[partitions]  rows per month partition (months(purchase_ts) is stored as months since 1970-01)")
    spark.sql(f"""SELECT add_months(DATE'1970-01-01', partition.purchase_ts_month) AS month, record_count, file_count
                  FROM {t}.partitions ORDER BY month""").show(30, truncate=False)

    # ---- physical layout in MinIO -------------------------------------------------------------
    section("Physical objects in MinIO (what is actually on disk)")
    loc = spark.sql(f"DESCRIBE TABLE EXTENDED {t}").where("col_name = 'Location'").first().data_type
    bucket, _, prefix = loc.removeprefix("s3://").partition("/")
    prefix = prefix.rstrip("/") + "/"
    log.info("table location: %s  (bucket=%s prefix=%s)", loc, bucket, prefix)
    s3 = s3_client()

    meta = list_prefix(s3, bucket, prefix + "metadata/")
    data = list_prefix(s3, bucket, prefix + "data/")
    groups = {
        "table metadata JSON (*.metadata.json)  — the root; the catalog points at the newest one":
            [m for m in meta if m[0].endswith(".metadata.json")],
        "manifest lists (snap-*.avro)           — one per snapshot": [m for m in meta if "/snap-" in m[0]],
        "manifest files (*-m*.avro)             — list data files + stats": [m for m in meta if m[0].endswith(".avro") and "/snap-" not in m[0]],
    }
    for title, items in groups.items():
        print(f"\n{title}: {len(items)} object(s)")
        for k, sz in items[:4]:
            print(f"   {k.removeprefix(prefix)}   ({sz:,} B)")
    parquet = [d for d in data if d[0].endswith(".parquet")]
    print(f"\ndata files (Parquet under data/): {len(parquet)} object(s), {sum(s for _, s in parquet) / 1e6:.1f} MB total")
    for k, sz in parquet[:4]:
        print(f"   {k.removeprefix(prefix)}   ({sz:,} B)")

    latest = max(groups[list(groups)[0]], key=lambda kv: kv[0])[0]
    doc = json.loads(s3.get_object(Bucket=bucket, Key=latest)["Body"].read())
    print(f"\nInside the newest table metadata JSON ({latest.removeprefix(prefix)}):")
    schema = next(sc for sc in doc["schemas"] if sc["schema-id"] == doc["current-schema-id"])
    names = {f["id"]: f["name"] for f in schema["fields"]}
    summary = {
        "format-version": doc.get("format-version"), "table-uuid": doc.get("table-uuid"),
        "current-snapshot-id": doc.get("current-snapshot-id"),
        "partition-spec": [f"{f['name']} = {f['transform']}({names[f['source-id']]})" for f in doc["partition-specs"][-1]["fields"]],
        "default-sort-order": [f"{f['transform']}({names[f['source-id']]}) {f['direction']} {f['null-order']}"
                               for f in doc["sort-orders"][-1]["fields"]] if doc.get("sort-orders") else None,
        "num_snapshots": len(doc.get("snapshots", [])),
        "snapshot[-1].manifest-list": doc["snapshots"][-1]["manifest-list"] if doc.get("snapshots") else None,
        "properties": doc.get("properties"),
    }
    print(json.dumps(summary, indent=2))
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
