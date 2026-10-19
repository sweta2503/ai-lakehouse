"""Locate the input CSVs (Olist if present, otherwise synthetic) and build the joined frame.

Lookup order:  data/*.csv  ->  data/olist/*.csv  ->  data/synthetic/*.csv  ->  generate synthetic now.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from common import DATA_DIR, ROOT, SAMPLE_ROWS, env_int, get_logger

log = get_logger("data")
ORDERS = "olist_orders_dataset.csv"
REVIEWS = "olist_order_reviews_dataset.csv"


def locate_csvs() -> tuple[Path, Path, str]:
    """Returns (orders_csv, reviews_csv, source) where source is 'olist' or 'synthetic'."""
    for sub, name in (("", "olist"), ("olist", "olist"), ("synthetic", "synthetic")):
        d = DATA_DIR / sub if sub else DATA_DIR
        if (d / ORDERS).exists() and (d / REVIEWS).exists():
            return d / ORDERS, d / REVIEWS, name
    log.warning("Olist CSVs not found in %s -> generating a synthetic dataset (see scripts/download_data.sh "
                "to use the real one)", DATA_DIR)
    out = DATA_DIR / "synthetic"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_synthetic_data.py"), "--out", str(out),
                    "--rows", str(env_int("SYNTHETIC_ROWS", 50000))], check=True)
    return out / ORDERS, out / REVIEWS, "synthetic"


def load_joined(spark):
    """orders JOIN reviews -> one row per review, with a cleaned `review_text` and `has_review_text` flag.

    Returns (DataFrame, source_name). Honors SAMPLE_ROWS (random but seeded sample).
    """
    from pyspark.sql import functions as F

    orders_csv, reviews_csv, source = locate_csvs()
    log.info("DATA SOURCE: %s  (%s, %s)", source.upper(), orders_csv.name, reviews_csv.name)

    orders = spark.read.csv(str(orders_csv), header=True)
    # Olist review messages contain quoted newlines and doubled quotes, hence multiLine + escape.
    reviews = spark.read.csv(str(reviews_csv), header=True, multiLine=True, quote='"', escape='"')

    ts = lambda c: F.to_timestamp(F.col(c))  # noqa: E731
    o = orders.select(
        "order_id", "customer_id", "order_status",
        ts("order_purchase_timestamp").alias("order_purchase_timestamp"),
        ts("order_approved_at").alias("order_approved_at"),
        ts("order_delivered_customer_date").alias("order_delivered_customer_date"),
        ts("order_estimated_delivery_date").alias("order_estimated_delivery_date"),
    )

    def clean(c):  # collapse whitespace/newlines, empty -> NULL
        return F.nullif(F.trim(F.regexp_replace(F.col(c), r"\s+", " ")), F.lit(""))

    r = reviews.select(
        "review_id", "order_id",
        F.col("review_score").cast("int").alias("review_score"),
        # title + message combined; concat_ws skips NULLs so either may be missing
        F.nullif(F.concat_ws(" - ", clean("review_comment_title"), clean("review_comment_message")),
                 F.lit("")).alias("review_text"),
        ts("review_creation_date").alias("review_creation_date"),
        ts("review_answer_timestamp").alias("review_answer_timestamp"),
    )
    df = (o.join(r, "order_id", "inner")
          .withColumn("has_review_text", F.col("review_text").isNotNull())
          .where(F.col("order_purchase_timestamp").isNotNull() & F.col("review_score").isNotNull()))

    if SAMPLE_ROWS:
        log.info("SAMPLE_ROWS=%s -> taking a seeded random sample", SAMPLE_ROWS)
        df = df.orderBy(F.rand(42)).limit(SAMPLE_ROWS)
    return df, source
