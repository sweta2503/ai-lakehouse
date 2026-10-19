"""M3 (2/2): Silver layer — embed every review that has text using the batched pandas_udf.

lake.silver.review_embeddings: review_id, order_id, purchase_ts, order_status, review_score, review_text,
embedding array<float> (384 floats, L2-normalised, so cosine similarity == dot product).

NOTE: Iceberg has no vector type. `array<float>` is just a list column; nothing in Iceberg knows it is
an embedding.
"""
import sys

from pyspark.sql import functions as F

from common import EMBEDDING_DIM, T_BRONZE, T_SILVER, get_logger, get_spark, local_cores, section, timed
from embeddings import batched_udf

log = get_logger("silver")


def main() -> int:
    section("M3: Silver embeddings")
    spark = get_spark("05_silver_embeddings")
    cores = local_cores()
    spark.sql("CREATE NAMESPACE IF NOT EXISTS lake.silver")

    src = spark.table(T_BRONZE).where("has_review_text")
    n_text = src.count()
    log.info("bronze rows with review text: %d (null/empty text is skipped on purpose)", n_text)

    # One Spark partition per core: each partition runs in one Python worker that loads the model once
    # and encodes Arrow batches of rows. More partitions than cores just adds scheduling overhead;
    # fewer leaves cores idle.
    df = (src.select("review_id", "order_id",
                     F.col("order_purchase_timestamp").alias("purchase_ts"),
                     "order_status", "review_score", "review_text")
          .repartition(cores)
          .withColumn("embedding", batched_udf()(F.col("review_text"))))

    with timed(f"embed + write {n_text} rows", log) as t:
        spark.sql(f"DROP TABLE IF EXISTS {T_SILVER} PURGE")
        df.writeTo(T_SILVER).using("iceberg").tableProperty("format-version", "2").create()
    log.info("throughput: %.0f rows/sec", n_text / t["seconds"])

    s = spark.table(T_SILVER)
    n = s.count()
    stats = s.select(F.size("embedding").alias("d"),
                     F.sqrt(F.aggregate("embedding", F.lit(0.0), lambda a, x: a + x * x)).alias("norm")
                     ).agg(F.min("d"), F.max("d"), F.min("norm"), F.max("norm")).first()
    log.info("silver rows=%d  dim(min,max)=(%d,%d)  L2 norm(min,max)=(%.5f,%.5f)", n, stats[0], stats[1], stats[2], stats[3])
    assert n == n_text and stats[0] == stats[1] == EMBEDDING_DIM
    assert abs(stats[2] - 1) < 1e-3 and abs(stats[3] - 1) < 1e-3, "embeddings are not unit norm"
    s.printSchema()
    s.select("order_id", "order_status", "review_score", "review_text").show(5, truncate=70)
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
