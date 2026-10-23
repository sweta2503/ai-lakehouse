"""M3b (optional, display only): translate the distinct review texts to English with a free local model.

Writes lake.silver.review_translations (review_text, review_text_en). It is a lookup table keyed by the original
text; the gold table, the embeddings and the benchmark are NOT changed. The UI and the query CLI look up the few
results they show. Incremental: only texts missing from the table are translated, so re-runs are cheap.

Machine translation of short, slangy reviews is imperfect; treat it as a reading aid.
"""
import os
import sys

from pyspark.sql import functions as F

from common import T_SILVER, T_TRANS, TRANSLATE_MODEL, env, get_logger, get_spark, section, timed
from translation import translate_udf

log = get_logger("translate")


def main() -> int:
    section("M3b: translate review text to English (local model)")
    # Each Spark Python worker holds its own copy of the translation model (~300 MB+), so on a busy laptop fewer
    # workers is faster than more (swapping is far slower than idle cores). Override with TRANSLATE_CORES.
    os.environ["SPARK_CORES"] = env("TRANSLATE_CORES", "4")
    spark = get_spark("05b_translate_reviews")
    cores = int(os.environ["SPARK_CORES"])
    todo = spark.table(T_SILVER).select("review_text").distinct()
    exists = spark.catalog.tableExists(T_TRANS)
    if exists:
        todo = todo.join(spark.table(T_TRANS).select("review_text"), "review_text", "left_anti")
    # many small tasks (model is cached per worker, so extra tasks are free): visible progress in the Spark UI
    todo = todo.repartition(cores * 16).cache()
    n = todo.count()
    log.info("model=%s  distinct texts to translate: %d", TRANSLATE_MODEL, n)
    if n:
        out = todo.withColumn("review_text_en", translate_udf()(F.col("review_text")))
        with timed(f"translate {n} texts", log) as t:
            if exists:
                out.writeTo(T_TRANS).append()
            else:
                out.writeTo(T_TRANS).using("iceberg").tableProperty("format-version", "2").create()
        log.info("throughput: %.0f texts/sec", n / t["seconds"])
    tr = spark.table(T_TRANS)
    total = tr.count()
    missing = spark.table(T_SILVER).select("review_text").distinct().join(tr, "review_text", "left_anti").count()
    log.info("translation table rows=%d  silver texts without translation=%d", total, missing)
    assert missing == 0
    print("\nSamples (original -> English):")
    tr.where(F.length("review_text") > 40).limit(6).show(truncate=70)
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
