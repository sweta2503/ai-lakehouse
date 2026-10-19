"""M4 (1/2): train the IVF "coarse quantizer" — k-means centroids over the embeddings.

HONESTY NOTE: Iceberg has no native vector type and no ANN index. What we build here is an *application-level*
IVF (inverted file) technique: cluster the vectors, remember which cluster each row belongs to, and at query
time only look inside the few clusters nearest the query. Iceberg's only job is to store `cluster_id` and to
skip files via its normal column statistics (see 07_gold_write.py).

Cosine distance + unit-norm vectors: Spark's KMeans with distanceMeasure='cosine' groups vectors by angle,
which is exactly what "cosine similarity == dot product" retrieval needs.
"""
import sys

import numpy as np
from pyspark.ml.clustering import KMeans
from pyspark.ml.functions import array_to_vector
from pyspark.sql.types import ArrayType, FloatType, IntegerType, LongType, StructField, StructType

from common import IVF_K, IVF_SEED, T_CENTROIDS, T_SILVER, get_logger, get_spark, section, timed

log = get_logger("ivf")


def main() -> int:
    section("M4: train IVF centroids (Spark MLlib KMeans)")
    spark = get_spark("06_build_ivf_index")
    spark.sql("CREATE NAMESPACE IF NOT EXISTS lake.gold")

    silver = spark.table(T_SILVER)
    n = silver.count()
    k = min(IVF_K, max(2, n // 20))  # guard for tiny SAMPLE_ROWS runs
    if k != IVF_K:
        log.warning("k reduced from %d to %d because only %d vectors exist", IVF_K, k, n)
    # array<float> -> MLlib Vector (what KMeans consumes)
    feats = silver.select(array_to_vector("embedding").alias("features")).cache()
    feats.count()

    km = KMeans(k=k, seed=IVF_SEED, maxIter=30, featuresCol="features", distanceMeasure="cosine")
    with timed(f"KMeans k={k} on {n} vectors", log):
        model = km.fit(feats)

    centers = np.array(model.clusterCenters(), dtype="float64")
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)  # normalise: score = dot(query, centroid) = cosine
    sizes = np.array(model.summary.clusterSizes)
    log.info("iterations=%d  cluster sizes: min=%d median=%d max=%d  (empty=%d)", model.summary.numIter,
             sizes.min(), int(np.median(sizes)), sizes.max(), int((sizes == 0).sum()))

    schema = StructType([StructField("cluster_id", IntegerType(), False),
                         StructField("centroid", ArrayType(FloatType()), False),
                         StructField("size", LongType(), False)])
    rows = [(i, centers[i].astype("float32").tolist(), int(sizes[i])) for i in range(k)]
    spark.sql(f"DROP TABLE IF EXISTS {T_CENTROIDS} PURGE")
    spark.createDataFrame(rows, schema).writeTo(T_CENTROIDS).using("iceberg").create()
    log.info("wrote %d centroids -> %s", k, T_CENTROIDS)
    spark.table(T_CENTROIDS).orderBy("cluster_id").show(5, truncate=40)
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
