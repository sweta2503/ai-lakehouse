"""Hybrid (SQL filter + vector) search over lake.gold.review_embeddings. Shared by 08, 09, tests and the bonus.

Two modes
---------
brute : score every row that passes the SQL filters.
ivf   : find the `nprobe` centroids closest to the query (in Python, from lake.gold.centroids), add
        `cluster_id IN (...)` to the filters, and score only those rows. Because the table is sorted by
        cluster_id, Iceberg's file min/max statistics let the planner skip the files of all other clusters.

Score = dot(query, embedding), computed with Spark SQL higher-order functions. Both are unit vectors, so
this is cosine similarity. Ties are broken by (review_id, order_id) — review_id alone is NOT unique in real
Olist data (1,603 rows share one) — so results are deterministic (important when
comparing brute force and IVF: identical vectors would otherwise tie in arbitrary order).
"""
from dataclasses import dataclass, field

import numpy as np
from pyspark.sql import functions as F

import time

from common import T_CENTROIDS, T_GOLD, T_TRANS

METRIC_KEYS = ["resultDataFiles", "skippedDataFiles", "totalDataManifests", "scannedDataManifests",
               "skippedDataManifests", "totalFileSize", "numOutputRows", "totalPlanningDuration"]


@dataclass
class SearchResult:
    rows: list
    seconds: float
    mode: str
    clusters: list[int] | None
    metrics: dict = field(default_factory=dict)
    df: object = None


def load_centroids(spark) -> np.ndarray:
    rows = spark.table(T_CENTROIDS).orderBy("cluster_id").collect()
    assert [r.cluster_id for r in rows] == list(range(len(rows))), "centroid ids must be 0..k-1"
    return np.array([r.centroid for r in rows], dtype="float32")


def nearest_clusters(q: np.ndarray, C: np.ndarray, nprobe: int) -> list[int]:
    """Centroids are unit-norm and so is q: dot product == cosine similarity."""
    sims = C @ q
    return [int(i) for i in np.argsort(-sims)[:nprobe]]


def translations(spark, texts) -> dict:
    """review_text -> English text for the given texts (display only; empty dict if no translation table).

    Kept outside the gold table on purpose: the index and embeddings use the ORIGINAL text, and a lookup of
    at most top-k strings does not touch the search benchmark.
    """
    texts = sorted(set(texts))
    if not texts or not spark.catalog.tableExists(T_TRANS):
        return {}
    rows = spark.table(T_TRANS).where(F.col("review_text").isin(*texts)).collect()
    return {r.review_text: r.review_text_en for r in rows}


def scan_metrics(df) -> dict:
    """Iceberg scan metrics of an already-executed DataFrame (needs spark.sql.adaptive.enabled=false)."""
    out: dict = {}
    leaves = df._jdf.queryExecution().executedPlan().collectLeaves()
    for i in range(leaves.size()):
        leaf = leaves.apply(i)
        if "BatchScan" not in leaf.nodeName():
            continue
        it = leaf.metrics().iterator()
        while it.hasNext():
            kv = it.next()
            if kv._1() in METRIC_KEYS:
                out[kv._1()] = out.get(kv._1(), 0) + int(kv._2().value())
    return out


def explain_text(df) -> str:
    mode = df.sparkSession._jvm.org.apache.spark.sql.execution.ExplainMode.fromString("formatted")
    return df._jdf.queryExecution().explainString(mode)


def build_query(spark, q: np.ndarray, *, clusters=None, since=None, until=None, status=None, top_k=10):
    spark.conf.set("spark.sql.adaptive.enabled", "false")  # so plan-level scan metrics are readable
    qarr = F.array(*[F.lit(float(x)) for x in q])
    score = F.aggregate(F.zip_with(F.col("embedding"), qarr, lambda a, b: a * b), F.lit(0.0), lambda acc, x: acc + x)
    df = spark.table(T_GOLD)
    if since:
        df = df.where(F.col("purchase_ts") >= F.lit(since).cast("timestamp"))
    if until:
        df = df.where(F.col("purchase_ts") < F.lit(until).cast("timestamp"))
    if status:
        df = df.where(F.col("order_status") == status)
    if clusters is not None:
        df = df.where(F.col("cluster_id").isin(*clusters))
    return (df.select("review_id", "order_id", "purchase_ts", "order_status", "review_score", "review_text",
                      "cluster_id", score.alias("score"))
            .orderBy(F.desc("score"), F.asc("review_id"), F.asc("order_id")).limit(top_k))


def search(spark, q: np.ndarray, *, mode="ivf", C=None, nprobe=3, since=None, until=None, status=None,
           top_k=10) -> SearchResult:
    t0 = time.perf_counter()
    clusters = None
    if mode == "ivf":
        C = load_centroids(spark) if C is None else C
        clusters = nearest_clusters(q, C, nprobe)
    elif mode != "brute":
        raise ValueError(mode)
    df = build_query(spark, q, clusters=clusters, since=since, until=until, status=status, top_k=top_k)
    rows = df.collect()
    secs = time.perf_counter() - t0
    return SearchResult(rows, secs, mode, clusters, scan_metrics(df), df)
