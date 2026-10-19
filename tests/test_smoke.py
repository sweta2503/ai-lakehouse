"""Smoke tests over the lakehouse as it currently exists. Run the pipeline first (`make all`, or for a fast
check: `make smoke`, which uses SAMPLE_ROWS=5000). Needs MinIO + the REST catalog up (`make up`)."""
import numpy as np
import pytest

from common import EMBEDDING_DIM, T_BRONZE, T_CENTROIDS, T_GOLD, T_SILVER, get_spark
from embeddings import encode_texts
from search import load_centroids, nearest_clusters, search


@pytest.fixture(scope="session")
def spark():
    s = get_spark("tests")
    missing = [t for t in (T_BRONZE, T_SILVER, T_CENTROIDS, T_GOLD) if not s.catalog.tableExists(t)]
    if missing:
        pytest.fail(f"tables missing: {missing}. Run the pipeline first (`make all` or `make smoke`).", pytrace=False)
    yield s
    s.stop()


@pytest.fixture(scope="session")
def C(spark):
    return load_centroids(spark)


def test_tables_exist_and_nonempty(spark):
    for t in (T_BRONZE, T_SILVER, T_CENTROIDS, T_GOLD):
        assert spark.table(t).count() > 0, t


def test_bronze_is_v2_partitioned_by_month(spark):
    props = {r.key: r.value for r in spark.sql(f"SHOW TBLPROPERTIES {T_BRONZE}").collect()}
    assert props["format-version"] == "2"
    spec = spark.sql(f"SELECT * FROM {T_BRONZE}.partitions").schema["partition"].dataType.fieldNames()
    assert spec == ["order_purchase_timestamp_month"]


def test_embeddings_only_for_nonempty_text(spark):
    n_text = spark.table(T_BRONZE).where("has_review_text").count()
    assert spark.table(T_SILVER).count() == n_text
    assert spark.table(T_SILVER).where("review_text IS NULL OR trim(review_text) = ''").count() == 0


def test_embeddings_are_384d_and_unit_norm(spark):
    rows = spark.table(T_SILVER).select("embedding").limit(500).collect()
    E = np.array([r.embedding for r in rows], dtype="float32")
    assert E.shape[1] == EMBEDDING_DIM == 384
    assert np.allclose(np.linalg.norm(E, axis=1), 1.0, atol=1e-3)


def test_every_gold_row_has_valid_cluster_id(spark, C):
    g = spark.table(T_GOLD)
    assert g.where("cluster_id IS NULL").count() == 0
    mn, mx = g.agg({"cluster_id": "min"}).first()[0], g.agg({"cluster_id": "max"}).first()[0]
    assert 0 <= mn and mx < len(C)
    assert g.count() == spark.table(T_SILVER).count()


def test_cluster_id_matches_nearest_centroid(spark, C):
    rows = spark.table(T_GOLD).select("embedding", "cluster_id").limit(300).collect()
    E = np.array([r.embedding for r in rows], dtype="float32")
    assert (np.argmax(E @ C.T, axis=1) == np.array([r.cluster_id for r in rows])).all()


def test_gold_has_sort_order_and_month_partitioning(spark):
    props = {r.key: r.value for r in spark.sql(f"SHOW TBLPROPERTIES {T_GOLD}").collect()}
    assert "cluster_id" in props.get("sort-order", "")
    assert spark.sql(f"SELECT * FROM {T_GOLD}.partitions").schema["partition"].dataType.fieldNames() == ["purchase_ts_month"]


QUERIES = ["package arrived damaged", "excellent quality, very happy", "I want a refund"]


@pytest.mark.parametrize("text", QUERIES)
def test_ivf_with_all_clusters_equals_brute_force(spark, C, text):
    q = encode_texts([text])[0]
    brute = search(spark, q, mode="brute", top_k=10)
    ivf = search(spark, q, mode="ivf", C=C, nprobe=len(C), top_k=10)
    assert [r.review_id for r in ivf.rows] == [r.review_id for r in brute.rows]


def test_ivf_small_nprobe_scans_fewer_or_equal_files(spark, C):
    q = encode_texts(["late delivery"])[0]
    brute = search(spark, q, mode="brute")
    ivf = search(spark, q, mode="ivf", C=C, nprobe=1)
    assert ivf.metrics["resultDataFiles"] <= brute.metrics["resultDataFiles"]


def test_hybrid_query_respects_filters(spark, C):
    q = encode_texts(["delivery took too long"])[0]
    for mode in ("brute", "ivf"):
        r = search(spark, q, mode=mode, C=C, nprobe=5, since="2018-01-01", status="delivered", top_k=10)
        assert r.rows, f"{mode}: no rows returned"
        assert all(row.order_status == "delivered" for row in r.rows)
        assert all(row.purchase_ts.year >= 2018 for row in r.rows)
        if mode == "ivf":
            assert {row.cluster_id for row in r.rows} <= set(r.clusters)
    scores = [row.score for row in r.rows]
    assert scores == sorted(scores, reverse=True)


def test_nearest_clusters_returns_nprobe_distinct(C):
    q = encode_texts(["anything"])[0]
    near = nearest_clusters(q, C, 5)
    assert len(set(near)) == 5
