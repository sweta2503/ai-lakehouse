"""BONUS (optional): load the same gold vectors into Qdrant and compare against our Iceberg IVF index.

Needs: `docker compose --profile bonus up -d qdrant` and `pip install -r requirements-bonus.txt` (make bonus).
Writes benchmarks/qdrant_comparison.{md,json}. The core pipeline does not depend on this.

What this is (and isn't): Qdrant is a purpose-built vector database with an HNSW index; we are comparing it
with a *laptop-scale, application-level IVF over Iceberg files queried through Spark*. Expect Qdrant to win on
latency by a wide margin. The interesting parts are recall, and what you give up/gain architecturally (one
copy of the data in the lakehouse, SQL governance and time travel vs. a second system to run and sync).
"""
import importlib
import json
import statistics
import sys
import time
from datetime import datetime, timezone

import numpy as np

from common import BENCH_DIR, QDRANT_URL, T_GOLD, get_logger, get_spark, section
from embeddings import encode_texts
from search import load_centroids, search

log = get_logger("qdrant")
COLLECTION = "reviews"
bench = importlib.import_module("09_benchmark")  # reuse the exact same 50 queries


def iso(d):
    """'2018-01-01' -> '2018-01-01T00:00:00Z' for Qdrant datetime filters."""
    return datetime.fromisoformat(d).replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if d else None


def main() -> int:
    try:
        from qdrant_client import QdrantClient, models
    except ImportError:
        log.error("qdrant-client missing: run `make bonus` (installs requirements-bonus.txt).")
        return 1

    section("BONUS: Qdrant vs Iceberg IVF")
    client = QdrantClient(url=QDRANT_URL, timeout=60)
    try:
        client.get_collections()
    except Exception as e:  # noqa: BLE001
        log.error("Qdrant not reachable at %s (%s). Start it: docker compose --profile bonus up -d qdrant", QDRANT_URL, e)
        return 1

    spark = get_spark("bonus_qdrant")
    C = load_centroids(spark)
    gold = spark.table(T_GOLD).select("review_id", "purchase_ts", "order_status", "embedding").toPandas()
    n = len(gold)
    log.info("loaded %d vectors from %s", n, T_GOLD)

    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(COLLECTION, vectors_config=models.VectorParams(size=len(gold.embedding[0]), distance=models.Distance.COSINE))
    client.create_payload_index(COLLECTION, "purchase_ts", models.PayloadSchemaType.DATETIME)
    client.create_payload_index(COLLECTION, "order_status", models.PayloadSchemaType.KEYWORD)
    t0 = time.perf_counter()
    B = 500
    for i in range(0, n, B):
        chunk = gold.iloc[i:i + B]
        client.upsert(COLLECTION, points=[
            models.PointStruct(id=int(i + j), vector=np.asarray(r.embedding, dtype="float32").tolist(),
                               payload={"review_id": r.review_id, "order_status": r.order_status,
                                        "purchase_ts": r.purchase_ts.strftime("%Y-%m-%dT%H:%M:%SZ")})
            for j, r in enumerate(chunk.itertuples())], wait=True)
    log.info("[timing] upserted %d points in %.1fs", n, time.perf_counter() - t0)
    while client.get_collection(COLLECTION).status != models.CollectionStatus.GREEN:
        time.sleep(0.5)  # wait for the HNSW index build

    queries = bench.build_queries()
    Q = encode_texts([t for t, _ in queries])

    def qfilter(f):
        must = []
        if f.get("since") or f.get("until"):
            must.append(models.FieldCondition(key="purchase_ts", range=models.DatetimeRange(
                gte=iso(f.get("since")), lt=iso(f.get("until")))))
        if f.get("status"):
            must.append(models.FieldCondition(key="order_status", match=models.MatchValue(value=f["status"])))
        return models.Filter(must=must) if must else None

    # Ground truth = our exact brute force through Spark/Iceberg (same filters, same top-10).
    truth = [search(spark, q, mode="brute", top_k=10, **f) for (t, f), q in zip(queries, Q)]
    truth_ids = [[r.review_id for r in x.rows] for x in truth]
    truth_floor = [x.rows[-1].score if x.rows else -1 for x in truth]  # 10th best score

    def run_qdrant(exact):
        lat, rec_id, rec_tie = [], [], []
        for (t, f), q, gt, floor in zip(queries, Q, truth_ids, truth_floor):
            t1 = time.perf_counter()
            res = client.query_points(COLLECTION, query=q.tolist(), query_filter=qfilter(f), limit=10,
                                      search_params=models.SearchParams(exact=exact), with_payload=True).points
            lat.append((time.perf_counter() - t1) * 1000)
            ids = [p.payload["review_id"] for p in res]
            rec_id.append(len(set(ids) & set(gt)) / len(gt) if gt else 1.0)
            # tie-aware: a result is as good as the truth if its score reaches the truth's 10th-best score
            good = sum(1 for p in res if p.score >= floor - 1e-4)
            rec_tie.append(min(good, len(gt)) / len(gt) if gt else 1.0)
        return {"mean_ms": round(statistics.mean(lat), 2), "p95_ms": round(bench.pct(lat, 95), 2),
                "recall_at_10_by_id": round(statistics.mean(rec_id), 4),
                "recall_at_10_tie_aware": round(statistics.mean(rec_tie), 4)}

    for _ in range(3):  # warm-up
        client.query_points(COLLECTION, query=Q[0].tolist(), limit=10)
    q_hnsw, q_exact = run_qdrant(False), run_qdrant(True)

    ivf = json.loads((BENCH_DIR / "ivf_benchmark.json").read_text())  # produced by 09_benchmark.py
    out = {"points": n, "queries": len(queries), "qdrant_hnsw": q_hnsw, "qdrant_exact": q_exact, "iceberg": ivf["results"]}
    (BENCH_DIR / "qdrant_comparison.json").write_text(json.dumps(out, indent=2))

    lines = ["| System | mean ms | p95 ms | recall@10 (by id) | recall@10 (tie-aware) |", "|---|---:|---:|---:|---:|",
             f"| Qdrant HNSW (default) | {q_hnsw['mean_ms']} | {q_hnsw['p95_ms']} | {q_hnsw['recall_at_10_by_id']:.3f} | {q_hnsw['recall_at_10_tie_aware']:.3f} |",
             f"| Qdrant exact | {q_exact['mean_ms']} | {q_exact['p95_ms']} | {q_exact['recall_at_10_by_id']:.3f} | {q_exact['recall_at_10_tie_aware']:.3f} |"]
    for r in ivf["results"]:
        lines.append(f"| Iceberg {r['config']} (Spark) | {r['mean_ms']} | {r['p95_ms']} | {r['recall_at_10']:.3f} | {r['recall_at_10_tie_aware']:.3f} |")
    md = f"""# Qdrant vs Iceberg IVF

{n} vectors, {len(queries)} queries, same SQL-style filters (date range + status) applied in both systems, top-10.
Ground truth: exact brute force over the Iceberg gold table. Iceberg rows are copied from `ivf_benchmark.json`
(same run as `09_benchmark.py`); Qdrant ran locally in Docker.

{chr(10).join(lines)}

Notes
- Qdrant latency is a client-side HTTP round trip to a purpose-built engine; the Iceberg numbers include a full
  Spark job per query. Latencies are not comparable as "which index is better", only as "what you pay for
  keeping vectors inside the lakehouse and querying them with Spark".
- "By id" recall counts a result only if it is the same review as the ground truth. The data has many identical
  texts (identical vectors); Qdrant breaks such ties arbitrarily, so the tie-aware column counts a result as correct
  when its score reaches the ground truth's 10th-best score.
"""
    (BENCH_DIR / "qdrant_comparison.md").write_text(md)
    print("\n" + md)
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
