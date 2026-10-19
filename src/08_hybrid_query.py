"""M5 (1/2): hybrid query = SQL filters (date, status) + vector similarity, optionally restricted to the nearest
IVF clusters.

Examples
  python src/08_hybrid_query.py --query "package arrived damaged" --since 2018-01-01 --status delivered --mode ivf --nprobe 3
  python src/08_hybrid_query.py --query "chegou atrasado" --mode brute          # Portuguese works too
"""
import argparse
import sys
import time

from common import T_GOLD, get_logger, get_spark, section, timed
from embeddings import encode_texts
from search import explain_text, load_centroids, nearest_clusters, search

log = get_logger("query")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--since", help="purchase date >= (YYYY-MM-DD)")
    ap.add_argument("--until", help="purchase date < (YYYY-MM-DD)")
    ap.add_argument("--status", help="order_status = ... (e.g. delivered)")
    ap.add_argument("--nprobe", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--mode", choices=["brute", "ivf"], default="ivf")
    ap.add_argument("--no-explain", action="store_true")
    a = ap.parse_args()

    section(f"M5: hybrid query  mode={a.mode}")
    spark = get_spark("08_hybrid_query")
    with timed("embed the query locally", log):
        q = encode_texts([a.query])[0]

    C = load_centroids(spark)
    if a.mode == "ivf":
        sims = C @ q
        near = nearest_clusters(q, C, a.nprobe)
        print(f"nearest {a.nprobe} of {len(C)} clusters: " + ", ".join(f"{c} (cos={sims[c]:.3f})" for c in near))

    search(spark, q, mode=a.mode, C=C, nprobe=a.nprobe, since=a.since, until=a.until, status=a.status, top_k=a.top_k)  # warm-up (JIT, REST metadata cache)
    r = search(spark, q, mode=a.mode, C=C, nprobe=a.nprobe, since=a.since, until=a.until, status=a.status, top_k=a.top_k)

    print(f"\nfilters: since={a.since} until={a.until} status={a.status}" +
          (f" cluster_id IN {r.clusters}" if r.clusters else "  (no cluster restriction)"))
    print(f"\nTop {a.top_k} for: \"{a.query}\"")
    for i, row in enumerate(r.rows, 1):
        print(f"{i:>2}. score={row.score:.3f} cluster={row.cluster_id:<2} {row.purchase_ts:%Y-%m-%d} "
              f"{row.order_status:<9} stars={row.review_score}  {row.review_text[:90]}")

    total_files = spark.sql(f"SELECT count(*) n FROM {T_GOLD}.files").first().n
    m = r.metrics
    print(f"\nIceberg scan metrics: scanned {m.get('resultDataFiles')} of {total_files} data files "
          f"(manifest-level skipped: {m.get('skippedDataFiles')}), rows read from scan: {m.get('numOutputRows')}, "
          f"scan planning: {m.get('totalPlanningDuration')} ms")
    print(f"query latency (after warm-up): {r.seconds * 1000:.0f} ms")

    if not a.no_explain:
        txt = explain_text(r.df)
        scan = [ln for ln in txt.splitlines() if "BatchScan" in ln or "filters=" in ln]
        print("\nEXPLAIN (scan node; note the pushed-down predicates, incl. cluster_id IN (...)):")
        print("\n".join(scan))
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
