"""M5 (2/2): benchmark brute force vs IVF (nprobe in 1,3,5,10) over ~50 diverse hybrid queries.

Reports mean / p95 latency, Iceberg data files scanned, rows read from the scan, and recall@10 against the
brute-force ground truth (same filters, exact scoring). Writes benchmarks/ivf_benchmark.{json,md}.
"""
import json
import statistics
import sys

import numpy as np

from common import BENCH_DIR, IVF_K, T_GOLD, get_logger, get_spark, section
from embeddings import encode_texts
from search import load_centroids, search

log = get_logger("bench")
TOP_K = 10
NPROBES = [1, 3, 5, 10]

# 25 phrasings (English paraphrases, a few Portuguese, a few off-theme) x 2 filter sets = 50 queries.
TEXTS = [
    "package arrived damaged", "the box was crushed and the item was broken", "delivery took forever",
    "my order is late and nobody tells me where it is", "excellent product, very happy", "great build quality",
    "I received the wrong product", "they sent a different color than I ordered", "I want a refund",
    "money back please, the return was approved long ago", "super fast shipping", "arrived earlier than expected",
    "the seller never answered my messages", "customer support was really helpful", "cheap and poor quality, fell apart",
    "good value for money", "one item was missing from the order", "chegou atrasado e sem rastreio",
    "produto veio quebrado", "ótima qualidade, recomendo", "wanted to cancel the order and get reimbursed",
    "the item does not look like the photos", "perfect gift, my kid loved it", "packaging was wet and torn",
    "would buy again",
]
FILTER_SETS = [
    dict(since="2018-01-01", status="delivered"),
    dict(since="2017-03-01", until="2017-10-01", status="delivered"),
    dict(since=None, status=None),
    dict(since="2018-04-01", status=None),
]


def build_queries():
    qs = []
    for i, t in enumerate(TEXTS):
        qs.append((t, FILTER_SETS[i % 4]))
        qs.append((t, FILTER_SETS[(i + 1 + i // 4) % 4]))
    return qs


def pct(vals, p):
    s = sorted(vals)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def main() -> int:
    section("M5: benchmark brute vs IVF")
    spark = get_spark("09_benchmark")
    C = load_centroids(spark)
    total_files = spark.sql(f"SELECT count(*) n FROM {T_GOLD}.files").first().n
    total_rows = spark.table(T_GOLD).count()
    queries = build_queries()
    log.info("%d queries | gold: %d rows, %d data files, k=%d clusters", len(queries), total_rows, total_files, len(C))
    Q = encode_texts([t for t, _ in queries])

    def run_config(mode, nprobe=None):
        res = []
        for (t, f), q in zip(queries, Q):
            res.append(search(spark, q, mode=mode, C=C, nprobe=nprobe or 0, top_k=TOP_K, **f))
        return res

    # warm up (JIT, REST metadata, codegen) on a few queries in both modes
    for (t, f), q in list(zip(queries, Q))[:4]:
        search(spark, q, mode="brute", top_k=TOP_K, **f)
        search(spark, q, mode="ivf", C=C, nprobe=3, top_k=TOP_K, **f)

    log.info("brute force (ground truth)...")
    truth = run_config("brute")
    truth_ids = [[r.review_id for r in x.rows] for x in truth]
    truth_floor = [x.rows[-1].score if x.rows else float("-inf") for x in truth]  # 10th-best exact score

    def summarize(name, res, nprobe):
        lat = [x.seconds * 1000 for x in res]
        recalls, recalls_tie = [], []
        for x, gt, floor in zip(res, truth_ids, truth_floor):
            got = {r.review_id for r in x.rows}
            recalls.append(len(got & set(gt)) / len(gt) if gt else 1.0)
            # tie-aware: a result counts if it scores at least as high as the exact 10th-best (identical
            # vectors are interchangeable, so which of them is returned is arbitrary)
            good = sum(1 for r in x.rows if r.score >= floor - 1e-4)
            recalls_tie.append(min(good, len(gt)) / len(gt) if gt else 1.0)
        files = [x.metrics.get("resultDataFiles", 0) for x in res]
        rows = [x.metrics.get("numOutputRows", 0) for x in res]
        return {"config": name, "nprobe": nprobe, "mean_ms": round(statistics.mean(lat), 1), "p95_ms": round(pct(lat, 95), 1),
                "mean_files_scanned": round(statistics.mean(files), 1), "mean_rows_scanned": round(statistics.mean(rows), 0),
                "recall_at_10": round(statistics.mean(recalls), 4), "recall_at_10_tie_aware": round(statistics.mean(recalls_tie), 4),
                "min_recall": round(min(recalls), 3)}

    rows_out = [summarize("brute force", truth, None)]
    for npb in NPROBES:
        log.info("ivf nprobe=%d...", npb)
        rows_out.append(summarize(f"IVF nprobe={npb}", run_config("ivf", npb), npb))

    base = rows_out[0]
    for r in rows_out:
        r["latency_speedup_vs_brute"] = round(base["mean_ms"] / r["mean_ms"], 2)
        r["files_scanned_pct_of_brute"] = round(100 * r["mean_files_scanned"] / base["mean_files_scanned"], 1)

    out = {"queries": len(queries), "top_k": TOP_K, "gold_rows": total_rows, "gold_data_files": total_files,
           "ivf_k": len(C), "results": rows_out}
    BENCH_DIR.mkdir(exist_ok=True)
    (BENCH_DIR / "ivf_benchmark.json").write_text(json.dumps(out, indent=2))

    lines = ["| Config | mean ms | p95 ms | speedup | data files scanned | % of brute files | rows read | recall@10 | recall@10 tie-aware |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows_out:
        lines.append(f"| {r['config']} | {r['mean_ms']} | {r['p95_ms']} | {r['latency_speedup_vs_brute']}x | "
                     f"{r['mean_files_scanned']} | {r['files_scanned_pct_of_brute']}% | {int(r['mean_rows_scanned'])} | "
                     f"{r['recall_at_10']:.3f} | {r['recall_at_10_tie_aware']:.3f} |")
    md = f"""# IVF benchmark (Iceberg + Spark)

{len(queries)} hybrid queries (SQL filters on date/status + vector similarity), top-{TOP_K}. Gold table: {total_rows} rows in
{total_files} data files, k={len(C)} clusters. Recall@10 is measured against exact brute force with the same filters.
"Data files scanned" and "rows read" come from Iceberg's scan metrics (they already include date-partition pruning,
which brute force benefits from too).

{chr(10).join(lines)}

**Scale caveat:** this is a laptop-sized table. Every query pays a fixed Spark job + Iceberg planning cost
regardless of how many vectors it scores, so on a table this small that overhead can hide (or even outweigh) the
savings from scanning fewer files; the files-scanned and rows-read columns show what IVF skips independent of that
overhead. Latency wins from IVF should grow with table size, but this run does not measure that. The reviews
(especially synthetic ones) are also much more clusterable than messy real-world text, so recall here may be
optimistic. "recall@10 tie-aware" counts a result as correct if it scores as high as the exact 10th-best result,
because many reviews share identical text and therefore identical vectors.
"""
    (BENCH_DIR / "ivf_benchmark.md").write_text(md)
    print("\n" + md)
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
