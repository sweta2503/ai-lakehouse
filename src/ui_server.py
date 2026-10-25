"""Search UI for the AI Lakehouse. One long-lived Spark session behind a tiny stdlib HTTP server.

    make ui      ->  http://localhost:8765      (Spark UI: http://localhost:4040, MinIO: http://localhost:9001)

Endpoints: GET /  (page) | GET /api/meta | POST /api/search  {query, mode, nprobe, since, until, status, top_k, compare}
Spark calls are serialised with a lock; this is a demo server, not a production one.
"""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from common import T_GOLD, get_logger, get_spark
from embeddings import encode_texts
from search import load_centroids, search

log = get_logger("ui")
PAGE = (Path(__file__).parent / "ui" / "index.html").read_bytes()
LOCK = threading.Lock()
STATE: dict = {}


def init():
    spark = get_spark("search_ui")
    C = load_centroids(spark)
    r = spark.sql(f"SELECT min(purchase_ts) lo, max(purchase_ts) hi, count(*) n FROM {T_GOLD}").first()
    STATE.update(
        spark=spark, C=C,
        statuses=[x.order_status for x in spark.sql(f"SELECT DISTINCT order_status FROM {T_GOLD} ORDER BY 1").collect()],
        total_files=spark.sql(f"SELECT count(*) n FROM {T_GOLD}.files").first().n,
        lo=r.lo.strftime("%Y-%m-%d"), hi=r.hi.strftime("%Y-%m-%d"), rows=r.n,
        spark_ui=f"http://localhost:{spark.sparkContext.uiWebUrl.rsplit(':', 1)[1]}")
    log.info("warming up (model + first query)...")
    q = encode_texts(["warm up"])[0]
    search(spark, q, mode="ivf", C=C, nprobe=3)
    log.info("ready")


def pack(res, total_files):
    m = res.metrics
    return {
        "mode": res.mode, "clusters": res.clusters, "latency_ms": round(res.seconds * 1000),
        "files_scanned": m.get("resultDataFiles"), "files_total": total_files,
        "rows_read": m.get("numOutputRows"),
        "rows": [{"review_id": r.review_id, "score": round(r.score, 4), "cluster": r.cluster_id,
                  "date": r.purchase_ts.strftime("%Y-%m-%d"), "status": r.order_status,
                  "stars": r.review_score, "text": r.review_text} for r in res.rows],
    }


def do_search(p):
    spark, C = STATE["spark"], STATE["C"]
    text = (p.get("query") or "").strip()
    if not text:
        raise ValueError("empty query")
    kw = dict(since=p.get("since") or None, until=p.get("until") or None, status=p.get("status") or None,
              top_k=int(p.get("top_k") or 10))
    nprobe = max(1, min(int(p.get("nprobe") or 3), len(C)))
    with LOCK:
        q = encode_texts([text])[0]
        sims = C @ q
        out = {"cluster_sims": [round(float(s), 3) for s in sims]}
        if p.get("compare"):
            b = search(spark, q, mode="brute", **kw)
            i = search(spark, q, mode="ivf", C=C, nprobe=nprobe, **kw)
            truth = [(r.review_id, r.order_id) for r in b.rows]
            got = {(r.review_id, r.order_id) for r in i.rows}
            floor = b.rows[-1].score if b.rows else float("-inf")
            tie = sum(1 for r in i.rows if r.score >= floor - 1e-4)
            n = len(truth) or 1
            out.update(brute=pack(b, STATE["total_files"]), ivf=pack(i, STATE["total_files"]),
                       recall=round(len(got & set(truth)) / n, 3), recall_tie=round(min(tie, n) / n, 3))
        else:
            r = search(spark, q, mode=p.get("mode", "ivf"), C=C, nprobe=nprobe, **kw)
            out["result"] = pack(r, STATE["total_files"])
    return out


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/api/meta":
            keys = ("statuses", "total_files", "lo", "hi", "rows", "spark_ui")
            self._send(200, json.dumps({**{k: STATE[k] for k in keys}, "k": len(STATE["C"])}).encode())
        else:
            self._send(404, b'{"error":"not found"}')

    def do_POST(self):
        if self.path != "/api/search":
            return self._send(404, b'{"error":"not found"}')
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            self._send(200, json.dumps(do_search(body)).encode())
        except Exception as e:  # noqa: BLE001
            log.exception("search failed")
            self._send(400, json.dumps({"error": str(e)}).encode())

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    init()
    port = int(os.environ.get("UI_PORT", 8765))
    print(f"\n  Search UI : http://localhost:{port}\n  Spark UI  : {STATE['spark_ui']}\n  MinIO     : http://localhost:9001\n")
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
