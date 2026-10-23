"""Shared helpers for the demo notebooks. Notebooks add NO pipeline logic: they import from src/ or run the
numbered scripts (whose filenames start with a digit, so they cannot be imported normally) via runpy/importlib.

First cell of every notebook:   from _setup import *

* puts src/ on sys.path and imports `common`, which loads .env exactly as the scripts do
  (real environment variables win; SAMPLE_ROWS etc. work the same way)
* run_script("02_bronze_ingest.py")  -> runs a numbered script in-process, keeps its log in a collapsed box
* get_spark() / stop_spark()         -> one Spark session per notebook; scripts manage their own and stop it
"""
import contextlib
import importlib.util
import io
import json
import logging
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.chdir(ROOT)  # scripts and relative paths assume the repo root

import common  # noqa: E402  (loads .env like the scripts do)
import pandas as pd  # noqa: E402
from IPython.display import HTML, Markdown, display  # noqa: E402

pd.set_option("display.max_colwidth", 100)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

T_BRONZE, T_SILVER, T_CENTROIDS, T_GOLD, T_TRANS = (common.T_BRONZE, common.T_SILVER, common.T_CENTROIDS,
                                                    common.T_GOLD, common.T_TRANS)
BENCH_DIR = common.BENCH_DIR
MINIO_CONSOLE = "http://localhost:9001"


def set_sample_rows(n):
    """Use only n rows (e.g. 5000) for the rest of this kernel; None = all rows. Same effect as SAMPLE_ROWS=n."""
    import data_source
    common.SAMPLE_ROWS = data_source.SAMPLE_ROWS = n
    if n:
        os.environ["SAMPLE_ROWS"] = str(n)
    else:
        os.environ.pop("SAMPLE_ROWS", None)
    print(f"SAMPLE_ROWS = {n}")


def active_spark():
    from pyspark.sql import SparkSession
    return SparkSession.getActiveSession()


@contextlib.contextmanager
def _quiet_jvm():
    """Hide the JVM's startup banner (Ivy resolution, hostname warnings) that would otherwise show as red stderr in
    every notebook. The JVM is launched once per kernel, so this only matters around the first Spark start."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved1, saved2, devnull = os.dup(1), os.dup(2), os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)  # Ivy prints one line to stdout, the rest to stderr
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(saved1, 1)
        os.dup2(saved2, 2)
        for fd in (saved1, saved2, devnull):
            os.close(fd)


def get_spark(app="notebook"):
    """The notebook's Spark session (re-created if a script stopped it)."""
    s = active_spark()
    if s is not None:
        return s
    with _quiet_jvm():
        return common.get_spark(app)


def stop_spark(quiet=False):
    s = active_spark()
    if s is not None:
        s.stop()
        if not quiet:
            print("Spark session stopped (the Spark UI on :4040 disappears with it).")


def run_script(name, *argv, show_log=True):
    """Run src/<name> exactly as `python src/<name> <argv>` would, in-process. Raises if it fails.

    stdout and log lines are captured into a collapsed <details> box so the notebook shows tables, not logs.
    Raises if the script fails (the log is shown).
    """
    stop_spark(quiet=True)  # scripts create and stop their own session
    path = SRC / name
    buf = io.StringIO()
    root = logging.getLogger()
    saved = root.handlers[:]
    h = logging.StreamHandler(buf)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)-5s %(message)s", "%H:%M:%S"))
    root.handlers = [h]
    old_argv, code, t0 = sys.argv, 0, time.perf_counter()
    sys.argv = [str(path), *map(str, argv)]
    try:
        with contextlib.redirect_stdout(buf), _quiet_jvm():
            try:
                runpy.run_path(str(path), run_name="__main__")
            except SystemExit as e:
                code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    finally:
        sys.argv = old_argv
        root.handlers = saved
    secs = time.perf_counter() - t0
    log = buf.getvalue()
    status = "OK" if code == 0 else f"FAILED (exit {code})"
    if show_log or code != 0:
        display(HTML(f"<b>{name}</b>: {status} in {secs:.1f}s &nbsp; <details><summary>show script log</summary>"
                     f"<pre style='max-height:420px;overflow:auto'>{log.replace('<', '&lt;')}</pre></details>"))
    if code != 0:
        raise RuntimeError(f"{name} failed with exit code {code}; see log above")


def load_script(name):
    """Import a numbered script as a module (so we can call its functions, not copy them). Its main() is not run."""
    spec = importlib.util.spec_from_file_location("nb_" + Path(name).stem, SRC / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sql(spark, query):
    """Run Spark SQL and return a pandas DataFrame (use for small result sets)."""
    return spark.sql(query).toPandas()


def side_by_side(dfs, titles):
    """Show tables next to each other when the pane is wide, stacked when it is narrow (e.g. a VS Code notebook).
    Long text columns (7th onward: the review and its English translation) get a readable width."""
    css = ("<style>.sbs{display:flex;flex-wrap:wrap;gap:18px}.sbs>div{flex:1 1 560px;min-width:0;overflow-x:auto}"
           ".sbs table{border-collapse:collapse}.sbs td,.sbs th{white-space:nowrap;padding:3px 8px;text-align:left;vertical-align:top}"
           ".sbs td:nth-child(n+7),.sbs th:nth-child(n+7){white-space:normal;min-width:260px;max-width:420px}</style>")
    cells = "".join(f"<div><h4 style='margin:4px 0'>{t}</h4>{d.to_html(index=False, escape=True)}</div>"
                    for d, t in zip(dfs, titles))
    display(HTML(f"{css}<div class='sbs'>{cells}</div>"))


def minio():
    import boto3
    from botocore.config import Config
    return boto3.client("s3", endpoint_url=common.S3_ENDPOINT, aws_access_key_id=common.S3_ACCESS_KEY,
                        aws_secret_access_key=common.S3_SECRET_KEY, region_name=common.AWS_REGION,
                        config=Config(s3={"addressing_style": "path"}))


def minio_objects(prefix=""):
    """Objects in the warehouse bucket under a prefix, as a DataFrame."""
    s3, rows, token = minio(), [], None
    while True:
        kw = dict(Bucket=common.WAREHOUSE_BUCKET, Prefix=prefix, **({"ContinuationToken": token} if token else {}))
        r = s3.list_objects_v2(**kw)
        rows += [{"key": o["Key"], "bytes": o["Size"]} for o in r.get("Contents", [])]
        token = r.get("NextContinuationToken")
        if not token:
            return pd.DataFrame(rows, columns=["key", "bytes"])


def minio_summary():
    """Objects and size per table prefix (e.g. gold/review_embeddings) in the warehouse bucket."""
    df = minio_objects()
    if df.empty:
        return pd.DataFrame(columns=["table", "objects", "MB"])
    df["table"] = df.key.str.split("/").str[:2].str.join("/")
    return (df.groupby("table").agg(objects=("key", "count"), MB=("bytes", lambda b: round(b.sum() / 1e6, 2)))
            .reset_index())
