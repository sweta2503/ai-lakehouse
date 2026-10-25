"""Shared helpers: config loading, SparkSession builder, timing, logging.

Everything configurable lives in `.env` (see `.env.example`); real environment
variables win over `.env` so `SAMPLE_ROWS=5000 make bronze` works.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        # Real env vars win, but treat an empty value as "unset" so .env defaults apply
        if not os.environ.get(key.strip()):
            os.environ[key.strip()] = val.strip()


_load_dotenv(ROOT / ".env")


def env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def env_int(name: str, default: int | None = None) -> int | None:
    v = env(name)
    return int(v) if v is not None else default


# ---- Config -------------------------------------------------------------------
ICEBERG_VERSION = env("ICEBERG_VERSION", "1.9.2")
SPARK_SCALA_SUFFIX = env("SPARK_SCALA_SUFFIX", "3.5_2.12")
S3_ENDPOINT = env("S3_ENDPOINT", "http://localhost:9000")
CATALOG_URI = env("CATALOG_URI", "http://localhost:8181")
AWS_REGION = env("AWS_REGION", "us-east-1")
S3_ACCESS_KEY = env("MINIO_ROOT_USER", "minioadmin")
S3_SECRET_KEY = env("MINIO_ROOT_PASSWORD", "minioadmin")
WAREHOUSE_BUCKET = env("WAREHOUSE_BUCKET", "warehouse")
EMBEDDING_MODEL = env("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = env_int("EMBEDDING_DIM", 384)
EMBED_BATCH_SIZE = env_int("EMBED_BATCH_SIZE", 64)
TRANSLATE_MODEL = env("TRANSLATE_MODEL", "unicamp-dl/translation-pt-en-t5")
IVF_K = env_int("IVF_K", 50)
IVF_SEED = env_int("IVF_SEED", 42)
TARGET_FILE_SIZE_BYTES = env_int("TARGET_FILE_SIZE_BYTES", 262144)
SAMPLE_ROWS = env_int("SAMPLE_ROWS")  # None = all rows
DATA_DIR = ROOT / env("DATA_DIR", "data")
BENCH_DIR = ROOT / "benchmarks"
QDRANT_URL = env("QDRANT_URL", "http://localhost:6333")
CATALOG = "lake"

# Tables (single place so scripts/tests never disagree)
T_BRONZE = f"{CATALOG}.bronze.reviews"
T_SILVER = f"{CATALOG}.silver.review_embeddings"
T_CENTROIDS = f"{CATALOG}.gold.centroids"
T_GOLD = f"{CATALOG}.gold.review_embeddings"


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(message)s",
                        datefmt="%H:%M:%S")
    return logging.getLogger(name)


@contextmanager
def timed(label: str, log: logging.Logger | None = None):
    """Context manager that prints how long a block took. Yields a dict with 'seconds'."""
    out: dict = {}
    t0 = time.perf_counter()
    try:
        yield out
    finally:
        out["seconds"] = time.perf_counter() - t0
        (log.info if log else print)(f"[timing] {label}: {out['seconds']:.2f}s")


# ---- Java / Spark ---------------------------------------------------------------
_BREW_JDK17 = [
    "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home",
    "/usr/local/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home",
    "/usr/lib/jvm/java-17-openjdk-amd64",
    "/usr/lib/jvm/java-17-openjdk-arm64",
]


def _is_java17(home: str) -> bool:
    java = Path(home) / "bin" / "java"
    if not java.exists():
        return False
    out = subprocess.run([str(java), "-version"], capture_output=True, text=True).stderr
    return 'version "17' in out


def ensure_java17() -> str:
    """Point JAVA_HOME at a Java 17 install (Spark 3.5 supports 8/11/17; we standardise on 17)."""
    cur = os.environ.get("JAVA_HOME")
    if cur and _is_java17(cur):
        return cur
    candidates: list[str] = []
    try:
        candidates.append(subprocess.run(["/usr/libexec/java_home", "-v", "17"], capture_output=True,
                                         text=True).stdout.strip())
    except FileNotFoundError:
        pass
    candidates += _BREW_JDK17
    for c in candidates:
        if c and _is_java17(c):
            os.environ["JAVA_HOME"] = c
            os.environ["PATH"] = f"{c}/bin:{os.environ['PATH']}"
            return c
    raise RuntimeError("Java 17 not found. Install it (macOS: `brew install openjdk@17`) or set JAVA_HOME.")


def local_cores() -> int:
    return env_int("SPARK_CORES") or os.cpu_count() or 4


def get_spark(app_name: str = "ai-lakehouse", shuffle_partitions: int | None = None):
    """SparkSession wired to the Iceberg REST catalog `lake` on MinIO.

    Notes on the S3 settings (all via Iceberg's S3FileIO; no hadoop-aws needed):
      * s3.endpoint + path-style: MinIO is not AWS, so there are no virtual-host buckets.
      * client.region: the AWS SDK v2 refuses to build a client without a region, even for MinIO.
      * The REST fixture returns/uses the in-docker endpoint `http://minio:9000`, which the host
        cannot resolve. We pin the endpoint client-side (see README -> Troubleshooting).
    """
    ensure_java17()
    # Make Python workers (pandas_udf) use this same venv interpreter.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    # macOS: forking after torch/Obj-C init can abort workers without this.
    os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Python workers must be able to `import embeddings` (UDFs reference it by module).
    src = str(ROOT / "src")
    if src not in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        os.environ["PYTHONPATH"] = src + os.pathsep + os.environ.get("PYTHONPATH", "")

    from pyspark.sql import SparkSession

    packages = ",".join([
        f"org.apache.iceberg:iceberg-spark-runtime-{SPARK_SCALA_SUFFIX}:{ICEBERG_VERSION}",
        f"org.apache.iceberg:iceberg-aws-bundle:{ICEBERG_VERSION}",
    ])
    c = f"spark.sql.catalog.{CATALOG}"
    cores = local_cores()
    builder = (
        SparkSession.builder.appName(app_name)
        .master(f"local[{cores}]")
        .config("spark.jars.packages", packages)
        .config("spark.jars.ivy", str(ROOT / ".ivy2"))
        .config("spark.driver.memory", env("SPARK_DRIVER_MEMORY", "4g"))
        .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        .config(c, "org.apache.iceberg.spark.SparkCatalog")
        .config(f"{c}.type", "rest")
        .config(f"{c}.uri", CATALOG_URI)
        .config(f"{c}.io-impl", "org.apache.iceberg.aws.s3.S3FileIO")
        .config(f"{c}.s3.endpoint", S3_ENDPOINT)
        .config(f"{c}.s3.path-style-access", "true")
        .config(f"{c}.s3.access-key-id", S3_ACCESS_KEY)
        .config(f"{c}.s3.secret-access-key", S3_SECRET_KEY)
        .config(f"{c}.client.region", AWS_REGION)
        .config("spark.sql.defaultCatalog", CATALOG)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions or env_int("SPARK_SHUFFLE_PARTITIONS", 8)))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def table_exists(spark, name: str) -> bool:
    return spark.catalog.tableExists(name)


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
