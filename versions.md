# Versions

Single source of truth for the Iceberg version is `ICEBERG_VERSION` in `.env` / `.env.example`;
`docker-compose.yml` (REST fixture image tag) and `src/common.py` (jar coordinates) both read it.

| Component | Version | Notes |
|---|---|---|
| Apache Iceberg | 1.9.2 | `iceberg-spark-runtime-3.5_2.12` + `iceberg-aws-bundle`, loaded via `spark.jars.packages` |
| Iceberg REST catalog | `apache/iceberg-rest-fixture:1.9.2` | same version as the client jars |
| Spark / PySpark | 3.5.6 | Scala 2.12 build |
| Java | 17 (OpenJDK 17.0.x) | `brew install openjdk@17`; `common.py` finds it without touching your PATH |
| Python | 3.11 | |
| MinIO | `bitnamilegacy/minio:2025.7.23` | official `minio/minio` images are no longer pullable |
| MinIO client | `bitnamilegacy/minio-client:2025.4.16` | one-shot bucket creation |
| Qdrant (bonus) | `qdrant/qdrant:v1.14.1` | |
| sentence-transformers | 3.4.1 | model: `paraphrase-multilingual-MiniLM-L12-v2` (384 dims) |
| torch / transformers | 2.6.0 / 4.49.0 | CPU only |
| numpy / pandas / pyarrow | 1.26.4 / 2.2.3 / 17.0.0 | numpy<2 for PySpark 3.5 compatibility |
