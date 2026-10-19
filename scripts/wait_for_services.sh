#!/usr/bin/env bash
# Block until MinIO and the Iceberg REST catalog answer on the host ports.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a
S3="${S3_ENDPOINT:-http://localhost:9000}"
CAT="${CATALOG_URI:-http://localhost:8181}"
TIMEOUT="${WAIT_TIMEOUT:-120}"

wait_for() {  # name url
  local name="$1" url="$2" waited=0
  until curl -fs -o /dev/null "$url"; do
    if [ "$waited" -ge "$TIMEOUT" ]; then
      echo "ERROR: $name did not become ready at $url within ${TIMEOUT}s" >&2
      echo "       try: docker compose logs" >&2
      exit 1
    fi
    sleep 2; waited=$((waited + 2))
  done
  echo "ok: $name ($url)"
}

wait_for "MinIO" "$S3/minio/health/live"
wait_for "Iceberg REST catalog" "$CAT/v1/config"
