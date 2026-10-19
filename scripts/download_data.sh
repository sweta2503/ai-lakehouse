#!/usr/bin/env bash
# Get the Olist Brazilian E-Commerce CSVs into data/ — or fall back to synthetic data.
#
# Kaggle needs a login, so this only works if the `kaggle` CLI is installed AND credentials exist
# (~/.kaggle/kaggle.json or KAGGLE_USERNAME + KAGGLE_KEY). Otherwise we print manual instructions
# and generate the synthetic dataset so the rest of the pipeline still runs.
set -uo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a
DATA_DIR="${DATA_DIR:-data}"
PY=".venv/bin/python"; [ -x "$PY" ] || PY="python3"
mkdir -p "$DATA_DIR"

have_olist() { [ -f "$DATA_DIR/olist_orders_dataset.csv" ] && [ -f "$DATA_DIR/olist_order_reviews_dataset.csv" ]; }

if have_olist; then
  echo "Olist CSVs already present in $DATA_DIR/ — nothing to do."; exit 0
fi

have_creds() { [ -f "$HOME/.kaggle/kaggle.json" ] || { [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; }; }
KAGGLE_BIN="$(command -v kaggle || ls .venv/bin/kaggle 2>/dev/null || true)"

if [ -n "$KAGGLE_BIN" ] && have_creds; then
  echo "Downloading Olist dataset with the kaggle CLI..."
  if "$KAGGLE_BIN" datasets download -d olistbr/brazilian-ecommerce -p "$DATA_DIR" --unzip && have_olist; then
    echo "Done: Olist CSVs are in $DATA_DIR/"; exit 0
  fi
  echo "kaggle download failed; falling back to synthetic data." >&2
else
  cat <<MSG
------------------------------------------------------------------------------
Olist CSVs not found and the Kaggle CLI/credentials are not available.

To use the REAL dataset:
  1. Sign in at https://www.kaggle.com and open
     https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
  2. Click Download, unzip, and copy these two files into ./$DATA_DIR/ :
       olist_orders_dataset.csv
       olist_order_reviews_dataset.csv
  Or: pip install kaggle, put your API token in ~/.kaggle/kaggle.json, re-run 'make data'.

Meanwhile, generating a SYNTHETIC dataset (same columns, English reviews) so you can proceed.
------------------------------------------------------------------------------
MSG
fi

"$PY" scripts/generate_synthetic_data.py --out "$DATA_DIR/synthetic"
