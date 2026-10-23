# AI Lakehouse — run `make up && make all`
SHELL := /bin/bash
PY := .venv/bin/python
UV := $(shell command -v uv 2>/dev/null)

# Query used by `make query` (override: make query Q="package arrived broken")
Q ?= package arrived damaged and the seller never answered
SINCE ?= 2018-01-01
STATUS ?= delivered
NPROBE ?= 3

.PHONY: help setup up down data bronze drift embed-bench silver translate index gold query bench inspect results ui notebooks notebooks-test bonus test smoke all clean

help:
	@grep -E '^[a-z-]+:' Makefile | grep -v '^\.PHONY' | cut -d: -f1 | tr '\n' ' '; echo

setup:  ## venv + pinned deps + .env
	@test -f .env || cp .env.example .env
	@test -d .venv || python3.11 -m venv .venv
ifdef UV
	$(UV) pip install --python $(PY) -r requirements.txt
else
	$(PY) -m pip install -r requirements.txt
endif

up:  ## MinIO + bucket + Iceberg REST catalog
	docker compose up -d
	scripts/wait_for_services.sh
	$(PY) src/01_check_connection.py

down:
	docker compose --profile bonus down

data:  ## Olist via kaggle if possible, else synthetic
	scripts/download_data.sh

bronze: ; $(PY) src/02_bronze_ingest.py
drift: ; $(PY) src/03_schema_drift.py
embed-bench: ; $(PY) src/04_embed_naive_vs_batched.py
silver: ; $(PY) src/05_silver_embeddings.py
translate: ; $(PY) src/05b_translate_reviews.py   ## OPTIONAL: English display text (free local model, slow: ~tens of minutes on a laptop)
index: ; $(PY) src/06_build_ivf_index.py
gold: ; $(PY) src/07_gold_write.py
query: ; $(PY) src/08_hybrid_query.py --query "$(Q)" --since $(SINCE) --status $(STATUS) --nprobe $(NPROBE) --mode ivf
bench: ; $(PY) src/09_benchmark.py
inspect: ; $(PY) src/10_inspect_iceberg_metadata.py
ui:  ## search UI at http://localhost:8765 (needs `make up` + a built gold table)
	$(PY) src/ui_server.py

notebooks:  ## register the .venv Jupyter kernel for the VS Code demo notebooks (see docs/vscode-demo.md)
ifdef UV
	$(UV) pip install --python $(PY) -r requirements.txt
else
	$(PY) -m pip install -r requirements.txt
endif
	$(PY) -m ipykernel install --user --name ai-lakehouse --display-name "AI Lakehouse (.venv)"
	@echo "Open notebooks/00_start_services.ipynb in VS Code and pick the kernel 'AI Lakehouse (.venv)'"

notebooks-test:  ## execute every notebook top to bottom (outputs go to /tmp, repo notebooks stay clean). Use SAMPLE_ROWS=5000 for a fast pass
	@mkdir -p /tmp/ai-lakehouse-nb-out
	@for nb in notebooks/[0-9]*.ipynb; do \
	  echo "== $$nb"; \
	  .venv/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=ai-lakehouse \
	    --ExecutePreprocessor.timeout=3600 --output-dir /tmp/ai-lakehouse-nb-out $$nb || exit 1; \
	done

results: ; $(PY) scripts/render_results.py   ## copy benchmarks/ into README

bonus:  ## optional: Qdrant comparison
ifdef UV
	$(UV) pip install --python $(PY) -r requirements-bonus.txt
else
	$(PY) -m pip install -r requirements-bonus.txt
endif
	docker compose --profile bonus up -d qdrant
	$(PY) src/bonus_qdrant_compare.py
	$(PY) scripts/render_results.py

test: ; $(PY) -m pytest -v tests

# Fast end-to-end check on 5,000 rows (skips the slow embedding benchmark), then the tests.
# WARNING: rebuilds bronze/silver/gold, replacing a full-size lakehouse.
smoke:
	SAMPLE_ROWS=5000 $(MAKE) data bronze drift silver index gold test

# Core flow. (bronze already ingests; drift appends a second batch afterwards.)
all: data bronze drift embed-bench silver index gold query bench inspect results

clean:  ## remove local state (keeps .venv and .ivy2 jar cache)
	docker compose --profile bonus down -v
	rm -rf spark-warehouse metastore_db derby.log benchmarks/*.json benchmarks/*.md data/synthetic data/landing
