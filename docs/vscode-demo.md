# Demoing in VS Code (click Run, no terminal commands)

The `notebooks/` folder has one notebook per step. Each one runs the real code in `src/` (it imports it or runs the numbered
script), shows the result as tables/charts, and ends with what you should see in the MinIO console.

## One-time setup

1. Install VS Code extensions: **Python**, **Jupyter** (both by Microsoft) and **Docker** (optional: shows your containers).
2. In a terminal, once:
   ```bash
   make setup        # venv + pinned deps (includes jupyter, ipykernel, matplotlib)
   make notebooks    # registers the Jupyter kernel "AI Lakehouse (.venv)"
   ```
3. Open the `ai-lakehouse` folder in VS Code, open `notebooks/00_start_services.ipynb`, click **Select Kernel** (top right)
   and choose **AI Lakehouse (.venv)**. If it is not listed, run *Developer: Reload Window* or pick *Python Environments...*
   and choose `.venv/bin/python`. Do this once per notebook if VS Code asks again.

Tested in VS Code: the first time, click **Select Kernel** then **Jupyter Kernel...** and pick **AI Lakehouse (.venv) (Python 3.11.x)**
(marked *Recommended*). For later notebooks VS Code offers that kernel at the top of the list. If you edit `notebooks/_setup.py`,
click **Restart** in the notebook toolbar so the change is picked up.

Java 17 is found automatically (Homebrew `openjdk@17`); you do not need it on your PATH. Docker Desktop must be running.

## Order to run the notebooks

Run each notebook top to bottom (**Run All**, or click each cell), in this order:

| # | Notebook | What the audience sees | Roughly |
|---|---|---|---|
| 00 | `00_start_services` | MinIO + REST catalog start; URL table | seconds |
| 01 | `01_check_connection` | Spark, catalog and MinIO talking | ~30 s |
| 02 | `02_bronze_ingest` | orders + reviews into an Iceberg table | ~30 s |
| 03 | `03_schema_drift` | new column, NULLs, time travel | ~30 s |
| 04 | `04_embedding_benchmark` | naive vs batched embedding speed | minutes |
| 05 | `05_silver_embeddings` | review text turned into vectors | ~1-2 min on all rows |
| 05b | `05b_translate` | *optional*, English display text | slow, set `RUN_TRANSLATION = False` to skip |
| 06 | `06_build_index` | k-means clusters | ~30 s |
| 07 | `07_gold_write` | sorted table, file-skipping preview | ~30 s |
| 08 | `08_search` | change the query string, brute vs IVF side by side | seconds per run |
| 09 | `09_benchmark` | recall vs files-scanned chart from saved results | seconds (reads saved files) |
| 10 | `10_inspect_iceberg` | snapshots, manifests, real objects in MinIO | ~30 s |
| 11 | `11_tests_and_results` | tests + final results | ~1 min |

Tips for presenting
- Keep **http://localhost:9001** (MinIO, login `minioadmin` / `minioadmin`) open in a browser next to VS Code and refresh it at
  the end of each notebook; the last cell of every notebook says what to look for.
- The **Spark UI** (http://localhost:4040) only exists while a notebook's Spark session is running. Each notebook ends with a
  cleanup cell that stops the session; skip that cell while you are showing the Spark UI.
- Notebooks are safe to re-run. Rerunning 02 rebuilds Bronze; 03 resets Bronze first if the drift was already applied.
- For a fast rehearsal on 5,000 rows, set `SAMPLE_ROWS=5000` in `.env` (or run `set_sample_rows(5000)` in the first cell of 02),
  then clear it again for the real-data run. **Notebooks 04 and 09 can overwrite `benchmarks/*.json`**: 04 always does; 09 only if you set
  `RUN_BENCHMARK = True`.
- Notebook 08 is the interactive one: edit `QUERY`, `NPROBE`, `SINCE`, `STATUS` in the first cell and run the rest.

## Check that all notebooks work

```bash
make notebooks-test        # executes every notebook top to bottom; outputs go to /tmp, the repo notebooks stay clean
```
