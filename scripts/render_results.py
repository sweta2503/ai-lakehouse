#!/usr/bin/env python3
"""Fill the <!-- BENCH:xxx:START/END --> blocks in README.md from the files in benchmarks/.
Numbers are never typed by hand: they are copied from what the pipeline measured."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
B = ROOT / "benchmarks"


def read(name):
    p = B / name
    return p.read_text() if p.exists() else None


def strip_title(md):
    return "\n".join(l for l in md.splitlines() if not l.startswith("# ")).strip()


blocks = {}
emb = read("embedding_benchmark.md")
blocks["embedding"] = strip_title(emb) if emb else "_Not generated yet: run `make embed-bench`._"
ivf = read("ivf_benchmark.md")
blocks["ivf"] = strip_title(ivf) if ivf else "_Not generated yet: run `make bench`._"
qd = read("qdrant_comparison.md")
blocks["qdrant"] = strip_title(qd) if qd else "_Bonus not run yet: `make bonus`._"
if (B / "embedding_benchmark.json").exists():
    j = json.loads((B / "embedding_benchmark.json").read_text())
    blocks["headline"] = (f"On this machine, batched `pandas_udf` was **{j['speedup_batched_vs_naive']}x** faster than the "
                          f"naive row-by-row UDF ({j['batched_rows_per_sec']:.0f} vs {j['naive_rows_per_sec']:.0f} rows/s "
                          f"over {j['rows']} rows).")
else:
    blocks["headline"] = "_Run `make embed-bench`._"

for f in (ROOT / "README.md",):
    if not f.exists():
        continue
    text = f.read_text()
    for key, body in blocks.items():
        text = re.sub(rf"(<!-- BENCH:{key}:START -->).*?(<!-- BENCH:{key}:END -->)",
                      lambda m: f"{m.group(1)}\n{body}\n{m.group(2)}", text, flags=re.S)
    f.write_text(text)
print("README results blocks refreshed from benchmarks/")
