"""Step 1b: make the dataset English.

Olist review titles/messages are Portuguese. This job translates every distinct text once (free local model, see
translation.py) and writes data/english/olist_order_reviews_dataset.csv. From then on the whole pipeline
(Bronze, embeddings, IVF index, search, UI, benchmark) works on English text; the Portuguese originals stay in the
extra columns review_comment_title_pt / review_comment_message_pt.

Resumable: finished translations are appended to data/english/translation_cache.jsonl every few hundred texts,
so you can stop it (Ctrl+C) and run it again. Takes roughly 30-40 minutes for the full 35k distinct texts on a laptop.

  python src/01b_translate_dataset.py            # translate everything, write the English CSV
  python src/01b_translate_dataset.py --peek 15  # just show 15 random examples (Portuguese -> English)
"""
import argparse
import json
import re
import sys
import time

import pandas as pd

from common import DATA_DIR, get_logger, section
from data_source import ORDERS, REVIEWS, locate_csvs
from translation import load_cache, translate_texts

log = get_logger("translate-dataset")
OUT_DIR = DATA_DIR / "english"
CACHE = OUT_DIR / "translation_cache.jsonl"
CHUNK = 512


def squash(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--peek", type=int, default=0, help="show N random translated examples and exit (writes nothing)")
    args = ap.parse_args()

    section("Step 1b: translate the Olist reviews to English (free local model)")
    _, reviews_csv, source = locate_csvs(prefer_english=False)
    if source != "olist":
        log.info("Dataset is '%s' (already English) - nothing to translate.", source)
        return 0

    df = pd.read_csv(reviews_csv, dtype=str, keep_default_na=False)
    cols = ["review_comment_title", "review_comment_message"]
    for c in cols:
        df[c + "_pt"] = df[c]
    texts = sorted({squash(v) for c in cols for v in df[c] if squash(v)})
    log.info("%d reviews, %d distinct non-empty texts to translate", len(df), len(texts))

    if args.peek:
        sample = pd.Series(texts).sample(args.peek, random_state=1).tolist()
        t0 = time.time()
        en, _ = translate_texts(sample)
        for pt, e in zip(sample, en):
            print(f"PT: {pt}\nEN: {e}\n")
        log.info("%.1f texts/sec", len(sample) / (time.time() - t0))
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = load_cache(CACHE)
    todo = [t for t in texts if t not in cache]
    log.info("%d already in cache, %d to go", len(cache), len(todo))
    # longest texts last so the progress rate looks steadier; sorted by length also keeps batches tight
    todo.sort(key=len)
    rejected_total, t0 = 0, time.time()
    with CACHE.open("a", encoding="utf-8") as f:
        for s in range(0, len(todo), CHUNK):
            chunk = todo[s:s + CHUNK]
            en, rej = translate_texts(chunk)
            rejected_total += rej
            for pt, e in zip(chunk, en):
                cache[pt] = e
                f.write(json.dumps({"pt": pt, "en": e}, ensure_ascii=False) + "\n")
            f.flush()
            done = s + len(chunk)
            rate = done / (time.time() - t0)
            log.info("translated %d/%d  (%.1f texts/sec, ~%.0f min left)", done, len(todo), rate,
                     (len(todo) - done) / rate / 60)

    for c in cols:
        df[c] = [cache.get(squash(v), "") if squash(v) else "" for v in df[c]]
    out = OUT_DIR / REVIEWS
    df.to_csv(out, index=False)
    kept = sum(1 for pt, en in cache.items() if pt == en)
    log.info("wrote %s (%d rows). Texts left in Portuguese by the guards (rejected output): %d this run",
             out, len(df), rejected_total)
    log.info("note: %d cached texts are identical before/after (names, 'ok', emoji, or rejected) - expected", kept)
    print("\nSamples (Portuguese -> English):")
    s = df[df["review_comment_message_pt"].str.len() > 40].sample(5, random_state=3)
    for _, r in s.iterrows():
        print(f"  PT: {squash(r['review_comment_message_pt'])[:110]}\n  EN: {squash(r['review_comment_message'])[:110]}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
