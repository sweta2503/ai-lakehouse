#!/usr/bin/env python3
"""Generate a synthetic stand-in for the Olist orders + reviews tables.

Same columns as `olist_orders_dataset.csv` and `olist_order_reviews_dataset.csv`, but the
review text is ENGLISH and built from themes (late delivery, damaged packaging, great quality,
wrong item, refund requests, ...), so semantic search has something real to find.

This is fake data. It exists so the repo runs without a Kaggle login. Numbers measured on it
(recall, latency) reflect its easy, well-separated clusters; real reviews are messier.

Usage: python scripts/generate_synthetic_data.py [--rows 50000] [--out data/synthetic] [--seed 7]
"""
from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

PRODUCTS = ["headphones", "blender", "running shoes", "phone case", "backpack", "desk lamp", "coffee maker",
            "yoga mat", "water bottle", "bluetooth speaker", "office chair", "kitchen knife set", "t-shirt",
            "watch", "toy car", "sofa cover", "keyboard", "monitor stand", "perfume", "bed sheets",
            "air fryer", "garden hose", "suitcase", "baby stroller", "board game"]

# theme -> (weight, [score choices], [sentences]) ; {p}=product, {d}=days
THEMES = {
    "late_delivery": (0.17, [1, 1, 2, 2, 3], [
        "The delivery took far too long, it arrived {d} days after the estimated date.",
        "My {p} was supposed to arrive last week and I am still waiting.",
        "Very late delivery. I needed the {p} for a gift and it came {d} days late.",
        "Shipping was a disaster, the package was stuck in transit for over {d} days.",
        "The order arrived way past the promised deadline, terrible logistics.",
        "I waited more than {d} days for my {p}. Unacceptable delay.",
        "Product delivered late, no tracking updates at all.",
        "The carrier delayed everything and the {p} showed up weeks after the due date.",
    ]),
    "damaged_packaging": (0.10, [1, 1, 2, 2, 3], [
        "The box arrived crushed and the {p} was damaged inside.",
        "Packaging was torn open and the {p} had scratches all over it.",
        "Item came broken because the package was not protected at all.",
        "The {p} arrived with a cracked case, clearly mishandled during shipping.",
        "Terrible packaging, the box was wet and dented when it got here.",
        "I opened the package and the {p} was smashed. No bubble wrap, nothing.",
        "The product was damaged on arrival and the packaging was falling apart.",
    ]),
    "great_quality": (0.20, [5, 5, 5, 4], [
        "Excellent quality, the {p} is exactly as described and feels premium.",
        "I am very happy with this {p}. Great build and works perfectly.",
        "Superb product, much better than I expected for the price.",
        "The {p} is sturdy and well made. Highly recommend it.",
        "Amazing quality! I would buy this {p} again without hesitation.",
        "Perfect {p}, great materials and nice finish. Totally worth it.",
        "Very good product, high quality and works like a charm.",
    ]),
    "wrong_item": (0.07, [1, 1, 2], [
        "I received the wrong item, I ordered a {p} and got something completely different.",
        "They sent me the wrong color and size, not what I ordered.",
        "The product in the box was not the {p} I purchased.",
        "Wrong product delivered. The listing said one thing and I received another.",
        "I ordered one {p} but a different model arrived, the order was mixed up.",
        "Got someone else's order by mistake, totally different from my purchase.",
    ]),
    "refund_request": (0.07, [1, 1, 2], [
        "I want my money back. Please process the refund for this {p}.",
        "Requesting a refund, the {p} does not work and I returned it {d} days ago.",
        "Still waiting for my refund after sending the product back.",
        "I asked for reimbursement three times and nobody replied. I want a full refund.",
        "Cancel my order and refund me immediately, this is not what I wanted.",
        "Return approved but the refund has not arrived yet, very frustrating.",
    ]),
    "fast_delivery": (0.12, [5, 5, 4], [
        "Super fast delivery, the {p} arrived two days before the deadline.",
        "Arrived much earlier than expected, great shipping.",
        "The package came quickly and was well packed.",
        "Delivered ahead of schedule, I am impressed with the speed.",
        "Fast shipping and the {p} was in perfect condition.",
        "Received in just {d} days, excellent service.",
    ]),
    "poor_quality": (0.08, [1, 2, 2, 3], [
        "The {p} feels cheap and stopped working after {d} days.",
        "Poor quality, the material is flimsy and it broke quickly.",
        "Not worth the money, the {p} looks nothing like the photos and is badly made.",
        "Disappointed, the {p} fell apart the first time I used it.",
        "Low quality product, it does not do what the description promised.",
        "The stitching came undone and the {p} is basically useless.",
    ]),
    "seller_service_bad": (0.05, [1, 2, 3], [
        "The seller never answered my messages about the {p}.",
        "Customer service was rude and did not solve my problem.",
        "No response from the store after several emails, horrible support.",
        "The seller ignored my complaint and blamed the carrier.",
    ]),
    "seller_service_good": (0.06, [5, 4, 5], [
        "The seller answered quickly and solved my question about the {p}.",
        "Great customer service, they replaced the item without any fuss.",
        "Very attentive seller, they kept me updated the whole time.",
        "Helpful support team, a pleasure to buy from this store.",
    ]),
    "good_value": (0.05, [4, 5], [
        "Great value for money, the {p} does everything I need at a fair price.",
        "Cheap and good, I did not expect this much for the price.",
        "Affordable {p} that works well, good cost benefit.",
    ]),
    "missing_item": (0.03, [1, 2], [
        "The package arrived but one of the items was missing.",
        "Only part of my order came, the {p} was not in the box.",
        "Missing accessories, the box did not include everything that was listed.",
    ]),
}
POSITIVE = ["great_quality", "fast_delivery", "seller_service_good", "good_value"]
NEGATIVE = ["late_delivery", "damaged_packaging", "wrong_item", "refund_request", "poor_quality",
            "seller_service_bad", "missing_item"]
TITLES = {
    "late_delivery": ["Late delivery", "Delayed", "Took forever"],
    "damaged_packaging": ["Damaged", "Broken on arrival", "Bad packaging"],
    "great_quality": ["Great!", "Excellent product", "Recommend"],
    "wrong_item": ["Wrong item", "Not what I ordered"],
    "refund_request": ["Refund", "Want my money back"],
    "fast_delivery": ["Fast", "Very quick"],
    "poor_quality": ["Poor quality", "Disappointed"],
    "seller_service_bad": ["Bad service"], "seller_service_good": ["Good service"],
    "good_value": ["Good value"], "missing_item": ["Missing item"],
}
STATUSES = ["delivered", "shipped", "canceled", "unavailable", "invoiced", "processing", "created", "approved"]
STATUS_P = [0.9700, 0.0110, 0.0063, 0.0060, 0.0032, 0.0030, 0.0003, 0.0002]
FMT = "%Y-%m-%d %H:%M:%S"


def sentence(rng, theme):
    tpl = THEMES[theme][2][rng.integers(len(THEMES[theme][2]))]
    return tpl.format(p=PRODUCTS[rng.integers(len(PRODUCTS))], d=int(rng.integers(2, 30)))


def make_review(rng):
    """Returns (title|None, message|None, score, primary_theme)."""
    names = list(THEMES)
    w = np.array([THEMES[n][0] for n in names])
    theme = names[rng.choice(len(names), p=w / w.sum())]
    score = int(rng.choice(THEMES[theme][1]))
    msg = sentence(rng, theme)
    if rng.random() < 0.15:  # mixed review: one good thing + one bad thing -> middling score
        other = rng.choice(POSITIVE if theme in NEGATIVE else NEGATIVE)
        msg = f"{sentence(rng, str(other))} But " + msg[0].lower() + msg[1:]
        score = int(rng.choice([2, 3, 3, 4]))
    elif rng.random() < 0.30:
        msg += " " + sentence(rng, theme)
    title = str(rng.choice(TITLES[theme])) if rng.random() < 0.15 else None
    if rng.random() < 0.28:  # many real Olist reviews have a score but no text
        msg = None
        title = title if rng.random() < 0.05 else None
        theme = "no_text"
    return title, msg, score, theme


def generate(rows: int, out: Path, seed: int = 7) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    start, end = datetime(2017, 1, 1), datetime(2018, 8, 31, 23, 59, 59)
    span = (end - start).total_seconds()
    # Marketplace grows over time -> skew purchase times toward the end of the window.
    frac = rng.beta(2.2, 1.4, rows)
    purchase = [start + timedelta(seconds=float(f * span)) for f in frac]
    statuses = rng.choice(STATUSES, size=rows, p=np.array(STATUS_P) / sum(STATUS_P))

    orders_path = out / "olist_orders_dataset.csv"
    reviews_path = out / "olist_order_reviews_dataset.csv"
    fmt = lambda d: d.strftime(FMT) if d else ""  # noqa: E731
    with open(orders_path, "w", newline="") as fo, open(reviews_path, "w", newline="") as fr:
        wo, wr = csv.writer(fo), csv.writer(fr)
        wo.writerow(["order_id", "customer_id", "order_status", "order_purchase_timestamp", "order_approved_at",
                     "order_delivered_carrier_date", "order_delivered_customer_date",
                     "order_estimated_delivery_date"])
        wr.writerow(["review_id", "order_id", "review_score", "review_comment_title", "review_comment_message",
                     "review_creation_date", "review_answer_timestamp"])
        for i in range(rows):
            oid, cid, rid = rng.bytes(16).hex(), rng.bytes(16).hex(), rng.bytes(16).hex()
            st, p = str(statuses[i]), purchase[i]
            title, msg, score, theme = make_review(rng)
            est = p + timedelta(days=int(rng.integers(15, 30)), seconds=int(rng.integers(0, 86400)))
            approved = carrier = delivered = None
            if st != "created":
                approved = p + timedelta(minutes=int(rng.exponential(600)) + 5)
            if st in ("shipped", "delivered"):
                carrier = approved + timedelta(hours=int(rng.integers(6, 72)))
            if st == "delivered":
                # late-delivery reviews really are late; everyone else is mostly on time
                late = theme == "late_delivery" or rng.random() < 0.07
                delivered = est + timedelta(days=int(rng.integers(2, 20))) if late else \
                    min(carrier + timedelta(days=int(rng.integers(2, 14))), est - timedelta(hours=3))
            wo.writerow([oid, cid, st, fmt(p), fmt(approved), fmt(carrier), fmt(delivered), fmt(est)])
            # Only delivered/shipped/canceled orders realistically get reviews, but Olist reviews exist
            # for almost every order, so we emit one per order.
            created = (delivered or est) + timedelta(days=int(rng.integers(0, 6)))
            answered = created + timedelta(hours=int(rng.integers(1, 96)))
            wr.writerow([rid, oid, score, title or "", msg or "", fmt(created), fmt(answered)])
    return orders_path, reviews_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=int(os.environ.get("SYNTHETIC_ROWS") or 50000))
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "synthetic")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    o, r = generate(a.rows, a.out, a.seed)
    print(f"wrote synthetic dataset ({a.rows} orders, 1 review each):\n  {o}\n  {r}")
