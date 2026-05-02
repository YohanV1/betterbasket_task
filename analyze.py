"""Lightweight post-run analysis: tier counts, score histogram, top brands matched.
Reads only the lightweight artifacts; doesn't re-normalize."""
import json
from collections import Counter
from pathlib import Path

import pandas as pd

ART = Path("artifacts")
full = pd.read_csv(ART / "matches_full.csv", low_memory=False)
matches = pd.read_csv(ART / "matches.csv", low_memory=False)
report = json.loads((ART / "run_report.json").read_text())

print("=== HEADLINE NUMBERS ===")
print(f"  A items processed:      {report['total_a_items']:,}")
print(f"  Matches emitted:        {report['matches_emitted']:,}")
print(f"  Threshold (>= 4,000):   {'PASS' if report['matches_emitted'] >= 4000 else 'FAIL'}")
print(f"  Total runtime:          {report['timings_s']['total_s']}s")
print()

print("=== TIER BREAKDOWN ===")
for tier, n in full["tier"].value_counts().items():
    pct = n / len(full)
    print(f"  {tier:>14s}: {n:>7,d}  ({pct:.1%})")
print()

print("=== COMPOSITE SCORE PERCENTILES (accepted) ===")
acc = full[full["tier"].isin(["auto_accept", "llm_accept"])]
for q in [0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]:
    print(f"  p{int(q*100):>3d}: {acc['composite'].quantile(q):.3f}")
print()

print("=== ASCII HISTOGRAM (accepted composite scores) ===")
bins = [0.5, 0.6, 0.7, 0.78, 0.85, 0.90, 0.95, 1.01]
labels = ["0.50-0.60", "0.60-0.70", "0.70-0.78", "0.78-0.85", "0.85-0.90", "0.90-0.95", "0.95-1.00"]
counts = []
for i in range(len(bins) - 1):
    lo, hi = bins[i], bins[i + 1]
    n = ((acc["composite"] >= lo) & (acc["composite"] < hi)).sum()
    counts.append(n)
maxc = max(counts) or 1
for label, n in zip(labels, counts):
    bar = "#" * int(40 * n / maxc)
    print(f"  {label}: {n:>6,d} {bar}")
print()

print("=== TIER × is_private_label CHECK (sanity: should be ~0 PL/national mismatches) ===")
print("(loading minimal A/B for PL flags...)")
import json as _json
def parse_info(v):
    try: return _json.loads(v) if isinstance(v, str) else {}
    except Exception: return {}

a_min = pd.read_csv("grocery_store_a_items_final.csv", low_memory=False,
                    usecols=["item_id", "name"])
b_min = pd.read_csv("grocery_store_b_items_final.csv", low_memory=False,
                    usecols=["item_id", "name"])
a_min["item_id"] = a_min["item_id"].astype(str)
b_min["item_id"] = b_min["item_id"].astype(str)
full["item_id_a"] = full["item_id_a"].astype(str)
full["item_id_b"] = full["item_id_b"].astype(str)

# crude PL check from name strings
from src.normalize import is_walmart_private_label, is_wegmans_private_label, clean_text
a_pl = {r["item_id"]: is_walmart_private_label(r["name"], "") for _, r in a_min.iterrows()}
b_pl = {r["item_id"]: is_wegmans_private_label(r["name"], "", "") for _, r in b_min.iterrows()}

acc2 = acc.copy()
acc2["a_pl"] = acc2["item_id_a"].map(a_pl).fillna(False)
acc2["b_pl"] = acc2["item_id_b"].map(b_pl).fillna(False)
print(f"  Both private label:    {((acc2['a_pl']) & (acc2['b_pl'])).sum():,}")
print(f"  Both national:         {((~acc2['a_pl']) & (~acc2['b_pl'])).sum():,}")
print(f"  Mixed (suspicious):    {(acc2['a_pl'] != acc2['b_pl']).sum():,}")

print("\n=== ACCEPT TIER × PL STATUS ===")
for t in ["auto_accept", "llm_accept"]:
    sub = acc2[acc2["tier"] == t]
    if len(sub) == 0: continue
    print(f"  {t}: n={len(sub):,}  pl-pl={((sub['a_pl']) & (sub['b_pl'])).sum():,}  "
          f"nat-nat={((~sub['a_pl']) & (~sub['b_pl'])).sum():,}")
