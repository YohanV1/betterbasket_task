"""Post-run validation. Loads the artifacts and inspects:
  - tier counts + match yield
  - confidence distribution
  - bucket breakdown
  - random spot-check samples (with names) at each confidence level
  - private-label vs national breakdown
"""
import json
from pathlib import Path

import pandas as pd

from src.normalize import normalize_dataframe

ART = Path("artifacts")
print("Loading artifacts...")
# Prefer the post-processed full file if present (contains cross_score + consistent)
full_pp = ART / "matches_full_pp.csv"
full = pd.read_csv(full_pp if full_pp.exists() else ART / "matches_full.csv",
                   low_memory=False)
matches = pd.read_csv(ART / "matches.csv", low_memory=False)
report = json.loads((ART / "run_report.json").read_text())
pp_report = None
if (ART / "postprocess_report.json").exists():
    pp_report = json.loads((ART / "postprocess_report.json").read_text())

print("\n=== RUN REPORT ===")
print(json.dumps(report, indent=2, default=str))

if pp_report:
    print("\n=== POST-PROCESS REPORT ===")
    print(json.dumps(pp_report, indent=2, default=str))

if (ART / "label_metrics_v.json").exists():
    print("\n=== LABEL METRICS (held-out 540-pair stratified set, gpt-5-nano) ===")
    print((ART / "label_metrics_v.json").read_text())

if (ART / "triangulation_report.json").exists():
    print("\n=== TRIANGULATION REPORT (Claude Sonnet 4.6 on labeled subsample) ===")
    print((ART / "triangulation_report.json").read_text())

print("\n=== TIER BREAKDOWN ===")
print(full["tier"].value_counts())
print(f"\nTotal A items: {len(full):,}")
print(f"Final accepted matches: {len(matches):,}  ({len(matches)/len(full):.1%} of A)")
print(f"Threshold for task: 4,000 -> {'PASS' if len(matches) >= 4000 else 'FAIL'}")

print("\n=== COMPOSITE SCORE DISTRIBUTION (accepted matches) ===")
acc_full = full[full["tier"].isin(["auto_accept", "llm_accept"])]
print(acc_full["composite"].describe())
print("\nQuantiles:")
for q in [0.05, 0.25, 0.5, 0.75, 0.9, 0.99]:
    print(f"  p{int(q*100)}: {acc_full['composite'].quantile(q):.3f}")

# Need names; reload + normalize a tiny slice for spot-checks
print("\n=== LOADING NAMES FOR SPOT-CHECK ===")
a = pd.read_csv("grocery_store_a_items_final.csv", low_memory=False)
b = pd.read_csv("grocery_store_b_items_final.csv", low_memory=False)
a_norm = normalize_dataframe(a, side="A")[["item_id", "name", "brand_norm",
                                            "size_key", "is_private_label", "cat0"]]
b_norm = normalize_dataframe(b, side="B")[["item_id", "name", "brand_norm",
                                            "size_key", "is_private_label", "cat0"]]
a_norm["item_id"] = a_norm["item_id"].astype(str)
b_norm["item_id"] = b_norm["item_id"].astype(str)
matches["item_id_a"] = matches["item_id_a"].astype(str)
matches["item_id_b"] = matches["item_id_b"].astype(str)
full["item_id_a"] = full["item_id_a"].astype(str)
full["item_id_b"] = full["item_id_b"].astype(str)

joined = (matches
          .merge(a_norm.add_suffix("_a").rename(columns={"item_id_a": "item_id_a"}),
                 left_on="item_id_a", right_on="item_id_a", how="left")
          .merge(b_norm.add_suffix("_b").rename(columns={"item_id_b": "item_id_b"}),
                 left_on="item_id_b", right_on="item_id_b", how="left"))

print("\n=== PL vs NATIONAL split among accepted ===")
print("Both private label:",
      ((joined["is_private_label_a"] == True) & (joined["is_private_label_b"] == True)).sum())
print("Both national:",
      ((joined["is_private_label_a"] == False) & (joined["is_private_label_b"] == False)).sum())
print("Mismatched (should be ~0):",
      (joined["is_private_label_a"] != joined["is_private_label_b"]).sum())

print("\n=== BREAKDOWN BY A.cat0 ===")
print(joined["cat0_a"].value_counts())

# Also pull tier from matches_full to distinguish auto vs LLM accepts
joined = joined.merge(full[["item_id_a", "tier"]], on="item_id_a", how="left")

print("\n=== ACCEPT TIER BREAKDOWN (within emitted matches.csv) ===")
print(joined["tier"].value_counts())

print("\n=== 30 RANDOM SPOT-CHECKS (national brand AUTO-accepts) ===")
nat_auto = joined[(joined["is_private_label_a"] == False)
                  & (joined["is_private_label_b"] == False)
                  & (joined["tier"] == "auto_accept")]
samp = nat_auto.sample(min(30, len(nat_auto)), random_state=2)
for _, r in samp.iterrows():
    print(f"  A: {r['name_a']!r}")
    print(f"  B: {r['name_b']!r}")
    print()

print("\n=== 20 RANDOM SPOT-CHECKS (private-label cross-matches, AUTO) ===")
pl_auto = joined[(joined["is_private_label_a"] == True)
                 & (joined["is_private_label_b"] == True)
                 & (joined["tier"] == "auto_accept")]
samp = pl_auto.sample(min(20, len(pl_auto)), random_state=3)
for _, r in samp.iterrows():
    print(f"  A: {r['name_a']!r}")
    print(f"  B: {r['name_b']!r}")
    print()

llm_acc = joined[joined["tier"] == "llm_accept"]
if len(llm_acc) > 0:
    print(f"\n=== 30 RANDOM SPOT-CHECKS (LLM-accepted, n={len(llm_acc):,}) ===")
    samp = llm_acc.sample(min(30, len(llm_acc)), random_state=4)
    for _, r in samp.iterrows():
        print(f"  A: {r['name_a']!r}")
        print(f"  B: {r['name_b']!r}")
        print()
