"""LLM verification of auto-accept matches.

Why this exists: the held-out 200-pair labeled set surfaced a real failure mode
in the pipeline. The composite score auto-accepts pairs at >= 0.78, but inside
the 0.78-0.95 band we measured ~58% precision. The wrong matches share a
specific pattern — same brand, same size, *different variant or product type*
(Cetaphil cream vs lotion; Mrs. Meyer's multi-surface vs dish soap; Hormel
chili WITH beans vs NO beans). These slip through because:

  - Embedding similarity is high (same brand + similar packaging + similar
    product description).
  - Cross-encoder (ms-marco) treats the pair as "relevant" since it was
    trained for retrieval, not entity resolution. Cross-scores are 0.999
    on both true and false matches.

The labels also showed that an LLM judge correctly identifies these failures
when shown both items side-by-side with structured size + brand-class info.
So the right move is to push the LLM into the auto-accept band as a
verification step. We use the same judge module as the tiebreak, but a
distinct cache namespace so prompts can evolve independently.

This is exactly the "active learning loop with internal matchers" pattern
the README describes — except the loop is closed inside this submission:
labels surfaced the failure mode, and the failure mode drove the design
of this stage.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.normalize import normalize_dataframe
from src.llm_judge import judge_batch

ARTIFACTS = Path("artifacts")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_full", default=str(ARTIFACTS / "matches_full_pp.csv"))
    p.add_argument("--out_full", default=str(ARTIFACTS / "matches_full_v.csv"))
    p.add_argument("--out_matches", default=str(ARTIFACTS / "matches.csv"))
    p.add_argument("--composite_lo", type=float, default=0.78,
                    help="Verify auto_accept items with composite >= this (default 0.78).")
    p.add_argument("--composite_hi", type=float, default=0.95,
                    help="Skip verification for items >= this (default 0.95). The labeled "
                         "set showed precision = 1.0 at composite >= 0.95.")
    p.add_argument("--workers", type=int, default=24)
    args = p.parse_args()

    print(f"Loading {args.in_full}...")
    result = pd.read_csv(args.in_full,
                          dtype={"item_id_a": str, "item_id_b": str,
                                  "runner_up_id_b": str},
                          low_memory=False)
    print(f"  {len(result):,} rows")

    # Identify auto_accept items needing verification
    needs_verify = (
        (result["tier"] == "auto_accept")
        & result["composite"].between(args.composite_lo, args.composite_hi, inclusive="left")
    )
    n_verify = int(needs_verify.sum())
    n_skip = int(((result["tier"] == "auto_accept") & ~needs_verify).sum())
    print(f"  auto_accept needing verification (composite {args.composite_lo:.2f}-{args.composite_hi:.2f}): {n_verify:,}")
    print(f"  auto_accept skipping verification (composite >= {args.composite_hi:.2f}): {n_skip:,}")

    if n_verify == 0:
        result.to_csv(args.out_full, index=False)
        print("nothing to verify, exiting")
        return

    # Build lookups
    print("\nLoading + normalizing CSVs for prompt context...")
    a = pd.read_csv("grocery_store_a_items_final.csv", low_memory=False)
    b = pd.read_csv("grocery_store_b_items_final.csv", low_memory=False)
    a_norm = normalize_dataframe(a, side="A")
    b_norm = normalize_dataframe(b, side="B")
    a_lookup = (a_norm.drop_duplicates("item_id").set_index("item_id")
                       [["name", "brand_norm", "size_key", "is_private_label"]]
                       .to_dict("index"))
    b_lookup = (b_norm.drop_duplicates("item_id").set_index("item_id")
                       [["name", "brand_norm", "size_key", "is_private_label"]]
                       .to_dict("index"))
    a_lookup = {str(k): v for k, v in a_lookup.items()}
    b_lookup = {str(k): v for k, v in b_lookup.items()}

    # Build inputs: pass both chosen and runner_up so the LLM can pick a different one
    rows = result[needs_verify]
    inputs: list[tuple[dict, list[dict]]] = []
    for _, r in rows.iterrows():
        aid = str(r["item_id_a"])
        a_payload = {"item_id": aid, **a_lookup.get(aid, {})}
        cand_ids = [str(r["item_id_b"])]
        if isinstance(r["runner_up_id_b"], str) and r["runner_up_id_b"]:
            cand_ids.append(str(r["runner_up_id_b"]))
        cands = [{"item_id": cid, **b_lookup[cid]} for cid in cand_ids if cid in b_lookup]
        inputs.append((a_payload, cands))

    print(f"\nLLM verification on {len(inputs):,} items, {args.workers} workers...")
    t0 = time.time()
    decisions = judge_batch(inputs, max_workers=args.workers, verbose=True)
    elapsed = round(time.time() - t0, 1)
    print(f"  done in {elapsed}s ({len(inputs)/elapsed:.1f}/s)")

    # Apply decisions
    n_kept = 0
    n_swapped = 0
    n_rejected = 0
    for (_, r), d in zip(rows.iterrows(), decisions):
        idx = r.name
        mid_raw = d.get("match_id")
        mid = "none" if mid_raw is None else str(mid_raw).strip()
        if mid == "none" or not mid:
            result.loc[idx, "tier"] = "verify_reject"
            n_rejected += 1
            continue
        cand_ids = {str(r["item_id_b"])}
        if isinstance(r["runner_up_id_b"], str):
            cand_ids.add(str(r["runner_up_id_b"]))
        if mid in cand_ids:
            if mid != str(r["item_id_b"]):
                result.loc[idx, "item_id_b"] = mid
                result.loc[idx, "tier"] = "verify_swap"
                n_swapped += 1
            else:
                result.loc[idx, "tier"] = "verify_accept"
                n_kept += 1
        else:
            result.loc[idx, "tier"] = "verify_reject"
            n_rejected += 1

    print(f"\nVerification outcomes: kept={n_kept:,}, swapped={n_swapped:,}, rejected={n_rejected:,}")

    # Updated tier counts
    print("\n=== UPDATED TIER COUNTS ===")
    print(result["tier"].value_counts())

    # Save updated full + new matches.csv
    result.to_csv(args.out_full, index=False)
    accepted = result[result["tier"].isin(
        ["auto_accept", "verify_accept", "verify_swap", "llm_accept"]
    )][["item_id_a", "item_id_b"]].copy()
    accepted.to_csv(args.out_matches, index=False)

    summary = {
        "verified_n": len(inputs),
        "kept": n_kept, "swapped": n_swapped, "rejected": n_rejected,
        "kept_pct": round(n_kept / len(inputs), 3),
        "skipped_high_score": n_skip,
        "final_accepted": int(len(accepted)),
        "elapsed_s": elapsed,
    }
    (ARTIFACTS / "verify_report.json").write_text(json.dumps(summary, indent=2))
    print("\n=== VERIFY SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {args.out_full} and {args.out_matches}")


if __name__ == "__main__":
    main()
