"""Post-processing: bidirectional consistency annotation.

Runs on `artifacts/matches_full.csv` from a finished pipeline run, adds the
A<->B consistency flag, and emits an updated `matches_full_pp.csv` for the
verification stage to consume.

Note on cross-encoder: an earlier version of this pipeline ran a
`cross-encoder/ms-marco-MiniLM-L-6-v2` reranker here. We removed it from the
production path because it filtered only 17 of 26k matches (0.07%) — ms-marco
was trained for retrieval relevance, not entity resolution, and scored both
true and false matches at ~0.999. The diagnosis is documented in the README
"What I'd do with another week" section, where fine-tuning a cross-encoder
specifically for entity resolution is the lead next-step item.

This script is split from the main pipeline so reruns are fast (no re-matching,
no re-LLM calls) and so the consistency annotation can be inspected.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.normalize import normalize_dataframe
from src import consistency

ARTIFACTS = Path("artifacts")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_full", default=str(ARTIFACTS / "matches_full.csv"))
    p.add_argument("--out_full", default=str(ARTIFACTS / "matches_full_pp.csv"))
    p.add_argument("--out_matches", default=str(ARTIFACTS / "matches.csv"))
    args = p.parse_args()

    print(f"Loading {args.in_full}...")
    result = pd.read_csv(args.in_full, dtype={"item_id_a": str, "item_id_b": str,
                                                "runner_up_id_b": str})
    print(f"  {len(result):,} rows")

    print("\nNormalizing original CSVs (for embedding cache lookup + names)...")
    a = pd.read_csv("grocery_store_a_items_final.csv", low_memory=False)
    b = pd.read_csv("grocery_store_b_items_final.csv", low_memory=False)
    a_norm = normalize_dataframe(a, side="A")
    b_norm = normalize_dataframe(b, side="B")

    timings: dict[str, float] = {}

    print("\nBidirectional consistency check...")
    t0 = time.time()
    result = consistency.add_consistency(result, a_norm, b_norm, verbose=True)
    timings["consistency_s"] = round(time.time() - t0, 1)

    # Decide accepted set (rule + LLM tier only at this stage; verify.py
    # narrows further). Consistency is annotated, never filters — multi-pack
    # collisions break it legitimately (see consistency.py docstring).
    accept_mask = result["tier"].isin(["auto_accept", "llm_accept"])
    base_accepted = int(accept_mask.sum())

    consistent_among_accepted = int((result.loc[accept_mask, "consistent"] == True).sum())

    result.to_csv(args.out_full, index=False)
    accepted_df = result[accept_mask][["item_id_a", "item_id_b"]].copy()
    accepted_df.to_csv(args.out_matches, index=False)

    summary = {
        "input_rows": int(len(result)),
        "rule_or_llm_accepted": base_accepted,
        "consistent_among_accepted": consistent_among_accepted,
        "consistency_rate": round(consistent_among_accepted / max(base_accepted, 1), 3),
        "timings_s": timings,
    }
    (ARTIFACTS / "postprocess_report.json").write_text(json.dumps(summary, indent=2))
    print("\n=== POST-PROCESS SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {args.out_full} and {args.out_matches}")


if __name__ == "__main__":
    main()
