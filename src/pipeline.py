"""End-to-end pipeline orchestrator: load -> normalize -> match -> LLM tiebreak -> emit CSV.

Outputs:
  artifacts/matches_full.csv     all decisions, all tiers (debug/inspection)
  artifacts/matches.csv          accepted matches only (item_id_a, item_id_b) — DELIVERABLE
  artifacts/run_report.json      counts, timing, threshold details
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from . import match, normalize


ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    a = pd.read_csv("grocery_store_a_items_final.csv", low_memory=False)
    b = pd.read_csv("grocery_store_b_items_final.csv", low_memory=False)
    return a, b


def run(a_sample_n: Optional[int] = None,
        use_llm: bool = True,
        use_xenc: bool = False,  # Removed from production: ms-marco isn't trained
                                  # for entity resolution and scored 0.999 on both
                                  # true and false matches; only filtered 17/26k.
                                  # See "What I'd do with another week" in README.
        use_consistency: bool = True,
        llm_workers: int = 8,
        verbose: bool = True) -> dict:
    t_start = time.time()
    timings: dict = {}

    print("Loading...")
    t0 = time.time()
    a_raw, b_raw = load_data()
    if a_sample_n:
        a_raw = a_raw.sample(a_sample_n, random_state=42).reset_index(drop=True)
    timings["load_s"] = round(time.time() - t0, 1)
    print(f"  A={len(a_raw):,} B={len(b_raw):,}  ({timings['load_s']}s)")

    print("Normalizing...")
    t0 = time.time()
    a_norm = normalize.normalize_dataframe(a_raw, side="A")
    b_norm = normalize.normalize_dataframe(b_raw, side="B")
    timings["normalize_s"] = round(time.time() - t0, 1)
    print(f"  done ({timings['normalize_s']}s)")

    print("Matching (embedding retrieval + rule scoring)...")
    t0 = time.time()
    result = match.run_matching(a_norm, b_norm, verbose=verbose)
    timings["match_s"] = round(time.time() - t0, 1)
    print(f"  done ({timings['match_s']}s)")
    print("\nTier breakdown (pre-LLM):")
    print(result["tier"].value_counts().to_string())

    # Save B name lookup for LLM and report. Drop dup item_ids (if any) — keep first.
    a_lookup = (a_norm.drop_duplicates("item_id", keep="first")
                       .set_index("item_id")[["name", "brand_norm", "size_key",
                                              "is_private_label"]].to_dict("index"))
    b_lookup = (b_norm.drop_duplicates("item_id", keep="first")
                       .set_index("item_id")[["name", "brand_norm", "size_key",
                                              "is_private_label"]].to_dict("index"))

    if use_llm:
        ambiguous = result[result["tier"] == "ambiguous"].copy()
        if len(ambiguous) > 0:
            print(f"\nLLM tiebreak on {len(ambiguous):,} ambiguous items...")
            from . import llm_judge
            t0 = time.time()
            inputs: list[tuple[dict, list[dict]]] = []
            for _, r in ambiguous.iterrows():
                a_id = r["item_id_a"]
                a_info = a_lookup.get(a_id, {})
                a_payload = {"item_id": a_id, **a_info}
                # Build candidate list: chosen + runner_up (if present)
                cand_ids = [r["item_id_b"]]
                if isinstance(r["runner_up_id_b"], str) and r["runner_up_id_b"]:
                    cand_ids.append(r["runner_up_id_b"])
                cands = []
                for cid in cand_ids:
                    if cid in b_lookup:
                        cands.append({"item_id": cid, **b_lookup[cid]})
                inputs.append((a_payload, cands))
            decisions = llm_judge.judge_batch(inputs, max_workers=llm_workers, verbose=verbose)
            timings["llm_s"] = round(time.time() - t0, 1)

            # Apply decisions
            n_kept = 0
            n_rejected = 0
            for (_, r), d in zip(ambiguous.iterrows(), decisions):
                idx = r.name
                # The model occasionally returns match_id as an int rather than string
                mid_raw = d.get("match_id")
                mid = "none" if mid_raw is None else str(mid_raw).strip()
                if mid == "none" or not mid:
                    result.loc[idx, "tier"] = "llm_reject"
                    n_rejected += 1
                else:
                    # Verify the LLM picked an actual candidate
                    cand_ids = {r["item_id_b"]}
                    if isinstance(r["runner_up_id_b"], str):
                        cand_ids.add(r["runner_up_id_b"])
                    if mid in cand_ids:
                        result.loc[idx, "item_id_b"] = mid
                        result.loc[idx, "tier"] = "llm_accept"
                        n_kept += 1
                    else:
                        # LLM hallucinated an id — fall closed
                        result.loc[idx, "tier"] = "llm_reject"
                        n_rejected += 1
            print(f"  LLM accepted={n_kept}, rejected={n_rejected}, time={timings['llm_s']}s")
        else:
            timings["llm_s"] = 0.0
    else:
        timings["llm_s"] = 0.0

    # ----- Post-processing: cross-encoder rerank + bidirectional consistency -----
    # Both are precision filters. They run on the matches that survived the
    # rule + LLM stages above. We don't drop anything — instead we annotate
    # each row so downstream can choose its precision/recall trade-off.
    if use_xenc:
        print("\nCross-encoder rerank...")
        try:
            t0 = time.time()
            from . import cross_encoder
            result = cross_encoder.rerank_matches(result, a_lookup, b_lookup,
                                                    verbose=verbose)
            timings["xenc_s"] = round(time.time() - t0, 1)
        except Exception as e:
            print(f"  cross-encoder failed: {e}; continuing without it")
            timings["xenc_s"] = 0.0
            result["cross_score"] = pd.NA
            result["xenc_pass"] = pd.NA
    else:
        timings["xenc_s"] = 0.0
        result["cross_score"] = pd.NA
        result["xenc_pass"] = pd.NA

    if use_consistency:
        print("\nBidirectional consistency check...")
        try:
            t0 = time.time()
            from . import consistency
            result = consistency.add_consistency(result, a_norm, b_norm,
                                                  verbose=verbose)
            timings["consistency_s"] = round(time.time() - t0, 1)
        except Exception as e:
            print(f"  consistency failed: {e}; continuing without it")
            timings["consistency_s"] = 0.0
            result["consistent"] = pd.NA
    else:
        timings["consistency_s"] = 0.0
        result["consistent"] = pd.NA

    # ----- Emit deliverables -----
    print("\nWriting artifacts...")
    result.to_csv(ARTIFACTS / "matches_full.csv", index=False)

    # The deliverable: rows that survived the rule pipeline AND the cross-encoder
    # (if it ran). Consistency is reported but not used to drop matches —
    # multi-pack pollution can break it legitimately.
    accept_mask = result["tier"].isin(["auto_accept", "llm_accept"])
    if use_xenc and "xenc_pass" in result.columns:
        # xenc_pass NA means cross-encoder didn't score it; we keep those
        # for safety. Only drop on explicit False.
        accept_mask = accept_mask & (result["xenc_pass"] != False)
    accepted = result[accept_mask][["item_id_a", "item_id_b"]].copy()
    accepted.to_csv(ARTIFACTS / "matches.csv", index=False)

    timings["total_s"] = round(time.time() - t_start, 1)
    summary = {
        "total_a_items": int(len(a_norm)),
        "total_b_items": int(len(b_norm)),
        "matches_emitted": int(len(accepted)),
        "tier_counts": result["tier"].value_counts().to_dict(),
        "timings_s": timings,
        "thresholds": {"tau_high": match.TAU_HIGH, "tau_low": match.TAU_LOW,
                       "name_sim_floor": match.NAME_SIM_FLOOR, "top_k": match.TOP_K},
        "weights": {"emb": match.W_EMB, "tfidf": match.W_TFIDF,
                    "size": match.W_SIZE, "brand": match.W_BRAND, "flags": match.W_FLAGS},
    }
    (ARTIFACTS / "run_report.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n=== DONE ({timings['total_s']}s) ===")
    print(f"Matches written: {len(accepted):,} -> artifacts/matches.csv")
    return summary
