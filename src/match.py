"""Core matching pipeline.

Strategy (cheap -> expensive):
  1. Normalize both sides (see normalize.py).
  2. Bucket each item into a coarse cross-retailer category (see category_map.py).
  3. Within each bucket, retrieve top-K candidates per A item using
     sentence-transformer embeddings (MiniLM, 384-d).  Embeddings are the primary
     retriever — they catch semantic equivalence that token-based methods miss.
  4. Re-score the top-K with a composite that adds discriminating features
     embeddings smooth over:
        - embedding similarity (primary, retrieval signal)
        - TF-IDF char/word similarity (token-level sharpness for size + brand tokens)
        - size match (hard penalty if mismatched)
        - brand match (national)  OR  both private-label
        - flag agreement (organic, frozen, decaf, ...)
  5. Auto-accept if composite >= TAU_HIGH; auto-reject below TAU_LOW.
  6. Return ambiguous middle-band candidates for downstream LLM tiebreaking.

Output is a DataFrame with one row per A item that found at least one candidate.
Each row carries the chosen B id (or None) and the confidence tier.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import normalize, category_map, embed

# ---------------- thresholds ----------------
TAU_HIGH = 0.78  # auto-accept
TAU_LOW = 0.50   # auto-reject
TOP_K = 8         # candidates per A item to keep for scoring (more recall now that
                  # the embedding retriever has good semantic precision upstream)
NAME_SIM_FLOOR = 0.30  # absolute name-sim floor before we even consider a candidate

# Composite weights — embeddings are primary, TF-IDF is the precision signal
W_EMB = 0.35    # semantic retrieval similarity (MiniLM)
W_TFIDF = 0.20  # token-level (char+word) sharpness — catches flavor/size variants
W_SIZE = 0.20
W_BRAND = 0.15
W_FLAGS = 0.10


# ---------------- data shape ----------------
@dataclass
class Candidate:
    item_id_b: str
    emb_sim: float
    tfidf_sim: float
    name_sim: float        # combined sim used for the floor check (= 0.5*emb + 0.5*tfidf)
    size_match: bool
    brand_match: bool
    flag_agree: float  # in [0,1]
    composite: float


@dataclass
class MatchResult:
    item_id_a: str
    chosen: Optional[Candidate]
    runner_up: Optional[Candidate]
    tier: str  # 'auto_accept' | 'ambiguous' | 'auto_reject' | 'no_candidates'


# ---------------- text features ----------------
def _safe_str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v)


def build_text_for_vectorizer(row: pd.Series) -> str:
    """Concatenate brand + name + size_key for richer signal.
    Including brand twice slightly upweights it."""
    bits = []
    brand = _safe_str(row.get("brand_norm"))
    if brand:
        bits.append(brand)
        bits.append(brand)  # weight
    name = _safe_str(row.get("name_norm"))
    if name:
        bits.append(name)
    size_key = _safe_str(row.get("size_key"))
    if size_key:
        bits.append(size_key)
    return " ".join(bits)


# ---------------- core matching per bucket ----------------
def match_bucket(a_df: pd.DataFrame, b_df: pd.DataFrame, bucket: str,
                 a_emb: np.ndarray, b_emb: np.ndarray) -> list[MatchResult]:
    """Match all A items in `a_df` against all B items in `b_df`.

    Pipeline within a bucket:
      1. Embedding retrieval -> top-K candidates per A.
      2. TF-IDF (char + word) computed only on the K candidates (cheap).
      3. Composite scoring with embedding + TF-IDF + size + brand + flag features.

    Both DFs and their embedding matrices must be aligned by row order.
    """
    if len(a_df) == 0 or len(b_df) == 0:
        return [
            MatchResult(item_id_a=str(r["item_id"]), chosen=None, runner_up=None,
                        tier="no_candidates")
            for _, r in a_df.iterrows()
        ]

    # Step 1: Embedding-based retrieval — primary signal
    top_idx, top_emb_sim = embed.topk_candidates(a_emb, b_emb, k=TOP_K)

    # Step 2: TF-IDF only over this bucket's items, for token-level features
    a_texts = a_df.apply(build_text_for_vectorizer, axis=1).tolist()
    b_texts = b_df.apply(build_text_for_vectorizer, axis=1).tolist()
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                min_df=1, max_df=0.95, sublinear_tf=True)
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                min_df=1, max_df=0.95, sublinear_tf=True)
    char_vec.fit(a_texts + b_texts)
    word_vec.fit(a_texts + b_texts)
    A_char = char_vec.transform(a_texts)
    B_char = char_vec.transform(b_texts)
    A_word = word_vec.transform(a_texts)
    B_word = word_vec.transform(b_texts)

    # Step 3: Score the top-K per A row
    results: list[MatchResult] = []
    n_a = a_emb.shape[0]
    for i in range(n_a):
        cand_idx = top_idx[i]
        cand_emb_sim = top_emb_sim[i]

        # TF-IDF sims only for top-K (small)
        a_char_v = A_char[i]
        a_word_v = A_word[i]
        sub_b_char = B_char[cand_idx]
        sub_b_word = B_word[cand_idx]
        cand_char_sim = cosine_similarity(a_char_v, sub_b_char).ravel()
        cand_word_sim = cosine_similarity(a_word_v, sub_b_word).ravel()
        cand_tfidf_sim = 0.5 * cand_char_sim + 0.5 * cand_word_sim

        # Combined retrieval signal we use for the floor check
        combined_sim = 0.5 * cand_emb_sim + 0.5 * cand_tfidf_sim
        keep = combined_sim >= NAME_SIM_FLOOR
        if not keep.any():
            results.append(MatchResult(item_id_a=str(a_df.iloc[i]["item_id"]),
                                       chosen=None, runner_up=None,
                                       tier="no_candidates"))
            continue
        cand_idx_kept = cand_idx[keep]
        cand_emb_kept = cand_emb_sim[keep]
        cand_tfidf_kept = cand_tfidf_sim[keep]
        cand_combined_kept = combined_sim[keep]

        a_row = a_df.iloc[i]
        scored = []
        for j, idx in enumerate(cand_idx_kept):
            b_row = b_df.iloc[idx]
            c = score_pair(a_row, b_row,
                            emb_sim=float(cand_emb_kept[j]),
                            tfidf_sim=float(cand_tfidf_kept[j]),
                            combined_sim=float(cand_combined_kept[j]))
            scored.append(c)
        scored.sort(key=lambda c: -c.composite)

        best = scored[0]
        runner = scored[1] if len(scored) > 1 else None

        if best.composite >= TAU_HIGH:
            tier = "auto_accept"
        elif best.composite < TAU_LOW:
            tier = "auto_reject"
        else:
            tier = "ambiguous"

        results.append(MatchResult(item_id_a=str(a_row["item_id"]),
                                   chosen=best, runner_up=runner, tier=tier))
    return results


def score_pair(a: pd.Series, b: pd.Series, emb_sim: float, tfidf_sim: float,
               combined_sim: float) -> Candidate:
    """Composite score for an A,B pair.

    `emb_sim` and `tfidf_sim` are both in [0,1] (cosine on normalized vectors,
    cosine on TF-IDF respectively). `combined_sim` is their average, kept for
    the floor check and saved as `name_sim` for downstream debugging.
    """
    # Size match (binary)
    a_key = a.get("size_key")
    b_key = b.get("size_key")
    if a_key and b_key:
        size_match = (a_key == b_key)
        size_score = 1.0 if size_match else 0.0
    else:
        # Unknown size on either side -> neutral
        size_match = False
        size_score = 0.5

    # Brand: either same national brand, or both private label
    a_brand = a.get("brand_norm") or ""
    b_brand = b.get("brand_norm") or ""
    a_pl = bool(a.get("is_private_label"))
    b_pl = bool(b.get("is_private_label"))
    if a_pl and b_pl:
        brand_match = True
        brand_score = 1.0
    elif (not a_pl) and (not b_pl) and a_brand and b_brand:
        # National-vs-national: must match
        brand_match = (a_brand == b_brand) or (a_brand in b_brand) or (b_brand in a_brand)
        brand_score = 1.0 if brand_match else 0.0
    else:
        # private label vs national label is essentially never the same product
        brand_match = False
        brand_score = 0.0

    # Flag agreement
    flag_keys = ["flag_organic", "flag_gluten_free", "flag_frozen",
                 "flag_decaf", "flag_diet", "flag_kosher"]
    agree = sum(1 for k in flag_keys if bool(a.get(k)) == bool(b.get(k)))
    flag_score = agree / len(flag_keys)

    composite = (
        W_EMB * emb_sim
        + W_TFIDF * tfidf_sim
        + W_SIZE * size_score
        + W_BRAND * brand_score
        + W_FLAGS * flag_score
    )

    # Hard penalty: opposite size known is a deal-breaker for grocery.
    if a_key and b_key and a_key != b_key:
        composite *= 0.65

    # Hard penalty: private vs national mismatch
    if a_pl != b_pl:
        composite *= 0.6

    return Candidate(
        item_id_b=str(b.get("item_id")),
        emb_sim=emb_sim,
        tfidf_sim=tfidf_sim,
        name_sim=combined_sim,
        size_match=size_match,
        brand_match=brand_match,
        flag_agree=flag_score,
        composite=composite,
    )


# ---------------- top-level driver ----------------
def run_matching(a_norm: pd.DataFrame, b_norm: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    a_norm = a_norm.copy().reset_index(drop=True)
    b_norm = b_norm.copy().reset_index(drop=True)
    a_norm["bucket"] = [
        category_map.bucket_for_a(c0, c1) for c0, c1 in zip(a_norm["cat0"], a_norm["cat1"])
    ]
    b_norm["bucket"] = [
        category_map.bucket_for_b(c0, c1) for c0, c1 in zip(b_norm["cat0"], b_norm["cat1"])
    ]

    # Embed everything once. We only embed items in buckets that have B inventory —
    # there's no point spending CPU on Walmart-only categories.
    relevant_a = a_norm[a_norm["bucket"].isin(category_map.BUCKETS_WITH_B_INVENTORY)]
    relevant_b = b_norm[b_norm["bucket"].isin(category_map.BUCKETS_WITH_B_INVENTORY)]
    if verbose:
        print(f"\n[embed] computing MiniLM embeddings for {len(relevant_a):,} A + "
              f"{len(relevant_b):,} B items in matchable buckets")
    a_emb_full = embed.embed_dataframe(relevant_a, side="A", verbose=verbose)
    b_emb_full = embed.embed_dataframe(relevant_b, side="B", verbose=verbose)

    # Map original index -> embedding row
    a_idx_to_row = {idx: i for i, idx in enumerate(relevant_a.index)}
    b_idx_to_row = {idx: i for i, idx in enumerate(relevant_b.index)}

    all_results: list[MatchResult] = []
    rows: list[dict] = []

    for bucket in sorted(a_norm["bucket"].unique()):
        a_sub = a_norm[a_norm["bucket"] == bucket]
        b_sub = b_norm[b_norm["bucket"] == bucket]
        if verbose:
            print(f"\n[bucket={bucket}] A={len(a_sub):,} B={len(b_sub):,}")
        if bucket not in category_map.BUCKETS_WITH_B_INVENTORY or len(b_sub) == 0:
            for _, r in a_sub.iterrows():
                all_results.append(MatchResult(
                    item_id_a=str(r["item_id"]), chosen=None, runner_up=None,
                    tier="no_candidates"))
            continue

        a_emb_rows = np.array([a_idx_to_row[i] for i in a_sub.index], dtype=np.int64)
        b_emb_rows = np.array([b_idx_to_row[i] for i in b_sub.index], dtype=np.int64)
        a_emb_bucket = a_emb_full[a_emb_rows]
        b_emb_bucket = b_emb_full[b_emb_rows]

        t0 = time.time()
        bucket_results = match_bucket(a_sub.reset_index(drop=True),
                                       b_sub.reset_index(drop=True), bucket,
                                       a_emb=a_emb_bucket, b_emb=b_emb_bucket)
        if verbose:
            print(f"  -> matched in {time.time()-t0:.1f}s "
                  f"(auto_accept={sum(r.tier=='auto_accept' for r in bucket_results)}, "
                  f"ambiguous={sum(r.tier=='ambiguous' for r in bucket_results)}, "
                  f"reject={sum(r.tier=='auto_reject' for r in bucket_results)})")
        all_results.extend(bucket_results)

    # Build dataframe of all results (one row per A item)
    for r in all_results:
        row = {"item_id_a": r.item_id_a, "tier": r.tier}
        if r.chosen is not None:
            row.update({
                "item_id_b": r.chosen.item_id_b,
                "composite": r.chosen.composite,
                "emb_sim": r.chosen.emb_sim,
                "tfidf_sim": r.chosen.tfidf_sim,
                "name_sim": r.chosen.name_sim,
                "size_match": r.chosen.size_match,
                "brand_match": r.chosen.brand_match,
                "flag_agree": r.chosen.flag_agree,
            })
        else:
            row.update({"item_id_b": None, "composite": None,
                        "emb_sim": None, "tfidf_sim": None, "name_sim": None,
                        "size_match": None, "brand_match": None, "flag_agree": None})
        if r.runner_up is not None:
            row["runner_up_id_b"] = r.runner_up.item_id_b
            row["runner_up_composite"] = r.runner_up.composite
        else:
            row["runner_up_id_b"] = None
            row["runner_up_composite"] = None
        rows.append(row)

    return pd.DataFrame(rows)
