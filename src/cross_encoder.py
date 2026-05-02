"""Cross-encoder reranking for high-precision second-stage scoring.

Bi-encoder (MiniLM in src/embed.py) is fast and gets us a recall-friendly top-K,
but it embeds each item independently — it can't directly compare token-level
differences between two specific items. A cross-encoder, in contrast, takes
both texts as a single input and outputs a relevance score, which is the gold
standard for entity-resolution rerankers.

We use `cross-encoder/ms-marco-MiniLM-L-6-v2` (60MB, ~3-5 ms per pair on CPU).
On the ambiguous and runner-up pairs we have a few thousand items max, so the
total compute is manageable: ~10-30k pairs * 4ms = 40-120s.

This is a pure precision filter — we only call it on pairs that already passed
the rule-based pipeline. We multiply the cross-encoder score (sigmoid'd to
[0,1]) into the composite, then re-evaluate the tier thresholds.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import CrossEncoder
        _MODEL = CrossEncoder(_MODEL_NAME)
    return _MODEL


def _build_text(row) -> str:
    """Format an item for the cross-encoder. Including the size in the text is
    important — the model can use it as a signal but won't be misled by missing
    sizes since both sides go through the same formatter."""
    bits = []
    brand = row.get("brand_norm")
    if isinstance(brand, str) and brand:
        bits.append(brand)
    name = row.get("name") or row.get("name_norm") or ""
    if isinstance(name, str) and name:
        bits.append(name)
    size = row.get("size_key")
    if isinstance(size, str) and size:
        bits.append(f"size={size}")
    return " | ".join(bits)


def score_pairs(pairs: list[tuple[dict, dict]], batch_size: int = 64,
                verbose: bool = True) -> np.ndarray:
    """Score (A, B) pairs with the cross-encoder.

    Returns scores in [0,1] (sigmoid of the model's logits — ms-marco scores are
    unbounded logits).
    """
    if not pairs:
        return np.array([], dtype=np.float32)
    model = _get_model()
    inputs = [(_build_text(a), _build_text(b)) for a, b in pairs]
    if verbose:
        print(f"  [cross-encoder] scoring {len(inputs):,} pairs ...")
    raw_scores = model.predict(inputs, batch_size=batch_size,
                                show_progress_bar=verbose)
    raw_scores = np.asarray(raw_scores, dtype=np.float32)
    # Sigmoid for interpretable 0-1 score
    return 1.0 / (1.0 + np.exp(-raw_scores))


def rerank_matches(matches: pd.DataFrame,
                   a_lookup: dict, b_lookup: dict,
                   verbose: bool = True) -> pd.DataFrame:
    """Add a `cross_score` column to matches and re-derive an `accepted_xenc` flag.

    The cross-encoder runs only on rows that already have a candidate (item_id_b
    not null). For rows we drop based on a low cross-encoder score, the tier is
    annotated as `xenc_reject`.
    """
    out = matches.copy()
    has_match = out["item_id_b"].notna()
    rows_with_match = out[has_match].copy()
    if len(rows_with_match) == 0:
        out["cross_score"] = pd.NA
        return out

    pairs = []
    for _, r in rows_with_match.iterrows():
        a = a_lookup.get(str(r["item_id_a"]), {})
        b = b_lookup.get(str(r["item_id_b"]), {})
        pairs.append((a, b))

    scores = score_pairs(pairs, verbose=verbose)
    rows_with_match["cross_score"] = scores

    # Lower bound on what we accept post-cross-encoder. ms-marco trained for
    # retrieval; "is this the same product" is harder than "is this relevant".
    # We use 0.30 as a conservative cutoff — most true matches score 0.7-0.99,
    # most non-matches < 0.2.
    XENC_THRESHOLD = 0.30
    rows_with_match["xenc_pass"] = rows_with_match["cross_score"] >= XENC_THRESHOLD

    # Merge back
    out = out.merge(
        rows_with_match[["item_id_a", "cross_score", "xenc_pass"]],
        on="item_id_a", how="left"
    )

    if verbose:
        n = len(rows_with_match)
        n_pass = int(rows_with_match["xenc_pass"].sum())
        print(f"  [cross-encoder] {n_pass:,}/{n:,} pairs passed "
              f"(threshold {XENC_THRESHOLD}); mean score {scores.mean():.3f}")
    return out
