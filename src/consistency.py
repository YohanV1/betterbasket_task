"""Bidirectional consistency check.

Premise: a real product match should be symmetric. If A_i's best match is B_j,
then within B_j's local neighborhood, A_i should also be its best (or among its
top few) match in A. When this fails, one of two things is true:

  1. The match is wrong — A_i and B_j look similar but aren't actually the same
     product. (Common: same brand, slightly different variant.)
  2. The match is right but A has multiple very-similar items mapping to the
     same B item. (Common: Walmart sells "(3 pack) X" and "(6 pack) X" — both
     legitimately match B's single "X". Only one can be "the" reverse-match.)

Case 1 we want to filter out. Case 2 we want to keep. We use a soft check:
  - If B_j -> top-K in A includes A_i (any rank), keep.
  - Else, demote to a flag (lower confidence) but don't drop unconditionally.

The check is cheap because we already have all embeddings in memory.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import embed, normalize, category_map


def add_consistency(matches: pd.DataFrame,
                    a_norm: pd.DataFrame, b_norm: pd.DataFrame,
                    k_reverse: int = 10,
                    verbose: bool = True) -> pd.DataFrame:
    """Adds a `consistent` boolean column to `matches`.

    For each accepted match (A_i, B_j), checks whether A_i is among B_j's top-K
    nearest A items by embedding similarity. Inconsistencies get flagged.
    """
    accepted = matches[matches["item_id_b"].notna()].copy()
    if len(accepted) == 0:
        matches["consistent"] = pd.NA
        return matches

    # We need embeddings keyed by item_id, restricted to matchable buckets
    a_norm = a_norm.copy().reset_index(drop=True)
    b_norm = b_norm.copy().reset_index(drop=True)
    a_norm["bucket"] = [
        category_map.bucket_for_a(c0, c1) for c0, c1 in zip(a_norm["cat0"], a_norm["cat1"])
    ]
    b_norm["bucket"] = [
        category_map.bucket_for_b(c0, c1) for c0, c1 in zip(b_norm["cat0"], b_norm["cat1"])
    ]
    relevant_a = a_norm[a_norm["bucket"].isin(category_map.BUCKETS_WITH_B_INVENTORY)]
    relevant_b = b_norm[b_norm["bucket"].isin(category_map.BUCKETS_WITH_B_INVENTORY)]

    if verbose:
        print(f"[consistency] loading cached embeddings...")
    a_emb = embed.embed_dataframe(relevant_a, side="A", verbose=False)
    b_emb = embed.embed_dataframe(relevant_b, side="B", verbose=False)

    # Maps from item_id -> embedding row index
    a_id_to_row = {str(iid): i for i, iid in enumerate(relevant_a["item_id"])}
    b_id_to_row = {str(iid): i for i, iid in enumerate(relevant_b["item_id"])}
    # Reverse map: embedding row -> item_id (only for A side, that's what we'll lookup)
    a_row_to_id = {i: str(iid) for iid, i in a_id_to_row.items()}

    consistent = np.zeros(len(accepted), dtype=bool)

    if verbose:
        print(f"[consistency] checking {len(accepted):,} matches against B->A top-{k_reverse}")

    # We need: for each unique B_j in accepted, the top-K nearest A items.
    # Then check membership of the corresponding A_i.
    unique_b_ids = accepted["item_id_b"].astype(str).unique()
    b_rows = np.array([b_id_to_row[bid] for bid in unique_b_ids if bid in b_id_to_row],
                       dtype=np.int64)
    if len(b_rows) == 0:
        matches["consistent"] = pd.NA
        return matches

    # Reverse retrieval: B->A
    b_emb_subset = b_emb[b_rows]
    rev_idx, _ = embed.topk_candidates(b_emb_subset, a_emb, k=k_reverse)
    b_id_to_topa: dict[str, set[str]] = {}
    for j, bid in enumerate(unique_b_ids):
        if bid not in b_id_to_row:
            continue
        top_a_rows = rev_idx[list(unique_b_ids).index(bid)] if False else rev_idx[j]
        b_id_to_topa[bid] = {a_row_to_id[int(r)] for r in top_a_rows}

    for i, (_, row) in enumerate(accepted.iterrows()):
        bid = str(row["item_id_b"])
        aid = str(row["item_id_a"])
        topa = b_id_to_topa.get(bid)
        consistent[i] = (topa is not None and aid in topa)

    accepted["consistent"] = consistent

    # Merge back. NA for non-accepted rows.
    out = matches.merge(accepted[["item_id_a", "consistent"]], on="item_id_a", how="left")
    if verbose:
        n_consistent = int(consistent.sum())
        n_total = len(accepted)
        print(f"[consistency] consistent: {n_consistent:,} / {n_total:,} "
              f"({n_consistent/n_total*100:.1f}%) of accepted matches")
    return out
