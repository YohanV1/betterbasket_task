"""Sentence-transformer embeddings for primary candidate retrieval.

Uses `sentence-transformers/all-MiniLM-L6-v2` (384-d) — small enough to run on CPU
in a few minutes for 290k items, and well-trained on retrieval-style tasks. We
embed once, cache to disk (np.float32 .npy), and reuse on every subsequent run.

Why embeddings over TF-IDF for retrieval:
- Catches semantic equivalence that TF-IDF misses ("Welch's 100% Grape Juice"
  ↔ "Welch's Concord Grape Juice", "Cheerios" ↔ "Cheerios Cereal").
- Robust to word-order shuffling that TF-IDF char n-grams handle but word
  n-grams penalize.
- Standard tool for product-matching / entity-resolution at scale.

We keep TF-IDF (char + word) as auxiliary features in the composite score —
TF-IDF is much sharper at exact size/brand token overlap that embeddings smooth
out, so the two are complementary.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

CACHE_DIR = Path("artifacts/embed_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Suppress HF telemetry noise; we don't need their analytics
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(_MODEL_NAME)
    return _MODEL


def _build_text(row: pd.Series) -> str:
    """The string we feed the encoder. Brand + name + size_key gives the model
    everything it needs without overwhelming with category boilerplate."""
    bits = []
    brand = row.get("brand_norm")
    if isinstance(brand, str) and brand:
        bits.append(brand)
    name = row.get("name_norm") or row.get("name") or ""
    if isinstance(name, str) and name:
        bits.append(name)
    size = row.get("size_key")
    if isinstance(size, str) and size:
        bits.append(size)
    return " | ".join(bits)


def _cache_key(df: pd.DataFrame, side: str) -> str:
    """A deterministic fingerprint of the (text, item_ids) so the cache invalidates
    when normalization changes. Keeps cache reuse safe across multi-pack tweaks."""
    h = hashlib.sha256()
    h.update(side.encode())
    h.update(_MODEL_NAME.encode())
    # Hash a sample of the texts + the row count — full hash is too slow on 290k.
    # The combination is unique enough for our purposes (collisions vanishingly rare).
    h.update(str(len(df)).encode())
    sample_idx = np.linspace(0, len(df) - 1, num=min(500, len(df)), dtype=int)
    for i in sample_idx:
        row = df.iloc[i]
        h.update(str(row.get("item_id")).encode())
        h.update(_build_text(row).encode())
    return h.hexdigest()[:16]


def embed_dataframe(df: pd.DataFrame, side: str, batch_size: int = 256,
                     verbose: bool = True) -> np.ndarray:
    """Return an (n, 384) float32 matrix of L2-normalized embeddings, one per row.

    Caches to disk keyed on (side, length, content fingerprint). Re-runs are free.
    """
    key = _cache_key(df, side)
    cache_file = CACHE_DIR / f"{side}_{key}.npy"
    if cache_file.exists():
        if verbose:
            print(f"  [embed/{side}] cache hit: {cache_file.name}")
        return np.load(cache_file)

    if verbose:
        print(f"  [embed/{side}] encoding {len(df):,} rows ...")
    model = _get_model()
    texts = df.apply(_build_text, axis=1).tolist()
    emb = model.encode(
        texts, batch_size=batch_size, show_progress_bar=verbose,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)
    np.save(cache_file, emb)
    if verbose:
        print(f"  [embed/{side}] wrote {cache_file.name} ({emb.nbytes/1e6:.1f} MB)")
    return emb


def topk_candidates(A: np.ndarray, B: np.ndarray, k: int = 10,
                    chunk: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """For each row of A, return (top_k_indices_into_B, top_k_similarities).

    Uses dense matmul (vectors are L2-normalized so dot product = cosine).
    Chunking keeps peak memory bounded.
    """
    n_a = A.shape[0]
    n_b = B.shape[0]
    k = min(k, n_b)
    top_idx = np.zeros((n_a, k), dtype=np.int32)
    top_sim = np.zeros((n_a, k), dtype=np.float32)

    for start in range(0, n_a, chunk):
        end = min(start + chunk, n_a)
        sims = A[start:end] @ B.T  # (chunk, n_b)
        if n_b <= k:
            order = np.argsort(-sims, axis=1)
        else:
            part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
            row_ix = np.arange(part.shape[0])[:, None]
            seg = sims[row_ix, part]
            ord_ = np.argsort(-seg, axis=1)
            order = part[row_ix, ord_]
        top_idx[start:end] = order[:, :k]
        row_ix = np.arange(end - start)[:, None]
        top_sim[start:end] = sims[row_ix, order[:, :k]]
    return top_idx, top_sim
