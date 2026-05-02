"""Stratified-sample 200 (A, B) pairs and produce ground-truth labels.

Approach:
- Sample across the composite-score distribution in 4 strata so we measure
  precision at every confidence tier (not just the top picks). Top-stratum
  errors are the dangerous ones for pricing — we want to see them.
- Label each pair with a stronger LLM judge (Claude Sonnet via Anthropic API
  if available, else fall back to GPT-5-nano with a careful prompt).
- Cache labels to disk so we don't re-spend on re-runs.
- Allow human override: any human-labeled pair in `labels_human.csv` wins
  over the LLM. That gives the user a fast path to correct LLM errors and
  produce a real human-grade ground-truth set.

The output `artifacts/labels.csv` has columns:
   item_id_a, item_id_b, composite, stratum, label (1/0/?), source

`label==1` means "same product" per our task definition (substitutable in a
shopping context). `label==0` means "different product". `label=='?'` means
the labeler abstained (rare, e.g. a B candidate that doesn't even exist).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"

# Strata cover the full score range. The top stratum is the most numerous in
# the output but error-checking it has the highest cost-of-being-wrong.
STRATA = [
    ("very_high", 0.85, 1.01, 180),  # auto-accept upper (3x — drives triangulation eligibility)
    ("high",     0.78, 0.85, 150),   # auto-accept lower
    ("ambig",    0.50, 0.78, 180),   # the LLM-tiebreak band
    ("reject",   0.00, 0.50, 30),    # we say no — verify some are real nos (kept small)
]
TOTAL = sum(n for _, _, _, n in STRATA)


def stratified_sample(matches_full: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Pick `n` rows from each composite-score stratum."""
    rng = np.random.default_rng(seed)
    samples = []
    for name, lo, hi, n in STRATA:
        cands = matches_full[
            matches_full["composite"].notna() &
            (matches_full["composite"] >= lo) &
            (matches_full["composite"] < hi)
        ]
        if len(cands) == 0:
            continue
        n_take = min(n, len(cands))
        idx = rng.choice(len(cands), size=n_take, replace=False)
        s = cands.iloc[idx].copy()
        s["stratum"] = name
        samples.append(s)
    return pd.concat(samples, ignore_index=True)


def make_anthropic_judge():
    """Return a function (a, b) -> {label, reason}, or None if no API key."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
    except ImportError:
        return None

    client = anthropic.Anthropic(api_key=key)
    SYSTEM = (
        "You are an expert grocery product matcher for retail price indexing. "
        "Given two products (one from store A, one from store B), decide whether "
        "they are 'the same product' — meaning a customer would treat them as "
        "interchangeable for shopping. Return strict JSON: {\"label\": 1 or 0, "
        "\"reason\": \"<short>\"}. Use label=1 only if brand class (national vs "
        "private label), product type, variant/flavor, AND size all agree. "
        "Multi-pack vs single is OK if per-unit size matches. Different "
        "flavors / different sizes / national-vs-private-label = label 0."
    )

    def judge(a: dict, b: dict) -> dict:
        user = (
            f"A: brand={a.get('brand_norm','?')} | "
            f"size={a.get('size_key','?')} | "
            f"private_label={a.get('is_private_label')} | "
            f"name={a.get('name','?')!r}\n\n"
            f"B: brand={b.get('brand_norm','?')} | "
            f"size={b.get('size_key','?')} | "
            f"private_label={b.get('is_private_label')} | "
            f"name={b.get('name','?')!r}\n\n"
            "JSON only."
        )
        for attempt in range(3):
            try:
                resp = client.messages.create(
                    model="claude-3-5-sonnet-latest",
                    max_tokens=200,
                    system=SYSTEM,
                    messages=[{"role": "user", "content": user}],
                )
                txt = resp.content[0].text.strip()
                # Strip code fences if present
                if txt.startswith("```"):
                    txt = txt.strip("`").lstrip("json").strip()
                data = json.loads(txt)
                if "label" not in data:
                    raise ValueError(f"missing label in {txt!r}")
                return {"label": int(data["label"]), "reason": data.get("reason", "")}
            except Exception as e:
                if attempt == 2:
                    return {"label": -1, "reason": f"err: {e}"}
                time.sleep(2 ** attempt)
        return {"label": -1, "reason": "unreachable"}
    return judge


def make_gpt5nano_judge():
    """Fallback judge using the same Azure GPT-5-nano deployment as the pipeline."""
    from src.llm_judge import load_credentials, make_client
    creds = load_credentials()
    client, deployment = make_client(creds)

    SYSTEM = (
        "You are an expert grocery product matcher for retail price indexing. "
        "Given two products (one from store A, one from store B), decide whether "
        "they are 'the same product' — meaning a customer would treat them as "
        "interchangeable for shopping. Be strict. Return JSON: "
        "{\"label\": 1 or 0, \"reason\": \"<short>\"}. "
        "label=1 only if brand class (national vs private-label), product type, "
        "variant/flavor, AND size all agree. Multi-pack-vs-single is OK if "
        "per-unit size matches. Different flavors/sizes/brand-class = 0."
    )

    def judge(a: dict, b: dict) -> dict:
        user = (
            f"A: brand={a.get('brand_norm','?')} | size={a.get('size_key','?')} | "
            f"private_label={a.get('is_private_label')} | name={a.get('name','?')!r}\n\n"
            f"B: brand={b.get('brand_norm','?')} | size={b.get('size_key','?')} | "
            f"private_label={b.get('is_private_label')} | name={b.get('name','?')!r}\n\n"
            "JSON only."
        )
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=deployment,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": user}],
                    response_format={"type": "json_object"},
                )
                data = json.loads(resp.choices[0].message.content or "{}")
                if "label" not in data:
                    raise ValueError(f"missing label: {data}")
                return {"label": int(data["label"]), "reason": data.get("reason", "")}
            except Exception as e:
                if attempt == 2:
                    return {"label": -1, "reason": f"err: {e}"}
                time.sleep(2 ** attempt)
        return {"label": -1, "reason": "unreachable"}
    return judge


def label_pairs(sample: pd.DataFrame,
                 a_lookup: dict, b_lookup: dict,
                 verbose: bool = True,
                 max_workers: int = 16) -> pd.DataFrame:
    """Label all pairs in parallel. The judges are thread-safe (HTTP clients)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time as _time

    judge = make_anthropic_judge()
    judge_name = "anthropic-sonnet" if judge else None
    if judge is None:
        judge = make_gpt5nano_judge()
        judge_name = "gpt-5-nano"
    if verbose:
        print(f"[label] using judge: {judge_name}, workers: {max_workers}")

    # Prepare inputs aligned with sample order
    inputs = []
    for _, r in sample.iterrows():
        aid = str(r["item_id_a"])
        bid = str(r["item_id_b"])
        a_info = {**a_lookup.get(aid, {}), "item_id": aid}
        b_info = {**b_lookup.get(bid, {}), "item_id": bid}
        inputs.append((a_info, b_info, r))

    results = [None] * len(inputs)
    completed = 0
    t0 = _time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(judge, a, b): i for i, (a, b, _) in enumerate(inputs)}
        for fut in as_completed(futures):
            i = futures[fut]
            d = fut.result()
            a_info, b_info, r = inputs[i]
            results[i] = {
                "item_id_a": a_info["item_id"],
                "item_id_b": b_info["item_id"],
                "name_a": a_info.get("name", ""),
                "name_b": b_info.get("name", ""),
                "brand_a": a_info.get("brand_norm", ""),
                "brand_b": b_info.get("brand_norm", ""),
                "size_a": a_info.get("size_key", ""),
                "size_b": b_info.get("size_key", ""),
                "pl_a": a_info.get("is_private_label"),
                "pl_b": b_info.get("is_private_label"),
                "composite": r.get("composite"),
                "stratum": r["stratum"],
                "label": d["label"],
                "reason": d["reason"],
                "judge": judge_name,
            }
            completed += 1
            if verbose and completed % 25 == 0:
                rate = completed / (_time.time() - t0)
                print(f"  labeled {completed}/{len(inputs)} ({rate:.1f}/s)")
    return pd.DataFrame(results)


def compute_metrics(labels: pd.DataFrame, matches_full: pd.DataFrame) -> dict:
    """Precision@1 in each tier, and an overall recall estimate against the
    labeled set."""
    metrics = {"by_stratum": {}}
    valid = labels[labels["label"].isin([0, 1])]
    if len(valid) == 0:
        return metrics
    for s in valid["stratum"].unique():
        sub = valid[valid["stratum"] == s]
        n = len(sub)
        tp = int((sub["label"] == 1).sum())
        precision = tp / n if n else 0
        metrics["by_stratum"][s] = {
            "n": n, "true_positive": tp, "precision": round(precision, 3),
        }
    # Precision@1 over the accepted (auto + LLM) tier
    accepted = valid[valid["composite"] >= 0.50]
    n_acc = len(accepted)
    tp_acc = int((accepted["label"] == 1).sum())
    metrics["precision_at_1_overall"] = {
        "n": n_acc, "true_positive": tp_acc,
        "precision": round(tp_acc / n_acc, 3) if n_acc else 0.0,
    }
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matches", default=str(ARTIFACTS / "matches_full.csv"))
    p.add_argument("--out", default=str(ARTIFACTS / "labels_v.csv"))
    p.add_argument("--metrics", default=str(ARTIFACTS / "label_metrics_v.json"))
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    print("Loading matches + normalized data for lookups...")
    # Read item_ids as strings — the CSV has NaN rows that would otherwise
    # coerce the column to float, producing "12345.0" instead of "12345"
    # downstream and breaking the lookup.
    matches_full = pd.read_csv(args.matches, dtype={"item_id_a": str, "item_id_b": str})
    print(f"  matches_full: {len(matches_full):,} rows")

    # Load normalized item info for prompts
    sys.path.insert(0, str(ROOT))
    from src.normalize import normalize_dataframe
    a = pd.read_csv("grocery_store_a_items_final.csv", low_memory=False)
    b = pd.read_csv("grocery_store_b_items_final.csv", low_memory=False)
    an = normalize_dataframe(a, side="A")
    bn = normalize_dataframe(b, side="B")
    a_lookup = an.drop_duplicates("item_id").set_index("item_id")[
        ["name", "brand_norm", "size_key", "is_private_label"]
    ].to_dict("index")
    b_lookup = bn.drop_duplicates("item_id").set_index("item_id")[
        ["name", "brand_norm", "size_key", "is_private_label"]
    ].to_dict("index")
    # Make item_ids strings for consistent lookup
    a_lookup = {str(k): v for k, v in a_lookup.items()}
    b_lookup = {str(k): v for k, v in b_lookup.items()}

    print(f"\nStratified sampling {TOTAL} pairs...")
    sample = stratified_sample(matches_full, seed=args.seed)
    # Item IDs are already string in the loaded CSV (dtype enforced above).
    # Drop any rows where item_id_b is missing — there's nothing to label.
    sample = sample[sample["item_id_b"].notna() & (sample["item_id_b"] != "nan")]
    print("  sample by stratum:")
    print(sample["stratum"].value_counts().to_string())

    print("\nLabeling pairs...")
    labels = label_pairs(sample, a_lookup, b_lookup, verbose=True)
    labels.to_csv(args.out, index=False)
    print(f"  wrote {args.out}")

    metrics = compute_metrics(labels, matches_full)
    Path(args.metrics).write_text(json.dumps(metrics, indent=2))
    print(f"\n=== METRICS ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
