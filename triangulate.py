"""Second-judge triangulation against the existing labeled set.

Why this exists: the headline precision number (0.50) was produced by
gpt-5-nano labeling pairs that gpt-5-nano (the verification stage) decided to
keep. The same instrument on both sides means part of the noise floor is one
model agreeing with itself on edge cases (kosher certifications, marketing
modifiers, debatable variants). To estimate that floor, this script re-judges
the same labeled pairs with Claude Sonnet as an independent second judge.

What it does:
  1. Load `artifacts/labels_v.csv` (200 stratified pairs, gpt-5-nano labels).
  2. Load `artifacts/matches_full_v.csv` to get each labeled pair's
     post-verification tier.
  3. Filter to pairs that (a) are in the final matches.csv (tier in
     {auto_accept, verify_accept, verify_swap, llm_accept}) and (b) have a
     valid gpt-5-nano label (0 or 1, not -1 abstain).
  4. Re-judge each survivor with Claude Sonnet using the same strict prompt.
  5. Report: Sonnet precision, gpt-5-nano precision on the same items,
     agreement rate, per-stratum breakdowns. Write `triangulation_report.json`.

Cost: ~30-50 small Sonnet calls × ~600 tokens in / ~150 tokens out ≈ $0.10-0.30.
Time: ~3 min serial, ~30s parallel.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python triangulate.py
    python triangulate.py --max_pairs 50    # cap (default: all eligible)
    python triangulate.py --workers 8       # parallel calls
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"
CACHE_DIR = ARTIFACTS / "triangulation_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FINAL_TIERS = {"auto_accept", "verify_accept", "verify_swap", "llm_accept"}

# Claude Sonnet model. Override via --model or ANTHROPIC_MODEL env var.
DEFAULT_MODEL = "claude-sonnet-4-6"

SYSTEM = (
    "You are an expert grocery product matcher for retail price indexing. "
    "Given two products (one from store A, one from store B), decide whether "
    "they are 'the same product' — meaning a customer would treat them as "
    "interchangeable for shopping. Return strict JSON: "
    "{\"label\": 1 or 0, \"reason\": \"<short>\"}. "
    "Use label=1 only if brand class (national vs private label), product "
    "type, variant/flavor, AND size all agree. Multi-pack vs single is OK if "
    "per-unit size matches. Different flavors / different sizes / "
    "national-vs-private-label = label 0. Marketing modifiers (e.g. 'kosher', "
    "'new look') alone don't make products different if substance + size match."
)


def _cache_key(item_id_a: str, item_id_b: str, model: str) -> str:
    import hashlib
    h = hashlib.sha256(f"{model}|{item_id_a}|{item_id_b}".encode()).hexdigest()[:16]
    return h


def judge_with_sonnet(client, model: str, row: pd.Series,
                      max_retries: int = 3) -> dict:
    """Returns {'label': 0/1/-1, 'reason': str}. Caches to disk."""
    aid = str(row["item_id_a"])
    bid = str(row["item_id_b"])
    key = _cache_key(aid, bid, model)
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except Exception:
            pass

    user = (
        f"A: brand={row.get('brand_a','?')} | "
        f"size={row.get('size_a','?')} | "
        f"private_label={row.get('pl_a')} | "
        f"name={row.get('name_a','?')!r}\n\n"
        f"B: brand={row.get('brand_b','?')} | "
        f"size={row.get('size_b','?')} | "
        f"private_label={row.get('pl_b')} | "
        f"name={row.get('name_b','?')!r}\n\n"
        "JSON only."
    )

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=300,
                system=SYSTEM,
                messages=[{"role": "user", "content": user}],
            )
            txt = resp.content[0].text.strip()
            if txt.startswith("```"):
                txt = txt.strip("`").lstrip("json").strip()
            data = json.loads(txt)
            if "label" not in data:
                raise ValueError(f"missing label in {txt!r}")
            out = {"label": int(data["label"]), "reason": data.get("reason", "")}
            cache_file.write_text(json.dumps(out))
            return out
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 8))
    return {"label": -1, "reason": f"err: {last_err}"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", default=str(ARTIFACTS / "labels_v.csv"))
    p.add_argument("--matches_full", default=str(ARTIFACTS / "matches_full_v.csv"))
    p.add_argument("--out", default=str(ARTIFACTS / "labels_triangulated.csv"))
    p.add_argument("--report", default=str(ARTIFACTS / "triangulation_report.json"))
    p.add_argument("--model", default=os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL))
    p.add_argument("--max_pairs", type=int, default=50,
                   help="Cap on pairs to triangulate (default 50).")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    # Resolve the API key: env var first, else fall back to anthropic_creds.yaml
    # (mirrors the openai_creds.yaml pattern already in the repo).
    api_key = os.environ.get("ANTHROPIC_API_KEY") or ""
    if not api_key:
        creds_path = ROOT / "anthropic_creds.yaml"
        if creds_path.exists():
            try:
                import yaml
                with open(creds_path) as f:
                    cfg = yaml.safe_load(f) or {}
                api_key = (cfg.get("anthropic", {}) or {}).get("api_key", "") or ""
            except Exception as e:
                print(f"WARN: failed to read {creds_path}: {e}")
    if not api_key:
        print("ERROR: no Anthropic API key found.")
        print("  Either set ANTHROPIC_API_KEY env var, or create anthropic_creds.yaml:")
        print("    anthropic:")
        print("      api_key: sk-ant-...")
        sys.exit(2)
    os.environ["ANTHROPIC_API_KEY"] = api_key  # so the SDK picks it up

    try:
        import anthropic
    except ImportError:
        print("ERROR: anthropic SDK not installed. Run: pip install anthropic")
        sys.exit(2)

    print(f"Loading labels from {args.labels}...")
    labels = pd.read_csv(args.labels, dtype={"item_id_a": str, "item_id_b": str})
    print(f"  {len(labels):,} labeled pairs (gpt-5-nano)")

    print(f"\nLoading post-verification tiers from {args.matches_full}...")
    full = pd.read_csv(args.matches_full,
                       dtype={"item_id_a": str, "item_id_b": str},
                       low_memory=False)
    print(f"  {len(full):,} rows in matches_full_v.csv")

    # Inner-join on (item_id_a, item_id_b) — labels.item_id_b reflects the
    # B that gpt-5-nano scored. After verify_swap that may differ from
    # full.item_id_b, but for triangulation we judge the pair the original
    # label was on, which is what's already in labels_v.csv.
    merged = labels.merge(
        full[["item_id_a", "tier"]],
        on="item_id_a", how="left",
    )

    eligible = merged[
        merged["label"].isin([0, 1])
        & merged["tier"].isin(FINAL_TIERS)
    ].copy()
    print(f"\nEligible (final-tier ∩ valid label): {len(eligible):,}")
    print(f"  by stratum:\n{eligible['stratum'].value_counts().to_string()}")

    if args.max_pairs and len(eligible) > args.max_pairs:
        # Stratified subsample to keep the cap respect the bands
        rng_state = 7
        per_stratum = max(1, args.max_pairs // eligible["stratum"].nunique())
        picks = []
        for s, sub in eligible.groupby("stratum"):
            picks.append(sub.sample(min(per_stratum, len(sub)), random_state=rng_state))
        # If we're under cap, fill in with random remainder
        chosen = pd.concat(picks)
        if len(chosen) < args.max_pairs:
            remainder = eligible.drop(chosen.index)
            extra = remainder.sample(min(args.max_pairs - len(chosen), len(remainder)),
                                     random_state=rng_state + 1)
            chosen = pd.concat([chosen, extra])
        eligible = chosen
        print(f"  subsampled to {len(eligible):,} (max_pairs={args.max_pairs})")

    print(f"\nJudging {len(eligible):,} pairs with Claude Sonnet ({args.model})...")
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    t0 = time.time()
    results = [None] * len(eligible)
    completed = 0
    rows = list(eligible.itertuples(index=False))
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        future_to_i = {
            ex.submit(judge_with_sonnet, client, args.model, pd.Series(r._asdict())): i
            for i, r in enumerate(rows)
        }
        for fut in as_completed(future_to_i):
            i = future_to_i[fut]
            results[i] = fut.result()
            completed += 1
            if completed % 5 == 0 or completed == len(eligible):
                rate = completed / (time.time() - t0 + 1e-9)
                print(f"  {completed}/{len(eligible)} ({rate:.1f}/s)")
    elapsed = round(time.time() - t0, 1)
    print(f"  done in {elapsed}s")

    eligible = eligible.copy()
    eligible["label_sonnet"] = [r["label"] for r in results]
    eligible["reason_sonnet"] = [r["reason"] for r in results]

    # ---------- metrics ----------
    valid_both = eligible[
        eligible["label"].isin([0, 1]) & eligible["label_sonnet"].isin([0, 1])
    ]
    n = len(valid_both)
    nano_p = round(valid_both["label"].mean(), 3) if n else 0.0
    sonnet_p = round(valid_both["label_sonnet"].mean(), 3) if n else 0.0
    agree = int((valid_both["label"] == valid_both["label_sonnet"]).sum())
    agree_rate = round(agree / n, 3) if n else 0.0
    delta = round(sonnet_p - nano_p, 3)

    by_stratum = {}
    for s, sub in valid_both.groupby("stratum"):
        ss_n = len(sub)
        by_stratum[s] = {
            "n": ss_n,
            "nano_precision": round(sub["label"].mean(), 3) if ss_n else 0.0,
            "sonnet_precision": round(sub["label_sonnet"].mean(), 3) if ss_n else 0.0,
            "agreement": round((sub["label"] == sub["label_sonnet"]).sum() / ss_n, 3)
                         if ss_n else 0.0,
        }

    report = {
        "model": args.model,
        "n_eligible": int(len(eligible)),
        "n_valid_both_judges": n,
        "gpt5nano_precision": nano_p,
        "sonnet_precision": sonnet_p,
        "delta_sonnet_minus_nano": delta,
        "agreement_rate": agree_rate,
        "by_stratum": by_stratum,
        "elapsed_s": elapsed,
    }

    eligible.to_csv(args.out, index=False)
    Path(args.report).write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 64)
    print("TRIANGULATION REPORT")
    print("=" * 64)
    print(json.dumps(report, indent=2))
    print(f"\nWrote {args.out}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
