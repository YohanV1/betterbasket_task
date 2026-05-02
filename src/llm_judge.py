"""GPT-5-nano tiebreaker for ambiguous matches.

Called only on the `tier == 'ambiguous'` middle band — items whose composite score
is between TAU_LOW and TAU_HIGH. The judge sees the A item and its top candidates
from B, then picks one or returns 'none'.

Design choices:
- Strict JSON output via `response_format={"type": "json_object"}`.
- Caches results to disk by hash(prompt) so reruns don't re-spend.
- Bounded concurrency via thread pool. The Azure deployment isn't documented for
  rate limits in the task, so we keep concurrency modest and retry with backoff.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import yaml
from openai import OpenAI

CACHE_DIR = Path("artifacts/llm_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_credentials(path: str = "openai_creds.yaml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg["openai"]


def make_client(creds: dict) -> tuple[OpenAI, str]:
    client = OpenAI(base_url=creds["endpoint"], api_key=creds["api_key"])
    return client, creds["deployment_name"]


SYSTEM_PROMPT = (
    "You are a grocery product-matching assistant for retail price indexing. Given a "
    "target product from store A and 2-5 candidate products from store B, pick the "
    "single B candidate that a customer would consider \"essentially the same product\".\n\n"
    "Each item line includes a structured `size=` field — TREAT THAT AS AUTHORITATIVE. "
    "If size=1gal on both sides, the size matches even if the name string omits it.\n\n"
    "Two valid kinds of matches:\n"
    "  1. EXACT (national brand): brand, variant/flavor, and size all agree.\n"
    "     - McCormick Black Pepper 4oz  <->  McCormick Black Pepper 4oz   ✓\n"
    "  2. NON-EXACT (private-label cross-match): both items are store private-label, "
    "     and product type/variant/size match. Brand names WILL differ — that's expected.\n"
    "     - Great Value Tomato Sauce 8oz       <->  Wegmans Tomato Sauce 8oz       ✓\n"
    "     - Great Value Whole Milk 1gal        <->  Wegmans Whole Milk 1gal        ✓\n"
    "     - Marketside Organic Baby Spinach 5oz <->  Wegmans Organic Baby Spinach 5oz ✓\n\n"
    "REJECT (return 'none') when ANY of:\n"
    "  - One item is national-brand and the other is private-label.\n"
    "  - Variant/flavor/type differs (peanut-butter vs cherry; whole milk vs 2%; etc.).\n"
    "  - Size mismatches (different size= values).\n"
    "  - Form/state differs (frozen vs fresh; ground vs whole-bean coffee).\n\n"
    "Output strict JSON only: {\"match_id\": \"<B id or 'none'>\", \"reason\": \"<short>\"}."
)


def build_user_prompt(a_row: dict, candidates: list[dict]) -> str:
    def fmt_item(prefix: str, r: dict) -> str:
        size = r.get("size_key") or "?"
        brand = r.get("brand_norm") or "?"
        pl = "(private-label)" if r.get("is_private_label") else "(national)"
        return (f"{prefix} id={r['item_id']} | brand={brand} {pl} | "
                f"size={size} | name=\"{r['name']}\"")

    lines = [fmt_item("A:", a_row), "", "Candidates:"]
    for c in candidates:
        lines.append(fmt_item("B-", c))
    lines.append("")
    lines.append("Which B id matches A? Or 'none'. JSON only.")
    return "\n".join(lines)


def cache_key(a_row: dict, candidates: list[dict]) -> str:
    payload = {
        "a": {k: a_row.get(k) for k in ("item_id", "name", "brand_norm", "size_key", "is_private_label")},
        "b": [{k: c.get(k) for k in ("item_id", "name", "brand_norm", "size_key", "is_private_label")} for c in candidates],
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def call_judge(client: OpenAI, deployment: str, a_row: dict, candidates: list[dict],
               max_retries: int = 3) -> dict:
    """Returns dict {match_id, reason}. Caches to disk."""
    key = cache_key(a_row, candidates)
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except Exception:
            pass

    user = build_user_prompt(a_row, candidates)
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
            )
            txt = resp.choices[0].message.content or "{}"
            data = json.loads(txt)
            # validate
            if not isinstance(data, dict) or "match_id" not in data:
                raise ValueError(f"bad response: {txt!r}")
            cache_file.write_text(json.dumps(data))
            return data
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 8))
    # If all retries failed, return 'none' to fail closed
    return {"match_id": "none", "reason": f"llm_error: {last_err}"}


def judge_batch(items: list[tuple[dict, list[dict]]],
                creds: Optional[dict] = None,
                max_workers: int = 8, verbose: bool = True) -> list[dict]:
    """Judge many (a_row, candidates) tuples in parallel. Returns list aligned with input."""
    if creds is None:
        creds = load_credentials()
    client, deployment = make_client(creds)

    results = [None] * len(items)
    completed = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_i = {
            ex.submit(call_judge, client, deployment, a, c): i
            for i, (a, c) in enumerate(items)
        }
        for fut in as_completed(future_to_i):
            i = future_to_i[fut]
            results[i] = fut.result()
            completed += 1
            if verbose and completed % 100 == 0:
                rate = completed / (time.time() - t0 + 1e-9)
                print(f"  judge: {completed}/{len(items)} ({rate:.1f}/s)")
    return results
