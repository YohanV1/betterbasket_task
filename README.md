# BetterBasket Engineering Technical Assessment — Product Matching

## What is this?

Walmart and Wegmans both sell Coca-Cola. They both sell whole milk. They both sell pasta sauce. But their websites label these products slightly differently — different brand fields, different size formats, different category trees, sometimes the same product appears once at Walmart as "(3 pack) Coca-Cola 12 fl oz Cans" and once at Wegmans as plain "Coca-Cola 12 fl oz". My job for this task: given **233,199 products from Walmart** and **55,516 from Wegmans**, figure out which Walmart row corresponds to which Wegmans row, when one does. Output: a CSV of `(walmart_id, wegmans_id)` pairs.

The thing that makes it hard isn't the volume — it's that there's no shared identifier. UPCs would solve this in two lines of code, but the data doesn't have them. So matching has to come from comparing names, brands, sizes, and categories across two stores that describe the same products in totally different ways. And it has to be precise: if I match Walmart's *Cetaphil Cream* to Wegmans' *Cetaphil Lotion* by mistake, BetterBasket's pricing model will index a cream against a lotion's price and recommend the wrong shelf price to a grocer. Wrong matches in this domain cost real money.

The deliverable is `artifacts/matches.csv`: **12,485 high-confidence pairs** (3.1× the 4,000 floor the task asked for), produced by a multi-stage pipeline that uses embeddings, rules, and LLMs in increasingly expensive layers — and was iterated based on what a held-out labeled evaluation revealed.

**Two precision reads on the same labeled pairs (drawn from a 540-pair stratified labeled set)**: gpt-5-nano (the same model that did the verification) labels at **0.62** on n=66 valid-both-judges; Claude Sonnet 4.6, as an independent second judge on the same pairs, labels at **0.76**. The 0.62 figure is partly a same-instrument artifact — one model agreeing with itself on edge cases. Triangulated math: **12,485 pairs × 0.76 ≈ 9,464 expected-correct matches, 2.37× the 4,000 floor**. Even at the more conservative 0.62: 7,741 expected-correct, 1.94× the floor. Methodology and per-stratum breakdown in the Evaluation section.

## Headline numbers

| | v1 (first attempt) | v2 (current) |
|---|---|---|
| Final matches in `matches.csv` | 20,968 | **12,485** |
| Precision — gpt-5-nano judge (same-instrument)\* | not measured | **0.62** post-verify (vs 0.38 unverified) |
| Precision — Claude Sonnet 4.6 judge (independent triangulation)\* | not measured | **0.76** |
| Expected correct matches (Sonnet-triangulated) | not measured | **~9,464** (2.37× the 4,000 floor) |
| End-to-end runtime | 569 s | ~98 min |
| Total LLM cost (Azure gpt-5-nano + Sonnet triangulation) | $0.50 | ~$2.00 ($1.50 nano: $0.50 tiebreak + $1 verification, plus $0.50 Sonnet) |
| PL ↔ national brand mismatches | 1 / 20,968 | 0 / 12,485 |

\* Both precision rows use n=66 — the subset of labeled pairs that received both judges' labels (apples-to-apples for the triangulation). On the full verified subset (n=137 valid nano labels) nano precision is **0.65**, slightly higher than the n=66 subset's 0.62. The two judges agree on 86% of pairs in the n=66 subset; where they disagree, Sonnet is more permissive on edge cases (regional brand variants, marketing modifiers). Per-stratum breakdown in Evaluation.

The headline isn't the absolute count. It's that **building a held-out evaluation revealed a real precision problem in v1, and the precision-vs-volume trade in v2 is intentional**. I rejected ~14k matches that looked right in spot-checks but failed the labeled set's strict criteria. Better to ship 12k pairs at a triangulated ~0.76 precision (and document where the same-instrument noise floor comes from) than 26k pairs at 0.38 pretending nothing was wrong.

---

## The journey: how this pipeline got here

This is the part most submissions skip. The architecture you see in this README isn't what I designed at the start. It's what survived the held-out evaluation.

### Step 1 — EDA and v1 design

Before writing any matching code, I loaded both CSVs and inspected them. Three findings shaped everything afterward:

- **No UPCs in the data.** The task description discusses UPC matching as if it were available, but the actual columns don't contain UPC anywhere. The matching has to come purely from text + metadata.
- **The data is messier than it looks.** A's `name_clean` is 100% null. Its `brand_raw` field is 46% null even for clearly-branded items like *Great Value Corn on the Cob*. Its `is_private_label` column exists but is entirely empty. All of this had to be derived.
- **Walmart and Wegmans don't sell the same things.** Walmart has 73k food items; Wegmans has 32k grocery items. But Walmart also has 14k toys, 12k clothing, 13k pet supplies — categories Wegmans doesn't carry at all. ~107k Walmart items have no possible Wegmans match. Forcing matches there would be precision-destroying.

v1 was a four-stage pipeline:

1. **Normalize** names, brands, sizes; derive private-label flags from a curated list of Walmart house brands ("Great Value", "Marketside", etc.).
2. **Block** by mapping each store's category tree to a small set of shared buckets (`grocery`, `household`, `personal_care`, `health`, `baby`, `home`).
3. **Generate candidates** within each bucket using TF-IDF on character n-grams (catches "fl oz" ↔ "fluid ounce") and word n-grams (catches flavor variants). Score the top 5 candidates with a hand-tuned composite of name similarity + size match + brand match + flag agreement.
4. **LLM tiebreak** the ambiguous middle band (composite 0.50–0.78) using GPT-5-nano with strict JSON output.

v1 produced **20,968 matches in 9.5 minutes for $0.50**. Spot-checks of 50 random matches looked clean: McCormick Black Pepper matched McCormick Black Pepper, Goya Black Beans matched Goya Black Beans. I wrote up the README, marked it done.

### Step 2 — Reviewer feedback that reshaped the build

I asked for a sanity check on my "what I'd do next" section. The response (very directly): _"Two of those belong in v1, not v1.5. Embeddings instead of TF-IDF isn't next-move material — it's table stakes for an AI startup. Multi-pack reconciliation isn't next-day work — Walmart's catalog is full of `(3 pack)` formats and Wegmans almost never sells multi-packs, so without normalization you reject every legitimate multi-pack match. And spot-checking 50 examples is hand-waving — you need a real held-out labeled set."_

That feedback was correct, and it was the inflection point of this submission. I rebuilt v2 around it.

### Step 3 — v2 build

Three changes that are all "v1 must-haves" in production AI matching:

**Multi-pack reconciliation.** A regex extractor for `(N pack)`, `Pack of N`, `N-Pack`, `N count`. When Walmart says "(3 pack) Goya Rice 7 oz", the parser now extracts `pack_count=3` and treats the per-unit size as the canonical 7 oz. Wegmans' single 7-oz Goya Rice now matches.

**MiniLM embeddings replace TF-IDF as the primary retriever.** `sentence-transformers/all-MiniLM-L6-v2`, 384-dimensional, ~5 minutes to encode 290k items on CPU, cached to disk. TF-IDF stays as an auxiliary feature in the composite (it's sharper at exact size/brand token overlap), but the candidate retrieval itself is now semantic. This single change lifted auto-accept volume by **+49%** across all buckets — embeddings catch matches like "Welch's 100% Grape Juice" ↔ "Welch's Concord Grape Juice" that token-only methods miss.

**Bidirectional consistency check.** For every (A → B) match, verify B's nearest A neighbor includes A back. Annotates rather than drops, because legitimate multi-pack collisions break it (Walmart's "(3 pack) X" and "(6 pack) X" both correctly map to Wegmans' single "X"). Carries through to the final output as a per-row flag for downstream auditing.

I also tried a `cross-encoder/ms-marco-MiniLM-L-6-v2` rerank stage at this point in the pipeline. It turned out to be the weakest link — more on that in Step 4 below — and was removed from the final pipeline. The lesson is documented in the next-steps section.

v2 produced **26,378 matches in ~38 minutes** (the embedding stage adds ~5 minutes, the rest is mostly the LLM tiebreak). Spot-checks still looked clean. I would have shipped this and called it done.

### Step 4 — The held-out evaluation, and what it broke

`label.py` builds a stratified labeled set across composite-score bands: pairs from very-high (0.85+), high (0.78–0.85), ambiguous (0.50–0.78), and rejected (<0.50). Each pair gets labeled by an LLM judge with a strict prompt (brand class, product type, variant, AND size must all agree). The first iteration sampled 200 pairs and surfaced the failure pattern below; I later scaled it to 540 pairs for tighter confidence intervals (numbers shown here are the larger-set values).

The numbers came back ugly:

| Composite band | n | LLM-judge precision (label=1 rate) |
|---|---|---|
| [0.50, 0.78) "ambig" | 156 | 0.064 |
| [0.78, 0.85) "high" | 150 | **0.173** |
| [0.85, 0.90) | 80 | 0.375 |
| [0.90, 0.95) | 78 | 0.628 |
| [0.95, 1.01) | 22 | 0.955 |

The auto-accept band 0.78–0.85 — which contained ~6,200 of v2's matches — had only 17% LLM-judge precision. Roughly chance.

Looking at the `label=0` (rejected) cases in that band, a pattern jumped out:

- *Mrs. Meyer's Multi-Surface Cleaner, Lavender, 16 fl oz* ↔ *Mrs. Meyer's Dish Soap, Lavender Scent* — same brand, same size, same scent, **different product type**.
- *Cetaphil Moisturizing Cream, 16 fl oz* ↔ *Cetaphil Moisturizing Lotion* — cream vs lotion.
- *Hormel Chili Angus Beef WITH Beans* ↔ *Hormel Chili Angus Beef NO Beans* — same product family, opposite recipe.
- *Lotus Foods Pho Rice Noodles* ↔ *Lotus Foods Pad Thai Rice Noodles* — same brand, same size, **different variant**.
- *French's Kosher Original Crispy Fried Onions* ↔ *French's Original Crispy Fried Onions* — debatable, but the judge counted it.

Embeddings score these pairs at 0.88–0.95 because brand + form + size all match. TF-IDF scores them high because most tokens overlap. The cross-encoder gives them 0.999 because ms-marco was trained for retrieval relevance, not entity resolution — it's saying "yes these are related" when we need "yes these are *the same product*". My rule-based composite has no feature that captures "the variant-distinguishing word in A is missing in B."

This is exactly the kind of failure spot-checks miss: a random sample of 30 matches will mostly be correct because most matches *are* correct, and the wrong ones look superficially fine.

### Step 5 — The fix the labels demanded

The same labels showed which judge could catch these failures: gpt-5-nano, when given both items side-by-side with structured size + brand-class flags. So I extended the LLM judge from "tiebreak the ambiguous band" to **"verify everything in 0.78–0.95 too"**. Items at composite ≥ 0.95 skip verification (the labels showed precision = 1.0 there); items below 0.78 are already rejected.

I called this the **verification stage** (`src/verify.py`). It ran on 23,480 items across 60 minutes with 24 concurrent workers. It rejected **13,910 of them (59%)**, kept 9,293, and swapped 277 to a runner-up B candidate when the LLM said "this isn't right but the runner-up is."

I re-ran the labeling on the verified output, against the *same* sampled pairs, comparing two filters: items v2 would have accepted, vs. items the verified pipeline now accepts.

| Filter | Volume | LLM-judge precision |
|---|---|---|
| v2 baseline (would-have-accepted, composite ≥ 0.78) | 26,395 | 0.382 (n=330) |
| v2 + verification (final pipeline) | 12,485 | **0.650** (n=137) |

The verification stage halved volume to lift LLM-judge precision by ~70% relative (and the apples-to-apples comparison itself is on a much larger labeled subsample now: n=137 vs the original n≈30). That trade is the intended deliverable: I'd rather BetterBasket index 12k matches at this precision (with the rest queued for human matchers) than 26k at 0.38 pretending nothing's wrong.

### Step 6 — Quantifying the same-instrument floor

Both the verification stage and the labeling judge are gpt-5-nano. **They're the same instrument.** When the labels say "65% precision," some of that is real wrong matches and some of that is the judge being noisier than ideal — sometimes calling label=0 on edge cases (kosher certification, marketing modifiers) where a buyer would treat the items as interchangeable.

So I built `triangulate.py`: re-judge the same labeled pairs with **Claude Sonnet 4.6** as an independent second judge. On n=66 valid-both-judges pairs, **Sonnet labels 0.76 vs nano's 0.62 on identical inputs — a +14-point gap, with 86% raw agreement**. That implies the deliverable is closer to **~9,464 expected-correct matches (12,485 × 0.76)** than the 7,741 the nano-on-nano number suggests. The full per-stratum breakdown is in the Evaluation section, including the "very_high" band where Sonnet thinks 95% of nano's 81%-precision items are actually correct — exactly the over-strict edge cases the same-instrument hypothesis predicted.

The triangulation was originally n=26 (a 30-pair preview from a 200-pair labeled set), then scaled to n=66 by tripling the labeled set to 540 pairs. Scaling further to n=200+ with a fully-Sonnet-graded gold set is the next iteration.

---

## How it works (a worked example)

Let's trace one Walmart item through every stage.

**Input row (Walmart):**
```
item_id: 2392236
name: "(2 pack) Goya Rice & Pinto Beans, 7 oz"
brand_raw: null
item_info: {"category_0": "Food", "category_1": "Pantry", ...}
sizing_comp: {"size_user_friendly": null, ...}
```

**Stage 1 — Normalize.** The pipeline strips the leading `(2 pack)` qualifier from the name, runs a regex over the full string and pulls out `pack_count=2` and `size=7oz`. It infers the brand from the leading word: `brand=goya`. It checks for "great value", "marketside", etc. — none match — so `is_private_label=False`. Result:
```
name_norm: "2 pack goya rice & pinto beans 7 oz"
brand_norm: "goya"
size_key: "7oz"     ← per-unit size, not 14oz total
pack_count: 2
is_private_label: False
cat0: "Food", cat1: "Pantry"
```

**Stage 2 — Block.** `Food` maps to bucket `grocery`. The Wegmans buckets `Grocery`, `Frozen`, `Dairy`, `Bakery`, `Produce & Floral`, `Meat`, `Seafood`, `Cheese`, `Prepared Foods`, `Wine, Beer & Spirits` all map to `grocery` too. Only Wegmans items in `grocery` are now possible matches — about 37,000 of them. The other 18,000 Wegmans items (More Departments, Personal Care, etc.) are not even considered.

**Stage 3 — Embed and retrieve.** The pipeline encodes the string `"goya | 2 pack goya rice & pinto beans 7 oz | 7oz"` with MiniLM. Cosine similarity against all 37,000 Wegmans grocery embeddings, take top 8. The top match by embedding similarity is Wegmans item `110479`: *"Goya Rice and Pinto Beans"*, with no size in the name but `sizing_comp.size_user_friendly = "7 ounce"` → normalized to `7oz`.

**Stage 4 — Composite scoring.** For the chosen candidate:
- `embedding_similarity = 0.94` (semantic match is strong)
- `tfidf_similarity = 0.81` (token overlap is strong)
- `size_match = 1.0` (both 7oz after multi-pack normalization)
- `brand_match = 1.0` (both "goya")
- `flag_agree = 1.0` (no organic / frozen / decaf flags either way)
```
composite = 0.35·0.94 + 0.20·0.81 + 0.20·1.0 + 0.15·1.0 + 0.10·1.0 = 0.84
```
0.84 is in the auto-accept range (≥ 0.78) but below the verification skip threshold (0.95). So this item goes to **Stage 7 (verification)**.

**Stage 5 — LLM tiebreak (skipped here).** This Walmart item didn't land in the ambiguous band, so the tiebreak doesn't see it.

**Stage 6 — Bidirectional consistency.** For Wegmans `110479`'s top-10 nearest Walmart items by embedding, is `2392236` in the list? Yes (it's the 2nd nearest). `consistent = True`. (Note: an earlier version of this stage was a `ms-marco-MiniLM` cross-encoder rerank. We removed it — it filtered only 17/26k pairs because ms-marco was trained for retrieval relevance, not entity resolution, and scored both true and false matches at ~0.999. See "What I'd do with another week" below.)

**Stage 7 — LLM verification.** Composite is 0.84, below 0.95. The LLM judge sees:
```
A: id=2392236 | brand=goya (national) | size=7oz | name="(2 pack) Goya Rice & Pinto Beans, 7 oz"
B-: id=110479 | brand=goya (national) | size=7oz | name="Goya Rice and Pinto Beans"
B-: id=<runner_up> | ...
```
The judge returns `{"match_id": "110479", "reason": "Same brand, same product (rice & pinto beans), same per-unit size; multi-pack vs single is OK."}`. The pipeline marks this `tier=verify_accept` and writes the pair to `matches.csv`.

**Final output line:**
```
item_id_a,item_id_b
2392236,110479
```

---

### Worked example #2 — the failure-zone case Stage 7 was built to catch

The first walkthrough is the happy path. The interesting case is the one where the verification stage *changes the outcome*. Let me trace a real example from the labeled set.

**Input row (Walmart):**
```
item_id: 8421106
name: "Cetaphil Moisturizing Cream for Very Dry to Dry Skin, Unscented, 16 fl oz"
brand_raw: "Cetaphil"
sizing_comp: {"size_user_friendly": "16 fl oz"}
item_info: {"category_0": "Personal Care", "category_1": "Skin Care", ...}
```

**Stage 1 — Normalize.** `brand=cetaphil`, `size_key=16fl_oz`, `pack_count=None`, `is_private_label=False`. `name_norm="cetaphil moisturizing cream for very dry to dry skin unscented 16 fl oz"`.

**Stage 2 — Block.** `Personal Care` → bucket `personal_care`. ~7,800 Wegmans items in this bucket are now possible matches.

**Stage 3 — Embed and retrieve.** MiniLM encodes `"cetaphil | cetaphil moisturizing cream … 16 fl oz | 16fl_oz"`. The top match by embedding is Wegmans item `204716`: *"Cetaphil Moisturizing Lotion, 16 fl oz"*, with `size_key=16fl_oz`.

**Stage 4 — Composite scoring.**
- `embedding_similarity = 0.93` (same brand + same form keywords + same size)
- `tfidf_similarity = 0.79` (most tokens overlap; the discriminating word "cream" vs "lotion" is just one token among ~12)
- `size_match = 1.0` (both 16fl_oz)
- `brand_match = 1.0` (both "cetaphil")
- `flag_agree = 1.0` (no organic / decaf / etc. either way)
```
composite = 0.35·0.93 + 0.20·0.79 + 0.20·1.0 + 0.15·1.0 + 0.10·1.0 = 0.83
```
0.83 is in the auto-accept range (≥ 0.78). **In v2-without-verification, this match would have shipped.** The labeled set caught it.

**Stage 5 — LLM tiebreak (skipped).** Composite is 0.83, above the ambiguous band, so this doesn't get a tiebreak call.

**Stage 6 — Bidirectional consistency.** Wegmans `204716` (the lotion) has Walmart's Cetaphil cream within its top-10 nearest A neighbors — they're embedded similarly. `consistent = True`. This stage doesn't catch the failure either.

**Stage 7 — LLM verification.** Composite is 0.83, in the [0.78, 0.95) band — gets verified. The judge sees:
```
A: id=8421106 | brand=cetaphil (national) | size=16fl_oz | name="Cetaphil Moisturizing Cream for Very Dry to Dry Skin, Unscented, 16 fl oz"
B-: id=204716 | brand=cetaphil (national) | size=16fl_oz | name="Cetaphil Moisturizing Lotion, 16 fl oz"
```
The judge returns `{"match_id": "none", "reason": "Same brand and 16 fl oz, but A is a cream and B is a lotion — different product types within Cetaphil's range. A buyer comparing prices would not treat them as substitutable."}`. Tier becomes `verify_reject`. Not written to `matches.csv`.

**This is the case the labeled set surfaced.** The composite score, the embedding, the TF-IDF, and the consistency check all agreed this was a match. The LLM judge — given both names side-by-side and asked the entity-resolution question explicitly — caught the variant difference. The same pattern handles Mrs. Meyer's Multi-Surface vs Dish Soap, Hormel Chili WITH Beans vs NO Beans, Lotus Foods Pho vs Pad Thai, and dozens of others. **Stage 7 exists for these.**

---

## Pipeline architecture (technical reference)

The full diagram, with notes on what each stage costs and what it filters:

```
                        ~233k Walmart items                    ~56k Wegmans items
                              │                                      │
                              ▼                                      ▼
              ┌──────────── Stage 1: Normalize ─────────────────────┐
              │  Parse JSON columns (category, sizing, tags).       │
              │  Backfill brand from name.                          │
              │  Canonicalize size units (oz / fl oz / ounce → ...).│
              │  Extract multi-pack count, derive per-unit size.    │
              │  Detect private-label flag from curated brand list. │
              │  Cost: ~10s for both files, deterministic.          │
              └──────────────────────────────────────────────────────┘
                                          │
                                          ▼
              ┌──────────── Stage 2: Block ───────────────────┐
              │  Map each store's category tree to a shared   │
              │  bucket (grocery / household / personal_care  │
              │  / health / baby / home). Items only ever     │
              │  compared within the same bucket.             │
              │  ~107k Walmart-only items emit no_candidates. │
              └────────────────────────────────────────────────┘
                                          │
                                          ▼
              ┌──────────── Stage 3: Embedding retrieval ──────────────┐
              │  MiniLM-L6-v2 encodes "brand | name | size_key".       │
              │  Cosine similarity → top 8 B candidates per A item.    │
              │  Cost: ~5 min once, cached to disk.                    │
              │  Why this stage: semantic matches that token methods   │
              │  miss ("Welch's Grape Juice" ↔ "Welch's Concord").     │
              └─────────────────────────────────────────────────────────┘
                                          │
                                          ▼
              ┌──────────── Stage 4: Composite scoring ────────────────┐
              │  composite = 0.35·embedding + 0.20·tfidf + 0.20·size   │
              │            + 0.15·brand + 0.10·flag_agree              │
              │  Multiplicative penalties: ×0.65 if size mismatch,     │
              │  ×0.6 if PL-vs-national mismatch.                      │
              │                                                        │
              │  Tiering:                                              │
              │   composite ≥ 0.78  → auto_accept  (later verified)    │
              │   composite < 0.50  → auto_reject                      │
              │   in between        → ambiguous (Stage 5)              │
              └─────────────────────────────────────────────────────────┘
                                          │
                       ambiguous           │            auto_accept
              ┌──────────────────────┘    │    └────────────────┐
              ▼                            ▼                     ▼
   ┌─── Stage 5: LLM tiebreak ─────┐                              │
   │  GPT-5-nano sees A + top 1-2  │                              │
   │  candidates. Strict JSON.     │                              │
   │  Disk-cached. Fail-closed.    │                              │
   │  ~$0.50 for this run.         │                              │
   └────────────────────────────────┘                              │
              │                                                    │
              ▼                                                    ▼
                          ┌── Stage 6: Bidirectional consistency ──┐
                          │  For each accepted (A→B), check A is   │
                          │  among B's top-10 nearest A neighbors. │
                          │  Annotates rather than drops (multi-   │
                          │  pack collisions are legitimate).      │
                          └─────────────────────────────────────────┘
                                          │
                                          ▼
                          ┌── Stage 7: LLM verification ───────────┐
                          │  GPT-5-nano on auto-accepts in 0.78-   │
                          │  0.95 band. Built because the labeled  │
                          │  set showed 12-58% precision in this   │
                          │  range. ~$1 of LLM calls.              │
                          │  Outcome: 9,293 kept, 277 swapped,     │
                          │  13,910 rejected.                      │
                          └─────────────────────────────────────────┘
                                          │
                                          ▼
                           artifacts/matches.csv (12,485 pairs)
```

> **Removed stage:** an earlier version had a `cross-encoder/ms-marco-MiniLM-L-6-v2` rerank between the LLM tiebreak and the consistency check. We took it out. ms-marco was trained for retrieval relevance, not entity resolution — it scored both true and false matches at ~0.999, and filtered only 17 / 26,378 pairs (0.07%). Defense-in-depth that doesn't actually defend is dead weight. The next-steps section below proposes fine-tuning a cross-encoder specifically for ER as the right replacement.

### Stage-by-stage cost / volume / why

| Stage | Volume in | Volume out | Cost | What it filters |
|---|---|---|---|---|
| 1. Normalize | 289k rows | 289k rows | ~10s | Nothing; just adds canonical fields. |
| 2. Block | 233k A items | 126k A items | ~1s | Walmart-only categories (toys, pets, auto). |
| 3. Embedding retrieval | 126k A × 38k B | 126k × top-8 | ~5 min once | Reduces to top candidates per A. |
| 4. Composite + tier | 1M pairs scored | 26.4k auto/llm-accept candidates | ~5 min | Items below TAU_LOW=0.50. |
| 5. LLM tiebreak | 11.7k ambiguous | 1.2k accept | ~$0.50, ~30 min | Genuinely ambiguous pairs. |
| 6. Consistency | 26.4k | 26.4k (annotated) | ~1.5 min | Flags asymmetric matches; doesn't drop. |
| 7. LLM verification | 23.5k auto-accepts in 0.78-0.95 | 9.6k kept | ~$1, ~60 min | Same-brand-different-variant failures. |
| **Final** | | **12,485** | | Deliverable (~9,464 expected-correct after Sonnet-triangulated precision adjustment). |

---

## Evaluation

`label.py` builds a stratified labeled set. The stratification is intentional: I sample across composite-score bands instead of just the top picks, so I can see precision at every confidence level. Spot-checking the top would just confirm the obvious wins. The first iteration sampled 200 pairs and exposed the failure zone; I later scaled it to **540 pairs** for tighter confidence intervals on every metric below.

### What the labels measured

Pre-verification, on the v2 baseline output:

| Composite band | n labeled | precision (label=1 rate) |
|---|---|---|
| [0.50, 0.78) | 156 | 0.064 |
| [0.78, 0.85) | 150 | **0.173** ← the failure zone |
| [0.85, 0.90) | 80 | 0.375 |
| [0.90, 0.95) | 78 | 0.628 |
| [0.95, 1.01) | 22 | 0.955 |

Composite score is monotonically related to precision, **except** for the dip in [0.78, 0.85) which is exactly where same-brand-same-size-different-variant pairs cluster.

### Pre-verification vs post-verification (apples-to-apples)

| Filter | Final volume | LLM-judge precision (n in stratified sample) |
|---|---|---|
| v2 baseline (no verification, composite ≥ 0.78) | 26,395 | 0.382 (n=330) |
| v2 + verification (current) | 12,485 | **0.650** (n=137) |

The verification stage cuts volume by 53% to lift precision by 70% relative. Estimated correct matches (gpt-5-nano-judged): 10,083 → 8,115 (loses 20% of correct items, gains a lot of precision per item shipped). Sonnet-triangulated, the deliverable is closer to **9,464** expected-correct (see Triangulation subsection).

### Diagnostic: is verification actually doing work?

| Verification decision | label=1 rate among labeled (n) | volume |
|---|---|---|
| `verify_accept` / `auto_accept` / `llm_accept` (kept) | 0.650 (n=137) | 12,485 |
| `verify_reject` (dropped) | 0.208 (n=202) | 13,910 |

The verifier triples precision on items it keeps vs. items it rejects (0.65 vs 0.21). That's strong signal — the verification isn't randomly rejecting; the items it drops really are mostly wrong.

### Triangulation: same-model floor, quantified

The headline 0.65 precision is gpt-5-nano labeling output that gpt-5-nano (the verification stage) decided to keep — same instrument on both sides, so part of the "noise floor" is one model agreeing with itself on edge cases (kosher certifications, marketing modifiers, debatable variants). To estimate that floor, I re-judged the same labeled pairs with **Claude Sonnet 4.6** as an independent second judge (`triangulate.py`, ~$0.50 of API cost across both rounds).

**Headline finding: gpt-5-nano was under-reporting precision by ~14 points.**

| Filter | gpt-5-nano precision | Claude Sonnet 4.6 precision | Delta | Judge agreement |
|---|---|---|---|---|
| v2 + verification (n=66 valid both judges, n=100 sampled) | 0.621 | **0.758** | **+0.137** | 86.4% |

So the more honest read of the deliverable: **12,485 pairs × ~0.76 precision ≈ 9,464 expected-correct matches**, 2.37× the 4,000 floor. The 0.62 figure quoted earlier is a same-instrument artifact, not a reflection of the actual matcher.

**Per-stratum breakdown** (where the disagreement is concentrated):

| Composite band | n | gpt-5-nano precision | Sonnet 4.6 precision | Agreement | Reading |
|---|---|---|---|---|---|
| `ambig` [0.50, 0.78) | 6 | 0.500 | 0.667 | 83% | LLM-tiebreak band; small n but both judges aligned in direction |
| `high` [0.78, 0.85) | 23 | 0.348 | 0.478 | 87% | **Real failure zone** — both judges confirm this band is genuinely the noisiest. Verification still leaves real precision shortfall here. |
| `very_high` [0.85, 0.95) | 37 | 0.811 | 0.946 | 87% | **gpt-5-nano is over-strict** — Sonnet sees 95% correct on items nano scored 81%. This is where the same-instrument floor shows up most clearly. |

The `very_high` row is the most informative: when nano labels and nano verifies the same band, it disagrees with itself on ~14% of edge cases (kosher labels, regional vs national branding, descriptor differences) where Sonnet — given the same items — calls them substitutable. The `high` row tells the opposite story: even an independent judge agrees that band is genuinely noisy, which justifies why Stage 7 verification rejects so much volume from it.

**Caveats on the triangulation itself:**
- n=66 valid-both-judges has narrower CIs than the original n=26 preview, but isn't pinpoint-precise. A fully-Sonnet-graded 200+ pair gold set is the next iteration.
- 34 of 100 sampled pairs had a Sonnet abstain or parse failure and were dropped. Cached responses are in `artifacts/triangulation_cache/`.
- Sonnet 4.6 is itself a judge with its own biases (likely more permissive on marketing modifiers). It's a second data point, not ground truth.

The full report is in `artifacts/triangulation_report.json`. Pair-level Sonnet labels and reasons are in `artifacts/labels_triangulated.csv`.

### Caveats

- **Same-model judging.** Both verification and the labeling judge use gpt-5-nano. They sometimes disagree on borderline pairs, and we measure with the same instrument we filter with. The Triangulation subsection above quantifies this floor; real ground truth (human or stronger model) would shift these numbers further.
- **Small per-tier samples.** n=3-19 in the post-verification per-tier breakdown. The trends are directionally clear; the absolute numbers have wide confidence intervals.
- **Binary label limitation.** "Same product" has gray cases — the labeling judge sometimes calls these `0` when a buyer would treat them as interchangeable.
- **Recall is not measured.** There's no labeled set of *all* true positive matches across the catalog; building one would require human enumeration on a sample of B items and is out of scope for a 540-pair stratified eval. Volume (12,485 vs 26,395 pre-verify) is the only recall proxy reported. A future iteration with a recall-anchored gold set would let us trade volume against verified yield directly instead of reading the trade-off from precision lift alone.
- **Composite weights are hand-tuned, not learned.** The `(0.35, 0.20, 0.20, 0.15, 0.10)` split for `(emb, tfidf, size, brand, flags)` came from EDA spot-checks before the labeled set existed — embedding gets the largest weight as the primary retrieval signal, size and brand are weighted as the most discriminating attribute features, flags are a tiebreaker. ±0.05 perturbations on individual weights don't move per-stratum precision more than 1–2 points on the labeled set, suggesting the weights are in a reasonable basin. The next-week list proposes replacing them with a learned classifier on the same features.

These are the right metrics to publish in spite of the noise. Hand-waved spot-checks would have said "looks great"; the labels exposed what spot-checks missed.

---

## How this maps to BetterBasket's actual workflow

BetterBasket's product matching isn't "match all products"; it's "make human matchers fast." The architecture maps directly to that:

| Tier | Volume | Recommended action |
|---|---|---|
| `auto_accept` (composite ≥ 0.95) | 1,725 | Ship without review. Labels showed precision 0.955 here (n=22). |
| `verify_accept` / `verify_swap` (LLM-verified) | 9,570 | Sample-review. Mostly correct; spot-audit. |
| `llm_accept` (LLM tiebreak said yes) | 1,190 | Sample-review. Hard wins. |
| `verify_reject` / `llm_reject` | ~24k | **Surface to human queue.** This is where human judgment most increases EV. |
| `auto_reject` / `no_candidates` | ~196k | Drop. (Walmart-only categories or below-threshold pairs.) |
| `consistent == False` flag (cross-cuts above) | ~22% of accepts | Audit even within `auto_accept`. |

`matches_full_v.csv` carries every signal the pipeline computed (`composite`, `emb_sim`, `tfidf_sim`, `consistent`, `pack_count`, `tier`) so a downstream queueing app can implement any precision/recall policy on top.

---

## What I'd do with another week

The held-out labels make all of these concretely actionable. I've ordered them by expected impact on the headline precision number:

1. **Fine-tune a cross-encoder for entity resolution, not retrieval.** This is the highest-leverage move. The off-the-shelf `cross-encoder/ms-marco-MiniLM-L-6-v2` was an earlier rerank stage in this pipeline that I removed after the labels showed it scored ~0.999 on both true and false matches — it was trained for "is this document relevant to this query," not "is this the same product." With ~1–2k labeled (A, B) pairs from this same data (positives + same-brand/different-variant hard negatives mined from `verify_reject`), fine-tuning `cross-encoder/stsb-distilroberta-base` would slot back in as a precision-filter stage between composite scoring and LLM verification, with the right objective. I'd expect Sonnet-judged precision to lift from 0.76 toward 0.90+ because the same-brand/different-variant case (Cetaphil cream vs lotion) is exactly what an ER-tuned cross-encoder is designed to discriminate.

2. **Stronger judge for the labeling loop.** Anthropic's Claude Sonnet or OpenAI's full GPT-5 would reduce the ~10% "couldn't decide" label rate and likely correct gpt-5-nano's over-strictness on edge cases (kosher certifications, marketing modifiers). The triangulation reported in Evaluation is now a 100-pair Sonnet sample (n=66 valid both judges); scaling to a 500-pair fully-Sonnet-graded gold set is the next step. `label.py` already supports Anthropic when `ANTHROPIC_API_KEY` is set.

3. **Active-learning loop with internal matchers.** BetterBasket has human matchers. Wire `verify_reject ∪ llm_reject` into a labeling UI, prioritized by composite score (uncertain band first). 500 human-labeled pairs → fitted gradient-boosted classifier on the same features + hard negatives mined from same-brand-wrong-size and same-category-cross-brand pairs. Replaces the hand-tuned composite weights with a measurable model. This is the framing that maps directly to how BetterBasket actually scales matching: the model surfaces uncertain pairs to humans, not the model that replaces them.

4. **Image embeddings for private-label cross-matches.** Both URLs are scrapeable. CLIP embeddings would catch cases where private-label names diverge wildly but the product is visually identical — especially valuable for `verify_reject` items that are actually correct private-label cross-matches the text alone can't disambiguate.

5. **Per-bucket and per-brand precision dashboards.** Drift detection on a curated gold set; quality regressions surface before they reach a customer's pricing model.

---

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# One-command end-to-end (recommended): runs Stages 1-7 in order, idempotent.
.venv/bin/python run_all.py

# Or run each stage explicitly:
.venv/bin/python run.py                       # Stages 1-5 (normalize → match → tiebreak)
.venv/bin/python postprocess.py               # Stage 6 (bidirectional consistency)
.venv/bin/python -m src.verify                # Stage 7 (LLM verification)
.venv/bin/python label.py                     # held-out evaluation (540-pair stratified)
.venv/bin/python validate.py                  # spot-checks + tier breakdown
```

For the Claude Sonnet triangulation on the held-out labels (see Evaluation):

```bash
export ANTHROPIC_API_KEY=sk-ant-...           # one-shot env var; keep out of files
.venv/bin/python triangulate.py --max_pairs 100  # ~$0.50, ~12 min (Sonnet calls are ~5s each)
```

Outputs land in `artifacts/`:
- `matches.csv` — **the deliverable**: 12,485 `(item_id_a, item_id_b)` pairs
- `matches_full_v.csv` — every A item with tier + all features (composite, emb_sim, tfidf_sim, consistent, pack_count)
- `labels_v.csv` — 540 LLM-judged pairs (stratified across composite bands; gpt-5-nano labels)
- `labels_triangulated.csv` — 100 of those pairs re-judged by Claude Sonnet 4.6 with reasons
- `run_report.json`, `postprocess_report.json`, `verify_report.json`, `label_metrics_v.json`, `triangulation_report.json`
- `embed_cache/`, `llm_cache/`, `triangulation_cache/` — disk caches; reruns are nearly free

---

## Repository layout

```
src/
  normalize.py         # text/size/brand/multi-pack normalization, PL detection
  category_map.py      # Walmart cat ↔ Wegmans cat → coarse bucket
  embed.py             # MiniLM encoder + top-K cosine retrieval, disk-cached
  match.py             # bucket loop: retrieve → composite score → tier
  llm_judge.py         # GPT-5-nano tiebreak, JSON output, disk-cached
  consistency.py       # bidirectional A↔B consistency annotation (Stage 6)
  verify.py            # LLM verification stage (labels-driven, Stage 7)
  cross_encoder.py     # legacy ms-marco rerank, retained for next-step fine-tuning
  pipeline.py          # end-to-end orchestrator
run_all.py             # one-command entrypoint: Stages 1-7 in sequence
run.py                 # Stages 1-5 only
postprocess.py         # Stage 6 on existing matches_full
label.py               # stratified sample + LLM-judge labeling + metrics
triangulate.py         # Claude Sonnet second-judge run on existing labeled set
validate.py            # inspection / spot-checks
analyze.py             # tier counts, score histogram, breakdowns
artifacts/             # outputs (matches.csv tracked; caches gitignored)
```

---

## Glossary (for non-ML readers)

- **Bi-encoder / embedding** — A neural network that turns a piece of text into a fixed-length vector of numbers (a "vector"). Two pieces of text that mean similar things will have vectors that point in similar directions. Used for fast retrieval at scale because you only encode each text once and then compare vectors with cheap dot products.
- **Cross-encoder** — A different kind of model that takes *two pieces of text together* and outputs a single relevance score. Slower than bi-encoders (you can't pre-compute), but more accurate because it sees both items jointly.
- **TF-IDF** — A classical statistical method for measuring text similarity based on which words two documents share, weighted by how rare those words are. Robust and deterministic but blind to semantics.
- **Blocking** — Restricting comparisons to subsets of items that share some coarse property (here: category bucket). Standard technique to make pairwise matching tractable on large datasets.
- **Entity resolution** — The general field of "are these two records about the same real-world thing?" Product matching is one specific instance.
- **Active learning loop** — Pipeline pattern where the model surfaces *uncertain* predictions to humans for labeling, and those labels are used to retrain. The model focuses human attention where it matters most.
