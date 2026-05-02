"""Normalization utilities: name cleaning, brand extraction, size canonicalization,
private-label detection, and attribute flag extraction.

Design notes:
- A's `name_clean`, `is_private_label`, and `tags` are 100% null — we derive them.
- A's `brand_raw` is ~46% null. We backfill from the leading words of `name`.
- Sizes appear in B's `sizing_comp.size_user_friendly` reliably; in A they are usually
  embedded in `name`. We try sizing_comp first, then a regex fallback on name.
- Unit normalization: B uses "ounce" / "fl. oz.", A uses "oz" / "fl oz". We canonicalize.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd

# ----- private label brands (from EDA + standard knowledge) -------------------
WALMART_PRIVATE_LABELS = {
    "great value", "marketside", "equate", "mainstays", "sam's choice",
    "better homes & gardens", "wonder nation", "athletic works", "hyper tough",
    "parent's choice", "way to celebrate", "modern moments", "garanimals",
    "freshness guaranteed", "ozark trail", "no boundaries", "george", "onn",
    "spring valley", "members mark",
}
WEGMANS_PRIVATE_LABELS = {"wegmans", "food you feel good about", "wegmans organic"}

# ----- size unit canonical map ------------------------------------------------
# All units are normalized to short, lowercase forms for exact-match comparison.
UNIT_CANONICAL = {
    "fl oz": "fl_oz", "fl. oz.": "fl_oz", "fl. oz": "fl_oz",
    "fluid ounce": "fl_oz", "fluid ounces": "fl_oz", "fluid oz": "fl_oz",
    "oz": "oz", "ounce": "oz", "ounces": "oz", "oz.": "oz",
    "lb": "lb", "lbs": "lb", "pound": "lb", "pounds": "lb",
    "g": "g", "gram": "g", "grams": "g",
    "kg": "kg", "kilogram": "kg", "kilograms": "kg",
    "ml": "ml", "milliliter": "ml", "milliliters": "ml",
    "l": "l", "liter": "l", "liters": "l", "litre": "l",
    "gal": "gal", "gallon": "gal", "gallons": "gal",
    "qt": "qt", "quart": "qt", "quarts": "qt",
    "pt": "pt", "pint": "pt", "pints": "pt",
    "ct": "ct", "count": "ct", "ct.": "ct", "pack": "ct", "pk": "ct",
    "each": "each", "ea": "each", "ea.": "each",
}

# Pluralization-tolerant regex; longest forms first to avoid premature matches.
_UNIT_PATTERNS = sorted(UNIT_CANONICAL.keys(), key=len, reverse=True)
SIZE_RE = re.compile(
    r"(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>"
    + "|".join(re.escape(u) for u in _UNIT_PATTERNS)
    + r")\b",
    re.IGNORECASE,
)

# ----- attribute flag patterns ------------------------------------------------
ORGANIC_RE = re.compile(r"\borganic\b", re.I)
GLUTEN_FREE_RE = re.compile(r"\bgluten[\s-]?free\b", re.I)
FROZEN_RE = re.compile(r"\bfrozen\b", re.I)
DECAF_RE = re.compile(r"\bdecaf(?:f?einated)?\b", re.I)
DIET_RE = re.compile(r"\b(diet|zero|no-?sugar|sugar[\s-]?free)\b", re.I)
NONFAT_RE = re.compile(r"\b(nonfat|fat[\s-]?free|skim|low[\s-]?fat|reduced[\s-]?fat|whole)\b", re.I)
KOSHER_RE = re.compile(r"\bkosher\b", re.I)


# ----- text normalization -----------------------------------------------------
def clean_text(s: str) -> str:
    """Lowercase, strip punctuation (except &), collapse whitespace."""
    if not isinstance(s, str):
        return ""
    s = s.lower()
    # Drop quotes/parentheses but keep & for "M&M", "Land O' Lakes" -> "land o lakes"
    s = re.sub(r"[‘’“”]", "", s)
    s = re.sub(r"[(),\"\[\]:;/\\]", " ", s)
    s = re.sub(r"[^a-z0-9& ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_json_safe(v) -> dict:
    if not isinstance(v, str):
        return {}
    try:
        d = json.loads(v)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


# ----- size parsing -----------------------------------------------------------
@dataclass(frozen=True)
class Size:
    val: Optional[float]
    unit: Optional[str]  # canonicalized

    def key(self) -> Optional[str]:
        if self.val is None or self.unit is None:
            return None
        # round 1.41 -> 1.41 ; 14.50 -> 14.5
        v = round(self.val, 2)
        if v == int(v):
            v = int(v)
        return f"{v}{self.unit}"


# Units in priority order: weight/volume beats count/each.
# When a name contains both "3 pack" and "4 fl oz", we want the fl oz.
_PRIORITY_UNITS = {"fl_oz", "oz", "lb", "g", "kg", "ml", "l", "gal", "qt", "pt", "ct", "each"}


def parse_size_str(s: str) -> Size:
    """Parse a freeform size string like '5.3 oz' or '12 fl. oz.' or '1 each'.
    If the string contains multiple sizes, prefer weight/volume over count/each."""
    if not isinstance(s, str):
        return Size(None, None)
    s = s.strip().lower()
    matches = list(SIZE_RE.finditer(s))
    if not matches:
        return Size(None, None)

    parsed: list[Size] = []
    for m in matches:
        try:
            val = float(m.group("val"))
        except ValueError:
            continue
        unit_raw = m.group("unit").lower()
        unit = UNIT_CANONICAL.get(unit_raw) or UNIT_CANONICAL.get(unit_raw.rstrip("."))
        if unit is None:
            continue
        parsed.append(Size(val, unit))

    if not parsed:
        return Size(None, None)

    # Prefer non-count units (weight/volume more discriminating than pack count)
    non_count = [p for p in parsed if p.unit not in ("ct", "each")]
    if non_count:
        return non_count[0]
    return parsed[0]


def extract_size(name: str, sizing_comp: dict) -> Size:
    """Prefer sizing_comp.size_user_friendly, else regex on the name."""
    s = sizing_comp.get("size_user_friendly") if isinstance(sizing_comp, dict) else None
    if isinstance(s, str) and s.strip():
        size = parse_size_str(s)
        if size.unit is not None:
            return size
    if isinstance(name, str):
        return parse_size_str(name)
    return Size(None, None)


# ----- multi-pack reconciliation ---------------------------------------------
# Walmart's catalog is full of "(3 pack) Goya Rice 7 oz" while Wegmans almost
# never sells multi-packs of grocery items — they sell singles. Without
# normalization we'd mismatch on size every time. We extract the pack count
# from the name, then compute a per-unit size that we can compare against B.
#
# Patterns we see in the data:
#   "(3 pack) Goya Rice 7 oz"
#   "(12 pack) Campbells Chunky Soup, 18.8 oz"
#   "Campbells Soup, 12 Pack, 10.75 oz"
#   "12-Pack Coca-Cola, 12 fl oz cans"
#   "Pack of 6 ..."
#   "6 Count, 1.4 oz Each"
#   "Set of 4 ..."
#
# Convention: when both a pack count and a per-unit size are stated, the
# per-unit size is what the "single product" would weigh — so we use that as
# the canonical size. When only a total size is stated alongside a pack
# count (rare), we divide.

# Patterns ordered by specificity. Each pattern captures the pack count.
_PACK_PATTERNS = [
    re.compile(r"\(\s*(\d+)\s*[-\s]?(?:pack|pk|count|ct)\s*\)", re.I),  # "(3 pack)"
    re.compile(r"\bpack\s+of\s+(\d+)\b", re.I),                           # "Pack of 6"
    re.compile(r"\bset\s+of\s+(\d+)\b", re.I),                            # "Set of 4"
    re.compile(r"(?<![a-z0-9])(\d+)\s*[-\s]?pack\b", re.I),               # "12-Pack" / "12 Pack"
    re.compile(r"(?<![a-z0-9])(\d+)\s*[-\s]?ct\b\.?", re.I),              # "12-ct" / "24 ct"
    re.compile(r"(?<![a-z0-9])(\d+)\s*count\b", re.I),                    # "12 count"
]


def extract_pack_count(name: str) -> Optional[int]:
    """Return the integer pack count if the name explicitly states one.

    We deliberately reject pack counts that come *after* a unit token
    (e.g. "12 oz, 24 ct" -> 24 is the pack count, but "24 ct" alone -> 24
    is the size). The patterns above already encode pack-specific phrasing
    so we don't need extra disambiguation here.
    """
    if not isinstance(name, str) or not name:
        return None
    # Avoid matching things like "0.46 fl oz" or "16 oz" that have a unit attached
    candidates: list[int] = []
    for pat in _PACK_PATTERNS:
        for m in pat.finditer(name):
            try:
                n = int(m.group(1))
            except ValueError:
                continue
            # Sanity: realistic pack counts are 2-144. Anything outside is noise.
            if 2 <= n <= 144:
                candidates.append(n)
    if not candidates:
        return None
    # If multiple matches disagree, take the smallest plausible (more conservative).
    return min(candidates)


def reconcile_pack(size: Size, pack_count: Optional[int], name: str) -> Size:
    """Given the parsed size + pack count, return the per-unit size.

    Heuristic:
    - If pack_count is None -> return size unchanged.
    - If size unit is "ct" or "each" -> the size IS the count, leave it.
    - If both pack count and a weight/volume size were extracted, the parsed
      size is almost always already per-unit (that's how grocery labels work).
      Returning it unchanged is correct.
    - Edge case: name explicitly says "Total: <size>" -> divide. We do this
      heuristically by checking for "total" near the size mention.
    """
    if pack_count is None or size.val is None or size.unit is None:
        return size
    if size.unit in ("ct", "each"):
        return size
    # If "total" appears in the name AND there's only one size match, divide.
    if isinstance(name, str) and re.search(r"\btotal\b", name, re.I):
        try:
            return Size(round(size.val / pack_count, 4), size.unit)
        except ZeroDivisionError:
            return size
    return size


# ----- brand extraction -------------------------------------------------------
def normalize_brand(b: str) -> str:
    if not isinstance(b, str):
        return ""
    return clean_text(b)


_LEADING_PARENS_RE = re.compile(r"^\s*\([^)]*\)\s*", re.I)
_LEADING_BRACKETS_RE = re.compile(r"^\s*\[[^\]]*\]\s*", re.I)


def _strip_leading_qualifiers(name: str) -> str:
    """Remove leading '(3 pack)', '(Fresh)', '[New]' style prefixes that aren't brands."""
    prev = None
    s = name
    while s != prev:
        prev = s
        s = _LEADING_PARENS_RE.sub("", s)
        s = _LEADING_BRACKETS_RE.sub("", s)
    return s


def guess_brand_from_name(name: str) -> str:
    """Take the first 1-2 words before the first comma. We prefer recall over precision
    here because the field gets normalized + private-label-checked downstream, and the
    real signal lives in the n-gram TF-IDF anyway."""
    if not isinstance(name, str):
        return ""
    head = _strip_leading_qualifiers(name).split(",")[0].strip()
    # Match known multi-word private-label brands first (longest first)
    head_lower = head.lower()
    multi_word_pls = [
        "great value", "way to celebrate", "better homes & gardens",
        "wonder nation", "athletic works", "hyper tough", "parent's choice",
        "modern moments", "sam's choice", "freshness guaranteed",
        "ozark trail", "no boundaries", "spring valley", "members mark",
    ]
    for pl in multi_word_pls:
        if head_lower.startswith(pl):
            return clean_text(pl)
    # Otherwise default to first 2 words (most national brands are 1-2 words)
    words = head.split()
    return clean_text(" ".join(words[:2]))


# ----- private-label detection ------------------------------------------------
def is_walmart_private_label(name: str, brand: str) -> bool:
    text = (clean_text(brand) + " " + clean_text(name)).strip()
    return any(pl in text for pl in WALMART_PRIVATE_LABELS)


def is_wegmans_private_label(name: str, brand: str, tags_raw: str) -> bool:
    text = (clean_text(brand) + " " + clean_text(name)).strip()
    if any(pl in text for pl in WEGMANS_PRIVATE_LABELS):
        return True
    # tags column on B contains 'wegmans_brand' for private label
    if isinstance(tags_raw, str) and "wegmans_brand" in tags_raw:
        return True
    return False


# ----- attribute flags --------------------------------------------------------
def attribute_flags(name: str, info: dict, tags_raw: str = "") -> dict:
    text = name if isinstance(name, str) else ""
    storage = (info or {}).get("storage_type") or ""
    blob = f"{text} {storage} {tags_raw or ''}".lower()
    return {
        "organic": bool(ORGANIC_RE.search(blob)),
        "gluten_free": bool(GLUTEN_FREE_RE.search(blob)),
        "frozen": bool(FROZEN_RE.search(blob)),
        "decaf": bool(DECAF_RE.search(blob)),
        "diet": bool(DIET_RE.search(blob)),
        "kosher": bool(KOSHER_RE.search(blob)),
    }


# ----- pulling it all together ------------------------------------------------
def normalize_dataframe(df: pd.DataFrame, side: str) -> pd.DataFrame:
    """Add canonical fields used downstream: name_norm, brand_norm, size, flags,
    is_private_label, cat0/cat1.

    `side` is 'A' or 'B'.
    """
    assert side in ("A", "B")

    info = df["item_info"].apply(parse_json_safe)
    sizing = df["sizing_comp"].apply(parse_json_safe)
    tags_str = df["tags"].fillna("") if "tags" in df.columns else pd.Series([""] * len(df))

    out = pd.DataFrame(index=df.index)
    out["item_id"] = df["item_id"].astype(str)
    out["name"] = df["name"].astype(str)
    out["name_norm"] = df["name"].apply(clean_text)
    out["brand_raw"] = df["brand_raw"].fillna("")

    # backfill brand from name when missing
    brand_norm = out["brand_raw"].apply(normalize_brand)
    needs_fill = brand_norm.eq("")
    brand_norm.loc[needs_fill] = df.loc[needs_fill, "name"].apply(guess_brand_from_name)
    out["brand_norm"] = brand_norm

    # size + multi-pack reconciliation
    raw_sizes = [extract_size(n, s) for n, s in zip(out["name"], sizing)]
    pack_counts = [extract_pack_count(n) for n in out["name"]]
    sizes = [reconcile_pack(sz, pc, n) for sz, pc, n in zip(raw_sizes, pack_counts, out["name"])]
    out["size_val"] = [s.val for s in sizes]
    out["size_unit"] = [s.unit for s in sizes]
    out["size_key"] = [s.key() for s in sizes]
    out["pack_count"] = pack_counts

    # category
    out["cat0"] = info.apply(lambda d: d.get("category_0"))
    out["cat1"] = info.apply(lambda d: d.get("category_1"))
    out["cat2"] = info.apply(lambda d: d.get("category_2"))

    # flags
    flags = [attribute_flags(n, i, t) for n, i, t in zip(out["name"], info, tags_str)]
    for k in ["organic", "gluten_free", "frozen", "decaf", "diet", "kosher"]:
        out[f"flag_{k}"] = [f[k] for f in flags]

    # private label
    if side == "A":
        out["is_private_label"] = [
            is_walmart_private_label(n, b) for n, b in zip(out["name"], out["brand_norm"])
        ]
    else:
        out["is_private_label"] = [
            is_wegmans_private_label(n, b, t)
            for n, b, t in zip(out["name"], out["brand_norm"], tags_str)
        ]

    return out
