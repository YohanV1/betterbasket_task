"""Map A's Walmart category tree to B's Wegmans category tree.

We use coarse "buckets" rather than exact category names because the two retailers
have entirely different taxonomies. Two items belong to the same bucket if a
shopper would consider them substitutable categories.

This map was derived from the EDA results — A and B's actual top categories.
"""
from __future__ import annotations

# Bucket -> set of A.cat0 values and set of B.cat0 values that belong to it.
# An item's bucket is determined by either its (cat0) or (cat0,cat1) match.

# A's high-level categories (from EDA): Food, Household Essentials, Personal Care,
#   Health and Medicine, Toys, Pets, Baby, Clothing, Home, Beauty, Home Improvement,
#   Sports & Outdoors, Party & Occasions, Office Supplies, Auto & Tires
#
# B's high-level categories (from EDA): More Departments (mixed), Grocery,
#   Wine Beer & Spirits, Frozen, Dairy, Produce & Floral, Bakery, Meat,
#   Prepared Foods, Cheese, Seafood
#
# B's "More Departments" sub-categories: Personal Care and Makeup, Health and Wellness,
#   Household Essentials, Party Celebrations & Gifts, Kitchen and Home,
#   Baby & Toddler, Bulk Foods, Deli, Seasonal Home

A_CAT0_TO_BUCKET = {
    "Food": "grocery",
    "Household Essentials": "household",
    "Personal Care": "personal_care",
    "Beauty": "personal_care",
    "Health and Medicine": "health",
    "Baby": "baby",
    # The rest below have very little overlap with Wegmans; we still bucket them
    # so we can attempt matches but they will almost always fail to find candidates.
    "Pets": "pets",
    "Toys": "toys",
    "Clothing": "clothing",
    "Home": "home",
    "Home Improvement": "home_improvement",
    "Sports & Outdoors": "sports",
    "Party & Occasions": "party",
    "Office Supplies": "office",
    "Auto & Tires": "auto",
}

# B-side: cat0 alone often suffices, but for "More Departments" we look at cat1.
B_CAT0_TO_BUCKET = {
    "Grocery": "grocery",
    "Frozen": "grocery",
    "Dairy": "grocery",
    "Produce & Floral": "grocery",
    "Bakery": "grocery",
    "Meat": "grocery",
    "Seafood": "grocery",
    "Prepared Foods": "grocery",
    "Cheese": "grocery",
    "Wine, Beer & Spirits": "grocery",  # A's Food includes alcohol
}

B_MORE_DEPT_CAT1_TO_BUCKET = {
    "Personal Care and Makeup": "personal_care",
    "Health and Wellness": "health",
    "Household Essentials": "household",
    "Baby & Toddler": "baby",
    "Bulk Foods": "grocery",
    "Deli": "grocery",
    "Kitchen and Home": "home",
    "Seasonal Home": "home",
    "Party Celebrations & Gifts": "party",
}


def bucket_for_a(cat0: str | None, cat1: str | None) -> str:
    return A_CAT0_TO_BUCKET.get(cat0, "other")


def bucket_for_b(cat0: str | None, cat1: str | None) -> str:
    if cat0 == "More Departments":
        return B_MORE_DEPT_CAT1_TO_BUCKET.get(cat1, "other")
    return B_CAT0_TO_BUCKET.get(cat0, "other")


# Buckets where B has any meaningful inventory worth searching against.
# (From EDA: pets/toys/clothing/home_improvement etc. have ~0 B items.)
BUCKETS_WITH_B_INVENTORY = {"grocery", "household", "personal_care", "health", "baby", "home"}
