"""One-command end-to-end pipeline.

Runs the four scripts that produce the deliverable, in order:

  1. run.py             Stages 1-5 (normalize -> embed -> match -> LLM tiebreak)
  2. postprocess.py     Stage 6   (bidirectional consistency annotation)
  3. src.verify         Stage 7   (LLM verification of the failure-zone band)
  4. validate.py        sanity-check the deliverable

Each step is idempotent given the disk caches in `artifacts/embed_cache/` and
`artifacts/llm_cache/` — re-running with caches warm finishes in seconds.
A fresh run takes ~98 minutes (most of it is Stage 7 LLM verification on 23k
items, which is rate-limited by the Azure deployment, not by compute).

Usage:
    python run_all.py                 # full end-to-end
    python run_all.py --skip-validate # skip the final spot-check report
    python run_all.py --dry-run       # print the plan without executing
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
PYTHON = sys.executable

STEPS = [
    ("Stage 1-5: match + LLM tiebreak", [PYTHON, "run.py"]),
    ("Stage 6: bidirectional consistency", [PYTHON, "postprocess.py"]),
    ("Stage 7: LLM verification", [PYTHON, "-m", "src.verify"]),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-validate", action="store_true",
                   help="Skip the final validate.py spot-check.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan without executing.")
    args = p.parse_args()

    plan = list(STEPS)
    if not args.skip_validate:
        plan.append(("validate (spot-checks + tier breakdown)", [PYTHON, "validate.py"]))

    print("=" * 72)
    print("BetterBasket product-matching pipeline (run_all.py)")
    print("=" * 72)
    for i, (name, cmd) in enumerate(plan, 1):
        print(f"  Step {i}/{len(plan)}: {name}")
        print(f"            $ {' '.join(cmd)}")
    print()

    if args.dry_run:
        print("--dry-run: not executing.")
        return

    t_start = time.time()
    for i, (name, cmd) in enumerate(plan, 1):
        print(f"\n{'=' * 72}")
        print(f"Step {i}/{len(plan)}: {name}")
        print(f"{'=' * 72}\n")
        t0 = time.time()
        result = subprocess.run(cmd, cwd=str(ROOT))
        elapsed = round(time.time() - t0, 1)
        if result.returncode != 0:
            print(f"\nStep {i} failed (exit {result.returncode}). "
                  f"Aborting after {elapsed}s.")
            sys.exit(result.returncode)
        print(f"\nStep {i} done in {elapsed}s.")

    total = round(time.time() - t_start, 1)
    print(f"\n{'=' * 72}")
    print(f"All steps complete in {total}s ({total/60:.1f} min).")
    print(f"Deliverable: {ROOT / 'artifacts' / 'matches.csv'}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
