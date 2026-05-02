"""Entry point. Run the full matching pipeline.

Usage:
    python run.py                      # full run with LLM tiebreak
    python run.py --no-llm             # skip LLM (rule-only baseline)
    python run.py --sample 5000        # run on a 5k-item sample of A
"""
import argparse
from src.pipeline import run

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--no-llm", action="store_true", help="Skip LLM tiebreaking")
    p.add_argument("--sample", type=int, default=None,
                   help="Sample N items from A instead of running on all")
    p.add_argument("--workers", type=int, default=8, help="LLM concurrency")
    args = p.parse_args()
    run(a_sample_n=args.sample, use_llm=not args.no_llm, llm_workers=args.workers)
