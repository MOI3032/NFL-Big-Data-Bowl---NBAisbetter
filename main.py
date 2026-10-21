"""Entry point.

    python main.py                 # all weeks (1-8)
    python main.py --weeks 1 2     # development subset
    python main.py --force         # rebuild the parquet cache
"""
import argparse

from src.pipeline import analyse, compute


def main():
    ap = argparse.ArgumentParser(description="Pocket Integrity Score & Blitz Time Pressure")
    ap.add_argument("--weeks", type=int, nargs="+", default=list(range(1, 9)))
    ap.add_argument("--force", action="store_true", help="ignore the parquet cache")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    res = compute(weeks=args.weeks, force=args.force)
    analyse(res, make_figures=not args.no_figures)


if __name__ == "__main__":
    main()
