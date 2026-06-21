from __future__ import annotations

import argparse
from pathlib import Path

from mct_research.comparison import find_summaries, load_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare completed tracking runs.")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output", default="results/comparison.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = find_summaries(args.results_root)
    if not summaries:
        raise SystemExit(f"No summary.json files found under {args.results_root}")

    comparison = load_comparison(summaries)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output, index=False, encoding="utf-8")
    markdown_path = output.with_suffix(".md")
    markdown_path.write_text(comparison.to_markdown(index=False), encoding="utf-8")
    print(comparison.to_string(index=False))
    print(f"Saved: {output}")
    print(f"Saved: {markdown_path}")


if __name__ == "__main__":
    main()
