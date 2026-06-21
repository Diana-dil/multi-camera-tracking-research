from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mct_research.evaluation import write_motchallenge_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert project CSV to MOTChallenge text format")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = write_motchallenge_file(pd.read_csv(args.input), Path(args.output))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
