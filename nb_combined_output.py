#!/usr/bin/env python3
"""
Combined Step 3 + Step 4 Naive Bayes results table for CS 4375 Project 1.

This script runs:
  - Step 3 Multinomial NB (BoW)
  - Step 4 Bernoulli NB

and prints a single combined table in instruction-style format:
  Dataset, NB Variant, Accuracy, Precision, Recall, F1
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

from multinomial_nb import run_dataset_experiment as run_multinomial_experiment
from bernoulli_nb import run_dataset_experiment as run_bernoulli_experiment


def parse_csv_list(raw: str) -> List[str]:
    """Parse a comma-separated string into a cleaned list.

    Args:
        raw: Comma-separated string such as "enron1,enron2,enron4".

    Returns:
        List[str]: Non-empty stripped entries preserving input order.
    """
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for combined NB reporting.

    Returns:
        argparse.Namespace: Parsed options for feature directory, datasets,
        and smoothing.
    """
    parser = argparse.ArgumentParser(
        description="Generate one combined output table for Step 3 and Step 4."
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=Path("step1_outputs"),
        help="Directory containing Step 1 CSV outputs.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="enron1,enron2,enron4",
        help="Comma-separated dataset list.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Laplace smoothing value for both NB variants.",
    )
    return parser.parse_args()


def normalize_dataset_name(name: str) -> str:
    """Format dataset names to match instruction table style.

    Args:
        name: Raw dataset name such as "enron1".

    Returns:
        str: Styled dataset name such as "Enron1".
    """
    if name.lower().startswith("enron"):
        suffix = name[len("enron") :]
        return f"Enron{suffix}"
    return name


def to_combined_row(raw_result: Dict[str, float | str]) -> Dict[str, float | str]:
    """Project a Step 3/4 result row into combined output columns.

    Args:
        raw_result: Result dictionary from Step 3 or Step 4 experiment.

    Returns:
        Dict[str, float | str]: Combined table row with standardized columns.
    """
    dataset = str(raw_result["dataset"])
    return {
        "Dataset": normalize_dataset_name(dataset),
        "NB Variant": str(raw_result["nb_variant"]),
        "Accuracy": float(raw_result["accuracy"]),
        "Precision": float(raw_result["precision"]),
        "Recall": float(raw_result["recall"]),
        "F1": float(raw_result["f1"]),
    }


def print_combined_table(rows: Sequence[Dict[str, float | str]]) -> None:
    """Print the combined Step 3/4 result table in terminal.

    Args:
        rows: Final combined rows.
    """
    print("\nCombined Naive Bayes Results")
    print("-" * 88)
    print(
        f"{'Dataset':<8} {'NB Variant':<20} "
        f"{'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}"
    )
    print("-" * 88)
    for row in rows:
        print(
            f"{str(row['Dataset']):<8} "
            f"{str(row['NB Variant']):<20} "
            f"{float(row['Accuracy']):>10.4f} "
            f"{float(row['Precision']):>10.4f} "
            f"{float(row['Recall']):>10.4f} "
            f"{float(row['F1']):>10.4f}"
        )
    print("-" * 88)


def main() -> None:
    """Entry point for combined Step 3 + Step 4 reporting."""
    args = parse_args()
    datasets = parse_csv_list(args.datasets)
    features_dir = args.features_dir

    if not features_dir.is_dir():
        raise FileNotFoundError(f"Features directory not found: {features_dir}")

    combined_rows: List[Dict[str, float | str]] = []

    for dataset in datasets:
        print(f"Running Multinomial NB for {dataset}...")
        step3_result = run_multinomial_experiment(
            dataset=dataset,
            features_dir=features_dir,
            alpha=args.alpha,
        )
        combined_rows.append(to_combined_row(step3_result))

        print(f"Running Bernoulli NB for {dataset}...")
        step4_result = run_bernoulli_experiment(
            dataset=dataset,
            features_dir=features_dir,
            alpha=args.alpha,
        )
        combined_rows.append(to_combined_row(step4_result))

    print_combined_table(combined_rows)
    print("\nFinished combined Step 3 + Step 4 report. Results are shown in terminal output only.")


if __name__ == "__main__":
    main()
