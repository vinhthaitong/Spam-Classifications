#!/usr/bin/env python3
"""
Step 4: Bernoulli Naive Bayes for CS 4375 Project 1.

This script implements Bernoulli (Binary/Discrete) Naive Bayes for text
classification on Bernoulli datasets generated in Step 1.

For each dataset (enron1, enron2, enron4), it:
  1. Loads Bernoulli training and test CSV files.
  2. Trains Bernoulli NB with add-one Laplace smoothing (alpha=1 by default).
  3. Computes predictions in log-space.
  4. Reports accuracy, precision, recall, and F1-score on test data in terminal.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: numpy.\n"
        "Install it with:\n"
        "  python3 -m pip install numpy"
    ) from exc


def parse_csv_list(raw: str) -> List[str]:
    """Parse a comma-separated string into a cleaned list.

    Args:
        raw: Comma-separated string such as "enron1,enron2,enron4".

    Returns:
        List[str]: Non-empty stripped entries.
    """
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Step 4 experiments.

    Returns:
        argparse.Namespace: Parsed options for feature paths, smoothing,
        and selected datasets.
    """
    parser = argparse.ArgumentParser(description="Run Step 4 Bernoulli Naive Bayes experiments.")
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
        help="Laplace smoothing value (readme requirement uses alpha=1).",
    )
    return parser.parse_args()


def resolve_features_dir(path: Path) -> Path:
    """Validate and resolve the Step 1 feature directory.

    Args:
        path: Candidate features directory.

    Returns:
        Path: Resolved feature directory.

    Raises:
        FileNotFoundError: If directory does not exist.
    """
    if not path.is_dir():
        raise FileNotFoundError(f"Features directory not found: {path}")
    return path


def load_feature_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load one Step 1 CSV file into feature matrix and label vector.

    Args:
        path: CSV path with format:
            feature_1,feature_2,...,feature_n,label

    Returns:
        Tuple[np.ndarray, np.ndarray, List[str]]:
            - X: Feature matrix (n_samples, n_features), dtype float64
            - y: Label vector (n_samples,), dtype int64
            - feature_names: Feature column names from CSV header
    """
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        feature_names = header[:-1]

    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    X = data[:, :-1]
    y = data[:, -1].astype(np.int64)
    return X, y, feature_names


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute binary classification metrics.

    Args:
        y_true: Ground-truth labels in {0,1}.
        y_pred: Predicted labels in {0,1}.

    Returns:
        Dict[str, float]: accuracy, precision, recall, and f1.
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def train_bernoulli_nb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
) -> Dict[str, np.ndarray]:
    """Train Bernoulli Naive Bayes with Laplace smoothing.

    Bernoulli event model per class c and word i:
      theta_{c,i} = P(x_i=1 | c)
                  = (N_{c,i} + alpha) / (N_c + 2*alpha)
    where:
      - N_c is the number of docs in class c
      - N_{c,i} is the number of class-c docs where word i is present

    Values are stored in log-space for numeric stability.

    Args:
        X_train: Binary training matrix (presence/absence per document).
        y_train: Training labels in {0,1}.
        alpha: Laplace smoothing value (typically 1.0).

    Returns:
        Dict[str, np.ndarray]:
            - log_priors: shape (2,), log P(c)
            - log_theta: shape (2, vocab_size), log P(x_i=1|c)
            - log_one_minus_theta: shape (2, vocab_size), log P(x_i=0|c)

    Raises:
        ValueError: If alpha is non-positive.
    """
    if alpha <= 0:
        raise ValueError("alpha must be > 0 for Laplace smoothing.")

    # Ensure binary input in case CSV contains non-binary values.
    X_bin = (X_train > 0).astype(np.float64)

    n_docs, vocab_size = X_bin.shape
    log_priors = np.zeros(2, dtype=np.float64)
    log_theta = np.zeros((2, vocab_size), dtype=np.float64)
    log_one_minus_theta = np.zeros((2, vocab_size), dtype=np.float64)

    for c in (0, 1):
        class_mask = y_train == c
        X_c = X_bin[class_mask]
        n_c = X_c.shape[0]
        if n_c == 0:
            raise ValueError(f"Class {c} has zero documents in training data.")

        log_priors[c] = np.log(n_c / n_docs)

        # Number of class-c documents with feature i present.
        docs_with_word = X_c.sum(axis=0)
        theta = (docs_with_word + alpha) / (n_c + 2.0 * alpha)

        # Log probabilities for present/absent feature events.
        log_theta[c] = np.log(theta)
        log_one_minus_theta[c] = np.log(1.0 - theta)

    return {
        "log_priors": log_priors,
        "log_theta": log_theta,
        "log_one_minus_theta": log_one_minus_theta,
    }


def predict_bernoulli_nb(X_test: np.ndarray, model: Dict[str, np.ndarray]) -> np.ndarray:
    """Predict labels with a trained Bernoulli NB model.

    For each class c and document x:
      score_c = log P(c) +
                sum_i [x_i * log P(x_i=1|c) + (1-x_i) * log P(x_i=0|c)]

    Computation remains in log-space and does not exponentiate scores.

    Args:
        X_test: Test matrix (binary or numeric; values >0 treated as 1).
        model: Trained model dictionary from train_bernoulli_nb.

    Returns:
        np.ndarray: Predicted labels in {0,1}.
    """
    X_bin = (X_test > 0).astype(np.float64)
    log_priors = model["log_priors"]
    log_theta = model["log_theta"]
    log_one_minus_theta = model["log_one_minus_theta"]

    present_scores = X_bin @ log_theta.T
    absent_scores = (1.0 - X_bin) @ log_one_minus_theta.T
    scores = log_priors + present_scores + absent_scores
    return np.argmax(scores, axis=1).astype(np.int64)


def run_dataset_experiment(
    dataset: str,
    features_dir: Path,
    alpha: float,
) -> Dict[str, float | str]:
    """Run Step 4 Bernoulli NB for one dataset using Bernoulli files only.

    Args:
        dataset: Dataset name (enron1, enron2, or enron4).
        features_dir: Directory containing Step 1 output CSV files.
        alpha: Laplace smoothing value.

    Returns:
        Dict[str, float | str]: Result row for this dataset.
    """
    train_csv = features_dir / f"{dataset}_bernoulli_train.csv"
    test_csv = features_dir / f"{dataset}_bernoulli_test.csv"
    if not train_csv.is_file() or not test_csv.is_file():
        raise FileNotFoundError(
            f"Missing Bernoulli CSV for dataset={dataset}. "
            f"Expected: {train_csv} and {test_csv}"
        )

    X_train, y_train, _ = load_feature_csv(train_csv)
    X_test, y_test, _ = load_feature_csv(test_csv)

    model = train_bernoulli_nb(X_train, y_train, alpha=alpha)
    y_pred = predict_bernoulli_nb(X_test, model)
    metrics = compute_metrics(y_true=y_test, y_pred=y_pred)

    return {
        "dataset": dataset,
        "nb_variant": "Bernoulli",
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "alpha": alpha,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
    }


def print_result_summary(results: Sequence[Dict[str, float | str]]) -> None:
    """Print a compact Step 4 result table to stdout.

    Args:
        results: Sequence of Step 4 result rows.
    """
    print("\nBernoulli Naive Bayes Results")
    print("-" * 84)
    print(
        f"{'Dataset':<8} {'NB Variant':<20} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}"
    )
    print("-" * 84)
    for row in results:
        print(
            f"{str(row['dataset']):<8} "
            f"{str(row['nb_variant']):<20} "
            f"{float(row['accuracy']):>8.4f} "
            f"{float(row['precision']):>8.4f} "
            f"{float(row['recall']):>8.4f} "
            f"{float(row['f1']):>8.4f}"
        )
    print("-" * 84)


def main() -> None:
    """Entry point for Step 4 Bernoulli NB experiments."""
    args = parse_args()
    features_dir = resolve_features_dir(args.features_dir)
    datasets = parse_csv_list(args.datasets)

    if abs(args.alpha - 1.0) > 1e-12:
        print(
            "Warning: project readme specifies add-one Laplace smoothing (alpha=1). "
            f"You provided alpha={args.alpha}."
        )

    results: List[Dict[str, float | str]] = []
    for dataset in datasets:
        print(f"Running Bernoulli NB for dataset={dataset} (Bernoulli features only)")
        result = run_dataset_experiment(
            dataset=dataset,
            features_dir=features_dir,
            alpha=args.alpha,
        )
        results.append(result)

    print_result_summary(results)
    print("\nFinished Step 4. Results are shown in terminal output only.")


if __name__ == "__main__":
    main()
