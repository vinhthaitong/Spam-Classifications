#!/usr/bin/env python3
"""
Step 2: Logistic Regression for CS 4375 Project 1.

This script trains and evaluates L2-regularized Logistic Regression models on
Step 1 feature CSV files for:
  - datasets: enron1, enron2, enron4
  - representations: bow, bernoulli
  - optimization variants: batch, minibatch, sgd

For each (dataset, representation, variant), it:
  1. Splits training CSV into 70% train / 30% validation.
  2. Tunes lambda (and minibatch size, when applicable) on validation data.
  3. Retrains on full training CSV using the best hyperparameters.
  4. Evaluates accuracy, precision, recall, and F1 on the test CSV.

Seed behavior:
  - Default seed is 42 (reproducible runs).
  - Use --seed <int> to set a specific seed.
  - Use --random-seed to generate one random seed for the run.
"""

from __future__ import annotations

import argparse
import csv
import secrets
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: numpy.\n"
        "Install it with:\n"
        "  python3 -m pip install numpy"
    ) from exc

VARIANT_DISPLAY = {
    "batch": "Batch GD",
    "minibatch": "Mini-batch GD",
    "sgd": "SGD",
}

REPRESENTATION_DISPLAY = {
    "bow": "BoW",
    "bernoulli": "Bern",
}

DEFAULT_SEED = 42


def parse_csv_list(raw: str) -> List[str]:
    """Parse a comma-separated string into a cleaned list.

    Args:
        raw: Comma-separated string, such as "enron1,enron2,enron4".

    Returns:
        List[str]: Non-empty, stripped entries preserving input order.
    """
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_float_list(raw: str) -> List[float]:
    """Parse a comma-separated string of floats.

    Args:
        raw: Comma-separated float string, such as "0.01,0.1,1.0,10.0".

    Returns:
        List[float]: Parsed float values.
    """
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_int_list(raw: str) -> List[int]:
    """Parse a comma-separated string of integers.

    Args:
        raw: Comma-separated integer string, such as "50,100".

    Returns:
        List[int]: Parsed integer values.
    """
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def normalize_variant_name(name: str) -> str:
    """Normalize a variant alias to the internal variant key.

    Accepted aliases:
        - batch, Batch GD
        - minibatch, Mini-batch GD
        - sgd, SGD

    Args:
        name: Raw variant name from CLI.

    Returns:
        str: Internal key in {"batch", "minibatch", "sgd"}.

    Raises:
        ValueError: If the name does not map to a supported variant.
    """
    key = name.strip().lower()
    mapping = {
        "batch": "batch",
        "batch gd": "batch",
        "minibatch": "minibatch",
        "mini-batch": "minibatch",
        "mini-batch gd": "minibatch",
        "sgd": "sgd",
    }
    if key not in mapping:
        raise ValueError(f"Unsupported variant '{name}'")
    return mapping[key]


def normalize_representation_name(name: str) -> str:
    """Normalize a representation alias to internal CSV naming.

    Accepted aliases:
        - bow, BoW
        - bernoulli, bern, Bern

    Args:
        name: Raw representation name from CLI.

    Returns:
        str: Internal key in {"bow", "bernoulli"}.

    Raises:
        ValueError: If the name does not map to a supported representation.
    """
    key = name.strip().lower()
    mapping = {
        "bow": "bow",
        "bernoulli": "bernoulli",
        "bern": "bernoulli",
    }
    if key not in mapping:
        raise ValueError(f"Unsupported representation '{name}'")
    return mapping[key]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Step 2 training/evaluation.

    Returns:
        argparse.Namespace: Parsed options for data paths, model settings,
        and experiment selections.
    """
    parser = argparse.ArgumentParser(description="Run Step 2 Logistic Regression experiments.")
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
        "--representations",
        type=str,
        default="bow,bernoulli",
        help="Comma-separated representation list.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="batch,minibatch,sgd",
        help="Comma-separated optimization variants.",
    )
    parser.add_argument(
        "--lambdas",
        type=str,
        default="0.01,0.1,1.0,10.0",
        help="Comma-separated L2 lambda values to tune.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate for gradient descent updates.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Number of epochs for model training.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="50,100",
        help="Comma-separated minibatch sizes (used when variant=minibatch).",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.3,
        help="Validation split ratio from training CSV (e.g., 0.3 means 70/30 split).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducible split and batch shuffling (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--random-seed",
        action="store_true",
        help="Generate a random seed for this run and print it.",
    )
    return parser.parse_args()


def resolve_features_dir(path: Path) -> Path:
    """Validate and resolve the feature directory path.

    Args:
        path: Candidate directory containing Step 1 CSV files.

    Returns:
        Path: Resolved directory path.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    if not path.is_dir():
        raise FileNotFoundError(f"Features directory not found: {path}")
    return path


def load_feature_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load one Step 1 CSV file into feature matrix and label vector.

    Args:
        path: CSV path in Step 1 format:
            - first row header: feature columns + label
            - subsequent rows: numeric feature values + binary label

    Returns:
        Tuple[np.ndarray, np.ndarray, List[str]]:
            - X: Feature matrix, shape (n_samples, n_features), dtype float64
            - y: Binary label vector, shape (n_samples,), dtype float64
            - feature_names: Ordered feature names from the CSV header
    """
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        feature_names = header[:-1]

    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    X = data[:, :-1]
    y = data[:, -1]
    return X, y, feature_names


def split_train_validation(
    X: np.ndarray,
    y: np.ndarray,
    validation_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split a training dataset into train and validation subsets.

    Args:
        X: Full training feature matrix.
        y: Full training labels.
        validation_ratio: Fraction for validation split (e.g., 0.3).
        seed: Random seed for deterministic shuffling.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - X_train, y_train, X_val, y_val

    Raises:
        ValueError: If split would create an empty train or validation set.
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    split_idx = int(round(n_samples * (1.0 - validation_ratio)))
    if split_idx <= 0 or split_idx >= n_samples:
        raise ValueError(
            f"Invalid validation split produced empty set: n_samples={n_samples}, "
            f"validation_ratio={validation_ratio}"
        )

    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Compute numerically stable sigmoid values.

    Args:
        z: Input logits array.

    Returns:
        np.ndarray: Element-wise sigmoid output in (0, 1).
    """
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def predict_probabilities(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Predict class-1 probabilities for input features.

    Args:
        X: Feature matrix.
        w: Weight vector.
        b: Bias scalar.

    Returns:
        np.ndarray: Predicted probabilities P(y=1|x).
    """
    return sigmoid(X @ w + b)


def predict_labels(probabilities: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert probabilities into binary predictions.

    Args:
        probabilities: Predicted probabilities in [0, 1].
        threshold: Classification threshold.

    Returns:
        np.ndarray: Binary predictions (0 or 1).
    """
    return (probabilities >= threshold).astype(np.int64)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute accuracy, precision, recall, and F1 for binary predictions.

    Args:
        y_true: Ground-truth labels (0/1).
        y_pred: Predicted labels (0/1).

    Returns:
        Dict[str, float]: Metric dictionary with keys:
            - accuracy
            - precision
            - recall
            - f1
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


def iterate_minibatches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    rng: np.random.Generator,
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """Yield minibatches from a dataset.

    Args:
        X: Feature matrix.
        y: Label vector.
        batch_size: Batch size used per update step.
        shuffle: Whether to shuffle sample order each epoch.
        rng: Random generator for shuffling.

    Yields:
        Iterable[Tuple[np.ndarray, np.ndarray]]: Batches (X_batch, y_batch).
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    if shuffle:
        rng.shuffle(indices)

    for start in range(0, n_samples, batch_size):
        batch_idx = indices[start : start + batch_size]
        yield X[batch_idx], y[batch_idx]


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    lambda_value: float,
    learning_rate: float,
    epochs: int,
    variant: str,
    seed: int,
    batch_size: int | None = None,
) -> Tuple[np.ndarray, float]:
    """Train L2-regularized Logistic Regression using a GD variant.

    Model:
        p = sigmoid(Xw + b)
    Objective:
        Binary cross-entropy + L2(w), where b is not regularized.

    Args:
        X: Training features.
        y: Binary labels.
        lambda_value: L2 regularization coefficient.
        learning_rate: Step size for gradient updates.
        epochs: Number of full passes over training data.
        variant: One of {"batch", "minibatch", "sgd"}.
        seed: Random seed for minibatch/SGD shuffling.
        batch_size: Batch size for minibatch variant only.

    Returns:
        Tuple[np.ndarray, float]:
            - w: Trained weight vector
            - b: Trained bias scalar

    Raises:
        ValueError: If unsupported variant or invalid minibatch config is used.
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features, dtype=np.float64)
    b = 0.0

    if variant == "batch":
        effective_batch_size = n_samples
        shuffle = False
    elif variant == "sgd":
        effective_batch_size = 1
        shuffle = True
    elif variant == "minibatch":
        if batch_size is None or batch_size <= 0:
            raise ValueError("minibatch variant requires a positive batch_size")
        effective_batch_size = batch_size
        shuffle = True
    else:
        raise ValueError(f"Unknown optimization variant: {variant}")

    rng = np.random.default_rng(seed)

    for _ in range(epochs):
        for X_batch, y_batch in iterate_minibatches(
            X=X,
            y=y,
            batch_size=effective_batch_size,
            shuffle=shuffle,
            rng=rng,
        ):
            m_batch = X_batch.shape[0]
            probs = predict_probabilities(X_batch, w, b)
            errors = probs - y_batch

            grad_w = (X_batch.T @ errors) / m_batch
            grad_w += (lambda_value / n_samples) * w
            grad_b = float(np.mean(errors))

            w -= learning_rate * grad_w
            b -= learning_rate * grad_b

    return w, b


def evaluate_model(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> Dict[str, float]:
    """Evaluate a trained Logistic Regression model on a dataset.

    Args:
        X: Evaluation feature matrix.
        y: Evaluation labels.
        w: Trained weight vector.
        b: Trained bias scalar.

    Returns:
        Dict[str, float]: accuracy/precision/recall/f1 metrics.
    """
    probabilities = predict_probabilities(X, w, b)
    predictions = predict_labels(probabilities, threshold=0.5)
    return compute_metrics(y_true=y, y_pred=predictions)


def is_better_candidate(candidate: Dict[str, float], best: Dict[str, float] | None) -> bool:
    """Decide whether a validation candidate is better than the current best.

    Ranking priority:
        1. Higher validation accuracy
        2. Higher validation F1
        3. Smaller lambda
        4. Smaller batch size (if both present)

    Args:
        candidate: Candidate hyperparameter result.
        best: Current best result, or None.

    Returns:
        bool: True if candidate should replace best.
    """
    if best is None:
        return True

    c_acc = candidate["val_accuracy"]
    b_acc = best["val_accuracy"]
    if c_acc != b_acc:
        return c_acc > b_acc

    c_f1 = candidate["val_f1"]
    b_f1 = best["val_f1"]
    if c_f1 != b_f1:
        return c_f1 > b_f1

    c_lambda = candidate["lambda"]
    b_lambda = best["lambda"]
    if c_lambda != b_lambda:
        return c_lambda < b_lambda

    c_batch = candidate.get("batch_size", 0)
    b_batch = best.get("batch_size", 0)
    return c_batch < b_batch


def tune_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lambdas: Sequence[float],
    learning_rate: float,
    epochs: int,
    variant: str,
    batch_sizes: Sequence[int],
    seed: int,
) -> Dict[str, float]:
    """Tune lambda (and minibatch size when relevant) on validation data.

    Args:
        X_train: Inner-train features for tuning.
        y_train: Inner-train labels for tuning.
        X_val: Validation features.
        y_val: Validation labels.
        lambdas: Candidate lambda values.
        learning_rate: Learning rate for training.
        epochs: Number of epochs per candidate.
        variant: Optimization variant ("batch", "minibatch", "sgd").
        batch_sizes: Candidate minibatch sizes (only used for minibatch variant).
        seed: Base random seed for reproducible candidate training.

    Returns:
        Dict[str, float]: Best candidate summary including
        lambda, optional batch_size, and validation metrics.
    """
    candidates: List[Tuple[float, int | None]] = []
    for lambda_value in lambdas:
        if variant == "minibatch":
            for batch_size in batch_sizes:
                candidates.append((lambda_value, batch_size))
        else:
            candidates.append((lambda_value, None))

    best_result: Dict[str, float] | None = None
    for i, (lambda_value, batch_size) in enumerate(candidates):
        w, b = train_logistic_regression(
            X=X_train,
            y=y_train,
            lambda_value=lambda_value,
            learning_rate=learning_rate,
            epochs=epochs,
            variant=variant,
            seed=seed + i,
            batch_size=batch_size,
        )
        val_metrics = evaluate_model(X_val, y_val, w, b)
        candidate: Dict[str, float] = {
            "lambda": float(lambda_value),
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
        }
        if batch_size is not None:
            candidate["batch_size"] = float(batch_size)

        if is_better_candidate(candidate, best_result):
            best_result = candidate

    assert best_result is not None
    return best_result


def run_variant_experiment(
    dataset: str,
    representation: str,
    variant: str,
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    lambdas: Sequence[float],
    batch_sizes: Sequence[int],
    learning_rate: float,
    epochs: int,
    validation_ratio: float,
    seed: int,
) -> Dict[str, float | str]:
    """Run a full Step 2 experiment for one optimization variant.

    Args:
        dataset: Dataset name (e.g., enron1).
        representation: Representation name (bow or bernoulli).
        variant: Optimization variant (batch/minibatch/sgd).
        X_train_full: Full training features from CSV.
        y_train_full: Full training labels from CSV.
        X_test: Test features from CSV.
        y_test: Test labels from CSV.
        lambdas: Candidate regularization values.
        batch_sizes: Candidate minibatch sizes.
        learning_rate: Learning rate for gradient updates.
        epochs: Number of epochs for training.
        validation_ratio: Fraction used for validation split.
        seed: Base random seed.

    Returns:
        Dict[str, float | str]: Result row containing selected hyperparameters,
        validation metrics, and final test metrics.
    """
    X_train, y_train, X_val, y_val = split_train_validation(
        X=X_train_full,
        y=y_train_full,
        validation_ratio=validation_ratio,
        seed=seed,
    )

    best = tune_hyperparameters(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        lambdas=lambdas,
        learning_rate=learning_rate,
        epochs=epochs,
        variant=variant,
        batch_sizes=batch_sizes,
        seed=seed,
    )

    best_lambda = float(best["lambda"])
    best_batch_size = int(best["batch_size"]) if "batch_size" in best else None

    w, b = train_logistic_regression(
        X=X_train_full,
        y=y_train_full,
        lambda_value=best_lambda,
        learning_rate=learning_rate,
        epochs=epochs,
        variant=variant,
        seed=seed + 777,
        batch_size=best_batch_size,
    )
    test_metrics = evaluate_model(X_test, y_test, w, b)

    result: Dict[str, float | str] = {
        "dataset": dataset,
        "representation": REPRESENTATION_DISPLAY.get(representation, representation),
        "variant": VARIANT_DISPLAY.get(variant, variant),
        "Best λ": best_lambda,
        "val_accuracy": float(best["val_accuracy"]),
        "val_precision": float(best["val_precision"]),
        "val_recall": float(best["val_recall"]),
        "val_f1": float(best["val_f1"]),
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "n_train": int(X_train_full.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train_full.shape[1]),
    }
    return result


def print_result_summary(results: Sequence[Dict[str, float | str]]) -> None:
    """Print a compact summary table to stdout.

    Args:
        results: Sequence of final experiment results.
    """
    print("\nLogistic Regression Results")
    print("-" * 92)
    print(
        f"{'Dataset':<8} {'Repr':<10} {'Variant':<14} "
        f"{'Best λ':>8} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}"
    )
    print("-" * 92)
    for row in results:
        print(
            f"{str(row['dataset']):<8} "
            f"{str(row['representation']):<10} "
            f"{str(row['variant']):<14} "
            f"{float(row['Best λ']):>8.4f} "
            f"{float(row['test_accuracy']):>8.4f} "
            f"{float(row['test_precision']):>8.4f} "
            f"{float(row['test_recall']):>8.4f} "
            f"{float(row['test_f1']):>8.4f}"
        )
    print("-" * 92)


def main() -> None:
    """Entry point for Step 2 experiments."""
    args = parse_args()
    features_dir = resolve_features_dir(args.features_dir)
    run_seed = int(args.seed)
    if args.random_seed:
        # Draw one seed per run so all experiments in this execution are consistent.
        run_seed = secrets.randbelow(2**32 - 1) + 1
        print(f"Using random seed: {run_seed}")
    else:
        print(f"Using seed: {run_seed}")

    datasets = parse_csv_list(args.datasets)
    raw_representations = parse_csv_list(args.representations)
    raw_variants = parse_csv_list(args.variants)
    lambdas = parse_float_list(args.lambdas)
    batch_sizes = parse_int_list(args.batch_sizes)
    representations = [normalize_representation_name(name) for name in raw_representations]
    variants = [normalize_variant_name(name) for name in raw_variants]

    results: List[Dict[str, float | str]] = []

    for dataset in datasets:
        for representation in representations:
            train_csv = features_dir / f"{dataset}_{representation}_train.csv"
            test_csv = features_dir / f"{dataset}_{representation}_test.csv"
            if not train_csv.is_file() or not test_csv.is_file():
                raise FileNotFoundError(
                    f"Missing Step 1 CSV for dataset={dataset}, representation={representation}. "
                    f"Expected: {train_csv} and {test_csv}"
                )

            print(f"\nLoading data: dataset={dataset}, representation={representation}")
            X_train_full, y_train_full, _ = load_feature_csv(train_csv)
            X_test, y_test, _ = load_feature_csv(test_csv)

            for variant in variants:
                print(
                    f"Running variant={VARIANT_DISPLAY.get(variant, variant)} "
                    f"(train={X_train_full.shape[0]}, test={X_test.shape[0]}, "
                    f"features={X_train_full.shape[1]})"
                )
                result = run_variant_experiment(
                    dataset=dataset,
                    representation=representation,
                    variant=variant,
                    X_train_full=X_train_full,
                    y_train_full=y_train_full,
                    X_test=X_test,
                    y_test=y_test,
                    lambdas=lambdas,
                    batch_sizes=batch_sizes,
                    learning_rate=args.learning_rate,
                    epochs=args.epochs,
                    validation_ratio=args.validation_ratio,
                    seed=run_seed,
                )
                results.append(result)

    print_result_summary(results)


if __name__ == "__main__":
    main()
