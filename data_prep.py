#!/usr/bin/env python3
"""
Step 1 data preparation for CS 4375 Project 1.

Creates 12 CSV files:
  - enron{1,2,4}_{bow,bernoulli}_{train,test}.csv

Each CSV has:
  - one row per email
  - one column per vocabulary word (built from training split only)
  - final column named "label" (1=spam, 0=ham)

Important naming note:
  - Step 1 auto-detects train/test folders using known Enron naming patterns.
  - If your extracted folder names are different, detection can fail.
  - In that case, either rename folders to a supported pattern or edit
    detect_dataset_splits() below to match your local names.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

# Try to use NLTK utilities for tokenization and stopword handling.
# If NLTK is not installed, we keep running with local fallback logic below.
"""
@book{bird2009nltk,
  title     = {Natural Language Processing with Python},
  author    = {Bird, Steven and Loper, Edward and Klein, Ewan},
  year      = {2009},
  publisher = {O'Reilly Media}
}
"""

try:
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.tokenize import word_tokenize
except ImportError:
    # Sentinel values: later code checks these to decide whether to use fallback.
    nltk_stopwords = None
    word_tokenize = None

# Regex tokenizer fallback: keeps lowercase alphanumeric tokens only.
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

# Minimal built-in stopword list used when NLTK stopwords are unavailable.
FALLBACK_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}


def get_default_stopwords() -> set[str]:
    """Load NLTK English stopwords, or fall back to a built-in minimal set."""
    if nltk_stopwords is None:
        return set(FALLBACK_STOPWORDS)
    try:
        return set(nltk_stopwords.words("english"))
    except LookupError:
        # NLTK installed but corpus is not downloaded.
        return set(FALLBACK_STOPWORDS)


DEFAULT_STOPWORDS = get_default_stopwords()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Step 1 dataset conversion.

    Returns:
        argparse.Namespace: Parsed options containing:
            - dataset_root: Path to extracted Enron directories.
            - output_dir: Destination directory for generated CSV files.
            - keep_stopwords: Whether to keep stopwords during preprocessing.
    """
    parser = argparse.ArgumentParser(description="Generate Step 1 CSV feature matrices.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset"),
        help="Directory containing extracted Enron folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("step1_outputs"),
        help="Directory where generated CSV files will be written.",
    )
    parser.add_argument(
        "--keep-stopwords",
        action="store_true",
        help="Keep stopwords (default behavior is to remove a small stopword list).",
    )
    return parser.parse_args()


def preprocess_text(text: str, stopwords: set[str] | None) -> List[str]:
    """Normalize and tokenize raw email text.

    Uses NLTK tokenization when available; falls back to regex tokenization
    if NLTK is missing or its tokenizer resource is not installed.

    Args:
        text: Raw email content.
        stopwords: Set of stopwords to remove, or None to keep all tokens.

    Returns:
        List[str]: Ordered token list after preprocessing.
    """
    if word_tokenize is not None:
        try:
            tokens = word_tokenize(text.lower())
            tokens = [t for t in tokens if t.isalnum()]
        except LookupError:
            # NLTK tokenizer resource (e.g., punkt/punkt_tab) missing.
            tokens = TOKEN_PATTERN.findall(text.lower())
    else:
        tokens = TOKEN_PATTERN.findall(text.lower())

    if stopwords is None:
        return tokens
    return [t for t in tokens if t not in stopwords]



def read_text_file(path: Path) -> str:
    """Read one email file as UTF-8 text.

    Some emails contain malformed bytes; decoding errors are ignored so
    the conversion pipeline can continue instead of crashing.

    Args:
        path: Path to a single .txt email file.

    Returns:
        str: Decoded text content.
    """
    # Some emails include noisy bytes; ignore undecodable characters.
    return path.read_text(encoding="utf-8", errors="ignore")


def split_exists(split_dir: Path) -> bool:
    """Check whether a dataset split directory has expected class folders.

    A valid split must contain both:
        - ham/
        - spam/

    Args:
        split_dir: Candidate directory for one split (train or test).

    Returns:
        bool: True if both class directories exist; otherwise False.
    """
    return (split_dir / "ham").is_dir() and (split_dir / "spam").is_dir()


def detect_dataset_splits(dataset_root: Path) -> Dict[str, Tuple[Path, Path]]:
    """Resolve train/test folders for enron1, enron2, and enron4.

    The provided dataset folder can have slightly different extracted layouts
    (for example, with "enron1 2" after repeated unzip operations). This
    function tries known path patterns and selects the first valid train/test
    pair for each dataset.

    Filename/folder-name dependency:
        Step 1 relies on these candidate folder names. If your machine has
        different extracted names, update the `candidates` mapping below or
        rename your folders to a supported pattern.

    Args:
        dataset_root: Root directory containing extracted dataset folders.

    Returns:
        Dict[str, Tuple[Path, Path]]: Mapping from dataset name to
        (train_dir, test_dir).

    Raises:
        FileNotFoundError: If one or more required datasets cannot be resolved.
    """
    candidates: Dict[str, List[Tuple[Path, Path]]] = {
        "enron1": [
            (dataset_root / "enron1_train" / "train", dataset_root / "enron1_test" / "test"),
        ],
        "enron2": [
            (dataset_root / "enron2_train", dataset_root / "enron2_test"),
        ],
        "enron4": [
            (dataset_root / "enron4_train" / "train", dataset_root / "enron4_test" / "test"),
        ],
    }

    resolved: Dict[str, Tuple[Path, Path]] = {}
    for dataset_name, pairs in candidates.items():
        for train_dir, test_dir in pairs:
            if split_exists(train_dir) and split_exists(test_dir):
                resolved[dataset_name] = (train_dir, test_dir)
                break

    missing = [name for name in ("enron1", "enron2", "enron4") if name not in resolved]
    if missing:
        raise FileNotFoundError(
            "Could not detect train/test folders for: "
            + ", ".join(missing)
            + f". Checked under {dataset_root}. "
            + "If your folder names are different, rename them to a supported "
            + "pattern or edit detect_dataset_splits() in data_prep.py."
        )

    return resolved


def iter_split_files(split_dir: Path) -> Iterable[Tuple[Path, int]]:
    """Yield file paths and labels for one split in deterministic order.

    Label convention:
        - ham  -> 0
        - spam -> 1

    Files are sorted so outputs remain stable across runs.

    Args:
        split_dir: Split directory containing ham/ and spam/.

    Yields:
        Tuple[Path, int]: (email_file_path, label).
    """
    ham_dir = split_dir / "ham"
    spam_dir = split_dir / "spam"

    for path in sorted(ham_dir.glob("*.txt")):
        yield path, 0
    for path in sorted(spam_dir.glob("*.txt")):
        yield path, 1


def load_split(split_dir: Path, stopwords: set[str] | None) -> List[Tuple[List[str], int]]:
    """Load and preprocess all emails in a split.

    For each file, this function:
        1. Reads text from disk.
        2. Applies tokenization/preprocessing.
        3. Stores tokens with its class label.

    Args:
        split_dir: Split directory (train or test).
        stopwords: Stopword set used by preprocess_text, or None.

    Returns:
        List[Tuple[List[str], int]]: List of (tokens, label) examples.
    """
    rows: List[Tuple[List[str], int]] = []
    for file_path, label in iter_split_files(split_dir):
        text = read_text_file(file_path)
        tokens = preprocess_text(text, stopwords)
        rows.append((tokens, label))
    return rows


def build_vocabulary(train_docs: Sequence[Tuple[List[str], int]]) -> List[str]:
    """Build a fixed vocabulary from training documents only.

    Using only the training split avoids data leakage from the test split.
    Tokens are sorted to keep feature column order deterministic.

    Args:
        train_docs: Sequence of (tokens, label) from the training split.

    Returns:
        List[str]: Sorted unique vocabulary tokens.
    """
    vocab = sorted({token for tokens, _ in train_docs for token in tokens})
    return vocab


def vectorize_tokens(
    tokens: Sequence[str],
    vocab_index: Dict[str, int],
    representation: str,
) -> List[int]:
    """Convert one tokenized email into a dense feature vector.

    Supported representations:
        - "bow": word counts (frequency).
        - "bernoulli": binary presence (0/1).

    Tokens not found in vocab_index are ignored.

    Args:
        tokens: Tokenized email.
        vocab_index: Mapping token -> feature column index.
        representation: Conversion mode ("bow" or "bernoulli").

    Returns:
        List[int]: Dense feature vector aligned with vocabulary order.

    Raises:
        ValueError: If representation is not one of the supported modes.
    """
    vector = [0] * len(vocab_index)

    if representation == "bow":
        counts = Counter(tokens)
        for token, count in counts.items():
            idx = vocab_index.get(token)
            if idx is not None:
                vector[idx] = count
        return vector

    if representation == "bernoulli":
        for token in set(tokens):
            idx = vocab_index.get(token)
            if idx is not None:
                vector[idx] = 1
        return vector

    raise ValueError(f"Unknown representation: {representation}")


def write_csv(
    output_path: Path,
    docs: Sequence[Tuple[List[str], int]],
    vocabulary: Sequence[str],
    vocab_index: Dict[str, int],
    representation: str,
) -> None:
    """Write one dataset split to CSV using a chosen representation.

    CSV format:
        - Header: all vocabulary tokens + final "label" column.
        - Each row: feature values + class label.

    Args:
        output_path: Destination CSV path.
        docs: Sequence of (tokens, label) examples to serialize.
        vocabulary: Ordered vocabulary (used for header columns).
        vocab_index: Mapping token -> feature index.
        representation: "bow" or "bernoulli" conversion mode.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([*vocabulary, "label"])
        for tokens, label in docs:
            features = vectorize_tokens(tokens, vocab_index, representation)
            writer.writerow([*features, label])


def generate_for_dataset(
    dataset_name: str,
    train_dir: Path,
    test_dir: Path,
    output_dir: Path,
    stopwords: set[str] | None,
) -> None:
    """Generate all Step 1 CSV outputs for one dataset.

    For the given dataset, this function:
        1. Loads and preprocesses train/test documents.
        2. Builds vocabulary from training documents only.
        3. Writes four CSV files:
           - {dataset}_bow_train.csv
           - {dataset}_bow_test.csv
           - {dataset}_bernoulli_train.csv
           - {dataset}_bernoulli_test.csv

    Args:
        dataset_name: Dataset identifier (enron1, enron2, or enron4).
        train_dir: Training split directory.
        test_dir: Test split directory.
        output_dir: Base output directory for CSV files.
        stopwords: Stopword set for preprocessing, or None.
    """
    train_docs = load_split(train_dir, stopwords)
    test_docs = load_split(test_dir, stopwords)

    vocabulary = build_vocabulary(train_docs)
    vocab_index = {token: idx for idx, token in enumerate(vocabulary)}

    for representation in ("bow", "bernoulli"):
        train_out = output_dir / f"{dataset_name}_{representation}_train.csv"
        test_out = output_dir / f"{dataset_name}_{representation}_test.csv"
        write_csv(train_out, train_docs, vocabulary, vocab_index, representation)
        write_csv(test_out, test_docs, vocabulary, vocab_index, representation)

    print(
        f"{dataset_name}: train={len(train_docs)} test={len(test_docs)} "
        f"vocab_size={len(vocabulary)}"
    )


def main() -> None:
    """Entry point for Step 1 conversion pipeline.

    Workflow:
        1. Parse command-line options.
        2. Resolve dataset split locations.
        3. Run conversion for enron1, enron2, enron4.
        4. Print summary and output location.
    """
    args = parse_args()
    dataset_root = args.dataset_root
    output_dir = args.output_dir
    stopwords = None if args.keep_stopwords else set(DEFAULT_STOPWORDS)

    splits = detect_dataset_splits(dataset_root)
    print("Detected dataset splits:")
    for dataset_name in ("enron1", "enron2", "enron4"):
        train_dir, test_dir = splits[dataset_name]
        print(f"  {dataset_name}: train={train_dir} test={test_dir}")

    print("\nGenerating CSV files...")
    for dataset_name in ("enron1", "enron2", "enron4"):
        train_dir, test_dir = splits[dataset_name]
        generate_for_dataset(
            dataset_name=dataset_name,
            train_dir=train_dir,
            test_dir=test_dir,
            output_dir=output_dir,
            stopwords=stopwords,
        )

    print(f"\nDone. Wrote files to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
