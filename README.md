# Spam Classification Project

This repository implements a full pipeline for:

1. **Step 1**: Data preparation (BoW + Bernoulli CSV generation)
2. **Step 2**: Logistic Regression (Batch GD, Mini-batch GD, SGD)
3. **Step 3**: Multinomial Naive Bayes (BoW only)
4. **Step 4**: Bernoulli Naive Bayes (Bernoulli features only)
5. **Combined NB table**: Step 3 + Step 4 in one terminal table

## Project Files

- `data_prep.py`: Builds feature CSVs from raw Enron email folders (Step 1)
- `logistic_reg.py`: Runs logistic regression experiments and outputs metrics (Step 2)
- `multinomial_nb.py`: Runs Multinomial Naive Bayes on BoW features (Step 3)
- `bernoulli_nb.py`: Runs Bernoulli Naive Bayes on Bernoulli features (Step 4)
- `nb_combined_output.py`: Combines Step 3 + Step 4 into one instruction-style table

## Requirements

- Python 3.9+ 
- `numpy` for Steps 2–4 and combined output script

Install dependency:

```bash
python3 -m pip install numpy
```

## Expected Data Location

Put extracted Enron folders under `dataset/`.

The Step 1 script supports common extracted naming patterns, including:

- `dataset/enron1_train` + `dataset/enron1_test`
- `dataset/enron2_train` + `dataset/enron2_test`
- `dataset/enron4_train` + `dataset/enron4_test`

and other variants from repeated unzip operations (e.g. `enron1 2`, etc.).

Important naming note:
- Step 1 depends on dataset folder names to detect each split (`enron1`, `enron2`, `enron4`).
- If your extracted folder names are different, Step 1 may not find them automatically.
- You can fix this in either way:
1. Rename your dataset folders to one of the supported patterns above.
2. Update the path patterns inside `data_prep.py` in `detect_dataset_splits()`.

## Full Run Commands

Run from project root:

```bash
cd /path/to/project
```

## Chronological Execution Order (Step -> Filename)

1. Step 1 -> `data_prep.py`
2. Step 2 -> `logistic_reg.py`
3. Step 3 -> `multinomial_nb.py`
4. Step 4 -> `bernoulli_nb.py`
5. Combined Step 3+4 table -> `nb_combined_output.py`

### Step 1: Build feature CSVs

```bash
python3 data_prep.py --dataset-root dataset --output-dir step1_outputs
```

Outputs (12 files):

- `enron1_bow_train.csv`, `enron1_bow_test.csv`
- `enron1_bernoulli_train.csv`, `enron1_bernoulli_test.csv`
- `enron2_bow_train.csv`, `enron2_bow_test.csv`
- `enron2_bernoulli_train.csv`, `enron2_bernoulli_test.csv`
- `enron4_bow_train.csv`, `enron4_bow_test.csv`
- `enron4_bernoulli_train.csv`, `enron4_bernoulli_test.csv`

### Step 2: Logistic Regression

```bash
python3 logistic_reg.py --features-dir step1_outputs
```
Note: if no seed option is passed, Step 2 uses default seed `42`.

Or run with an auto-generated random seed:

```bash
python3 logistic_reg.py --features-dir step1_outputs --random-seed
```
Or run with a specific seed number:

```bash
python3 logistic_reg.py --features-dir step1_outputs --seed 123
```

Default behavior:

- Datasets: `enron1,enron2,enron4`
- Representations: `bow,bernoulli`
- Variants: `batch,minibatch,sgd`
- Lambda tuning: `0.01,0.1,1.0,10.0`
- Train/validation split: 70/30
- Default seed: 42 (use `--random-seed` to auto-generate one)
- Final test metrics: Accuracy, Precision, Recall, F1
- Output: shown in terminal only (no Step 2 CSV export)

Variant aliases supported:

- `batch` or `Batch GD`
- `minibatch` or `Mini-batch GD`
- `sgd` or `SGD`

Representation aliases supported:

- `bow` or `BoW`
- `bernoulli` / `bern` / `Bern`

### Step 3: Multinomial Naive Bayes (BoW only)

```bash
python3 multinomial_nb.py --features-dir step1_outputs
```

### Step 4: Bernoulli Naive Bayes (Bernoulli features only)

```bash
python3 bernoulli_nb.py --features-dir step1_outputs
```

### Combined Step 3 + Step 4 Table

```bash
python3 nb_combined_output.py --features-dir step1_outputs
```

Combined columns:

- `Dataset`
- `NB Variant`
- `Accuracy`
- `Precision`
- `Recall`
- `F1`

## Output Overview

- Step 1 writes feature CSVs (`*_bow_*.csv`, `*_bernoulli_*.csv`).
- Step 2 prints Logistic Regression results in terminal.
- Step 3 prints Multinomial NB results in terminal.
- Step 4 prints Bernoulli NB results in terminal.
- Combined Step 3+4 script prints one instruction-style NB table in terminal.

## Common Issues

### 1) `ModuleNotFoundError: No module named 'numpy'`

```bash
python3 -m pip install numpy
```

### 2) Step 1 cannot detect dataset splits

- Confirm extracted folders are under `dataset/`
- Confirm folder names match one of Step 1 supported naming patterns.
- If needed, point to a different root:

```bash
python3 data_prep.py --dataset-root <your_dataset_dir> --output-dir step1_outputs
```

- If names still differ, either rename folders or edit `detect_dataset_splits()` in `data_prep.py`.

### 3) Different machine / different absolute path

The scripts use relative paths by default. They are portable if you:

1. `cd` into project root
2. Keep dataset under `dataset/` (or pass explicit path flags)

## Citation Notes
- Python Software Foundation. *Python Language Reference, version 3*. https://docs.python.org/3/
- Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). *Array programming with NumPy*. Nature, 585, 357-362. https://doi.org/10.1038/s41586-020-2649-2
- Bird, Steven, Edward Loper, and Ewan Klein (2009). *Natural Language Processing with Python*. O'Reilly Media.
