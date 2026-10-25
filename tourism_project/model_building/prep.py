
"""Prepare the Tourism Package Prediction dataset for model training."""

from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split


# Resolve paths relative to the project directory.
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "tourism.csv"
ARTIFACT_DIR = PROJECT_DIR / "artifacts"

TARGET = "ProdTaken"
ID_COLUMNS = ["CustomerID"]

# Exclude features that may only be known after a sales interaction.
POST_INTERACTION_COLUMNS = [
    "DurationOfPitch",
    "NumberOfFollowups",
    "PitchSatisfactionScore",
    "ProductPitched",
]

TEST_SIZE = 0.20
RANDOM_STATE = 42


def prepare_data():
    """Clean, split, and save training/testing datasets."""

    print("Tourism Package Prediction - Data Preparation")
    print("-" * 55)

    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Ensure tourism.csv is in tourism_project/data/."
        )

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    if df.empty:
        raise ValueError("The input dataset is empty.")

    if TARGET not in df.columns:
        raise ValueError(f"Required target column '{TARGET}' is missing.")

    if df[TARGET].isna().any():
        raise ValueError(
            f"Target '{TARGET}' contains missing values. "
            "Resolve them before splitting."
        )

    if df[TARGET].nunique() < 2:
        raise ValueError(
            f"Target '{TARGET}' must contain at least two classes."
        )

    # Remove duplicate records.
    original_rows = len(df)
    df = df.drop_duplicates().copy()
    duplicates_removed = original_rows - len(df)

    # Remove rows with missing target labels; already checked above.
    # Keep other missing feature values for imputation in the model pipeline.
    columns_to_drop = [
        col
        for col in ID_COLUMNS + POST_INTERACTION_COLUMNS
        if col in df.columns and col != TARGET
    ]
    df = df.drop(columns=columns_to_drop)

    # Ensure the target is binary as specified in the data dictionary.
    observed_targets = set(df[TARGET].unique())
    if not observed_targets.issubset({0, 1}):
        raise ValueError(
            f"Expected binary target values 0 and 1; found {observed_targets}."
        )

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    # Stratification preserves the target class ratio in both splits.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Save each split with its target column included.
    train_df = X_train.copy()
    train_df[TARGET] = y_train

    test_df = X_test.copy()
    test_df[TARGET] = y_test

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = ARTIFACT_DIR / "train.csv"
    test_path = ARTIFACT_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Source dataset: {DATA_PATH}")
    print(f"Original rows: {original_rows:,}")
    print(f"Duplicate rows removed: {duplicates_removed:,}")
    print(f"Rows after cleaning: {len(df):,}")
    print(f"Columns removed: {columns_to_drop}")
    print(f"Training rows: {len(train_df):,}")
    print(f"Testing rows: {len(test_df):,}")
    print(f"Training target distribution:\n{y_train.value_counts().sort_index()}")
    print(f"Testing target distribution:\n{y_test.value_counts().sort_index()}")
    print(f"\nSaved training data: {train_path}")
    print(f"Saved testing data:  {test_path}")
    print("\nData preparation completed successfully.")


if __name__ == "__main__":
    try:
        prepare_data()
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
