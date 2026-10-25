
"""Register and validate the Tourism Package Prediction dataset."""

from pathlib import Path
import sys

import pandas as pd


# Resolve paths relative to this script, not the current working directory.
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "tourism.csv"

EXPECTED_COLUMNS = {
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
    "PitchSatisfactionScore",
    "ProductPitched",
    "NumberOfFollowups",
    "DurationOfPitch",
}


def register_dataset():
    """Validate the repository dataset and print its summary."""

    print("Tourism Package Prediction - Dataset Registration")
    print("-" * 55)
    print(f"Dataset path: {DATA_PATH}")

    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Ensure tourism.csv is committed to tourism_project/data/."
        )

    df = pd.read_csv(DATA_PATH)

    # Normalize column names to avoid failures from accidental whitespace.
    df.columns = df.columns.str.strip()

    missing_columns = sorted(EXPECTED_COLUMNS - set(df.columns))

    if missing_columns:
        raise ValueError(
            "Dataset registration failed. Missing expected columns: "
            + ", ".join(missing_columns)
        )

    if df.empty:
        raise ValueError("Dataset registration failed: CSV contains no rows.")

    if "ProdTaken" in df.columns:
        unexpected_targets = set(df["ProdTaken"].dropna().unique()) - {0, 1}
        if unexpected_targets:
            raise ValueError(
                f"Invalid values in target column ProdTaken: "
                f"{sorted(unexpected_targets, key=str)}"
            )

    print("\nDataset registered and validated successfully.")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    print(f"Expected columns verified: {len(EXPECTED_COLUMNS)}")
    print(f"Duplicate rows: {df.duplicated().sum():,}")
    print(f"Missing values: {df.isna().sum().sum():,}")

    if "ProdTaken" in df.columns:
        print("\nTarget distribution (ProdTaken):")
        print(df["ProdTaken"].value_counts(dropna=False).sort_index())

    print("\nColumn names:")
    print(", ".join(df.columns))


if __name__ == "__main__":
    try:
        register_dataset()
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
