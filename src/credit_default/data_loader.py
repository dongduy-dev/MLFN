"""
data_loader.py
==============
Reusable module for loading the raw UCI Credit Card Default dataset.

Key design decisions:
- Paths are relative to the project root (no hard-coded absolutes).
- The raw .xls file has TWO header rows:
    Row 0: generic labels (X1, X2, ..., X23, Y)
    Row 1: meaningful column names (ID, LIMIT_BAL, ..., default payment next month)
  We use header=1 to read the meaningful names.
- No renaming, recoding, imputation or transformation is applied.
- The raw dataset is treated as immutable.
"""

import hashlib
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Path configuration (relative to project root)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = _PROJECT_ROOT / "data" / "raw"
RAW_DATASET_FILENAME = "default of credit card clients.xls"
RAW_DATASET_PATH = RAW_DATA_DIR / RAW_DATASET_FILENAME


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return _PROJECT_ROOT


def get_raw_dataset_path() -> Path:
    """Return the absolute path to the raw dataset file."""
    return RAW_DATASET_PATH


def load_raw_dataset() -> pd.DataFrame:
    """
    Load the raw UCI Credit Card Default dataset without any modification.

    Returns
    -------
    pd.DataFrame
        The raw dataset with original column names from header row 1.

    Raises
    ------
    FileNotFoundError
        If the raw dataset file does not exist at the expected location.
    """
    path = get_raw_dataset_path()

    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at: {path}\n"
            f"Expected file: '{RAW_DATASET_FILENAME}'\n"
            f"Expected location: {RAW_DATA_DIR}\n"
            f"Please download the dataset from the UCI Machine Learning Repository:\n"
            f"https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients"
        )

    # header=1 skips the generic X1..Y row (row 0) and uses the
    # meaningful column names in row 1 (ID, LIMIT_BAL, SEX, ...).
    df = pd.read_excel(path, header=1, engine="xlrd")

    return df


def compute_file_sha256(filepath: Path | None = None) -> str:
    """
    Compute the SHA-256 hash of a file.

    Parameters
    ----------
    filepath : Path, optional
        File to hash.  Defaults to the raw dataset.

    Returns
    -------
    str
        Hex-encoded SHA-256 digest.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    if filepath is None:
        filepath = get_raw_dataset_path()

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()
