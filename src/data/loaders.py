import pandas as pd
from pathlib import Path
from typing import Union, List, Optional


def load_sample_dataset(csv_path: Union[str, Path] = "dataset/sample.csv") -> pd.DataFrame:
    """
    Load the small development sample dataset. Do not load the large CSV.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Sample dataset not found at {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def load_csv_with_limit(csv_path: Union[str, Path], n_rows: Optional[int] = None, use_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load only the first N rows from a CSV file. If n_rows is None, pandas will
    read everything (not recommended for huge files). Optionally select columns.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at {csv_path}")
    df = pd.read_csv(csv_path, nrows=n_rows, usecols=use_cols)
    return df
