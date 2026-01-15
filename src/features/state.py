from typing import List
import pandas as pd


BASE_FEATURES: List[str] = [
    "VL",
    "CD4",
    "Rel CD4",
    "Gender",
    "Ethnic",
    "Base Drug Combo",
    "Comp. INI",
    "Comp. NNRTI",
    "Extra PI",
    "Extra pk-En",
]

ACTION_COL = "Drug (M)"
OUTCOME_COLS = ["VL (M)", "CD4 (M)"]
ID_COL = "PatientID"
TIME_COL = "Timestep"


def add_lag_features(df: pd.DataFrame, lags: int = 1) -> pd.DataFrame:
    df = df.sort_values([ID_COL, TIME_COL]).copy()
    for col in BASE_FEATURES + [ACTION_COL] + OUTCOME_COLS:
        for k in range(1, lags + 1):
            df[f"{col}_lag{k}"] = df.groupby(ID_COL)[col].shift(k)
    df = df.dropna().reset_index(drop=True)
    return df


def build_state_action_table(raw: pd.DataFrame) -> pd.DataFrame:
    df = add_lag_features(raw, lags=1)
    return df
