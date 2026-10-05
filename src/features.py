"""Feature sets. Pick one with `train.py --features <name>`."""
import pandas as pd

from config import CATEGORICAL, NUMERIC, SERVICE_RATINGS


def raw(df: pd.DataFrame) -> pd.DataFrame:
    """The 21 original columns, categoricals kept as pandas `category` for LightGBM."""
    return df[NUMERIC + SERVICE_RATINGS + CATEGORICAL].copy()


FEATURE_SETS = {
    "raw": raw,
}


def build(name: str, df: pd.DataFrame) -> pd.DataFrame:
    if name not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {name!r}; choose from {sorted(FEATURE_SETS)}")
    return FEATURE_SETS[name](df)
