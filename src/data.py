"""Load the competition CSVs with consistent dtypes."""
import pandas as pd

from config import CATEGORICAL, DATA_DIR, ID, TARGET


def _read(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the data first:\n"
            "  kaggle competitions download playground-series-s6e10 -p data && unzip data/*.zip -d data"
        )
    return pd.read_csv(path)


def load_train() -> pd.DataFrame:
    df = _read("train.csv")
    df[TARGET] = df[TARGET].astype(str).str.lower().eq("true").astype("int8")
    return _set_categories(df)


def load_test() -> pd.DataFrame:
    return _set_categories(_read("test.csv"))


def load_sample_submission() -> pd.DataFrame:
    return _read("sample_submission.csv")


def _set_categories(df: pd.DataFrame) -> pd.DataFrame:
    for col in CATEGORICAL:
        df[col] = df[col].astype("category")
    return df.set_index(ID)
