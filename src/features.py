"""Feature sets. Pick one with `train.py --features <name>`."""
import pandas as pd

from config import CATEGORICAL, NUMERIC, SERVICE_RATINGS


def raw(df: pd.DataFrame) -> pd.DataFrame:
    """The 21 original columns, categoricals kept as pandas `category` for LightGBM."""
    return df[NUMERIC + SERVICE_RATINGS + CATEGORICAL].copy()


def fe(df: pd.DataFrame) -> pd.DataFrame:
    """raw + summaries of the 13 service ratings, delay features and segment combinations."""
    out = raw(df)
    ratings = df[SERVICE_RATINGS]
    out["svc_mean"] = ratings.mean(axis=1)
    out["svc_min"] = ratings.min(axis=1)
    out["svc_max"] = ratings.max(axis=1)
    out["svc_std"] = ratings.std(axis=1)
    out["svc_n5"] = (ratings == 5).sum(axis=1)
    out["svc_n0"] = (ratings == 0).sum(axis=1)       # 0 = "not applicable" in the original survey
    out["svc_low"] = ((ratings > 0) & (ratings <= 2)).sum(axis=1)
    out["boarding_x_wifi"] = df["Online boarding"] * df["Inflight wifi service"]

    dep, arr = df["Departure Delay in Minutes"], df["Arrival Delay in Minutes"]
    out["delay_total"] = dep + arr.fillna(dep)
    out["delay_recovered"] = dep - arr                 # >0: plane made up time in the air
    out["delay_per_1000km"] = out["delay_total"] / (df["Flight Distance"] / 1000)

    out["class_x_travel"] = (df["Class"].astype(str) + "|" + df["Type of Travel"].astype(str)).astype("category")
    out["customer_x_travel"] = (df["Customer Type"].astype(str) + "|" + df["Type of Travel"].astype(str)).astype("category")
    return out


FEATURE_SETS = {
    "raw": raw,
    "fe": fe,
}


def build(name: str, df: pd.DataFrame) -> pd.DataFrame:
    if name not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {name!r}; choose from {sorted(FEATURE_SETS)}")
    return FEATURE_SETS[name](df)
