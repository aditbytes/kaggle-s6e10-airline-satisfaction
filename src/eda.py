"""Exploratory data analysis -> reports/eda.md

Usage: python src/eda.py
"""
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import CATEGORICAL, NUMERIC, REPORT_DIR, SERVICE_RATINGS, TARGET
from data import load_test, load_train


def md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    def fmt(v):
        return floatfmt.format(v) if isinstance(v, float) else str(v)
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [fmt(v) for v in row]) + " |")
    return "\n".join(lines)


def single_feature_auc(train: pd.DataFrame) -> pd.DataFrame:
    """How well each feature alone ranks satisfied vs not (0.5 = useless, 1.0 = perfect)."""
    y = train[TARGET]
    rows = {}
    for col in NUMERIC + SERVICE_RATINGS + CATEGORICAL:
        x = train[col]
        if col in CATEGORICAL:  # score each category by its satisfaction rate
            x = x.map(train.groupby(col, observed=True)[TARGET].mean()).astype(float)
        auc = roc_auc_score(y, x.fillna(x.median()))
        rows[col] = {"auc": max(auc, 1 - auc), "direction": "+" if auc >= 0.5 else "-"}
    return pd.DataFrame(rows).T.sort_values("auc", ascending=False).rename_axis("feature")


def main():
    train, test = load_train(), load_test()
    out = ["# EDA: Predicting Airline Satisfaction", ""]

    out += ["## Shape and target", "",
            f"- Train: **{len(train):,}** rows · Test: **{len(test):,}** rows · Features: **{test.shape[1]}**",
            f"- Satisfied: **{train[TARGET].mean():.2%}** (balanced enough; stratify folds anyway)",
            f"- Exact duplicate feature rows in train: **{train.drop(columns=TARGET).duplicated().sum():,}**", ""]

    nulls = pd.DataFrame({"train": train.isna().sum(), "test": test.isna().sum()}).dropna()
    nulls = nulls[(nulls.train > 0) | (nulls.test > 0)].astype(int).rename_axis("column")
    out += ["## Missing values", "", md_table(nulls), "",
            "LightGBM handles NaN natively; no imputation needed for tree models.", ""]

    out += ["## Single-feature AUC", "",
            "Each feature on its own, as a ranking score for `satisfaction`.", "",
            md_table(single_feature_auc(train)), ""]

    for col in CATEGORICAL:
        g = train.groupby(col, observed=True)[TARGET].agg(rows="size", satisfied_rate="mean")
        g["rows"] = g["rows"].astype(int)
        out += [f"## {col}", "", md_table(g), ""]

    by_target = train.groupby(TARGET)[SERVICE_RATINGS].mean().T
    by_target.columns = ["mean if not satisfied", "mean if satisfied"]
    by_target["gap"] = by_target.iloc[:, 1] - by_target.iloc[:, 0]
    out += ["## Service ratings (0-5) by outcome", "",
            md_table(by_target.sort_values("gap", ascending=False).rename_axis("rating"), "{:.2f}"), ""]

    shift = pd.DataFrame({"train mean": train[NUMERIC + SERVICE_RATINGS].mean(),
                          "test mean": test[NUMERIC + SERVICE_RATINGS].mean()})
    shift["diff %"] = (shift["test mean"] / shift["train mean"] - 1) * 100
    out += ["## Train vs test drift", "",
            "Large differences would mean CV may not reflect the leaderboard.", "",
            md_table(shift.rename_axis("feature"), "{:.3f}"), ""]

    REPORT_DIR.mkdir(exist_ok=True)
    path = REPORT_DIR / "eda.md"
    path.write_text("\n".join(out))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
