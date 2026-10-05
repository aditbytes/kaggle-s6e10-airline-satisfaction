"""Build a submission CSV from one or more experiments.

With several experiments, predictions are rank-averaged (ROC AUC only cares about order,
so ranks blend models whose probabilities are on different scales).

Usage:
  python src/submit.py exp001_lgbm_raw
  python src/submit.py exp001_lgbm_raw exp002_lgbm_fe --name blend_001_002
"""
import argparse

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from config import COMPETITION, ID, OUTPUT_DIR, SUBMISSION_DIR, TARGET
from data import load_sample_submission, load_test, load_train


def load(exp: str, kind: str) -> np.ndarray:
    path = OUTPUT_DIR / exp / f"{kind}.npy"
    if not path.exists():
        raise SystemExit(f"{path} missing; run: python src/train.py --exp {exp}")
    return np.load(path)


def blend(preds: list[np.ndarray]) -> np.ndarray:
    if len(preds) == 1:
        return preds[0]
    return np.mean([rankdata(p) / len(p) for p in preds], axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("experiments", nargs="+")
    p.add_argument("--name", help="submission file name (default: the experiment name)")
    args = p.parse_args()
    name = args.name or "__".join(args.experiments)

    y = load_train()[TARGET].to_numpy()
    for exp in args.experiments:
        print(f"  {exp:30s} OOF AUC {roc_auc_score(y, load(exp, 'oof')):.6f}")
    cv = roc_auc_score(y, blend([load(e, "oof") for e in args.experiments]))
    print(f"  {'=> ' + name:30s} OOF AUC {cv:.6f}")

    sub = load_sample_submission()
    pred = blend([load(e, "test") for e in args.experiments])
    if len(pred) != len(sub) or not (load_test().index.to_numpy() == sub[ID].to_numpy()).all():
        raise SystemExit("test rows and sample submission ids are not in the same order")
    sub[TARGET] = pred
    assert sub[ID].is_unique and sub[TARGET].between(0, 1).all()

    SUBMISSION_DIR.mkdir(exist_ok=True)
    path = SUBMISSION_DIR / f"{name}.csv"
    sub.to_csv(path, index=False)
    print(f"wrote {path.name} ({len(sub):,} rows)\n\nSubmit with:\n"
          f"  kag submit {COMPETITION} {path.relative_to(SUBMISSION_DIR.parent)} --cv {cv:.6f} -m \"{name}\"")


if __name__ == "__main__":
    main()
