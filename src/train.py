"""Train LightGBM with stratified K-fold CV and report the out-of-fold ROC AUC.

Usage:
  python src/train.py --exp exp001_lgbm_raw --features raw
  python src/train.py --exp smoke --quick        # 50k rows, a few seconds
"""
import argparse
import json
import time
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

import features
from config import EXPERIMENT_DIR, N_FOLDS, OUTPUT_DIR, ROOT, SEED, TARGET
from cv import make_folds
from data import load_test, load_train
from models import EARLY_STOPPING_ROUNDS, LGBM_PARAMS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True, help="experiment name, e.g. exp001_lgbm_raw")
    p.add_argument("--features", default="raw", help="feature set from features.py")
    p.add_argument("--lr", type=float, help="override the learning rate")
    p.add_argument("--quick", action="store_true", help="smoke test on a 50k-row sample")
    return p.parse_args()


def main():
    args = parse_args()
    started = time.time()
    train, test = load_train(), load_test()
    if args.quick:
        train, test = train.sample(50_000, random_state=SEED), test.head(10_000)

    y = train[TARGET].to_numpy()
    X = features.build(args.features, train)
    X_test = features.build(args.features, test)
    folds = make_folds(train[TARGET])
    params = {**LGBM_PARAMS, **({"learning_rate": args.lr} if args.lr else {})}
    print(f"{args.exp}: {X.shape[0]:,} rows x {X.shape[1]} features ({args.features}), lr={params['learning_rate']}")

    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_scores, best_iters = [], []
    importance = np.zeros(X.shape[1])
    for k in range(N_FOLDS):
        t = time.time()
        tr, va = folds != k, folds == k
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X[tr], y[tr],
            eval_X=X[va],
            eval_y=y[va],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        oof[va] = model.predict_proba(X[va])[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1] / N_FOLDS
        fold_scores.append(roc_auc_score(y[va], oof[va]))
        best_iters.append(model.best_iteration_)
        importance += model.booster_.feature_importance("gain") / N_FOLDS
        print(f"  fold {k}: AUC {fold_scores[-1]:.6f}  trees {best_iters[-1]:>5}  {time.time() - t:5.1f}s")

    cv = roc_auc_score(y, oof)
    minutes = (time.time() - started) / 60
    print(f"OOF AUC {cv:.6f}  (folds {np.mean(fold_scores):.6f} ± {np.std(fold_scores):.6f})  {minutes:.1f} min")
    if args.quick:
        return  # smoke tests don't overwrite real results

    out = OUTPUT_DIR / args.exp
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "oof.npy", oof)
    np.save(out / "test.npy", test_pred)

    gain = importance / importance.sum()
    top = sorted(zip(X.columns, gain), key=lambda t: -t[1])
    EXPERIMENT_DIR.mkdir(exist_ok=True)
    result = {
        "experiment": args.exp,
        "model": "LightGBM",
        "features": args.features,
        "n_features": X.shape[1],
        "oof_auc": round(cv, 6),
        "fold_auc": [round(s, 6) for s in fold_scores],
        "fold_auc_std": round(float(np.std(fold_scores)), 6),
        "best_iterations": best_iters,
        "params": params,
        "feature_importance_gain": {name: round(float(g), 4) for name, g in top},
        "minutes": round(minutes, 1),
        "finished": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
    (EXPERIMENT_DIR / f"{args.exp}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"saved predictions to {out.relative_to(ROOT)}/ and results to experiments/{args.exp}.json")


if __name__ == "__main__":
    main()
