"""Model hyperparameters."""
from config import SEED

LGBM_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "n_estimators": 10_000,   # upper bound; early stopping picks the real number per fold
    "num_leaves": 63,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
}
EARLY_STOPPING_ROUNDS = 200
